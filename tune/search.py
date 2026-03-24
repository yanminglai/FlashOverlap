'''
    Using torchrun for distributed running:
        e.g., MUSA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 search.py --m 4096 --n 8192 --k 4096 --comm_op all_reduce
'''

import sys
import torch
import torch_musa
import argparse
import json
import os
from pathlib import Path
import torch.distributed as dist
import numpy as np

def dbg(msg):
    if not os.getenv('FLASH_DEBUG'):
        return
    rank = os.getenv('LOCAL_RANK', '?')
    print(f"[rank={rank}] {msg}", flush=True, file=sys.stderr)

_script_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_script_dir, "../build/lib/libst_pybinding.so"))

# Global distributed state
_rank = None
_local_rank = None
_world_size = None


def init_dist():
    global _rank, _local_rank, _world_size
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '29500'))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    dist.init_process_group(
        backend='mccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=world_size,
        rank=rank,
    )
    torch.musa.set_device(local_rank)
    _rank = rank
    _local_rank = local_rank
    _world_size = world_size


def _get_mccl_id():
    """Generate MCCL ID on rank 0 and broadcast to all ranks."""
    dbg("_get_mccl_id: enter")
    device = torch.device(f'musa:{_local_rank}')
    if _rank == 0:
        dbg("_get_mccl_id: generating...")
        mccl_id = torch.ops.flashoverlap_op.generate_mccl_id()
        dbg(f"_get_mccl_id: generated {mccl_id[:3]}...")
        mccl_id_tsr = torch.tensor(mccl_id, device=device)
    else:
        mccl_id_tsr = torch.zeros(16, dtype=torch.int64, device=device)
    dbg("_get_mccl_id: broadcasting...")
    dist.broadcast(mccl_id_tsr, src=0)
    dbg("_get_mccl_id: done")
    return mccl_id_tsr.cpu().tolist()


def div_up(x: int, y: int):
    return (x + y - 1) // y

def load_json(M: int, N: int, K: int):
    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'
    
    assert Path(file_path).exists(), "Please run preprocess.py first!"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data["BM"], data["BN"], data["dur"], data["Algo"]

def save_solution(M: int, N: int, K: int, BM: int, BN: int, gemm_dur: float, Algo: int, hint: list, cSeg: list):
    if _rank != 0:
        return
    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    file_path = f'../configs/m{M}n{N}k{K}_{gpu_name}.json'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data["hint"] = hint
    data["cSeg"] = cSeg
    data["rLDN"] = 1
    data["BM"] = BM
    data["BN"] = BN
    data["dur"] = gemm_dur
    data["Algo"] = Algo
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def generate_row_remap_array(
    M, N, BM, BN, S_list, world_size, device="musa"
):
    total_tiles = (M * N) // (BM * BN)
    assert sum(S_list) == total_tiles, "sum(S_list) must equal total number of tiles"
    
    original_row_ids = torch.arange(M * N // BN, dtype=torch.int, device=device)
    reordered_row_id = torch.empty_like(original_row_ids)
    
    current_row = 0
    for S in S_list:
        chunk_size = S * BM
        chunk_row_ids = original_row_ids[current_row : current_row + chunk_size]
        
        mod_values = chunk_row_ids % world_size
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    
    return remap


def compute_hint(M: int, N: int, K: int,
    BM: int, BN: int, Algo: list, wSize: int, comm_op: str):
    """Each rank runs the hint computation; result broadcast from rank 0."""
    dbg(f"compute_hint: enter BM={BM} BN={BN} Algo={Algo} wSize={wSize}")
    rank, world_size = _rank, _world_size

    TileNum = div_up(M, BM) * div_up(N, BN)
    WaveNum = div_up(TileNum, wSize)
    dbg(f"compute_hint: TileNum={TileNum} WaveNum={WaveNum}")

    cSeg = []
    for i in range(WaveNum):
        this_seg = min(wSize, TileNum - i * wSize)
        cSeg.append(this_seg)

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.musa(rank)

    mccl_id = _get_mccl_id()
    dbg("compute_hint: creating OverlapImpl")
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    dbg("compute_hint: mccl_init")
    gemm_class.mccl_init(rank, world_size, mccl_id)
    dbg("compute_hint: mutlass_init")
    gemm_class.mutlass_init()
    dbg("compute_hint: overlap_init")
    gemm_class.overlap_init()
    dbg("compute_hint: init done")

    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    MonitoredMatrix = torch.zeros(((M+BM-1)//BM + 1, (N+BN-1)//BN), dtype=torch.int, device="musa")
    ReorderedArray = torch.arange(0, TileNum, dtype=torch.int, device="musa").reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="musa")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = 100
    _sample = 20

    kernel_error = False
    try:
        if comm_op == "all_reduce":
            dbg(f"compute_hint: warmup {_warm_up} iters...")
            for wi in range(_warm_up):
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            torch.musa.synchronize()
            dbg("compute_hint: warmup done, sampling...")

            samples = torch.empty((_sample, TileNum), dtype=torch.int, device="musa")
            for i in range(_sample):
                MonitoredMatrix[0] = 0
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
                samples[i, :] = MonitoredMatrix[1:, :].view(-1)
            torch.musa.synchronize()
            dbg("compute_hint: sampling done")

        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
            torch.musa.synchronize()

            samples = torch.empty((_sample, TileNum), dtype=torch.int, device="musa")
            for i in range(_sample):
                MonitoredMatrix[0] = 0
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, True)
                samples[i, :] = MonitoredMatrix[1:, :].view(-1)
            torch.musa.synchronize()

        else:
            assert comm_op in ["all_reduce", "reduce_scatter"], \
                f"comm_op must be 'all_reduce' or 'reduce_scatter', but got '{comm_op}'"
    except RuntimeError as e:
        dbg(f"compute_hint: kernel error during warmup/sampling: {e}")
        kernel_error = True

    # Analysis: either skip (kernel error) or perform majority-vote per tile
    hint = []
    is_consistency = False
    if not kernel_error:
        # Move samples to CPU for analysis (avoids MUSA async error issues)
        samples_cpu = samples.cpu()

        # Per-tile majority vote: assign each tile to its most frequent wave
        tile_wave = samples_cpu // wSize  # shape: (_sample, TileNum), values 0..WaveNum-1
        tile_wave = tile_wave.clamp(max=WaveNum - 1)

        # For each tile, find the mode (most common wave across samples)
        tile_modes = torch.mode(tile_wave, dim=0).values  # shape: (TileNum,)

        # Compute confidence: how many samples agreed on the mode for each tile
        mode_counts = torch.zeros(TileNum, dtype=torch.long)
        for ti in range(TileNum):
            mode_counts[ti] = (tile_wave[:, ti] == tile_modes[ti]).sum()

        # Build per-wave tile lists, sorted by confidence (descending)
        wave_tiles = {w: [] for w in range(WaveNum)}
        for ti in range(TileNum):
            wave_tiles[tile_modes[ti].item()].append((mode_counts[ti].item(), ti))
        for w in wave_tiles:
            wave_tiles[w].sort(reverse=True)

        # Redistribute: assign exactly the target count per wave,
        # keeping the most confident tiles, moving surplus to unassigned pool
        unassigned = []
        assigned = {w: [] for w in range(WaveNum)}
        for w in range(WaveNum):
            target = wSize if w < WaveNum - 1 else TileNum - w * wSize
            tiles = wave_tiles[w]
            assigned[w] = [t[1] for t in tiles[:target]]
            unassigned.extend([t[1] for t in tiles[target:]])

        # Fill waves that are still short from unassigned tiles
        redistributed = 0
        for w in range(WaveNum):
            target = wSize if w < WaveNum - 1 else TileNum - w * wSize
            while len(assigned[w]) < target and unassigned:
                assigned[w].append(unassigned.pop())
                redistributed += 1

        # Build hint from assigned tiles
        is_consistency = True
        for w in range(WaveNum):
            target = wSize if w < WaveNum - 1 else TileNum - w * wSize
            if len(assigned[w]) < target:
                dbg(f"compute_hint: wave {w}/{WaveNum} FAIL: need {target}, got {len(assigned[w])}")
                is_consistency = False
                break
            dbg(f"compute_hint: wave {w}/{WaveNum} OK: {len(assigned[w])} tiles")
            hint.extend(assigned[w])

        dbg(f"compute_hint: redistributed {redistributed}/{TileNum} tiles")

    dbg(f"compute_hint: kernel_error={kernel_error} is_consistency={is_consistency} hint_len={len(hint)}")
    # Broadcast result from rank 0 to all ranks so every rank agrees
    consistency_tsr = torch.tensor([int(is_consistency)], dtype=torch.int64, device="musa")
    dbg("compute_hint: broadcasting consistency...")
    dist.broadcast(consistency_tsr, src=0)
    is_consistency = bool(consistency_tsr.item())
    dbg(f"compute_hint: consistency broadcast done, is_consistency={is_consistency}")

    if is_consistency:
        if rank == 0:
            hint_tsr = torch.tensor(hint, dtype=torch.int64, device="musa")
            hint_len_tsr = torch.tensor([len(hint)], dtype=torch.int64, device="musa")
        else:
            hint_len_tsr = torch.zeros(1, dtype=torch.int64, device="musa")
        dist.broadcast(hint_len_tsr, src=0)
        hint_len = hint_len_tsr.item()
        if rank != 0:
            hint_tsr = torch.zeros(hint_len, dtype=torch.int64, device="musa")
        dist.broadcast(hint_tsr, src=0)
        hint = hint_tsr.cpu().tolist()

    dbg("compute_hint: final barrier...")
    dist.barrier()
    dbg("compute_hint: done")
    return (is_consistency, hint)


def interpolate_latency(samples, x, comm_op):
    world_size = _world_size
    if not isinstance(samples, torch.Tensor):
        samples = torch.tensor(samples, dtype=torch.float32)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    data_sizes = samples[:, 0].numpy()
    bandwidths = samples[:, 1].numpy()
    x_np = x.numpy()

    y_np = np.interp(x_np, data_sizes, bandwidths)
    y = torch.tensor(y_np, dtype=torch.float32).item()

    if comm_op == "all_reduce":
        # busbw stored in GB/s; latency_s = msgsize * 2*(n-1)/n / (busbw * GiB)
        latency = x * 2 * 2 * (world_size - 1) / world_size / y / (1024 ** 3) * 1000
    elif comm_op == "reduce_scatter":
        latency = x * 2 * (world_size - 1) / world_size / y / (1024 ** 3) * 1000

    return latency.item()

def predict_lat(M: int, N: int, gemm_dur: float, 
    comm_array: torch.Tensor, gp: list, tile_num: int, comm_op: str):

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    sm_count = props.multi_processor_count
    
    acc_comm_dur = 0
    acc_comp_dur = 0
    iter_num = len(gp)

    if iter_num == 1:
        acc_comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[0], comm_op) + gemm_dur
        return acc_comm_dur

    old_wave_num = (tile_num + sm_count - 1) // sm_count
    new_wave_num = (tile_num + sm_count - 3) // (sm_count - 2)
    gemm_dur = gemm_dur / old_wave_num * new_wave_num

    for i in range(iter_num):
        if i == 0:
            comm_dur = 0
        else:
            comm_dur = interpolate_latency(comm_array, M*N // tile_num * gp[i - 1], comm_op)
        acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + comm_dur 
        acc_comp_dur += gemm_dur / new_wave_num * ((gp[i] + sm_count - 3) // (sm_count - 2))
    acc_comm_dur = max(acc_comp_dur, acc_comm_dur) + interpolate_latency(comm_array, M*N // tile_num * gp[-1], comm_op)

    return acc_comm_dur

def reorder_indices(S, hint):
    original = list(range(S))
    new_order = [-1] * S
    for i, element in enumerate(hint):
        new_order[element] = i
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    return torch.tensor(new_order, dtype=torch.int, device="musa")


def perf_running(M: int, N: int, K: int, 
    BM: int, BN: int, Algo: int, 
    cSeg: list, hint: list, comm_op: str):
    """Each rank runs the benchmark; returns max duration across all ranks."""
    dbg(f"perf_running: enter cSeg={cSeg}")
    rank, world_size = _rank, _world_size

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.musa(rank)

    TileNum = div_up(M, BM) * div_up(N, BN)

    mccl_id = _get_mccl_id()
    dbg("perf_running: creating OverlapImpl")
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    dbg("perf_running: mccl_init")
    gemm_class.mccl_init(rank, world_size, mccl_id)
    dbg("perf_running: mutlass_init")
    gemm_class.mutlass_init()
    dbg("perf_running: overlap_init")
    gemm_class.overlap_init()
    dbg("perf_running: init done")

    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    MonitoredMatrix = torch.zeros(((N+BN-1)//BN), dtype=torch.int, device="musa")
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="musa")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)
    
    _warm_up = 20
    _freq = 200

    if len(cSeg) == 1:
        # No overlapping
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce(A, B, C, Algo)

            gemm_class.gemm_allreduce(A, B, C, Algo)

            start_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce(A, B, C, Algo)
                end_event[i].record()
            torch.musa.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            MonitoredMatrix[0] = 0
            gemm_class.gemm_reducescatter(A, B, C, D, Algo)

            start_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter(A, B, C, D, Algo)
                end_event[i].record()
            torch.musa.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)

    else:
        if comm_op == "all_reduce":
            for _ in range(_warm_up):
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_allreduce_overlap(A, B, C, MonitoredMatrix, ReorderedArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.musa.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        elif comm_op == "reduce_scatter":
            for _ in range(_warm_up):
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)

            start_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            end_event = [torch.musa.Event(enable_timing=True) for i in range(_freq)]
            for i in range(_freq):
                start_event[i].record()
                gemm_class.gemm_reducescatter_overlap(A, B, C, D, MonitoredMatrix, ReorderedArray, RowArray, 1, cSeg_CPU, cSeg_GPU, Algo, False)
                end_event[i].record()
            torch.musa.synchronize()
            dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
        else:
            dur = torch.zeros((_freq))

    local_dur = torch.mean(dur).item()

    # Get max duration across all ranks
    dur_tensor = torch.tensor([local_dur], dtype=torch.float64, device="musa")
    dist.all_reduce(dur_tensor, op=dist.ReduceOp.MAX)
    return dur_tensor.item()


def integer_partitions(n):
    result = []
    def helper(remaining, path):
        if remaining == 0:
            result.append(path)
            return
        for i in range(1, remaining + 1):
            helper(remaining - i, path + [i])
    helper(n, [])
    return result

def exhaustive_search(M: int, N: int, K: int, comm_op: str):
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    sm_count = props.multi_processor_count

    hint = None
    for t in range(min(5, len(BM_list))):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, (sm_count - 2))

        result = compute_hint(M, N, K, BM, BN, Algo, (sm_count - 2), comm_op)

        if result[0] == True:
            hint = result[1]
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    if _rank == 0:
        print("Start exhaustive searching.")

    min_dur = 1e5

    group_size_list = integer_partitions(wave_num)
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * (sm_count - 2)
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * (sm_count - 2), tile_num - acc)
        dur = perf_running(M, N, K, BM, BN, Algo, gp, hint, comm_op)
        if _rank == 0:
            print(gp, "%.4f" % (dur))

        if dur < min_dur:
            min_dur = dur
            cSeg = gp
        
    if _rank == 0:
        print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg)
    if _rank == 0:
        print("Solution saved.")


def fast_search(M: int, N: int, K: int, comm_array: torch.Tensor, comm_op: str):
    dbg(f"fast_search: enter M={M} N={N} K={K}")
    BM_list, BN_list, gemm_dur_list, Algo_list = load_json(M, N, K)
    dbg(f"fast_search: loaded json, {len(BM_list)} configs")

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    sm_count = props.multi_processor_count

    hint = None
    for t in range(min(10, len(BM_list))):
        BM = BM_list[t]
        BN = BN_list[t]
        gemm_dur = gemm_dur_list[t]
        Algo = Algo_list[t]

        tile_num = div_up(M, BM) * div_up(N, BN)
        wave_num = div_up(tile_num, (sm_count - 2))

        min_group_size = div_up(wave_num, 10)

        dbg(f"fast_search: try t={t} BM={BM} BN={BN} Algo={Algo} tile_num={tile_num} wave_num={wave_num} min_group_size={min_group_size}")
        result = compute_hint(M, N, K, BM, BN, Algo, min_group_size * (sm_count - 2), comm_op)
        dbg(f"fast_search: compute_hint returned is_consistency={result[0]}")

        if result[0] == True:
            hint = result[1]
            dbg(f"fast_search: found hint with {len(hint)} entries at t={t}")
            break

    assert hint != None, "Tuning fails! Try to increase min_group_size manully."
    if _rank == 0:
        print("Start predictive searching.")
    
    min_dur = 1e5
    normalized_wave_num = div_up(wave_num, min_group_size)
    group_size_list = integer_partitions(normalized_wave_num)
    
    group_choice = len(group_size_list)
    for i in range(group_choice):
        gp = group_size_list[i]
        iter_num = len(gp)
        acc = 0
        if iter_num > 5 and gp[0] > 2:
            continue
        for j in range(iter_num):
            if j < iter_num - 1:
                gp[j] = gp[j] * (sm_count - 2) * min_group_size
                acc += gp[j]
            else:
                gp[j] = min(gp[j] * (sm_count - 2) * min_group_size, tile_num - acc)
        est_dur = predict_lat(M, N, gemm_dur, comm_array, gp, tile_num, comm_op)
        
        if est_dur < min_dur:
            min_dur = est_dur
            cSeg = gp
    if _rank == 0:
        print("Search process finished.")

    dbg(f"fast_search: calling perf_running with cSeg={cSeg}")
    searched_lat = perf_running(M, N, K, BM, BN, Algo, cSeg, hint, comm_op)
    dbg(f"fast_search: perf_running returned {searched_lat}")
    if _rank == 0:
        print("Searched latency: %.4f" % searched_lat)
        print("Best solution: ", cSeg)
    save_solution(M, N, K, BM, BN, gemm_dur, Algo, hint, cSeg)
    if _rank == 0:
        print("Solution saved.")


def main():
    init_dist()

    parser = argparse.ArgumentParser()
    parser.add_argument('--m_dim', type=int, default=4096)
    parser.add_argument('--k_dim', type=int, default=8192)
    parser.add_argument('--n_dim', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    parser.add_argument('--predictive_search', action='store_true')
    args = parser.parse_args()

    if args.predictive_search or args.m_dim * args.n_dim > 33554432:
        comm_array = torch.load(f"../configs/bandwidth_{args.comm_op}_tp{_world_size}.pt")
        if _rank == 0:
            print("Bandwidth curve captured.")
        fast_search(args.m_dim, args.n_dim, args.k_dim, comm_array, args.comm_op)
    else:
        exhaustive_search(args.m_dim, args.n_dim, args.k_dim, args.comm_op)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()