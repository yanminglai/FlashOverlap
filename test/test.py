"""
Speed test using torchrun distributed launch.

Usage:
    MUSA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test.py \
        --m_dim 4096 --n_dim 4096 --k_dim 4096 --comm_op all_reduce
"""

import sys
import torch
import torch_musa
import json
import argparse
import os
import torch.distributed as dist

def dbg(msg):
    rank = os.getenv('LOCAL_RANK', '?')
    print(f"[rank={rank}] {msg}", flush=True, file=sys.stderr)

_script_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_script_dir, "../build/lib/libst_pybinding.so"))

WARM_UP = 20
REP = 200

# Global distributed state
_rank = 0
_local_rank = 0
_world_size = 1


def init_dist():
    """Initialize distributed using torchrun env vars."""
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
    dbg(f"init_dist done: rank={rank}, local_rank={local_rank}, world_size={world_size}")


def _get_mccl_id():
    """Generate MCCL ID on rank 0 and broadcast to all ranks."""
    device = torch.device(f'musa:{_local_rank}')
    if _rank == 0:
        mccl_id = torch.ops.flashoverlap_op.generate_mccl_id()
        mccl_id_tsr = torch.tensor(mccl_id, device=device)
    else:
        mccl_id_tsr = torch.zeros(16, dtype=torch.int64, device=device)
    dist.broadcast(mccl_id_tsr, src=0)
    return mccl_id_tsr.cpu().tolist()

def div_up(x: int, y: int):
    return (x + y - 1) // y

def reorder_indices(S, hint):
    # Generate the original array of indices [0, 1, ..., S-1]
    original = list(range(S))
    
    # Create an empty list to store the new order of indices
    new_order = [-1] * S
    
    # Place the indices of the hint list in the first positions of the new order
    for i, element in enumerate(hint):
        new_order[element] = i
    
    # Place the remaining indices in the new order
    remaining_elements = [x for x in original if x not in hint]
    for i, element in enumerate(remaining_elements, start=len(hint)):
        new_order[element] = i
    
    return torch.tensor(new_order, dtype=torch.int, device="musa")

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
        
        # Compute row_id % world_size for the current chunk
        mod_values = chunk_row_ids % world_size
        
        # Sort the chunk based on mod_values (stable sort)
        _, sorted_indices = torch.sort(mod_values, stable=True)
        reordered_chunk = chunk_row_ids[sorted_indices]
        
        reordered_row_id[current_row : current_row + chunk_size] = reordered_chunk
        current_row += chunk_size
    
    # Compute remap: remap[original_row_id] = new_row_id
    remap = torch.empty_like(original_row_ids)
    remap[reordered_row_id] = torch.arange(len(reordered_row_id), dtype=torch.int, device=device)
    
    return remap

def perf_running(M: int, N: int, K: int,
    BM: int, BN: int, Algo: int, cSeg: list, hint: list,
    comm_op: str):
    """Run overlap perf benchmark on current rank, return max duration across ranks."""
    rank, world_size = _rank, _world_size

    cSeg_CPU = torch.tensor(cSeg, dtype=torch.int32)
    cSeg_GPU = cSeg_CPU.musa(rank)

    TileNum = div_up(M, BM) * div_up(N, BN)

    mccl_id = _get_mccl_id()
    gemm_class = torch.classes.flashoverlap_class.OverlapImpl()
    gemm_class.mccl_init(rank, world_size, mccl_id)
    gemm_class.mutlass_init()
    gemm_class.overlap_init()

    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    MonitoredMatrix = torch.zeros(((N+BN-1)//BN), dtype=torch.int, device="musa")
    ReorderedArray = reorder_indices(TileNum, hint).reshape(((M+BM-1)//BM, (N+BN-1)//BN))

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="musa")
        RowArray = generate_row_remap_array(M, N, BM, BN, cSeg, world_size)

    _warm_up = WARM_UP
    _freq = REP

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

    # Take max across all ranks
    local_dur = torch.tensor([torch.mean(dur).item()], device="musa")
    dist.all_reduce(local_dur, op=dist.ReduceOp.MAX)
    return local_dur.item()

def perf_comm(M: int, N: int, comm_type: str):
    """Benchmark communication on current rank, return rank-0 duration."""
    rank, world_size = _rank, _world_size

    mccl_id = _get_mccl_id()
    comm_class = torch.classes.flashoverlap_class.OverlapImpl()
    comm_class.mccl_init(rank, world_size, mccl_id)
    comm_class.mutlass_init()

    C = torch.empty((M, N), dtype=torch.float16, device="musa")
    if comm_type == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="musa")

    if comm_type == "all_reduce":
        for _ in range(WARM_UP):
            comm_class.mccl_allreduce(C)
        start_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.mccl_allreduce(C)
            end_event[i].record()
        torch.musa.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_type == "reduce_scatter":
        for _ in range(WARM_UP):
            comm_class.mccl_reducescatter(C)
        start_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            comm_class.mccl_reducescatter(C)
            end_event[i].record()
        torch.musa.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))

    # Broadcast rank-0's result
    local_dur = torch.tensor([torch.mean(dur).item()], device="musa")
    dist.broadcast(local_dur, src=0)
    return local_dur.item()

def perf_baseline(M: int, N: int, K: int, comm_op: str):
    """Run baseline (no overlap) perf benchmark on current rank, return max duration."""
    rank, world_size = _rank, _world_size

    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    if comm_op == "reduce_scatter":
        D = torch.empty((M // world_size, N), dtype=torch.float16, device="musa")

    mccl_id = _get_mccl_id()
    gemm_comm = torch.classes.flashoverlap_class.BaselineImpl()
    gemm_comm.mccl_init(rank, world_size, mccl_id)
    gemm_comm.mublas_init()

    if comm_op == "all_reduce":
        for _ in range(WARM_UP):
            gemm_comm.gemm_allreduce(A, B, C)
        start_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            gemm_comm.gemm_allreduce(A, B, C)
            end_event[i].record()
        torch.musa.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    elif comm_op == "reduce_scatter":
        for _ in range(WARM_UP):
            gemm_comm.gemm_reducescatter(A, B, C, D)
        start_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        end_event = [torch.musa.Event(enable_timing=True) for i in range(REP)]
        for i in range(REP):
            start_event[i].record()
            gemm_comm.gemm_reducescatter(A, B, C, D)
            end_event[i].record()
        torch.musa.synchronize()
        dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    else:
        dur = torch.zeros((REP))

    # Take max across all ranks
    local_dur = torch.tensor([torch.mean(dur).item()], device="musa")
    dist.all_reduce(local_dur, op=dist.ReduceOp.MAX)
    return local_dur.item()

def main():
    init_dist()

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    sm_count = props.multi_processor_count
    wave_size = sm_count - 2

    parser = argparse.ArgumentParser()
    parser.add_argument('--m_dim', type=int, default=4096)
    parser.add_argument('--k_dim', type=int, default=8192)
    parser.add_argument('--n_dim', type=int, default=8192)
    parser.add_argument('--comm_op', type=str, default='all_reduce')
    args = parser.parse_args()

    comm_op = args.comm_op

    m, n, k = args.m_dim, args.n_dim, args.k_dim

    file_path = os.path.join(_script_dir, f'../configs/m{m}n{n}k{k}_{gpu_name}.json')

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tile_num = m // data["BM"] * n // data["BN"]
    wave_num = (tile_num + wave_size - 1) // wave_size

    gemm_dur = data["dur"]

    dbg("Starting comm benchmark...")
    comm_dur = perf_comm(m, n, comm_op)
    dbg("Starting overlap benchmark...")
    overlap_dur = perf_running(m, n, k,
        data["BM"], data["BN"], data["Algo"], data["cSeg"], data["hint"], comm_op)
    dbg("Starting baseline benchmark...")
    baseline_dur = perf_baseline(m, n, k, comm_op)

    speedup = baseline_dur / overlap_dur

    if _rank == 0:
        print(f"""
        {'Item':<20} {'Value':>15}
        {'-----':<20} {'-----':>15}
        {'m':<20} {m:>15}
        {'n':<20} {n:>15}
        {'k':<20} {k:>15}
        {'tile_num':<20} {tile_num:>15}
        {'gemm_dur (ms)':<20} {gemm_dur:>15.4f}
        {'comm_dur (ms)':<20} {comm_dur:>15.4f}
        {'baseline_dur (ms)':<20} {baseline_dur:>15.4f}
        {'overlap_dur (ms)':<20} {overlap_dur:>15.4f}
        {'speedup':<20} {speedup:>15.4f}
        """)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()




