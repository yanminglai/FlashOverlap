"""
Bandwidth benchmark using torchrun (DeepEP-style distributed launch).

Usage:
    MUSA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 bandwidth.py --comm_op all_reduce
"""

import sys
import torch
import torch_musa
import argparse
import torch.distributed as dist
import matplotlib.pyplot as plt
import os

def dbg(msg):
    if not os.getenv('FLASH_DEBUG'):
        return
    rank = os.getenv('LOCAL_RANK', '?')
    print(f"[rank={rank}] {msg}", flush=True, file=sys.stderr)

_script_dir = os.path.dirname(os.path.abspath(__file__))
dbg("loading libst_pybinding.so ...")
torch.ops.load_library(os.path.join(_script_dir, "../build/lib/libst_pybinding.so"))
dbg("library loaded")

WARM_UP = 20
REP = 200


def init_dist():
    """Initialize distributed using torchrun env vars directly."""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '29500'))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.getenv('LOCAL_RANK', 0))

    params = {
        'backend': 'mccl',
        'init_method': f'tcp://{ip}:{port}',
        'world_size': world_size,
        'rank': rank,
    }
    dbg(f"init_process_group with params={params}")
    dist.init_process_group(**params)
    dbg("init_process_group done")
    torch.musa.set_device(local_rank)
    dbg(f"set_device({local_rank}) done")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def perf_comm(M: int, N: int, comm_op: str, comm_class):
    """Benchmark a single communication size on the current rank."""
    C = torch.empty((M, N), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)

    if comm_op == "all_reduce":
        comm_func = comm_class.mccl_allreduce
    elif comm_op == "reduce_scatter":
        comm_func = comm_class.mccl_reducescatter
    else:
        raise ValueError(f"Unsupported comm_op: {comm_op}")

    # Warmup
    for _ in range(WARM_UP):
        comm_func(C)

    # Timed runs
    start_events = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
    end_events = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
    for i in range(REP):
        start_events[i].record()
        comm_func(C)
        end_events[i].record()
    torch.musa.synchronize()

    dur = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
    return torch.mean(dur).item()


def main():
    dbg("calling init_dist()")
    rank, local_rank, world_size = init_dist()
    dbg(f"init_dist done: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Generate and broadcast MCCL ID for FlashOverlap comm class
    device = torch.device(f'musa:{local_rank}')
    if rank == 0:
        dbg("generating mccl_id ...")
        mccl_id = torch.ops.flashoverlap_op.generate_mccl_id()
        dbg(f"mccl_id generated: {mccl_id[:3]}...")
        mccl_id_tsr = torch.tensor(mccl_id, device=device)
    else:
        mccl_id_tsr = torch.zeros(16, dtype=torch.int64, device=device)
    dbg("broadcasting mccl_id ...")
    dist.broadcast(mccl_id_tsr, src=0)
    dbg("mccl_id broadcast done")
    mccl_id = mccl_id_tsr.cpu().tolist()

    # Initialize FlashOverlap comm class
    dbg("creating OverlapImpl ...")
    comm_class = torch.classes.flashoverlap_class.OverlapImpl()
    dbg("OverlapImpl created, barrier ...")
    dist.barrier()
    dbg("barrier done, mccl_init ...")
    comm_class.mccl_init(rank, world_size, mccl_id)
    dbg("mccl_init done, mutlass_init ...")
    comm_class.mutlass_init()
    dbg("mutlass_init done")

    parser = argparse.ArgumentParser()
    parser.add_argument('--comm_op', type=str, default="all_reduce",
                        choices=["all_reduce", "reduce_scatter"])
    args = parser.parse_args()

    data_sizes = [(int(2**(20 + 0.25*i)) // 1024 * 1024) for i in range(36)]
    bandwidths = []
    comm_array = torch.zeros((len(data_sizes), 2))

    for i, size in enumerate(data_sizes):
        if rank == 0:
            print(f"Testing size: {size / 2**20:.1f} MB")

        avg_time = perf_comm(1024, size // 1024, args.comm_op, comm_class)

        if rank == 0:
            data_size_bytes = size * 2  # float16 = 2 bytes
            if args.comm_op == "all_reduce":
                total_data_transferred = data_size_bytes * 2 * (world_size - 1)
            else:  # reduce_scatter
                total_data_transferred = data_size_bytes * (world_size - 1)

            bandwidth = (total_data_transferred / avg_time) / (1024 ** 3)
            bandwidths.append(bandwidth)
            comm_array[i, 0] = size
            comm_array[i, 1] = bandwidth

        dist.barrier()

    if rank == 0:
        plt.plot(data_sizes, bandwidths, marker='o')
        plt.xlabel('Data Size (elements)')
        plt.ylabel('Bandwidth (GB/s)')
        plt.title('Bandwidth vs Data Size')
        plt.grid(True)
        plt.savefig('bandwidth.png', dpi=300, bbox_inches='tight')

        os.makedirs(os.path.join(_script_dir, "../configs"), exist_ok=True)
        torch.save(comm_array, os.path.join(_script_dir,
                   f"../configs/bandwidth_{args.comm_op}_tp{world_size}.pt"))
        print("Done. Saved bandwidth.png and configs.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()