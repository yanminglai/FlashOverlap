"""
Correctness test for GEMM+ReduceScatter+RMSNorm using torchrun.

Usage:
    MUSA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 correctness_rs.py \
        --m_dim 4096 --n_dim 4096 --k_dim 8192
"""

import sys
import torch
import torch_musa
import torch.nn as nn
import torch.distributed as dist
import argparse
import os
import json
from RMSNorm import RMSNorm, ReorderRMSNorm
from RowParallelLinear import RowParallelLayer, OverlapRowParallelLayer
from utils import reorder_rows_by_world_size

def dbg(msg):
    if not os.getenv('FLASH_DEBUG'):
        return
    rank = os.getenv('LOCAL_RANK', '?')
    print(f"[rank={rank}] {msg}", flush=True, file=sys.stderr)

_script_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_script_dir, "../build/lib/libst_pybinding.so"))

# Global distributed state
_rank = 0
_local_rank = 0
_world_size = 1


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
    dbg(f"init_dist done: rank={rank}, local_rank={local_rank}, world_size={world_size}")


def _get_mccl_id():
    device = torch.device(f'musa:{_local_rank}')
    if _rank == 0:
        mccl_id = torch.ops.flashoverlap_op.generate_mccl_id()
        mccl_id_tsr = torch.tensor(mccl_id, device=device)
    else:
        mccl_id_tsr = torch.zeros(16, dtype=torch.int64, device=device)
    dist.broadcast(mccl_id_tsr, src=0)
    return mccl_id_tsr.cpu().tolist()


def create_tp_group(world_size, rank, tp_size):
    group_id = rank // tp_size
    start_rank = group_id * tp_size
    end_rank = (group_id + 1) * tp_size
    ranks = list(range(start_rank, end_rank))
    tp_group = dist.new_group(ranks=ranks)
    return tp_group


def main():
    init_dist()
    rank, world_size = _rank, _world_size

    parser = argparse.ArgumentParser()
    parser.add_argument('--m_dim', type=int, default=4096)
    parser.add_argument('--k_dim', type=int, default=8192)
    parser.add_argument('--n_dim', type=int, default=4096)
    args = parser.parse_args()

    M, N, K = args.m_dim, args.n_dim, args.k_dim

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    config_file = os.path.join(_script_dir, f'../configs/m{M}n{N}k{K}_{gpu_name}.json')

    with open(config_file, 'r') as file:
        config = json.load(file)

    mccl_id = _get_mccl_id()

    A0 = torch.ones((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    A1 = reorder_rows_by_world_size(A0, world_size)
    B = torch.ones((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    W = torch.ones((N), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)

    rmsnorm_layer = RMSNorm(N)
    rmsnorm_layer.weight = nn.Parameter(W)
    reorder_rmsnorm_layer = ReorderRMSNorm(
        N, M // world_size, config["BM"] // world_size, config["BN"], config["hint"])
    reorder_rmsnorm_layer.weight = nn.Parameter(W)

    tp_group = create_tp_group(world_size, rank, world_size)
    linear_layer = RowParallelLayer(K, N, "reduce_scatter", tp_group)
    linear_layer.weight = nn.Parameter(B)
    overlap_linear_layer = OverlapRowParallelLayer(
        rank, world_size, K, N, M, config, "reduce_scatter", mccl_id)
    overlap_linear_layer.weight = nn.Parameter(B)

    torch.musa.synchronize()
    x1 = linear_layer(A1)
    x2 = overlap_linear_layer(A0)

    y1 = rmsnorm_layer(x1)
    y2 = reorder_rmsnorm_layer(x2)

    diff = torch.abs(y1 - y2)

    max_idx_flat = torch.argmax(diff)
    max_idx_unravel = torch.unravel_index(max_idx_flat, diff.shape)
    val1 = y1[max_idx_unravel]
    val2 = y2[max_idx_unravel]

    if rank == 0:
        print("Index of max absolute difference:", max_idx_unravel)
        print("Value in y1 at that index:", val1.item())
        print("Value in y2 at that index:", val2.item())
        print("Absolute difference:", diff[max_idx_unravel].item())

    all_close = torch.allclose(y1, y2, atol=5e-2, rtol=5e-2)
    torch.musa.synchronize()

    if rank == 0:
        print("[GEMM+ReduceScatter+RMSNorm] all close : ", all_close)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()