import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import os
import json
from RMSNorm import RMSNorm, ReorderRMSNorm
from RowParallelLinear import RowParallelLayer, OverlapRowParallelLayer

torch.ops.load_library("../build/lib/libst_pybinding.so")

def create_tp_group(world_size, rank, tp_size): 
    group_id = rank // tp_size  
    start_rank = group_id * tp_size
    end_rank = (group_id + 1) * tp_size
    ranks = list(range(start_rank, end_rank))
    tp_group = dist.new_group(ranks=ranks)
    return tp_group

def per_gpu_process(rank, world_size, mccl_id, M, N, K, config):
    torch.musa.set_device(rank)

    A = torch.ones((M, K), dtype=torch.float16, device="musa")
    B = torch.ones((N, K), dtype=torch.float16, device="musa")
    W = torch.ones((N), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    
    rmsnorm_layer = RMSNorm(N)
    rmsnorm_layer.weight = nn.Parameter(W)
    reorder_rmsnorm_layer = ReorderRMSNorm(
        N, M, config["BM"], config["BN"], config["hint"])
    reorder_rmsnorm_layer.weight = nn.Parameter(W)
    
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "23456"
    dist.init_process_group(backend="mccl", init_method="env://")
    tp_group = create_tp_group(world_size, rank, world_size)
    linear_layer = RowParallelLayer(K, N, "all_reduce", tp_group)
    linear_layer.weight = nn.Parameter(B)
    overlap_linear_layer = OverlapRowParallelLayer(
        rank, world_size, K, N, M, config, "all_reduce", mccl_id)
    overlap_linear_layer.weight = nn.Parameter(B)

    torch.musa.synchronize()
    x1 = linear_layer(A)
    x2 = overlap_linear_layer(A)

    y1 = rmsnorm_layer(x1)
    y2 = reorder_rmsnorm_layer(x2)

    all_close = torch.allclose(y1, y2, atol=1e-2, rtol=1e-2)
    torch.musa.synchronize()
    dist.destroy_process_group()
    
    print("[GEMM+AllReduce+RMSNorm] all close : ", all_close)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--k', type=int, default=8192)
    parser.add_argument('--n', type=int, default=4096)
    args = parser.parse_args()

    world_size = torch.musa.device_count()
    if world_size < 2:
        raise RuntimeError("At least 2 GPUs are required for this program.")

    # Use the custom MCCL initialization wrapper to get a unique MCCL ID
    # mccl_id = NcclInit()
    mccl_id = torch.ops.flashoverlap_op.generate_mccl_id()
    torch.musa.synchronize()

    print(f"MCCL ID generated: {mccl_id[0]}")

    device = torch.musa.current_device()
    props = torch.musa.get_device_properties(device)
    gpu_name = props.name[7:11].lower()
    config_file = f'../configs/m{args.m}n{args.n}k{args.k}_{gpu_name}.json'

    with open(config_file, 'r') as file:
        config = json.load(file)

    # Spawn processes
    mp.spawn(
            per_gpu_process,
            args=(world_size, mccl_id, args.m, args.n, args.k, config),
            nprocs=world_size
        )

if __name__ == "__main__":
    main()