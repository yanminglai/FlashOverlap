"""
Sweep M/N/K to compare MUTlass vs muBLAS GEMM performance.

Usage:
    MUSA_VISIBLE_DEVICES=0 python sweep_gemm.py
    MUSA_VISIBLE_DEVICES=0 python sweep_gemm.py --k_dim 8192
    MUSA_VISIBLE_DEVICES=0 python sweep_gemm.py --m_list 1024 2048 4096 8192 --n_list 1024 2048 4096 8192
"""

import sys
import os
import torch
import torch_musa
import argparse

_script_dir = os.path.dirname(os.path.abspath(__file__))
torch.ops.load_library(os.path.join(_script_dir, "../build/lib/libst_pybinding.so"))

WARM_UP = 20
REP = 200
NUM_ALGO = 6  # Algo 0..5


def bench_mublas(M, N, K):
    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    g = torch.classes.flashoverlap_class.BaselineImpl()
    g.mublas_init()

    for _ in range(WARM_UP):
        g.mublas_gemm(A, B, C)
    starts = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
    ends = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
    for i in range(REP):
        starts[i].record()
        g.mublas_gemm(A, B, C)
        ends[i].record()
    torch.musa.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return sum(times) / len(times)


def bench_mutlass(M, N, K, algo):
    A = torch.empty((M, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    B = torch.empty((N, K), dtype=torch.float16, device="musa").normal_(mean=0., std=0.5)
    C = torch.empty((M, N), dtype=torch.float16, device="musa")

    g = torch.classes.flashoverlap_class.OverlapImpl()
    g.mutlass_init()

    try:
        for _ in range(WARM_UP):
            g.mutlass_gemm(A, B, C, algo)
        starts = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
        ends = [torch.musa.Event(enable_timing=True) for _ in range(REP)]
        for i in range(REP):
            starts[i].record()
            g.mutlass_gemm(A, B, C, algo)
            ends[i].record()
        torch.musa.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
        return sum(times) / len(times)
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m_list', type=int, nargs='+',
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument('--n_list', type=int, nargs='+',
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument('--k_dim', type=int, default=None,
                        help='Fixed K; if not set, K sweeps same as M list')
    parser.add_argument('--k_list', type=int, nargs='+', default=None,
                        help='Explicit K list to sweep')
    args = parser.parse_args()

    m_list = args.m_list
    n_list = args.n_list
    if args.k_list:
        k_list = args.k_list
    elif args.k_dim:
        k_list = [args.k_dim]
    else:
        k_list = m_list

    torch.musa.set_device(0)

    # Header
    print(f"{'M':>6} {'N':>6} {'K':>6} {'Algo':>4}  "
          f"{'muBLAS(ms)':>10} {'MUTlass(ms)':>11} {'ratio':>6}  {'note':>10}")
    print("-" * 78)

    results = []

    for K in k_list:
        for M in m_list:
            for N in n_list:
                # muBLAS once per (M,N,K)
                mublas_t = bench_mublas(M, N, K)

                best_algo = -1
                best_mutlass = float('inf')

                for algo in range(NUM_ALGO):
                    mt = bench_mutlass(M, N, K, algo)
                    if mt is not None and mt < best_mutlass:
                        best_mutlass = mt
                        best_algo = algo

                if best_mutlass == float('inf'):
                    print(f"{M:>6} {N:>6} {K:>6} {'FAIL':>4}  "
                          f"{mublas_t:>10.4f} {'N/A':>11} {'N/A':>6}  {'':>10}")
                    continue

                ratio = best_mutlass / mublas_t
                note = ""
                if ratio <= 1.1:
                    note = "<== CLOSE"
                elif ratio <= 1.3:
                    note = "<= OK"

                print(f"{M:>6} {N:>6} {K:>6} {best_algo:>4}  "
                      f"{mublas_t:>10.4f} {best_mutlass:>11.4f} {ratio:>6.2f}  {note:>10}")

                results.append((M, N, K, best_algo, mublas_t, best_mutlass, ratio))

    # Summary: top 10 closest
    if results:
        results.sort(key=lambda x: x[6])
        print("\n--- Top 10 closest (MUTlass/muBLAS ratio) ---")
        print(f"{'M':>6} {'N':>6} {'K':>6} {'Algo':>4}  "
              f"{'muBLAS(ms)':>10} {'MUTlass(ms)':>11} {'ratio':>6}")
        for r in results[:10]:
            M, N, K, algo, mb, mt, ratio = r
            print(f"{M:>6} {N:>6} {K:>6} {algo:>4}  "
                  f"{mb:>10.4f} {mt:>11.4f} {ratio:>6.2f}")


if __name__ == "__main__":
    main()
