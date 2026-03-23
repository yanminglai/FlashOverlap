#pragma once

#include <musa_fp16.h>
#include <musa_runtime.h>

template <int TileM, int TileN, int TileK>
void mutlass_gemm(int M, int N, int K, const half* A, const half* B, half* D, musaStream_t stream);