#pragma once

#include <musa_fp16.h>
#include <musa_runtime.h>

template <int TileM, int TileN, int TileK>
void mutlass_gemm_scatter(int M, int N, int K, int ReLDN, int* CommThr,
    half* A, half* B, half* D, int* MM, int* RA, int* RE, bool Monitor, musaStream_t stream);