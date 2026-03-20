#pragma once

#include "../gemm/gemm.h"
#include "../overlap/gemm_signal.h"
#include "../overlap/gemm_scatter.h"

typedef void (*SignalFuncPtr)(
    int M, int N, int K, 
    int ReLDN, int* CommThr, 
    half* A, half* B, half* D, 
    int* MM, int* RA, bool Monitor, 
    musaStream_t stream
);

typedef void (*GemmFuncPtr)(
    int M, int N, int K, 
    const half* A, const half* B, half* D, 
    musaStream_t stream
);

typedef void (*ScatterFuncPtr)(
    int M, int N, int K, 
    int ReLDN, int* CommThr, 
    half* A, half* B, half* D, 
    int* MM, int* RA, int* RE, bool Monitor, 
    musaStream_t stream
);