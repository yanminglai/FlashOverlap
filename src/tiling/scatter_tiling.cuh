#include "gemm_dispatcher.h"

ScatterFuncPtr scatter_func_table[] = {
    &mutlass_gemm_scatter<128, 128, 32>,
    &mutlass_gemm_scatter<128, 128, 64>,
    &mutlass_gemm_scatter<128, 256, 32>,
    &mutlass_gemm_scatter<128, 256, 64>,
    &mutlass_gemm_scatter<256, 128, 32>,
    &mutlass_gemm_scatter<256, 128, 64>,
};
