#include "gemm_dispatcher.h"

GemmFuncPtr gemm_func_table[] = {
    &mutlass_gemm<128, 128, 32>,
    &mutlass_gemm<128, 128, 64>,
    &mutlass_gemm<128, 256, 32>,
    &mutlass_gemm<128, 256, 64>,
    &mutlass_gemm<256, 128, 32>,
    &mutlass_gemm<256, 128, 64>,
};
