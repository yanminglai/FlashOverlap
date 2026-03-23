#include "gemm_dispatcher.h"

SignalFuncPtr signal_func_table[] = {
    &mutlass_gemm_signal<128, 128, 32>,
    &mutlass_gemm_signal<128, 128, 64>,
    &mutlass_gemm_signal<128, 256, 32>,
    &mutlass_gemm_signal<128, 256, 64>,
    &mutlass_gemm_signal<256, 128, 32>,
    &mutlass_gemm_signal<256, 128, 64>,
};
