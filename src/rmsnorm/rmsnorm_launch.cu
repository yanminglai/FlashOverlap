// Kernel launch wrappers — compiled with -x musa for <<<>>> syntax.
// No PyTorch headers here to avoid c10::Half / __half conflicts.

#include <musa_fp16.h>
#include <musa_runtime.h>
#include <cstdint>

#include "rmsnorm.cuh"
#include "rmsnorm_launch.h"

void launch_rmsnorm_kernel(void* x, void* rw, void* o, int bs, int dim) {
    rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 16), 1)>>>(
        (half*)x, (half*)rw, (half*)o, bs, dim);
}

void launch_reorder_rmsnorm_kernel(void* x, void* rw, void* o,
    int bs, int dim, int64_t BM, int64_t BN, int64_t rldn, int* RA) {
    int ldn = dim / BN;
    reorder_rmsnorm_kernel<<<dim3(bs), dim3(DIV_UP(dim, 16), 1)>>>(
        (half*)x, (half*)rw, (half*)o, bs, dim, BM, BN, ldn, rldn, RA);
}
