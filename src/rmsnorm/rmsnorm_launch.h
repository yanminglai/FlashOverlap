#pragma once

#include <cstdint>

void launch_rmsnorm_kernel(void* x, void* rw, void* o, int bs, int dim);

void launch_reorder_rmsnorm_kernel(void* x, void* rw, void* o,
    int bs, int dim, int64_t BM, int64_t BN, int64_t rldn, int* RA);
