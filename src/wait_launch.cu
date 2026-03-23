// Kernel launch wrapper — compiled with -x musa for <<<>>> syntax.
#include "wait.cuh"
#include "wait_launch.h"

void launch_kernel_wait_flag(int that, int* addr, musaStream_t stream) {
    kernel_wait_flag<<<1, 1, 0, stream>>>(that, addr);
}
