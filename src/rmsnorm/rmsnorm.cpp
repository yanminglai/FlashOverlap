// PyTorch wrapper for RMSNorm kernels.
// Compiled as plain CXX (no -x musa) to avoid c10::Half / __half conflicts.

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <torch/extension.h>

#include "rmsnorm_launch.h"

void reorder_rmsnorm(at::Tensor X, at::Tensor RX, at::Tensor RW, 
    int64_t BM, int64_t BN, int64_t rldn, at::Tensor RA){

  int bs = X.size(0);
  int dim = X.size(1);

  launch_reorder_rmsnorm_kernel(
        X.data_ptr<at::Half>(),
        RW.data_ptr<at::Half>(),
        RX.data_ptr<at::Half>(),
        bs, dim, BM, BN, rldn, 
        RA.data_ptr<int>()
    );
}

void rmsnorm(at::Tensor X, at::Tensor O, at::Tensor RW){

  int bs = X.size(0);
  int dim = X.size(1);

  launch_rmsnorm_kernel(
        X.data_ptr<at::Half>(),
        RW.data_ptr<at::Half>(),
        O.data_ptr<at::Half>(),
        bs, dim
    );
}