#pragma once

#include <mccl.h>
#include <vector>
#include <cublas_v2.h>

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

class BaselineImpl : public torch::CustomClassHolder {
    public:
        BaselineImpl();
        ~BaselineImpl();

        void McclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void CublasInit();

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);
        void Gemm(at::Tensor A, at::Tensor B, at::Tensor C);

        void McclAllReduce(at::Tensor C);
        void McclReduceScatter(at::Tensor C);
        void McclAll2All(at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);
        
    private:
        mcclComm_t comm;
        int64_t my_rank;
        int64_t my_size;

        cublasHandle_t my_handle;
        musaStream_t my_stream;
};