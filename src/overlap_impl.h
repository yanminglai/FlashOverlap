#pragma once

#include <mccl.h>
#include <vector>

#include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

class OverlapImpl : public torch::CustomClassHolder {
    public:
        OverlapImpl();
        ~OverlapImpl();

        void MutlassInit();
        void McclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id);
        void OverlapInit();

        void Gemm(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);

        void GemmAllReduceOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);
        void GemmReduceScatterOverlap(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t Algo, bool if_monitor);

        void GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo);
        void GemmReduceScatter(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo);
        void GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo, at::Tensor mLen_CPU);

        void SegAllReduce(at::Tensor C, at::Tensor cSEG_CPU, int64_t SegNum);
        void McclAllReduce(at::Tensor C);
        void McclReduceScatter(at::Tensor C);
        void McclAll2All(at::Tensor C, at::Tensor D, at::Tensor mLen_CPU);

    private:
        musaStream_t gemm_stream;
        musaStream_t comm_stream;
        musaEvent_t gemm_finished;

        mcclComm_t comm;
        int64_t my_rank;
        int64_t my_size;
};