#include "mccl_utils.h"
#include <musa.h>
#include <musa_runtime.h>
#include <musa_fp16.h>
#include <math.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>

#include "baseline_impl.h"

#define DIV_UP(x, y) (((x) + (y) - 1) / (y))
#define MAX_GROUP_SIZE 16

/// Baseline Implementation: cuBLAS for GEMM and MCCL for AllReduce
BaselineImpl::BaselineImpl(){
    cublasCreate(&this->my_handle);
    this->my_rank = 0;
    this->my_size = 1;
}

BaselineImpl::~BaselineImpl(){
    cublasDestroy(this->my_handle);
    // mcclCommDestroy(this->comm);
}

void BaselineImpl::GemmAllReduce(at::Tensor A, at::Tensor B, at::Tensor C){

    // Check if MCCL is initialized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for MCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch AllReduce after GEMM
    MCCL_CHECK(mcclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), mcclFloat16, mcclSum, this->comm, this->my_stream));
}

void BaselineImpl::GemmReduceScatter(
        at::Tensor A, // [M, K]
        at::Tensor B, // [N, K]
        at::Tensor C, // [M, N]
        at::Tensor D  // [M / world_size, N]
        ){

    // Check if MCCL is initialized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for MCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch ReduceScatter after GEMM
    size_t recvcount = (M * N) / this->my_size;
    MCCL_CHECK(mcclReduceScatter((void *)c_ptr, (void *)d_ptr, recvcount, 
        mcclFloat16, mcclSum, this->comm, this->my_stream));
}

void BaselineImpl::GemmAll2All(at::Tensor A, at::Tensor B, at::Tensor C,
    at::Tensor D, at::Tensor mLen_CPU){

    // Check if MCCL is initialized
    if (this->comm == nullptr) {
        return;
    }

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    assert(mLen_CPU.size(0) == this->my_size);
    assert(mLen_CPU.size(1) == this->my_size);

    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    // prepare for MCCL
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    // Launch GEMM
    cublasGemmEx(this->my_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half),
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // Launch All2All after GEMM
    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    MCCL_CHECK(mcclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        MCCL_CHECK(mcclSend((void *)(c_ptr + src_acc_addr), sendcount, mcclFloat16, i, this->comm, this->my_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        MCCL_CHECK(mcclRecv((void *)(d_ptr + dst_acc_addr), recvcount, mcclFloat16, i, this->comm, this->my_stream));
        dst_acc_addr += recvcount;
    }
    MCCL_CHECK(mcclGroupEnd());
}

void BaselineImpl::McclInit(const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id)
{
    this->my_rank = tp_rank;
    this->my_size = tp_size;
    
    mcclUniqueId tp_uid;
    memcpy(tp_uid.internal, &tp_id[0], MCCL_UNIQUE_ID_BYTES);

    if (this->my_size == 1) {
        this->comm = nullptr;
        return;
    }
    MCCL_CHECK(mcclCommInitRank(&this->comm, this->my_size, tp_uid, this->my_rank));
}

void BaselineImpl::CublasInit(){

    // prepare for GEMM
    this->my_stream = at::musa::getCurrentMUSAStream().stream();
    cublasSetStream(this->my_handle, this->my_stream);
    cublasSetMathMode(this->my_handle, CUBLAS_TENSOR_OP_MATH);
}

void BaselineImpl::Gemm(at::Tensor A, at::Tensor B, at::Tensor C){

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    cublasHandle_t cublasH = at::cuda::getCurrentCUDABlasHandle();
    cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH);

    float alpha = 1.0f;
    float beta = 0.0f;
    half alpha_half = __float2half(alpha);
    half beta_half = __float2half(beta);

    cublasGemmEx(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N, M, K,
                    (const void*)reinterpret_cast<half *>(&alpha_half), 
                    (const void*)reinterpret_cast<half *>(B.data_ptr<at::Half>()),
                    CUDA_R_16F, K, 
                    (const void*)reinterpret_cast<half *>(A.data_ptr<at::Half>()), 
                    CUDA_R_16F, K,  
                    (const void*)reinterpret_cast<half *>(&beta_half),
                    (void*)reinterpret_cast<half *>(C.data_ptr<at::Half>()), 
                    CUDA_R_16F, N,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void BaselineImpl::McclAllReduce(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    mcclAllReduce((void *)c_ptr, (void *)c_ptr, (M * N), mcclFloat16, mcclSum, this->comm, this->my_stream);
}

void BaselineImpl::McclReduceScatter(at::Tensor C){

    int M = C.size(0);
    int N = C.size(1);

    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());

    size_t recvcount = (M * N) / this->my_size;
    MCCL_CHECK(mcclReduceScatter((void *)c_ptr, (void *)(c_ptr + this->my_rank * recvcount), recvcount, 
        mcclFloat16, mcclSum, this->comm, this->my_stream));
}

void BaselineImpl::McclAll2All(at::Tensor C, 
    at::Tensor D, // [world_size - 1, M, N]
    at::Tensor mLen_CPU // [world_size, world_size]
    ){
    
    int M = C.size(0);
    int N = C.size(1);

    assert(mLen_CPU.size(0) == this->my_size);
    assert(mLen_CPU.size(1) == this->my_size);

    int* mlen_cpu_ptr = mLen_CPU.data_ptr<int>();
    half* c_ptr = reinterpret_cast<half *>(C.data_ptr<at::Half>());
    half* d_ptr = reinterpret_cast<half *>(D.data_ptr<at::Half>());

    // First SEND
    int src_acc_addr = 0;
    // Then RECV
    int dst_acc_addr = 0;
    MCCL_CHECK(mcclGroupStart());
    for (int i = 0; i < this->my_size; i++){
        if (i == this->my_rank){continue;}
        size_t sendcount = mlen_cpu_ptr[this->my_rank * this->my_size + i] * N;
        MCCL_CHECK(mcclSend((void *)(c_ptr + src_acc_addr), sendcount, mcclFloat16, i, this->comm, this->my_stream));
        src_acc_addr += sendcount;

        size_t recvcount = mlen_cpu_ptr[i * this->my_size + this->my_rank] * N;
        MCCL_CHECK(mcclRecv((void *)(d_ptr + dst_acc_addr), recvcount, mcclFloat16, i, this->comm, this->my_stream));
        dst_acc_addr += recvcount;
    }
    MCCL_CHECK(mcclGroupEnd());
}
