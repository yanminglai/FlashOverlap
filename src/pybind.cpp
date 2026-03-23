#include "baseline_impl.h"
#include "overlap_impl.h"
#include "mccl_utils.h"
#include "rmsnorm/rmsnorm.h"

#include <torch/script.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

template<typename T>
void McclInitWrapper(const c10::intrusive_ptr<T>& self, 
    const int64_t tp_rank, const int64_t tp_size, const std::vector<int64_t> tp_id){
    self->McclInit(tp_rank, tp_size, tp_id);
}

template<typename T>
void MublasInitWrapper(const c10::intrusive_ptr<T>& self){
    self->MublasInit();
}

template<typename T>
void MutlassGemmAllReduceWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){
    self->GemmAllReduce(A, B, C, Algo);
}

template<typename T>
void MutlassGemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo){
    self->GemmReduceScatter(A, B, C, D, Algo);
}

template<typename T>
void MutlassAll2AllWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, int64_t Algo, at::Tensor mLen_CPU){
    self->GemmAll2All(A, B, C, D, Algo, mLen_CPU);
}

template<typename T>
void GemmAllReduceWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C){
    self->GemmAllReduce(A, B, C);
}

template<typename T>
void GemmReduceScatterWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D){
    self->GemmReduceScatter(A, B, C, D);
}

template<typename T>
void GemmAll2AllWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU){
    self->GemmAll2All(A, B, C, D, mLen_CPU);
}

template<typename T>
void MutlassInitWrapper(const c10::intrusive_ptr<T>& self){
    self->MutlassInit();
}

template<typename T>
void GemmWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C){
    self->Gemm(A, B, C);
}

template<typename T>
void MutlassGemmWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, int64_t Algo){
    self->Gemm(A, B, C, Algo);
}

template<typename T>
void OverlapInitWrapper(const c10::intrusive_ptr<T>& self){
    self->OverlapInit();
}

template<typename T>
void AllReduceOverlapWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor MM, at::Tensor RA, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo, bool if_monitor){
    self->GemmAllReduceOverlap(A, B, C, MM, RA, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo, if_monitor);
}

template<typename T>
void ReduceScatterOverlapWrapper(const c10::intrusive_ptr<T>& self, 
    at::Tensor A, at::Tensor B, at::Tensor C, at::Tensor D, at::Tensor MM, at::Tensor RA, at::Tensor RE, int64_t rLDN, 
    at::Tensor cSEG_CPU, at::Tensor cSEG_GPU, int64_t TilingAlgo, bool if_monitor){
    self->GemmReduceScatterOverlap(A, B, C, D, MM, RA, RE, rLDN, cSEG_CPU, cSEG_GPU, TilingAlgo, if_monitor);
}

template<typename T>
void McclAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C){
    self->McclAllReduce(C);
}

template<typename T>
void SegAllReduceWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C,
    at::Tensor cSEG_CPU, int64_t SegNum){
    self->SegAllReduce(C, cSEG_CPU, SegNum);
}

template<typename T>
void McclReduceScatterWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C){
    self->McclReduceScatter(C);
}

template<typename T>
void McclAll2AllWrapper(const c10::intrusive_ptr<T>& self, at::Tensor C, at::Tensor D, at::Tensor mLen_CPU){
    self->McclAll2All(C, D, mLen_CPU);
}

TORCH_LIBRARY(flashoverlap_class, m) {

    // Class
    m.class_<BaselineImpl>("BaselineImpl")
        .def(torch::init())
        .def("mccl_init", &McclInitWrapper<BaselineImpl>)
        .def("mublas_init", &MublasInitWrapper<BaselineImpl>)
        .def("gemm_allreduce", &GemmAllReduceWrapper<BaselineImpl>)
        .def("gemm_reducescatter", &GemmReduceScatterWrapper<BaselineImpl>)
        .def("gemm_all2all", &GemmAll2AllWrapper<BaselineImpl>)
        .def("mublas_gemm", &GemmWrapper<BaselineImpl>)
        .def("mccl_allreduce", &McclAllReduceWrapper<BaselineImpl>)
        .def("mccl_reducescatter", &McclReduceScatterWrapper<BaselineImpl>)
        .def("mccl_all2all", &McclAll2AllWrapper<BaselineImpl>)
    ;
    m.class_<OverlapImpl>("OverlapImpl")
        .def(torch::init())
        .def("mutlass_init", &MutlassInitWrapper<OverlapImpl>)
        .def("mutlass_gemm", &MutlassGemmWrapper<OverlapImpl>)
        .def("gemm_allreduce_overlap", &AllReduceOverlapWrapper<OverlapImpl>)
        .def("gemm_reducescatter_overlap", &ReduceScatterOverlapWrapper<OverlapImpl>)
        .def("mccl_init", &McclInitWrapper<OverlapImpl>)
        .def("gemm_allreduce", &MutlassGemmAllReduceWrapper<OverlapImpl>)
        .def("gemm_reducescatter", &MutlassGemmReduceScatterWrapper<OverlapImpl>)
        .def("gemm_all2all", &MutlassAll2AllWrapper<OverlapImpl>)
        .def("overlap_init", &OverlapInitWrapper<OverlapImpl>)
        .def("mccl_allreduce", &McclAllReduceWrapper<OverlapImpl>)
        .def("mccl_reducescatter", &McclReduceScatterWrapper<OverlapImpl>)
        .def("mccl_all2all", &McclAll2AllWrapper<OverlapImpl>)
        .def("seg_allreduce", &SegAllReduceWrapper<OverlapImpl>)
    ;
}

TORCH_LIBRARY(flashoverlap_op, m) {
    m.def("generate_mccl_id", &generate_mccl_id);
    m.def("reorder_rmsnorm", &reorder_rmsnorm);
}