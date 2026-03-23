#include <musa_fp16.h>
#include <iostream>

#include "gemm_scatter.h"
#include "gemm_with_scatter.h"

#include "mutlass/mutlass.h"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/kernel/gemm_universal.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/gemm/dispatch_policy.hpp"
#include "mutlass/epilogue/dispatch_policy.hpp"
#include "mutlass/util/packed_stride.hpp"
#include "mutlass/util/device_memory.h"

using namespace mute;

#define MUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    mutlass::Status error = status;                                                              \
    if (error != mutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got mutlass error: " << mutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }

template <int TileM, int TileN, int TileK>
void mutlass_gemm_scatter(int M, int N, int K, int ReLDN, int* CommThr,
                          half* A, half* B, half* D, int* MM, int* RA,
                          int* RE, bool Monitor, musaStream_t stream) {
    using namespace mutlass;

    using ElementA           = half_t;
    using LayoutA            = layout::RowMajor;
    using ElementB           = half_t;
    using LayoutB            = layout::ColumnMajor;
    using ElementD           = half_t;
    using ElementAccumulator = float;

    static constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
    static constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8

    using ArchTag   = arch::Mp31;
    using OpClass   = arch::OpClassTensorOp;
    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

    // Build mainloop collective (same as standard GEMM)
    using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        Shape<_1, _1, _1>,
        gemm::collective::StageCountAuto,
        gemm::KernelTme
    >::CollectiveOp;

    // Custom scatter epilogue
    using StrideC = Stride<int, _1, int>;
    using StrideD = Stride<int, _1, int>;
    using EpiSchedule = epilogue::NoSmem;

    using CollectiveEpilogue = epilogue::collective::ScatterEpilogue<
        StrideC, StrideD, EpiSchedule, TileM, TileN
    >;

    // Assemble kernel
    using GemmKernel = gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;
    using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Compute strides
    using MainloopStrideA = typename Gemm::GemmKernel::StrideA;
    using MainloopStrideB = typename Gemm::GemmKernel::StrideB;

    auto stride_A = make_mute_packed_stride(MainloopStrideA{}, make_shape(M, K, 1));
    auto stride_B = make_mute_packed_stride(MainloopStrideB{}, make_shape(N, K, 1));

    int monitor_columns = N / TileN;

    // Build arguments
    typename Gemm::Arguments arguments{
        gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const ElementA*>(A), stride_A,
         reinterpret_cast<const ElementB*>(B), stride_B},
        {1.0f, 0.0f,
         nullptr, StrideC{},                              // No source C
         reinterpret_cast<ElementD*>(D), StrideD{},
         MM,                                              // ptr_monitor_matrix
         RA,                                              // ptr_reorder_array
         monitor_columns,                                 // monitor_columns
         ReLDN,                                           // reorder_columns
         CommThr,                                         // ptr_comm_seg_array
         RE,                                              // ptr_row_array
         Monitor},                                        // if_monitor
        KernelHardwareInfo{}
    };

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    mutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    MUTLASS_CHECK(gemm_op.can_implement(arguments));
    MUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    MUTLASS_CHECK(gemm_op.run(stream));
}

#define MUTLASS_SCATTER_INIT(TileM, TileN, TileK)                           \
    template void mutlass_gemm_scatter<TileM, TileN, TileK>(                \
        int M, int N, int K, int ReLDN, int* CommThr,                       \
        half* A, half* B, half* D, int* MM, int* RA,                        \
        int* RE, bool Monitor, musaStream_t stream)

#include "../inc/scatter_instances.inc"

#undef MUTLASS_SCATTER_INIT