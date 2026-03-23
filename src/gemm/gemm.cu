#include <musa_fp16.h>
#include <iostream>

#include "gemm.h"

#include "mutlass/mutlass.h"
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/kernel/gemm_universal.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"
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
void mutlass_gemm(int M, int N, int K, const half* A, const half* B, half* D, musaStream_t stream) {

    using namespace mutlass;

    // Element types
    using ElementA           = half_t;
    using LayoutA            = layout::RowMajor;
    using ElementB           = half_t;
    using LayoutB            = layout::ColumnMajor;
    using ElementD           = half_t;
    using LayoutD            = layout::RowMajor;
    using ElementAccumulator = float;
    using ElementCompute     = float;

    static constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
    static constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8
    static constexpr int AlignmentD = 16 / sizeof(ElementD);  // 8

    using ArchTag   = arch::Mp22;
    using OpClass   = arch::OpClassTensorOp;
    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

    // Build mainloop collective
    using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
        ArchTag, OpClass,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        Shape<_1, _1, _1>,                              // ClusterShape
        gemm::collective::StageCountAuto,
        gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

    // Build epilogue collective (standard linear combination, no source C)
    using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
        ArchTag, OpClass,
        TileShape,
        Shape<_1, _1, _1>,
        epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        void, LayoutD, AlignmentD,                       // C = void (beta = 0, skip source load)
        ElementD, LayoutD, AlignmentD,
        epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;

    // Assemble kernel
    using GemmKernel = gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue
    >;
    using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;

    // Compute strides
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = make_mute_packed_stride(StrideA{}, make_shape(M, K, 1));
    auto stride_B = make_mute_packed_stride(StrideB{}, make_shape(N, K, 1));
    auto stride_D = make_mute_packed_stride(StrideD{}, make_shape(M, N, 1));

    // Build arguments
    typename Gemm::Arguments arguments{
        gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const ElementA*>(A), stride_A,
         reinterpret_cast<const ElementB*>(B), stride_B},
        {{1.0f, 0.0f},
         nullptr, {},                                    // No source C
         reinterpret_cast<ElementD*>(D), stride_D},
        KernelHardwareInfo{}
    };

    Gemm gemm_op;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    mutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    MUTLASS_CHECK(gemm_op.can_implement(arguments));
    MUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    MUTLASS_CHECK(gemm_op.run(stream));
}

#define MUTLASS_GEMM_INIT(TileM, TileN, TileK)                  \
    template void mutlass_gemm<TileM, TileN, TileK>(            \
        int M, int N, int K, const half* A, const half* B,      \
        half* D, musaStream_t stream)

#include "../inc/gemm_instances.inc"

#undef MUTLASS_GEMM_INIT