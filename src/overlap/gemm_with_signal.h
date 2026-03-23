/***************************************************************************************************
 * Adapted from NVIDIA example for MUTlass 3.x
 * Original by hkeee. MUTlass port for FlashOverlap.
 *
 * Custom epilogue that writes output to reordered tile locations and
 * atomically signals the monitor matrix for communication-computation overlap.
 **************************************************************************************************/

#pragma once

#include <cmath>
#include <iostream>

#include "mutlass/mutlass.h"
#include "mutlass/gemm/gemm.h"
#include "mutlass/epilogue/collective/detail.hpp"

#include "mute/tensor.hpp"
#include "mute/numeric/numeric_types.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Custom epilogue for FlashOverlap signal injection.
/// Writes output to reordered tile locations and atomically updates the
/// monitor matrix so the communication stream can detect tile completion.
template <
  class StrideC_,
  class StrideD_,
  class EpilogueSchedule_,
  int TileM_,
  int TileN_
>
class SignalEpilogue {
public:
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;

  using ElementAccumulator = float;
  using ElementCompute     = float;
  using ElementOutput      = mutlass::half_t;
  using ElementC           = mutlass::half_t;
  using ElementD           = mutlass::half_t;
  using StrideC = StrideC_;
  using StrideD = StrideD_;

  static constexpr int kTileM = TileM_;
  static constexpr int kTileN = TileN_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static constexpr int NumBarriers = 0;

  struct SharedStorage {};

  struct Arguments {
    float alpha = 1.0f;
    float beta  = 0.0f;
    ElementC const* ptr_C = nullptr;
    StrideC dC{};
    ElementD* ptr_D = nullptr;
    StrideD dD{};
    // Signal-specific parameters
    int* ptr_monitor_matrix  = nullptr;
    int* ptr_reorder_array   = nullptr;
    int  monitor_columns     = 0;   // N / TileN (original tile grid columns)
    int  reorder_columns     = 0;   // rLDN (columns in reordered layout)
    int* ptr_comm_seg_array  = nullptr;
    bool if_monitor          = false;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size([[maybe_unused]] ProblemShape const&, [[maybe_unused]] Arguments const&) {
    return 0;
  }

  template <class ProblemShape>
  static mutlass::Status
  initialize_workspace([[maybe_unused]] ProblemShape const&, [[maybe_unused]] Arguments const&,
                       [[maybe_unused]] void*, [[maybe_unused]] musaStream_t) {
    return mutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  MUTLASS_HOST_DEVICE static bool
  can_implement([[maybe_unused]] ProblemShape const&, [[maybe_unused]] Arguments const&) {
    return true;
  }

  MUTLASS_HOST_DEVICE
  SignalEpilogue(Params const& params_, [[maybe_unused]] SharedStorage const& = SharedStorage())
      : params(params_) {}

  MUTLASS_DEVICE
  bool is_source_needed() { return false; }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  MUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      mute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char* smem_buf)
  {
    using namespace mute;
    using X = Underscore;

    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    // --- Compute reordered destination tile ---
    int tile_idx = m_coord * params.monitor_columns + n_coord;
    int reordered_tile_idx = params.ptr_reorder_array[tile_idx];
    int new_m_tile = reordered_tile_idx / params.reorder_columns;
    int new_n_tile = reordered_tile_idx % params.reorder_columns;

    // The output D is logically reshaped:
    //   reshaped_M = M * N / (reorder_columns * TileN)
    //   reshaped_N = reorder_columns * TileN
    int reshaped_N = params.reorder_columns * kTileN;

    // Create the full reshaped output tensor
    int reshaped_M = (M * N) / reshaped_N;
    auto stride_d = make_stride(reshaped_N, Int<1>{}, reshaped_M * reshaped_N);
    Tensor mD_mnl = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(reshaped_M, reshaped_N, L), stride_d);
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(X{},X{},X{}), Step<_1,_1,X>{});

    // Slice at the reordered tile coordinates
    Tensor gD = gD_mnl(_,_,new_m_tile,new_n_tile,l_coord);

    // Partition for this thread
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);

    // Identity tensor for predication
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    // Write output: D = alpha * accumulator
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accumulators); ++i) {
      if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
        tCgD(i) = ElementD(params.alpha * float(accumulators(i)));
      }
    }

    // --- Signal: atomically update monitor matrix ---
    if (threadIdx.x == 0) {
      int reordered_tile_2d = new_m_tile * params.reorder_columns + new_n_tile;

      // Find which communication segment this tile belongs to
      int idx_bound = params.ptr_comm_seg_array[0];
      int iter_idx = 0;
      while (idx_bound <= reordered_tile_2d) {
        iter_idx += 1;
        idx_bound += params.ptr_comm_seg_array[iter_idx];
      }

      atomicAdd(&params.ptr_monitor_matrix[iter_idx], 1);

      if (params.if_monitor) {
        int global_order = atomicAdd(&params.ptr_monitor_matrix[params.monitor_columns - 1], 1);
        params.ptr_monitor_matrix[params.monitor_columns + reordered_tile_2d] = global_order;
      }
    }
  }

private:
  Params params;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace collective
} // namespace epilogue
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////