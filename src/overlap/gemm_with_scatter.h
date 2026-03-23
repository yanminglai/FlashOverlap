/***************************************************************************************************
 * Adapted from NVIDIA example for MUTlass 3.x
 * Original by hkeee. MUTlass port for FlashOverlap.
 *
 * Custom epilogue that writes output to scattered (reordered + row-remapped)
 * locations and atomically signals the monitor matrix.
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

/// Custom epilogue for FlashOverlap scatter+signal injection.
/// Like SignalEpilogue, but additionally scatters output rows via a row mapping array.
template <
  class StrideC_,
  class StrideD_,
  class EpilogueSchedule_,
  int TileM_,
  int TileN_
>
class ScatterEpilogue {
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
    // Scatter-specific parameters
    int* ptr_monitor_matrix  = nullptr;
    int* ptr_reorder_array   = nullptr;
    int  monitor_columns     = 0;
    int  reorder_columns     = 0;
    int* ptr_comm_seg_array  = nullptr;
    int* ptr_row_array       = nullptr;  // Row scatter mapping
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
  ScatterEpilogue(Params const& params_, [[maybe_unused]] SharedStorage const& = SharedStorage())
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

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;

    // --- Compute reordered destination tile ---
    int tile_idx = m_coord * params.monitor_columns + n_coord;
    int reordered_tile_idx = params.ptr_reorder_array[tile_idx];
    int new_m_tile = reordered_tile_idx / params.reorder_columns;
    int new_n_tile = reordered_tile_idx % params.reorder_columns;

    int reshaped_N = params.reorder_columns * kTileN;

    // --- Build coordinate mapping for this thread ---
    auto cD = make_identity_tensor(make_shape(get<0>(blk_shape_MNK), get<1>(blk_shape_MNK)));
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCcD = thr_mma.partition_C(cD);

    // --- Write output with scatter: per-element pointer computation ---
    MUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(accumulators); ++i) {
      auto coord = tCcD(i);
      int m_local = get<0>(coord);  // row within tile
      int n_local = get<1>(coord);  // col within tile

      if (m_local >= get<0>(residue_mnk) || n_local >= get<1>(residue_mnk)) continue;

      // Compute output value
      ElementD result = ElementD(params.alpha * float(accumulators(i)));

      // Apply scatter: map the output row through the row array
      int abs_row = new_m_tile * kTileM + m_local;
      int scattered_row = params.ptr_row_array[abs_row];
      int abs_col = new_n_tile * kTileN + n_local;

      params.ptr_D[scattered_row * reshaped_N + abs_col] = result;
    }

    // --- Signal: atomically update monitor matrix ---
    if (threadIdx.x == 0) {
      int reordered_tile_2d = new_m_tile * params.reorder_columns + new_n_tile;

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