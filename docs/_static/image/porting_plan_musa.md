# FlashOverlap CUDA → MUSA 平台移植方案

## 1. 概述

将 FlashOverlap 项目从 NVIDIA CUDA 平台整体移植到 Moore Threads MUSA 平台。采用**直接 API 替换策略**（参照同目录下 DeepEP 的已验证移植模式），替换关系如下：

| NVIDIA 组件 | MUSA 平替 | 状态 |
|-------------|----------|------|
| CUDA Runtime | MUSA Runtime | 系统已安装 |
| NCCL | MCCL | 系统已安装 |
| cuBLAS | muBLAS | 系统已安装 |
| CUTLASS | MUTlass (https://github.com/MooreThreads/mutlass) | 需引入 |
| nvcc 编译器 | mcc 编译器 | 系统已安装 |
| GPU 架构 sm_80/sm_86/sm_89 | mp_31 | — |
| PyTorch CUDA (at::cuda) | torch_musa (at::musa) | 需安装 |

---

## 2. 项目代码结构与移植范围

```
FlashOverlap/
├── CMakeLists.txt              ← 全面改写
├── cmake/Modules/FindNCCL.cmake ← 替换为 FindMCCL.cmake
├── src/
│   ├── CMakeLists.txt           ← 全面改写
│   ├── pybind.cpp               ← 少量修改
│   ├── baseline_impl.h/cu       ← NCCL→MCCL, cuBLAS→muBLAS, CUDA→MUSA
│   ├── overlap_impl.h/cu        ← NCCL→MCCL, CUDA→MUSA, CUTLASS→MUTlass
│   ├── nccl_utils.h/cu          ← NCCL→MCCL
│   ├── wait.cuh                 ← CUDA intrinsics→MUSA intrinsics
│   ├── gemm/
│   │   ├── gemm.h/cu            ← CUTLASS→MUTlass (标准GEMM)
│   ├── overlap/
│   │   ├── gemm_signal.h/cu     ← CUTLASS→MUTlass (计算-通信重叠)
│   │   ├── gemm_scatter.h/cu    ← CUTLASS→MUTlass (计算-通信重叠)
│   │   ├── gemm_with_signal.h   ← 自定义Epilogue Visitor移植 ⚠️核心难点
│   │   ├── gemm_with_scatter.h  ← 自定义Epilogue Visitor移植 ⚠️核心难点
│   │   └── gemm_with_epilogue_visitor.h ← 基础Epilogue Visitor移植
│   ├── rmsnorm/
│   │   ├── rmsnorm.cuh/cu/h     ← CUDA kernel→MUSA kernel
│   │   └── utils.h              ← warp shuffle→MUSA warp shuffle
│   ├── tiling/
│   │   ├── gemm_dispatcher.h    ← 函数指针表更新
│   │   ├── gemm_tiling.cuh      ← CUTLASS→MUTlass
│   │   ├── signal_tiling.cuh    ← CUTLASS→MUTlass
│   │   └── scatter_tiling.cuh   ← CUTLASS→MUTlass
│   ├── inc/
│   │   ├── gemm_instances.inc   ← 100+实例配置，可能需重新调优
│   │   ├── signal_instances.inc
│   │   ├── scatter_instances.inc
│   │   └── monitor_instances.inc
│   └── 3rdparty/
│       └── cutlass/ → mutlass/  ← 子模块替换
├── example/*.py                 ← torch.cuda→torch.musa
├── test/*.py                    ← torch.cuda→torch.musa
└── tune/*.py                    ← torch.cuda→torch.musa
```

---

## 3. 移植步骤

### Phase 1: 构建系统迁移

#### 步骤 1: 改写根目录 CMakeLists.txt

**当前代码（关键行）：**
```cmake
project(flashoverlap LANGUAGES CXX CUDA)
find_package(CUDAToolkit 11.4 REQUIRED)
set(COMMON_LIBS CUDA::cudart)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ...")
```

**移植为：**
```cmake
project(flashoverlap LANGUAGES CXX)

# MUSA Toolkit
list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)
find_package(MUSAToolkit REQUIRED)
set(CMAKE_CXX_COMPILER "${MUSAToolkit_TARGET_DIR}/bin/mcc")

# MUSA runtime 替代 CUDA::cudart
set(COMMON_LIBS ${MUSA_LIBRARIES})  # 或 musart

# 编译 flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -x musa --offload-arch=mp_31")
```

**具体改动点：**
- `project(... LANGUAGES CXX CUDA)` → `project(... LANGUAGES CXX)`
- `find_package(CUDAToolkit 11.4 REQUIRED)` → `find_package(MUSAToolkit REQUIRED)` + 设置 module path
- 编译器设为 `mcc`
- `CUDA::cudart` → MUSA 运行时库
- `CMAKE_CUDA_FLAGS` → `CMAKE_CXX_FLAGS`（mcc 通过 CXX flags 传递 MUSA 选项）
- PyTorch 检测保留，但 `find_package(Torch)` 需确保与 torch_musa 兼容
- CXX11 ABI 兼容逻辑保留不变
- NCCL 查找替换为 MCCL 查找

#### 步骤 2: 改写 src/CMakeLists.txt

**当前代码：**
```cmake
file(GLOB gemm_files ${PROJECT_SOURCE_DIR}/src/gemm/*.cu)
add_library(gemm STATIC ${gemm_files})
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/include)
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/cutlass/tools/util/include)
```

**移植为：**
```cmake
file(GLOB gemm_files ${PROJECT_SOURCE_DIR}/src/gemm/*.cu)
add_library(gemm STATIC ${gemm_files})
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/include)
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/tools/util/include)
target_compile_options(gemm PRIVATE -x musa -fgpu-rdc --offload-arch=mp_31)
```

对 `overlap`、`norm`、`baseline_impl`、`overlap_impl`、`nccl_tool` 目标做类似处理。链接时需用 `lld` 链接器（MUSA 要求）。

#### 步骤 3: 创建 cmake/Modules/FindMCCL.cmake

参照 `FindNCCL.cmake` 的结构，查找 MCCL 的头文件和库文件：
```cmake
find_path(MCCL_INCLUDE_DIR NAMES mccl.h PATHS /usr/include /usr/local/include)
find_library(MCCL_LIBRARIES NAMES mccl PATHS /usr/lib /usr/local/lib)
```

---

### Phase 2: NCCL → MCCL 迁移

#### 步骤 4-6: 替换 NCCL 头文件、类型和 API

**涉及文件：** `nccl_utils.h`, `nccl_utils.cu`, `baseline_impl.h`, `baseline_impl.cu`, `overlap_impl.h`, `overlap_impl.cu`

**替换规则（全局查找替换）：**

| 原始 (NCCL) | 替换为 (MCCL) |
|-------------|-------------|
| `#include <nccl.h>` | `#include <mccl.h>` |
| `ncclComm_t` | `mcclComm_t` |
| `ncclResult_t` | `mcclResult_t` |
| `ncclUniqueId` | `mcclUniqueId` |
| `ncclGetUniqueId` | `mcclGetUniqueId` |
| `ncclCommInitRank` | `mcclCommInitRank` |
| `ncclAllReduce` | `mcclAllReduce` |
| `ncclReduceScatter` | `mcclReduceScatter` |
| `ncclSend` | `mcclSend` |
| `ncclRecv` | `mcclRecv` |
| `ncclGroupStart` | `mcclGroupStart` |
| `ncclGroupEnd` | `mcclGroupEnd` |
| `ncclFloat16` | `mcclFloat16` |
| `ncclSum` | `mcclSum` |
| `ncclSuccess` | `mcclSuccess` |
| `ncclGetErrorString` | `mcclGetErrorString` |
| `NCCL_CHECK` | `MCCL_CHECK` |

**示例 — nccl_utils.h 移植前后：**

```cpp
// 移植前
#include <nccl.h>
#define NCCL_CHECK(cmd) \
    do { ncclResult_t result = cmd; \
         if (result != ncclSuccess) { ... ncclGetErrorString(result) ... } \
    } while (0)

// 移植后
#include <mccl.h>
#define MCCL_CHECK(cmd) \
    do { mcclResult_t result = cmd; \
         if (result != mcclSuccess) { ... mcclGetErrorString(result) ... } \
    } while (0)
```

> **参考**: DeepEP 中 `csrc/deep_ep.hpp` 和 `csrc/kernels/exception.cuh` 已验证此替换模式可行。

---

### Phase 3: CUDA Runtime API → MUSA Runtime API

#### 步骤 7-9: 替换 CUDA 运行时头文件、API 和 PyTorch 互操作

**涉及文件：** `wait.cuh`, `baseline_impl.h/cu`, `overlap_impl.h/cu`, `nccl_utils.h`, `gemm/gemm.h`, `gemm/gemm.cu`, `overlap/gemm_signal.cu`, `overlap/gemm_scatter.cu`

**替换规则：**

| 原始 (CUDA) | 替换为 (MUSA) |
|-------------|-------------|
| `#include <cuda_runtime_api.h>` | `#include <musa_runtime_api.h>` |
| `#include <cuda_runtime.h>` | `#include <musa_runtime.h>` |
| `#include <cuda_fp16.h>` | `#include <musa_fp16.h>` |
| `cudaStream_t` | `musaStream_t` |
| `cudaEvent_t` | `musaEvent_t` |
| `cudaStreamCreateWithPriority` | `musaStreamCreateWithPriority` |
| `cudaStreamNonBlocking` | `musaStreamNonBlocking` |
| `cudaEventCreateWithFlags` | `musaEventCreateWithFlags` |
| `cudaEventDisableTiming` | `musaEventDisableTiming` |
| `cudaEventRecord` | `musaEventRecord` |
| `cudaStreamWaitEvent` | `musaStreamWaitEvent` |
| `cudaEventDestroy` | `musaEventDestroy` |

**PyTorch 互操作替换：**

| 原始 | 替换为 |
|------|--------|
| `#include <ATen/cuda/CUDAContext.h>` | `#include "torch_musa/csrc/aten/musa/MUSAContext.h"` |
| `at::cuda::getCurrentCUDAStream()` | `at::musa::getCurrentMUSAStream()` |
| `torch::kCUDA` | `torch::kMUSA` |

**示例 — overlap_impl.cu 中的 stream 管理：**

```cpp
// 移植前
cudaStreamCreateWithPriority(&this->comm_stream, cudaStreamNonBlocking, -5);
cudaEventCreateWithFlags(&this->gemm_finished, cudaEventDisableTiming);
cudaEventRecord(this->gemm_finished, this->comm_stream);
cudaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);

// 移植后
musaStreamCreateWithPriority(&this->comm_stream, musaStreamNonBlocking, -5);
musaEventCreateWithFlags(&this->gemm_finished, musaEventDisableTiming);
musaEventRecord(this->gemm_finished, this->comm_stream);
musaStreamWaitEvent(this->gemm_stream, this->gemm_finished, 0);
```

---

### Phase 4: cuBLAS → muBLAS 迁移

#### 步骤 10: 替换 BaselineImpl 中的 cuBLAS 调用

**涉及文件：** `baseline_impl.h`, `baseline_impl.cu`

**替换规则（✅ 已通过真实代码确认）：**

| 原始 (cuBLAS) | 替换为 (muBLAS) |
|--------------|----------------|
| `#include <cublas_v2.h>` | `#include <mublas.h>` |
| `cublasHandle_t` | `mublasHandle_t` |
| `cublasStatus_t` | `mublasStatus` |
| `cublasCreate` | `mublasCreate` |
| `cublasDestroy` | `mublasDestroy` |
| `cublasSetStream` | `mublasSetStream` |
| `cublasSetMathMode` | `mublasSetMathMode`（需确认是否存在，tensor op 可能默认启用） |
| `CUBLAS_TENSOR_OP_MATH` | 可能已默认启用，先尝试移除该调用 |
| `cublasGemmEx` | `mublasGemmEx` |
| `CUBLAS_STATUS_SUCCESS` | `MUBLAS_STATUS_SUCCESS` |
| `CUBLAS_OP_T` / `CUBLAS_OP_N` | `MUBLAS_OP_T` / `MUBLAS_OP_N` |
| `CUDA_R_16F` | `MUSA_R_16F`（FP16）/ `MUSA_R_16BF`（BF16） |
| `CUBLAS_COMPUTE_32F` | `MUBLAS_COMPUTE_32F` |
| `CUBLAS_GEMM_DEFAULT` | `MUBLAS_GEMM_DEFAULT` |

**确认的 muBLAS 调用示例（来自真实项目代码）：**

```cpp
// muBLAS GEMM 调用
mublasStatus status = mublasSetStream(handle_, main_stream);
assert(status == MUBLAS_STATUS_SUCCESS);

status = mublasGemmEx(
    handle_,
    MUBLAS_OP_N, MUBLAS_OP_N,
    n_, m_, k_,
    &alpha_,
    weight_, MUSA_R_16BF, n_,
    input_, MUSA_R_16BF, k_,
    &beta_,
    output_, MUSA_R_16BF, n_,
    MUBLAS_COMPUTE_32F,
    MUBLAS_GEMM_DEFAULT);
assert(status == MUBLAS_STATUS_SUCCESS);
```

> **注意**: FlashOverlap 使用 FP16（`half`），对应类型应为 `MUSA_R_16F`（FP16），而非 `MUSA_R_16BF`（BF16）。

**额外发现 — MUSA 驱动层 Stream 等待 API：**

```cpp
// muStreamWaitValue64 可用于 stream-ordered 值等待（类似 cuStreamWaitValue64）
muStreamWaitValue64(main_stream, (MUdeviceptr)ptr, value, MU_STREAM_WAIT_VALUE_EQ);
```

> 此 API 对 FlashOverlap 的 `wait.cuh` 中基于 `atomicCAS` 的 polling 等待机制可能是更高效的替代方案，值得在后续优化阶段考虑。

---

### Phase 5: CUTLASS 2.x → MUTlass 3.x 重构（⚠️核心工作量，非简单替换）

> **审计结论：MUTlass 采用 CUTLASS 3.x Collective API，不提供 2.x 向后兼容。FlashOverlap 的 GEMM 模块需要架构级重写。** 详见第 5 节审计结果。

#### 步骤 11: 引入 MUTlass 子模块（✅ 已完成）

```bash
cd src/3rdparty
git clone https://github.com/MooreThreads/mutlass
```

更新 CMakeLists.txt 中的 include 路径：
- `src/3rdparty/cutlass/include` → `src/3rdparty/mutlass/include`
- `src/3rdparty/cutlass/tools/util/include` → `src/3rdparty/mutlass/tools/util/include`

#### 步骤 12: 重写标准 GEMM (gemm/gemm.cu) — 从 2.x 直接模板参数改为 3.x CollectiveBuilder

**改动性质：架构级重写**（不是简单的命名空间替换）

原始 CUTLASS 2.x 使用 15+ 个直接模板参数（ThreadblockShape, WarpShape, InstructionShape, NumStages, SwizzleSize, SplitK 等）。MUTlass 3.x 使用 `CollectiveBuilder` 自动配置或手动构建 `CollectiveMma`。

**新代码结构（以 MP22 FP16 为例）：**
```cpp
#include "mutlass/gemm/device/gemm_universal_adapter.h"
#include "mutlass/gemm/kernel/gemm_universal.hpp"
#include "mutlass/gemm/collective/collective_builder.hpp"
#include "mutlass/epilogue/collective/collective_builder.hpp"

template <int TileM, int TileN, int TileK>
void mutlass_gemm(int M, int N, int K, const half* A, const half* B, half* D, musaStream_t stream) {
    using namespace mutlass;
    using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
    
    using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
        arch::Mp22, arch::OpClassTensorOp,
        half_t, layout::RowMajor, 8,
        half_t, layout::ColumnMajor, 8,
        float,
        TileShape, Shape<_1,_1,_1>,
        gemm::collective::StageCountAuto,
        gemm::collective::KernelScheduleAuto
    >::CollectiveOp;
    
    using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
        arch::Mp22, arch::OpClassTensorOp,
        TileShape, Shape<_1,_1,_1>,
        epilogue::collective::EpilogueTileAuto,
        float, float,
        half_t, layout::RowMajor, 8,
        half_t, layout::RowMajor, 8,
        epilogue::collective::EpilogueScheduleAuto
    >::CollectiveOp;
    
    using GemmKernel = gemm::kernel::GemmUniversal<
        Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;
    
    // ... initialize arguments and run
}
```

**关键差异：**
- 不再手动指定 WarpShape / InstructionShape / NumStages — 由 `CollectiveBuilder` 根据硬件自动选择
- `TileShape<M,N,K>` 是唯一的主要配置旋钮
- Accumulator 建议使用 `float` 而非 `half_t`（MUTlass MP22 的 32×32 MMA 生成 FP32 累加器）

#### 步骤 13: 重写自定义 Epilogue Visitor（⚠️最复杂）

**改动性质：接口重写**（回调语义保留，但模板结构完全不同）

FlashOverlap 的三个自定义文件需改写为 MUTlass 3.x `FusionCallbacks` 特化：

| 原始文件 (CUTLASS 2.x) | 新实现方式 (MUTlass 3.x) |
|------------------------|------------------------|
| `gemm_with_epilogue_visitor.h` (GemmWithEpilogueVisitor 基类) | **删除** — 由 `CollectiveEpilogue` 取代 |
| `gemm_with_signal.h` (EpilogueVisitorSignaling) | 特化 `FusionCallbacks<DispatchPolicy, FlashOverlapSignalOp, ...>` |
| `gemm_with_scatter.h` (EpilogueVisitorScatter) | 特化 `FusionCallbacks<DispatchPolicy, FlashOverlapScatterOp, ...>` |

**MUTlass 3.x 的回调机制（mp22_epilogue_evt.hpp）：**
```
callbacks.begin_epilogue()
  for each epilogue tile (epi_m, epi_n):
    callbacks.begin_step(epi_m, epi_n, step_idx)
    for each element epi_v:
      output[epi_v] = callbacks.visit(accumulator[epi_v], epi_v, epi_m, epi_n)
    callbacks.end_step(epi_m, epi_n, step_idx)
    → store output tile to global memory
callbacks.end_epilogue()
```

**信号注入实现要点：**
- `visit()` → 执行 D = alpha * acc (标准线性组合)
- `end_step()` → 在 output tile 写入后，用 `atomicAdd` 更新 monitor matrix
- tile 坐标通过 `VisitorArgs.blk_coord_mnkl` 获取
- 自定义指针（monitor matrix, reorder array 等）通过 `Arguments` 结构传入

#### 步骤 14: 重写 GemmSignal/GemmScatter 封装

`gemm_signal.cu` 和 `gemm_scatter.cu` 需完全重写：
- 不再使用 `GemmSignal<...>` / `GemmScatter<...>` 组合模板
- 改为 `GemmUniversal<ProblemShape, CollectiveMainloop, CustomCollectiveEpilogue>`
- `CustomCollectiveEpilogue` 使用步骤 13 中的 `FusionCallbacks` 特化

#### 步骤 15: 重新设计实例配置体系

**原始 12 参数体系在 3.x 中不适用。** 新体系以 `TileShape<M,N,K>` 为核心：

```cpp
// 新实例宏 (仅需 3 个 TileShape 参数)
MUTLASS_GEMM_INIT(128, 128, 32);
MUTLASS_GEMM_INIT(256, 128, 32);
MUTLASS_GEMM_INIT(128, 256, 32);
MUTLASS_GEMM_INIT(256, 128, 64);
MUTLASS_GEMM_INIT(128, 128, 64);
// ... 约 15-20 个配置 (vs 原来 100+)
```

WarpShape、InstructionShape、NumStages、SwizzleSize 由 `CollectiveBuilder` 自动选择。

需同步更新：
- `inc/gemm_instances.inc`
- `inc/signal_instances.inc`
- `inc/scatter_instances.inc`
- `inc/monitor_instances.inc`
- `tool/generate_instances.py`

#### 步骤 16: 更新 tiling/dispatch

函数指针表（`gemm_tiling.cuh`, `signal_tiling.cuh`, `scatter_tiling.cuh`）和分发器（`gemm_dispatcher.h`）需要重构：
- 函数签名变化（不再有 12 个模板参数）
- Algo 索引到 TileShape 的映射关系需重新定义

---

### Phase 6: RMSNorm 内核迁移

#### 步骤 17: 替换 CUDA intrinsics

**涉及文件：** `rmsnorm/utils.h`, `rmsnorm/rmsnorm.cuh`

**Warp Shuffle 替换：**
```cpp
// 移植前
__shfl_down_sync(0xffffffff, val, offset);
__shfl_xor_sync(uint32_t(-1), val, mask);
__shfl_sync(uint32_t(-1), val, lane);

// 移植后 — MUSA 通常保持相同 intrinsic 名称
// 需确认 MUSA 的 warp size（如果也是 32 则直接兼容）
__shfl_down_sync(0xffffffff, val, offset);
__shfl_xor_sync(uint32_t(-1), val, mask);
__shfl_sync(uint32_t(-1), val, lane);
```

**其他 intrinsics：**
- `__syncthreads()` — MUSA 通常支持
- `__forceinline__` — MUSA mcc 编译器通常支持
- `float4` 向量化读写 — 需验证 MUSA 对齐要求

#### 步骤 18: 移植 wait kernel (wait.cuh)

```cpp
// 移植前
#include <cuda_runtime_api.h>
__global__ __forceinline__ void kernel_wait_flag(const int that, int* addr) {
    while (atomicCAS(addr, that, 0) != that) {
        __nanosleep(100);  // ← 可能不可用
    }
}

// 移植后
#include <musa_runtime_api.h>
__global__ __forceinline__ void kernel_wait_flag(const int that, int* addr) {
    while (atomicCAS(addr, that, 0) != that) {
        // MUSA 如果不支持 __nanosleep，可替换为空循环或 __threadfence()
        // 方案A: 直接使用（如果 mcc 支持）
        __nanosleep(100);
        // 方案B: 如果不支持则用空操作替代
        // asm volatile("" ::: "memory");
    }
}
```

---

### Phase 7: Python 绑定 & 集成

#### 步骤 19: 更新 pybind.cpp

`pybind.cpp` 本身主要是模板包装和 `TORCH_LIBRARY` 注册，不直接调用 CUDA API，改动极小。确认以下头文件可用即可：
- `baseline_impl.h` / `overlap_impl.h` / `nccl_utils.h` / `rmsnorm/rmsnorm.h` 已完成移植

#### 步骤 20: 更新 Python 脚本

**涉及文件：** `example/*.py`, `test/*.py`, `tune/*.py`

**替换规则：**
```python
# 移植前
torch.cuda.set_device(rank)
x = torch.randn(..., device='cuda')
torch.cuda.synchronize()

# 移植后
torch.musa.set_device(rank)
x = torch.randn(..., device='musa')
torch.musa.synchronize()
```

---

## 4. 验证计划

| 阶段 | 验证内容 | 方法 |
|------|---------|------|
| 构建 | CMake 配置 + mcc 编译通过 | `mkdir build && cd build && cmake .. && make` |
| 单元 | 单 GPU GEMM 正确性 | 运行 `test/test.py` |
| 通信 | MCCL AllReduce/ReduceScatter | 运行 `example/correctness_ar.py` (2+ GPU) |
| 通信 | MCCL AllReduce/ReduceScatter | 运行 `example/correctness_rs.py` (2+ GPU) |
| 功能 | Overlap vs Baseline 数值一致性 | 对比两种实现的输出 |
| 性能 | 计算-通信重叠实际生效 | 运行 `tune/bandwidth.py`，对比吞吐 |
| 多节点 | 多节点 AllToAll | 运行 `test/test_multinode.py` |

---

## 5. MUTlass 审计结果（⚠️关键发现）

> 以下为对 MUTlass 0.3.0 源码的完整审计结果。**结论：MUTlass 采用 CUTLASS 3.x 架构，不提供 CUTLASS 2.x 向后兼容**，FlashOverlap 的 GEMM 代码需要**整体重构**而非简单的命名空间替换。

### 5.1 API 代际差异（核心发现）

| 对比项 | FlashOverlap 使用 (CUTLASS 2.x) | MUTlass 提供 (3.x) | 兼容性 |
|--------|-------------------------------|--------------------|----|
| **GemmUniversal 模板** | `device::GemmUniversal<ElementA, LayoutA, ElementB, LayoutB, ..., ThreadblockShape, WarpShape, InstructionShape, ...>` (15+ 直接模板参数) | `kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, TileScheduler>` (4 参数，通过 Collective 封装) | ❌ **不兼容** |
| **GEMM 配置方式** | 直接指定 ThreadblockShape、WarpShape、InstructionShape、NumStages、SwizzleSize | 通过 `CollectiveBuilder` 自动选择，或手动构建 `CollectiveMma` | ❌ **不兼容** |
| **Epilogue** | `EpilogueWithVisitor` threadblock 级模板 | `CollectiveEpilogue` + `FusionCallbacks` 集合级模板 | ❌ **接口不同，但回调机制保留** |
| **Layout 表示** | `layout::RowMajor` / `layout::ColumnMajor` 标签类 | 同时支持标签类和 rank-3 `Stride` | ✅ 标签类保留 |
| **2.x 适配层** | — | `GemmUniversalAdapter` 仅包装 3.x 内核，**无 2.x 特化** | ❌ **无法直接使用 2.x 内核** |

**结论：不能简单地将 `cutlass::` 全局替换为 `mutlass::`。需要将 FlashOverlap 的 GEMM 代码从 CUTLASS 2.x 风格重构为 MUTlass 3.x Collective API。**

### 5.2 架构标签

**文件**: `include/mutlass/arch/arch.h`

| CUTLASS | MUTlass | 说明 |
|---------|---------|------|
| `cutlass::arch::Sm80` | `mutlass::arch::Mp22` | MUSA 2.0 架构 (Quyuan/曲院) |
| `cutlass::arch::Sm90` | `mutlass::arch::Mp31` | MUSA 3.1 架构 |

- `Mp22` — 128 线程/warp，32×32 指令形状，2 级流水线
- `Mp31` — 32 线程/warp，支持 TME (Tensor Memory Extension)，多级流水线，warp specialized

### 5.3 MMA 指令形状

**FlashOverlap 原始**: `InstructionShape<16, 8, 16>` (NVIDIA Tensor Core mma.sync)

**MUTlass 支持**:

| 架构 | 线程数/单元 | 指令形状 | 数据类型 |
|------|-----------|---------|---------|
| **MP22** | 128 | **32×32×16** (FP16/BF16), 32×32×8 (TF32), 32×32×32 (INT8) | FP16, BF16, TF32, INT8 |
| **MP31 标准** | 32 | **16×8×8**, **16×8×16**, **8×16×16**, **16×16×16**, **16×16×32** | FP16, BF16, INT8, FP8 |
| **MP31 SQMMA** | 层级化 | 自动选择: M∈{16,32,64,128}, N∈{16,32,64,128}, K∈{16,32,64,128} | 同上 |

> **MP31 标准模式**包含与 NVIDIA 相近的 16×8×16 形状，但 MP22 使用完全不同的 32×32×16 形状。

### 5.4 Epilogue Visitor 机制

**CUTLASS 2.x (FlashOverlap 当前使用)**:
```
EpilogueWithVisitor → EpilogueVisitorSignaling {
    begin_epilogue()
    begin_row()
    visit(accumulator, row, col)   // 逐元素处理
    end_row()
    end_epilogue()                 // 段落完成后原子信号
}
```

**MUTlass 3.x (需要迁移到)**:
```
CollectiveEpilogue<..., FusionCallbacks_> {
    mp22_epilogue_evt.hpp 执行流程:
        callbacks.begin_epilogue()
        for epi_m, epi_n:
            callbacks.begin_step(epi_m, epi_n, step_idx)
            for epi_v:
                tSR_rD(epi_v) = callbacks.visit(accumulator, epi_v, epi_m, epi_n)
            callbacks.end_step(epi_m, epi_n, step_idx)  // ← 在此注入原子信号
            store D to global memory
        callbacks.end_epilogue()
}
```

**好消息：**
- ✅ `begin_epilogue()`、`visit()`、`end_epilogue()` 回调接口**保留**
- ✅ 新增 `begin_step()/end_step()` 提供更细粒度的 per-tile 回调
- ✅ `VisitorArgs` 中可访问 `blk_coord_mnkl`（threadblock tile 坐标）—— 这正是 FlashOverlap 信号注入所需
- ✅ 支持自定义 `Arguments` 结构传递 monitor matrix 指针等辅助数据
- ✅ `end_step()` 中可使用 `atomicAdd` 做原子信号更新
- ✅ EVT (Epilogue Visitor Tree) 实现位于 `mp22_epilogue_evt.hpp`

**需重写部分：**
- ❌ `EpilogueVisitorSignaling` 需改为 `FusionCallbacks` 特化
- ❌ `EpilogueVisitorScatter` 需改为 `FusionCallbacks` 特化
- ❌ `GemmWithEpilogueVisitor` 基类不再需要（由 `CollectiveEpilogue` 取代）

### 5.5 其他 API 审计结果

| API | MUTlass 中存在 | 文件路径 |
|-----|---------------|---------|
| `mutlass::half_t` | ✅ | `include/mutlass/half.h` (使用 `musa_fp16.h`) |
| `mutlass::layout::RowMajor/ColumnMajor` | ✅ | `include/mutlass/layout/matrix.h` |
| `mutlass::gemm::GemmShape<M,N,K>` | ✅ | `include/mutlass/gemm_coord.h` |
| `mutlass::gemm::GemmCoord` | ✅ | `include/mutlass/gemm_coord.h` |
| `mutlass::sizeof_bits<T>` | ✅ | `include/mutlass/numeric_size.h` |
| `mutlass::epilogue::thread::LinearCombination` | ✅ | `include/mutlass/epilogue/thread/linear_combination.h` |
| `mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle` | ✅ (仅用于适配器) | `include/mutlass/gemm/threadblock/threadblock_swizzle.h` |
| `mutlass::Status` / `mutlassGetStatusString()` | ✅ | `include/mutlass/mutlass.h` |
| `mutlass::gemm::GemmUniversalMode::kGemm` | ✅ | `include/mutlass/gemm/gemm_enumerated_types.h` |
| `mutlass::device_memory` | ✅ | `tools/util/include/mutlass/util/device_memory.h` |
| `mutlass::arch::OpClassTensorOp` | ✅ | `include/mutlass/arch/mma.h` |
| `CollectiveBuilder` (GEMM) | ✅ | `include/mutlass/gemm/collective/collective_builder.hpp` |
| `CollectiveBuilder` (Epilogue) | ✅ | `include/mutlass/epilogue/collective/` 下的 builders |

### 5.6 Warp Size 特殊说明

从 `include/mutlass/mutlass.h`:
```cpp
NumThreadsPerWarpBeforeMP31 = 128   // MP22: 128 线程/warp（与 NVIDIA 32 线程不同！）
NumThreadsPerWarp = 32              // MP31+: 32 线程/warp（与 NVIDIA 相同）
NumThreadsPerWarpSquad = 128        // MP31 warp squad: 4 warp 协作
```

> **重要**: MP22 的 warp size 是 128 而非 32。这意味着 FlashOverlap 中 `rmsnorm/utils.h` 的 warp shuffle 和 `blockReduceSum` 使用 `0xffffffff` (32-bit mask) 在 MP22 上可能需要调整。如果目标是 MP31 则 warp size 为 32，与 NVIDIA 相同。

---

## 6. 风险评估（审计后更新）

### 极高风险

| # | 风险项 | 说明 | 缓解措施 |
|---|--------|------|---------|
| 1 | **CUTLASS 2.x → MUTlass 3.x API 代际重构** | MUTlass **不提供 CUTLASS 2.x 兼容层**。FlashOverlap 的整个 GEMM 模块（标准 GEMM + Signal + Scatter，共 ~10 个核心文件）必须从 CUTLASS 2.x 直接模板参数风格**重构为 MUTlass 3.x Collective API 风格**。这不是简单的查找替换，而是架构级重写 | 分两步走：先用 `CollectiveBuilder` 实现标准 GEMM 验证功能正确性，再移植自定义 epilogue |
| 2 | **自定义 Epilogue Visitor 重写** | `EpilogueVisitorSignaling` 和 `EpilogueVisitorScatter` 需从 CUTLASS 2.x threadblock epilogue 接口改写为 MUTlass 3.x `FusionCallbacks` 接口。虽然回调方法名保留，但参数签名、tile 迭代模型、数据访问方式完全不同 | 参考 `mp22_epilogue_evt.hpp` 中的回调接口，逐步实现。`VisitorArgs.blk_coord_mnkl` 可获取 tile 坐标，`end_step()` 可做原子信号 |

### 高风险

| # | 风险项 | 说明 | 缓解措施 |
|---|--------|------|---------|
| 3 | **实例配置体系废弃** | 原有 100+ 个基于 `(ThreadblockM, ThreadblockN, ThreadblockK, WarpM, WarpN, WarpK, InstructionM, InstructionN, InstructionK, NumStages, SwizzleSize, SplitK)` 的实例配置**在 MUTlass 3.x 中不适用**。3.x 使用 `CollectiveBuilder` + `TileShape_MNK` + `StageCountAuto` 自动配置，或手动构建 `CollectiveMma` | 重新设计实例体系：以 `TileShape<M,N,K>` 为主要变量，利用 `CollectiveBuilder` 自动优化其余参数。大幅减少手动实例数量 |
| 4 | **SplitK 模式支持** | CUTLASS 2.x 的 SplitK 通过 `GemmUniversalMode::kGemmSplitKParallel` 实现，MUTlass 3.x 通过 `TileScheduler` 实现，API 完全不同 | 检查 MUTlass `TileScheduler` 是否支持 K-splitting；如不支持可暂时禁用 SplitK |

### 中风险

| # | 风险项 | 说明 | 缓解措施 |
|---|--------|------|---------|
| 5 | **MP22 warp size 128 vs NVIDIA 32** | RMSNorm 中 `__shfl_*_sync(0xffffffff, ...)` 使用 32-bit mask 假设 warp=32，在 MP22 上不正确 | 如目标为 MP31 (warp=32) 则无需修改；如需支持 MP22 则需更新 mask |
| 6 | **`__nanosleep()` 可用性** | wait kernel 中使用，MUSA 可能不支持 | 替换为 `asm volatile("" ::: "memory")` 或 `__threadfence()` |
| 7 | **muBLAS API 兼容性** | `cublasGemmEx` → `mublasGemmEx` 精确签名需确认 | 查阅 MUSA SDK 文档 |
| 8 | **Stream 优先级行为** | 计算-通信重叠依赖 stream 优先级差异化 | 测试 `musaStreamCreateWithPriority` 行为 |
| 9 | **torch_musa 集成** | `at::musa::getCurrentMUSAStream()` 等 API 需与实际安装版本匹配 | 确认 torch_musa 版本 |

### 低风险

| # | 风险项 | 说明 |
|---|--------|------|
| 10 | MCCL API 兼容性 | DeepEP 已验证此替换模式可行 |
| 11 | Python 层修改 | 简单的设备字符串替换 |
| 12 | 构建系统 | DeepEP 提供了清晰的 CMake 模板参考 |

---

## 7. 建议的执行顺序（审计后更新）

```
Phase A: 基础平台层移植 (Week 1-2)
  ├── 构建系统迁移 (CMakeLists.txt)
  ├── CUDA Runtime → MUSA Runtime (所有源文件)
  ├── NCCL → MCCL (nccl_utils, baseline_impl, overlap_impl)
  ├── cuBLAS → muBLAS (baseline_impl)
  ├── RMSNorm 内核移植 (rmsnorm/)
  ├── wait kernel 移植 (wait.cuh)
  ├── Python 脚本更新 (example/, test/, tune/)
  └── ✅ 验证: BaselineImpl (muBLAS+MCCL) 在 MUSA 上可运行

Phase B: 标准 GEMM 重构 (Week 2-3)
  ├── 用 MUTlass 3.x CollectiveBuilder API 重写 gemm/gemm.cu
  │   ├── 定义 CollectiveMainloop (half_t, RowMajor/ColumnMajor, Mp22 或 Mp31)
  │   ├── 定义 CollectiveEpilogue (LinearCombination)
  │   ├── 组装 GemmKernel = GemmUniversal<ProblemShape, Mainloop, Epilogue>
  │   └── 使用 GemmUniversalAdapter 提供设备级 API
  ├── 重写 gemm/gemm.h 模板接口 (不再需要 12 个模板参数)
  ├── 重新设计实例配置体系
  │   ├── 以 TileShape<M,N,K> 为主要变量
  │   ├── 利用 CollectiveBuilder 自动选择 MMA + pipeline stages
  │   └── 更新 inc/gemm_instances.inc
  ├── 更新 tiling/gemm_tiling.cuh 函数指针表
  └── ✅ 验证: OverlapImpl.Gemm() 标准 GEMM 正确

Phase C: 自定义 Epilogue 重构 (Week 3-5) ⚠️核心难点
  ├── 实现 FlashOverlapSignalOp (FusionOperation 子类)
  ├── 特化 FusionCallbacks<..., FlashOverlapSignalOp, ...>
  │   ├── Arguments: monitor_matrix 指针, reorder_array, comm_seg_array
  │   ├── visit(): D = alpha * acc + beta * C (标准 LinearCombination)
  │   ├── end_step(): atomicAdd 更新 monitor matrix (从 VisitorArgs.blk_coord 获取 tile 坐标)
  │   └── end_epilogue(): 最终信号更新
  ├── 实现 FlashOverlapScatterOp + 对应 FusionCallbacks
  │   ├── visit(): 应用 reorder array 进行 scatter 写入
  │   └── end_step(): 原子信号
  ├── 重写 overlap/gemm_signal.cu 和 gemm_scatter.cu
  │   └── 使用 CollectiveEpilogue<..., FusionCallbacks_t> 替代 GemmWithEpilogueVisitor
  ├── 删除旧的 gemm_with_epilogue_visitor.h (不再需要)
  ├── 更新 inc/signal_instances.inc, scatter_instances.inc, monitor_instances.inc
  ├── 更新 tiling/signal_tiling.cuh, scatter_tiling.cuh
  └── ✅ 验证: GemmAllReduceOverlap / GemmReduceScatterOverlap 正确

Phase D: 端到端集成 & 性能调优 (Week 5-7)
  ├── 运行 correctness_ar.py / correctness_rs.py 验证数值正确性
  ├── 运行 bandwidth 测试验证计算-通信重叠收益
  ├── TileShape 配置搜索 (tune/search.py)
  ├── Stream 优先级调优
  └── 最终性能 benchmark
```

---

## 8. Phase 5 重构详细设计（CUTLASS 2.x → MUTlass 3.x）

### 8.1 标准 GEMM 重构示例

**FlashOverlap 原始代码** (`gemm/gemm.cu`, CUTLASS 2.x):
```cpp
using DeviceGemmBasic = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,          // cutlass::half_t, RowMajor
    ElementB, LayoutB,          // cutlass::half_t, ColumnMajor
    ElementC, LayoutC,          // cutlass::half_t, RowMajor
    ElementAccumulator,         // cutlass::half_t
    OperatorClass,              // OpClassTensorOp
    ArchTag,                    // Sm80
    ThreadblockShape,           // GemmShape<128,128,32>
    WarpShape,                  // GemmShape<64,64,32>
    InstructionShape,           // GemmShape<16,8,16>
    EpilogueOp,                 // LinearCombination
    SwizzleThreadBlock,         // GemmIdentityThreadblockSwizzle<SwizzleSize>
    NumStages,                  // 3/4/5
    AlignmentA, AlignmentB>;    // 8
```

**MUTlass 3.x 重构后** (推荐方式):
```cpp
using namespace mutlass;

// 方式 A: 使用 CollectiveBuilder (推荐，自动优化)
using CollectiveMainloop = typename gemm::collective::CollectiveBuilder<
    arch::Mp22,                           // 或 arch::Mp31
    arch::OpClassTensorOp,
    half_t, layout::RowMajor, 8,          // A: type, layout, alignment
    half_t, layout::ColumnMajor, 8,       // B: type, layout, alignment
    float,                                // Accumulator (建议用 float，精度更高)
    Shape<_128, _128, _32>,               // TileShape_MNK
    Shape<_1, _1, _1>,                    // ClusterShape (单 CTA)
    gemm::collective::StageCountAuto,     // 自动选择流水线级数
    gemm::collective::KernelScheduleAuto  // 自动选择调度策略
>::CollectiveOp;

using CollectiveEpilogue = typename epilogue::collective::CollectiveBuilder<
    arch::Mp22, arch::OpClassTensorOp,
    Shape<_128, _128, _32>, Shape<_1, _1, _1>,
    epilogue::collective::EpilogueTileAuto,
    float, float,                              // Accumulator, Compute
    half_t, layout::RowMajor, 8,               // C source
    half_t, layout::RowMajor, 8,               // D output
    epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using GemmKernel = gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,                  // ProblemShape (M,N,K,batch)
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;

// 启动 GEMM
auto arguments = typename Gemm::Arguments{
    gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},                                         // problem size
    {A, stride_A, B, stride_B},                            // mainloop args
    {{1.0f, 0.0f}, nullptr, stride_C, D, stride_D},       // epilogue args
    KernelHardwareInfo{}
};
Gemm gemm;
gemm.initialize(arguments, workspace);
gemm.run(stream);
```

### 8.2 自定义 Epilogue (Signal) 重构示例

**需要实现的 FusionCallbacks 特化：**

```cpp
// 1. 定义融合操作类型
struct FlashOverlapSignalOp : mutlass::epilogue::fusion::FusionOperation {
    using ElementOutput = mutlass::half_t;
    using ElementCompute = float;
    using ElementSource = mutlass::half_t;
};

// 2. 特化 FusionCallbacks
template <class CtaTile_MNK, class EpilogueTile_MN>
struct mutlass::epilogue::fusion::FusionCallbacks<
    mutlass::gemm::MainloopMp22TwoStage,   // DispatchPolicy
    FlashOverlapSignalOp,                   // Operation
    CtaTile_MNK,
    EpilogueTile_MN
> {
    struct Arguments {
        float alpha = 1.0f, beta = 0.0f;
        int* ptr_monitor_matrix;
        int* ptr_reorder_array;
        int  monitor_columns;
        int  reorder_columns;
        int* ptr_comm_seg_array;
        bool if_monitor;
    };
    
    struct Params { /* 从 Arguments 转换 */ };
    struct SharedStorage {};

    struct Callbacks {
        Params const& params;
        // blk_coord 来自 VisitorArgs
        int m_coord, n_coord;
        
        MUTLASS_DEVICE void begin_epilogue() {}
        
        MUTLASS_DEVICE void begin_step(int epi_m, int epi_n, int step_idx) {}
        
        MUTLASS_DEVICE mutlass::half_t visit(
            float accumulator, int epi_v, int epi_m, int epi_n) {
            // D = alpha * acc (标准线性组合)
            return mutlass::half_t(params.alpha * accumulator);
        }
        
        MUTLASS_DEVICE void end_step(int epi_m, int epi_n, int step_idx) {
            // ⚡ 核心: 在 output tile 写入后注入通信信号
            if (threadIdx.x == 0 && params.if_monitor) {
                int tile_row = m_coord;  // 当前 threadblock 的 M 维度 tile 索引
                int reordered_row = params.ptr_reorder_array[tile_row];
                int seg = params.ptr_comm_seg_array[reordered_row];
                atomicAdd(&params.ptr_monitor_matrix[seg], 1);
            }
        }
        
        MUTLASS_DEVICE void end_epilogue() {}
        
        MUTLASS_DEVICE bool is_C_load_needed() { return false; }
    };
    
    Callbacks get_callbacks(auto const& visitor_args) {
        auto [m, n, k, l] = visitor_args.blk_coord_mnkl;
        return Callbacks{params, m, n};
    }
};
```

### 8.3 实例配置重新设计

**CUTLASS 2.x 风格 (废弃)**:
```
CUTLASS_GEMM_SPLITK_INIT(128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 1, 1);
// 12 个参数: TB_M, TB_N, TB_K, Warp_M, Warp_N, Warp_K, Inst_M, Inst_N, Inst_K, Stages, Swizzle, SplitK
```

**MUTlass 3.x 风格 (新)**:
```cpp
// 仅需 TileShape 作为主要配置变量，其余由 CollectiveBuilder 自动选择
MUTLASS_GEMM_INIT(128, 128, 32);   // TileShape<128, 128, 32>
MUTLASS_GEMM_INIT(256, 128, 32);   // TileShape<256, 128, 32>
MUTLASS_GEMM_INIT(128, 256, 32);   // TileShape<128, 256, 32>
MUTLASS_GEMM_INIT(256, 128, 64);   // TileShape<256, 128, 64>
// ... 大幅减少变体数量 (从 100+ 降至 ~20)
```

---

## 9. 前置确认清单（审计后更新）

| # | 确认项 | 审计结果 |
|---|--------|---------|
| 1 | MUTlass 中是否存在 `GemmUniversal` 模板 | ✅ 存在但为 **3.x API** (`kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>`)，不兼容 2.x |
| 2 | MUTlass 中是否存在 epilogue visitor 扩展接口 | ✅ 存在 `FusionCallbacks` 模板 + `mp22_epilogue_evt.hpp`，回调方法保留 (`begin_epilogue/visit/end_step/end_epilogue`) |
| 3 | GPU 架构标签名 | ✅ `mutlass::arch::Mp22` (Quyuan) / `mutlass::arch::Mp31` |
| 4 | Tensor 指令形状 | ✅ MP22: 32×32×16 (FP16); MP31: 16×8×16, 16×16×16 等 (FP16) |
| 5 | Warp size | ✅ MP22: 128, MP31: 32 |
| 6 | half_t 类型 | ✅ `mutlass::half_t` (使用 `musa_fp16.h`) |
| 7 | Layout 类型 | ✅ `mutlass::layout::RowMajor/ColumnMajor` 保留 |
| 8 | LinearCombination epilogue | ✅ `mutlass::epilogue::thread::LinearCombination` 保留 |
| 9 | Status / 错误处理 | ✅ `mutlass::Status` + `mutlassGetStatusString()` |
| 10 | 工具库 | ✅ `tools/util/include/` 包含 device_memory, host_tensor, command_line 等 |

**仍需实际验证（编码阶段）：**
- [ ] muBLAS 精确 API 签名
- [ ] MUSA 是否支持 `__nanosleep()`
- [ ] torch_musa `at::musa::` 命名空间 API
- [ ] 目标硬件是 MP22 还是 MP31（影响所有 GEMM 配置）
- [ ] `FusionCallbacks` 在 `end_step()` 中做 `atomicAdd` 的实际可行性

---

## 10. 各阶段执行状态

### Phase 1: 构建系统迁移 — ✅ 已完成

**提交**: `1585c3c feat: migrate build system from CUDA to MUSA platform`

改动文件：
- `CMakeLists.txt` — 全面改写（nvcc → mcc, CUDAToolkit → MUSAToolkit, CUDA::cudart → MUSA 运行时）
- `src/CMakeLists.txt` — include 路径 cutlass → mutlass, 编译选项 `-x musa --offload-arch=mp_31`
- `cmake/Modules/FindMCCL.cmake` — 新建（替代 FindNCCL.cmake）

### Phase 2: NCCL → MCCL 迁移 — ✅ 已完成

**提交**: `b101ff8 feat: migrate NCCL to MCCL communication library`

改动文件：
- `nccl_utils.h` → `mccl_utils.h` (重命名 + 全局替换 nccl → mccl)
- `nccl_utils.cu` → `mccl_utils.cu`
- `baseline_impl.h/cu` — NCCL 类型/函数 → MCCL
- `overlap_impl.h/cu` — NCCL 类型/函数 → MCCL
- `pybind.cpp` — 更新 include 和 wrapper 函数名

### Phase 3: CUDA Runtime → MUSA Runtime — ✅ 已完成

**提交**: `d3a942c feat: migrate Runtime API to MUSA`

16 个文件全面替换：
- 头文件: `cuda_runtime.h` → `musa_runtime.h`, `cuda_fp16.h` → `musa_fp16.h`
- 类型: `cudaStream_t` → `musaStream_t`, `cudaEvent_t` → `musaEvent_t`
- 函数: `cudaStreamCreate*` → `musaStreamCreate*`, `cudaEventCreate*` → `musaEventCreate*` 等
- PyTorch: `ATen/cuda/CUDAContext.h` → `torch_musa/.../MUSAContext.h`, `at::cuda::` → `at::musa::`

### Phase 4: cuBLAS → muBLAS — ✅ 已完成

**提交**: `49cad95 feat: migrate mublas`

改动文件：
- `baseline_impl.h` — `cublasHandle_t` → `mublasHandle_t`
- `baseline_impl.cu` — `cublasCreate/Destroy/SetStream/GemmEx` → `mublasCreate/Destroy/SetStream/GemmEx`, 类型常量 `CUBLAS_*` → `MUBLAS_*`, `CUDA_R_16F` → `MUSA_R_16F`
- `overlap_impl.h` — 头文件更新
- `pybind.cpp` — 头文件更新

### Phase 5: CUTLASS 2.x → MUTlass 3.x 重构 — ✅ 已完成

**改动性质**: 架构级重写（非简单命名空间替换）。原始代码使用 CUTLASS 2.x 直接模板参数 API（GemmUniversal 15+ 参数、ThreadblockShape/WarpShape/InstructionShape），MUTlass 仅提供 3.x Collective API，无 2.x 兼容层。

**17 个文件, -2669 / +689 行**:

| 文件 | 改动说明 |
|------|---------|
| `src/gemm/gemm.h` | 函数声明: `cutlass_gemm_splitk<12 params>` → `mutlass_gemm<TileM, TileN, TileK>` |
| `src/gemm/gemm.cu` | 架构重写: CUTLASS 2.x `GemmUniversal<15+ params>` → MUTlass 3.x `CollectiveBuilder` + `GemmUniversalAdapter` |
| `src/overlap/gemm_signal.h` | 函数声明: `cutlass_gemm_signal<12 params>` → `mutlass_gemm_signal<TileM, TileN, TileK>` |
| `src/overlap/gemm_signal.cu` | 架构重写: 同 gemm.cu + 自定义 `SignalEpilogue` 替代 `EpilogueWithVisitor` |
| `src/overlap/gemm_scatter.h` | 函数声明: `cutlass_gemm_scatter<12 params>` → `mutlass_gemm_scatter<TileM, TileN, TileK>` |
| `src/overlap/gemm_scatter.cu` | 架构重写: 同上 + 自定义 `ScatterEpilogue` |
| `src/overlap/gemm_with_signal.h` | 自定义 Epilogue: CUTLASS 2.x `EpilogueVisitorSignaling` → MUTlass 3.x `mutlass::epilogue::collective::SignalEpilogue` |
| `src/overlap/gemm_with_scatter.h` | 自定义 Epilogue: CUTLASS 2.x `EpilogueVisitorScatter` → MUTlass 3.x `mutlass::epilogue::collective::ScatterEpilogue` |
| `src/overlap/gemm_with_epilogue_visitor.h` | **已删除** — CUTLASS 2.x 基类，不再被任何文件引用 |
| `src/overlap_impl.h` | `#include "cutlass/util/device_memory.h"` → `"mutlass/util/device_memory.h"` |
| `src/tiling/gemm_tiling.cuh` | 函数指针表: 108 条 `&cutlass_gemm_splitk<12 params>` → 6 条 `&mutlass_gemm<3 params>` |
| `src/tiling/signal_tiling.cuh` | 函数指针表: 108 条 → 6 条 (同上模式) |
| `src/tiling/scatter_tiling.cuh` | 函数指针表: 108 条 → 6 条 (同上模式) |
| `src/inc/gemm_instances.inc` | 实例宏: 108 条 `CUTLASS_GEMM_SPLITK_INIT(12 params)` → 6 条 `MUTLASS_GEMM_INIT(TileM, TileN, TileK)` |
| `src/inc/signal_instances.inc` | 实例宏: 102 条 → 6 条 `MUTLASS_SIGNAL_INIT(TileM, TileN, TileK)` |
| `src/inc/scatter_instances.inc` | 实例宏: 102 条 → 6 条 `MUTLASS_SCATTER_INIT(TileM, TileN, TileK)` |
| `src/inc/monitor_instances.inc` | 清理为注释占位（原文件未被任何源文件 include，属于死代码） |

**关键设计决策**:
- 模板参数从 12 个（ThreadblockM/N/K, WarpM/N/K, InstructionM/N/K, NumStages, SwizzleSize, SplitK）缩减为 3 个（TileM, TileN, TileK），`CollectiveBuilder` 自动选择 MMA 形状、流水线级数等
- 实例总数从 108 缩减为 6（覆盖所有唯一 TileShape 组合：128×128×32, 128×128×64, 128×256×32, 128×256×64, 256×128×32, 256×128×64）
- Accumulator 类型从 `half_t` 改为 `float`（MUTlass MP22 推荐，精度更高）
- 架构标签: `cutlass::arch::Sm80` → `mutlass::arch::Mp22`

**遗留说明**:
- `pybind.cpp` 中 Python API 名称 `"cutlass_init"`/`"cutlass_gemm"` 及 C++ wrapper 函数名 `CutlassInitWrapper`/`CutlassGemmWrapper` 保留原名 — 将在 Phase 7（Python 层）一并改名
- `overlap_impl.cu` 中 `CutlassInit()` 方法名同理
- Algo 索引空间从 0-107 缩减为 0-5，需在 Phase 7 更新 `tune/gen_config.py` 的调优脚本适配到新索引

### Phase 6: RMSNorm 内核迁移 — ✅ 已完成

**改动极小** — RMSNorm 使用的 GPU device intrinsics（`__shfl_down_sync`、`__shfl_xor_sync`、`__shfl_sync`、`__syncthreads`、`__half2float`、`__float2half`、`__hadd`、`__hadd2`、`__hmul2`、`rsqrtf`、`__fdividef`、`float4` 向量化访存、kernel launch `<<<>>>` 语法）均为 MUSA mcc 编译器原生支持的标准 GPU 编程原语，无需修改。

MUSA MP31 架构的 warp size 为 32，与 NVIDIA 相同，因此 `0xffffffff` warp mask 和 `WARP_SIZE=32` 宏定义完全兼容。

**1 个文件改动**:

| 文件 | 改动说明 |
|------|---------|
| `src/rmsnorm/rmsnorm.cuh` | 移除未使用的 CUDA 头文件 `#include <curand_kernel.h>` 和 `#include <driver_functions.h>` |

其余文件无需改动：
- `src/rmsnorm/utils.h` — 纯 `__device__` 函数，使用标准 warp shuffle / FP16 intrinsics，mcc 原生支持
- `src/rmsnorm/rmsnorm.cu` — Phase 3 已迁移头文件（`musa.h`, `musa_runtime.h`, `torch_musa`），kernel launch 语法 mcc 原生支持
- `src/rmsnorm/rmsnorm.h` — 纯 PyTorch 接口，Phase 3 已迁移

### Phase 7: Python 绑定 & 集成 — ✅ 已完成

**改动范围**: 14 个 Python 文件 + `src/pybind.cpp` + `tool/generate_instances.py`

**关键修复（运行时必需）**:

| 替换类别 | 原始 | 替换为 |
|---------|------|--------|
| PyTorch Device API | `torch.cuda.*` | `torch.musa.*` |
| Device 字符串 | `device="cuda"` / `f'cuda:{rank}'` / `.cuda(rank)` | `device="musa"` / `f'musa:{rank}'` / `.musa(rank)` |
| GEMM 初始化 | `.cutlass_init()` | `.mutlass_init()` |
| GEMM 计算 | `.cutlass_gemm(...)` | `.mutlass_gemm(...)` |
| 通信初始化 | `.nccl_init(...)` | `.mccl_init(...)` |
| AllReduce | `.nccl_allreduce(...)` | `.mccl_allreduce(...)` |
| ReduceScatter | `.nccl_reducescatter(...)` | `.mccl_reducescatter(...)` |
| BLAS 初始化 | `.cublas_init()` | `.mublas_init()` |
| 通信 ID 生成 | `torch.ops.flashoverlap_op.generate_nccl_id()` | `torch.ops.flashoverlap_op.generate_mccl_id()` |
| 分布式后端 | `backend="nccl"` / `backend='nccl'` | `backend="mccl"` / `backend='mccl'` |
| 环境变量 | `CUDA_VISIBLE_DEVICES` | `MUSA_VISIBLE_DEVICES` |

**C++ pybind.cpp 内部重命名**:

| 原始 | 替换为 |
|------|--------|
| `CutlassInitWrapper` | `MutlassInitWrapper` |
| `CutlassGemmWrapper` | `MutlassGemmWrapper` |
| `CutlassGemmAllReduceWrapper` | `MutlassGemmAllReduceWrapper` |
| `CutlassGemmReduceScatterWrapper` | `MutlassGemmReduceScatterWrapper` |
| `CutlassAll2AllWrapper` | `MutlassAll2AllWrapper` |
| `self->CutlassInit()` | `self->MutlassInit()` |
| `OverlapImpl::CutlassInit()` | `OverlapImpl::MutlassInit()` |

**Python 变量名重命名（一致性）**:
- `nccl_id` → `mccl_id`
- `nccl_id_tsr` → `mccl_id_tsr`
- `generate_and_broadcast_nccl_id()` → `generate_and_broadcast_mccl_id()`
- `cutlass_gemm` → `gemm_class` (tune/gen_config.py, tune/profile_config.py)
- `cutlass_dur` → `gemm_dur` (tune/gen_config.py, tune/profile_config.py)

**代码生成工具更新**:
- `tool/generate_instances.py` — 从 CUTLASS 2.x 12 参数格式重写为 MUTlass 3.x 3 参数格式（TileM, TileN, TileK）

**涉及文件**:

| 文件 | 改动说明 |
|------|---------|
| `src/pybind.cpp` | API 名称 + C++ wrapper 函数名全面 Cutlass→Mutlass |
| `src/overlap_impl.h` | `CutlassInit()` → `MutlassInit()` 方法声明 |
| `src/overlap_impl.cu` | `CutlassInit()` → `MutlassInit()` 方法定义 + 注释更新 |
| `example/correctness_ar.py` | torch.cuda→torch.musa, nccl→mccl, cutlass→mutlass |
| `example/correctness_rs.py` | 同上 |
| `example/RMSNorm.py` | 同上 |
| `example/RowParallelLinear.py` | 同上 |
| `example/utils.py` | 同上 |
| `test/test.py` | 同上 + cublas_init→mublas_init, profiler 注释更新 |
| `test/test_multinode.py` | 同上 |
| `tune/bandwidth.py` | torch.cuda→torch.musa, nccl→mccl |
| `tune/bandwidth_multinode.py` | 同上 |
| `tune/gen_config.py` | 同上 + cutlass_dur→gemm_dur, cutlass_gemm→gemm_class, path 默认值更新 |
| `tune/profile_config.py` | 同上 + cutlass_dur→gemm_dur, cutlass_gemm→gemm_class |
| `tune/search.py` | torch.cuda→torch.musa, nccl→mccl |
| `tune/search_multinode.py` | 同上 |
| `tool/generate_instances.py` | CUTLASS 2.x 12 参数 → MUTlass 3.x 3 参数重写 |

**遗留文件**:
- `src/nccl_utils.cu` / `src/nccl_utils.h` — Phase 2 遗留的旧副本（已被 `mccl_utils.*` 替代，无代码引用），建议删除
