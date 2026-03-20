# ====== Helper: Mark .cu files for MUSA compilation via mcc ======
# mcc needs -x musa to enable MUSA language extensions (__global__, __device__, etc.)
# and LANGUAGE CXX since cmake doesn't natively know how to handle .cu with mcc.
function(set_musa_source_properties)
  foreach(src ${ARGN})
    set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
    set_source_files_properties(${src} PROPERTIES
      COMPILE_FLAGS "-x musa --offload-arch=${MUSA_ARCH}")
  endforeach()
endfunction()

# ====== GEMM library (mutlass-based) ======
file(GLOB gemm_files ${PROJECT_SOURCE_DIR}/src/gemm/*.cu)
set_musa_source_properties(${gemm_files})
add_library(gemm STATIC ${gemm_files})
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/include)
target_include_directories(gemm PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/tools/util/include)
set_property(TARGET gemm PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====== Overlap library (mutlass-based custom epilogue) ======
file(GLOB overlap_files ${PROJECT_SOURCE_DIR}/src/overlap/*.cu)
set_musa_source_properties(${overlap_files})
add_library(overlap STATIC ${overlap_files})
target_include_directories(overlap PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/include)
target_include_directories(overlap PUBLIC ${PROJECT_SOURCE_DIR}/src/3rdparty/mutlass/tools/util/include)
set_property(TARGET overlap PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====== RMSNorm library ======
file(GLOB norm_files ${PROJECT_SOURCE_DIR}/src/rmsnorm/*.cu)
set_musa_source_properties(${norm_files})
add_library(norm STATIC ${norm_files})
set_property(TARGET norm PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====== BaselineImpl (muBLAS + MCCL path) ======
set_musa_source_properties(baseline_impl.cu)
add_library(baseline_impl baseline_impl.cu)
set_property(TARGET baseline_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries (baseline_impl PUBLIC gemm overlap norm)

# ====== OverlapImpl (mutlass GEMM + MCCL overlap path) ======
set_musa_source_properties(overlap_impl.cu)
add_library(overlap_impl overlap_impl.cu)
set_property(TARGET overlap_impl PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries (overlap_impl PUBLIC gemm overlap norm)

# ====== MCCL utilities ======
set_musa_source_properties(mccl_utils.cu)
add_library(mccl_tool mccl_utils.cu)
set_property(TARGET mccl_tool PROPERTY POSITION_INDEPENDENT_CODE ON)

# ====== Python binding shared library ======
add_library(st_pybinding SHARED pybind.cpp)
target_link_libraries(
    st_pybinding
    baseline_impl
    overlap_impl
    mccl_tool
)

set_target_properties(st_pybinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)
