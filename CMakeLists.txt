cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# ====== MUSA Toolkit Path (must be before project()) ======
if (NOT DEFINED CUSTOM_MUSA_PATH)
  set(CUSTOM_MUSA_PATH "/usr/local/musa")
endif()
# Use mcc (MUSA C/C++ Compiler) as the CXX compiler
if (NOT DEFINED CMAKE_CXX_COMPILER)
  set(CMAKE_CXX_COMPILER "${CUSTOM_MUSA_PATH}/bin/mcc")
endif()

project(flashoverlap LANGUAGES CXX)

if (DEFINED ENV{CONDA_PREFIX})
  # use conda environment
  link_directories($ENV{CONDA_PREFIX}/lib)
  include_directories($ENV{CONDA_PREFIX}/include)
endif()

# ====== MUSA Toolkit Discovery ======
list(APPEND CMAKE_MODULE_PATH ${CUSTOM_MUSA_PATH}/cmake)
find_package(MUSAToolkit REQUIRED)
message(STATUS "MUSAToolkit_TARGET_DIR: ${MUSAToolkit_TARGET_DIR}")

# MUSA runtime library (replaces CUDA::cudart)
find_library(MUSART_LIBRARY musart
  PATHS ${CUSTOM_MUSA_PATH}
  PATH_SUFFIXES lib/x86_64-linux-gnu lib/x64 lib64 lib
  NO_DEFAULT_PATH)
if(NOT MUSART_LIBRARY)
  message(FATAL_ERROR "musart library not found in ${CUSTOM_MUSA_PATH}")
endif()
message(STATUS "MUSART: ${MUSART_LIBRARY}")

# muBLAS library (replaces cuBLAS)
find_library(MUBLAS_LIBRARY mublas
  PATHS ${CUSTOM_MUSA_PATH}
  PATH_SUFFIXES lib/x86_64-linux-gnu lib/x64 lib64 lib
  NO_DEFAULT_PATH)
if(MUBLAS_LIBRARY)
  message(STATUS "MUBLAS: ${MUBLAS_LIBRARY}")
else()
  message(WARNING "mublas library not found, will rely on torch_musa to provide it")
endif()

# MUSA target architecture (mp_22 for Quyuan 1, mp_31 for Quyuan 2)
set(MUSA_ARCH "mp_31" CACHE STRING "MUSA target architecture")
message(STATUS "MUSA_ARCH: ${MUSA_ARCH}")

# Add filesystem library for gcc < 9 (harmless with mcc/clang, condition won't match)
link_libraries( "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>" )

# Set up C++ standard
set(CXX_STD "17" CACHE STRING "C++ standard")
set(CMAKE_CXX_STANDARD ${CXX_STD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (BUILD_MODE)
  string(TOUPPER ${BUILD_MODE} BUILD_MODE)
  if (BUILD_MODE STREQUAL "DEBUG")
    set(DEBUG ON)
  elseif (BUILD_MODE STREQUAL "RELEASE")
    set(RELEASE ON)
  else()
    message(FATAL_ERROR "Unknown build mode: ${BUILD_MODE}")
  endif()
else()
  message("No build type selected, defaulting to RELEASE mode")
  message("Use -DBUILD_MODE=DEBUG or -DBUILD_MODE=RELEASE to specify build type")
  set(RELEASE ON)
endif()

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/src
)
set(COMMON_LIB_DIRS "")

# ====== MCCL for multi-GPU communication (replaces NCCL) ======
message("Building with MCCL")
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
find_package(MCCL REQUIRED)
set(CMAKE_MODULE_PATH "") # prevent the bugs for pytorch building

list(APPEND COMMON_HEADER_DIRS ${MCCL_INCLUDE_DIR})
list(APPEND COMMON_LIB_DIRS ${MCCL_LIBRARIES})

# ====== MUSA libraries (replaces CUDA::cudart) ======
set(COMMON_LIBS ${MUSART_LIBRARY})
if(MUBLAS_LIBRARY)
  list(APPEND COMMON_LIBS ${MUBLAS_LIBRARY})
endif()

# MUSA toolkit headers
include_directories(${MUSAToolkit_TARGET_DIR}/include)

# ====== Python Setup ======
set(PYTHON_PATH "python" CACHE STRING "Python path")
execute_process(COMMAND ${PYTHON_PATH} "-c" "import sysconfig;
print(sysconfig.get_paths()['include']);"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Python config Error.")
endif()
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

# ====== PyTorch + torch_musa Setup ======
execute_process(COMMAND ${PYTHON_PATH} "-c" "import os; import torch;
print(os.path.dirname(torch.__file__), end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
	message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
list(APPEND COMMON_HEADER_DIRS "${TORCH_INCLUDE_DIRS}")
list(APPEND COMMON_LIBS "${TORCH_LIBRARIES}")

# Find torch_musa package path for headers and libraries
execute_process(COMMAND ${PYTHON_PATH} "-c"
  "import sysconfig; print(sysconfig.get_paths()['purelib'], end='')"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYTHON_PACKAGE_PATH)
if (NOT _PYTHON_SUCCESS MATCHES 0)
  message(FATAL_ERROR "Cannot determine Python package path.")
endif()

# torch_musa headers (for at::musa::, MUSAContext.h, torch::kMUSA etc.)
list(APPEND COMMON_HEADER_DIRS "${PYTHON_PACKAGE_PATH}")

# torch_musa libraries
find_library(TorchMusa_LIBRARY NAMES musa_python
  PATHS "${PYTHON_PACKAGE_PATH}/torch_musa/lib")
find_library(MusaKernels_LIBRARY NAMES musa_kernels
  PATHS "${PYTHON_PACKAGE_PATH}/torch_musa/lib")
if(TorchMusa_LIBRARY AND MusaKernels_LIBRARY)
  list(APPEND COMMON_LIBS ${TorchMusa_LIBRARY} ${MusaKernels_LIBRARY})
  message(STATUS "TorchMusa: ${TorchMusa_LIBRARY}")
  message(STATUS "MusaKernels: ${MusaKernels_LIBRARY}")
else()
  message(WARNING "torch_musa libraries not found, some features may not link correctly")
endif()

# Let COMMON_HEADER_DIRS & COMMON_LIB_DIRS take effect
include_directories(${COMMON_HEADER_DIRS})
link_directories(${COMMON_LIB_DIRS})
link_libraries(${COMMON_LIBS})

# Should turn off CXX11 ABI if pytorch is built with CXX11 ABI off
execute_process(COMMAND ${PYTHON_PATH} "-c" "import torch;
print(torch._C._GLIBCXX_USE_CXX11_ABI,end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE USE_CXX11_ABI)
message("-- USE_CXX11_ABI=${USE_CXX11_ABI}")
if (USE_CXX11_ABI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# Let all executable targets go to bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add subdirectories
add_subdirectory(src)
