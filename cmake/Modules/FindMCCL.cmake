# Find the MCCL (MUSA Collective Communication Library) libraries
#
# Based on FindNCCL.cmake, adapted for Moore Threads MCCL.
#
# The following variables are optionally searched for defaults
#  MCCL_ROOT: Base directory where all MCCL components are found
#  MCCL_INCLUDE_DIR: Directory where MCCL header is found
#  MCCL_LIB_DIR: Directory where MCCL library is found
#
# The following are set after configuration is done:
#  MCCL_FOUND
#  MCCL_INCLUDE_DIRS
#  MCCL_LIBRARIES

set(MCCL_INCLUDE_DIR $ENV{MCCL_INCLUDE_DIR} CACHE PATH "Folder contains MCCL headers")
set(MCCL_LIB_DIR $ENV{MCCL_LIB_DIR} CACHE PATH "Folder contains MCCL libraries")

if ($ENV{MCCL_ROOT_DIR})
  message(WARNING "MCCL_ROOT_DIR is deprecated. Please set MCCL_ROOT instead.")
endif()

# Search hints: MCCL_ROOT, MCCL_ROOT_DIR env, MUSA toolkit path
list(APPEND MCCL_ROOT $ENV{MCCL_ROOT_DIR} $ENV{MCCL_ROOT})
if (DEFINED CUSTOM_MUSA_PATH)
  list(APPEND MCCL_ROOT ${CUSTOM_MUSA_PATH})
endif()
list(APPEND MCCL_ROOT /usr/local/musa /usr/local /usr)

# Compatible layer for CMake <3.12
list(APPEND CMAKE_PREFIX_PATH ${MCCL_ROOT})

find_path(MCCL_INCLUDE_DIRS
  NAMES mccl.h
  HINTS ${MCCL_INCLUDE_DIR} ${MCCL_ROOT}
  PATH_SUFFIXES include)

if (USE_STATIC_MCCL)
  message(STATUS "USE_STATIC_MCCL is set. Linking with static MCCL library.")
  set(MCCL_LIBNAME "mccl_static")
else()
  set(MCCL_LIBNAME "mccl")
endif()

find_library(MCCL_LIBRARIES
  NAMES ${MCCL_LIBNAME}
  HINTS ${MCCL_LIB_DIR} ${MCCL_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MCCL DEFAULT_MSG MCCL_INCLUDE_DIRS MCCL_LIBRARIES)

if(MCCL_FOUND)
  message(STATUS "Found MCCL (include: ${MCCL_INCLUDE_DIRS}, library: ${MCCL_LIBRARIES})")
  mark_as_advanced(MCCL_ROOT_DIR MCCL_INCLUDE_DIRS MCCL_LIBRARIES)
endif()
