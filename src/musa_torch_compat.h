// Workaround for c10::Half / MUSA __half incompatibility.
//
// MCC defines __CUDA__ which triggers PyTorch's __ldg overload for c10::Half
// (in c10/util/Half-inl.h), but MUSA's __half lacks implicit conversion to
// c10::Half. Temporarily suppress __CUDA__ while including PyTorch headers.
//
// Usage: #include "musa_torch_compat.h" instead of directly including
// <ATen/ATen.h>, <torch/extension.h>, etc. in .cu files compiled with -x musa.
#pragma once

#ifdef __MUSA__
#pragma push_macro("__CUDA__")
#undef __CUDA__
#endif

#include <ATen/ATen.h>
#include <torch/extension.h>

#ifdef __MUSA__
#pragma pop_macro("__CUDA__")
#endif
