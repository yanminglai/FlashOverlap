#pragma once

#include <mccl.h>
#include <torch/extension.h>
#include <musa_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#define MCCL_CHECK(cmd)                                                                                           \
    do {                                                                                                          \
        mcclResult_t result = cmd;                                                                                \
        if (result != mcclSuccess) {                                                                              \
            printf("[ERROR] MCCL error %s:%d '%s' : %s\n", __FILE__, __LINE__, #cmd, mcclGetErrorString(result)); \
            exit(-1);                                                                                             \
        }                                                                                                         \
    } while (0)

std::vector<int64_t> generate_mccl_id();
