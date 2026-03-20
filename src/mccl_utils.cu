/***************************************************************************************************
 * Adapted from SwiftTransformer(https://github.com/LLMServe/SwiftTransformer/blob/main/src/csrc/util/py_nccl.cc)
 **************************************************************************************************/

#include "mccl_utils.h"

std::vector<int64_t> generate_mccl_id() {
    mcclUniqueId mccl_id;
    mcclGetUniqueId(&mccl_id);
    std::vector<int64_t> ret;
    ret.resize(MCCL_UNIQUE_ID_BYTES / sizeof(int64_t));
    memcpy(ret.data(), mccl_id.internal, MCCL_UNIQUE_ID_BYTES);
    return ret;
}
