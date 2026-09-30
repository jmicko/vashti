#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdint>

inline bool kuno_qwen_mlp_f16_experiment(bool qwen21, bool vulkan) {
    const char* value = std::getenv("KUNO_EXPERIMENTAL_QWEN_MLP_F16");
    return qwen21 && vulkan && value != nullptr && std::strcmp(value, "1") == 0;
}
inline uint64_t kuno_qwen_working_reserve(bool cuda, bool mlp_f16) {
    // MLP conversion adds 4.5 GiB; round the 9+4.5 GiB Vulkan baseline up
    // to 14 GiB. This remains conservative, not a measured peak guarantee.
    return (mlp_f16 ? 14ull : (cuda ? 3ull : 9ull)) * 1024 * 1024 * 1024;
}
inline bool kuno_qwen_mlp_f16_host_fits(uint64_t available) {
    // Existing pinned models/workspace reserve: 24 GiB. Add 4.5 GiB of
    // owned dense MLPs and bounded conversion temporaries; round up to 32.
    return available >= 32ull * 1024 * 1024 * 1024;
}
