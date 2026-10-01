#pragma once
#include <cstdint>

// Qwen 2.1 caches K/V for 32 layers at width 4096: 1 MiB/token in FP32,
// half that in FP16. Precision is separate from the working reserve: CUDA
// can stream weights with 3 GiB for a segment, while Vulkan conservatively
// retains 9 GiB for the full diffusion weights and working space.
// UTF-8 bytes conservatively bound prompt tokens; fixed wrappers get 128 more.
inline uint64_t kuno_qwen_prefix_cache_bytes(uint64_t reference_pixels,
                                             uint64_t prompt_bytes, bool guided, bool fp16) {
    if (prompt_bytes > UINT64_MAX - 128 || reference_pixels / 256 > UINT64_MAX - prompt_bytes - 128)
        return UINT64_MAX;
    const uint64_t tokens = reference_pixels / 256 + prompt_bytes + 128;
    const uint64_t per_token = (fp16 ? 512ull : 1024ull) * 1024 * (guided ? 2 : 1);
    return tokens > UINT64_MAX / per_token ? UINT64_MAX : tokens * per_token;
}
inline bool kuno_qwen_prefix_cache_fits(uint64_t budget, uint64_t reference_pixels,
                                       uint64_t prompt_bytes, bool guided, bool fp16 = false,
                                       uint64_t working_reserve = 9ull * 1024 * 1024 * 1024) {
    const uint64_t cache = kuno_qwen_prefix_cache_bytes(reference_pixels, prompt_bytes, guided, fp16);
    return budget >= working_reserve && cache <= budget - working_reserve;
}

inline bool kuno_qwen_host_prefix_cache_fits(uint64_t gpu_budget, uint64_t host_available,
                                            uint64_t reference_pixels, uint64_t prompt_bytes, bool guided) {
    constexpr uint64_t gib = 1024ull * 1024 * 1024;
    const uint64_t cache = kuno_qwen_prefix_cache_bytes(reference_pixels, prompt_bytes, guided, true);
    // Keep small caches on the GPU. For large edit caches, reserve room for
    // diffusion weights and workspace. The host reserve covers both pinned
    // Qwen model sets, VAE/workspace allocations, and additional headroom.
    const bool crowded = gpu_budget < 10 * gib || cache > gpu_budget - 10 * gib;
    return reference_pixels != 0 && cache >= gib && crowded && gpu_budget >= 3 * gib &&
           host_available >= 24 * gib && cache <= host_available - 24 * gib;
}
