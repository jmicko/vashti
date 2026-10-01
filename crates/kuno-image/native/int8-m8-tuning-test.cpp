// Pure CPU acceptance for bounded internal Vulkan tuning values. No backend is
// initialized, and absent or malformed values must retain the legacy kernel.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>

#include "vulkan-int8-m8-tuning.h"

int main() {
    struct Accepted { const char* value; uint32_t m; uint32_t n; };
    const Accepted accepted[] = {{"2", 2, 2}, {"4", 4, 4}, {"4x8", 4, 8}, {"8x4", 8, 4}, {"8x8", 8, 8}};
    for (const auto& test : accepted) {
        const auto tiles = ggml_vk_parse_int8_m8_tiles(test.value);
        assert(tiles.m == test.m && tiles.n == test.n);
    }
    const char* rejected[] = {
        nullptr, "", "0", "1", "8", "2x2", "4x4", "4X8", " 4x8", "4x8 ",
        "4x08", "04x8", "4x8junk", "4x8\n", "8x0", "999999999999999999999999999999", "4x8x8"
    };
    for (const auto* value : rejected) {
        const auto tiles = ggml_vk_parse_int8_m8_tiles(value);
        assert(tiles.m == 0 && tiles.n == 0);
    }
    std::printf("int8_m8_tuning_passed accepted=%zu rejected=%zu\n",
                sizeof(accepted) / sizeof(accepted[0]), sizeof(rejected) / sizeof(rejected[0]));
}
