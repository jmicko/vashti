// CPU-only byte-offset proof. No SYCL headers, devices, allocations or model data.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "cpy-offset.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <limits>

static uint64_t checks = 0;
static void check(const std::array<int64_t, 4> & ne,
                  const std::array<size_t, 4> & nb, const int64_t block = 1) {
    const int64_t count = ne[0] * ne[1] * ne[2] * ne[3];
    assert(count > 0 && ne[0] % block == 0);
    for (int64_t i3 = 0; i3 < ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < ne[0]; i0 += block) {
                    const int64_t index = ((i3 * ne[2] + i2) * ne[1] + i1) * ne[0] + i0;
                    const size_t expected = size_t(i0 / block) * nb[0] + size_t(i1) * nb[1] +
                                            size_t(i2) * nb[2] + size_t(i3) * nb[3];
                    assert(ggml_sycl_cpy_byte_offset(index, ne[0], ne[1], ne[2],
                                                     nb[0], nb[1], nb[2], nb[3], block) == expected);
                    ++checks;
                }
}

int main() {
    static_assert(sizeof(size_t) == 8, "64-bit byte strides required");
    std::array<int, 4> axes{0, 1, 2, 3};
    const std::array<int64_t, 4> dims{7, 5, 3, 2};
    const std::array<size_t, 4> strides{4, 36, 216, 864};
    do {
        std::array<int64_t, 4> ne{};
        std::array<size_t, 4> nb{};
        for (int i = 0; i < 4; ++i) { ne[i] = dims[axes[i]]; nb[i] = strides[axes[i]]; }
        check(ne, nb);
    } while (std::next_permutation(axes.begin(), axes.end()));
    // Padded/two-row views isolate source and destination >2 and >4 GiB offsets
    // while keeping the logical tensor tiny. Quantized block offsets stay wide.
    for (uint64_t boundary : {UINT64_C(1) << 31, UINT64_C(1) << 32, UINT64_C(1) << 33}) {
        for (int axis = 1; axis < 4; ++axis) {
            std::array<int64_t, 4> ne{32, 1, 1, 1}; ne[axis] = 2;
            std::array<size_t, 4> nb{4, 128, 128, 128}; nb[axis] = boundary + 64;
            check(ne, nb);
            ne[0] = 96; nb[0] = 34; check(ne, nb, 32);
            nb[0] = 18; check(ne, nb, 32);
            ne[0] = 768; nb[0] = 84; check(ne, nb, 256);
        }
    }
    // Exact original-generation final F32 activation: index fits signed32,
    // byte offset exceeds signed32. Check every boundary and the last element.
    const int64_t pixels = INT64_C(1184) * 1776, total = pixels * 288;
    assert(total <= INT32_MAX);
    for (int64_t i : {INT64_C(0), INT64_C(1), INT64_C(536870911),
                      INT64_C(536870912), INT64_C(536870913), total - 1}) {
        const size_t offset = ggml_sycl_cpy_byte_offset(i, 1184, 1776, 288,
                                                       4, 1184 * 4, size_t(pixels) * 4,
                                                       size_t(total) * 4);
        assert(offset == size_t(i) * 4);
        ++checks;
    }
    assert(ggml_sycl_cpy_byte_offset(total - 1, 1184, 1776, 288, 4, 4736,
                                    size_t(pixels) * 4, size_t(total) * 4) == UINT64_C(2422407164));
    // >INT32 element-index proof without allocating or submitting a giant range.
    const int64_t index = INT64_C(2147483648) + 17;
    assert(ggml_sycl_cpy_byte_offset(index, 64, 65536, 1024, 4, 256,
                                    UINT64_C(16777216), UINT64_C(17179869184)) == size_t(index) * 4);
    ++checks;
    std::printf("copy_offset_cpu_pass checks=%llu device_execution=0\n", (unsigned long long)checks);
}
