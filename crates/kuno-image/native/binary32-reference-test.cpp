// Independent integer-reference validation. No GGML/backend/GPU dependency.
#include "binary32-reference.hpp"
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#ifdef __SSE__
#include <xmmintrin.h>
#endif

static void check(bool value) { if (!value) throw 1; }
static float value(uint32_t bits) { float result; std::memcpy(&result, &bits, 4); return result; }
static uint32_t bits(float input) { uint32_t result; std::memcpy(&result, &input, 4); return result; }
static uint32_t mixed(uint32_t index) {
    index ^= index >> 16; index *= 0x7feb352dU;
    index ^= index >> 15; index *= 0x846ca68bU;
    return index ^ (index >> 16);
}
static uint32_t sample(uint32_t index, unsigned exponent) { return (mixed(index) & 0x807fffffU) | (exponent << 23); }
static uint32_t host_mul(uint32_t a, uint32_t b) { volatile float result = value(a) * value(b); return bits(result); }
static uint32_t host_add(uint32_t a, uint32_t b) { volatile float result = value(a) + value(b); return bits(result); }

int main() {
    try {
        rlimit limit{0, 0}; check(setrlimit(RLIMIT_CORE, &limit) == 0);
        check(std::fesetround(FE_TONEAREST) == 0 && std::fegetround() == FE_TONEAREST);
#ifdef __SSE__
        check((_mm_getcsr() & ((1U << 15) | (1U << 6))) == 0);
#endif
        struct Triple { uint32_t a, b, expected; };
        const Triple products[] = {
            {0x3f800001U, 0x3fc00000U, 0x3fc00002U},
            {0x3f800003U, 0x3fc00000U, 0x3fc00004U},
            {0xbf800001U, 0x3fc00000U, 0xbfc00002U},
            {0xbf800003U, 0x3fc00000U, 0xbfc00004U},
            {0x3f800001U, 0x3f7fffffU, 0x3f800000U},
            {0x3f800001U, 0x3f800001U, 0x3f800002U},
            {0x00800000U, 0x3f000000U, 0x00400000U},
            {0x00800001U, 0x3f000000U, 0x00400000U},
            {0x00000001U, 0x3f000000U, 0x00000000U},
            {0x00000003U, 0x3f000000U, 0x00000002U},
            {0x80000001U, 0x3f000000U, 0x80000000U},
            {0x7f7fffffU, 0x40000000U, 0x7f800000U},
            {0xff7fffffU, 0x40000000U, 0xff800000U},
            {0x00000000U, 0xbf800000U, 0x80000000U},
            {0x80000000U, 0xbf800000U, 0x00000000U}
        };
        const Triple sums[] = {
            {0x3f800000U, 0x33800000U, 0x3f800000U},
            {0x3f800001U, 0x33800000U, 0x3f800002U},
            {0xbf800000U, 0xb3800000U, 0xbf800000U},
            {0xbf800001U, 0xb3800000U, 0xbf800002U},
            {0x00800000U, 0x807fffffU, 0x00000001U},
            {0x00800000U, 0x007fffffU, 0x00ffffffU},
            {0x00000001U, 0x00000001U, 0x00000002U},
            {0x3dcccccdU, 0xbdcccccdU, 0x00000000U},
            {0x80000000U, 0x80000000U, 0x80000000U},
            {0x00000000U, 0x80000000U, 0x00000000U},
            {0x7f7fffffU, 0x7f7fffffU, 0x7f800000U},
            {0xff7fffffU, 0xff7fffffU, 0xff800000U},
            {0x3f800000U, 0x00000001U, 0x3f800000U},
            {0x3f800000U, 0x80000001U, 0x3f800000U},
            {0x3f800000U, 0x30000000U, 0x3f800000U}, // Distance 31.
            {0x3f800000U, 0x2f800000U, 0x3f800000U}, // Shortcut distance 32.
            {0xbf800000U, 0x2f800000U, 0xbf800000U},
            {0x7f800000U, 0xbf800000U, 0x7f800000U}
        };
        size_t known = 0;
        for (const auto & c : products) {
            check(kuno_binary32::mul(c.a, c.b) == c.expected && host_mul(c.a, c.b) == c.expected); ++known;
        }
        for (const auto & c : sums) {
            check(kuno_binary32::add(c.a, c.b) == c.expected && host_add(c.a, c.b) == c.expected);
            check(kuno_binary32::add(c.b, c.a) == c.expected && host_add(c.b, c.a) == c.expected); known += 2;
        }
        check(kuno_binary32::rounded_shift(1, 1) == 0 && kuno_binary32::rounded_shift(3, 1) == 2);
        check(kuno_binary32::rounded_shift(uint64_t(1) << 63, 64) == 0);
        check(kuno_binary32::rounded_shift((uint64_t(1) << 63) + 1, 64) == 1);
        check(kuno_binary32::rounded_shift(UINT64_MAX, 65) == 0); known += 5;
        bool rejected = false;
        try { (void)kuno_binary32::mul(0x7fc00001U, 0x3f800000U); } catch (const kuno_binary32::Error &) { rejected = true; }
        check(rejected); rejected = false;
        try { (void)kuno_binary32::add(0x7f800000U, 0xff800000U); } catch (const kuno_binary32::Error &) { rejected = true; }
        check(rejected); known += 2;
        size_t fma_distinct = 0, rounded_up = 0, rounded_down = 0, subnormal_inputs = 0, overflow_products = 0;
        constexpr uint32_t samples = 250000;
        for (uint32_t i = 0; i < samples; ++i) {
            const auto a = sample(i + 0x243f6a88U, i % 255U);
            const auto g = sample(i + 0x85a308d3U, (i * 37U) % 255U);
            const auto r = sample(i + 0x13198a2eU, (i * 61U) % 255U);
            const auto product = kuno_binary32::mul(a, g);
            check(product == host_mul(a, g));
            check(kuno_binary32::add(a, r) == host_add(a, r));
            const auto sum = kuno_binary32::add(r, product);
            check(sum == host_add(r, product));
            check(kuno_binary32::add(product, r) == sum);
            subnormal_inputs += ((a & 0x7f800000U) == 0 && (a & 0x007fffffU) != 0);
            overflow_products += (product & 0x7fffffffU) == 0x7f800000U;
            if (kuno_binary32::finite(product)) {
                const double exact_product = double(value(a)) * double(value(g));
                rounded_up += double(value(product)) > exact_product;
                rounded_down += double(value(product)) < exact_product;
            }
            fma_distinct += bits(std::fma(value(a), value(g), value(r))) != sum;
        }
        check(fma_distinct && rounded_up && rounded_down && subnormal_inputs && overflow_products);
        std::printf("binary32_reference known=%zu samples=%u fma_distinct=%zu rounded_up=%zu rounded_down=%zu subnormal_inputs=%zu overflow_products=%zu pass=1\n",
                    known, samples, fma_distinct, rounded_up, rounded_down, subnormal_inputs, overflow_products);
        return 0;
    } catch (...) { std::fputs("binary32_reference_error phase=1\n", stderr); return 2; }
}
