// Test-only nearest-even binary32 reference. Integer operations preserve every
// discarded bit; no production code or backend relies on this implementation.
#pragma once
#include <cstdint>

namespace kuno_binary32 {
struct Error {};
inline bool finite(uint32_t bits) { return (bits & 0x7f800000U) != 0x7f800000U; }
struct Parts { bool negative; uint32_t significand; int power; };
inline Parts parts(uint32_t bits) {
    if (!finite(bits)) throw Error{};
    const unsigned exponent = (bits >> 23) & 255U;
    return {bool(bits >> 31), (bits & 0x007fffffU) | (exponent ? 0x00800000U : 0U),
            exponent ? int(exponent) - 150 : -149};
}
inline unsigned highest(uint64_t value) {
    if (!value) throw Error{};
    unsigned result = 0;
    while (value >>= 1) ++result;
    return result;
}
inline uint64_t rounded_shift(uint64_t value, unsigned shift) {
    if (!shift) return value;
    if (shift > 64) return 0;
    if (shift == 64) return value > (uint64_t(1) << 63) ? 1 : 0;
    const auto quotient = value >> shift;
    const auto remainder = value & ((uint64_t(1) << shift) - 1);
    const auto halfway = uint64_t(1) << (shift - 1);
    return quotient + (remainder > halfway || (remainder == halfway && (quotient & 1)));
}
inline uint32_t round(bool negative, uint64_t significand, int power) {
    const uint32_t sign = negative ? 0x80000000U : 0;
    if (!significand) return sign;
    const int exponent = power + int(highest(significand));
    if (exponent >= 128) return sign | 0x7f800000U;
    int target_power = exponent - 23;
    if (target_power < -149) target_power = -149;
    const int shift = target_power - power;
    uint64_t rounded;
    if (shift >= 0) rounded = rounded_shift(significand, unsigned(shift));
    else {
        if (-shift >= 64 || significand > (UINT64_MAX >> unsigned(-shift))) throw Error{};
        rounded = significand << unsigned(-shift);
    }
    if (!rounded) return sign;
    if (target_power == -149 && rounded < 0x00800000U) return sign | uint32_t(rounded);
    if (rounded >= 0x01000000U) { rounded >>= 1; ++target_power; }
    const int encoded_exponent = target_power + 150;
    if (encoded_exponent >= 255) return sign | 0x7f800000U;
    if (encoded_exponent < 1 || rounded < 0x00800000U || rounded >= 0x01000000U) throw Error{};
    return sign | (uint32_t(encoded_exponent) << 23) | (uint32_t(rounded) & 0x007fffffU);
}
inline uint32_t mul(uint32_t a, uint32_t b) {
    const auto left = parts(a), right = parts(b);
    return round(left.negative != right.negative,
                 uint64_t(left.significand) * right.significand, left.power + right.power);
}
inline uint32_t add(uint32_t a, uint32_t b) {
    // Finite inputs can overflow the preceding MUL. Its infinity added to a
    // finite residual is exact; NaNs and two infinities are outside this oracle.
    if (!finite(a) || !finite(b)) {
        if ((a & 0x7fffffffU) == 0x7f800000U && finite(b)) return a;
        if ((b & 0x7fffffffU) == 0x7f800000U && finite(a)) return b;
        throw Error{};
    }
    auto left = parts(a), right = parts(b);
    if (!left.significand && !right.significand) return (a & b) & 0x80000000U;
    if (!left.significand) return b;
    if (!right.significand) return a;
    if (left.power < right.power) { const auto temp = left; left = right; right = temp; const auto bits = a; a = b; b = bits; }
    const unsigned distance = unsigned(left.power - right.power);
    // A binary32 significand has at most 24 bits. Beyond 31 powers, the smaller
    // operand is strictly below half of the larger operand's binary32 ULP.
    // The larger operand is already representable, so nearest-even retains it.
    if (distance > 31) return a;
    const uint64_t l = uint64_t(left.significand) << distance, r = right.significand;
    if (left.negative == right.negative) return round(left.negative, l + r, right.power);
    if (l == r) return 0; // Exact cancellation is positive zero in nearest-even.
    return l > r ? round(left.negative, l - r, right.power) : round(right.negative, r - l, right.power);
}
} // namespace kuno_binary32
