//! Original six integer decoders. Every eight-value run has the same separately
//! rounded F32 factors, signed-byte values, RNE F16 bits and all-eight fault gate
//! as the existing direct8 image producer. Typed pair casts are lowered to the
//! original native pack/unpack operations before driver pipeline creation.
use super::*;
#[derive(CubeType, Clone)]
pub(crate) struct Decoded {
    d: f32,
    lo: u32,
    hi: u32,
}
#[cube]
pub(crate) fn byte(p: &Array<u32>, at: usize) -> u32 {
    (p[at / 4] >> u32::cast_from(at % 4 * 8)) & 255
}
#[cube]
pub(crate) fn halfword(p: &Array<u32>, at: usize) -> u32 {
    (p[at / 4] >> u32::cast_from(at % 4 * 8)) & 65535
}
#[cube]
pub(crate) fn word(p: &Array<u32>, at: usize) -> u32 {
    let shift = u32::cast_from(at % 4 * 8);
    let mut result = p[at / 4] >> shift;
    if shift != 0 {
        result |= p[at / 4 + 1] << (32 - shift);
    }
    result
}
#[cube]
pub(crate) fn half_to_float(bits: u32) -> f32 {
    let sign = (bits & 32768) << 16;
    let e = (bits >> 10) & 31;
    let m = bits & 1023;
    let mut out = sign;
    if e == 0 {
        if m != 0 {
            // Nonzero ten-bit mantissa: shift is exactly 1..10. Normalize
            // once instead of expanding a data-dependent loop at every row.
            let shift = m.leading_zeros() - 21;
            out |= ((113u32 - shift) << 23) | (((m << shift) & 1023) << 13);
        }
    } else if e == 31 {
        out |= 2139095040u32 | (m << 13);
    } else {
        out |= ((e + 112) << 23) | (m << 13);
    }
    f32::reinterpret(out)
}
/// Exact round-to-nearest-ties-even conversion; signed zero, subnormals,
/// normal carry and overflow are represented as bits before the final bitcast.
#[cube]
pub(crate) fn half_bits(value: f32) -> u32 {
    let bits = u32::reinterpret(value);
    let sign = (bits >> 16) & 32768;
    let magnitude = bits & 2147483647;
    let e = i32::cast_from(magnitude >> 23) - 112;
    let mut out = 0u32;
    if magnitude >= 2139095040u32 {
        out = 31744;
        if (magnitude & 8388607) != 0 {
            out = 32256;
        }
    } else if e >= 31 {
        out = 31744;
    } else if e <= 0 {
        if e >= -10 {
            let shift = u32::cast_from(14 - e);
            let significand = (magnitude & 8388607) | 8388608;
            out = (significand + (1u32 << (shift - 1)) - 1 + ((significand >> shift) & 1)) >> shift;
        }
    } else {
        out = ((magnitude + 4095 + ((magnitude >> 13) & 1)) >> 13) - 114688;
    }
    sign | out
}
/// These typed two-lane casts are lowered by kuno_quant_half_pack to the
/// exact original GLSL.std.450 operations before either native module path.
/// The compiler proof rejects missing packing, unsupported widths or a raw
/// narrowing FConvert. The independent software helpers above stay unchanged.
#[cube]
pub(crate) fn native_half_to_float(bits: u32) -> f32 {
    let pair = Vector::<f16, Const<2>>::reinterpret(bits);
    let widened = Vector::<f32, Const<2>>::cast_from(pair);
    widened[0]
}
#[cube]
pub(crate) fn native_pack_pair(x: f32, y: f32) -> u32 {
    let mut pair = Vector::<f32, Const<2>>::empty();
    pair[0] = x;
    pair[1] = y;
    let narrowed = Vector::<f16, Const<2>>::cast_from(pair);
    u32::reinterpret(narrowed)
}
#[cube]
pub(crate) fn signed_word(magnitude: u32, signs: u32) -> u32 {
    let mut out = 0u32;
    #[unroll]
    for j in 0..4u32 {
        let v = (magnitude >> (j * 8)) & 255;
        let mut s = i32::cast_from(v);
        if ((signs >> j) & 1) != 0 {
            s = -s;
        }
        out |= (u32::cast_from(s) & 255) << (j * 8);
    }
    out
}
#[cube]
pub(crate) fn iq4_word(codes: u32, high: bool) -> u32 {
    let mut out = 0u32;
    // Use the explicit runtime IR selector. The nested if-expression in an
    // unrolled shift was lowered as zero, so quarters2/3 read low nibbles.
    // The actual SPIR-V validator now requires this real0/4 selection.
    let nibble_shift = select(high, 4u32, 0u32);
    #[unroll]
    for j in 0..4u32 {
        let shift = j * 8 + nibble_shift;
        let index = (codes >> shift) & 15;
        let mut lut = 3215825025u32; // bfad9881: -127,-104,-83,-65
        if index >= 4 {
            lut = 4142587343u32;
        } // f6eaddcf
        if index >= 8 {
            lut = 639175937u32;
        } // 26190d01
        if index >= 12 {
            lut = 1901675829u32;
        } // 71594535
        out |= ((lut >> ((index % 4) * 8)) & 255) << (j * 8);
    }
    out
}
#[cube]
pub(crate) fn q2_word(codes: u32) -> u32 {
    let mut out = 0u32;
    #[unroll]
    for j in 0..4u32 {
        let v = i32::cast_from((codes >> (j * 2)) & 3) - 1;
        out |= (u32::cast_from(v) & 255) << (j * 8);
    }
    out
}
#[cube]
pub(crate) fn decode8(
    p: &Array<u32>,
    grid: &Array<u32>,
    row: usize,
    block: usize,
    quarter: usize,
    #[comptime] dtype: u32,
) -> Decoded {
    let group = block % 8;
    let mut d = 0.0f32;
    let mut lo = 0u32;
    let mut hi = 0u32;
    if comptime!(dtype == 18) {
        let base = row + block / 8 * 98;
        let codes = crate::strata_matrix::quant_cohort::decoder::halfword(
            p,
            base + 2 + group * 8 + quarter * 2,
        );
        let aux = crate::strata_matrix::quant_cohort::decoder::word(p, base + 66 + group * 4);
        let sc = (aux >> u32::cast_from(quarter * 7)) & 127;
        let signs = sc | ((sc.count_ones() & 1) << 7);
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(
            crate::strata_matrix::quant_cohort::decoder::halfword(p, base),
        ) * (0.5 + f32::cast_from(aux >> 28))
            * 0.5;
        lo = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[usize::cast_from(codes & 255)],
            signs,
        );
        hi = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[usize::cast_from(codes >> 8)],
            signs >> 4,
        );
    } else if comptime!(dtype == 21) {
        let base = row + block / 8 * 110;
        let codes = crate::strata_matrix::quant_cohort::decoder::halfword(
            p,
            base + 2 + group * 8 + quarter * 2,
        );
        let high = crate::strata_matrix::quant_cohort::decoder::byte(p, base + 66 + group);
        let first = (codes & 255) | (((high >> u32::cast_from(quarter * 2)) & 1) << 8);
        let second = (codes >> 8) | (((high >> u32::cast_from(quarter * 2 + 1)) & 1) << 8);
        let signs =
            crate::strata_matrix::quant_cohort::decoder::byte(p, base + 74 + group * 4 + quarter);
        let scale = (crate::strata_matrix::quant_cohort::decoder::byte(p, base + 106 + group / 2)
            >> u32::cast_from(group % 2 * 4))
            & 15;
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(
            crate::strata_matrix::quant_cohort::decoder::halfword(p, base),
        ) * f32::cast_from(1 + 2 * scale);
        lo = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[256 + usize::cast_from(first)],
            signs,
        );
        hi = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[256 + usize::cast_from(second)],
            signs >> 4,
        );
    } else if comptime!(dtype == 22) {
        let base = row + block / 8 * 82;
        let index =
            crate::strata_matrix::quant_cohort::decoder::byte(p, base + 2 + group * 4 + quarter)
                | (((crate::strata_matrix::quant_cohort::decoder::byte(p, base + 66 + group)
                    >> u32::cast_from(quarter * 2))
                    & 3)
                    << 8);
        let signs =
            crate::strata_matrix::quant_cohort::decoder::byte(p, base + 34 + group * 4 + quarter);
        let scale = (crate::strata_matrix::quant_cohort::decoder::byte(p, base + 74 + group)
            >> u32::cast_from(quarter / 2 * 4))
            & 15;
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(
            crate::strata_matrix::quant_cohort::decoder::halfword(p, base),
        ) * (0.5 + f32::cast_from(scale))
            * 0.25;
        lo = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[768 + usize::cast_from(index) * 2],
            signs,
        );
        hi = crate::strata_matrix::quant_cohort::decoder::signed_word(
            grid[769 + usize::cast_from(index) * 2],
            signs >> 4,
        );
    } else if comptime!(dtype == 23) {
        let base = row + block / 8 * 136;
        let header = crate::strata_matrix::quant_cohort::decoder::word(p, base);
        let low = (crate::strata_matrix::quant_cohort::decoder::byte(p, base + 4 + group / 2)
            >> u32::cast_from(group % 2 * 4))
            & 15;
        let scale = low | (((header >> u32::cast_from(16 + group * 2)) & 3) << 4);
        let codes = base + 8 + group * 16 + quarter % 2 * 8;
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(header & 65535)
            * f32::cast_from(i32::cast_from(scale) - 32);
        lo = crate::strata_matrix::quant_cohort::decoder::iq4_word(
            crate::strata_matrix::quant_cohort::decoder::word(p, codes),
            quarter >= 2,
        );
        hi = crate::strata_matrix::quant_cohort::decoder::iq4_word(
            crate::strata_matrix::quant_cohort::decoder::word(p, codes + 4),
            quarter >= 2,
        );
    } else if comptime!(dtype == 20) {
        let base = row + block * 18;
        let codes = base + 2 + quarter % 2 * 8;
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(
            crate::strata_matrix::quant_cohort::decoder::halfword(p, base),
        );
        lo = crate::strata_matrix::quant_cohort::decoder::iq4_word(
            crate::strata_matrix::quant_cohort::decoder::word(p, codes),
            quarter >= 2,
        );
        hi = crate::strata_matrix::quant_cohort::decoder::iq4_word(
            crate::strata_matrix::quant_cohort::decoder::word(p, codes + 4),
            quarter >= 2,
        );
    } else if comptime!(dtype == 42) {
        let base = row + block / 2 * 18;
        let codes = crate::strata_matrix::quant_cohort::decoder::halfword(
            p,
            base + 2 + block % 2 * 8 + quarter * 2,
        );
        d = crate::strata_matrix::quant_cohort::decoder::native_half_to_float(
            crate::strata_matrix::quant_cohort::decoder::halfword(p, base),
        );
        lo = crate::strata_matrix::quant_cohort::decoder::q2_word(codes & 255);
        hi = crate::strata_matrix::quant_cohort::decoder::q2_word(codes >> 8);
    }
    Decoded { d, lo, hi }
}
#[cube]
pub(crate) fn store8(
    p: &Array<u32>,
    grid: &Array<u32>,
    row: usize,
    col: usize,
    out: &mut SharedMemory<Vector<f16, Const<4>>>,
    first: usize,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] dtype: u32,
) {
    let decoded = crate::strata_matrix::quant_cohort::decoder::decode8(
        p,
        grid,
        row,
        col / 32,
        col % 32 / 8,
        dtype,
    );
    let mut a = Vector::<f16, Const<4>>::new(f16::new(0.0));
    let mut b = Vector::<f16, Const<4>>::new(f16::new(0.0));
    let mut good = true;
    #[unroll]
    for pair in 0..4usize {
        let mut packed = decoded.lo;
        if pair >= 2 {
            packed = decoded.hi;
        }
        let c0 = (packed >> u32::cast_from(pair % 2 * 16)) & 255;
        let c1 = (packed >> u32::cast_from(pair % 2 * 16 + 8)) & 255;
        let mut s0 = i32::cast_from(c0);
        let mut s1 = i32::cast_from(c1);
        if c0 >= 128 {
            s0 -= 256;
        }
        if c1 >= 128 {
            s1 -= 256;
        }
        let x = decoded.d * f32::cast_from(s0);
        let y = decoded.d * f32::cast_from(s1);
        let rounded = crate::strata_matrix::quant_cohort::decoder::native_pack_pair(x, y);
        let hx = rounded & 65535;
        let hy = rounded >> 16;
        good = good
            && x.abs() <= f32::MAX
            && y.abs() <= f32::MAX
            && (hx & 31744) != 31744
            && (hy & 31744) != 31744;
        let v = Vector::<f16, Const<2>>::reinterpret(hx | (hy << 16));
        if pair < 2 {
            a[pair * 2] = v[0];
            a[pair * 2 + 1] = v[1];
        } else {
            b[(pair - 2) * 2] = v[0];
            b[(pair - 2) * 2 + 1] = v[1];
        }
    }
    if !good {
        a = Vector::<f16, Const<4>>::new(f16::new(0.0));
        b = Vector::<f16, Const<4>>::new(f16::new(0.0));
        fault[0].fetch_or(128);
    }
    out[first] = a;
    out[first + 1] = b;
}
