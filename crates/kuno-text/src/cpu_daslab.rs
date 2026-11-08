// Original b11232 packed IQ2_S/IQ3_S/Q5_K/Q2_0, F32 activations and the
// unchanged original64-lane tree. Eight weights at a time live in registers;
// this path never expands a row or requantizes an activation. See ../NOTICE.
use super::*;
use crate::quant::{IQ2_S_GRID, IQ3_S_GRID};

#[inline]
fn scales(dtype: TensorType, block: &[u8]) -> Result<(f32, f32)> {
    let scale = finite(
        crate::quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]])),
        "CPU expert non-finite packed weight",
    )?;
    let minimum = if dtype == TensorType::Q5K {
        finite(
            crate::quant::f16_to_f32(u16::from_le_bytes([block[2], block[3]])),
            "CPU expert non-finite packed weight",
        )?
    } else {
        0.0
    };
    Ok((scale, minimum))
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn signed_magnitudes(grid: __m128i, signs: u8, factor: f32) -> __m256 {
    let magnitudes = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(grid));
    let signs = _mm256_and_si256(
        _mm256_srlv_epi32(
            _mm256_set1_epi32(i32::from(signs)),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        ),
        _mm256_set1_epi32(1),
    );
    // For finite weights, sign multiplication by +/-1 toggles exactly this
    // bit, including signed zero, after the original factor*magnitude round.
    _mm256_xor_ps(
        _mm256_mul_ps(_mm256_set1_ps(factor), magnitudes),
        _mm256_castsi256_ps(_mm256_slli_epi32::<31>(signs)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn vector(
    dtype: TensorType,
    block: &[u8],
    column: usize,
    scale: f32,
    minimum: f32,
) -> __m256 {
    match dtype {
        TensorType::Iq2S => {
            let group = column / 32;
            let octet = column % 32 / 8;
            let index = usize::from(block[2 + group * 4 + octet])
                | ((usize::from(block[66 + group]) << (8 - 2 * octet)) & 0x300);
            let word = IQ2_S_GRID[index];
            let grid = _mm_setr_epi32(word as i32, (word >> 32) as i32, 0, 0);
            let subscale = (block[74 + group] >> (4 * (octet / 2))) & 15;
            let factor = scale * (0.5 + f32::from(subscale)) * 0.25;
            // SAFETY: caller has AVX2; original82-byte block/group/octets
            // bound both the public1024-entry table and full8-bit sign field.
            unsafe { signed_magnitudes(grid, block[34 + group * 4 + octet], factor) }
        }
        TensorType::Iq3S => {
            let group = column / 32;
            let octet = column % 32 / 8;
            let first = 2 + group * 8 + octet * 2;
            let high = usize::from(block[66 + group]);
            let a = IQ3_S_GRID[usize::from(block[first]) | ((high << (8 - 2 * octet)) & 256)];
            let b = IQ3_S_GRID[usize::from(block[first + 1]) | ((high << (7 - 2 * octet)) & 256)];
            let grid = _mm_setr_epi32(a as i32, b as i32, 0, 0);
            let subscale = (block[106 + group / 2] >> (4 * (group % 2))) & 15;
            let factor = scale * f32::from(1 + 2 * subscale);
            // SAFETY: caller has AVX2; original110-byte block bounds the two
            // public512-entry quartets and all8 original stored sign bits.
            unsafe { signed_magnitudes(grid, block[74 + group * 4 + octet], factor) }
        }
        TensorType::Q5K => {
            let group = column / 64;
            let side = column % 64 / 32;
            let first = column % 32;
            let (factor, bias) = q4_factors(&block[4..16], group * 2 + side, scale, minimum);
            // SAFETY: original176-byte block, group<4, side<2, first<=24:
            // both8-byte loads are complete inside their32/128-byte fields.
            let (low, high) = unsafe {
                (
                    _mm_loadl_epi64(block.as_ptr().add(48 + group * 32 + first).cast()),
                    _mm_loadl_epi64(block.as_ptr().add(16 + first).cast()),
                )
            };
            let low = _mm256_cvtepu8_epi32(low);
            let low = if side == 0 {
                _mm256_and_si256(low, _mm256_set1_epi32(15))
            } else {
                _mm256_srli_epi32::<4>(low)
            };
            let high = _mm256_and_si256(
                _mm256_srlv_epi32(
                    _mm256_cvtepu8_epi32(high),
                    _mm256_set1_epi32((2 * group + side) as i32),
                ),
                _mm256_set1_epi32(1),
            );
            let quant = _mm256_or_si256(low, _mm256_slli_epi32::<4>(high));
            _mm256_sub_ps(
                _mm256_mul_ps(_mm256_set1_ps(factor), _mm256_cvtepi32_ps(quant)),
                _mm256_set1_ps(bias),
            )
        }
        TensorType::Q2_0 => {
            // Original64-value blocks have consecutive four2-bit codes per
            // byte. In particular the fourth code is+2, not a ternary alias.
            let a = i32::from(block[2 + column / 4]);
            let b = i32::from(block[3 + column / 4]);
            let codes = _mm256_and_si256(
                _mm256_srlv_epi32(
                    _mm256_setr_epi32(a, a, a, a, b, b, b, b),
                    _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6),
                ),
                _mm256_set1_epi32(3),
            );
            let signed = _mm256_sub_epi32(codes, _mm256_set1_epi32(1));
            _mm256_mul_ps(_mm256_cvtepi32_ps(signed), _mm256_set1_ps(scale))
        }
        _ => unreachable!(),
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot(
    dtype: TensorType,
    packed: &[u8],
    input: &[f32],
    cancel: &AtomicBool,
) -> Result<f32> {
    if !supported(dtype) {
        return Err(Error::InvalidFormat("CPU packed DASLab SIMD layout"));
    }
    let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
    let width = dtype.block_elements() as usize;
    for (index, block) in packed
        .chunks_exact(dtype.block_bytes() as usize)
        .enumerate()
    {
        check_cancel(cancel)?;
        let (scale, minimum) = scales(dtype, block)?;
        for local in (0..width).step_by(8) {
            let column = index * width + local;
            // SAFETY: dot_row proved complete packed/input geometry; local+8
            // is within this original64/256-value block. AVX2 is enabled.
            let (weight, activation) = unsafe {
                (
                    vector(dtype, block, local, scale, minimum),
                    _mm256_loadu_ps(input.as_ptr().add(column)),
                )
            };
            let bank = column % LANES / 8;
            sums.0[bank] = _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
        }
    }
    // SAFETY: all64 original lane accumulators were initialized, AVX2 enabled.
    unsafe { finish(&sums) }
}

fn supported(dtype: TensorType) -> bool {
    matches!(
        dtype,
        TensorType::Iq2S | TensorType::Iq3S | TensorType::Q5K | TensorType::Q2_0
    )
}

#[cfg(test)]
pub(super) fn test_decode(dtype: TensorType, packed: &[u8], out: &mut [f32]) -> Result<()> {
    if !avx2_available() {
        return Err(Error::InvalidFormat("CPU expert AVX2 unavailable"));
    }
    if !supported(dtype) {
        return Err(Error::InvalidFormat("CPU packed DASLab SIMD layout"));
    }
    if packed.len() != dtype.block_bytes() as usize || out.len() != dtype.block_elements() as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    // Validate geometry AND both Q5 half scales before any output write.
    let (scale, minimum) = scales(dtype, packed)?;
    for local in (0..out.len()).step_by(8) {
        // SAFETY: runtime ISA detection, complete original block and disjoint
        //8-lane owning output span validated before the first write.
        unsafe {
            _mm256_storeu_ps(
                out.as_mut_ptr().add(local),
                vector(dtype, packed, local, scale, minimum),
            );
        }
    }
    Ok(())
}
