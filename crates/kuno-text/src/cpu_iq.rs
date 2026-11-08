// Original GGML b11232 IQ layouts/codebooks, independently pinned in
// tests/fixtures/iq-b11232. Decode only eight weights at a time into registers.
// No full-weight expansion, activation requantization or reduction change.
use super::*;
use crate::quant::{IQ3_XXS_GRID, IQ4_NL_VALUES};

#[inline]
fn scale(block: &[u8]) -> Result<f32> {
    finite(
        crate::quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]])),
        "CPU expert non-finite packed weight",
    )
}

#[inline]
fn iq3_word(block: &[u8], group: usize) -> u32 {
    let first = 66 + group * 4;
    u32::from_le_bytes(block[first..first + 4].try_into().unwrap())
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq3_vector(block: &[u8], group: usize, octet: usize, word: u32, factor: f32) -> __m256 {
    let first = 2 + group * 8 + octet * 2;
    let a = IQ3_XXS_GRID[usize::from(block[first])];
    let b = IQ3_XXS_GRID[usize::from(block[first + 1])];
    // The two original little-endian quartets occupy the low eight byte lanes.
    let grid = _mm_setr_epi32(a as i32, b as i32, 0, 0);
    let magnitude = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(grid));
    let code = (word >> (7 * octet)) & 127;
    let signs = code | ((code.count_ones() & 1) << 7);
    let bits = _mm256_and_si256(
        _mm256_srlv_epi32(
            _mm256_set1_epi32(signs as i32),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
        ),
        _mm256_set1_epi32(1),
    );
    // Multiplication by +/-1 is exactly a sign-bit toggle for every finite
    // weight, including signed zero. Keep factor*magnitude as its own f32
    // multiply before applying that sign, exactly as the scalar/C decoder.
    _mm256_xor_ps(
        _mm256_mul_ps(_mm256_set1_ps(factor), magnitude),
        _mm256_castsi256_ps(_mm256_slli_epi32::<31>(bits)),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn iq4_vectors(codes: &[u8], factor: f32) -> [__m256; 4] {
    // SAFETY: the canonical signed table and original nibble span each contain
    // exactly16 bytes. AVX2 includes the byte-shuffle instruction used here.
    let (table, bytes) = unsafe {
        (
            _mm_loadu_si128(IQ4_NL_VALUES.as_ptr().cast()),
            _mm_loadu_si128(codes.as_ptr().cast()),
        )
    };
    let mask = _mm_set1_epi8(15);
    let low = _mm_shuffle_epi8(table, _mm_and_si128(bytes, mask));
    let high = _mm_shuffle_epi8(table, _mm_and_si128(_mm_srli_epi16::<4>(bytes), mask));
    let factor = _mm256_set1_ps(factor);
    [
        _mm256_mul_ps(factor, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low))),
        _mm256_mul_ps(
            factor,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(low))),
        ),
        _mm256_mul_ps(factor, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high))),
        _mm256_mul_ps(
            factor,
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(high))),
        ),
    ]
}

#[inline]
fn iq4_factor(block: &[u8], group: usize, scale: f32) -> f32 {
    let low = (block[4 + group / 2] >> (4 * (group % 2))) & 15;
    let high = (u16::from_le_bytes([block[2], block[3]]) >> (2 * group)) & 3;
    let subscale = i16::from(low) | ((high as i16) << 4);
    scale * f32::from(subscale - 32)
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn accumulate(sums: &mut Sums, weights: [__m256; 4], input: &[f32], first: usize) {
    for (vector, weight) in weights.into_iter().enumerate() {
        let column = first + vector * 8;
        let bank = column % LANES / 8;
        // SAFETY: caller passes an original complete32-value span inside input.
        let activation = unsafe { _mm256_loadu_ps(input.as_ptr().add(column)) };
        sums.0[bank] = _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn dot(
    dtype: TensorType,
    packed: &[u8],
    input: &[f32],
    cancel: &AtomicBool,
) -> Result<f32> {
    let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
    match dtype {
        TensorType::Iq3Xxs => {
            for (index, block) in packed.chunks_exact(98).enumerate() {
                check_cancel(cancel)?;
                let scale = scale(block)?;
                for group in 0..8 {
                    let word = iq3_word(block, group);
                    let factor = scale * (0.5 + (word >> 28) as f32) * 0.5;
                    for octet in 0..4 {
                        let column = index * 256 + group * 32 + octet * 8;
                        // SAFETY: full original98-byte block, octet<4, group<8;
                        // column+8 lies in its complete256-value input span.
                        let (weight, activation) = unsafe {
                            (
                                iq3_vector(block, group, octet, word, factor),
                                _mm256_loadu_ps(input.as_ptr().add(column)),
                            )
                        };
                        let bank = column % LANES / 8;
                        sums.0[bank] =
                            _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
                    }
                }
            }
        }
        TensorType::Iq4Nl => {
            for (index, block) in packed.chunks_exact(18).enumerate() {
                check_cancel(cancel)?;
                let scale = scale(block)?;
                // SAFETY: original18-byte block contains16 packed nibble bytes;
                // first=index*32 identifies its complete32-value input span.
                unsafe {
                    accumulate(
                        &mut sums,
                        iq4_vectors(&block[2..18], scale),
                        input,
                        index * 32,
                    );
                }
            }
        }
        TensorType::Iq4Xs => {
            for (index, block) in packed.chunks_exact(136).enumerate() {
                check_cancel(cancel)?;
                let scale = scale(block)?;
                for group in 0..8 {
                    let first = 8 + group * 16;
                    let factor = iq4_factor(block, group, scale);
                    // SAFETY: group<8 bounds16 bytes inside the original block;
                    // index*256+group*32 identifies its complete input span.
                    unsafe {
                        accumulate(
                            &mut sums,
                            iq4_vectors(&block[first..first + 16], factor),
                            input,
                            index * 256 + group * 32,
                        );
                    }
                }
            }
        }
        _ => return Err(Error::InvalidFormat("CPU packed IQ SIMD layout")),
    }
    // SAFETY: AVX2 enabled, all64 original lane accumulators are initialized.
    unsafe { finish(&sums) }
}

#[cfg(test)]
pub(super) fn test_decode(dtype: TensorType, packed: &[u8], out: &mut [f32]) -> Result<()> {
    if !avx2_available() {
        return Err(Error::InvalidFormat("CPU expert AVX2 unavailable"));
    }
    if !matches!(
        dtype,
        TensorType::Iq3Xxs | TensorType::Iq4Nl | TensorType::Iq4Xs
    ) {
        return Err(Error::InvalidFormat("CPU packed IQ SIMD layout"));
    }
    if packed.len() != dtype.block_bytes() as usize || out.len() != dtype.block_elements() as usize
    {
        return Err(Error::InvalidBufferLength);
    }
    // Reject bad geometry and non-finite scales before the first output write.
    let scale = scale(packed)?;
    // SAFETY: AVX2 detected, the entire original block/output were validated.
    unsafe { decode_block(dtype, packed, out, scale) };
    Ok(())
}

#[cfg(test)]
#[target_feature(enable = "avx2")]
unsafe fn decode_block(dtype: TensorType, block: &[u8], out: &mut [f32], scale: f32) {
    if dtype == TensorType::Iq3Xxs {
        for group in 0..8 {
            let word = iq3_word(block, group);
            let factor = scale * (0.5 + (word >> 28) as f32) * 0.5;
            for octet in 0..4 {
                // SAFETY: complete original98-byte block and256-value output;
                // each disjoint8-lane store is within group*32+octet*8+8<=256.
                unsafe {
                    _mm256_storeu_ps(
                        out.as_mut_ptr().add(group * 32 + octet * 8),
                        iq3_vector(block, group, octet, word, factor),
                    );
                }
            }
        }
    } else {
        let groups = if dtype == TensorType::Iq4Nl { 1 } else { 8 };
        for group in 0..groups {
            let (first, factor) = if dtype == TensorType::Iq4Nl {
                (2, scale)
            } else {
                (8 + group * 16, iq4_factor(block, group, scale))
            };
            // SAFETY: preflight established a complete original nibble span.
            let vectors = unsafe { iq4_vectors(&block[first..first + 16], factor) };
            for (vector, weight) in vectors.into_iter().enumerate() {
                // SAFETY: each disjoint8-lane store stays inside the32-value
                // subgroup, and all subgroups fit the validated complete output.
                unsafe { _mm256_storeu_ps(out.as_mut_ptr().add(group * 32 + vector * 8), weight) };
            }
        }
    }
}
