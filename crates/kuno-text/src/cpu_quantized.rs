//! Explicit experimental Q8 activation dots for original packed CPU experts.
//! This changes rounding/activation precision: it is NOT original64 parity.
//! Quantize each 32-value input block once per operator, reuse across rows.
//! Only Q4_K, Q5_0 and Q8_0 are accepted; never select this after a failure.
use super::*;

const GROUP: usize = 32;
const GROUPS: usize = MAX_HIDDEN / GROUP;

struct ResidualInput {
    quants: [i8; MAX_HIDDEN],
    scales: [f32; GROUPS],
    sums: [i32; GROUPS],
}
impl ResidualInput {
    fn empty() -> Self {
        Self {
            quants: [0; MAX_HIDDEN],
            scales: [0.; GROUPS],
            sums: [0; GROUPS],
        }
    }
}
impl Zeroize for ResidualInput {
    fn zeroize(&mut self) {
        self.quants.zeroize();
        self.scales.zeroize();
        self.sums.zeroize();
    }
}
pub(super) struct Input {
    quants: [i8; MAX_HIDDEN],
    scales: [f32; GROUPS],
    sums: [i32; GROUPS],
    columns: usize,
    refinements: [ResidualInput; 2],
    refined: bool,
}
impl Input {
    pub(super) fn empty() -> Self {
        Self {
            quants: [0; MAX_HIDDEN],
            scales: [0.; GROUPS],
            sums: [0; GROUPS],
            columns: 0,
            refinements: std::array::from_fn(|_| ResidualInput::empty()),
            refined: false,
        }
    }
    pub(super) fn fill(&mut self, values: &[f32], cancel: &AtomicBool) -> Result<()> {
        self.zeroize();
        if values.is_empty() || values.len() > MAX_HIDDEN || !values.len().is_multiple_of(GROUP) {
            return Err(Error::InvalidBufferLength);
        }
        for (group, block) in values.chunks_exact(GROUP).enumerate() {
            check_cancel(cancel)?;
            let mut maximum = Zeroizing::new(0f32);
            for &value in block {
                finite(value, "CPU quantized non-finite input")?;
                *maximum = maximum.max(value.abs());
            }
            let scale = *maximum / 127.;
            self.scales[group] = scale;
            // All-zero and underflowed subnormal blocks become zero. This is
            // an explicit approximation, never claimed as lossless decoding.
            if scale > 0. {
                for (index, &value) in block.iter().enumerate() {
                    let quant = (value / scale).round_ties_even().clamp(-127., 127.) as i8;
                    self.quants[group * GROUP + index] = quant;
                    self.sums[group] += i32::from(quant);
                }
            }
        }
        self.columns = values.len();
        Ok(())
    }
    pub(super) fn fill_refined(&mut self, values: &[f32], cancel: &AtomicBool) -> Result<()> {
        self.fill(values, cancel)?;
        for (group, block) in values.chunks_exact(GROUP).enumerate() {
            check_cancel(cancel)?;
            let mut residual = Zeroizing::new([0f32; GROUP]);
            for (i, &v) in block.iter().enumerate() {
                residual[i] = v - f32::from(self.quants[group * GROUP + i]) * self.scales[group];
            }
            for refinement in &mut self.refinements {
                check_cancel(cancel)?;
                let scale = residual.iter().map(|v| v.abs()).fold(0f32, f32::max) / 127.;
                refinement.scales[group] = scale;
                if scale > 0. {
                    for (i, value) in residual.iter_mut().enumerate() {
                        let q = (*value / scale).round_ties_even().clamp(-127., 127.) as i8;
                        refinement.quants[group * GROUP + i] = q;
                        refinement.sums[group] += i32::from(q);
                        *value -= f32::from(q) * scale;
                    }
                }
            }
        }
        self.refined = true;
        Ok(())
    }
}
impl Zeroize for Input {
    fn zeroize(&mut self) {
        self.quants.zeroize();
        self.scales.zeroize();
        self.sums.zeroize();
        self.columns.zeroize();
        self.refinements.zeroize();
        self.refined.zeroize();
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        self.zeroize();
    }
}

pub(super) fn supported(kind: TensorType) -> bool {
    matches!(kind, TensorType::Q4K | TensorType::Q5_0 | TensorType::Q8_0)
}

#[allow(unsafe_code)]
pub(super) fn dot(
    kind: TensorType,
    packed: &[u8],
    input: &Input,
    cancel: &AtomicBool,
) -> Result<f32> {
    if !supported(kind) {
        return Err(Error::InvalidFormat("CPU quantized format unsupported"));
    }
    if input.columns == 0 || kind.row_bytes(input.columns as u64)? != packed.len() as u64 {
        return Err(Error::InvalidBufferLength);
    }
    if !avx2_available() {
        return Err(Error::InvalidFormat("CPU quantized AVX2 unavailable"));
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // SAFETY: runtime ISA check; checked original packed row geometry and
        // fixed, bounded Q8 input owner. Loads below never cross block bounds.
        unsafe { avx::run(kind, packed, input, cancel) }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    Err(Error::InvalidFormat("CPU quantized AVX2 unavailable"))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod avx {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    unsafe fn integer_dot(unsigned: __m256i, signed: __m256i) -> i32 {
        // Maximum adjacent sum is 2*128*127=32512, below i16 saturation.
        // Q4/Q5 unsigned magnitudes are smaller; Q8 uses abs(weight)<=128.
        let pairs = _mm256_maddubs_epi16(unsigned, signed);
        let groups = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
        // Exact i32 sum stays in registers. All32 products fit within520192;
        // no application stack array or changed rounding is introduced.
        let halves = _mm_add_epi32(
            _mm256_castsi256_si128(groups),
            _mm256_extracti128_si256::<1>(groups),
        );
        let pairs = _mm_hadd_epi32(halves, halves);
        _mm_cvtsi128_si32(_mm_hadd_epi32(pairs, pairs))
    }
    #[target_feature(enable = "avx2")]
    unsafe fn projection(
        codes: __m256i,
        signs: Option<__m256i>,
        input: &Input,
        index: usize,
        factor: f32,
        bias: f32,
    ) -> f32 {
        // Original packed weight codes reused across three activation passes;
        // no weight requantization or private row dequantization is introduced.
        let unsigned = codes;
        let signed = unsafe { _mm256_loadu_si256(input.quants.as_ptr().add(index * GROUP).cast()) };
        let signed = if let Some(signs) = signs {
            _mm256_sign_epi8(signed, signs)
        } else {
            signed
        };
        let mut result = Zeroizing::new(
            input.scales[index]
                * (factor * unsafe { integer_dot(unsigned, signed) } as f32
                    - bias * input.sums[index] as f32),
        );
        if input.refined {
            for refinement in &input.refinements {
                let signed = unsafe {
                    _mm256_loadu_si256(refinement.quants.as_ptr().add(index * GROUP).cast())
                };
                let signed = if let Some(signs) = signs {
                    _mm256_sign_epi8(signed, signs)
                } else {
                    signed
                };
                *result += refinement.scales[index]
                    * (factor * unsafe { integer_dot(unsigned, signed) } as f32
                        - bias * refinement.sums[index] as f32);
            }
        }
        *result
    }
    fn half(bytes: &[u8]) -> Result<f32> {
        finite(
            crate::quant::f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])),
            "CPU quantized non-finite packed scale",
        )
    }
    fn scale(bytes: &[u8], group: usize) -> u8 {
        if group < 4 {
            bytes[group] & 63
        } else {
            (bytes[group + 4] & 15) | ((bytes[group - 4] >> 6) << 4)
        }
    }
    fn minimum(bytes: &[u8], group: usize) -> u8 {
        if group < 4 {
            bytes[group + 4] & 63
        } else {
            (bytes[group + 4] >> 4) | ((bytes[group] >> 6) << 4)
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn run(
        kind: TensorType,
        packed: &[u8],
        input: &Input,
        cancel: &AtomicBool,
    ) -> Result<f32> {
        let mut result = Zeroizing::new(0f32);
        match kind {
            TensorType::Q4K => {
                for (block, bytes) in packed.chunks_exact(144).enumerate() {
                    check_cancel(cancel)?;
                    let d = half(bytes)?;
                    let dmin = half(&bytes[2..])?;
                    for group in 0..8 {
                        let index = block * 8 + group;
                        // SAFETY: 32-byte nibble block at16+(group/2)*32
                        // inside144 bytes; input has exactly8 groups/block.
                        let codes = unsafe {
                            let raw = _mm256_loadu_si256(
                                bytes.as_ptr().add(16 + (group / 2) * 32).cast(),
                            );
                            let nibble = if group % 2 == 0 {
                                raw
                            } else {
                                _mm256_srli_epi16::<4>(raw)
                            };
                            _mm256_and_si256(nibble, _mm256_set1_epi8(15))
                        };
                        let factor = d * f32::from(scale(&bytes[4..16], group));
                        let bias = dmin * f32::from(minimum(&bytes[4..16], group));
                        *result += unsafe { projection(codes, None, input, index, factor, bias) };
                    }
                }
            }
            TensorType::Q5_0 => {
                for (block, bytes) in packed.chunks_exact(22).enumerate() {
                    check_cancel(cancel)?;
                    let d = half(bytes)?;
                    let high = u32::from_le_bytes(
                        bytes[2..6]
                            .try_into()
                            .map_err(|_| Error::InvalidBufferLength)?,
                    );
                    // Original16 nibble bytes become32 public unsigned
                    // codes in registers. Each four-bit mask is spread to the
                    // high bit of four bytes without changing original order.
                    let codes = unsafe {
                        let low = _mm_loadu_si128(bytes.as_ptr().add(6).cast());
                        let joined = _mm256_inserti128_si256::<1>(
                            _mm256_castsi128_si256(low),
                            _mm_srli_epi16::<4>(low),
                        );
                        let nibbles = _mm256_and_si256(joined, _mm256_set1_epi8(15));
                        let bits = _mm256_srlv_epi32(
                            _mm256_set1_epi32(high as i32),
                            _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28),
                        );
                        let flags = _mm256_and_si256(
                            _mm256_mullo_epi32(
                                _mm256_and_si256(bits, _mm256_set1_epi32(15)),
                                _mm256_set1_epi32(0x02040810),
                            ),
                            _mm256_set1_epi32(0x10101010),
                        );
                        _mm256_or_si256(nibbles, flags)
                    };
                    *result += unsafe { projection(codes, None, input, block, d, 16. * d) };
                }
            }
            TensorType::Q8_0 => {
                for (block, bytes) in packed.chunks_exact(34).enumerate() {
                    check_cancel(cancel)?;
                    let d = half(bytes)?;
                    // Use abs(signed weights), flipping activation signs.
                    // XOR128 offset dots saturate maddubs on extreme pairs;
                    // this approach includes weight=-128 without saturation.
                    // SAFETY:34-byte original block contains32 signed weights;
                    // quantizer clips input to[-127,127], so sign cannot wrap.
                    *result += unsafe {
                        let weights = _mm256_loadu_si256(bytes.as_ptr().add(2).cast());
                        projection(_mm256_abs_epi8(weights), Some(weights), input, block, d, 0.)
                    };
                }
            }
            _ => return Err(Error::InvalidFormat("CPU quantized format unsupported")),
        }
        check_cancel(cancel)?;
        finite(*result, "CPU quantized non-finite projection")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quantization_bounds_zero_subnormal_geometry_and_erasure() {
        let cancel = AtomicBool::new(false);
        let mut input = Input::empty();
        let values: Vec<_> = (0..MAX_HIDDEN)
            .map(|i| ((i * 19 % 263) as f32 - 131.) / 13.)
            .collect();
        input.fill(&values, &cancel).unwrap();
        for (index, &value) in values.iter().enumerate() {
            let scale = input.scales[index / GROUP];
            assert!(
                (value - f32::from(input.quants[index]) * scale).abs() <= scale * 0.50001 + 1e-6
            );
        }
        input.fill(&[f32::from_bits(1); GROUP], &cancel).unwrap();
        assert_eq!(input.columns, GROUP);
        assert!(input.quants.iter().all(|&v| v == 0));
        input.zeroize();
        assert_eq!(input.columns, 0);
        assert!(input.scales.iter().all(|&v| v.to_bits() == 0));
        assert!(input.sums.iter().all(|&v| v == 0));
        for size in [0, 31, MAX_HIDDEN + 32] {
            assert!(input.fill(&vec![0.; size], &cancel).is_err());
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(input.fill(&[value; GROUP], &cancel).is_err());
        }
        cancel.store(true, Ordering::Release);
        assert!(input.fill(&[1.; GROUP], &cancel).is_err());
    }
    #[test]
    fn residual_original_rows_match_independent_f64_and_erase_every_pass() {
        if !avx2_available() {
            return;
        }
        let cancel = AtomicBool::new(false);
        for kind in [TensorType::Q4K, TensorType::Q5_0, TensorType::Q8_0] {
            for case in 0..13 {
                let mut packed = vec![0u8; kind.row_bytes(MAX_HIDDEN as u64).unwrap() as usize];
                for (i, b) in packed.iter_mut().enumerate() {
                    *b = ((i * 37 + case * 29) % 256) as u8;
                }
                for block in packed.chunks_mut(kind.block_bytes() as usize) {
                    block[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                    if kind == TensorType::Q4K {
                        block[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
                    }
                }
                let values: Vec<_> = (0..MAX_HIDDEN)
                    .map(|i| ((i * 41 + case * 19) % 287) as f32 / 37. - 3.)
                    .collect();
                let mut input = Input::empty();
                input.fill_refined(&values, &cancel).unwrap();
                let mut weights = vec![0.; MAX_HIDDEN];
                kind.decode(&packed, &mut weights).unwrap();
                let reference: f64 = weights
                    .iter()
                    .zip(&values)
                    .map(|(&w, &v)| f64::from(w) * f64::from(v))
                    .sum();
                let norm: f64 = weights
                    .iter()
                    .zip(&values)
                    .map(|(&w, &v)| (f64::from(w) * f64::from(v)).abs())
                    .sum();
                let actual = dot(kind, &packed, &input, &cancel).unwrap();
                assert!((f64::from(actual) - reference).abs() <= 1e-5 + 1e-5 * norm);
                input.zeroize();
                assert!(!input.refined);
                assert_eq!(input.columns, 0);
                assert!(
                    input
                        .refinements
                        .iter()
                        .all(|r| r.quants.iter().all(|&v| v == 0)
                            && r.scales.iter().all(|&v| v.to_bits() == 0)
                            && r.sums.iter().all(|&v| v == 0))
                );
            }
        }
    }
    #[test]
    fn integer_dots_match_independent_dequantized_oracle_including_saturation_edges() {
        if !avx2_available() {
            return;
        }
        let cancel = AtomicBool::new(false);
        for kind in [TensorType::Q4K, TensorType::Q5_0, TensorType::Q8_0] {
            for case in 0..13 {
                let columns = 256;
                let mut packed = vec![0u8; kind.row_bytes(columns).unwrap() as usize];
                for (i, byte) in packed.iter_mut().enumerate() {
                    *byte = ((i * 37 + case * 29) % 256) as u8;
                }
                let block_bytes = kind.block_bytes() as usize;
                for block in packed.chunks_mut(block_bytes) {
                    block[..2].copy_from_slice(&0x3800u16.to_le_bytes());
                    if kind == TensorType::Q4K {
                        block[2..4].copy_from_slice(&0x2400u16.to_le_bytes());
                    }
                    if kind == TensorType::Q8_0 && case < 4 {
                        block[2..].fill(if case % 2 == 0 { 128 } else { 127 });
                    }
                }
                let values: Vec<_> = (0..columns)
                    .map(|i| {
                        if case < 4 {
                            if (i + case as u64).is_multiple_of(2) {
                                -127.
                            } else {
                                127.
                            }
                        } else {
                            ((i * 41 + case as u64 * 19) % 287) as f32 / 37. - 3.
                        }
                    })
                    .collect();
                let mut input = Input::empty();
                input.fill(&values, &cancel).unwrap();
                let mut weights = vec![0.; columns as usize];
                kind.decode(&packed, &mut weights).unwrap();
                let reference: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(i, &w)| {
                        f64::from(w)
                            * f64::from(input.scales[i / GROUP])
                            * f64::from(input.quants[i])
                    })
                    .sum();
                let actual = dot(kind, &packed, &input, &cancel).unwrap();
                assert!((f64::from(actual) - reference).abs() <= 0.001 + reference.abs() * 2e-5);
                packed[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
                assert!(dot(kind, &packed, &input, &cancel).is_err());
            }
        }
    }
}
