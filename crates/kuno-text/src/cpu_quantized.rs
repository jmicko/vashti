//! Explicit experimental Q8 activation dots for original packed CPU experts.
//! This changes rounding/activation precision: it is NOT original64 parity.
//! Quantize each 32-value input block once per operator, reuse across rows.
//! Only Q4_K, Q5_0 and Q8_0 are accepted; never select this after a failure.
use super::*;

const GROUP: usize = 32;
const GROUPS: usize = MAX_HIDDEN / GROUP;

pub(super) struct Input {
    quants: [i8; MAX_HIDDEN],
    scales: [f32; GROUPS],
    sums: [i32; GROUPS],
    columns: usize,
}
impl Input {
    pub(super) fn empty() -> Self {
        Self {
            quants: [0; MAX_HIDDEN],
            scales: [0.; GROUPS],
            sums: [0; GROUPS],
            columns: 0,
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
}
impl Zeroize for Input {
    fn zeroize(&mut self) {
        self.quants.zeroize();
        self.scales.zeroize();
        self.sums.zeroize();
        self.columns.zeroize();
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
        let mut lanes = Zeroizing::new([0i32; 8]);
        // SAFETY: eight initialized i32 lanes, exactly one vector store.
        unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast(), groups) };
        lanes.iter().sum()
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
                        let (codes, x) = unsafe {
                            let raw = _mm256_loadu_si256(
                                bytes.as_ptr().add(16 + (group / 2) * 32).cast(),
                            );
                            let nibble = if group % 2 == 0 {
                                raw
                            } else {
                                _mm256_srli_epi16::<4>(raw)
                            };
                            (
                                _mm256_and_si256(nibble, _mm256_set1_epi8(15)),
                                _mm256_loadu_si256(input.quants.as_ptr().add(index * GROUP).cast()),
                            )
                        };
                        // SAFETY: enclosing function guarantees AVX2.
                        let product = unsafe { integer_dot(codes, x) };
                        let factor = d * f32::from(scale(&bytes[4..16], group));
                        let bias = dmin * f32::from(minimum(&bytes[4..16], group));
                        *result += input.scales[index]
                            * (factor * product as f32 - bias * input.sums[index] as f32);
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
                    let mut codes = Zeroizing::new([0u8; GROUP]);
                    for index in 0..16 {
                        codes[index] = (bytes[6 + index] & 15) | (((high >> index) as u8 & 1) << 4);
                        codes[index + 16] =
                            (bytes[6 + index] >> 4) | (((high >> (index + 16)) as u8 & 1) << 4);
                    }
                    // SAFETY: exact32-byte guarded code and input blocks.
                    let product = unsafe {
                        integer_dot(
                            _mm256_loadu_si256(codes.as_ptr().cast()),
                            _mm256_loadu_si256(input.quants.as_ptr().add(block * GROUP).cast()),
                        )
                    };
                    *result += d * input.scales[block] * (product - 16 * input.sums[block]) as f32;
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
                    let product = unsafe {
                        let weights = _mm256_loadu_si256(bytes.as_ptr().add(2).cast());
                        let x = _mm256_loadu_si256(input.quants.as_ptr().add(block * GROUP).cast());
                        integer_dot(_mm256_abs_epi8(weights), _mm256_sign_epi8(x, weights))
                    };
                    *result += d * input.scales[block] * product as f32;
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
