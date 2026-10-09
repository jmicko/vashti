//! Bounded, transient decoded-weight cache for small public-lab matrices.
//!
//! Only original model weights are cached. Inputs, outputs, and intermediate
//! activations are never retained. The inspected descriptor is pinned for the
//! cache lifetime; a same-name tensor from another artifact cannot hit it.
use kuno_text::cpu::Matrix;
use kuno_text::{Error, GgufFile, Result, TensorView};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

const MAX_BUDGET: usize = 64 * 1024 * 1024;
const MAX_OUTPUTS: usize = 64;
const MAX_ENTRIES: usize = 1024;
const MAX_PACKED_ROW: usize = 256 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Key {
    descriptor: usize,
    offset: u64,
    dtype: u32,
    inputs: usize,
    outputs: usize,
    experts: usize,
    expert: usize,
}

struct Entry {
    key: Key,
    weights: Zeroizing<Vec<f32>>,
    bytes: usize,
}

/// FIFO eviction stores no access clock/history. The budget counts decoded
/// Vec capacity, with at most 1024 small metadata entries and one bounded
/// packed row (256 KiB) in addition. Each matrix contains at most 64 rows.
pub struct SmallCpu<'a> {
    model: &'a GgufFile,
    budget: usize,
    bytes: usize,
    entries: VecDeque<Entry>,
    avx2: bool,
}

impl<'a> SmallCpu<'a> {
    pub fn new(model: &'a GgufFile, budget: usize) -> Result<Self> {
        if !(4..=MAX_BUDGET).contains(&budget) {
            return Err(Error::LimitExceeded("small CPU cache budget"));
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let avx2 = std::is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let avx2 = false;
        Ok(Self {
            model,
            budget,
            bytes: 0,
            entries: VecDeque::new(),
            avx2,
        })
    }

    pub fn cached_bytes(&self) -> usize {
        self.bytes
    }

    pub fn cached_matrices(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        // Entry's Zeroizing<Vec<f32>> erases the complete retained capacity.
        self.entries.clear();
        self.bytes = 0;
    }

    /// Returns None only for matrices beyond the 64-output shape threshold.
    /// Eligible requests validate descriptor identity, file consistency, input
    /// length/finiteness, and expert bounds before reading or executing weights.
    pub fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        if !tensor.belongs_to(self.model) {
            return Err(Error::InvalidFormat("small CPU descriptor ownership"));
        }
        let matrix = Matrix::new(tensor)?;
        if matrix.outputs() > MAX_OUTPUTS {
            return Ok(None);
        }
        if input.len() != matrix.inputs() || expert >= matrix.experts() {
            return Err(Error::InvalidBufferLength);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidFormat("non-finite small CPU input"));
        }
        check_cancel(cancel)?;
        if let Err(error) = self.model.verify_file_unchanged() {
            self.clear();
            return Err(error);
        }
        let key = Key {
            // The owner is pinned, so its immutable descriptor address cannot
            // be freed/reused while this cache lives. No pointer dereference.
            descriptor: std::ptr::from_ref(tensor.info()) as usize,
            offset: tensor.info().offset(),
            dtype: tensor.info().dtype().gguf_id(),
            inputs: matrix.inputs(),
            outputs: matrix.outputs(),
            experts: matrix.experts(),
            expert,
        };
        let index = match self.entries.iter().position(|entry| entry.key == key) {
            Some(index) => index,
            None => {
                let count = matrix
                    .inputs()
                    .checked_mul(matrix.outputs())
                    .ok_or(Error::ArithmeticOverflow)?;
                let requested = count
                    .checked_mul(size_of::<f32>())
                    .ok_or(Error::ArithmeticOverflow)?;
                if requested > self.budget {
                    return Err(Error::LimitExceeded("small CPU matrix bytes"));
                }
                let row_bytes = usize::try_from(tensor.info().row_bytes())
                    .map_err(|_| Error::ArithmeticOverflow)?;
                if row_bytes > MAX_PACKED_ROW {
                    return Err(Error::LimitExceeded("small CPU packed row"));
                }
                self.make_room(requested)?;
                let mut weights = zeroed::<f32>(count)?;
                let bytes = weights
                    .capacity()
                    .checked_mul(size_of::<f32>())
                    .ok_or(Error::ArithmeticOverflow)?;
                if bytes > self.budget {
                    return Err(Error::LimitExceeded("small CPU allocation capacity"));
                }
                self.make_room(bytes)?;
                let mut packed = zeroed::<u8>(row_bytes)?;
                let first_row = expert
                    .checked_mul(matrix.outputs())
                    .ok_or(Error::ArithmeticOverflow)?;
                for row in 0..matrix.outputs() {
                    check_cancel(cancel)?;
                    let source_row = first_row
                        .checked_add(row)
                        .ok_or(Error::ArithmeticOverflow)?;
                    tensor.read_row(
                        u64::try_from(source_row).map_err(|_| Error::ArithmeticOverflow)?,
                        &mut packed,
                    )?;
                    let first = row
                        .checked_mul(matrix.inputs())
                        .ok_or(Error::ArithmeticOverflow)?;
                    let target = &mut weights[first..first + matrix.inputs()];
                    tensor.info().dtype().decode(&packed, target)?;
                    if target.iter().any(|value| !value.is_finite()) {
                        return Err(Error::InvalidFormat("non-finite small CPU weights"));
                    }
                }
                check_cancel(cancel)?;
                self.bytes = self
                    .bytes
                    .checked_add(bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                self.entries.push_back(Entry {
                    key,
                    weights,
                    bytes,
                });
                self.entries.len() - 1
            }
        };
        let weights = &self.entries[index].weights;
        let mut output = zeroed::<f32>(matrix.outputs())?;
        for (row, value) in output.iter_mut().enumerate() {
            check_cancel(cancel)?;
            let first = row
                .checked_mul(matrix.inputs())
                .ok_or(Error::ArithmeticOverflow)?;
            *value = dot(
                &weights[first..first + matrix.inputs()],
                input,
                cancel,
                self.avx2,
            )?;
        }
        check_cancel(cancel)?;
        // No partial output escapes on error/cancellation. The successful Vec
        // is caller-owned; the cache never keeps an output copy.
        Ok(Some(std::mem::take(&mut *output)))
    }

    fn make_room(&mut self, bytes: usize) -> Result<()> {
        while self.entries.len() >= MAX_ENTRIES
            || self
                .bytes
                .checked_add(bytes)
                .is_none_or(|total| total > self.budget)
        {
            let entry = self
                .entries
                .pop_front()
                .ok_or(Error::LimitExceeded("small CPU cache capacity"))?;
            self.bytes = self
                .bytes
                .checked_sub(entry.bytes)
                .ok_or(Error::InvalidFormat("small CPU cache accounting"))?;
            // Model-only conversion is erased on every eviction too.
            drop(entry);
        }
        Ok(())
    }
}

impl Drop for SmallCpu<'_> {
    fn drop(&mut self) {
        self.clear();
    }
}

fn zeroed<T: Clone + Default + Zeroize>(len: usize) -> Result<Zeroizing<Vec<T>>> {
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(len)
        .map_err(|_| Error::LimitExceeded("small CPU allocation"))?;
    values.resize(len, T::default());
    Ok(values)
}

fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::InvalidFormat("small CPU execution cancelled"))
    } else {
        Ok(())
    }
}

#[allow(unsafe_code)]
fn dot(weights: &[f32], input: &[f32], cancel: &AtomicBool, avx2: bool) -> Result<f32> {
    if weights.len() != input.len() {
        return Err(Error::InvalidBufferLength);
    }
    check_cancel(cancel)?;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let sum = if avx2 {
        // SAFETY: SmallCpu sets this flag only after runtime AVX2 detection;
        // equal slice lengths above and chunk bounds prove every vector load.
        unsafe { dot_avx2(weights, input, cancel)? }
    } else {
        dot_scalar(weights, input, cancel)?
    };
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let sum = {
        let _ = avx2;
        dot_scalar(weights, input, cancel)?
    };
    check_cancel(cancel)?;
    if !sum.is_finite() {
        return Err(Error::InvalidFormat("non-finite small CPU output"));
    }
    Ok(sum)
}

fn dot_scalar(weights: &[f32], input: &[f32], cancel: &AtomicBool) -> Result<f32> {
    let mut sum = 0.;
    for (index, (weight, value)) in weights.iter().zip(input).enumerate() {
        if index.is_multiple_of(1024) {
            check_cancel(cancel)?;
        }
        sum += weight * value;
    }
    Ok(sum)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code)]
unsafe fn dot_avx2(weights: &[f32], input: &[f32], cancel: &AtomicBool) -> Result<f32> {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut accumulators = [_mm256_setzero_ps(); 4];
    let chunks = weights.len() / 32;
    for block in 0..chunks {
        if block.is_multiple_of(128) {
            check_cancel(cancel)?;
        }
        for (lane, accumulator) in accumulators.iter_mut().enumerate() {
            let first = block * 32 + lane * 8;
            // SAFETY: block<weights.len()/32, lane<4, equal input length;
            // each unaligned read covers eight valid contiguous f32 elements.
            let (weight, value) = unsafe {
                (
                    _mm256_loadu_ps(weights.as_ptr().add(first)),
                    _mm256_loadu_ps(input.as_ptr().add(first)),
                )
            };
            // Explicit multiply/add: no FMA feature requirement or C/C++.
            *accumulator = _mm256_add_ps(*accumulator, _mm256_mul_ps(weight, value));
        }
    }
    let combined = _mm256_add_ps(
        _mm256_add_ps(accumulators[0], accumulators[1]),
        _mm256_add_ps(accumulators[2], accumulators[3]),
    );
    let mut lanes = Zeroizing::new([0f32; 8]);
    // SAFETY: the destination contains exactly eight initialized f32 slots.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), combined) };
    let mut sum = lanes.iter().sum::<f32>();
    for (weight, value) in weights[chunks * 32..].iter().zip(&input[chunks * 32..]) {
        sum += weight * value;
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_text::{GgufLimits, TensorType};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicU64;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    struct Fixture {
        path: PathBuf,
        file: GgufFile,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn make_fixture(
        kind: TensorType,
        inputs: usize,
        outputs: usize,
        experts: usize,
        payload: &[u8],
    ) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "kuno-small-cpu-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend((if experts == 1 { 2u32 } else { 3 }).to_le_bytes());
        bytes.extend((inputs as u64).to_le_bytes());
        bytes.extend((outputs as u64).to_le_bytes());
        if experts != 1 {
            bytes.extend((experts as u64).to_le_bytes());
        }
        bytes.extend(kind.gguf_id().to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend(payload);
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let file = GgufFile::open(&path, GgufLimits::default()).unwrap();
        Fixture { path, file }
    }

    fn oracle(weights: &[f32], input: &[f32]) -> (f64, f64) {
        weights
            .iter()
            .zip(input)
            .fold((0., 0.), |(sum, absolute), (w, x)| {
                let term = f64::from(*w) * f64::from(*x);
                (sum + term, absolute + term.abs())
            })
    }

    fn assert_dot(actual: f32, weights: &[f32], input: &[f32]) {
        let (expected, absolute) = oracle(weights, input);
        let tolerance = 1e-6 + absolute * f64::from(f32::EPSILON) * 32.;
        assert!((f64::from(actual) - expected).abs() <= tolerance);
    }

    #[test]
    fn scalar_and_detected_avx2_match_f64_oracle_including_tails() {
        for count in [1, 7, 8, 9, 13, 31, 32, 33, 65, 257, 4097, 65536] {
            let weights: Vec<_> = (0..count)
                .map(|i| ((i * 13 % 97) as f32 - 48.) * 0.125)
                .collect();
            let input: Vec<_> = (0..count)
                .map(|i| ((i * 19 % 79) as f32 - 39.) * 0.0625)
                .collect();
            let cancel = AtomicBool::new(false);
            assert_dot(
                dot(&weights, &input, &cancel, false).unwrap(),
                &weights,
                &input,
            );
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if std::is_x86_feature_detected!("avx2") {
                assert_dot(
                    dot(&weights, &input, &cancel, true).unwrap(),
                    &weights,
                    &input,
                );
            }
        }
    }

    #[test]
    fn nonsquare_expert_cache_is_reused_and_never_retains_input() {
        let inputs = 13;
        let outputs = 4;
        let experts = 3;
        let weights: Vec<_> = (0..inputs * outputs * experts)
            .map(|i| ((i * 13 % 97) as f32 - 48.) * 0.125)
            .collect();
        let payload: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let fixture = make_fixture(TensorType::F32, inputs, outputs, experts, &payload);
        let mut cache = SmallCpu::new(&fixture.file, MAX_BUDGET).unwrap();
        let cancel = AtomicBool::new(false);
        for expert in [0, 2, 0, 1, 2] {
            let input: Vec<_> = (0..inputs).map(|i| i as f32 * 0.25 - 1.).collect();
            let result = cache
                .run(fixture.file.tensor("w").unwrap(), expert, &input, &cancel)
                .unwrap()
                .unwrap();
            for (row, value) in result.iter().enumerate() {
                let first = (expert * outputs + row) * inputs;
                assert_dot(*value, &weights[first..first + inputs], &input);
            }
        }
        assert_eq!(cache.cached_matrices(), 3);
        assert_eq!(cache.cached_bytes(), weights.len() * 4);
        let pointer = cache.entries[0].weights.as_ptr();
        let result = cache
            .run(fixture.file.tensor("w").unwrap(), 0, &[0.; 13], &cancel)
            .unwrap()
            .unwrap();
        assert!(result.iter().all(|v| v.to_bits() == 0));
        assert_eq!(pointer, cache.entries[0].weights.as_ptr());
        assert_eq!(
            cache.entries[0].weights.as_slice(),
            &weights[..inputs * outputs]
        );
    }

    fn constant_packed(kind: TensorType) -> (usize, Vec<u8>, f32) {
        match kind {
            TensorType::F32 => (13, 13f32.to_le_bytes().repeat(13), 13.),
            TensorType::F16 => (13, 0x3800u16.to_le_bytes().repeat(13), 0.5),
            TensorType::Bf16 => (13, 0xc000u16.to_le_bytes().repeat(13), -2.),
            TensorType::Q5_0 => {
                let mut block = 0x3800u16.to_le_bytes().to_vec();
                block.extend(u32::MAX.to_le_bytes());
                block.extend([0xff; 16]);
                (160, block.repeat(5), 7.5)
            }
            TensorType::Q8_0 => {
                let mut block = 0x3800u16.to_le_bytes().to_vec();
                block.extend([2; 32]);
                (64, block.repeat(2), 1.)
            }
            TensorType::Q4K => {
                let mut block = 0x3800u16.to_le_bytes().to_vec();
                block.extend(0x3400u16.to_le_bytes());
                block.extend([1; 8]);
                block.extend([0x11; 4]);
                block.extend([0x22; 128]);
                (256, block, 0.75)
            }
            TensorType::Q6K => {
                let mut block = vec![0x11; 128];
                block.extend([0xaa; 64]);
                block.extend([2; 16]);
                block.extend(0x3800u16.to_le_bytes());
                (256, block, 1.)
            }
            TensorType::Iq3Xxs
            | TensorType::Iq4Nl
            | TensorType::Iq4Xs
            | TensorType::Iq2S
            | TensorType::Iq3S
            | TensorType::Q5K
            | TensorType::Q2_0 => {
                panic!("IQ nonlinear formats require the independent quant oracle fixture")
            }
        }
    }

    #[test]
    fn exact_known_weight_bits_for_every_artifact_format_and_q5_ple_rows() {
        for kind in [
            TensorType::F32,
            TensorType::F16,
            TensorType::Bf16,
            TensorType::Q5_0,
            TensorType::Q8_0,
            TensorType::Q4K,
            TensorType::Q6K,
        ] {
            let (inputs, packed, expected) = constant_packed(kind);
            if kind == TensorType::Q5_0 {
                assert_eq!(packed.len(), 110);
            }
            let fixture = make_fixture(kind, inputs, 3, 2, &packed.repeat(6));
            let mut cache = SmallCpu::new(&fixture.file, MAX_BUDGET).unwrap();
            let input: Vec<_> = (0..inputs).map(|i| i as f32 * 0.03125 - 1.).collect();
            let result = cache
                .run(
                    fixture.file.tensor("w").unwrap(),
                    1,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap()
                .unwrap();
            assert!(
                cache.entries[0]
                    .weights
                    .iter()
                    .all(|v| v.to_bits() == expected.to_bits())
            );
            for value in result {
                assert_dot(value, &vec![expected; inputs], &input);
            }
        }
    }

    #[test]
    fn budget_eviction_shape_fallback_and_bounds() {
        let fixture = make_fixture(TensorType::F32, 9, 2, 3, &1f32.to_le_bytes().repeat(54));
        let mut cache = SmallCpu::new(&fixture.file, 72).unwrap();
        let cancel = AtomicBool::new(false);
        for expert in [0, 1, 2, 0] {
            assert_eq!(
                cache
                    .run(fixture.file.tensor("w").unwrap(), expert, &[1.; 9], &cancel)
                    .unwrap()
                    .unwrap(),
                vec![9., 9.]
            );
            assert_eq!(cache.cached_matrices(), 1);
            assert_eq!(cache.cached_bytes(), 72);
            assert_eq!(cache.entries[0].key.expert, expert);
        }
        assert!(
            cache
                .run(fixture.file.tensor("w").unwrap(), 3, &[1.; 9], &cancel)
                .is_err()
        );
        assert!(
            cache
                .run(fixture.file.tensor("w").unwrap(), 0, &[1.; 8], &cancel)
                .is_err()
        );
        let larger = make_fixture(TensorType::F32, 1, 65, 1, &1f32.to_le_bytes().repeat(65));
        let mut cache = SmallCpu::new(&larger.file, 4).unwrap();
        assert!(
            cache
                .run(larger.file.tensor("w").unwrap(), 0, &[1.], &cancel)
                .unwrap()
                .is_none()
        );
        assert_eq!(cache.cached_bytes(), 0);
        assert!(SmallCpu::new(&larger.file, 0).is_err());
        assert!(SmallCpu::new(&larger.file, MAX_BUDGET + 1).is_err());
    }

    #[test]
    fn descriptor_identity_and_changed_artifact_fail_closed() {
        let a = make_fixture(TensorType::F32, 3, 1, 1, &1f32.to_le_bytes().repeat(3));
        let b = make_fixture(TensorType::F32, 3, 1, 1, &2f32.to_le_bytes().repeat(3));
        let mut cache = SmallCpu::new(&a.file, 64).unwrap();
        let cancel = AtomicBool::new(false);
        assert_eq!(
            cache
                .run(a.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
                .unwrap()
                .unwrap(),
            vec![3.]
        );
        assert!(
            cache
                .run(b.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
                .is_err()
        );
        OpenOptions::new()
            .append(true)
            .open(&a.path)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(matches!(
            cache.run(a.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel),
            Err(Error::FileChanged)
        ));
        assert_eq!(cache.cached_bytes(), 0);
        assert_eq!(cache.cached_matrices(), 0);
    }

    #[test]
    fn cancellation_and_nonfinite_fail_without_partial_output_or_weight_insert() {
        let fixture = make_fixture(TensorType::F32, 3, 2, 1, &1f32.to_le_bytes().repeat(6));
        let mut cache = SmallCpu::new(&fixture.file, 64).unwrap();
        let cancel = AtomicBool::new(true);
        assert!(
            cache
                .run(fixture.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
                .is_err()
        );
        assert_eq!(cache.cached_bytes(), 0);
        cancel.store(false, Ordering::Relaxed);
        assert!(
            cache
                .run(
                    fixture.file.tensor("w").unwrap(),
                    0,
                    &[f32::NAN; 3],
                    &cancel
                )
                .is_err()
        );
        assert_eq!(cache.cached_bytes(), 0);
        cache
            .run(fixture.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
            .unwrap();
        cancel.store(true, Ordering::Relaxed);
        assert!(
            cache
                .run(fixture.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
                .is_err()
        );
        assert_eq!(cache.cached_matrices(), 1); // Existing public weights remain safe.
        assert!(dot(&[1.; 33], &[1.; 33], &cancel, false).is_err());
        let bad = make_fixture(TensorType::F32, 3, 1, 1, &f32::NAN.to_le_bytes().repeat(3));
        let mut bad_cache = SmallCpu::new(&bad.file, 64).unwrap();
        cancel.store(false, Ordering::Relaxed);
        assert!(
            bad_cache
                .run(bad.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
                .is_err()
        );
        assert_eq!(bad_cache.cached_bytes(), 0);
        let overflow = make_fixture(TensorType::F32, 3, 1, 1, &f32::MAX.to_le_bytes().repeat(3));
        let mut overflow_cache = SmallCpu::new(&overflow.file, 64).unwrap();
        assert!(
            overflow_cache
                .run(overflow.file.tensor("w").unwrap(), 0, &[2.; 3], &cancel)
                .is_err()
        );
    }
}
