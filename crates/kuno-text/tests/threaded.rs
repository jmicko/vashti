// The root task owns lib.rs integration. Re-export its public modules so this
// included candidate module is tested before adding the production declaration.
pub use kuno_text::{Error, Result, TensorView, cpu, model};

#[path = "../src/threaded.rs"]
mod threaded;

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use kuno_text::{GgufFile, GgufLimits, TensorType};
use model::{MatrixExecutor, ScalarExecutor};
use threaded::ThreadedExecutor;

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

fn fixture(
    kind: TensorType,
    inputs: usize,
    outputs: usize,
    experts: usize,
    payload: Vec<u8>,
) -> Fixture {
    let path = std::env::temp_dir().join(format!(
        "kuno-threaded-fixture-{}-{}.gguf",
        std::process::id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(1_u64.to_le_bytes());
    bytes.push(b'w');
    bytes.extend((if experts == 1 { 2_u32 } else { 3 }).to_le_bytes());
    bytes.extend((inputs as u64).to_le_bytes());
    bytes.extend((outputs as u64).to_le_bytes());
    if experts != 1 {
        bytes.extend((experts as u64).to_le_bytes());
    }
    bytes.extend(kind.gguf_id().to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
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

fn f32_fixture(inputs: usize, outputs: usize, experts: usize) -> Fixture {
    let payload = (0..inputs * outputs * experts)
        .flat_map(|index| {
            let value = ((index * 13 % 97) as f32 - 48.) * 0.125;
            value.to_le_bytes()
        })
        .collect();
    fixture(TensorType::F32, inputs, outputs, experts, payload)
}

#[test]
fn nonsquare_dense_and_selected_experts_are_bit_identical_to_scalar() {
    let fixture = f32_fixture(13, 97, 3);
    let input: Vec<_> = (0..13).map(|index| index as f32 * 0.25 - 1.).collect();
    for threads in [1, 2, 4] {
        let mut threaded = ThreadedExecutor::new(threads).unwrap();
        assert_eq!(threaded.threads(), threads);
        for expert in [0, 2, 1, 0] {
            let expected = ScalarExecutor
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let actual = threaded
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    let fixture = f32_fixture(7, 65, 1);
    let input = [1.; 7];
    let expected = ScalarExecutor
        .run(
            fixture.file.tensor("w").unwrap(),
            0,
            &input,
            &AtomicBool::new(false),
        )
        .unwrap();
    let actual = ThreadedExecutor::new(3)
        .unwrap()
        .run(
            fixture.file.tensor("w").unwrap(),
            0,
            &input,
            &AtomicBool::new(false),
        )
        .unwrap();
    assert_eq!(actual, expected);
}

fn quantized_payload(kind: TensorType, inputs: usize, outputs: usize, experts: usize) -> Vec<u8> {
    let blocks = inputs / kind.block_elements() as usize;
    let mut payload = Vec::new();
    for row in 0..outputs * experts {
        for block in 0..blocks {
            let mut bytes = vec![0; kind.block_bytes() as usize];
            match kind {
                TensorType::Q5_0 => {
                    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
                    let high = 0xa5a5_5a5a_u32.rotate_left((row % 32) as u32);
                    bytes[2..6].copy_from_slice(&high.to_le_bytes());
                    for (index, value) in bytes[6..].iter_mut().enumerate() {
                        *value = (index * 19 + row * 7 + block * 3) as u8;
                    }
                }
                TensorType::Q8_0 => {
                    bytes[..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                    for (index, value) in bytes[2..].iter_mut().enumerate() {
                        *value = (index * 11 + row * 7 + block * 3) as u8;
                    }
                }
                TensorType::Q4K => {
                    bytes[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
                    bytes[2..4].copy_from_slice(&0x3400_u16.to_le_bytes());
                    for (index, value) in bytes[4..].iter_mut().enumerate() {
                        *value = (index * 17 + row * 5 + block * 3) as u8;
                    }
                }
                TensorType::Q6K => {
                    bytes[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
                    for (index, value) in bytes[..208].iter_mut().enumerate() {
                        *value = (index * 13 + row * 3 + block * 7) as u8;
                    }
                }
                _ => unreachable!(),
            }
            payload.extend(bytes);
        }
    }
    payload
}

#[test]
fn mixed_quantization_formats_preserve_serial_dot_and_expert_offsets() {
    let mut threaded = ThreadedExecutor::new(4).unwrap();
    for kind in [
        TensorType::Q5_0,
        TensorType::Q8_0,
        TensorType::Q4K,
        TensorType::Q6K,
    ] {
        let inputs = kind.block_elements() as usize * 3;
        let outputs = 67;
        let experts = 3;
        let fixture = fixture(
            kind,
            inputs,
            outputs,
            experts,
            quantized_payload(kind, inputs, outputs, experts),
        );
        let input: Vec<_> = (0..inputs)
            .map(|index| (index % 17) as f32 * 0.125 - 1.)
            .collect();
        for expert in [2, 0, 1] {
            let expected = ScalarExecutor
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let actual = threaded
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn thread_count_and_every_validation_error_are_bounded_and_clear_output() {
    assert!(ThreadedExecutor::new(0).is_err());
    assert!(ThreadedExecutor::new(33).is_err());
    let fixture = f32_fixture(3, 65, 2);
    let executor = ThreadedExecutor::new(2).unwrap();
    for (expert, input, cancelled) in [
        (2, vec![1.; 3], false),
        (0, vec![1.; 2], false),
        (0, vec![f32::NAN; 3], false),
        (0, vec![1.; 3], true),
    ] {
        let mut output = [99.; 65];
        assert!(
            executor
                .apply(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &mut output,
                    &AtomicBool::new(cancelled)
                )
                .is_err()
        );
        assert_eq!(output, [0.; 65]);
    }
    let mut wrong_output = [99.; 64];
    assert!(
        executor
            .apply(
                fixture.file.tensor("w").unwrap(),
                0,
                &[1.; 3],
                &mut wrong_output,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(wrong_output, [0.; 64]);
}

#[test]
fn file_changes_nonfinite_weights_and_overflow_clear_all_parallel_output() {
    let executor = ThreadedExecutor::new(4).unwrap();
    let mut payload = Vec::new();
    for row in 0..97 {
        for column in 0..3 {
            payload.extend(
                (if row == 60 && column == 1 {
                    f32::NAN
                } else {
                    1.
                })
                .to_le_bytes(),
            );
        }
    }
    let fixture = fixture(TensorType::F32, 3, 97, 1, payload);
    let mut output = [99.; 97];
    assert!(
        executor
            .apply(
                fixture.file.tensor("w").unwrap(),
                0,
                &[1.; 3],
                &mut output,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(output, [0.; 97]);
    let fixture = f32_fixture(3, 97, 1);
    let file = OpenOptions::new().write(true).open(&fixture.path).unwrap();
    file.set_len(fixture.file.file_len() - 1).unwrap();
    output.fill(99.);
    assert!(
        executor
            .apply(
                fixture.file.tensor("w").unwrap(),
                0,
                &[1.; 3],
                &mut output,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(output, [0.; 97]);
    let fixture = self::fixture(
        TensorType::F32,
        3,
        97,
        1,
        std::iter::repeat_n(f32::MAX.to_le_bytes(), 3 * 97)
            .flatten()
            .collect(),
    );
    output.fill(99.);
    assert!(
        executor
            .apply(
                fixture.file.tensor("w").unwrap(),
                0,
                &[2.; 3],
                &mut output,
                &AtomicBool::new(false)
            )
            .is_err()
    );
    assert_eq!(output, [0.; 97]);
}

#[test]
fn cancellation_from_another_thread_clears_output_before_return_and_pool_remains_usable() {
    let inputs = 65_536;
    let outputs = 128;
    let fixture = fixture(
        TensorType::F32,
        inputs,
        outputs,
        1,
        std::iter::repeat_n(1.0_f32.to_le_bytes(), inputs * outputs)
            .flatten()
            .collect(),
    );
    let executor = ThreadedExecutor::new(2).unwrap();
    let cancel = AtomicBool::new(false);
    let barrier = std::sync::Barrier::new(2);
    let input = vec![1.; inputs];
    let mut output = [99.; 128];
    std::thread::scope(|scope| {
        scope.spawn(|| {
            barrier.wait();
            std::thread::sleep(std::time::Duration::from_millis(1));
            cancel.store(true, Ordering::Relaxed);
        });
        barrier.wait();
        assert!(
            executor
                .apply(
                    fixture.file.tensor("w").unwrap(),
                    0,
                    &input,
                    &mut output,
                    &cancel
                )
                .is_err()
        );
    });
    assert_eq!(output, [0.; 128]);
    cancel.store(false, Ordering::Relaxed);
    let small = f32_fixture(3, 65, 1);
    let mut output = [0.; 65];
    executor
        .apply(
            small.file.tensor("w").unwrap(),
            0,
            &[1.; 3],
            &mut output,
            &cancel,
        )
        .unwrap();
    let expected = ScalarExecutor
        .run(small.file.tensor("w").unwrap(), 0, &[1.; 3], &cancel)
        .unwrap();
    assert_eq!(output.as_slice(), expected);
}

#[test]
fn float16_and_bfloat16_tiles_match_scalar_on_nonsquare_experts() {
    for (kind, one, minus_one) in [
        (TensorType::F16, 0x3c00_u16, 0xbc00_u16),
        (TensorType::Bf16, 0x3f80_u16, 0xbf80_u16),
    ] {
        let inputs = 13;
        let outputs = 71;
        let experts = 2;
        let payload = (0..inputs * outputs * experts)
            .flat_map(|index| {
                if index % 3 == 0 {
                    one.to_le_bytes()
                } else {
                    minus_one.to_le_bytes()
                }
            })
            .collect();
        let fixture = fixture(kind, inputs, outputs, experts, payload);
        let input: Vec<_> = (0..inputs).map(|index| index as f32 * 0.125 - 1.).collect();
        let mut executor = ThreadedExecutor::new(4).unwrap();
        for expert in [1, 0] {
            let actual = executor
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            let expected = ScalarExecutor
                .run(
                    fixture.file.tensor("w").unwrap(),
                    expert,
                    &input,
                    &AtomicBool::new(false),
                )
                .unwrap();
            assert_eq!(actual, expected);
        }
    }
}
