//! Opt-in complete public split binding and target carry scheduling reference.
//! Projections are synthetic; only bounded embedding/PLE rows and original
//! static vectors are read. No GPU, generation, full-weight numerical parity,
//! speculative transaction, performance or private erasure claim is made.
use kuno_text::model::{MatrixExecutor, Qwen4Exp};
use kuno_text::ops::{self, GdnState};
use kuno_text::{Error, GgufFile, GgufLimits, Result, TensorType, TensorView};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::Zeroizing;

#[derive(Default)]
struct SyntheticExecutor {
    projections: usize,
    head_calls: [usize; 3],
    normalized_head_input: Option<Zeroizing<Vec<f32>>>,
    mixed_head_input: Option<Zeroizing<Vec<f32>>>,
    fail_head: bool,
    cancel_head: bool,
}
impl MatrixExecutor for SyntheticExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        _expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        assert_eq!(input.len() as u64, tensor.info().dimensions()[0]);
        self.projections += 1;
        let mut output = vec![0.; tensor.info().dimensions()[1] as usize];
        match tensor.info().name() {
            "output_hc_down.weight" => {
                self.head_calls[0] += 1;
                self.normalized_head_input = Some(Zeroizing::new(input.to_vec()));
            }
            "output_hc_up.weight" => self.head_calls[1] += 1,
            "output.weight" => {
                self.head_calls[2] += 1;
                self.mixed_head_input = Some(Zeroizing::new(input.to_vec()));
                if self.fail_head {
                    return Err(Error::InvalidFormat("public synthetic target head failure"));
                }
                output[..input.len()].copy_from_slice(input);
                if self.cancel_head {
                    cancel.store(true, Ordering::Relaxed);
                }
            }
            _ => {}
        }
        Ok(output)
    }

    // This scheduling test has no real projection Q/K/V. Avoid spending host
    // time running a real delta recurrence on that synthetic all-zero payload.
    #[allow(clippy::too_many_arguments)]
    fn delta_rule(
        &mut self,
        _state: &mut GdnState,
        _query: &[f32],
        _key: &[f32],
        value: &[f32],
        _log_decay: &[f32],
        _beta: &[f32],
        _cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        Ok(vec![0.; value.len()])
    }
}

fn assert_same_bits(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
#[ignore = "requires verified public UD-Q3_K_XL shards via KUNO_TEST_Q3_DIR; synthetic projections only"]
fn installed_target_carry_is_pre_head_and_preserves_single_path_position_and_recovery() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("KUNO_TEST_Q3_DIR").expect("KUNO_TEST_Q3_DIR is required"),
    );
    let paths = [
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
    ]
    .map(|name| directory.join(name));
    let file = GgufFile::open_shards(&paths, GgufLimits::default()).unwrap();
    assert_eq!(file.file_len(), 89_986_353_824);
    assert_eq!(file.tensors().len(), 1224);
    let model = Qwen4Exp::bind(&file).unwrap();
    let graph = model.bound_graph();
    assert_eq!(
        graph.tensor("token_embd.weight").unwrap().info().dtype(),
        TensorType::Q8_0
    );
    assert_eq!(
        graph.tensor("output.weight").unwrap().info().dtype(),
        TensorType::Q6K
    );
    let mut sequence = model.sequence(2).unwrap();
    let cancel = AtomicBool::new(false);
    let mut executor = SyntheticExecutor::default();
    let mut layers = Vec::with_capacity(48);
    let carry = model
        .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |layer| {
            layers.push(layer)
        })
        .unwrap();
    assert_eq!(layers, (0..48).collect::<Vec<_>>());
    assert_eq!((carry.position(), sequence.position()), (0, 1));
    assert!(carry.binding_id() == graph.binding_id());
    assert_eq!(carry.hidden().len(), 10240);
    assert_eq!(carry.logits().len(), 248320);
    assert_eq!(executor.head_calls, [1; 3]);

    // Every synthetic block output and PLE projection is zero, so the real
    // embedding remains the wide carry through every actual HC combine.
    let mut embedding = Zeroizing::new(vec![0.; 2560]);
    graph
        .tensor("token_embd.weight")
        .unwrap()
        .decode_row(0, &mut embedding)
        .unwrap();
    for stream in carry.hidden().chunks_exact(2560) {
        assert_eq!(stream, &*embedding);
    }
    let normalized = Zeroizing::new(
        ops::rms_norm(
            carry.hidden(),
            graph.static_vector("output_hc_norm.weight").unwrap(),
            2560,
            1e-6,
        )
        .unwrap(),
    );
    assert_same_bits(
        executor.normalized_head_input.as_ref().unwrap(),
        &normalized,
    );
    assert_eq!(executor.mixed_head_input.as_ref().unwrap().len(), 2560);

    // The old API runs exactly the same trunk/head and returns the same logits.
    sequence.reset();
    executor = SyntheticExecutor::default();
    let legacy = Zeroizing::new(
        model
            .step(&mut executor, &mut sequence, 0, &cancel, |_| {})
            .unwrap(),
    );
    assert_same_bits(&legacy, carry.logits());
    assert_eq!(executor.head_calls, [1; 3]);
    let second = model
        .step_with_carry(&mut executor, &mut sequence, 1, &cancel, |_| {})
        .unwrap();
    assert_eq!((second.position(), sequence.position()), (1, 2));
    assert_eq!(carry.position(), 0);
    let calls = executor.projections;
    assert!(
        model
            .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
            .is_err()
    );
    assert_eq!((sequence.position(), executor.projections), (2, calls));

    // A failure or late cancellation after all QSA/PLE state advanced cannot
    // publish a carry, resume that state, or bypass the existing reset gate.
    for cancel_head in [false, true] {
        sequence.reset();
        executor = SyntheticExecutor {
            fail_head: !cancel_head,
            cancel_head,
            ..Default::default()
        };
        assert!(
            model
                .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(sequence.position(), 0);
        assert_eq!(executor.head_calls, [1; 3]);
        cancel.store(false, Ordering::Relaxed);
        let calls = executor.projections;
        executor.fail_head = false;
        executor.cancel_head = false;
        assert!(
            model
                .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(executor.projections, calls);
        sequence.reset();
        let recovered = model
            .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
            .unwrap();
        assert_same_bits(recovered.logits(), carry.logits());
        assert_same_bits(recovered.hidden(), carry.hidden());
    }
    let other = Qwen4Exp::bind(&file).unwrap();
    let position = sequence.position();
    let calls = executor.projections;
    assert!(
        other
            .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
            .is_err()
    );
    assert_eq!(
        (sequence.position(), executor.projections),
        (position, calls)
    );
    graph.verify_unchanged().unwrap();
}
