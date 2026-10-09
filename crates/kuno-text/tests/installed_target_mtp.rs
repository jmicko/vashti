//! Protected-runner opt-in for real public Q3 target -> Q8 MTP conditioning.
//! Fixed public token0 ("!") only; x1 is its target greedy successor. No text
//! input, content serialization, GPU, speculative commit or speed measurement.
//! This couples existing graphs, not an independent full-head numeric oracle.
use kuno_text::model::{MatrixExecutor, Qwen4Exp, greedy};
use kuno_text::mtp::MtpGraph;
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{Error, GgufFile, GgufLimits, Result, TensorView};
use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

const TOKEN0: u32 = 0;
const VOCAB: usize = 248320;
const WIDE: usize = 10240;
const THREADS: usize = 8;

#[derive(Clone, Copy)]
struct OutputCounts {
    logits_nonzero: usize,
    hidden_nonzero: usize,
}

fn output_counts(
    logits: &[f32],
    hidden: &[f32],
) -> std::result::Result<OutputCounts, &'static str> {
    if logits.len() != VOCAB || hidden.len() != WIDE {
        return Err("incomplete public graph output");
    }
    if !logits.iter().chain(hidden).all(|value| value.is_finite()) {
        return Err("non-finite public graph output");
    }
    let counts = OutputCounts {
        logits_nonzero: logits.iter().filter(|value| value.abs() > 1e-6).count(),
        hidden_nonzero: hidden.iter().filter(|value| value.abs() > 1e-6).count(),
    };
    if counts.logits_nonzero == 0 || counts.hidden_nonzero == 0 {
        return Err("zero public graph output");
    }
    Ok(counts)
}

fn bit_mismatches(a: &[f32], b: &[f32]) -> std::result::Result<usize, &'static str> {
    if a.len() != b.len() {
        return Err("incomplete public replay output");
    }
    Ok(a.iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count())
}

fn format_counts(file: &GgufFile) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    for tensor in file.tensors().values() {
        *counts.entry(tensor.dtype().gguf_id()).or_insert(0) += 1;
    }
    counts
}

struct NoInference;
impl MatrixExecutor for NoInference {
    fn run(&mut self, _: TensorView<'_>, _: usize, _: &[f32], _: &AtomicBool) -> Result<Vec<f32>> {
        panic!("foreign graph reached projection in public fixture");
    }
}

#[test]
#[ignore = "protected runner only: verified KUNO_TEST_Q3_DIR and KUNO_TEST_MTP; real CPU weights, no GPU"]
fn real_public_target_carry_conditions_q8_mtp_at_the_next_position_and_replays() {
    let directory = std::path::PathBuf::from(
        std::env::var_os("KUNO_TEST_Q3_DIR").expect("KUNO_TEST_Q3_DIR is required"),
    );
    let paths = [
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
        "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
    ]
    .map(|name| directory.join(name));
    let target_file = GgufFile::open_shards(&paths, GgufLimits::default()).unwrap();
    assert_eq!(target_file.file_len(), 89_986_353_824);
    assert_eq!(target_file.tensors().len(), 1224);
    assert_eq!(
        format_counts(&target_file),
        BTreeMap::from([
            (0, 557),
            (8, 502),
            (14, 1),
            (18, 94),
            (20, 44),
            (23, 2),
            (30, 24)
        ])
    );
    let target = Qwen4Exp::bind(&target_file).unwrap();
    let mtp_file = GgufFile::open(
        std::env::var_os("KUNO_TEST_MTP").expect("KUNO_TEST_MTP is required"),
        GgufLimits::default(),
    )
    .unwrap();
    assert_eq!(mtp_file.file_len(), 4_137_429_120);
    assert_eq!(mtp_file.tensors().len(), 34);
    assert_eq!(
        format_counts(&mtp_file),
        BTreeMap::from([(0, 11), (8, 21), (30, 2)])
    );
    let mtp = MtpGraph::bind(&mtp_file).unwrap();
    let cancel = AtomicBool::new(false);
    let zero_preceding = Zeroizing::new(vec![0.; WIDE]);

    let mut target_sequence = target.sequence(2).unwrap();
    let mut draft_sequence = mtp.sequence(2).unwrap();
    // The target and sidecar have separate binding domains. Check each actual
    // sequence owner independently; their raw binding numbers are not compared.
    let other_target = Qwen4Exp::bind(&target_file).unwrap();
    assert!(other_target.bound_graph().binding_id() != target.bound_graph().binding_id());
    assert!(matches!(
        other_target.step_with_carry(
            &mut NoInference,
            &mut target_sequence,
            TOKEN0,
            &cancel,
            |_| {}
        ),
        Err(Error::InvalidFormat("invalid native sequence input"))
    ));
    assert_eq!(target_sequence.position(), 0);
    drop(other_target);
    let other_mtp = MtpGraph::bind(&mtp_file).unwrap();
    assert!(matches!(
        other_mtp.step(
            &mut NoInference,
            &mut draft_sequence,
            TOKEN0,
            &zero_preceding,
            &cancel
        ),
        Err(Error::InvalidFormat("invalid MTP sequence input"))
    ));
    assert_eq!(draft_sequence.position(), 0);
    drop(other_mtp);

    let mut executor = ThreadedExecutor::new(THREADS).unwrap();
    let mut completed_layers = 0;
    let target0 = target
        .step_with_carry(
            &mut executor,
            &mut target_sequence,
            TOKEN0,
            &cancel,
            |layer| {
                assert_eq!(layer, completed_layers);
                completed_layers += 1;
            },
        )
        .unwrap();
    assert_eq!(completed_layers, 48);
    assert!(target0.binding_id() == target.bound_graph().binding_id());
    assert_eq!((target0.position(), target_sequence.position()), (0, 1));
    let target_counts = output_counts(target0.logits(), target0.hidden()).unwrap();
    let token1 = greedy(target0.logits()).unwrap();
    // No target-state clone, second target position or speculative rollback is
    // required for this conditioning smoke. Release its real recurrence now.
    drop(target_sequence);

    // Position0 has the defined zero preceding-hidden row. The draft's own h0
    // is deliberately not used for position1: that row uses the real target h0.
    assert_eq!(draft_sequence.position(), 0);
    let draft0 = mtp
        .step(
            &mut executor,
            &mut draft_sequence,
            TOKEN0,
            &zero_preceding,
            &cancel,
        )
        .unwrap();
    let draft0_counts = output_counts(&draft0.logits, &draft0.hidden).unwrap();
    assert_eq!(draft_sequence.position(), target0.position() + 1);
    let draft1 = mtp
        .step(
            &mut executor,
            &mut draft_sequence,
            token1,
            target0.hidden(),
            &cancel,
        )
        .unwrap();
    let draft1_counts = output_counts(&draft1.logits, &draft1.hidden).unwrap();
    assert_eq!(draft_sequence.position(), 2);
    drop(draft_sequence);
    drop(executor);
    target_file.verify_file_unchanged().unwrap();
    mtp_file.verify_file_unchanged().unwrap();

    // A fresh legacy target sequence is both the second and final target pass
    // and the exact old-API parity reference. No third expensive pass or full
    // target state checkpoint copy is introduced merely to check the adapter.
    let mut fresh_executor = ThreadedExecutor::new(THREADS).unwrap();
    let mut fresh_target_sequence = target.sequence(1).unwrap();
    let legacy = Zeroizing::new(
        target
            .step(
                &mut fresh_executor,
                &mut fresh_target_sequence,
                TOKEN0,
                &cancel,
                |_| {},
            )
            .unwrap(),
    );
    assert_eq!(fresh_target_sequence.position(), 1);
    let target_mismatches = bit_mismatches(target0.logits(), &legacy).unwrap();
    assert_eq!(target_mismatches, 0);
    drop(fresh_target_sequence);
    drop(legacy);

    let mut fresh_draft_sequence = mtp.sequence(2).unwrap();
    let repeated0 = mtp
        .step(
            &mut fresh_executor,
            &mut fresh_draft_sequence,
            TOKEN0,
            &zero_preceding,
            &cancel,
        )
        .unwrap();
    output_counts(&repeated0.logits, &repeated0.hidden).unwrap();
    assert_eq!(fresh_draft_sequence.position(), target0.position() + 1);
    let repeated1 = mtp
        .step(
            &mut fresh_executor,
            &mut fresh_draft_sequence,
            token1,
            target0.hidden(),
            &cancel,
        )
        .unwrap();
    output_counts(&repeated1.logits, &repeated1.hidden).unwrap();
    assert_eq!(fresh_draft_sequence.position(), 2);
    let draft_mismatches = bit_mismatches(&draft0.logits, &repeated0.logits).unwrap()
        + bit_mismatches(&draft0.hidden, &repeated0.hidden).unwrap()
        + bit_mismatches(&draft1.logits, &repeated1.logits).unwrap()
        + bit_mismatches(&draft1.hidden, &repeated1.hidden).unwrap();
    assert_eq!(draft_mismatches, 0);
    target_file.verify_file_unchanged().unwrap();
    mtp_file.verify_file_unchanged().unwrap();

    // Numeric counters only. Neither tokens, content, logits, residual values,
    // route histories nor speculative-throughput claims are serialized.
    eprintln!(
        "target_passes=2 draft_passes=4 target_positions=1 draft_positions=2 binding_rejections=2 target_logit_values={} target_hidden_values={} target_logits_nonzero={} target_hidden_nonzero={} draft0_logits_nonzero={} draft0_hidden_nonzero={} draft1_logits_nonzero={} draft1_hidden_nonzero={} target_replay_mismatches={} draft_replay_mismatches={}",
        VOCAB,
        WIDE,
        target_counts.logits_nonzero,
        target_counts.hidden_nonzero,
        draft0_counts.logits_nonzero,
        draft0_counts.hidden_nonzero,
        draft1_counts.logits_nonzero,
        draft1_counts.hidden_nonzero,
        target_mismatches,
        draft_mismatches
    );
}

#[test]
fn full_output_gate_rejects_partial_nonfinite_and_all_zero_payloads() {
    let mut logits = Zeroizing::new(vec![1.; VOCAB]);
    let mut hidden = Zeroizing::new(vec![1.; WIDE]);
    let counts = output_counts(&logits, &hidden).unwrap();
    assert_eq!(
        (counts.logits_nonzero, counts.hidden_nonzero),
        (VOCAB, WIDE)
    );
    assert!(output_counts(&logits[..VOCAB - 1], &hidden).is_err());
    assert!(output_counts(&logits, &hidden[..WIDE - 1]).is_err());
    logits[VOCAB - 1] = f32::NAN;
    assert!(output_counts(&logits, &hidden).is_err());
    logits[VOCAB - 1] = 1.;
    hidden[WIDE - 1] = f32::INFINITY;
    assert!(output_counts(&logits, &hidden).is_err());
    hidden.fill(0.);
    assert!(output_counts(&logits, &hidden).is_err());
    hidden.fill(1.);
    logits.fill(0.);
    assert!(output_counts(&logits, &hidden).is_err());
}

#[test]
fn replay_gate_compares_every_bit_and_rejects_length_changes_without_values() {
    assert_eq!(bit_mismatches(&[1., 0., 3.], &[1., 0., 3.]).unwrap(), 0);
    assert_eq!(bit_mismatches(&[1., 0., 3.], &[1., -0., 4.]).unwrap(), 2);
    assert!(bit_mismatches(&[1.], &[1., 2.]).is_err());
}
