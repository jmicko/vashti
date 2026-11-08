//! Fixed public original Strata Q2 GGUF MTP full-vector oracle; no prompt arguments.
//!
//! Aligned version2 teacher cells0..3 consume target T[c+1] + real target R[c]
//! at RoPE cell c, predicting T[c+2], with GLOBAL10240 input hidden RMS. There
//! is no dummy/zero-HC cell. Provisional cells0/1 use target T1/R0 followed by
//! their own greedy0/HC0. Primary convention/source links are in strata_mtp_public_reference.
//! Dense weights remain BF16, raw Gemma norms fold exactly once, and the
//! actual DAS target binds borrowed IQ4_XS embedding/Q6_K full output head.
//! Requires core global-input-norm correction724ca2ff7ca307399b0665e795e19159a22ac2f8;
//! the protected runner must pin the compiled source before generating v2.
//! Full publisher/target-fixture SHA256 checks belong to the protected runner.
//! CPU scalar helpers have no whole-stack private-erasure acceptance; this
//! writer is exclusively a compiled public fixture, not an inference route.

#[path = "../../../apps/kuno/benchmarks/native-text/src/daslab_carry_reference.rs"]
pub mod daslab_carry_reference;
#[path = "../../../apps/kuno/benchmarks/native-text/src/daslab_profile.rs"]
pub mod daslab_profile;
#[path = "../../../apps/kuno/benchmarks/native-text/src/strata_mtp_public_reference.rs"]
pub mod strata_mtp_public_reference;

use kuno_text::model::{MatrixExecutor, Qwen4Exp, greedy};
use kuno_text::mtp::{MtpGraph, MtpOutput, MtpProfile, MtpSequence};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{Error, GgufFile, GgufLimits, Qwen35Tokenizer, TensorView, TokenizerLimits};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use strata_mtp_public_reference::{CONTEXT, Conditioning, HIDDEN, Reference, Row, TEACHER_ROWS};

// Five split-FC calls replace four EH calls; each routed expert uses fused
// gate/up +down (20 calls); shared BF16 scalar adds one matrix call.
const STEP_PROJECTIONS: usize = 43;
const APPEND_PROJECTIONS: usize = 9;
const EXPECTED_PROJECTIONS: usize = 12 * STEP_PROJECTIONS + 2 * APPEND_PROJECTIONS;

struct Inputs {
    sidecar: PathBuf,
    directory: PathBuf,
    target: PathBuf,
    output: PathBuf,
}
fn registration(
    args: &[OsString],
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    default_build: bool,
) -> Result<Inputs, &'static str> {
    if args.len() != 1 || !default_build {
        return Err("public MTP writer requires default build and no prompt arguments");
    }
    let mut settings = BTreeMap::new();
    for (key, value) in environment {
        if !(key.as_encoded_bytes().starts_with(b"KUNO_LAB_")
            || key.as_encoded_bytes().starts_with(b"KUNO_TEST_")
            || key.as_encoded_bytes().starts_with(b"KUNO_PUBLIC_"))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_DASLAB_STRATA_MTP_PUBLIC_ONLY"
                    | "KUNO_TEST_STRATA_MTP"
                    | "KUNO_TEST_DASLAB_DIR"
                    | "KUNO_PUBLIC_DASLAB_CARRY"
                    | "KUNO_PUBLIC_STRATA_MTP_REFERENCE"
            )
        ) || value.is_empty()
            || settings.insert(key, value).is_some()
        {
            return Err("public MTP writer rejects other inputs/settings");
        }
    }
    if settings.remove(OsStr::new("KUNO_LAB_DASLAB_STRATA_MTP_PUBLIC_ONLY")) != Some("1".into()) {
        return Err("public MTP writer requires explicit public-only gate");
    }
    Ok(Inputs {
        directory: settings
            .remove(OsStr::new("KUNO_TEST_DASLAB_DIR"))
            .ok_or("verified original DASLab two-shard directory required")?
            .into(),
        sidecar: settings
            .remove(OsStr::new("KUNO_TEST_STRATA_MTP"))
            .ok_or("verified original Strata Q2 GGUF required")?
            .into(),
        target: settings
            .remove(OsStr::new("KUNO_PUBLIC_DASLAB_CARRY"))
            .ok_or("verified original DAS target carry fixture required")?
            .into(),
        output: settings
            .remove(OsStr::new("KUNO_PUBLIC_STRATA_MTP_REFERENCE"))
            .ok_or("fresh fixed public MTP output required")?
            .into(),
    })
}

struct Counted {
    inner: ThreadedExecutor,
    calls: usize,
}
impl MatrixExecutor for Counted {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> kuno_text::Result<Vec<f32>> {
        self.calls = self.calls.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        self.inner.run(tensor, expert, input, cancel)
    }
}
fn check_progress(
    executor: &Counted,
    before: usize,
    expected_calls: usize,
    sequence: &MtpSequence,
    expected_cursor: usize,
) -> kuno_text::Result<()> {
    if executor.calls.checked_sub(before) != Some(expected_calls)
        || sequence.position() != expected_cursor
    {
        return Err(Error::InvalidFormat(
            "public MTP complete projection/cursor",
        ));
    }
    Ok(())
}
fn row(
    mode: Conditioning,
    position: usize,
    input: u32,
    output: MtpOutput,
) -> kuno_text::Result<Row> {
    Row::new(
        mode,
        position,
        input,
        position + 1,
        output.logits,
        output.hidden,
    )
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = registration(
        &std::env::args_os().collect::<Vec<_>>(),
        std::env::vars_os(),
        !cfg!(any(
            feature = "cpu-q8-two-pass",
            feature = "cpu-work-stealing-lab"
        )),
    )?;
    let target = daslab_carry_reference::Reference::load(&inputs.target)?;
    let file = GgufFile::open(&inputs.sidecar, GgufLimits::default())?;
    strata_mtp_public_reference::require_sidecar(&file)?;
    let target_paths = strata_mtp_public_reference::target_paths(&inputs.directory);
    for (path, bytes) in target_paths.iter().zip(daslab_carry_reference::SHARD_BYTES) {
        if std::fs::metadata(path)?.len() != bytes {
            return Err("public Strata MTP actual target shard extent".into());
        }
    }
    let target_file = GgufFile::open_shards(&target_paths, GgufLimits::default())?;
    strata_mtp_public_reference::require_target(&target_file)?;
    let tokenizer = Qwen35Tokenizer::from_gguf(&target_file, TokenizerLimits::default())?;
    if tokenizer.encode("!", true)? != [0] {
        return Err("public Strata MTP fixed target token0 rejected".into());
    }
    let target_graph = Qwen4Exp::bind(&target_file)?;
    let target_binding = target_graph.bound_graph().binding_id();
    let graph = MtpGraph::bind_strata_q2(&file, target_graph.bound_graph())?;
    if graph.profile() != MtpProfile::StrataQ2 || graph.target_binding_id() != Some(target_binding)
    {
        return Err("public Strata MTP actual borrowed target binding rejected".into());
    }
    let mut executor = Counted {
        inner: ThreadedExecutor::new(strata_mtp_public_reference::THREADS)?,
        calls: 0,
    };
    if executor.inner.threads() != strata_mtp_public_reference::THREADS {
        return Err("public MTP fixed CPU threads rejected".into());
    }
    let cancel = AtomicBool::new(false);

    // Two bindings of identical source still have separate sequence domains.
    // Foreign sequence rejection must precede all projections and cache writes.
    {
        let foreign_graph = MtpGraph::bind_strata_q2(&file, target_graph.bound_graph())?;
        let mut foreign = foreign_graph.sequence(CONTEXT)?;
        let before = executor.calls;
        if graph
            .step(
                &mut executor,
                &mut foreign,
                target.token(0)?,
                target.hidden(0)?,
                &cancel,
            )
            .is_ok()
        {
            return Err("public MTP foreign binding accepted".into());
        }
        check_progress(&executor, before, 0, &foreign, 0)?;
        foreign.reset();
    }

    let mut rows = Vec::new();
    rows.try_reserve_exact(strata_mtp_public_reference::ROWS)?;
    if rows.capacity() != strata_mtp_public_reference::ROWS {
        return Err("public MTP exact row capacity".into());
    }
    let mut sequence = graph.sequence(CONTEXT)?;
    for position in 0..TEACHER_ROWS {
        let input = target.token(position)?;
        let before = executor.calls;
        let output = graph.step(
            &mut executor,
            &mut sequence,
            input,
            target.hidden(position)?,
            &cancel,
        )?;
        check_progress(&executor, before, STEP_PROJECTIONS, &sequence, position + 1)?;
        rows.push(row(Conditioning::TeacherForced, position, input, output)?);
    }
    sequence.reset();
    check_progress(&executor, executor.calls, 0, &sequence, 0)?;

    let before = executor.calls;
    let input = target.token(0)?;
    let output = graph.step(
        &mut executor,
        &mut sequence,
        input,
        target.hidden(0)?,
        &cancel,
    )?;
    check_progress(&executor, before, STEP_PROJECTIONS, &sequence, 1)?;
    rows.push(row(Conditioning::ProvisionalTarget, 0, input, output)?);
    let first = &rows[TEACHER_ROWS];
    let input = first.token();
    let before = executor.calls;
    let output = graph.step(&mut executor, &mut sequence, input, first.hidden(), &cancel)?;
    check_progress(&executor, before, STEP_PROJECTIONS, &sequence, 2)?;
    rows.push(row(Conditioning::ProvisionalDraft, 1, input, output)?);
    sequence.reset();
    let reference = Reference::new(rows, &target)?;

    // Fresh allocation, original CPU execution and exact complete-vector bits.
    // Use the replay's own draft output for its second provisional conditioning.
    let mut fresh = graph.sequence(CONTEXT)?;
    let mut replay_first = None;
    for position in 0..TEACHER_ROWS {
        let input = target.token(position)?;
        let before = executor.calls;
        let output = graph.step(
            &mut executor,
            &mut fresh,
            input,
            target.hidden(position)?,
            &cancel,
        )?;
        check_progress(&executor, before, STEP_PROJECTIONS, &fresh, position + 1)?;
        reference.verify(
            Conditioning::TeacherForced,
            position,
            input,
            fresh.position(),
            &output.logits,
            &output.hidden,
        )?;
        reference.verify_bits(
            Conditioning::TeacherForced,
            position,
            &output.logits,
            &output.hidden,
        )?;
        if position == 0 {
            // This guarded output is also the fresh first provisional cell.
            // Retain its actual HC for draft1; never clone the vocabulary.
            replay_first = Some(output);
        }
    }
    fresh.reset();
    let before = executor.calls;
    graph.append_kv(
        &mut executor,
        &mut fresh,
        target.token(0)?,
        target.hidden(0)?,
        &cancel,
    )?;
    check_progress(&executor, before, APPEND_PROJECTIONS, &fresh, 1)?;
    let first = replay_first.ok_or("public MTP fresh first cell missing")?;
    reference.verify(
        Conditioning::ProvisionalTarget,
        0,
        target.token(0)?,
        fresh.position(),
        &first.logits,
        &first.hidden,
    )?;
    reference.verify_bits(
        Conditioning::ProvisionalTarget,
        0,
        &first.logits,
        &first.hidden,
    )?;
    let input = greedy(&first.logits)?;
    let before = executor.calls;
    let second = graph.step(&mut executor, &mut fresh, input, &first.hidden, &cancel)?;
    check_progress(&executor, before, STEP_PROJECTIONS, &fresh, 2)?;
    reference.verify(
        Conditioning::ProvisionalDraft,
        1,
        input,
        fresh.position(),
        &second.logits,
        &second.hidden,
    )?;
    reference.verify_bits(
        Conditioning::ProvisionalDraft,
        1,
        &second.logits,
        &second.hidden,
    )?;
    drop(second);
    drop(first);
    fresh.reset();
    check_progress(&executor, executor.calls, 0, &fresh, 0)?;

    // Independent KV-only catch-up of actual cell0 followed by teacher cell1
    // must reproduce the complete normal step's teacher1 outputs exactly.
    // This second append preserves the fixed12step/2append projection bound.
    let before = executor.calls;
    graph.append_kv(
        &mut executor,
        &mut fresh,
        target.token(0)?,
        target.hidden(0)?,
        &cancel,
    )?;
    check_progress(&executor, before, APPEND_PROJECTIONS, &fresh, 1)?;
    let before = executor.calls;
    let input = target.token(1)?;
    let catchup = graph.step(&mut executor, &mut fresh, input, target.hidden(1)?, &cancel)?;
    check_progress(&executor, before, STEP_PROJECTIONS, &fresh, 2)?;
    reference.verify(
        Conditioning::TeacherForced,
        1,
        input,
        fresh.position(),
        &catchup.logits,
        &catchup.hidden,
    )?;
    reference.verify_bits(
        Conditioning::TeacherForced,
        1,
        &catchup.logits,
        &catchup.hidden,
    )?;
    drop(catchup);
    fresh.reset();
    check_progress(&executor, executor.calls, 0, &fresh, 0)?;
    if executor.calls != EXPECTED_PROJECTIONS {
        return Err("public MTP complete execution count".into());
    }
    strata_mtp_public_reference::require_sidecar(&file)?;
    strata_mtp_public_reference::require_target(&target_file)?;
    graph.verify_unchanged()?;
    let agreements = reference.agreements(&target)?;
    // Create no partial file before full source/binding/cursor/fresh-bit gates.
    // A later I/O failure cannot pass the strict exact-length/header parser.
    reference.write_new(&inputs.output)?;
    println!(
        "public_daslab_strata_mtp_reference_complete=true profile=strata_original_gguf_q2_0 convention_version=2 raw_gemma_fold_count=1 borrowed_target_tensors=2 separate_fcs=2 hidden_norm_width=10240 dummy_cells=0 teacher_rows=4 provisional_rows=2 executions=12 kv_appends=2 projections={} replay_bit_mismatches=0 logits_values={} hidden_values={} teacher_greedy_agreements={} teacher_greedy_compared={} provisional_first_greedy_agreements={} provisional_second_greedy_agreements={} provisional_greedy_compared=2 bytes={} abs_tolerance={} rel_tolerance={} private_request_accepted=false whole_owning_erasure_accepted=false quality_accepted=false performance_accepted=false",
        executor.calls,
        strata_mtp_public_reference::VOCAB,
        HIDDEN,
        agreements.teacher_forced,
        agreements.teacher_compared,
        agreements.provisional_first,
        agreements.provisional_second,
        strata_mtp_public_reference::FILE_BYTES,
        strata_mtp_public_reference::ABS_TOLERANCE,
        strata_mtp_public_reference::REL_TOLERANCE,
    );
    Ok(())
}
fn main() -> Result<(), &'static str> {
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_daslab_strata_mtp_writer_failure panic=true")
    }));
    run().map_err(|_| "fixed public DASLab MTP writer failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn environment() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_DASLAB_STRATA_MTP_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_STRATA_MTP", "/public-original-q2.gguf"),
            ("KUNO_TEST_DASLAB_DIR", "/public-daslab-shards"),
            ("KUNO_PUBLIC_DASLAB_CARRY", "/public-target-carry"),
            ("KUNO_PUBLIC_STRATA_MTP_REFERENCE", "/fresh-public-mtp"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .into()
    }
    #[test]
    fn only_explicit_default_public_writer_configuration_is_admitted() {
        let args = ["writer".into()];
        assert!(registration(&args, environment(), true).is_ok());
        assert!(registration(&args, environment(), false).is_err());
        assert!(registration(&[], environment(), true).is_err());
        assert!(registration(&["writer".into(), "prompt".into()], environment(), true).is_err());
        for index in 0..environment().len() {
            let mut missing = environment();
            missing.remove(index);
            assert!(registration(&args, missing, true).is_err());
            let mut empty = environment();
            empty[index].1 = "".into();
            assert!(registration(&args, empty, true).is_err());
        }
        for key in [
            "KUNO_LAB_FAST_GPU",
            "KUNO_TEST_GGUF",
            "KUNO_PUBLIC_Q3_REFERENCE",
        ] {
            let mut other = environment();
            other.push((key.into(), "0".into()));
            assert!(registration(&args, other, true).is_err());
        }
        let mut disabled = environment();
        disabled[0].1 = "0".into();
        assert!(registration(&args, disabled, true).is_err());
        let mut duplicate = environment();
        duplicate.push(duplicate[0].clone());
        assert!(registration(&args, duplicate, true).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn non_unicode_inherited_setting_is_rejected() {
        use std::os::unix::ffi::OsStringExt;
        let mut other = environment();
        other.push((OsString::from_vec(b"KUNO_LAB_\xff".to_vec()), "0".into()));
        assert!(registration(&["writer".into()], other, true).is_err());
    }
    #[test]
    fn fixed_projection_and_wire_extents_are_explicit() {
        assert_eq!(EXPECTED_PROJECTIONS, 534);
        assert_eq!(strata_mtp_public_reference::FILE_BYTES, 6_206_016);
        assert_eq!(strata_mtp_public_reference::SIDECAR_BYTES, 889_017_216);
        assert_eq!(strata_mtp_public_reference::TENSOR_COUNT, 31);
        assert_eq!(daslab_carry_reference::POSITIONS, TEACHER_ROWS);
        assert_eq!(daslab_carry_reference::HIDDEN, HIDDEN);
        assert_eq!(
            daslab_carry_reference::VOCAB,
            strata_mtp_public_reference::VOCAB
        );
    }
}
