//! Conditional fixed-public Q3 prefix ceiling with correct original experts.
//! Training/preload are outside timing. No request history is persisted and no
//! admission ledger is reset: each phase has a fresh complete original owner.
use crate::{executor::GpuExecutor, q3_public_reference::Reference, resident_model::ResidentModel};
use kuno_text::model::greedy;
use std::{sync::atomic::AtomicBool, time::Instant};
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const POSITIONS: usize = 23;
const DECODE: usize = POSITIONS - 1;
const CONTEXT: usize = POSITIONS + 1;

fn prepare<'f>(
    r: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    let logits = r.prefill_chunk(
        e,
        &[0],
        crate::resident_model::batch::Purpose::LastLogits,
        cancel,
    )?;
    if r.set_lab_native_projection_tile(e, 1)? != 1 {
        return Err("Q3 ideal projection policy".into());
    }
    crate::apply_native_decode_kernels(r, e, "single-all", cancel)?;
    crate::apply_count1_hybrid_cap(r, e, Some(10), true, cancel)?;
    Ok(logits)
}

fn close(r: &mut ResidentModel<'_, '_>, e: &mut GpuExecutor<'_>) -> Result<serde_json::Value> {
    r.close()?;
    if !r.erased_after_failure() {
        return Err("Q3 ideal full owning close incomplete".into());
    }
    let mut audits = Vec::new();
    crate::report_public_owned_zero_audit(r, true, &mut audits)?;
    e.close_transient_resident()?;
    if e.ordered_source_pending_bytes() != 0 || e.oracle_trace.is_some() {
        return Err("Q3 ideal source/trace close incomplete".into());
    }
    audits.pop().ok_or_else(|| "Q3 ideal audit absent".into())
}

pub(crate) fn run<'m, 'f: 'm>(
    mut make: impl FnMut(usize) -> Result<(ResidentModel<'m, 'f>, GpuExecutor<'f>)>,
    reference: &Reference,
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    reference.require_positions(POSITIONS)?;
    let (mut training, mut train_executor) = make(CONTEXT)?;
    let mut logits = prepare(&mut training, &mut train_executor, cancel)?;
    training.begin_lab_q3_oracle_trace(&mut train_executor)?;
    let mut trained_ids = Zeroizing::new(Vec::with_capacity(POSITIONS));
    for position in 0..POSITIONS {
        reference.verify(position, &logits)?;
        let token = greedy(&logits)?;
        trained_ids.push(token);
        if position + 1 < POSITIONS {
            logits = crate::resident_decode_step(
                &mut training,
                &mut train_executor,
                token,
                cancel,
                true,
            )?;
        }
    }
    drop(logits);
    let trace = training.finish_lab_oracle_training(&mut train_executor)?;
    let counts = trace.counts();
    let training_audit = close(&mut training, &mut train_executor)?;
    drop(training);
    drop(train_executor);

    let (mut replay, mut executor) = make(CONTEXT)?;
    let initial = prepare(&mut replay, &mut executor, cancel)?;
    reference.verify(0, &initial)?;
    let first = greedy(&initial)?;
    drop(initial);
    let preload = replay.preload_lab_oracle(&mut executor, &trace, cancel)?;
    replay.begin_lab_oracle_replay(&mut executor, trace)?;
    let before = (
        executor.expert_hits,
        replay.completed_cpu_misses(&executor)?,
        executor.expert_uploads,
        executor.uploaded_bytes,
    );
    let resources = crate::public_resources::snapshot();
    let mut replayed_ids = Zeroizing::new(Vec::with_capacity(POSITIONS));
    replayed_ids.push(first);
    let clock = Instant::now();
    for position in 1..POSITIONS {
        let output = crate::resident_decode_step(
            &mut replay,
            &mut executor,
            replayed_ids[position - 1],
            cancel,
            true,
        )?;
        reference.verify(position, &output)?;
        let token = greedy(&output)?;
        if token != trained_ids[position] {
            return Err("Q3 ideal replay original prefix mismatch".into());
        }
        replayed_ids.push(token);
    }
    replay.finish_lab_oracle_replay(&mut executor)?;
    replay.end_lab_request_io_reuse()?;
    let elapsed_ns = clock.elapsed().as_nanos();
    let resources = crate::public_resources::delta(resources, crate::public_resources::snapshot())
        .ok_or("Q3 ideal resource counters unavailable")?;
    let after = (
        executor.expert_hits,
        replay.completed_cpu_misses(&executor)?,
        executor.expert_uploads,
        executor.uploaded_bytes,
    );
    let delta = |a: u64, b: u64| a.checked_sub(b).ok_or("Q3 ideal counter regression");
    let work = [
        delta(after.0, before.0)?,
        delta(after.1, before.1)?,
        delta(after.2, before.2)?,
        delta(after.3, before.3)?,
    ];
    if work != [DECODE as u64 * 48 * 10, 0, 0, 0] {
        return Err("Q3 ideal timed execution was not entirely resident".into());
    }
    let replay_audit = close(&mut replay, &mut executor)?;
    println!("public_q3_ideal_probe_closed=true decode_steps={DECODE} elapsed_ns={elapsed_ns}");
    Ok(serde_json::json!({
        "scope":"fixed-public-Q3-prefix23-conditional-ideal-residency",
        "positions":POSITIONS,"decode_steps":DECODE,"context_tokens":CONTEXT,"seed":424242,
        "train_full_vectors_accepted":true,"replay_full_vectors_accepted":true,
        "trace_membership_verified":true,"owned_io_cleanup_accepted":true,
        "full32_reference_accepted":false,"ends_before_known_router_divergence":true,
        "expert_union_counts":counts,"preload_original_bytes":preload.bytes,
        "elapsed_ns":elapsed_ns,"gpu_hits":work[0],"cpu_misses":work[1],
        "expert_uploads":work[2],"public_weight_upload_bytes":work[3],"resources":resources,
        "train_greedy_ids":&*trained_ids,"replay_greedy_ids":&*replayed_ids,
        "training_owned_zero_audit":training_audit,"replay_owned_zero_audit":replay_audit,
        "timing_includes":"full-vector checks, original trace membership and audited request I/O close",
        "timing_excludes":"training, binding, prefill, preload, later full owning close and recovery",
    }))
}
