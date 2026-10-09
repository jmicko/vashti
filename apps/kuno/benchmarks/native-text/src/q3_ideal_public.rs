//! Conditional fixed-public Q3 residency replay with correct original experts.
//! Training/preload are outside timing. No request history is persisted and no
//! admission ledger is reset: each phase has a fresh complete original owner.
use crate::{
    executor::GpuExecutor, oracle_trace::Fixture, q3_public_reference::Reference,
    resident_model::ResidentModel,
};
use kuno_text::model::greedy;
use std::{sync::atomic::AtomicBool, time::Instant};
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
struct FixtureRun {
    positions: usize,
    decode: usize,
    context: usize,
    scope: &'static str,
    full32: bool,
}
impl FixtureRun {
    fn new(fixture: Fixture) -> Result<Self> {
        let (scope, full32) = match fixture {
            Fixture::Q3Prefix23 => (
                "fixed-public-Q3-prefix23-conditional-ideal-residency",
                false,
            ),
            Fixture::DaslabPrefix13 => (
                "fixed-public-DASLab-prefix13-conditional-ideal-residency",
                false,
            ),
            Fixture::Daslab32 => ("fixed-public-DASLab32-conditional-ideal-residency", true),
            Fixture::Q4Rainwater => return Err("Q3 ideal fixture selection".into()),
        };
        Ok(Self {
            positions: fixture.prefix(),
            decode: fixture.decode(),
            context: fixture.context(),
            scope,
            full32,
        })
    }
}

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
    fixture: Fixture,
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    let selected = FixtureRun::new(fixture)?;
    let positions = selected.positions;
    let decode = selected.decode;
    reference.require_positions(positions)?;
    let (mut training, mut train_executor) = make(selected.context)?;
    let mut logits = prepare(&mut training, &mut train_executor, cancel)?;
    training.begin_lab_oracle_trace_for(&mut train_executor, fixture)?;
    let mut trained_ids = Zeroizing::new(Vec::with_capacity(positions));
    for position in 0..positions {
        reference.verify(position, &logits)?;
        let token = greedy(&logits)?;
        trained_ids.push(token);
        if position + 1 < positions {
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

    let (mut replay, mut executor) = make(selected.context)?;
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
    let mut replayed_ids = Zeroizing::new(Vec::with_capacity(positions));
    replayed_ids.push(first);
    let clock = Instant::now();
    for position in 1..positions {
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
    if work != [decode as u64 * 48 * 10, 0, 0, 0] {
        return Err("Q3 ideal timed execution was not entirely resident".into());
    }
    let replay_audit = close(&mut replay, &mut executor)?;
    println!("public_q3_ideal_probe_closed=true decode_steps={decode} elapsed_ns={elapsed_ns}");
    Ok(serde_json::json!({
        "scope":selected.scope,
        "positions":positions,"decode_steps":decode,"context_tokens":selected.context,"seed":424242,
        "train_full_vectors_accepted":true,"replay_full_vectors_accepted":true,
        "trace_membership_verified":true,"owned_io_cleanup_accepted":true,
        "full32_reference_accepted":selected.full32,"ends_before_known_router_divergence":!selected.full32,
        "expert_union_counts":counts,"preload_original_bytes":preload.bytes,
        "elapsed_ns":elapsed_ns,"gpu_hits":work[0],"cpu_misses":work[1],
        "expert_uploads":work[2],"public_weight_upload_bytes":work[3],"resources":resources,
        "train_greedy_ids":&*trained_ids,"replay_greedy_ids":&*replayed_ids,
        "training_owned_zero_audit":training_audit,"replay_owned_zero_audit":replay_audit,
        "timing_includes":"full-vector checks, original trace membership and audited request I/O close",
        "timing_excludes":"training, binding, prefill, preload, later full owning close and recovery",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_fixture_geometry_and_receipt_scope_cannot_claim_the_wrong_window() {
        let prefix = FixtureRun::new(Fixture::Q3Prefix23).unwrap();
        assert_eq!(
            (prefix.positions, prefix.decode, prefix.context),
            (23, 22, 24)
        );
        assert_eq!(
            prefix.scope,
            "fixed-public-Q3-prefix23-conditional-ideal-residency"
        );
        assert!(!prefix.full32);
        let das_prefix = FixtureRun::new(Fixture::DaslabPrefix13).unwrap();
        assert_eq!(
            (das_prefix.positions, das_prefix.decode, das_prefix.context),
            (13, 12, 14)
        );
        assert_eq!(
            das_prefix.scope,
            "fixed-public-DASLab-prefix13-conditional-ideal-residency"
        );
        assert!(!das_prefix.full32);
        let full = FixtureRun::new(Fixture::Daslab32).unwrap();
        assert_eq!((full.positions, full.decode, full.context), (32, 31, 33));
        assert_eq!(
            full.scope,
            "fixed-public-DASLab32-conditional-ideal-residency"
        );
        assert!(full.full32);
        assert!(FixtureRun::new(Fixture::Q4Rainwater).is_err());
        assert_eq!(
            (
                Fixture::Q4Rainwater.start(),
                Fixture::Q4Rainwater.decode(),
                Fixture::Q4Rainwater.prefix(),
                Fixture::Q4Rainwater.context()
            ),
            (75, 7, 8, 203)
        );
    }
}
