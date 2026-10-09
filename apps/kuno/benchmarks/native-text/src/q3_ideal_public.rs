//! Conditional fixed-public Q3 residency replay with correct original experts.
//! Training/preload are outside timing. No request history is persisted and no
//! admission ledger is reset: each phase has a fresh complete original owner.
use crate::{
    executor::GpuExecutor, oracle_trace::Fixture, q3_public_reference::Reference,
    resident_model::ResidentModel,
};
use kuno_text::model::greedy;
use std::{collections::BTreeMap, sync::atomic::AtomicBool, time::Instant};
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
    resident_router: bool,
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    let selected = FixtureRun::new(fixture)?;
    let positions = selected.positions;
    let decode = selected.decode;
    reference.require_positions(positions)?;
    let (mut training, mut train_executor) = make(selected.context)?;
    let training_arithmetic = (
        train_executor.q3_integer_moe,
        train_executor.q3_skip_empty_segments,
    );
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
    if resident_router
        && training_arithmetic != (executor.q3_integer_moe, executor.q3_skip_empty_segments)
    {
        return Err("Q3 ideal resident training and replay arithmetic policies differ".into());
    }
    let initial = prepare(&mut replay, &mut executor, cancel)?;
    reference.verify(0, &initial)?;
    let first = greedy(&initial)?;
    drop(initial);
    let preload = replay.preload_lab_oracle(&mut executor, &trace, cancel)?;
    replay.begin_lab_oracle_replay(&mut executor, trace)?;
    if resident_router {
        replay.begin_lab_resident_router(&mut executor)?;
    }
    let before = (
        executor.expert_hits,
        replay.completed_cpu_misses(&executor)?,
        executor.expert_uploads,
        executor.uploaded_bytes,
    );
    let head_before = (
        executor.q6_integer_head_prepares,
        executor.q6_integer_head_tiles,
    );
    let resident_router_audits_before = executor.resident_router_audits;
    let resources = crate::public_resources::snapshot();
    let phase_before = replay.phase_ns.clone();
    let bindings_before = cubecl::wgpu::transient_binding_cache_stats();
    let mut reference_check_ns = 0u128;
    let mut greedy_check_ns = 0u128;
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
        let check_start = Instant::now();
        reference.verify(position, &output)?;
        reference_check_ns += check_start.elapsed().as_nanos();
        let check_start = Instant::now();
        let token = greedy(&output)?;
        greedy_check_ns += check_start.elapsed().as_nanos();
        if token != trained_ids[position] {
            return Err("Q3 ideal replay original prefix mismatch".into());
        }
        replayed_ids.push(token);
    }
    replay.finish_lab_oracle_replay(&mut executor)?;
    replay.end_lab_request_io_reuse()?;
    let elapsed_ns = clock.elapsed().as_nanos();
    let phase_ns: BTreeMap<_, _> = replay
        .phase_ns
        .iter()
        .map(|(&key, &value)| {
            value
                .checked_sub(*phase_before.get(key).unwrap_or(&0))
                .map(|delta| (key, delta))
                .ok_or("Q3 ideal phase counter regression")
        })
        .collect::<std::result::Result<_, _>>()?;
    let bindings_after = cubecl::wgpu::transient_binding_cache_stats();
    let binding_stats = bindings_after
        .iter()
        .zip(bindings_before)
        .map(|(&after, before)| {
            after
                .checked_sub(before)
                .ok_or("Q3 ideal binding counter regression")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
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
    let resident_router_audits = delta(
        executor.resident_router_audits,
        resident_router_audits_before,
    )?;
    if resident_router_audits != if resident_router { decode as u64 } else { 0 } {
        return Err("Q3 ideal complete per-token resident router audits".into());
    }
    let head_work = [
        delta(executor.q6_integer_head_prepares, head_before.0)?,
        delta(executor.q6_integer_head_tiles, head_before.1)?,
    ];
    let expected_head = if executor.q6_integer_head {
        [decode as u64, decode as u64 * 16]
    } else {
        [0, 0]
    };
    if head_work != expected_head {
        return Err("Q6 ideal replay complete prepared head contract".into());
    }
    let q6_integer_head = executor.q6_integer_head;
    let q3_integer_moe = executor.q3_integer_moe;
    let requested_skip_empty_segments = executor.q3_skip_empty_segments;
    let q3_shared_original_moe = executor.q3_shared_original_moe;
    let resident_router_policy = serde_json::json!({
        "arithmetic":if resident_router {Some(if q3_integer_moe {"prepared-two-residual-Q8"} else {"original-F32"})} else {None},
        "training_arithmetic":if resident_router {Some(if training_arithmetic.0 {"prepared-two-residual-Q8"} else {"original-F32"})} else {None},
        "requested_skip_empty_segments":resident_router.then_some(requested_skip_empty_segments),
        "replay_segment_policy":resident_router.then_some("both-original-72-slot-segments"),
        "mapper_preflight_info_charge_bytes":resident_router.then_some(crate::executor::batch_moe::RESIDENT_ROUTER_PREFLIGHT_INFO),
        "integer_clear_info_charge_bytes":if resident_router && q3_integer_moe {crate::executor::batch_moe::RESIDENT_ROUTER_INTEGER_CLEAR_INFO} else {0},
        "runtime_plan_bytes":resident_router.then_some(crate::executor::batch_moe::resident_router_runtime(q3_integer_moe)),
    });
    let replay_audit = close(&mut replay, &mut executor)?;
    println!("public_q3_ideal_probe_closed=true decode_steps={decode} elapsed_ns={elapsed_ns}");
    Ok(serde_json::json!({
        "scope":selected.scope,
        "positions":positions,"decode_steps":decode,"context_tokens":selected.context,"seed":424242,
        "train_full_vectors_accepted":true,"replay_full_vectors_accepted":true,
        "trace_membership_verified":true,"owned_io_cleanup_accepted":true,
        "full32_reference_accepted":selected.full32,"ends_before_known_router_divergence":!selected.full32,
        "expert_union_counts":counts,"preload_original_bytes":preload.bytes,
        "q6_integer_head":q6_integer_head,"q6_integer_head_prepares":head_work[0],"q6_integer_head_tiles":head_work[1],
        "resident_gpu_routing":resident_router,"resident_gpu_routing_default":false,
        "resident_router_owning_audits":resident_router_audits,
        "resident_router_audited_layers":resident_router_audits*48,
        "resident_router_policy":resident_router_policy,
        "q3_integer_moe":q3_integer_moe,"q3_shared_original_moe":q3_shared_original_moe,
        "elapsed_ns":elapsed_ns,"gpu_hits":work[0],"cpu_misses":work[1],
        "expert_uploads":work[2],"public_weight_upload_bytes":work[3],"resources":resources,
        "host_phase_ns":phase_ns,"binding_cache_stats":binding_stats,
        "reference_check_ns":reference_check_ns,"greedy_check_ns":greedy_check_ns,
        "phase_counters_are_host_wall_time_not_gpu_kernel_time":true,
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
