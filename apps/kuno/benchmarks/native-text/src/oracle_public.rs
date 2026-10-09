//! Compiled-public exact expert replay. Model arithmetic/router are unchanged;
//! only the original weights needed by seven correct decode steps are preloaded.
use crate::{executor::GpuExecutor, oracle_trace as trace, resident_model::ResidentModel};
use kuno_text::model::greedy;
use std::{sync::atomic::AtomicBool, time::Instant};
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
struct Snapshot {
    gpu: u64,
    cpu: u64,
    uploads: u64,
    bytes: u64,
}
impl Snapshot {
    fn take(r: &ResidentModel<'_, '_>, e: &GpuExecutor<'_>) -> kuno_text::Result<Self> {
        Ok(Self {
            gpu: e.expert_hits,
            cpu: r.completed_cpu_misses(e)?,
            uploads: e.expert_uploads,
            bytes: e.uploaded_bytes,
        })
    }
    fn delta(self, old: Self) -> kuno_text::Result<Self> {
        let checked = |new: u64, before: u64| {
            new.checked_sub(before)
                .ok_or(kuno_text::Error::ArithmeticOverflow)
        };
        Ok(Self {
            gpu: checked(self.gpu, old.gpu)?,
            cpu: checked(self.cpu, old.cpu)?,
            uploads: checked(self.uploads, old.uploads)?,
            bytes: checked(self.bytes, old.bytes)?,
        })
    }
}

fn prepare<'f>(
    r: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    prompt: &[u32],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    let mut logits = Zeroizing::new(Vec::new());
    let chunks = prompt.len().div_ceil(8);
    for (n, ids) in prompt.chunks(8).enumerate() {
        let purpose = if n + 1 == chunks {
            crate::resident_model::batch::Purpose::LastLogits
        } else {
            crate::resident_model::batch::Purpose::Advance
        };
        logits = r.prefill_chunk(e, ids, purpose, cancel)?;
    }
    if r.set_lab_native_projection_tile(e, 1)? != 1 {
        return Err("oracle native projection policy".into());
    }
    crate::apply_native_decode_kernels(r, e, "single-q4-all", cancel)?;
    crate::apply_count1_hybrid_cap(r, e, Some(10), true, cancel)?;
    Ok(logits)
}

fn close(r: &mut ResidentModel<'_, '_>, e: &mut GpuExecutor<'_>) -> Result<serde_json::Value> {
    r.close()?;
    if !r.erased_after_failure() {
        return Err("oracle full owning close incomplete".into());
    }
    let mut audits = Vec::new();
    crate::report_public_owned_zero_audit(r, true, &mut audits)?;
    e.close_transient_resident()?;
    if e.ordered_source_pending_bytes() != 0 || e.oracle_trace.is_some() || e.adaptive.is_some() {
        return Err("oracle source/trace close incomplete".into());
    }
    audits
        .pop()
        .ok_or_else(|| "oracle zero audit missing".into())
}

pub(crate) fn run<'m, 'f: 'm>(
    mut make: impl FnMut(usize) -> Result<(ResidentModel<'m, 'f>, GpuExecutor<'f>)>,
    prompt: &[u32],
    public_prefix: &[u32],
    cancel: &AtomicBool,
) -> Result<serde_json::Value> {
    if prompt.len() != trace::START || public_prefix.len() != trace::PREFIX {
        return Err("oracle requires the original compiled public prompt/prefix".into());
    }
    let (mut training, mut train_executor) = make(trace::CONTEXT)?;
    let mut logits = prepare(&mut training, &mut train_executor, prompt, cancel)?;
    training.begin_lab_oracle_trace(&mut train_executor)?;
    let mut trained_ids = Zeroizing::new(Vec::with_capacity(trace::PREFIX));
    for (n, &expected) in public_prefix.iter().enumerate() {
        let actual = greedy(&logits)?;
        if actual != expected {
            return Err("oracle training original prefix mismatch".into());
        }
        trained_ids.push(actual);
        if n + 1 < trace::PREFIX {
            logits = crate::resident_decode_step(
                &mut training,
                &mut train_executor,
                actual,
                cancel,
                true,
            )?;
        }
    }
    drop(logits);
    let replay_trace = training.finish_lab_oracle_training(&mut train_executor)?;
    let counts = replay_trace.counts();
    let training_audit = close(&mut training, &mut train_executor)?;
    drop(training);
    drop(train_executor);

    let (mut replay, mut executor) = make(trace::CONTEXT)?;
    let initial = prepare(&mut replay, &mut executor, prompt, cancel)?;
    if greedy(&initial)? != public_prefix[0] {
        return Err("oracle fresh replay prefill mismatch".into());
    }
    drop(initial);
    let preload = replay.preload_lab_oracle(&mut executor, &replay_trace, cancel)?;
    replay.begin_lab_oracle_replay(&mut executor, replay_trace)?;
    let mut replayed_ids = Zeroizing::new(Vec::with_capacity(trace::PREFIX));
    replayed_ids.push(public_prefix[0]);
    let before = Snapshot::take(&replay, &executor)?;
    let resources = crate::public_resources::snapshot();
    let clock = Instant::now();
    for step in 0..trace::DECODE {
        let output = crate::resident_decode_step(
            &mut replay,
            &mut executor,
            replayed_ids[step],
            cancel,
            true,
        )?;
        let actual = greedy(&output)?;
        if actual != public_prefix[step + 1] {
            return Err("oracle replay original prefix mismatch".into());
        }
        replayed_ids.push(actual);
    }
    let elapsed_ns = clock.elapsed().as_nanos();
    let resources = crate::public_resources::delta(resources, crate::public_resources::snapshot())
        .ok_or("oracle resource counters unavailable")?;
    let work = Snapshot::take(&replay, &executor)?.delta(before)?;
    if work.gpu != trace::DECODE as u64 * 48 * 10
        || work.cpu != 0
        || work.uploads != 0
        || work.bytes != 0
    {
        return Err("oracle timed execution was not entirely resident".into());
    }
    replay.finish_lab_oracle_replay(&mut executor)?;
    let replay_audit = close(&mut replay, &mut executor)?;
    println!("public_oracle_probe_closed=true");
    Ok(serde_json::json!({
        "scope":"fixed-public-exact-expert-replay-ceiling",
        "prompt_tokens":trace::START,"prefix_tokens":trace::PREFIX,
        "decode_steps":trace::DECODE,"context_tokens":trace::CONTEXT,"seed":424242,
        "train_prefix_matches":true,"replay_prefix_matches":true,
        "trace_membership_verified":true,"owned_io_cleanup_accepted":true,
        "expert_union_counts":counts,"preload_original_bytes":preload.bytes,
        "elapsed_ns":elapsed_ns,"gpu_hits":work.gpu,"cpu_misses":work.cpu,
        "expert_uploads":work.uploads,"public_weight_upload_bytes":work.bytes,
        "adaptive_promotions":0,"resources":resources,
        "train_greedy_ids":&*trained_ids,"replay_greedy_ids":&*replayed_ids,
        "training_owned_zero_audit":training_audit,"replay_owned_zero_audit":replay_audit,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn work_delta_does_not_hide_transfers_or_counter_underflow() {
        let old = Snapshot {
            gpu: 10,
            cpu: 2,
            uploads: 1,
            bytes: 100,
        };
        let next = Snapshot {
            gpu: 11,
            cpu: 3,
            uploads: 2,
            bytes: 200,
        };
        let work = next.delta(old).unwrap();
        assert_eq!(
            (work.gpu, work.cpu, work.uploads, work.bytes),
            (1, 1, 1, 100)
        );
        assert!(old.delta(next).is_err());
    }
}
