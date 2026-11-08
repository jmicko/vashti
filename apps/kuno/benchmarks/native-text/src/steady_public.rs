//! Explicit compiled-public settling diagnostic, separate from the original
//! complete-model acceptance and its exact allocation/owner inventory.
use crate::executor::GpuExecutor;
use crate::resident_model::ResidentModel;
use kuno_text::model::greedy;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroizing;

pub(crate) const STEPS: usize = 1024;
pub(crate) const WINDOW: usize = 128;
pub(crate) const PREFIX: usize = 128;
pub(crate) const CONTEXT: usize = 75 + PREFIX + STEPS;

#[derive(Clone, Copy)]
struct Snapshot {
    gpu: u64,
    cpu: u64,
    promotions: u64,
    bytes: u64,
    wall: u128,
}
impl Snapshot {
    fn take(r: &ResidentModel<'_, '_>, e: &GpuExecutor<'_>) -> kuno_text::Result<Self> {
        Ok(Self {
            gpu: e.expert_hits,
            cpu: r.completed_cpu_misses(e)?,
            promotions: e.adaptive.as_ref().map_or(0, |c| c.stats.promotions),
            bytes: e.adaptive.as_ref().map_or(0, |c| c.stats.bytes),
            wall: e.adaptive.as_ref().map_or(0, |c| c.stats.wall_ns),
        })
    }
    fn delta(self, before: Self) -> kuno_text::Result<Self> {
        Ok(Self {
            gpu: self
                .gpu
                .checked_sub(before.gpu)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
            cpu: self
                .cpu
                .checked_sub(before.cpu)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
            promotions: self
                .promotions
                .checked_sub(before.promotions)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
            bytes: self
                .bytes
                .checked_sub(before.bytes)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
            wall: self
                .wall
                .checked_sub(before.wall)
                .ok_or(kuno_text::Error::ArithmeticOverflow)?,
        })
    }
}

pub(crate) fn run<'f>(
    r: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    prompt: &[u32],
    public_prefix: &[u32],
    cancel: &AtomicBool,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if prompt.len() != 75 || public_prefix.len() != PREFIX {
        return Err("steady probe requires the complete fixed public prefix".into());
    }
    let adaptive = e.adaptive.is_some();
    let direct = std::env::var("KUNO_LAB_ADAPTIVE_DIRECT_UPLOAD").as_deref() == Ok("1");
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
        return Err("steady native projection policy".into());
    }
    crate::apply_native_decode_kernels(r, e, "single-q4-all", cancel)?;
    crate::apply_count1_hybrid_cap(r, e, Some(10), true, cancel)?;
    for (n, &expected) in public_prefix.iter().enumerate() {
        if greedy(&logits)? != expected {
            return Err("steady context original prefix mismatch".into());
        }
        if n + 1 < PREFIX {
            logits = crate::resident_decode_step(r, e, expected, cancel, true)?;
        }
    }
    drop(logits);
    let mut outputs = Zeroizing::new(Vec::with_capacity(STEPS));
    let mut windows = Vec::with_capacity(STEPS / WINDOW);
    for first in (0..STEPS).step_by(WINDOW) {
        let before = Snapshot::take(r, e)?;
        let resources = crate::public_resources::snapshot();
        let clock = Instant::now();
        for step in first..first + WINDOW {
            // Feed the last prefix token first, then repeat the fixed PUBLIC
            // sequence. Neither sampling nor model output chooses later inputs.
            let input = public_prefix[(step + PREFIX - 1) % PREFIX];
            let output = crate::resident_decode_step(r, e, input, cancel, true)?;
            outputs.push(greedy(&output)?);
        }
        let elapsed_ns = clock.elapsed().as_nanos();
        let resources =
            crate::public_resources::delta(resources, crate::public_resources::snapshot())
                .ok_or("steady resource counters unavailable")?;
        let work = Snapshot::take(r, e)?.delta(before)?;
        if work.gpu + work.cpu != WINDOW as u64 * 48 * 10 || work.promotions > WINDOW as u64 * 4 {
            return Err("steady completed work partition".into());
        }
        let window = serde_json::json!({
            "first_step":first,"steps":WINDOW,"elapsed_ns":elapsed_ns,
            "gpu_hits":work.gpu,"cpu_misses":work.cpu,"promotions":work.promotions,
            "upload_bytes":work.bytes,"promotions_wall_ns":work.wall,"resources":resources,
        });
        println!("public_steady_window={window}");
        windows.push(window);
    }
    r.close()?;
    if !r.erased_after_failure() {
        return Err("steady full owning close incomplete".into());
    }
    let mut audits = Vec::new();
    crate::report_public_owned_zero_audit(r, true, &mut audits)?;
    e.close_transient_resident()?;
    if e.ordered_source_pending_bytes() != 0 || e.adaptive.is_some() {
        return Err("steady source/cache close incomplete".into());
    }
    let audit = audits
        .pop()
        .ok_or("steady complete owning zero audit missing")?;
    println!("public_steady_probe_closed=true");
    Ok(serde_json::json!({
        "scope":"fixed-public-repeated-prefix-steady-diagnostic",
        "teacher_forced":true,"prefix_tokens":PREFIX,"prefix_matches":true,
        "steps":STEPS,"window_tokens":WINDOW,"context_tokens":CONTEXT,"seed":424242,
        "adaptive_experts":adaptive,"adaptive_direct_upload":direct,
        "windows":windows,"greedy_ids":&*outputs,
        "owned_io_cleanup_accepted":true,"owned_zero_audit":audit,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuation_starts_at_unconsumed_last_prefix_and_spans_eight_windows() {
        assert_eq!(CONTEXT, 1227);
        assert_eq!(STEPS / WINDOW, 8);
        assert_eq!((PREFIX - 1) % PREFIX, 127);
        assert_eq!((1 + PREFIX - 1) % PREFIX, 0);
        assert_eq!((STEPS - 1 + PREFIX - 1) % PREFIX, 126);
        let before = Snapshot {
            gpu: 10,
            cpu: 20,
            promotions: 1,
            bytes: 100,
            wall: 40,
        };
        let after = Snapshot {
            gpu: 11,
            cpu: 22,
            promotions: 2,
            bytes: 200,
            wall: 80,
        };
        let delta = after.delta(before).unwrap();
        assert_eq!(
            (
                delta.gpu,
                delta.cpu,
                delta.promotions,
                delta.bytes,
                delta.wall
            ),
            (1, 2, 1, 100, 40)
        );
        assert!(before.delta(after).is_err());
    }
}
