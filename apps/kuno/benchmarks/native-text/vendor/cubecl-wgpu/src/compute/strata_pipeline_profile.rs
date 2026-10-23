//! Fixed-public host wall time only. No pipeline identity, bindings or GPU work.
use std::{
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};

pub const LABELS: [&str; 2] = ["server_cache_miss_prepare", "native_driver_pipeline_create"];
pub const COLUMNS: [&str; 4] = [
    "success_count",
    "success_wall_ns",
    "failure_count",
    "failure_wall_ns",
];
static ACTIVE: AtomicBool = AtomicBool::new(false);
static TOTALS: [[AtomicU64; 4]; 2] = [const { [const { AtomicU64::new(0) }; 4] }; 2];

fn selected(native: bool, public: bool, mode: &str) -> bool {
    native && public && matches!(mode, "prefill-layer" | "prefill-host-layer")
}

/// Kept alive only by the existing compiled-public profiling worker. Dropping
/// the guard disables future samples on normal return, error or unwind.
pub struct PublicPipelineProfile(());
impl Drop for PublicPipelineProfile {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}
pub(super) fn begin(native: bool) -> Option<PublicPipelineProfile> {
    if !native {
        return None;
    }
    let public = std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1");
    let mode = std::env::var("KUNO_STRATA_NATIVE_PROFILE").unwrap_or_default();
    if !selected(native, public, &mode)
        || ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
    {
        return None;
    }
    Some(PublicPipelineProfile(()))
}
pub fn totals() -> Option<[[u64; 4]; 2]> {
    ACTIVE.load(Ordering::Acquire).then(|| {
        std::array::from_fn(|i| std::array::from_fn(|j| TOTALS[i][j].load(Ordering::Relaxed)))
    })
}

fn record(counters: &[AtomicU64; 4], success: bool, elapsed: u64) {
    let at = if success { 0 } else { 2 };
    for (counter, value) in [(&counters[at], 1), (&counters[at + 1], elapsed)] {
        let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
            Some(old.saturating_add(value))
        });
    }
}
#[derive(Clone, Copy)]
pub(super) enum Stage {
    Prepare = 0,
    NativeCreate = 1,
}
pub(super) struct Timer {
    stage: Stage,
    start: Option<Instant>,
}
impl Timer {
    pub(super) fn new(stage: Stage) -> Self {
        Self {
            stage,
            start: ACTIVE.load(Ordering::Acquire).then(Instant::now),
        }
    }
    pub(super) fn finish(mut self, success: bool) {
        self.record(success);
    }
    fn record(&mut self, success: bool) {
        if let Some(start) = self.start.take() {
            let elapsed = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            record(&TOTALS[self.stage as usize], success, elapsed);
        }
    }
}
impl Drop for Timer {
    fn drop(&mut self) {
        self.record(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pipeline_timing_requires_native_and_public_authorization_and_exact_profile_modes() {
        assert!(begin(false).is_none());
        for mode in ["prefill-layer", "prefill-host-layer"] {
            assert!(selected(true, true, mode));
            for (native, public) in [(false, false), (false, true), (true, false)] {
                assert!(!selected(native, public, mode));
            }
        }
        for mode in [
            "",
            "round-8",
            "warm-round-64",
            "prefill-layer-extra",
            "prefill-host-layer::other",
        ] {
            assert!(!selected(true, true, mode));
        }
    }
    #[test]
    fn disabled_pipeline_timer_retains_no_clock_and_records_no_success_or_failure() {
        let before: [[u64; 4]; 2] =
            std::array::from_fn(|i| std::array::from_fn(|j| TOTALS[i][j].load(Ordering::Relaxed)));
        for stage in [Stage::Prepare, Stage::NativeCreate] {
            let timer = Timer::new(stage);
            assert!(timer.start.is_none());
            timer.finish(true);
            drop(Timer::new(stage));
        }
        let after: [[u64; 4]; 2] =
            std::array::from_fn(|i| std::array::from_fn(|j| TOTALS[i][j].load(Ordering::Relaxed)));
        assert_eq!(after, before);
    }
    #[test]
    fn pipeline_timing_aggregates_success_and_failure_separately_without_wrapping() {
        let counters = [const { AtomicU64::new(0) }; 4];
        record(&counters, true, 11);
        record(&counters, true, 17);
        record(&counters, false, 5);
        assert_eq!(
            counters.each_ref().map(|v| v.load(Ordering::Relaxed)),
            [2, 28, 1, 5]
        );
        for v in &counters {
            v.store(u64::MAX - 1, Ordering::Relaxed);
        }
        record(&counters, true, 10);
        record(&counters, true, 10);
        record(&counters, false, 10);
        record(&counters, false, 10);
        assert_eq!(
            counters.each_ref().map(|v| v.load(Ordering::Relaxed)),
            [u64::MAX; 4]
        );
    }
}
