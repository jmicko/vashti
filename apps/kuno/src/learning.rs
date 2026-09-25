//! Bounded, content-free evidence from failed placement attempts. An unloaded
//! baseline is provisional: it cannot establish who held memory during a load.
use serde::Serialize;
use std::collections::VecDeque;

const TOLERANCE: u64 = 128 * 1024 * 1024;
const HISTORY_LIMIT: usize = 64;

/// Keep the final three readings, allowing delayed driver cleanup to settle.
/// An expired sampling interval must not manufacture a missing reading.
pub(crate) async fn settle<F, Fut>(
    mut probe: F,
    baseline: Option<MemorySample>,
) -> Vec<Option<MemorySample>>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Option<MemorySample>>,
{
    use tokio::time::{Duration, Instant, sleep, timeout};
    let interval = Duration::from_millis(200);
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut samples = Vec::with_capacity(3);
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if !samples.is_empty() && remaining < interval {
            break;
        }
        if samples.len() == 3 {
            samples.remove(0);
        }
        samples.push(timeout(remaining, probe()).await.unwrap_or(None));
        if let Some(sample) = stable(&samples)
            && baseline.is_none_or(|old| {
                old.total_bytes != sample.total_bytes
                    || old.free_bytes.saturating_sub(sample.free_bytes) <= TOLERANCE
            })
        {
            break;
        }
        if Instant::now() >= deadline {
            break;
        }
        sleep(interval).await;
    }
    samples
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct MemorySample {
    pub total_bytes: u64,
    pub free_bytes: u64,
}
impl MemorySample {
    pub fn new(total_bytes: u64, free_bytes: u64) -> Option<Self> {
        (total_bytes > 0 && free_bytes <= total_bytes).then_some(Self {
            total_bytes,
            free_bytes,
        })
    }
}

/// All readings must be present and agree on capacity. Use the least free
/// reading in the window so a brief release cannot inflate available capacity.
pub fn stable(samples: &[Option<MemorySample>]) -> Option<MemorySample> {
    if samples.len() < 3 {
        return None;
    }
    let first = samples[0]?;
    let mut low = first.free_bytes;
    let mut high = low;
    for sample in samples {
        let sample = (*sample)?;
        if MemorySample::new(sample.total_bytes, sample.free_bytes).is_none()
            || sample.total_bytes != first.total_bytes
        {
            return None;
        }
        low = low.min(sample.free_bytes);
        high = high.max(sample.free_bytes);
    }
    (high - low <= TOLERANCE).then_some(MemorySample {
        total_bytes: first.total_bytes,
        free_bytes: low,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Unexpected memory remains after the owned worker has been reaped. This
    /// may be another process or delayed driver cleanup; attribution is unknown.
    UnexplainedMemoryPressure,
    TelemetryUnavailable,
    MemoryUnstable,
    BaselineUnavailable,
    /// Even a recovered baseline is NOT proof that the failed combination is
    /// unsafe. A transient external consumer could already have released VRAM.
    RecoveredNeedsControlledRetest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    CpuSpill,
    EstimateExceeded,
    HeadroomExhausted,
    CapacityUnavailable,
    RuntimeOrVerification,
}
impl FailureKind {
    pub fn from_code(code: &str) -> Self {
        match code {
            "worker_cpu_spill_detected" | "managed_cpu_spill_detected" => Self::CpuSpill,
            "managed_peak_estimate_exceeded" => Self::EstimateExceeded,
            "managed_memory_headroom_exhausted" => Self::HeadroomExhausted,
            "managed_capacity_unavailable" | "managed_memory_release_timeout" => {
                Self::CapacityUnavailable
            }
            _ => Self::RuntimeOrVerification,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ResidentWorkload {
    pub model_digest: String,
    pub context: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct Failure {
    // Artifact identity and context only. Never a model alias or request text.
    pub model_digest: String,
    pub context: u32,
    pub kind: FailureKind,
    pub previous_residents: usize,
    pub resident_workloads: Vec<ResidentWorkload>,
    pub before_cleanup: Option<MemorySample>,
    pub after_cleanup: Vec<Option<MemorySample>>,
    pub outcome: Outcome,
    pub isolated_retry: bool,
}

#[derive(Clone, Default, Debug, Serialize)]
pub struct History {
    pub provisional_unloaded_baseline: Option<MemorySample>,
    pub failure_count: u64,
    pub failures: VecDeque<Failure>,
}
impl History {
    /// Called only with no owned runtime alive. Never label the calibration
    /// clean; external allocations can exist even at initial startup.
    pub fn calibrate(&mut self, samples: &[Option<MemorySample>]) {
        if let Some(sample) = stable(samples) {
            self.provisional_unloaded_baseline = Some(match self.provisional_unloaded_baseline {
                Some(old) if old.total_bytes == sample.total_bytes => MemorySample {
                    free_bytes: old.free_bytes.max(sample.free_bytes),
                    ..sample
                },
                _ => sample,
            });
        }
    }

    pub fn record(&mut self, mut failure: Failure) -> bool {
        failure.outcome = self.classify(&failure.after_cleanup);
        failure.isolated_retry = failure.isolated_retry
            && failure.previous_residents > 0
            && failure.kind != FailureKind::RuntimeOrVerification
            && failure.outcome == Outcome::RecoveredNeedsControlledRetest;
        let retry = failure.isolated_retry;
        // Classify against the PRIOR calibration before considering a new low
        // baseline. Otherwise a game occupying VRAM could normalize its own use.
        self.calibrate(&failure.after_cleanup);
        self.failure_count = self.failure_count.saturating_add(1);
        if self.failures.len() == HISTORY_LIMIT {
            self.failures.pop_front();
        }
        self.failures.push_back(failure);
        retry
    }

    fn classify(&self, samples: &[Option<MemorySample>]) -> Outcome {
        if samples.len() < 3 || samples.iter().any(Option::is_none) {
            return Outcome::TelemetryUnavailable;
        }
        let Some(sample) = stable(samples) else {
            return Outcome::MemoryUnstable;
        };
        let Some(baseline) = self.provisional_unloaded_baseline else {
            return Outcome::BaselineUnavailable;
        };
        if sample.total_bytes != baseline.total_bytes {
            return Outcome::BaselineUnavailable;
        }
        if baseline.free_bytes.saturating_sub(sample.free_bytes) > TOLERANCE {
            Outcome::UnexplainedMemoryPressure
        } else {
            Outcome::RecoveredNeedsControlledRetest
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1024 * 1024 * 1024;
    fn sample(free: u64) -> Option<MemorySample> {
        MemorySample::new(16 * GIB, free * GIB)
    }
    fn failure(samples: Vec<Option<MemorySample>>) -> Failure {
        Failure {
            model_digest: "a".repeat(64),
            context: 4096,
            kind: FailureKind::CpuSpill,
            previous_residents: 2,
            resident_workloads: vec![ResidentWorkload {
                model_digest: "b".repeat(64),
                context: 2048,
            }],
            before_cleanup: sample(1),
            after_cleanup: samples,
            outcome: Outcome::BaselineUnavailable,
            isolated_retry: true,
        }
    }
    #[test]
    fn external_or_delayed_allocations_do_not_poison_baseline_or_trigger_retry() {
        let mut h = History::default();
        h.calibrate(&[sample(15); 3]);
        assert!(!h.record(failure(vec![sample(9); 3])));
        assert_eq!(h.failures[0].outcome, Outcome::UnexplainedMemoryPressure);
        assert_eq!(h.provisional_unloaded_baseline, sample(15));
        // External use can disappear before cleanup: still never "unsafe".
        assert!(h.record(failure(vec![sample(15); 3])));
        assert_eq!(
            h.failures[1].outcome,
            Outcome::RecoveredNeedsControlledRetest
        );
    }
    #[test]
    fn missing_unstable_and_changed_capacity_are_inconclusive() {
        let mut h = History::default();
        assert!(!h.record(failure(vec![sample(15); 3])));
        assert_eq!(h.failures[0].outcome, Outcome::BaselineUnavailable);
        for (samples, expected) in [
            (
                vec![sample(15), None, sample(15)],
                Outcome::TelemetryUnavailable,
            ),
            (
                vec![sample(12), sample(14), sample(15)],
                Outcome::MemoryUnstable,
            ),
            (
                vec![MemorySample::new(32 * GIB, 31 * GIB); 3],
                Outcome::BaselineUnavailable,
            ),
        ] {
            assert!(!h.record(failure(samples)));
            assert_eq!(h.failures.back().unwrap().outcome, expected);
        }
        assert_eq!(MemorySample::new(0, 0), None);
        assert_eq!(MemorySample::new(1, 2), None);
    }
    #[test]
    fn retry_is_bounded_and_never_applies_to_runtime_errors_or_isolated_failures() {
        let mut h = History::default();
        h.calibrate(&[sample(15); 3]);
        let mut f = failure(vec![sample(15); 3]);
        f.isolated_retry = false;
        assert!(!h.record(f.clone()));
        f.isolated_retry = true;
        f.previous_residents = 0;
        assert!(!h.record(f.clone()));
        f.previous_residents = 2;
        f.kind = FailureKind::RuntimeOrVerification;
        assert!(!h.record(f));
    }
    #[test]
    fn history_is_bounded_and_contains_only_resource_evidence() {
        let mut h = History::default();
        for _ in 0..100 {
            h.record(failure(vec![sample(15); 3]));
        }
        assert_eq!(h.failure_count, 100);
        assert_eq!(h.failures.len(), 64);
        let json = serde_json::to_value(h).unwrap();
        assert!(json["failures"][0].get("prompt").is_none());
        let mut keys: Vec<_> = json["failures"][0]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "after_cleanup",
                "before_cleanup",
                "context",
                "isolated_retry",
                "kind",
                "model_digest",
                "outcome",
                "previous_residents",
                "resident_workloads"
            ]
        );
    }

    #[tokio::test(start_paused = true)]
    async fn delayed_release_settles_but_remaining_pressure_survives_the_deadline() {
        let mut calls = 0;
        let samples = settle(
            || {
                calls += 1;
                std::future::ready(sample(if calls == 1 { 9 } else { 15 }))
            },
            sample(15),
        )
        .await;
        assert_eq!(calls, 4);
        assert_eq!(stable(&samples), sample(15));
        let started = tokio::time::Instant::now();
        let samples = settle(|| std::future::ready(sample(9)), sample(15)).await;
        assert_eq!(samples, vec![sample(9); 3]);
        assert!(started.elapsed() >= std::time::Duration::from_millis(2800));
        assert!(started.elapsed() <= std::time::Duration::from_secs(3));
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_and_missing_probes_have_a_bounded_inconclusive_result() {
        let started = tokio::time::Instant::now();
        let samples = settle(
            || async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                sample(15)
            },
            sample(15),
        )
        .await;
        assert_eq!(samples, vec![None]);
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(3));
        let samples = settle(|| std::future::ready(None), sample(15)).await;
        assert_eq!(samples, vec![None; 3]);
    }
}
