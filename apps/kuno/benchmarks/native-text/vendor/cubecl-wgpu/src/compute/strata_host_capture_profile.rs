//! Compiled-public-only capture intervals. No GPU commands or content samples.
use std::{
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::Instant,
};

pub const LABELS: [&str; 8] = [
    "inventory_signature",
    "snapshot_prepare",
    "queue_plan",
    "copy_dependencies",
    "compute_dependencies",
    "retained_prepare",
    "command_encoding",
    "finalize",
];
static TOTALS: [[AtomicU64; 4]; 8] = [const { [const { AtomicU64::new(0) }; 4] }; 8];
pub(super) fn selected(
    gpu_active: bool,
    enabled: bool,
    active: bool,
    public: bool,
    host_mode: bool,
) -> bool {
    gpu_active || (enabled && active && public && host_mode)
}
pub fn totals() -> [[u64; 4]; 8] {
    std::array::from_fn(|i| std::array::from_fn(|j| TOTALS[i][j].load(Relaxed)))
}
#[cfg(target_os = "linux")]
fn cpu_ns() -> Option<u64> {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: valid exclusively owned timespec; this samples only this thread's
    // CPU clock, never a stack, register, GPU allocation or inference payload.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) } != 0
        || ts.tv_sec < 0
        || !(0..1_000_000_000).contains(&ts.tv_nsec)
    {
        return None;
    }
    (ts.tv_sec as u64)
        .checked_mul(1_000_000_000)?
        .checked_add(ts.tv_nsec as u64)
}
#[cfg(not(target_os = "linux"))]
fn cpu_ns() -> Option<u64> {
    None
}

pub(super) struct Timer(Option<(Instant, Option<u64>)>);
impl Timer {
    pub fn new(public_active: bool) -> Self {
        Self(public_active.then(|| (Instant::now(), cpu_ns())))
    }
    pub fn mark(&mut self, phase: usize) {
        if let Some((wall, cpu)) = &mut self.0 {
            let now_wall = Instant::now();
            let now_cpu = cpu_ns();
            let elapsed = now_wall
                .duration_since(*wall)
                .as_nanos()
                .min(u64::MAX as u128) as u64;
            let thread = cpu.and_then(|before| now_cpu.and_then(|after| after.checked_sub(before)));
            // Fixed labels and anonymous durations only. Unavailable CPU time
            // is explicitly counted and cannot masquerade as zero CPU work.
            for (counter, value) in TOTALS[phase].iter().zip([
                1,
                elapsed,
                thread.unwrap_or(0),
                u64::from(thread.is_none()),
            ]) {
                counter.fetch_add(value, Relaxed);
            }
            *wall = now_wall;
            *cpu = now_cpu;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_only_selection_requires_every_original_public_guard() {
        for bits in 0..16 {
            let args = std::array::from_fn::<_, 4, _>(|i| bits & (1 << i) != 0);
            assert_eq!(
                selected(false, args[0], args[1], args[2], args[3]),
                bits == 15
            );
            assert!(selected(true, args[0], args[1], args[2], args[3]));
        }
    }
    #[test]
    fn host_mode_cannot_enable_gpu_timestamps_or_disable_original_rolling_path() {
        let s = include_str!("strata_graph.rs");
        let gpu =
            &s[s.find("fn profile_active()").unwrap()..s.find("fn host_profile_active()").unwrap()];
        assert!(!gpu.contains("prefill-host-layer"));
        let host =
            &s[s.find("fn host_profile_active()").unwrap()..s.find("type OperatorTotals").unwrap()];
        assert!(
            host.contains("prefill-host-layer") && host.contains("host_capture_profile::selected")
        );
        assert!(s.contains("let profiling = profile_active() && compute_count >= 2048;"));
        assert!(s.contains("!profile_active()"));
    }
    #[test]
    fn disabled_capture_samples_nothing_and_publishes_no_duration() {
        let before = totals()[7];
        let mut t = Timer::new(false);
        t.mark(7);
        assert_eq!(totals()[7], before);
        assert!(t.0.is_none());
    }
    #[test]
    fn real_thread_and_wall_clocks_have_distinct_explicit_availability() {
        let before = totals()[0];
        let mut t = Timer::new(true);
        let mut n = 1u64;
        for i in 1..10000 {
            n = std::hint::black_box(n.wrapping_mul(i));
        }
        std::hint::black_box(n);
        t.mark(0);
        let after = totals()[0];
        assert_eq!(after[0], before[0] + 1);
        assert!(after[1] > before[1]);
        #[cfg(target_os = "linux")]
        {
            assert!(after[2] > before[2]);
            assert_eq!(after[3], before[3]);
        }
        #[cfg(not(target_os = "linux"))]
        {
            assert_eq!(after[2], before[2]);
            assert_eq!(after[3], before[3] + 1);
        }
    }
    #[test]
    fn actual_graph_gates_all_eight_marks_with_existing_public_only_admission() {
        let s = include_str!("strata_graph.rs");
        assert_eq!(
            s.matches("host_capture_profile::Timer::new(host_profile_active())")
                .count(),
            1
        );
        for i in 0..8 {
            assert_eq!(s.matches(&format!("host_clock.mark({i});")).count(), 1);
        }
        let gate =
            &s[s.find("fn profile_active()").unwrap()..s.find("type OperatorTotals").unwrap()];
        assert!(
            gate.contains("KUNO_LAB_LANGUAGE_PUBLIC_ONLY")
                && gate.contains("KUNO_STRATA_NATIVE_PROFILE")
                && gate.contains("PROFILE_ACTIVE.load")
        );
        assert_eq!(LABELS.len(), 8);
    }
}
