// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
//! Cleanup-only policy shared by the real stream and host fault fixtures.
//! Establish this guard BEFORE scheduler execution: that execution can flush
//! and enter the ordinary unbounded submission regulator. Explicit cleanup
//! supplies its own checked/bounded drain after submitting the encoder.
use cubecl_runtime::server::ServerError;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Default)]
pub(crate) struct CleanupBoundary {
    active: AtomicBool,
    retired: AtomicBool,
}
pub(crate) struct CleanupGuard {
    state: Arc<CleanupBoundary>,
    accepted: bool,
}
impl CleanupBoundary {
    pub(crate) fn begin(self: &Arc<Self>, live_leases: usize) -> Result<CleanupGuard, ServerError> {
        // Reject BEFORE scheduler/encoder mutation or changing boundary state.
        if live_leases != 0 {
            return Err(super::transient_io::held_lease_fault());
        }
        if self.active.swap(true, Ordering::AcqRel) {
            return Err(super::transient_io::fault(
                "private I/O cleanup already active",
            ));
        }
        Ok(CleanupGuard {
            state: self.clone(),
            accepted: false,
        })
    }
    pub(crate) fn retired(&self) -> bool {
        self.retired.load(Ordering::Acquire)
    }
    pub(crate) fn retire(&self) {
        self.retired.store(true, Ordering::Release);
    }
    pub(crate) fn cleanup_timeout(&self) -> Option<std::time::Duration> {
        self.active
            .load(Ordering::Acquire)
            .then_some(std::time::Duration::from_secs(30))
    }
    pub(crate) fn ordinary_regulation(&self, wait: impl FnOnce()) {
        if !self.active.load(Ordering::Acquire) {
            wait();
        }
    }
}
impl CleanupGuard {
    pub(crate) fn run<T>(
        mut self,
        operation: impl FnOnce() -> Result<T, ServerError>,
    ) -> Result<T, ServerError> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
            Ok(Ok(value)) => {
                self.accepted = true;
                Ok(value)
            }
            Ok(Err(error)) => Err(error),
            Err(_) => Err(super::transient_io::cleanup_panic_fault()),
        }
    }
}
impl Drop for CleanupGuard {
    fn drop(&mut self) {
        if !self.accepted {
            self.state.retired.store(true, Ordering::Release);
        }
        self.state.active.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn cleanup_bypasses_ordinary_wait_and_success_restores_inference_policy() {
        let state = Arc::new(CleanupBoundary::default());
        let waits = Cell::new(0);
        assert_eq!(state.cleanup_timeout(), None);
        state.ordinary_regulation(|| waits.set(waits.get() + 1));
        state
            .begin(0)
            .unwrap()
            .run(|| {
                assert_eq!(
                    state.cleanup_timeout(),
                    Some(std::time::Duration::from_secs(30))
                );
                // Represents both scheduler B::flush and stream's final flush.
                state.ordinary_regulation(|| panic!("unbounded scheduler wait"));
                state.ordinary_regulation(|| panic!("unbounded final-flush wait"));
                Ok(())
            })
            .unwrap();
        assert!(!state.retired());
        assert_eq!(state.cleanup_timeout(), None);
        state.ordinary_regulation(|| waits.set(waits.get() + 1));
        assert_eq!(waits.get(), 2);
    }

    #[test]
    fn error_panic_and_abandonment_restore_boundary_and_retire_stream() {
        for failure in [0, 1, 2] {
            let state = Arc::new(CleanupBoundary::default());
            let guard = state.begin(0).unwrap();
            if failure == 2 {
                drop(guard);
            } else {
                assert!(
                    guard
                        .run::<()>(|| {
                            if failure == 0 {
                                Err(super::super::transient_io::fault(
                                    "synthetic bounded drain failure",
                                ))
                            } else {
                                panic!("synthetic cleanup panic")
                            }
                        })
                        .is_err()
                );
            }
            assert!(state.retired());
            assert!(!state.active.load(Ordering::Acquire));
            let waits = Cell::new(0);
            state.ordinary_regulation(|| waits.set(1));
            assert_eq!(waits.get(), 1);
            // Retirement is permanent even if best-effort retry cleans buffers.
            state.begin(0).unwrap().run(|| Ok(())).unwrap();
            assert!(state.retired());
        }
    }

    #[test]
    fn live_leases_reject_before_any_mutation_and_nested_entry_keeps_outer_guard() {
        let state = Arc::new(CleanupBoundary::default());
        assert!(state.begin(1).is_err());
        assert!(!state.active.load(Ordering::Acquire));
        assert!(!state.retired());
        let guard = state.begin(0).unwrap();
        assert!(state.begin(0).is_err());
        assert!(state.active.load(Ordering::Acquire));
        guard.run(|| Ok(())).unwrap();
        assert!(!state.active.load(Ordering::Acquire));
        assert!(!state.retired());
    }
}
