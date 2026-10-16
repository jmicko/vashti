//! Original private-arena cleanup exclusion for an external submitted reader.
//! Completion is credited only by the actual owning queue/fence consumer.
use std::{
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
#[derive(Clone)]
pub(crate) struct Block(Arc<AtomicBool>);
impl Block {
    pub(crate) fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }
    pub(crate) fn pending(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
    pub(crate) fn begin(&self, retired: Arc<AtomicBool>) -> Result<Flight, &'static str> {
        if retired.load(Ordering::Acquire)
            || self
                .0
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err("private external flight unavailable");
        }
        Ok(Flight {
            block: self.clone(),
            retired,
            active: true,
            _same_thread: PhantomData,
        })
    }
}
pub(crate) struct Flight {
    block: Block,
    retired: Arc<AtomicBool>,
    active: bool,
    _same_thread: PhantomData<Rc<()>>,
}
impl Flight {
    /// Caller owns the original queue completion, including failed submissions
    /// proved not pending. An attempted wait is not a completion receipt.
    pub(crate) fn joined(&mut self) -> Result<(), &'static str> {
        if !self.active || !self.block.0.swap(false, Ordering::AcqRel) {
            return Err("private external flight completion");
        }
        self.active = false;
        Ok(())
    }
}
impl Drop for Flight {
    fn drop(&mut self) {
        if self.active {
            self.retired.store(true, Ordering::Release);
        }
        // Unknown completion leaves the original arena blocked and quarantined.
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_cleanup_exclusion_needs_explicit_completion_and_rejects_stale_credit() {
        let b = Block::new();
        let retired = Arc::new(AtomicBool::new(false));
        let mut first = b.begin(retired.clone()).unwrap();
        assert!(b.pending());
        assert!(b.begin(retired.clone()).is_err());
        first.joined().unwrap();
        assert!(!b.pending());
        let mut next = b.begin(retired.clone()).unwrap();
        assert!(first.joined().is_err());
        assert!(b.pending());
        drop(first);
        assert!(!retired.load(Ordering::Acquire));
        next.joined().unwrap();
        drop(next);
        assert!(!b.pending());
    }
    #[test]
    fn lost_external_completion_keeps_original_cleanup_blocked_and_retires() {
        let b = Block::new();
        let retired = Arc::new(AtomicBool::new(false));
        drop(b.begin(retired.clone()).unwrap());
        assert!(b.pending());
        assert!(retired.load(Ordering::Acquire));
        assert!(b.begin(retired).is_err());
    }
}
