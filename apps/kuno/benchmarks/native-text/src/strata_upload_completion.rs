//! One actual PUBLIC upload completion owns its destination roots until joined.
//! Unknown/error/panic completion quarantines owners; no readiness is invented.
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

pub(super) struct Pending<'a, R, E> {
    roots: &'a mut Vec<R>,
    fallback: Option<Box<dyn FnOnce() -> Result<(), E> + 'a>>,
    completion: Option<Box<dyn FnOnce() -> Result<(), E> + 'a>>,
    armed: bool,
}
impl<'a, R, E> Pending<'a, R, E> {
    /// Arm before scheduling/creating a completion future, so even a panic in
    /// that synchronous setup takes an actual drain through the owning client.
    pub(super) fn new(
        roots: &'a mut Vec<R>,
        fallback: impl FnOnce() -> Result<(), E> + 'a,
    ) -> Self {
        Self {
            roots,
            fallback: Some(Box::new(fallback)),
            completion: None,
            armed: true,
        }
    }
    pub(super) fn submitted(&mut self, join: impl FnOnce() -> Result<(), E> + 'a) {
        assert!(
            self.armed && self.completion.is_none(),
            "original public upload completion"
        );
        self.completion = Some(Box::new(join));
    }
    fn quarantine(&mut self) {
        // The process/driver owns final teardown after terminal load failure.
        // Never free roots or claim a reusable/erased model after unknown work.
        std::mem::forget(std::mem::take(self.roots));
        self.armed = false;
    }
    pub(super) fn finish(mut self) -> Result<(), E> {
        let join = self
            .completion
            .take()
            .or_else(|| self.fallback.take())
            .expect("armed public upload completion");
        match catch_unwind(AssertUnwindSafe(join)) {
            Ok(Ok(())) => {
                self.armed = false;
                Ok(())
            }
            Ok(Err(error)) => {
                self.quarantine();
                Err(error)
            }
            Err(payload) => {
                self.quarantine();
                resume_unwind(payload)
            }
        }
    }
}
impl<R, E> Drop for Pending<'_, R, E> {
    fn drop(&mut self) {
        if self.armed {
            let joined = self
                .completion
                .take()
                .or_else(|| self.fallback.take())
                .is_some_and(|join| matches!(catch_unwind(AssertUnwindSafe(join)), Ok(Ok(()))));
            if !joined {
                self.quarantine();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};
    struct Root(Rc<Cell<usize>>);
    impl Drop for Root {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    fn roots(count: usize, drops: &Rc<Cell<usize>>) -> Vec<Root> {
        (0..count).map(|_| Root(drops.clone())).collect()
    }

    #[test]
    fn actual_completion_joins_once_and_preserves_destination_owners() {
        let drops = Rc::new(Cell::new(0));
        let joins = Cell::new(0);
        let mut owned = roots(3, &drops);
        let mut pending = Pending::new(&mut owned, || -> Result<(), ()> { panic!("fallback") });
        pending.submitted(|| {
            joins.set(joins.get() + 1);
            Ok(())
        });
        pending.finish().unwrap();
        assert_eq!((joins.get(), owned.len(), drops.get()), (1, 3, 0));
        drop(owned);
        assert_eq!(drops.get(), 3);
    }

    #[test]
    fn next_cpu_read_failure_or_panic_still_joins_original_completion() {
        for panic in [false, true] {
            let drops = Rc::new(Cell::new(0));
            let joins = Cell::new(0);
            let mut owned = roots(2, &drops);
            let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), ()> {
                let mut pending =
                    Pending::new(&mut owned, || -> Result<(), ()> { panic!("fallback") });
                pending.submitted(|| {
                    joins.set(joins.get() + 1);
                    Ok(())
                });
                if panic {
                    panic!("fixed public next-read fault")
                } else {
                    return Err(());
                }
            }));
            assert_eq!(result.is_err(), panic);
            assert_eq!((joins.get(), owned.len(), drops.get()), (1, 2, 0));
        }
    }

    #[test]
    fn synchronous_setup_failure_uses_real_fallback_drain_before_release() {
        let drops = Rc::new(Cell::new(0));
        let joins = Cell::new(0);
        let mut owned = roots(1, &drops);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _pending = Pending::new(&mut owned, || -> Result<(), ()> {
                    joins.set(1);
                    Ok(())
                });
                panic!("fixed public scheduling fault");
            }))
            .is_err()
        );
        assert_eq!((joins.get(), owned.len(), drops.get()), (1, 1, 0));
    }

    #[test]
    fn failed_or_panicked_join_quarantines_every_original_root() {
        for mode in 0..4 {
            let drops = Rc::new(Cell::new(0));
            let mut owned = roots(3, &drops);
            let run = catch_unwind(AssertUnwindSafe(|| {
                let mut pending = Pending::new(&mut owned, || -> Result<(), ()> { Err(()) });
                if mode < 3 {
                    pending.submitted(|| {
                        if mode == 1 {
                            panic!("fixed public completion fault");
                        }
                        Err(())
                    });
                }
                if mode == 2 || mode == 3 {
                    drop(pending);
                } else {
                    assert!(pending.finish().is_err());
                }
            }));
            assert_eq!(run.is_err(), mode == 1);
            assert_eq!((owned.len(), drops.get()), (0, 0));
        }
    }
}
