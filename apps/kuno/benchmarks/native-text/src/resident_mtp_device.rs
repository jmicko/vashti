//! Native first/count1 front and request exit authority from original
//! verifier device rows. A front completion does not authorize a draft follow.
use super::window::WindowStart;
use super::*;
use crate::resident_model::batch::target_rows::DeviceCarry;

impl ResidentMtp<'_, '_> {
    pub(crate) fn append_device_front(
        &mut self,
        start: usize,
        tokens: &[u32],
        source: &DeviceCarry<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let input = WindowStart::Device {
            start,
            tokens,
            source,
        };
        let cell = self.window_start_position(&input, 1)?;
        self.last_forward.take();
        self.last_front.take();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.start_window(&input)?;
            self.queue_native_front(
                tokens.len(),
                Some(NativeFrontInputs::Device(tokens, source)),
                start,
                cancel,
            )?;
            self.guard_private(cell + 1)?;
            self.verify_source()?;
            u::cancel(cancel)?;
            self.complete_fault()?;
            self.position = cell + 1;
            self.last_front = Some(LastFront(LastForward {
                binding: self.binding,
                start: cell,
            }));
            self.poisoned = false;
            Ok(())
        }));
        self.accept_device_cache(result, source)
    }
    /// Native generate.cpp exits after target commit at EOS/cap; it does not
    /// run another MTP catch-up. Original cache ownership/erasure remains with
    /// close(), but no provisional forward/front can authorize a later follow.
    pub(crate) fn finish_native_request(&mut self) -> Result<()> {
        if !self.native_confidence_supported()
            || self.closed
            || self.poisoned
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
        {
            return Err(Error::InvalidFormat("native MTP request exit scope"));
        }
        self.last_forward.take();
        self.last_front.take();
        Ok(())
    }
    fn accept_device_cache(
        &mut self,
        result: std::thread::Result<Result<()>>,
        source: &DeviceCarry<'_>,
    ) -> Result<()> {
        match result {
            Ok(Ok(())) => {
                // SAFETY: each producer has joined its actual full fault packet
                // and completed ledger fence after every original-source copy.
                unsafe {
                    source.completed();
                }
                Ok(())
            }
            failure => {
                self.fail();
                Err(match failure {
                    Ok(Err(e)) => e,
                    _ => Error::InvalidFormat("native MTP device cache failure"),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_device_end_first_prompt_cell_is_distinct_from_accepted_catchup() {
        use super::super::window::device_front_position;
        if !crate::strata_intel_core::enabled() {
            assert!(device_front_position(0, 0, 1, false).is_err());
            return;
        }
        for start in [0, 1, 34, 4095] {
            assert_eq!(
                device_front_position(start, start, 1, false).unwrap(),
                start
            );
            assert!(device_front_position(start, start, 2, false).is_err());
            assert!(device_front_position(start + 1, start, 1, false).is_err());
        }
        assert!(device_front_position(4096, 4096, 1, false).is_err());
    }
    #[test]
    fn native_device_end_front_stamp_does_not_publish_full_forward_authority() {
        let front = Some(LastFront(LastForward {
            binding: 99,
            start: 4,
        }));
        let full: Option<LastForward> = None;
        assert!(front.as_ref().unwrap().valid(99, 5));
        assert!(!front.as_ref().unwrap().valid(98, 5));
        assert!(!front.as_ref().unwrap().valid(99, 4));
        assert!(full.is_none()); // follow_draft only consumes this distinct type.
    }
}
