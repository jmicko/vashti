//! Native first/count1 front and request exit authority from original
//! verifier device rows. A front completion does not authorize a draft follow.
use super::window::WindowStart;
use super::*;
use crate::resident_model::batch::target_rows::{DeviceCarry, KnownInputCarry};

pub(crate) struct NativeRequestResetReceipt {
    pub(crate) private: crate::private_arena::OwnedZeroAudit,
    binding: u64,
    fault: BufferId,
}
impl NativeRequestResetReceipt {
    pub(crate) fn check_original(&self, model: &ResidentMtp<'_, '_>) -> Result<()> {
        if model.binding != self.binding
            || model.id(B::Fault) != self.fault
            || model.position != 0
            || model.closed
            || model.poisoned
            || model.retired.load(Ordering::Acquire)
        {
            return Err(Error::InvalidFormat(
                "native MTP retained reset original owner",
            ));
        }
        model.verify_source()
    }
}
impl ResidentMtp<'_, '_> {
    /// Whole known-input teacher forcing, with original residuals kept on the
    /// device. It publishes only KV progress, never a prediction/forward stamp.
    /// The caller holds the actual target MAIN->draft visibility scope and R
    /// borrow until this consumer's real complete fault/ledger observation.
    pub(crate) fn append_known_device_group(
        &mut self,
        start: usize,
        tokens: &[u32],
        source: &KnownInputCarry<'_>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || self.graph.profile() != MtpProfile::StrataQ2
            || self.poisoned
            || self.closed
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
            || !(1..=crate::strata_port::verify_capacity()).contains(&tokens.len())
            || self.position != start
            || start
                .checked_add(tokens.len())
                .is_none_or(|end| end > k::CONTEXT)
            || tokens.iter().any(|&token| token as usize >= k::VOCAB)
        {
            return Err(Error::InvalidFormat("known MTP device teacher scope"));
        }
        source.validate_teacher(start, tokens)?;
        u::cancel(cancel)?;
        self.verify_source()?;
        self.physical()?;
        self.last_forward.take();
        self.last_front.take();
        self.poisoned = true;
        self.dispatches = 0;
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.clear_scratch()?;
            self.queue_native_front(
                tokens.len(),
                Some(NativeFrontInputs::Known(tokens, source)),
                start,
                cancel,
            )?;
            self.guard_private(start + tokens.len())?;
            self.complete_fault()?;
            self.verify_source()?;
            u::cancel(cancel)
        }));
        match attempted {
            Ok(Ok(())) => {
                // SAFETY: the original same-client full fault read and ledger
                // fence above complete ALL queued source copies/front consumers.
                unsafe {
                    source.completed();
                }
                self.position = start + tokens.len();
                self.poisoned = false;
                Ok(())
            }
            failed => {
                self.fail();
                Err(match failed {
                    Ok(Err(error)) => error,
                    _ => Error::InvalidFormat("known MTP device teacher failed"),
                })
            }
        }
    }
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
    /// Healthy completed-request reset, preserving original public weights,
    /// graph identities and actual private owners. No failed context recovery.
    pub(crate) fn reset_completed_native_request(
        &mut self,
        cancel: &AtomicBool,
    ) -> Result<NativeRequestResetReceipt> {
        u::cancel(cancel)?;
        self.finish_native_request()
            .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
        if self.ledger.shared().is_none() || self.native_front_pending.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("native MTP retained reset scope"));
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || -> Result<crate::private_arena::OwnedZeroAudit> {
                self.physical()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                self.verify_source()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                // Observe the original fault before rewind/erasure can clear it.
                self.complete_fault()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                self.rewind(0)
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                self.poisoned = true;
                let private = self
                    .arena
                    .erase_retained_public_owners()
                    .map_err(error)
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                let (buffers, bytes) = private_owned_geometry()?;
                if private.bytes != bytes || private.buffers != buffers {
                    return Err(Error::InvalidFormat("native MTP retained reset inventory"));
                }
                self.verify_source()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                self.complete_fault()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                self.audit_io()
                    .map_err(|error| crate::strata_engine::report_private_reset_error(error))?;
                u::cancel(cancel)?;
                self.position = 0;
                self.poisoned = false;
                Ok(private)
            },
        ));
        match result {
            Ok(Ok(private)) => Ok(NativeRequestResetReceipt {
                private,
                binding: self.binding,
                fault: self.id(B::Fault),
            }),
            other => {
                self.poisoned = true;
                self.retired.store(true, Ordering::Release);
                self.ledger.poison();
                match other {
                    Ok(Err(e)) => Err(e),
                    _ => Err(Error::InvalidFormat("native MTP retained reset failed")),
                }
            }
        }
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
    fn retained_owner_scan_has_a_separate_host_quote_from_normal_mtp_packets() {
        let largest = WORDS
            .into_iter()
            .map(|n| BufferLayout::new(n).unwrap().owned_bytes)
            .max()
            .unwrap();
        let window = largest.min(crate::private_arena::ZERO_AUDIT_WINDOW);
        if crate::strata_intel_core::enabled() {
            // The actual32K cache root exceeds the normal packet allowance.
            // Its real inspection loop maps at most one bounded window.
            assert!(window > HOST_BYTES);
            assert!(2 * window <= crate::private_arena::OWNED_ZERO_AUDIT_HOST_BYTES);
        } else {
            assert!(window <= HOST_BYTES);
        }
    }
    #[test]
    fn native_device_end_first_prompt_cell_is_distinct_from_accepted_catchup() {
        use super::super::window::device_front_position;
        if !crate::strata_intel_core::enabled() {
            assert!(device_front_position(0, 0, 1, false).is_err());
            return;
        }
        for start in [0, 1, 34, 4095, k::CONTEXT - 1] {
            assert_eq!(
                device_front_position(start, start, 1, false).unwrap(),
                start
            );
            assert!(device_front_position(start, start, 2, false).is_err());
            assert!(device_front_position(start + 1, start, 1, false).is_err());
        }
        assert!(device_front_position(k::CONTEXT, k::CONTEXT, 1, false).is_err());
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
