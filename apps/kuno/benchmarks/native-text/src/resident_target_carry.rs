//! Borrow the actual completed verifier R owner for a joined MTP round.
//! A receipt is not a raw handle or erase authority. Its source is authenticated
//! again under an immutable parent-model borrow before any GPU command.
use super::*;
use std::cell::Cell;

pub(crate) const HOST_BYTES: usize = 8192;

pub(crate) struct DeviceCarryReceipt {
    source: Zeroizing<BufferId>,
    identity: CarryIdentity,
    predictions: Zeroizing<Vec<u32>>,
}
/// Pure state identity; generic binding is used only by CPU tests. Production
/// instantiation requires the original opaque graph binding and source BufferId.
struct CarryIdentity<B = GraphBindingId> {
    binding: B,
    generation: Zeroizing<u64>,
    geometry: Zeroizing<[usize; 2]>,
}
impl<B: PartialEq> CarryIdentity<B> {
    fn validate(
        &self,
        generation: u64,
        binding: B,
        position: usize,
        accepted: usize,
        predictions: usize,
    ) -> Result<()> {
        if *self.generation != generation
            || self.binding != binding
            || !(1..=FOUR_ROWS).contains(&self.geometry[1])
            || predictions != self.geometry[1]
            || !(1..=self.geometry[1]).contains(&accepted)
            || self.geometry[0].checked_add(accepted) != Some(position)
        {
            return Err(rejected!());
        }
        Ok(())
    }
}
pub(super) fn next_generation(old: u64) -> Result<u64> {
    old.checked_add(1).ok_or(Error::ArithmeticOverflow)
}
/// A queued source borrow can be released only after the consumer's real fence.
/// Unknown completion retires the ORIGINAL context; it never frees that source.
struct CopyCompletion<'a> {
    retired: &'a AtomicBool,
    in_flight: Cell<bool>,
}
impl Drop for CopyCompletion<'_> {
    fn drop(&mut self) {
        if self.in_flight.get() {
            self.retired.store(true, Ordering::Release);
        }
    }
}

pub(super) fn receipt(
    owner: &Owners,
    binding: GraphBindingId,
    source: BufferId,
    payload: &Payload,
) -> Result<Option<DeviceCarryReceipt>> {
    if !crate::strata_intel_core::enabled() {
        return Ok(None);
    }
    let predictions = payload.greedy.as_ref().ok_or_else(|| rejected!())?;
    if predictions.len() != payload.coordinates[1] || predictions.capacity() > FOUR_ROWS {
        return Err(rejected!());
    }
    Ok(Some(DeviceCarryReceipt {
        source: Zeroizing::new(source),
        identity: CarryIdentity {
            binding,
            generation: Zeroizing::new(owner.carry_generation.get()),
            geometry: Zeroizing::new(*payload.coordinates),
        },
        predictions: Zeroizing::new(predictions.to_vec()),
    }))
}

/// Immutable arena borrow prevents source reset/reuse while MTP reads it. All
/// copies borrow the complete source; only read intervals are sliced. No clone,
/// serialization, free, reset or source write operation is exposed.
pub(crate) struct DeviceCarry<'a> {
    arena: &'a PrivateArena,
    receipt: &'a DeviceCarryReceipt,
    accepted: usize,
    completion: CopyCompletion<'a>,
}
impl DeviceCarry<'_> {
    pub(crate) fn validate_predictions(&self, start: usize, tokens: &[u32]) -> Result<()> {
        validate_predictions(
            self.receipt.identity.geometry[0],
            self.accepted,
            &self.receipt.predictions,
            start,
            tokens,
        )
    }
    /// Copies a real R prefix or selected last row into an existing MTP owner.
    /// The caller retains this grant until its actual private output/fault fence.
    pub(crate) fn copy_rows(
        &self,
        destination: &PrivateArena,
        id: BufferId,
        fault: BufferId,
        first: usize,
        count: usize,
    ) -> Result<()> {
        let client = destination.client().map_err(u::fault)?;
        if !std::ptr::eq(client.info(), self.arena.client().map_err(u::fault)?.info())
            || self.completion.retired.load(Ordering::Acquire)
        {
            return Err(rejected!());
        }
        let words = copy_span(first, count, self.accepted)?;
        let source = gr::full_private(self.arena, *self.receipt.source, MAX_BATCH * HC_WIDTH)?;
        let layout = destination.layout(id).map_err(u::fault)?;
        let dest = gr::full_private(destination, id, layout.logical_words)?;
        let fault = gr::full_private(destination, fault, 1)?;
        if words.end > MAX_BATCH * HC_WIDTH || words.len() > layout.logical_words {
            return Err(rejected!());
        }
        gr::physical(self.arena, &[source.clone(), dest.clone(), fault])?;
        let read = source.offset_start((words.start * 4) as u64).offset_end(
            ((self
                .arena
                .layout(*self.receipt.source)
                .map_err(u::fault)?
                .owned_words
                - words.end)
                * 4) as u64,
        );
        let write = dest.offset_end(((layout.owned_words - words.len()) * 4) as u64);
        self.completion.in_flight.set(true); // Before enqueue; failure grants no clean reuse.
        u::copy_words::launch::<WgpuRuntime>(
            client,
            u::grid(words.len())?,
            CubeDim::new_1d(64),
            u::raw_arg(read, words.len())?,
            u::raw_arg(write, words.len())?,
        );
        Ok(())
    }
    /// # Safety
    /// The sole MTP consumer must have completed its real same-client output/
    /// sticky-fault fence after ALL copies. Enqueue alone is not completion.
    pub(crate) unsafe fn completed(&self) {
        self.completion.in_flight.set(false);
    }
}
impl Drop for DeviceCarry<'_> {
    fn drop(&mut self) {
        self.accepted.zeroize();
    }
}
fn validate_predictions(
    original_start: usize,
    accepted: usize,
    predictions: &[u32],
    start: usize,
    tokens: &[u32],
) -> Result<()> {
    if original_start != start
        || !(1..=predictions.len()).contains(&accepted)
        || tokens.len() != accepted
        || tokens != &predictions[..accepted]
    {
        return Err(rejected!());
    }
    Ok(())
}
fn copy_span(first: usize, count: usize, accepted: usize) -> Result<std::ops::Range<usize>> {
    if !(1..=FOUR_ROWS).contains(&accepted)
        || count == 0
        || first.checked_add(count).is_none_or(|end| end > accepted)
    {
        return Err(rejected!());
    }
    Ok(first * HC_WIDTH..(first + count) * HC_WIDTH)
}
impl<'m, 'f> ResidentModel<'m, 'f> {
    pub(crate) fn borrow_target_carry<'a>(
        &'a self,
        executor: &GpuExecutor<'f>,
        receipt: &'a DeviceCarryReceipt,
        accepted: usize,
    ) -> Result<DeviceCarry<'a>> {
        if !crate::strata_intel_core::enabled() {
            return Err(rejected!());
        }
        self.completed_native_projection_boundary(executor)?;
        let native = self.native_batch.as_ref().ok_or_else(|| rejected!())?;
        let owner = native.target_rows.as_ref().ok_or_else(|| rejected!())?;
        owner.check_no_live_output()?;
        receipt.identity.validate(
            owner.carry_generation.get(),
            self.graph.binding_id(),
            self.position,
            accepted,
            receipt.predictions.len(),
        )?;
        // BufferId authenticates the actual arena identity/epoch/slot, even for
        // two model instances sharing the same public graph and cursor.
        let source = gr::full_private(&self.arena, *receipt.source, MAX_BATCH * HC_WIDTH)?;
        if !native.outer.residual.contains(&*receipt.source) {
            return Err(rejected!());
        }
        gr::physical(&self.arena, &[source])?;
        Ok(DeviceCarry {
            arena: &self.arena,
            receipt,
            accepted,
            completion: CopyCompletion {
                retired: &self.retired,
                in_flight: Cell::new(false),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_device_carry_identity_authenticates_every_accepted_prefix_and_epoch() {
        for n in 1..=FOUR_ROWS {
            let id = CarryIdentity {
                binding: 91u64,
                generation: Zeroizing::new(7),
                geometry: Zeroizing::new([20, n]),
            };
            for accepted in 1..=n {
                id.validate(7, 91, 20 + accepted, accepted, n).unwrap();
                assert!(id.validate(6, 91, 20 + accepted, accepted, n).is_err());
                assert!(id.validate(7, 90, 20 + accepted, accepted, n).is_err());
                assert!(id.validate(7, 91, 21 + accepted, accepted, n).is_err());
                assert!(id.validate(7, 91, 20 + accepted, accepted, n + 1).is_err());
            }
            for bad in [0, n + 1, usize::MAX] {
                assert!(id.validate(7, 91, 20, bad, n).is_err());
            }
        }
        assert_eq!(next_generation(9).unwrap(), 10);
        assert!(next_generation(u64::MAX).is_err());
    }
    #[test]
    fn native_device_carry_predictions_cannot_substitute_rejected_or_changed_tokens() {
        for n in 1..=FOUR_ROWS {
            let predicted: Vec<_> = (10..10 + n as u32).collect();
            for accepted in 1..=n {
                validate_predictions(20, accepted, &predicted, 20, &predicted[..accepted]).unwrap();
                let mut changed = predicted[..accepted].to_vec();
                changed[accepted - 1] += 1;
                assert!(validate_predictions(20, accepted, &predicted, 20, &changed).is_err());
                assert!(
                    validate_predictions(20, accepted, &predicted, 21, &predicted[..accepted])
                        .is_err()
                );
                assert!(validate_predictions(20, 0, &predicted, 20, &[]).is_err());
                assert!(validate_predictions(20, n + 1, &predicted, 20, &predicted).is_err());
            }
        }
    }
    #[test]
    fn native_device_carry_views_cover_only_prefix_and_selected_row_of_full_owner() {
        let full = BufferLayout::new(MAX_BATCH * HC_WIDTH).unwrap();
        for accepted in 1..=FOUR_ROWS {
            for first in 0..accepted {
                for count in 1..=accepted - first {
                    let r = copy_span(first, count, accepted).unwrap();
                    assert_eq!(r.start, first * HC_WIDTH);
                    assert_eq!(r.len(), count * HC_WIDTH);
                    assert!(r.end <= full.logical_words);
                    assert_eq!(
                        r.start * 4 + r.len() * 4 + (full.owned_words - r.end) * 4,
                        full.owned_bytes
                    );
                }
            }
            assert!(copy_span(accepted, 1, accepted).is_err());
            assert!(copy_span(0, accepted + 1, accepted).is_err());
            assert!(copy_span(usize::MAX, 1, accepted).is_err());
            assert!(copy_span(0, 0, accepted).is_err());
        }
    }
    #[test]
    fn native_device_carry_unknown_completion_retires_without_resetting_prior_poison() {
        let retired = AtomicBool::new(false);
        drop(CopyCompletion {
            retired: &retired,
            in_flight: Cell::new(false),
        });
        assert!(!retired.load(Ordering::Acquire));
        {
            let state = CopyCompletion {
                retired: &retired,
                in_flight: Cell::new(false),
            };
            state.in_flight.set(true);
            state.in_flight.set(true);
            state.in_flight.set(false); // synthetic completed fence, not enqueue
        }
        assert!(!retired.load(Ordering::Acquire));
        drop(CopyCompletion {
            retired: &retired,
            in_flight: Cell::new(true),
        });
        assert!(retired.load(Ordering::Acquire));
        drop(CopyCompletion {
            retired: &retired,
            in_flight: Cell::new(false),
        });
        assert!(retired.load(Ordering::Acquire));
        assert!(
            std::mem::size_of::<DeviceCarryReceipt>()
                + std::mem::size_of::<DeviceCarry<'_>>()
                + FOUR_ROWS * 4
                + 3 * std::mem::size_of::<Handle>()
                < HOST_BYTES
        );
    }
}
