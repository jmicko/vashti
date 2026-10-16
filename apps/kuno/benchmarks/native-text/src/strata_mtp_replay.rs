//! Original capture-derived host accounting, not output/completion authority.
//! Cached fields contain only immutable shapes and content-free operation counts.
use super::*;
use cubecl_runtime::server::NativeStrataBranch;

const CAPACITY: usize = 128; // (8*7 rounds +6 steps)*2 sampler inventories.
const METADATA_CAP: usize = 32 * 1024;
#[derive(Clone, Copy)]
pub(super) struct Entry {
    key: NativeStrataBranch,
    cells: usize,
    dispatches: u64,
    copies: u64,
    first_cost: usize,
    final_dispatches: usize,
}
pub(super) struct Capture {
    key: NativeStrataBranch,
    before: Traffic,
    entry_dispatches: usize,
    first_cost: Option<usize>,
}
pub(super) struct Proposed {
    pub dispatches: usize,
    pub traffic: Traffic,
    pub advance: usize,
}
impl Capture {
    pub(super) fn begin(
        key: NativeStrataBranch,
        before: Traffic,
        dispatches: usize,
    ) -> Result<Self> {
        if key.mtp_geometry().is_none() || dispatches > MAX_TOKEN_DISPATCHES {
            return Err(reject());
        }
        Ok(Self {
            key,
            before,
            entry_dispatches: dispatches,
            first_cost: None,
        })
    }
    pub(super) fn first_reset(&mut self, dispatches: usize) -> Result<()> {
        if self.first_cost.is_some()
            || self.key.mtp_geometry().is_none_or(|g| g.2 <= 1)
            || dispatches > MAX_TOKEN_DISPATCHES
        {
            return Err(reject());
        }
        self.first_cost = Some(
            dispatches
                .checked_sub(self.entry_dispatches)
                .ok_or_else(reject)?,
        );
        Ok(())
    }
    pub(super) fn finish(self, after: Traffic, dispatches: usize) -> Result<Entry> {
        let (_, _, cells) = self.key.mtp_geometry().ok_or_else(reject)?;
        let cells = cells as usize;
        let explicit = after
            .explicit_dispatches
            .checked_sub(self.before.explicit_dispatches)
            .ok_or_else(reject)?;
        let copies = after
            .followup_copy_calls
            .checked_sub(self.before.followup_copy_calls)
            .ok_or_else(reject)?;
        let mut expected = self.before;
        expected.explicit_dispatches = after.explicit_dispatches;
        expected.followup_copy_calls = after.followup_copy_calls;
        // A replayable math region contains no actor host reads/uploads,
        // observations or repair publications. Those stay outside recording.
        if after != expected
            || explicit == 0
            || copies != (cells - 1) as u64
            || dispatches > MAX_TOKEN_DISPATCHES
        {
            return Err(reject());
        }
        let first_cost = if cells == 1 {
            if self.first_cost.is_some() {
                return Err(reject());
            }
            let cost = usize::try_from(explicit).map_err(u::fault)?;
            if self.entry_dispatches.checked_add(cost) != Some(dispatches) {
                return Err(reject());
            }
            cost
        } else {
            self.first_cost.ok_or_else(reject)?
        };
        let entry = Entry {
            key: self.key,
            cells,
            dispatches: explicit,
            copies,
            first_cost,
            final_dispatches: dispatches,
        };
        entry.propose(self.before, self.entry_dispatches)?;
        Ok(entry)
    }
}
impl Entry {
    pub(super) fn cells(&self) -> usize {
        self.cells
    }
    pub(super) fn propose(&self, mut before: Traffic, dispatches: usize) -> Result<Proposed> {
        if dispatches
            .checked_add(self.first_cost)
            .is_none_or(|n| n > MAX_TOKEN_DISPATCHES)
        {
            return Err(reject());
        }
        let after = if self.cells == 1 {
            dispatches
                .checked_add(usize::try_from(self.dispatches).map_err(u::fault)?)
                .ok_or_else(reject)?
        } else {
            self.final_dispatches
        };
        if after > MAX_TOKEN_DISPATCHES {
            return Err(reject());
        }
        before.explicit_dispatches = before
            .explicit_dispatches
            .checked_add(self.dispatches)
            .ok_or(Error::ArithmeticOverflow)?;
        before.followup_copy_calls = before
            .followup_copy_calls
            .checked_add(self.copies)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(Proposed {
            dispatches: after,
            traffic: before,
            advance: self.cells - 1,
        })
    }
}
pub(super) struct Cache(Box<[Option<Entry>; CAPACITY]>);
impl Cache {
    pub(super) fn new() -> Result<Self> {
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(CAPACITY)
            .map_err(|_| Error::LimitExceeded("MTP replay metadata allocation"))?;
        if entries.capacity() * size_of::<Option<Entry>>() > METADATA_CAP / 2 {
            return Err(reject());
        }
        entries.resize_with(CAPACITY, || None);
        Ok(Self(
            entries
                .into_boxed_slice()
                .try_into()
                .map_err(|_| reject())?,
        ))
    }
    pub(super) fn get(&self, key: NativeStrataBranch) -> Option<Entry> {
        self.0.iter().flatten().find(|e| e.key == key).copied()
    }
    pub(super) fn insert(&mut self, entry: Entry) -> Result<()> {
        let slot = self
            .0
            .iter_mut()
            .find(|e| e.is_some_and(|e| e.key == entry.key));
        if let Some(slot) = slot {
            *slot = Some(entry);
            return Ok(());
        }
        let slot = self.0.iter_mut().find(|e| e.is_none()).ok_or_else(reject)?;
        *slot = Some(entry);
        Ok(())
    }
}
const _: () = {
    assert!(2 * size_of::<[Option<Entry>; CAPACITY]>() <= METADATA_CAP);
};
impl ResidentMtp<'_, '_> {
    /// Only this private owner can hand completed native work back to its CPU
    /// bookkeeping. A miss mutates no proposed counters/sampler state; errors
    /// retain the original request's terminal cleanup semantics.
    pub(super) fn try_native_frame_replay(
        &mut self,
        key: NativeStrataBranch,
        origin: usize,
        first: &mut Option<PreparedNativeFront>,
        cancel: &AtomicBool,
    ) -> Result<bool> {
        let Some(entry) = self.native_replay.get(key) else {
            return Ok(false);
        };
        let (rows, index, cells) = key.mtp_geometry().ok_or_else(reject)?;
        let expected = origin
            .checked_add(usize::from(rows) - 1)
            .ok_or_else(reject)?;
        if !self.native_confidence_supported()
            || !self.fused_sweeps
            || self.closed
            || self.retired.load(Ordering::Acquire)
            || self.position != expected
            || entry.cells() != cells as usize
        {
            return Err(reject());
        }
        if index == 0 {
            let ids = &*self.native_front.as_ref().ok_or_else(reject)?.ids;
            first.as_ref().ok_or_else(reject)?.check(
                self.binding,
                ids,
                self.position,
                &self.native_front_pending,
            )?;
        } else if first.is_some() || self.native_front_pending.load(Ordering::Acquire) {
            return Err(reject());
        }
        self.verify_source()?;
        self.physical()?;
        u::cancel(cancel)?;
        let next = entry.propose(self.traffic.get(), self.dispatches)?;
        let position = self
            .position
            .checked_add(next.advance)
            .ok_or(Error::ArithmeticOverflow)?;
        if position >= k::CONTEXT {
            return Err(reject());
        }
        let sampler = crate::strata_sampler::request::prepare_draft_replay(
            self.client()?,
            self.binding,
            self.position,
            entry.cells(),
            index as usize,
        )?;
        if key.with_coupled_sampling(sampler.coupled()) != key {
            return Err(reject());
        }
        let invocation = cubecl_runtime::server::NativeStrataInvocation::mtp(
            u32::try_from(origin).map_err(error)?,
        )
        .map_err(error)?;
        if !self
            .client()?
            .replay_native_strata_branch(key, invocation)
            .map_err(error)?
        {
            return Ok(false);
        }
        u::cancel(cancel)?;
        // GPU completion alone is not numerical acceptance. The caller still
        // performs the actual packet/fault/padding/source/ledger observation.
        if index == 0 {
            let prepared = first.take().ok_or_else(reject)?;
            let ids = &*self.native_front.as_ref().ok_or_else(reject)?.ids;
            prepared.check(self.binding, ids, self.position, &self.native_front_pending)?;
            prepared.queued()?;
        }
        sampler.complete(self.client()?)?;
        self.position = position;
        self.dispatches = next.dispatches;
        self.traffic.set(next.traffic);
        Ok(true)
    }
}
fn reject() -> Error {
    Error::InvalidFormat("MTP original capture accounting mismatch")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(cells: u8, owner: u64, coupled: bool) -> NativeStrataBranch {
        NativeStrataBranch::mtp_round_for_context(
            2,
            cells,
            crate::mtp_resident_kernels::CONTEXT as u16,
        )
        .unwrap()
        .with_model_owner(owner)
        .unwrap()
        .with_coupled_sampling(coupled)
    }
    #[test]
    fn single_frame_reuses_real_math_counts_with_a_different_entry_budget() {
        let before = Traffic {
            explicit_dispatches: 100,
            host_read_calls: 7,
            ..Traffic::default()
        };
        let mut after = before;
        after.explicit_dispatches += 73;
        let entry = Capture::begin(key(1, 91, false), before, 10)
            .unwrap()
            .finish(after, 83)
            .unwrap();
        let mut current = before;
        current.host_read_calls = 999;
        let next = entry.propose(current, 20).unwrap();
        assert_eq!(next.dispatches, 93);
        assert_eq!(next.advance, 0);
        assert_eq!(next.traffic.explicit_dispatches, 173);
        assert_eq!(next.traffic.host_read_calls, 999);
        assert!(entry.propose(current, MAX_TOKEN_DISPATCHES - 73).is_ok());
        assert!(entry.propose(current, MAX_TOKEN_DISPATCHES - 72).is_err());
        current.explicit_dispatches = u64::MAX;
        assert!(entry.propose(current, 0).is_err());
    }
    #[test]
    fn whole_chain_preserves_the_original_first_budget_resets_and_final_counter() {
        let before = Traffic {
            explicit_dispatches: 100,
            ..Traffic::default()
        };
        let mut capture = Capture::begin(key(3, 91, true), before, 20).unwrap();
        capture.first_reset(100).unwrap();
        assert!(capture.first_reset(100).is_err());
        let mut after = before;
        after.explicit_dispatches += 300;
        after.followup_copy_calls += 2;
        let entry = capture.finish(after, 120).unwrap();
        let next = entry.propose(before, 15).unwrap();
        assert_eq!(next.dispatches, 120);
        assert_eq!(next.advance, 2);
        assert_eq!(next.traffic.explicit_dispatches, 400);
        assert_eq!(next.traffic.followup_copy_calls, 2);
        assert!(entry.propose(before, MAX_TOKEN_DISPATCHES - 80).is_ok());
        assert!(entry.propose(before, MAX_TOKEN_DISPATCHES - 79).is_err());
        assert!(
            Capture::begin(key(3, 91, true), before, 20)
                .unwrap()
                .finish(after, 120)
                .is_err()
        );
    }
    #[test]
    fn accounting_rejects_host_observations_uploads_repairs_and_missing_copies() {
        let base = Traffic::default();
        for mutate in [
            |t: &mut Traffic| t.host_read_calls += 1,
            |t: &mut Traffic| t.private_upload_bytes += 1,
            |t: &mut Traffic| t.draft_window_reads += 1,
            |t: &mut Traffic| t.batched_repair_calls += 1,
        ] {
            let mut after = base;
            after.explicit_dispatches = 64;
            mutate(&mut after);
            assert!(
                Capture::begin(key(1, 91, false), base, 0)
                    .unwrap()
                    .finish(after, 64)
                    .is_err()
            );
        }
        let mut capture = Capture::begin(key(2, 91, false), base, 0).unwrap();
        capture.first_reset(64).unwrap();
        let after = Traffic {
            explicit_dispatches: 128,
            ..base
        };
        assert!(capture.finish(after, 64).is_err());
    }
    #[test]
    fn bounded_cache_separates_models_and_coupled_graphs_without_retaining_positions() {
        let mut cache = Cache::new().unwrap();
        for rows in 1..=8 {
            for coupled in [false, true] {
                for cells in 1..=7 {
                    let k = NativeStrataBranch::mtp_round_for_context(
                        rows,
                        cells,
                        crate::mtp_resident_kernels::CONTEXT as u16,
                    )
                    .unwrap()
                    .with_model_owner(91)
                    .unwrap()
                    .with_coupled_sampling(coupled);
                    let mut c = Capture::begin(k, Traffic::default(), 0).unwrap();
                    if cells > 1 {
                        c.first_reset(64).unwrap();
                    }
                    cache
                        .insert(
                            c.finish(
                                Traffic {
                                    explicit_dispatches: u64::from(cells) * 64,
                                    followup_copy_calls: u64::from(cells - 1),
                                    ..Traffic::default()
                                },
                                64,
                            )
                            .unwrap(),
                        )
                        .unwrap();
                }
            }
        }
        assert!(cache.get(key(1, 91, false)).is_some());
        assert!(cache.get(key(1, 91, true)).is_some());
        assert!(cache.get(key(1, 92, false)).is_none());
        assert!(size_of::<Cache>() <= size_of::<usize>() * 2);
    }
}
