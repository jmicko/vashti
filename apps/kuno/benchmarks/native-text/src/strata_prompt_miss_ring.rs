//! Enqueue-only scheduling for the original eight raw transfer slots. A
//! resident group never consumes a slot. Consumed means its ORIGINAL kernels
//! were enqueued, NOT that GPU use completed: original Graph range/timeline
//! dependencies still gate every physical overwrite.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
const GROUPS: usize = 512;
const SLOTS: usize = 8;
const RESIDENT: usize = usize::MAX;
// Checked inside the existing complete batch's 24MiB parent envelope, not
// additional GPU memory or a larger original8MiB command-tape allowance.
pub(super) const HOST_BYTES: usize = 8192;
fn reject() -> Error {
    Error::InvalidFormat("native miss-only prefill ring")
}

pub(super) struct Ring {
    slots: Zeroizing<Vec<usize>>,
    issued: usize,
    consumed: usize,
    live: usize,
    peak: usize,
    misses: usize,
    batched: bool,
    refill: usize,
    cohorts: usize,
}
impl Ring {
    pub(super) fn new(streamed: &[bool]) -> Result<Self> {
        Self::new_policy(streamed, cfg!(feature = "strata-copy-cohorts"))
    }
    fn new_policy(streamed: &[bool], batched: bool) -> Result<Self> {
        if streamed.is_empty() || streamed.len() > GROUPS {
            return Err(reject());
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(streamed.len())
            .map_err(|_| reject())?;
        // Include the simultaneously live exact-size input classification Vec,
        // both Vec headers and fixed stack/struct scratch conservatively.
        if slots
            .capacity()
            .checked_mul(size_of::<usize>())
            .and_then(|n| n.checked_add(streamed.len()))
            .and_then(|n| n.checked_add(2 * size_of::<Vec<usize>>() + 1024))
            .is_none_or(|n| n > HOST_BYTES)
        {
            return Err(reject());
        }
        let mut misses = 0;
        for &host in streamed {
            slots.push(if host {
                let slot = misses % SLOTS;
                misses += 1;
                slot
            } else {
                RESIDENT
            });
        }
        Ok(Self {
            slots: Zeroizing::new(slots),
            issued: 0,
            consumed: 0,
            live: 0,
            peak: 0,
            misses,
            batched,
            refill: SLOTS,
            cohorts: usize::from(misses > 0),
        })
    }
    pub(super) fn issue_next(&mut self) -> Result<Option<(usize, Option<usize>)>> {
        if self.issued == self.slots.len() {
            return Ok(None);
        }
        let raw = self.slots[self.issued];
        if raw != RESIDENT && self.live == SLOTS {
            return Ok(None);
        }
        if self.live > SLOTS || self.consumed > self.issued {
            return Err(reject());
        }
        if raw != RESIDENT {
            if self.batched {
                if self.refill == 0 {
                    if self.live > SLOTS / 2 {
                        return Ok(None);
                    }
                    self.refill = SLOTS - self.live;
                    self.cohorts += 1;
                }
                self.refill -= 1;
            }
            self.live += 1;
            self.peak = self.peak.max(self.live);
        }
        let group = self.issued;
        self.issued += 1;
        Ok(Some((group, (raw != RESIDENT).then_some(raw))))
    }
    /// Checked non-consuming view of already issued contiguous cohort entries.
    /// Only consumed() advances retirement; physical hazards remain separate.
    #[cfg(feature = "strata-expert-cohorts")]
    pub(super) fn peek_slot(&self, group: usize) -> Result<Option<Option<usize>>> {
        if group < self.consumed || group >= self.slots.len() { return Err(reject()); }
        if group >= self.issued { return Ok(None); }
        let slot = self.slots[group];
        Ok(Some((slot != RESIDENT).then_some(slot)))
    }
    pub(super) fn ready_slot(&self, group: usize) -> Result<Option<usize>> {
        if group != self.consumed || group >= self.issued {
            return Err(reject());
        }
        let slot = self.slots[group];
        Ok((slot != RESIDENT).then_some(slot))
    }
    pub(super) fn consumed(&mut self, group: usize) -> Result<()> {
        if self.ready_slot(group)?.is_some() {
            self.live = self.live.checked_sub(1).ok_or_else(reject)?;
        }
        self.consumed += 1;
        Ok(())
    }
    pub(super) fn finish(&self) -> Result<()> {
        if self.consumed != self.slots.len() || self.issued != self.slots.len() || self.live != 0 {
            return Err(reject());
        }
        crate::strata_prefill_profile::count("expert_streamed_groups", self.misses as u64);
        crate::strata_prefill_profile::count(
            "expert_resident_groups",
            (self.slots.len() - self.misses) as u64,
        );
        crate::strata_prefill_profile::count("expert_raw_slots_peak_sum", self.peak as u64);
        if self.batched {
            crate::strata_prefill_profile::count("expert_raw_copy_cohorts", self.cohorts as u64);
        }
        Ok(())
    }
    fn erase(&mut self) {
        // Includes the full allocation capacity, not only populated slots.
        // Routing-derived metadata never survives a normal/error/panic drop.
        self.slots.zeroize();
        self.issued.zeroize();
        self.consumed.zeroize();
        self.live.zeroize();
        self.peak.zeroize();
        self.misses.zeroize();
        self.batched.zeroize();
        self.refill.zeroize();
        self.cohorts.zeroize();
    }
}
impl Drop for Ring {
    fn drop(&mut self) {
        self.erase();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_owned_slot_capacity_and_scalar_state_are_erased_before_deallocation() {
        let mut r = Ring::new_policy(&[true, false, true, false, true], false).unwrap();
        while r.issue_next().unwrap().is_some() {}
        let capacity = r.slots.capacity();
        assert!(r.slots.iter().any(|x| *x != 0));
        r.erase();
        assert!(r.slots.is_empty());
        assert_eq!(r.slots.capacity(), capacity);
        // zeroize1.9 Vec explicitly clears every spare-capacity byte as
        // MaybeUninit. The allocation remains genuinely owned and alive here.
        let bytes = unsafe {
            std::slice::from_raw_parts(r.slots.as_ptr().cast::<u8>(), capacity * size_of::<usize>())
        };
        assert!(bytes.iter().all(|b| *b == 0));
        assert_eq!([r.issued, r.consumed, r.live, r.peak, r.misses], [0; 5]);
    }
    #[test]
    fn all_mixed_sequences_keep_eight_actual_misses_ahead_and_never_reuse_a_live_slot() {
        for len in [1, 7, 8, 9, 31, 64, 512] {
            for pattern in 0..12 {
                let mode = (0..len)
                    .map(|i| match pattern {
                        0 => false,
                        1 => true,
                        _ => (i * 17 + pattern * 23) % pattern == 0,
                    })
                    .collect::<Vec<_>>();
                let mut r = Ring::new_policy(&mode, false).unwrap();
                let mut occupants = [None; 8];
                let mut rank = 0;
                let mut issued = 0;
                for group in 0..len {
                    while let Some((next, slot)) = r.issue_next().unwrap() {
                        assert_eq!(next, issued);
                        issued += 1;
                        assert_eq!(slot.is_some(), mode[next]);
                        if let Some(slot) = slot {
                            assert_eq!(slot, rank % 8);
                            rank += 1;
                            assert!(occupants[slot].is_none());
                            occupants[slot] = Some(next);
                        }
                    }
                    // Available staging is always full unless all remaining
                    // groups have already been examined (including residents).
                    assert!(issued == len || r.live == 8);
                    let slot = r.ready_slot(group).unwrap();
                    assert_eq!(slot.is_some(), mode[group]);
                    if let Some(slot) = slot {
                        assert_eq!(occupants[slot], Some(group));
                        occupants[slot] = None;
                    }
                    r.consumed(group).unwrap();
                }
                r.finish().unwrap();
                assert!(occupants.iter().all(Option::is_none));
                assert_eq!(r.misses, mode.iter().filter(|x| **x).count());
                assert_eq!(r.peak, 8.min(r.misses));
            }
        }
    }
    #[test]
    fn four_miss_cohorts_keep_the_same_eight_slots_routes_and_no_live_overwrite() {
        for len in [1, 7, 8, 9, 31, 64, 512] {
            for pattern in 0..12 {
                let mode = (0..len)
                    .map(|i| match pattern {
                        0 => false,
                        1 => true,
                        _ => (i * 17 + pattern * 23) % pattern == 0,
                    })
                    .collect::<Vec<_>>();
                let mut ring = Ring::new_policy(&mode, true).unwrap();
                let mut live = [None; SLOTS];
                let mut rank = 0;
                let mut next = 0;
                for group in 0..len {
                    while let Some((issued, slot)) = ring.issue_next().unwrap() {
                        assert_eq!(issued, next);
                        next += 1;
                        assert_eq!(slot.is_some(), mode[issued]);
                        if let Some(slot) = slot {
                            assert_eq!(slot, rank % SLOTS);
                            rank += 1;
                            assert!(live[slot].is_none());
                            live[slot] = Some(issued);
                        }
                    }
                    assert!(next == len || ring.live > SLOTS / 2);
                    if let Some(slot) = ring.ready_slot(group).unwrap() {
                        assert_eq!(live[slot], Some(group));
                        live[slot] = None;
                    }
                    ring.consumed(group).unwrap();
                }
                ring.finish().unwrap();
                assert!(live.iter().all(Option::is_none));
                let misses = mode.iter().filter(|&&v| v).count();
                assert_eq!(
                    ring.cohorts,
                    usize::from(misses > 0) + misses.saturating_sub(SLOTS).div_ceil(SLOTS / 2)
                );
                assert_eq!(ring.peak, misses.min(SLOTS));
                ring.erase();
                assert!(!ring.batched);
                assert_eq!((ring.refill, ring.cohorts), (0, 0));
            }
        }
    }
    #[test]
    fn cohort_policy_waits_for_four_original_enqueues_without_claiming_gpu_completion() {
        let mut r = Ring::new_policy(&[true; 24], true).unwrap();
        while r.issue_next().unwrap().is_some() {}
        assert_eq!(r.issued, 8);
        for i in 0..3 {
            r.consumed(i).unwrap();
            assert!(r.issue_next().unwrap().is_none());
        }
        r.consumed(3).unwrap();
        while r.issue_next().unwrap().is_some() {}
        assert_eq!(r.issued, 12);
        assert_eq!(r.live, 8);
        assert_eq!(r.cohorts, 2);
        assert_eq!(r.ready_slot(4).unwrap(), Some(4));
        assert!(r.ready_slot(8).is_err());
        assert!(r.finish().is_err());
    }
    #[test]
    fn resident_groups_do_not_limit_lookahead_and_foreign_or_unissued_consumption_rejects() {
        let mut mode = vec![false; 512];
        for i in (0..512).step_by(32) {
            mode[i] = true;
        }
        let mut r = Ring::new_policy(&mode, false).unwrap();
        while r.issue_next().unwrap().is_some() {}
        assert_eq!(r.live, 8);
        assert_eq!(r.issued, 256); // first eight misses plus intervening residents
        assert!(r.ready_slot(1).is_err());
        assert!(r.consumed(256).is_err());
        assert!(r.finish().is_err());
        assert!(Ring::new_policy(&[], false).is_err());
        assert!(Ring::new_policy(&[false; 513], false).is_err());
        let mut one = Ring::new_policy(&[true], false).unwrap();
        assert!(one.consumed(0).is_err());
        one.issue_next().unwrap();
        one.consumed(0).unwrap();
        assert!(one.consumed(0).is_err());
        one.finish().unwrap();
    }
}

#[cfg(all(test,feature="strata-expert-cohorts"))]
mod cohort_tests {
    use super::*;
    #[test]
    fn cohort_peek_never_consumes_and_refill_requires_actual_original_consumers() {
        let input=[true,true,false,true,true,true,true,true,true,true,false,true,true];
        let mut ring=Ring::new(&input).unwrap();
        while ring.issue_next().unwrap().is_some(){}
        let consumed=ring.consumed;let live=ring.live;
        for i in 0..ring.issued {assert!(ring.peek_slot(i).unwrap().is_some());}
        assert_eq!((ring.consumed,ring.live),(consumed,live));
        assert!(ring.peek_slot(ring.issued).unwrap().is_none());
        assert!(ring.issue_next().unwrap().is_none());
        for group in 0..input.len(){
            while ring.issue_next().unwrap().is_some(){}
            let peek=ring.peek_slot(group).unwrap().unwrap();assert_eq!(peek,ring.ready_slot(group).unwrap());
            ring.consumed(group).unwrap();assert!(ring.peek_slot(group).is_err());
        }
        ring.finish().unwrap();
    }
}
