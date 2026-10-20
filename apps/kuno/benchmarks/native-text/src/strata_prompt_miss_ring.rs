//! Enqueue-only scheduling for the original eight raw transfer slots. A
//! resident group never consumes a slot. Consumed means its ORIGINAL kernels
//! were enqueued, NOT that GPU use completed: original Graph range/timeline
//! dependencies still gate every physical overwrite.
use kuno_text::{Error, Result};
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
    slots: Vec<usize>,
    issued: usize,
    consumed: usize,
    live: usize,
    peak: usize,
    misses: usize,
}
impl Ring {
    pub(super) fn new(streamed: &[bool]) -> Result<Self> {
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
            slots,
            issued: 0,
            consumed: 0,
            live: 0,
            peak: 0,
            misses,
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
            self.live += 1;
            self.peak = self.peak.max(self.live);
        }
        let group = self.issued;
        self.issued += 1;
        Ok(Some((group, (raw != RESIDENT).then_some(raw))))
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
                let mut r = Ring::new(&mode).unwrap();
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
    fn resident_groups_do_not_limit_lookahead_and_foreign_or_unissued_consumption_rejects() {
        let mut mode = vec![false; 512];
        for i in (0..512).step_by(32) {
            mode[i] = true;
        }
        let mut r = Ring::new(&mode).unwrap();
        while r.issue_next().unwrap().is_some() {}
        assert_eq!(r.live, 8);
        assert_eq!(r.issued, 256); // first eight misses plus intervening residents
        assert!(r.ready_slot(1).is_err());
        assert!(r.consumed(256).is_err());
        assert!(r.finish().is_err());
        assert!(Ring::new(&[]).is_err());
        assert!(Ring::new(&[false; 513]).is_err());
        let mut one = Ring::new(&[true]).unwrap();
        assert!(one.consumed(0).is_err());
        one.issue_next().unwrap();
        one.consumed(0).unwrap();
        assert!(one.consumed(0).is_err());
        one.finish().unwrap();
    }
}
