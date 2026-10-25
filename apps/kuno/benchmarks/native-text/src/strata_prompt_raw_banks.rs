//! Ordered execution units for two genuinely separate whole raw owners.
//! These enqueue states grant no GPU completion, source or overwrite authority.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
const GROUPS: usize = 512;
const SLOTS: usize = 8;
const RESIDENT: u8 = 8;
const LEGACY_GROUPS: usize = 4;
// All original group boundaries are <=512. Keep the upload-unit kind in its
// existing two-u16 record rather than adding another routing allocation.
const LEGACY_UNIT: u16 = 1 << 15;
const REGULAR_UNIT: u16 = 1 << 14;
const LARGE_UNIT: u16 = 1 << 13;
// Replacement for the mutually exclusive original miss-ring's8KiB scratch
// inside the unchanged complete native batch's24MiB host envelope.
pub(super) const HOST_BYTES: usize = 8192;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Unit {
    pub index: usize,
    pub first: usize,
    pub end: usize,
    pub bank: usize,
    pub legacy: bool,
    pub regular: bool,
    pub large: bool,
}
fn reject() -> Error {
    Error::InvalidFormat("native whole raw bank unit plan")
}
pub(super) struct Plan {
    units: Zeroizing<Vec<[u16; 2]>>,
    slots: Zeroizing<Vec<u8>>,
    issued: usize,
    consumed: usize,
    next_group: usize,
    occupied: [Option<usize>; 2],
    live: usize,
    peak: usize,
    misses: usize,
    copy_units: usize,
}
impl Plan {
    pub(super) fn new(members: &[u32], streamed: &[bool]) -> Result<Self> {
        Self::with_lookahead(members, streamed, cfg!(feature = "strata-raw-lookahead"))
    }
    fn with_lookahead(members: &[u32], streamed: &[bool], lookahead: bool) -> Result<Self> {
        Self::with_policy(members, streamed, lookahead, false, false)
    }
    #[cfg(feature = "strata-regular-cohorts")]
    pub(super) fn new_regular(members: &[u32], streamed: &[bool]) -> Result<Self> {
        Self::with_policy(members, streamed, true, true, false)
    }
    #[cfg(feature="strata-large-cohorts")]
    pub(super) fn new_large(members: &[u32], streamed: &[bool]) -> Result<Self> {
        Self::with_policy(members, streamed, true, true, true)
    }
    fn with_policy(members: &[u32], streamed: &[bool], lookahead: bool, regular_cohorts: bool, large_cohorts: bool) -> Result<Self> {
        if members.is_empty()
            || members.len() > GROUPS
            || members.len() != streamed.len()
            || members.iter().any(|&n| !(1..=4096).contains(&n))
        {
            return Err(reject());
        }
        let mut units = Zeroizing::new(Vec::new());
        let mut slots = Zeroizing::new(Vec::new());
        units
            .try_reserve_exact(members.len())
            .map_err(|_| reject())?;
        slots
            .try_reserve_exact(members.len())
            .map_err(|_| reject())?;
        // Includes the simultaneously live classification allocation, all Vec
        // headers, fixed plan state and 1KiB conservative current-unit scratch.
        let bytes = units
            .capacity()
            .checked_mul(size_of::<[u16; 2]>())
            .and_then(|n| n.checked_add(slots.capacity()))
            .and_then(|n| n.checked_add(GROUPS)) // caller checks classification capacity <=512
            .and_then(|n| n.checked_add(size_of::<Self>() + size_of::<Vec<bool>>() + 1024))
            .ok_or_else(reject)?;
        if units.capacity() > GROUPS || slots.capacity() > GROUPS || bytes > HOST_BYTES {
            return Err(reject());
        }
        let mut first = 0;
        let mut misses = 0;
        while first < members.len() {
            let mut end = first + 1;
            if members[first] < 32 {
                while end < members.len() && end < first + SLOTS && members[end] < 32 {
                    end += 1;
                }
            } else if regular_cohorts && members[first] < 256 {
                let mut minimum = members[first].div_ceil(16);
                let mut maximum = minimum;
                while end < members.len() && end < first + SLOTS && (32..256).contains(&members[end]) {
                    let blocks = members[end].div_ceil(16);
                    let lo = minimum.min(blocks);
                    let hi = maximum.max(blocks);
                    // A fixed maximum-Y dispatch has at most twice its active
                    // tile work. Keep order; never reorder or skip a group.
                    if hi > 2 * lo { break; }
                    minimum = lo;
                    maximum = hi;
                    end += 1;
                }
            } else if large_cohorts {
                // Consecutive original groups only. Padding belongs to the
                // producer, so charge its entire Pad32 extent before admission.
                let mut rows = members[first].div_ceil(32) * 32;
                while end < members.len() && end-first < SLOTS && members[end]>=256 {
                    let next=members[end].div_ceil(32)*32;
                    if rows+next>4096 {break;}
                    rows+=next;end+=1;
                }
            } else if lookahead {
                while end < members.len() && end < first + LEGACY_GROUPS
                    && members[end] >= if regular_cohorts { 256 } else { 32 } {
                    end += 1;
                }
            }
            let regular = regular_cohorts && (32..256).contains(&members[first]);
            let large = large_cohorts && members[first]>=256;
            let legacy = lookahead && members[first] >= 32 && !regular && !large;
            units.push([
                first as u16,
                end as u16 | if legacy { LEGACY_UNIT } else { 0 }
                    | if regular { REGULAR_UNIT } else { 0 }
                    | if large { LARGE_UNIT } else { 0 },
            ]);
            let mut next_slot = 0;
            for &host in &streamed[first..end] {
                slots.push(if host {
                    let slot = next_slot;
                    next_slot += 1;
                    misses += 1;
                    slot
                } else {
                    RESIDENT
                });
            }
            if next_slot as usize > SLOTS {
                return Err(reject());
            }
            first = end;
        }
        Ok(Self {
            units,
            slots,
            issued: 0,
            consumed: 0,
            next_group: 0,
            occupied: [None; 2],
            live: 0,
            peak: 0,
            misses,
            copy_units: 0,
        })
    }
    fn unit(&self, index: usize) -> Result<Unit> {
        let &[first, end] = self.units.get(index).ok_or_else(reject)?;
        Ok(Unit {
            index,
            first: first as usize,
            end: (end & !(LEGACY_UNIT | REGULAR_UNIT | LARGE_UNIT)) as usize,
            bank: index % 2,
            legacy: end & LEGACY_UNIT != 0,
            regular: end & REGULAR_UNIT != 0,
            large: end & LARGE_UNIT != 0,
        })
    }
    fn authenticate(&self, unit: Unit) -> Result<()> {
        if self.unit(unit.index)? != unit
            || self.occupied[unit.bank] != Some(unit.index)
            || unit.index >= self.issued
        {
            return Err(reject());
        }
        Ok(())
    }
    /// At most two original units can be enqueued ahead. Issuing a unit marks
    /// its host slots live, not ready/completed; real Graph hazards remain.
    pub(super) fn issue_next(&mut self) -> Result<Option<Unit>> {
        if self.issued == self.units.len() {
            return Ok(None);
        }
        let unit = self.unit(self.issued)?;
        if self.occupied[unit.bank].is_some() {
            return Ok(None);
        }
        let misses = self.slots[unit.first..unit.end]
            .iter()
            .filter(|&&v| v != RESIDENT)
            .count();
        self.live = self.live.checked_add(misses).ok_or_else(reject)?;
        if self.live > 2 * SLOTS {
            return Err(reject());
        }
        self.peak = self.peak.max(self.live);
        self.copy_units += usize::from(misses > 0);
        self.occupied[unit.bank] = Some(unit.index);
        self.issued += 1;
        Ok(Some(unit))
    }
    pub(super) fn current(&self, group: usize) -> Result<Unit> {
        let unit = self.unit(self.consumed)?;
        self.authenticate(unit)?;
        if group != self.next_group || !(unit.first..unit.end).contains(&group) {
            return Err(reject());
        }
        Ok(unit)
    }
    pub(super) fn slots(&self, unit: Unit) -> Result<&[u8]> {
        self.authenticate(unit)?;
        Ok(&self.slots[unit.first..unit.end])
    }
    pub(super) fn slot(&self, unit: Unit, group: usize) -> Result<Option<usize>> {
        self.authenticate(unit)?;
        if !(unit.first..unit.end).contains(&group) {
            return Err(reject());
        }
        Ok((self.slots[group] != RESIDENT).then_some(self.slots[group] as usize))
    }
    /// Called only after ALL genuine unit kernels were enqueued. This allows
    /// unit+2's host copies; every actual overwrite still waits real readers.
    pub(super) fn consumed(&mut self, unit: Unit) -> Result<()> {
        self.authenticate(unit)?;
        // A real small cohort (or the unchanged original singleton path) closes
        // all of its groups together. A lookahead legacy upload unit must use
        // complete_group after EACH original final DOWN instead.
        if unit.legacy || self.next_group != unit.first {
            return Err(reject());
        }
        self.retire(unit, unit.end)
    }
    /// Advance only the next genuinely enqueued original legacy group. This is
    /// host bookkeeping, never a claim that a copy or kernel has completed.
    /// The original native group-end marker remains the caller's prerequisite.
    #[cfg(any(test, feature = "strata-raw-lookahead"))]
    pub(super) fn complete_group(&mut self, unit: Unit, group: usize) -> Result<bool> {
        self.authenticate(unit)?;
        if !unit.legacy
            || unit.index != self.consumed
            || group != self.next_group
            || !(unit.first..unit.end).contains(&group)
        {
            return Err(reject());
        }
        let next = group + 1;
        if next == unit.end {
            self.retire(unit, next)?;
            Ok(true)
        } else {
            self.next_group = next;
            Ok(false)
        }
    }
    fn retire(&mut self, unit: Unit, next: usize) -> Result<()> {
        self.authenticate(unit)?;
        if unit.index != self.consumed || next != unit.end {
            return Err(reject());
        }
        let misses = self.slots[unit.first..unit.end]
            .iter()
            .filter(|&&v| v != RESIDENT)
            .count();
        self.live = self.live.checked_sub(misses).ok_or_else(reject)?;
        self.occupied[unit.bank] = None;
        self.consumed += 1;
        self.next_group = next;
        Ok(())
    }
    pub(super) fn finish(&self) -> Result<(usize, usize, usize, usize)> {
        if self.issued != self.units.len()
            || self.consumed != self.units.len()
            || self.next_group != self.slots.len()
            || self.live != 0
            || self.occupied != [None; 2]
        {
            return Err(reject());
        }
        Ok((
            self.misses,
            self.slots.len() - self.misses,
            self.peak,
            self.copy_units,
        ))
    }
    #[cfg(any(test, feature = "strata-raw-lookahead"))]
    pub(super) fn legacy_geometry(&self) -> Result<[usize; 4]> {
        self.finish()?;
        let mut result = [0; 4];
        for &[first, encoded_end] in self.units.iter() {
            if encoded_end & LEGACY_UNIT != 0 {
                let groups = usize::from((encoded_end & !LEGACY_UNIT) - first);
                result[0] += 1;
                result[1] += groups;
                if groups > 1 {
                    result[2] += 1;
                    result[3] += groups;
                }
            }
        }
        Ok(result)
    }
    fn erase(&mut self) {
        self.units.zeroize();
        self.slots.zeroize();
        self.issued.zeroize();
        self.consumed.zeroize();
        self.next_group.zeroize();
        self.occupied.zeroize();
        self.live.zeroize();
        self.peak.zeroize();
        self.misses.zeroize();
        self.copy_units.zeroize();
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        self.erase();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn authentic_small_units_and_single_legacy_groups_preserve_all_original_order_and_slots() {
        for len in [1, 7, 8, 9, 31, 64, 512] {
            for pattern in 0..10 {
                let members: Vec<_> = (0..len)
                    .map(|i| {
                        if pattern == 0 {
                            1
                        } else if (i + pattern) % 5 == 0 {
                            32 + (i % 4096) as u32
                        } else {
                            1 + (i % 31) as u32
                        }
                    })
                    .collect();
                let host: Vec<_> = (0..len)
                    .map(|i| pattern == 1 || pattern > 1 && (i * 17 + pattern) % 3 != 0)
                    .collect();
                // Keep the ORIGINAL singleton policy and its exact oracle even
                // when this test binary also enables the optional lookahead.
                let mut p = Plan::with_lookahead(&members, &host, false).unwrap();
                let mut seen = vec![0; len];
                let mut first = 0;
                while first < len {
                    while p.issue_next().unwrap().is_some() {}
                    let unit = p.current(first).unwrap();
                    assert!(unit.end <= len && unit.end > first && unit.end - first <= 8);
                    if members[first] >= 32 {
                        assert_eq!(unit.end, first + 1);
                    } else {
                        assert!(members[first..unit.end].iter().all(|&n| n < 32));
                    }
                    let mut slot = 0;
                    for group in unit.first..unit.end {
                        seen[group] += 1;
                        if host[group] {
                            assert_eq!(p.slot(unit, group).unwrap(), Some(slot));
                            slot += 1;
                        } else {
                            assert_eq!(p.slot(unit, group).unwrap(), None);
                        }
                    }
                    p.consumed(unit).unwrap();
                    first = unit.end;
                }
                let (misses, resident, peak, cohorts) = p.finish().unwrap();
                assert_eq!(misses, host.iter().filter(|&&h| h).count());
                assert_eq!(misses + resident, len);
                assert!(peak <= 16);
                assert!(cohorts <= p.units.len());
                assert!(seen.iter().all(|&n| n == 1));
            }
        }
    }
    #[test]
    fn both_whole_banks_prefetch_then_only_the_consumed_bank_can_be_refilled() {
        let mut p = Plan::new(&[1; 32], &[true; 32]).unwrap();
        let a = p.issue_next().unwrap().unwrap();
        let b = p.issue_next().unwrap().unwrap();
        assert_eq!((a.bank, b.bank, p.live, p.peak), (0, 1, 16, 16));
        assert!(p.issue_next().unwrap().is_none());
        assert!(p.consumed(b).is_err());
        assert!(p.finish().is_err());
        p.consumed(a).unwrap();
        assert!(p.slots(a).is_err());
        let c = p.issue_next().unwrap().unwrap();
        assert_eq!((c.index, c.bank, p.live), (2, 0, 16));
        let mut changed = b;
        changed.bank = 0;
        assert!(p.consumed(changed).is_err());
        assert!(p.current(c.first).is_err());
        p.consumed(b).unwrap();
        let d = p.issue_next().unwrap().unwrap();
        p.consumed(c).unwrap();
        p.consumed(d).unwrap();
        assert_eq!(p.finish().unwrap(), (32, 0, 16, 4));
    }
    #[test]
    fn raw_bank_plan_rejects_invalid_geometry_and_erases_full_reserved_metadata() {
        assert!(Plan::new(&[], &[]).is_err());
        assert!(Plan::new(&[1], &[]).is_err());
        assert!(Plan::new(&[0], &[false]).is_err());
        assert!(Plan::new(&[4097], &[false]).is_err());
        assert!(Plan::new(&[1; 513], &[false; 513]).is_err());
        let mut p = Plan::new(&[1; 512], &[true; 512]).unwrap();
        p.issue_next().unwrap();
        let uc = p.units.capacity();
        let sc = p.slots.capacity();
        assert!(
            uc * 4 + sc + 512 + size_of::<Plan>() + size_of::<Vec<bool>>() + 1024 <= HOST_BYTES
        );
        p.erase();
        assert!(p.units.is_empty() && p.slots.is_empty());
        assert_eq!(
            (
                p.issued,
                p.consumed,
                p.next_group,
                p.live,
                p.peak,
                p.misses,
                p.copy_units
            ),
            (0, 0, 0, 0, 0, 0, 0)
        );
        assert_eq!(p.occupied, [None; 2]);
        let a = unsafe { std::slice::from_raw_parts(p.units.as_ptr().cast::<u8>(), uc * 4) };
        let b = unsafe { std::slice::from_raw_parts(p.slots.as_ptr(), sc) };
        assert!(a.iter().chain(b).all(|&v| v == 0));
    }
}

#[cfg(all(test, feature = "strata-raw-lookahead"))]
#[path = "strata_prompt_raw_lookahead_tests.rs"]
mod lookahead_tests;

#[cfg(all(test, feature = "strata-raw-lookahead"))]
#[path = "strata_prompt_raw_lookahead_queue_tests.rs"]
mod queue_tests;

#[cfg(all(test,feature="strata-large-cohorts"))]
#[test]
fn large_cohort_two_bank_plan_charges_padding_before_issue_and_preserves_order() {
    let members=[31,32,255,256,257,265,288,511,1025,1025,1025,1025,4095,4096];
    let mut plan=Plan::new_large(&members,&[true;14]).unwrap();
    let mut next=0;
    while next<members.len() {
        while plan.issue_next().unwrap().is_some() {}
        let unit=plan.current(next).unwrap();assert!(!unit.legacy);
        if unit.large {
            assert!(members[unit.first..unit.end].iter().all(|&m|m>=256));
            assert!(members[unit.first..unit.end].iter().map(|m|m.div_ceil(32)*32).sum::<u32>()<=4096);
            assert!(unit.end-unit.first<=8);
        }
        assert!(plan.live<=16);next=unit.end;plan.consumed(unit).unwrap();
    }
    assert!(plan.issue_next().unwrap().is_none());assert_eq!(plan.live,0);
}
