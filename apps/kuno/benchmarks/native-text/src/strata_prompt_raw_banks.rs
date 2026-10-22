//! Ordered execution units for two genuinely separate whole raw owners.
//! These enqueue states grant no GPU completion, source or overwrite authority.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
const GROUPS: usize = 512;
const SLOTS: usize = 8;
const RESIDENT: u8 = 8;
// Replacement for the mutually exclusive original miss-ring's8KiB scratch
// inside the unchanged complete native batch's24MiB host envelope.
pub(super) const HOST_BYTES: usize = 8192;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Unit {
    pub index: usize,
    pub first: usize,
    pub end: usize,
    pub bank: usize,
}
fn reject() -> Error {
    Error::InvalidFormat("native whole raw bank unit plan")
}
pub(super) struct Plan {
    units: Zeroizing<Vec<[u16; 2]>>,
    slots: Zeroizing<Vec<u8>>,
    issued: usize,
    consumed: usize,
    occupied: [Option<usize>; 2],
    live: usize,
    peak: usize,
    misses: usize,
    copy_units: usize,
}
impl Plan {
    pub(super) fn new(members: &[u32], streamed: &[bool]) -> Result<Self> {
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
            }
            units.push([first as u16, end as u16]);
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
            end: end as usize,
            bank: index % 2,
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
    pub(super) fn current(&self, first: usize) -> Result<Unit> {
        let unit = self.unit(self.consumed)?;
        self.authenticate(unit)?;
        if unit.first != first {
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
        if unit.index != self.consumed {
            return Err(reject());
        }
        let misses = self.slots[unit.first..unit.end]
            .iter()
            .filter(|&&v| v != RESIDENT)
            .count();
        self.live = self.live.checked_sub(misses).ok_or_else(reject)?;
        self.occupied[unit.bank] = None;
        self.consumed += 1;
        Ok(())
    }
    pub(super) fn finish(&self) -> Result<(usize, usize, usize, usize)> {
        if self.issued != self.units.len()
            || self.consumed != self.units.len()
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
    fn erase(&mut self) {
        self.units.zeroize();
        self.slots.zeroize();
        self.issued.zeroize();
        self.consumed.zeroize();
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
                let mut p = Plan::new(&members, &host).unwrap();
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
            (p.issued, p.consumed, p.live, p.peak, p.misses, p.copy_units),
            (0, 0, 0, 0, 0, 0)
        );
        assert_eq!(p.occupied, [None; 2]);
        let a = unsafe { std::slice::from_raw_parts(p.units.as_ptr().cast::<u8>(), uc * 4) };
        let b = unsafe { std::slice::from_raw_parts(p.slots.as_ptr(), sc) };
        assert!(a.iter().chain(b).all(|&v| v == 0));
    }
}
