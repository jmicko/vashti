//! Transient repeat-miss admission for a small rotating part of original banks.
//! No route history is serialized. Current misses remain CPU work; candidates
//! are filled only at an all-family completed token boundary.
use crate::bank_slots::BankSlots;
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
pub(crate) const HOST: usize = 192 * 1024;
pub(crate) const ROTATING: usize = 16;
pub(crate) const BUDGET: usize = 4;
const LAYERS: usize = 48;
const EXPERTS: usize = 512;
const WINDOW: u16 = 32;
#[derive(Default)]
pub(crate) struct Stats {
    pub promotions: u64,
    pub bytes: u64,
    pub read_ns: u128,
    pub upload_ns: u128,
    pub fence_ns: u128,
    pub boundaries: u64,
    pub completed_batches: u64,
    pub private_peak: usize,
    pub source_peak: u64,
}
pub(crate) struct Cache {
    seen: Zeroizing<Vec<u16>>,
    repeats: Zeroizing<Vec<u16>>,
    rotating: Zeroizing<Vec<[usize; ROTATING]>>,
    tick: Zeroizing<u16>,
    cursor: Zeroizing<usize>,
    pub stats: Stats,
}
impl Cache {
    pub(crate) fn new(
        mut bank: impl FnMut(usize) -> Result<Zeroizing<Vec<usize>>>,
    ) -> Result<Self> {
        let mut rotating = Zeroizing::new(Vec::with_capacity(LAYERS));
        for layer in 0..LAYERS {
            let slots = bank(layer)?;
            rotating.push(
                slots
                    .as_slice()
                    .try_into()
                    .map_err(|_| Error::InvalidBufferLength)?,
            );
        }
        let cache = Self {
            seen: Zeroizing::new(vec![0; LAYERS * EXPERTS]),
            repeats: Zeroizing::new(vec![0; LAYERS * EXPERTS]),
            rotating,
            tick: Zeroizing::new(1),
            cursor: Zeroizing::new(0),
            stats: Stats::default(),
        };
        if cache.owned_bytes() > HOST {
            return Err(Error::LimitExceeded("adaptive private host"));
        }
        Ok(cache)
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        (self.seen.capacity() + self.repeats.capacity()) * 2
            + self.rotating.capacity() * std::mem::size_of::<[usize; ROTATING]>()
            + std::mem::size_of::<Self>()
    }
    pub(crate) fn observe(
        &mut self,
        layer: usize,
        misses: impl Iterator<Item = usize>,
    ) -> Result<()> {
        if layer >= LAYERS {
            return Err(Error::InvalidBufferLength);
        }
        for expert in misses {
            if expert >= EXPERTS {
                return Err(Error::InvalidBufferLength);
            }
            let index = layer * EXPERTS + expert;
            // Duplicate observations in one token cannot create repetition.
            if self.seen[index] == *self.tick {
                continue;
            }
            self.repeats[index] = if self.seen[index] != 0
                && self
                    .tick
                    .checked_sub(self.seen[index])
                    .is_some_and(|d| d <= WINDOW)
            {
                self.repeats[index].saturating_add(1)
            } else {
                1
            };
            self.seen[index] = *self.tick;
        }
        Ok(())
    }
    pub(crate) fn slots(&self, layer: usize) -> Result<&[usize]> {
        self.rotating
            .get(layer)
            .map(|s| s.as_slice())
            .ok_or(Error::InvalidBufferLength)
    }
    pub(crate) fn is_rotating(&self, layer: usize, slot: usize) -> Result<bool> {
        Ok(self.slots(layer)?.contains(&slot))
    }
    pub(crate) fn candidate(&self, layer: usize, bank: &BankSlots) -> Result<Option<usize>> {
        if layer >= LAYERS || bank.capacity() != 144 || bank.experts() != EXPERTS {
            return Err(Error::InvalidFormat("adaptive original bank geometry"));
        }
        let mut best = None;
        let mut score = 0;
        for expert in 0..EXPERTS {
            let index = layer * EXPERTS + expert;
            if self.seen[index] == *self.tick
                && self.repeats[index] >= 2
                && self.repeats[index] > score
                && bank
                    .resident_slot(expert)
                    .map_err(|_| Error::InvalidFormat("adaptive original residency"))?
                    .is_none()
            {
                best = Some(expert);
                score = self.repeats[index];
            }
        }
        Ok(best)
    }
    pub(crate) fn first_layer(&self) -> usize {
        *self.cursor
    }
    pub(crate) fn advance(&mut self, next: usize) -> Result<()> {
        *self.tick = self.tick.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        *self.cursor = next % LAYERS;
        self.stats.boundaries += 1;
        Ok(())
    }
    pub(crate) fn erase(&mut self) {
        self.seen.zeroize();
        self.repeats.zeroize();
        self.rotating.zeroize();
        self.tick.zeroize();
        self.cursor.zeroize();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (Cache, BankSlots) {
        let mut bank = BankSlots::new(144, 512).unwrap();
        let p = bank.plan(&(0..144).collect::<Vec<_>>()).unwrap();
        bank.commit(p).unwrap();
        (
            Cache::new(|_| Ok(bank.cold_slots(ROTATING).unwrap())).unwrap(),
            bank,
        )
    }
    #[test]
    fn one_off_duplicate_stale_and_resident_misses_do_not_promote() {
        let (mut c, mut b) = setup();
        assert!(c.owned_bytes() <= HOST);
        c.observe(0, [200, 200].into_iter()).unwrap();
        assert_eq!(c.candidate(0, &b).unwrap(), None);
        c.advance(1).unwrap();
        c.observe(0, [200].into_iter()).unwrap();
        assert_eq!(c.candidate(0, &b).unwrap(), Some(200));
        let p = b.plan_in_slots(&[200], c.slots(0).unwrap()).unwrap();
        assert!(c.owned_bytes() + p.owned_metadata_bytes().unwrap() + 4096 <= HOST);
        b.commit(p).unwrap();
        assert_eq!(c.candidate(0, &b).unwrap(), None);
        for _ in 0..33 {
            c.advance(0).unwrap();
        }
        c.observe(0, [300].into_iter()).unwrap();
        c.advance(1).unwrap();
        for _ in 0..33 {
            c.advance(0).unwrap();
        }
        c.observe(0, [300].into_iter()).unwrap();
        assert_eq!(c.candidate(0, &b).unwrap(), None);
        c.erase();
        assert!(c.seen.is_empty() && c.repeats.is_empty() && c.rotating.is_empty());
        assert_eq!((*c.tick, *c.cursor), (0, 0));
    }
    #[test]
    fn grouped_original_plans_fit_reservation_and_only_rotating_sources_can_return_to_cpu() {
        let (cache, mut bank) = setup();
        let mut live = cache.owned_bytes() + 4096;
        let mut plans = Vec::new();
        for layer in 0..BUDGET {
            assert!(cache.is_rotating(layer, 0).unwrap());
            assert!(!cache.is_rotating(layer, 128).unwrap());
            let p = bank
                .plan_in_slots(&[200 + layer], cache.slots(layer).unwrap())
                .unwrap();
            live += p.owned_metadata_bytes().unwrap();
            plans.push(p);
        }
        assert!(live <= HOST);
        assert_eq!(plans.len(), BUDGET);
        assert!(cache.is_rotating(48, 0).is_err());
    }
    #[test]
    fn candidate_requires_current_token_and_prefers_repeated_recent_misses() {
        let (mut c, b) = setup();
        for _ in 0..3 {
            c.observe(47, [200].into_iter()).unwrap();
            c.advance(0).unwrap();
        }
        assert_eq!(c.candidate(47, &b).unwrap(), None);
        c.observe(47, [201].into_iter()).unwrap();
        c.advance(0).unwrap();
        c.observe(47, [200, 201].into_iter()).unwrap();
        assert_eq!(c.candidate(47, &b).unwrap(), Some(200));
        assert!(c.observe(48, [200].into_iter()).is_err());
        assert!(c.observe(0, [512].into_iter()).is_err());
    }
}
