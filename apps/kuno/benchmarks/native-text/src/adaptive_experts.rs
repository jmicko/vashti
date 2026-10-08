//! Model-local decaying frequency/recency for fully adaptive original banks.
//! No route histories leave memory. Fills serve future tokens only, after an
//! all-family owning completion; no slot is permanently pinned.
use crate::bank_slots::BankSlots;
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
pub(crate) const HOST: usize = 192 * 1024;
pub(crate) const CAPACITY: usize = 144;
pub(crate) const BUDGET: usize = 4;
const LAYERS: usize = 48;
const EXPERTS: usize = 512;
pub(crate) const HALF_LIFE: u16 = 16;
pub(crate) const SCALE: u16 = 256;
pub(crate) const ADMIT_SCORE: u16 = 384;
// Rounded Q16 factors for2^(-remainder/16), including exact unity.
const FRACTIONAL_DECAY: [u32; 16] = [
    65536, 62757, 60097, 57549, 55109, 52773, 50535, 48393, 46341, 44376, 42495, 40693, 38968,
    37316, 35734, 34219,
];
fn decay(count: u16, elapsed: u16) -> u16 {
    let periods = elapsed / HALF_LIFE;
    if periods >= 16 {
        return 0;
    }
    // Multiply before shifting; dropping fractional age on every touch would
    // recreate lifetime LFU for experts called at intervals below16 tokens.
    (((u32::from(count) * FRACTIONAL_DECAY[usize::from(elapsed % HALF_LIFE)]) >> 16) >> periods)
        as u16
}
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
    pub wall_ns: u128,
    pub policy_ns: u128,
    pub fill_ns: u128,
    pub publication_ns: u128,
    pub direct_batches: u64,
    pub direct_components: u64,
    pub direct_bytes: u64,
}
pub(crate) struct Cache {
    seen: Zeroizing<Vec<u16>>,
    frequency: Zeroizing<Vec<u16>>,
    tick: Zeroizing<u16>,
    cursor: Zeroizing<usize>,
    pub(crate) ready: bool,
    pub stats: Stats,
}
impl Cache {
    pub(crate) fn new() -> Result<Self> {
        let cache = Self {
            seen: Zeroizing::new(vec![0; LAYERS * EXPERTS]),
            frequency: Zeroizing::new(vec![0; LAYERS * EXPERTS]),
            tick: Zeroizing::new(0),
            cursor: Zeroizing::new(0),
            ready: false,
            stats: Stats::default(),
        };
        if cache.owned_bytes() > HOST {
            return Err(Error::LimitExceeded("adaptive private host"));
        }
        Ok(cache)
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        (self.seen.capacity() + self.frequency.capacity()) * 2 + std::mem::size_of::<Self>()
    }
    fn decayed(&self, index: usize, epoch: u16) -> u16 {
        decay(
            self.frequency[index],
            epoch.saturating_sub(self.seen[index]),
        )
    }
    pub(crate) fn record(
        &mut self,
        layer: usize,
        position: u64,
        ids: impl Iterator<Item = usize>,
    ) -> Result<()> {
        let epoch = u16::try_from(position)
            .ok()
            .filter(|&p| p > 0)
            .ok_or(Error::ArithmeticOverflow)?;
        if layer >= LAYERS {
            return Err(Error::InvalidBufferLength);
        }
        for expert in ids {
            if expert >= EXPERTS {
                return Err(Error::InvalidBufferLength);
            }
            let index = layer * EXPERTS + expert;
            if self.seen[index] > epoch {
                return Err(Error::InvalidFormat("adaptive route position regressed"));
            }
            if self.seen[index] == epoch {
                continue;
            }
            self.frequency[index] = self.decayed(index, epoch).saturating_add(SCALE);
            self.seen[index] = epoch;
        }
        *self.tick = (*self.tick).max(epoch);
        Ok(())
    }
    pub(crate) fn score(&self, layer: usize, expert: usize) -> (u16, u16) {
        let index = layer * EXPERTS + expert;
        (self.decayed(index, *self.tick), self.seen[index])
    }
    pub(crate) fn candidate(
        &self,
        layer: usize,
        bank: &BankSlots,
    ) -> Result<Option<(usize, usize)>> {
        if !self.ready
            || layer >= LAYERS
            || bank.capacity() != CAPACITY
            || bank.experts() != EXPERTS
        {
            return Err(Error::InvalidFormat("adaptive original bank geometry"));
        }
        let mut victim = Zeroizing::new(None::<usize>);
        let mut victim_score = Zeroizing::new((u16::MAX, u16::MAX));
        for slot in 0..bank.capacity() {
            let resident = bank
                .resident_expert(slot)
                .map_err(|_| Error::InvalidFormat("adaptive original residency"))?;
            let Some(expert) = resident else {
                *victim = Some(slot);
                *victim_score = (0, 0);
                break;
            };
            let score = self.score(layer, expert);
            // All current selections are protected, including GPU hits.
            if score.1 == *self.tick {
                continue;
            }
            if score < *victim_score {
                *victim = Some(slot);
                *victim_score = score;
            }
        }
        let Some(slot) = *victim else {
            return Ok(None);
        };
        let mut best = Zeroizing::new(None::<usize>);
        let mut score = Zeroizing::new(*victim_score);
        for expert in 0..EXPERTS {
            let current = self.score(layer, expert);
            if current.1 == *self.tick
                && current.0 >= ADMIT_SCORE
                && current > *score
                && bank
                    .resident_slot(expert)
                    .map_err(|_| Error::InvalidFormat("adaptive original residency"))?
                    .is_none()
            {
                *best = Some(expert);
                *score = current;
            }
        }
        Ok(best.map(|expert| (expert, slot)))
    }
    pub(crate) fn first_layer(&self) -> usize {
        *self.cursor
    }
    pub(crate) fn advance(&mut self, next: usize) -> Result<()> {
        if !self.ready {
            return Err(Error::InvalidFormat("adaptive prefill incomplete"));
        }
        *self.cursor = next % LAYERS;
        self.stats.boundaries += 1;
        Ok(())
    }
    pub(crate) fn erase(&mut self) {
        self.seen.zeroize();
        self.frequency.zeroize();
        self.tick.zeroize();
        self.cursor.zeroize();
        self.ready = false;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn setup() -> (Cache, BankSlots) {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        let p = bank.plan(&(0..CAPACITY).collect::<Vec<_>>()).unwrap();
        bank.commit(p).unwrap();
        let mut cache = Cache::new().unwrap();
        cache.ready = true;
        (cache, bank)
    }
    #[test]
    fn fifty_expert_shift_can_use_fifty_slots_without_churn() {
        let (mut c, mut b) = setup();
        c.record(0, 1, 200..250).unwrap();
        assert!(c.candidate(0, &b).unwrap().is_none());
        c.record(0, 2, 200..250).unwrap();
        for _ in 0..50 {
            let (expert, slot) = c.candidate(0, &b).unwrap().unwrap();
            let p = b.plan_in_slots(&[expert], &[slot]).unwrap();
            b.commit(p).unwrap();
        }
        assert!(c.candidate(0, &b).unwrap().is_none());
        assert!((200..250).all(|id| b.resident_slot(id).unwrap().is_some()));
    }
    #[test]
    fn actual_gpu_hits_are_protected_and_old_frequency_decays() {
        let (mut c, b) = setup();
        for position in 1..=20 {
            c.record(0, position, [0].into_iter()).unwrap();
        }
        assert!((12 * SCALE..15 * SCALE).contains(&c.score(0, 0).0));
        assert_eq!(c.score(0, 0).1, 20);
        c.record(0, 21, [200].into_iter()).unwrap();
        c.record(0, 22, [0, 200].into_iter()).unwrap();
        let (_, slot) = c.candidate(0, &b).unwrap().unwrap();
        assert_ne!(slot, 0);
        c.record(0, 148, [201].into_iter()).unwrap();
        assert!(c.score(0, 0).0 < SCALE / 8);
        assert!(c.candidate(0, &b).unwrap().is_none());
    }
    #[test]
    fn one_off_duplicate_invalid_and_regressing_observations_are_rejected() {
        let (mut c, b) = setup();
        assert!(c.owned_bytes() <= HOST);
        c.record(0, 1, [200, 200].into_iter()).unwrap();
        assert_eq!(c.score(0, 200), (SCALE, 1));
        assert!(c.candidate(0, &b).unwrap().is_none());
        assert!(c.record(48, 2, [200].into_iter()).is_err());
        assert!(c.record(0, 2, [512].into_iter()).is_err());
        assert!(c.record(0, 0, [200].into_iter()).is_err());
        c.record(0, 2, [200].into_iter()).unwrap();
        assert!(c.record(0, 1, [200].into_iter()).is_err());
        c.erase();
        assert!(c.seen.is_empty() && c.frequency.is_empty());
        assert_eq!((*c.tick, *c.cursor, c.ready), (0, 0, false));
    }
    #[test]
    fn fractional_decay_has_exact_halves_and_bounded_reference_error() {
        assert_eq!(decay(1024, 16), 512);
        assert_eq!(decay(1024, 32), 256);
        assert_eq!(decay(u16::MAX, 256), 0);
        for interval in [1u64, 8, 15] {
            let (mut c, _) = setup();
            let mut expected = 0.0f64;
            let factor = 2f64.powf(-(interval as f64) / f64::from(HALF_LIFE));
            for n in 1..=100 {
                c.record(0, n * interval, [200].into_iter()).unwrap();
                expected = expected * factor + f64::from(SCALE);
                let actual = f64::from(c.score(0, 200).0);
                assert!((actual - expected).abs() < 26.0);
            }
            assert!(c.score(0, 200).0 < 24 * SCALE);
        }
    }
    #[test]
    fn new_working_set_overtakes_old_popularity_without_lifetime_pin() {
        let (mut c, _) = setup();
        for pos in 1..=100 {
            c.record(0, pos, [0].into_iter()).unwrap();
        }
        for pos in 101..=120 {
            c.record(0, pos, [200].into_iter()).unwrap();
        }
        assert!(c.score(0, 200) > c.score(0, 0));
        let (mut c, b) = setup();
        c.record(0, 1, [200].into_iter()).unwrap();
        c.record(0, 17, [200].into_iter()).unwrap();
        assert_eq!(c.score(0, 200).0, ADMIT_SCORE);
        assert!(c.candidate(0, &b).unwrap().is_some());
    }
    #[test]
    fn grouped_plans_and_history_fit_existing_reservation() {
        let (c, mut b) = setup();
        let mut live = c.owned_bytes() + 4096;
        let mut plans = Vec::new();
        for layer in 0..BUDGET {
            let p = b.plan_in_slots(&[200 + layer], &[layer]).unwrap();
            live += p.owned_metadata_bytes().unwrap();
            plans.push(p);
        }
        assert!(live <= HOST);
        assert_eq!(plans.len(), BUDGET);
    }
}

#[cfg(test)]
#[path = "adaptive_convergence_tests.rs"]
mod convergence_tests;
