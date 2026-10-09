//! Exact route union for one compiled-public, seven-step laboratory replay.
//! No expert IDs leave this transient owner; no substitutions or router changes.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};

pub(crate) const HOST: usize = 32 * 1024;
pub(crate) const START: usize = 75;
pub(crate) const DECODE: usize = 7;
pub(crate) const PREFIX: usize = DECODE + 1;
pub(crate) const CONTEXT: usize = 203;
const LAYERS: usize = 48;
const WORDS: usize = 8;
const RANKS: usize = 10;
const ALL_LAYERS: u64 = (1u64 << LAYERS) - 1;

fn rejected() -> Error {
    Error::InvalidFormat("public oracle trace rejected")
}

pub(crate) struct Trace {
    bits: Zeroizing<Vec<u64>>,
    coverage: Zeroizing<[u64; DECODE]>,
    replay: bool,
}

impl Trace {
    pub(crate) const HOST: usize = crate::oracle_trace::HOST;

    pub(crate) fn training() -> Result<Self> {
        let mut bits = Zeroizing::new(Vec::new());
        bits.try_reserve_exact(LAYERS * WORDS)
            .map_err(|_| Error::LimitExceeded("public oracle trace host"))?;
        bits.resize(LAYERS * WORDS, 0);
        let trace = Self {
            bits,
            coverage: Zeroizing::new([0; DECODE]),
            replay: false,
        };
        trace.validate_shape()?;
        Ok(trace)
    }

    pub(crate) fn owned_bytes(&self) -> usize {
        self.bits.capacity() * std::mem::size_of::<u64>() + std::mem::size_of::<Self>()
    }

    fn validate_shape(&self) -> Result<()> {
        if self.bits.len() != LAYERS * WORDS || self.owned_bytes() > HOST {
            return Err(rejected());
        }
        Ok(())
    }

    pub(crate) fn counts(&self) -> Vec<usize> {
        self.bits
            .chunks_exact(WORDS)
            .map(|row| row.iter().map(|word| word.count_ones() as usize).sum())
            .collect()
    }

    fn validate_union(&self) -> Result<()> {
        self.validate_shape()?;
        if self.bits.chunks_exact(WORDS).any(|row| {
            !(RANKS..=RANKS * DECODE)
                .contains(&row.iter().map(|word| word.count_ones() as usize).sum())
        }) {
            return Err(rejected());
        }
        Ok(())
    }

    pub(crate) fn complete(&self) -> Result<()> {
        self.validate_union()?;
        if self.coverage.iter().any(|&mask| mask != ALL_LAYERS) {
            return Err(rejected());
        }
        Ok(())
    }

    pub(crate) fn validate_ready_replay(&self) -> Result<()> {
        self.validate_union()?;
        if !self.replay || self.coverage.iter().any(|&mask| mask != 0) {
            return Err(rejected());
        }
        Ok(())
    }

    pub(crate) fn validated_replay_copy(&self) -> Result<Self> {
        self.complete()?;
        if self.replay {
            return Err(rejected());
        }
        let mut trace = Self::training()?;
        trace.bits.copy_from_slice(&self.bits);
        trace.replay = true;
        trace.validate_ready_replay()?;
        Ok(trace)
    }

    pub(crate) fn union(&self, layer: usize) -> Result<Zeroizing<Vec<usize>>> {
        self.validate_union()?;
        let row = self
            .bits
            .get(layer.checked_mul(WORDS).ok_or(Error::ArithmeticOverflow)?..)
            .and_then(|remaining| remaining.get(..WORDS))
            .ok_or(rejected())?;
        let count = row.iter().map(|word| word.count_ones() as usize).sum();
        let mut selected = Zeroizing::new(Vec::new());
        selected
            .try_reserve_exact(count)
            .map_err(|_| Error::LimitExceeded("public oracle union host"))?;
        for (index, &word) in row.iter().enumerate() {
            for bit in 0..64 {
                if word & (1u64 << bit) != 0 {
                    selected.push(index * 64 + bit);
                }
            }
        }
        if selected.len() != count || selected.len() > RANKS * DECODE {
            return Err(rejected());
        }
        Ok(selected)
    }

    pub(crate) fn record(&mut self, layer: usize, position: u64, ids: &[usize]) -> Result<()> {
        self.validate_shape()?;
        let token = position
            .checked_sub((START + 1) as u64)
            .and_then(|p| usize::try_from(p).ok())
            .filter(|&p| p < DECODE)
            .ok_or(rejected())?;
        if layer >= LAYERS || ids.len() != RANKS || self.coverage[token] & (1u64 << layer) != 0 {
            return Err(rejected());
        }
        let mut current = Zeroizing::new([0u64; WORDS]);
        for &expert in ids {
            if expert >= WORDS * 64 || current[expert / 64] & (1u64 << (expert % 64)) != 0 {
                return Err(rejected());
            }
            current[expert / 64] |= 1u64 << (expert % 64);
        }
        let row = &mut self.bits[layer * WORDS..(layer + 1) * WORDS];
        if self.replay
            && row
                .iter()
                .zip(current.iter())
                .any(|(allowed, selected)| selected & !allowed != 0)
        {
            return Err(rejected());
        }
        if !self.replay {
            for (allowed, selected) in row.iter_mut().zip(current.iter()) {
                *allowed |= selected;
            }
        }
        self.coverage[token] |= 1u64 << layer;
        Ok(())
    }

    pub(crate) fn is_replay(&self) -> bool {
        self.replay
    }

    pub(crate) fn erase(&mut self) {
        self.bits.zeroize();
        self.coverage.zeroize();
        self.replay.zeroize();
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        self.erase();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(disjoint: bool) -> Trace {
        let mut trace = Trace::training().unwrap();
        for token in 0..DECODE {
            for layer in 0..LAYERS {
                let ids = Zeroizing::new(std::array::from_fn::<_, RANKS, _>(|rank| {
                    if disjoint { token * RANKS + rank } else { rank }
                }));
                trace
                    .record(layer, (START + token + 1) as u64, &*ids)
                    .unwrap();
            }
        }
        trace
    }

    #[test]
    fn exact_coverage_phase_and_bounds() {
        let mut trace = Trace::training().unwrap();
        assert!(trace.owned_bytes() <= Trace::HOST);
        assert!(trace.complete().is_err());
        assert!(trace.validated_replay_copy().is_err());
        let ids = std::array::from_fn::<_, RANKS, _>(|rank| rank);
        for position in [START as u64, (START + DECODE + 1) as u64, u64::MAX] {
            assert!(trace.record(0, position, &ids).is_err());
        }
        assert!(trace.record(LAYERS, (START + 1) as u64, &ids).is_err());
        assert!(trace.record(0, (START + 1) as u64, &ids[..9]).is_err());
        assert!(trace.record(0, (START + 1) as u64, &[0; RANKS]).is_err());
        let mut invalid = ids;
        invalid[9] = 512;
        assert!(trace.record(0, (START + 1) as u64, &invalid).is_err());
        assert_eq!(trace.counts(), vec![0; LAYERS]);
        trace.record(0, (START + 1) as u64, &ids).unwrap();
        assert!(trace.record(0, (START + 1) as u64, &ids).is_err());
        assert!(trace.complete().is_err());
    }

    #[test]
    fn complete_union_is_bounded_and_replay_checks_every_actual_id() {
        for disjoint in [false, true] {
            let trace = trained(disjoint);
            trace.complete().unwrap();
            assert_eq!(trace.counts(), vec![if disjoint { 70 } else { 10 }; LAYERS]);
            assert!(trace.union(LAYERS).is_err());
            assert!(trace.validate_ready_replay().is_err());
            let mut replay = trace.validated_replay_copy().unwrap();
            replay.validate_ready_replay().unwrap();
            assert!(replay.complete().is_err());
            assert!(
                replay
                    .record(0, (START + 1) as u64, &(500..510).collect::<Vec<_>>())
                    .is_err()
            );
            replay.validate_ready_replay().unwrap();
            for token in 0..DECODE {
                for layer in 0..LAYERS {
                    let ids = Zeroizing::new(std::array::from_fn::<_, RANKS, _>(|rank| {
                        if disjoint { token * RANKS + rank } else { rank }
                    }));
                    replay
                        .record(layer, (START + token + 1) as u64, &*ids)
                        .unwrap();
                }
            }
            replay.complete().unwrap();
            assert!(replay.validate_ready_replay().is_err());
            assert!(replay.validated_replay_copy().is_err());
        }
    }

    #[test]
    fn erase_wipes_route_storage_and_invalidates_owner() {
        let mut trace = trained(true).validated_replay_copy().unwrap();
        trace
            .record(0, (START + 1) as u64, &(0..10).collect::<Vec<_>>())
            .unwrap();
        trace.erase();
        assert!(trace.bits.is_empty());
        assert_eq!(*trace.coverage, [0; DECODE]);
        assert!(!trace.replay);
        assert!(trace.complete().is_err());
        assert!(trace.validate_ready_replay().is_err());
    }

    #[test]
    fn missing_layer_and_corrupt_union_cannot_be_copied_or_preloaded() {
        let mut trace = trained(true);
        trace.coverage[DECODE - 1] &= !(1u64 << (LAYERS - 1));
        assert!(trace.complete().is_err());
        assert!(trace.validated_replay_copy().is_err());
        let mut replay = trained(true).validated_replay_copy().unwrap();
        replay.bits[0] = u64::MAX;
        replay.bits[1] = u64::MAX;
        assert!(replay.validate_ready_replay().is_err());
        assert!(replay.union(0).is_err());
        replay.bits.truncate(LAYERS * WORDS - 1);
        assert!(replay.validate_ready_replay().is_err());
        assert!(
            replay
                .record(0, (START + 1) as u64, &(0..10).collect::<Vec<_>>())
                .is_err()
        );
    }

    #[test]
    fn whole_union_plan_protects_needed_residents_until_final_commit() {
        let trace = trained(true);
        let union = trace.union(0).unwrap();
        let mut bank = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let initial = bank.plan(&(0..144).collect::<Vec<_>>()).unwrap();
        bank.commit(initial).unwrap();
        let needed = Zeroizing::new((100..170).collect::<Vec<_>>());
        let plan = bank.plan(&needed).unwrap();
        bank.validate_pending(&plan, &needed).unwrap();
        assert!(
            plan.uploads
                .iter()
                .all(|(expert, slot)| *expert >= 144 && *slot < 100)
        );
        assert!(
            trace.owned_bytes()
                + union.capacity() * std::mem::size_of::<usize>()
                + plan.owned_metadata_bytes().unwrap()
                + 4096
                <= 192 * 1024
        );
        // Partial/failed GPU windows may not publish the whole plan.
        drop(plan);
        for expert in 144..170 {
            assert!(bank.resident_slot(expert).unwrap().is_none());
        }
        let plan = bank.plan(&needed).unwrap();
        bank.commit(plan).unwrap();
        assert!(
            needed
                .iter()
                .all(|&expert| bank.resident_slot(expert).unwrap().is_some())
        );
        assert!(bank.plan(&(0..145).collect::<Vec<_>>()).is_err());
    }
}
