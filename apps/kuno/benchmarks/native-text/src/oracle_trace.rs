//! Exact route union for one compiled-public laboratory replay.
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

/// Fixed public fixtures only. START is the completed prefill cursor, so Q3's
/// token0 prefill ends at cursor1; the selected fixed window then covers either
/// cursors2..23 (Unsloth prefix) or cursors2..32 (DASLab complete reference).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Fixture {
    Q4Rainwater,
    Q3Prefix23,
    Daslab32,
}
impl Fixture {
    pub(crate) fn start(self) -> usize {
        match self {
            Self::Q4Rainwater => START,
            Self::Q3Prefix23 | Self::Daslab32 => 1,
        }
    }
    pub(crate) fn decode(self) -> usize {
        match self {
            Self::Q4Rainwater => DECODE,
            Self::Q3Prefix23 => 22,
            Self::Daslab32 => 31,
        }
    }
    pub(crate) fn prefix(self) -> usize {
        self.decode() + 1
    }
    pub(crate) fn context(self) -> usize {
        match self {
            Self::Q4Rainwater => CONTEXT,
            Self::Q3Prefix23 => 24,
            Self::Daslab32 => 33,
        }
    }
    fn max_union(self) -> usize {
        (RANKS * self.decode()).min(144)
    }
}

enum Coverage {
    Q4(Zeroizing<[u64; DECODE]>),
    Q3(Zeroizing<[u64; 22]>),
    Daslab(Zeroizing<[u64; 31]>),
}
impl Coverage {
    fn new(fixture: Fixture) -> Self {
        match fixture {
            Fixture::Q4Rainwater => Self::Q4(Zeroizing::new([0; DECODE])),
            Fixture::Q3Prefix23 => Self::Q3(Zeroizing::new([0; 22])),
            Fixture::Daslab32 => Self::Daslab(Zeroizing::new([0; 31])),
        }
    }
    fn fixture(&self) -> Fixture {
        match self {
            Self::Q4(_) => Fixture::Q4Rainwater,
            Self::Q3(_) => Fixture::Q3Prefix23,
            Self::Daslab(_) => Fixture::Daslab32,
        }
    }
}
impl std::ops::Deref for Coverage {
    type Target = [u64];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Q4(masks) => masks.as_slice(),
            Self::Q3(masks) => masks.as_slice(),
            Self::Daslab(masks) => masks.as_slice(),
        }
    }
}
impl std::ops::DerefMut for Coverage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Q4(masks) => masks.as_mut_slice(),
            Self::Q3(masks) => masks.as_mut_slice(),
            Self::Daslab(masks) => masks.as_mut_slice(),
        }
    }
}
impl Zeroize for Coverage {
    fn zeroize(&mut self) {
        match self {
            Self::Q4(masks) => masks.zeroize(),
            Self::Q3(masks) => masks.zeroize(),
            Self::Daslab(masks) => masks.zeroize(),
        }
    }
}

fn rejected() -> Error {
    Error::InvalidFormat("public oracle trace rejected")
}

pub(crate) struct Trace {
    bits: Zeroizing<Vec<u64>>,
    coverage: Coverage,
    replay: bool,
}

impl Trace {
    pub(crate) const HOST: usize = crate::oracle_trace::HOST;

    pub(crate) fn training() -> Result<Self> {
        Self::training_for(Fixture::Q4Rainwater)
    }

    pub(crate) fn q3_prefix23_training() -> Result<Self> {
        Self::training_for(Fixture::Q3Prefix23)
    }

    pub(crate) fn daslab32_training() -> Result<Self> {
        Self::training_for(Fixture::Daslab32)
    }

    fn training_for(fixture: Fixture) -> Result<Self> {
        let mut bits = Zeroizing::new(Vec::new());
        bits.try_reserve_exact(LAYERS * WORDS)
            .map_err(|_| Error::LimitExceeded("public oracle trace host"))?;
        bits.resize(LAYERS * WORDS, 0);
        let trace = Self {
            bits,
            coverage: Coverage::new(fixture),
            replay: false,
        };
        trace.validate_shape()?;
        Ok(trace)
    }

    pub(crate) fn owned_bytes(&self) -> usize {
        self.bits.capacity() * std::mem::size_of::<u64>() + std::mem::size_of::<Self>()
    }

    pub(crate) fn fixture(&self) -> Fixture {
        self.coverage.fixture()
    }
    pub(crate) fn start(&self) -> usize {
        self.fixture().start()
    }
    pub(crate) fn decode(&self) -> usize {
        self.fixture().decode()
    }
    pub(crate) fn prefix(&self) -> usize {
        self.fixture().prefix()
    }
    pub(crate) fn context(&self) -> usize {
        self.fixture().context()
    }

    fn validate_shape(&self) -> Result<()> {
        if self.bits.len() != LAYERS * WORDS
            || self.coverage.len() != self.decode()
            || self.prefix() != self.decode() + 1
            || self.start() + self.decode() >= self.context()
            || self.owned_bytes() > HOST
        {
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
            !(RANKS..=self.fixture().max_union())
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
        let mut trace = Self::training_for(self.fixture())?;
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
        if selected.len() != count || selected.len() > self.fixture().max_union() {
            return Err(rejected());
        }
        Ok(selected)
    }

    pub(crate) fn record(&mut self, layer: usize, position: u64, ids: &[usize]) -> Result<()> {
        self.validate_shape()?;
        let token = position
            .checked_sub((self.start() + 1) as u64)
            .and_then(|p| usize::try_from(p).ok())
            .filter(|&p| p < self.decode())
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
        let max_union = self.fixture().max_union();
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
            let union: usize = row
                .iter()
                .zip(current.iter())
                .map(|(allowed, selected)| (allowed | selected).count_ones() as usize)
                .sum();
            if union > max_union {
                return Err(rejected());
            }
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

    fn q3_trained_full_capacity() -> Trace {
        let mut trace = Trace::q3_prefix23_training().unwrap();
        for token in 0..trace.decode() {
            let ids = std::array::from_fn::<_, RANKS, _>(|rank| match token {
                0..=13 => token * RANKS + rank,
                14 if rank < 4 => 140 + rank,
                _ => rank,
            });
            for layer in 0..LAYERS {
                trace.record(layer, (2 + token) as u64, &ids).unwrap();
            }
        }
        trace
    }

    #[test]
    fn q3_fixed_geometry_covers_only_actual22_decode_cursors_and_wipes_every_mask() {
        let mut trace = Trace::q3_prefix23_training().unwrap();
        assert!(trace.fixture() == Fixture::Q3Prefix23);
        assert_eq!(
            (
                trace.start(),
                trace.decode(),
                trace.prefix(),
                trace.context()
            ),
            (1, 22, 23, 24)
        );
        assert_eq!(trace.coverage.len(), 22);
        assert_eq!(trace.bits.len(), 384);
        assert!(trace.owned_bytes() <= HOST);
        let ids = std::array::from_fn::<_, RANKS, _>(|rank| rank);
        for position in [0, 1, 24, u64::MAX] {
            assert!(trace.record(0, position, &ids).is_err());
        }
        assert_eq!(trace.counts(), vec![0; LAYERS]);
        trace.record(0, 2, &ids).unwrap();
        assert!(trace.record(0, 2, &ids).is_err());
        assert!(trace.complete().is_err());
        trace = q3_trained_full_capacity();
        trace.complete().unwrap();
        assert_eq!(trace.counts(), vec![144; LAYERS]);
        let mut replay = trace.validated_replay_copy().unwrap();
        assert_eq!(
            (
                replay.start(),
                replay.decode(),
                replay.prefix(),
                replay.context()
            ),
            (1, 22, 23, 24)
        );
        replay.validate_ready_replay().unwrap();
        assert!(
            replay
                .record(0, 2, &(500..510).collect::<Vec<_>>())
                .is_err()
        );
        replay.validate_ready_replay().unwrap();
        for token in 0..22 {
            for layer in 0..LAYERS {
                replay.record(layer, (2 + token) as u64, &ids).unwrap();
            }
        }
        replay.complete().unwrap();
        replay.erase();
        assert!(replay.bits.is_empty());
        assert_eq!(*replay.coverage, [0; 22]);
        assert!(!replay.is_replay());
        assert!(replay.complete().is_err());
    }

    #[test]
    fn q3_overfull_union_rejection_does_not_publish_ids_or_coverage() {
        let mut trace = Trace::q3_prefix23_training().unwrap();
        for token in 0..14 {
            trace
                .record(
                    0,
                    (token + 2) as u64,
                    &(token * 10..(token + 1) * 10).collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let original = trace.bits.to_vec();
        assert_eq!(trace.counts()[0], 140);
        assert!(
            trace
                .record(0, 16, &(140..150).collect::<Vec<_>>())
                .is_err()
        );
        assert_eq!(*trace.bits, original);
        assert_eq!(trace.coverage[14], 0);
        assert!(trace.complete().is_err());
        let mut complete = q3_trained_full_capacity();
        complete.coverage[21] &= !(1u64 << 47);
        assert!(complete.complete().is_err());
        assert!(complete.validated_replay_copy().is_err());
    }

    #[test]
    fn q3_full144_union_keeps_one_bounded_plan_until_all_windows_complete() {
        let trace = q3_trained_full_capacity();
        let union = trace.union(0).unwrap();
        assert_eq!(union.len(), 144);
        let mut bank = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let initial = bank.plan(&(200..344).collect::<Vec<_>>()).unwrap();
        bank.commit(initial).unwrap();
        let plan = bank.plan(&union).unwrap();
        bank.validate_pending(&plan, &union).unwrap();
        assert_eq!(plan.uploads.len(), 144);
        assert_eq!(plan.uploads.chunks(10).count(), 15);
        for window in plan.uploads.chunks(10) {
            assert!(window.len() <= 10);
            bank.validate_pending(&plan, &union).unwrap();
            assert!(
                union
                    .iter()
                    .all(|&id| bank.resident_slot(id).unwrap().is_none())
            );
        }
        assert!(
            trace.owned_bytes()
                + union.capacity() * std::mem::size_of::<usize>()
                + plan.owned_metadata_bytes().unwrap()
                + 4096
                <= 192 * 1024
        );
        bank.commit(plan).unwrap();
        assert!(
            union
                .iter()
                .all(|&id| bank.resident_slot(id).unwrap().is_some())
        );
        assert!(bank.plan(&(0..145).collect::<Vec<_>>()).is_err());
    }

    fn daslab_trained_full_capacity() -> Trace {
        let mut trace = Trace::daslab32_training().unwrap();
        for token in 0..trace.decode() {
            let ids = std::array::from_fn::<_, RANKS, _>(|rank| match token {
                0..=13 => token * RANKS + rank,
                14 if rank < 4 => 140 + rank,
                _ => rank,
            });
            for layer in 0..LAYERS {
                trace.record(layer, (2 + token) as u64, &ids).unwrap();
            }
        }
        trace
    }

    #[test]
    fn daslab_fixed31_masks_cover_all32_outputs_and_replay_wipes_every_owner() {
        let mut trace = Trace::daslab32_training().unwrap();
        assert!(trace.fixture() == Fixture::Daslab32);
        assert_eq!(
            (
                trace.start(),
                trace.decode(),
                trace.prefix(),
                trace.context()
            ),
            (1, 31, 32, 33)
        );
        assert_eq!(trace.coverage.len(), 31);
        assert_eq!(trace.bits.len(), 384);
        assert!(trace.owned_bytes() <= HOST);
        let ids = std::array::from_fn::<_, RANKS, _>(|rank| rank);
        for position in [0, 1, 33, u64::MAX] {
            assert!(trace.record(0, position, &ids).is_err());
        }
        assert_eq!(trace.counts(), vec![0; LAYERS]);
        trace.record(0, 32, &ids).unwrap();
        assert!(trace.record(0, 32, &ids).is_err());
        assert!(trace.validated_replay_copy().is_err());

        trace = daslab_trained_full_capacity();
        trace.complete().unwrap();
        assert!(trace.coverage.iter().all(|&mask| mask == ALL_LAYERS));
        assert_eq!(trace.counts(), vec![144; LAYERS]);
        let mut replay = trace.validated_replay_copy().unwrap();
        assert!(replay.fixture() == Fixture::Daslab32);
        assert_eq!(replay.coverage.len(), 31);
        replay.validate_ready_replay().unwrap();
        assert!(replay.complete().is_err());
        assert!(replay.validated_replay_copy().is_err());
        assert!(
            replay
                .record(0, 2, &(140..150).collect::<Vec<_>>())
                .is_err()
        );
        replay.validate_ready_replay().unwrap();
        for token in 0..31 {
            for layer in 0..LAYERS {
                replay.record(layer, (2 + token) as u64, &ids).unwrap();
            }
        }
        replay.complete().unwrap();
        assert!(replay.validate_ready_replay().is_err());
        for owner in [&mut trace, &mut replay] {
            owner.erase();
            assert!(owner.bits.is_empty());
            assert_eq!(*owner.coverage, [0; 31]);
            assert!(!owner.is_replay());
            assert!(owner.complete().is_err());
        }
    }

    #[test]
    fn daslab_union145_and_missing_final_layer_fail_without_partial_publication() {
        let mut trace = Trace::daslab32_training().unwrap();
        for token in 0..14 {
            trace
                .record(
                    0,
                    (token + 2) as u64,
                    &(token * 10..(token + 1) * 10).collect::<Vec<_>>(),
                )
                .unwrap();
        }
        let original = trace.bits.to_vec();
        let too_many = [140, 141, 142, 143, 144, 0, 1, 2, 3, 4];
        assert!(trace.record(0, 16, &too_many).is_err());
        assert_eq!(*trace.bits, original);
        assert_eq!(trace.coverage[14], 0);
        let fits = [140, 141, 142, 143, 0, 1, 2, 3, 4, 5];
        trace.record(0, 16, &fits).unwrap();
        assert_eq!(trace.counts()[0], 144);
        let original = trace.bits.to_vec();
        assert!(
            trace
                .record(0, 17, &[144, 0, 1, 2, 3, 4, 5, 6, 7, 8])
                .is_err()
        );
        assert_eq!(*trace.bits, original);
        assert_eq!(trace.coverage[15], 0);
        let mut complete = daslab_trained_full_capacity();
        complete.coverage[30] &= !(1u64 << 47);
        assert!(complete.complete().is_err());
        assert!(complete.validated_replay_copy().is_err());
    }

    #[test]
    fn daslab_full_union_plan_remains_bounded_and_unpublished_until_complete() {
        let trace = daslab_trained_full_capacity();
        let union = trace.union(47).unwrap();
        let mut bank = crate::bank_slots::BankSlots::new(144, 512).unwrap();
        let initial = bank.plan(&(200..344).collect::<Vec<_>>()).unwrap();
        bank.commit(initial).unwrap();
        let plan = bank.plan(&union).unwrap();
        bank.validate_pending(&plan, &union).unwrap();
        assert_eq!(plan.uploads.len(), 144);
        assert!(plan.uploads.chunks(10).all(|window| window.len() <= 10));
        assert!(
            trace.owned_bytes()
                + union.capacity() * std::mem::size_of::<usize>()
                + plan.owned_metadata_bytes().unwrap()
                + 4096
                <= 192 * 1024
        );
        drop(plan);
        assert!(
            union
                .iter()
                .all(|&id| bank.resident_slot(id).unwrap().is_none())
        );
        let complete = bank.plan(&union).unwrap();
        bank.commit(complete).unwrap();
        assert!(
            union
                .iter()
                .all(|&id| bank.resident_slot(id).unwrap().is_some())
        );
    }
}
