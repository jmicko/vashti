//! One-layer numeric adapter, not a model publication or timing permit.
//! The borrowed original bank pins its complete physical owners through all
//! consumers/final preflight. The lab retains that owner through checked
//! erasure/audit and disposes it before reporting. Failed fills require disposal.
use super::HOST_BRIDGE_LIMIT;
use crate::{
    bank_slots::{BankSlots, Plan, StagedSlots, routed_capacity},
    cpu_gpu_expert_schedule::{RANKS, Split},
    expert_banks::{BankInstance, ExpertBanks},
};
use kuno_text::{Error, Result};
use zeroize::Zeroizing;

pub(super) const METADATA_LIMIT: usize = HOST_BRIDGE_LIMIT / 4;
fn rejected() -> Error {
    Error::InvalidFormat("staged CPU/GPU selection rejected")
}

// No Debug, mutable plan access, completion token or source publication API.
// Keeping both independently canonical plans proves actual CPU miss ownership
// and unchanged resident-hit slots before every fill/launch/final preflight.
struct State {
    instance: BankInstance,
    ids: Zeroizing<[usize; RANKS]>,
    cpu_cap: Zeroizing<usize>,
    staged: StagedSlots,
    all: Plan,
    gpu: Plan,
    split: Split,
}
impl State {
    fn prepare(
        instance: BankInstance,
        source: &BankSlots,
        ids: &[usize; RANKS],
        cpu_cap: usize,
    ) -> Result<Self> {
        if source.experts() != 512 || !routed_capacity(source.capacity()) || cpu_cap > RANKS {
            return Err(rejected());
        }
        let bound = BankSlots::bounded_original_metadata_bytes(source.capacity(), 512)
            .map_err(|_| rejected())?
            .checked_add(
                StagedSlots::bounded_native_metadata_bytes(source.capacity())
                    .map_err(|_| rejected())?
                    .checked_mul(2)
                    .ok_or_else(rejected)?,
            )
            .and_then(|n| n.checked_add(std::mem::size_of::<Selection<'_>>()))
            .ok_or_else(rejected)?;
        if bound > METADATA_LIMIT {
            return Err(rejected());
        }
        let mut staged = StagedSlots::begin(source).map_err(|_| rejected())?;
        let all = staged.plan(source, ids).map_err(|_| rejected())?;
        staged
            .validate_pending(source, &all, ids)
            .map_err(|_| rejected())?;
        let mut misses = Zeroizing::new([0usize; RANKS]);
        for (i, &(id, _)) in all.uploads.iter().enumerate() {
            misses[i] = id;
        }
        let split = Split::from_actual_misses(ids, &misses[..all.uploads.len()], cpu_cap)?;
        // Protect every real GPU hit before choosing any GPU miss victim.
        // CPU-assigned misses never enter this canonical fill plan.
        let gpu = staged.plan(source, split.gpu()).map_err(|_| rejected())?;
        let state = Self {
            instance,
            ids: Zeroizing::new(*ids),
            cpu_cap: Zeroizing::new(cpu_cap),
            staged,
            all,
            gpu,
            split,
        };
        state.validate(instance, source)?;
        state.metadata_bytes(source)?;
        Ok(state)
    }
    fn validate(&self, instance: BankInstance, source: &BankSlots) -> Result<()> {
        if instance != self.instance {
            return Err(rejected());
        }
        self.staged
            .validate_pending(source, &self.all, &*self.ids)
            .map_err(|_| rejected())?;
        self.staged
            .validate_pending(source, &self.gpu, self.split.gpu())
            .map_err(|_| rejected())?;
        let mut misses = Zeroizing::new([0usize; RANKS]);
        for (i, &(id, _)) in self.all.uploads.iter().enumerate() {
            misses[i] = id;
        }
        let expected =
            Split::from_actual_misses(&self.ids, &misses[..self.all.uploads.len()], *self.cpu_cap)?;
        if expected.cpu() != self.split.cpu()
            || expected.gpu() != self.split.gpu()
            || expected.map() != self.split.map()
            || self.gpu.uploads.len() != self.all.uploads.len() - self.split.cpu().len()
        {
            return Err(rejected());
        }
        for (rank, &id) in self.ids.iter().enumerate() {
            let miss = misses[..self.all.uploads.len()].contains(&id);
            let owner = self.split.map()[rank * 2];
            let index = self.split.map()[rank * 2 + 1] as usize;
            if owner == 1 {
                if !miss
                    || self.split.cpu().get(index) != Some(&id)
                    || self.gpu.uploads.iter().any(|&(expert, _)| expert == id)
                {
                    return Err(rejected());
                }
            } else if owner == 0 {
                if self.split.gpu().get(index) != Some(&id) {
                    return Err(rejected());
                }
                let slot = *self.gpu.slots.get(index).ok_or_else(rejected)?;
                if miss {
                    if !self.gpu.uploads.contains(&(id, slot)) {
                        return Err(rejected());
                    }
                } else if self.all.slots[rank] != slot {
                    return Err(rejected());
                }
            } else {
                return Err(rejected());
            }
        }
        self.metadata_bytes(source)?;
        Ok(())
    }
    fn metadata_bytes(&self, source: &BankSlots) -> Result<usize> {
        // Conservative double accounting of the shared staged state is
        // intentional; both Plan.next/canonical vectors are actually retained.
        let bytes = source
            .original_metadata_bytes()
            .map_err(|_| rejected())?
            .checked_add(
                self.staged
                    .retained_native_metadata_bytes(&self.all)
                    .map_err(|_| rejected())?,
            )
            .and_then(|n| {
                self.staged
                    .retained_native_metadata_bytes(&self.gpu)
                    .ok()
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<Selection<'_>>()))
            .ok_or_else(rejected)?;
        if bytes > METADATA_LIMIT {
            return Err(rejected());
        }
        Ok(bytes)
    }
}

/// Borrowing this exact owner pins original bank handles, instance and source.
/// This selector cannot advance or publish either original or staged ledgers.
pub(super) struct Selection<'bank> {
    bank: &'bank ExpertBanks,
    state: State,
}
impl<'bank> Selection<'bank> {
    pub(super) fn prepare(
        bank: &'bank ExpertBanks,
        ids: &[usize; RANKS],
        cpu_miss_cap: usize,
    ) -> Result<Self> {
        Ok(Self {
            bank,
            state: State::prepare(bank.instance(), &bank.slots, ids, cpu_miss_cap)?,
        })
    }
    pub(super) fn validate(&self, bank: &ExpertBanks) -> Result<()> {
        if !std::ptr::eq(self.bank, bank) {
            return Err(rejected());
        }
        self.state.validate(bank.instance(), &bank.slots)
    }
    pub(super) fn split(&self) -> &Split {
        &self.state.split
    }
    pub(super) fn gpu_plan(&self) -> &Plan {
        &self.state.gpu
    }
    pub(super) fn misses(&self) -> usize {
        self.state.all.uploads.len()
    }
    pub(super) fn is_miss(&self, rank: usize) -> Result<bool> {
        let id = self.state.ids.get(rank).ok_or_else(rejected)?;
        Ok(self
            .state
            .all
            .uploads
            .iter()
            .any(|&(expert, _)| expert == *id))
    }
    pub(super) fn metadata_bytes(&self) -> Result<usize> {
        self.state.metadata_bytes(&self.bank.slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_banks::fixture_bank_instance;
    fn primed() -> BankSlots {
        let mut source = BankSlots::new(16, 512).unwrap();
        let p = source
            .plan(&[0, 481, 2, 483, 4, 485, 6, 487, 8, 489])
            .unwrap();
        source.commit(p).unwrap();
        source
    }
    fn source_slots(source: &mut BankSlots) -> Vec<usize> {
        let p = source
            .plan(&[0, 481, 2, 483, 4, 485, 6, 487, 8, 489])
            .unwrap();
        assert!(p.uploads.is_empty());
        p.slots.clone()
    }
    #[test]
    fn actual_noncontiguous_misses_preserve_all_ranks_and_protected_hit_slots() {
        let ids = std::array::from_fn(|r| r);
        for cap in [0, 1, 5, 10] {
            let instance = fixture_bank_instance();
            let mut source = primed();
            let before = source_slots(&mut source);
            let origin_proof = source
                .plan(&[0, 481, 2, 483, 4, 485, 6, 487, 8, 489])
                .unwrap();
            let selection = State::prepare(instance, &source, &ids, cap).unwrap();
            let cpu = &[1, 3, 5, 7, 9][..cap.min(5)];
            assert_eq!(selection.split.cpu(), cpu);
            assert_eq!(selection.all.uploads.len(), 5);
            assert_eq!(selection.gpu.uploads.len(), 5 - cpu.len());
            for rank in 0..RANKS {
                let owner = selection.split.map()[rank * 2];
                let index = selection.split.map()[rank * 2 + 1] as usize;
                let list = if owner == 1 {
                    selection.split.cpu()
                } else {
                    selection.split.gpu()
                };
                assert_eq!(list[index], ids[rank]);
                assert_eq!(owner == 1, cpu.contains(&ids[rank]));
                if rank % 2 == 0 {
                    assert_eq!(selection.gpu.slots[index], before[rank]);
                }
            }
            selection.validate(instance, &source).unwrap();
            drop(selection);
            assert_eq!(source_slots(&mut source), before); // no LRU publication
            source
                .validate_pending(&origin_proof, &[0, 481, 2, 483, 4, 485, 6, 487, 8, 489])
                .unwrap();
        }
    }
    #[test]
    fn endpoints_skip_empty_owner_without_fake_gpu_routes() {
        let ids = std::array::from_fn(|r| r);
        let instance = fixture_bank_instance();
        let mut source = BankSlots::new(16, 512).unwrap();
        let all_cpu = State::prepare(instance, &source, &ids, 10).unwrap();
        assert_eq!(all_cpu.split.cpu(), ids);
        assert!(all_cpu.split.gpu().is_empty());
        assert!(all_cpu.gpu.slots.is_empty());
        assert!(all_cpu.gpu.uploads.is_empty());
        assert_eq!(
            all_cpu.split.map(),
            &[1, 0, 1, 1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8, 1, 9]
        );
        drop(all_cpu);
        let p = source.plan(&ids).unwrap();
        assert_eq!(p.uploads.len(), 10);
        source.commit(p).unwrap();
        let all_hit = State::prepare(instance, &source, &ids, 10).unwrap();
        assert!(all_hit.split.cpu().is_empty());
        assert_eq!(all_hit.split.gpu(), ids);
        assert!(all_hit.gpu.uploads.is_empty());
        all_hit.validate(instance, &source).unwrap();
    }
    #[test]
    fn original_identity_revision_and_canonical_plan_tampering_reject() {
        let ids = std::array::from_fn(|r| r);
        let instance = fixture_bank_instance();
        let mut source = primed();
        let mut selected = State::prepare(instance, &source, &ids, 1).unwrap();
        assert!(selected.validate(fixture_bank_instance(), &source).is_err());
        assert!(selected.validate(instance, &primed()).is_err());
        selected.gpu.slots.swap(0, 1);
        assert!(selected.validate(instance, &source).is_err());
        drop(selected);
        let mut selected = State::prepare(instance, &source, &ids, 1).unwrap();
        selected.all.uploads.swap(0, 1);
        assert!(selected.validate(instance, &source).is_err());
        drop(selected);
        let mut selected = State::prepare(instance, &source, &ids, 1).unwrap();
        selected.split = Split::new(&ids, 5, 1).unwrap();
        assert!(selected.validate(instance, &source).is_err());
        drop(selected);
        let selected = State::prepare(instance, &source, &ids, 1).unwrap();
        let changed = source.plan(&[400]).unwrap();
        source.commit(changed).unwrap();
        assert!(selected.validate(instance, &source).is_err());
    }
    #[test]
    fn full_actual_bank_protects_old_hits_before_any_gpu_miss_victim() {
        let ids = std::array::from_fn(|r| r);
        let mut source = primed();
        let fill = source.plan(&[490, 491, 492, 493, 494, 495]).unwrap();
        source.commit(fill).unwrap();
        let before = source.plan(&[0, 2, 4, 6, 8]).unwrap();
        assert!(before.uploads.is_empty());
        let selected = State::prepare(fixture_bank_instance(), &source, &ids, 1).unwrap();
        assert_eq!(selected.gpu.uploads.len(), 4);
        for (rank, &slot) in before.slots.iter().enumerate() {
            let original_rank = rank * 2;
            let index = selected.split.map()[original_rank * 2 + 1] as usize;
            assert_eq!(selected.gpu.slots[index], slot);
            assert!(
                selected
                    .gpu
                    .uploads
                    .iter()
                    .all(|&(_, victim)| victim != slot)
            );
        }
        assert!(selected.gpu.uploads.iter().all(|&(expert, _)| expert != 1));
        drop(selected);
        source.validate_pending(&before, &[0, 2, 4, 6, 8]).unwrap();
    }
    #[test]
    fn actual_retained_capacities_and_invalid_selection_are_bounded() {
        let ids = std::array::from_fn(|r| r);
        for capacity in [10, 16, 72, 96] {
            let instance = fixture_bank_instance();
            let source = BankSlots::new(capacity, 512).unwrap();
            let selected = State::prepare(instance, &source, &ids, 5).unwrap();
            assert!(selected.metadata_bytes(&source).unwrap() < METADATA_LIMIT);
        }
        let source = primed();
        assert!(State::prepare(fixture_bank_instance(), &source, &ids, 11).is_err());
        let mut duplicate = ids;
        duplicate[9] = 0;
        assert!(State::prepare(fixture_bank_instance(), &source, &duplicate, 1).is_err());
        let mut outside = ids;
        outside[9] = 512;
        assert!(State::prepare(fixture_bank_instance(), &source, &outside, 1).is_err());
    }
}
