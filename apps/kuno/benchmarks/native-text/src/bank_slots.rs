//! Host-only, bounded expert-bank residency planning.
//!
//! All requested experts are protected before choosing any victim. Planning is
//! transactional: slots/LRU change only on commit, after the caller finishes all
//! uploads. No tensor bytes, GPU handles, logs or persistent profiles live here.
//! If a failed upload partially overwrites a resident GPU slot, discard/rebuild
//! that physical bank; dropping a plan cannot restore its old GPU bytes.

use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_BANK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankError {
    Capacity,
    ExpertCount,
    SelectionTooLarge,
    ExpertOutOfRange,
    DuplicateExpert,
    CounterOverflow,
    NoVictim,
    StalePlan,
    ModifiedPlan,
}

impl std::fmt::Display for BankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "expert bank capacity must be between one and 64",
            Self::ExpertCount => "expert bank count must be between capacity and 512",
            Self::SelectionTooLarge => "expert selection exceeds bank capacity",
            Self::ExpertOutOfRange => "expert selection is out of range",
            Self::DuplicateExpert => "expert selection contains duplicate IDs",
            Self::CounterOverflow => "expert bank counter overflow",
            Self::NoVictim => "no unprotected expert bank slot is available",
            Self::StalePlan => "expert bank plan is stale or belongs to another bank",
            Self::ModifiedPlan => "expert bank plan was modified after construction",
        })
    }
}

impl std::error::Error for BankError {}

pub type Result<T> = std::result::Result<T, BankError>;

#[derive(Clone, Copy, PartialEq, Eq)]
struct Resident {
    expert: usize,
    last_used: u64,
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    slots: Vec<Option<Resident>>,
    by_expert: Vec<Option<usize>>,
    tick: u64,
    revision: u64,
}

/// Transient model-local residency and LRU; Debug is omitted to avoid dumping
/// request-derived expert histories. Production's 512 experts/10..64 slots fit
/// these bounds; smaller banks are supported for independent fixture tests.
pub struct BankSlots {
    bank_id: u64,
    state: State,
}

/// Ordered slot IDs and required uploads for one selected set. Uploads contain
/// only misses, in the request's order. Public fields are checked on commit.
/// Dropping a plan leaves committed host state unchanged.
pub struct Plan {
    pub slots: Vec<usize>,
    pub uploads: Vec<(usize, usize)>,
    bank_id: u64,
    base_revision: u64,
    canonical_slots: Vec<usize>,
    canonical_uploads: Vec<(usize, usize)>,
    next: State,
}

impl BankSlots {
    pub fn new(capacity: usize, experts: usize) -> Result<Self> {
        if !(1..=64).contains(&capacity) {
            return Err(BankError::Capacity);
        }
        if experts < capacity || experts > 512 {
            return Err(BankError::ExpertCount);
        }
        let bank_id = NEXT_BANK_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| BankError::CounterOverflow)?;
        Ok(Self {
            bank_id,
            state: State {
                slots: vec![None; capacity],
                by_expert: vec![None; experts],
                tick: 0,
                revision: 0,
            },
        })
    }

    pub fn capacity(&self) -> usize {
        self.state.slots.len()
    }

    pub fn experts(&self) -> usize {
        self.state.by_expert.len()
    }

    /// Construct a plan without mutating committed residency or recency.
    pub fn plan(&mut self, selected: &[usize]) -> Result<Plan> {
        if selected.len() > self.capacity() {
            return Err(BankError::SelectionTooLarge);
        }
        let mut protected = vec![false; self.experts()];
        for &expert in selected {
            let seen = protected
                .get_mut(expert)
                .ok_or(BankError::ExpertOutOfRange)?;
            if *seen {
                return Err(BankError::DuplicateExpert);
            }
            *seen = true;
        }
        let delta = u64::try_from(selected.len()).map_err(|_| BankError::CounterOverflow)?;
        let tick = self
            .state
            .tick
            .checked_add(delta)
            .ok_or(BankError::CounterOverflow)?;
        let revision = if selected.is_empty() {
            self.state.revision
        } else {
            self.state
                .revision
                .checked_add(1)
                .ok_or(BankError::CounterOverflow)?
        };
        let mut next = self.state.clone();
        let mut slots = Vec::with_capacity(selected.len());
        let mut uploads = Vec::with_capacity(selected.len());
        for (index, &expert) in selected.iter().enumerate() {
            let slot = if let Some(slot) = next.by_expert[expert] {
                slot
            } else {
                let slot = next
                    .slots
                    .iter()
                    .position(Option::is_none)
                    .or_else(|| {
                        next.slots
                            .iter()
                            .enumerate()
                            .filter_map(|(slot, resident)| {
                                resident
                                    .as_ref()
                                    .filter(|resident| !protected[resident.expert])
                                    .map(|resident| (resident.last_used, slot))
                            })
                            .min()
                            .map(|(_, slot)| slot)
                    })
                    .ok_or(BankError::NoVictim)?;
                if let Some(victim) = next.slots[slot] {
                    next.by_expert[victim.expert] = None;
                }
                next.by_expert[expert] = Some(slot);
                uploads.push((expert, slot));
                slot
            };
            let last_used = self
                .state
                .tick
                .checked_add(index as u64)
                .and_then(|value| value.checked_add(1))
                .ok_or(BankError::CounterOverflow)?;
            next.slots[slot] = Some(Resident { expert, last_used });
            slots.push(slot);
        }
        next.tick = tick;
        next.revision = revision;
        Ok(Plan {
            canonical_slots: slots.clone(),
            canonical_uploads: uploads.clone(),
            slots,
            uploads,
            bank_id: self.bank_id,
            base_revision: self.state.revision,
            next,
        })
    }

    /// Commit only after all planned uploads succeed. Stale/cross-bank or altered
    /// plans fail without changing residency. GPU upload rollback is external.
    pub fn commit(&mut self, plan: Plan) -> Result<()> {
        if plan.bank_id != self.bank_id || plan.base_revision != self.state.revision {
            return Err(BankError::StalePlan);
        }
        if plan.slots != plan.canonical_slots || plan.uploads != plan.canonical_uploads {
            return Err(BankError::ModifiedPlan);
        }
        self.state = plan.next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn commit(bank: &mut BankSlots, experts: &[usize]) -> (Vec<usize>, Vec<(usize, usize)>) {
        let plan = bank.plan(experts).unwrap();
        let result = (plan.slots.clone(), plan.uploads.clone());
        bank.commit(plan).unwrap();
        result
    }

    #[test]
    fn repeated_hits_preserve_request_order_and_need_no_upload() {
        let mut bank = BankSlots::new(3, 8).unwrap();
        assert_eq!(
            commit(&mut bank, &[5, 2, 7]),
            (vec![0, 1, 2], vec![(5, 0), (2, 1), (7, 2)])
        );
        assert_eq!(commit(&mut bank, &[7, 5, 2]), (vec![2, 0, 1], vec![]));
        assert_eq!(bank.state.tick, 6);
        assert_eq!(bank.state.revision, 2);
    }

    #[test]
    fn missing_first_expert_never_evicts_an_older_requested_later_hit() {
        let mut bank = BankSlots::new(2, 5).unwrap();
        commit(&mut bank, &[1, 2]); // 1 is oldest; 2 is the only legal victim.
        let before = bank.state.clone();
        let plan = bank.plan(&[3, 1]).unwrap();
        assert_eq!(plan.slots, [1, 0]);
        assert_eq!(plan.uploads, [(3, 1)]);
        assert!(bank.state == before);
        bank.commit(plan).unwrap();
        assert_eq!(bank.state.by_expert[1], Some(0));
        assert_eq!(bank.state.by_expert[2], None);
    }

    #[test]
    fn all_requested_hits_are_protected_before_multiple_missing_victims() {
        let mut bank = BankSlots::new(4, 10).unwrap();
        commit(&mut bank, &[0, 1, 2, 3]);
        assert_eq!(
            commit(&mut bank, &[5, 0, 6, 2]),
            (vec![1, 0, 3, 2], vec![(5, 1), (6, 3)])
        );
        assert_eq!(commit(&mut bank, &[2, 6, 0, 5]).1, []);
    }

    #[test]
    fn recency_from_hits_controls_victims_and_empty_requests_do_not_touch_ticks() {
        let mut bank = BankSlots::new(3, 6).unwrap();
        commit(&mut bank, &[0, 1, 2]);
        commit(&mut bank, &[0]);
        assert_eq!(commit(&mut bank, &[4]).1, [(4, 1)]);
        let before = bank.state.clone();
        assert_eq!(commit(&mut bank, &[]), (vec![], vec![]));
        assert!(bank.state == before);
    }

    #[test]
    fn dropping_or_rejecting_a_plan_does_not_mutate_committed_state() {
        let mut bank = BankSlots::new(2, 4).unwrap();
        commit(&mut bank, &[0, 1]);
        let before = bank.state.clone();
        drop(bank.plan(&[2, 1]).unwrap());
        assert!(bank.state == before);
        for selected in [&[0, 0][..], &[4][..], &[0, 1, 2][..]] {
            assert!(bank.plan(selected).is_err());
            assert!(bank.state == before);
        }
        let mut altered = bank.plan(&[2, 1]).unwrap();
        altered.slots[0] = 1;
        assert_eq!(bank.commit(altered), Err(BankError::ModifiedPlan));
        assert!(bank.state == before);
        let mut altered = bank.plan(&[2, 1]).unwrap();
        altered.uploads[0] = (3, 0);
        assert_eq!(bank.commit(altered), Err(BankError::ModifiedPlan));
        assert!(bank.state == before);
    }

    #[test]
    fn stale_and_cross_bank_plans_cannot_replace_current_residency() {
        let mut bank = BankSlots::new(2, 4).unwrap();
        let first = bank.plan(&[0]).unwrap();
        let stale = bank.plan(&[1]).unwrap();
        bank.commit(first).unwrap();
        let before = bank.state.clone();
        assert_eq!(bank.commit(stale), Err(BankError::StalePlan));
        assert!(bank.state == before);
        let mut other = BankSlots::new(2, 4).unwrap();
        assert_eq!(
            other.commit(bank.plan(&[0]).unwrap()),
            Err(BankError::StalePlan)
        );
        assert!(other.state.slots.iter().all(Option::is_none));
    }

    #[test]
    fn counter_overflow_and_capacity_errors_leave_state_unchanged() {
        assert!(matches!(BankSlots::new(0, 512), Err(BankError::Capacity)));
        assert!(matches!(BankSlots::new(65, 512), Err(BankError::Capacity)));
        assert!(matches!(BankSlots::new(2, 1), Err(BankError::ExpertCount)));
        assert!(matches!(
            BankSlots::new(1, 513),
            Err(BankError::ExpertCount)
        ));
        let mut bank = BankSlots::new(2, 4).unwrap();
        bank.state.tick = u64::MAX;
        let before = bank.state.clone();
        assert!(matches!(bank.plan(&[0]), Err(BankError::CounterOverflow)));
        assert!(bank.state == before);
        bank.state.tick = u64::MAX - 1;
        let before = bank.state.clone();
        assert!(matches!(
            bank.plan(&[0, 1]),
            Err(BankError::CounterOverflow)
        ));
        assert!(bank.state == before);
        bank.state.tick = 0;
        bank.state.revision = u64::MAX;
        let before = bank.state.clone();
        assert!(matches!(bank.plan(&[0]), Err(BankError::CounterOverflow)));
        assert!(bank.state == before);
    }

    #[test]
    fn bounded_full_size_requests_never_relocate_hits_or_overwrite_selected_experts() {
        let mut bank = BankSlots::new(64, 512).unwrap();
        let mut random = 19_u64;
        for _ in 0..1000 {
            let mut selected = Vec::new();
            while selected.len() < 10 {
                random = random.wrapping_mul(6364136223846793005).wrapping_add(1);
                let expert = (random >> 32) as usize % 512;
                if !selected.contains(&expert) {
                    selected.push(expert);
                }
            }
            let before = bank.state.clone();
            let plan = bank.plan(&selected).unwrap();
            assert_eq!(plan.slots.len(), selected.len());
            let unique: std::collections::BTreeSet<_> = plan.slots.iter().copied().collect();
            assert_eq!(unique.len(), selected.len());
            let missing: Vec<_> = selected
                .iter()
                .copied()
                .filter(|&expert| before.by_expert[expert].is_none())
                .collect();
            assert_eq!(
                plan.uploads
                    .iter()
                    .map(|&(expert, _)| expert)
                    .collect::<Vec<_>>(),
                missing
            );
            for (&expert, &slot) in selected.iter().zip(&plan.slots) {
                if let Some(previous) = before.by_expert[expert] {
                    assert_eq!(slot, previous);
                }
                assert_eq!(plan.next.by_expert[expert], Some(slot));
                assert_eq!(plan.next.slots[slot].unwrap().expert, expert);
            }
            bank.commit(plan).unwrap();
        }
    }
}
