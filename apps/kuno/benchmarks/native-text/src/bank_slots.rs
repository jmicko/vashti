//! Host-only, bounded expert-bank residency planning.
//!
//! All requested experts are protected before choosing any victim. Planning is
//! transactional: slots/LRU change only on commit, after the caller finishes all
//! uploads. No tensor bytes, GPU handles, logs or persistent profiles live here.
//! If a failed upload partially overwrites a resident GPU slot, discard/rebuild
//! that physical bank; dropping a plan cannot restore its old GPU bytes.

use std::sync::atomic::{AtomicU64, Ordering};
use zeroize::{Zeroize, Zeroizing};

// 72 original packed Qwen experts fit the same pooled pages as 64.
// The resident-only 96 experiment uses public 64 + 32 pages;128 uses64 +64.
pub const MAX_CAPACITY: usize = 144;
pub const MAX_FLAT_CAPACITY: usize = 72;
pub fn segmented_capacities(capacity: usize) -> Option<[usize; 2]> {
    match capacity {
        96 => Some([64, 32]),
        128 => Some([64, 64]),
        144 => Some([72, 72]),
        _ => None,
    }
}
pub fn segmented_capacity(capacity: usize) -> bool {
    matches!(capacity, 96 | 128 | 144)
}
pub fn routed_capacity(capacity: usize) -> bool {
    (MIN_ROUTED_CAPACITY..=MAX_FLAT_CAPACITY).contains(&capacity) || segmented_capacity(capacity)
}
pub const MIN_ROUTED_CAPACITY: usize = 10;

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
    Allocation,
}

impl std::fmt::Display for BankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "expert bank capacity must be between one and 144",
            Self::ExpertCount => "expert bank count must be between capacity and 512",
            Self::SelectionTooLarge => "expert selection exceeds bank capacity",
            Self::ExpertOutOfRange => "expert selection is out of range",
            Self::DuplicateExpert => "expert selection contains duplicate IDs",
            Self::CounterOverflow => "expert bank counter overflow",
            Self::NoVictim => "no unprotected expert bank slot is available",
            Self::StalePlan => "expert bank plan is stale or belongs to another bank",
            Self::ModifiedPlan => "expert bank plan was modified after construction",
            Self::Allocation => "expert bank bounded metadata allocation failed",
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

impl Zeroize for Resident {
    fn zeroize(&mut self) {
        self.expert.zeroize();
        self.last_used.zeroize();
    }
}

#[derive(Clone, PartialEq, Eq)]
struct State {
    slots: Vec<Option<Resident>>,
    by_expert: Vec<Option<usize>>,
    tick: u64,
    revision: u64,
}

impl State {
    fn wipe(&mut self) {
        // Each vector has fixed, public capacity and never grows after creation.
        // Vec::zeroize covers initialized entries and the entire spare capacity.
        self.slots.zeroize();
        self.by_expert.zeroize();
        self.tick.zeroize();
        self.revision.zeroize();
    }
}
impl Drop for State {
    fn drop(&mut self) {
        self.wipe();
    }
}

/// Transient model-local residency and LRU; Debug is omitted to avoid dumping
/// request-derived expert histories. Production's 512 experts/10..72 or 96 slots fit
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

impl Plan {
    fn wipe_routes(&mut self) {
        self.slots.zeroize();
        self.uploads.zeroize();
        self.canonical_slots.zeroize();
        self.canonical_uploads.zeroize();
        self.bank_id.zeroize();
        self.base_revision.zeroize();
    }
}
impl Drop for Plan {
    fn drop(&mut self) {
        self.wipe_routes();
        // The candidate or swapped-out committed State has its own wiping Drop.
    }
}

/// Pure parsing keeps overrides testable without process-global environment
/// mutation. Smaller fixture-only banks are not valid routed-model overrides.
pub fn configured_capacity(raw: Option<&str>, default: usize) -> Result<usize> {
    let capacity = match raw {
        Some(raw) if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) => {
            raw.parse().map_err(|_| BankError::Capacity)?
        }
        Some(_) => return Err(BankError::Capacity),
        None => default,
    };
    if !routed_capacity(capacity) {
        return Err(BankError::Capacity);
    }
    Ok(capacity)
}

impl BankSlots {
    pub fn new(capacity: usize, experts: usize) -> Result<Self> {
        if !(1..=MAX_CAPACITY).contains(&capacity) {
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
    /// Read committed residency only at the parent's completed boundary.
    /// Caller must not serialize or retain this request-derived selection.
    pub(crate) fn resident_expert(&self, slot: usize) -> Result<Option<usize>> {
        self.state
            .slots
            .get(slot)
            .map(|r| r.map(|r| r.expert))
            .ok_or(BankError::Capacity)
    }

    /// Actual committed state storage, including both Vec capacities and the
    /// inline owner/state/counters. Public capacity only; no route data escapes.
    pub(crate) fn original_metadata_bytes(&self) -> Result<usize> {
        self.state
            .slots
            .capacity()
            .checked_mul(std::mem::size_of::<Option<Resident>>())
            .and_then(|n| {
                n.checked_add(
                    self.state
                        .by_expert
                        .capacity()
                        .checked_mul(std::mem::size_of::<Option<usize>>())?,
                )
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(BankError::CounterOverflow)
    }

    pub(crate) fn bounded_original_metadata_bytes(
        capacity: usize,
        experts: usize,
    ) -> Result<usize> {
        if !(1..=MAX_CAPACITY).contains(&capacity) || experts < capacity || experts > 512 {
            return Err(BankError::Capacity);
        }
        capacity
            .checked_mul(std::mem::size_of::<Option<Resident>>())
            .and_then(|n| n.checked_add(experts.checked_mul(std::mem::size_of::<Option<usize>>())?))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(BankError::CounterOverflow)
    }

    /// Construct a plan without mutating committed residency or recency.
    pub fn plan(&mut self, selected: &[usize]) -> Result<Plan> {
        if selected.len() > self.capacity() {
            return Err(BankError::SelectionTooLarge);
        }
        let mut protected = Zeroizing::new(vec![false; self.experts()]);
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
        let mut slots = Zeroizing::new(Vec::with_capacity(selected.len()));
        let mut uploads = Zeroizing::new(Vec::with_capacity(selected.len()));
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
            canonical_slots: slots.to_vec(),
            canonical_uploads: uploads.to_vec(),
            slots: std::mem::take(&mut *slots),
            uploads: std::mem::take(&mut *uploads),
            bank_id: self.bank_id,
            base_revision: self.state.revision,
            next,
        })
    }

    /// Validate a real pending plan against its complete original protected
    /// selection BEFORE any public fill. Canonical arrays alone cannot prove
    /// that a cached hit belongs to the caller's requested original expert.
    pub fn validate_pending(&self, plan: &Plan, selected: &[usize]) -> Result<()> {
        if plan.bank_id != self.bank_id || plan.base_revision != self.state.revision {
            return Err(BankError::StalePlan);
        }
        if plan.slots != plan.canonical_slots
            || plan.uploads != plan.canonical_uploads
            || selected.len() != plan.slots.len()
            || selected.len() > self.capacity()
            || plan.next.slots.len() != self.capacity()
            || plan.next.by_expert.len() != self.experts()
        {
            return Err(BankError::ModifiedPlan);
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
        if plan.next.tick != tick || plan.next.revision != revision {
            return Err(BankError::ModifiedPlan);
        }
        let mut upload_index = Zeroizing::new(0usize);
        for (i, &expert) in selected.iter().enumerate() {
            if expert >= self.experts() {
                return Err(BankError::ExpertOutOfRange);
            }
            if selected[..i].contains(&expert) {
                return Err(BankError::DuplicateExpert);
            }
            let slot = plan.slots[i];
            let recency = self
                .state
                .tick
                .checked_add(i as u64)
                .and_then(|n| n.checked_add(1))
                .ok_or(BankError::CounterOverflow)?;
            if slot >= self.capacity()
                || plan.next.by_expert[expert] != Some(slot)
                || plan.next.slots[slot]
                    .is_none_or(|r| r.expert != expert || r.last_used != recency)
            {
                return Err(BankError::ModifiedPlan);
            }
            if let Some(hit) = self.state.by_expert[expert] {
                if slot != hit {
                    return Err(BankError::ModifiedPlan);
                }
            } else {
                if plan.uploads.get(*upload_index) != Some(&(expert, slot)) {
                    return Err(BankError::ModifiedPlan);
                }
                *upload_index += 1;
            }
        }
        if *upload_index != plan.uploads.len() {
            return Err(BankError::ModifiedPlan);
        }
        // Prove the complete candidate forward/reverse ledger, not only misses.
        for (slot, resident) in plan.next.slots.iter().enumerate() {
            if let Some(resident) = resident
                && (resident.expert >= self.experts()
                    || resident.last_used > tick
                    || plan.next.by_expert[resident.expert] != Some(slot))
            {
                return Err(BankError::ModifiedPlan);
            }
        }
        for (expert, slot) in plan.next.by_expert.iter().enumerate() {
            if let Some(slot) = slot
                && (*slot >= self.capacity()
                    || plan.next.slots[*slot].is_none_or(|r| r.expert != expert))
            {
                return Err(BankError::ModifiedPlan);
            }
        }
        Ok(())
    }

    /// Commit only after all planned uploads succeed. Stale/cross-bank or altered
    /// plans fail without changing residency. GPU upload rollback is external.
    pub fn commit(&mut self, mut plan: Plan) -> Result<()> {
        if plan.bank_id != self.bank_id || plan.base_revision != self.state.revision {
            return Err(BankError::StalePlan);
        }
        if plan.slots != plan.canonical_slots || plan.uploads != plan.canonical_uploads {
            return Err(BankError::ModifiedPlan);
        }
        // Swapping preserves ownership: Plan Drop erases the previous committed
        // state and both route copies on success, and candidate state on error.
        std::mem::swap(&mut self.state, &mut plan.next);
        Ok(())
    }
}

/// Private candidate ledger for one MAX8 chunk. Original slots are never
/// mutated by planning or completed-wave advances. The owning GPU transaction
/// must hold exclusive physical banks, validate origin before every mutation,
/// and poison/drop those banks if any fill/consumer/cleanup fails.
/// No Clone/Debug/serialization/persistence; candidate State wipes on Drop.
pub struct StagedSlots {
    source_bank_id: u64,
    source_revision: u64,
    candidate: BankSlots,
    accepted_waves: usize,
}
impl StagedSlots {
    pub fn begin(source: &BankSlots) -> Result<Self> {
        // Establish wiping owners BEFORE copying any private residency data.
        let mut slots = Zeroizing::new(Vec::new());
        slots
            .try_reserve_exact(source.capacity())
            .map_err(|_| BankError::Allocation)?;
        slots.extend_from_slice(&source.state.slots);
        let mut by_expert = Zeroizing::new(Vec::new());
        by_expert
            .try_reserve_exact(source.experts())
            .map_err(|_| BankError::Allocation)?;
        by_expert.extend_from_slice(&source.state.by_expert);
        Ok(Self {
            source_bank_id: source.bank_id,
            source_revision: source.state.revision,
            candidate: BankSlots {
                bank_id: source.bank_id,
                state: State {
                    slots: std::mem::take(&mut *slots),
                    by_expert: std::mem::take(&mut *by_expert),
                    tick: source.state.tick,
                    revision: source.state.revision,
                },
            },
            accepted_waves: 0,
        })
    }
    pub fn validate_origin(&self, source: &BankSlots) -> Result<()> {
        if self.source_bank_id != source.bank_id
            || self.source_revision != source.state.revision
            || self.candidate.capacity() != source.capacity()
            || self.candidate.experts() != source.experts()
        {
            return Err(BankError::StalePlan);
        }
        Ok(())
    }
    pub fn plan(&mut self, source: &BankSlots, selected: &[usize]) -> Result<Plan> {
        self.validate_origin(source)?;
        if self.accepted_waves >= 8 {
            return Err(BankError::SelectionTooLarge);
        }
        self.candidate.plan(selected)
    }
    pub fn validate_pending(
        &self,
        source: &BankSlots,
        plan: &Plan,
        selected: &[usize],
    ) -> Result<()> {
        self.validate_origin(source)?;
        self.candidate.validate_pending(plan, selected)
    }
    /// Caller precondition: this exact plan's full copies/consumers completed,
    /// owning controls/readiness/output/status/guards accepted, all Bytes leases
    /// dropped and checked I/O cleanup accepted, cancellation/context clear.
    /// This advances ONLY request-private candidate metadata, never the source.
    pub fn advance_after_completed_consumers(
        &mut self,
        source: &BankSlots,
        plan: Plan,
        selected: &[usize],
    ) -> Result<()> {
        self.validate_pending(source, &plan, selected)?;
        let next = self
            .accepted_waves
            .checked_add(1)
            .ok_or(BankError::CounterOverflow)?;
        if next > 8 {
            return Err(BankError::SelectionTooLarge);
        }
        self.candidate.commit(plan)?;
        self.accepted_waves = next;
        Ok(())
    }
    pub fn accepted_waves(&self) -> usize {
        self.accepted_waves
    }

    /// Crate-private native family commit only. Caller has preflighted EVERY
    /// model family/physical bank and accepted the actual whole-model fence.
    /// Routed operator success or a dependency fence never grants this right.
    pub(crate) fn publish_after_whole_model_acceptance(
        mut self,
        source: &mut BankSlots,
        expected_waves: usize,
    ) -> Result<()> {
        self.validate_origin(source)?;
        if expected_waves == 0 || expected_waves != self.accepted_waves {
            return Err(BankError::ModifiedPlan);
        }
        std::mem::swap(&mut source.state, &mut self.candidate.state);
        Ok(())
    }
    /// Report actual retained heap capacities, including canonical route copies.
    /// Called before retaining a native pending receipt; allocator over-reserve
    /// cannot silently consume the declared request-private host envelope.
    pub(crate) fn retained_native_metadata_bytes(&self, plan: &Plan) -> Result<usize> {
        let mul =
            |count: usize, bytes: usize| count.checked_mul(bytes).ok_or(BankError::CounterOverflow);
        let add = |a: usize, b: usize| a.checked_add(b).ok_or(BankError::CounterOverflow);
        let state = |s: &State| -> Result<usize> {
            add(
                mul(s.slots.capacity(), std::mem::size_of::<Option<Resident>>())?,
                mul(s.by_expert.capacity(), std::mem::size_of::<Option<usize>>())?,
            )
        };
        let states = add(state(&self.candidate.state)?, state(&plan.next)?)?;
        let slots = mul(
            add(plan.slots.capacity(), plan.canonical_slots.capacity())?,
            std::mem::size_of::<usize>(),
        )?;
        let uploads = mul(
            add(plan.uploads.capacity(), plan.canonical_uploads.capacity())?,
            std::mem::size_of::<(usize, usize)>(),
        )?;
        add(
            add(add(states, slots)?, uploads)?,
            std::mem::size_of::<Self>() + std::mem::size_of::<Plan>(),
        )
    }

    pub(crate) fn bounded_native_metadata_bytes(capacity: usize) -> Result<usize> {
        if !routed_capacity(capacity) {
            return Err(BankError::Capacity);
        }
        let state = capacity
            .checked_mul(std::mem::size_of::<Option<Resident>>())
            .and_then(|n| n.checked_add(512 * std::mem::size_of::<Option<usize>>()))
            .ok_or(BankError::CounterOverflow)?;
        state
            .checked_mul(2)
            .and_then(|n| {
                n.checked_add(
                    2 * 80 * (std::mem::size_of::<usize>() + std::mem::size_of::<(usize, usize)>())
                        + std::mem::size_of::<Self>()
                        + std::mem::size_of::<Plan>(),
                )
            })
            .ok_or(BankError::CounterOverflow)
    }

    // No routed-only GPU fixture may publish original-bank metadata. A future
    // whole-model owner must place publication behind its real whole-chunk
    // fence. This pure host seam proves identity/revision/swap semantics only.
    #[cfg(test)]
    fn publish_host_fixture_only(
        mut self,
        source: &mut BankSlots,
        expected_waves: usize,
    ) -> Result<()> {
        self.validate_origin(source)?;
        if expected_waves == 0 || expected_waves != self.accepted_waves {
            return Err(BankError::ModifiedPlan);
        }
        std::mem::swap(&mut source.state, &mut self.candidate.state);
        Ok(())
    }
}
impl Drop for StagedSlots {
    fn drop(&mut self) {
        self.source_bank_id.zeroize();
        self.source_revision.zeroize();
        self.accepted_waves.zeroize();
        // candidate.State owns complete wiping Drop, including spare capacity.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn original_state_bound_includes_actual_capacity_and_inline_counters() {
        let mut bank = BankSlots::new(128, 512).unwrap();
        let bound = BankSlots::bounded_original_metadata_bytes(128, 512).unwrap();
        assert_eq!(bank.original_metadata_bytes().unwrap(), bound);
        assert!(48 * bound <= 640 * 1024);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(48 * bound, 544_128);
        assert_eq!(
            bound,
            std::mem::size_of::<BankSlots>()
                + 128 * std::mem::size_of::<Option<Resident>>()
                + 512 * std::mem::size_of::<Option<usize>>()
        );
        bank.state.slots.reserve(128);
        assert!(bank.original_metadata_bytes().unwrap() > bound);
        assert!(BankSlots::bounded_original_metadata_bytes(MAX_CAPACITY + 1, 512).is_err());
    }

    fn commit(bank: &mut BankSlots, experts: &[usize]) -> (Vec<usize>, Vec<(usize, usize)>) {
        let plan = bank.plan(experts).unwrap();
        let result = (plan.slots.clone(), plan.uploads.clone());
        bank.commit(plan).unwrap();
        result
    }

    fn assert_zero_capacity<T>(values: &Vec<T>) {
        assert!(values.is_empty());
        // After Vec::zeroize, every byte of the entire capacity is initialized
        // to zero. Inspect before deallocation; no use-after-free test tricks.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values.as_ptr().cast::<u8>(),
                values.capacity() * std::mem::size_of::<T>(),
            )
        };
        assert!(bytes.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn route_and_lru_cleanup_covers_complete_retained_host_capacities() {
        let mut bank = BankSlots::new(3, 8).unwrap();
        commit(&mut bank, &[5, 2, 7]);
        let mut plan = bank.plan(&[6, 5, 2]).unwrap();
        assert!(!plan.uploads.is_empty());
        plan.wipe_routes();
        assert_zero_capacity(&plan.slots);
        assert_zero_capacity(&plan.uploads);
        assert_zero_capacity(&plan.canonical_slots);
        assert_zero_capacity(&plan.canonical_uploads);
        assert_eq!((plan.bank_id, plan.base_revision), (0, 0));
        plan.next.wipe();
        assert_zero_capacity(&plan.next.slots);
        assert_zero_capacity(&plan.next.by_expert);
        assert_eq!((plan.next.tick, plan.next.revision), (0, 0));
        // Wiping an uncommitted candidate leaves the original owner intact.
        assert_eq!(bank.state.by_expert[5], Some(0));
        bank.state.wipe();
        assert_zero_capacity(&bank.state.slots);
        assert_zero_capacity(&bank.state.by_expert);
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
    fn routed_capacity_override_preserves_defaults_and_rejects_invalid_before_planning() {
        assert_eq!(configured_capacity(None, 64).unwrap(), 64);
        assert_eq!(configured_capacity(None, 16).unwrap(), 16);
        assert_eq!(configured_capacity(Some("72"), 64).unwrap(), 72);
        assert_eq!(configured_capacity(Some("96"), 64).unwrap(), 96);
        assert_eq!(configured_capacity(Some("128"), 64).unwrap(), 128);
        assert_eq!(configured_capacity(Some("144"), 64).unwrap(), 144);
        assert_eq!(configured_capacity(Some("10"), 16).unwrap(), 10);
        for invalid in [
            "",
            "0",
            "9",
            "73",
            "95",
            "97",
            "512",
            "-1",
            "+72",
            " 72",
            "72 ",
            "1.5",
            "abc",
            "184467440737095516160",
        ] {
            assert_eq!(
                configured_capacity(Some(invalid), 64),
                Err(BankError::Capacity)
            );
        }
        assert_eq!(configured_capacity(None, 9), Err(BankError::Capacity));
    }

    #[test]
    fn extended_slots_are_used_and_requested_hits_remain_protected() {
        for capacity in [96, 128, 144] {
            let mut bank = BankSlots::new(capacity, 512).unwrap();
            let initial: Vec<_> = (0..capacity).collect();
            let p = bank.plan(&initial).unwrap();
            assert_eq!(p.slots, initial);
            bank.commit(p).unwrap();
            let new = capacity + 4;
            let hit = capacity - 1;
            let p = bank
                .plan(&[new, hit, 70, 69, 68, 67, 66, 65, 64, 63])
                .unwrap();
            assert_eq!(p.uploads, [(new, 0)]);
            assert_eq!(&p.slots[1..], &[hit, 70, 69, 68, 67, 66, 65, 64, 63]);
            bank.commit(p).unwrap();
            assert_eq!(bank.state.by_expert[new], Some(0));
            assert_eq!(bank.state.by_expert[hit], Some(hit));
            assert_eq!(bank.state.by_expert[0], None);
        }
    }

    #[test]
    fn counter_overflow_and_capacity_errors_leave_state_unchanged() {
        assert!(matches!(BankSlots::new(0, 512), Err(BankError::Capacity)));
        assert!(matches!(
            BankSlots::new(MAX_CAPACITY + 1, 512),
            Err(BankError::Capacity)
        ));
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
    #[test]
    fn staged_real_generation_protects_hits_and_does_not_publish_between_waves() {
        let mut source = BankSlots::new(16, 512).unwrap();
        let seed: Vec<_> = (0..16).collect();
        let p = source.plan(&seed).unwrap();
        source.commit(p).unwrap();
        let original = source.state.clone();
        let mut staged = StagedSlots::begin(&source).unwrap();
        let first = [16, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        let p = staged.plan(&source, &first).unwrap();
        assert_eq!(p.uploads, [(16, 9)]); // All requested hits protected before first victim.
        staged.validate_pending(&source, &p, &first).unwrap();
        assert!(source.state == original);
        staged
            .advance_after_completed_consumers(&source, p, &first)
            .unwrap();
        assert_eq!(staged.accepted_waves(), 1);
        assert!(source.state == original);
        let second = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26];
        let p = staged.plan(&source, &second).unwrap();
        staged
            .advance_after_completed_consumers(&source, p, &second)
            .unwrap();
        assert_eq!(staged.accepted_waves(), 2);
        assert!(source.state == original);
        assert_eq!(staged.candidate.state.by_expert[16], None);
        // Pure host only: eventual authorized whole-model publication swaps
        // candidate state and wipes the previous ledger through owning Drop.
        staged.publish_host_fixture_only(&mut source, 2).unwrap();
        assert!(source.state != original);
        assert_eq!(source.state.by_expert[17], Some(10));
        assert_eq!(source.state.revision, original.revision + 2);
    }

    #[test]
    fn staged_drop_cross_bank_stale_and_modified_pending_plans_cannot_publish() {
        let mut source = BankSlots::new(16, 512).unwrap();
        let selected: Vec<_> = (0..10).collect();
        let original = source.state.clone();
        {
            let mut s = StagedSlots::begin(&source).unwrap();
            let p = s.plan(&source, &selected).unwrap();
            s.advance_after_completed_consumers(&source, p, &selected)
                .unwrap();
        }
        assert!(source.state == original);
        let mut s = StagedSlots::begin(&source).unwrap();
        let foreign = BankSlots::new(16, 512).unwrap();
        assert_eq!(s.validate_origin(&foreign), Err(BankError::StalePlan));
        let mut p = s.plan(&source, &selected).unwrap();
        p.slots.swap(0, 1);
        assert_eq!(
            s.validate_pending(&source, &p, &selected),
            Err(BankError::ModifiedPlan)
        );
        drop(p);
        let p = s.plan(&source, &selected).unwrap();
        let mut wrong = selected.clone();
        wrong[0] = 20;
        assert_eq!(
            s.validate_pending(&source, &p, &wrong),
            Err(BankError::ModifiedPlan)
        );
        drop(p);
        let mut p = s.plan(&source, &selected).unwrap();
        p.next.by_expert[0] = Some(1);
        assert_eq!(
            s.validate_pending(&source, &p, &selected),
            Err(BankError::ModifiedPlan)
        );
        drop(p);
        let p = source.plan(&selected).unwrap();
        source.commit(p).unwrap();
        assert_eq!(s.validate_origin(&source), Err(BankError::StalePlan));
        assert_eq!(
            s.publish_host_fixture_only(&mut source, 1),
            Err(BankError::StalePlan)
        );
    }

    #[test]
    fn staged_plan_cannot_advance_twice_or_publish_an_incomplete_chunk() {
        let mut source = BankSlots::new(16, 512).unwrap();
        let selected: Vec<_> = (0..10).collect();
        let original = source.state.clone();
        let mut s = StagedSlots::begin(&source).unwrap();
        let stale = s.plan(&source, &selected).unwrap();
        let accepted = s.plan(&source, &selected).unwrap();
        s.advance_after_completed_consumers(&source, accepted, &selected)
            .unwrap();
        assert_eq!(
            s.validate_pending(&source, &stale, &selected),
            Err(BankError::StalePlan)
        );
        assert_eq!(
            s.publish_host_fixture_only(&mut source, 2),
            Err(BankError::ModifiedPlan)
        );
        assert!(source.state == original);
    }
}
