//! Setup-only PUBLIC host-mirror preparation. Default keeps every committed
//! GPU resident in its actual slot and fills only empty slots. The separate
//! CPU13 diagnostic fixes only the resident set; both use original weights.
//! The parent mints the opaque native completion authority from the SAME live
//! target model immediately before calling; this module cannot mint that proof.
//! No routing trace, guessed expert substitution, source freeze or persistence.
use super::*;
use crate::{
    bank_slots::{BankSlots, Plan},
    private_arena::{BufferId, PrivateArena},
    resident_model::{batch::CompletedNativeProjectionBoundary, batch_io as resident_batch_io},
};
use kuno_text::model::BoundGraph;
use zeroize::Zeroizing;

const LAYERS: usize = 48;
const CAPACITY: usize = 144;
const EXPERTS: usize = 512;
// Same bounded temporary private-plan allowance as original oracle preload.
// Original committed-bank storage is already in the model's bank reservation;
// public packed payload capacities retain their separate source-page charges.
const PLAN_HOST_LIMIT: usize = 192 * 1024;
const PLAN_MISC_BYTES: usize = 4096;
const WRITE_METADATA_BYTES: usize = 8192;
const SOURCE_LIMIT: u64 = 40 * 1024 * 1024;

#[derive(Clone, Copy)]
enum Placement {
    PreserveResidents,
    // Compiled CPU13 placement diagnostic only. Routing is never changed.
    ValidateHostComplement,
}
impl Placement {
    fn selection(self, bank: &BankSlots) -> Result<Zeroizing<Vec<usize>>> {
        let mut selected = completion_selection(bank)?;
        if matches!(self, Self::ValidateHostComplement) {
            for (expert, id) in selected.iter_mut().enumerate() {
                *id = expert;
            }
        }
        Ok(selected)
    }

    fn validate(self, bank: &BankSlots, plan: &Plan, selected: &[usize]) -> Result<()> {
        match self {
            Self::PreserveResidents => validate_empty_plan(bank, plan, selected),
            Self::ValidateHostComplement => validate_canonical_plan(bank, plan, selected),
        }
    }
}

fn reject() -> Error {
    Error::InvalidFormat("public host bank completion")
}

/// Copy current residents in SLOT order, then fill the remaining selection
/// with the lowest absent IDs. Selecting every old resident protects it from
/// the original planner's victim path even when resident IDs are noncontiguous.
fn completion_selection(bank: &BankSlots) -> Result<Zeroizing<Vec<usize>>> {
    if bank.capacity() != CAPACITY || bank.experts() != EXPERTS {
        return Err(reject());
    }
    let mut selected = Zeroizing::new(Vec::new());
    selected
        .try_reserve_exact(CAPACITY)
        .map_err(|_| Error::LimitExceeded("public host completion selection"))?;
    if selected.capacity() != CAPACITY {
        return Err(Error::LimitExceeded("public host completion selection"));
    }
    for slot in 0..CAPACITY {
        if let Some(expert) = bank.resident_expert(slot).map_err(|_| reject())? {
            if expert >= EXPERTS
                || selected.contains(&expert)
                || bank.resident_slot(expert).map_err(|_| reject())? != Some(slot)
            {
                return Err(reject());
            }
            selected.push(expert);
        }
    }
    for expert in 0..EXPERTS {
        if selected.len() == CAPACITY {
            break;
        }
        if bank.resident_slot(expert).map_err(|_| reject())?.is_none() {
            selected.push(expert);
        }
    }
    if selected.len() != CAPACITY {
        return Err(reject());
    }
    Ok(selected)
}

/// Authenticate the COMPLETE original pending plan, not only its upload list.
/// Every cached hit must retain its actual old slot; every write must target an
/// originally empty slot. This validation remains live until the owning fence.
fn validate_empty_plan(bank: &BankSlots, plan: &Plan, selected: &[usize]) -> Result<()> {
    bank.validate_pending(plan, selected)
        .map_err(|_| reject())?;
    if selected.len() != CAPACITY {
        return Err(reject());
    }
    let mut empty = 0;
    for slot in 0..CAPACITY {
        match bank.resident_expert(slot).map_err(|_| reject())? {
            Some(expert) => {
                let index = selected
                    .iter()
                    .position(|&id| id == expert)
                    .ok_or_else(reject)?;
                if plan.slots[index] != slot {
                    return Err(reject());
                }
            }
            None => empty += 1,
        }
    }
    if plan.uploads.len() != empty {
        return Err(reject());
    }
    for &(expert, slot) in &plan.uploads {
        if bank.resident_expert(slot).map_err(|_| reject())?.is_some()
            || bank.resident_slot(expert).map_err(|_| reject())?.is_some()
        {
            return Err(reject());
        }
    }
    Ok(())
}

/// Fixed resident SET 0..143; retained selected hits keep their original slots.
/// Original checked LRU replacement may remove only IDs outside that set. All
/// missing IDs retain their actual original source tensors/packed weights.
fn validate_canonical_plan(bank: &BankSlots, plan: &Plan, selected: &[usize]) -> Result<()> {
    bank.validate_pending(plan, selected)
        .map_err(|_| reject())?;
    if bank.capacity() != CAPACITY
        || bank.experts() != EXPERTS
        || selected.len() != CAPACITY
        || selected.iter().copied().ne(0..CAPACITY)
    {
        return Err(reject());
    }
    let mut missing = 0;
    for (expert, &slot) in plan.slots.iter().enumerate() {
        match bank.resident_slot(expert).map_err(|_| reject())? {
            Some(old) if old == slot => {}
            Some(_) => return Err(reject()),
            None => missing += 1,
        }
    }
    if plan.uploads.len() != missing {
        return Err(reject());
    }
    for &(expert, slot) in &plan.uploads {
        if expert >= CAPACITY
            || bank.resident_slot(expert).map_err(|_| reject())?.is_some()
            || bank
                .resident_expert(slot)
                .map_err(|_| reject())?
                .is_some_and(|old| old < CAPACITY)
        {
            return Err(reject());
        }
    }
    Ok(())
}

fn plan_host_bytes(selected: &Vec<usize>, plan: &Plan, request_capacity: usize) -> Result<usize> {
    let bytes = selected
        .capacity()
        .checked_mul(std::mem::size_of::<usize>())
        .and_then(|n| n.checked_add(plan.owned_metadata_bytes().ok()?))
        .and_then(|n| n.checked_add(request_capacity.checked_mul(std::mem::size_of::<u64>())?))
        .and_then(|n| n.checked_add(WRITE_METADATA_BYTES))
        .and_then(|n| n.checked_add(PLAN_MISC_BYTES))
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes > PLAN_HOST_LIMIT {
        return Err(Error::LimitExceeded("public host completion plan metadata"));
    }
    Ok(bytes)
}

fn add_stats(total: &mut expert_banks::FillStats, add: &expert_banks::FillStats) -> Result<()> {
    total.bytes = total
        .bytes
        .checked_add(add.bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    total.read_ns = total
        .read_ns
        .checked_add(add.read_ns)
        .ok_or(Error::ArithmeticOverflow)?;
    total.upload_ns = total
        .upload_ns
        .checked_add(add.upload_ns)
        .ok_or(Error::ArithmeticOverflow)?;
    total.experts = total
        .experts
        .checked_add(add.experts)
        .ok_or(Error::ArithmeticOverflow)?;
    Ok(())
}

impl GpuExecutor<'_> {
    fn check_public_host_completion(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        self.check_target_rows_policy()?;
        if !self.target_rows_reserved
            || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || self.bank_slots != CAPACITY
            || self.banks.len() != LAYERS
            || self.resident_banks_frozen
            || self.target_frozen_host_source_identity.is_some()
            || (self.target_batch_router_reserved && !self.target_frozen_host_router_reserved)
            || self.batch_moe.is_some()
            || self.oracle_trace.is_some()
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.secondary_frequency.is_some()
            || self.transient_resident.is_none()
            || !self.model.weights_mapped()
        {
            return Err(reject());
        }
        let admission = self.resident_admission.as_ref().ok_or_else(reject)?;
        admission.active()?;
        // Setup follows the cold scope's checked end. Active request reuse
        // would recycle dirty runtime owners instead of the full sanitizer.
        if admission.request_io_nonce.load(Ordering::Acquire) != 0 {
            return Err(reject());
        }
        graph.verify_unchanged()?;
        expert_banks::OriginalPackedProfile::DaslabIq3.validate_graph(graph)?;
        // Verify the bound graph's complete source identity, including dense
        // tensors; equal names or dimensions from another GgufFile are not proof.
        for name in graph.tensor_names() {
            if !graph.tensor(name)?.belongs_to(self.model) {
                return Err(reject());
            }
        }
        for layer in 0..LAYERS {
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            let bank = self.banks.get(&key).ok_or_else(reject)?;
            if bank.profile() != self.packed_profile
                || bank.slots.capacity() != CAPACITY
                || bank.slots.experts() != EXPERTS
            {
                return Err(reject());
            }
            bank.check_client(&self.client)?;
            let [gate, up, down] = bank.source_names().map(|name| graph.tensor(name));
            bank.check_names(RoutedWeights {
                gate: gate?,
                up: up?,
                down: down?,
            })?;
        }
        self.model.verify_file_unchanged()
    }

    /// Parent-only completion capability from the same healthy, accepted live
    /// target owner. The original completed proof is opaque and deliberately
    /// carries no serialized cursor or request IDs. Parent must obtain it at a
    /// cursor >0, with no live output lease, immediately before this call.
    /// Source/admission/client checks here supplement that existing authority.
    #[allow(dead_code)] // Parent owns registration and capability forwarding.
    pub(crate) fn complete_public_host_banks(
        &mut self,
        boundary: &CompletedNativeProjectionBoundary,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<expert_banks::FillStats> {
        self.complete_public_host_banks_for(
            boundary,
            graph,
            arena,
            fault,
            cancel,
            Placement::PreserveResidents,
        )
    }

    /// Separate CPU13 numerical diagnostic ONLY. Parent requires the original
    /// compiled public context13/warm-reset reservation, a freshly completed
    /// accepted cold13 boundary from this SAME model and the ended reuse scope.
    /// It fixes only the VRAM resident set to 0..143 so correct IDs144..511 must
    /// use the subsequent authenticated host complement. It neither reads nor
    /// changes router selections, probabilities, weights or accepted vectors.
    /// The opaque completion proof does not encode context; parent owns that
    /// existing protected diagnostic guard. Ordinary code200 uses the method
    /// above and keeps its actual warm resident set unchanged.
    #[allow(dead_code)] // Parent owns the protected CPU13 diagnostic wrapper.
    pub(crate) fn complete_public_host_validation_banks(
        &mut self,
        boundary: &CompletedNativeProjectionBoundary,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<expert_banks::FillStats> {
        self.complete_public_host_banks_for(
            boundary,
            graph,
            arena,
            fault,
            cancel,
            Placement::ValidateHostComplement,
        )
    }

    fn complete_public_host_banks_for(
        &mut self,
        _boundary: &CompletedNativeProjectionBoundary,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
        placement: Placement,
    ) -> Result<expert_banks::FillStats> {
        self.check_public_host_completion(graph)?;
        let client = arena.client().map_err(|_| reject())?;
        self.check_client(client)?;
        if !arena.public_owned_zero_audit_enabled() {
            return Err(reject());
        }
        let admission = self.resident_admission.as_ref().ok_or_else(reject)?.clone();
        admission.check_client(client)?;
        let result = self
            .complete_public_host_banks_inner(graph, arena, fault, cancel, &admission, placement);
        if result.is_err() {
            // Any begun source/queue/audit failure is terminal; partial
            // slot writes cannot be published or retried as accepted residents.
            self.poison_resident();
        }
        result
    }

    fn complete_public_host_banks_inner(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
        admission: &ResidentAdmission,
        placement: Placement,
    ) -> Result<expert_banks::FillStats> {
        // Prove actual full fault owner/context and completed runtime audit even
        // when every bank is already full and no source window is necessary.
        resident_batch_io::read_fault(arena, admission, fault, cancel)?;
        let mut total = expert_banks::FillStats::default();
        for layer in 0..LAYERS {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("GPU execution cancelled"));
            }
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            let bank = self.banks.get_mut(&key).ok_or_else(reject)?;
            let selected = placement.selection(&bank.slots)?;
            let plan = bank.slots.plan(&selected).map_err(|_| reject())?;
            placement.validate(&bank.slots, &plan, &selected)?;
            plan_host_bytes(&selected, &plan, 0)?;
            let [gate, up, down] = bank.source_names().map(|name| graph.tensor(name));
            let weights = RoutedWeights {
                gate: gate?,
                up: up?,
                down: down?,
            };
            bank.check_names(weights)?;
            bank.check_client(&self.client)?;
            for window in plan.uploads.chunks(10) {
                placement.validate(&bank.slots, &plan, &selected)?;
                let requests = fill_requests(
                    [bank.gate(), bank.up(), bank.down()].map(|b| b.expert_bytes() as u64),
                    window.len(),
                    false,
                    self.expert_batch_bytes,
                )?;
                plan_host_bytes(&selected, &plan, requests.capacity())?;
                let charged = requests.iter().try_fold(0u64, |n, &b| {
                    n.checked_add(main_admission::page_bytes(b).map_err(admission_error)?)
                        .ok_or(Error::ArithmeticOverflow)
                })?;
                if charged > SOURCE_LIMIT {
                    return Err(Error::LimitExceeded("public host completion source window"));
                }
                let add = admission.allocate(&requests, || {
                    let mut add = expert_banks::FillStats::default();
                    let mut writes = Vec::new();
                    writes
                        .try_reserve_exact(window.len() * 3)
                        .map_err(|_| Error::LimitExceeded("public expert write metadata"))?;
                    for &(expert, slot) in window {
                        placement.validate(&bank.slots, &plan, &selected)?;
                        for (packed, tensor) in [
                            (bank.gate(), weights.gate),
                            (bank.up(), weights.up),
                            (bank.down(), weights.down),
                        ] {
                            let (view, data, read) = expert_banks::prepare_public_existing(
                                packed, tensor, expert, slot, cancel,
                            )?;
                            add.bytes = add
                                .bytes
                                .checked_add(data.len() as u64)
                                .ok_or(Error::ArithmeticOverflow)?;
                            add.read_ns = add
                                .read_ns
                                .checked_add(read)
                                .ok_or(Error::ArithmeticOverflow)?;
                            writes.push((view, data));
                        }
                        add.experts = add
                            .experts
                            .checked_add(1)
                            .ok_or(Error::ArithmeticOverflow)?;
                    }
                    placement.validate(&bank.slots, &plan, &selected)?;
                    let (bytes, upload) =
                        expert_banks::write_public_existing(&self.client, writes, charged, cancel)?;
                    if bytes != add.bytes {
                        return Err(Error::InvalidBufferLength);
                    }
                    add.upload_ns = upload;
                    Ok(add)
                })?;
                // ORIGINAL completed owning audit precedes ANY metadata commit.
                resident_batch_io::read_fault(arena, admission, fault, cancel)?;
                placement.validate(&bank.slots, &plan, &selected)?;
                add_stats(&mut total, &add)?;
            }
            placement.validate(&bank.slots, &plan, &selected)?;
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("GPU execution cancelled"));
            }
            // A full existing bank needs no new commit/recency generation. For
            // nonempty plans all original windows have actually completed.
            if !plan.uploads.is_empty() {
                bank.slots.commit(plan).map_err(|_| reject())?;
            }
        }
        self.check_public_host_completion(graph)?;
        self.expert_uploads = self
            .expert_uploads
            .checked_add(total.experts)
            .ok_or(Error::ArithmeticOverflow)?;
        self.uploaded_bytes = self
            .uploaded_bytes
            .checked_add(total.bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_read_ns = self
            .weight_read_ns
            .checked_add(total.read_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        self.weight_upload_ns = self
            .weight_upload_ns
            .checked_add(total.upload_ns)
            .ok_or(Error::ArithmeticOverflow)?;
        Ok(total)
    }

    /// Read actual bank owners only after the parent has completed the above
    /// proof. This accessor does not fabricate a native boundary or a mirror
    /// source identity, and returns no separately retained selection/snapshot.
    #[allow(dead_code)] // Parent alone creates the immutable mirror plan.
    pub(crate) fn public_host_completed_banks(
        &self,
        graph: BoundGraph<'_, '_>,
    ) -> Result<[&ExpertBanks; LAYERS]> {
        self.check_public_host_completion(graph)?;
        let first = self
            .banks
            .get("blk.0.ffn_gate_exps.weight")
            .ok_or_else(reject)?;
        let mut banks = [first; LAYERS];
        for (layer, result) in banks.iter_mut().enumerate() {
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            let bank = self.banks.get(&key).ok_or_else(reject)?;
            for slot in 0..CAPACITY {
                let expert = bank
                    .slots
                    .resident_expert(slot)
                    .map_err(|_| reject())?
                    .ok_or_else(reject)?;
                if bank.slots.resident_slot(expert).map_err(|_| reject())? != Some(slot) {
                    return Err(reject());
                }
            }
            *result = bank;
        }
        Ok(banks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(bank: &mut BankSlots, ids: &[usize]) {
        let plan = bank.plan(ids).unwrap();
        bank.validate_pending(&plan, ids).unwrap();
        bank.commit(plan).unwrap();
    }

    #[test]
    fn completion_keeps_noncontiguous_residents_in_their_real_slots() {
        for ids in [
            vec![],
            vec![511, 7, 300, 0, 143],
            (368..512).rev().collect(),
        ] {
            let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
            seed(&mut bank, &ids);
            let selected = completion_selection(&bank).unwrap();
            assert_eq!(&selected[..ids.len()], &ids);
            let absent: Vec<_> = (0..EXPERTS)
                .filter(|id| !ids.contains(id))
                .take(CAPACITY - ids.len())
                .collect();
            assert_eq!(&selected[ids.len()..], &absent);
            let plan = bank.plan(&selected).unwrap();
            validate_empty_plan(&bank, &plan, &selected).unwrap();
            assert_eq!(plan.uploads.len(), CAPACITY - ids.len());
            for &(id, slot) in &plan.uploads {
                assert!(bank.resident_slot(id).unwrap().is_none());
                assert!(bank.resident_expert(slot).unwrap().is_none());
            }
            assert!(plan_host_bytes(&selected, &plan, 30).unwrap() < PLAN_HOST_LIMIT);
            bank.commit(plan).unwrap();
            for (slot, &id) in ids.iter().enumerate() {
                assert_eq!(bank.resident_expert(slot).unwrap(), Some(id));
            }
            assert!((0..CAPACITY).all(|slot| bank.resident_expert(slot).unwrap().is_some()));
        }
    }

    #[test]
    fn diagnostic_resident_set_is_exact_without_changing_preserve_selection() {
        for ids in [
            vec![],
            vec![511, 7, 300, 0, 143],
            (368..512).rev().collect(),
        ] {
            let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
            seed(&mut bank, &ids);
            let preserved = Placement::PreserveResidents.selection(&bank).unwrap();
            assert_eq!(&preserved[..ids.len()], &ids);
            let selected = Placement::ValidateHostComplement.selection(&bank).unwrap();
            assert!(selected.iter().copied().eq(0..CAPACITY));
            let plan = bank.plan(&selected).unwrap();
            Placement::ValidateHostComplement
                .validate(&bank, &plan, &selected)
                .unwrap();
            assert!(plan_host_bytes(&selected, &plan, 30).unwrap() < PLAN_HOST_LIMIT);
            for &(expert, slot) in &plan.uploads {
                assert!(expert < CAPACITY);
                assert!(bank.resident_slot(expert).unwrap().is_none());
                assert!(
                    bank.resident_expert(slot)
                        .unwrap()
                        .is_none_or(|old| old >= CAPACITY)
                );
            }
            if ids.iter().any(|&id| id >= CAPACITY) {
                // The normal entry MUST still refuse this replacement plan.
                assert!(
                    Placement::PreserveResidents
                        .validate(&bank, &plan, &selected)
                        .is_err()
                );
            }
            bank.commit(plan).unwrap();
            for (slot, &expert) in ids.iter().enumerate().filter(|&(_, &id)| id < CAPACITY) {
                assert_eq!(bank.resident_expert(slot).unwrap(), Some(expert));
            }
            for slot in 0..CAPACITY {
                assert!(bank.resident_expert(slot).unwrap().unwrap() < CAPACITY);
            }
            for expert in 0..EXPERTS {
                assert_eq!(
                    bank.resident_slot(expert).unwrap().is_some(),
                    expert < CAPACITY
                );
            }
            assert_eq!(
                Placement::PreserveResidents.selection(&bank).unwrap().len(),
                CAPACITY
            );
        }
    }

    #[test]
    fn diagnostic_rejects_any_other_set_or_modified_actual_replacement() {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        seed(&mut bank, &[511, 7, 300]);
        let normal = Placement::PreserveResidents.selection(&bank).unwrap();
        let plan = bank.plan(&normal).unwrap();
        assert!(
            Placement::ValidateHostComplement
                .validate(&bank, &plan, &normal)
                .is_err()
        );
        let selected = Placement::ValidateHostComplement.selection(&bank).unwrap();
        let mut plan = bank.plan(&selected).unwrap();
        plan.uploads[0].1 = 1; // overwrite the protected real ID7 hit
        assert!(
            Placement::ValidateHostComplement
                .validate(&bank, &plan, &selected)
                .is_err()
        );
        drop(plan);
        assert_eq!(bank.resident_expert(1).unwrap(), Some(7));
        assert_eq!(bank.resident_expert(0).unwrap(), Some(511));
        let plan = bank.plan(&selected).unwrap();
        seed(&mut bank, &[7]);
        assert!(
            Placement::ValidateHostComplement
                .validate(&bank, &plan, &selected)
                .is_err()
        );
    }

    #[test]
    fn no_upload_is_published_by_dropped_plan_or_failed_validation() {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        seed(&mut bank, &[511, 18, 202]);
        let selected = completion_selection(&bank).unwrap();
        let mut plan = bank.plan(&selected).unwrap();
        plan.uploads[0].1 = 0; // never overwrite an existing resident
        assert!(validate_empty_plan(&bank, &plan, &selected).is_err());
        drop(plan);
        assert_eq!(bank.resident_expert(0).unwrap(), Some(511));
        assert!(bank.resident_expert(3).unwrap().is_none());
        let plan = bank.plan(&selected).unwrap();
        seed(&mut bank, &[511]); // even all-hit recency changes invalidate it
        assert!(validate_empty_plan(&bank, &plan, &selected).is_err());
        assert_eq!(bank.resident_expert(0).unwrap(), Some(511));
        assert!(bank.resident_expert(3).unwrap().is_none());
    }

    #[test]
    fn completion_rejects_wrong_geometry_and_metadata_growth() {
        for (capacity, experts) in [(72, 512), (144, 256), (160, 512)] {
            let bank = BankSlots::new(capacity, experts).unwrap();
            assert!(completion_selection(&bank).is_err());
        }
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        let selected = completion_selection(&bank).unwrap();
        let plan = bank.plan(&selected).unwrap();
        assert!(plan_host_bytes(&selected, &plan, usize::MAX).is_err());
        assert!(plan_host_bytes(&selected, &plan, PLAN_HOST_LIMIT).is_err());
        let mut incomplete = selected.to_vec();
        incomplete.pop();
        assert!(validate_empty_plan(&bank, &plan, &incomplete).is_err());
    }

    #[test]
    fn separately_rounded_source_windows_retain_original_triple_precharge() {
        for parts in [
            [870400, 870400, 921600],
            [655360, 655360, 460800],
            [870400, 870400, 460800],
        ] {
            for misses in 1..=10 {
                let requests =
                    fill_requests(parts, misses, false, expert_banks::DEFAULT_BATCH_BYTES).unwrap();
                assert_eq!(requests.len(), misses * 3);
                let sum: u64 = requests
                    .iter()
                    .map(|&n| main_admission::page_bytes(n).unwrap())
                    .sum();
                assert_eq!(
                    sum,
                    parts
                        .iter()
                        .map(|&n| main_admission::page_bytes(n).unwrap())
                        .sum::<u64>()
                        * misses as u64
                );
                assert!(sum <= SOURCE_LIMIT);
                assert_ne!(
                    sum,
                    main_admission::page_bytes(parts.iter().sum::<u64>() * misses as u64).unwrap()
                );
            }
        }
        assert!(fill_requests([4, 4, 4], 11, false, expert_banks::DEFAULT_BATCH_BYTES).is_err());
    }
}
