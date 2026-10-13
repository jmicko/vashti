//! Setup-only PUBLIC host-mirror preparation. Default keeps every committed
//! GPU resident in its actual slot and fills only empty slots. The separate
//! CPU13 diagnostic fixes only the resident set; both use original weights.
//! The parent mints the opaque native completion authority from the SAME live
//! target model immediately before calling; this module cannot mint that proof.
//! No routing trace, guessed expert substitution, source freeze or persistence.
use super::*;
#[path = "public_host_cpu_gpu_launch.rs"]
mod cpu_gpu_launch;
use crate::{
    bank_slots::{BankSlots, CompletionGeneration, Plan},
    private_arena::{BufferId, PrivateArena},
    resident_model::{batch::CompletedNativeProjectionBoundary, batch_io as resident_batch_io},
};
use kuno_text::model::BoundGraph;
use zeroize::Zeroizing;

const LAYERS: usize = 48;
const CAPACITY: usize = crate::daslab_profile::BANK_SLOTS;
const EXPERTS: usize = 512;
// Same bounded temporary private-plan allowance as original oracle preload.
// Original committed-bank storage is already in the model's bank reservation;
// public packed payload capacities retain their separate source-page charges.
const PLAN_HOST_LIMIT: usize = 192 * 1024;
const PLAN_MISC_BYTES: usize = 4096;
const WRITE_METADATA_BYTES: usize = 8192;
const SOURCE_LIMIT: u64 = 40 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum PublicHostBankFillPolicy {
    #[default]
    Original,
    StrataRankedFiller,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CompletedPublicHostBankFill {
    pub(crate) policy: PublicHostBankFillPolicy,
    pub(crate) completed_layers: u64,
    pub(crate) filler_experts: u64,
    pub(crate) preserved_residents: u64,
}
impl CompletedPublicHostBankFill {
    fn selected(filler_experts: u64) -> Result<Self> {
        let preserved_residents = (LAYERS * CAPACITY) as u64;
        let preserved_residents = preserved_residents
            .checked_sub(filler_experts)
            .ok_or_else(reject)?;
        Ok(Self {
            policy: PublicHostBankFillPolicy::StrataRankedFiller,
            completed_layers: LAYERS as u64,
            filler_experts,
            preserved_residents,
        })
    }
    pub(crate) fn validate(self) -> Result<()> {
        match self.policy {
            PublicHostBankFillPolicy::Original if self == Self::default() => Ok(()),
            PublicHostBankFillPolicy::StrataRankedFiller
                if self.completed_layers == LAYERS as u64
                    && self.filler_experts.checked_add(self.preserved_residents)
                        == Some((LAYERS * CAPACITY) as u64) =>
            {
                Ok(())
            }
            _ => Err(reject()),
        }
    }
}
/// Minted only after every original selected write/audit/commit completed.
/// Consumed into the existing immutable Plan; the inline tokens cannot expose IDs.
pub(crate) struct BankFillPublication {
    completed: CompletedPublicHostBankFill,
    generations: [CompletionGeneration; LAYERS],
}
// Inline publication tokens and per-layer PUBLIC prefix share the original
// completion misc allowance. No new application heap or private owner charge.
const _: () = assert!(std::mem::size_of::<BankFillPublication>() + CAPACITY * 2 <= PLAN_MISC_BYTES);
impl BankFillPublication {
    pub(crate) fn completed(
        &self,
        banks: [&expert_banks::ExpertBanks; LAYERS],
    ) -> Result<CompletedPublicHostBankFill> {
        self.validate_generations(banks.map(|bank| &bank.slots))?;
        Ok(self.completed)
    }
    fn validate_generations(&self, banks: [&BankSlots; LAYERS]) -> Result<()> {
        self.completed.validate()?;
        for (origin, bank) in self.generations.iter().zip(banks) {
            origin.validate(bank).map_err(|_| reject())?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Placement {
    PreserveResidents,
    StrataRankedFiller,
    // Compiled CPU13 placement diagnostic only. Routing is never changed.
    ValidateHostComplement,
}
impl Placement {
    fn selection(self, bank: &BankSlots, layer: usize) -> Result<Zeroizing<Vec<usize>>> {
        if matches!(self, Self::StrataRankedFiller) {
            return ranked_completion_selection(bank, &public_strata_bank_profile::prefix(layer)?);
        }
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
            Self::StrataRankedFiller if crate::strata_port::profile_first() => {
                validate_profile_selected_plan(bank, plan, selected)
            }
            Self::PreserveResidents | Self::StrataRankedFiller => {
                validate_empty_plan(bank, plan, selected)
            }
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

/// Existing residents retain SLOT order; only the fixed PUBLIC prefix fills
/// empties. Its144 distinct IDs contain at least144-r nonresidents for any r.
fn ranked_completion_selection(bank: &BankSlots, prefix: &[u16]) -> Result<Zeroizing<Vec<usize>>> {
    public_strata_bank_profile::validate_prefix(prefix)?;
    if bank.capacity() != CAPACITY || bank.experts() != EXPERTS {
        return Err(reject());
    }
    let mut selected = Zeroizing::new(Vec::new());
    selected
        .try_reserve_exact(CAPACITY)
        .map_err(|_| Error::LimitExceeded("public host completion selection"))?;
    if selected.capacity() != CAPACITY {
        return Err(reject());
    }
    if crate::strata_port::profile_first() {
        selected.extend(prefix.iter().map(|&id| usize::from(id)));
        return Ok(selected);
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
    for &expert in prefix {
        if selected.len() == CAPACITY {
            break;
        }
        let expert = usize::from(expert);
        if bank.resident_slot(expert).map_err(|_| reject())?.is_none() {
            selected.push(expert);
        }
    }
    if selected.len() != CAPACITY {
        return Err(reject());
    }
    Ok(selected)
}

/// A profile-owned selected set replaces only unwanted public experts at a
/// completed setup boundary. Pending canonical plan protects every wanted hit.
fn validate_profile_selected_plan(bank: &BankSlots, plan: &Plan, selected: &[usize]) -> Result<()> {
    if !crate::strata_port::enabled()
        || bank.capacity() != CAPACITY
        || bank.experts() != EXPERTS
        || selected.len() != CAPACITY
        || selected.iter().any(|&id| id >= EXPERTS)
    {
        return Err(reject());
    }
    bank.validate_pending(plan, selected)
        .map_err(|_| reject())?;
    let mut missing = 0;
    for (i, &id) in selected.iter().enumerate() {
        match bank.resident_slot(id).map_err(|_| reject())? {
            Some(old) if old == plan.slots[i] => {}
            Some(_) => return Err(reject()),
            None => missing += 1,
        }
    }
    if plan.uploads.len() != missing {
        return Err(reject());
    }
    for &(id, slot) in &plan.uploads {
        if !selected.contains(&id)
            || bank.resident_slot(id).map_err(|_| reject())?.is_some()
            || bank
                .resident_expert(slot)
                .map_err(|_| reject())?
                .is_some_and(|old| selected.contains(&old))
        {
            return Err(reject());
        }
    }
    Ok(())
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

fn placement_plan_host_bytes(
    placement: Placement,
    selected: &Vec<usize>,
    plan: &Plan,
    requests: usize,
) -> Result<usize> {
    let bytes = plan_host_bytes(selected, plan, requests)?;
    let bytes = bytes
        .checked_add(if matches!(placement, Placement::StrataRankedFiller) {
            PUBLIC_STRATA_BANK_PROFILE_BYTES
        } else {
            0
        })
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

    /// Same original writes/fault proofs; only empty-slot selection differs.
    pub(crate) fn complete_public_host_ranked_banks(
        &mut self,
        boundary: &CompletedNativeProjectionBoundary,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        fault: BufferId,
        cancel: &AtomicBool,
    ) -> Result<(expert_banks::FillStats, BankFillPublication)> {
        let result = (|| {
            let fills = self.complete_public_host_banks_for(
                boundary,
                graph,
                arena,
                fault,
                cancel,
                Placement::StrataRankedFiller,
            )?;
            let banks = self.public_host_completed_banks(graph)?;
            let publication = BankFillPublication {
                completed: CompletedPublicHostBankFill::selected(fills.experts)?,
                generations: std::array::from_fn(|layer| {
                    banks[layer].slots.completion_generation()
                }),
            };
            publication.completed(banks)?;
            Ok((fills, publication))
        })();
        if result.is_err() {
            self.poison_resident();
        }
        result
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
        resident_batch_io::read_fault(arena, admission, fault, cancel)
            .map_err(|_| Error::InvalidFormat("public host completion owning fault checkpoint"))?;
        let mut total = expert_banks::FillStats::default();
        for layer in 0..LAYERS {
            if cancel.load(Ordering::Acquire) {
                return Err(Error::InvalidFormat("GPU execution cancelled"));
            }
            let key = format!("blk.{layer}.ffn_gate_exps.weight");
            let bank = self.banks.get_mut(&key).ok_or_else(reject)?;
            let selected = placement.selection(&bank.slots, layer)?;
            let plan = bank.slots.plan(&selected).map_err(|_| reject())?;
            placement.validate(&bank.slots, &plan, &selected)?;
            placement_plan_host_bytes(placement, &selected, &plan, 0)?;
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
                placement_plan_host_bytes(placement, &selected, &plan, requests.capacity())?;
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
                            )
                            .map_err(|_| {
                                Error::InvalidFormat("public host completion original source read")
                            })?;
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
                        expert_banks::write_public_existing(&self.client, writes, charged, cancel)
                            .map_err(|_| {
                                Error::InvalidFormat("public host completion original source write")
                            })?;
                    if bytes != add.bytes {
                        return Err(Error::InvalidBufferLength);
                    }
                    add.upload_ns = upload;
                    Ok(add)
                })?;
                // ORIGINAL completed owning audit precedes ANY metadata commit.
                resident_batch_io::read_fault(arena, admission, fault, cancel).map_err(|_| {
                    Error::InvalidFormat("public host completion owning fault checkpoint")
                })?;
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

    /// Read-only post-attachment authentication for the scoped CPU/GPU public
    /// operator comparison. It returns no bank/source handles and deliberately
    /// leaves the pre-attachment completion accessor unchanged.
    pub(crate) fn check_public_host_cpu_borrow(
        &self,
        plan: &crate::public_host_expert_plan::Plan<'_, '_>,
        graph: BoundGraph<'_, '_>,
        client: &ComputeClient<WgpuRuntime>,
    ) -> Result<()> {
        self.check_client(client)?;
        self.check_target_rows_policy()?;
        if !self.target_rows_reserved
            || self.packed_profile != expert_banks::OriginalPackedProfile::DaslabIq3
            || self.bank_slots != CAPACITY
            || self.banks.len() != LAYERS
            || !self.resident_banks_frozen
            || self.target_frozen_host_source_identity != Some(plan.identity())
            || self.transient_resident.is_none()
            || self.oracle_trace.is_some()
            || self.adaptive.is_some()
            || self.secondary.is_some()
            || self.secondary_frequency.is_some()
        {
            return Err(reject());
        }
        let first = self
            .banks
            .get("blk.0.ffn_gate_exps.weight")
            .ok_or_else(reject)?;
        let mut banks = [first; LAYERS];
        for (layer, result) in banks.iter_mut().enumerate() {
            *result = self
                .banks
                .get(&format!("blk.{layer}.ffn_gate_exps.weight"))
                .ok_or_else(reject)?;
        }
        plan.check_frozen(graph, self.model, banks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranked_filler_all_layers_and_resident_counts_preserve_exact_slots() {
        for layer in 0..LAYERS {
            let prefix = public_strata_bank_profile::prefix(layer).unwrap();
            for residents in 0..=CAPACITY {
                // Independently fixed, noncontiguous public test IDs, not routes.
                let ids: Vec<_> = (0..residents)
                    .map(|i| (i * 7 + layer * 13) % EXPERTS)
                    .collect();
                let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
                seed(&mut bank, &ids);
                let selected = Placement::StrataRankedFiller
                    .selection(&bank, layer)
                    .unwrap();
                assert_eq!(&selected[..residents], &ids);
                let expected: Vec<_> = prefix
                    .iter()
                    .map(|&id| usize::from(id))
                    .filter(|id| !ids.contains(id))
                    .take(CAPACITY - residents)
                    .collect();
                assert_eq!(&selected[residents..], &expected);
                let generation = bank.completion_generation();
                let plan = bank.plan(&selected).unwrap();
                Placement::StrataRankedFiller
                    .validate(&bank, &plan, &selected)
                    .unwrap();
                assert_eq!(plan.uploads.len(), CAPACITY - residents);
                let original_charge = plan_host_bytes(&selected, &plan, 30).unwrap();
                assert_eq!(
                    placement_plan_host_bytes(Placement::StrataRankedFiller, &selected, &plan, 30)
                        .unwrap(),
                    original_charge + PUBLIC_STRATA_BANK_PROFILE_BYTES
                );
                assert_eq!(
                    placement_plan_host_bytes(Placement::PreserveResidents, &selected, &plan, 30)
                        .unwrap(),
                    original_charge
                );
                if plan.uploads.is_empty() {
                    generation.validate(&bank).unwrap();
                } else {
                    bank.commit(plan).unwrap();
                    assert!(generation.validate(&bank).is_err());
                }
                for (slot, id) in ids.iter().enumerate() {
                    assert_eq!(bank.resident_expert(slot).unwrap(), Some(*id));
                }
                assert!((0..CAPACITY).all(|slot| bank.resident_expert(slot).unwrap().is_some()));
            }
        }
    }

    #[test]
    fn ranked_filler_rejects_corrupt_prefix_and_changed_pending_plan() {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        seed(&mut bank, &[511, 300, 7]);
        let mut prefix = public_strata_bank_profile::prefix(0).unwrap();
        prefix[1] = prefix[0];
        assert!(ranked_completion_selection(&bank, &prefix).is_err());
        prefix[1] = 512;
        assert!(ranked_completion_selection(&bank, &prefix).is_err());
        let selected = Placement::StrataRankedFiller.selection(&bank, 0).unwrap();
        let mut plan = bank.plan(&selected).unwrap();
        plan.uploads[0].0 = 511;
        assert!(
            Placement::StrataRankedFiller
                .validate(&bank, &plan, &selected)
                .is_err()
        );
    }

    #[test]
    fn ranked_publication_rejects_reordered_and_recency_changed_banks() {
        let mut banks: [BankSlots; LAYERS] =
            std::array::from_fn(|_| BankSlots::new(CAPACITY, EXPERTS).unwrap());
        for bank in &mut banks {
            seed(bank, &(0..CAPACITY).collect::<Vec<_>>());
        }
        let publication = BankFillPublication {
            completed: CompletedPublicHostBankFill::selected(6912).unwrap(),
            generations: std::array::from_fn(|layer| banks[layer].completion_generation()),
        };
        publication
            .validate_generations(std::array::from_fn(|i| &banks[i]))
            .unwrap();
        assert!(
            publication
                .validate_generations(std::array::from_fn(|i| &banks[(i + 1) % LAYERS]))
                .is_err()
        );
        seed(&mut banks[17], &(0..CAPACITY).collect::<Vec<_>>());
        assert!(
            publication
                .validate_generations(std::array::from_fn(|i| &banks[i]))
                .is_err()
        );
        let bank = &mut banks[0];
        let selected = Placement::StrataRankedFiller.selection(bank, 0).unwrap();
        let plan = bank.plan(&selected).unwrap();
        let original = plan_host_bytes(&selected, &plan, 0).unwrap();
        let near_limit = (PLAN_HOST_LIMIT - original) / std::mem::size_of::<u64>();
        assert!(plan_host_bytes(&selected, &plan, near_limit).is_ok());
        assert!(
            placement_plan_host_bytes(Placement::StrataRankedFiller, &selected, &plan, near_limit)
                .is_err()
        );
    }

    #[test]
    fn ranked_publication_counts_and_inline_witness_stay_bounded() {
        assert!(std::mem::size_of::<BankFillPublication>() < PLAN_MISC_BYTES);
        assert_eq!(
            CompletedPublicHostBankFill::default().policy,
            PublicHostBankFillPolicy::Original
        );
        for filler in 0..=(LAYERS * CAPACITY) as u64 {
            let completed = CompletedPublicHostBankFill::selected(filler).unwrap();
            completed.validate().unwrap();
            assert_eq!(
                completed.filler_experts + completed.preserved_residents,
                6912
            );
        }
        assert!(CompletedPublicHostBankFill::selected(6913).is_err());
        let mut bad = CompletedPublicHostBankFill::selected(10).unwrap();
        bad.completed_layers = 47;
        assert!(bad.validate().is_err());
        bad = CompletedPublicHostBankFill::selected(10).unwrap();
        bad.policy = PublicHostBankFillPolicy::Original;
        assert!(bad.validate().is_err());
        let mut a = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        let b = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        let origin = a.completion_generation();
        assert!(origin.validate(&b).is_err());
        seed(&mut a, &[7]);
        assert!(origin.validate(&a).is_err());
        let origin = a.completion_generation();
        seed(&mut a, &[7]); // Same map, real recency-only commit must reject reuse.
        assert!(origin.validate(&a).is_err());
    }

    #[cfg(feature = "strata-rust-port")]
    #[test]
    fn profile_owned_plan_protects_wanted_hits_and_rejects_redirected_writes() {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        seed(&mut bank, &(0..CAPACITY).collect::<Vec<_>>());
        let wanted = public_strata_bank_profile::prefix(0)
            .unwrap()
            .into_iter()
            .map(usize::from)
            .collect::<Vec<_>>();
        let mut plan = bank.plan(&wanted).unwrap();
        validate_profile_selected_plan(&bank, &plan, &wanted).unwrap();
        assert!(!plan.uploads.is_empty());
        let original = plan.uploads[0];
        let wanted_hit = wanted
            .iter()
            .find_map(|&id| bank.resident_slot(id).unwrap())
            .unwrap();
        plan.uploads[0].1 = wanted_hit;
        assert!(validate_profile_selected_plan(&bank, &plan, &wanted).is_err());
        plan.uploads[0] = original;
        validate_profile_selected_plan(&bank, &plan, &wanted).unwrap();
        bank.commit(plan).unwrap();
        let actual = (0..CAPACITY)
            .map(|slot| bank.resident_expert(slot).unwrap().unwrap())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(actual, wanted.into_iter().collect());
    }

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
            let preserved = Placement::PreserveResidents.selection(&bank, 0).unwrap();
            assert_eq!(&preserved[..ids.len()], &ids);
            let selected = Placement::ValidateHostComplement
                .selection(&bank, 0)
                .unwrap();
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
                Placement::PreserveResidents
                    .selection(&bank, 0)
                    .unwrap()
                    .len(),
                CAPACITY
            );
        }
    }

    #[test]
    fn diagnostic_rejects_any_other_set_or_modified_actual_replacement() {
        let mut bank = BankSlots::new(CAPACITY, EXPERTS).unwrap();
        seed(&mut bank, &[511, 7, 300]);
        let normal = Placement::PreserveResidents.selection(&bank, 0).unwrap();
        let plan = bank.plan(&normal).unwrap();
        assert!(
            Placement::ValidateHostComplement
                .validate(&bank, &plan, &normal)
                .is_err()
        );
        let selected = Placement::ValidateHostComplement
            .selection(&bank, 0)
            .unwrap();
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
