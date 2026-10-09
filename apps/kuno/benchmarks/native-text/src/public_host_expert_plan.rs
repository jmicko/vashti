//! Metadata-only plan for the pinned DASLab public host-expert complement.
//!
//! The parent must capture actual completed, fully populated 144-slot banks,
//! freeze fills/evictions, and retain its physical/source/completion proofs.
//! This planner neither reads payloads nor proves loaded GPU bytes. It allocates
//! no GPU memory, imports no buffers, chooses no routes and publishes no state.
//! Request-derived residency maps and opaque generation snapshots are transient
//! wiping owners; they have no Debug/serialization or persistence interface.
//!
//! Original VRAM slots remain 0..143. Missing IDs in ascending order occupy
//! virtual slots 144..511, packed identically in all three roles into pages
//! 144+144+80. These virtual addresses are NOT ordinary BankSlots/GroupMap slots.
//! A future source-aware map/receipt must validate them explicitly. Ordinary
//! 144-slot planning and source-generation rules are unchanged.
//!
//! Physical bytes are a checked allocation envelope, not Vulkan telemetry.
//! An importer must independently reject requirements exceeding each page's
//! envelope, retain/charge its complete backing and padding, and detach all
//! consumers before release. Public RAM is separate from GPU MAIN/private IO.

use crate::{
    bank_slots::{BankSlots, StagedSlots},
    daslab_profile,
    expert_banks::{BankInstance, ExpertBanks, OriginalPackedProfile},
};
use kuno_text::{
    Error, GgufFile, Result, TensorType, TensorView,
    model::{BoundGraph, GraphBindingId, RoutedWeights},
};
use std::sync::atomic::{AtomicU64, Ordering};
use zeroize::{Zeroize, Zeroizing};

pub(crate) const LAYERS: usize = 48;
pub(crate) const EXPERTS: usize = 512;
pub(crate) const VRAM_SLOTS: usize = 144;
pub(crate) const HOST_SLOTS: usize = EXPERTS - VRAM_SLOTS;
pub(crate) const PAGE_COUNTS: [usize; 3] = [144, 144, 80];
pub(crate) const MAX_BINDING_BYTES: u64 = 128 * 1024 * 1024;
pub(crate) const LOGICAL_PUBLIC_RAM_BYTES: u64 = 36_147_609_600;
/// Includes fixed map owners, opaque generation snapshots and validation peak.
pub(crate) const METADATA_HOST_CAP: u64 = 2 * 1024 * 1024;
/// One original expert at a time, including the largest 921600-byte down role.
pub(crate) const SOURCE_WINDOW_HOST_BYTES: u64 = 1024 * 1024;
const MAX_ALIGNMENT: u64 = 1024 * 1024;
const UNSET: u16 = u16::MAX;

fn reject() -> Error {
    Error::InvalidFormat("public host expert source plan rejected")
}

/// One validated plan's authority; equal geometry/source bindings are not
/// interchangeable. No address reuse, public constructor or serialization.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlanIdentity(u64);
static NEXT_PLAN_IDENTITY: AtomicU64 = AtomicU64::new(1);
fn fresh_identity(next: &AtomicU64) -> Result<PlanIdentity> {
    next.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        if n == 0 { None } else { n.checked_add(1) }
    })
    .map(PlanIdentity)
    .map_err(|_| Error::ArithmeticOverflow)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Gate,
    Up,
    Down,
}
impl Role {
    fn index(self) -> usize {
        match self {
            Self::Gate => 0,
            Self::Up => 1,
            Self::Down => 2,
        }
    }
}
const ROLES: [Role; 3] = [Role::Gate, Role::Up, Role::Down];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Vram,
    Host,
}

/// Immutable address; its caller must keep it transient and never log IDs/maps.
/// For VRAM, page/local slot retain the original two 72-slot segments.
#[derive(PartialEq, Eq)]
pub(crate) struct Address {
    tier: Tier,
    virtual_slot: usize,
    page: usize,
    local_slot: usize,
}
impl Address {
    pub(crate) fn tier(&self) -> Tier {
        self.tier
    }
    pub(crate) fn virtual_slot(&self) -> usize {
        self.virtual_slot
    }
    pub(crate) fn page(&self) -> usize {
        self.page
    }
    pub(crate) fn local_slot(&self) -> usize {
        self.local_slot
    }
}
impl Drop for Address {
    fn drop(&mut self) {
        self.tier = Tier::Vram;
        self.virtual_slot.zeroize();
        self.page.zeroize();
        self.local_slot.zeroize();
    }
}
fn address(slot: usize) -> Result<Address> {
    if slot >= EXPERTS {
        return Err(reject());
    }
    if slot < VRAM_SLOTS {
        return Ok(Address {
            tier: Tier::Vram,
            virtual_slot: slot,
            page: slot / 72,
            local_slot: slot % 72,
        });
    }
    let ordinal = slot - VRAM_SLOTS;
    Ok(Address {
        tier: Tier::Host,
        virtual_slot: slot,
        page: ordinal / 144,
        local_slot: ordinal % 144,
    })
}

/// Fixed heap owners are populated in place: moving the enclosing plan never
/// relocates a filled inline residency array into an unwiped allocation.
struct Maps {
    by_expert: Zeroizing<Box<[u16]>>,
    by_virtual: Zeroizing<Box<[u16]>>,
}
impl Maps {
    fn from_slots(slots: &[Option<usize>]) -> Result<Self> {
        if slots.len() != VRAM_SLOTS {
            return Err(reject());
        }
        let mut m = Self {
            by_expert: Zeroizing::new(Box::new([UNSET; EXPERTS]) as Box<[u16]>),
            by_virtual: Zeroizing::new(Box::new([UNSET; EXPERTS]) as Box<[u16]>),
        };
        for (slot, &expert) in slots.iter().enumerate() {
            let expert = expert.ok_or(reject())?;
            if expert >= EXPERTS || m.by_expert[expert] != UNSET {
                return Err(reject());
            }
            m.by_expert[expert] = slot as u16;
            m.by_virtual[slot] = expert as u16;
        }
        let mut next = VRAM_SLOTS;
        for expert in 0..EXPERTS {
            if m.by_expert[expert] == UNSET {
                m.by_expert[expert] = next as u16;
                m.by_virtual[next] = expert as u16;
                next += 1;
            }
        }
        if next != EXPERTS {
            return Err(reject());
        }
        for (expert, &slot) in m.by_expert.iter().enumerate() {
            if slot as usize >= EXPERTS || m.by_virtual[slot as usize] as usize != expert {
                return Err(reject());
            }
        }
        Ok(m)
    }
    fn from_bank(slots: &BankSlots) -> Result<Self> {
        if slots.capacity() != VRAM_SLOTS || slots.experts() != EXPERTS {
            return Err(reject());
        }
        let mut residents = Zeroizing::new([None; VRAM_SLOTS]);
        for (slot, cell) in residents.iter_mut().enumerate() {
            let expert = slots
                .resident_expert(slot)
                .map_err(|_| reject())?
                .ok_or(reject())?;
            if slots.resident_slot(expert).map_err(|_| reject())? != Some(slot) {
                return Err(reject());
            }
            *cell = Some(expert);
        }
        let result = Self::from_slots(&*residents)?;
        for expert in 0..EXPERTS {
            let slot = result.by_expert[expert] as usize;
            let expected = (slot < VRAM_SLOTS).then_some(slot);
            if slots.resident_slot(expert).map_err(|_| reject())? != expected {
                return Err(reject());
            }
        }
        Ok(result)
    }
    fn check_bank(&self, slots: &BankSlots) -> Result<()> {
        if slots.capacity() != VRAM_SLOTS || slots.experts() != EXPERTS {
            return Err(reject());
        }
        for expert in 0..EXPERTS {
            let slot = self.by_expert[expert] as usize;
            let expected = (slot < VRAM_SLOTS).then_some(slot);
            if slots.resident_slot(expert).map_err(|_| reject())? != expected {
                return Err(reject());
            }
        }
        for slot in 0..VRAM_SLOTS {
            if slots.resident_expert(slot).map_err(|_| reject())?
                != Some(self.by_virtual[slot] as usize)
            {
                return Err(reject());
            }
        }
        Ok(())
    }
}

/// Public per-role page geometry, without an expert ID or source payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Page {
    dtype: u32,
    row_bytes: u64,
    expert_bytes: u64,
    first_virtual_slot: usize,
    slots: usize,
    logical_bytes: u64,
    physical_envelope_bytes: u64,
}
impl Page {
    pub(crate) fn dtype(self) -> u32 {
        self.dtype
    }
    pub(crate) fn row_bytes(self) -> u64 {
        self.row_bytes
    }
    pub(crate) fn expert_bytes(self) -> u64 {
        self.expert_bytes
    }
    pub(crate) fn first_virtual_slot(self) -> usize {
        self.first_virtual_slot
    }
    pub(crate) fn slots(self) -> usize {
        self.slots
    }
    pub(crate) fn logical_bytes(self) -> u64 {
        self.logical_bytes
    }
    pub(crate) fn physical_envelope_bytes(self) -> u64 {
        self.physical_envelope_bytes
    }
    /// Importer pre-allocation gate; complete physical padding stays owned.
    pub(crate) fn check_actual_allocation(self, bytes: u64, alignment: u64) -> Result<()> {
        if alignment == 0
            || !alignment.is_power_of_two()
            || bytes < self.logical_bytes
            || bytes > self.physical_envelope_bytes
            || !bytes.is_multiple_of(alignment)
        {
            return Err(reject());
        }
        Ok(())
    }
}
fn round_physical(bytes: u64, alignment: u64) -> Result<u64> {
    if !(4..=MAX_ALIGNMENT).contains(&alignment) || !alignment.is_power_of_two() {
        return Err(reject());
    }
    bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or(Error::ArithmeticOverflow)
}
fn role_pages(
    layer: usize,
    role: Role,
    dtype: u32,
    dimensions: &[u64],
    row_bytes: u64,
    tensor_bytes: u64,
    alignment: u64,
) -> Result<[Page; 3]> {
    let expected = daslab_profile::routed_types(layer).ok_or(reject())?[role.index()];
    let dims = if role == Role::Down {
        [640, 2560, 512]
    } else {
        [2560, 640, 512]
    };
    let format = TensorType::from_gguf_id(dtype)?;
    if dtype != expected || dimensions != dims || row_bytes != format.row_bytes(dims[0])? {
        return Err(reject());
    }
    let expert = row_bytes
        .checked_mul(dims[1])
        .ok_or(Error::ArithmeticOverflow)?;
    if expert == 0
        || expert > SOURCE_WINDOW_HOST_BYTES
        || !expert.is_multiple_of(4)
        || expert.checked_mul(EXPERTS as u64) != Some(tensor_bytes)
    {
        return Err(reject());
    }
    pages(dtype, row_bytes, expert, alignment)
}
fn pages(dtype: u32, row_bytes: u64, expert: u64, alignment: u64) -> Result<[Page; 3]> {
    let mut result = [Page {
        dtype,
        row_bytes,
        expert_bytes: expert,
        first_virtual_slot: 0,
        slots: 0,
        logical_bytes: 0,
        physical_envelope_bytes: 0,
    }; 3];
    let mut first = VRAM_SLOTS;
    for (p, count) in result.iter_mut().zip(PAGE_COUNTS) {
        let bytes = expert
            .checked_mul(count as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        if bytes == 0 || bytes > MAX_BINDING_BYTES || !bytes.is_multiple_of(4) {
            return Err(reject());
        }
        *p = Page {
            dtype,
            row_bytes,
            expert_bytes: expert,
            first_virtual_slot: first,
            slots: count,
            logical_bytes: bytes,
            physical_envelope_bytes: round_physical(bytes, alignment)?,
        };
        first = first.checked_add(count).ok_or(Error::ArithmeticOverflow)?;
    }
    if first != EXPERTS {
        return Err(reject());
    }
    Ok(result)
}

/// Tensor-relative, byte-exact original range for ONE complementary expert.
/// It is never an absolute file offset and cannot accidentally cross shards.
pub(crate) struct Range<'file> {
    tensor: TensorView<'file>,
    source_offset: u64,
    destination_offset: u64,
    bytes: u64,
    page: usize,
}
impl<'file> Range<'file> {
    pub(crate) fn tensor(&self) -> TensorView<'file> {
        self.tensor
    }
    pub(crate) fn source_offset(&self) -> u64 {
        self.source_offset
    }
    pub(crate) fn destination_offset(&self) -> u64 {
        self.destination_offset
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) fn page(&self) -> usize {
        self.page
    }
}
impl Drop for Range<'_> {
    fn drop(&mut self) {
        self.source_offset.zeroize();
        self.destination_offset.zeroize();
        self.bytes.zeroize();
        self.page.zeroize();
    }
}
fn offsets(expert: usize, local: usize, page: Page, tensor_bytes: u64) -> Result<(u64, u64)> {
    if expert >= EXPERTS || local >= page.slots {
        return Err(reject());
    }
    let source = (expert as u64)
        .checked_mul(page.expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let destination = (local as u64)
        .checked_mul(page.expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    if source
        .checked_add(page.expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?
        > tensor_bytes
        || destination
            .checked_add(page.expert_bytes)
            .ok_or(Error::ArithmeticOverflow)?
            > page.logical_bytes
    {
        return Err(reject());
    }
    Ok((source, destination))
}

struct Layer<'file> {
    instance: BankInstance,
    origin: StagedSlots,
    maps: Maps,
    sources: [TensorView<'file>; 3],
    pages: [[Page; 3]; 3],
}

/// Measure the opaque snapshot's real heap capacities through its existing
/// owning receipt API. The empty plan never changes either ledger; counting
/// its additional candidate state conservatively includes validation peak.
fn generation_snapshot(slots: &BankSlots) -> Result<(StagedSlots, usize)> {
    let mut origin = StagedSlots::begin(slots).map_err(|_| reject())?;
    let empty = origin.plan(slots, &[]).map_err(|_| reject())?;
    origin
        .validate_pending(slots, &empty, &[])
        .map_err(|_| reject())?;
    let bytes = origin
        .retained_native_metadata_bytes(&empty)
        .map_err(|_| reject())?;
    if bytes > StagedSlots::bounded_native_metadata_bytes(VRAM_SLOTS).map_err(|_| reject())? {
        return Err(reject());
    }
    drop(empty);
    Ok((origin, bytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Budget {
    pub logical_public_ram_bytes: u64,
    pub physical_public_ram_envelope_bytes: u64,
    /// Separately identifiable parts of the conservative 2MiB metadata reserve.
    pub mapping_metadata_bytes: u64,
    pub source_generation_metadata_bytes: u64,
    pub metadata_host_reserved_bytes: u64,
    pub source_window_host_bytes: u64,
    pub allocation_alignment: u64,
}
impl Budget {
    pub(crate) fn total_host_reserved_bytes(self) -> Result<u64> {
        self.physical_public_ram_envelope_bytes
            .checked_add(self.metadata_host_reserved_bytes)
            .and_then(|n| n.checked_add(self.source_window_host_bytes))
            .ok_or(Error::ArithmeticOverflow)
    }
}

pub(crate) struct Plan<'model, 'file> {
    identity: PlanIdentity,
    graph: BoundGraph<'model, 'file>,
    model: &'file GgufFile,
    binding: GraphBindingId,
    layers: Vec<Layer<'file>>,
    budget: Budget,
}
impl<'model, 'file> Plan<'model, 'file> {
    /// Only actual bank owners enter this interface. Parent completed boundary,
    /// physical identity and loaded-byte proofs precede this metadata snapshot.
    pub(crate) fn from_completed_banks(
        graph: BoundGraph<'model, 'file>,
        model: &'file GgufFile,
        banks: [&ExpertBanks; LAYERS],
        allocation_alignment: u64,
    ) -> Result<Self> {
        round_physical(4, allocation_alignment)?;
        if model.shard_count() != 2 || model.tensors().len() != daslab_profile::TENSOR_COUNT {
            return Err(reject());
        }
        model.verify_file_unchanged()?;
        OriginalPackedProfile::DaslabIq3.validate_graph(graph)?;
        for name in graph.tensor_names() {
            if !graph.tensor(name)?.belongs_to(model) {
                return Err(reject());
            }
        }
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(LAYERS)
            .map_err(|_| Error::LimitExceeded("host expert metadata"))?;
        if layers.capacity() != LAYERS {
            return Err(reject());
        }
        let mut logical = 0u64;
        let mut physical = 0u64;
        let mut generations = 0usize;
        for (layer, bank) in banks.iter().enumerate() {
            let sources = sources(graph, layer)?;
            check_bank(bank, sources)?;
            let maps = Maps::from_bank(&bank.slots)?;
            let (origin, generation_bytes) = generation_snapshot(&bank.slots)?;
            generations = generations
                .checked_add(generation_bytes)
                .ok_or(Error::ArithmeticOverflow)?;
            let mut rp = [[Page {
                dtype: 0,
                row_bytes: 0,
                expert_bytes: 0,
                first_virtual_slot: 0,
                slots: 0,
                logical_bytes: 0,
                physical_envelope_bytes: 0,
            }; 3]; 3];
            for role in ROLES {
                let info = sources[role.index()].info();
                rp[role.index()] = role_pages(
                    layer,
                    role,
                    info.dtype().gguf_id(),
                    info.dimensions(),
                    info.row_bytes(),
                    info.byte_len(),
                    allocation_alignment,
                )?;
                for page in rp[role.index()] {
                    logical = logical
                        .checked_add(page.logical_bytes)
                        .ok_or(Error::ArithmeticOverflow)?;
                    physical = physical
                        .checked_add(page.physical_envelope_bytes)
                        .ok_or(Error::ArithmeticOverflow)?;
                }
            }
            layers.push(Layer {
                instance: bank.instance(),
                origin,
                maps,
                sources,
                pages: rp,
            });
        }
        let mapping = layers
            .capacity()
            .checked_mul(std::mem::size_of::<Layer<'file>>() + 2 * EXPERTS * 2)
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::ArithmeticOverflow)?;
        if logical != LOGICAL_PUBLIC_RAM_BYTES
            || (mapping as u64)
                .checked_add(generations as u64)
                .ok_or(Error::ArithmeticOverflow)?
                > METADATA_HOST_CAP
        {
            return Err(reject());
        }
        let plan = Self {
            identity: fresh_identity(&NEXT_PLAN_IDENTITY)?,
            graph,
            model,
            binding: graph.binding_id(),
            layers,
            budget: Budget {
                logical_public_ram_bytes: logical,
                physical_public_ram_envelope_bytes: physical,
                mapping_metadata_bytes: mapping as u64,
                source_generation_metadata_bytes: generations as u64,
                metadata_host_reserved_bytes: METADATA_HOST_CAP,
                source_window_host_bytes: SOURCE_WINDOW_HOST_BYTES,
                allocation_alignment,
            },
        };
        plan.check_frozen(graph, model, banks)?;
        Ok(plan)
    }
    pub(crate) fn identity(&self) -> PlanIdentity {
        self.identity
    }
    /// Exact generation freeze, including recency-only ledger commits. A later
    /// source-aware consumer must not reuse ordinary mutation receipts here.
    pub(crate) fn check_frozen(
        &self,
        graph: BoundGraph<'_, '_>,
        model: &GgufFile,
        banks: [&ExpertBanks; LAYERS],
    ) -> Result<()> {
        if graph.binding_id() != self.binding
            || !std::ptr::eq(model, self.model)
            || self.layers.len() != LAYERS
        {
            return Err(reject());
        }
        self.graph.verify_unchanged()?;
        model.verify_file_unchanged()?;
        for (i, (layer, bank)) in self.layers.iter().zip(banks).enumerate() {
            if layer.instance != bank.instance() {
                return Err(reject());
            }
            layer
                .origin
                .validate_origin(&bank.slots)
                .map_err(|_| reject())?;
            layer.maps.check_bank(&bank.slots)?;
            let original = sources(graph, i)?;
            check_bank(bank, original)?;
            for (now, pinned) in original.into_iter().zip(layer.sources) {
                if !now.belongs_to(model) || now.info() != pinned.info() {
                    return Err(reject());
                }
            }
        }
        Ok(())
    }
    pub(crate) fn binding(&self) -> GraphBindingId {
        self.binding
    }
    pub(crate) fn budget(&self) -> Budget {
        self.budget
    }
    pub(crate) fn address(&self, layer: usize, expert: usize) -> Result<Address> {
        let l = self.layers.get(layer).ok_or(reject())?;
        address(*l.maps.by_expert.get(expert).ok_or(reject())? as usize)
    }
    pub(crate) fn page(&self, layer: usize, role: Role, page: usize) -> Result<Page> {
        self.layers
            .get(layer)
            .and_then(|l| l.pages[role.index()].get(page))
            .copied()
            .ok_or(reject())
    }
    /// Sorted complement ordinal0..367, the same expert for every role.
    pub(crate) fn range(&self, layer: usize, role: Role, ordinal: usize) -> Result<Range<'file>> {
        if ordinal >= HOST_SLOTS {
            return Err(reject());
        }
        let l = self.layers.get(layer).ok_or(reject())?;
        let slot = VRAM_SLOTS + ordinal;
        let a = address(slot)?;
        let page = l.pages[role.index()][a.page];
        let tensor = l.sources[role.index()];
        let expert = l.maps.by_virtual[slot] as usize;
        let (source_offset, destination_offset) =
            offsets(expert, a.local_slot, page, tensor.info().byte_len())?;
        Ok(Range {
            tensor,
            source_offset,
            destination_offset,
            bytes: page.expert_bytes,
            page: a.page,
        })
    }
}
fn sources<'file>(graph: BoundGraph<'_, 'file>, layer: usize) -> Result<[TensorView<'file>; 3]> {
    if layer >= LAYERS {
        return Err(reject());
    }
    Ok([
        graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    ])
}
fn check_bank(bank: &ExpertBanks, sources: [TensorView<'_>; 3]) -> Result<()> {
    if bank.profile() != OriginalPackedProfile::DaslabIq3
        || bank.hidden != 2560
        || bank.ff != 640
        || bank.experts != EXPERTS
        || bank.slots.capacity() != VRAM_SLOTS
    {
        return Err(reject());
    }
    bank.check_names(RoutedWeights {
        gate: sources[0],
        up: sources[1],
        down: sources[2],
    })?;
    for (b, tensor) in [bank.gate(), bank.up(), bank.down()]
        .into_iter()
        .zip(sources)
    {
        let info = tensor.info();
        let expert = info
            .row_bytes()
            .checked_mul(info.dimensions()[1])
            .ok_or(Error::ArithmeticOverflow)?;
        if b.expert_bytes() as u64 != expert || b.segments().len() != 2 {
            return Err(reject());
        }
        for (i, s) in b.segments().iter().enumerate() {
            if s.first_slot() != i * 72
                || s.capacity() != 72
                || (s.words() as u64).checked_mul(4) != expert.checked_mul(72)
            {
                return Err(reject());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plan_identity_is_unique_and_exhaustion_never_wraps_or_reuses() {
        let next = AtomicU64::new(1);
        let first = fresh_identity(&next).unwrap();
        let second = fresh_identity(&next).unwrap();
        assert!(first != second);
        assert_eq!(next.load(Ordering::Relaxed), 3);
        let exhausted = AtomicU64::new(u64::MAX - 1);
        assert_eq!(fresh_identity(&exhausted).unwrap().0, u64::MAX - 1);
        assert!(fresh_identity(&exhausted).is_err());
        assert!(fresh_identity(&exhausted).is_err());
        assert_eq!(exhausted.load(Ordering::Relaxed), u64::MAX);
        assert!(fresh_identity(&AtomicU64::new(0)).is_err());
    }
    #[test]
    fn concurrent_plan_identities_are_not_reused() {
        let next = AtomicU64::new(1);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    let next = &next;
                    scope.spawn(move || {
                        (0..32)
                            .map(|_| fresh_identity(next).unwrap().0)
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            let mut ids: Vec<_> = workers
                .into_iter()
                .flat_map(|worker| worker.join().unwrap())
                .collect();
            ids.sort_unstable();
            assert_eq!(ids, (1..=256).collect::<Vec<_>>());
        });
    }
    fn slots() -> [Option<usize>; VRAM_SLOTS] {
        std::array::from_fn(|i| Some((i * 3 + 11) % EXPERTS))
    }
    fn role(layer: usize, r: Role, alignment: u64) -> [Page; 3] {
        let dtype = daslab_profile::routed_types(layer).unwrap()[r.index()];
        let dims = if r == Role::Down {
            [640, 2560, 512]
        } else {
            [2560, 640, 512]
        };
        let rb = TensorType::from_gguf_id(dtype)
            .unwrap()
            .row_bytes(dims[0])
            .unwrap();
        role_pages(
            layer,
            r,
            dtype,
            &dims,
            rb,
            rb * dims[1] * dims[2],
            alignment,
        )
        .unwrap()
    }
    #[test]
    fn complete_actual_mapping_has_sorted_complement_and_exactly_once_coverage() {
        let s = slots();
        let map = Maps::from_slots(&s).unwrap();
        let mut seen = [false; EXPERTS];
        for (v, &expert) in map.by_virtual.iter().enumerate() {
            assert!(!seen[expert as usize]);
            seen[expert as usize] = true;
            assert_eq!(map.by_expert[expert as usize] as usize, v);
            let a = address(v).unwrap();
            assert_eq!(a.virtual_slot(), v);
            if v < VRAM_SLOTS {
                assert_eq!(a.tier(), Tier::Vram);
                assert_eq!(Some(expert as usize), s[v]);
            } else {
                assert_eq!(a.tier(), Tier::Host);
                assert!(a.local_slot() < PAGE_COUNTS[a.page()]);
            }
        }
        assert!(seen.into_iter().all(|v| v));
        assert!(map.by_virtual[VRAM_SLOTS..].windows(2).all(|w| w[0] < w[1]));
        for (slot, page, local) in [
            (0, 0, 0),
            (71, 0, 71),
            (72, 1, 0),
            (143, 1, 71),
            (144, 0, 0),
            (287, 0, 143),
            (288, 1, 0),
            (431, 1, 143),
            (432, 2, 0),
            (511, 2, 79),
        ] {
            let a = address(slot).unwrap();
            assert_eq!((a.page(), a.local_slot()), (page, local));
        }
        assert!(address(512).is_err());
    }
    #[test]
    fn partial_duplicate_out_of_range_and_wrong_capacity_are_rejected() {
        let mut s = slots();
        s[70] = None;
        assert!(Maps::from_slots(&s).is_err());
        s = slots();
        s[70] = s[69];
        assert!(Maps::from_slots(&s).is_err());
        s = slots();
        s[70] = Some(512);
        assert!(Maps::from_slots(&s).is_err());
        assert!(Maps::from_slots(&slots()[..143]).is_err());
        assert!(Maps::from_bank(&BankSlots::new(144, 512).unwrap()).is_err());
    }
    #[test]
    fn actual_bank_origin_rejects_other_instances_and_any_later_generation() {
        let mut b = BankSlots::new(144, 512).unwrap();
        let selected: Vec<_> = slots().into_iter().map(Option::unwrap).collect();
        let p = b.plan(&selected).unwrap();
        b.commit(p).unwrap();
        let map = Maps::from_bank(&b).unwrap();
        map.check_bank(&b).unwrap();
        let (origin, generation_bytes) = generation_snapshot(&b).unwrap();
        assert!(generation_bytes <= StagedSlots::bounded_native_metadata_bytes(144).unwrap());
        assert_eq!(origin.accepted_waves(), 0);
        origin.validate_origin(&b).unwrap();
        assert!(
            origin
                .validate_origin(&BankSlots::new(144, 512).unwrap())
                .is_err()
        );
        let p = b.plan(&selected[..10]).unwrap();
        assert!(p.uploads.is_empty());
        b.commit(p).unwrap();
        map.check_bank(&b).unwrap();
        assert!(origin.validate_origin(&b).is_err());
        let p = b.plan(&[500]).unwrap();
        b.commit(p).unwrap();
        assert!(map.check_bank(&b).is_err());
    }
    #[test]
    fn all_actual_layers_roles_ranges_and_host_budgets_are_exact() {
        let map = Maps::from_slots(&slots()).unwrap();
        let (mut logical, mut physical, mut largest) = (0u64, 0u64, 0u64);
        for layer in 0..LAYERS {
            for r in ROLES {
                let ps = role(layer, r, 65536);
                for p in ps {
                    logical += p.logical_bytes();
                    physical += p.physical_envelope_bytes();
                    largest = largest.max(p.logical_bytes());
                    assert!(p.logical_bytes() <= MAX_BINDING_BYTES);
                    p.check_actual_allocation(p.physical_envelope_bytes(), 65536)
                        .unwrap();
                }
                for ordinal in 0..HOST_SLOTS {
                    let expert = map.by_virtual[144 + ordinal] as usize;
                    let a = address(144 + ordinal).unwrap();
                    let p = ps[a.page()];
                    let (src, dst) =
                        offsets(expert, a.local_slot(), p, p.expert_bytes() * 512).unwrap();
                    assert_eq!(src, p.expert_bytes() * expert as u64);
                    assert_eq!(dst, p.expert_bytes() * a.local_slot() as u64);
                }
            }
        }
        assert_eq!(logical, LOGICAL_PUBLIC_RAM_BYTES);
        assert_eq!(largest, 132_710_400);
        assert!(physical >= logical);
        assert!(physical - logical < (LAYERS * 3 * 3 * 65536) as u64);
    }
    #[test]
    fn original_tensor_geometry_dtype_and_allocation_envelopes_fail_closed() {
        let p = role(0, Role::Down, 4096)[0];
        assert_eq!(p.dtype(), 20);
        assert_eq!(p.row_bytes(), 360);
        assert_eq!(p.expert_bytes(), 921600);
        assert_eq!(p.slots(), 144);
        assert_eq!(p.first_virtual_slot(), 144);
        for (ty, dims, rb, bytes) in [
            (42, [640, 2560, 512], 360, 471859200),
            (20, [2560, 640, 512], 360, 471859200),
            (20, [640, 2560, 511], 360, 471859200),
            (20, [640, 2560, 512], 359, 471859200),
            (20, [640, 2560, 512], 360, 471859199),
        ] {
            assert!(role_pages(0, Role::Down, ty, &dims, rb, bytes, 4096).is_err());
        }
        assert!(role_pages(48, Role::Down, 20, &[640, 2560, 512], 360, 471859200, 4096).is_err());
        assert!(
            p.check_actual_allocation(p.logical_bytes() - 4, 4096)
                .is_err()
        );
        assert!(
            p.check_actual_allocation(p.physical_envelope_bytes() + 4096, 4096)
                .is_err()
        );
        assert!(
            p.check_actual_allocation(p.physical_envelope_bytes(), 3)
                .is_err()
        );
    }
    #[test]
    fn checked_ranges_rounding_and_resource_overflow_are_rejected() {
        for a in [0, 1, 3, 6, MAX_ALIGNMENT * 2] {
            assert!(round_physical(64, a).is_err());
        }
        assert!(round_physical(u64::MAX, 4).is_err());
        assert!(pages(20, 360, u64::MAX, 4096).is_err());
        assert!(pages(20, 360, MAX_BINDING_BYTES, 4096).is_err());
        let p = role(0, Role::Down, 4096)[0];
        assert!(offsets(512, 0, p, p.expert_bytes() * 512).is_err());
        assert!(offsets(0, 144, p, p.expert_bytes() * 512).is_err());
        assert!(offsets(511, 0, p, p.expert_bytes() * 512 - 1).is_err());
        let b = Budget {
            logical_public_ram_bytes: 1,
            physical_public_ram_envelope_bytes: u64::MAX,
            mapping_metadata_bytes: 0,
            source_generation_metadata_bytes: 0,
            metadata_host_reserved_bytes: 1,
            source_window_host_bytes: 1,
            allocation_alignment: 4,
        };
        assert!(b.total_host_reserved_bytes().is_err());
    }
}
