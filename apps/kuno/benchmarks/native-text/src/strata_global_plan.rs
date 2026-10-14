//! MIT Strata fb58e0 native-pack sized cache/global profile placement.
//! Public model setup only: no request routes, scores, tokens or persisted state.
//! A ranked prefix is chosen in bytes, then every role gets bounded physical
//! pieces. Consumers must use this SAME plan for residency and host complements.
use kuno_text::{
    Error, Result,
    model::{BoundGraph, GraphBindingId},
};

pub(crate) const LAYERS: usize = 48;
pub(crate) const EXPERTS: usize = 512;
pub(crate) const RANKED_PAIRS: usize = LAYERS * EXPERTS;
pub(crate) const ALIGNMENT: u64 = 256;
pub(crate) const MAX_PIECE_BYTES: u64 = 128 * 1024 * 1024;
// Original constructor's complete source/target/inspection envelope. Keep one
// public upload window bounded instead of allocating a whole-model RAM mirror.
pub(crate) const MAX_GPU_CHUNK_BYTES: u64 = 16 * 1024 * 1024;
const PROFILE: &[u8] = include_bytes!("../fixtures/strata-bank-profile/expert-profile.bin");
fn reject() -> Error {
    Error::InvalidFormat("Strata global expert allocation")
}
fn align(bytes: u64, alignment: u64) -> Result<u64> {
    if bytes == 0 || !alignment.is_power_of_two() {
        return Err(reject());
    }
    bytes
        .checked_add(alignment - 1)
        .map(|n| n & !(alignment - 1))
        .ok_or(Error::ArithmeticOverflow)
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RankedPair {
    pub layer: usize,
    pub expert: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Slot {
    pub pair: RankedPair,
    pub offset: u64,
    pub bytes: u64,
    pub aligned_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Resident,
    Host,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Piece {
    pub tier: Tier,
    pub first: usize,
    pub count: usize,
    pub expert_bytes: u64,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Address {
    pub tier: Tier,
    pub virtual_slot: usize,
    pub piece: usize,
    pub local: usize,
    pub offset: u64,
    pub bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Chunk {
    pub first_slot: usize,
    pub slots: usize,
    pub logical_bytes: u64,
    pub physical_bytes: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cell {
    pub chunk: usize,
    pub offset: u64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    Resident {
        chunk: usize,
        offset: u64,
        bytes: u64,
    },
    Host {
        piece: usize,
        offset: u64,
        bytes: u64,
    },
}
pub(crate) struct Layer {
    pub resident: usize,
    pub role_expert_bytes: [u64; 3],
    /// Resident pair order is exactly its order in the selected global prefix.
    /// Missing experts follow in ID order; every expert has one source address.
    by_virtual: Vec<u16>,
    by_expert: Vec<u16>,
    pub roles: [Vec<Piece>; 3],
    resident_slots: Vec<usize>,
}
impl Layer {
    pub(crate) fn resident_ids(&self) -> &[u16] {
        &self.by_virtual[..self.resident]
    }
    pub(crate) fn expert_at(&self, slot: usize) -> Result<usize> {
        self.by_virtual
            .get(slot)
            .copied()
            .map(usize::from)
            .ok_or_else(reject)
    }
    pub(crate) fn address(&self, expert: usize, role: usize) -> Result<Address> {
        let slot = usize::from(*self.by_expert.get(expert).ok_or_else(reject)?);
        let pieces = self.roles.get(role).ok_or_else(reject)?;
        let (piece, p) = pieces
            .iter()
            .enumerate()
            .find(|(_, p)| slot >= p.first && slot - p.first < p.count)
            .ok_or_else(reject)?;
        let local = slot - p.first;
        Ok(Address {
            tier: p.tier,
            virtual_slot: slot,
            piece,
            local,
            offset: (local as u64)
                .checked_mul(p.expert_bytes)
                .ok_or(Error::ArithmeticOverflow)?,
            bytes: p.expert_bytes,
        })
    }
}
static NEXT_PLAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
pub(crate) struct Plan {
    instance: u64,
    binding: Option<GraphBindingId>,
    pub layers: Vec<Layer>,
    /// Strata's arena positions have heterogeneous aligned blob sizes; retaining
    /// their prefix order is needed for pointer-table and global cache semantics.
    pub slots: Vec<Slot>,
    pub chunks: Vec<Chunk>,
    pub cells: Vec<Cell>,
    pub budget: u64,
    pub arena_bytes: u64,
    pub resident_logical_bytes: u64,
    pub resident_piece_bytes: u64,
    pub host_logical_bytes: u64,
    pub host_piece_bytes: u64,
}
fn ranked_profile() -> Result<Vec<RankedPair>> {
    if PROFILE.len() != 24 + RANKED_PAIRS * 4 + RANKED_PAIRS * 4 || &PROFILE[..4] != b"STRP" {
        return Err(reject());
    }
    let word = |at| u32::from_le_bytes(PROFILE[at..at + 4].try_into().unwrap()) as usize;
    if [word(4), word(8), word(12), word(16), word(20)]
        != [1, LAYERS, EXPERTS, RANKED_PAIRS, RANKED_PAIRS]
    {
        return Err(reject());
    }
    let mut result = Vec::with_capacity(RANKED_PAIRS);
    let mut seen = vec![false; RANKED_PAIRS];
    for rank in 0..RANKED_PAIRS {
        let at = 24 + rank * 4;
        let layer = u16::from_le_bytes(PROFILE[at..at + 2].try_into().unwrap()) as usize;
        let expert = u16::from_le_bytes(PROFILE[at + 2..at + 4].try_into().unwrap()) as usize;
        if layer >= LAYERS
            || expert >= EXPERTS
            || seen[layer * EXPERTS + expert]
            || word(24 + RANKED_PAIRS * 4 + (layer * EXPERTS + expert) * 4) != rank
        {
            return Err(reject());
        }
        seen[layer * EXPERTS + expert] = true;
        result.push(RankedPair { layer, expert });
    }
    Ok(result)
}
fn pieces(resident: usize, expert_bytes: u64, alignment: u64) -> Result<Vec<Piece>> {
    if resident > EXPERTS || expert_bytes == 0 || expert_bytes > MAX_PIECE_BYTES {
        return Err(reject());
    }
    let mut result = Vec::new();
    // Vulkan descriptor spans are bounded. Empty tiers have NO invented dummy
    // weight allocation; BDA consumers must tolerate an absent tier/piece.
    for (tier, start, end) in [(Tier::Host, resident, EXPERTS)] {
        let max = (MAX_PIECE_BYTES / expert_bytes) as usize;
        let mut first = start;
        while first < end {
            let count = (end - first).min(max);
            let logical_bytes = expert_bytes
                .checked_mul(count as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            result.push(Piece {
                tier,
                first,
                count,
                expert_bytes,
                logical_bytes,
                physical_bytes: align(logical_bytes, alignment)?,
            });
            first += count;
        }
    }
    Ok(result)
}
fn arena_chunks(slots: &[Slot], alignment: u64) -> Result<(Vec<Chunk>, Vec<Cell>, u64)> {
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut cells = Vec::with_capacity(slots.len());
    let mut total = 0u64;
    for (index, slot) in slots.iter().enumerate() {
        if slot.aligned_bytes > MAX_GPU_CHUNK_BYTES {
            return Err(reject());
        }
        if chunks
            .last()
            .is_none_or(|c| c.logical_bytes + slot.aligned_bytes > MAX_GPU_CHUNK_BYTES)
        {
            chunks.push(Chunk {
                first_slot: index,
                slots: 0,
                logical_bytes: 0,
                physical_bytes: 0,
            });
        }
        let chunk_index = chunks.len() - 1;
        let c = chunks.last_mut().unwrap();
        cells.push(Cell {
            chunk: chunk_index,
            offset: c.logical_bytes,
        });
        c.logical_bytes += slot.aligned_bytes;
        c.slots += 1;
    }
    for c in &mut chunks {
        c.physical_bytes = align(c.logical_bytes, alignment)?;
        total = total
            .checked_add(c.physical_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
    }
    Ok((chunks, cells, total))
}
impl Plan {
    pub(crate) fn instance(&self) -> u64 {
        self.instance
    }
    pub(crate) fn from_graph(
        graph: BoundGraph<'_, '_>,
        budget: u64,
        alignment: u64,
    ) -> Result<Self> {
        graph.verify_unchanged()?;
        if graph.geometry().layers != LAYERS || graph.geometry().experts != EXPERTS {
            return Err(reject());
        }
        let mut roles = [[0u64; 3]; LAYERS];
        for (layer, bytes) in roles.iter_mut().enumerate() {
            let expected = crate::daslab_profile::routed_types(layer).ok_or_else(reject)?;
            for (role, name) in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]
            .into_iter()
            .enumerate()
            {
                let t = graph.tensor(&format!("blk.{layer}.{name}"))?;
                let d = t.info().dimensions();
                let dimensions = if role == 2 {
                    [640, 2560, 512]
                } else {
                    [2560, 640, 512]
                };
                if d != dimensions || t.info().dtype().gguf_id() != expected[role] {
                    return Err(reject());
                }
                bytes[role] = t
                    .info()
                    .row_bytes()
                    .checked_mul(dimensions[1])
                    .ok_or(Error::ArithmeticOverflow)?;
                if bytes[role].checked_mul(EXPERTS as u64) != Some(t.info().byte_len()) {
                    return Err(reject());
                }
            }
        }
        let mut plan = Self::from_roles(roles, budget, alignment)?;
        plan.binding = Some(graph.binding_id());
        Ok(plan)
    }
    pub(crate) fn check_binding(&self, graph: BoundGraph<'_, '_>) -> Result<()> {
        graph.verify_unchanged()?;
        if self.binding != Some(graph.binding_id()) {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn uniform_budget(graph: BoundGraph<'_, '_>, slots: usize) -> Result<u64> {
        if slots > EXPERTS {
            return Err(reject());
        }
        let mut total = 0u64;
        for layer in 0..LAYERS {
            for name in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                let t = graph.tensor(&format!("blk.{layer}.{name}"))?;
                let d = t.info().dimensions();
                let expected = if name == "ffn_down_exps.weight" {
                    [640, 2560, 512]
                } else {
                    [2560, 640, 512]
                };
                if d != expected {
                    return Err(reject());
                }
                let bytes = t
                    .info()
                    .row_bytes()
                    .checked_mul(t.info().dimensions()[1])
                    .and_then(|n| n.checked_mul(slots as u64))
                    .ok_or(Error::ArithmeticOverflow)?;
                total = total.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
            }
        }
        Ok(total)
    }
    fn from_roles(roles: [[u64; 3]; LAYERS], budget: u64, alignment: u64) -> Result<Self> {
        if !alignment.is_power_of_two()
            || !(ALIGNMENT..=1024 * 1024).contains(&alignment)
            || roles
                .iter()
                .flatten()
                .any(|&b| b == 0 || !b.is_multiple_of(4) || b > 1024 * 1024)
        {
            return Err(reject());
        }
        let mut blobs = [0u64; LAYERS];
        for (blob, rs) in blobs.iter_mut().zip(roles) {
            for bytes in rs {
                *blob = blob.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
            }
        }
        let profile = ranked_profile()?;
        let mut slots = Vec::new();
        let mut resident_ids: Vec<Vec<u16>> = (0..LAYERS).map(|_| Vec::new()).collect();
        let mut resident_slots: Vec<Vec<usize>> = (0..LAYERS).map(|_| Vec::new()).collect();
        let mut arena_bytes = 0u64;
        for pair in profile {
            let bytes = blobs[pair.layer];
            let aligned_bytes = align(bytes, ALIGNMENT)?;
            let next = arena_bytes
                .checked_add(aligned_bytes)
                .ok_or(Error::ArithmeticOverflow)?;
            // Faithful prefix policy: stop at the first pair that does not fit,
            // rather than silently skipping a large/hot expert for smaller ones.
            if next > budget {
                break;
            }
            resident_slots[pair.layer].push(slots.len());
            slots.push(Slot {
                pair,
                offset: arena_bytes,
                bytes,
                aligned_bytes,
            });
            resident_ids[pair.layer].push(pair.expert as u16);
            arena_bytes = next;
        }
        let (chunks, cells, arena_piece_bytes) = arena_chunks(&slots, alignment)?;
        let mut layers = Vec::with_capacity(LAYERS);
        let (
            mut resident_logical_bytes,
            mut resident_piece_bytes,
            mut host_logical_bytes,
            mut host_piece_bytes,
        ) = (0u64, 0u64, 0u64, 0u64);
        for (layer, mut by_virtual) in resident_ids.into_iter().enumerate() {
            let resident = by_virtual.len();
            let mut by_expert = vec![u16::MAX; EXPERTS];
            for (slot, &id) in by_virtual.iter().enumerate() {
                by_expert[id as usize] = slot as u16;
            }
            for (expert, slot) in by_expert.iter_mut().enumerate() {
                if *slot == u16::MAX {
                    *slot = by_virtual.len() as u16;
                    by_virtual.push(expert as u16);
                }
            }
            let role_pieces = roles[layer].map(|bytes| pieces(resident, bytes, alignment));
            let [gate, up, down] = role_pieces;
            let role_pieces = [gate?, up?, down?];
            for p in role_pieces.iter().flatten() {
                let (logical, physical) = if p.tier == Tier::Resident {
                    (&mut resident_logical_bytes, &mut resident_piece_bytes)
                } else {
                    (&mut host_logical_bytes, &mut host_piece_bytes)
                };
                *logical = logical
                    .checked_add(p.logical_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                *physical = physical
                    .checked_add(p.physical_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
            }
            layers.push(Layer {
                resident_slots: std::mem::take(&mut resident_slots[layer]),
                resident,
                role_expert_bytes: roles[layer],
                by_virtual,
                by_expert,
                roles: role_pieces,
            });
        }
        resident_logical_bytes = slots.iter().try_fold(0u64, |n, s| {
            n.checked_add(s.bytes).ok_or(Error::ArithmeticOverflow)
        })?;
        resident_piece_bytes = arena_piece_bytes;
        if resident_logical_bytes > arena_bytes
            || arena_bytes > budget
            || resident_piece_bytes > budget
        {
            return Err(Error::LimitExceeded(
                "global expert pieces exceed admitted budget",
            ));
        }
        let instance = NEXT_PLAN
            .fetch_update(
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
                |n| n.checked_add(1),
            )
            .map_err(|_| Error::ArithmeticOverflow)?;
        Ok(Self {
            instance,
            binding: None,
            layers,
            slots,
            chunks,
            cells,
            budget,
            arena_bytes,
            resident_logical_bytes,
            resident_piece_bytes,
            host_logical_bytes,
            host_piece_bytes,
        })
    }
    pub(crate) fn source(&self, layer: usize, expert: usize, role: usize) -> Result<Source> {
        let l = self.layers.get(layer).ok_or_else(reject)?;
        let virtual_slot = usize::from(*l.by_expert.get(expert).ok_or_else(reject)?);
        let bytes = *l.role_expert_bytes.get(role).ok_or_else(reject)?;
        if virtual_slot < l.resident {
            let index = l.resident_slots[virtual_slot];
            let cell = self.cells[index];
            let role_first = l.role_expert_bytes[..role].iter().sum::<u64>();
            let offset = cell
                .offset
                .checked_add(role_first)
                .ok_or(Error::ArithmeticOverflow)?;
            if offset
                .checked_add(bytes)
                .is_none_or(|n| n > self.chunks[cell.chunk].logical_bytes)
            {
                return Err(reject());
            }
            Ok(Source::Resident {
                chunk: cell.chunk,
                offset,
                bytes,
            })
        } else {
            let a = l.address(expert, role)?;
            Ok(Source::Host {
                piece: a.piece,
                offset: a.offset,
                bytes: a.bytes,
            })
        }
    }
    /// Startup-only bounded original GGUF reads. The destination is one admitted
    /// PUBLIC upload window, not a decoded mirror or a second on-disk model.
    pub(crate) fn load_chunk(
        &self,
        graph: BoundGraph<'_, '_>,
        index: usize,
        destination: &mut [u8],
    ) -> Result<()> {
        self.check_binding(graph)?;
        let chunk = self.chunks.get(index).ok_or_else(reject)?;
        if destination.len() as u64 != chunk.physical_bytes {
            return Err(Error::InvalidBufferLength);
        }
        destination.fill(0);
        for slot_index in chunk.first_slot..chunk.first_slot + chunk.slots {
            let slot = self.slots[slot_index];
            let cell = self.cells[slot_index];
            let l = &self.layers[slot.pair.layer];
            if cell.chunk != index {
                return Err(reject());
            }
            let mut at = usize::try_from(cell.offset).map_err(|_| Error::ArithmeticOverflow)?;
            for (role, name) in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]
            .into_iter()
            .enumerate()
            {
                let t = graph.tensor(&format!("blk.{}.{name}", slot.pair.layer))?;
                let bytes = usize::try_from(l.role_expert_bytes[role])
                    .map_err(|_| Error::ArithmeticOverflow)?;
                let end = at.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
                let target = destination
                    .get_mut(at..end)
                    .ok_or(Error::InvalidBufferLength)?;
                t.read_range(
                    (slot.pair.expert as u64)
                        .checked_mul(bytes as u64)
                        .ok_or(Error::ArithmeticOverflow)?,
                    target,
                )?;
                at = end;
            }
        }
        graph.verify_unchanged()?;
        Ok(())
    }
    pub(crate) fn receipt(&self) -> serde_json::Value {
        serde_json::json!({"global_profile_prefix":true,"ranked_pairs":self.slots.len(),"budget_bytes":self.budget,"strata_arena_bytes":self.arena_bytes,"resident_logical_bytes":self.resident_logical_bytes,"resident_piece_bytes":self.resident_piece_bytes,"resident_chunks":self.chunks.len(),"public_load_window_bytes":self.chunks.iter().map(|c|c.physical_bytes).max().unwrap_or(0),"host_logical_bytes":self.host_logical_bytes,"host_piece_bytes":self.host_piece_bytes,"layer_resident_counts":self.layers.iter().map(|l|l.resident).collect::<Vec<_>>(),"gpu_root_consumers_connected":false,"request_routes_or_scores_used":false})
    }
}
/// Exact original profile metadata, also used by independent host-only fixtures.
#[cfg(test)]
fn pinned_roles() -> [[u64; 3]; LAYERS] {
    std::array::from_fn(|layer| {
        crate::daslab_profile::routed_types(layer)
            .unwrap()
            .map(|dtype| {
                let role = if matches!(dtype, 20 | 42) { 2 } else { 0 };
                let (columns, rows) = if role == 2 { (640, 2560) } else { (2560, 640) };
                kuno_text::TensorType::from_gguf_id(dtype)
                    .unwrap()
                    .row_bytes(columns)
                    .unwrap()
                    * rows
            })
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_byte_budget_preserves_real_global_prefix_and_complete_complement() {
        let roles = pinned_roles();
        let budget = roles.iter().flatten().sum::<u64>() * 240;
        let plan = Plan::from_roles(roles, budget, ALIGNMENT).unwrap();
        // Independently inspected original GGUF/header and pinned-profile result.
        assert_eq!(budget, 23_574_528_000);
        assert_eq!(plan.slots.len(), 11570);
        assert_eq!(plan.arena_bytes, 23_573_836_800);
        assert_eq!(
            plan.layers.iter().map(|l| l.resident).collect::<Vec<_>>(),
            vec![
                342, 299, 291, 176, 225, 240, 273, 277, 240, 242, 259, 252, 258, 279, 232, 160,
                187, 246, 275, 237, 244, 259, 281, 217, 246, 230, 268, 244, 261, 288, 254, 198,
                211, 251, 270, 243, 245, 235, 280, 200, 210, 217, 228, 181, 203, 204, 213, 199
            ]
        );
        let full = roles.iter().flatten().sum::<u64>() * 512;
        assert_eq!(plan.resident_logical_bytes + plan.host_logical_bytes, full);
        for (layer, l) in plan.layers.iter().enumerate() {
            let mut seen = [false; EXPERTS];
            for slot in 0..EXPERTS {
                let id = l.expert_at(slot).unwrap();
                assert!(!seen[id]);
                seen[id] = true;
                for role in 0..3 {
                    match plan.source(layer, id, role).unwrap() {
                        Source::Resident {
                            chunk,
                            offset,
                            bytes,
                        } => {
                            assert!(slot < l.resident);
                            assert_eq!(bytes, roles[layer][role]);
                            assert!(offset + bytes <= plan.chunks[chunk].logical_bytes);
                        }
                        Source::Host {
                            piece,
                            offset,
                            bytes,
                        } => {
                            assert!(slot >= l.resident);
                            assert_eq!(bytes, roles[layer][role]);
                            assert!(offset + bytes <= l.roles[role][piece].logical_bytes);
                        }
                    }
                }
            }
            assert!(
                l.roles
                    .iter()
                    .flatten()
                    .all(|p| p.logical_bytes <= MAX_PIECE_BYTES && p.count > 0)
            );
            let expected = ranked_profile()
                .unwrap()
                .into_iter()
                .take(plan.slots.len())
                .filter(|p| p.layer == layer)
                .map(|p| p.expert as u16)
                .collect::<Vec<_>>();
            assert_eq!(l.resident_ids(), expected);
        }
    }
    #[test]
    fn empty_partial_and_full_cache_never_invent_or_duplicate_a_weight_source() {
        let roles = pinned_roles();
        let first = ranked_profile().unwrap()[0];
        let bytes = roles[first.layer].iter().sum::<u64>();
        for budget in [0, bytes - 1, bytes, u64::MAX] {
            let p = Plan::from_roles(roles, budget, ALIGNMENT).unwrap();
            assert!(p.arena_bytes <= budget);
            for layer in &p.layers {
                for role in 0..3 {
                    assert_eq!(
                        layer.roles[role].iter().map(|p| p.count).sum::<usize>(),
                        EXPERTS - layer.resident
                    );
                }
            }
            if budget < bytes {
                assert!(p.slots.is_empty());
                assert_eq!(p.resident_piece_bytes, 0);
            }
            if budget == u64::MAX {
                assert_eq!(p.slots.len(), RANKED_PAIRS);
                assert_eq!(p.host_piece_bytes, 0);
            }
        }
    }
}
