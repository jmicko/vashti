//! Bounded experimental actual-score GPU routing for one count1..3 MAX8 wave.
//! No model/source authorization, allocation, readback or publication occurs
//! here. Parent freezes actual original expert↔slot and six complete physical
//! owners on one stream until deferred CPU audit/all-family completion. A miss,
//! nonfinite/range score or dirty owner sets sticky fault without MAP/PROB/
//! AUDIT publication. GPU selected-softmax F32 is explicitly different from
//! the original full-softmax F64→F32 router; audit requires exact original IDs
//! and separately declared probability bounds, never bit-equivalence.
//! Near ties are provisional: exact original IDs/order must pass the deferred
//! actual-score audit before any all-family commit. Rust exp
//! precision is unspecified and WGSL allows exp error; see:
//! https://doc.rust-lang.org/std/primitive.f64.html#method.exp
//! https://gpuweb.github.io/gpuweb/wgsl/#accuracy-of-floating-point-operations

use cubecl::ir::{ElemType, StorageType, UIntKind};
use cubecl::prelude::{ComputeClient, CubeCount, CubeDim};
use cubecl::wgpu::{AutoCompiler, WgpuRuntime};
use cubecl_runtime::compiler::{CompilationError, CubeTask};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
use cubecl_runtime::server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo};
use kuno_text::{Error, Result};
use zeroize::Zeroizing;

pub(crate) const EXPERTS: usize = 512;
pub(crate) const LAYERS: usize = 48;
pub(crate) const SLOTS: usize = 144;
pub(crate) const MAX_COUNT: usize = 4;
pub(crate) const RANKS: usize = 10;
pub(crate) const MAX_ROUTES: usize = 80;
pub(crate) const ROUTER_WORDS: usize = 8192;
pub(crate) const TABLE_WORDS: usize = 32768;
pub(crate) const MAP_WORDS: usize = 804;
pub(crate) const PROB_WORDS: usize = 80;
pub(crate) const SELECTION_WORDS: usize = 244;
pub(crate) const META_WORDS: usize = 18;
pub(crate) const FRAME_WORDS: usize =
    ROUTER_WORDS + SELECTION_WORDS + MAP_WORDS + PROB_WORDS + META_WORDS;
pub(crate) const AUDIT_WORDS: usize = LAYERS * FRAME_WORDS;
pub(crate) const AUDIT_OWNED_WORDS: usize = 524288;
pub(crate) const PREFLIGHT_WORKGROUPS: u32 = (AUDIT_OWNED_WORDS - AUDIT_WORDS).div_ceil(256) as u32;
pub(crate) const OWNED_BYTES: [u64; 6] = [32768, 131072, 32768, 32768, 2097152, 32768];
pub(crate) const MAIN_BYTES: u64 = 2359296;
pub(crate) const NEW_TABLE_AUDIT_BYTES: u64 = 2228224;
pub(crate) const INFO_CALLS_PER_LAYER: usize = 2;
pub(crate) const INFO_BYTES_PER_LAYER: u64 = INFO_CALLS_PER_LAYER as u64 * 2048;
pub(crate) const INFO_BYTES_PER_CHUNK: u64 = LAYERS as u64 * INFO_BYTES_PER_LAYER;
pub(crate) const NEAR_TIE_MARGIN: f32 = 0.;
pub(crate) const PROB_ABS: f64 = 2e-6;
pub(crate) const PROB_REL: f64 = 2e-5;
pub(crate) const FAULT_SHAPE: u32 = 1;
pub(crate) const FAULT_MISSING: u32 = 4;
pub(crate) const FAULT_NUMERIC: u32 = 2;
const SELECTION: usize = ROUTER_WORDS;
const MAP: usize = SELECTION + SELECTION_WORDS;
const PROB: usize = MAP + MAP_WORDS;
const META: usize = PROB + PROB_WORDS;
const MAGIC: u32 = 0x4b425231;
const VERSION: u32 = 2;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Addressing {
    Resident144,
    FrozenHost512,
}
impl Addressing {
    fn capacity(self) -> usize {
        match self {
            Self::Resident144 => SLOTS,
            Self::FrozenHost512 => EXPERTS,
        }
    }
    fn shader(self) -> String {
        let original = include_str!("resident_batch_router_gpu_four.wgsl");
        match self {
            Self::Resident144 => original.into(),
            Self::FrozenHost512 => original.replace(
                "const SOURCE_CAPACITY:u32=144u;",
                "const SOURCE_CAPACITY:u32=512u;",
            ),
        }
    }
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Kernel {
    preflight: bool,
    addressing: Addressing,
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if self.preflight {
            "ResidentBatchRouterFourPreflight"
        } else {
            "ResidentBatchRouterFourMapAudit"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(if self.preflight { 256 } else { 64 }))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: if self.preflight { "preflight" } else { "main" }.into(),
            debug_name: None,
            source: self.addressing.shader(),
            repr: None,
            cube_dim: CubeDim::new_1d(if self.preflight { 256 } else { 64 }),
            debug_info: None,
        })
    }
}
fn coordinates(layer: usize, start: u64, count: usize) -> Result<()> {
    if layer >= LAYERS
        || count != MAX_COUNT
        || start.checked_add(count as u64).is_none_or(|end| end > 4096)
    {
        return Err(Error::InvalidFormat("batch device router coordinates"));
    }
    Ok(())
}
pub(crate) fn validate_owners(
    owners: &[Handle; 6],
    layer: usize,
    start: u64,
    count: usize,
) -> Result<()> {
    coordinates(layer, start, count)?;
    for (h, bytes) in owners.iter().zip(OWNED_BYTES) {
        if h.stream != owners[0].stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != bytes
            || h.size_in_used() != bytes
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    Ok(())
}
/// Validate actual full table and padding before parent uploads/freezes it.
/// Partial residency uses UINT_MAX, never an invented or replacement expert.
pub(crate) fn validate_table(table: &[u32]) -> Result<()> {
    validate_table_with_addressing(table, Addressing::Resident144)
}
pub(crate) fn validate_frozen_source_table(table: &[u32]) -> Result<()> {
    validate_table_with_addressing(table, Addressing::FrozenHost512)
}
fn validate_table_with_addressing(table: &[u32], addressing: Addressing) -> Result<()> {
    if table.len() != TABLE_WORDS || table[LAYERS * EXPERTS..].iter().any(|&x| x != 0) {
        return Err(Error::InvalidFormat("batch device router table owner"));
    }
    let mut seen = Zeroizing::new([false; EXPERTS]);
    for row in table[..LAYERS * EXPERTS].chunks_exact(EXPERTS) {
        seen.fill(false);
        for &slot in row {
            if slot == u32::MAX && addressing == Addressing::Resident144 {
                continue;
            }
            if slot as usize >= addressing.capacity() || seen[slot as usize] {
                return Err(Error::InvalidFormat("batch device router table slots"));
            }
            seen[slot as usize] = true;
        }
    }
    Ok(())
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    if limits.max_storage_buffer_binding_size < OWNED_BYTES[4]
        || limits.max_buffer_size < OWNED_BYTES[4]
        || limits.max_storage_buffers_per_shader_stage < 6
        || limits.max_bindings_per_bind_group < 7
        || limits.max_bind_groups < 1
        || limits.max_uniform_buffer_binding_size < 32
        || limits.max_uniform_buffers_per_shader_stage < 1
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroups_per_dimension < PREFLIGHT_WORKGROUPS
        || limits.max_compute_workgroup_storage_size < 8192
    {
        return Err(Error::InvalidFormat("batch device router runtime limits"));
    }
    Ok(())
}
/// Six complete owners: actual MAX8 logits, frozen48×512 table, MAP, PROB,
/// deferred48-frame audit, original shared whole-chunk fault. MAP/PROB/current
/// frame and all tails must be zero. Caller proves physical disjointness/client
/// and immutable bank/source identity. Two32-byte INFO dispatches must both be
/// statically charged, even when a sticky fault suppresses publication.
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 6],
    layer: usize,
    start: u64,
    count: usize,
) -> Result<()> {
    launch_with_addressing(client, owners, layer, start, count, Addressing::Resident144)
}
/// Same actual-score/numeric operator, but virtual source addresses are the
/// authenticated complete original512 mapping, never BankSlots capacity512.
pub(crate) fn launch_frozen_sources(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 6],
    layer: usize,
    start: u64,
    count: usize,
) -> Result<()> {
    launch_with_addressing(
        client,
        owners,
        layer,
        start,
        count,
        Addressing::FrozenHost512,
    )
}
fn launch_with_addressing(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 6],
    layer: usize,
    start: u64,
    count: usize,
    addressing: Addressing,
) -> Result<()> {
    validate_owners(&owners, layer, start, count)?;
    let arguments = || {
        let mut args = KernelArguments::new();
        for h in &owners {
            args = args.with_buffer(h.clone().binding());
        }
        args.with_info(MetadataBindingInfo::custom(vec![
            layer as u64 | (start << 32),
            count as u64 | (u64::from(NEAR_TIE_MARGIN.to_bits()) << 32),
            u64::from(VERSION),
            0,
        ]))
    };
    // SAFETY: checked full extents and coordinates are identical across the
    // ordered phases. Parent retains all physical/source/epoch proofs. Failed
    // preflight can only set fault; mapper does not publish through that fault.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel {
                preflight: true,
                addressing,
            }),
            CubeCount::Static(PREFLIGHT_WORKGROUPS, 1, 1),
            arguments(),
        );
        client.launch_unchecked(
            Box::new(Kernel {
                preflight: false,
                addressing,
            }),
            CubeCount::Static(1, 1, 1),
            arguments(),
        );
    }
    Ok(())
}

use super::{AuditedRoutes, Routes};
fn word(bytes: &[u8], index: usize) -> u32 {
    u32::from_le_bytes(
        bytes[index * 4..index * 4 + 4]
            .try_into()
            .expect("bounded audit word"),
    )
}
/// Deferred original router comparison from the actual copied GPU score bits.
/// All2MiB audit padding and128KiB table padding are checked on every call;
/// parent additionally requires every48 expected frame exactly once and the
/// original whole-owner fault/physical/readback proof before state publication.
pub(crate) fn audit_frame(
    bytes: &[u8],
    table: &[u32],
    layer: usize,
    start: u64,
    count: usize,
) -> Result<AuditedRoutes> {
    audit_frame_with_addressing(
        bytes,
        table,
        layer,
        start,
        count,
        Addressing::Resident144,
        None,
    )
}
pub(crate) fn audit_frame_frozen_sources(
    bytes: &[u8],
    table: &[u32],
    layer: usize,
    start: u64,
    count: usize,
    geometry: crate::quant_matvec::grouped_moe::Geometry,
) -> Result<AuditedRoutes> {
    if !geometry.frozen_host_sources() {
        return Err(Error::InvalidFormat(
            "batch device router frozen source geometry",
        ));
    }
    geometry
        .validate_count(count)
        .map_err(Error::InvalidFormat)?;
    audit_frame_with_addressing(
        bytes,
        table,
        layer,
        start,
        count,
        Addressing::FrozenHost512,
        Some(geometry),
    )
}
fn audit_frame_with_addressing(
    bytes: &[u8],
    table: &[u32],
    layer: usize,
    start: u64,
    count: usize,
    addressing: Addressing,
    geometry: Option<crate::quant_matvec::grouped_moe::Geometry>,
) -> Result<AuditedRoutes> {
    coordinates(layer, start, count)?;
    match addressing {
        Addressing::Resident144 => validate_table(table)?,
        Addressing::FrozenHost512 => validate_frozen_source_table(table)?,
    }
    if bytes.len() != AUDIT_OWNED_WORDS * 4 || bytes[AUDIT_WORDS * 4..].iter().any(|&x| x != 0) {
        return Err(Error::InvalidFormat("batch device router audit owner"));
    }
    let base = layer * FRAME_WORDS;
    let mut routes = AuditedRoutes(Zeroizing::new(Routes {
        count,
        groups: 0,
        experts: [0; MAX_ROUTES],
        probabilities: [0.; MAX_ROUTES],
        pairs: [(0, 0); 40],
        probability_bit_mismatches: 0,
        max_probability_abs_error: 0.,
        minimum_gap: f32::MAX,
    }));
    let mut scores = Zeroizing::new([0.; EXPERTS]);
    let mut raw = Zeroizing::new([0usize; EXPERTS]);
    let mut metadata = Zeroizing::new([0u32; META_WORDS]);
    metadata[..8].copy_from_slice(&[
        MAGIC,
        VERSION,
        layer as u32,
        start as u32,
        count as u32,
        addressing.capacity() as u32,
        0,
        0,
    ]);
    metadata[17] = NEAR_TIE_MARGIN.to_bits();
    for row in 0..count {
        for id in 0..EXPERTS {
            scores[id] = f32::from_bits(word(bytes, base + row * EXPERTS + id));
            if !scores[id].is_finite() {
                return Err(Error::InvalidFormat("batch device router actual scores"));
            }
            raw[id] = id;
        }
        raw.sort_unstable_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .expect("checked finite scores")
                .then(a.cmp(&b))
        });
        metadata[8 + row * 2] = raw[10] as u32;
        metadata[9 + row * 2] = scores[raw[10]].to_bits();
        for rank in 0..RANKS {
            let gap = scores[raw[rank]] - scores[raw[rank + 1]];
            if !gap.is_finite()
                || gap < NEAR_TIE_MARGIN
                || scores[raw[rank]] - scores[raw[0]] < -80.
            {
                return Err(Error::InvalidFormat("batch device router finite range"));
            }
            routes.0.minimum_gap = routes.0.minimum_gap.min(gap);
        }
        let expected = kuno_text::ops::router_topk_guarded(&scores[..], RANKS, true)
            .map_err(|_| Error::InvalidFormat("batch device router original numeric selection"))?;
        for (rank, &(expected_id, expected_prob)) in expected.as_slice().iter().enumerate() {
            let route = row * RANKS + rank;
            let id = word(bytes, base + SELECTION + 4 + route * 3);
            let bits = word(bytes, base + SELECTION + 5 + route * 3);
            let prob = f32::from_bits(bits);
            if id != expected_id
                || id != raw[rank] as u32
                || word(bytes, base + SELECTION + 6 + route * 3) != scores[id as usize].to_bits()
                || !prob.is_finite()
                || prob <= 0.
                || prob > 1.
            {
                return Err(Error::InvalidFormat(
                    "batch device router original IDs/selection",
                ));
            }
            let error = (f64::from(prob) - f64::from(expected_prob)).abs();
            if error > PROB_ABS + PROB_REL * f64::from(expected_prob).abs() {
                return Err(Error::InvalidFormat(
                    "batch device router probability arithmetic",
                ));
            }
            routes.0.probability_bit_mismatches += usize::from(bits != expected_prob.to_bits());
            routes.0.max_probability_abs_error = routes.0.max_probability_abs_error.max(error);
            routes.0.experts[route] = id;
            routes.0.probabilities[route] = prob;
            if !routes.pairs().iter().any(|&(expert, _)| expert == id) {
                let slot = table[layer * EXPERTS + id as usize];
                if slot as usize >= addressing.capacity()
                    || routes.pairs().iter().any(|&(_, old)| old == slot)
                {
                    return Err(Error::InvalidFormat(
                        "batch device router actual missing slot",
                    ));
                }
                let group = routes.0.groups;
                routes.0.pairs[group] = (id, slot);
                routes.0.groups += 1;
            }
        }
    }
    for index in count * EXPERTS..ROUTER_WORDS {
        if word(bytes, base + index) != 0 {
            return Err(Error::InvalidFormat(
                "batch device router source inactive/padding",
            ));
        }
    }
    metadata[6] = routes.groups() as u32;
    metadata[16] = routes.minimum_gap().to_bits();
    for (i, &expected) in metadata.iter().enumerate() {
        if word(bytes, base + META + i) != expected {
            return Err(Error::InvalidFormat("batch device router audit metadata"));
        }
    }
    let header = [
        count as u32,
        0,
        (count * RANKS) as u32,
        routes.groups() as u32,
    ];
    for (i, &expected) in header.iter().enumerate() {
        if word(bytes, base + SELECTION + i) != expected {
            return Err(Error::InvalidFormat("batch device router selection header"));
        }
    }
    for i in 4 + count * RANKS * 3..SELECTION_WORDS {
        if word(bytes, base + SELECTION + i) != 0 {
            return Err(Error::InvalidFormat("batch device router selection tail"));
        }
    }
    let canonical = match addressing {
        Addressing::Resident144 => crate::quant_matvec::grouped_moe::GroupMap::new(
            count,
            0,
            count,
            &routes.experts()[..count * RANKS],
            &routes.probabilities()[..count * RANKS],
            routes.pairs(),
            SLOTS,
        ),
        Addressing::FrozenHost512 => {
            crate::quant_matvec::grouped_moe::GroupMap::new_frozen_sources(
                count,
                &routes.experts()[..count * RANKS],
                &routes.probabilities()[..count * RANKS],
                routes.pairs(),
                geometry.ok_or(Error::InvalidFormat(
                    "batch device router frozen geometry missing",
                ))?,
            )
        }
    }
    .map_err(|_| Error::InvalidFormat("batch device router canonical map"))?;
    for (i, &expected) in canonical.words().iter().enumerate() {
        if word(bytes, base + MAP + i) != expected {
            return Err(Error::InvalidFormat("batch device router consumed map"));
        }
    }
    for (i, expected) in canonical.probabilities().iter().enumerate() {
        if word(bytes, base + PROB + i) != expected.to_bits() {
            return Err(Error::InvalidFormat(
                "batch device router consumed probability",
            ));
        }
    }
    drop(canonical);
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_matvec::grouped_moe::{Geometry, GroupMap, Projection};
    fn geometry(layer: usize) -> Geometry {
        let [gate, up, down] = crate::daslab_profile::routed_types(layer).unwrap();
        Geometry::new(
            [
                Projection::new(gate, 2560, 640, 512).unwrap(),
                Projection::new(up, 2560, 640, 512).unwrap(),
                Projection::new(down, 640, 2560, 512).unwrap(),
            ],
            8,
            144,
        )
        .unwrap()
        .with_frozen_host_sources()
        .unwrap()
        .with_inactive_partition_exit()
        .unwrap()
        .with_four_members()
        .unwrap()
    }
    fn table() -> Zeroizing<Vec<u32>> {
        let mut out = Zeroizing::new(vec![0; TABLE_WORDS]);
        for layer in 0..LAYERS {
            for expert in 0..EXPERTS {
                out[layer * EXPERTS + expert] = ((expert * 31 + layer * 7) % 512) as u32;
            }
        }
        out
    }
    // Independent actual-score fixture: original core router returns its IDs
    // and probabilities; the original canonical map builds consumed wires.
    // No fixture carries a forced expert choice into a model or GPU request.
    fn fixture(
        layer: usize,
        start: u64,
        count: usize,
        ties: bool,
    ) -> (Zeroizing<Vec<u8>>, Zeroizing<Vec<u32>>) {
        let table = table();
        let mut words = Zeroizing::new(vec![0u32; AUDIT_OWNED_WORDS]);
        let base = layer * FRAME_WORDS;
        let boundary = [0usize, 143, 144, 287, 288, 431, 432, 511, 71, 72, 17];
        let mut experts = Zeroizing::new([0u32; MAX_ROUTES]);
        let mut probabilities = Zeroizing::new([0f32; MAX_ROUTES]);
        let mut pairs = Zeroizing::new(Vec::with_capacity(40));
        let mut meta = Zeroizing::new([0u32; META_WORDS]);
        meta[..8].copy_from_slice(&[
            MAGIC,
            VERSION,
            layer as u32,
            start as u32,
            count as u32,
            512,
            0,
            0,
        ]);
        meta[17] = NEAR_TIE_MARGIN.to_bits();
        let mut minimum_gap = f32::MAX;
        for row in 0..count {
            let mut scores = Zeroizing::new(std::array::from_fn::<_, 512, _>(|id| {
                -8. - id as f32 / 1024.
            }));
            if ties {
                for id in 0..12 {
                    scores[id] = if id & 1 == 0 { 0. } else { -0. };
                }
            } else {
                for (rank, &id) in boundary.iter().enumerate() {
                    let id = (id + row * 37) % 512;
                    scores[id] = 8. - rank as f32 / 8.;
                }
            }
            for (i, &score) in scores.iter().enumerate() {
                words[base + row * EXPERTS + i] = score.to_bits();
            }
            let selected = kuno_text::ops::router_topk_guarded(&*scores, RANKS, true).unwrap();
            let mut raw = Zeroizing::new(std::array::from_fn::<_, 512, _>(|id| id));
            raw.sort_unstable_by(|&a, &b| {
                scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b))
            });
            meta[8 + row * 2] = raw[10] as u32;
            meta[9 + row * 2] = scores[raw[10]].to_bits();
            for rank in 0..RANKS {
                minimum_gap = minimum_gap.min(scores[raw[rank]] - scores[raw[rank + 1]]);
                let (id, probability) = selected.as_slice()[rank];
                assert_eq!(id as usize, raw[rank]);
                let route = row * RANKS + rank;
                experts[route] = id;
                probabilities[route] = probability;
                if !pairs.iter().any(|&(old, _)| old == id) {
                    pairs.push((id, table[layer * EXPERTS + id as usize]));
                }
                words[base + SELECTION + 4 + route * 3] = id;
                words[base + SELECTION + 5 + route * 3] = probability.to_bits();
                words[base + SELECTION + 6 + route * 3] = scores[id as usize].to_bits();
            }
        }
        let map = GroupMap::new_frozen_sources(
            count,
            &experts[..count * RANKS],
            &probabilities[..count * RANKS],
            &pairs,
            geometry(layer),
        )
        .unwrap();
        words[base + MAP..base + MAP + MAP_WORDS].copy_from_slice(map.words());
        for (i, p) in map.probabilities().iter().enumerate() {
            words[base + PROB + i] = p.to_bits();
        }
        words[base + SELECTION..base + SELECTION + 4].copy_from_slice(&[
            count as u32,
            0,
            (count * RANKS) as u32,
            pairs.len() as u32,
        ]);
        meta[6] = pairs.len() as u32;
        meta[16] = minimum_gap.to_bits();
        words[base + META..base + META + META_WORDS].copy_from_slice(&*meta);
        let mut bytes = Zeroizing::new(Vec::with_capacity(words.len() * 4));
        for &word in words.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        (bytes, table)
    }
    #[test]
    fn four_router_full_source_owner_meta18_original_rank_wires() {
        assert_eq!((META_WORDS, FRAME_WORDS, AUDIT_WORDS), (18, 9338, 448224));
        for layer in [0, 23, 47] {
            for ties in [false, true] {
                let (mut bytes, table) = fixture(layer, 200, 4, ties);
                let routes =
                    audit_frame_frozen_sources(&bytes, &table, layer, 200, 4, geometry(layer))
                        .unwrap();
                assert_eq!(
                    (routes.count(), routes.probability_bit_mismatches()),
                    (4, 0)
                );
                assert!(routes.groups() <= 40);
                let base = layer * FRAME_WORDS;
                for offset in [
                    META + 14,
                    META + 15,
                    META + 16,
                    META + 17,
                    SELECTION + 4 + 39 * 3,
                    MAP + 803,
                    PROB + 79,
                    ROUTER_WORDS - 1,
                ] {
                    let first = (base + offset) * 4;
                    let original = bytes[first..first + 4].to_vec();
                    let corrupted = word(&bytes, base + offset) ^ 1;
                    bytes[first..first + 4].copy_from_slice(&corrupted.to_le_bytes());
                    assert!(
                        audit_frame_frozen_sources(&bytes, &table, layer, 200, 4, geometry(layer))
                            .is_err()
                    );
                    bytes[first..first + 4].copy_from_slice(&original);
                }
                assert!(
                    super::super::audit_frame_frozen_sources(
                        &bytes,
                        &table,
                        layer,
                        200,
                        3,
                        geometry(layer)
                    )
                    .is_err()
                );
                bytes[AUDIT_WORDS * 4] = 1;
                assert!(
                    audit_frame_frozen_sources(&bytes, &table, layer, 200, 4, geometry(layer))
                        .is_err()
                );
            }
        }
    }
    #[test]
    fn four_router_shader_keeps_typed_metadata_and_uniform_full_owner_gate() {
        let source = Addressing::FrozenHost512.shader();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        for entry in &module.entry_points {
            wgpu::naga::back::spv::write_vec(
                &module,
                &info,
                &wgpu::naga::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&wgpu::naga::back::spv::PipelineOptions {
                    shader_stage: wgpu::naga::ShaderStage::Compute,
                    entry_point: entry.name.clone(),
                }),
            )
            .unwrap();
        }
        assert!(source.contains("9338u"));
        assert!(source.contains("448224u"));
        assert_ne!(
            Kernel {
                addressing: Addressing::FrozenHost512,
                preflight: false
            }
            .id(),
            super::super::Kernel {
                addressing: super::super::Addressing::FrozenHost512,
                preflight: false
            }
            .id()
        );
    }
}
