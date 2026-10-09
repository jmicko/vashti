//! Bounded experimental actual-score GPU routing for one count1..3 MAX8 wave.
//! No model/source authorization, allocation, readback or publication occurs
//! here. Parent freezes actual original expert↔slot and six complete physical
//! owners on one stream until deferred CPU audit/all-family completion. A miss,
//! nonfinite/near-tie score or dirty owner sets sticky fault without MAP/PROB/
//! AUDIT publication. GPU selected-softmax F32 is explicitly different from
//! the original full-softmax F64→F32 router; audit requires exact original IDs
//! and separately declared probability bounds, never bit-equivalence.
//! Near-tie rejection is conservative, not an exp/rounding proof. Rust exp
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
use zeroize::{Zeroize, Zeroizing};

pub(crate) const EXPERTS: usize = 512;
pub(crate) const LAYERS: usize = 48;
pub(crate) const SLOTS: usize = 144;
pub(crate) const MAX_COUNT: usize = 3;
pub(crate) const RANKS: usize = 10;
pub(crate) const MAX_ROUTES: usize = 80;
pub(crate) const ROUTER_WORDS: usize = 8192;
pub(crate) const TABLE_WORDS: usize = 32768;
pub(crate) const MAP_WORDS: usize = 804;
pub(crate) const PROB_WORDS: usize = 80;
pub(crate) const SELECTION_WORDS: usize = 244;
pub(crate) const META_WORDS: usize = 16;
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
pub(crate) const NEAR_TIE_MARGIN: f32 = 1e-4;
pub(crate) const PROB_ABS: f64 = 2e-6;
pub(crate) const PROB_REL: f64 = 2e-5;
pub(crate) const FAULT_SHAPE: u32 = 1;
pub(crate) const FAULT_NUMERIC: u32 = 2;
pub(crate) const FAULT_MISSING: u32 = 4;
pub(crate) const FAULT_NEAR_TIE: u32 = 8;
const SELECTION: usize = ROUTER_WORDS;
const MAP: usize = SELECTION + SELECTION_WORDS;
const PROB: usize = MAP + MAP_WORDS;
const META: usize = PROB + PROB_WORDS;
const MAGIC: u32 = 0x4b425231;
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Kernel(bool);
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        if self.0 {
            "ResidentBatchRouterPreflight"
        } else {
            "ResidentBatchRouterMapAudit"
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(if self.0 { 256 } else { 64 }))
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
            entrypoint_name: if self.0 { "preflight" } else { "main" }.into(),
            debug_name: None,
            source: include_str!("resident_batch_router_gpu.wgsl").into(),
            repr: None,
            cube_dim: CubeDim::new_1d(if self.0 { 256 } else { 64 }),
            debug_info: None,
        })
    }
}
fn coordinates(layer: usize, start: u64, count: usize) -> Result<()> {
    if layer >= LAYERS
        || !(1..=MAX_COUNT).contains(&count)
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
    if table.len() != TABLE_WORDS || table[LAYERS * EXPERTS..].iter().any(|&x| x != 0) {
        return Err(Error::InvalidFormat("batch device router table owner"));
    }
    let mut seen = Zeroizing::new([false; SLOTS]);
    for row in table[..LAYERS * EXPERTS].chunks_exact(EXPERTS) {
        seen.fill(false);
        for &slot in row {
            if slot == u32::MAX {
                continue;
            }
            if slot as usize >= SLOTS || seen[slot as usize] {
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
            Box::new(Kernel(true)),
            CubeCount::Static(PREFLIGHT_WORKGROUPS, 1, 1),
            arguments(),
        );
        client.launch_unchecked(
            Box::new(Kernel(false)),
            CubeCount::Static(1, 1, 1),
            arguments(),
        );
    }
    Ok(())
}

struct Routes {
    count: usize,
    groups: usize,
    experts: [u32; MAX_ROUTES],
    probabilities: [f32; MAX_ROUTES],
    pairs: [(u32, u32); MAX_COUNT * RANKS],
    probability_bit_mismatches: usize,
    max_probability_abs_error: f64,
    minimum_gap: f32,
}
impl Zeroize for Routes {
    fn zeroize(&mut self) {
        self.count.zeroize();
        self.groups.zeroize();
        self.experts.zeroize();
        self.probabilities.zeroize();
        self.pairs.zeroize();
        self.probability_bit_mismatches.zeroize();
        self.max_probability_abs_error.zeroize();
        self.minimum_gap.zeroize();
    }
}
/// Bounded wiping result. Exact parent pending/slot identity and all-family
/// commit are still required; successful wire/numeric audit is not publication.
pub(crate) struct AuditedRoutes(Zeroizing<Routes>);
impl AuditedRoutes {
    pub(crate) fn count(&self) -> usize {
        self.0.count
    }
    pub(crate) fn groups(&self) -> usize {
        self.0.groups
    }
    pub(crate) fn experts(&self) -> &[u32; MAX_ROUTES] {
        &self.0.experts
    }
    pub(crate) fn probabilities(&self) -> &[f32; MAX_ROUTES] {
        &self.0.probabilities
    }
    pub(crate) fn pairs(&self) -> &[(u32, u32)] {
        &self.0.pairs[..self.0.groups]
    }
    pub(crate) fn probability_bit_mismatches(&self) -> usize {
        self.0.probability_bit_mismatches
    }
    pub(crate) fn max_probability_abs_error(&self) -> f64 {
        self.0.max_probability_abs_error
    }
    pub(crate) fn minimum_gap(&self) -> f32 {
        self.0.minimum_gap
    }
}
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
    coordinates(layer, start, count)?;
    validate_table(table)?;
    if bytes.len() != AUDIT_OWNED_WORDS * 4 || bytes[AUDIT_WORDS * 4..].iter().any(|&x| x != 0) {
        return Err(Error::InvalidFormat("batch device router audit owner"));
    }
    let base = layer * FRAME_WORDS;
    let mut routes = AuditedRoutes(Zeroizing::new(Routes {
        count,
        groups: 0,
        experts: [0; MAX_ROUTES],
        probabilities: [0.; MAX_ROUTES],
        pairs: [(0, 0); MAX_COUNT * RANKS],
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
        SLOTS as u32,
        0,
        0,
    ]);
    metadata[15] = NEAR_TIE_MARGIN.to_bits();
    for row in 0..count {
        for id in 0..EXPERTS {
            scores[id] = f32::from_bits(word(bytes, base + row * EXPERTS + id));
            if !scores[id].is_finite() {
                return Err(Error::InvalidFormat("batch device router actual scores"));
            }
            raw[id] = id;
        }
        raw.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        metadata[8 + row * 2] = raw[10] as u32;
        metadata[9 + row * 2] = scores[raw[10]].to_bits();
        for rank in 0..RANKS {
            let gap = scores[raw[rank]] - scores[raw[rank + 1]];
            if !gap.is_finite()
                || gap <= NEAR_TIE_MARGIN
                || scores[raw[rank]] - scores[raw[0]] < -80.
            {
                return Err(Error::InvalidFormat("batch device router near tie/range"));
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
                if slot as usize >= SLOTS || routes.pairs().iter().any(|&(_, old)| old == slot) {
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
    metadata[14] = routes.minimum_gap().to_bits();
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
    let canonical = crate::quant_matvec::grouped_moe::GroupMap::new(
        count,
        0,
        count,
        &routes.experts()[..count * RANKS],
        &routes.probabilities()[..count * RANKS],
        routes.pairs(),
        SLOTS,
    )
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
    use cubecl::stream_id::StreamId;
    #[test]
    fn geometry_charges_and_prelaunch_checks() {
        assert_eq!(FRAME_WORDS, 9336);
        assert_eq!(AUDIT_WORDS, 448128);
        assert_eq!(OWNED_BYTES.iter().sum::<u64>(), MAIN_BYTES);
        assert_eq!(OWNED_BYTES[1] + OWNED_BYTES[4], NEW_TABLE_AUDIT_BYTES);
        assert_eq!(INFO_BYTES_PER_CHUNK, 196608);
        assert_eq!(NEAR_TIE_MARGIN.to_bits(), 0x38d1b717);
        for count in 1..=3 {
            let owners = OWNED_BYTES.map(|n| Handle::new(StreamId { value: 7 }, n));
            assert!(validate_owners(&owners, 47, 4096 - count as u64, count).is_ok());
            for index in 0..6 {
                let mut bad = owners.clone();
                bad[index] = bad[index].clone().offset_start(4);
                assert!(validate_owners(&bad, 0, 0, count).is_err());
                let mut bad = owners.clone();
                bad[index].stream.value = 8;
                assert!(validate_owners(&bad, 0, 0, count).is_err());
            }
        }
        for (layer, start, count) in [
            (48, 0, 1),
            (0, 0, 0),
            (0, 0, 4),
            (0, 4095, 2),
            (0, u64::MAX, 1),
        ] {
            assert!(coordinates(layer, start, count).is_err());
        }
    }
    #[test]
    fn both_shader_phases_are_dense_guarded_and_compile_to_spirv() {
        let source = include_str!("resident_batch_router_gpu.wgsl");
        let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        for (index, entry) in module.entry_points.iter().enumerate() {
            assert_eq!(
                entry.workgroup_size,
                if entry.name == "main" {
                    [64, 1, 1]
                } else {
                    [256, 1, 1]
                }
            );
            for (h, var) in module.global_variables.iter() {
                if var.binding.is_some() {
                    assert!(!info.get_entry_point(index)[h].is_empty());
                }
            }
            let spv = wgpu::naga::back::spv::write_vec(
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
            assert!(!spv.is_empty());
        }
        let gate = source.rfind("if atomicLoad(&bad)!=0u").unwrap();
        for publication in ["map[index]=", "probability[index]=", "audit[frame+index]="] {
            assert!(source.find(publication).unwrap() > gate);
        }
        assert!(source.contains("gap<=bitcast<f32>(p.margin)"));
        assert!(source.contains("storageBarrier();"));
        assert_eq!(PREFLIGHT_WORKGROUPS, 298);
    }
}
