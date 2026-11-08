//! Grouped MIT Strata fb58e0 MTP front/prompt KV graph.
//! Rust owns MAX8 private workspace; original public pages remain borrowed.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{CubeCount, CubeDim},
    wgpu::AutoCompiler,
};
use cubecl_runtime::server::{NativeStrataParameter, NativeStrataValue};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Embed,
    EmbedMapped,
    Norm,
    Broadcast,
    TargetBroadcast,
    Positions,
    Append,
    Guard,
    Rewind,
    Identity,
    Select,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Norm => 1024,
            Self::Append => 128,
            Self::EmbedMapped => 32,
            _ => 64,
        }
    }
    fn source(self) -> String {
        let all = include_str!("strata_mtp_front.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut source = format!(
            "const CONTEXT_CAPACITY:u32={}u;const CACHE_STORAGE_WORDS:u32={}u;\n",
            crate::mtp_resident_kernels::CONTEXT,
            crate::mtp_resident_kernels::CACHE_STORAGE_WORDS
        );
        source.push_str(section("COMMON"));
        source.push_str(section(match self {
            Self::Embed => "EMBED",
            Self::EmbedMapped => "EMBED_MAPPED",
            Self::Norm => "NORM",
            Self::Broadcast => "BROADCAST",
            Self::TargetBroadcast => "TARGET_BROADCAST",
            Self::Positions => "POSITIONS",
            Self::Append => "APPEND",
            Self::Guard => "GUARD",
            Self::Rewind => "REWIND",
            Self::Identity => "IDENTITY",
            Self::Select => "SELECT",
        }));
        source
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Embed => "StrataMtpFrontEmbedding",
            Self::EmbedMapped => "StrataOriginalMappedEmbedding",
            Self::Norm => "StrataMtpFrontNorm",
            Self::Broadcast => "StrataMtpFrontBroadcast",
            Self::TargetBroadcast => "StrataTargetEmbeddingBroadcast",
            Self::Positions => "StrataMtpFrontPositions",
            Self::Append => "StrataMtpFrontAppend",
            Self::Guard => "StrataMtpPackedCacheGuard",
            Self::Rewind => "StrataMtpPackedCacheRewind",
            Self::Identity => "StrataMtpDenseIdentity",
            Self::Select => "StrataMtpAcceptedFrontSelect",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(self.lanes()))
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
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
/// All roots are validated by the parent as distinct, full, admitted pages.
/// INFO is runtime geometry; no prompt or result is copied to host memory.
pub(crate) fn launch(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    launch_impl(
        arena,
        kernel,
        grid,
        handles,
        info,
        8,
        position_field(kernel),
        Vec::new(),
    )
}
fn position_field(kernel: Kernel) -> Option<NativeStrataParameter> {
    matches!(
        kernel,
        Kernel::Embed
            | Kernel::EmbedMapped
            | Kernel::Norm
            | Kernel::Broadcast
            | Kernel::Positions
            | Kernel::Append
            | Kernel::Identity
    )
    .then_some(NativeStrataParameter {
        word: 1,
        value: NativeStrataValue::PhasePosition { offset: 0 },
    })
}
/// Packed-cache boundary is distinct from a readable token cell and may equal context end.
pub(crate) fn guard(
    arena: &PrivateArena,
    grid: [usize; 3],
    handles: Vec<Handle>,
    position: usize,
    phase_offset: Option<i16>,
) -> Result<()> {
    if position > crate::mtp_resident_kernels::CONTEXT {
        return Err(Error::InvalidBufferLength);
    }
    let field = phase_offset.map(|offset| NativeStrataParameter {
        word: 1,
        value: NativeStrataValue::PhaseBoundary { offset },
    });
    launch_impl(
        arena,
        Kernel::Guard,
        grid,
        handles,
        [1, position as u32, 0, 0, 0, 0, 0, 0],
        8,
        field,
        Vec::new(),
    )
}
pub(crate) fn launch_prompt(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    if !crate::strata_prompt::active()
        || !matches!(
            kernel,
            Kernel::Embed
                | Kernel::Norm
                | Kernel::Broadcast
                | Kernel::TargetBroadcast
                | Kernel::Positions
                | Kernel::Append
        )
    {
        return Err(Error::InvalidBufferLength);
    }
    launch_impl(
        arena,
        kernel,
        grid,
        handles,
        info,
        crate::strata_prompt::groups::CAPACITY as u32,
        position_field(kernel),
        Vec::new(),
    )
}
fn dispatch_grid(kernel: Kernel, grid: [usize; 3]) -> Result<[usize; 3]> {
    const LIMIT: usize = 65535;
    if grid.contains(&0) {
        return Err(Error::InvalidBufferLength);
    }
    if matches!(kernel, Kernel::Guard | Kernel::Rewind) && grid[1..] == [1, 1] && grid[0] > LIMIT {
        let rows = grid[0].div_ceil(LIMIT);
        if rows <= LIMIT {
            return Ok([LIMIT, rows, 1]);
        }
    }
    if grid.iter().any(|&n| n > LIMIT) {
        return Err(Error::InvalidBufferLength);
    }
    Ok(grid)
}
pub(crate) fn launch_mapped_embedding(
    arena: &PrivateArena,
    handles: Vec<Handle>,
    reads: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    if handles.len() != 5 || reads.len() != crate::strata_shared_embedding::PAGE_COUNT {
        return Err(Error::InvalidBufferLength);
    }
    let cap = if crate::strata_prompt::active() {
        crate::strata_prompt::groups::CAPACITY
    } else {
        8
    };
    launch_impl(
        arena,
        Kernel::EmbedMapped,
        [10, info[0] as usize, 1],
        handles,
        info,
        cap as u32,
        position_field(Kernel::EmbedMapped),
        reads,
    )
}
#[allow(clippy::too_many_arguments)] // Actual descriptors, physical-source dependencies and invocation controls.
fn launch_impl(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
    capacity: u32,
    native: Option<NativeStrataParameter>,
    reads: Vec<Handle>,
) -> Result<()> {
    let grid = dispatch_grid(kernel, grid)?;
    let client = arena.client().map_err(u::fault)?;
    if !crate::strata_intel_core::enabled()
        || !(1..=capacity).contains(&info[0])
        || grid.contains(&0)
        || handles.is_empty()
        || handles
            .iter()
            .chain(&reads)
            .any(|h| h.stream != handles[0].stream)
        || kernel.lanes() > client.properties().hardware.max_units_per_cube
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    let mut metadata = MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    );
    if let Some(field) = native {
        metadata = metadata.with_native_parameter(field.word, field.value);
    }
    args = args
        .with_read_dependencies(reads.into_iter().map(Handle::binding).collect())
        .with_info(metadata);
    // SAFETY: admitted MAX8 front owner validates complete geometry and retains
    // original public weights/private roots through publication and erasure.
    unsafe {
        client.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Embed,
        Kernel::EmbedMapped,
        Kernel::Norm,
        Kernel::Broadcast,
        Kernel::TargetBroadcast,
        Kernel::Positions,
        Kernel::Append,
        Kernel::Guard,
        Kernel::Rewind,
        Kernel::Identity,
        Kernel::Select,
    ]
    .into_iter()
    .map(|k| {
        k.compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
    })
    .collect()
}

pub(crate) const TOKEN: usize = 0;
pub(crate) const CARRY: usize = 1;
pub(crate) const EMBED: usize = 2;
pub(crate) const EN: usize = 3;
pub(crate) const HN: usize = 4;
pub(crate) const E2: usize = 5;
pub(crate) const RES: usize = 6;
pub(crate) const NORM: usize = 7;
pub(crate) const SCALES: usize = 8;
pub(crate) const PART: usize = 9;
pub(crate) const DOWN: usize = 10;
pub(crate) const LOW: usize = 11;
pub(crate) const INJ: usize = 12;
pub(crate) const MIXED: usize = 13;
pub(crate) const KEY: usize = 14;
pub(crate) const VALUE: usize = 15;
pub(crate) const ROTATED: usize = 16;
pub(crate) const PLANES: usize = 17;
pub(crate) const POS: usize = 18;
pub(crate) const STATUS: usize = 19;
pub(crate) const RES1: usize = 20;
pub(crate) const BLOCK: usize = 21;
pub(crate) const EXPERT_INPUT: usize = 22;
pub(crate) const EXPERT_GU: usize = 23;
pub(crate) const EXPERT_HIDDEN: usize = 24;
pub(crate) const SHARED_FAULT: usize = 25;
pub(crate) const WORDS: [usize; 26] = [
    8,
    81920,
    20480,
    20480,
    81920,
    20480,
    81920,
    81920,
    64,
    207360,
    2560,
    2560,
    32,
    20480,
    4096,
    4096,
    4096,
    32768,
    crate::mtp_resident_kernels::CONTEXT,
    crate::mtp_resident_kernels::ATTENTION_WORDS,
    81920,
    20480,
    800,
    12800,
    2000,
    1, // Shared fault folds on the owning draft lane before combine.
];
pub(crate) struct Workspace {
    pub ids: zeroize::Zeroizing<[BufferId; 26]>,
}
impl Workspace {
    pub fn requests() -> Vec<u64> {
        if crate::strata_intel_core::enabled() {
            WORDS.map(|n| n as u64 * 4).to_vec()
        } else {
            Vec::new()
        }
    }
    pub fn bytes() -> Result<u64> {
        Self::requests().into_iter().try_fold(0, |n, b| {
            Ok(n + crate::private_arena::BufferLayout::new(b as usize / 4)
                .map_err(u::fault)?
                .owned_bytes as u64)
        })
    }
    pub fn allocate(arena: &mut PrivateArena) -> Result<Option<Self>> {
        if !crate::strata_intel_core::enabled() {
            return Ok(None);
        }
        let mut ids = Vec::new();
        for n in WORDS {
            ids.push(arena.allocate_zeroed(n).map_err(u::fault)?);
        }
        Ok(Some(Self {
            ids: zeroize::Zeroizing::new(ids.try_into().map_err(|_| Error::InvalidBufferLength)?),
        }))
    }
    pub fn handles(&self, arena: &PrivateArena) -> Result<Vec<Handle>> {
        self.ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                if self.ids[..i].contains(&id)
                    || arena.layout(id).map_err(u::fault)?
                        != crate::private_arena::BufferLayout::new(WORDS[i]).map_err(u::fault)?
                {
                    return Err(Error::InvalidBufferLength);
                }
                arena.handle(id).cloned().map_err(u::fault)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn actual_mapped_embedding_shader_lowers_original_addresses_with_contiguous_descriptors() {
        let source = Kernel::EmbedMapped.source();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        let mut bindings = module
            .global_variables
            .iter()
            .filter_map(|(_, g)| g.binding.as_ref().map(|b| b.binding))
            .collect::<Vec<_>>();
        bindings.sort_unstable();
        assert_eq!(bindings, [0, 1, 2, 3, 4, 5]);
        assert_eq!(module.entry_points[0].workgroup_size, [32, 1, 1]);
        let roots = module
            .global_variables
            .iter()
            .filter(|(_, g)| g.name.as_deref() == Some("gate0"))
            .collect::<Vec<_>>();
        assert_eq!(roots.len(), 1);
        assert_eq!(
            roots[0].1.space,
            wgpu::naga::AddressSpace::Storage {
                access: wgpu::naga::StorageAccess::LOAD
            }
        );
        assert_eq!(crate::strata_shared_embedding::WORD_FIRST / 4, 1536);
        assert_eq!(crate::strata_shared_embedding::PAGE_COUNT, 16);
        assert!(position_field(Kernel::EmbedMapped).is_some());
        let words = cubecl::wgpu::native_expert_shader_words(&source).unwrap();
        assert_eq!(words[0], 0x07230203);
        assert!(words.len() > 128);
    }
    #[test]
    fn mapped_embedding_native_lanes_match_original_decoder_and_cover_float4_runs_once() {
        for salt in 0..8 {
            let mut packed = [0u8; 136];
            packed[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
            for (i, b) in packed[2..].iter_mut().enumerate() {
                *b = (i * 29 + salt * 11) as u8;
            }
            let mut expected = [0f32; 256];
            kuno_text::TensorType::Iq4Xs
                .decode(&packed, &mut expected)
                .unwrap();
            let values = [
                -127f32, -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69.,
                89., 113.,
            ];
            let mut seen = [false; 256];
            for lane in 0..32 {
                let il = lane / 8;
                let ib = lane % 8;
                let low = (packed[4 + ib / 2] >> ((ib % 2) * 4)) & 15;
                let high = (u16::from_le_bytes(packed[2..4].try_into().unwrap()) >> (ib * 2)) & 3;
                let scale = 0.03125 * ((low as i32 | ((high as i32) << 4)) - 32) as f32;
                for j in 0..4 {
                    let byte = packed[8 + 16 * ib + 4 * il + j];
                    for half in 0..2 {
                        let at = 32 * ib + 4 * il + j + half * 16;
                        assert!(!seen[at]);
                        seen[at] = true;
                        let code = (byte >> (4 * half)) & 15;
                        assert_eq!(
                            (scale * values[code as usize]).to_bits(),
                            expected[at].to_bits()
                        );
                    }
                }
            }
            assert!(seen.into_iter().all(|b| b));
        }
    }
    #[test]
    fn native_grouped_round_front_shaders_lower_without_device_or_new_owners() {
        use wgpu::naga as n;
        for source in sources() {
            let module = n::front::wgsl::parse_str(&source.source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source.source)));
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source.source)));
            let binary = n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            assert!(!binary.is_empty());
            if source.source == Kernel::Select.source() {
                assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
                assert_eq!(
                    module
                        .global_variables
                        .iter()
                        .filter(|(_, v)| v.binding.is_some())
                        .count(),
                    5
                );
                assert!(source.source.contains("row!=p.count-1u"));
            }
        }
        // All three selected row intervals are disjoint from row0. A single
        // read_write binding per full plane avoids aliasing two storage views.
        for n in 2..=8 {
            for (root, width) in [(RES, 10240), (INJ, 4), (MIXED, 2560)] {
                let source = (n - 1) * width..n * width;
                assert!(source.start >= width && source.end <= WORDS[root]);
            }
        }
    }
    #[test]
    fn native_four_cell_page_mapping_covers_packed_words_and_preserves_rejected_prefix() {
        let mut seen = vec![false; 4096 * 132];
        for token in 0usize..4096 {
            for head in 0..2 {
                for dim in 0..66 {
                    let row = ((token / 4) * 2 + head) * 4 + token % 4;
                    let at = row * 66 + dim;
                    assert!(!seen[at]);
                    seen[at] = true;
                    assert_eq!((at / 66 / 8) * 4 + (at / 66) % 4, token);
                }
            }
        }
        assert!(seen.iter().all(|&v| v));
        for accepted in [0, 1, 3, 4, 5, 63, 64, 4095, 4096] {
            let cleared = (0..4096 * 132)
                .filter(|&word| (word / 66 / 8) * 4 + (word / 66) % 4 >= accepted)
                .count();
            assert_eq!(cleared, (4096 - accepted) * 132);
        }
    }
    #[test]
    fn grouped_embedding_lane_layout_matches_original_iq4xs_decoder() {
        for salt in 0..8 {
            let mut packed = [0u8; 136];
            packed[..2].copy_from_slice(&half::f16::from_f32(0.03125).to_bits().to_le_bytes());
            for (i, v) in packed[2..].iter_mut().enumerate() {
                *v = (i * 29 + salt * 11) as u8;
            }
            let mut decoded = [0f32; 256];
            kuno_text::TensorType::Iq4Xs
                .decode(&packed, &mut decoded)
                .unwrap();
            let table = [
                -127f32, -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69.,
                89., 113.,
            ];
            for (d, &expected) in decoded.iter().enumerate() {
                let sub = d / 32;
                let low = (packed[4 + sub / 2] >> ((sub % 2) * 4)) & 15;
                let high = (u16::from_le_bytes(packed[2..4].try_into().unwrap()) >> (sub * 2)) & 3;
                let scale = 0.03125 * ((low as i32 | ((high as i32) << 4)) - 32) as f32;
                let byte = packed[8 + sub * 16 + d % 16];
                let nib = (byte >> if d % 32 >= 16 { 4 } else { 0 }) & 15;
                assert_eq!((scale * table[nib as usize]).to_bits(), expected.to_bits());
            }
        }
    }
    #[test]
    fn all_native_front_rows_stream_groups_and_last_context_cells_are_in_bounds() {
        for count in 1usize..=8 {
            for first in (0..count * 4).step_by(8) {
                let n = (count * 4 - first).min(8);
                assert!(first * 2560 + n * 2560 <= WORDS[HN]);
                assert!(first * 2560 + n * 2560 <= WORDS[RES]);
                assert!(n * (2560 / 2 + 2560 / 16) <= WORDS[PLANES]);
            }
            for start in [0, 1, 2047, 4096 - count] {
                assert!(start + count <= WORDS[POS]);
                assert!(
                    (start + count) * 512 <= crate::mtp_resident_kernels::CACHE_WORDS
                        || !crate::strata_intel_core::enabled()
                );
            }
            assert!(count * 10240 <= WORDS[CARRY]);
            assert!(count * 512 <= WORDS[KEY]);
            assert!(count * 2 <= WORDS[STATUS]);
        }
        // Bounds cover both actual active intervals and the fixed native GR
        // workspace; these are host proofs, not physical-device acceptance.
        assert_eq!(WORDS[PART], 80 * 8 * 324);
        if crate::strata_intel_core::enabled() {
            // Fixed batch scratch stays bounded independently of the original
            // full-context attention owner, which the constructor also charges.
            let context = crate::private_arena::BufferLayout::new(WORDS[STATUS])
                .unwrap()
                .owned_bytes as u64;
            assert!(Workspace::bytes().unwrap().checked_sub(context).unwrap() < 8 * 1024 * 1024);
        } else {
            assert_eq!(Workspace::bytes().unwrap(), 0);
        }
    }
}

#[cfg(test)]
mod context32768_tests {
    use super::*;
    #[test]
    fn packed_full_sweep_covers_last_context_word_with_portable_two_dimensional_grid() {
        let words = crate::mtp_resident_kernels::CACHE_STORAGE_WORDS;
        for kernel in [Kernel::Guard, Kernel::Rewind] {
            let grid = dispatch_grid(kernel, [words.div_ceil(64), 1, 1]).unwrap();
            assert!(grid.iter().all(|&n| n <= 65535));
            let mut next = 0;
            for y in 0..grid[1] {
                for x in 0..grid[0] * 64 {
                    let i = x + y * grid[0] * 64;
                    if i < words {
                        assert_eq!(i, next);
                        next += 1;
                    }
                }
            }
            assert_eq!(next, words);
            assert!(kernel.source().contains("id.x+id.y*groups.x*64u"));
        }
        assert!(dispatch_grid(Kernel::Positions, [65536, 1, 1]).is_err());
        assert!(dispatch_grid(Kernel::Guard, [65536, 2, 1]).is_err());
        assert!(dispatch_grid(Kernel::Guard, [1, 0, 1]).is_err());
    }
    #[test]
    fn front_positions_and_closing_owner_cover_context_without_expanding_batch_math() {
        let context = crate::mtp_resident_kernels::CONTEXT;
        assert_eq!(WORDS[POS], context);
        assert_eq!(WORDS[STATUS], context * 24);
        assert_eq!(WORDS[KEY], 8 * 512);
        assert_eq!(WORDS[VALUE], 8 * 512);
        assert_eq!(WORDS[ROTATED], 8 * 512);
        let token = context - 1;
        for head in 0..2 {
            for word in 0..66 {
                let row = ((token / 4) * 2 + head) * 4 + token % 4;
                assert!(row * 66 + word < crate::mtp_resident_kernels::CACHE_STORAGE_WORDS);
                assert_eq!((row / 8) * 4 + row % 4, token);
            }
        }
    }
}
