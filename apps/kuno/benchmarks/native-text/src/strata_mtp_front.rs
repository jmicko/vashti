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
    Norm,
    Broadcast,
    Positions,
    Append,
    Guard,
    Rewind,
    Identity,
    Select,
}
impl Kernel {
    fn lanes(self) -> u32 {
        if matches!(self, Self::Norm) { 1024 } else { 64 }
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
        let mut source = section("COMMON").to_owned();
        source.push_str(section(match self {
            Self::Embed => "EMBED",
            Self::Norm => "NORM",
            Self::Broadcast => "BROADCAST",
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
            Self::Norm => "StrataMtpFrontNorm",
            Self::Broadcast => "StrataMtpFrontBroadcast",
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
    launch_impl(arena, kernel, grid, handles, info, 8)
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
            Kernel::Embed | Kernel::Norm | Kernel::Broadcast | Kernel::Positions | Kernel::Append
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
    )
}
fn launch_impl(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: [usize; 3],
    handles: Vec<Handle>,
    info: [u32; 8],
    capacity: u32,
) -> Result<()> {
    let client = arena.client().map_err(u::fault)?;
    if !crate::strata_intel_core::enabled()
        || !(1..=capacity).contains(&info[0])
        || grid.contains(&0)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || kernel.lanes() > client.properties().hardware.max_units_per_cube
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(MetadataBindingInfo::custom(
        info.chunks_exact(2)
            .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
            .collect(),
    ));
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
        Kernel::Norm,
        Kernel::Broadcast,
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
pub(crate) const WORDS: [usize; 25] = [
    8, 81920, 20480, 20480, 81920, 20480, 81920, 81920, 64, 207360, 2560, 2560, 32, 20480, 4096,
    4096, 4096, 32768, 4096, 98304, 81920, 20480, 800, 12800, 2000,
];
pub(crate) struct Workspace {
    pub ids: zeroize::Zeroizing<[BufferId; 25]>,
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
        let mut seen = vec![false; 4096 * 256];
        for token in 0usize..4096 {
            for head in 0..2 {
                for dim in 0..128 {
                    let row = ((token / 4) * 2 + head) * 4 + token % 4;
                    let at = row * 128 + dim;
                    assert!(!seen[at]);
                    seen[at] = true;
                    assert_eq!((at / 1024) * 4 + (at / 128) % 4, token);
                }
            }
        }
        assert!(seen.iter().all(|&v| v));
        for accepted in [0, 1, 3, 4, 5, 63, 64, 4095, 4096] {
            let cleared = (0..4096 * 256)
                .filter(|&word| (word / 1024) * 4 + (word / 128) % 4 >= accepted)
                .count();
            assert_eq!(cleared, (4096 - accepted) * 256);
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
            assert!(Workspace::bytes().unwrap() < 8 * 1024 * 1024);
        } else {
            assert_eq!(Workspace::bytes().unwrap(), 0);
        }
    }
}
