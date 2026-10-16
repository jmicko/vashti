//! Connected native Intel QSA: INT8 KV, fused norm/RoPE, chunked attention.
//! MIT Strata fb58e0: native_rope.dp.cpp, qsa_decode_attn.dp.cpp, kv append.
//! All private owners/strides are supplied by the admitted MAX8 QSA workspace.
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
#[path = "strata_prompt_qsa.rs"]
pub(crate) mod prompt;
#[path = "strata_qsa_select.rs"]
pub(crate) mod selection;

// Original64-value quantization groups: 1024 signed codes,16 half scales,
// followed by the128 FP16-rounded F32 index-key values.
pub(crate) const INDEX_OFFSET: usize = 264;
pub(crate) const CACHE_WORDS: usize = INDEX_OFFSET + 128;
pub(crate) const PACK_GROUPS: usize = 8;
pub(crate) const CHUNKS: usize = 33; // ceil((2048 + incomplete tail <=3)/64)
pub(crate) const PARTIAL_WORDS: usize = 24 * CHUNKS * 256;
pub(crate) const STATS_WORDS: usize = 24 * CHUNKS * 2;
pub(crate) const fn cache_words() -> usize {
    if crate::strata_intel_core::enabled() {
        CACHE_WORDS
    } else {
        crate::qsa_gpu::TOKEN
    }
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Kernel {
    Positions,
    Norm,
    PromptNorm,
    Pack,
    PromptPack,
    Pool,
    RetainedPool,
    PublishPool,
    Score,
    PromptScore,
    Attention,
    PromptAttention,
    PromptMerge,
    MtpAttention,
    MtpMerge,
    Merge,
    Gate,
    Append,
    PromptAppend,
    PromptPackFault,
    PromptAppendFault,
    PromptAttentionFault,
    PromptMergeFault,
    PromptScoreFault,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Pack | Self::PromptPack | Self::PromptPackFault => 128,
            Self::Norm
            | Self::PromptNorm
            | Self::Attention
            | Self::Merge
            | Self::MtpAttention
            | Self::MtpMerge
            | Self::PromptAttention
            | Self::PromptMerge
            | Self::PromptAttentionFault
            | Self::PromptMergeFault
            | Self::Score
            | Self::PromptScore => 256,
            Self::PromptScoreFault => 256,
            _ => 64,
        }
    }
    fn source(self) -> String {
        let all = include_str!("strata_qsa.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut s = format!(
            "const CONTEXT_CAPACITY:u32={}u;\n",
            crate::qsa_gpu::MAX_CONTEXT
        );
        s.push_str(section("COMMON"));
        if self == Self::Positions {
            s.push_str(&format!(
                "const TOKEN_STATUS:u32={}u;const POSITION_MAX:u32={}u;\n",
                crate::qsa_gpu::TOKEN,
                crate::qsa_gpu::MAX_POSITION,
            ));
        }
        s.push_str(section(match self {
            Self::Positions => "POSITIONS",
            Self::Norm | Self::PromptNorm => "NORM",
            Self::Pack | Self::PromptPack | Self::PromptPackFault => "PACK",
            Self::Pool => "POOL",
            Self::RetainedPool => "RETAINED_POOL",
            Self::PublishPool => "PUBLISH_POOL",
            Self::Score | Self::PromptScore | Self::PromptScoreFault => "SCORE",
            Self::Attention
            | Self::MtpAttention
            | Self::PromptAttention
            | Self::PromptAttentionFault => "ATTENTION",
            Self::Merge | Self::MtpMerge | Self::PromptMerge | Self::PromptMergeFault => "MERGE",
            Self::Gate => "GATE",
            Self::Append | Self::PromptAppend | Self::PromptAppendFault => "APPEND",
        }));
        if matches!(self, Self::PromptScore | Self::PromptScoreFault) {
            s = s.replace(
                "token*p.input_stride+block*128u+lane*4u",
                "block*128u+lane*4u",
            );
        }
        if matches!(
            self,
            Self::PromptAttention
                | Self::PromptMerge
                | Self::PromptAttentionFault
                | Self::PromptMergeFault
        ) {
            // Keep partial/stats indexed by the local <=32-query tile. Q,
            // selection, output and closing flags retain full chunk indices.
            // In particular selection's 2051-word stride is not256-aligned:
            // rebasing its descriptor would produce illegal Vulkan offsets.
            s = s.replace(
                "let t=group.z;let n=cells(t,p.start);",
                "let t=group.z;let query_row=t+p.reserved;let n=cells(query_row,p.start);",
            );
            s = s.replace(
                "let t=group.y;if sw",
                "let t=group.y;let query_row=t+p.reserved;if sw",
            );
            s = s.replace("cells(t,p.start)+63u", "cells(query_row,p.start)+63u");
            s = s.replace("p.start+t+1u", "p.start+query_row+1u");
            for binding in ["query", "selected", "output", "status"] {
                s = s.replace(&format!("{binding}[t*"), &format!("{binding}[query_row*"));
            }
            if matches!(self, Self::PromptAttention | Self::PromptAttentionFault) {
                // Prompt append is ordered before index scoring/attention.
                // Every causal cell now comes from the authoritative cache.
                let a = s.find("fn kv(").unwrap();
                let z = a + s[a..].find("@compute").unwrap();
                s.replace_range(a..z,"fn kv(token:u32,offset:u32)->f32 {return q8_value(cache[token*CACHE_WORDS+offset/4u],cache[token*CACHE_WORDS+256u+offset/128u],offset);}\n");
            }
        }
        let edits: &[(&str, &str)] = match self {
            Self::PromptNorm => &[(
                "if sw!=32u || token>=p.count || head>=nh || group.z!=0u || (p.width!=128u && p.width!=256u) || (p.block_positions!=0u && head<p.start/4u) {return;}",
                "if sw!=32u || token>=p.count || head>=nh || group.z!=0u || (p.width!=128u && p.width!=256u) || (p.block_positions!=0u && head<p.start/4u) {if tid==0u&&token<p.count&&head<p.heads {status[token*p.status_stride+head]=0u;}return;}",
            )],
            Self::PromptScoreFault => &[
                (
                    "@group(0) @binding(5) var<uniform> p:Params;",
                    "@group(0) @binding(5) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding(6) var<uniform> p:Params;",
                ),
                (
                    "if sw!=32u || token>=p.count",
                    "if sw!=32u{if lane==0u{atomicOr(&fault[0],1u);}return;}if token>=p.count",
                ),
            ],
            Self::PromptPackFault => &[
                (
                    "@binding(4) var<storage,read_write> status:array<u32>;",
                    "@binding(4) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "fn record_valid(t:u32,element:u32,ok:bool) {status[t*p.status_stride+element]=select(0u,1u,ok);}",
                    "fn record_valid(t:u32,element:u32,ok:bool) {if !ok{atomicOr(&fault[0],1u);}}",
                ),
            ],
            Self::PromptAppendFault => &[
                (
                    "@binding(3) var<storage,read_write> status:array<u32>;",
                    "@binding(3) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "fn record_valid(t:u32,element:u32,ok:bool) {status[t*p.status_stride+element]=select(0u,1u,ok);}",
                    "fn record_valid(t:u32,element:u32,ok:bool) {if !ok{atomicOr(&fault[0],1u);}}",
                ),
            ],
            Self::PromptAttentionFault => &[
                (
                    "@binding(6) var<storage,read_write> status:array<u32>;",
                    "@binding(6) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c]=select(0u,1u,finite(score));",
                    "",
                ),
                (
                    "for(var i=tid;i<here*12u;i+=256u) {let h=i/here;let c=i%here;status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c]=0u;}",
                    "if tid==0u{atomicOr(&fault[0],1u);}",
                ),
                (
                    "if sw!=32u || t>=p.count",
                    "if sw!=32u{if tid==0u{atomicOr(&fault[0],1u);}return;}if t>=p.count",
                ),
            ],
            Self::PromptMergeFault => &[
                (
                    "@binding(3) var<storage,read_write> status:array<u32>;",
                    "@binding(3) var<storage,read_write> fault:array<atomic<u32>>;",
                ),
                (
                    "status[query_row*p.status_stride+h]=valid;",
                    "if valid!=1u{atomicOr(&fault[0],1u);}",
                ),
                (
                    "if sw!=32u || h>=24u",
                    "if sw!=32u{if d==0u{atomicOr(&fault[0],1u);}return;}if h>=24u",
                ),
            ],
            _ => &[],
        };
        for &(old, new) in edits {
            assert!(s.contains(old), "prompt QSA closing source drift");
            s = s.replace(old, new);
        }
        if matches!(self, Self::MtpAttention | Self::MtpMerge) {
            s = s.replace(
                "let n=start+token+1u;return min(n,2051u);",
                "return start+token+1u;",
            );
            let context = crate::mtp_resident_kernels::CONTEXT;
            let chunks = context.div_ceil(64);
            s = s
                .replace("33u", &format!("{chunks}u"))
                .replace("202752u", &format!("{}u", 24 * chunks * 256))
                .replace("1584u", &format!("{}u", 24 * chunks * 2))
                .replace("2051u", &format!("{context}u"));
            if matches!(self, Self::MtpAttention) {
                let a = s.find("fn kv(").unwrap();
                let z = a + s[a..].find("@compute").unwrap();
                s.replace_range(a..z,"fn kv(token:u32,offset:u32)->f32 {let component=offset%512u;let row=((token/4u)*2u+component/256u)*4u+token%4u;let d=component%256u;let at=row*66u+d/4u;let scale_at=row*66u+64u+d/128u;var bits=0u;var scales=0u;if offset<512u {bits=cache[at];scales=cache[scale_at];}else{bits=candidate[at];scales=candidate[scale_at];}return q8_value(bits,scales,d); }\n");
            }
        }
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Positions => "StrataQsaNativePositions",
            Self::Norm => "StrataQsaNormRope",
            Self::PromptNorm => "StrataQsaPromptMtpNormRope",
            Self::Pack => "StrataQsaPackInt8",
            Self::PromptPack => "StrataPromptQsaPackInt8",
            Self::Pool => "StrataQsaRawIndexerPool",
            Self::RetainedPool => "StrataQsaRetainedIndexerPrefix",
            Self::PublishPool => "StrataQsaPublishIndexerBlocks",
            Self::Score => "StrataQsaIndexerScore",
            Self::PromptScore => "StrataPromptSharedIndexerScore",
            Self::Attention => "StrataQsaChunkAttention",
            Self::PromptAttention => "StrataPromptAttentionBatch32",
            Self::PromptMerge => "StrataPromptMergeBatch32",
            Self::MtpAttention => "StrataMtpChunkAttention",
            Self::MtpMerge => "StrataMtpChunkMerge",
            Self::Merge => "StrataQsaChunkMerge",
            Self::Gate => "StrataQsaGate",
            Self::Append => "StrataQsaAppendInt8",
            Self::PromptAppend => "StrataPromptQsaAppendInt8",
            Self::PromptPackFault => "StrataPromptFullQsaPack",
            Self::PromptAppendFault => "StrataPromptFullQsaAppend",
            Self::PromptAttentionFault => "StrataPromptFullAttentionBatch32",
            Self::PromptMergeFault => "StrataPromptFullMergeBatch32",
            Self::PromptScoreFault => "StrataPromptFullIndexerScore",
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
    info: [u32; 12],
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || grid.contains(&0)
        || grid.iter().any(|&x| x > 65535)
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || !(1..=if matches!(
            kernel,
            Kernel::PromptAttention
                | Kernel::PromptMerge
                | Kernel::PromptAttentionFault
                | Kernel::PromptMergeFault
        ) && crate::strata_prompt::active()
        {
            32
        } else if matches!(
            kernel,
            Kernel::PromptNorm
                | Kernel::PromptScore
                | Kernel::PromptPack
                | Kernel::PromptAppend
                | Kernel::PromptPackFault
                | Kernel::PromptAppendFault
                | Kernel::PromptScoreFault
        ) && crate::strata_prompt::active()
        {
            crate::strata_prompt::groups::CAPACITY as u32
        } else {
            8
        })
            .contains(&info[0])
        || info[1].checked_add(info[0]).is_none_or(|end| end > info[2])
        || info[2] > crate::qsa_gpu::MAX_CONTEXT as u32
        || if matches!(
            kernel,
            Kernel::PromptAttention
                | Kernel::PromptMerge
                | Kernel::PromptAttentionFault
                | Kernel::PromptMergeFault
        ) {
            info[11]
                .checked_add(info[0])
                .is_none_or(|end| end > crate::strata_prompt::groups::CAPACITY as u32)
                || info[1]
                    .checked_add(info[11])
                    .and_then(|first| first.checked_add(info[0]))
                    .is_none_or(|end| end > info[2])
        } else {
            info[11] != 0
        }
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    args = args.with_info(
        MetadataBindingInfo::custom(
            info.chunks_exact(2)
                .map(|p| p[0] as u64 | ((p[1] as u64) << 32))
                .collect(),
        )
        .with_native_parameter(
            1,
            cubecl_runtime::server::NativeStrataValue::PhasePosition { offset: 0 },
        ),
    );
    // SAFETY: authenticated QSA parent validates fixed widths and complete private
    // owners before enqueue; kernels additionally bound runtime active intervals.
    unsafe {
        arena.client().map_err(u::fault)?.launch_unchecked(
            Box::new(kernel),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}
pub(crate) fn handles(arena: &PrivateArena, ids: &[BufferId]) -> Result<Vec<Handle>> {
    ids.iter().map(|&id| u::handle(arena, id)).collect()
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Positions,
        Kernel::Norm,
        Kernel::PromptNorm,
        Kernel::Pack,
        Kernel::PromptPack,
        Kernel::Pool,
        Kernel::RetainedPool,
        Kernel::PublishPool,
        Kernel::Score,
        Kernel::PromptScore,
        Kernel::Attention,
        Kernel::PromptAttention,
        Kernel::PromptMerge,
        Kernel::MtpAttention,
        Kernel::MtpMerge,
        Kernel::Merge,
        Kernel::Gate,
        Kernel::Append,
        Kernel::PromptAppend,
        Kernel::PromptPackFault,
        Kernel::PromptAppendFault,
        Kernel::PromptAttentionFault,
        Kernel::PromptMergeFault,
        Kernel::PromptScoreFault,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_int8_current_and_committed_geometry_covers_all_real_causal_rows() {
        // Token owners retain both code planes, packed half scales and raw
        // index keys. Candidate and committed rows use identical local offsets.
        for count in 1usize..=8 {
            for start in [0, 1, 3, 63, 64, 2047, 2048, 4096 - count] {
                for t in 0..count {
                    let visible = start + t + 1;
                    let selected = (visible / 4).min(512) * 4 + visible % 4;
                    let chunks = selected.div_ceil(64);
                    assert!((1..=CHUNKS).contains(&chunks));
                    for token in [0, visible - 1] {
                        let in_current = token >= start;
                        let row = if in_current { token - start } else { token };
                        let bound = if in_current {
                            count * CACHE_WORDS
                        } else {
                            4096 * CACHE_WORDS
                        };
                        for dim in [0, 255] {
                            for kvh in 0..2 {
                                for value in [0, 1] {
                                    let element = value * 512 + kvh * 256 + dim;
                                    let code = row * CACHE_WORDS + element / 4;
                                    let scale = row * CACHE_WORDS + 256 + element / 128;
                                    assert!(code < bound && scale < bound);
                                    assert_eq!(element / 4, value * 128 + kvh * 64 + dim / 4);
                                    assert!(element / 128 < 8);
                                }
                            }
                        }
                    }
                    // Only newly completed blocks pool raw keys; any old key
                    // is among the final three rows of the pre-window tail.
                    for block in start / 4..visible / 4 {
                        for j in 0..4 {
                            let raw = block * 4 + j;
                            assert!(raw < visible);
                            if raw < start {
                                assert!(start - raw <= 3);
                            }
                        }
                    }
                    for kvh in 0..2 {
                        for chunk in 0..chunks {
                            for head in 0..12 {
                                let slot = kvh * CHUNKS + chunk;
                                assert!(
                                    (t * PARTIAL_WORDS + (slot * 12 + head) * 256 + 255)
                                        < count * PARTIAL_WORDS
                                );
                                assert!(
                                    (t * STATS_WORDS + (slot * 12 + head) * 2 + 1)
                                        < count * STATS_WORDS
                                );
                            }
                        }
                    }
                }
                // Rejected suffix erasure preserves every byte of the accepted
                // prefix, including codes, scales and historical index keys.
                for accepted in 1..=count {
                    let from = (start + accepted) * CACHE_WORDS;
                    let end = (start + count) * CACHE_WORDS;
                    assert!(from <= end && end <= 4096 * CACHE_WORDS);
                }
            }
        }
    }
    #[test]
    fn chunk_merge_host_oracle_matches_unpartitioned_softmax_on_partial_tail() {
        // Independent F64 global-softmax oracle; Rust mirror of native chunk
        // max/sum/FMA checks the prescribed merge, not device execution parity.
        for n in [1usize, 3, 63, 64, 65, 511, 2048, 2051] {
            let logits = (0..n)
                .map(|i| (((i * 37) % 211) as f64 - 105.) / 13.)
                .collect::<Vec<_>>();
            let values = (0..n)
                .map(|i| half::f16::from_f32(((i * 53) % 97) as f32 / 31. - 1.5).to_f32())
                .collect::<Vec<_>>();
            let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let den = logits.iter().map(|&x| (x - maximum).exp()).sum::<f64>();
            let expected = logits
                .iter()
                .zip(&values)
                .map(|(&x, &v)| (x - maximum).exp() * v as f64)
                .sum::<f64>()
                / den;
            let mut partial = Vec::new();
            for (ls, vs) in logits.chunks(64).zip(values.chunks(64)) {
                let m = ls
                    .iter()
                    .map(|&x| x as f32)
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut d = 0.;
                let mut a = 0.;
                for (&x, &v) in ls.iter().zip(vs) {
                    let p = (x as f32 - m).exp();
                    d += p;
                    a = p.mul_add(v, a);
                }
                partial.push((m, d, a));
            }
            let m = partial
                .iter()
                .map(|p| p.0)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut d = 0.;
            let mut a = 0.;
            for (mx, den, acc) in partial {
                let w = (mx - m).exp();
                d = den.mul_add(w, d);
                a = acc.mul_add(w, a);
            }
            assert!(((a / d) as f64 - expected).abs() < 3e-6);
        }
    }
}
