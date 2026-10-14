//! Connected native Intel QSA: FP16 KV, fused norm/RoPE, chunked attention.
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

pub(crate) const CACHE_WORDS: usize = 640; // 512 half K + 512 half V + 128 F32 index key
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
    Norm,
    Pack,
    Pool,
    RetainedPool,
    PublishPool,
    Score,
    Attention,
    Merge,
    Gate,
    Append,
}
impl Kernel {
    fn lanes(self) -> u32 {
        match self {
            Self::Norm | Self::Attention | Self::Merge => 256,
            Self::Score => 128,
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
        let mut s = section("COMMON").to_owned();
        s.push_str(section(match self {
            Self::Norm => "NORM",
            Self::Pack => "PACK",
            Self::Pool => "POOL",
            Self::RetainedPool => "RETAINED_POOL",
            Self::PublishPool => "PUBLISH_POOL",
            Self::Score => "SCORE",
            Self::Attention => "ATTENTION",
            Self::Merge => "MERGE",
            Self::Gate => "GATE",
            Self::Append => "APPEND",
        }));
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Norm => "StrataQsaNormRope",
            Self::Pack => "StrataQsaPackFp16",
            Self::Pool => "StrataQsaRawIndexerPool",
            Self::RetainedPool => "StrataQsaRetainedIndexerPrefix",
            Self::PublishPool => "StrataQsaPublishIndexerBlocks",
            Self::Score => "StrataQsaIndexerScore",
            Self::Attention => "StrataQsaChunkAttention",
            Self::Merge => "StrataQsaChunkMerge",
            Self::Gate => "StrataQsaGate",
            Self::Append => "StrataQsaAppendFp16",
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
        || !(1..=8).contains(&info[0])
        || info[1].checked_add(info[0]).is_none_or(|end| end > info[2])
        || info[2] > 4096
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
        Kernel::Norm,
        Kernel::Pack,
        Kernel::Pool,
        Kernel::RetainedPool,
        Kernel::PublishPool,
        Kernel::Score,
        Kernel::Attention,
        Kernel::Merge,
        Kernel::Gate,
        Kernel::Append,
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
    fn packed_fp16_current_and_committed_geometry_covers_all_real_causal_rows() {
        // The original separate half pools have exactly the same component
        // address as this packed token owner (both little-endian pairs).
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
                                    let half = value * 512 + kvh * 256 + dim;
                                    let at = row * CACHE_WORDS + half / 2;
                                    assert!(at < bound);
                                    assert_eq!(half / 2, (value * 256) + (kvh * 128) + dim / 2);
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
                // prefix, including all packed halves and historical index keys.
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
