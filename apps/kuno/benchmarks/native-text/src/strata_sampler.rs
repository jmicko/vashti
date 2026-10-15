//! MIT Strata fb58e0 Intel sampler: original penalties, GPU split top-k and
//! ordered F32 truncation/temperature/Philox tail. Caller owns private buffers.
//! No content allocation, output storage or host-logit sampling in this launcher.
//! Device parameters are a private, mutable buffer: graph identity contains no
//! request values. Request owners and target/draft connection remain incomplete.
#![allow(dead_code)] // Owning request/target/draft hookup is the next coupled block.
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::*,
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments},
};
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};
pub(crate) const VOCAB: usize = 248320;
pub(crate) const BLOCK: usize = 4096;
pub(crate) const BLOCKS: usize = VOCAB.div_ceil(BLOCK);
pub(crate) const KMAX: usize = 64;
pub(crate) const HISTORY_MAX: usize = 4096;
/// Request values only: no Debug/serialization or kernel-cache identity.
pub(crate) struct Params {
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub min_keep: usize,
    pub penalty_last_n: usize,
    pub penalty_repeat: f32,
    pub penalty_freq: f32,
    pub penalty_present: f32,
    pub seed: u64,
    pub counter: u64,
    pub greedy: bool,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            top_k: 20,
            top_p: 0.95,
            min_p: 0.,
            temperature: 1.,
            min_keep: 1,
            penalty_last_n: 0,
            penalty_repeat: 1.,
            penalty_freq: 0.,
            penalty_present: 0.,
            seed: 0,
            counter: 0,
            greedy: false,
        }
    }
}
impl Drop for Params {
    fn drop(&mut self) {
        self.top_k.zeroize();
        self.top_p.zeroize();
        self.min_p.zeroize();
        self.temperature.zeroize();
        self.min_keep.zeroize();
        self.penalty_last_n.zeroize();
        self.penalty_repeat.zeroize();
        self.penalty_freq.zeroize();
        self.penalty_present.zeroize();
        self.seed.zeroize();
        self.counter.zeroize();
        self.greedy.zeroize();
    }
}
fn reject() -> Error {
    Error::InvalidFormat("native sampler geometry/parameters")
}
impl Params {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.top_k < 0
            || self.min_keep > i32::MAX as usize
            || self.penalty_last_n > i32::MAX as usize
            || !self.top_p.is_finite()
            || self.top_p <= 0.
            || self.top_p > 1.
            || !self.min_p.is_finite()
            || !(0. ..=1.).contains(&self.min_p)
            || !self.temperature.is_finite()
            || self.temperature < 0.
            || !self.penalty_repeat.is_finite()
            || self.penalty_repeat <= 0.
            || !self.penalty_freq.is_finite()
            || !self.penalty_present.is_finite()
        {
            return Err(reject());
        }
        Ok(())
    }
    pub(crate) fn greedy(&self) -> bool {
        self.greedy || self.temperature <= 0.
    }
    pub(crate) fn k(&self) -> usize {
        if self.greedy() {
            1
        } else if self.top_k == 0 || self.top_k as usize > KMAX {
            KMAX
        } else {
            self.top_k as usize
        }
    }
    pub(crate) fn words(
        &self,
        rows: usize,
        history: usize,
        first: usize,
        stride: usize,
    ) -> Result<Zeroizing<Vec<u64>>> {
        self.validate()?;
        if rows == 0
            || rows > crate::strata_port::verify_capacity()
            || history > HISTORY_MAX
            || stride < 2
            || first > u32::MAX as usize
            || stride > u32::MAX as usize
            || self.penalty_last_n > 0 && history == 0
            || first
                .checked_add(
                    (rows - 1)
                        .checked_mul(stride)
                        .ok_or(Error::ArithmeticOverflow)?,
                )
                .and_then(|n| n.checked_add(1))
                .is_none_or(|last| last >= u32::MAX as usize)
        {
            return Err(reject());
        }
        let mut w = Zeroizing::new(vec![0u32; 32]);
        w[..20].copy_from_slice(&[
            rows as u32,
            VOCAB as u32,
            self.k() as u32,
            BLOCKS as u32,
            history as u32,
            self.penalty_last_n as u32,
            self.min_keep as u32,
            u32::from(self.greedy()),
            self.top_p.to_bits(),
            self.min_p.to_bits(),
            self.temperature.to_bits(),
            self.penalty_repeat.to_bits(),
            self.penalty_freq.to_bits(),
            self.penalty_present.to_bits(),
            self.seed as u32,
            (self.seed >> 32) as u32,
            self.counter as u32,
            (self.counter >> 32) as u32,
            first as u32,
            stride as u32,
        ]);
        Ok(Zeroizing::new(
            w.chunks_exact(2)
                .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
                .collect(),
        ))
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Kernel {
    Greedy { penalties: bool },
    Part,
    Merge,
}
impl Kernel {
    pub(crate) fn source(self) -> String {
        format!(
            "{}\n{}",
            include_str!("strata-sampler-common.wgsl"),
            match self {
                Self::Greedy { .. } => include_str!("strata-sampler-greedy.wgsl"),
                Self::Part => include_str!("strata-sampler-part.wgsl"),
                Self::Merge => include_str!("strata-sampler-merge.wgsl"),
            }
        )
        .replace(
            "BITMAP_WORDS",
            if matches!(self, Self::Greedy { penalties: false }) {
                "1"
            } else {
                "7760"
            },
        )
    }
    fn threads(self) -> u32 {
        match self {
            Self::Greedy { .. } => 1024,
            Self::Part => 128,
            Self::Merge => 32,
        }
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Greedy { penalties: true } => "StrataNativeSamplerGreedyPenalties",
            Self::Greedy { penalties: false } => "StrataNativeSamplerGreedy",
            Self::Part => "StrataNativeSamplerSplitPart",
            Self::Merge => "StrataNativeSamplerSplitMerge",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(self.threads()))
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
            cube_dim: CubeDim::new_1d(self.threads()),
            debug_info: None,
        })
    }
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Greedy { penalties: false },
        Kernel::Greedy { penalties: true },
        Kernel::Part,
        Kernel::Merge,
    ]
    .into_iter()
    .map(|k| CompiledKernel {
        entrypoint_name: "main".into(),
        debug_name: None,
        source: k.source(),
        repr: None,
        cube_dim: CubeDim::new_1d(k.threads()),
        debug_info: None,
    })
    .collect()
}
pub(crate) fn scratch_bytes(rows: usize) -> Result<u64> {
    if rows == 0 || rows > crate::strata_port::verify_capacity() {
        return Err(reject());
    }
    Ok((rows * BLOCKS * KMAX * 8) as u64)
}
/// Original device inputs/outputs. No CPU fallback or implicit GPU allocation.
/// # Safety
/// Same protected sole-worker client, configured owned transient I/O, retained
/// complete owning PrivateArena grants and original vocabulary. Device admits
/// 1024 threads/32KiB shared memory and native32 subgroup. All views are private,
/// all prior writers are ordered, and owners survive both native queues' join.
/// Owners are [logits, per-row history, split candidates, id/probability output,
/// sticky fault, request parameters]. The last buffer must contain exactly
/// `params.words(rows, history, first, stride)` at execution. Refresh its DEVICE
/// contents at each target window/draft position; do not capture request values
/// into graph metadata. Views authorize no allocation/free/erasure by themselves.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch(
    client: &ComputeClient<WgpuRuntime>,
    owners: [Handle; 6],
    params: &Params,
    rows: usize,
    history: usize,
    first: usize,
    stride: usize,
) -> Result<()> {
    if !crate::strata_intel_core::enabled() {
        return Err(reject());
    }
    let _checked_words = params.words(rows, history, first, stride)?;
    let need = [
        (rows * VOCAB * 4) as u64,
        (rows * history * 4).max(4) as u64,
        if params.greedy() {
            4
        } else {
            scratch_bytes(rows)?
        },
        first
            .checked_add(
                (rows - 1)
                    .checked_mul(stride)
                    .ok_or(Error::ArithmeticOverflow)?,
            )
            .and_then(|n| n.checked_add(2))
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::ArithmeticOverflow)? as u64,
        4,
        128,
    ];
    for (i, h) in owners.iter().enumerate() {
        if h.stream != owners[0].stream
            || h.size_in_used() < need[i]
            || !h.size_in_used().is_multiple_of(4)
        {
            return Err(reject());
        }
        for old in &owners[..i] {
            if old.clone().binding().memory.allocation_id()
                == h.clone().binding().memory.allocation_id()
            {
                let a = old.offset_start.unwrap_or(0);
                let b = h.offset_start.unwrap_or(0);
                if a < b
                    .checked_add(h.size_in_used())
                    .ok_or(Error::ArithmeticOverflow)?
                    && b < a
                        .checked_add(old.size_in_used())
                        .ok_or(Error::ArithmeticOverflow)?
                {
                    return Err(reject());
                }
            }
        }
    }
    let greedy = [Kernel::Greedy {
        penalties: params.penalty_last_n > 0 && history > 0,
    }];
    for kernel in if params.greedy() {
        &greedy[..]
    } else {
        &[Kernel::Part, Kernel::Merge][..]
    } {
        let mut args = KernelArguments::new();
        for h in &owners {
            args = args.with_buffer(h.clone().binding());
        }
        let grid = if *kernel == Kernel::Part {
            [BLOCKS as u32, rows as u32, 1]
        } else {
            [rows as u32, 1, 1]
        };
        unsafe {
            client.launch_unchecked(
                Box::new(*kernel),
                CubeCount::Static(grid[0], grid[1], grid[2]),
                args,
            );
        }
    }
    Ok(())
}
#[cfg(test)]
#[path = "strata_sampler_tests.rs"]
mod tests;

/// Pinned sampler.hpp::penalty_rows. A target row follows the real consumed
/// tail AND its own input prefix. Provisional inputs beyond that row never
/// affect penalties. Empty front slots are -1, never a fabricated token0.
/// The returned transient host staging is wiped on drop, including no-history.
pub(crate) fn penalty_rows(
    tail: &[u32],
    window: &[u32],
    width: usize,
) -> Result<Zeroizing<Vec<i32>>> {
    if window.is_empty()
        || window.len() > crate::strata_port::verify_capacity()
        || width > HISTORY_MAX
        || tail.len() > HISTORY_MAX
        || tail.iter().chain(window).any(|&id| id >= VOCAB as u32)
    {
        return Err(reject());
    }
    if width == 0 {
        return Ok(Zeroizing::new(vec![-1]));
    }
    let mut history = Zeroizing::new(vec![-1; window.len() * width]);
    for row in 0..window.len() {
        let available = tail.len() + row + 1;
        let take = width.min(available);
        for j in 0..take {
            let i = available - take + j;
            history[row * width + width - take + j] = if i < tail.len() {
                tail[i]
            } else {
                window[i - tail.len()]
            } as i32;
        }
    }
    Ok(history)
}
/// Coupled draft0 has precisely the target's row0 history. Later draft history
/// is this base plus actual device-produced draft tokens (not host guesses).
pub(crate) fn coupled_base(tail: &[u32], next: u32, width: usize) -> Result<Zeroizing<Vec<i32>>> {
    penalty_rows(tail, std::slice::from_ref(&next), width)
}
/// MTP cell c predicts T[c+2], checked against target row c+1. This is NOT c+2.
pub(crate) fn coupled_counter(cell: usize) -> Result<u64> {
    u64::try_from(cell)
        .ok()
        .and_then(|c| c.checked_add(1))
        .ok_or(Error::ArithmeticOverflow)
}
