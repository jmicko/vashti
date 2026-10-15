//! MIT Strata fb58e0 verify_kernels.dp.cpp::row_top_prob_split_kernel.
//! Original8x128 geometry, XOR32 sums and ascending32-warp final sum.
//! Caller lends two disjoint views of the COMPLETE private Candidates owner.
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
pub(crate) const WORD: usize = 64;
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Kernel {
    Argmax,
    Probability,
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Argmax => "StrataNativeMtpArgmax",
            Self::Probability => "StrataNativeMtpTopProbability",
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
        Ok(compiled(*self))
    }
}
impl Kernel {
    fn lanes(self) -> u32 {
        if self == Self::Argmax { 1024 } else { 128 }
    }
}
fn compiled(k: Kernel) -> CompiledKernel<AutoCompiler> {
    CompiledKernel {
        entrypoint_name: "main".into(),
        debug_name: None,
        source: match k {
            Kernel::Argmax => include_str!("strata-mtp-argmax.wgsl"),
            Kernel::Probability => include_str!("strata-mtp-probability.wgsl"),
        }
        .into(),
        repr: None,
        cube_dim: CubeDim::new_1d(k.lanes()),
        debug_info: None,
    }
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [Kernel::Argmax, Kernel::Probability]
        .into_iter()
        .map(compiled)
        .collect()
}
/// # Safety
/// Original full-vocabulary logits/selected argmax and fault are authenticated
/// private roots in the same protected client. Candidates is its complete,
/// newly zeroed32KiB owner, retained through both queues' join/arena erasure.
pub(crate) unsafe fn launch(
    client: &ComputeClient<WgpuRuntime>,
    logits: Handle,
    selected: Handle,
    candidates: Handle,
    fault: Handle,
) -> Result<()> {
    if !crate::strata_intel_core::enabled()
        || logits.size_in_used() < (248320 * 4) as u64
        || selected.size_in_used() < 4
        || fault.size_in_used() < 4
        || candidates.size() != 32768
        || candidates.size_in_used() != 32768
        || candidates.offset_start.unwrap_or(0) != 0
        || candidates.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidBufferLength);
    }
    let offset = (WORD * 4) as u64;
    let scratch = candidates.clone().offset_end(32768 - offset);
    let probability = candidates
        .clone()
        .offset_start(offset)
        .offset_end(32768 - offset - 4);
    let handles = [logits, selected, scratch, probability, fault];
    for (i, h) in handles.iter().enumerate() {
        if h.stream != handles[0].stream {
            return Err(Error::InvalidBufferLength);
        }
        for a in &handles[..i] {
            if a.clone().binding().memory.allocation_id()
                == h.clone().binding().memory.allocation_id()
            {
                let x = a.offset_start.unwrap_or(0);
                let y = h.offset_start.unwrap_or(0);
                if x < y + h.size_in_used() && y < x + a.size_in_used() {
                    return Err(Error::InvalidBufferLength);
                }
            }
        }
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel::Probability),
            CubeCount::Static(8, 1, 1),
            args,
        );
    }
    Ok(())
}

/// # Safety
/// Original full private logits/selected/fault roots, protected sole client,
/// all complete owner grants retained until the parent's output/fault join.
pub(crate) unsafe fn argmax(
    client: &ComputeClient<WgpuRuntime>,
    logits: Handle,
    selected: Handle,
    fault: Handle,
) -> Result<()> {
    let handles = [logits, selected, fault];
    let need = [248320 * 4, 4, 4];
    if !crate::strata_intel_core::enabled() {
        return Err(Error::InvalidBufferLength);
    }
    for (i, h) in handles.iter().enumerate() {
        if h.size_in_used() < need[i]
            || h.stream != handles[0].stream
            || handles[..i].iter().any(|a| {
                a.clone().binding().memory.allocation_id()
                    == h.clone().binding().memory.allocation_id()
            })
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    unsafe {
        client.launch_unchecked(Box::new(Kernel::Argmax), CubeCount::Static(1, 1, 1), args);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_confidence_shader_memory_order_and_geometry_lower_without_device() {
        use wgpu::naga as n;
        for (index, s) in sources().into_iter().enumerate() {
            let module = n::front::wgsl::parse_str(&s.source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s.source)));
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s.source)));
            assert_eq!(module.entry_points[0].workgroup_size, [s.cube_dim.x, 1, 1]);
            let words = n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options {
                    lang_version: (1, 3),
                    zero_initialize_workgroup_memory:
                        n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                    ..Default::default()
                },
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            if index == 1 {
                let mut constants = std::collections::HashMap::new();
                let mut barriers = Vec::new();
                let mut atomic_rmw = false;
                let mut at = 5;
                while at < words.len() {
                    let len = (words[at] >> 16) as usize;
                    let op = words[at] & 65535;
                    let inst = &words[at..at + len];
                    if op == 43 && len == 4 {
                        constants.insert(inst[2], inst[3]);
                    }
                    if op == 224 {
                        barriers.push((inst[1], inst[2], inst[3]));
                    }
                    atomic_rmw |= op == 234;
                    at += len;
                }
                assert!(atomic_rmw);
                assert_eq!(
                    barriers
                        .iter()
                        .filter(|(exec, scope, semantics)| constants.get(exec) == Some(&2)
                            && constants.get(scope) == Some(&1)
                            && constants.get(semantics) == Some(&72))
                        .count(),
                    2
                );
            }
        }
    }
    #[test]
    fn native_confidence_xor_warp_sum_matches_full_vocab_probability() {
        fn probability(logits: &[f32], maximum: f32) -> f32 {
            let mut thread = [0f32; 1024];
            for (t, s) in thread.iter_mut().enumerate() {
                for i in (t..logits.len()).step_by(1024) {
                    *s += (logits[i] - maximum).exp();
                }
            }
            let mut parts = [0f32; 32];
            for (w, part) in parts.iter_mut().enumerate() {
                let mut lanes: [f32; 32] = thread[w * 32..w * 32 + 32].try_into().unwrap();
                for off in [16, 8, 4, 2, 1] {
                    let before = lanes;
                    for lane in 0..32 {
                        lanes[lane] += before[lane ^ off];
                    }
                }
                *part = lanes[0];
            }
            1. / parts.into_iter().fold(0f32, |s, x| s + x)
        }
        let uniform = vec![0f32; 248320];
        assert_eq!(probability(&uniform, 0.), 1. / 248320f32);
        let values: Vec<_> = (0..248320).map(|i| (i % 37) as f32 * 0.11).collect();
        let max = 36f32 * 0.11;
        let reference = 1.
            / values
                .iter()
                .map(|&x| f64::from(x - max).exp())
                .sum::<f64>();
        let actual = f64::from(probability(&values, max));
        assert!((actual - reference).abs() / reference < 0.00002);
    }
}
