//! MIT Strata Intel prefill.cpp's native FP16 expert arithmetic, connected to
//! original global Sources, the real router map and the parent private owners.
//! Serial Vulkan stage reuse is explicit; full chunk/ring/queue scheduling is
//! still a separate required part of the complete prefill port.
use super::*;
use crate::{quant_matvec as q, strata_global_sources::Sources};
use kuno_text::model::BoundGraph;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Op {
    Gather,
    Dequant(u32),
    Gemm,
    Activate,
}
impl Op {
    fn lanes(self) -> u32 {
        if self == Self::Gemm { 64 } else { 256 }
    }
    fn source(self) -> String {
        let all = include_str!("strata_prompt_experts.wgsl");
        let section = |name: &str| {
            all.split_once(&format!("// {name}_BEGIN\n"))
                .unwrap()
                .1
                .split_once(&format!("// {name}_END"))
                .unwrap()
                .0
        };
        let mut s = section("COMMON").to_owned();
        if let Self::Dequant(dtype) = self {
            let helpers = include_str!("iq_moe_integer.wgsl")
                .split_once("fn iq4_byte")
                .unwrap()
                .1
                .split_once("// Original byte-addressed")
                .unwrap()
                .0;
            s.push_str(&format!("fn iq4_byte{helpers}"));
            let (decode, value) = if matches!(dtype, q::IQ4_NL | q::Q2_0) {
                let source = q::frozen_down_integer::decode(dtype).replace("down", "gate");
                (
                    source,
                    "let e=gate_prepare(row*p.row_bytes,col/32u);let codes=gate_packed(e,(col%32u)/4u);return e.d*signed_byte((codes>>((col%4u)*8u))&255u);",
                )
            } else {
                (
                    q::frozen_gu_integer::decode("gate", dtype),
                    "let e=gate_prepare(row*p.row_bytes,col/32u);let codes=gate_packed(e,(col%32u)/4u);return gate_factor(e,(col%32u)/16u)*signed_byte((codes>>((col%4u)*8u))&255u);",
                )
            };
            let offset = match dtype {
                q::IQ3_XXS => 0,
                q::IQ3_S => 256,
                q::IQ2_S => 768,
                _ => 0,
            };
            let decode = decode.replace("iq3_grid[", &format!("strata_grid[{offset}u+"));
            s.push_str(&crate::strata_expert_fused::replace_reads(decode, "gate"));
            s.push_str(&format!("fn value(row:u32,col:u32)->f32 {{{value}}}\n"));
        }
        s.push_str(section(match self {
            Self::Gather => "GATHER",
            Self::Dequant(_) => "DEQUANT",
            Self::Gemm => "GEMM",
            Self::Activate => "ACTIVATE",
        }));
        s
    }
}
impl KernelMetadata for Op {
    fn name(&self) -> &'static str {
        match self {
            Self::Gather => "StrataPromptExpertGather",
            Self::Dequant(_) => "StrataPromptExpertF16Image",
            Self::Gemm => "StrataPromptExpertGemm",
            Self::Activate => "StrataPromptExpertF16Swiglu",
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
impl CubeTask<AutoCompiler> for Op {
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
fn dispatch(
    client: &ComputeClient<WgpuRuntime>,
    op: Op,
    grid: [usize; 3],
    mut args: KernelArguments,
    words: [u32; 16],
) -> Result<()> {
    if grid.contains(&0) || grid.iter().any(|&v| v > 65535) || !active() {
        return Err(Error::InvalidBufferLength);
    }
    args = args.with_info(MetadataBindingInfo::custom(
        words
            .chunks_exact(2)
            .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
            .collect(),
    ));
    // SAFETY: parent authenticates the original geometry and distinct private
    // owners. GPU map guards select real rows; stages are reused in queue order.
    unsafe {
        client.launch_unchecked(
            Box::new(op),
            CubeCount::Static(grid[0] as u32, grid[1] as u32, grid[2] as u32),
            args,
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue(
    client: &ComputeClient<WgpuRuntime>,
    graph: BoundGraph<'_, '_>,
    sources: &Sources,
    layer: usize,
    geometry: q::grouped_moe::Geometry,
    owners: &[Handle; 7],
    fault: &Handle,
    count: usize,
) -> Result<()> {
    sources.validate_graph_ready(graph)?;
    let stage = stages()?;
    if !(1..=8).contains(&count)
        || fault.size_in_used() != 4
        || owners
            .iter()
            .chain(stage.iter())
            .chain([fault])
            .any(|h| h.stream != owners[0].stream)
        || owners[0].size_in_used() != 32768
        || owners[1].size_in_used() < (count * 2560 * 4) as u64
        || owners[2].size_in_used() != 262144
        || owners[3].size_in_used() != 1048576
        || owners[6].size_in_used() != 32768
    {
        return Err(Error::InvalidBufferLength);
    }
    let projections = geometry.projections();
    for (role, p) in projections.iter().enumerate() {
        if p.experts() != 512
            || p.columns() != if role == 2 { 640 } else { 2560 }
            || p.rows() != if role == 2 { 2560 } else { 640 }
            || !matches!(
                (role, p.dtype()),
                (0 | 1, q::IQ2_S | q::IQ3_S | q::IQ3_XXS | q::IQ4_XS) | (2, q::IQ4_NL | q::Q2_0)
            )
            || (p.rows() * p.columns() * 2) as u64 > stage[1].size_in_used()
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    // Each role's original HOST bytes occupy stage2. Stage0 holds gathered
    // FP16 activations, stage1 one expert's FP16 matrix. Resident bytes are read
    // from their real original owner; no second whole model is materialized.
    for (role, p) in projections.iter().enumerate() {
        if role == 2 {
            let words = [
                count as u32,
                640,
                640,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            let args = KernelArguments::new().with_buffers(vec![
                owners[0].clone().binding(),
                owners[3].clone().binding(),
                owners[2].clone().binding(),
                owners[6].clone().binding(),
                fault.clone().binding(),
            ]);
            dispatch(
                client,
                Op::Activate,
                [(count * 10 * 640 / 2).div_ceil(256), 1, 1],
                args,
                words,
            )?;
        }
        crate::strata_global_experts::fetch_role(
            client, graph, sources, layer, geometry, &owners[0], &owners[6], &stage[2], count, role,
        )?;
        for group in 0..count * 10 {
            let words = [
                count as u32,
                p.columns() as u32,
                p.rows() as u32,
                p.row_bytes() as u32,
                group as u32,
                role as u32,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            let args = KernelArguments::new().with_buffers(vec![
                owners[0].clone().binding(),
                owners[if role == 2 { 2 } else { 1 }].clone().binding(),
                stage[0].clone().binding(),
                fault.clone().binding(),
            ]);
            dispatch(
                client,
                Op::Gather,
                [(count * p.columns() / 2).div_ceil(256), 1, 1],
                args,
                words,
            )?;
            let args = KernelArguments::new().with_buffers(vec![
                sources.prototype(graph, layer, role)?.binding(),
                owners[0].clone().binding(),
                stage[2].clone().binding(),
                stage[1].clone().binding(),
                owners[6].clone().binding(),
                fault.clone().binding(),
            ]);
            let args = crate::strata_intel_core::bind(args, owners[0].stream)?;
            let args = sources.bind(layer, args)?;
            dispatch(
                client,
                Op::Dequant(p.dtype()),
                [(p.rows() * p.columns() / 2).div_ceil(256), 1, 1],
                args,
                words,
            )?;
            let args = KernelArguments::new().with_buffers(vec![
                owners[0].clone().binding(),
                stage[0].clone().binding(),
                stage[1].clone().binding(),
                owners[3].clone().binding(),
                owners[6].clone().binding(),
                fault.clone().binding(),
            ]);
            dispatch(client, Op::Gemm, [p.rows().div_ceil(8), 1, 1], args, words)?;
        }
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Op::Gather,
        Op::Dequant(q::IQ2_S),
        Op::Dequant(q::IQ3_S),
        Op::Dequant(q::IQ3_XXS),
        Op::Dequant(q::IQ4_XS),
        Op::Dequant(q::IQ4_NL),
        Op::Dequant(q::Q2_0),
        Op::Gemm,
        Op::Activate,
    ]
    .into_iter()
    .map(|op| {
        op.compile(
            &mut AutoCompiler::Wgsl(Default::default()),
            &Default::default(),
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap()
    })
    .collect()
}
