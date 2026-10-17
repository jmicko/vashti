//! MIT Strata fb58e0 MTP S2 expert and native router/shared consumers.
//! Original Q2 GGUF pages stay borrowed; request-local quantized activation
//! planes carry transposed bytes, F32 scales and signed sums for native S2 math.
use crate::{private_arena::PrivateArena, resident_util as u};
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
    Prepare,
    ActivateQuant,
    Gu,
    Down,
    RouterDot,
    Route,
    SharedActivate,
    SharedScale,
    Combine,
}
impl Kernel {
    fn dim(self) -> CubeDim {
        CubeDim::new_1d(if matches!(self, Self::SharedScale | Self::Combine) {
            128
        } else {
            256
        })
    }
    fn source(self) -> String {
        let all = include_str!("strata_mtp_moe.wgsl");
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
            Self::Prepare | Self::ActivateQuant => "QUANT",
            Self::Gu | Self::Down => "EXPERT",
            Self::RouterDot => "DOT",
            Self::Route => "ROUTE",
            Self::SharedActivate => "SHARED",
            Self::SharedScale => "SCALE",
            Self::Combine => "COMBINE",
        }));
        s.replace(
            "ACTIVATE",
            if self == Self::ActivateQuant {
                "true"
            } else {
                "false"
            },
        )
        .replace("DOWN", if self == Self::Down { "true" } else { "false" })
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare => "StrataMtpS2Prepare",
            Self::ActivateQuant => "StrataMtpS2SwiGluQuant",
            Self::Gu => "StrataMtpS2GateUpPairs",
            Self::Down => "StrataMtpS2DownPairs",
            Self::RouterDot => "StrataMtpNativeBf16Dot",
            Self::Route => "StrataMtpNativeRouter",
            Self::SharedActivate => "StrataMtpNativeSharedSwiGluQ8",
            Self::SharedScale => "StrataMtpNativeSharedScale",
            Self::Combine => "StrataMtpNativeCombine",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().info(*self)
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
            cube_dim: self.dim(),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(
    arena: &PrivateArena,
    kernel: Kernel,
    grid: usize,
    handles: Vec<Handle>,
    info: [u32; 8],
) -> Result<()> {
    let client = arena.client().map_err(u::fault)?;
    if !crate::strata_intel_core::enabled()
        || grid == 0
        || grid > 65535
        || handles.is_empty()
        || handles.iter().any(|h| h.stream != handles[0].stream)
        || client.properties().hardware.max_units_per_cube < 256
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
    // SAFETY: connected MTP owner validates original role/page provenance and
    // admitted full private spans. Each kernel also checks its live bounds.
    unsafe {
        client.launch_unchecked(Box::new(kernel), CubeCount::Static(grid as u32, 1, 1), args);
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Kernel::Prepare,
        Kernel::ActivateQuant,
        Kernel::Gu,
        Kernel::Down,
        Kernel::RouterDot,
        Kernel::Route,
        Kernel::SharedActivate,
        Kernel::SharedScale,
        Kernel::Combine,
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
    fn shared_scale_and_combined_rows_have_native_vec4_geometry_and_original_fault_roles() {
        use wgpu::naga as n;
        for (kernel, bindings) in [(Kernel::SharedScale, 4), (Kernel::Combine, 7)] {
            let source = kernel.source();
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [128, 1, 1]);
            assert_eq!(kernel.dim().x, 128);
            let entry = info.get_entry_point(0);
            let mut actual = std::collections::BTreeMap::new();
            for (id, global) in module.global_variables.iter() {
                if let Some(binding) = &global.binding {
                    actual.insert(
                        binding.binding,
                        (
                            entry[id].contains(n::valid::GlobalUse::READ),
                            entry[id].contains(n::valid::GlobalUse::WRITE),
                        ),
                    );
                }
            }
            assert_eq!(actual.len(), bindings);
            if kernel == Kernel::SharedScale {
                assert_eq!(actual[&0], (true, true)); // Shared row is stored rounded before combine.
                assert_eq!(actual[&1], (true, false)); // Original scalar logit.
                assert!(actual[&2].1); // Separate shared fault; no routed fault dependency.
            } else {
                assert_eq!(actual[&0], (true, false)); // Authentic ordered ten routed rows.
                assert_eq!(actual[&1], (true, false)); // Already scaled shared vec4 row.
                assert!(actual[&4].1);
                assert!(actual[&5].1);
                assert!(
                    module.entry_points[0]
                        .function
                        .expressions
                        .iter()
                        .any(|(_, e)| matches!(
                            e,
                            n::Expression::Math {
                                fun: n::MathFunction::Fma,
                                ..
                            }
                        ))
                );
            }
        }
    }
    // Independent canonical GGUF decoder is the oracle; reproduce the native
    // quantizer/transpose and integer lane arithmetic using public synthetic data.
    fn quant(x: &[f32]) -> Vec<[u32; 10]> {
        x.chunks_exact(32)
            .map(|xs| {
                let scale = xs.iter().fold(0f32, |m, &x| m.max(x.abs())) / 127.;
                let inv = if scale > 0. { 1. / scale } else { 0. };
                let mut q = [0i32; 32];
                for (dst, &x) in q.iter_mut().zip(xs) {
                    let t = x * inv;
                    *dst = t.round() as i32;
                    *dst = (*dst).clamp(-127, 127);
                }
                let mut p = [0u32; 10];
                for (j, out) in p[..8].iter_mut().enumerate() {
                    for b in 0..4 {
                        let at = (j / 4) * 16 + b * 4 + j % 4;
                        *out |= ((q[at] as u32) & 255) << (b * 8);
                    }
                }
                p[8] = scale.to_bits();
                p[9] = q.iter().sum::<i32>() as u32;
                p
            })
            .collect()
    }
    fn lane_dot(w: &[u8], p: &[[u32; 10]], lane: usize) -> f32 {
        let mut acc = 0f32;
        for c in (lane..p.len()).step_by(32) {
            let block = &w[(c / 2) * 18..];
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let mut sum = 0i32;
            for j in 0..8 {
                let off = 2 + (c % 2) * 8 + (j / 4) * 4;
                let code = u32::from_le_bytes(block[off..off + 4].try_into().unwrap());
                let cw = (code >> ((j % 4) * 2)) & 0x03030303;
                for b in 0..4 {
                    sum += ((cw >> (b * 8)) & 255) as i32
                        * (((p[c][j] >> (b * 8)) & 255) as u8 as i8) as i32;
                }
            }
            acc += (scale * f32::from_bits(p[c][8])) * (sum - p[c][9] as i32) as f32;
        }
        acc
    }
    #[test]
    fn s2_transpose_scaled_integer_pairs_match_independent_canonical_q2_weights() {
        for n in [640usize, 2560] {
            for salt in 0..9 {
                let x = (0..n)
                    .map(|i| ((i * 17 + salt * 13) % 79) as f32 / 19. - 2.)
                    .collect::<Vec<_>>();
                let p = quant(&x);
                for row in 0..2 {
                    let mut w = vec![0u8; n / 64 * 18];
                    for (b, block) in w.chunks_exact_mut(18).enumerate() {
                        let scale = half::f16::from_f32((b % 9 + 1) as f32 / 31.);
                        block[..2].copy_from_slice(&scale.to_bits().to_le_bytes());
                        for (j, v) in block[2..].iter_mut().enumerate() {
                            *v = (b * 53 + j * 29 + row * 17 + salt * 11) as u8;
                        }
                    }
                    let mut decoded = vec![0f32; n];
                    kuno_text::TensorType::Q2_0
                        .decode(&w, &mut decoded)
                        .unwrap();
                    let mut expected = 0f64;
                    for (c, block) in p.iter().enumerate() {
                        let mut signed_sum = 0;
                        for j in 0..32 {
                            let word = (j / 16) * 4 + j % 4;
                            let byte = (j % 16) / 4;
                            let q = ((block[word] >> (byte * 8)) & 255) as u8 as i8;
                            signed_sum += q as i32;
                            expected += decoded[c * 32 + j] as f64
                                * (q as f64 * f32::from_bits(block[8]) as f64);
                        }
                        assert_eq!(signed_sum, block[9] as i32);
                    }
                    let mut lanes = std::array::from_fn::<_, 32, _>(|l| lane_dot(&w, &p, l));
                    for d in [16, 8, 4, 2, 1] {
                        let old = lanes;
                        for l in 0..32 - d {
                            lanes[l] += old[l + d];
                        }
                    }
                    assert!(
                        (lanes[0] as f64 - expected).abs() < 1e-4 * (1. + expected.abs()),
                        "n={n} row={row}"
                    );
                }
            }
        }
        assert!(quant(&[0f32; 32])[0].iter().all(|&v| v == 0));
    }
}
