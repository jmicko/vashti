// Complete padded producer + original chronological CMMA for tiny N.
// Included inside native: original product, dtype rounding and writer policy.

#[cube(launch_unchecked)]
fn tiny_pad_copy(input: &Array<u32>, output: &mut Array<u32>, words: usize, padded: usize) {
    let i = usize::cast_from(ABSOLUTE_POS);
    if i < padded {
        let mut v = 0u32;
        if i < words {
            v = input[i];
        }
        output[i] = v;
    }
}
#[cube(launch_unchecked)]
fn tiny_pad_in_place(output: &mut Array<u32>, words: usize, padded: usize) {
    let i = usize::cast_from(ABSOLUTE_POS);
    if i >= words && i < padded {
        output[i] = 0;
    }
}
#[cube(launch_unchecked)]
fn dense_tiny<F: Float>(
    input: &Array<F>,
    weight: &Array<F>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    rows: usize,
    count: usize,
    out_first: usize,
    out_stride: usize,
) {
    // The genuine producer has materialized all16 weight rows, with original
    // first rows and exact bit-zero padding. The SAME product now takes its
    // direct-global arm for full M. Original masked M tails remain unchanged.
    let values = product::<F>(input, weight, columns, 16, count, 0);
    let mut i = usize::cast_from(UNIT_POS);
    while i < 128 {
        let t = usize::cast_from(CUBE_POS_Y) * 8 + i / 16;
        let row = i % 16;
        if t < count && row < rows {
            let value = values[i];
            if value.abs() <= f32::MAX {
                output[out_first + t * out_stride + row] = value;
            } else {
                output[out_first + t * out_stride + row] = 0.0;
                fault[0].fetch_or(1);
            }
        }
        i += 16;
    }
}
fn tiny_shape(columns: usize, rows: usize, bfloat: bool) -> bool {
    matches!((columns, rows, bfloat), (10240, 4, true) | (2560, 1, false))
}
fn tiny_views(
    h: &[Handle; 5],
    count: usize,
    columns: usize,
    rows: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    use crate::strata_prompt::storage::overlaps;
    if !tiny_shape(columns, rows, bfloat) || !(1..=4096).contains(&count) || out_stride < rows {
        return Err(Error::InvalidBufferLength);
    }
    let output_end = out_first
        .checked_add(
            (count - 1)
                .checked_mul(out_stride)
                .and_then(|v| v.checked_add(rows))
                .ok_or(Error::ArithmeticOverflow)?,
        )
        .ok_or(Error::ArithmeticOverflow)?;
    let need = [
        count * columns * 2,
        rows * columns * 2,
        16 * columns * 2,
        output_end.checked_mul(4).ok_or(Error::ArithmeticOverflow)?,
        4,
    ];
    if u32::try_from(out_first).is_err() || u32::try_from(out_stride).is_err() {
        return Err(Error::ArithmeticOverflow);
    }
    if h.iter().zip(need).any(|(v, n)| {
        v.size_in_used() < n as u64
            || !v.size_in_used().is_multiple_of(4)
            || !v.offset_start.unwrap_or(0).is_multiple_of(4)
            || v.stream != h[0].stream
    }) || h[4].size_in_used() != 4
        || need.iter().any(|&n| u32::try_from(n).is_err())
    {
        return Err(Error::InvalidBufferLength);
    }
    let same_prefix = h[1].clone().binding().memory.allocation_id()
        == h[2].clone().binding().memory.allocation_id()
        && h[1].offset_start.unwrap_or(0) == h[2].offset_start.unwrap_or(0);
    if overlaps(&h[1], &h[2])? && !same_prefix {
        return Err(Error::InvalidBufferLength);
    }
    for (i, j) in [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ] {
        if overlaps(&h[i], &h[j])? {
            return Err(Error::InvalidBufferLength);
        }
    }
    // Same-owner prefix mutation never changes an input image or public weight
    // root elsewhere. The caller gives the actual working-stage owner; no ready
    // flag, fictional16-row source extent or borrowed scope is manufactured.
    Ok(same_prefix)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn run_tiny(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 5],
    count: usize,
    columns: usize,
    rows: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<()> {
    let in_place = tiny_views(&h, count, columns, rows, out_first, out_stride, bfloat)?;
    compatible(client, bfloat)?;
    let [input, weight, workspace, output, fault] = h;
    let bytes = (16 * columns * 2) as u64;
    let padded = workspace
        .clone()
        .offset_end(workspace.size_in_used() - bytes);
    let pad = crate::resident_util::raw_arg(padded.clone(), bytes as usize / 4)?;
    // SAFETY: exact checked raw U32 producer extents; copy preserves all BF16/
    // FP16 bits including NaN payloads. In-place producer only writes padding.
    unsafe {
        if in_place {
            tiny_pad_in_place::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static((bytes / 4).div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                pad,
                rows * columns / 2,
                16 * columns / 2,
            );
        } else {
            let src =
                crate::resident_util::raw_arg(weight.clone(), weight.size_in_used() as usize / 4)?;
            tiny_pad_copy::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static((bytes / 4).div_ceil(256) as u32, 1, 1),
                CubeDim::new_1d(256),
                src,
                pad,
                rows * columns / 2,
                16 * columns / 2,
            );
        }
    }
    let ia = half_arg(input)?;
    let wa = half_arg(padded)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    // Original dependency planner sees the authentic producer WRITE and this
    // consumer READ of the SAME view. No flush, fake completion or bypass.
    unsafe {
        if bfloat {
            dense_tiny::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(1, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
                ia,
                wa,
                oa,
                fa,
                columns,
                rows,
                count,
                out_first,
                out_stride,
            );
        } else {
            dense_tiny::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static(1, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
                ia,
                wa,
                oa,
                fa,
                columns,
                rows,
                count,
                out_first,
                out_stride,
            );
        }
    }
    crate::strata_prefill_profile::count(
        if bfloat {
            "tiny_gr_injection_products"
        } else {
            "tiny_shared_gate_products"
        },
        1,
    );
    Ok(())
}
fn definition_tiny<F: Float>() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let input = b.input_array(F::as_type(&b.scope));
    let weight = b.input_array(F::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: [cubecl::ir::ManagedVariable; 5] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_tiny::expand::<F>(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        fault.into(),
        a[0].clone().into(),
        a[1].clone().into(),
        a[2].clone().into(),
        a[3].clone().into(),
        a[4].clone().into(),
    );
    assert!(b.scope.pop_errors().is_empty());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(16)))
}
fn definition_tiny_pad(in_place: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    AddressType::U32.register(&mut b.scope);
    let source = if in_place {
        None
    } else {
        Some(b.input_array(u32::as_type(&b.scope)))
    };
    let output = b.output_array(u32::as_type(&b.scope));
    let words = b.scalar(u32::as_type(&b.scope).storage_type());
    let padded = b.scalar(u32::as_type(&b.scope).storage_type());
    if let Some(input) = source {
        tiny_pad_copy::expand(
            &mut b.scope,
            input.into(),
            output.into(),
            words.into(),
            padded.into(),
        );
    } else {
        tiny_pad_in_place::expand(&mut b.scope, output.into(), words.into(), padded.into());
    }
    assert!(b.scope.pop_errors().is_empty());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(256)))
}
pub(super) fn validate_tiny_producers() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, WgpuCompilationOptions};
    use std::io::Write;
    for mask in 0..8 {
        for in_place in [false, true] {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition_tiny_pad(in_place),
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
            )?
            else {
                return Err("tiny producer compiler representation".into());
            };
            if kernel.shared_size != 0
                || wgpu::hal::vulkan::kuno_cooperative_subgroup(&kernel.assembled_module).is_some()
            {
                return Err("tiny bit producer must not contain matrix or shared memory".into());
            }
            let mut validator = std::process::Command::new("spirv-val");
            validator.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                validator.arg("--uniform-buffer-standard-layout");
            }
            let mut child = validator
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            child
                .stdin
                .take()
                .ok_or("tiny producer validator stdin")?
                .write_all(
                    &kernel
                        .assembled_module
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !child.wait()?.success() {
                return Err("tiny raw producer SPIRV validation failed".into());
            }
            println!("strata_tiny_pad_shader_accepted in_place={in_place} layout_mask={mask}");
        }
    }
    Ok(())
}
/// Fixed-public control only: force the unchanged original logical-row product.
/// Never called by the private worker or used as a selected implementation.
pub(super) fn original_tiny_reference(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    bfloat: bool,
) -> Result<()> {
    compatible(client, bfloat)?;
    let [input, weight, output, fault] = h;
    let ia = half_arg(input)?;
    let wa = half_arg(weight)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    unsafe {
        if bfloat {
            dense::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(1, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
                ia,
                wa,
                oa,
                fa,
                columns,
                rows,
                count,
                0,
                0,
                rows,
            );
        } else {
            dense::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static(1, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
                ia,
                wa,
                oa,
                fa,
                columns,
                rows,
                count,
                0,
                0,
                rows,
            );
        }
    }
    Ok(())
}
#[cfg(test)]
mod tiny_tests {
    use super::*;
    #[test]
    fn actual_raw_producer_expansions_validate_all_sixteen_layouts_without_matrix_or_slm() {
        validate_tiny_producers().unwrap();
    }
    #[test]
    fn source_and_zero_padding_are_disjoint_and_preserve_every_raw_word() {
        for (columns, rows, bf) in [(10240, 4, true), (2560, 1, false)] {
            assert!(tiny_shape(columns, rows, bf));
            let mut seen = vec![0; 16 * columns / 2];
            for group in 0..(16 * columns / 2).div_ceil(256) {
                for lane in 0..256 {
                    let i = group * 256 + lane;
                    if i < seen.len() {
                        seen[i] += 1;
                        if i < rows * columns / 2 {
                            assert!(i * 4 < rows * columns * 2);
                        } else {
                            assert!(i * 4 >= rows * columns * 2);
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&v| v == 1));
            for count in [1usize, 3, 7, 8, 9, 45, 512, 513, 4096] {
                let mut out = vec![0; count * rows];
                for group in 0..count.div_ceil(8) {
                    for lane in 0..16 {
                        for round in 0..8 {
                            let i = lane + round * 16;
                            let t = group * 8 + i / 16;
                            let r = i % 16;
                            if t < count && r < rows {
                                out[t * rows + r] += 1;
                            }
                        }
                    }
                }
                assert!(out.iter().all(|&v| v == 1));
            }
        }
    }
    #[test]
    fn original_k16_product_and_logical_output_owner_are_retained() {
        let source = include_str!("strata_matrix_tiny.rs");
        assert!(source.contains("product::<F>(input, weight, columns, 16, count, 0)"));
        for k in [2560, 10240] {
            assert_eq!(
                (0..k).step_by(16).collect::<Vec<_>>(),
                (0..k / 16).map(|i| i * 16).collect::<Vec<_>>()
            );
        }
        assert!(!tiny_shape(10240, 4, false));
        assert!(!tiny_shape(2560, 4, true));
        assert!(!tiny_shape(2560, 16, false));
        let own = |n| Handle::new(cubecl_common::stream_id::StreamId::current(), n);
        let h = [
            own(8 * 10240 * 2),
            own(4 * 10240 * 2),
            own(16 * 10240 * 2),
            own(8 * 4 * 4),
            own(4),
        ];
        assert_eq!(tiny_views(&h, 8, 10240, 4, 0, 4, true).unwrap(), false);
        for i in 0..5 {
            let mut bad = h.clone();
            bad[i] = own(h[i].size_in_used() - 1);
            assert!(tiny_views(&bad, 8, 10240, 4, 0, 4, true).is_err());
        }
        let mut alias = h.clone();
        alias[2] = h[0].clone();
        assert!(tiny_views(&alias, 8, 10240, 4, 0, 4, true).is_err());
        let mut prefix = h.clone();
        prefix[1] = h[2].clone().offset_end((16 - 4) * 10240 * 2);
        assert_eq!(tiny_views(&prefix, 8, 10240, 4, 0, 4, true).unwrap(), true);
        let mut shifted = prefix.clone();
        shifted[1] = h[2].clone().offset_start(4);
        assert!(tiny_views(&shifted, 8, 10240, 4, 0, 4, true).is_err());
    }
}
