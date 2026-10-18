//! Native Intel matrix instructions for the pinned Strata rounded prompt path.
//! Data stays in the already admitted BF16/FP16 images and F32 result owners.
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{Error, Result};

#[cfg(feature = "strata-intel-core")]
#[path = "strata_matrix_numeric_probe.rs"]
mod numeric;
#[cfg(feature = "strata-intel-core")]
pub(crate) use numeric::cases as public_numeric;

pub(crate) async fn request_device(
    adapter: &wgpu::Adapter,
    descriptor: &wgpu::DeviceDescriptor<'_>,
) -> std::result::Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    #[cfg(feature = "strata-intel-core")]
    {
        if !descriptor
            .required_features
            .contains(crate::strata_intel_core::device_features())
        {
            return Err("native inference device features rejected".into());
        }
        if descriptor.required_limits.max_storage_buffer_binding_size
            < crate::strata_prompt::allocation::ROOT_BYTES
            || descriptor.required_limits.max_buffer_size
                < crate::strata_prompt::allocation::ROOT_BYTES
        {
            return Err("native prompt owner device limits rejected".into());
        }
        cubecl::wgpu::vulkan::request_strata_device(
            adapter,
            descriptor.required_features,
            descriptor.required_limits.clone(),
        )
        .ok_or_else(|| "native Intel matrix device setup rejected".into())
    }
    #[cfg(not(feature = "strata-intel-core"))]
    {
        Ok(adapter.request_device(descriptor).await?)
    }
}
#[cfg(feature = "strata-intel-core")]
mod native {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind, features::MmaConfig};
    use half::{bf16, f16};

    /// One8x16 tile per16-lane subgroup. Shared loads handle partial rows and
    /// output pages without reading outside an original packed weight owner.
    #[cube]
    fn product<F: Float>(
        input: &Array<F>,
        weight: &Array<F>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
    ) -> SharedMemory<f32> {
        let mut a = SharedMemory::<F>::new(128usize);
        let mut b = SharedMemory::<F>::new(256usize);
        let mut out = SharedMemory::<f32>::new(128usize);
        // SAFETY: every loop initializes both operands with load before use;
        // host validation guarantees a nonempty reduction dimension.
        let am = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        let bm = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::B,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
            )
        };
        let acc = cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        );
        let token = usize::cast_from(CUBE_POS_Y) * 8;
        let row = usize::cast_from(CUBE_POS_X) * 16;
        let mut k = 0usize;
        while k < columns {
            let mut i = usize::cast_from(UNIT_POS);
            while i < 128 {
                let t = token + i / 16;
                let col = k + i % 16;
                let mut v = F::new(0.0);
                if t < count && col < columns {
                    v = input[t * columns + col];
                }
                a[i] = v;
                i += 16;
            }
            i = usize::cast_from(UNIT_POS);
            while i < 256 {
                let r = row + i / 16;
                let col = k + i % 16;
                let mut v = F::new(0.0);
                if r < rows && col < columns {
                    v = weight[weight_first + r * columns + col];
                }
                b[i] = v;
                i += 16;
            }
            sync_cube();
            cmma::load(&am, &a.to_slice(), 16);
            cmma::load(&bm, &b.to_slice(), 16);
            cmma::execute::<F, F, f32, f32>(&am, &bm, &acc, &acc);
            sync_cube();
            k += 16;
        }
        cmma::store(
            &mut out.to_slice_mut(),
            &acc,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        out
    }
    #[cube(launch_unchecked)]
    pub fn dense<F: Float>(
        input: &Array<F>,
        weight: &Array<F>,
        output: &mut Array<f32>,
        fault: &mut Array<Atomic<u32>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        out_first: usize,
        out_stride: usize,
    ) {
        let values = product::<F>(input, weight, columns, rows, count, weight_first);
        let mut i = usize::cast_from(UNIT_POS);
        while i < 128 {
            let t = usize::cast_from(CUBE_POS_Y) * 8 + i / 16;
            let row = usize::cast_from(CUBE_POS_X) * 16 + i % 16;
            if t < count && row < rows {
                let v = values[i];
                if v.abs() <= f32::MAX {
                    output[out_first + t * out_stride + row] = v;
                } else {
                    output[out_first + t * out_stride + row] = 0.0;
                    fault[0].fetch_or(1);
                }
            }
            i += 16;
        }
    }
    #[cube(launch_unchecked)]
    pub fn expert(
        input: &Array<f16>,
        weight: &Array<f16>,
        output: &mut Array<f32>,
        map: &Array<u32>,
        status: &mut Array<u32>,
        fault: &mut Array<Atomic<u32>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        group: usize,
        #[comptime] down: bool,
    ) {
        let values = product::<f16>(input, weight, columns, rows, count, weight_first);
        let mut i = usize::cast_from(UNIT_POS);
        while i < 128 {
            let member = usize::cast_from(CUBE_POS_Y) * 8 + i / 16;
            let row = usize::cast_from(CUBE_POS_X) * 16 + i % 16;
            if member < count && row < rows {
                let route =
                    usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let mut base = member * 1280;
                    if down {
                        base = route * 2560;
                    }
                    let v = values[i];
                    let valid = v.abs() <= f32::MAX;
                    if valid {
                        output[base + row] = v;
                    } else {
                        output[base + row] = 0.0;
                        fault[0].fetch_or(128);
                    }
                    if down && row == 0 {
                        let mut good = 0u32;
                        if valid && status[route] == 1 {
                            good = 1;
                        }
                        // Same full-capacity plane as prompt activation and
                        // the rank-ordered reducer, including routes >20479.
                        status[comptime!(crate::strata_prompt::groups::ROUTES) + route] = good;
                    }
                } else {
                    fault[0].fetch_or(128);
                }
            }
            i += 16;
        }
    }
    pub(super) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
        vec![
            ("dense_fp16", definition_dense::<f16>()),
            ("dense_bf16", definition_dense::<bf16>()),
            ("expert_gu", definition_expert(false)),
            ("expert_down", definition_expert(true)),
        ]
    }
    fn settings() -> KernelSettings {
        KernelSettings::default().cube_dim(CubeDim::new_1d(16))
    }
    fn definition_dense<F: Float>() -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = F::as_type(&b.scope);
        let input = b.input_array(t);
        let weight = b.input_array(t);
        let output = b.output_array(f32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let args: [cubecl::ir::ManagedVariable; 6] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        dense::expand::<F>(
            &mut b.scope,
            input.into(),
            weight.into(),
            output.into(),
            fault.into(),
            args[0].clone().into(),
            args[1].clone().into(),
            args[2].clone().into(),
            args[3].clone().into(),
            args[4].clone().into(),
            args[5].clone().into(),
        );
        b.build(settings())
    }
    fn definition_expert(down: bool) -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = f16::as_type(&b.scope);
        let input = b.input_array(t);
        let weight = b.input_array(t);
        let output = b.output_array(f32::as_type(&b.scope));
        let map = b.input_array(u32::as_type(&b.scope));
        let status = b.output_array(u32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let args: [cubecl::ir::ManagedVariable; 5] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        expert::expand(
            &mut b.scope,
            input.into(),
            weight.into(),
            output.into(),
            map.into(),
            status.into(),
            fault.into(),
            args[0].clone().into(),
            args[1].clone().into(),
            args[2].clone().into(),
            args[3].clone().into(),
            args[4].clone().into(),
            down,
        );
        b.build(settings())
    }
    fn half_arg(handle: Handle) -> Result<ArrayArg<WgpuRuntime>> {
        if handle.size_in_used() == 0 || !handle.size_in_used().is_multiple_of(2) {
            return Err(Error::InvalidBufferLength);
        }
        let words = handle.size_in_used() as usize / 2;
        // SAFETY: typed half elements, not the u32 elements of raw_arg.
        Ok(unsafe { ArrayArg::from_raw_parts(handle, words) })
    }
    fn compatible(client: &ComputeClient<WgpuRuntime>, bfloat: bool) -> Result<()> {
        let kind = ElemType::Float(if bfloat {
            FloatKind::BF16
        } else {
            FloatKind::F16
        });
        if !client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: kind.into(),
                b_type: kind.into(),
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 8,
                n: 16,
                k: 16,
            })
        {
            return Err(Error::InvalidFormat(
                "native Intel8x16x16 matrix capability",
            ));
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_dense(
        client: &ComputeClient<WgpuRuntime>,
        h: [Handle; 4],
        count: usize,
        columns: usize,
        rows: usize,
        weight_first: usize,
        out_first: usize,
        out_stride: usize,
        bfloat: bool,
    ) -> Result<()> {
        compatible(client, bfloat)?;
        let [input, weight, output, fault] = h;
        let ia = half_arg(input)?;
        let wa = half_arg(weight)?;
        let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
        let fa = crate::resident_util::raw_arg(fault, 1)?;
        // SAFETY: caller checked original owner intervals; padded tile operands
        // are synthesized in workgroup memory and stores mask inactive rows.
        unsafe {
            if bfloat {
                dense::launch_unchecked::<bf16, WgpuRuntime>(
                    client,
                    CubeCount::Static(rows.div_ceil(16) as u32, count.div_ceil(8) as u32, 1),
                    CubeDim::new_1d(16),
                    ia,
                    wa,
                    oa,
                    fa,
                    columns,
                    rows,
                    count,
                    weight_first,
                    out_first,
                    out_stride,
                );
            } else {
                dense::launch_unchecked::<f16, WgpuRuntime>(
                    client,
                    CubeCount::Static(rows.div_ceil(16) as u32, count.div_ceil(8) as u32, 1),
                    CubeDim::new_1d(16),
                    ia,
                    wa,
                    oa,
                    fa,
                    columns,
                    rows,
                    count,
                    weight_first,
                    out_first,
                    out_stride,
                );
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_expert(
        client: &ComputeClient<WgpuRuntime>,
        h: [Handle; 6],
        count: usize,
        columns: usize,
        rows: usize,
        weight_first: usize,
        group: usize,
        down: bool,
    ) -> Result<()> {
        compatible(client, false)?;
        let [input, weight, output, map, status, fault] = h;
        let ia = half_arg(input)?;
        let wa = half_arg(weight)?;
        let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
        let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
        let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
        let fa = crate::resident_util::raw_arg(fault, 1)?;
        // SAFETY: the actual prompt grouping/row map and all physical packed
        // images were checked by the original-source expert caller.
        unsafe {
            expert::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(rows.div_ceil(16) as u32, count.div_ceil(8) as u32, 1),
                CubeDim::new_1d(16),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                down,
            );
        }
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<()> {
    if !crate::strata_prompt::active()
        || count == 0
        || count > crate::strata_prompt::groups::CAPACITY * 4
        || columns == 0
        || columns > 16384
        || !columns.is_multiple_of(16)
        || rows == 0
        || rows > 16384
        || out_stride < rows
        || [weight_first, out_first, out_stride]
            .iter()
            .any(|&n| u32::try_from(n).is_err())
        || h.iter().any(|v| v.stream != h[0].stream)
        || h[3].size_in_used() != 4
        || (count * columns * 2) as u64 > h[0].size_in_used()
        || weight_first
            .checked_add(rows * columns)
            .is_none_or(|n| n as u64 * 2 > h[1].size_in_used())
        || (count - 1)
            .checked_mul(out_stride)
            .and_then(|n| n.checked_add(rows))
            .and_then(|n| out_first.checked_add(n))
            .is_none_or(|n| n as u64 * 4 > h[2].size_in_used())
    {
        return Err(Error::InvalidBufferLength);
    }
    #[cfg(feature = "strata-intel-core")]
    {
        native::run_dense(
            client,
            h,
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
            bfloat,
        )
    }
    #[cfg(not(feature = "strata-intel-core"))]
    {
        let _ = (client, bfloat);
        Err(Error::InvalidBufferLength)
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> Result<()> {
    if !crate::strata_prompt::active()
        || count == 0
        || count > crate::strata_prompt::groups::ROUTE_CAPACITY
        || group >= 512
        || u32::try_from(weight_first).is_err()
        || columns != if down { 640 } else { 2560 }
        || rows != if down { 2560 } else { 1280 }
        || h.iter().any(|v| v.stream != h[0].stream)
        || h[5].size_in_used() != 4
        || h[3].size_in_used() != (crate::strata_prompt::groups::MAP_WORDS * 4) as u64
        || h[4].size_in_used() != (crate::strata_prompt::groups::STATUS_WORDS * 4) as u64
        || (count * columns * 2) as u64 > h[0].size_in_used()
        || weight_first
            .checked_add(rows * columns)
            .is_none_or(|n| n as u64 * 2 > h[1].size_in_used())
        || (if down {
            crate::strata_prompt::groups::ROUTES * 2560
        } else {
            count * 1280
        } * 4) as u64
            > h[2].size_in_used()
    {
        return Err(Error::InvalidBufferLength);
    }
    #[cfg(feature = "strata-intel-core")]
    {
        native::run_expert(client, h, count, columns, rows, weight_first, group, down)
    }
    #[cfg(not(feature = "strata-intel-core"))]
    {
        let _ = client;
        Err(Error::InvalidBufferLength)
    }
}

/// Host-only compilation of the real launched Cube expansions. No runtime,
/// GPU device, request data or model allocation is constructed by this check.
pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "strata-intel-core")]
    {
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{
            Compiler, WgpuCompilationOptions,
            ir::{ElemType, StorageType, UIntKind},
        };
        use std::io::Write;
        // Device extensions are independent: the B70 need not select either
        // the all-disabled or all-enabled combination of these layout flags.
        for layout_mask in 0u8..8 {
            let standard_layout = layout_mask & 1 != 0;
            for (name, definition) in native::definitions() {
                let mut compiler = AutoCompiler::SpirV(Default::default());
                let mut options = WgpuCompilationOptions {
                    supports_vulkan: true,
                    ..Default::default()
                };
                options.vulkan.max_spirv_version = (1, 6);
                options.vulkan.supports_uniform_standard_layout = standard_layout;
                options.vulkan.supports_uniform_unsized_array = layout_mask & 2 != 0;
                options.vulkan.supports_explicit_smem = layout_mask & 4 != 0;
                let compiled = Compiler::compile(
                    &mut compiler,
                    definition,
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )?;
                let AutoRepresentation::SpirV(kernel) = compiled else {
                    return Err("matrix compiler representation".into());
                };
                let canonical =
                    wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
                let canonical = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&canonical)?
                    .unwrap_or(canonical);
                let words = &canonical;
                if wgpu::hal::vulkan::kuno_cooperative_subgroup(words) != Some(16) {
                    return Err("matrix pipeline subgroup geometry".into());
                }
                let mut at = 5;
                let mut matrix = false;
                let mut multiply = false;
                let mut fault = false;
                while at < words.len() {
                    let size = (words[at] >> 16) as usize;
                    if size == 0 || at.checked_add(size).is_none_or(|end| end > words.len()) {
                        return Err("matrix instruction extent".into());
                    }
                    let op = words[at] & 65535;
                    matrix |= op == 4456;
                    multiply |= op == 4459;
                    fault |= op == 241;
                    at += size;
                }
                if !matrix || !multiply || !fault {
                    return Err("matrix multiply/fault instructions absent".into());
                }
                let mut validator = std::process::Command::new("spirv-val");
                validator.args(["--target-env", "vulkan1.3"]);
                if standard_layout {
                    // The compiler emits std430 INFO arrays only when the
                    // actual device enabled uniformBufferStandardLayout.
                    validator.arg("--uniform-buffer-standard-layout");
                }
                let mut command = validator
                    .arg("-")
                    .stdin(std::process::Stdio::piped())
                    .spawn()?;
                command
                    .stdin
                    .take()
                    .ok_or("matrix validator stdin")?
                    .write_all(
                        &canonical
                            .iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                if !command.wait()?.success() {
                    return Err("native matrix SPIRV validation failed".into());
                }
                println!(
                    "strata_matrix_shader_accepted name={name} subgroup=16 layout_mask={layout_mask}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use wgpu::hal::vulkan::kuno_native_workgroup_layout as layout;

    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn actual_fp16_cooperative_module_requests_preservation_and_rejects_conflicting_control() {
        use cubecl::ir::{ElemType, StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
        let definition = super::native::definitions()
            .into_iter()
            .find(|(name, _)| *name == "dense_fp16")
            .unwrap()
            .1;
        let mut options = WgpuCompilationOptions {
            supports_vulkan: true,
            ..Default::default()
        };
        options.vulkan.max_spirv_version = (1, 6);
        let AutoRepresentation::SpirV(kernel) = Compiler::compile(
            &mut AutoCompiler::SpirV(Default::default()),
            definition,
            &options,
            ExecutionMode::Unchecked,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        )
        .unwrap() else {
            panic!("native SPIRV required")
        };
        let precise = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&kernel.assembled_module)
            .unwrap()
            .expect("ordinary FP16 cooperative module");
        assert_eq!(
            wgpu::hal::vulkan::kuno_cooperative_subgroup(&precise),
            Some(16)
        );
        assert_eq!(
            wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&precise)
                .unwrap()
                .unwrap(),
            precise
        );
        let mut at = 5;
        let mut entry = None;
        while at < precise.len() {
            let n = (precise[at] >> 16) as usize;
            if precise[at] & 65535 == 16 && n == 4 && precise[at + 2..at + 4] == [4459, 16] {
                assert!(entry.replace(precise[at + 1]).is_none());
            }
            at += n;
        }
        let entry = entry.expect("actual DenormPreserve16 mode");
        let mut conflicting = precise;
        conflicting.extend_from_slice(&[(4 << 16) | 16, entry, 4460, 16]);
        assert!(wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&conflicting).is_err());
        assert!(wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&[0; 5]).is_err());
        assert!(
            wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&[
                0x07230203, 0x00010600, 0, 1, 0, 0
            ])
            .is_err()
        );
    }

    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn compiled_expert_down_writes_the_same_full_prompt_status_plane_the_reducer_reads() {
        use cubecl::ir::{ElemType, StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
        use std::collections::BTreeMap;

        // Inspect the real cooperative kernel, not the legacy scalar WGSL
        // diagnostic. Its first status read is activation[route]; its sole
        // status store must be down[route], in the consumer's second plane.
        let reducer = crate::strata_prompt::groups::shader_prefix();
        assert!(reducer.contains(&format!(
            "const PROMPT_ROUTES:u32={}u;",
            crate::strata_prompt::groups::ROUTES
        )));
        assert!(
            include_str!("strata_prompt_reduce.wgsl").contains("status[PROMPT_ROUTES+route]==1u")
        );

        for mask in 0..8u8 {
            let mut options = WgpuCompilationOptions {
                supports_vulkan: true,
                ..Default::default()
            };
            options.vulkan.max_spirv_version = (1, 6);
            options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
            options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
            options.vulkan.supports_explicit_smem = mask & 4 != 0;
            let definition = super::native::definitions()
                .into_iter()
                .find(|(name, _)| *name == "expert_down")
                .unwrap()
                .1;
            let compiled = Compiler::compile(
                &mut AutoCompiler::SpirV(Default::default()),
                definition,
                &options,
                ExecutionMode::Unchecked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap();
            let AutoRepresentation::SpirV(kernel) = compiled else {
                panic!("actual native matrix compiler required");
            };
            let words = layout(&kernel.assembled_module).unwrap();
            let mut at = 5;
            let mut bindings = BTreeMap::new();
            let mut constants = BTreeMap::new();
            let mut adds = BTreeMap::new();
            let mut pointers = BTreeMap::new();
            let mut loads = Vec::new();
            let mut stores = Vec::new();
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                assert!(n > 0 && at + n <= words.len());
                let operands = &words[at + 1..at + n];
                match words[at] & 65535 {
                    71 if operands.len() == 3 && operands[1] == 33 => {
                        bindings.insert(operands[0], operands[2]); // Binding.
                    }
                    43 if operands.len() == 3 => {
                        constants.insert(operands[1], operands[2]); // OpConstant.
                    }
                    128 if operands.len() == 4 => {
                        adds.insert(operands[1], [operands[2], operands[3]]); // IAdd.
                    }
                    65 if operands.len() == 5 => {
                        pointers.insert(operands[1], (operands[2], operands[4]));
                    }
                    61 if operands.len() >= 3 => loads.push(operands[2]),
                    62 if operands.len() >= 2 => stores.push(operands[0]),
                    _ => {}
                }
                at += n;
            }
            let status = *bindings.iter().find(|(_, b)| **b == 4).unwrap().0;
            let indices = |accesses: &[u32]| {
                accesses
                    .iter()
                    .filter_map(|p| pointers.get(p))
                    .filter_map(|&(base, index)| (base == status).then_some(index))
                    .collect::<Vec<_>>()
            };
            let reads = indices(&loads);
            let writes = indices(&stores);
            assert_eq!(reads.len(), 1, "mask={mask}");
            assert_eq!(writes.len(), 1, "mask={mask}");
            let operands = adds[&writes[0]];
            let offset = if operands[0] == reads[0] {
                constants[&operands[1]]
            } else {
                assert_eq!(operands[1], reads[0]);
                constants[&operands[0]]
            } as usize;
            assert_eq!(offset, crate::strata_prompt::groups::ROUTES, "mask={mask}");
            for route in [0, 9, 20479, 20480, crate::strata_prompt::groups::ROUTES - 1] {
                assert!(offset + route >= crate::strata_prompt::groups::ROUTES);
                assert!(offset + route < crate::strata_prompt::groups::STATUS_WORDS);
            }
        }
    }

    // Public, content-free type graphs exercise the unsafe native boundary:
    // changing a host/explicit byte ABI would silently address the wrong data.
    fn module(instructions: &[&[u32]]) -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010600, 0, 32, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    #[test]
    fn implicit_workgroup_stride_is_removed_but_storage_stride_survives() {
        let wg_stride = [4 << 16 | 71, 2, 6, 4];
        let host_stride = [4 << 16 | 71, 5, 6, 16];
        let declarations: &[&[u32]] = &[
            &[4 << 16 | 28, 2, 1, 10],
            &[4 << 16 | 32, 3, 4, 2],
            &[4 << 16 | 59, 3, 4, 4],
            &[4 << 16 | 28, 5, 1, 10],
            &[4 << 16 | 32, 6, 12, 5],
            &[4 << 16 | 59, 6, 7, 12],
        ];
        let mut original = vec![wg_stride.as_slice(), host_stride.as_slice()];
        original.extend_from_slice(declarations);
        let mut expected = vec![host_stride.as_slice()];
        expected.extend_from_slice(declarations);
        let canonical = layout(&module(&original)).unwrap();
        assert_eq!(canonical, module(&expected));
        assert_eq!(layout(&canonical).unwrap(), canonical);
    }

    #[test]
    fn explicit_workgroup_block_preserves_member_byte_layout() {
        let original = module(&[
            &[4 << 16 | 71, 2, 6, 16],
            &[3 << 16 | 71, 3, 2],
            &[5 << 16 | 72, 3, 0, 35, 0],
            &[4 << 16 | 28, 2, 1, 10],
            &[3 << 16 | 30, 3, 2],
            &[4 << 16 | 32, 4, 4, 3],
            &[4 << 16 | 59, 4, 5, 4],
        ]);
        assert_eq!(layout(&original).unwrap(), original);
    }

    #[test]
    fn shared_workgroup_and_interface_array_is_rejected_without_rewriting() {
        let original = module(&[
            &[4 << 16 | 71, 2, 6, 4],
            &[4 << 16 | 28, 2, 1, 10],
            &[4 << 16 | 32, 3, 4, 2],
            &[4 << 16 | 59, 3, 4, 4],
            &[4 << 16 | 32, 5, 12, 2],
            &[4 << 16 | 59, 5, 6, 12],
        ]);
        assert_eq!(
            layout(&original),
            Err("workgroup/interface layout alias requires separate types")
        );
    }

    #[test]
    fn malformed_native_instructions_are_rejected_before_vulkan() {
        assert!(layout(&[]).is_err());
        assert!(layout(&module(&[&[0]])).is_err());
        assert!(layout(&module(&[&[4 << 16 | 71, 2, 6]])).is_err());
    }

    #[test]
    fn cooperative_subgroup_override_requires_the_whole_declared_geometry() {
        use wgpu::hal::vulkan::kuno_cooperative_subgroup as subgroup;
        let matrix = [7 << 16 | 4456, 1, 2, 3, 4, 5, 6];
        let size16 = [6 << 16 | 16, 7, 17, 16, 1, 1];
        let size32 = [6 << 16 | 16, 7, 17, 32, 1, 1];
        assert_eq!(subgroup(&module(&[&matrix, &size16])), Some(16));
        assert_eq!(subgroup(&module(&[&matrix, &size32])), None);
        assert_eq!(subgroup(&module(&[&matrix])), None);
        assert_eq!(subgroup(&module(&[&size16])), None);
        assert_eq!(subgroup(&module(&[&matrix, &size16, &[0]])), None);
    }

    #[test]
    fn original_shader_requirement_metadata_preserves_default_fixed32_and_cooperative16() {
        use wgpu::hal::vulkan::{KunoSubgroupRequirement, ShaderModule};
        let requirement = KunoSubgroupRequirement::default();
        assert_eq!(requirement.required(), None);
        requirement.require32().unwrap();
        requirement.require32().unwrap();
        assert_eq!(requirement.required(), Some(32));
        assert_eq!(KunoSubgroupRequirement::default().required(), None);
        // Metadata-only fixtures: no device/module creation or unsafe attach.
        // These selectors are the actual methods used by both pipeline paths;
        // a null raw handle grants no ownership or physical execution proof.
        let raw = ShaderModule::Raw(ash::vk::ShaderModule::null(), requirement.required());
        assert_eq!(raw.kuno_required_subgroup_size(), Some(32));
        let ordinary = ShaderModule::Raw(ash::vk::ShaderModule::null(), Default::default());
        assert_eq!(ordinary.kuno_required_subgroup_size(), None);
        let late = KunoSubgroupRequirement::default();
        late.require32().unwrap();
        let intermediate = ShaderModule::Intermediate {
            naga_shader: Default::default(),
            runtime_checks: Default::default(),
            required_subgroup: late.required(),
        };
        assert_eq!(intermediate.kuno_required_subgroup_size(), Some(32));
        let matrix = ShaderModule::RawSubgroup16(ash::vk::ShaderModule::null());
        assert_eq!(matrix.kuno_required_subgroup_size(), Some(16));
        // The actual ordinary AND retained pipeline builders consume this
        // selector; fixed width must clear varying flags and own stable pNext.
        use wgpu::hal::vulkan::kuno_subgroup_stage_options as options;
        for module in [&ordinary, &raw, &intermediate, &matrix] {
            for supported in [false, true] {
                let width = module.kuno_required_subgroup_size();
                let (flags, info) = options(width, supported);
                assert_eq!(
                    flags.contains(
                        ash::vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE
                    ),
                    width.is_none() && supported
                );
                assert_eq!(info.as_ref().map(|v| v.required_subgroup_size), width);
                if let Some(info) = info {
                    assert!(info.p_next.is_null());
                    assert_eq!(info.s_type,ash::vk::StructureType::PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
                }
            }
        }
        requirement.require32().unwrap();
        assert_eq!(matrix.kuno_required_subgroup_size(), Some(16));
    }
}
