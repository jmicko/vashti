//! Opt-in full-tile native dense CMMA; original packed operands and F32 owner.
//! No expert routing, host allocation, request storage, copy or dtype conversion.
use cubecl::{
    ir::{ElemType, FloatKind, features::MmaConfig},
    prelude::*,
    server::Handle,
    wgpu::WgpuRuntime,
};
use half::{bf16, f16};
use kuno_text::{Error, Result};

pub(super) const SHARED_BYTES: usize = 40960;
const BLOCK: usize = 128;
const THREADS: u32 = 256;

#[cube]
#[allow(clippy::too_many_arguments)]
fn stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    columns: usize,
    weight_first: usize,
    token_base: usize,
    row_base: usize,
    k: usize,
    buffer: usize,
) {
    // Full128x128x32 stages only. Two genuine vec4s per job preserve original
    // FP16/BF16 bits; every one of the1024 vectors per operand is written once.
    #[unroll]
    for round in 0..2usize {
        let job = usize::cast_from(UNIT_POS) + round * 256;
        let offset = k + (job % 4) * 8;
        let row = job / 4;
        let source_a = ((token_base + row) * columns + offset) / 4;
        let source_b = (weight_first + (row_base + row) * columns + offset) / 4;
        a[buffer * 1024 + job * 2] = input[source_a];
        a[buffer * 1024 + job * 2 + 1] = input[source_a + 1];
        b[buffer * 1024 + job * 2] = weight[source_b];
        b[buffer * 1024 + job * 2 + 1] = weight[source_b + 1];
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn publish(
    acc: &cmma::Matrix<f32>,
    scratch: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    token_base: usize,
    row_base: usize,
    out_first: usize,
    out_stride: usize,
    #[comptime] tile_m: usize,
    #[comptime] tile_n: usize,
) {
    let subgroup = usize::cast_from(PLANE_POS);
    cmma::store(
        &mut scratch.slice_mut(subgroup * 128, subgroup * 128 + 128),
        acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    // Every cooperative store completes before any lane consumes its fragment.
    sync_cube();
    #[unroll]
    for round in 0..8usize {
        let i = usize::cast_from(UNIT_POS_PLANE) + round * 16;
        let token = token_base + (subgroup / 4) * 32 + comptime!(tile_m * 8) + i / 16;
        let row = row_base + (subgroup % 4) * 32 + comptime!(tile_n * 16) + i % 16;
        let value = scratch[subgroup * 128 + i];
        if value.abs() <= f32::MAX {
            output[out_first + token * out_stride + row] = value;
        } else {
            output[out_first + token * out_stride + row] = 0.0;
            fault[0].fetch_or(1);
        }
    }
    // No subgroup may overwrite its shared fragment before all consumers finish.
    // Both barriers are uniform and execute for every publication phase.
    sync_cube();
}

#[cube(launch_unchecked)]
fn dense_wide<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    // Two typed stages per operand:2*128*32*2 bytes each. Scratch remains a
    // separate real8KiB allocation; admission never assumes lifetime aliasing.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut scratch = SharedMemory::<f32>::new(2048usize);
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a2 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a3 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b0 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b1 = unsafe {
        cmma::Matrix::<F>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let c0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c4 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c5 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c6 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c7 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let subgroup = usize::cast_from(PLANE_POS);
    let token_base = usize::cast_from(CUBE_POS_Y) * 128;
    let row_base = usize::cast_from(CUBE_POS_X) * 128;
    stage::<F>(
        input,
        weight,
        &mut a,
        &mut b,
        columns,
        weight_first,
        token_base,
        row_base,
        0usize,
        0usize,
    );
    sync_cube();
    let mut k = 0usize;
    let mut current = 0usize;
    while k < columns {
        let next = 1usize - current;
        // Same accepted immediate global-load→SLM-store software-prefetch ordering.
        // This expresses overlap opportunity, not guaranteed asynchronous execution.
        if k + 32 < columns {
            stage::<F>(
                input,
                weight,
                &mut a,
                &mut b,
                columns,
                weight_first,
                token_base,
                row_base,
                k + 32,
                next,
            );
        }
        #[unroll]
        for step in 0..2usize {
            let af = current * 1024 + (subgroup / 4) * 256 + step * 4;
            let bf = current * 1024 + (subgroup % 4) * 256 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(af + 64, a.len()), 32);
            cmma::load(&a2, &a.to_slice().slice(af + 128, a.len()), 32);
            cmma::load(&a3, &a.to_slice().slice(af + 192, a.len()), 32);
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 32);
            cmma::load(&b1, &b.to_slice().slice(bf + 128, b.len()), 32);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
            cmma::execute::<F, F, f32, f32>(&a2, &b0, &c4, &c4);
            cmma::execute::<F, F, f32, f32>(&a2, &b1, &c5, &c5);
            cmma::execute::<F, F, f32, f32>(&a3, &b0, &c6, &c6);
            cmma::execute::<F, F, f32, f32>(&a3, &b1, &c7, &c7);
        }
        // Current reads end and all next-stage writes become visible before switch.
        sync_cube();
        current = next;
        k += 32;
    }
    publish(
        &c0,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        0usize,
        0usize,
    );
    publish(
        &c1,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        0usize,
        1usize,
    );
    publish(
        &c2,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        1usize,
        0usize,
    );
    publish(
        &c3,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        1usize,
        1usize,
    );
    publish(
        &c4,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        2usize,
        0usize,
    );
    publish(
        &c5,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        2usize,
        1usize,
    );
    publish(
        &c6,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        3usize,
        0usize,
    );
    publish(
        &c7,
        &mut scratch,
        output,
        fault,
        token_base,
        row_base,
        out_first,
        out_stride,
        3usize,
        1usize,
    );
}

#[allow(clippy::too_many_arguments)]
fn selected(
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    shared: usize,
    max_units: u32,
    max_dim: (u32, u32, u32),
    plane_min: u32,
    plane_max: u32,
) -> bool {
    count >= 512
        && count.is_multiple_of(BLOCK)
        && rows != 0
        && rows.is_multiple_of(BLOCK)
        && columns != 0
        && columns.is_multiple_of(32)
        && weight_first.is_multiple_of(4)
        && shared >= SHARED_BYTES
        && max_units >= THREADS
        && max_dim.0 >= THREADS
        && max_dim.1 >= 1
        && max_dim.2 >= 1
        && plane_min <= 16
        && plane_max >= 16
}
fn checked_bytes(elements: usize, bytes: usize) -> Result<u64> {
    u64::try_from(
        elements
            .checked_mul(bytes)
            .ok_or(Error::ArithmeticOverflow)?,
    )
    .map_err(|_| Error::ArithmeticOverflow)
}
fn vector_views(h: &[Handle; 4]) -> bool {
    h[..2].iter().all(|v| {
        v.size_in_used() != 0
            && v.size_in_used().is_multiple_of(8)
            && v.offset_start.unwrap_or(0).is_multiple_of(8)
    })
}
fn check_bounds(
    h: &[Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) -> Result<()> {
    let input = count
        .checked_mul(columns)
        .ok_or(Error::ArithmeticOverflow)?;
    let weight = rows
        .checked_mul(columns)
        .and_then(|n| weight_first.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    let output = count
        .checked_sub(1)
        .and_then(|n| n.checked_mul(out_stride))
        .and_then(|n| n.checked_add(rows))
        .and_then(|n| out_first.checked_add(n))
        .ok_or(Error::ArithmeticOverflow)?;
    if out_stride < rows
        || [
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
            input,
            weight,
            output,
        ]
        .iter()
        .any(|&v| u32::try_from(v).is_err())
        || h.iter().any(|v| v.stream != h[0].stream)
        || h[3].size_in_used() != 4
        || checked_bytes(input, 2)? > h[0].size_in_used()
        || checked_bytes(weight, 2)? > h[1].size_in_used()
        || checked_bytes(output, 4)? > h[2].size_in_used()
        || !vector_views(h)
        || !h[2].size_in_used().is_multiple_of(4)
        || !h[2].offset_start.unwrap_or(0).is_multiple_of(4)
        || !h[3].offset_start.unwrap_or(0).is_multiple_of(4)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    let hw = &client.properties().hardware;
    if !selected(
        count,
        columns,
        rows,
        weight_first,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
    ) {
        return Ok(false);
    }
    // These may still be valid scalar half views for the original fallback.
    // Never force a vector pointer or fail an otherwise admitted scalar view.
    if !vector_views(&h) {
        return Ok(false);
    }
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
        return Ok(false);
    }
    check_bounds(
        &h,
        count,
        columns,
        rows,
        weight_first,
        out_first,
        out_stride,
    )?;
    let gx = rows / BLOCK;
    let gy = count / BLOCK;
    if gx > hw.max_cube_count.0 as usize
        || gy > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("native wide dense grid"));
    }
    let [input, weight, output, fault] = h;
    // SAFETY: true aligned vec4 half owners; no scalar-length/vector-pointer forgery.
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 8) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 8) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    // SAFETY: complete original logical owners, all grid/offset/index limits,
    // actual sixteen-subgroup capabilities and full-tile geometry checked above.
    unsafe {
        if bfloat {
            dense_wide::launch_unchecked::<bf16, WgpuRuntime>(
                client,
                CubeCount::Static(gx as u32, gy as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                fa,
                columns,
                weight_first,
                out_first,
                out_stride,
            );
        } else {
            dense_wide::launch_unchecked::<f16, WgpuRuntime>(
                client,
                CubeCount::Static(gx as u32, gy as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                fa,
                columns,
                weight_first,
                out_first,
                out_stride,
            );
        }
    }
    Ok(true)
}

fn definition<F: Float>() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<F, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let args: [cubecl::ir::ManagedVariable; 4] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    dense_wide::expand::<F>(
        &mut b.scope,
        input.into(),
        weight.into(),
        output.into(),
        fault.into(),
        args[0].clone().into(),
        args[1].clone().into(),
        args[2].clone().into(),
        args[3].clone().into(),
    );
    let errors = b.scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}
pub(super) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    vec![
        ("dense_wide_native_fp16", definition::<f16>()),
        ("dense_wide_native_bf16", definition::<bf16>()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_tile_selection_requires_real_caps_and_preserves_tails() {
        let yes = |c, k, n, w, s, u, d, pmin, pmax| selected(c, k, n, w, s, u, d, pmin, pmax);
        assert!(yes(512, 32, 128, 0, 40960, 256, (256, 1, 1), 16, 32));
        for (c, k, n, w, s, u, d, pmin, pmax) in [
            (511, 32, 128, 0, 40960, 256, (256, 1, 1), 16, 32),
            (513, 32, 128, 0, 40960, 256, (256, 1, 1), 16, 32),
            (512, 16, 128, 0, 40960, 256, (256, 1, 1), 16, 32),
            (512, 32, 320, 0, 40960, 256, (256, 1, 1), 16, 32),
            (512, 32, 128, 1, 40960, 256, (256, 1, 1), 16, 32),
            (512, 32, 128, 0, 40959, 256, (256, 1, 1), 16, 32),
            (512, 32, 128, 0, 40960, 255, (256, 1, 1), 16, 32),
            (512, 32, 128, 0, 40960, 256, (255, 1, 1), 16, 32),
            (512, 32, 128, 0, 40960, 256, (256, 0, 1), 16, 32),
            (512, 32, 128, 0, 40960, 256, (256, 1, 0), 16, 32),
            (512, 32, 128, 0, 40960, 256, (256, 1, 1), 32, 32),
            (512, 32, 128, 0, 40960, 256, (256, 1, 1), 8, 8),
        ] {
            assert!(!yes(c, k, n, w, s, u, d, pmin, pmax));
        }
    }
    #[test]
    fn full_vec4_stages_and_scratch_publications_cover_each_original_cell_once() {
        let mut a = vec![0; 1024];
        for lane in 0..256 {
            for round in 0..2 {
                let job = lane + round * 256;
                let row = job / 4;
                let col = job % 4 * 8;
                assert!(row < 128 && col + 8 <= 32);
                for v in 0..2 {
                    a[job * 2 + v] += 1;
                    assert_eq!(job * 2 + v, (row * 32 + col) / 4 + v);
                }
            }
        }
        assert!(a.iter().all(|&n| n == 1));
        let mut out = vec![0; 128 * 128];
        for subgroup in 0..16 {
            for m in 0..4 {
                for n in 0..2 {
                    let mut scratch = vec![0; 128];
                    for lane in 0..16 {
                        for round in 0..8 {
                            let i = lane + round * 16;
                            scratch[i] += 1;
                            let token = subgroup / 4 * 32 + m * 8 + i / 16;
                            let row = subgroup % 4 * 32 + n * 16 + i % 16;
                            assert!(token < 128 && row < 128);
                            out[token * 128 + row] += 1;
                        }
                    }
                    assert!(scratch.iter().all(|&n| n == 1));
                }
            }
        }
        assert!(out.iter().all(|&n| n == 1));
        for columns in [32, 64, 96, 2560, 10240] {
            let mut reductions = vec![];
            for k in (0..columns).step_by(32) {
                for step in 0..2 {
                    reductions.push(k + step * 16);
                }
            }
            assert_eq!(reductions, (0..columns).step_by(16).collect::<Vec<_>>());
        }
    }
    #[test]
    fn original_logical_owners_offsets_stream_and_complete_output_bounds_remain_required() {
        let owner = |bytes| Handle::new(cubecl_common::stream_id::StreamId::current(), bytes);
        let h = [
            owner(512 * 32 * 2),
            owner(128 * 32 * 2),
            owner(512 * 128 * 4),
            owner(4),
        ];
        assert!(check_bounds(&h, 512, 32, 128, 0, 0, 128).is_ok());
        for index in 0..4 {
            let mut bad = h.clone();
            bad[index] = owner(h[index].size_in_used() - if index == 3 { 1 } else { 4 });
            assert!(check_bounds(&bad, 512, 32, 128, 0, 0, 128).is_err());
        }
        assert!(check_bounds(&h, 512, 32, 128, 0, 0, 127).is_err());
        assert!(check_bounds(&h, 512, 32, 128, 0, 1, 128).is_err());
        assert!(check_bounds(&h, 512, 32, 128, usize::MAX, 0, 128).is_err());
        assert!(check_bounds(&h, 512, 32, 128, 0, 0, usize::MAX).is_err());
        let mut different_stream = h.clone();
        different_stream[1] = Handle::new(
            cubecl_common::stream_id::StreamId {
                value: h[0].stream.value ^ 1,
            },
            h[1].size_in_used(),
        );
        assert!(check_bounds(&different_stream, 512, 32, 128, 0, 0, 128).is_err());
        let mut unaligned = h.clone();
        unaligned[0] = owner(h[0].size_in_used() + 4).offset_start(4);
        assert!(check_bounds(&unaligned, 512, 32, 128, 0, 0, 128).is_err());
        assert!(!vector_views(&unaligned));
        let mut scalar_padding = h.clone();
        scalar_padding[0] = owner(h[0].size_in_used() + 4);
        assert!(!vector_views(&scalar_padding));
        assert!(vector_views(&h));
        let padded = [
            owner(512 * 32 * 2 + 16).offset_start(8).offset_end(8),
            owner(128 * 32 * 2 + 24).offset_start(8).offset_end(8),
            owner((512 * 131 + 5) * 4),
            owner(12).offset_start(4).offset_end(4),
        ];
        assert!(check_bounds(&padded, 512, 32, 128, 4, 5, 131).is_ok());
    }
    #[test]
    fn actual_full_tile_module_keeps_16_subgroups_ieee_stages_publication_and_faults() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        for mask in 0..8 {
            for (name, definition) in [
                ("fp16", definition::<f16>()),
                ("bf16", definition::<bf16>()),
            ] {
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
                    definition,
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap() else {
                    panic!("actual SPIR-V required")
                };
                assert_eq!(kernel.shared_size, SHARED_BYTES);
                let words =
                    wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)
                        .unwrap();
                let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
                    .unwrap()
                    .unwrap_or(words);
                assert_eq!(
                    wgpu::hal::vulkan::kuno_cooperative_subgroup(&words),
                    Some(16)
                );
                let mut at = 5;
                let mut local = None;
                let mut builtins = Vec::new();
                let mut products = 0;
                let mut loads = 0;
                let mut stores = 0;
                let mut barriers = 0;
                let mut fault = false;
                while at < words.len() {
                    let n = (words[at] >> 16) as usize;
                    let op = words[at] & 65535;
                    let args = &words[at + 1..at + n];
                    if op == 16 && args.get(1) == Some(&17) {
                        local = Some(args[2..5].to_vec());
                    }
                    if op == 71 && args.get(1) == Some(&11) {
                        builtins.push(args[2]);
                    }
                    products += usize::from(op == 4459);
                    loads += usize::from(op == 4457);
                    stores += usize::from(op == 4458);
                    barriers += usize::from(op == 224);
                    fault |= op == 241;
                    assert!(
                        op != 71 || args.get(1) != Some(&40),
                        "relaxed FPFastMathMode"
                    );
                    at += n;
                }
                assert_eq!(local, Some(vec![256, 1, 1]));
                assert!(builtins.contains(&40) && builtins.contains(&41));
                assert_eq!(products, 16);
                assert_eq!(loads, 12);
                assert_eq!(stores, 8);
                assert_eq!(barriers, 18);
                assert!(fault);
                let mut command = std::process::Command::new("spirv-val");
                command.args(["--target-env", "vulkan1.3"]);
                if mask & 1 != 0 {
                    command.arg("--uniform-buffer-standard-layout");
                }
                let mut child = command
                    .arg("-")
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .unwrap();
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(
                        &words
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                assert!(child.wait().unwrap().success(), "{name}/{mask}");
                println!(
                    "compiled_public_native_wide name={name} mask={mask} shared_bytes={} products={products} loads={loads} stores={stores} barriers={barriers}",
                    kernel.shared_size
                );
            }
        }
    }
}
