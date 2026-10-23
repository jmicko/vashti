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

const BLOCK: usize = 128;
#[cfg(not(feature = "strata-native-wide32"))]
const THREADS: u32 = 256;
#[cfg(feature = "strata-native-wide32")]
const THREADS: u32 = 512;
const SUBGROUPS: usize = THREADS as usize / 16;
const SUBGROUP_ROWS: usize = BLOCK / (SUBGROUPS / 4);
const M_FRAGMENTS: usize = SUBGROUP_ROWS / 8;
const STAGE_ROUNDS: usize = 512 / THREADS as usize;
const STAGE_THREADS: usize = THREADS as usize;
pub(super) const SHARED_BYTES: usize = 32768 + SUBGROUPS * 128 * 4;

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
    #[comptime] gr_down: bool,
) {
    // Full128x128x32 stages only. Two genuine vec4s per job preserve original
    // FP16/BF16 bits; every one of the1024 vectors per operand is written once.
    #[unroll]
    for round in 0..comptime!(STAGE_ROUNDS) {
        let job = usize::cast_from(UNIT_POS) + round * comptime!(STAGE_THREADS);
        let offset = k + (job % 4) * 8;
        let row = job / 4;
        let source_a = ((token_base + row) * columns + offset) / 4;
        let source_b = (weight_first + (row_base + row) * columns + offset) / 4;
        a[buffer * 1024 + job * 2] = input[source_a];
        a[buffer * 1024 + job * 2 + 1] = input[source_a + 1];
        if comptime!(gr_down) {
            // The sole extra shape is the authentic320-row GR down image.
            // Do not read the missing last64 rows, even from a larger owner.
            let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
            let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
            if row_base + row < 320 {
                lo = weight[source_b];
                hi = weight[source_b + 1];
            }
            b[buffer * 1024 + job * 2] = lo;
            b[buffer * 1024 + job * 2 + 1] = hi;
        } else {
            b[buffer * 1024 + job * 2] = weight[source_b];
            b[buffer * 1024 + job * 2 + 1] = weight[source_b + 1];
        }
    }
}

#[cube]
fn publication_barrier() {
    if comptime!(cfg!(feature = "strata-native-subgroup-publish")) {
        // Both real Vulkan compiler paths add AcquireRelease|WorkgroupMemory.
        // Only this subgroup owns and consumes its128-F32 publication slot.
        sync_plane();
    } else {
        sync_cube();
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
    #[comptime] gr_down: bool,
) {
    let subgroup = usize::cast_from(PLANE_POS);
    cmma::store(
        &mut scratch.slice_mut(subgroup * 128, subgroup * 128 + 128),
        acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    // Every cooperative store completes before any lane consumes its fragment.
    publication_barrier();
    #[unroll]
    for round in 0..8usize {
        let i = usize::cast_from(UNIT_POS_PLANE) + round * 16;
        let token =
            token_base + (subgroup / 4) * comptime!(SUBGROUP_ROWS) + comptime!(tile_m * 8) + i / 16;
        let row = row_base + (subgroup % 4) * 32 + comptime!(tile_n * 16) + i % 16;
        // Padded feature cells never publish or set a fault. The original
        // finite/fault writer is unchanged for every authentic logical cell.
        if !comptime!(gr_down) || row < 320 {
            let value = scratch[subgroup * 128 + i];
            if value.abs() <= f32::MAX {
                output[out_first + token * out_stride + row] = value;
            } else {
                output[out_first + token * out_stride + row] = 0.0;
                fault[0].fetch_or(1);
            }
        }
    }
    // No subgroup may overwrite its shared fragment before all consumers finish.
    // Both barriers are uniform within the owning subgroup and retain shared
    // memory acquire/release ordering for every publication phase.
    publication_barrier();
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
    #[comptime] gr_down: bool,
) {
    // Two typed stages per operand:2*128*32*2 bytes each. Scratch remains a
    // separate real8/16KiB allocation; admission never assumes lifetime aliasing.
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut scratch = SharedMemory::<f32>::new(comptime!(SUBGROUPS * 128));
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
    let a2 = if comptime!(M_FRAGMENTS == 4) {
        unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    } else {
        a0
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a3 = if comptime!(M_FRAGMENTS == 4) {
        unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    } else {
        a1
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
    let c4 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c0
    };
    let c5 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c1
    };
    let c6 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c2
    };
    let c7 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c3
    };
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
        gr_down,
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
                gr_down,
            );
        }
        #[unroll]
        for step in 0..2usize {
            let af = current * 1024 + (subgroup / 4) * comptime!(SUBGROUP_ROWS * 8) + step * 4;
            let bf = current * 1024 + (subgroup % 4) * 256 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(af + 0, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(af + 64, a.len()), 32);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::load(&a2, &a.to_slice().slice(af + 128, a.len()), 32);
                cmma::load(&a3, &a.to_slice().slice(af + 192, a.len()), 32);
            }
            cmma::load(&b0, &b.to_slice().slice(bf + 0, b.len()), 32);
            cmma::load(&b1, &b.to_slice().slice(bf + 128, b.len()), 32);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
            if comptime!(M_FRAGMENTS == 4) {
                cmma::execute::<F, F, f32, f32>(&a2, &b0, &c4, &c4);
                cmma::execute::<F, F, f32, f32>(&a2, &b1, &c5, &c5);
                cmma::execute::<F, F, f32, f32>(&a3, &b0, &c6, &c6);
                cmma::execute::<F, F, f32, f32>(&a3, &b1, &c7, &c7);
            }
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
        gr_down,
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
        gr_down,
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
        gr_down,
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
        gr_down,
    );
    if comptime!(M_FRAGMENTS == 4) {
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
            gr_down,
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
            gr_down,
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
            gr_down,
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
            gr_down,
        );
    }
}

#[cfg(feature="strata-dense-direct-store")]
include!("strata_matrix_native_direct_store.rs");
#[cfg(feature="strata-dense-slm40")]
include!("strata_matrix_dense_slm40.rs");
#[cfg(feature = "strata-deep-matrix")]
include!("strata_matrix_dense_deep.rs");
#[cfg(feature = "strata-deep-matrix")]
include!("strata_matrix_dense_asymmetric.rs");
#[cfg(feature="strata-dense-prefix")]
include!("strata_matrix_dense_prefix.rs");
#[cfg(feature="strata-hc-down-inject")]
include!("strata_matrix_hc_down_inject.rs");

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
    selected_min(count,columns,rows,weight_first,shared,max_units,max_dim,plane_min,plane_max,512)
}
fn selected_min(
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    shared: usize,
    max_units: u32,
    max_dim: (u32, u32, u32),
    plane_min: u32,
    plane_max: u32,
    minimum: usize,
) -> bool {
    count >= minimum
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
fn gr_down_shape(
    bfloat: bool,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) -> bool {
    cfg!(feature = "strata-gr-wide-tail")
        && bfloat
        && columns == 10240
        && rows == 320
        && weight_first == 0
        && out_first == 0
        && out_stride == 320
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
    try_dense_impl(client,h,count,columns,rows,weight_first,out_first,out_stride,bfloat,cfg!(feature="strata-dense-slm40"),false)
}
#[cfg(feature="strata-dense-slm40")]
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense_compact(
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
    try_dense_impl(client,h,count,columns,rows,weight_first,out_first,out_stride,bfloat,false,false)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense_impl(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
    #[allow(unused_variables)] padded: bool,
    prefix: bool,
) -> Result<bool> {
    let gr_down = gr_down_shape(bfloat, columns, rows, weight_first, out_first, out_stride);
    let hw = &client.properties().hardware;
    if !selected_min(
        count,
        columns,
        if gr_down {
            rows.div_ceil(BLOCK) * BLOCK
        } else {
            rows
        },
        weight_first,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
        hw.plane_size_min,
        hw.plane_size_max,
        if prefix {128} else {512},
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
    let gx = rows.div_ceil(BLOCK);
    let gy = count / BLOCK;
    if gx > hw.max_cube_count.0 as usize
        || gy > hw.max_cube_count.1 as usize
        || hw.max_cube_count.2 < 1
    {
        return Err(Error::LimitExceeded("native wide dense grid"));
    }
    #[cfg(feature="strata-dense-direct-store")]
    if let Some(grid)=direct_selection(&h[2],count,rows,out_first,out_stride,hw.max_cube_count) {
        if gr_down && !prefix {crate::strata_prefill_profile::count("gr_wide_tail_products",1);}
        #[cfg(feature="strata-dense-slm40")]
        if padded {
            run_slm40_store(client,h,count,columns,rows,weight_first,out_first,out_stride,bfloat,gr_down,grid)?;
            return Ok(true);
        }
        run_direct_store(client,h,count,columns,rows,weight_first,out_first,out_stride,bfloat,gr_down,grid)?;
        return Ok(true);
    }
    let [input, weight, output, fault] = h;
    // SAFETY: true aligned vec4 half owners; no scalar-length/vector-pointer forgery.
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 8) };
    let wa =
        unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 8) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    if gr_down && !prefix {
        crate::strata_prefill_profile::count("gr_wide_tail_products", 1);
    }
    // SAFETY: complete original logical owners, all grid/offset/index limits,
    // actual sixteen/thirty-two-subgroup geometry and full tiles checked above.
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
                gr_down,
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
                false,
            );
        }
    }
    Ok(true)
}

fn definition<F: Float>(gr_down: bool) -> KernelDefinition {
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
        gr_down,
    );
    let errors = b.scope.pop_errors();
    assert!(errors.is_empty(), "{errors:?}");
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}
pub(super) fn definitions() -> Vec<(&'static str, KernelDefinition)> {
    let mut out = vec![
        ("dense_wide_native_fp16", definition::<f16>(false)),
        ("dense_wide_native_bf16", definition::<bf16>(false)),
    ];
    if cfg!(feature = "strata-gr-wide-tail") {
        out.push(("dense_wide_native_gr_down_bf16", definition::<bf16>(true)));
    }
    out
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    #[test]
    fn full_tile_selection_requires_real_caps_and_preserves_tails() {
        let yes = |c, k, n, w, s, u, d, pmin, pmax| selected(c, k, n, w, s, u, d, pmin, pmax);
        assert!(yes(
            512,
            32,
            128,
            0,
            SHARED_BYTES,
            THREADS,
            (THREADS, 1, 1),
            16,
            32
        ));
        for (c, k, n, w, shared, units, dim, pmin, pmax) in [
            (
                511,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                513,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                16,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                320,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                1,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES - 1,
                THREADS,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS - 1,
                (THREADS, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS - 1, 1, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 0, 1),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 0),
                16,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                32,
                32,
            ),
            (
                512,
                32,
                128,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                8,
                8,
            ),
        ] {
            assert!(!yes(c, k, n, w, shared, units, dim, pmin, pmax));
        }
        assert_eq!(SUBGROUPS * M_FRAGMENTS * 8 * 32, BLOCK * BLOCK);
        assert_eq!(SHARED_BYTES, 32768 + SUBGROUPS * 128 * 4);
        assert_eq!(THREADS as usize * STAGE_ROUNDS, 512);
    }
    #[test]
    fn full_vec4_stages_and_scratch_publications_cover_each_original_cell_once() {
        let mut a = vec![0; 1024];
        for lane in 0..THREADS as usize {
            for round in 0..STAGE_ROUNDS {
                let job = lane + round * THREADS as usize;
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
        for subgroup in 0..SUBGROUPS {
            for m in 0..M_FRAGMENTS {
                for n in 0..2 {
                    let mut scratch = vec![0; 128];
                    for lane in 0..16 {
                        for round in 0..8 {
                            let i = lane + round * 16;
                            scratch[i] += 1;
                            let token = subgroup / 4 * SUBGROUP_ROWS + m * 8 + i / 16;
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
    #[cfg(feature = "strata-gr-wide-tail")]
    #[test]
    fn gr_down_selection_retains_actual_hardware_and_unpadded_owner_bounds() {
        assert!(gr_down_shape(true, 10240, 320, 0, 0, 320));
        for (bf, k, n, w, o, st) in [
            (false, 10240, 320, 0, 0, 320),
            (true, 10236, 320, 0, 0, 320),
            (true, 10240, 319, 0, 0, 320),
            (true, 10240, 384, 0, 0, 384),
            (true, 10240, 320, 4, 0, 320),
            (true, 10240, 320, 0, 1, 320),
            (true, 10240, 320, 0, 0, 321),
        ] {
            assert!(!gr_down_shape(bf, k, n, w, o, st));
        }
        for count in [512usize, 640, 4096] {
            assert!(selected(
                count,
                10240,
                384,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32
            ));
            let own = |n| Handle::new(cubecl_common::stream_id::StreamId::current(), n);
            let h = [
                own((count * 10240 * 2) as u64),
                own(320 * 10240 * 2),
                own((count * 320 * 4) as u64),
                own(4),
            ];
            assert!(check_bounds(&h, count, 10240, 320, 0, 0, 320).is_ok());
            assert!(check_bounds(&h, count, 10240, 384, 0, 0, 320).is_err());
            for i in 0..4 {
                let mut bad = h.clone();
                bad[i] = own(h[i].size_in_used() - 1);
                assert!(check_bounds(&bad, count, 10240, 320, 0, 0, 320).is_err());
            }
        }
        for count in [0, 45, 149, 511, 513, 639, 4095] {
            assert!(!selected(
                count,
                10240,
                384,
                0,
                SHARED_BYTES,
                THREADS,
                (THREADS, 1, 1),
                16,
                32
            ));
        }
        assert!(!selected(
            512,
            10240,
            384,
            0,
            SHARED_BYTES - 1,
            THREADS,
            (THREADS, 1, 1),
            16,
            32
        ));
    }
    #[cfg(feature = "strata-gr-wide-tail")]
    #[test]
    fn gr_down_last_tile_has_no_padded_weight_reads_or_output_writes() {
        for k in (0..10240).step_by(32) {
            let mut seen = vec![0u8; 320 * 8];
            let mut padding = 0;
            for base in [0usize, 128, 256] {
                for lane in 0..THREADS as usize {
                    for round in 0..STAGE_ROUNDS {
                        let job = lane + round * THREADS as usize;
                        let row = base + job / 4;
                        let col = k + (job % 4) * 8;
                        for v in 0..2 {
                            if row < 320 {
                                let source = (row * 10240 + col) / 4 + v;
                                assert!(source < 320 * 10240 / 4);
                                assert_eq!(source / (10240 / 4), row);
                                seen[row * 8 + (col - k) / 4 + v] += 1;
                            } else {
                                padding += 1;
                            }
                        }
                    }
                }
            }
            assert!(seen.iter().all(|&v| v == 1));
            assert_eq!(padding, 64 * 8);
        }
        let mut out = vec![0u8; 128 * 320];
        let mut masked = 0;
        for base in [0usize, 128, 256] {
            for sg in 0..SUBGROUPS {
                for m in 0..M_FRAGMENTS {
                    for n in 0..2 {
                        for lane in 0..16 {
                            for r in 0..8 {
                                let i = lane + r * 16;
                                let t = sg / 4 * SUBGROUP_ROWS + m * 8 + i / 16;
                                let row = base + sg % 4 * 32 + n * 16 + i % 16;
                                if row < 320 {
                                    out[t * 320 + row] += 1;
                                } else {
                                    masked += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert!(out.iter().all(|&v| v == 1));
        assert_eq!(masked, 128 * 64);
    }
    pub(crate) fn assert_publication_scope_contract(words: &[u32]) {
        fn constants(words: &[u32]) -> std::collections::BTreeMap<u32, u32> {
            let mut values = std::collections::BTreeMap::new();
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                if words[at] & 65535 == 43 && n == 4 {
                    values.insert(words[at + 2], words[at + 3]);
                }
                at += n;
            }
            values
        }
        let result = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(words).unwrap();
        if !cfg!(feature = "strata-native-subgroup-publish") {
            assert!(result.is_none());
            return;
        }
        let after = result.expect("actual selected subgroup publication");
        let values = constants(words);
        let mut restored = after.clone();
        let mut subgroup = 0;
        let mut workgroup = 0;
        let mut at = 5;
        let mut first = None;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            if words[at] & 65535 == 224 {
                let scope = values[&words[at + 1]];
                assert_eq!(values[&words[at + 2]], scope);
                if scope == 3 {
                    subgroup += 1;
                    first.get_or_insert(at);
                    assert_eq!(values[&words[at + 3]], 0);
                    assert_eq!(values[&after[at + 3]], 264);
                    restored[at + 3] = words[at + 3];
                } else {
                    assert_eq!(scope, 2);
                    workgroup += 1;
                }
            }
            at += n;
        }
        assert_eq!(subgroup, M_FRAGMENTS * 4);
        assert_eq!(workgroup, 2);
        assert_eq!(
            restored, words,
            "ONLY existing subgroup memory-semantics operands change"
        );
        assert!(
            wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&after)
                .unwrap()
                .is_none()
        );
        let at = first.unwrap();
        let mut bad = words.to_vec();
        bad[at + 2] = words[at + 3];
        assert!(wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&bad).is_err());
        let mut bad = words.to_vec();
        bad[at + 3] = words[at + 1];
        assert!(wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&bad).is_err());
        let mut bad = words.to_vec();
        let mut at = 5;
        while at < bad.len() {
            let n = (bad[at] >> 16) as usize;
            if bad[at] & 65535 == 43 && n == 4 && bad[at + 3] == 264 {
                bad[at + 3] = 265;
            }
            at += n;
        }
        assert!(wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&bad).is_err());
    }

    #[test]
    fn subgroup_publication_slots_are_disjoint_and_have_only_local_consumers() {
        let mut owners = vec![None; SUBGROUPS * 128];
        for sg in 0..SUBGROUPS {
            for lane in 0..16usize {
                for round in 0..8usize {
                    let at = sg * 128 + lane + round * 16;
                    assert!(owners[at].replace(sg).is_none());
                    assert_eq!(at / 128, sg);
                }
            }
        }
        assert!(owners.iter().all(Option::is_some));
        for sg in 0..SUBGROUPS {
            for i in 0..128usize {
                assert_eq!(owners[sg * 128 + i], Some(sg));
            }
        }
    }
    #[test]
    fn actual_full_tile_module_keeps_real_subgroups_ieee_stages_publication_and_faults() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        for mask in 0..8 {
            for (name, definition) in [
                ("fp16", definition::<f16>(false)),
                ("bf16", definition::<bf16>(false)),
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
                assert_publication_scope_contract(&words);
                let words = wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)
                    .unwrap()
                    .unwrap_or(words);
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
                assert_eq!(local, Some(vec![THREADS, 1, 1]));
                assert!(builtins.contains(&40) && builtins.contains(&41));
                assert_eq!(products, M_FRAGMENTS * 4);
                assert_eq!(loads, (M_FRAGMENTS + 2) * 2);
                assert_eq!(stores, M_FRAGMENTS * 2);
                assert_eq!(barriers, M_FRAGMENTS * 4 + 2);
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
                    "compiled_public_native_wide threads={THREADS} name={name} mask={mask} shared_bytes={} products={products} loads={loads} stores={stores} barriers={barriers}",
                    kernel.shared_size
                );
            }
        }
    }
}
