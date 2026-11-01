//! Dense NT GEMM for Intel Xe2 through Vulkan cooperative matrices.
//!
//! output[out_first + t * out_stride + r] = sum_k input[t, k] * weight[weight_first + r, k]
//! with f16 operands (K contiguous for both) and f32 accumulation/output.
//!
//! Workgroup tile 128 tokens x 128 rows, 16 subgroups of 16 lanes, each owning a
//! 32 x 32 block as 4 x 2 accumulators of 8 x 16. K advances 32 per stage through
//! double-buffered, row-padded shared memory filled with vec4 loads. Non-finite
//! outputs are written as 0 and set the fault word, matching the existing kernels.
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{Error, Result};

const TILE: usize = 128;
const KSTEP: usize = 32;
/// Staged row pitch in vec4 (40 halves: 32 data + 8 padding against bank conflicts).
const PV: usize = 10;
const PITCH: u32 = 40;
/// vec4 slots per operand stage.
const STAGE: usize = TILE * PV;

#[cube]
#[allow(clippy::too_many_arguments)]
fn stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    cols4: usize,
    t0: usize,
    r0: usize,
    wf4: usize,
    k4: usize,
    buf: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    #[unroll]
    for j in 0..4usize {
        let idx = lane + j * 256;
        let row = idx / 8;
        let c = idx % 8;
        a[buf * 1280 + row * 10 + c] = input[(t0 + row) * cols4 + k4 + c];
        b[buf * 1280 + row * 10 + c] = weight[wf4 + (r0 + row) * cols4 + k4 + c];
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn emit(
    acc: &cmma::Matrix<f32>,
    out: &mut SharedMemory<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    sg: usize,
    token: usize,
    row: usize,
    out_first: usize,
    out_stride: usize,
) {
    cmma::store(
        &mut out.slice_mut(sg * 128, sg * 128 + 128),
        acc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    let mut i = usize::cast_from(UNIT_POS_PLANE);
    while i < 128 {
        let v = out[sg * 128 + i];
        let at = out_first + (token + i / 16) * out_stride + row + i % 16;
        let finite = v.abs() <= f32::MAX;
        output[at] = select(finite, v, 0.0);
        if !finite {
            fault[0].fetch_or(1);
        }
        i += 16;
    }
    sync_cube();
}

#[cube(launch_unchecked)]
pub fn gemm_nt128<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2560usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2560usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 4) * 32;
    let bn = (sg % 4) * 32;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c00 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c01 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c10 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c11 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c20 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c21 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c30 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c31 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt);
        }
        #[unroll]
        for step in 0..2usize {
            let ab = cur * 1280 + am * 10 + step * 4;
            let bb = cur * 1280 + bn * 10 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 40);
            cmma::load(&a1, &a.to_slice().slice(ab + 80, a.len()), 40);
            cmma::load(&a2, &a.to_slice().slice(ab + 160, a.len()), 40);
            cmma::load(&a3, &a.to_slice().slice(ab + 240, a.len()), 40);
            cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 40);
            cmma::load(&b1, &b.to_slice().slice(bb + 160, b.len()), 40);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c00, &c00);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c01, &c01);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c10, &c10);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c11, &c11);
            cmma::execute::<F, F, f32, f32>(&a2, &b0, &c20, &c20);
            cmma::execute::<F, F, f32, f32>(&a2, &b1, &c21, &c21);
            cmma::execute::<F, F, f32, f32>(&a3, &b0, &c30, &c30);
            cmma::execute::<F, F, f32, f32>(&a3, &b1, &c31, &c31);
        }
        sync_cube();
        cur = nxt;
        k += 32;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c00, &mut out, output, fault, sg, t, r, out_first, out_stride);
    emit(&c01, &mut out, output, fault, sg, t, r + 16, out_first, out_stride);
    emit(&c10, &mut out, output, fault, sg, t + 8, r, out_first, out_stride);
    emit(&c11, &mut out, output, fault, sg, t + 8, r + 16, out_first, out_stride);
    emit(&c20, &mut out, output, fault, sg, t + 16, r, out_first, out_stride);
    emit(&c21, &mut out, output, fault, sg, t + 16, r + 16, out_first, out_stride);
    emit(&c30, &mut out, output, fault, sg, t + 24, r, out_first, out_stride);
    emit(&c31, &mut out, output, fault, sg, t + 24, r + 16, out_first, out_stride);
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn stage512<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    cols4: usize,
    t0: usize,
    r0: usize,
    wf4: usize,
    k4: usize,
    buf: usize,
) {
    let lane = usize::cast_from(UNIT_POS);
    #[unroll]
    for j in 0..2usize {
        let idx = lane + j * 512;
        let row = idx / 8;
        let c = idx % 8;
        a[buf * 1024 + row * 8 + c] = input[(t0 + row) * cols4 + k4 + c];
        b[buf * 1024 + row * 8 + c] = weight[wf4 + (r0 + row) * cols4 + k4 + c];
    }
}

/// 32 subgroups, each a 16 x 32 block (2 A x 2 B fragments, 4 accumulators).
#[cube(launch_unchecked)]
pub fn gemm_nt128_sg32<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 4) * 16;
    let bn = (sg % 4) * 32;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c00 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c01 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c10 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c11 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt);
        }
        #[unroll]
        for step in 0..2usize {
            let ab = cur * 1024 + am * 8 + step * 4;
            let bb = cur * 1024 + bn * 8 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(ab + 64, a.len()), 32);
            cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 32);
            cmma::load(&b1, &b.to_slice().slice(bb + 128, b.len()), 32);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c00, &c00);
            cmma::execute::<F, F, f32, f32>(&a0, &b1, &c01, &c01);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c10, &c10);
            cmma::execute::<F, F, f32, f32>(&a1, &b1, &c11, &c11);
        }
        sync_cube();
        cur = nxt;
        k += 32;
    }
    let mut out = SharedMemory::<f32>::new(4096usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c00, &mut out, output, fault, sg, t, r, out_first, out_stride);
    emit(&c01, &mut out, output, fault, sg, t, r + 16, out_first, out_stride);
    emit(&c10, &mut out, output, fault, sg, t + 8, r, out_first, out_stride);
    emit(&c11, &mut out, output, fault, sg, t + 8, r + 16, out_first, out_stride);
}

/// 32 subgroups, each a 32 x 16 block (4 A x 1 B fragments, 4 accumulators):
/// A fragments get vectorized SLM loads on Mesa, B fragments scalar ones, so
/// favor A. Subgroup grid 4 (tokens) x 8 (rows).
#[cube(launch_unchecked)]
pub fn gemm_nt128_a4<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 8) * 32;
    let bn = (sg % 8) * 16;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt);
        }
        #[unroll]
        for step in 0..2usize {
            let ab = cur * 1024 + am * 8 + step * 4;
            let bb = cur * 1024 + bn * 8 + step * 4;
            cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 32);
            cmma::load(&a1, &a.to_slice().slice(ab + 64, a.len()), 32);
            cmma::load(&a2, &a.to_slice().slice(ab + 128, a.len()), 32);
            cmma::load(&a3, &a.to_slice().slice(ab + 192, a.len()), 32);
            cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 32);
            cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<F, F, f32, f32>(&a1, &b0, &c1, &c1);
            cmma::execute::<F, F, f32, f32>(&a2, &b0, &c2, &c2);
            cmma::execute::<F, F, f32, f32>(&a3, &b0, &c3, &c3);
        }
        sync_cube();
        cur = nxt;
        k += 32;
    }
    let mut out = SharedMemory::<f32>::new(4096usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c0, &mut out, output, fault, sg, t, r, out_first, out_stride);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride);
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn a4_step<F: Float>(
    a: &SharedMemory<Vector<F, Const<4>>>,
    b: &SharedMemory<Vector<F, Const<4>>>,
    a0: &cmma::Matrix<F>,
    a1: &cmma::Matrix<F>,
    a2: &cmma::Matrix<F>,
    a3: &cmma::Matrix<F>,
    b0: &cmma::Matrix<F>,
    c0: &cmma::Matrix<f32>,
    c1: &cmma::Matrix<f32>,
    c2: &cmma::Matrix<f32>,
    c3: &cmma::Matrix<f32>,
    ab: usize,
    bb: usize,
) {
    #[unroll]
    for step in 0..2usize {
        cmma::load(a0, &a.to_slice().slice(ab + step * 4, a.len()), 32);
        cmma::load(a1, &a.to_slice().slice(ab + step * 4 + 64, a.len()), 32);
        cmma::load(a2, &a.to_slice().slice(ab + step * 4 + 128, a.len()), 32);
        cmma::load(a3, &a.to_slice().slice(ab + step * 4 + 192, a.len()), 32);
        cmma::load(b0, &b.to_slice().slice(bb + step * 4, b.len()), 32);
        cmma::execute::<F, F, f32, f32>(a0, b0, c0, c0);
        cmma::execute::<F, F, f32, f32>(a1, b0, c1, c1);
        cmma::execute::<F, F, f32, f32>(a2, b0, c2, c2);
        cmma::execute::<F, F, f32, f32>(a3, b0, c3, c3);
    }
}

/// `gemm_nt128_a4` with the K loop unrolled by two stages (K % 64 == 0), so
/// both shared-memory buffers are addressed by constants.
#[cube(launch_unchecked)]
pub fn gemm_nt128_a4u<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 8) * 32;
    let bn = (sg % 8) * 16;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let ab0 = am * 8;
    let bb0 = bn * 8;
    stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0);
    sync_cube();
    let mut k = 0usize;
    while k < columns {
        // Stage 1 <- k + 32 while computing stage 0.
        stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, 1);
        a4_step::<F>(&a, &b, &a0, &a1, &a2, &a3, &b0, &c0, &c1, &c2, &c3, ab0, bb0);
        sync_cube();
        // Stage 0 <- k + 64 while computing stage 1.
        if k + 64 < columns {
            stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 64) / 4, 0);
        }
        a4_step::<F>(&a, &b, &a0, &a1, &a2, &a3, &b0, &c0, &c1, &c2, &c3, ab0 + 1024, bb0 + 1024);
        sync_cube();
        k += 64;
    }
    let mut out = SharedMemory::<f32>::new(4096usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c0, &mut out, output, fault, sg, t, r, out_first, out_stride);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride);
}

/// `gemm_nt128_a4` with separate fragments for the two K16 steps, so the
/// scheduler can issue all fragment loads ahead of the products.
/// 32 subgroups, each a 32 x 16 block (4 A x 1 B fragments, 4 accumulators):
/// A fragments get vectorized SLM loads on Mesa, B fragments scalar ones, so
/// favor A. Subgroup grid 4 (tokens) x 8 (rows).
#[cube(launch_unchecked)]
pub fn gemm_nt128_a4p<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 8) * 32;
    let bn = (sg % 8) * 16;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    // SAFETY: every operand fragment is loaded before each product.
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage512::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt);
        }
        let ab = cur * 1024 + am * 8;
        let bb = cur * 1024 + bn * 8;
        cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 32);
        cmma::load(&a1, &a.to_slice().slice(ab + 64, a.len()), 32);
        cmma::load(&a2, &a.to_slice().slice(ab + 128, a.len()), 32);
        cmma::load(&a3, &a.to_slice().slice(ab + 192, a.len()), 32);
        cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 32);
        cmma::load(&a4, &a.to_slice().slice(ab + 4, a.len()), 32);
        cmma::load(&a5, &a.to_slice().slice(ab + 68, a.len()), 32);
        cmma::load(&a6, &a.to_slice().slice(ab + 132, a.len()), 32);
        cmma::load(&a7, &a.to_slice().slice(ab + 196, a.len()), 32);
        cmma::load(&b1, &b.to_slice().slice(bb + 4, b.len()), 32);
        cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
        cmma::execute::<F, F, f32, f32>(&a1, &b0, &c1, &c1);
        cmma::execute::<F, F, f32, f32>(&a2, &b0, &c2, &c2);
        cmma::execute::<F, F, f32, f32>(&a3, &b0, &c3, &c3);
        cmma::execute::<F, F, f32, f32>(&a4, &b1, &c0, &c0);
        cmma::execute::<F, F, f32, f32>(&a5, &b1, &c1, &c1);
        cmma::execute::<F, F, f32, f32>(&a6, &b1, &c2, &c2);
        cmma::execute::<F, F, f32, f32>(&a7, &b1, &c3, &c3);
        sync_cube();
        cur = nxt;
        k += 32;
    }
    let mut out = SharedMemory::<f32>::new(4096usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c0, &mut out, output, fault, sg, t, r, out_first, out_stride);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride);
}

fn vec4_arg(handle: Handle) -> Result<ArrayArg<WgpuRuntime>> {
    if handle.size_in_used() == 0
        || !handle.size_in_used().is_multiple_of(8)
        || !handle.offset_start.unwrap_or(0).is_multiple_of(8)
    {
        return Err(Error::InvalidBufferLength);
    }
    let vectors = handle.size_in_used() as usize / 8;
    // SAFETY: aligned vec4<f16> binding of the whole handle.
    Ok(unsafe { ArrayArg::from_raw_parts(handle, vectors) })
}

/// Runs the tiled GEMM when the shape fits its constraints; `Ok(false)` means
/// the caller should use another kernel. Bounds were checked by `dense`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dense(
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
    if bfloat
        || !count.is_multiple_of(TILE)
        || !rows.is_multiple_of(TILE)
        || !columns.is_multiple_of(KSTEP)
        || !weight_first.is_multiple_of(4)
        || std::env::var_os("KUNO_CLAUDE_GEMM_OFF").is_some()
    {
        return Ok(false);
    }
    let _ = (PITCH, STAGE);
    let [input, weight, output, fault] = h;
    let ia = vec4_arg(input)?;
    let wa = vec4_arg(weight)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    // SAFETY: `dense` validated every operand interval for this shape; the
    // grid covers exactly count x rows in full 128 x 128 tiles.
    unsafe {
        let grid = CubeCount::Static((rows / TILE) as u32, (count / TILE) as u32, 1);
        if std::env::var_os("KUNO_CLAUDE_GEMM_SG16").is_some() {
            gemm_nt128::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(256), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride,
            );
        } else if std::env::var_os("KUNO_CLAUDE_GEMM_COLB").is_some() {
            gemm_nt128_sg32::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride,
            );
        } else if std::env::var_os("KUNO_CLAUDE_GEMM_A4").is_some() || !columns.is_multiple_of(64) {
            gemm_nt128_a4::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride,
            );
        } else if std::env::var_os("KUNO_CLAUDE_GEMM_A4U").is_some() {
            gemm_nt128_a4u::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride,
            );
        } else {
            gemm_nt128_a4p::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride,
            );
        }
    }
    Ok(true)
}
