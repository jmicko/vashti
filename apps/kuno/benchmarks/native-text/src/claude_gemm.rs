//! Dense NT GEMM for Intel Xe2 through Vulkan cooperative matrices.
//!
//! output[out_first + t * out_stride + r] = sum_k input[t, k] * weight[weight_first + r, k]
//! with f16 or bf16 operands (K contiguous for both) and f32 accumulation/output.
//!
//! Workgroup tile 128 tokens x 128 rows, 32 subgroups of 16 lanes, each owning a
//! 32 x 16 block as 4 A x 1 B fragments (four 8 x 16 accumulators). K advances 32
//! per stage through double-buffered shared memory filled with vec4 loads; tokens
//! and rows past the bounds stage as zeros and are not stored. Non-finite outputs
//! are written as 0 and set the fault word, matching the existing kernels.
//!
//! Measured on B70 / Mesa 26.2.6 (see `--public-dense-bench`), which shaped this:
//! - Mesa compiles at 128 GRF and spills beyond ~4 accumulators per subgroup.
//! - SLM loads of A (row-major) fragments vectorize; B (column-major) fragments
//!   lower to scalar d32 loads and row-major B to d16 loads, so favor A per B.
//! - 32 x 16 per subgroup: 55-57 TFLOPS vs 24-29 for the previous kernels;
//!   16 x 32: ~42; 32 x 32 (8 accumulators): ~18 with spills.
//! - Workgroups > 256 lanes must use subgroup builtins (PLANE_POS / UNIT_POS_PLANE).
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{Error, Result};

const TILE: usize = 128;
/// Benchmark-only switch: route `dense` through the original variant chain.
pub(crate) static BYPASS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
const KSTEP: usize = 32;

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
    count: usize,
    rows: usize,
    #[comptime] guarded: bool,
) {
    let lane = usize::cast_from(UNIT_POS);
    #[unroll]
    for j in 0..2usize {
        let idx = lane + j * 512;
        let row = idx / 8;
        let c = idx % 8;
        if comptime!(guarded) {
            let zero = Vector::<F, Const<4>>::empty().fill(F::new(0.0));
            let mut va = zero;
            if t0 + row < count {
                va = input[(t0 + row) * cols4 + k4 + c];
            }
            a[buf * 1024 + (c / 4) * 512 + row * 4 + c % 4] = va;
            let mut vb = zero;
            if r0 + row < rows {
                vb = weight[wf4 + (r0 + row) * cols4 + k4 + c];
            }
            b[buf * 1024 + row * 8 + c] = vb;
        } else {
            a[buf * 1024 + (c / 4) * 512 + row * 4 + c % 4] = input[(t0 + row) * cols4 + k4 + c];
            b[buf * 1024 + row * 8 + c] = weight[wf4 + (r0 + row) * cols4 + k4 + c];
        }
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
    count: usize,
    rows: usize,
    #[comptime] guarded: bool,
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
        let t = token + i / 16;
        let r = row + i % 16;
        if !comptime!(guarded) || (t < count && r < rows) {
            let v = out[sg * 128 + i];
            let finite = v.abs() <= f32::MAX;
            output[out_first + t * out_stride + r] = select(finite, v, 0.0);
            if !finite {
                fault[0].fetch_or(1);
            }
        }
        i += 16;
    }
    sync_cube();
}

#[cube(launch_unchecked)]
pub fn gemm_nt<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] guarded: bool,
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
    let a4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0, count, rows, guarded);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt, count, rows, guarded);
        }
        // Two K16 steps with separate fragments so loads can run ahead.
        // A is staged fragment-contiguous: [K16 half][row][16 elements].
        let aa = cur * 1024 + am * 4;
        let bb = cur * 1024 + bn * 8;
        cmma::load(&a0, &a.to_slice().slice(aa + 0, a.len()), 16);
        cmma::load(&a1, &a.to_slice().slice(aa + 32, a.len()), 16);
        cmma::load(&a2, &a.to_slice().slice(aa + 64, a.len()), 16);
        cmma::load(&a3, &a.to_slice().slice(aa + 96, a.len()), 16);
        cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 32);
        cmma::load(&a4, &a.to_slice().slice(aa + 512, a.len()), 16);
        cmma::load(&a5, &a.to_slice().slice(aa + 544, a.len()), 16);
        cmma::load(&a6, &a.to_slice().slice(aa + 576, a.len()), 16);
        cmma::load(&a7, &a.to_slice().slice(aa + 608, a.len()), 16);
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
    emit(&c0, &mut out, output, fault, sg, t, r, out_first, out_stride, count, rows, guarded);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride, count, rows, guarded);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride, count, rows, guarded);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride, count, rows, guarded);
}


/// Large-GRF variant (Mesa with Xe2 Large GRF Mode, 256 registers/thread):
/// 256-lane workgroups (16 subgroups) on the same 128 x 128 tile, each
/// subgroup owning 64 x 16 as 8 A x 1 B fragments (eight accumulators).
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage8<F: Float>(
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
    count: usize,
    rows: usize,
    #[comptime] guarded: bool,
) {
    let lane = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
    #[unroll]
    for j in 0..4usize {
        let idx = lane + j * 256;
        let row = idx / 8;
        let c = idx % 8;
        if comptime!(guarded) {
            let zero = Vector::<F, Const<4>>::empty().fill(F::new(0.0));
            let mut va = zero;
            if t0 + row < count {
                va = input[(t0 + row) * cols4 + k4 + c];
            }
            a[buf * 1024 + (c / 4) * 512 + row * 4 + c % 4] = va;
            let mut vb = zero;
            if r0 + row < rows {
                vb = weight[wf4 + (r0 + row) * cols4 + k4 + c];
            }
            b[buf * 1024 + row * 8 + c] = vb;
        } else {
            a[buf * 1024 + (c / 4) * 512 + row * 4 + c % 4] = input[(t0 + row) * cols4 + k4 + c];
            b[buf * 1024 + row * 8 + c] = weight[wf4 + (r0 + row) * cols4 + k4 + c];
        }
    }
}

#[cube(launch_unchecked)]
pub fn gemm_nt8<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    columns: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] guarded: bool,
) {
    let mut a = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 8) * 64;
    let bn = (sg % 8) * 16;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a8 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a9 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a10 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a11 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a12 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a13 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a14 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a15 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c4 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c5 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c6 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c7 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    stage8::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, 0, 0, count, rows, guarded);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 32 < columns {
            stage8::<F>(input, weight, &mut a, &mut b, cols4, t0, r0, wf4, (k + 32) / 4, nxt, count, rows, guarded);
        }
        // A is staged fragment-contiguous: [K16 half][row][16 elements].
        let aa = cur * 1024 + am * 4;
        let bb = cur * 1024 + bn * 8;
        cmma::load(&a0, &a.to_slice().slice(aa + 0, a.len()), 16);
        cmma::load(&a1, &a.to_slice().slice(aa + 32, a.len()), 16);
        cmma::load(&a2, &a.to_slice().slice(aa + 64, a.len()), 16);
        cmma::load(&a3, &a.to_slice().slice(aa + 96, a.len()), 16);
        cmma::load(&a4, &a.to_slice().slice(aa + 128, a.len()), 16);
        cmma::load(&a5, &a.to_slice().slice(aa + 160, a.len()), 16);
        cmma::load(&a6, &a.to_slice().slice(aa + 192, a.len()), 16);
        cmma::load(&a7, &a.to_slice().slice(aa + 224, a.len()), 16);
        cmma::load(&b0, &b.to_slice().slice(bb + 0, b.len()), 32);
        cmma::load(&a8, &a.to_slice().slice(aa + 512, a.len()), 16);
        cmma::load(&a9, &a.to_slice().slice(aa + 544, a.len()), 16);
        cmma::load(&a10, &a.to_slice().slice(aa + 576, a.len()), 16);
        cmma::load(&a11, &a.to_slice().slice(aa + 608, a.len()), 16);
        cmma::load(&a12, &a.to_slice().slice(aa + 640, a.len()), 16);
        cmma::load(&a13, &a.to_slice().slice(aa + 672, a.len()), 16);
        cmma::load(&a14, &a.to_slice().slice(aa + 704, a.len()), 16);
        cmma::load(&a15, &a.to_slice().slice(aa + 736, a.len()), 16);
        cmma::load(&b1, &b.to_slice().slice(bb + 4, b.len()), 32);
        cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
        cmma::execute::<F, F, f32, f32>(&a1, &b0, &c1, &c1);
        cmma::execute::<F, F, f32, f32>(&a2, &b0, &c2, &c2);
        cmma::execute::<F, F, f32, f32>(&a3, &b0, &c3, &c3);
        cmma::execute::<F, F, f32, f32>(&a4, &b0, &c4, &c4);
        cmma::execute::<F, F, f32, f32>(&a5, &b0, &c5, &c5);
        cmma::execute::<F, F, f32, f32>(&a6, &b0, &c6, &c6);
        cmma::execute::<F, F, f32, f32>(&a7, &b0, &c7, &c7);
        cmma::execute::<F, F, f32, f32>(&a8, &b1, &c0, &c0);
        cmma::execute::<F, F, f32, f32>(&a9, &b1, &c1, &c1);
        cmma::execute::<F, F, f32, f32>(&a10, &b1, &c2, &c2);
        cmma::execute::<F, F, f32, f32>(&a11, &b1, &c3, &c3);
        cmma::execute::<F, F, f32, f32>(&a12, &b1, &c4, &c4);
        cmma::execute::<F, F, f32, f32>(&a13, &b1, &c5, &c5);
        cmma::execute::<F, F, f32, f32>(&a14, &b1, &c6, &c6);
        cmma::execute::<F, F, f32, f32>(&a15, &b1, &c7, &c7);
        sync_cube();
        cur = nxt;
        k += 32;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    let t = t0 + am;
    let r = r0 + bn;
    emit(&c0, &mut out, output, fault, sg, t + 0, r, out_first, out_stride, count, rows, guarded);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride, count, rows, guarded);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride, count, rows, guarded);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride, count, rows, guarded);
    emit(&c4, &mut out, output, fault, sg, t + 32, r, out_first, out_stride, count, rows, guarded);
    emit(&c5, &mut out, output, fault, sg, t + 40, r, out_first, out_stride, count, rows, guarded);
    emit(&c6, &mut out, output, fault, sg, t + 48, r, out_first, out_stride, count, rows, guarded);
    emit(&c7, &mut out, output, fault, sg, t + 56, r, out_first, out_stride, count, rows, guarded);
}


/// Direct variant for Mesa with Xe2 2D block loads (INTEL_CMAT_BLOCK2D):
/// fragments load straight from global memory into the DPAS layout; no
/// shared-memory staging or barriers in the K loop. 256 lanes, tile 128 x
/// 128, each subgroup 32 x 32 (4 A x 2 B, eight accumulators). Full tiles.
#[cube(launch_unchecked)]
pub fn gemm_direct<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] columns: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 4) * 32;
    let bn = (sg % 4) * 32;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128 + am;
    let r0 = usize::cast_from(CUBE_POS_X) * 128 + bn;
    let cols4 = columns / 4;
    let wf4 = weight_first / 4;
    let a0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c4 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c5 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c6 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c7 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let ilen = input.len();
    let wlen = weight.len();
    let mut k4 = 0usize;
    while k4 < cols4 {
        cmma::load(&a0, &input.to_slice().slice((t0 + 0) * cols4 + k4, ilen), comptime!(columns as u32));
        cmma::load(&a1, &input.to_slice().slice((t0 + 8) * cols4 + k4, ilen), comptime!(columns as u32));
        cmma::load(&a2, &input.to_slice().slice((t0 + 16) * cols4 + k4, ilen), comptime!(columns as u32));
        cmma::load(&a3, &input.to_slice().slice((t0 + 24) * cols4 + k4, ilen), comptime!(columns as u32));
        cmma::load(&b0, &weight.to_slice().slice(wf4 + (r0 + 0) * cols4 + k4, wlen), comptime!(columns as u32));
        cmma::load(&b1, &weight.to_slice().slice(wf4 + (r0 + 16) * cols4 + k4, wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
        cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
        cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
        cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
        cmma::execute::<F, F, f32, f32>(&a2, &b0, &c4, &c4);
        cmma::execute::<F, F, f32, f32>(&a2, &b1, &c5, &c5);
        cmma::execute::<F, F, f32, f32>(&a3, &b0, &c6, &c6);
        cmma::execute::<F, F, f32, f32>(&a3, &b1, &c7, &c7);
        k4 += 4;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    emit(&c0, &mut out, output, fault, sg, t0 + 0, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c1, &mut out, output, fault, sg, t0 + 0, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c2, &mut out, output, fault, sg, t0 + 8, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c3, &mut out, output, fault, sg, t0 + 8, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c4, &mut out, output, fault, sg, t0 + 16, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c5, &mut out, output, fault, sg, t0 + 16, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c6, &mut out, output, fault, sg, t0 + 24, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c7, &mut out, output, fault, sg, t0 + 24, r0 + 16, out_first, out_stride, count, rows, false);
}







/// Direct 2D-block variant: subgroup tile 32 x 32 (4 A x 2 B), K unrolled x4.
/// Indices are (tile base + K) + constant, so Mesa can turn constants into
/// immediate block offsets and share one payload per matrix and step.
#[cube(launch_unchecked)]
pub fn gemm_direct_42u4<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] columns: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let t0 = usize::cast_from(CUBE_POS_X) * 128 + (sg / 4) * 32;
    let r0 = usize::cast_from(CUBE_POS_Y) * 128 + (sg % 4) * 32;
    let cols4 = comptime!(columns / 4);
    let abase = t0 * cols4;
    let bbase = weight_first / 4 + r0 * cols4;
    let ilen = input.len();
    let wlen = weight.len();
    let a0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a2_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a2_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b2_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b2_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a3_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a3_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b3_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b3_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut k4 = 0usize;
    while k4 < cols4 {
        let ak = abase + k4;
        let bk = bbase + k4;
        cmma::load(&a0_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&b0_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_1, &c3_1, &c3_1);
        cmma::load(&a1_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&b1_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::load(&b1_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_1, &c3_1, &c3_1);
        cmma::load(&a2_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 8), ilen), comptime!(columns as u32));
        cmma::load(&a2_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 8), ilen), comptime!(columns as u32));
        cmma::load(&a2_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 8), ilen), comptime!(columns as u32));
        cmma::load(&a2_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 8), ilen), comptime!(columns as u32));
        cmma::load(&b2_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 8), wlen), comptime!(columns as u32));
        cmma::load(&b2_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 8), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a2_0, &b2_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a2_0, &b2_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a2_1, &b2_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a2_1, &b2_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a2_2, &b2_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a2_2, &b2_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a2_3, &b2_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a2_3, &b2_1, &c3_1, &c3_1);
        cmma::load(&a3_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 12), ilen), comptime!(columns as u32));
        cmma::load(&a3_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 12), ilen), comptime!(columns as u32));
        cmma::load(&a3_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 12), ilen), comptime!(columns as u32));
        cmma::load(&a3_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 12), ilen), comptime!(columns as u32));
        cmma::load(&b3_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 12), wlen), comptime!(columns as u32));
        cmma::load(&b3_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 12), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a3_0, &b3_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a3_0, &b3_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a3_1, &b3_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a3_1, &b3_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a3_2, &b3_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a3_2, &b3_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a3_3, &b3_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a3_3, &b3_1, &c3_1, &c3_1);
        k4 += 16;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    emit(&c0_0, &mut out, output, fault, sg, t0 + 0, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c0_1, &mut out, output, fault, sg, t0 + 0, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c1_0, &mut out, output, fault, sg, t0 + 8, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c1_1, &mut out, output, fault, sg, t0 + 8, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c2_0, &mut out, output, fault, sg, t0 + 16, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c2_1, &mut out, output, fault, sg, t0 + 16, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c3_0, &mut out, output, fault, sg, t0 + 24, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c3_1, &mut out, output, fault, sg, t0 + 24, r0 + 16, out_first, out_stride, count, rows, false);
}

/// Direct 2D-block variant: subgroup tile 32 x 64 (4 A x 4 B), K unrolled x2.
/// Indices are (tile base + K) + constant, so Mesa can turn constants into
/// immediate block offsets and share one payload per matrix and step.
#[cube(launch_unchecked)]
pub fn gemm_direct_44u2<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] columns: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let t0 = usize::cast_from(CUBE_POS_X) * 128 + (sg / 4) * 32;
    let r0 = usize::cast_from(CUBE_POS_Y) * 256 + (sg % 4) * 64;
    let cols4 = comptime!(columns / 4);
    let abase = t0 * cols4;
    let bbase = weight_first / 4 + r0 * cols4;
    let ilen = input.len();
    let wlen = weight.len();
    let a0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut k4 = 0usize;
    while k4 < cols4 {
        let ak = abase + k4;
        let bk = bbase + k4;
        cmma::load(&a0_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&b0_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_2, &weight.to_slice().slice(bk + comptime!(32 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_3, &weight.to_slice().slice(bk + comptime!(48 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_2, &c0_2, &c0_2);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_3, &c0_3, &c0_3);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_2, &c1_2, &c1_2);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_3, &c1_3, &c1_3);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_2, &c2_2, &c2_2);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_3, &c2_3, &c2_3);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_1, &c3_1, &c3_1);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_2, &c3_2, &c3_2);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_3, &c3_3, &c3_3);
        cmma::load(&a1_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&b1_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::load(&b1_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::load(&b1_2, &weight.to_slice().slice(bk + comptime!(32 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::load(&b1_3, &weight.to_slice().slice(bk + comptime!(48 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_2, &c0_2, &c0_2);
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_3, &c0_3, &c0_3);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_2, &c1_2, &c1_2);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_3, &c1_3, &c1_3);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_2, &c2_2, &c2_2);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_3, &c2_3, &c2_3);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_1, &c3_1, &c3_1);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_2, &c3_2, &c3_2);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_3, &c3_3, &c3_3);
        k4 += 8;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    emit(&c0_0, &mut out, output, fault, sg, t0 + 0, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c0_1, &mut out, output, fault, sg, t0 + 0, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c0_2, &mut out, output, fault, sg, t0 + 0, r0 + 32, out_first, out_stride, count, rows, false);
    emit(&c0_3, &mut out, output, fault, sg, t0 + 0, r0 + 48, out_first, out_stride, count, rows, false);
    emit(&c1_0, &mut out, output, fault, sg, t0 + 8, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c1_1, &mut out, output, fault, sg, t0 + 8, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c1_2, &mut out, output, fault, sg, t0 + 8, r0 + 32, out_first, out_stride, count, rows, false);
    emit(&c1_3, &mut out, output, fault, sg, t0 + 8, r0 + 48, out_first, out_stride, count, rows, false);
    emit(&c2_0, &mut out, output, fault, sg, t0 + 16, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c2_1, &mut out, output, fault, sg, t0 + 16, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c2_2, &mut out, output, fault, sg, t0 + 16, r0 + 32, out_first, out_stride, count, rows, false);
    emit(&c2_3, &mut out, output, fault, sg, t0 + 16, r0 + 48, out_first, out_stride, count, rows, false);
    emit(&c3_0, &mut out, output, fault, sg, t0 + 24, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c3_1, &mut out, output, fault, sg, t0 + 24, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c3_2, &mut out, output, fault, sg, t0 + 24, r0 + 32, out_first, out_stride, count, rows, false);
    emit(&c3_3, &mut out, output, fault, sg, t0 + 24, r0 + 48, out_first, out_stride, count, rows, false);
}

/// Direct 2D-block variant: subgroup tile 64 x 32 (8 A x 2 B), K unrolled x2.
/// Indices are (tile base + K) + constant, so Mesa can turn constants into
/// immediate block offsets and share one payload per matrix and step.
#[cube(launch_unchecked)]
pub fn gemm_direct_82u2<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] columns: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let t0 = usize::cast_from(CUBE_POS_X) * 256 + (sg / 4) * 64;
    let r0 = usize::cast_from(CUBE_POS_Y) * 128 + (sg % 4) * 32;
    let cols4 = comptime!(columns / 4);
    let abase = t0 * cols4;
    let bbase = weight_first / 4 + r0 * cols4;
    let ilen = input.len();
    let wlen = weight.len();
    let a0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let a1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_4 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_5 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_6 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1_7 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b1_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c4_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c4_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c5_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c5_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c6_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c6_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c7_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c7_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut k4 = 0usize;
    while k4 < cols4 {
        let ak = abase + k4;
        let bk = bbase + k4;
        cmma::load(&a0_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_4, &input.to_slice().slice(ak + comptime!(32 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_5, &input.to_slice().slice(ak + comptime!(40 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_6, &input.to_slice().slice(ak + comptime!(48 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_7, &input.to_slice().slice(ak + comptime!(56 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&b0_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_1, &c3_1, &c3_1);
        cmma::execute::<F, F, f32, f32>(&a0_4, &b0_0, &c4_0, &c4_0);
        cmma::execute::<F, F, f32, f32>(&a0_4, &b0_1, &c4_1, &c4_1);
        cmma::execute::<F, F, f32, f32>(&a0_5, &b0_0, &c5_0, &c5_0);
        cmma::execute::<F, F, f32, f32>(&a0_5, &b0_1, &c5_1, &c5_1);
        cmma::execute::<F, F, f32, f32>(&a0_6, &b0_0, &c6_0, &c6_0);
        cmma::execute::<F, F, f32, f32>(&a0_6, &b0_1, &c6_1, &c6_1);
        cmma::execute::<F, F, f32, f32>(&a0_7, &b0_0, &c7_0, &c7_0);
        cmma::execute::<F, F, f32, f32>(&a0_7, &b0_1, &c7_1, &c7_1);
        cmma::load(&a1_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_4, &input.to_slice().slice(ak + comptime!(32 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_5, &input.to_slice().slice(ak + comptime!(40 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_6, &input.to_slice().slice(ak + comptime!(48 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&a1_7, &input.to_slice().slice(ak + comptime!(56 * (columns / 4) + 4), ilen), comptime!(columns as u32));
        cmma::load(&b1_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::load(&b1_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 4), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a1_0, &b1_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a1_1, &b1_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a1_2, &b1_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a1_3, &b1_1, &c3_1, &c3_1);
        cmma::execute::<F, F, f32, f32>(&a1_4, &b1_0, &c4_0, &c4_0);
        cmma::execute::<F, F, f32, f32>(&a1_4, &b1_1, &c4_1, &c4_1);
        cmma::execute::<F, F, f32, f32>(&a1_5, &b1_0, &c5_0, &c5_0);
        cmma::execute::<F, F, f32, f32>(&a1_5, &b1_1, &c5_1, &c5_1);
        cmma::execute::<F, F, f32, f32>(&a1_6, &b1_0, &c6_0, &c6_0);
        cmma::execute::<F, F, f32, f32>(&a1_6, &b1_1, &c6_1, &c6_1);
        cmma::execute::<F, F, f32, f32>(&a1_7, &b1_0, &c7_0, &c7_0);
        cmma::execute::<F, F, f32, f32>(&a1_7, &b1_1, &c7_1, &c7_1);
        k4 += 8;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    emit(&c0_0, &mut out, output, fault, sg, t0 + 0, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c0_1, &mut out, output, fault, sg, t0 + 0, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c1_0, &mut out, output, fault, sg, t0 + 8, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c1_1, &mut out, output, fault, sg, t0 + 8, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c2_0, &mut out, output, fault, sg, t0 + 16, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c2_1, &mut out, output, fault, sg, t0 + 16, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c3_0, &mut out, output, fault, sg, t0 + 24, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c3_1, &mut out, output, fault, sg, t0 + 24, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c4_0, &mut out, output, fault, sg, t0 + 32, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c4_1, &mut out, output, fault, sg, t0 + 32, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c5_0, &mut out, output, fault, sg, t0 + 40, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c5_1, &mut out, output, fault, sg, t0 + 40, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c6_0, &mut out, output, fault, sg, t0 + 48, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c6_1, &mut out, output, fault, sg, t0 + 48, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c7_0, &mut out, output, fault, sg, t0 + 56, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c7_1, &mut out, output, fault, sg, t0 + 56, r0 + 16, out_first, out_stride, count, rows, false);
}

/// Direct 2D-block variant: subgroup tile 32 x 32 (4 A x 2 B), K unrolled x1.
/// Indices are (tile base + K) + constant, so Mesa can turn constants into
/// immediate block offsets and share one payload per matrix and step.
#[cube(launch_unchecked)]
pub fn gemm_direct_42u1<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    count: usize,
    rows: usize,
    #[comptime] columns: usize,
) {
    let sg = usize::cast_from(PLANE_POS);
    let t0 = usize::cast_from(CUBE_POS_X) * 128 + (sg / 4) * 32;
    let r0 = usize::cast_from(CUBE_POS_Y) * 128 + (sg % 4) * 32;
    let cols4 = comptime!(columns / 4);
    let abase = t0 * cols4;
    let bbase = weight_first / 4 + r0 * cols4;
    let ilen = input.len();
    let wlen = weight.len();
    let a0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_2 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a0_3 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0_0 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b0_1 = unsafe { cmma::Matrix::<F>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let c0_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c0_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3_1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut k4 = 0usize;
    while k4 < cols4 {
        let ak = abase + k4;
        let bk = bbase + k4;
        cmma::load(&a0_0, &input.to_slice().slice(ak + comptime!(0 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_1, &input.to_slice().slice(ak + comptime!(8 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_2, &input.to_slice().slice(ak + comptime!(16 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&a0_3, &input.to_slice().slice(ak + comptime!(24 * (columns / 4) + 0), ilen), comptime!(columns as u32));
        cmma::load(&b0_0, &weight.to_slice().slice(bk + comptime!(0 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::load(&b0_1, &weight.to_slice().slice(bk + comptime!(16 * (columns / 4) + 0), wlen), comptime!(columns as u32));
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_0, &c0_0, &c0_0);
        cmma::execute::<F, F, f32, f32>(&a0_0, &b0_1, &c0_1, &c0_1);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_0, &c1_0, &c1_0);
        cmma::execute::<F, F, f32, f32>(&a0_1, &b0_1, &c1_1, &c1_1);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_0, &c2_0, &c2_0);
        cmma::execute::<F, F, f32, f32>(&a0_2, &b0_1, &c2_1, &c2_1);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_0, &c3_0, &c3_0);
        cmma::execute::<F, F, f32, f32>(&a0_3, &b0_1, &c3_1, &c3_1);
        k4 += 4;
    }
    let mut out = SharedMemory::<f32>::new(2048usize);
    emit(&c0_0, &mut out, output, fault, sg, t0 + 0, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c0_1, &mut out, output, fault, sg, t0 + 0, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c1_0, &mut out, output, fault, sg, t0 + 8, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c1_1, &mut out, output, fault, sg, t0 + 8, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c2_0, &mut out, output, fault, sg, t0 + 16, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c2_1, &mut out, output, fault, sg, t0 + 16, r0 + 16, out_first, out_stride, count, rows, false);
    emit(&c3_0, &mut out, output, fault, sg, t0 + 24, r0 + 0, out_first, out_stride, count, rows, false);
    emit(&c3_1, &mut out, output, fault, sg, t0 + 24, r0 + 16, out_first, out_stride, count, rows, false);
}

fn vec4_arg(handle: Handle) -> Result<ArrayArg<WgpuRuntime>> {
    if handle.size_in_used() == 0
        || !handle.size_in_used().is_multiple_of(8)
        || !handle.offset_start.unwrap_or(0).is_multiple_of(8)
    {
        return Err(Error::InvalidBufferLength);
    }
    let vectors = handle.size_in_used() as usize / 8;
    // SAFETY: aligned vec4<f16/bf16> binding of the whole handle.
    Ok(unsafe { ArrayArg::from_raw_parts(handle, vectors) })
}

/// Runs the tiled GEMM when the shape suits it; `Ok(false)` means the caller
/// should use another kernel. Operand bounds were checked by `dense`.
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
    // Partial tiles are guarded; tiny outputs are better served elsewhere.
    if rows < 64
        || !rows.is_multiple_of(16)
        || !columns.is_multiple_of(KSTEP)
        || !weight_first.is_multiple_of(4)
        || BYPASS.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os("KUNO_CLAUDE_GEMM_OFF").is_some()
    {
        return Ok(false);
    }
    let [input, weight, output, fault] = h;
    let ia = vec4_arg(input)?;
    let wa = vec4_arg(weight)?;
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    let fa = crate::resident_util::raw_arg(fault, 1)?;
    let grid = CubeCount::Static(rows.div_ceil(TILE) as u32, count.div_ceil(TILE) as u32, 1);
    let guarded = !count.is_multiple_of(TILE) || !rows.is_multiple_of(TILE);
    // SAFETY: `dense` validated every operand interval for this shape; loads and
    // stores outside count x rows are masked in the kernel.
    // Production selection: Mesa with Xe2 2D block loads (INTEL_CMAT_BLOCK2D)
    // runs the direct kernel on full tiles; everything else keeps the
    // shared-memory kernel below.
    let direct = std::env::var_os("INTEL_CMAT_BLOCK2D").is_some()
        && std::env::var_os("KUNO_CLAUDE_GEMM_DIRECT").is_none();
    if direct && count % 128 == 0 && rows % 128 == 0 && columns % 16 == 0 {
        let grid = CubeCount::Static((count / 128) as u32, (rows / 128) as u32, 1);
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct_42u1::launch_unchecked::<half::bf16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            } else {
                gemm_direct_42u1::launch_unchecked::<half::f16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            }
        }
        return Ok(true);
    }
    if std::env::var("KUNO_CLAUDE_GEMM_DIRECT").as_deref() == Ok("42u4") && count % 128 == 0 && rows % 128 == 0 && columns % 64 == 0 {
        let grid = CubeCount::Static((count / 128) as u32, (rows / 128) as u32, 1);
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct_42u4::launch_unchecked::<half::bf16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            } else {
                gemm_direct_42u4::launch_unchecked::<half::f16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            }
        }
        return Ok(true);
    }
    if std::env::var("KUNO_CLAUDE_GEMM_DIRECT").as_deref() == Ok("44u2") && count % 128 == 0 && rows % 256 == 0 && columns % 32 == 0 {
        let grid = CubeCount::Static((count / 128) as u32, (rows / 256) as u32, 1);
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct_44u2::launch_unchecked::<half::bf16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            } else {
                gemm_direct_44u2::launch_unchecked::<half::f16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            }
        }
        return Ok(true);
    }
    if std::env::var("KUNO_CLAUDE_GEMM_DIRECT").as_deref() == Ok("82u2") && count % 256 == 0 && rows % 128 == 0 && columns % 32 == 0 {
        let grid = CubeCount::Static((count / 256) as u32, (rows / 128) as u32, 1);
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct_82u2::launch_unchecked::<half::bf16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            } else {
                gemm_direct_82u2::launch_unchecked::<half::f16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            }
        }
        return Ok(true);
    }
    if std::env::var("KUNO_CLAUDE_GEMM_DIRECT").as_deref() == Ok("42u1") && count % 128 == 0 && rows % 128 == 0 && columns % 16 == 0 {
        let grid = CubeCount::Static((count / 128) as u32, (rows / 128) as u32, 1);
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct_42u1::launch_unchecked::<half::bf16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            } else {
                gemm_direct_42u1::launch_unchecked::<half::f16, WgpuRuntime>(client, grid, CubeDim::new_2d(16, 16), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns);
            }
        }
        return Ok(true);
    }
    if std::env::var("KUNO_CLAUDE_GEMM_DIRECT").as_deref() == Ok("42") && !guarded {
        // SAFETY: full tiles only; every fragment lies inside the operands.
        unsafe {
            if bfloat {
                gemm_direct::launch_unchecked::<half::bf16, WgpuRuntime>(
                    client, grid, CubeDim::new_1d(256), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns,
                );
            } else {
                gemm_direct::launch_unchecked::<half::f16, WgpuRuntime>(
                    client, grid, CubeDim::new_1d(256), ia, wa, oa, fa,
                    weight_first, out_first, out_stride, count, rows, columns,
                );
            }
        }
        return Ok(true);
    }
    if std::env::var_os("KUNO_CLAUDE_GEMM8").is_some() {
        // SAFETY: as below.
        unsafe {
            if bfloat {
                gemm_nt8::launch_unchecked::<half::bf16, WgpuRuntime>(
                    client, grid, CubeDim::new_1d(256), ia, wa, oa, fa,
                    columns, weight_first, out_first, out_stride, count, rows, guarded,
                );
            } else {
                gemm_nt8::launch_unchecked::<half::f16, WgpuRuntime>(
                    client, grid, CubeDim::new_1d(256), ia, wa, oa, fa,
                    columns, weight_first, out_first, out_stride, count, rows, guarded,
                );
            }
        }
        return Ok(true);
    }
    unsafe {
        if bfloat {
            gemm_nt::launch_unchecked::<half::bf16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride, count, rows, guarded,
            );
        } else {
            gemm_nt::launch_unchecked::<half::f16, WgpuRuntime>(
                client, grid, CubeDim::new_1d(512), ia, wa, oa, fa,
                columns, weight_first, out_first, out_stride, count, rows, guarded,
            );
        }
    }
    Ok(true)
}
