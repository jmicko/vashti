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
            a[buf * 1024 + row * 8 + c] = va;
            let mut vb = zero;
            if r0 + row < rows {
                vb = weight[wf4 + (r0 + row) * cols4 + k4 + c];
            }
            b[buf * 1024 + row * 8 + c] = vb;
        } else {
            a[buf * 1024 + row * 8 + c] = input[(t0 + row) * cols4 + k4 + c];
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
    emit(&c0, &mut out, output, fault, sg, t, r, out_first, out_stride, count, rows, guarded);
    emit(&c1, &mut out, output, fault, sg, t + 8, r, out_first, out_stride, count, rows, guarded);
    emit(&c2, &mut out, output, fault, sg, t + 16, r, out_first, out_stride, count, rows, guarded);
    emit(&c3, &mut out, output, fault, sg, t + 24, r, out_first, out_stride, count, rows, guarded);
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
