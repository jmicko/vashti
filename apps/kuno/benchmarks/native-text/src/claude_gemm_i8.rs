//! Int8 NT GEMM probe for Intel Xe2 (s8 x s8 -> s32 cooperative matrices,
//! 8 x 16 x 32). Operands are packed 4 x i8 per u32 in global memory (8-bit
//! storage buffers are not enabled) and reinterpreted into int8 shared memory.
//!
//! output[t * rows + r] = sum_k input[t, k] * weight[r, k] (i32)
//! Tile 128 tokens x 128 rows, 32 subgroups of 16 lanes each owning 32 x 16
//! (4 A x 1 B, like the f16 GEMM), K advances 64 per double-buffered stage.
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use cubecl_runtime::server::Handle;
use kuno_text::{Error, Result};

#[cube]
fn stage_i8(
    input: &Array<Vector<u32, Const<4>>>,
    weight: &Array<Vector<u32, Const<4>>>,
    a: &mut SharedMemory<Vector<i8, Const<4>>>,
    b: &mut SharedMemory<Vector<i8, Const<4>>>,
    k16: usize,
    t0: usize,
    r0: usize,
    stage16: usize,
    buf: usize,
) {
    // 128 rows x 64 bytes = 512 vec4<u32> per operand: one per lane.
    // Subgroup builtins, as the native 512-lane cooperative contract requires.
    let lane = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
    let row = lane / 4;
    let c = lane % 4;
    let va = input[(t0 + row) * k16 + stage16 + c];
    let vb = weight[(r0 + row) * k16 + stage16 + c];
    let at = buf * 2048 + row * 16 + c * 4;
    #[unroll]
    for j in 0..4usize {
        a[at + j] = Vector::<i8, Const<4>>::reinterpret(va[j]);
        b[at + j] = Vector::<i8, Const<4>>::reinterpret(vb[j]);
    }
}

#[cube(launch_unchecked)]
pub fn gemm_i8(
    input: &Array<Vector<u32, Const<4>>>,
    weight: &Array<Vector<u32, Const<4>>>,
    output: &mut Array<i32>,
    columns: usize,
    rows: usize,
) {
    let mut a = SharedMemory::<Vector<i8, Const<4>>>::new(4096usize);
    let mut b = SharedMemory::<Vector<i8, Const<4>>>::new(4096usize);
    let sg = usize::cast_from(PLANE_POS);
    let am = (sg / 8) * 32;
    let bn = (sg % 8) * 16;
    let t0 = usize::cast_from(CUBE_POS_Y) * 128;
    let r0 = usize::cast_from(CUBE_POS_X) * 128;
    let k16 = columns / 16;
    let a0 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a2 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a3 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a4 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a5 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a6 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let a7 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 32usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 32usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<i8>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 32usize, cmma::MatrixLayout::ColMajor) };
    let c0 = cmma::Matrix::<i32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 32usize, cmma::MatrixLayout::Undefined, 0i32);
    let c1 = cmma::Matrix::<i32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 32usize, cmma::MatrixLayout::Undefined, 0i32);
    let c2 = cmma::Matrix::<i32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 32usize, cmma::MatrixLayout::Undefined, 0i32);
    let c3 = cmma::Matrix::<i32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 32usize, cmma::MatrixLayout::Undefined, 0i32);
    stage_i8(input, weight, &mut a, &mut b, k16, t0, r0, 0, 0);
    sync_cube();
    let mut k = 0usize;
    let mut cur = 0usize;
    while k < columns {
        let nxt = 1usize - cur;
        if k + 64 < columns {
            stage_i8(input, weight, &mut a, &mut b, k16, t0, r0, (k + 64) / 16, nxt);
        }
        // Rows are 64 bytes = 16 vec4<i8>; the second K32 half is +8.
        let ab = cur * 2048 + am * 16;
        let bb = cur * 2048 + bn * 16;
        cmma::load(&a0, &a.to_slice().slice(ab, a.len()), 64);
        cmma::load(&a1, &a.to_slice().slice(ab + 128, a.len()), 64);
        cmma::load(&a2, &a.to_slice().slice(ab + 256, a.len()), 64);
        cmma::load(&a3, &a.to_slice().slice(ab + 384, a.len()), 64);
        cmma::load(&b0, &b.to_slice().slice(bb, b.len()), 64);
        cmma::load(&a4, &a.to_slice().slice(ab + 8, a.len()), 64);
        cmma::load(&a5, &a.to_slice().slice(ab + 136, a.len()), 64);
        cmma::load(&a6, &a.to_slice().slice(ab + 264, a.len()), 64);
        cmma::load(&a7, &a.to_slice().slice(ab + 392, a.len()), 64);
        cmma::load(&b1, &b.to_slice().slice(bb + 8, b.len()), 64);
        cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c0, &c0);
        cmma::execute::<i8, i8, i32, i32>(&a1, &b0, &c1, &c1);
        cmma::execute::<i8, i8, i32, i32>(&a2, &b0, &c2, &c2);
        cmma::execute::<i8, i8, i32, i32>(&a3, &b0, &c3, &c3);
        cmma::execute::<i8, i8, i32, i32>(&a4, &b1, &c0, &c0);
        cmma::execute::<i8, i8, i32, i32>(&a5, &b1, &c1, &c1);
        cmma::execute::<i8, i8, i32, i32>(&a6, &b1, &c2, &c2);
        cmma::execute::<i8, i8, i32, i32>(&a7, &b1, &c3, &c3);
        sync_cube();
        cur = nxt;
        k += 64;
    }
    let t = t0 + am;
    let r = r0 + bn;
    let first = t * rows + r;
    let stride = u32::cast_from(rows);
    cmma::store(&mut output.slice_mut(first, output.len()), &c0, stride, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(first + 8 * rows, output.len()), &c1, stride, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(first + 16 * rows, output.len()), &c2, stride, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(first + 24 * rows, output.len()), &c3, stride, cmma::MatrixLayout::RowMajor);
}

/// Full-tile shapes only (probe): count, rows multiples of 128, columns of 64.
pub(crate) fn launch(
    client: &ComputeClient<WgpuRuntime>,
    input: Handle,
    weight: Handle,
    output: Handle,
    count: usize,
    columns: usize,
    rows: usize,
) -> Result<()> {
    if !count.is_multiple_of(128) || !rows.is_multiple_of(128) || !columns.is_multiple_of(64) {
        return Err(Error::InvalidBufferLength);
    }
    let ia = unsafe { ArrayArg::from_raw_parts(input.clone(), input.size_in_used() as usize / 16) };
    let wa = unsafe { ArrayArg::from_raw_parts(weight.clone(), weight.size_in_used() as usize / 16) };
    let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
    unsafe {
        gemm_i8::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static((rows / 128) as u32, (count / 128) as u32, 1),
            CubeDim::new_1d(512),
            ia,
            wa,
            oa,
            columns,
            rows,
        );
    }
    Ok(())
}

/// Synthetic int8 GEMM timing and exactness against a CPU reference.
pub(crate) fn bench(client: &ComputeClient<WgpuRuntime>) -> std::result::Result<(), String> {
    for (label, count, k, n) in [("gu", 4096usize, 2560usize, 1280usize), ("down", 4096, 640, 2560), ("wide", 4096, 2560, 10240)] {
        let a: Vec<i8> = (0..count * k).map(|i| (((i * 2654435761) >> 7) % 255) as i32 as i8).collect();
        let w: Vec<i8> = (0..n * k).map(|i| (((i * 40503 + 7) >> 3) % 251) as i32 as i8).collect();
        let bytes = |v: &[i8]| v.iter().map(|x| *x as u8).collect::<Vec<u8>>();
        let ia = client.create_from_slice(&bytes(&a));
        let wa = client.create_from_slice(&bytes(&w));
        let out = client.empty(count * n * 4);
        launch(client, ia.clone(), wa.clone(), out.clone(), count, k, n).map_err(|e| format!("{e:?}"))?;
        cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;
        let got = client.read_one(out.clone().offset_end(out.size_in_used() - 4 * 1024 * 1024)).map_err(|e| format!("{e:?}"))?;
        let mut bad = 0usize;
        for idx in (0..1024 * 1024).step_by(997) {
            let (t, r) = (idx / n, idx % n);
            let mut s = 0i64;
            for kk in 0..k {
                s += i64::from(a[t * k + kk]) * i64::from(w[r * k + kk]);
            }
            let g = i32::from_le_bytes(got[idx * 4..idx * 4 + 4].try_into().unwrap());
            if i64::from(g) != s {
                bad += 1;
            }
        }
        let reps = 20;
        let started = std::time::Instant::now();
        for _ in 0..reps {
            launch(client, ia.clone(), wa.clone(), out.clone(), count, k, n).map_err(|e| format!("{e:?}"))?;
        }
        cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;
        let s = started.elapsed().as_secs_f64() / reps as f64;
        eprintln!(
            "public_i8_bench {label} count={count} k={k} n={n} ms={:.3} tops={:.1} mismatches={bad}",
            s * 1e3,
            2.0 * (count * k * n) as f64 / s / 1e12
        );
    }
    Ok(())
}

/// Register-only DPAS throughput: each subgroup loads its fragments once from
/// shared memory, then runs `iters` x 8 products into 4 (or 8) accumulators.
#[cube(launch_unchecked)]
pub fn dpas_peak(output: &mut Array<f32>, iters: u32, #[comptime] eight: bool) {
    let mut s = SharedMemory::<half::f16>::new(4096usize);
    let lane = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
    let mut i = lane;
    while i < 4096 {
        s[i] = half::f16::cast_from(i % 7) * half::f16::new(0.001);
        i += 256;
    }
    sync_cube();
    let a0 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let a1 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::A, 8usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor) };
    let b0 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    let b1 = unsafe { cmma::Matrix::<half::f16>::uninitialized(cmma::MatrixIdent::B, 8usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor) };
    cmma::load(&a0, &s.to_slice().slice(0, 4096), 16);
    cmma::load(&a1, &s.to_slice().slice(256, 4096), 16);
    cmma::load(&b0, &s.to_slice().slice(512, 4096), 16);
    cmma::load(&b1, &s.to_slice().slice(1024, 4096), 16);
    let c0 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c1 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c2 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c3 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c4 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c5 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c6 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let c7 = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0);
    let mut n = 0u32;
    while n < iters {
        cmma::execute::<half::f16, half::f16, f32, f32>(&a0, &b0, &c0, &c0);
        cmma::execute::<half::f16, half::f16, f32, f32>(&a1, &b0, &c1, &c1);
        cmma::execute::<half::f16, half::f16, f32, f32>(&a0, &b1, &c2, &c2);
        cmma::execute::<half::f16, half::f16, f32, f32>(&a1, &b1, &c3, &c3);
        if comptime!(eight) {
            cmma::execute::<half::f16, half::f16, f32, f32>(&a0, &b0, &c4, &c4);
            cmma::execute::<half::f16, half::f16, f32, f32>(&a1, &b0, &c5, &c5);
            cmma::execute::<half::f16, half::f16, f32, f32>(&a0, &b1, &c6, &c6);
            cmma::execute::<half::f16, half::f16, f32, f32>(&a1, &b1, &c7, &c7);
        }
        n += 1;
    }
    let sg = usize::cast_from(CUBE_POS_X) * 16 + usize::cast_from(PLANE_POS);
    let base = sg * 1024;
    cmma::store(&mut output.slice_mut(base, base + 128), &c0, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(base + 128, base + 256), &c1, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(base + 256, base + 384), &c2, 16, cmma::MatrixLayout::RowMajor);
    cmma::store(&mut output.slice_mut(base + 384, base + 512), &c3, 16, cmma::MatrixLayout::RowMajor);
    if comptime!(eight) {
        cmma::store(&mut output.slice_mut(base + 512, base + 640), &c4, 16, cmma::MatrixLayout::RowMajor);
        cmma::store(&mut output.slice_mut(base + 640, base + 768), &c5, 16, cmma::MatrixLayout::RowMajor);
        cmma::store(&mut output.slice_mut(base + 768, base + 896), &c6, 16, cmma::MatrixLayout::RowMajor);
        cmma::store(&mut output.slice_mut(base + 896, base + 1024), &c7, 16, cmma::MatrixLayout::RowMajor);
    }
}

pub(crate) fn dpas_bench(client: &ComputeClient<WgpuRuntime>) -> std::result::Result<(), String> {
    let groups = 1024u32;
    let out = client.empty(groups as usize * 16 * 1024 * 4);
    for eight in [false, true] {
        for wg in [256u32, 512] {
            let iters = 4096u32;
            let launch = || unsafe {
                dpas_peak::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(groups * 256 / wg, 1, 1),
                    CubeDim::new_1d(wg),
                    crate::resident_util::raw_arg(out.clone(), out.size_in_used() as usize / 4).unwrap(),
                    iters,
                    eight,
                );
            };
            launch();
            cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;
            let started = std::time::Instant::now();
            for _ in 0..5 {
                launch();
            }
            cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))?;
            let s = started.elapsed().as_secs_f64() / 5.0;
            // subgroups x iters x products x (8 x 16 x 16 x 2) flops
            let flops = (groups * 16) as f64 * iters as f64 * if eight { 8.0 } else { 4.0 } * 4096.0;
            eprintln!("public_dpas_peak accumulators={} wg={wg} ms={:.3} tflops={:.1}", if eight { 8 } else { 4 }, s * 1e3, flops / s / 1e12);
        }
    }
    Ok(())
}
