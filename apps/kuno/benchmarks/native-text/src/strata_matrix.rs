//! Native Intel matrix instructions for the pinned Strata rounded prompt path.
//! Data stays in the already admitted BF16/FP16 images and F32 result owners.
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{Error, Result};

#[cfg(feature = "strata-cubek-gemm")]
#[path = "strata_cubek_gemm.rs"]
mod cubek;

#[cfg(feature = "strata-native-wide")]
#[path = "strata_matrix_native_wide.rs"]
mod native_wide;

#[cfg(feature = "strata-intel-core")]
#[path = "strata_matrix_numeric_probe.rs"]
mod numeric;
#[cfg(feature = "strata-intel-core")]
pub(crate) use numeric::cases as public_numeric;

#[cfg(feature = "strata-cubek-gemm")]
pub(crate) fn public_expert_numeric(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut crate::private_arena::PrivateArena,
) -> std::result::Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
    cubek::public_expert_numeric(client, arena)
}

#[cfg(feature = "strata-intel-core")]
#[path = "strata_matrix_timing_probe.rs"]
mod timing;
#[cfg(feature = "strata-intel-core")]
pub(crate) use timing::cases as public_timing;

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
        // Complete tiles are already contiguous, aligned FP16/BF16 images.
        // Cooperative loads can consume these original global owners directly;
        // bouncing them through SLM adds loads, stores and barriers at every K16.
        // Keep the original masked staging path for any partial tile/reduction.
        if token + 8 <= count && row + 16 <= rows && columns.is_multiple_of(16) {
            let mut k = 0usize;
            while k < columns {
                let first_a = token * columns + k;
                let first_b = weight_first + row * columns + k;
                cmma::load(
                    &am,
                    &input.to_slice().slice(first_a, input.len()),
                    u32::cast_from(columns),
                );
                cmma::load(
                    &bm,
                    &weight.to_slice().slice(first_b, weight.len()),
                    u32::cast_from(columns),
                );
                cmma::execute::<F, F, f32, f32>(&am, &bm, &acc, &acc);
                k += 16;
            }
        } else {
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
    // Two K64 operand buffers: A[2][32][64] and B[2][64][64].
    // Half/BF16 vectors preserve their original bits; no F32 round trip.
    #[cube]
    #[allow(clippy::too_many_arguments)]
    fn macro_stage<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        a: &mut SharedMemory<Vector<F, Const<4>>>,
        b: &mut SharedMemory<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        token_base: usize,
        row_base: usize,
        k: usize,
        buffer: usize,
    ) {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + 32 <= count && k + 64 <= columns {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (token * columns + col) / 4;
                a[buffer * 512 + job * 2] = input[source];
                a[buffer * 512 + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..4usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * 512 + job * 2] = lo;
                a[buffer * 512 + job * 2 + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 64 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[buffer * 1024 + job * 2] = weight[source];
                b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if row < rows && col < columns {
                    let source = (weight_first + row * columns + col) / 4;
                    lo = weight[source];
                    hi = weight[source + 1];
                }
                b[buffer * 1024 + job * 2] = lo;
                b[buffer * 1024 + job * 2 + 1] = hi;
            }
        }
    }

    /// Four subgroups, each owning four8x16 results. Two A and two B
    /// fragments feed four accumulators, doubling operand reuse without
    /// changing the ordered K16 reduction of any output. The complete
    /// workgroup covers32 input rows by64 output features.
    #[cube]
    fn macro_product<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
    ) -> SharedMemory<f32> {
        let mut a = SharedMemory::<Vector<F, Const<4>>>::new(1024usize);
        let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
        // SAFETY: both operands are loaded before the corresponding products.
        let a0 = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        // SAFETY: both operands are loaded before the corresponding products.
        let a1 = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        // SAFETY: both operands are loaded before the corresponding products.
        let b0 = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::B,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::ColMajor,
            )
        };
        // SAFETY: both operands are loaded before the corresponding products.
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
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let token_base = usize::cast_from(CUBE_POS_Y) * 32;
        let row_base = usize::cast_from(CUBE_POS_X) * 64;
        // Buffer0 is fully initialized, including every masked tail, before
        // any subgroup reads it. Each lane copies eight consecutive half
        // values as two genuine vec4 loads/stores without float conversion.
        macro_stage::<F>(
            input,
            weight,
            &mut a,
            &mut b,
            columns,
            rows,
            count,
            weight_first,
            token_base,
            row_base,
            0,
            0,
        );
        sync_cube();
        let mut k = 0usize;
        let mut current = 0usize;
        while k < columns {
            let next = 1usize - current;
            // Software prefetch: independent next-stage global loads/stores
            // are issued before current-stage MMA. This expresses overlap
            // opportunity, not a guaranteed asynchronous DMA operation.
            if k + 64 < columns {
                macro_stage::<F>(
                    input,
                    weight,
                    &mut a,
                    &mut b,
                    columns,
                    rows,
                    count,
                    weight_first,
                    token_base,
                    row_base,
                    k + 64,
                    next,
                );
            }
            // Expose all four ordered K16 groups together. This keeps the
            // same four accumulators and exact reduction order, but avoids
            // a runtime loop around the native matrix instructions.
            #[unroll]
            for step in 0..4usize {
                let inner = step * 16;
                if k + inner < columns {
                    let af = current * 512 + (subgroup / 2) * 256 + inner / 4;
                    let bf = current * 1024 + (subgroup % 2) * 512 + inner / 4;
                    // Offsets index real vec4 storage; cmma's stride remains
                    // the64 scalar elements in each staged row. The compiler
                    // divides by the authentic vector width for the pointer.
                    cmma::load(&a0, &a.to_slice().slice(af, a.len()), 64);
                    cmma::load(&a1, &a.to_slice().slice(af + 128, a.len()), 64);
                    cmma::load(&b0, &b.to_slice().slice(bf, b.len()), 64);
                    cmma::load(&b1, &b.to_slice().slice(bf + 256, b.len()), 64);
                    cmma::execute::<F, F, f32, f32>(&a0, &b0, &c0, &c0);
                    cmma::execute::<F, F, f32, f32>(&a0, &b1, &c1, &c1);
                    cmma::execute::<F, F, f32, f32>(&a1, &b0, &c2, &c2);
                    cmma::execute::<F, F, f32, f32>(&a1, &b1, &c3, &c3);
                }
            }
            // All old readers finish before either stage is reused. Every
            // next-stage write is visible before it becomes current. Loop
            // conditions and the barrier are uniform across the workgroup.
            sync_cube();
            current = next;
            k += 64;
        }
        // Every lane has completed the final K64 handoff barrier, and no
        // operand slice is used beyond this uniform loop exit. Matrix fragments
        // retain their values in registers. End the authentic SLM lifetimes
        // before allocating output; the compiler may reuse physical offsets
        // only on devices supporting explicit workgroup memory layout.
        // SAFETY: uniform control flow, no surviving operand-pointer uses.
        unsafe {
            a.free();
            b.free();
        }
        let mut out = SharedMemory::<f32>::new(2048usize);
        cmma::store(
            &mut out.slice_mut(subgroup * 512, subgroup * 512 + 128),
            &c0,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            &mut out.slice_mut(subgroup * 512 + 128, subgroup * 512 + 256),
            &c1,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            &mut out.slice_mut(subgroup * 512 + 256, subgroup * 512 + 384),
            &c2,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        cmma::store(
            &mut out.slice_mut(subgroup * 512 + 384, subgroup * 512 + 512),
            &c3,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        out
    }
    #[cube(launch_unchecked)]
    pub fn dense_macro<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        output: &mut Array<f32>,
        fault: &mut Array<Atomic<u32>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        out_first: usize,
        out_stride: usize,
    ) {
        let values = macro_product::<F>(input, weight, columns, rows, count, weight_first);
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let mut i = usize::cast_from(UNIT_POS) % 16;
        while i < 512 {
            let t = usize::cast_from(CUBE_POS_Y) * 32
                + (subgroup / 2) * 16
                + (i / 256) * 8
                + (i % 128) / 16;
            let row = usize::cast_from(CUBE_POS_X) * 64
                + (subgroup % 2) * 32
                + ((i / 128) % 2) * 16
                + i % 16;
            if t < count && row < rows {
                let v = values[subgroup * 512 + i];
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
    pub fn expert_macro(
        input: &Array<Vector<f16, Const<4>>>,
        weight: &Array<Vector<f16, Const<4>>>,
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
        let values = macro_product::<f16>(input, weight, columns, rows, count, weight_first);
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let mut i = usize::cast_from(UNIT_POS) % 16;
        while i < 512 {
            let member = usize::cast_from(CUBE_POS_Y) * 32
                + (subgroup / 2) * 16
                + (i / 256) * 8
                + (i % 128) / 16;
            let row = usize::cast_from(CUBE_POS_X) * 64
                + (subgroup % 2) * 32
                + ((i / 128) % 2) * 16
                + i % 16;
            if member < count && row < rows {
                let route =
                    usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let mut base = member * 1280;
                    if down {
                        base = route * 2560;
                    }
                    let v = values[subgroup * 512 + i];
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
    #[cube]
    #[allow(clippy::too_many_arguments)]
    fn narrow_stage<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        a: &mut SharedMemory<Vector<F, Const<4>>>,
        b: &mut SharedMemory<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        token_base: usize,
        row_base: usize,
        k: usize,
        buffer: usize,
        #[comptime] tile: usize,
    ) {
        // A full stage is uniform for the workgroup. Resolve that condition
        // once, then expose the fixed per-lane jobs to the compiler. Publish
        // each pair immediately; no register-first operand array stays live
        // across the matrix products.
        if token_base + tile <= count && k + 64 <= columns {
            #[unroll]
            for round in 0..comptime!(tile / 8) {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (token * columns + col) / 4;
                a[buffer * comptime!(tile * 16) + job * 2] = input[source];
                a[buffer * comptime!(tile * 16) + job * 2 + 1] = input[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..comptime!(tile / 8) {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let token = token_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if token < count && col < columns {
                    // Genuine vector bindings and multiple-of16 reductions
                    // guarantee both whole vectors are within the owner.
                    let source = (token * columns + col) / 4;
                    lo = input[source];
                    hi = input[source + 1];
                }
                a[buffer * comptime!(tile * 16) + job * 2] = lo;
                a[buffer * comptime!(tile * 16) + job * 2 + 1] = hi;
            }
        }
        if row_base + 64 <= rows && k + 64 <= columns {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let source = (weight_first + row * columns + col) / 4;
                b[buffer * 1024 + job * 2] = weight[source];
                b[buffer * 1024 + job * 2 + 1] = weight[source + 1];
            }
        } else {
            #[unroll]
            for round in 0..8usize {
                let job = usize::cast_from(UNIT_POS) + round * 64;
                let row = row_base + job / 8;
                let col = k + (job % 8) * 8;
                let mut lo = Vector::<F, Const<4>>::new(F::new(0.0));
                let mut hi = Vector::<F, Const<4>>::new(F::new(0.0));
                if row < rows && col < columns {
                    let source = (weight_first + row * columns + col) / 4;
                    lo = weight[source];
                    hi = weight[source + 1];
                }
                b[buffer * 1024 + job * 2] = lo;
                b[buffer * 1024 + job * 2 + 1] = hi;
            }
        }
    }

    /// Expert groups below32 tokens use8/16-row tiles and share vector-staged
    /// A across four feature subgroups. They never execute
    /// the four-accumulator M32 product for a one-row group.
    #[cube]
    fn narrow_product<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        #[comptime] tile: usize,
    ) -> SharedMemory<f32> {
        let mut a = SharedMemory::<Vector<F, Const<4>>>::new(comptime!(tile * 32));
        let mut b = SharedMemory::<Vector<F, Const<4>>>::new(2048usize);
        // SAFETY: the complete current operand stage is initialized before
        // every product, including zero-masked token/feature/reduction tails.
        let a0 = unsafe {
            cmma::Matrix::<F>::uninitialized(
                cmma::MatrixIdent::A,
                8usize,
                16usize,
                16usize,
                cmma::MatrixLayout::RowMajor,
            )
        };
        let a1 = if comptime!(tile == 16) {
            // SAFETY: only the16-row variant loads and uses this fragment.
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
        let bm = unsafe {
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
        let c1 = if comptime!(tile == 16) {
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
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let token_base = usize::cast_from(CUBE_POS_Y) * tile;
        let row_base = usize::cast_from(CUBE_POS_X) * 64;
        narrow_stage::<F>(
            input,
            weight,
            &mut a,
            &mut b,
            columns,
            rows,
            count,
            weight_first,
            token_base,
            row_base,
            0,
            0,
            tile,
        );
        sync_cube();
        let mut k = 0usize;
        let mut current = 0usize;
        while k < columns {
            let next = 1usize - current;
            if k + 64 < columns {
                narrow_stage::<F>(
                    input,
                    weight,
                    &mut a,
                    &mut b,
                    columns,
                    rows,
                    count,
                    weight_first,
                    token_base,
                    row_base,
                    k + 64,
                    next,
                    tile,
                );
            }
            #[unroll]
            for step in 0..4usize {
                let inner = step * 16;
                if k + inner < columns {
                    let af = current * comptime!(tile * 16) + inner / 4;
                    let bf = current * 1024 + subgroup * 256 + inner / 4;
                    cmma::load(&a0, &a.to_slice().slice(af, a.len()), 64);
                    cmma::load(&bm, &b.to_slice().slice(bf, b.len()), 64);
                    cmma::execute::<F, F, f32, f32>(&a0, &bm, &c0, &c0);
                    if comptime!(tile == 16) {
                        cmma::load(&a1, &a.to_slice().slice(af + 128, a.len()), 64);
                        cmma::execute::<F, F, f32, f32>(&a1, &bm, &c1, &c1);
                    }
                }
            }
            // A single uniform handoff perK64 replaces the scalar partial
            // path's two workgroup barriers perK16. Old readers finish before
            // stage reuse; every next-stage store is visible before the switch.
            sync_cube();
            current = next;
            k += 64;
        }
        // Every lane has completed the final K64 handoff barrier, and no
        // operand slice is used beyond this uniform loop exit. Matrix fragments
        // retain their values in registers. End the authentic SLM lifetimes
        // before allocating output; the compiler may reuse physical offsets
        // only on devices supporting explicit workgroup memory layout.
        // SAFETY: uniform control flow, no surviving operand-pointer uses.
        unsafe {
            a.free();
            b.free();
        }
        let mut out = SharedMemory::<f32>::new(comptime!(tile * 64));
        let first = subgroup * comptime!(tile * 16);
        cmma::store(
            &mut out.slice_mut(first, first + 128),
            &c0,
            16,
            cmma::MatrixLayout::RowMajor,
        );
        if comptime!(tile == 16) {
            cmma::store(
                &mut out.slice_mut(first + 128, first + 256),
                &c1,
                16,
                cmma::MatrixLayout::RowMajor,
            );
        }
        sync_cube();
        out
    }
    #[cube(launch_unchecked)]
    pub fn dense_narrow<F: Float>(
        input: &Array<Vector<F, Const<4>>>,
        weight: &Array<Vector<F, Const<4>>>,
        output: &mut Array<f32>,
        fault: &mut Array<Atomic<u32>>,
        columns: usize,
        rows: usize,
        count: usize,
        weight_first: usize,
        out_first: usize,
        out_stride: usize,
        #[comptime] tile: usize,
    ) {
        let values = narrow_product::<F>(input, weight, columns, rows, count, weight_first, tile);
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let mut i = usize::cast_from(UNIT_POS) % 16;
        while i < comptime!(tile * 16) {
            let t = usize::cast_from(CUBE_POS_Y) * tile + i / 16;
            let row = usize::cast_from(CUBE_POS_X) * 64 + subgroup * 16 + i % 16;
            if t < count && row < rows {
                let v = values[subgroup * comptime!(tile * 16) + i];
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
    pub fn expert_narrow(
        input: &Array<Vector<f16, Const<4>>>,
        weight: &Array<Vector<f16, Const<4>>>,
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
        #[comptime] tile: usize,
    ) {
        let values = narrow_product::<f16>(input, weight, columns, rows, count, weight_first, tile);
        let subgroup = usize::cast_from(UNIT_POS) / 16;
        let mut i = usize::cast_from(UNIT_POS) % 16;
        while i < comptime!(tile * 16) {
            let member = usize::cast_from(CUBE_POS_Y) * tile + i / 16;
            let row = usize::cast_from(CUBE_POS_X) * 64 + subgroup * 16 + i % 16;
            if member < count && row < rows {
                let route =
                    usize::cast_from(map[1540 + usize::cast_from(map[1028 + group]) + member]);
                if route < usize::cast_from(map[0]) * 10 {
                    let mut base = member * 1280;
                    if down {
                        base = route * 2560;
                    }
                    let v = values[subgroup * comptime!(tile * 16) + i];
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
            ("dense_macro_fp16", definition_dense_macro::<f16>()),
            ("dense_macro_bf16", definition_dense_macro::<bf16>()),
            ("expert_macro_gu", definition_expert_macro(false)),
            ("expert_macro_down", definition_expert_macro(true)),
            ("dense_narrow8_fp16", definition_dense_narrow::<f16>(8)),
            ("dense_narrow8_bf16", definition_dense_narrow::<bf16>(8)),
            ("expert_narrow8_gu", definition_expert_narrow(false, 8)),
            ("expert_narrow8_down", definition_expert_narrow(true, 8)),
            ("dense_narrow16_fp16", definition_dense_narrow::<f16>(16)),
            ("dense_narrow16_bf16", definition_dense_narrow::<bf16>(16)),
            ("expert_narrow16_gu", definition_expert_narrow(false, 16)),
            ("expert_narrow16_down", definition_expert_narrow(true, 16)),
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
    fn definition_dense_macro<F: Float>() -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = Vector::<F, Const<4>>::as_type(&b.scope);
        let input = b.input_array(t);
        let weight = b.input_array(t);
        let output = b.output_array(f32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let args: [cubecl::ir::ManagedVariable; 6] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        dense_macro::expand::<F>(
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
        b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
    }
    fn definition_expert_macro(down: bool) -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = Vector::<f16, Const<4>>::as_type(&b.scope);
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
        expert_macro::expand(
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
        b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
    }
    fn definition_dense_narrow<F: Float>(tile: usize) -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = Vector::<F, Const<4>>::as_type(&b.scope);
        let input = b.input_array(t);
        let weight = b.input_array(t);
        let output = b.output_array(f32::as_type(&b.scope));
        let fault = b.output_array(
            cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
        );
        let args: [cubecl::ir::ManagedVariable; 6] =
            std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
        dense_narrow::expand::<F>(
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
            tile,
        );
        b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
    }
    fn definition_expert_narrow(down: bool, tile: usize) -> KernelDefinition {
        let mut b = KernelBuilder::default();
        AddressType::U32.register(&mut b.scope);
        let t = Vector::<f16, Const<4>>::as_type(&b.scope);
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
        expert_narrow::expand(
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
            tile,
        );
        b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(64)))
    }
    fn half_arg(handle: Handle) -> Result<ArrayArg<WgpuRuntime>> {
        if handle.size_in_used() == 0 || !handle.size_in_used().is_multiple_of(2) {
            return Err(Error::InvalidBufferLength);
        }
        let words = handle.size_in_used() as usize / 2;
        // SAFETY: typed half elements, not the u32 elements of raw_arg.
        Ok(unsafe { ArrayArg::from_raw_parts(handle, words) })
    }
    const MACRO_OPERAND_BYTES: usize = 2 * (32 + 64) * 64 * 2;
    const MACRO_SHARED_BYTES: usize = MACRO_OPERAND_BYTES + 2048 * 4;
    #[allow(clippy::too_many_arguments)]
    fn macro_geometry(
        count: usize,
        rows: usize,
        columns: usize,
        weight_first: usize,
        shared_bytes: usize,
        bindings: [(u64, u64); 2],
    ) -> bool {
        count >= 32
            && rows >= 32
            && columns >= 16
            && columns.is_multiple_of(16)
            && weight_first.is_multiple_of(4)
            && shared_bytes >= MACRO_SHARED_BYTES
            && bindings.iter().all(|&(bytes, offset)| {
                bytes != 0 && bytes.is_multiple_of(8) && offset.is_multiple_of(8)
            })
    }
    fn macro_compatible(
        client: &ComputeClient<WgpuRuntime>,
        input: &Handle,
        weight: &Handle,
        count: usize,
        rows: usize,
        columns: usize,
        weight_first: usize,
    ) -> bool {
        macro_geometry(
            count,
            rows,
            columns,
            weight_first,
            client.properties().hardware.max_shared_memory_size,
            [input, weight].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0))),
        )
    }
    const NARROW8_SHARED_BYTES: usize = 2 * (8 + 64) * 64 * 2 + 8 * 64 * 4;
    const NARROW16_SHARED_BYTES: usize = 2 * (16 + 64) * 64 * 2 + 16 * 64 * 4;
    pub(super) fn compiled_shared_bytes(name: &str, explicit: bool) -> Option<usize> {
        let (full, operands) = if name.contains("narrow8") {
            (NARROW8_SHARED_BYTES, NARROW8_SHARED_BYTES - 8 * 64 * 4)
        } else if name.contains("narrow16") {
            (NARROW16_SHARED_BYTES, NARROW16_SHARED_BYTES - 16 * 64 * 4)
        } else if name.contains("macro") {
            (MACRO_SHARED_BYTES, MACRO_OPERAND_BYTES)
        } else {
            return None;
        };
        Some(if explicit { operands } else { full })
    }
    #[allow(clippy::too_many_arguments)]
    fn narrow_geometry(
        count: usize,
        rows: usize,
        columns: usize,
        weight_first: usize,
        shared_bytes: usize,
        bindings: [(u64, u64); 2],
    ) -> Option<usize> {
        let tile = match count {
            1..=8 => 8,
            9..=31 => 16,
            _ => return None,
        };
        let bytes = if tile == 8 {
            NARROW8_SHARED_BYTES
        } else {
            NARROW16_SHARED_BYTES
        };
        (rows >= 32
            && columns >= 16
            && columns.is_multiple_of(16)
            && weight_first.is_multiple_of(4)
            && shared_bytes >= bytes
            && bindings.iter().all(|&(bytes, offset)| {
                bytes != 0 && bytes.is_multiple_of(8) && offset.is_multiple_of(8)
            }))
        .then_some(tile)
    }
    fn narrow_compatible(
        client: &ComputeClient<WgpuRuntime>,
        input: &Handle,
        weight: &Handle,
        count: usize,
        rows: usize,
        columns: usize,
        weight_first: usize,
    ) -> Option<usize> {
        narrow_geometry(
            count,
            rows,
            columns,
            weight_first,
            client.properties().hardware.max_shared_memory_size,
            [input, weight].map(|h| (h.size_in_used(), h.offset_start.unwrap_or(0))),
        )
    }
    fn macro_half_arg(handle: Handle) -> Result<ArrayArg<WgpuRuntime>> {
        if handle.size_in_used() == 0
            || !handle.size_in_used().is_multiple_of(8)
            || !handle.offset_start.unwrap_or(0).is_multiple_of(8)
        {
            return Err(Error::InvalidBufferLength);
        }
        let vectors = handle.size_in_used() as usize / 8;
        // SAFETY: authentic aligned vec4<F16/BF16> binding, eight bytes per
        // element. No scalar length or forged vector pointer is supplied.
        Ok(unsafe { ArrayArg::from_raw_parts(handle, vectors) })
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
        #[cfg(feature = "strata-native-wide")]
        if super::native_wide::try_dense(
            client,
            h.clone(),
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
            bfloat,
        )? {
            return Ok(());
        }
        #[cfg(feature = "strata-cubek-gemm")]
        if super::cubek::try_dense(
            client,
            h.clone(),
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
            bfloat,
        )? {
            return Ok(());
        }
        let [input, weight, output, fault] = h;
        let use_macro =
            macro_compatible(client, &input, &weight, count, rows, columns, weight_first);
        let narrow = narrow_compatible(client, &input, &weight, count, rows, columns, weight_first);
        let ia = if use_macro || narrow.is_some() {
            macro_half_arg(input)?
        } else {
            half_arg(input)?
        };
        let wa = if use_macro || narrow.is_some() {
            macro_half_arg(weight)?
        } else {
            half_arg(weight)?
        };
        let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
        let fa = crate::resident_util::raw_arg(fault, 1)?;
        // SAFETY: caller checked original owner intervals; padded tile operands
        // are synthesized in workgroup memory and stores mask inactive rows.
        unsafe {
            if bfloat {
                if use_macro {
                    dense_macro::launch_unchecked::<bf16, WgpuRuntime>(
                        client,
                        CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(32) as u32, 1),
                        CubeDim::new_1d(64),
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
                } else if let Some(tile) = narrow {
                    dense_narrow::launch_unchecked::<bf16, WgpuRuntime>(
                        client,
                        CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(tile) as u32, 1),
                        CubeDim::new_1d(64),
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
                        tile,
                    );
                } else {
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
                }
            } else {
                if use_macro {
                    dense_macro::launch_unchecked::<f16, WgpuRuntime>(
                        client,
                        CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(32) as u32, 1),
                        CubeDim::new_1d(64),
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
                } else if let Some(tile) = narrow {
                    dense_narrow::launch_unchecked::<f16, WgpuRuntime>(
                        client,
                        CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(tile) as u32, 1),
                        CubeDim::new_1d(64),
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
                        tile,
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
        #[cfg(feature = "strata-cubek-gemm")]
        if super::cubek::try_expert(
            client,
            h.clone(),
            count,
            columns,
            rows,
            weight_first,
            group,
            down,
        )? {
            return Ok(());
        }
        let [input, weight, output, map, status, fault] = h;
        let use_macro =
            macro_compatible(client, &input, &weight, count, rows, columns, weight_first);
        let narrow = narrow_compatible(client, &input, &weight, count, rows, columns, weight_first);
        let ia = if use_macro || narrow.is_some() {
            macro_half_arg(input)?
        } else {
            half_arg(input)?
        };
        let wa = if use_macro || narrow.is_some() {
            macro_half_arg(weight)?
        } else {
            half_arg(weight)?
        };
        let oa = crate::resident_util::raw_arg(output.clone(), output.size_in_used() as usize / 4)?;
        let ma = crate::resident_util::raw_arg(map.clone(), map.size_in_used() as usize / 4)?;
        let sa = crate::resident_util::raw_arg(status.clone(), status.size_in_used() as usize / 4)?;
        let fa = crate::resident_util::raw_arg(fault, 1)?;
        // SAFETY: the actual prompt grouping/row map and all physical packed
        // images were checked by the original-source expert caller.
        unsafe {
            if use_macro {
                expert_macro::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(32) as u32, 1),
                    CubeDim::new_1d(64),
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
            } else if let Some(tile) = narrow {
                expert_narrow::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(rows.div_ceil(64) as u32, count.div_ceil(tile) as u32, 1),
                    CubeDim::new_1d(64),
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
                    tile,
                );
            } else {
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
        }
        Ok(())
    }
    #[cfg(test)]
    mod stage_tests {
        use super::*;
        #[test]
        fn narrow_selection_requires_original_vector_owners_and_actual_shape_shared_limit() {
            let actual = [(80 * 1024 * 1024, 0), (80 * 1024 * 1024, 256)];
            assert_eq!(NARROW8_SHARED_BYTES, 20480);
            assert_eq!(NARROW16_SHARED_BYTES, 24576);
            for count in 1..=31 {
                let tile = if count <= 8 { 8 } else { 16 };
                let bytes = if tile == 8 {
                    NARROW8_SHARED_BYTES
                } else {
                    NARROW16_SHARED_BYTES
                };
                assert_eq!(narrow_geometry(count, 33, 16, 4, bytes, actual), Some(tile));
                assert_eq!(narrow_geometry(count, 33, 16, 4, bytes - 1, actual), None);
                for (rows, columns, first) in [(31, 16, 4), (33, 15, 4), (33, 17, 4), (33, 16, 1)] {
                    assert_eq!(
                        narrow_geometry(count, rows, columns, first, bytes, actual),
                        None
                    );
                }
                for bad in [(0, 0), (6, 0), (8, 2)] {
                    for index in 0..2 {
                        let mut b = actual;
                        b[index] = bad;
                        assert_eq!(narrow_geometry(count, 33, 16, 4, bytes, b), None);
                    }
                }
            }
            for count in [0, 32, 64] {
                assert_eq!(narrow_geometry(count, 33, 16, 4, 32768, actual), None);
            }
            assert!(macro_geometry(32, 33, 16, 4, 32768, actual));
        }

        #[test]
        fn narrow_epilogue_covers_each_live_output_once_without_m32_work() {
            for count in [1usize, 3, 7, 8, 9, 15, 16, 17, 31] {
                let tile = if count <= 8 { 8 } else { 16 };
                assert!(tile <= 16);
                if count <= 8 {
                    assert_eq!(tile, 8);
                }
                for rows in [32usize, 33, 65, 1280, 2560] {
                    let mut visits = vec![0u8; count * rows];
                    for gy in 0..count.div_ceil(tile) {
                        for gx in 0..rows.div_ceil(64) {
                            for sg in 0..4 {
                                for lane in 0..16 {
                                    for i in (lane..tile * 16).step_by(16) {
                                        let t = gy * tile + i / 16;
                                        let row = gx * 64 + sg * 16 + i % 16;
                                        if t < count && row < rows {
                                            visits[t * rows + row] += 1;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    assert!(visits.iter().all(|&n| n == 1), "count{count} rows{rows}");
                }
            }
        }

        #[test]
        fn narrow_vector_stages_preserve_every_operand_tail_and_ordered_k16() {
            for count in [1usize, 3, 7, 8, 9, 16, 17, 31] {
                let tile = if count <= 8 { 8 } else { 16 };
                let a_span = tile * 16;
                for rows in [32usize, 33, 65] {
                    for columns in [16usize, 64, 80, 640, 2560] {
                        let weight_first = 4;
                        for token_base in (0..count).step_by(tile) {
                            for row_base in (0..rows).step_by(64) {
                                let mut a = vec![None; 2 * a_span];
                                let mut b = vec![None; 2048];
                                let fill =
                                    |k: usize,
                                     buffer: usize,
                                     a: &mut [Option<usize>],
                                     b: &mut [Option<usize>]| {
                                        let mut av = vec![0u8; a_span];
                                        let mut bv = vec![0u8; 1024];
                                        let full_a =
                                            token_base + tile <= count && k + 64 <= columns;
                                        let full_b = row_base + 64 <= rows && k + 64 <= columns;
                                        for lane in 0..64 {
                                            for round in 0..tile / 8 {
                                                let job = lane + round * 64;
                                                let t = token_base + job / 8;
                                                let col = k + job % 8 * 8;
                                                for half in 0..2 {
                                                    let at = job * 2 + half;
                                                    av[at] += 1;
                                                    a[buffer * a_span + at] = if full_a
                                                        || (t < count && col < columns)
                                                    {
                                                        assert!(t < count && col + 7 < columns);
                                                        let source = (t * columns + col) / 4 + half;
                                                        assert!(source * 4 + 3 < count * columns);
                                                        Some(source)
                                                    } else {
                                                        None
                                                    };
                                                }
                                            }
                                            for round in 0..8 {
                                                let job = lane + round * 64;
                                                let row = row_base + job / 8;
                                                let col = k + job % 8 * 8;
                                                for half in 0..2 {
                                                    let at = job * 2 + half;
                                                    bv[at] += 1;
                                                    b[buffer * 1024 + at] = if full_b
                                                        || (row < rows && col < columns)
                                                    {
                                                        assert!(row < rows && col + 7 < columns);
                                                        let source =
                                                            (weight_first + row * columns + col)
                                                                / 4
                                                                + half;
                                                        assert!(
                                                            source * 4 + 3
                                                                < weight_first + rows * columns
                                                        );
                                                        Some(source)
                                                    } else {
                                                        None
                                                    };
                                                }
                                            }
                                        }
                                        assert!(av.into_iter().chain(bv).all(|n| n == 1));
                                    };
                                fill(0, 0, &mut a, &mut b);
                                let mut current = 0;
                                let mut order = Vec::new();
                                for k in (0..columns).step_by(64) {
                                    let next = 1 - current;
                                    if k + 64 < columns {
                                        fill(k + 64, next, &mut a, &mut b);
                                    }
                                    for inner in (0..64).step_by(16).filter(|i| k + i < columns) {
                                        order.push(k + inner);
                                        for sg in 0..4 {
                                            for tr in 0..tile {
                                                for col in 0..16 {
                                                    let t = token_base + tr;
                                                    let expected = if t < count {
                                                        Some((t * columns + k + inner + col) / 4)
                                                    } else {
                                                        None
                                                    };
                                                    let at = current * a_span
                                                        + tr * 16
                                                        + (inner + col) / 4;
                                                    assert_eq!(a[at], expected);
                                                }
                                            }
                                            for feature in 0..16 {
                                                for col in 0..16 {
                                                    let row = row_base + sg * 16 + feature;
                                                    let expected = if row < rows {
                                                        Some(
                                                            (weight_first
                                                                + row * columns
                                                                + k
                                                                + inner
                                                                + col)
                                                                / 4,
                                                        )
                                                    } else {
                                                        None
                                                    };
                                                    let at = current * 1024
                                                        + (sg * 16 + feature) * 16
                                                        + (inner + col) / 4;
                                                    assert_eq!(b[at], expected);
                                                }
                                            }
                                        }
                                    }
                                    current = next;
                                }
                                assert_eq!(order, (0..columns).step_by(16).collect::<Vec<_>>());
                            }
                        }
                    }
                }
            }
        }

        #[test]
        fn vector_binding_geometry_prices_actual_storage_and_keeps_small_fallback() {
            let actual = [(80 * 1024 * 1024, 0), (80 * 1024 * 1024, 256)];
            assert!(macro_geometry(32, 32, 16, 4, 32768, actual));
            for (count, rows, columns, first, shared) in [
                (31, 32, 16, 4, 32768),
                (32, 31, 16, 4, 32768),
                (32, 32, 15, 4, 32768),
                (32, 32, 16, 1, 32768),
                (32, 32, 16, 4, 32767),
            ] {
                assert!(!macro_geometry(count, rows, columns, first, shared, actual));
            }
            for bad in [(0, 0), (6, 0), (8, 2)] {
                for index in 0..2 {
                    let mut b = actual;
                    b[index] = bad;
                    assert!(!macro_geometry(32, 32, 16, 4, 32768, b));
                }
            }
            assert_eq!(MACRO_OPERAND_BYTES, 24576);
            assert_eq!(MACRO_SHARED_BYTES, 32768);
        }
        fn assert_actual_shared_alias_and_fenced_lifetimes(
            words: &[u32],
            expected: usize,
            explicit: bool,
        ) {
            use std::collections::{BTreeMap, BTreeSet};
            let mut instructions = Vec::new();
            let mut at = 5;
            while at < words.len() {
                let n = (words[at] >> 16) as usize;
                assert!(n > 0 && at + n <= words.len());
                instructions.push((words[at] & 65535, words[at + 1..at + n].to_vec()));
                at += n;
            }
            let mut types = BTreeMap::new();
            let mut pointers = BTreeMap::new();
            let mut constants = BTreeMap::new();
            let mut shared = Vec::new();
            let mut offsets = BTreeMap::new();
            let mut aliased = BTreeSet::new();
            let mut labels = BTreeMap::new();
            for (i, (op, a)) in instructions.iter().enumerate() {
                match op {
                    21 | 22 | 23 | 28 | 30 => {
                        types.insert(a[0], (*op, a[1..].to_vec()));
                    }
                    32 => {
                        pointers.insert(a[0], a[2]);
                    }
                    43 => {
                        constants.insert(a[1], a[2] as usize);
                    }
                    59 if a[2] == 4 => shared.push((a[1], a[0])),
                    71 if a.len() == 2 && a[1] == 20 => {
                        aliased.insert(a[0]);
                    }
                    72 if a.len() == 4 && a[1] == 0 && a[2] == 35 => {
                        offsets.insert(a[0], a[3] as usize);
                    }
                    248 => {
                        labels.insert(a[0], i);
                    }
                    _ => {}
                }
            }
            assert_eq!(shared.len(), 3);
            let mut ranges = Vec::new();
            let mut output = None;
            for &(variable, pointer) in &shared {
                let root = pointers[&pointer];
                let (array, offset) = if explicit {
                    assert_eq!(types[&root].0, 30);
                    assert!(aliased.contains(&variable));
                    (types[&root].1[0], offsets[&root])
                } else {
                    (root, 0)
                };
                assert_eq!(types[&array].0, 28);
                let element = types[&array].1[0];
                let elements = constants[&types[&array].1[1]];
                let (width, is_output) = if types[&element].0 == 23 {
                    assert_eq!(types[&element].1[1], 4);
                    let scalar = types[&element].1[0];
                    assert_eq!(types[&scalar].0, 22);
                    assert_eq!(types[&scalar].1[0], 16);
                    (8, false)
                } else {
                    assert_eq!(types[&element], (22, vec![32]));
                    assert!(output.replace(variable).is_none());
                    (4, true)
                };
                ranges.push((variable, offset, offset + elements * width, is_output));
            }
            let output = output.unwrap();
            if explicit {
                assert_eq!(ranges.iter().map(|r| r.2).max(), Some(expected));
                let out = ranges.iter().find(|r| r.3).unwrap();
                let operands = ranges.iter().filter(|r| !r.3).collect::<Vec<_>>();
                assert!(operands[0].2 <= operands[1].1 || operands[1].2 <= operands[0].1);
                assert!(operands.iter().any(|r| r.1 <= out.1 && out.2 <= r.2));
            } else {
                assert_eq!(ranges.iter().map(|r| r.2 - r.1).sum::<usize>(), expected);
            }
            let shared_ids = shared.iter().map(|r| r.0).collect::<BTreeSet<_>>();
            let mut origins = BTreeMap::new();
            let mut operand_access = BTreeSet::new();
            let mut output_stores = Vec::new();
            let mut barriers = BTreeSet::new();
            let mut edges = vec![Vec::new(); instructions.len()];
            for (i, (op, a)) in instructions.iter().enumerate() {
                edges[i] = match op {
                    249 => vec![labels[&a[0]]],
                    250 => vec![labels[&a[1]], labels[&a[2]]],
                    251 => std::iter::once(labels[&a[1]])
                        .chain(a[3..].iter().step_by(2).map(|l| labels[l]))
                        .collect(),
                    56 | 252..=255 => Vec::new(),
                    _ if i + 1 < instructions.len() => vec![i + 1],
                    _ => Vec::new(),
                };
                if matches!(op, 65..=67 | 70) {
                    origins.insert(a[1], origins.get(&a[2]).copied().unwrap_or(a[2]));
                }
                let access = match op {
                    61 | 4457 => Some(a[2]),
                    62 | 4458 => Some(a[0]),
                    _ => None,
                };
                if let Some(pointer) = access {
                    let root = origins.get(&pointer).copied().unwrap_or(pointer);
                    if shared_ids.contains(&root) {
                        if root != output {
                            operand_access.insert(i);
                        } else if matches!(op, 62 | 4458) {
                            output_stores.push(i);
                        }
                    }
                }
                if *op == 224 {
                    assert_eq!(constants[&a[0]], 2);
                    assert_eq!(constants[&a[1]], 2);
                    assert_eq!(constants[&a[2]] & 264, 264);
                    barriers.insert(i);
                }
            }
            assert!(!operand_access.is_empty() && !output_stores.is_empty());
            // Cut real barriers from the instruction CFG: there must be no
            // path from any operand access to any output store without a
            // workgroup acquire/release handoff. This includes loop backedges
            // and zero-iteration paths, rather than comparing lexical offsets.
            let mut reverse = vec![Vec::new(); edges.len()];
            for (i, next) in edges.iter().enumerate() {
                for &j in next {
                    reverse[j].push(i);
                }
            }
            let mut pending = output_stores.clone();
            let mut reached = BTreeSet::new();
            while let Some(i) = pending.pop() {
                if barriers.contains(&i) || !reached.insert(i) {
                    continue;
                }
                assert!(
                    !operand_access.contains(&i),
                    "unfenced SLM lifetime overlap"
                );
                pending.extend(&reverse[i]);
            }
            // No old operand access can execute once output publication has
            // started, even across the epilogue barrier and output-loop branches.
            let mut pending = output_stores;
            let mut reached = BTreeSet::new();
            while let Some(i) = pending.pop() {
                if !reached.insert(i) {
                    continue;
                }
                assert!(!operand_access.contains(&i), "operand use after SLM reuse");
                pending.extend(&edges[i]);
            }
        }

        #[test]
        fn actual_staged_compiler_preserves_vector_storage_envelopes_and_order() {
            use cubecl::ir::{ElemType, StorageType, UIntKind};
            use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
            use cubecl::{Compiler, WgpuCompilationOptions};
            use std::collections::BTreeMap;
            for mask in 0..8 {
                for (name, definition) in definitions()
                    .into_iter()
                    .filter(|(name, _)| compiled_shared_bytes(name, false).is_some())
                {
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
                        panic!("actual SPIRV required")
                    };
                    assert_eq!(
                        kernel.shared_size,
                        compiled_shared_bytes(name, mask & 4 != 0).unwrap(),
                        "{name} mask{mask}"
                    );
                    assert_actual_shared_alias_and_fenced_lifetimes(
                        &kernel.assembled_module,
                        compiled_shared_bytes(name, mask & 4 != 0).unwrap(),
                        mask & 4 != 0,
                    );
                    if mask == 4 && name == "dense_macro_fp16" {
                        // A real compiled module without its handoff barriers
                        // remains syntactically valid, but cannot justify alias
                        // lifetimes. Exercise that rejection without GPU work.
                        let mut unfenced = kernel.assembled_module[..5].to_vec();
                        let mut at = 5;
                        while at < kernel.assembled_module.len() {
                            let n = (kernel.assembled_module[at] >> 16) as usize;
                            if kernel.assembled_module[at] & 65535 != 224 {
                                unfenced.extend_from_slice(&kernel.assembled_module[at..at + n]);
                            }
                            at += n;
                        }
                        assert!(
                            std::panic::catch_unwind(|| {
                                assert_actual_shared_alias_and_fenced_lifetimes(
                                    &unfenced,
                                    MACRO_OPERAND_BYTES,
                                    true,
                                );
                            })
                            .is_err()
                        );
                    }
                    let mut types = BTreeMap::new();
                    let mut pointers = BTreeMap::new();
                    let mut variables = BTreeMap::new();
                    let mut function_variables = Vec::new();
                    let mut loops = 0;
                    let mut products = 0;
                    let mut bindings = BTreeMap::new();
                    let mut strides = BTreeMap::new();
                    let mut barriers = 0;
                    let words = &kernel.assembled_module;
                    let mut at = 5;
                    while at < words.len() {
                        let count = (words[at] >> 16) as usize;
                        assert!(count != 0 && at + count <= words.len());
                        let op = words[at] & 65535;
                        let args = &words[at + 1..at + count];
                        match op {
                            22 | 23 | 28 | 29 | 30 | 4456 => {
                                types.insert(args[0], (op, args[1..].to_vec()));
                            }
                            32 => {
                                pointers.insert(args[0], args[2]);
                            }
                            59 => {
                                variables.insert(args[1], args[0]);
                                if args[2] == 7 {
                                    function_variables.push(args[0]);
                                }
                            }
                            71 if args.len() == 3 && args[1] == 33 => {
                                bindings.insert(args[2], args[0]);
                            }
                            71 if args.len() == 3 && args[1] == 6 => {
                                strides.insert(args[0], args[2]);
                            }
                            224 => barriers += 1,
                            246 => loops += 1,
                            4459 => products += 1,
                            _ => {}
                        }
                        at += count;
                    }
                    for binding in [0, 1] {
                        let variable = bindings[&binding];
                        let structure = pointers[&variables[&variable]];
                        assert_eq!(types[&structure].0, 30);
                        let array = types[&structure].1[0];
                        assert!(matches!(types[&array].0, 28 | 29));
                        assert_eq!(strides[&array], 8);
                        let vector = types[&array].1[0];
                        assert_eq!(types[&vector].0, 23);
                        assert_eq!(types[&vector].1[1], 4);
                        let element = types[&vector].1[0];
                        assert_eq!(types[&element].0, 22);
                        assert_eq!(types[&element].1[0], 16);
                    }
                    assert_eq!(
                        barriers, 3,
                        "first-load, stage handoff and epilogue fences: {name}"
                    );
                    assert_eq!(loops, 2, "only K64 and output loops: {name} mask{mask}");
                    let expected_products = if name.contains("narrow8") {
                        4
                    } else if name.contains("narrow16") {
                        8
                    } else {
                        16
                    };
                    assert_eq!(
                        products, expected_products,
                        "four ordered K16 groups: {name} mask{mask}"
                    );
                    let accumulators = function_variables
                        .iter()
                        .filter(|pointer| {
                            let ty = pointers[pointer];
                            types.get(&ty).is_some_and(|(op, args)| {
                                *op == 4456 && types[&args[0]].0 == 22 && types[&args[0]].1[0] == 32
                            })
                        })
                        .count();
                    assert_eq!(
                        accumulators,
                        expected_products / 4,
                        "actual live F32 accumulator variables: {name}"
                    );
                    for pointer in function_variables {
                        let ty = pointers[&pointer];
                        assert!(
                            types.get(&ty).is_none_or(|(op, _)| !matches!(op, 28 | 29)),
                            "no live register-first operand array: {name} mask{mask}"
                        );
                    }
                }
            }
        }
        #[test]
        fn vector_staging_covers_tails_once_and_pingpong_preserves_every_k16() {
            // Indices, not floating arithmetic: validate full live vectors,
            // masked storage and stage generations independently of device math.
            for count in [32usize, 35, 65] {
                for rows in [32usize, 33, 65] {
                    for columns in [16usize, 64, 80, 640, 2560] {
                        for token_base in (0..count).step_by(32) {
                            for row_base in (0..rows).step_by(64) {
                                let mut a = vec![None; 1024];
                                let mut b = vec![None; 2048];
                                let fill =
                                    |k: usize,
                                     buffer: usize,
                                     a: &mut [Option<usize>],
                                     b: &mut [Option<usize>]| {
                                        let mut writes_a = vec![0u8; 512];
                                        let mut writes_b = vec![0u8; 1024];
                                        let full_a = token_base + 32 <= count && k + 64 <= columns;
                                        let full_b = row_base + 64 <= rows && k + 64 <= columns;
                                        for lane in 0..64 {
                                            for round in 0..4 {
                                                let job = lane + round * 64;
                                                let t = token_base + job / 8;
                                                let col = k + job % 8 * 8;
                                                for half in 0..2 {
                                                    let i = job * 2 + half;
                                                    writes_a[i] += 1;
                                                    a[buffer * 512 + i] = if full_a
                                                        || (t < count && col < columns)
                                                    {
                                                        assert!(t < count && col + 7 < columns);
                                                        let source = (t * columns + col) / 4 + half;
                                                        assert!(source * 4 + 3 < count * columns);
                                                        Some(source)
                                                    } else {
                                                        None
                                                    };
                                                }
                                            }
                                            for round in 0..8 {
                                                let job = lane + round * 64;
                                                let row = row_base + job / 8;
                                                let col = k + job % 8 * 8;
                                                for half in 0..2 {
                                                    let i = job * 2 + half;
                                                    writes_b[i] += 1;
                                                    b[buffer * 1024 + i] = if full_b
                                                        || (row < rows && col < columns)
                                                    {
                                                        assert!(row < rows && col + 7 < columns);
                                                        let source =
                                                            (row * columns + col) / 4 + half;
                                                        assert!(source * 4 + 3 < rows * columns);
                                                        Some(source)
                                                    } else {
                                                        None
                                                    };
                                                }
                                            }
                                        }
                                        assert!(
                                            writes_a.into_iter().chain(writes_b).all(|v| v == 1)
                                        );
                                    };
                                fill(0, 0, &mut a, &mut b);
                                let mut order = Vec::new();
                                let mut current = 0;
                                for k in (0..columns).step_by(64) {
                                    let next = 1 - current;
                                    if k + 64 < columns {
                                        fill(k + 64, next, &mut a, &mut b);
                                    }
                                    for inner in (0..64).step_by(16).filter(|i| k + i < columns) {
                                        order.push(k + inner);
                                        for sg in 0..4 {
                                            for tr in 0..16 {
                                                for col in 0..16 {
                                                    let t = token_base + (sg / 2) * 16 + tr;
                                                    let expected = if t < count {
                                                        Some((t * columns + k + inner + col) / 4)
                                                    } else {
                                                        None
                                                    };
                                                    let i = current * 512
                                                        + ((sg / 2) * 16 + tr) * 16
                                                        + (inner + col) / 4;
                                                    assert_eq!(a[i], expected);
                                                }
                                            }
                                            for row in 0..32 {
                                                for col in 0..16 {
                                                    let r = row_base + (sg % 2) * 32 + row;
                                                    let expected = if r < rows {
                                                        Some((r * columns + k + inner + col) / 4)
                                                    } else {
                                                        None
                                                    };
                                                    let i = current * 1024
                                                        + ((sg % 2) * 32 + row) * 16
                                                        + (inner + col) / 4;
                                                    assert_eq!(b[i], expected);
                                                }
                                            }
                                        }
                                    }
                                    // Every reader is now complete before old stage reuse.
                                    current = next;
                                }
                                assert_eq!(order, (0..columns).step_by(16).collect::<Vec<_>>());
                            }
                        }
                    }
                }
            }
        }
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
            let definitions = native::definitions();
            #[cfg(feature = "strata-native-wide")]
            let definitions = definitions.into_iter().chain(native_wide::definitions());
            for (name, definition) in definitions {
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
                let expected_shared = native::compiled_shared_bytes(name, layout_mask & 4 != 0);
                #[cfg(feature = "strata-native-wide")]
                let expected_shared = if name.starts_with("dense_wide_native_") {
                    Some(native_wide::SHARED_BYTES)
                } else {
                    expected_shared
                };
                if expected_shared.is_some_and(|n| kernel.shared_size != n) {
                    return Err("staged matrix original shared envelope".into());
                }
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
            for definition in super::native::definitions()
                .into_iter()
                .filter(|(name, _)| {
                    matches!(
                        *name,
                        "expert_down"
                            | "expert_macro_down"
                            | "expert_narrow8_down"
                            | "expert_narrow16_down"
                    )
                })
                .map(|(_, definition)| definition)
            {
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
    fn macro_tile_covers_each_output_once_and_masks_all_partial_rows() {
        for count in [16usize, 17, 35, 63, 64, 65, 129, 257] {
            for rows in [32usize, 33, 63, 64, 65, 1280, 2560] {
                let mut visits = vec![0u8; count * rows];
                for gy in 0..count.div_ceil(32) {
                    for gx in 0..rows.div_ceil(64) {
                        for subgroup in 0..4 {
                            for i in 0..512 {
                                let token =
                                    gy * 32 + (subgroup / 2) * 16 + (i / 256) * 8 + (i % 128) / 16;
                                let row =
                                    gx * 64 + (subgroup % 2) * 32 + ((i / 128) % 2) * 16 + i % 16;
                                if token < count && row < rows {
                                    visits[token * rows + row] += 1;
                                }
                            }
                        }
                    }
                }
                assert!(visits.iter().all(|&v| v == 1), "count={count} rows={rows}");
            }
        }
    }

    #[test]
    fn incomplete_cooperative_type_fixture_does_not_authorize_an_override() {
        use wgpu::hal::vulkan::kuno_cooperative_subgroup as subgroup;
        let matrix = [7 << 16 | 4456, 1, 2, 3, 4, 5, 6];
        let size16 = [6 << 16 | 16, 7, 17, 16, 1, 1];
        let size128 = [6 << 16 | 16, 7, 17, 128, 1, 1];
        assert_eq!(subgroup(&module(&[&matrix, &size16])), None);
        assert_eq!(subgroup(&module(&[&matrix, &size128])), None);
        assert_eq!(subgroup(&module(&[&matrix])), None);
        assert_eq!(subgroup(&module(&[&size16])), None);
        assert_eq!(subgroup(&module(&[&matrix, &size16, &[0]])), None);
    }

    #[cfg(feature = "strata-intel-core")]
    #[test]
    fn every_actual_small_and_macro_kernel_requires_sixteen_lane_subgroups() {
        use cubecl::ir::{ElemType, StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, ExecutionMode, WgpuCompilationOptions};
        for (name, definition) in super::native::definitions() {
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
                panic!("actual native matrix SPIRV required");
            };
            assert_eq!(
                wgpu::hal::vulkan::kuno_cooperative_subgroup(&kernel.assembled_module),
                Some(16),
                "name={name}",
            );
        }
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
