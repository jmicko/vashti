//! Fixed public device acceptance for resident-vector operators and full-span
//! buffer ownership. This is not a production privacy or worker acceptance.
use crate::{
    private_arena::{BufferId, PrivateArena},
    residual_gpu as gpu,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::ops::{self, GatedResidualShape};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::Zeroize;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn floats(a: &PrivateArena, id: BufferId) -> Result<Vec<f32>> {
    let len = a.layout(id)?.logical_words;
    let mut bytes = a.client()?.read_one(a.handle(id)?.clone())?;
    if bytes.len() < len * 4 {
        return Err("resident download length".into());
    }
    let values = bytes[..len * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    bytes.zeroize();
    Ok(values)
}
fn statuses(a: &PrivateArena, id: BufferId) -> Result<Vec<u32>> {
    let len = a.layout(id)?.logical_words;
    let mut bytes = a.client()?.read_one(a.handle(id)?.clone())?;
    if bytes.len() < len * 4 {
        return Err("resident status length".into());
    }
    let values = bytes[..len * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    bytes.zeroize();
    Ok(values)
}
fn parity(a: &PrivateArena, out: BufferId, status: BufferId, expected: &[f32]) -> Result<f32> {
    let mut values = floats(a, out)?;
    let mut flags = statuses(a, status)?;
    gpu::validate_download(&values, &flags, expected.len(), flags.len())?;
    let mut error = 0f32;
    for (&actual, &reference) in values.iter().zip(expected) {
        error = error.max((actual - reference).abs());
        if (actual - reference).abs() > 2e-5 + 3e-5 * reference.abs() {
            return Err("resident operator parity".into());
        }
    }
    values.zeroize();
    flags.zeroize();
    Ok(error)
}

pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let retired = Arc::new(AtomicBool::new(false));
    for (hc, width, lr) in [(3usize, 5usize, 7usize), (4, 2560, 320)] {
        let shape = GatedResidualShape::new(hc, width)?;
        let layout = gpu::ResidualLayout::new(shape, lr)?;
        let mut arena = PrivateArena::new(client.clone(), 32 * 1024 * 1024, retired.clone())?;
        let mut original: Vec<f32> = (0..hc * width)
            .map(|i| ((i % 31) as f32 - 15.) / 13.)
            .collect();
        let mut gamma: Vec<f32> = (0..hc * width)
            .map(|i| 0.4 + (i % 17) as f32 / 17.)
            .collect();
        let mut down: Vec<f32> = (0..lr).map(|i| ((i % 29) as f32 - 14.) / 9.).collect();
        let mut up: Vec<f32> = (0..hc * width)
            .map(|i| ((i % 23) as f32 - 11.) / 7.)
            .collect();
        let mut block: Vec<f32> = (0..width).map(|i| ((i % 19) as f32 - 9.) / 11.).collect();
        let mut inject: Vec<f32> = (0..hc).map(|i| i as f32 - 1.5).collect();
        layout.validate_norm(original.len(), gamma.len(), 1e-6)?;
        let input = arena.upload_f32(&original)?;
        let g = arena.upload_f32(&gamma)?;
        let dn = arena.upload_f32(&down)?;
        let u = arena.upload_f32(&up)?;
        let b = arena.upload_f32(&block)?;
        let inj = arena.upload_f32(&inject)?;
        let norm = arena.allocate_zeroed(hc * width)?;
        let norm_status = arena.allocate_zeroed(hc)?;
        let low = arena.allocate_zeroed(lr)?;
        let low_status = arena.allocate_zeroed(lr)?;
        let mixed = arena.allocate_zeroed(width)?;
        let mix_status = arena.allocate_zeroed(width)?;
        let combined = arena.allocate_zeroed(hc * width)?;
        let combine_status = arena.allocate_zeroed(hc * width)?;
        // SAFETY: layout checked, separately owned padded spans are all distinct.
        unsafe {
            gpu::grouped_rms_norm::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.norm_groups, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(arena.handle(input)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(g)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(norm)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(norm_status)?.clone(), hc),
                1e-6,
                hc,
                width,
            );
            gpu::activate_down::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.down_groups, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(arena.handle(dn)?.clone(), lr),
                ArrayArg::from_raw_parts(arena.handle(low)?.clone(), lr),
                ArrayArg::from_raw_parts(arena.handle(low_status)?.clone(), lr),
                hc,
                lr,
            );
            gpu::mix::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.mix_groups, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(arena.handle(norm)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(u)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(mixed)?.clone(), width),
                ArrayArg::from_raw_parts(arena.handle(mix_status)?.clone(), width),
                hc,
                width,
            );
            gpu::combine::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.combine_groups, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(arena.handle(input)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(b)?.clone(), width),
                ArrayArg::from_raw_parts(arena.handle(inj)?.clone(), hc),
                ArrayArg::from_raw_parts(arena.handle(combined)?.clone(), hc * width),
                ArrayArg::from_raw_parts(arena.handle(combine_status)?.clone(), hc * width),
                hc,
                width,
            );
        }
        let mut ref_norm = ops::rms_norm(&original, &gamma, width, 1e-6)?;
        let independent = gpu::reference_rms(&original, &gamma, width, 1e-6)?;
        if ref_norm
            .iter()
            .zip(independent)
            .any(|(&a, b)| (a - b).abs() > 1e-6)
        {
            return Err("resident independent normalization reference".into());
        }
        let mut ref_low = shape.activate_down(&down)?;
        let mut ref_mix = shape.mix(&ref_norm, &up)?;
        let mut ref_combined = shape.combine(&original, &block, &inject)?;
        let errors = [
            parity(&arena, norm, norm_status, &ref_norm)?,
            parity(&arena, low, low_status, &ref_low)?,
            parity(&arena, mixed, mix_status, &ref_mix)?,
            parity(&arena, combined, combine_status, &ref_combined)?,
        ];
        println!(
            "public_resident_residual hc={hc} width={width} low_rank={lr} max_errors={errors:?} buffers={} charged_bytes={} budget_bytes={}",
            arena.buffer_count(),
            arena.charged_bytes(),
            arena.budget_bytes()
        );
        // Inspection-only sentinel: deliberately retain a raw clone solely to
        // read full-span erasure after close. Never enqueue graph work with it.
        let sentinel_words = vec![0xa596_317bu32; 8193];
        let sentinel = arena.upload_u32(&sentinel_words)?;
        let span = arena.layout(sentinel)?.owned_words;
        let retained = arena.handle(sentinel)?.clone();
        let before = client.read_one(retained.clone())?;
        if before.len() != span * 4
            || before[..8193 * 4]
                .chunks_exact(4)
                .any(|b| u32::from_le_bytes(b.try_into().unwrap()) != 0xa596_317b)
            || before[8193 * 4..].iter().any(|&b| b != 0)
        {
            return Err("arena full-span initialization failed".into());
        }
        arena.close()?;
        let mut after = client.read_one(retained)?;
        if after.len() != span * 4
            || after.iter().any(|&b| b != 0)
            || arena.handle(sentinel).is_ok()
            || retired.load(Ordering::Acquire)
        {
            return Err("arena full-span close failed".into());
        }
        after.zeroize();
        println!("public_arena_close logical_words=8193 erased_words={span} context_retired=false");
        for values in [
            &mut original,
            &mut gamma,
            &mut down,
            &mut up,
            &mut block,
            &mut inject,
            &mut ref_norm,
            &mut ref_low,
            &mut ref_mix,
            &mut ref_combined,
        ] {
            values.zeroize();
        }
    }
    let mut arena = PrivateArena::new(client.clone(), 1024 * 1024, retired)?;
    let old = arena.allocate_zeroed(3)?;
    arena.reset()?;
    let next = arena.allocate_zeroed(3)?;
    if arena.handle(old).is_ok() || arena.handle(next).is_err() {
        return Err("arena reset epoch failed".into());
    }
    arena.close()?;
    Ok(())
}

/// Entire recurrent block has device-owned old/next states and no intermediate
/// readback. Download all candidates only for this public numerical oracle.
pub fn gdn(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    use crate::{gdn_gpu as recurrence, gdn_pointwise as pointwise, private_arena};
    use kuno_text::ops::{CausalConvState, GdnShape, GdnState};
    for shape in [GdnShape::new(128, 2, 6)?, GdnShape::qwen38()] {
        let layout = pointwise::PointwiseLayout::new(shape)?;
        let recurrent = recurrence::GdnLayout::new(shape)?;
        let retired = Arc::new(AtomicBool::new(false));
        let mut arena = PrivateArena::new(client.clone(), 32 * 1024 * 1024, retired.clone())?;
        let mut old_ring = arena.allocate_zeroed(layout.ring_elements)?;
        let mut next_ring = arena.allocate_zeroed(layout.ring_elements)?;
        let mut old_state = arena.allocate_zeroed(recurrent.state_elements)?;
        let mut next_state = arena.allocate_zeroed(recurrent.state_elements)?;
        let activation = arena.allocate_zeroed(layout.channels)?;
        let prepared = arena.allocate_zeroed(layout.channels)?;
        let controls = arena.allocate_zeroed(layout.control_elements)?;
        let output = arena.allocate_zeroed(layout.value_elements)?;
        let gated = arena.allocate_zeroed(layout.value_elements)?;
        let conv_status = arena.allocate_zeroed(layout.channels)?;
        let qkv_status = arena.allocate_zeroed(layout.channels)?;
        let control_status = arena.allocate_zeroed(shape.value_heads)?;
        let state_status = arena.allocate_zeroed(layout.value_elements)?;
        let closing_status = arena.allocate_zeroed(layout.value_elements)?;
        let weights: Vec<f32> = (0..layout.conv_weight_elements)
            .map(|i| ((i % 17) as f32 - 8.) / 19.)
            .collect();
        let gamma: Vec<f32> = (0..128).map(|i| 0.7 + (i % 11) as f32 / 11.).collect();
        let bias = vec![0f32; shape.value_heads];
        let mut stored_a = vec![-0.5f32; shape.value_heads];
        stored_a[0] = -1e8;
        let mut params = bias.clone();
        params.extend_from_slice(&stored_a);
        let w = arena.upload_f32(&weights)?;
        let g = arena.upload_f32(&gamma)?;
        let p = arena.upload_f32(&params)?;
        let mut conv = CausalConvState::new(layout.channels, 4, 1)?;
        let mut state = GdnState::new(shape)?;
        let mut maximum = 0f32;
        let mut state_error = 0f32;
        for step in 0..7 {
            let mut x: Vec<f32> = (0..layout.channels)
                .map(|i| ((i * 7 + step * 3) % 31) as f32 / 29. - 0.5)
                .collect();
            let mut alpha: Vec<f32> = (0..shape.value_heads)
                .map(|i| i as f32 / 37. - 0.6)
                .collect();
            alpha[0] = -20.;
            let mut beta: Vec<f32> = (0..shape.value_heads)
                .map(|i| (i % 9) as f32 / 7. - 0.5)
                .collect();
            let mut z: Vec<f32> = (0..layout.value_elements)
                .map(|i| (i % 13) as f32 / 9. - 0.8)
                .collect();
            pointwise::validate_conv_inputs(layout, &x, &weights, &vec![0.; layout.ring_elements])?;
            pointwise::validate_control_inputs(layout, &alpha, &beta, &params)?;
            pointwise::validate_conv_lengths(
                layout,
                layout.channels,
                weights.len(),
                layout.ring_elements,
                layout.ring_elements,
                layout.channels,
                layout.channels,
            )?;
            pointwise::validate_qkv_lengths(
                layout,
                layout.channels,
                layout.channels,
                layout.channels,
                pointwise::EPSILON,
            )?;
            pointwise::validate_control_lengths(
                layout,
                alpha.len(),
                beta.len(),
                params.len(),
                layout.control_elements,
                shape.value_heads,
            )?;
            let shared = pointwise::validate_output_lengths(
                layout,
                layout.value_elements,
                z.len(),
                gamma.len(),
                layout.value_elements,
                layout.value_elements,
                pointwise::EPSILON,
            )?;
            let xi = arena.upload_f32(&x)?;
            let al = arena.upload_f32(&alpha)?;
            let be = arena.upload_f32(&beta)?;
            let zi = arena.upload_f32(&z)?;
            let arg = |id: BufferId| -> Result<ArrayArg<WgpuRuntime>> {
                // SAFETY: arena owns live full spans; kernel uses validated logical lengths.
                Ok(unsafe {
                    ArrayArg::from_raw_parts(
                        arena.handle(id)?.clone(),
                        arena.layout(id)?.logical_words,
                    )
                })
            };
            for id in [
                conv_status,
                qkv_status,
                control_status,
                state_status,
                closing_status,
            ] {
                let full = arena.layout(id)?;
                // SAFETY: erase covers the complete owning page, including padding.
                private_arena::erase_full_span::launch::<WgpuRuntime>(
                    client,
                    CubeCount::Static(full.workgroups_x, full.workgroups_y, 1),
                    CubeDim::new_1d(64),
                    unsafe {
                        ArrayArg::from_raw_parts(arena.handle(id)?.clone(), full.owned_words)
                    },
                );
            }
            pointwise::conv_silu::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.conv_cubes, 1, 1),
                CubeDim::new_1d(64),
                arg(xi)?,
                arg(w)?,
                arg(old_ring)?,
                arg(next_ring)?,
                arg(activation)?,
                arg(conv_status)?,
                layout.channels,
            );
            pointwise::prepare_qkv::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.qkv_heads as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(activation)?,
                arg(prepared)?,
                arg(qkv_status)?,
                pointwise::EPSILON,
                shape.key_heads,
                shape.value_heads,
            );
            pointwise::prepare_controls::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(layout.control_cubes, 1, 1),
                CubeDim::new_1d(64),
                arg(al)?,
                arg(be)?,
                arg(p)?,
                arg(controls)?,
                arg(control_status)?,
                shape.value_heads,
            );
            let region = |first: usize, len: usize| -> Result<ArrayArg<WgpuRuntime>> {
                let full = arena.layout(prepared)?;
                if first + len > full.logical_words || !(first * 4).is_multiple_of(256) {
                    return Err("GDN prepared view geometry".into());
                }
                let h = arena
                    .handle(prepared)?
                    .clone()
                    .offset_start((first * 4) as u64)
                    .offset_end(((full.owned_words - first - len) * 4) as u64);
                if h.size_in_used() != (len * 4) as u64 {
                    return Err("GDN prepared view length".into());
                }
                // SAFETY: disjoint READ-ONLY Q/K/V ranges, padded owner retained.
                Ok(unsafe { ArrayArg::from_raw_parts(h, len) })
            };
            recurrence::gdn_step::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(recurrent.workgroups_x, 1, 1),
                CubeDim::new_1d(64),
                region(0, layout.qk_elements)?,
                region(layout.qk_elements, layout.qk_elements)?,
                region(2 * layout.qk_elements, layout.value_elements)?,
                arg(controls)?,
                arg(old_state)?,
                arg(next_state)?,
                arg(output)?,
                arg(state_status)?,
                shape.head_dim,
                shape.key_heads,
                shape.value_heads,
            );
            pointwise::output_gate::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(shape.value_heads as u32, 1, 1),
                CubeDim::new_1d(64),
                arg(output)?,
                arg(zi)?,
                arg(g)?,
                arg(gated)?,
                arg(closing_status)?,
                pointwise::EPSILON,
                shape.value_heads,
                shared,
            );
            let flags = [
                statuses(&arena, conv_status)?,
                statuses(&arena, qkv_status)?,
                statuses(&arena, control_status)?,
                statuses(&arena, state_status)?,
                statuses(&arena, closing_status)?,
            ];
            if flags.iter().any(|s| s.iter().any(|&v| v != 1)) {
                let bad: Vec<_> = flags
                    .iter()
                    .map(|s| {
                        (
                            s.len(),
                            s.iter().filter(|&&v| v != 1).count(),
                            s.iter().position(|&v| v != 1),
                        )
                    })
                    .collect();
                println!(
                    "public_resident_gdn_failure step={step} shape={shape:?} stage_bad_counts={bad:?}"
                );
                println!(
                    "public_resident_gdn_debug activation={:?} prepared={:?} controls={:?} output={:?}",
                    &floats(&arena, activation)?[..8],
                    &floats(&arena, prepared)?[..8],
                    &floats(&arena, controls)?[..6],
                    &floats(&arena, output)?[..8]
                );
            }
            pointwise::validate_token_status(
                layout, &flags[0], &flags[1], &flags[2], &flags[3], &flags[4],
            )?;
            let mut actual = floats(&arena, gated)?;
            let mut candidate = floats(&arena, next_state)?;
            let mut raw = floats(&arena, next_ring)?;
            let mut actual_prepared = floats(&arena, prepared)?;
            let mut actual_controls = floats(&arena, controls)?;
            pointwise::validate_candidates(
                layout,
                &raw,
                &actual_prepared,
                &actual_controls,
                &actual,
            )?;
            let mut u = conv.step(&x, &weights)?;
            u.iter_mut().for_each(|v| *v = ops::silu(*v));
            pointwise::validate_qkv_inputs(layout, &u, pointwise::EPSILON)?;
            let mut q = ops::l2_norm_heads(&u[..layout.qk_elements], 128, pointwise::EPSILON)?;
            q.iter_mut().for_each(|v| *v /= 128f32.sqrt());
            let k = ops::l2_norm_heads(
                &u[layout.qk_elements..2 * layout.qk_elements],
                128,
                pointwise::EPSILON,
            )?;
            let decay = ops::gdn_log_decay(&alpha, &bias, &stored_a)?;
            let probabilities: Vec<f32> = beta.iter().copied().map(ops::sigmoid).collect();
            let mut expected_raw =
                state.step(&q, &k, &u[2 * layout.qk_elements..], &decay, &probabilities)?;
            pointwise::validate_output_inputs(
                layout,
                &expected_raw,
                &z,
                &gamma,
                pointwise::EPSILON,
            )?;
            let mut expected =
                ops::gdn_output_gate(&expected_raw, &z, &gamma, 128, pointwise::EPSILON)?;
            for (&a, &e) in actual.iter().zip(&expected) {
                maximum = maximum.max((a - e).abs());
                if !a.is_finite() || (a - e).abs() > 2e-5 + 5e-5 * e.abs() {
                    return Err("resident GDN closing parity".into());
                }
            }
            for (&a, &e) in candidate.iter().zip(state.values()) {
                state_error = state_error.max((a - e).abs());
                if !a.is_finite() || (a - e).abs() > 2e-6 + 5e-5 * e.abs() {
                    return Err("resident GDN state parity".into());
                }
            }
            // Commit both pointers only after every stage and oracle accepts.
            std::mem::swap(&mut old_ring, &mut next_ring);
            std::mem::swap(&mut old_state, &mut next_state);
            for values in [
                &mut x,
                &mut alpha,
                &mut beta,
                &mut z,
                &mut actual,
                &mut candidate,
                &mut raw,
                &mut actual_prepared,
                &mut actual_controls,
                &mut u,
                &mut q,
                &mut expected_raw,
                &mut expected,
            ] {
                values.zeroize();
            }
        }
        println!(
            "public_resident_gdn shape={shape:?} steps=7 max_state_abs_error={state_error} max_gated_abs_error={maximum} oracle={} charged_bytes={}",
            pointwise::ORACLE_SHA,
            arena.charged_bytes()
        );
        arena.close()?;
        if retired.load(Ordering::Acquire) {
            return Err("resident GDN context retired".into());
        }
    }
    Ok(())
}
