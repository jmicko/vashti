//! Compiled-public inputs through the actual prompt router and ordered footer.
//! No token/model request content, custom shader, or fabricated owner metadata.
use super::*;
use crate::{gr_batch_probe as probe, strata_prompt::groups};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn fail(label: &'static str) -> Box<dyn std::error::Error> {
    eprintln!("public_prompt_route_fault {label}");
    label.into()
}

/// Fixed source layout includes exact ties and nontrivial per-token winners.
fn logits(count: usize) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..count * 512)
            .map(|i| {
                if (i / 512).is_multiple_of(3) {
                    0.
                } else {
                    (((i % 512 * 37 + i / 512 * 23) % 512) as f32 - 256.) / 32.
                }
            })
            .collect(),
    )
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> ProbeResult<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    // All requested token tails plus explicit terminal-fault/header controls.
    for (count, flags, control) in [
        (1, 0, "valid"),
        (8, 0, "valid"),
        (9, 0, "valid"),
        (35, 0, "valid"),
        (257, 1, "valid"),
        (4096, 0, "valid"),
        (9, 1, "nan"),
        (9, 0, "bad_remap"),
        (9, 1, "prior_fault"),
    ] {
        let layer = 7usize;
        let mut source = logits(count);
        if control == "nan" {
            source[8 * 512 + 17] = f32::NAN;
        }
        let mut remap = Zeroizing::new(
            (0..32768)
                .map(|i| ((i * 13 + 17) % 512) as u32)
                .collect::<Vec<_>>(),
        );
        if control == "bad_remap" {
            remap[layer * 512] = 512;
        }
        let logit_id = if control == "nan" {
            // The ordinary float-upload admission correctly rejects NaNs.
            // Only this compiled-public negative control supplies raw bits so
            // the unchanged production GPU finite/fault guard is exercised.
            let bits = Zeroizing::new(source.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            arena.upload_u32(&bits)?
        } else {
            arena.upload_f32(&source)?
        };
        let remap_id = arena.upload_u32(&remap)?;
        let map_id = arena.allocate_zeroed(groups::RAW_WORDS)?;
        let probability_id = arena.allocate_zeroed(groups::PROB_WORDS)?;
        let fault_id = arena.upload_u32(&[if control == "prior_fault" { 32 } else { 0 }])?;
        let usage_id = arena.allocate_zeroed(32768)?;
        let args = || -> kuno_text::Result<KernelArguments> {
            let mut args = KernelArguments::new();
            for id in [
                logit_id,
                remap_id,
                map_id,
                probability_id,
                fault_id,
                usage_id,
            ] {
                args = args.with_buffer(u::handle(arena, id)?.binding());
            }
            // Exact production parameter bytes; no replay override is needed
            // for this immediate public invocation with a fixed nonzero epoch.
            Ok(args.with_info(MetadataBindingInfo::custom(vec![
                count as u64 | ((layer as u64) << 32),
                1 | ((flags as u64) << 32),
            ])))
        };
        // SAFETY: genuine distinct arena roots, production descriptors/extents,
        // fixed compiled-public dimensions. Same owners order both dispatches.
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel::PromptRoute),
                CubeCount::Static(prompt_route_groups(count)?, 1, 1),
                args()?,
            );
        }
        if count == 1 {
            let before = probe::read_native_public(arena, map_id)?;
            if before[..4] != [0; 4] {
                return Err(fail("header_published_before_footer"));
            }
        }
        unsafe {
            client.launch_unchecked(
                Box::new(Kernel::PromptRouteFinalize),
                CubeCount::Static(1, 1, 1),
                args()?,
            );
        }
        let map = probe::read_native_public(arena, map_id)?;
        let probability = probe::read_native_public(arena, probability_id)?;
        let fault = probe::read_native_public(arena, fault_id)?;
        let usage = probe::read_native_public(arena, usage_id)?;
        let mut max_error = 0f64;
        if control == "valid" {
            if fault.iter().any(|&word| word != 0) || map[..4] != [count as u32, 0, count as u32, 0]
            {
                return Err(fail("valid_header_or_fault"));
            }
            let mut reference_usage = Zeroizing::new(vec![0u32; usage.len()]);
            for token in 0..count {
                let row = &source[token * 512..][..512];
                let mut ids = [0usize; 512];
                for (i, id) in ids.iter_mut().enumerate() {
                    *id = i;
                }
                ids.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then_with(|| a.cmp(&b)));
                let maximum = f64::from(row[ids[0]]);
                let total = ids[..10]
                    .iter()
                    .map(|&id| (f64::from(row[id]) - maximum).exp())
                    .sum::<f64>();
                for (rank, &id) in ids[..10].iter().enumerate() {
                    let route = token * 10 + rank;
                    if map[groups::LOGICAL + route] != id as u32
                        || map[1540 + route] != remap[layer * 512 + id]
                    {
                        return Err(fail("logical_or_physical_id_mismatch"));
                    }
                    let expected = (f64::from(row[id]) - maximum).exp() / total;
                    let actual = f64::from(f32::from_bits(probability[route]));
                    let error = (actual - expected).abs();
                    max_error = max_error.max(error);
                    if !actual.is_finite() || error > 1e-6 + 1e-5 * expected.abs() {
                        return Err(fail("selected_probability_mismatch"));
                    }
                    if flags & 1 != 0 {
                        reference_usage[layer * 512 + id] += 1;
                    }
                }
            }
            for (i, &word) in map.iter().enumerate() {
                if i >= 4
                    && !(1540..1540 + count * 10).contains(&i)
                    && !(groups::LOGICAL..groups::LOGICAL + count * 10).contains(&i)
                    && word != 0
                {
                    return Err(fail("map_full_owner_tail_changed"));
                }
            }
            if probability[count * 10..].iter().any(|&word| word != 0)
                || usage
                    .iter()
                    .zip(reference_usage.iter())
                    .any(|(&bits, &n)| f32::from_bits(bits) != n as f32)
            {
                return Err(fail("probability_tail_or_usage_mismatch"));
            }
            // Also retain the actual CPU grouping validation used by inference.
            groups::group(&map[..groups::LOGICAL + count * 10], count)?;
        } else {
            let expected = if control == "prior_fault" { 32 } else { 128 };
            if fault[0] != expected
                || fault[1..].iter().any(|&word| word != 0)
                || map[..4] != [0; 4]
                || usage.iter().any(|&word| word != 0)
            {
                return Err(fail("terminal_fault_or_invalid_header_mismatch"));
            }
        }
        println!("public_prompt_route_numeric count={count} control={control} max_abs={max_error}");
        reports.push(
            serde_json::json!({"family":"prompt-route-finalizer", "count":count,
            "control":control,"usage_enabled":flags&1!=0,"exact_ids":true,"max_abs":max_error,
            "actual_native_kernels":true,"full_owner_and_tail_checked":true}),
        );
        arena.reset()?;
    }
    Ok(reports)
}
