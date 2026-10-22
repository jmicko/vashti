//! Real8-group mixed-family matrices against the original GPU products.
//! Independent original F64 acceptance remains in public_oracle controls.
use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_prompt::groups as g,
};
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn sanitize(arena: &PrivateArena) -> Checked<()> {
    let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
    let bytes = receipt
        .upload_bytes
        .checked_add(receipt.uniform_bytes)
        .and_then(|v| v.checked_add(receipt.readback_bytes))
        .ok_or("public cohort pair IO audit overflow")?;
    if bytes == 0 || receipt.inspected_bytes != bytes || receipt.inspected_buffers == 0 {
        return Err("public cohort pair complete transient IO erasure".into());
    }
    Ok(())
}
fn compare(arena: &PrivateArena, a: BufferId, b: BufferId) -> Checked<usize> {
    let words = arena.layout(a)?.owned_bytes / 4;
    if arena.layout(b)?.owned_bytes / 4 != words {
        return Err("public cohort paired whole-owner extent".into());
    }
    const WINDOW: usize = 4 * 1024 * 1024;
    for start in (0..words).step_by(WINDOW) {
        let count = WINDOW.min(words - start);
        let view = |id| -> Checked<Handle> {
            Ok(arena
                .handle(id)?
                .clone()
                .offset_start((start * 4) as u64)
                .offset_end(((words - start - count) * 4) as u64))
        };
        let x = u::read_handle_prefix_u32(arena, view(a)?, count)?;
        let y = u::read_handle_prefix_u32(arena, view(b)?, count)?;
        let equal = *x == *y;
        drop(x);
        drop(y);
        sanitize(arena)?;
        if !equal {
            return Err("public cohort exact original matrix/owner/status mismatch".into());
        }
    }
    Ok(words)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    down: bool,
    poison: bool,
) -> Checked<serde_json::Value> {
    let counts = [1usize, 3, 7, 8, 9, 16, 17, 31];
    let total = 4096usize;
    let mut raw = Zeroizing::new(vec![0u32; g::LOGICAL + total * 10]);
    raw[..3].copy_from_slice(&[total as u32, 0, total as u32]);
    for token in 0..total {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let logical = if rank < 8 && token >= total - counts[rank] {
                400 + rank
            } else {
                1 + rank
            };
            raw[g::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let mut map = g::group(&raw, total)?;
    let first = g::validate(&map, total)? - 8;
    if poison {
        let offset = map[1028 + first + 6] as usize;
        map[1540 + offset] = g::ROUTES as u32;
    }
    let cols = if down { 640 } else { 2560 };
    let rows = if down { 2560 } else { 1280 };
    let mut words = Zeroizing::new([0u32; 48]);
    words[0] = total as u32;
    words[1] = cols;
    words[2] = rows;
    words[4] = first as u32;
    words[5] = if down { 2 } else { 0 };
    words[6] = 100;
    words[7] = 100;
    words[8] = 300;
    words[9] = 8;
    words[10] = 31;
    let mut prefix = 0;
    for (i, n) in counts.into_iter().enumerate() {
        if map[516 + first + i] as usize != n {
            return Err("public cohort paired authentic group fixture".into());
        }
        words[16 + i * 4..20 + i * 4].copy_from_slice(&[(first + i) as u32, n as u32, prefix, 8]);
        prefix += n.div_ceil(if n <= 8 { 8 } else { 16 }) as u32 * if n <= 8 { 8 } else { 16 };
    }
    let mut image = Zeroizing::new(vec![
        0u32;
        8 * crate::strata_prompt::experts::cohorts::IMAGE_BYTES
            as usize
            / 4
    ]);
    let mut input = Zeroizing::new(vec![0u32; prefix as usize * cols as usize / 2]);
    for i in 0..8 {
        let row_prefix = words[18 + i * 4] as usize;
        let members = counts[i];
        for member in 0..members {
            for col in 0..cols as usize {
                let value = half::f16::from_f32(
                    0.125 + ((member * 13 + col * 7 + i * 3) % 43) as f32 / 19.0,
                );
                let at = (row_prefix + member) * cols as usize + col;
                input[at / 2] |= u32::from(value.to_bits()) << (16 * (at % 2));
            }
        }
        let half_first = i * 4_915_200 + if down { 3_276_800 } else { 0 };
        for row in 0..rows as usize {
            for col in 0..cols as usize {
                let value = if poison && i == 3 && row == 0 && col == 0 {
                    half::f16::NAN
                } else {
                    half::f16::from_f32(
                        (((row * 17 + col * 11 + i * 19) % 67) as f32 - 33.0) / 41.0,
                    )
                };
                let at = half_first + row * cols as usize + col;
                image[at / 2] |= u32::from(value.to_bits()) << (16 * (at % 2));
            }
        }
    }
    let src = arena.upload_u32(&input)?;
    let weight = arena.upload_u32(&image)?;
    let map_id = arena.upload_u32(&map)?;
    let output_bytes = if down { g::PART_BYTES } else { g::GU_BYTES };
    let a = arena.allocate_zeroed(output_bytes as usize / 4)?;
    let b = arena.allocate_zeroed(output_bytes as usize / 4)?;
    let status = Zeroizing::new(vec![1u32; g::STATUS_WORDS]);
    let sa = arena.upload_u32(&status)?;
    let sb = arena.upload_u32(&status)?;
    let fa = arena.allocate_zeroed(1)?;
    let fb = arena.allocate_zeroed(1)?;
    for i in 0..8 {
        let prefix = words[18 + i * 4] as u64;
        let first_half = i as u64 * 4_915_200 + if down { 3_276_800 } else { 0 };
        let ih = u::handle(arena, src)?
            .offset_start(prefix * u64::from(cols) * 2)
            .offset_end(0);
        let wh = u::handle(arena, weight)?
            .offset_start(first_half * 2)
            .offset_end(0);
        let oh = if down {
            u::handle(arena, a)?
        } else {
            u::handle(arena, a)?
                .offset_start(prefix * 1280 * 4)
                .offset_end(0)
        };
        super::super::native::run_expert(
            client,
            [
                ih,
                wh,
                oh,
                u::handle(arena, map_id)?,
                u::handle(arena, sa)?,
                u::handle(arena, fa)?,
            ],
            counts[i],
            cols as usize,
            rows as usize,
            0,
            first + i,
            down,
        )?;
    }
    for tile in [8usize, 16] {
        words[11] = tile as u32;
        launch(
            client,
            [
                u::handle(arena, src)?,
                u::handle(arena, weight)?,
                u::handle(arena, b)?,
                u::handle(arena, map_id)?,
                u::handle(arena, sb)?,
                u::handle(arena, fb)?,
            ],
            &words,
            down,
            tile,
        )?;
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let checked = compare(arena, a, b)?;
    compare(arena, sa, sb)?;
    compare(arena, fa, fb)?;
    for (id, expected) in [(src, &*input), (weight, &*image), (map_id, &*map)] {
        let root = arena.handle(id)?.clone();
        let owned = arena.layout(id)?.owned_bytes / 4;
        for start in (0..owned).step_by(4 * 1024 * 1024) {
            let n = (4 * 1024 * 1024).min(owned - start);
            let view = root
                .clone()
                .offset_start((start * 4) as u64)
                .offset_end(((owned - start - n) * 4) as u64);
            let actual = u::read_handle_prefix_u32(arena, view, n)?;
            if actual
                .iter()
                .enumerate()
                .any(|(i, v)| *v != expected.get(start + i).copied().unwrap_or(0))
            {
                return Err("public cohort paired original source owner changed".into());
            }
            drop(actual);
            sanitize(arena)?;
        }
    }
    Ok(
        serde_json::json!({"family":"cohort-paired-original-matrix","down":down,"groups":8,"counts":counts,"poison_and_invalid_route":poison,"actual_original_and_cohort_gpu_programs":true,"original_map_builder_and_validator":true,"nonzero_group_prefix_and_image_slot7":true,"bit_mismatches":0,"full_output_words_checked":checked,"full_owner_status_fault_sources_and_tail_checked":true}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("public cohort pair original empty audit arena required".into());
    }
    let mut reports = Vec::new();
    for down in [false, true] {
        for poison in [false, true] {
            let result = one(client, arena, down, poison);
            arena.reset()?;
            reports.push(result?);
        }
    }
    Ok(reports)
}
