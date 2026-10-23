//! Whole-owner paired real-format proof: original decoded image + original
//! accepted CMMA versus compact original bytes + fused staged CMMA. All public
//! request/source values stay in wiping buffers. No private production logging.
use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
    strata_prompt::groups as g,
};
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WINDOW_WORDS: usize = 4 * 1024 * 1024;
// A public oracle only: never attached to a private inference request. The
// largest original DOWN owner is400MiB. Keep its reference in scrubbed host
// memory rather than charging a second400MiB GPU owner to the original arena.
const REFERENCE_BYTES: usize = 400 * 1024 * 1024;
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
fn window(arena: &PrivateArena, id: BufferId, start: usize) -> Checked<(Handle, usize)> {
    let words = arena.layout(id)?.owned_words;
    if start >= words {
        return Err("public cohort pair full-owner window".into());
    }
    let count = WINDOW_WORDS.min(words - start);
    let view = arena
        .handle(id)?
        .clone()
        .offset_start((start * 4) as u64)
        .offset_end(((words - start - count) * 4) as u64);
    Ok((view, count))
}
fn snapshot(arena: &PrivateArena, id: BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let words = arena.layout(id)?.owned_words;
    if words
        .checked_mul(4)
        .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort pair bounded host reference".into());
    }
    let mut reference = Zeroizing::new(Vec::new());
    reference.try_reserve_exact(words)?;
    if reference
        .capacity()
        .checked_mul(4)
        .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort pair bounded host reference capacity".into());
    }
    reference.resize(words, 0);
    for start in (0..words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let data = u::read_handle_prefix_u32(arena, view, count)?;
        reference[start..start + count].copy_from_slice(&data);
        drop(data);
        sanitize(arena)?;
    }
    Ok(reference)
}
fn erase_output(
    client: &ComputeClient<WgpuRuntime>,
    arena: &PrivateArena,
    id: BufferId,
) -> Checked<()> {
    let layout = arena.layout(id)?;
    let owner = arena.handle(id)?.clone();
    if owner.offset_start.unwrap_or(0) != 0
        || owner.offset_end.unwrap_or(0) != 0
        || owner.size_in_used() != layout.owned_bytes as u64
        || owner.size() != layout.owned_bytes as u64
    {
        return Err("public cohort pair exact output owner erase".into());
    }
    // This is the unchanged original arena eraser and its exact full-span
    // launch geometry. Ownership stays with the arena until reset completes.
    crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(owner, layout.owned_words) },
    );
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    for start in (0..layout.owned_words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let data = u::read_handle_prefix_u32(arena, view, count)?;
        let zero = data.iter().all(|word| *word == 0);
        drop(data);
        sanitize(arena)?;
        if !zero {
            return Err("public cohort pair output owner not completely zero".into());
        }
    }
    Ok(())
}
fn compare_reference(arena: &PrivateArena, id: BufferId, reference: &[u32]) -> Checked<usize> {
    let words = arena.layout(id)?.owned_words;
    if words != reference.len()
        || words
            .checked_mul(4)
            .is_none_or(|bytes| bytes > REFERENCE_BYTES)
    {
        return Err("public cohort paired reference whole-owner extent".into());
    }
    for start in (0..words).step_by(WINDOW_WORDS) {
        let (view, count) = window(arena, id, start)?;
        let actual = u::read_handle_prefix_u32(arena, view, count)?;
        let expected = &reference[start..start + count];
        let mismatch = if actual.as_slice() == expected {
            None
        } else {
            actual
                .iter()
                .zip(expected)
                .enumerate()
                .find_map(|(i, (got, want))| (got != want).then_some((start + i, *want, *got)))
        };
        drop(actual);
        sanitize(arena)?;
        if let Some((word, original, selected)) = mismatch {
            return Err(format!("public cohort exact original matrix/owner/status mismatch word={word} original_bits={original:08x} cohort_bits={selected:08x}").into());
        }
    }
    Ok(words)
}

fn check_source(arena: &PrivateArena, id: BufferId, expected: &[u32]) -> Checked<()> {
    for start in (0..arena.layout(id)?.owned_words).step_by(WINDOW_WORDS) {
        let (view, n) = window(arena, id, start)?;
        let actual = u::read_handle_prefix_u32(arena, view, n)?;
        let good = actual
            .iter()
            .enumerate()
            .all(|(j, v)| *v == expected.get(start + j).copied().unwrap_or(0));
        drop(actual);
        sanitize(arena)?;
        if !good {
            return Err("quant cohort source whole owner changed".into());
        }
    }
    Ok(())
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
    dtype: u32,
    poison: bool,
    table: &Handle,
    pipeline: bool,
) -> Checked<serde_json::Value> {
    let down = matches!(dtype, 20 | 42);
    let counts = [1usize, 3, 7, 8, 9, 16, 17, 31];
    let total = 32usize;
    let suffix = if down {
        "ffn_down_exps.weight"
    } else {
        "ffn_gate_exps.weight"
    };
    let (name, tensor) = file
        .tensors()
        .iter()
        .find(|(n, t)| n.ends_with(suffix) && t.dtype().gguf_id() == dtype)
        .ok_or("quant cohort original model format missing")?;
    let cols = if down { 640usize } else { 2560 };
    let rows = if down { 2560usize } else { 1280 };
    let rb = quant_row_bytes(dtype, down);
    if tensor.dtype().row_bytes(cols as u64)? as usize != rb {
        return Err("quant cohort model row extent".into());
    }
    let encoded_rows = if down { 2560usize } else { 640 };
    let role_bytes = encoded_rows * rb;
    let gu_bytes = if down { 640 * 820 } else { role_bytes };
    let db = if down { role_bytes } else { 2560 * 360 };
    let stride = gu_bytes * 2 + db;
    let mut raw = Zeroizing::new(vec![0u32; g::LOGICAL + total * 10]);
    raw[..3].copy_from_slice(&[total as u32, 0, total as u32]);
    for token in 0..total {
        for rank in 0..10 {
            let logical = if rank < 8 && token >= total - counts[rank] {
                400 + rank
            } else {
                1 + rank
            };
            raw[g::LOGICAL + token * 10 + rank] = logical as u32;
            raw[1540 + token * 10 + rank] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let mut map = g::group(&raw, total)?;
    let first = g::validate(&map, total)? - 8;
    if poison {
        let off = map[1028 + first + 6] as usize;
        map[1540 + off] = (total * 10) as u32;
    }
    let mut words = Zeroizing::new([0u32; 48]);
    words[..11].copy_from_slice(&[
        total as u32,
        cols as u32,
        rows as u32,
        rb as u32,
        first as u32,
        if down { 2 } else { 0 },
        (gu_bytes / 4) as u32,
        (gu_bytes / 4) as u32,
        (stride / 4) as u32,
        8,
        31,
    ]);
    let mut prefix = 0usize;
    for (i, n) in counts.into_iter().enumerate() {
        if map[516 + first + i] as usize != n {
            return Err("quant cohort authentic real count groups".into());
        }
        words[16 + i * 4..20 + i * 4].copy_from_slice(&[
            (first + i) as u32,
            n as u32,
            prefix as u32,
            8,
        ]);
        prefix += n.div_ceil(if n <= 8 { 8 } else { 16 }) * if n <= 8 { 8 } else { 16 };
    }
    let mut packed = Zeroizing::new(vec![
        0u8;
        crate::strata_prompt::layout::WORKING_BYTES as usize
    ]);
    let mut image = Zeroizing::new(vec![
        0u32;
        crate::strata_prompt::layout::WORKING_BYTES as usize / 4
    ]);
    let mut weight_fault = 0u32;
    for slot in 0..8 {
        let mut gate = Zeroizing::new(vec![0u8; role_bytes]);
        file.tensor(name)
            .ok_or("quant cohort original tensor")?
            .read_range((slot * role_bytes) as u64, &mut gate)?;
        let mut up = gate.clone();
        if !down {
            file.tensor(&name.replace(suffix, "ffn_up_exps.weight"))
                .ok_or("quant cohort original up")?
                .read_range((slot * role_bytes) as u64, &mut up)?;
        }
        if poison && slot == 3 {
            gate[..2].copy_from_slice(&0x7e00u16.to_le_bytes());
        }
        let base = slot * stride + if down { gu_bytes * 2 } else { 0 };
        packed[base..base + role_bytes].copy_from_slice(&gate);
        if !down {
            packed[base + gu_bytes..base + gu_bytes + role_bytes].copy_from_slice(&up);
        }
        let mut a = Zeroizing::new(vec![0f32; encoded_rows * cols]);
        let mut b = Zeroizing::new(vec![0f32; encoded_rows * cols]);
        tensor.dtype().decode(&gate, &mut a)?;
        tensor.dtype().decode(&up, &mut b)?;
        let first_half = slot * 4_915_200 + if down { 3_276_800 } else { 0 };
        for row in 0..rows {
            for col in (0..cols).step_by(8) {
                let source = if down || row % 2 == 0 { &a } else { &b };
                let r = if down { row } else { row / 2 };
                let values = &source[r * cols + col..r * cols + col + 8];
                let halves = std::array::from_fn::<_, 8, _>(|j| half::f16::from_f32(values[j]));
                let good =
                    values.iter().all(|v| v.is_finite()) && halves.iter().all(|v| v.is_finite());
                if !good {
                    weight_fault = 128;
                }
                for j in 0..8 {
                    let at = first_half + row * cols + col + j;
                    let bits = if good { halves[j].to_bits() } else { 0 };
                    image[at / 2] |= u32::from(bits) << (16 * (at % 2));
                }
            }
        }
    }
    let mut input = Zeroizing::new(vec![0u32; prefix * cols / 2]);
    for slot in 0..8 {
        let off = words[18 + slot * 4] as usize;
        for m in 0..counts[slot] {
            for k in 0..cols {
                let v =
                    half::f16::from_f32((((slot * 7 + m * 13 + k * 3) % 67) as f32 - 33.) / 64.);
                let at = (off + m) * cols + k;
                input[at / 2] |= u32::from(v.to_bits()) << (16 * (at % 2));
            }
        }
    }
    let ids = [
        arena.upload_u32(&input)?,
        arena.upload_u32(&image)?,
        arena.upload_u32(&map)?,
    ];
    let packed_words = Zeroizing::new(
        packed
            .chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    drop(packed);
    let output =
        arena.allocate_zeroed((if down { g::PART_BYTES } else { g::GU_BYTES }) as usize / 4)?;
    let statuses = Zeroizing::new(vec![1u32; g::STATUS_WORDS]);
    let sa = arena.upload_u32(&statuses)?;
    let fa = arena.upload_u32(&[weight_fault])?;
    let mut old_words = *words;
    old_words[3] = 0;
    for tile in [8usize, 16] {
        old_words[11] = tile as u32;
        crate::strata_matrix::cohorts::small_direct::launch(
            client,
            [
                u::handle(arena, ids[0])?,
                u::handle(arena, ids[1])?,
                u::handle(arena, output)?,
                u::handle(arena, ids[2])?,
                u::handle(arena, sa)?,
                u::handle(arena, fa)?,
            ],
            &old_words,
            down,
            tile,
        )?;
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let reference = snapshot(arena, output)?;
    let status_reference = snapshot(arena, sa)?;
    let fault_reference = snapshot(arena, fa)?;
    for (id, expected) in [(ids[0], &*input), (ids[1], &*image), (ids[2], &*map)] {
        check_source(arena, id, expected)?;
    }
    erase_output(client, arena, output)?;
    arena.reset()?;
    // Original phase has completed and every original owner was erased. The
    // same bounded arena now owns the fused phase; no second400MiB output or
    // simultaneous original-F16 and compact slab consumes unpriced VRAM.
    let ids = [arena.upload_u32(&input)?, arena.upload_u32(&map)?];
    let compact = arena.upload_u32(&packed_words)?;
    let output =
        arena.allocate_zeroed((if down { g::PART_BYTES } else { g::GU_BYTES }) as usize / 4)?;
    let sb = arena.upload_u32(&statuses)?;
    let fb = arena.allocate_zeroed(1)?;
    let table_before = u::read_handle_prefix_u32(arena, table.clone(), 8192)?;
    sanitize(arena)?;
    for tile in [8usize, 16, 32] {
        words[11] = tile as u32;
        let selected = if pipeline { launch } else { launch_original };
        selected(
            client,
            [
                u::handle(arena, ids[0])?,
                u::handle(arena, compact)?,
                u::handle(arena, output)?,
                u::handle(arena, ids[1])?,
                u::handle(arena, sb)?,
                u::handle(arena, fb)?,
                table.clone(),
            ],
            &words,
            down,
            tile,
            dtype,
        )?;
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let checked = compare_reference(arena, output, &reference)?;
    drop(reference);
    compare_reference(arena, sb, &status_reference)?;
    compare_reference(arena, fb, &fault_reference)?;
    for (id, expected) in [
        (ids[0], &*input),
        (ids[1], &*map),
        (compact, &*packed_words),
    ] {
        check_source(arena, id, expected)?;
    }
    let table_after = u::read_handle_prefix_u32(arena, table.clone(), 8192)?;
    let table_unchanged = *table_before == *table_after;
    drop(table_before);
    drop(table_after);
    sanitize(arena)?;
    if !table_unchanged {
        return Err("quant cohort full immutable public table changed".into());
    }
    let mut result = serde_json::json!({"family":"quant-cohort-original-format-matrix","dtype":dtype,"down":down,"groups":8,"counts":counts,"poison_and_invalid_route":poison,
        "actual_fused_quantized_cmma":true,"original_decoded_image_and_gpu_product_oracle":true,"all_eight_slots_and_nonzero_prefix":true,
        "exact_original8_rounding_and_fault_gate":true,"chronological_k16":true,"bit_mismatches":0,"full_output_words_checked":checked,
        "full_owner_status_fault_sources_and_tail_checked":true,"single_original_gpu_output_owner":true,"original_full_arena_reset_between_phases":true,"full_public_table_owner_unchanged":true,"full_output_owner_zero_verified_between_programs":true});
    if pipeline {
        result["actual_producer_consumer_pipeline"] = true.into();
        result["workgroup_output_columns"] = 32.into();
        result["producer_subgroups"] = 2.into();
        result["consumer_subgroups"] = 2.into();
    }
    Ok(result)
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> Checked<Vec<serde_json::Value>> {
    cases_mode(client, arena, file, false)
}
pub(crate) fn pipeline_cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> Checked<Vec<serde_json::Value>> {
    cases_mode(client, arena, file, true)
}
fn cases_mode(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
    pipeline: bool,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("quant cohort original empty public audit arena required".into());
    }
    let mut table_words = Vec::new();
    for dtype in [18u32, 21, 22] {
        table_words.extend(crate::quant_matvec::iq_moe_integer::original_grid_words(
            dtype,
        ));
    }
    if table_words.len() != 2816 {
        return Err("quant cohort original table extent".into());
    }
    table_words.resize(8192, 0);
    let table =
        client.create_public_owned(table_words.into_iter().flat_map(u32::to_le_bytes).collect());
    if table.size_in_used() != 32768 {
        return Err("quant cohort actual public table owner".into());
    }
    let mut out = Vec::new();
    for dtype in [18u32, 21, 22, 23, 20, 42] {
        for poison in [false, true] {
            let result = one(client, arena, file, dtype, poison, &table, pipeline).map_err(
                |e| -> Box<dyn std::error::Error> {
                    format!("quant cohort dtype{dtype} poison{poison}: {e}").into()
                },
            );
            arena.reset()?;
            out.push(result?);
        }
    }
    Ok(out)
}
