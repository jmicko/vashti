//! All8 output slots, distinct ORIGINAL expert weights, simultaneous GPU/HOST
//! and mixed GU-role residency in the same genuine dequantization dispatch.
use super::*;
use crate::{private_arena::PrivateArena, resident_util as u};
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
    dtype: u32,
    table: &Handle,
) -> Checked<serde_json::Value> {
    let down = matches!(dtype, q::IQ4_NL | q::Q2_0);
    let rows = if down { 2560usize } else { 640 };
    let cols = if down { 640usize } else { 2560 };
    let suffix = if down {
        "ffn_down_exps.weight"
    } else {
        "ffn_gate_exps.weight"
    };
    let (name, tensor) = file
        .tensors()
        .iter()
        .find(|(name, t)| name.ends_with(suffix) && t.dtype().gguf_id() == dtype)
        .ok_or("public cohort original format missing")?;
    let row_bytes = tensor.dtype().row_bytes(cols as u64)? as usize;
    let bytes = rows * row_bytes;
    let gu_bytes = if down { 256 } else { bytes };
    let down_bytes = if down { bytes } else { 256 };
    let stride = 2 * gu_bytes + down_bytes;
    let mut raw = Zeroizing::new(vec![0u8; allocation::RAW_RING_BYTES as usize]);
    let mut expected = Zeroizing::new(vec![0u32; storage::WORKING_BYTES as usize / 4]);
    let mut handles = Vec::new();
    let mut original_sources = Vec::new();
    for slot in 0..8 {
        let mut gate = vec![0u8; bytes];
        file.tensor(name)
            .unwrap()
            .read_range((slot * bytes) as u64, &mut gate)?;
        let mut up = gate.clone();
        if !down {
            file.tensor(&name.replace(suffix, "ffn_up_exps.weight"))
                .ok_or("public cohort original up missing")?
                .read_range((slot * bytes) as u64, &mut up)?;
        }
        if slot % 2 == 0 {
            let at = slot / 2 * stride;
            if down {
                raw[at + 2 * gu_bytes..at + stride].copy_from_slice(&gate);
            } else {
                raw[at..at + gu_bytes].copy_from_slice(&gate);
                raw[at + gu_bytes..at + 2 * gu_bytes].copy_from_slice(&up);
            }
        }
        let mut a = Zeroizing::new(vec![0f32; rows * cols]);
        let mut b = Zeroizing::new(vec![0f32; rows * cols]);
        tensor.dtype().decode(&gate, &mut a)?;
        tensor.dtype().decode(&up, &mut b)?;
        let output_rows = if down { rows } else { rows * 2 };
        let first = slot * DQ_WORDS + if down { GU_WORDS } else { 0 };
        for row in 0..output_rows {
            let original = if down || row % 2 == 0 { &a } else { &b };
            let r = if down { row } else { row / 2 };
            for col in (0..cols).step_by(2) {
                let i = r * cols + col;
                expected[first + (row * cols + col) / 2] =
                    u32::from(half::f16::from_f32(original[i]).to_bits())
                        | (u32::from(half::f16::from_f32(original[i + 1]).to_bits()) << 16);
            }
        }
        handles.push(super::super::public_numeric::public(client, gate.clone())?);
        handles.push(super::super::public_numeric::public(client, up.clone())?);
        original_sources.extend([gate, up]);
    }
    let physical = client
        .native_strata_source_addresses(handles.clone())
        .map_err(u::fault)?;
    let mut addresses = vec![0u32; 8192];
    for slot in 0..8 {
        for role in if down {
            &[2usize][..]
        } else {
            &[0usize, 1][..]
        } {
            let index = slot * 2 + usize::from(*role == 1);
            let address = physical[index];
            let at = (*role * 512 + slot) * 4;
            let host = slot % 2 == 0 && (down || slot % 4 == 0 || *role == 1);
            addresses[at] = address[0] as u32;
            addresses[at + 1] = (address[0] >> 32) as u32;
            addresses[at + 2] = (bytes / 4) as u32;
            addresses[at + 3] = u32::from(host);
        }
    }
    let address_handle = super::super::public_numeric::public(
        client,
        addresses.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    const TOTAL: usize = 256;
    let counts = [32usize, 33, 63, 64, 65, 127, 128, 255];
    let mut raw_routes = Zeroizing::new(vec![0u32; groups::LOGICAL + TOTAL * 10]);
    raw_routes[..3].copy_from_slice(&[TOTAL as u32, 0, TOTAL as u32]);
    for token in 0..TOTAL {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let logical = if rank < 8 && token < counts[rank] {
                400 + rank
            } else {
                1 + rank
            };
            raw_routes[groups::LOGICAL + route] = logical as u32;
            raw_routes[1540 + route] = if logical >= 400 {
                (logical - 400) as u32
            } else {
                (100 + logical) as u32
            };
        }
    }
    let map = groups::group(&raw_routes, TOTAL)?;
    let first = groups::validate(&map, TOTAL)? - 8;
    let map_id = arena.upload_u32(&map)?;
    let raw_words = Zeroizing::new(
        raw.chunks_exact(4)
            .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    let raw_id = arena.upload_u32(&raw_words)?;
    let output = arena.allocate_zeroed(storage::WORKING_BYTES as usize / 4)?;
    let fault = arena.allocate_zeroed(1)?;
    let mut words = Zeroizing::new([0u32; 48]);
    words[0] = TOTAL as u32;
    words[4] = first as u32;
    words[1] = cols as u32;
    words[2] = if down { rows as u32 } else { 2 * rows as u32 };
    words[3] = row_bytes as u32;
    words[5] = if down { 2 } else { 0 };
    words[6] = (gu_bytes / 4) as u32;
    words[7] = (gu_bytes / 4) as u32;
    words[8] = (stride / 4) as u32;
    words[9] = 8;
    words[10] = 255;
    let mut prefix = 0u32;
    for slot in 0..8 {
        words[16 + slot * 4..20 + slot * 4].copy_from_slice(&[
            (first + slot) as u32,
            counts[slot] as u32,
            prefix,
            if slot % 2 == 0 { slot as u32 / 2 } else { 8 },
        ]);
        prefix += counts[slot].div_ceil(16) as u32 * 16;
    }
    validate_words(&words)?;
    let mut args = KernelArguments::new().with_buffer(handles[0].clone().binding());
    if !down {
        args = args.with_buffer(handles[1].clone().binding());
    }
    args = args
        .with_buffers(vec![
            u::handle(arena, map_id)?.binding(),
            u::handle(arena, raw_id)?.binding(),
            u::handle(arena, output)?.binding(),
            u::handle(arena, fault)?.binding(),
            table.clone().binding(),
            address_handle.binding(),
        ])
        .with_read_dependencies(handles.iter().cloned().map(Handle::binding).collect());
    launch(
        client,
        RegularOp::Image { dtype, down },
        [
            (words[1] as usize * words[2] as usize / Op::selected_dequant_values())
                .div_ceil(Op::selected_dequant_lanes()),
            1,
            8,
        ],
        args,
        &words,
    )?;
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    for (id, reference) in [(output, &*expected), (raw_id, &*raw_words), (map_id, &*map)] {
        let whole = arena.handle(id)?.clone();
        let owned = arena.layout(id)?.owned_bytes / 4;
        for first in (0..owned).step_by(4 * 1024 * 1024) {
            let count = (4 * 1024 * 1024).min(owned - first);
            let view = whole
                .clone()
                .offset_start((first * 4) as u64)
                .offset_end(((owned - first - count) * 4) as u64);
            let data = u::read_handle_prefix_u32(arena, view, count)?;
            let equal = data
                .iter()
                .enumerate()
                .all(|(i, v)| *v == reference.get(first + i).copied().unwrap_or(0));
            drop(data);
            super::probe::sanitize(arena)?;
            if !equal {
                return Err("public cohort multi-slot exact original image/source/tail".into());
            }
        }
    }
    for (handle, source) in handles.iter().zip(&original_sources) {
        let words = source.len() / 4;
        let actual = u::read_handle_prefix_u32(arena, handle.clone(), words)?;
        let equal = actual
            .iter()
            .zip(source.chunks_exact(4))
            .all(|(a, b)| *a == u32::from_le_bytes(b.try_into().unwrap()));
        drop(actual);
        super::probe::sanitize(arena)?;
        if !equal {
            return Err("public cohort original public weight source changed".into());
        }
    }
    let data = u::read_handle_prefix_u32(
        arena,
        arena.handle(fault)?.clone(),
        arena.layout(fault)?.owned_bytes / 4,
    )?;
    let zero = data.iter().all(|word| *word == 0);
    drop(data);
    super::probe::sanitize(arena)?;
    if !zero {
        return Err("public cohort multi-slot source fault".into());
    }
    Ok(
        serde_json::json!({"family":"regular-cohort-multi-original-dequant","dtype":dtype,"down":down,"groups":8,"distinct_original_model_experts":8,"actual_direct8":cfg!(feature="strata-dequant-direct8"),"values_per_invocation":Op::selected_dequant_values(),"workgroup_invocations":Op::selected_dequant_lanes(),"simultaneous_host_gpu_roles":true,"mixed_gate_up_residency":!down,"all_output_image_slots_0_through_7":true,"resident_sentinel8":true,"original_map_builder_and_validator":true,"bit_mismatches":0,"full_owner_sources_and_unused_image_tail_checked":true}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("public cohort multi-DQ original empty audit arena".into());
    }
    let mut table = Vec::new();
    for dtype in [q::IQ3_XXS, q::IQ3_S, q::IQ2_S] {
        table.extend(q::iq_moe_integer::original_grid_words(dtype));
    }
    table.resize(8192, 0);
    let table = super::super::public_numeric::public(
        client,
        table.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    let mut results = Vec::new();
    for dtype in [
        q::IQ3_XXS,
        q::IQ3_S,
        q::IQ2_S,
        q::IQ4_XS,
        q::IQ4_NL,
        q::Q2_0,
    ] {
        let result = one(client, arena, file, dtype, &table);
        arena.reset()?;
        results.push(result?);
    }
    Ok(results)
}
