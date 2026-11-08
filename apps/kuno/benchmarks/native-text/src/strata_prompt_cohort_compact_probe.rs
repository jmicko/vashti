//! Exact ORIGINAL encoded bytes through real compact MAIN producers, including
//! all8 logical slots, distinct model experts, RAW/resident mixed roles and
//! malformed source/whole-owner controls. No decoded mirror authorizes output.
use super::super::{allocation, groups, storage};
use super::*;
use crate::{private_arena::PrivateArena, resident_util as u};
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
    dtype: u32,
    table: &Handle,
    control: &str,
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
    let invalid_slot = match control {
        "source_extent" | "source_tier" => Some(2usize),
        "resident_sentinel" => Some(1),
        "map_members" => Some(3),
        _ => None,
    };
    let rejected_owner = matches!(control, "raw_owner" | "packed_owner");
    let is_valid = control.starts_with("valid");
    let all_resident = control == "valid_resident";
    let all_raw = control == "valid_raw";
    let raw_slot = |slot: usize| {
        if all_resident || (!all_raw && slot % 2 == 1) {
            8
        } else if all_raw {
            (slot * 5 + 3) % 8
        } else {
            slot / 2
        }
    };
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
        if raw_slot(slot) < 8 {
            let at = raw_slot(slot) * stride;
            if down {
                raw[at + 2 * gu_bytes..at + stride].copy_from_slice(&gate);
            } else {
                raw[at..at + gu_bytes].copy_from_slice(&gate);
                raw[at + gu_bytes..at + 2 * gu_bytes].copy_from_slice(&up);
            }
        }
        if !rejected_owner && invalid_slot != Some(slot) {
            let first = slot * stride / 4;
            for (offset, payload) in if down {
                vec![(2 * gu_bytes / 4, &gate)]
            } else {
                vec![(0, &gate), (gu_bytes / 4, &up)]
            } {
                for (i, value) in payload.chunks_exact(4).enumerate() {
                    expected[first + offset + i] = u32::from_le_bytes(value.try_into().unwrap());
                }
            }
        }
        handles.push(super::super::super::public_numeric::public(
            client,
            gate.clone(),
        )?);
        handles.push(super::super::super::public_numeric::public(
            client,
            up.clone(),
        )?);
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
            let host = all_raw
                || (!all_resident && slot % 2 == 0 && (down || slot % 4 == 0 || *role == 1));
            addresses[at] = address[0] as u32;
            addresses[at + 1] = (address[0] >> 32) as u32;
            addresses[at + 2] = (bytes / 4) as u32;
            addresses[at + 3] = u32::from(host);
            if slot == 2 && control == "source_extent" {
                addresses[at + 2] += 4;
            }
            if slot == 2 && control == "source_tier" {
                addresses[at + 3] = 2;
            }
            if slot == 1 && control == "resident_sentinel" {
                addresses[at + 3] = 1;
            }
        }
    }
    let address_handle = super::super::super::public_numeric::public(
        client,
        addresses.into_iter().flat_map(u32::to_le_bytes).collect(),
    )?;
    let mut routing = Zeroizing::new(vec![0u32; groups::LOGICAL + 10]);
    routing[..3].copy_from_slice(&[1, 0, 1]);
    for route in 0..10 {
        routing[groups::LOGICAL + route] = route as u32;
        routing[1540 + route] = route as u32;
    }
    let mut map = groups::group(&routing, 1)?;
    groups::validate(&map, 1)?;
    if control == "map_members" {
        map[516 + 3] = 2;
    }
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
    words[0] = 1;
    words[1] = cols as u32;
    words[2] = if down { rows as u32 } else { 2 * rows as u32 };
    words[3] = row_bytes as u32;
    words[5] = if down { 2 } else { 0 };
    words[6] = (gu_bytes / 4) as u32;
    words[7] = (gu_bytes / 4) as u32;
    words[8] = (stride / 4) as u32;
    words[9] = 8;
    words[10] = 1;
    for slot in 0..8 {
        words[16 + slot * 4..20 + slot * 4].copy_from_slice(&[
            slot as u32,
            1,
            slot as u32 * 8,
            raw_slot(slot) as u32,
        ]);
    }
    validate_words(&words)?;
    let mut args = KernelArguments::new().with_buffer(handles[0].clone().binding());
    if !down {
        args = args.with_buffer(handles[1].clone().binding());
    }
    args = args
        .with_buffers(vec![
            u::handle(arena, map_id)?.binding(),
            if control == "raw_owner" {
                u::handle(arena, raw_id)?.offset_end(256).binding()
            } else {
                u::handle(arena, raw_id)?.binding()
            },
            if control == "packed_owner" {
                u::handle(arena, output)?.offset_end(256).binding()
            } else {
                u::handle(arena, output)?.binding()
            },
            u::handle(arena, fault)?.binding(),
            table.clone().binding(),
            address_handle.binding(),
        ])
        .with_read_dependencies(handles.iter().cloned().map(Handle::binding).collect());
    super::launch(client, super::CompactOp { dtype, down }, args, &words)?;
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
            super::super::probe::sanitize(arena)?;
            if !equal {
                return Err("public compact exact original bits/source/full owner tail".into());
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
        super::super::probe::sanitize(arena)?;
        if !equal {
            return Err("public cohort original public weight source changed".into());
        }
    }
    let data = u::read_handle_prefix_u32(
        arena,
        arena.handle(fault)?.clone(),
        arena.layout(fault)?.owned_bytes / 4,
    )?;
    let expected_fault = if is_valid { 0 } else { 128 };
    let zero =
        data.first().copied() == Some(expected_fault) && data.iter().skip(1).all(|word| *word == 0);
    drop(data);
    super::super::probe::sanitize(arena)?;
    if !zero {
        return Err("public cohort multi-slot source fault".into());
    }
    Ok(
        serde_json::json!({"family":"cohort-original-encoded-compact", "dtype":dtype,
        "down":down,"control":control,"groups":8,"distinct_original_model_experts":8,
        "actual_compact_copy":true,"workgroup_invocations":256,"words_per_invocation":4,
        "simultaneous_host_gpu_roles":!all_resident&&!all_raw,"mixed_gate_up_residency":!down&&!all_resident&&!all_raw,
        "all_resident":all_resident,"all_raw":all_raw,"actual_raw_slot_permutation":all_raw,
        "logical_slots_0_through_7":true,"resident_sentinel8":!all_raw,
        "original_map_builder_and_validator":true,"bit_mismatches":0,
        "fault":expected_fault,"full_owner_sources_and_unused_compact_tail_checked":true}),
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
    let table = super::super::super::public_numeric::public(
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
        for control in ["valid", "valid_resident", "valid_raw"] {
            let result = one(client, arena, file, dtype, &table, control);
            arena.reset()?;
            results.push(result?);
        }
    }
    for dtype in [q::IQ3_S, q::IQ4_NL] {
        for control in [
            "raw_owner",
            "packed_owner",
            "source_extent",
            "source_tier",
            "resident_sentinel",
            "map_members",
        ] {
            let result = one(client, arena, file, dtype, &table, control);
            arena.reset()?;
            results.push(result?);
        }
    }
    Ok(results)
}
