//! Additive fixed-public packed LARGE layer activation oracle. The original
//! per-expert ActivateCheckGu GPU program supplies the exact row-major result;
//! an independent host16x16 enumeration permutes it for comparison, including
//! every Pad32 zero gutter and complete physical owner. Original probes stay.

use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(super) fn sanitize(arena: &PrivateArena) -> Checked<()> {
    let receipt = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
    let bytes = receipt
        .upload_bytes
        .checked_add(receipt.uniform_bytes)
        .and_then(|v| v.checked_add(receipt.readback_bytes))
        .ok_or("public cohort IO audit overflow")?;
    if bytes == 0 || receipt.inspected_bytes != bytes || receipt.inspected_buffers == 0 {
        return Err("public cohort complete transient IO erasure".into());
    }
    Ok(())
}
fn read(arena: &PrivateArena, id: BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let result = u::read_handle_prefix_u32(
        arena,
        arena.handle(id)?.clone(),
        arena.layout(id)?.owned_bytes / 4,
    )?;
    sanitize(arena)?;
    Ok(result)
}

fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    poison: bool,
) -> Checked<serde_json::Value> {
    // This probe covers ONLY the new large layer activation, never standalone
    // gather, regular cohorts or FP16 GU input.
    let total_tokens = 512;
    let counts = [256usize, 257, 264, 265, 287, 288, 383, 511];
    let mut raw = Zeroizing::new(vec![0u32; groups::LOGICAL + total_tokens * 10]);
    raw[..3].copy_from_slice(&[total_tokens as u32, 0, total_tokens as u32]);
    for token in 0..total_tokens {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let logical = if rank < 8 && token < counts[rank] {
                400 + rank
            } else {
                1 + rank
            };
            raw[groups::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let mut map = groups::group(&raw, total_tokens)?;
    let total = groups::validate(&map, total_tokens)?;
    let first = total - 8;
    if poison {
        let offset = map[1028 + first + 6] as usize;
        map[1540 + offset] = (total_tokens * 10) as u32;
    }
    let mut words = Zeroizing::new([0u32; WORDS]);
    words[0] = total_tokens as u32;
    words[1] = 2560;
    words[2] = 1280;
    words[4] = first as u32;
    words[6] = 100;
    words[7] = 100;
    words[8] = 300;
    words[9] = 8;
    words[10] = *counts.iter().max().unwrap() as u32;
    words[12] = 0;
    words[13] = 1;
    words[14] = 1;
    let mut prefix = 0;
    for (i, members) in counts.into_iter().enumerate() {
        if map[516 + first + i] as usize != members {
            return Err("public cohort authentic fixture groups".into());
        }
        words[16 + i * 4..20 + i * 4].copy_from_slice(&[
            (first + i) as u32,
            members as u32,
            prefix,
            8,
        ]);
        prefix += members.div_ceil(32) as u32 * 32;
    }
    crate::strata_prompt::experts::large::packed::validate_words(&words)?;
    let source_words = groups::GU_BYTES as usize / 4;
    let mut source = Zeroizing::new(
        (0..source_words)
            .map(|i| (((i % 41) as f32 - 20.0) / 32.0).to_bits())
            .collect::<Vec<_>>(),
    );
    if poison {
        source[0] = f32::NAN.to_bits();
    }
    let source0 = arena.upload_u32(&source)?;
    let source1 = arena.upload_u32(&source)?;
    let map_id = arena.upload_u32(&map)?;
    let bytes = H_BYTES;
    let original = arena.allocate_zeroed(bytes as usize / 4)?;
    let selected = arena.allocate_zeroed(bytes as usize / 4)?;
    let status0 = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let status1 = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let fault0 = arena.allocate_zeroed(1)?;
    let fault1 = arena.allocate_zeroed(1)?;
    let original_op = Op::ActivateCheckGu;
    for i in 0..8 {
        let group = first + i;
        let prefix = words[18 + i * 4] as u64;
        let rowbytes = 1280;
        let output = u::handle(arena, original)?
            .offset_start(prefix * rowbytes)
            .offset_end(0);
        let input = u::handle(arena, source0)?
            .offset_start(prefix * 5120)
            .offset_end(0);
        let mut args = KernelArguments::new().with_buffers(vec![
            u::handle(arena, map_id)?.binding(),
            input.binding(),
            output.binding(),
        ]);
        args = args.with_buffer(u::handle(arena, status0)?.binding());
        args = args.with_buffer(u::handle(arena, fault0)?.binding());
        let mut old = [0u32; 16];
        old[0] = total_tokens as u32;
        old[1] = 2560;
        old[2] = 1280;
        old[4] = group as u32;
        old[12] = 0;
        args = args.with_info(MetadataBindingInfo::custom(
            old.chunks_exact(2)
                .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                .collect(),
        ));
        let grid = member_dispatches(counts[i], total_tokens)?[1];
        unsafe {
            client.launch_unchecked(
                Box::new(original_op),
                CubeCount::Static(grid as u32, 1, 1),
                args,
            );
        }
    }
    let mut args = KernelArguments::new().with_buffers(vec![
        u::handle(arena, map_id)?.binding(),
        u::handle(arena, source1)?.binding(),
        u::handle(arena, selected)?.binding(),
    ]);
    args = args.with_buffer(u::handle(arena, status1)?.binding());
    args = args.with_buffer(u::handle(arena, fault1)?.binding());
    {
        use crate::strata_prompt::experts::large::{LargeOp, packed};
        packed::launch(
            client,
            LargeOp::Activate,
            [(prefix as usize * 320).div_ceil(256), 1, 1],
            args,
            &words,
        )?;
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let a = read(arena, original)?;
    let b = read(arena, selected)?;
    let expected = packed_hidden_reference(&a, &words)?;
    if *expected != *b {
        return Err("packed LARGE activation independent original bits/gutters/tail".into());
    }
    if *read(arena, status0)? != *read(arena, status1)? {
        return Err("public cohort paired status".into());
    }
    for id in [fault0, fault1] {
        let actual = read(arena, id)?;
        if actual
            .iter()
            .enumerate()
            .any(|(i, x)| *x != if i == 0 && poison { 128 } else { 0 })
        {
            return Err("public cohort paired fault".into());
        }
    }
    if poison {
        source[0] = 0;
    }
    for id in [source0, source1] {
        let actual = read(arena, id)?;
        if actual
            .iter()
            .enumerate()
            .any(|(i, v)| *v != source.get(i).copied().unwrap_or(0))
        {
            return Err("public cohort source full owner changed".into());
        }
    }
    let actual = read(arena, map_id)?;
    if actual
        .iter()
        .enumerate()
        .any(|(i, v)| *v != map.get(i).copied().unwrap_or(0))
    {
        return Err("public cohort route owner changed".into());
    }
    Ok(
        serde_json::json!({"family":"large-packed-cohort-paired-original-activation","activate":true,"flat_layer_tiles":true,"mixed_f16_input":false,"nonfinite_and_invalid_route":poison,"groups":8,"counts":counts,"original_map_builder_and_validator":true,"packed_large_format":1,"input_gu_is_original_f32":true,"hidden_tile_rows":16,"reduction_tile_columns":16,"independent_original_gpu_tile_enumeration":true,"pad32_zero_gutters_checked":true,"actual_original_and_cohort_gpu_programs":true,"bit_mismatches":0,"full_output_words_checked":a.len(),"full_owner_status_fault_sources_and_tail_checked":true,"disjoint_x_hidden":true}),
    )
}
/// Enumerate physical tiles rather than reusing the production offset formula.
/// Each cohort is independently permuted; its original parent prefix is applied
/// outside the tile order. No row from a neighboring cohort can stand in for a
/// gutter. Untouched owning-tail words must remain zero in BOTH GPU outputs.
fn packed_hidden_reference(original: &[u32], words: &[u32; WORDS]) -> Checked<Zeroizing<Vec<u32>>> {
    let mut expected = Zeroizing::new(vec![0u32; original.len()]);
    let mut logical_end = 0;
    for group in 0..words[9] as usize {
        let members = words[17 + group * 4] as usize;
        let prefix = words[18 + group * 4] as usize;
        let padded = members.div_ceil(32) * 32;
        let first = prefix * 320;
        let end = (prefix + padded) * 320;
        if first != logical_end || end > original.len() {
            return Err("packed LARGE independent original hidden segment extent".into());
        }
        if original[first + members * 320..end]
            .iter()
            .any(|word| *word != 0)
        {
            return Err("original activation Pad32 gutter not zero".into());
        }
        let mut destination = first;
        for row_block in (0..padded).step_by(16) {
            for column_block in (0..640).step_by(16) {
                for tile_row in 0..16 {
                    for pair in 0..8 {
                        expected[destination] = original
                            [first + (row_block + tile_row) * 320 + column_block / 2 + pair];
                        destination += 1;
                    }
                }
            }
        }
        if destination != end {
            return Err("packed LARGE independent hidden tile coverage".into());
        }
        logical_end = end;
    }
    if original[logical_end..].iter().any(|word| *word != 0) {
        return Err("original activation unused full owner tail changed".into());
    }
    Ok(expected)
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("packed LARGE activation empty audited arena required".into());
    }
    let mut reports = Vec::new();
    for poison in [false, true] {
        let result = one(client, arena, poison);
        arena.reset()?;
        reports.push(result?);
    }
    Ok(reports)
}
