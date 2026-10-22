//! Fixed-public paired original/cohort producers with authentic routing maps.
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
    activate: bool,
    mixed: bool,
    poison: bool,
) -> Checked<serde_json::Value> {
    const TOTAL: usize = 32;
    let counts = [1usize, 3, 7, 8, 9, 16, 17, 31];
    let mut raw = Zeroizing::new(vec![0u32; groups::LOGICAL + TOTAL * 10]);
    raw[..3].copy_from_slice(&[TOTAL as u32, 0, TOTAL as u32]);
    for token in 0..TOTAL {
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
    let mut map = groups::group(&raw, TOTAL)?;
    let total = groups::validate(&map, TOTAL)?;
    let first = total - 8;
    if poison {
        let offset = map[1028 + first + 6] as usize;
        map[1540 + offset] = (TOTAL * 10) as u32;
    }
    let mut words = Zeroizing::new([0u32; WORDS]);
    words[0] = TOTAL as u32;
    words[1] = 2560;
    words[2] = 1280;
    words[4] = first as u32;
    words[6] = 100;
    words[7] = 100;
    words[8] = 300;
    words[9] = 8;
    words[10] = 31;
    words[12] = u32::from(mixed);
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
        let tile = if members <= 8 { 8 } else { 16 };
        prefix += members.div_ceil(tile) as u32 * tile as u32;
    }
    validate_words(&words)?;
    let source_words = if activate {
        groups::GU_BYTES as usize / 4
    } else {
        TOTAL * 2560 / if mixed { 2 } else { 1 }
    };
    let mut source = Zeroizing::new(
        (0..source_words)
            .map(|i| {
                if mixed {
                    let a = half::f16::from_f32((((i * 2) % 41) as f32 - 20.0) / 32.0);
                    let b = half::f16::from_f32((((i * 2 + 1) % 41) as f32 - 20.0) / 32.0);
                    u32::from(a.to_bits()) | (u32::from(b.to_bits()) << 16)
                } else {
                    (((i % 41) as f32 - 20.0) / 32.0).to_bits()
                }
            })
            .collect::<Vec<_>>(),
    );
    if poison {
        source[0] = if mixed {
            (source[0] & 0xffff0000) | 0x7e00
        } else {
            f32::NAN.to_bits()
        };
    }
    let source0 = arena.upload_u32(&source)?;
    let source1 = arena.upload_u32(&source)?;
    let map_id = arena.upload_u32(&map)?;
    let bytes = if activate { H_BYTES } else { X_BYTES };
    let original = arena.allocate_zeroed(bytes as usize / 4)?;
    let selected = arena.allocate_zeroed(bytes as usize / 4)?;
    let status0 = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let status1 = arena.allocate_zeroed(groups::STATUS_WORDS)?;
    let fault0 = arena.allocate_zeroed(1)?;
    let fault1 = arena.allocate_zeroed(1)?;
    let original_op = if activate {
        Op::ActivateCheckGu
    } else {
        Op::Gather
    };
    for i in 0..8 {
        let group = first + i;
        let prefix = words[18 + i * 4] as u64;
        let rowbytes = if activate { 1280 } else { 5120 };
        let output = u::handle(arena, original)?
            .offset_start(prefix * rowbytes)
            .offset_end(0);
        let input = if activate {
            u::handle(arena, source0)?
                .offset_start(prefix * 5120)
                .offset_end(0)
        } else {
            u::handle(arena, source0)?
        };
        let mut args = KernelArguments::new().with_buffers(vec![
            u::handle(arena, map_id)?.binding(),
            input.binding(),
            output.binding(),
        ]);
        if activate {
            args = args.with_buffer(u::handle(arena, status0)?.binding());
        }
        args = args.with_buffer(u::handle(arena, fault0)?.binding());
        let mut old = [0u32; 16];
        old[0] = TOTAL as u32;
        old[1] = 2560;
        old[2] = 1280;
        old[4] = group as u32;
        old[12] = u32::from(mixed);
        args = args.with_info(MetadataBindingInfo::custom(
            old.chunks_exact(2)
                .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                .collect(),
        ));
        let grid = member_dispatches(counts[i], TOTAL)?[usize::from(activate)];
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
    if activate {
        args = args.with_buffer(u::handle(arena, status1)?.binding());
    }
    args = args.with_buffer(u::handle(arena, fault1)?.binding());
    launch(
        client,
        if activate {
            CohortOp::Activate
        } else {
            CohortOp::Gather
        },
        [
            (31 * if activate { 320usize } else { 1280usize }).div_ceil(256),
            1,
            8,
        ],
        args,
        &words,
    )?;
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let a = read(arena, original)?;
    let b = read(arena, selected)?;
    if *a != *b {
        return Err("public cohort paired producer original bits/tail".into());
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
    if activate && poison {
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
        serde_json::json!({"family":"cohort-paired-original-producer","activate":activate,"mixed_f16_input":mixed,"nonfinite_and_invalid_route":poison,"groups":8,"counts":counts,"original_map_builder_and_validator":true,"actual_original_and_cohort_gpu_programs":true,"bit_mismatches":0,"full_output_words_checked":a.len(),"full_owner_status_fault_sources_and_tail_checked":true,"disjoint_x_hidden":true}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("public cohort original empty audit arena required".into());
    }
    let mut reports = Vec::new();
    for (activate, mixed) in [(false, false), (false, true), (true, false)] {
        for poison in [false, true] {
            let result = one(client, arena, activate, mixed, poison);
            arena.reset()?;
            reports.push(result?);
        }
    }
    Ok(reports)
}
