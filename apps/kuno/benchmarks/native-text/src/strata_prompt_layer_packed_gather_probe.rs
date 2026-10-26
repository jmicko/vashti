//! Additive packed-vs-original full-owner layer producer controls.
use super::*;
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use cubecl::prelude::ArrayArg;
use zeroize::Zeroizing;
type Checked<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WINDOW: usize = 4 * 1024 * 1024;
fn scrub(arena: &PrivateArena) -> Checked<()> {
    let r = cubecl::future::block_on(arena.client()?.sanitize_transient_io_audited())?;
    let bytes = r.upload_bytes + r.uniform_bytes + r.readback_bytes;
    if bytes == 0 || bytes != r.inspected_bytes || r.inspected_buffers == 0 {
        return Err("layer producer transient IO erasure incomplete".into());
    }
    Ok(())
}
fn snapshot(arena: &PrivateArena, id: BufferId) -> Checked<Zeroizing<Vec<u32>>> {
    let words = arena.layout(id)?.owned_words;
    if words * 4 > 400 * 1024 * 1024 {
        return Err("layer producer host reference cap".into());
    }
    let mut out = Zeroizing::new(vec![0; words]);
    for start in (0..words).step_by(WINDOW) {
        let n = WINDOW.min(words - start);
        let view = arena
            .handle(id)?
            .clone()
            .offset_start((start * 4) as u64)
            .offset_end(((words - start - n) * 4) as u64);
        let v = u::read_handle_prefix_u32(arena, view, n)?;
        out[start..start + n].copy_from_slice(&v);
        drop(v);
        scrub(arena)?;
    }
    Ok(out)
}
fn one(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    count: usize,
    half: bool,
    poison: bool,
) -> Checked<serde_json::Value> {
    let mut raw = Zeroizing::new(vec![0u32; groups::LOGICAL + count * 10]);
    raw[..3].copy_from_slice(&[count as u32, 0, count as u32]);
    for token in 0..count {
        for rank in 0..10 {
            let route = token * 10 + rank;
            let logical = if rank < 3 {
                rank
            } else if rank < 5 {
                3 + token % 20
            } else {
                23 + (token * 11 + rank * 13) % 489
            };
            raw[groups::LOGICAL + route] = logical as u32;
            raw[1540 + route] = ((logical * 13 + 17) % 512) as u32;
        }
    }
    let mut map = groups::group(&raw, count)?;
    let (n, padded) = groups::validate_layer(&map, count)?;
    if poison {
        map[1540 + count * 10 - 1] = (count * 10) as u32;
    }
    let mut source = Zeroizing::new(
        (0..count * 2560 / if half { 2 } else { 1 })
            .map(|i| {
                let value = |at: usize| ((at % 41) as f32 - 20.0) / 32.0;
                if half {
                    u32::from(half::f16::from_f32(value(i * 2)).to_bits())
                        | (u32::from(half::f16::from_f32(value(i * 2 + 1)).to_bits()) << 16)
                } else {
                    value(i).to_bits()
                }
            })
            .collect::<Vec<_>>(),
    );
    if poison {
        source[0] = if half {
            (source[0] & 0xffff0000) | 0x7e00
        } else {
            f32::NAN.to_bits()
        };
    }
    let mi = arena.upload_u32(&map)?;
    let si = arena.upload_u32(&source)?;
    let x = arena.allocate_zeroed(groups::LAYER_X_BYTES as usize / 4)?;
    let h = arena.allocate_zeroed(5 * 1024 * 1024 / 4)?;
    let f0 = arena.allocate_zeroed(1)?;
    let f1 = arena.allocate_zeroed(1)?;
    for group in 0..n {
        let prefix = map[groups::LAYER_PREFIX + group] as u64;
        let members = map[516 + group] as usize;
        let span = members.div_ceil(32) * 32 * 5120;
        let output = u::handle(arena, x)?
            .offset_start(prefix * 5120)
            .offset_end(groups::LAYER_X_BYTES - prefix * 5120 - span as u64);
        let mut old = [0u32; 16];
        old[0] = count as u32;
        old[1] = 2560;
        old[2] = 1280;
        old[4] = group as u32;
        old[12] = u32::from(half);
        let args = KernelArguments::new()
            .with_buffers(vec![
                u::handle(arena, mi)?.binding(),
                u::handle(arena, si)?.binding(),
                output.binding(),
                u::handle(arena, f0)?.binding(),
            ])
            .with_info(MetadataBindingInfo::custom(
                old.chunks_exact(2)
                    .map(|p| u64::from(p[0]) | (u64::from(p[1]) << 32))
                    .collect(),
            ));
        unsafe {
            client.launch_unchecked(
                Box::new(experts::Op::Gather),
                CubeCount::Static((members * 5) as u32, 1, 1),
                args,
            );
        }
    }
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    let original = snapshot(arena, x)?;
    let mut expected = original.clone();
    // Independent nested tile enumeration, with prefix outside each segment.
    for group in 0..n {
        let members = map[516 + group] as usize;
        if members < 256 {
            continue;
        }
        let first = map[groups::LAYER_PREFIX + group] as usize * 1280;
        let rows = members.div_ceil(32) * 32;
        let mut dest = first;
        for tile_row in (0..rows).step_by(16) {
            for tile_col in (0..1280).step_by(8) {
                for row in tile_row..tile_row + 16 {
                    for col in tile_col..tile_col + 8 {
                        expected[dest] = original[first + row * 1280 + col];
                        dest += 1;
                    }
                }
            }
        }
        if dest != first + rows * 1280 {
            return Err("packed layer complete segment oracle".into());
        }
    }
    drop(original);
    let layout = arena.layout(x)?;
    crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
        client,
        CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
        CubeDim::new_1d(64),
        unsafe { ArrayArg::from_raw_parts(arena.handle(x)?.clone(), layout.owned_words) },
    );
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    probe_util::source(arena, x, &[])?;
    launch_packed(
        client,
        [
            u::handle(arena, mi)?,
            u::handle(arena, si)?,
            u::handle(arena, x)?,
            u::handle(arena, h)?,
            u::handle(arena, f1)?,
        ],
        &[
            count as u32,
            n as u32,
            u32::from(half),
            padded as u32,
            1,
            0,
            0,
            0,
        ],
    )?;
    cubecl::future::block_on(client.sync()).map_err(u::fault)?;
    probe_util::source(arena, x, &expected)?;
    for f in [f0, f1] {
        probe_util::source(arena, f, &[if poison { 128 } else { 0 }])?;
    }
    for (id, v) in [(mi, &*map), (si, &*source)] {
        probe_util::source(arena, id, v)?;
    }
    probe_util::source(arena, h, &[])?;
    Ok(
        serde_json::json!({"family":"whole-layer-packed-gather-original-per-expert-pair","count":count,"groups":n,"padded_rows":padded,"half":half,"poison_and_invalid_route":poison,"actual_previous_and_new_gpu_programs":true,"bit_mismatches":0,"full_output_words_checked":expected.len(),"global_x_bytes":groups::LAYER_X_BYTES,"single_output_owner":true,"complete_zero_between_programs":true,"full_source_fault_hidden_and_tail_checked":true,"readback_window_bytes":WINDOW*4,"zeroizing_host_reference_bytes":expected.capacity()*4,"original_arena_budget_bytes":arena.budget_bytes(),"actual_arena_charged_bytes":arena.charged_bytes()}),
    )
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
) -> Checked<Vec<serde_json::Value>> {
    if !arena.public_owned_zero_audit_enabled() || arena.buffer_count() != 0 {
        return Err("layer producer empty audited arena required".into());
    }
    let mut out = Vec::new();
    for count in [512, 4096] {
        for half in [false, true] {
            for poison in [false, true] {
                let result = one(client, arena, count, half, poison);
                arena.reset()?;
                out.push(result?);
            }
        }
    }
    Ok(out)
}
