//! Compiled-public raw operator comparison; no fabricated model/prompt scope.
use super::*;
use crate::{gr_batch_probe::read_native_public, private_arena::PrivateArena};
use zeroize::Zeroizing;
type P<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(super) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> P<Vec<serde_json::Value>> {
    let tensor = file
        .tensor("blk.0.hc_attn_norm.weight")
        .ok_or("GR original norm tensor")?;
    if tensor.info().dimensions() != [10240] || tensor.info().dtype().gguf_id() != 0 {
        return Err("GR original F32 norm geometry".into());
    }
    let mut bytes = Zeroizing::new(vec![0u8; 40960]);
    tensor.read_range(0, &mut bytes)?;
    let gamma = Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    let mut cases = Vec::new();
    for count in [1usize, 8, 148, 512] {
        for apply in [false, true] {
            for ip in [false, true] {
                cases.push((count, apply, ip, "finite"));
            }
        }
    }
    for ip in [false, true] {
        for poison in ["nan-residual", "inf-block"] {
            cases.push((8, true, ip, poison));
        }
    }
    let mut reports = Vec::new();
    for (count, apply, in_place, control) in cases {
        let first = 10240;
        let len = first + count * 10240 + 64;
        let mut r = Zeroizing::new(
            (0..len)
                .map(|i| (((i * 7 % 101) as f32 - 50.) / 32.).to_bits())
                .collect::<Vec<_>>(),
        );
        let mut previous = Zeroizing::new(
            (0..len)
                .map(|i| (((i * 13 % 127) as f32 - 63.) / 64.).to_bits())
                .collect::<Vec<_>>(),
        );
        let mut block = Zeroizing::new(
            (0..count * 2560 + 64)
                .map(|i| (((i * 3 % 43) as f32 - 21.) / 32.).to_bits())
                .collect::<Vec<_>>(),
        );
        let injection = Zeroizing::new(
            (0..count * 4 + 64)
                .map(|i| ((i % 17) as f32 / 16.).to_bits())
                .collect::<Vec<_>>(),
        );
        if control == "nan-residual" {
            let at = first + (count - 1) * 10240 + 17;
            if in_place {
                r[at] = f32::NAN.to_bits();
            } else {
                previous[at] = f32::NAN.to_bits();
            }
        }
        if control == "inf-block" {
            block[(count - 1) * 2560 + 19] = f32::INFINITY.to_bits();
        }
        let prev = arena.upload_u32(&previous)?;
        let bid = arena.upload_u32(&block)?;
        let inj = arena.upload_u32(&injection)?;
        let gid = arena.upload_u32(&gamma)?;
        let mut all = Vec::new();
        for new in [false, true] {
            let rid = arena.upload_u32(&r)?;
            let out = arena.allocate_zeroed(count * 5120 + 64)?;
            let scales = arena.allocate_zeroed(count * 4 + 64)?;
            let fault = arena.allocate_zeroed(1)?;
            let mut h = vec![
                u::handle(arena, prev)?,
                u::handle(arena, bid)?,
                u::handle(arena, inj)?,
                u::handle(arena, gid)?,
                u::handle(arena, rid)?,
                u::handle(arena, out)?,
                u::handle(arena, scales)?,
                u::handle(arena, fault)?,
            ];
            if in_place {
                h.remove(0);
            }
            let mut info = [0u32; 16];
            info[0] = count as u32;
            info[4] = first as u32;
            info[9] = u32::from(apply);
            info[11] = 1e-6f32.to_bits();
            dispatch(client, new, in_place, h, info)?;
            let rb = read_native_public(arena, rid)?;
            let ob = read_native_public(arena, out)?;
            let sb = read_native_public(arena, scales)?;
            let fb = read_native_public(arena, fault)?;
            if rb[..first] != r[..first]
                || rb[first + count * 10240..r.len()] != r[first + count * 10240..]
                || rb[r.len()..].iter().any(|&x| x != 0)
                || ob[count * 5120..].iter().any(|&x| x != 0)
                || sb[count * 4..].iter().any(|&x| x != 0)
                || fb[0] != u32::from(control != "finite")
                || fb[1..].iter().any(|&x| x != 0)
            {
                return Err("GR whole owner/padding/fault contract".into());
            }
            all.push([rb, ob, sb, fb]);
        }
        if all[0] != all[1] {
            return Err("GR register original full bitwise R/image/scales/fault mismatch".into());
        }
        for (id, original) in [
            (prev, &*previous),
            (bid, &*block),
            (inj, &*injection),
            (gid, &*gamma),
        ] {
            let after = read_native_public(arena, id)?;
            if after[..original.len()] != original[..]
                || after[original.len()..].iter().any(|&x| x != 0)
            {
                return Err("GR original read-only source owner changed".into());
            }
        }
        reports.push(serde_json::json!({"family":"gr-register-norm-original-pair","count":count,"apply":apply,"in_place":in_place,"first":first,"control":control,"bit_mismatches":0,"full_R_normalized_scales_fault_sources_padding_checked":true,"actual_original_and_register_kernels":true,"original_norm_weight":true,"logical_scalar_registers":10,"fabricated_prompt_or_batch_scope":false,"full_owned_words_checked":all[0].iter().map(|v|v.len()).sum::<usize>()}));
        arena.reset()?;
    }
    Ok(reports)
}
