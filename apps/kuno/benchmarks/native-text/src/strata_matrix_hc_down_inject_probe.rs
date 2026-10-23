//! Fixed-public paired real original down + padded tiny-injection comparison.
//! No model or prompt Scope is fabricated; inputs/weights are real arena owners.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const K: usize = 10240;
fn exact_view(h: Handle, bytes: usize) -> Handle {
    let end = h.size_in_used() - bytes as u64;
    h.offset_end(end)
}
fn original_weights(file: &kuno_text::GgufFile, name: &str, rows: usize) -> R<Zeroizing<Vec<u32>>> {
    let t = file.tensor(name).ok_or("HC original tensor missing")?;
    if t.info().dimensions() != [K as u64, rows as u64] || t.info().dtype().gguf_id() != 30 {
        return Err("HC original BF16 geometry".into());
    }
    let mut bytes = Zeroizing::new(vec![0u8; rows * K * 2]);
    t.read_range(0, &mut bytes)?;
    let mut words = Zeroizing::new(
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    // Real owner poison beyond the exact BF16 descriptor. Any fabricated
    // padded weight/source view is exposed by non-finite results or mutation.
    words.extend(std::iter::repeat_n(0x7fc17fc1, 64));
    Ok(words)
}
fn source_unchanged(
    arena: &PrivateArena,
    id: crate::private_arena::BufferId,
    original: &[u32],
) -> R<()> {
    let after = probe::read_native_public(arena, id)?;
    if after[..original.len()] != *original || after[original.len()..].iter().any(|&v| v != 0) {
        return Err("HC entire original source/padded owner changed".into());
    }
    Ok(())
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> R<Vec<serde_json::Value>> {
    let original_down = original_weights(file, "blk.0.hc_attn_down.weight", 320)?;
    let original_inject = original_weights(file, "blk.0.hc_attn_inject.weight", 4)?;
    let fixtures = [
        (128usize, "finite", 0usize),
        (148, "finite", 0),
        (512, "finite", 0),
        (513, "finite", 0),
        (148, "finite-offset", 4),
        (148, "nan-down-weight", 0),
        (148, "nan-inject-weight", 0),
        (148, "inf-down-weight", 0),
        (148, "negative-inf-inject-weight", 0),
        (148, "nan-input-prefix", 0),
        (148, "nan-input-tail", 0),
        (128, "inf-input", 0),
    ];
    let mut reports = Vec::new();
    for (count, control, first) in fixtures {
        let mut down_weight = Zeroizing::new(original_down.to_vec());
        let mut inject_weight = Zeroizing::new(original_inject.to_vec());
        if control.ends_with("down-weight") {
            let poison = if control.starts_with("nan") {
                0x7fc1
            } else {
                0x7f80
            };
            down_weight[0] = (down_weight[0] & 0xffff0000) | poison;
        }
        if control.ends_with("inject-weight") {
            let poison = if control.starts_with("nan") {
                0x7fc1
            } else {
                0xff80
            };
            inject_weight[0] = (inject_weight[0] & 0xffff0000) | poison;
        }
        let mut input = Zeroizing::new(vec![0u32; count * K / 2 + 64]);
        for i in 0..count * K {
            let h = (i as u32).wrapping_mul(0x9e3779b9).rotate_left(13);
            let bits = if i % 257 == 0 {
                0x8000
            } else {
                (if h & 1 == 0 { 0x3f00 } else { 0xbf00 }) | ((h >> 16) & 127)
            };
            input[i / 2] |= bits << (16 * (i % 2));
        }
        input[count * K / 2..].fill(0x7fc17fc1);
        if control.starts_with("nan-input") || control == "inf-input" {
            let token = if control.ends_with("tail") {
                count - 1
            } else {
                0
            };
            let at = token * K / 2;
            let bits = if control == "inf-input" {
                0x7f80
            } else {
                0x7fc1
            };
            input[at] = (input[at] & 0xffff0000) | bits;
        }
        let sentinel = 0xa57a0f11;
        let down_initial = Zeroizing::new(vec![sentinel; first + count * 320 + 64]);
        let inject_initial = Zeroizing::new(vec![sentinel; first + count * 4 + 64]);
        let scratch_initial = Zeroizing::new(vec![sentinel; 16 * K / 2 + 64]);
        let input_id = arena.upload_u32(&input)?;
        let dw_id = arena.upload_u32(&down_weight)?;
        let iw_id = arena.upload_u32(&inject_weight)?;
        let actual_down = arena.upload_u32(&down_initial)?;
        let old_down = arena.upload_u32(&down_initial)?;
        let actual_inject = arena.upload_u32(&inject_initial)?;
        let old_inject = arena.upload_u32(&inject_initial)?;
        let scratch = arena.upload_u32(&scratch_initial)?;
        let old_scratch = arena.upload_u32(&scratch_initial)?;
        let fault = arena.allocate_zeroed(1)?;
        let old_fault = arena.allocate_zeroed(1)?;
        let input_h = exact_view(u::handle(arena, input_id)?, count * K * 2);
        let dw = exact_view(u::handle(arena, dw_id)?, 320 * K * 2);
        let iw = exact_view(u::handle(arena, iw_id)?, 4 * K * 2);
        if !native_wide::try_hc_down_inject(
            client,
            [
                input_h.clone(),
                dw.clone(),
                iw.clone(),
                u::handle(arena, actual_down)?,
                u::handle(arena, actual_inject)?,
                u::handle(arena, fault)?,
            ],
            u::handle(arena, scratch)?,
            count,
            first,
            first,
        )? {
            return Err("HC actual coherent fused selection rejected".into());
        }
        // Unchanged real prompt primitives, with the SAME original full count
        // and exact original BF16 source descriptors. No fictional source rows.
        native::run_dense(
            client,
            [
                input_h.clone(),
                dw,
                u::handle(arena, old_down)?,
                u::handle(arena, old_fault)?,
            ],
            count,
            K,
            320,
            0,
            first,
            320,
            true,
        )?;
        native::run_tiny(
            client,
            [
                input_h,
                iw,
                u::handle(arena, old_scratch)?,
                u::handle(arena, old_inject)?,
                u::handle(arena, old_fault)?,
            ],
            count,
            K,
            4,
            first,
            4,
            true,
        )?;
        let actual = [
            probe::read_native_public(arena, actual_down)?,
            probe::read_native_public(arena, actual_inject)?,
            probe::read_native_public(arena, fault)?,
        ];
        let old = [
            probe::read_native_public(arena, old_down)?,
            probe::read_native_public(arena, old_inject)?,
            probe::read_native_public(arena, old_fault)?,
        ];
        if actual != old {
            let mismatch = (0..3)
                .map(|j| {
                    actual[j]
                        .iter()
                        .zip(old[j].iter())
                        .filter(|(a, b)| a != b)
                        .count()
                })
                .collect::<Vec<_>>();
            return Err(format!("HC original bitwise fields_down_injection_fault={mismatch:?} control={control} count={count}").into());
        }
        let nonfinite = !control.starts_with("finite");
        if actual[2][0] != u32::from(nonfinite) || actual[2][1..].iter().any(|&v| v != 0) {
            return Err("HC original sticky fault or complete fault owner".into());
        }
        for (j, rows, initial) in [(0, 320, &*down_initial), (1, 4, &*inject_initial)] {
            for (i, &bits) in actual[j].iter().enumerate() {
                if i < first || i >= first + count * rows {
                    let expected = initial.get(i).copied().unwrap_or(0);
                    if bits != expected {
                        return Err("HC output prefix/padded owner changed".into());
                    }
                } else if !f32::from_bits(bits).is_finite() {
                    return Err("HC finite consumer leaked nonfinite output".into());
                }
            }
        }
        source_unchanged(arena, input_id, &input)?;
        source_unchanged(arena, dw_id, &down_weight)?;
        source_unchanged(arena, iw_id, &inject_weight)?;
        let staged = probe::read_native_public(arena, scratch)?;
        if count % 128 == 0 {
            if staged[..scratch_initial.len()] != scratch_initial[..]
                || staged[scratch_initial.len()..].iter().any(|&v| v != 0)
            {
                return Err("HC full prefix must not touch old tiny workspace".into());
            }
        } else {
            let words = 4 * K / 2;
            let padded = 16 * K / 2;
            if staged[..words] != inject_weight[..words]
                || staged[words..padded].iter().any(|&v| v != 0)
                || staged[padded..scratch_initial.len()] != scratch_initial[padded..]
                || staged[scratch_initial.len()..].iter().any(|&v| v != 0)
            {
                return Err("HC authentic old tiny suffix workspace/padding contract".into());
            }
        }
        reports.push(
            serde_json::json!({"family":"hc-down-inject-original-paired","count":count,
            "prefix_count":count/128*128,"suffix_count":count%128,"first":first,"control":control,
            "bit_mismatches":0,"original_down_and_tiny_gpu_checked":true,
            "full_owner_sources_poison_padding_outputs_fault_checked":true,
            "original_k16_order":true,"actual_separate_source_and_output_owners":true,
            "producer_shared_bytes":45056,"production_payload_allocations":0,
            "fabricated_prompt_or_batch_scope":false}),
        );
        arena.reset()?;
    }
    Ok(reports)
}
