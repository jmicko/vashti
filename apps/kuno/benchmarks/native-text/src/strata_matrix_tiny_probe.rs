//! Compiled-public original-model tiny products, paired with the original GPU
//! primitive and an independent unchanged F64 oracle. No Session or fake scope.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn pack(v: f32, bf: bool) -> (u16, f32) {
    if bf {
        let x = half::bf16::from_f32(v);
        (x.to_bits(), x.to_f32())
    } else {
        let x = half::f16::from_f32(v);
        (x.to_bits(), x.to_f32())
    }
}
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> R<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    for (columns, rows, bf, name) in [
        (10240usize, 4usize, true, "blk.0.hc_attn_inject.weight"),
        (2560, 1, false, "blk.0.ffn_gate_inp_shexp.weight"),
    ] {
        let tensor = file.tensor(name).ok_or("tiny original model tensor")?;
        if tensor.info().dimensions() != [columns as u64, rows as u64]
            && !(rows == 1 && tensor.info().dimensions() == [columns as u64])
        {
            return Err("tiny original tensor geometry".into());
        }
        let row_bytes = usize::try_from(tensor.info().dtype().row_bytes(columns as u64)?)?;
        let mut raw = Zeroizing::new(vec![0u8; rows * row_bytes]);
        tensor.read_range(0, &mut raw)?;
        let mut decoded = Zeroizing::new(vec![0f32; rows * columns]);
        tensor.info().dtype().decode(&raw, &mut decoded)?;
        for (count, control) in [1usize, 3, 7, 8, 9, 45, 512, 513]
            .into_iter()
            .map(|c| (c, "finite-ieee"))
            .chain([(9, "nan-weight"), (9, "nan-input")])
        {
            for in_place in [false, true] {
                if bf && in_place {
                    continue;
                }
                let mut weights = Zeroizing::new(vec![0f32; rows * columns]);
                let mut packed = Zeroizing::new(vec![0u32; rows * columns / 2]);
                for (i, &v) in decoded.iter().enumerate() {
                    let (bits, x) = pack(v, bf);
                    weights[i] = x;
                    packed[i / 2] |= u32::from(bits) << (16 * (i % 2));
                }
                if control == "nan-weight" {
                    weights[0] = f32::NAN;
                    let bits = if bf { 0x7fc1 } else { 0x7e01 };
                    packed[0] = (packed[0] & 0xffff0000) | bits;
                }
                let mut inputs = Zeroizing::new(vec![0f32; count * columns]);
                let mut image = Zeroizing::new(vec![0u32; count * columns / 2]);
                for (i, v) in inputs.iter_mut().enumerate() {
                    let x = if i % 257 == 0 {
                        -0.0
                    } else {
                        (((i * 7 + i / columns * 13) % 43) as f32 - 21.) / 16.
                    };
                    let (bits, x) = pack(x, bf);
                    *v = x;
                    image[i / 2] |= u32::from(bits) << (16 * (i % 2));
                }
                if control == "nan-input" {
                    inputs[0] = f32::NAN;
                    let bits = if bf { 0x7fc1 } else { 0x7e01 };
                    image[0] = (image[0] & 0xffff0000) | bits;
                }
                let pad_words = 16 * columns / 2;
                let sentinel = 0xa57a0f11u32;
                let mut initial = Zeroizing::new(vec![sentinel; pad_words + 64]);
                if in_place {
                    initial[..packed.len()].copy_from_slice(&packed);
                }
                let input = arena.upload_u32(&image)?;
                let weight = arena.upload_u32(&packed)?;
                let scratch = arena.upload_u32(&initial)?;
                let output = arena.allocate_zeroed(count * rows + 64)?;
                let old_output = arena.allocate_zeroed(count * rows + 64)?;
                let fault = arena.allocate_zeroed(1)?;
                let old_fault = arena.allocate_zeroed(1)?;
                let scratch_h = u::handle(arena, scratch)?;
                let weight_h = if in_place {
                    scratch_h
                        .clone()
                        .offset_end(scratch_h.size_in_used() - (packed.len() * 4) as u64)
                } else {
                    u::handle(arena, weight)?
                };
                native::run_tiny(
                    client,
                    [
                        u::handle(arena, input)?,
                        weight_h,
                        scratch_h,
                        u::handle(arena, output)?,
                        u::handle(arena, fault)?,
                    ],
                    count,
                    columns,
                    rows,
                    0,
                    rows,
                    bf,
                )?;
                native::original_tiny_reference(
                    client,
                    [
                        u::handle(arena, input)?,
                        u::handle(arena, weight)?,
                        u::handle(arena, old_output)?,
                        u::handle(arena, old_fault)?,
                    ],
                    count,
                    columns,
                    rows,
                    bf,
                )?;
                let actual = probe::read_native_public(arena, output)?;
                let old = probe::read_native_public(arena, old_output)?;
                let flags = probe::read_native_public(arena, fault)?;
                let old_flags = probe::read_native_public(arena, old_fault)?;
                let staged = probe::read_native_public(arena, scratch)?;
                let after_input = probe::read_native_public(arena, input)?;
                let after_weights = probe::read_native_public(arena, weight)?;
                if actual != old
                    || flags != old_flags
                    || flags[0] != u32::from(control != "finite-ieee")
                    || flags[1..].iter().any(|&v| v != 0)
                {
                    return Err("tiny original GPU bitwise or fault parity".into());
                }
                if staged[..packed.len()] != packed[..]
                    || staged[packed.len()..pad_words].iter().any(|&v| v != 0)
                    || staged[pad_words..].iter().any(|&v| v != sentinel)
                    || after_input[..image.len()] != image[..]
                    || after_input[image.len()..].iter().any(|&v| v != 0)
                    || after_weights[..packed.len()] != packed[..]
                    || after_weights[packed.len()..].iter().any(|&v| v != 0)
                {
                    return Err("tiny original sources padded bit image or full-owner guard".into());
                }
                let mut errors = 0usize;
                let mut max_abs = 0f32;
                for (i, &bits) in actual.iter().enumerate() {
                    if i >= count * rows {
                        if bits != 0 {
                            return Err("tiny output full owning tail".into());
                        }
                        continue;
                    }
                    let reference = weights[i % rows * columns..][..columns]
                        .iter()
                        .zip(&inputs[i / rows * columns..][..columns])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>() as f32;
                    if !reference.is_finite() {
                        errors += usize::from(bits != 0);
                        continue;
                    }
                    let value = f32::from_bits(bits);
                    let error = (value - reference).abs();
                    errors +=
                        usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
                    max_abs = max_abs.max(error);
                }
                if errors != 0 {
                    return Err("tiny unchanged F64 numerical tolerance".into());
                }
                reports.push(serde_json::json!({"family":"tiny-original-paired-matrix","columns":columns,"rows":rows,
                    "bfloat":bf,"count":count,"control":control,"in_place":in_place,"bit_mismatches":0,"f64_mismatches":errors,"max_abs":max_abs,
                    "original_gpu_and_f64_both_checked":true,"original_chronological_k16_product":true,"actual_raw_producer_and_matrix_consumer":true,
                    "full_owner_sources_padding_and_fault_checked":true,"source_words":packed.len(),"padded_words":pad_words,
                    "no_new_production_allocation":true,"fabricated_prompt_or_batch_scope":false}));
                arena.reset()?;
            }
        }
    }
    Ok(reports)
}
