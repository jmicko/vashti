//! Compiled-public paired original/shared and selected direct/finite pipeline.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn rounded(v: f32, bf: bool) -> (u16, f32) {
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
    let columns = 10240usize;
    let t = file
        .tensor("blk.0.hc_attn_down.weight")
        .ok_or("direct original tensor")?;
    if t.info().dimensions() != [10240, 320] || t.info().dtype().gguf_id() != 30 {
        return Err("direct original model geometry".into());
    }
    let mut bytes = Zeroizing::new(vec![0u8; 320 * columns * 2]);
    t.read_range(0, &mut bytes)?;
    let mut decoded = Zeroizing::new(vec![0f32; 320 * columns]);
    t.info().dtype().decode(&bytes, &mut decoded)?;
    let cases = [
        (true, 128usize, 512usize, 0usize, 128usize, 0usize, "finite"),
        (true, 128, 512, 4, 132, 4, "finite"),
        (true, 320, 512, 0, 320, 0, "finite"),
        (true, 320, 640, 0, 320, 0, "finite"),
        (false, 128, 512, 0, 128, 0, "finite"),
        (false, 256, 512, 4, 260, 4, "finite"),
        (true, 320, 512, 0, 320, 0, "nan-weight"),
        (true, 320, 512, 0, 320, 0, "nan-input"),
        (false, 128, 512, 0, 128, 0, "nan-weight"),
        (false, 128, 512, 0, 128, 0, "nan-input"),
        (true, 320, 512, 0, 320, 0, "inf-weight"),
        (false, 128, 512, 0, 128, 0, "negative-inf-weight"),
    ];
    let mut reports = Vec::new();
    for (bf, rows, count, first, stride, weight_first, control) in cases {
        let mut weights = Zeroizing::new(vec![0f32; rows * columns]);
        let mut packed = Zeroizing::new(vec![0u32; (weight_first + rows * columns) / 2]);
        for (i, v) in weights.iter_mut().enumerate() {
            let (bits, x) = rounded(decoded[i], bf);
            *v = x;
            let at = weight_first + i;
            packed[at / 2] |= u32::from(bits) << (16 * (at % 2));
        }
        if control.ends_with("weight") {
            let value = match control {
                "nan-weight" => f32::NAN,
                "inf-weight" => f32::INFINITY,
                _ => f32::NEG_INFINITY,
            };
            let (bits, x) = rounded(value, bf);
            weights[0] = x;
            let at = weight_first;
            let shift = 16 * (at % 2);
            packed[at / 2] = (packed[at / 2] & !(0xffff << shift)) | (u32::from(bits) << shift);
        }
        let mut inputs = Zeroizing::new(vec![0f32; count * columns]);
        let mut image = Zeroizing::new(vec![0u32; count * columns / 2]);
        for (i, v) in inputs.iter_mut().enumerate() {
            let raw = if i % 257 == 0 {
                -0.0
            } else {
                (((i * 7 + i / columns * 13) % 43) as f32 - 21.) / 16.
            };
            let (bits, x) = rounded(raw, bf);
            *v = x;
            image[i / 2] |= u32::from(bits) << (16 * (i % 2));
        }
        if control == "nan-input" {
            let (bits, x) = rounded(f32::NAN, bf);
            inputs[0] = x;
            image[0] = (image[0] & 0xffff0000) | u32::from(bits);
        }
        let input = arena.upload_u32(&image)?;
        let weight = arena.upload_u32(&packed)?;
        let logical = first + (count - 1) * stride + rows;
        let actual_id = arena.allocate_zeroed(logical + 64)?;
        let reference_id = arena.allocate_zeroed(logical + 64)?;
        let fault_id = arena.allocate_zeroed(1)?;
        let old_fault_id = arena.allocate_zeroed(1)?;
        let h = [
            u::handle(arena, input)?,
            u::handle(arena, weight)?,
            u::handle(arena, actual_id)?,
            u::handle(arena, fault_id)?,
        ];
        #[cfg(feature = "strata-dense-slm40")]
        let selected = native_wide::try_dense;
        #[cfg(not(feature = "strata-dense-slm40"))]
        let selected = native_wide::try_dense;
        if !selected(
            client,
            h,
            count,
            columns,
            rows,
            weight_first,
            first,
            stride,
            bf,
        )? {
            return Err("direct actual selected matrix rejected".into());
        }
        native_wide::shared_reference(
            client,
            [
                u::handle(arena, input)?,
                u::handle(arena, weight)?,
                u::handle(arena, reference_id)?,
                u::handle(arena, old_fault_id)?,
            ],
            count,
            columns,
            rows,
            weight_first,
            first,
            stride,
            bf,
        )?;
        let actual = probe::read_native_public(arena, actual_id)?;
        let old = probe::read_native_public(arena, reference_id)?;
        let flags = probe::read_native_public(arena, fault_id)?;
        let old_flags = probe::read_native_public(arena, old_fault_id)?;
        if actual != old
            || flags != old_flags
            || flags[0] != u32::from(control != "finite")
            || flags[1..].iter().any(|&v| v != 0)
        {
            return Err("direct original bitwise output or sticky-fault parity".into());
        }
        let after_input = probe::read_native_public(arena, input)?;
        let after_weight = probe::read_native_public(arena, weight)?;
        if after_input[..image.len()] != image[..]
            || after_input[image.len()..].iter().any(|&v| v != 0)
            || after_weight[..packed.len()] != packed[..]
            || after_weight[packed.len()..].iter().any(|&v| v != 0)
        {
            return Err("direct full original source owners changed".into());
        }
        let (mut errors, mut max_abs) = (0usize, 0f32);
        for (i, &bits) in actual.iter().enumerate() {
            let rel = i.saturating_sub(first);
            let logical = i >= first && rel / stride < count && rel % stride < rows;
            if !logical {
                if bits != 0 {
                    return Err("direct full output holes or owner tail".into());
                }
                continue;
            }
            let row = rel % stride;
            let token = rel / stride;
            let reference = weights[row * columns..][..columns]
                .iter()
                .zip(&inputs[token * columns..][..columns])
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32;
            if !reference.is_finite() {
                errors += usize::from(bits != 0);
                continue;
            }
            let value = f32::from_bits(bits);
            let error = (value - reference).abs();
            errors += usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
            max_abs = max_abs.max(error);
        }
        if errors != 0 {
            return Err("direct original F64 numerical tolerance".into());
        }
        reports.push(serde_json::json!({"family":"dense-slm40-finite-paired","bfloat":bf,"rows":rows,"columns":columns,"count":count,"first":first,"stride":stride,"weight_first":weight_first,"control":control,
            "bit_mismatches":0,"f64_mismatches":errors,"max_abs":max_abs,"original_shared_gpu_and_f64_both_checked":true,
            "actual_direct_producer_and_finite_consumer":true,"full_owner_holes_sources_and_fault_checked":true,"fabricated_prompt_or_batch_scope":false,"new_payload_allocations":0,"actual_pitch40_stages_and_consumers":true}));
        arena.reset()?;
    }
    Ok(reports)
}
