//! Fixed-public check of the ACTUAL cooperative-matrix product used by prompt
//! inference. It calls the same primitive without claiming a prompt workspace
//! or mutating normal dense admission. All comparisons and activations stay RAM.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn rounded(value: f32, bfloat: bool) -> (u16, f32) {
    if bfloat {
        let v = half::bf16::from_f32(value);
        (v.to_bits(), v.to_f32())
    } else {
        let v = half::f16::from_f32(value);
        (v.to_bits(), v.to_f32())
    }
}

pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> ProbeResult<Vec<serde_json::Value>> {
    let (rows, columns) = (33usize, 10240usize);
    let tensor = file
        .tensor("blk.0.hc_attn_down.weight")
        .ok_or("public matrix original weight")?;
    if tensor.info().dimensions() != [10240, 320] || tensor.info().dtype().gguf_id() != 30 {
        return Err("public matrix original geometry".into());
    }
    let mut original = Zeroizing::new(vec![0u8; rows * columns * 2]);
    tensor.read_range(0, &mut original)?;
    let mut decoded = Zeroizing::new(vec![0.; rows * columns]);
    tensor.info().dtype().decode(&original, &mut decoded)?;
    let mut reports = Vec::new();
    for bfloat in [true, false] {
        let mut weights = Zeroizing::new(vec![0.; decoded.len()]);
        let mut packed = Zeroizing::new(vec![0u32; decoded.len() / 2]);
        for (i, &v) in decoded.iter().enumerate() {
            let (bits, value) = rounded(v, bfloat);
            weights[i] = value;
            packed[i / 2] |= u32::from(bits) << (16 * (i % 2));
        }
        let public = client.create_from_slice(u32::as_bytes(&packed));
        for count in [1, 8, 35] {
            let mut inputs = Zeroizing::new(vec![0.; count * columns]);
            let mut image = Zeroizing::new(vec![0u32; inputs.len() / 2]);
            for (i, v) in inputs.iter_mut().enumerate() {
                let x = (((i * 7 + i / columns * 13) % 43) as f32 - 21.) / 16.;
                let (bits, value) = rounded(x, bfloat);
                *v = value;
                image[i / 2] |= u32::from(bits) << (16 * (i % 2));
            }
            let input = arena.upload_u32(&image)?;
            let output = arena.allocate_zeroed(count * rows)?;
            let fault = arena.allocate_zeroed(1)?;
            // Same original compiled primitive, authentic checked arena roots.
            // No model recording or source ownership is manufactured here.
            native::run_dense(
                client,
                [
                    u::handle(arena, input)?,
                    public.clone(),
                    u::handle(arena, output)?,
                    u::handle(arena, fault)?,
                ],
                count,
                columns,
                rows,
                0,
                0,
                rows,
                bfloat,
            )?;
            let actual = probe::read_native_public(arena, output)?;
            if probe::read_native_public(arena, fault)?
                .iter()
                .any(|&v| v != 0)
            {
                return Err("public matrix actual native fault".into());
            }
            let mut mismatches = 0usize;
            let mut max_abs = 0f32;
            for (i, &bits) in actual.iter().enumerate() {
                if i >= count * rows {
                    if bits != 0 {
                        return Err("public matrix full-owner padding".into());
                    }
                    continue;
                }
                let w = &weights[i % rows * columns..][..columns];
                let x = &inputs[i / rows * columns..][..columns];
                let reference = w
                    .iter()
                    .zip(x)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum::<f64>() as f32;
                let value = f32::from_bits(bits);
                let error = (value - reference).abs();
                mismatches +=
                    usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
                max_abs = max_abs.max(error);
            }
            println!(
                "public_native_matrix_numeric bfloat={bfloat} count={count} rows={rows} columns={columns} mismatches={mismatches} max_abs={max_abs}"
            );
            reports.push(
                serde_json::json!({"family":"cooperative-matrix","bfloat":bfloat,
                "count":count,"rows":rows,"columns":columns,"mismatches":mismatches,
                "max_abs":max_abs,"actual_native_kernels":true,"full_owner_and_tail_checked":true}),
            );
            arena.reset()?;
            if mismatches != 0 {
                return Err("public native matrix numerical parity failed".into());
            }
        }
    }
    Ok(reports)
}
