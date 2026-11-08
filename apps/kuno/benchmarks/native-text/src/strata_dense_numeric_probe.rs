//! Actual native Q8 producer and packed dense consumers, fixed synthetic
//! activations and original model rows. Independent canonical CPU decoding;
//! only aggregate errors survive. No inference path or throughput measurement.
use super::*;
use crate::{resident_util as u, strata_dense as dense};

type Quantized = (Zeroizing<Vec<f32>>, Zeroizing<Vec<u32>>);
fn quantize(input: &[f32]) -> Result<Quantized> {
    if input.is_empty() || !input.len().is_multiple_of(32) {
        return Err("public dense Q8 input extent".into());
    }
    let mut values = Zeroizing::new(vec![0.; input.len()]);
    let mut planes = Zeroizing::new(vec![0; input.len() / 2 + input.len() / 16]);
    for (block, row) in input.chunks_exact(32).enumerate() {
        let max = row.iter().fold(0_f32, |m, v| m.max(v.abs()));
        let scale = max / 127.;
        let half_scale = half::f16::from_f32(scale).to_f32();
        for (lane, &x) in row.iter().enumerate() {
            let q = if scale == 0. {
                0
            } else {
                (x / scale).round().clamp(-127., 127.) as i32
            };
            values[block * 32 + lane] = half_scale * q as f32;
            planes[block * 8 + lane / 4] |= (q as u32 & 255) << (8 * (lane % 4));
        }
        planes[input.len() / 2 + block] = half_scale.to_bits();
        // Fixed dyadic synthetic inputs sum exactly in both sequential and
        // subgroup-XOR orders. This independently checks the native sum plane.
        planes[input.len() / 2 + input.len() / 32 + block] =
            half::f16::from_f32(row.iter().sum::<f32>())
                .to_f32()
                .to_bits();
    }
    Ok((values, planes))
}

pub(super) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &GgufFile,
) -> Result<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    for dtype in [8, 12, 13, 14, 20, 23] {
        // Deterministic original inventory selection, never arbitrary caller
        // tensors. Thirty-three rows exercises the native partial subgroup.
        let Some((name, info)) = file.tensors().iter().find(|(_, info)| {
            let d = info.dimensions();
            info.dtype().gguf_id() == dtype
                && d.len() == 2
                && d[1] >= 33
                && d[0] <= 16384
                && d[0] > 0
                && d[0].is_multiple_of(if matches!(dtype, 8 | 20) { 128 } else { 256 })
        }) else {
            continue;
        };
        let columns = info.dimensions()[0] as usize;
        let rows = 33;
        let row_bytes = crate::quant_matvec::packed_row_bytes(columns, dtype);
        let logical = rows * row_bytes;
        let mut packed = Zeroizing::new(vec![0u8; logical.next_multiple_of(16)]);
        file.tensor(name)
            .ok_or("public dense original tensor missing")?
            .read_range(0, &mut packed[..logical])?;
        let mut weights = Zeroizing::new(vec![0.; rows * columns]);
        info.dtype().decode(&packed[..logical], &mut weights)?;
        let public = client.create_from_slice(&packed);
        for count in [1, 4, 8] {
            let input = fixture(columns, dtype as usize);
            let input_id = arena.upload_f32(&input)?;
            let stride = columns / 2 + columns / 16;
            let planes = arena.allocate_zeroed(MAX_BATCH * stride)?;
            let output = arena.allocate_zeroed(MAX_BATCH * rows)?;
            let fault = arena.allocate_zeroed(1)?;
            let spans = BatchSpans::new(count, 0, 0, columns, 0, rows);
            dense::prepare(
                client,
                columns,
                u::handle(arena, input_id)?,
                u::handle(arena, planes)?,
                u::handle(arena, fault)?,
                &spans,
            )?;
            dense::project(
                client,
                dtype,
                rows,
                columns,
                public.clone(),
                u::handle(arena, planes)?,
                u::handle(arena, output)?,
                u::handle(arena, fault)?,
                0,
                rows,
                count,
            )?;
            if native::read_id(arena, fault)?.iter().any(|&x| x != 0) {
                return Err("public dense native fault".into());
            }
            let actual_planes = native::read_id(arena, planes)?;
            let actual = native::read_id(arena, output)?;
            let mut expected_planes = Zeroizing::new(vec![0u32; actual_planes.len()]);
            let mut expected = Zeroizing::new(vec![0f32; actual.len()]);
            for token in 0..count {
                let (values, image) = quantize(&input[token * columns..][..columns])?;
                expected_planes[token * stride..][..stride].copy_from_slice(&image);
                for (row, w) in weights.chunks_exact(columns).enumerate() {
                    expected[token * rows + row] = w
                        .iter()
                        .zip(values.iter())
                        .map(|(&w, &x)| f64::from(w) * f64::from(x))
                        .sum::<f64>() as f32;
                }
            }
            let plane_mismatches = actual_planes
                .iter()
                .zip(expected_planes.iter())
                .filter(|(a, b)| a != b)
                .count();
            let mut mismatches = 0;
            let mut max_abs = 0f32;
            for (i, (&bits, &reference)) in actual.iter().zip(expected.iter()).enumerate() {
                let value = f32::from_bits(bits);
                let error = (value - reference).abs();
                if i >= count * rows {
                    if bits != 0 {
                        return Err("public dense inactive/full-owner padding".into());
                    }
                } else {
                    mismatches +=
                        usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
                    max_abs = max_abs.max(error);
                }
            }
            println!(
                "public_native_dense_numeric dtype={dtype} columns={columns} rows={rows} count={count} plane_mismatches={plane_mismatches} mismatches={mismatches} max_abs={max_abs}"
            );
            reports.push(
                serde_json::json!({"dtype":dtype,"columns":columns,"rows":rows,
                "count":count,"plane_mismatches":plane_mismatches,"mismatches":mismatches,
                "max_abs":max_abs,"actual_native_kernels":true,"full_owner_and_tail_checked":true}),
            );
            arena.reset()?;
            if plane_mismatches != 0 || mismatches != 0 {
                return Err("public native dense numerical parity failed".into());
            }
        }
    }
    if reports.is_empty() {
        return Err("public dense original formats missing".into());
    }
    Ok(reports)
}
