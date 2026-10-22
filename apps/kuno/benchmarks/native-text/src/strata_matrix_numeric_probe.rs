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
    let (max_rows, columns) = (
        if cfg!(any(
            feature = "strata-cubek-wide",
            feature = "strata-native-wide"
        )) {
            320usize
        } else {
            65usize
        },
        10240usize,
    );
    let tensor = file
        .tensor("blk.0.hc_attn_down.weight")
        .ok_or("public matrix original weight")?;
    if tensor.info().dimensions() != [10240, 320] || tensor.info().dtype().gguf_id() != 30 {
        return Err("public matrix original geometry".into());
    }
    let mut original = Zeroizing::new(vec![0u8; max_rows * columns * 2]);
    tensor.read_range(0, &mut original)?;
    let mut decoded = Zeroizing::new(vec![0.; max_rows * columns]);
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
        // Existing independent IEEE cases plus exact/full macro tiles and
        // both token/feature tails. The same checked primitive is selected.
        let original_shapes = [
            (1, 33),
            (8, 33),
            (35, 33),
            (257, 33),
            (16, 64),
            (64, 64),
            (65, 65),
            // Additional full IEEE checks for both narrow staged shapes and
            // their boundary/tail owners. Keep all seven original cases above
            // and the original reference/tolerance unchanged.
            (3, 33),
            (7, 33),
            (9, 64),
            (16, 33),
            (17, 65),
            (31, 64),
            // Keep the prior26 controls and exercise aligned output rows with
            // non-tile-aligned M as well, including a multi-stage long tail.
            (35, 64),
            (257, 64),
        ];
        let wide_shapes: &[(usize, usize)] = if cfg!(any(
            feature = "strata-cubek-wide",
            feature = "strata-native-wide"
        )) {
            // Full M/N, M tail, then a simultaneous M/N tail from the
            // original complete320-row weight. Preserve the same F64 oracle.
            &[(512, 128), (513, 128), (513, 320)]
        } else {
            &[]
        };
        // Keep all36 original cases. These genuine model-shape cases use the
        // unpadded320-row original BF16 tensor, including the missing64 rows.
        let gr_shapes: &[(usize, usize)] = if bfloat && cfg!(feature = "strata-gr-wide-tail") {
            &[(512, 320), (640, 320), (512, 320)]
        } else {
            &[]
        };
        for (case, &(count, rows)) in original_shapes
            .iter()
            .chain(wide_shapes)
            .chain(gr_shapes)
            .enumerate()
        {
            let gr_case = case >= original_shapes.len() + wide_shapes.len();
            let poison = gr_case && case == original_shapes.len() + wide_shapes.len() + 2;
            let mut case_weights = Zeroizing::new(weights.to_vec());
            let mut case_packed = Zeroizing::new(packed.to_vec());
            if poison {
                for (at, bits) in [(0, 0x7fc1u16), (319 * columns, 0x7f80u16)] {
                    case_weights[at] = half::bf16::from_bits(bits).to_f32();
                    let shift = 16 * (at % 2);
                    case_packed[at / 2] =
                        (case_packed[at / 2] & !(0xffff << shift)) | (u32::from(bits) << shift);
                }
            }
            let case_public = if poison {
                client.create_from_slice(u32::as_bytes(&case_packed))
            } else {
                public.clone()
            };
            let mut inputs = Zeroizing::new(vec![0.; count * columns]);
            let mut image = Zeroizing::new(vec![0u32; inputs.len() / 2]);
            for (i, v) in inputs.iter_mut().enumerate() {
                let x = (((i * 7 + i / columns * 13) % 43) as f32 - 21.) / 16.;
                let (bits, value) = rounded(x, bfloat);
                *v = value;
                image[i / 2] |= u32::from(bits) << (16 * (i % 2));
            }
            let input = arena.upload_u32(&image)?;
            let output = arena.allocate_zeroed(count * rows + if gr_case { 64 } else { 0 })?;
            let fault = arena.allocate_zeroed(1)?;
            // Same original compiled primitive, authentic checked arena roots.
            // No model recording or source ownership is manufactured here.
            let handles = [
                u::handle(arena, input)?,
                case_public,
                u::handle(arena, output)?,
                u::handle(arena, fault)?,
            ];
            #[cfg(feature = "strata-native-wide")]
            let actual_native_wide_launch = super::native_wide::try_dense(
                client,
                handles.clone(),
                count,
                columns,
                rows,
                0,
                0,
                rows,
                bfloat,
            )?;
            #[cfg(not(feature = "strata-native-wide"))]
            let actual_native_wide_launch = false;
            #[cfg(feature = "strata-cubek-gemm")]
            let actual_cubek_launch = super::cubek::try_dense(
                client,
                handles.clone(),
                count,
                columns,
                rows,
                0,
                0,
                rows,
                bfloat,
            )?;
            #[cfg(not(feature = "strata-cubek-gemm"))]
            let actual_cubek_launch = false;
            if !actual_cubek_launch && !actual_native_wide_launch {
                native::run_dense(client, handles, count, columns, rows, 0, 0, rows, bfloat)?;
            }
            #[cfg(feature = "strata-cubek-wide")]
            let actual_cubek_wide_launch =
                actual_cubek_launch && super::cubek::public_dense_uses_wide(client, count);
            #[cfg(not(feature = "strata-cubek-wide"))]
            let actual_cubek_wide_launch = false;
            if cfg!(feature = "strata-cubek-wide") && count >= 512 && !actual_cubek_wide_launch {
                return Err("public wide matrix actual selection rejected".into());
            }
            if cfg!(feature = "strata-native-wide")
                && actual_native_wide_launch
                    != (count >= 512 && count % 128 == 0 && (rows % 128 == 0 || gr_case))
            {
                return Err("public native wide actual selection rejected".into());
            }
            let actual = probe::read_native_public(arena, output)?;
            if probe::read_native_public(arena, fault)?
                .iter()
                .any(|&v| v != u32::from(poison))
            {
                return Err("public matrix actual native fault".into());
            }
            let mut mismatches = 0usize;
            let mut max_abs = 0f32;
            let mut ftz_mismatches = 0usize;
            let mut ftz_max_abs = 0f32;
            for (i, &bits) in actual.iter().enumerate() {
                if i >= count * rows {
                    if bits != 0 {
                        return Err("public matrix full-owner padding".into());
                    }
                    continue;
                }
                let w = &case_weights[i % rows * columns..][..columns];
                let x = &inputs[i / rows * columns..][..columns];
                let reference = w
                    .iter()
                    .zip(x)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum::<f64>() as f32;
                let value = f32::from_bits(bits);
                if poison && !reference.is_finite() {
                    mismatches += usize::from(bits != 0);
                    continue;
                }
                let error = (value - reference).abs();
                mismatches +=
                    usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
                max_abs = max_abs.max(error);
                if !bfloat {
                    // Additional diagnosis only: do not replace the IEEE
                    // reference or its unchanged acceptance threshold.
                    let minimum = half::f16::MIN_POSITIVE.to_f32();
                    let ftz = w
                        .iter()
                        .zip(x)
                        .map(|(&w, &x)| {
                            if w.abs() < minimum || x.abs() < minimum {
                                0.
                            } else {
                                f64::from(w) * f64::from(x)
                            }
                        })
                        .sum::<f64>() as f32;
                    let e = (value - ftz).abs();
                    ftz_mismatches +=
                        usize::from(!value.is_finite() || e > 1e-4 + 3e-4 * ftz.abs());
                    ftz_max_abs = ftz_max_abs.max(e);
                }
            }
            println!(
                "public_native_matrix_numeric bfloat={bfloat} count={count} rows={rows} columns={columns} mismatches={mismatches} max_abs={max_abs} ftz_mismatches={ftz_mismatches} ftz_max_abs={ftz_max_abs}"
            );
            reports.push(
                serde_json::json!({"family":"cooperative-matrix","bfloat":bfloat,
                "count":count,"rows":rows,"columns":columns,"mismatches":mismatches,
                "max_abs":max_abs,"ftz_oracle_mismatches":ftz_mismatches,"ftz_max_abs":ftz_max_abs,
                "actual_cubek_launch":actual_cubek_launch,
                "actual_cubek_wide_launch":actual_cubek_wide_launch,
                "actual_native_wide_launch":actual_native_wide_launch,
                "gr_wide_tail_case":gr_case,"nonfinite_weight_poison":poison,
                "ftz_is_additional_diagnosis_only":true,"actual_native_kernels":true,"full_owner_and_tail_checked":true}),
            );
            arena.reset()?;
            if mismatches != 0 {
                return Err("public native matrix numerical parity failed".into());
            }
        }
    }
    Ok(reports)
}
