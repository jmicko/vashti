//! Compiled-public expert-shaped timings of the original matrix primitive.
//! No Session, caller input or altered admission. Numeric samples are an extra
//! diagnostic; the complete original-weight matrix acceptance remains required.
use super::*;
use crate::{gr_batch_probe as probe, private_arena::PrivateArena, resident_util as u};
use std::time::Instant;
use zeroize::Zeroizing;

type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const COUNTS: [usize; 11] = [1, 3, 7, 8, 16, 31, 32, 64, 80, 160, 320];
const WARM: usize = 3;
const REPEATS: usize = 64;

struct Weights {
    rows: usize,
    columns: usize,
    dtype: u32,
    values: Zeroizing<Vec<f32>>,
    packed: Zeroizing<Vec<u32>>,
}

fn original(file: &kuno_text::GgufFile, down: bool) -> ProbeResult<Weights> {
    let (rows, columns, suffix) = if down {
        (2560usize, 640usize, "ffn_down_exps.weight")
    } else {
        (640usize, 2560usize, "ffn_gate_exps.weight")
    };
    let (name, tensor) = file
        .tensors()
        .iter()
        .find(|(name, tensor)| {
            name.ends_with(suffix) && tensor.dimensions() == [columns as u64, rows as u64, 512]
        })
        .ok_or("public matrix timing original tensor")?;
    let row_bytes = usize::try_from(tensor.dtype().row_bytes(columns as u64)?)?;
    let mut raw = Zeroizing::new(vec![0u8; rows * row_bytes]);
    file.tensor(name)
        .ok_or("public matrix timing original view")?
        .read_range(0, &mut raw)?;
    let mut gate = Zeroizing::new(vec![0f32; rows * columns]);
    tensor.dtype().decode(&raw, &mut gate)?;
    let mut up = Zeroizing::new(Vec::new());
    if !down {
        let name = name.replace(suffix, "ffn_up_exps.weight");
        let tensor_up = file
            .tensor(&name)
            .ok_or("public matrix timing original up")?;
        if tensor_up.info().dimensions() != tensor.dimensions()
            || tensor_up.info().dtype() != tensor.dtype()
        {
            return Err("public matrix timing original up geometry".into());
        }
        tensor_up.read_range(0, &mut raw)?;
        up.resize(gate.len(), 0.);
        tensor.dtype().decode(&raw, &mut up)?;
    }
    let output_rows = if down { rows } else { rows * 2 };
    let mut values = Zeroizing::new(vec![0f32; output_rows * columns]);
    let mut packed = Zeroizing::new(vec![0u32; values.len() / 2]);
    for row in 0..output_rows {
        let source = if down || row % 2 == 0 { &gate } else { &up };
        let source_row = if down { row } else { row / 2 };
        for col in 0..columns {
            let v = half::f16::from_f32(source[source_row * columns + col]);
            if !v.is_finite() {
                return Err("public matrix timing nonfinite original weight".into());
            }
            let at = row * columns + col;
            values[at] = v.to_f32();
            packed[at / 2] |= u32::from(v.to_bits()) << (16 * (at % 2));
        }
    }
    Ok(Weights {
        rows: output_rows,
        columns,
        dtype: tensor.dtype().gguf_id(),
        values,
        packed,
    })
}

fn sync(client: &ComputeClient<WgpuRuntime>) -> ProbeResult<()> {
    cubecl::future::block_on(client.sync()).map_err(|_| "public matrix timing sync failed".into())
}

fn check(
    arena: &PrivateArena,
    output: crate::private_arena::BufferId,
    fault: crate::private_arena::BufferId,
    weights: &Weights,
    input: &[f32],
    count: usize,
) -> ProbeResult<(usize, f32)> {
    let actual = probe::read_native_public(arena, output)?;
    let result = check_values(&actual, weights, input, count)?;
    if probe::read_native_public(arena, fault)?
        .iter()
        .any(|&bits| bits != 0)
    {
        return Err("public matrix timing actual fault".into());
    }
    Ok(result)
}

fn check_values(
    actual: &[u32],
    weights: &Weights,
    input: &[f32],
    count: usize,
) -> ProbeResult<(usize, f32)> {
    if count == 0
        || count > 320
        || weights.rows < 65
        || input.len() != count * weights.columns
        || weights.values.len() != weights.rows * weights.columns
    {
        return Err("public matrix timing input geometry".into());
    }
    let live = count * weights.rows;
    if actual.len() < live
        || actual[..live]
            .iter()
            .any(|&bits| !f32::from_bits(bits).is_finite())
        || actual[live..].iter().any(|&bits| bits != 0)
    {
        return Err("public matrix timing full output owner".into());
    }
    let mut tokens = vec![0, 7.min(count - 1), count / 2, count - 1];
    tokens.sort_unstable();
    tokens.dedup();
    let mut rows = vec![0, 15, 16, 31, 32, 63, 64, weights.rows - 1];
    rows.sort_unstable();
    rows.dedup();
    let mut samples = 0;
    let mut max_abs = 0f32;
    for token in tokens {
        for &row in &rows {
            let w = &weights.values[row * weights.columns..][..weights.columns];
            let x = &input[token * weights.columns..][..weights.columns];
            let reference = w
                .iter()
                .zip(x)
                .map(|(&w, &x)| f64::from(w) * f64::from(x))
                .sum::<f64>() as f32;
            let value = f32::from_bits(actual[token * weights.rows + row]);
            let error = (value - reference).abs();
            if !reference.is_finite() || error > 1e-4 + 3e-4 * reference.abs() {
                return Err("public matrix timing sampled IEEE parity".into());
            }
            max_abs = max_abs.max(error);
            samples += 1;
        }
    }
    Ok((samples, max_abs))
}

fn timed(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    weights: &Weights,
    public: &Handle,
    down: bool,
    count: usize,
) -> ProbeResult<serde_json::Value> {
    let mut input = Zeroizing::new(vec![0f32; count * weights.columns]);
    let mut packed = Zeroizing::new(vec![0u32; input.len() / 2]);
    for (i, v) in input.iter_mut().enumerate() {
        let x = (((i * 7 + i / weights.columns * 13) % 43) as f32 - 21.) / 16.;
        let h = half::f16::from_f32(x);
        *v = h.to_f32();
        packed[i / 2] |= u32::from(h.to_bits()) << (16 * (i % 2));
    }
    let input_id = arena.upload_u32(&packed)?;
    let output = arena.allocate_zeroed(count * weights.rows)?;
    let fault = arena.allocate_zeroed(1)?;
    let handles = [
        u::handle(arena, input_id)?,
        public.clone(),
        u::handle(arena, output)?,
        u::handle(arena, fault)?,
    ];
    let launch = || {
        native::run_dense(
            client,
            handles.clone(),
            count,
            weights.columns,
            weights.rows,
            0,
            0,
            weights.rows,
            false,
        )
    };
    for _ in 0..WARM {
        launch()?;
    }
    sync(client)?;
    let _ = check(arena, output, fault, weights, &input, count)?;
    // The preflight read and its audited transient cleanup completed before
    // timing. Queue completion is genuine; enqueue time is not GPU time.
    sync(client)?;
    let start = Instant::now();
    for _ in 0..REPEATS {
        launch()?;
    }
    let enqueue_ns = start.elapsed().as_nanos();
    sync(client)?;
    let complete_ns = start.elapsed().as_nanos();
    let (samples, max_abs) = check(arena, output, fault, weights, &input, count)?;
    if complete_ns == 0 || enqueue_ns > complete_ns {
        return Err("public matrix timing clock".into());
    }
    let flops = 2u64 * count as u64 * weights.rows as u64 * weights.columns as u64;
    Ok(serde_json::json!({
        "family":"expert-shaped-original-matrix-timing",
        "role":if down {"down"} else {"gate-up"},
        "count":count,"rows":weights.rows,"columns":weights.columns,
        "original_gguf_dtype":weights.dtype,"bfloat":false,
        "warmups":WARM,"repeats":REPEATS,
        "enqueue_ns":u64::try_from(enqueue_ns)?,
        "complete_ns":u64::try_from(complete_ns)?,
        "ns_per_call":complete_ns as f64 / REPEATS as f64,
        "flops_per_call":flops,
        "gflops_per_second":flops as f64 * REPEATS as f64 / complete_ns as f64,
        "actual_native_primitive":true,"actual_completion_fence":true,
        "includes_host_enqueue":true,"full_session_loaded":false,
        "sampled_ieee_oracle":true,"samples":samples,"max_abs":max_abs,
        "full_owner_finite_padding_fault_checked":true,
        "does_not_replace_complete_numerical_acceptance":true
    }))
}

/// Caller configures the existing compiled-public transient policy and owns
/// final close/audit. Private roots remain in the arena on every per-shape
/// outcome until reset erases them; public weights stay live through reset.
pub(crate) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    file: &kuno_text::GgufFile,
) -> ProbeResult<Vec<serde_json::Value>> {
    if !std::ptr::eq(client.info(), arena.client()?.info()) {
        return Err("public matrix timing original client".into());
    }
    let mut reports = Vec::new();
    for down in [false, true] {
        let weights = original(file, down)?;
        let public = client.create_from_slice(u32::as_bytes(&weights.packed));
        for count in COUNTS {
            let attempt = timed(client, arena, &weights, &public, down, count);
            let erased = arena.reset();
            erased?;
            reports.push(attempt?);
        }
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(count: usize) -> (Weights, Vec<f32>, Vec<u32>) {
        let (rows, columns) = (65, 16);
        let values = Zeroizing::new(
            (0..rows * columns)
                .map(|i| ((i * 7 % 23) as f32 - 11.) / 16.)
                .collect::<Vec<_>>(),
        );
        let input = (0..count * columns)
            .map(|i| ((i * 13 % 17) as f32 - 8.) / 16.)
            .collect::<Vec<_>>();
        let mut actual = vec![0u32; count * rows + 16];
        for token in 0..count {
            for row in 0..rows {
                actual[token * rows + row] = (values[row * columns..][..columns]
                    .iter()
                    .zip(&input[token * columns..][..columns])
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum::<f64>() as f32)
                    .to_bits();
            }
        }
        (
            Weights {
                rows,
                columns,
                dtype: 0,
                values,
                packed: Zeroizing::new(Vec::new()),
            },
            input,
            actual,
        )
    }

    #[test]
    fn fixed_samples_cover_token_and_feature_boundaries_with_original_ieee_tolerance() {
        for count in COUNTS {
            let (weights, input, actual) = fixture(count);
            let (samples, max_abs) = check_values(&actual, &weights, &input, count).unwrap();
            assert!(samples >= 7);
            assert_eq!(max_abs, 0.);
            for (token, row) in [(0, 0), (count - 1, 64), (count / 2, 31)] {
                let mut wrong = actual.clone();
                wrong[token * weights.rows + row] = 1e8f32.to_bits();
                assert!(check_values(&wrong, &weights, &input, count).is_err());
            }
        }
    }

    #[test]
    fn full_owner_nonfinite_padding_and_short_read_are_rejected_beyond_sampled_dots() {
        let (weights, input, actual) = fixture(32);
        let mut wrong = actual.clone();
        wrong[weights.rows + 2] = f32::NAN.to_bits(); // Not a sampled dot.
        assert!(check_values(&wrong, &weights, &input, 32).is_err());
        let mut wrong = actual.clone();
        wrong[32 * weights.rows] = (-0.0f32).to_bits();
        assert!(check_values(&wrong, &weights, &input, 32).is_err());
        assert!(check_values(&actual[..32 * weights.rows - 1], &weights, &input, 32).is_err());
        assert!(check_values(&actual, &weights, &input, 0).is_err());
    }
}
