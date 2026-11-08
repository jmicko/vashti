//! Fixed synthetic inputs and original model weights through the ACTUAL
//! selected fused native kernels. CPU oracle comparisons remain in RAM.
//! No request input, graph substitution or speed measurement.
use super::*;
use crate::strata_gr as native;

pub(super) fn read_id(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
    let layout = arena.layout(id)?;
    let mut bytes = arena.client()?.read_one(arena.handle(id)?.clone())?;
    let words = decode_read(&mut bytes, layout.owned_bytes);
    drop(bytes);
    audit(arena.client()?)?;
    words
}

fn difference(
    actual: &[u32],
    expected: &[[Zeroizing<Vec<f32>>; 7]],
    stage: usize,
    count: usize,
    width: usize,
) -> Result<(usize, f32)> {
    if count == 0
        || count > MAX_BATCH
        || expected.len() != MAX_BATCH
        || actual.len() < MAX_BATCH * width
        || expected.iter().any(|row| row[stage].len() != width)
    {
        return Err("native GR numeric owner geometry".into());
    }
    let mut mismatches = 0;
    let mut max = 0_f32;
    for (i, &bits) in actual.iter().enumerate() {
        if i < count * width {
            let value = f32::from_bits(bits);
            let reference = expected[i / width][stage][i % width];
            let error = (value - reference).abs();
            mismatches += usize::from(!value.is_finite() || error > 1e-4 + 3e-4 * reference.abs());
            max = max.max(error);
        } else if bits != 0 {
            return Err("native GR inactive or complete owning padding changed".into());
        }
    }
    Ok((mismatches, max))
}

pub(super) fn cases(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    public: &[PublicStage],
) -> Result<Vec<serde_json::Value>> {
    if !std::ptr::eq(client.info(), arena.client()?.info()) {
        return Err("native GR numeric client identity".into());
    }
    let mut reports = Vec::new();
    for (seed, p) in public.iter().enumerate() {
        let expected = oracle(p, seed)?;
        for count in [1, 4, 8] {
            let residual = arena.upload_f32(&fixture(10240, seed))?;
            let block = arena.upload_f32(&fixture(2560, seed + 3))?;
            let output = arena.allocate_zeroed(81920)?;
            let fault = arena.allocate_zeroed(1)?;
            let buffers = native::Buffers {
                normalized: arena.allocate_zeroed(81920)?,
                scales: arena.allocate_zeroed(64)?,
                partial: arena.allocate_zeroed(207360)?,
                down: arena.allocate_zeroed(2560)?,
                low: arena.allocate_zeroed(2560)?,
                injection: arena.allocate_zeroed(32)?,
                mixed: arena.allocate_zeroed(20480)?,
            };
            native::read(
                native::Read {
                    arena,
                    residual,
                    previous: None,
                    gamma: p.gamma.clone(),
                    buffers,
                    fault,
                    count,
                    first: 0,
                    epsilon: p.epsilon,
                    inject: p.injection.is_some(),
                },
                [
                    p.down.handle.clone(),
                    p.up.handle.clone(),
                    p.injection
                        .as_ref()
                        .map_or_else(|| p.down.handle.clone(), |m| m.handle.clone()),
                ],
            )?;
            if p.injection.is_some() {
                native::write(
                    arena,
                    residual,
                    block,
                    buffers.injection,
                    output,
                    fault,
                    count,
                )?;
            }
            let faults = read_id(arena, fault)?;
            if faults.iter().any(|&word| word != 0) {
                return Err("native GR actual fault packet rejected".into());
            }
            let mut all_matched = true;
            for (name, id, stage, width) in [
                ("normalized", buffers.normalized, 0, 10240),
                ("down", buffers.down, 1, 320),
                ("low", buffers.low, 2, 320),
                ("mixed", buffers.mixed, 4, 2560),
                ("injection", buffers.injection, 5, 4),
                ("combined", output, 6, 10240),
            ] {
                if stage >= 5 && p.injection.is_none() {
                    continue;
                }
                let actual = read_id(arena, id)?;
                let (mismatches, max_abs) = difference(&actual, &expected, stage, count, width)?;
                println!(
                    "public_native_gr_numeric base={} count={count} stage={name} mismatches={mismatches} max_abs={max_abs}",
                    p.base
                );
                all_matched &= mismatches == 0;
                reports.push(serde_json::json!({"base":p.base,"count":count,
                    "stage":name,"mismatches":mismatches,"max_abs":max_abs,
                    "actual_native_kernels":true,"full_owner_and_tail_checked":true}));
            }
            // Complete erasure precedes numerical acceptance, including a
            // failed comparison. No dirty owner receives recovery credit.
            arena.reset()?;
            if !all_matched {
                return Err("native GR CPU numerical parity failed".into());
            }
        }
    }
    // Keep the original caller's complete close/audit gate authoritative.
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_comparison_rejects_wrong_values_and_checks_unused_full_owner_tail() {
        let expected: Vec<_> = (0..8)
            .map(|_| std::array::from_fn(|_| Zeroizing::new(vec![1.; 4])))
            .collect();
        let mut actual = vec![0; 40];
        actual[..4].fill(1_f32.to_bits());
        assert_eq!(difference(&actual, &expected, 0, 1, 4).unwrap(), (0, 0.));
        actual[0] = 2_f32.to_bits();
        assert_eq!(difference(&actual, &expected, 0, 1, 4).unwrap(), (1, 1.));
        actual[0] = f32::NAN.to_bits();
        assert_eq!(difference(&actual, &expected, 0, 1, 4).unwrap().0, 1);
        actual[39] = 1;
        assert!(difference(&actual, &expected, 0, 1, 4).is_err());
    }
}
