//! Independent fixed-public F64 Q8 attention and operation-derived F32 bounds.
//! No shader/source parsing, GPU authority, tolerance fit or private input path.

pub(super) const QUERY: usize = 6144;
pub(super) const SELECT: usize = 2051;
pub(super) const PARTIAL: usize = 202752;
pub(super) const STATS: usize = 1584;
const ROWS: usize = 8;
const CACHE: usize = 264;
const U: f64 = 1.0 / 16777216.0;

fn gamma(rounds: usize) -> f64 {
    let nu = rounds as f64 * U;
    nu / (1.0 - nu)
}

fn half(bits: u16) -> Result<f64, &'static str> {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    match exponent {
        0 => Ok(sign * f64::from(fraction) / 16777216.0),
        31 => Err("public QSA F64 nonfinite half scale"),
        _ => Ok(sign * f64::from(1024 + fraction) * 2f64.powi(i32::from(exponent) - 25)),
    }
}

fn value(cache: &[u32], token: usize, element: usize) -> Result<f64, &'static str> {
    let raw = cache[token * CACHE + element / 4];
    let code = ((raw >> (8 * (element % 4))) as u8 as i8) as f64;
    let pair = cache[token * CACHE + 256 + element / 128];
    let scale = half((pair >> (16 * ((element / 64) % 2))) as u16)?;
    if scale < 0.0 {
        return Err("public QSA F64 negative half scale");
    }
    let decoded = code * scale;
    // These fixed fixtures use <=7-bit integer codes and finite FP16 scales:
    // their product is represented exactly in normal F32. Reject a changed
    // fixture needing another decode-rounding term rather than omit that term.
    if !decoded.is_finite() || f64::from(decoded as f32) != decoded {
        return Err("public QSA F64 Q8 decode is not exactly F32");
    }
    Ok(decoded)
}

#[derive(Clone, Copy)]
pub(super) struct Geometry {
    pub start: usize,
    pub count: usize,
    pub capacity: usize,
    pub query_offset: usize,
}

/// Fill caller-owned wiping arrays in mathematical [row,KV,chunk,head,feature]
/// order, independently of subgroup lane/reduction/store implementation.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate(
    g: Geometry,
    query: &[u32],
    cache: &[u32],
    selected: &[u32],
    partial: &mut [f64],
    stats: &mut [f64],
    partial_bound: &mut [f64],
    stats_bound: &mut [f64],
) -> Result<(), &'static str> {
    evaluate_rows(
        g,
        query,
        cache,
        selected,
        partial,
        stats,
        partial_bound,
        stats_bound,
        ROWS,
    )
}

/// Additional full-batch control; original entry retains its exact8-row domain
/// and the identical mathematical loops and operation-derived bounds.
#[allow(clippy::too_many_arguments)]
pub(super) fn evaluate_large(
    g: Geometry,
    query: &[u32],
    cache: &[u32],
    selected: &[u32],
    partial: &mut [f64],
    stats: &mut [f64],
    partial_bound: &mut [f64],
    stats_bound: &mut [f64],
) -> Result<(), &'static str> {
    evaluate_rows(
        g,
        query,
        cache,
        selected,
        partial,
        stats,
        partial_bound,
        stats_bound,
        32,
    )
}
#[allow(clippy::too_many_arguments)]
fn evaluate_rows(
    g: Geometry,
    query: &[u32],
    cache: &[u32],
    selected: &[u32],
    partial: &mut [f64],
    stats: &mut [f64],
    partial_bound: &mut [f64],
    stats_bound: &mut [f64],
    rows: usize,
) -> Result<(), &'static str> {
    if !(1..=rows).contains(&g.count)
        || ![0, 32, 64].contains(&g.query_offset)
        || g.capacity > 32768
        || g.start
            .checked_add(g.query_offset)
            .and_then(|n| n.checked_add(g.count))
            .is_none_or(|end| end > g.capacity)
        || query.len() != (g.query_offset + rows) * QUERY
        || cache.len() != g.capacity * CACHE
        || selected.len() != (g.query_offset + rows) * SELECT
        || partial.len() != rows * PARTIAL
        || partial_bound.len() != partial.len()
        || stats.len() != rows * STATS
        || stats_bound.len() != stats.len()
    {
        return Err("public QSA F64 original operand geometry");
    }
    partial.fill(0.0);
    stats.fill(0.0);
    partial_bound.fill(0.0);
    stats_bound.fill(0.0);
    // This is fixed PUBLIC data only. Decode each original cache word once;
    // mathematical loops below never use a shader-produced intermediate.
    let mut decoded = vec![0.0; g.capacity * 1024];
    for token in 0..g.capacity {
        for element in 0..1024 {
            decoded[token * 1024 + element] = value(cache, token, element)?;
        }
    }
    for row in 0..g.count {
        let qrow = row + g.query_offset;
        let visible = g.start + qrow + 1;
        let cells = visible.min(SELECT);
        for kv in 0..2 {
            for chunk in 0..cells.div_ceil(64) {
                let first = chunk * 64;
                let here = (cells - first).min(64);
                let mut tokens = [0usize; 64];
                for cell in 0..here {
                    let token = selected[qrow * SELECT + first + cell] as usize;
                    if token >= visible || token >= g.capacity {
                        return Err("public QSA F64 noncausal selected cell");
                    }
                    tokens[cell] = token;
                }
                for head in 0..12 {
                    let mut scores = [0.0f64; 64];
                    let mut score_bounds = [0.0f64; 64];
                    for cell in 0..here {
                        let mut dot = 0.0;
                        let mut dot_l1 = 0.0;
                        for feature in 0..256 {
                            let q = f64::from(f32::from_bits(
                                query[qrow * QUERY + (kv * 12 + head) * 256 + feature],
                            ));
                            if !q.is_finite() {
                                return Err("public QSA F64 nonfinite live query");
                            }
                            let k = decoded[tokens[cell] * 1024 + kv * 256 + feature];
                            let product = q * k;
                            dot += product;
                            dot_l1 += product.abs();
                        }
                        scores[cell] = dot / 16.0;
                        // Each scalar path: one rounded product, seven local
                        // adds, five shuffle adds; gamma13 also covers fused
                        // contraction. WGSL division permits2.5ULP: <=5u|s|.
                        let dot_bound =
                            gamma(13) * dot_l1 / 16.0 + 512.0 * f64::EPSILON * dot_l1 / 16.0;
                        score_bounds[cell] = dot_bound + 5.0 * U * (scores[cell].abs() + dot_bound);
                    }
                    let maximum = scores[..here]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let max_bound = score_bounds[..here].iter().copied().fold(0.0, f64::max);
                    let mut weights = [0.0f64; 64];
                    let mut weight_bounds = [0.0f64; 64];
                    for cell in 0..here {
                        let x = scores[cell] - maximum;
                        let delta = score_bounds[cell] + max_bound;
                        let x_bound = delta + gamma(1) * (x.abs() + delta);
                        weights[cell] = x.exp();
                        // WGSL exp accuracy is (3+2|x|)ULP. One normal F32 ULP
                        // is <=2u times magnitude; include the largest allowed
                        // argument and exponent value, plus propagated QK/max
                        // and subtraction error. No measured error enters.
                        let largest_exp = (x + x_bound).exp();
                        if weights[cell] < f64::from(f32::MIN_POSITIVE) || !largest_exp.is_finite()
                        {
                            return Err("public QSA F64 normal exp bound domain");
                        }
                        weight_bounds[cell] = weights[cell] * x_bound.exp_m1()
                            + (3.0 + 2.0 * (x.abs() + x_bound)) * 2.0 * U * largest_exp
                            + 8.0 * f64::EPSILON * largest_exp;
                    }
                    let total: f64 = weights[..here].iter().sum();
                    let total_error: f64 = weight_bounds[..here].iter().sum();
                    let stat = row * STATS + ((kv * 33 + chunk) * 12 + head) * 2;
                    stats[stat] = maximum;
                    stats_bound[stat] = max_bound;
                    stats[stat + 1] = total;
                    // Pairwise a+b then five subgroup adds on positive terms.
                    stats_bound[stat + 1] = total_error + gamma(6) * (total + total_error);
                    for feature in 0..256 {
                        let mut sum = 0.0;
                        let mut l1 = 0.0;
                        let mut weight_error = 0.0;
                        for cell in 0..here {
                            let v = decoded[tokens[cell] * 1024 + 512 + kv * 256 + feature];
                            let term = weights[cell] * v;
                            sum += term;
                            l1 += term.abs();
                            weight_error += weight_bounds[cell] * v.abs();
                        }
                        let at = row * PARTIAL + ((kv * 33 + chunk) * 12 + head) * 256 + feature;
                        partial[at] = sum;
                        //64 explicit FMAs. WGSL permits inherited multiply+add
                        // precision; gamma128 safely includes unfused results.
                        partial_bound[at] = weight_error
                            + gamma(128) * (l1 + weight_error)
                            + 128.0 * f64::EPSILON * l1;
                    }
                }
            }
        }
    }
    decoded.fill(0.0);
    Ok(())
}

#[derive(Default, Debug)]
pub(super) struct Comparison {
    pub checked: usize,
    pub mismatches: usize,
    pub nonfinite: usize,
    pub max_abs: f64,
    pub normalized_max_error: f64,
    pub max_bound: f64,
    pub max_error_to_bound: f64,
}

pub(super) fn compare(expected: &[f64], bounds: &[f64], actual: &[u32]) -> Comparison {
    let mut out = Comparison::default();
    if bounds.len() != expected.len() || actual.len() < expected.len() {
        out.mismatches = 1;
        return out;
    }
    let mut norm = 0f64;
    for (index, &bits) in actual.iter().enumerate() {
        let reference = expected.get(index).copied().unwrap_or(0.0);
        let bound = bounds.get(index).copied().unwrap_or(0.0);
        let value = f64::from(f32::from_bits(bits));
        out.checked += 1;
        if !reference.is_finite() || !bound.is_finite() || bound < 0.0 || !value.is_finite() {
            out.nonfinite += 1;
            out.mismatches += 1;
            continue;
        }
        norm = norm.max(reference.abs());
        let error = (value - reference).abs();
        out.max_abs = out.max_abs.max(error);
        out.max_bound = out.max_bound.max(bound);
        if error > bound {
            out.mismatches += 1;
        }
        if bound > 0.0 {
            out.max_error_to_bound = out.max_error_to_bound.max(error / bound);
        }
    }
    out.normalized_max_error = out.max_abs / norm.max(1e-30);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Public {
        query: Vec<u32>,
        cache: Vec<u32>,
        selected: Vec<u32>,
    }
    fn fixture(g: Geometry) -> Public {
        let mut query = vec![f32::NAN.to_bits(); (g.query_offset + ROWS) * QUERY];
        for row in g.query_offset..g.query_offset + g.count {
            for feature in 0..QUERY {
                let i = row * QUERY + feature;
                query[i] = ((((i * 17 + row * 3) % 43) as f32 - 21.0) / 31.0
                    + ((i / 256 % 7) as f32 - 3.0) / 127.0)
                    .to_bits();
            }
        }
        let mut cache = vec![0u32; g.capacity * CACHE];
        let scales = [0x2113u32, 0x2487, 0x299d, 0x2c53];
        for token in 0..g.capacity {
            for word in 0..256 {
                for byte in 0..4 {
                    let code = ((token * 7 + (word * 4 + byte) * 13) % 127) as i32 - 63;
                    cache[token * CACHE + word] |= u32::from(code as i8 as u8) << (byte * 8);
                }
            }
            for word in 0..8 {
                cache[token * CACHE + 256 + word] =
                    scales[(token + word * 2) % 4] | (scales[(token + word * 2 + 1) % 4] << 16);
            }
        }
        let mut selected = vec![u32::MAX; (g.query_offset + ROWS) * SELECT];
        for row in g.query_offset..g.query_offset + g.count {
            let visible = g.start + row + 1;
            let width = visible.min(SELECT);
            for cell in 0..width {
                selected[row * SELECT + cell] = (visible - width + cell) as u32;
            }
        }
        Public {
            query,
            cache,
            selected,
        }
    }
    fn outputs(g: Geometry, f: &Public) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut p = vec![0.0; ROWS * PARTIAL];
        let mut s = vec![0.0; ROWS * STATS];
        let mut pb = p.clone();
        let mut sb = s.clone();
        evaluate(
            g,
            &f.query,
            &f.cache,
            &f.selected,
            &mut p,
            &mut s,
            &mut pb,
            &mut sb,
        )
        .unwrap();
        (p, s, pb, sb)
    }
    fn rounded(values: &[f64]) -> Vec<u32> {
        values.iter().map(|&v| (v as f32).to_bits()).collect()
    }
    #[test]
    fn independent_q8_unpack_and_f64_math_accept_rounded_results_at_actual_offset() {
        assert_eq!(half(0x3c00).unwrap(), 1.0);
        assert_eq!(half(1).unwrap(), 2f64.powi(-24));
        assert!(half(0x7c00).is_err());
        let g = Geometry {
            start: 1,
            count: 2,
            capacity: 128,
            query_offset: 32,
        };
        let f = fixture(g);
        let (p, s, pb, sb) = outputs(g, &f);
        assert_eq!(compare(&p, &pb, &rounded(&p)).mismatches, 0);
        assert_eq!(compare(&s, &sb, &rounded(&s)).mismatches, 0);
        assert!(pb.iter().copied().fold(0.0, f64::max) < 0.001);
        assert!(sb.iter().copied().fold(0.0, f64::max) < 0.001);
        // Every complete owner row not launched remains an exact zero.
        assert!(p[g.count * PARTIAL..].iter().all(|&v| v == 0.0));
        assert!(s[g.count * STATS..].iter().all(|&v| v == 0.0));
    }
    #[test]
    fn wrong_query_offset_k_value_head_order_and_tail_are_rejected() {
        let g = Geometry {
            start: 31,
            count: 1,
            capacity: 128,
            query_offset: 32,
        };
        let mut f = fixture(g);
        let (p, s, pb, sb) = outputs(g, &f);
        let mut bad = rounded(&p);
        bad.swap(0, 256); // Different original query heads, not a lane tolerance.
        assert!(compare(&p, &pb, &bad).mismatches > 0);
        let mut bad = rounded(&s);
        bad.swap(0, 24); // Different64-cell chunk, including inactive padding.
        assert!(compare(&s, &sb, &bad).mismatches > 0);
        let mut bad = rounded(&p);
        bad[PARTIAL] = 1f32.to_bits();
        assert!(compare(&p, &pb, &bad).mismatches > 0);
        let mut bad = rounded(&s);
        bad.push(1f32.to_bits());
        assert!(compare(&s, &sb, &bad).mismatches > 0);
        let mut bad = rounded(&p);
        bad[0] = f32::NAN.to_bits();
        assert_eq!(compare(&p, &pb, &bad).nonfinite, 1);
        // A wrong row offset sees deliberately poisoned query rows.
        let wrong = Geometry {
            query_offset: 0,
            ..g
        };
        let mut p2 = p.clone();
        let mut s2 = s.clone();
        let mut pb2 = pb.clone();
        let mut sb2 = sb.clone();
        assert!(
            evaluate(
                wrong,
                &f.query[..ROWS * QUERY],
                &f.cache,
                &f.selected[..ROWS * SELECT],
                &mut p2,
                &mut s2,
                &mut pb2,
                &mut sb2
            )
            .is_err()
        );
        // Corrupt an actual selected token's K code; the expected mathematical
        // reference stays original. It must not become another GPU-derived oracle.
        f.cache[0] ^= 0x3f;
        let (bad_p, bad_s, _, _) = outputs(g, &f);
        assert!(
            compare(&p, &pb, &rounded(&bad_p)).mismatches > 0
                || compare(&s, &sb, &rounded(&bad_s)).mismatches > 0
        );
    }
    #[test]
    fn all_three_actual_nondyadic_geometries_have_finite_operation_bounds() {
        for (start, count, query_offset) in [(17, 5, 0), (31, 3, 32), (1983, 8, 64)] {
            let g = Geometry {
                start,
                count,
                capacity: 4096,
                query_offset,
            };
            let f = fixture(g);
            let (p, s, pb, sb) = outputs(g, &f);
            let pc = compare(&p, &pb, &rounded(&p));
            let sc = compare(&s, &sb, &rounded(&s));
            assert_eq!(pc.mismatches, 0);
            assert_eq!(sc.mismatches, 0);
            assert!(pc.max_bound < 0.001 && sc.max_bound < 0.001);
            println!(
                "public_f64_oracle_geometry start={start} count={count} offset={query_offset} partial_max_bound={} stats_max_bound={}",
                pc.max_bound, sc.max_bound
            );
        }
    }
}
