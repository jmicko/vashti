//! MIT Strata fb58e0 tools/mtp_rt.py::q8_0 runtime weight conversion.
//! Original GGUF rows stay borrowed. This creates no converted model artifact.
use kuno_text::{Error, Result};

pub(crate) fn block(values: &[f32]) -> Result<[u8; 34]> {
    if values.len() != 32 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat(
            "Strata public runtime conversion block",
        ));
    }
    let scale = values.iter().map(|v| v.abs()).fold(0f32, f32::max) / 127.0;
    let half = half::f16::from_f32(scale);
    // Preserve the packer's F32 reciprocal followed by multiply. Division per
    // value can choose a different code at a rounding boundary.
    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    if !half.is_finite() || !inverse.is_finite() {
        return Err(Error::InvalidFormat("Strata Q8 weight scale"));
    }
    let mut packed = [0; 34];
    packed[..2].copy_from_slice(&half.to_bits().to_le_bytes());
    for (out, &value) in packed[2..].iter_mut().zip(values) {
        // Preserve the packer's F32 addition before floor too. Rust round
        // differs for the F32 immediately below0.5, whose +0.5 rounds to1.
        let scaled = value * inverse;
        let code = (scaled.abs() + 0.5).floor().copysign(scaled);
        if !code.is_finite() || !(-127.0..=127.0).contains(&code) {
            return Err(Error::InvalidFormat("Strata Q8 weight code"));
        }
        *out = code as i8 as u8;
    }
    Ok(packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_runtime_weights_match_pinned_numpy_golden_including_reciprocal_boundary() {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("strata_weight_q8_fixtures.json")).unwrap();
        let mut differing_division = false;
        for case in fixtures["cases"].as_array().unwrap() {
            let values: Vec<_> = case["f32_bits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| f32::from_bits(v.as_u64().unwrap() as u32))
                .collect();
            let expected: Vec<_> = case["bytes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect();
            let actual = block(&values).unwrap();
            assert_eq!(actual.as_slice(), expected, "{}", case["name"]);
            let scale = values.iter().map(|v| v.abs()).fold(0f32, f32::max) / 127.0;
            if scale > 0.0 {
                differing_division |= values
                    .iter()
                    .zip(&actual[2..])
                    .any(|(&v, &q)| (v / scale).round() as i8 as u8 != q);
            }
        }
        assert!(differing_division); // The fixture distinguishes the actual old operation order.
    }
    #[test]
    fn malformed_or_nonfinite_weight_blocks_reject_before_upload() {
        assert!(block(&[0.0; 31]).is_err());
        assert!(block(&[0.0; 33]).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
            let mut values = [0.0; 32];
            values[13] = value;
            assert!(block(&values).is_err());
        }
        assert_eq!(block(&[-0.0; 32]).unwrap(), [0; 34]);
        // A finite model scale that cannot produce a finite reciprocal has no
        // valid packed-code contract; do not silently saturate infinity to i8.
        assert!(block(&[f32::from_bits(1); 32]).is_ok()); // Scale underflows to0 as in the original packer.
        assert!(block(&[f32::from_bits(128); 32]).is_err());
    }
}
