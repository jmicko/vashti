//! Four or 32 fixed public original-Q3 positions from the independent CPU path.
//! An explicit 23-position prefix uses the complete validated 32-position file;
//! it ends before the known router discontinuity and is not 32-position acceptance.
//! No request-derived input, vocabulary text or expert history is persisted.
use kuno_text::model::greedy;
use kuno_text::{Error, Result};
use std::io::Read;
use std::path::Path;
use zeroize::Zeroizing;

const VOCAB: usize = 248320;
const RECORD_BYTES: usize = 16 + 4 * VOCAB;
const DEFAULT_POSITIONS: usize = 4;
const LONG_POSITIONS: usize = 32;
const PREFIX_POSITIONS: usize = 23;

/// Numeric diagnostics are reachable only through this fixed-public fixture.
/// Keep the original f32 acceptance equation: the f64 summaries below are
/// explanatory, and never decide whether a vector is accepted.
struct LogitDiagnostics {
    first_bad: Option<(usize, u32, u32)>,
    nonfinite_class: u8,
    max_abs_error: f64,
    max_normalized_error: f64,
}

impl LogitDiagnostics {
    #[cold]
    #[inline(never)]
    fn inspect(actual: &[f32], reference: &[f32]) -> Self {
        let mut out = Self {
            first_bad: None,
            nonfinite_class: 0,
            max_abs_error: 0.,
            max_normalized_error: 0.,
        };
        for (index, (&a, &r)) in actual.iter().zip(reference.iter()).enumerate() {
            let limit = 0.006 + 0.0003 * r.abs();
            if !a.is_finite() {
                if out.nonfinite_class == 0 {
                    out.nonfinite_class = if a.is_nan() {
                        1
                    } else if a.is_sign_positive() {
                        2
                    } else {
                        3
                    };
                }
            } else {
                // Finite f32 subtraction can overflow. The diagnostic remains
                // finite, while the unchanged f32 gate below still rejects it.
                let error = (f64::from(a) - f64::from(r)).abs();
                out.max_abs_error = out.max_abs_error.max(error);
                out.max_normalized_error = out.max_normalized_error.max(error / f64::from(limit));
            }
            if (!a.is_finite() || (a - r).abs() > limit) && out.first_bad.is_none() {
                out.first_bad = Some((index, r.to_bits(), a.to_bits()));
            }
        }
        out
    }

    fn report_failure(&self, position: usize, class: &'static str, greedy_matches: bool) {
        // VOCAB is the bounded absent-index sentinel; no sampled token IDs,
        // vocabulary text, model rows or backend errors are formatted.
        let (index, expected, actual) = self.first_bad.unwrap_or((VOCAB, 0, 0));
        println!(
            "public_q3_full_logits_failure position={position} class={class} values={VOCAB} first_bad_present={} first_bad_index={index} expected_bits=0x{expected:08x} actual_bits=0x{actual:08x} max_abs_error={} max_normalized_error={} nonfinite_class={} greedy_matches={greedy_matches}",
            self.first_bad.is_some(),
            self.max_abs_error,
            self.max_normalized_error,
            self.nonfinite_class,
        );
    }
}

fn positions_for_length(bytes: u64) -> Result<usize> {
    match bytes {
        n if n == (8 + DEFAULT_POSITIONS * RECORD_BYTES) as u64 => Ok(DEFAULT_POSITIONS),
        n if n == (8 + LONG_POSITIONS * RECORD_BYTES) as u64 => Ok(LONG_POSITIONS),
        _ => Err(Error::InvalidFormat("public Q3 reference length")),
    }
}

pub(crate) struct Reference {
    logits: Vec<Zeroizing<Vec<f32>>>,
    tokens: Vec<u32>,
}
impl Reference {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let positions = positions_for_length(file.metadata()?.len())?;
        let mut bytes = Zeroizing::new(vec![0; 8 + positions * RECORD_BYTES]);
        file.read_exact(&mut bytes)?;
        if file.read(&mut [0u8; 1])? != 0 {
            return Err(Error::InvalidFormat("public Q3 reference length"));
        }
        Self::parse(&bytes, positions)
    }

    fn parse(bytes: &[u8], positions: usize) -> Result<Self> {
        if positions_for_length(bytes.len() as u64)? != positions {
            return Err(Error::InvalidFormat("public Q3 reference length"));
        }
        if &bytes[..8] != b"KQ3R0001" {
            return Err(Error::InvalidFormat("public Q3 reference version"));
        }
        let mut tokens = vec![0; positions];
        let mut logits = Vec::with_capacity(positions);
        let mut cursor = 8;
        for position in 0..positions {
            let fields: Vec<_> = bytes[cursor..cursor + 16]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            if fields[0] != position as u32
                || fields[1]
                    != if position == 0 {
                        0
                    } else {
                        tokens[position - 1]
                    }
                || fields[2] as usize >= VOCAB
                || fields[3] != VOCAB as u32
            {
                return Err(Error::InvalidFormat("public Q3 reference positions"));
            }
            tokens[position] = fields[2];
            cursor += 16;
            let values = Zeroizing::new(
                bytes[cursor..cursor + VOCAB * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>(),
            );
            if !values.iter().all(|v| v.is_finite()) || greedy(&values)? != tokens[position] {
                return Err(Error::InvalidFormat("public Q3 reference logits"));
            }
            cursor += VOCAB * 4;
            logits.push(values);
        }
        Ok(Self { logits, tokens })
    }
    pub(crate) fn len(&self) -> usize {
        self.logits.len()
    }
    pub(crate) fn require_positions(&self, positions: usize) -> Result<()> {
        let required = match positions {
            DEFAULT_POSITIONS | LONG_POSITIONS => positions,
            PREFIX_POSITIONS => LONG_POSITIONS,
            _ => return Err(Error::InvalidFormat("public Q3 selected position count")),
        };
        if self.len() != required {
            return Err(Error::InvalidFormat("public Q3 selected position count"));
        }
        Ok(())
    }
    pub(crate) fn verify(&self, position: usize, actual: &[f32]) -> Result<()> {
        let reference = self.logits.get(position).ok_or(Error::OutOfBounds)?;
        if actual.len() != VOCAB {
            println!(
                "public_q3_full_logits_failure position={position} class=width expected_values={VOCAB} actual_values={} greedy_matches=false",
                actual.len(),
            );
            return Err(Error::InvalidFormat("public Q3 greedy mismatch"));
        }
        let selected = match greedy(actual) {
            Ok(selected) => selected,
            Err(error) => {
                LogitDiagnostics::inspect(actual, reference).report_failure(
                    position,
                    "nonfinite",
                    false,
                );
                return Err(error);
            }
        };
        if selected != self.tokens[position] {
            LogitDiagnostics::inspect(actual, reference).report_failure(position, "greedy", false);
            return Err(Error::InvalidFormat("public Q3 greedy mismatch"));
        }
        // Accepted benchmark vectors retain the original f32 hot loop. The
        // full diagnostic scan/divisions occur only after an unchanged failure.
        let mut max_error = 0f32;
        for (&a, &r) in actual.iter().zip(reference.iter()) {
            if !a.is_finite() || (a - r).abs() > 0.006 + 0.0003 * r.abs() {
                LogitDiagnostics::inspect(actual, reference)
                    .report_failure(position, "logits", true);
                return Err(Error::InvalidFormat("public Q3 full logits mismatch"));
            }
            max_error = max_error.max((a - r).abs());
        }
        println!(
            "public_q3_full_logits position={position} values={VOCAB} max_abs_error={max_error} greedy_matches=true"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn public_bytes(positions: usize) -> Vec<u8> {
        let mut bytes = vec![0; 8 + positions * RECORD_BYTES];
        bytes[..8].copy_from_slice(b"KQ3R0001");
        for position in 0..positions {
            let first = 8 + position * RECORD_BYTES;
            // Distinct maxima make every position/input chain independently
            // observable, including position31 and the predecessor of it.
            for (index, value) in [
                position as u32,
                position as u32,
                position as u32 + 1,
                VOCAB as u32,
            ]
            .into_iter()
            .enumerate()
            {
                bytes[first + index * 4..first + index * 4 + 4]
                    .copy_from_slice(&value.to_le_bytes());
            }
            let max = first + 16 + (position + 1) * 4;
            bytes[max..max + 4].copy_from_slice(&1f32.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn exact4_and32_lengths_bound_every_other_reference_size() {
        assert_eq!(positions_for_length(3_973_192).unwrap(), 4);
        assert_eq!(positions_for_length(31_785_480).unwrap(), 32);
        for positions in [0, 1, 3, 5, 8, 23, 31, 33, 64] {
            assert!(positions_for_length((8 + positions * RECORD_BYTES) as u64).is_err());
        }
        for positions in [4, 32] {
            let length = (8 + positions * RECORD_BYTES) as u64;
            for bytes in [length - 1, length + 1, u64::MAX] {
                assert!(positions_for_length(bytes).is_err());
            }
        }
    }

    #[test]
    fn every_public_position_loads_and_verifies_full_logits_and_matching_count() {
        for positions in [4, 32] {
            let reference = Reference::parse(&public_bytes(positions), positions).unwrap();
            assert_eq!(reference.len(), positions);
            reference.require_positions(positions).unwrap();
            assert!(
                reference
                    .require_positions(if positions == 4 { 32 } else { 4 })
                    .is_err()
            );
            assert!(reference.require_positions(8).is_err());
            for position in 0..positions {
                reference
                    .verify(position, &reference.logits[position])
                    .unwrap();
            }
            assert!(reference.verify(positions, &[]).is_err());
        }
    }

    #[test]
    fn malformed_header_record_fields_and_last_full_logits_are_rejected() {
        let positions = 32;
        let mut bytes = public_bytes(positions);
        bytes[7] = b'2';
        assert!(Reference::parse(&bytes, positions).is_err());
        bytes[7] = b'1';
        assert!(Reference::parse(&bytes, 4).is_err());
        assert!(Reference::parse(&bytes[..bytes.len() - 1], positions).is_err());
        for position in [0, 3, 4, 30, 31] {
            let first = 8 + position * RECORD_BYTES;
            for (field, value) in [
                (0, 999u32),
                (1, 999),
                (2, VOCAB as u32),
                (3, VOCAB as u32 - 1),
            ] {
                let offset = first + field * 4;
                let original = bytes[offset..offset + 4].to_vec();
                bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                assert!(Reference::parse(&bytes, positions).is_err());
                bytes[offset..offset + 4].copy_from_slice(&original);
            }
        }
        let first = 8 + 31 * RECORD_BYTES;
        let last = first + RECORD_BYTES - 4;
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 2.] {
            bytes[last..last + 4].copy_from_slice(&value.to_le_bytes());
            assert!(Reference::parse(&bytes, positions).is_err());
        }
    }

    #[test]
    fn actual_last_vector_requires_full_width_finite_values_greedy_and_all_lanes() {
        let reference = Reference::parse(&public_bytes(32), 32).unwrap();
        let mut values = reference.logits[31].to_vec();
        assert!(reference.verify(31, &values[..VOCAB - 1]).is_err());
        for (index, value) in [(VOCAB - 1, f32::NAN), (VOCAB - 1, 0.007), (33, 2.)] {
            let original = values[index];
            values[index] = value;
            assert!(reference.verify(31, &values).is_err());
            values[index] = original;
        }
        reference.verify(31, &values).unwrap();
    }

    #[test]
    fn explicit23_prefix_requires_the_complete32_reference_not_a_prefix_file() {
        let reference = Reference::parse(&public_bytes(LONG_POSITIONS), LONG_POSITIONS).unwrap();
        reference.require_positions(PREFIX_POSITIONS).unwrap();
        reference.require_positions(LONG_POSITIONS).unwrap();
        assert!(reference.require_positions(DEFAULT_POSITIONS).is_err());
        for position in 0..PREFIX_POSITIONS {
            reference
                .verify(position, &reference.logits[position])
                .unwrap();
        }
        let short = Reference::parse(&public_bytes(DEFAULT_POSITIONS), DEFAULT_POSITIONS).unwrap();
        assert!(short.require_positions(PREFIX_POSITIONS).is_err());
        short.require_positions(DEFAULT_POSITIONS).unwrap();
        assert!(short.require_positions(LONG_POSITIONS).is_err());
        assert!(Reference::parse(&public_bytes(PREFIX_POSITIONS), PREFIX_POSITIONS).is_err());
        for positions in [0, 1, 8, 22, 24, 31, 33] {
            assert!(reference.require_positions(positions).is_err());
        }
    }

    #[test]
    fn prefix_keeps_original_full_vector_gates_and_validates_the_unused_suffix() {
        let original = public_bytes(LONG_POSITIONS);
        let reference = Reference::parse(&original, LONG_POSITIONS).unwrap();
        reference.require_positions(PREFIX_POSITIONS).unwrap();
        // Last prefix vector and first omitted vector retain identical gates;
        // selecting a prefix does not mutate or truncate the reference owner.
        for position in [PREFIX_POSITIONS - 1, PREFIX_POSITIONS, LONG_POSITIONS - 1] {
            let mut actual = reference.logits[position].to_vec();
            actual[VOCAB - 1] = 0.007;
            assert!(matches!(
                reference.verify(position, &actual),
                Err(Error::InvalidFormat("public Q3 full logits mismatch"))
            ));
            actual[VOCAB - 1] = f32::NAN;
            assert!(reference.verify(position, &actual).is_err());
        }
        reference.require_positions(LONG_POSITIONS).unwrap();
        for position in [PREFIX_POSITIONS, LONG_POSITIONS - 1] {
            let mut malformed = original.clone();
            let final_lane = 8 + (position + 1) * RECORD_BYTES - 4;
            malformed[final_lane..final_lane + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            assert!(Reference::parse(&malformed, LONG_POSITIONS).is_err());
        }
        assert!(Reference::parse(&original[..original.len() - 1], LONG_POSITIONS).is_err());
    }

    #[test]
    fn diagnostics_find_first_failed_lane_but_measure_the_complete_vector() {
        let expected = [0., 2., -3., 0.];
        let actual = [0.005, 2.1, -3., 10.];
        let diagnostics = LogitDiagnostics::inspect(&actual, &expected);
        assert_eq!(
            diagnostics.first_bad,
            Some((1, 2f32.to_bits(), 2.1f32.to_bits()))
        );
        assert_eq!(diagnostics.max_abs_error, 10.);
        assert_eq!(diagnostics.max_normalized_error, 10. / f64::from(0.006f32));
        assert_eq!(diagnostics.nonfinite_class, 0);
    }

    #[test]
    fn diagnostics_keep_the_original_f32_boundary_and_reject_overflow() {
        let boundary = 0.006f32;
        let below = f32::from_bits(boundary.to_bits() - 1);
        let above = f32::from_bits(boundary.to_bits() + 1);
        for accepted in [below, boundary] {
            assert!(
                LogitDiagnostics::inspect(&[accepted], &[0.])
                    .first_bad
                    .is_none()
            );
        }
        assert_eq!(
            LogitDiagnostics::inspect(&[above], &[0.]).first_bad,
            Some((0, 0f32.to_bits(), above.to_bits()))
        );
        let overflow = LogitDiagnostics::inspect(&[f32::MAX], &[-f32::MAX]);
        assert_eq!(
            overflow.first_bad,
            Some((0, (-f32::MAX).to_bits(), f32::MAX.to_bits()))
        );
        assert!(overflow.max_abs_error.is_finite());
        assert!(overflow.max_normalized_error.is_finite());
        assert!(overflow.max_normalized_error > 1.);
        assert_eq!(overflow.nonfinite_class, 0);
    }

    #[test]
    fn diagnostics_classify_nonfinite_bits_without_losing_later_finite_error() {
        for (value, class) in [(f32::NAN, 1), (f32::INFINITY, 2), (f32::NEG_INFINITY, 3)] {
            let diagnostics = LogitDiagnostics::inspect(&[0., value, 10.], &[0., 0., 0.]);
            assert_eq!(
                diagnostics.first_bad,
                Some((1, 0f32.to_bits(), value.to_bits()))
            );
            assert_eq!(diagnostics.nonfinite_class, class);
            assert_eq!(diagnostics.max_abs_error, 10.);
            assert!(diagnostics.max_normalized_error.is_finite());
        }
    }

    #[test]
    fn diagnostics_do_not_relax_greedy_or_sampling_error_precedence() {
        let reference = Reference {
            logits: vec![Zeroizing::new(vec![0.; VOCAB])],
            tokens: vec![0],
        };
        let mut actual = vec![0.; VOCAB];
        actual[1] = 0.001;
        assert!(
            LogitDiagnostics::inspect(&actual, &reference.logits[0])
                .first_bad
                .is_none()
        );
        assert!(matches!(
            reference.verify(0, &actual),
            Err(Error::InvalidFormat("public Q3 greedy mismatch"))
        ));
        actual[VOCAB - 1] = f32::NAN;
        assert!(matches!(
            reference.verify(0, &actual),
            Err(Error::InvalidFormat("invalid sampling logits"))
        ));
        assert!(matches!(
            reference.verify(0, &actual[..VOCAB - 1]),
            Err(Error::InvalidFormat("public Q3 greedy mismatch"))
        ));
    }
}
