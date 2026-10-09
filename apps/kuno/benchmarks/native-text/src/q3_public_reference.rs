//! Four or32 fixed public original-Q3 positions from the independent CPU path.
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
        if !matches!(positions, DEFAULT_POSITIONS | LONG_POSITIONS) || self.len() != positions {
            return Err(Error::InvalidFormat("public Q3 selected position count"));
        }
        Ok(())
    }
    pub(crate) fn verify(&self, position: usize, actual: &[f32]) -> Result<()> {
        let reference = self.logits.get(position).ok_or(Error::OutOfBounds)?;
        if actual.len() != VOCAB || greedy(actual)? != self.tokens[position] {
            return Err(Error::InvalidFormat("public Q3 greedy mismatch"));
        }
        let mut max_error = 0f32;
        for (&a, &r) in actual.iter().zip(reference.iter()) {
            if !a.is_finite() || (a - r).abs() > 0.006 + 0.0003 * r.abs() {
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
        for positions in [0, 1, 3, 5, 8, 31, 33, 64] {
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
}
