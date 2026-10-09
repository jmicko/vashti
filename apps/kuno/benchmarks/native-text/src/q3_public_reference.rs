//! Four fixed public original-Q3 positions from the independent CPU path.
//! No request-derived input, vocabulary text or expert history is persisted.
use kuno_text::model::greedy;
use kuno_text::{Error, Result};
use std::io::Read;
use std::path::Path;
use zeroize::Zeroizing;

const VOCAB: usize = 248320;
const POSITIONS: usize = 4;
const BYTES: usize = 8 + POSITIONS * (16 + 4 * VOCAB);

pub(crate) struct Reference {
    logits: Vec<Zeroizing<Vec<f32>>>,
    tokens: [u32; POSITIONS],
}
impl Reference {
    pub(crate) fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        if file.metadata()?.len() != BYTES as u64 {
            return Err(Error::InvalidFormat("public Q3 reference length"));
        }
        let mut bytes = Zeroizing::new(vec![0; BYTES]);
        file.read_exact(&mut bytes)?;
        if &bytes[..8] != b"KQ3R0001" {
            return Err(Error::InvalidFormat("public Q3 reference version"));
        }
        let mut tokens = [0; POSITIONS];
        let mut logits = Vec::with_capacity(POSITIONS);
        let mut cursor = 8;
        for position in 0..POSITIONS {
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
