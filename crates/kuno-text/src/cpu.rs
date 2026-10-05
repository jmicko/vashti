//! Bounded scalar matrix execution for the first Rust graph.
//!
//! This deliberately keeps the artifact's original packing. It is a numerical
//! reference, not an optimized expert cache or a full-model allocation.
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{Error, Result, TensorView};

const MAX_INPUT_FEATURES: u64 = 65_536;
const MAX_OUTPUT_FEATURES: u64 = 262_144;
const MAX_EXPERTS: u64 = 512;
const TILE_READ_BYTES: usize = 512 * 1024;
const TILE_ROWS: usize = 32;

/// Matrix rows are GGUF dimension one; expert dimension two is outermost.
#[derive(Clone, Copy)]
pub struct Matrix<'a> {
    tensor: TensorView<'a>,
    inputs: usize,
    outputs: usize,
    experts: usize,
}

impl<'a> Matrix<'a> {
    pub fn new(tensor: TensorView<'a>) -> Result<Self> {
        let shape = tensor.info().dimensions();
        if !(2..=3).contains(&shape.len()) {
            return Err(Error::InvalidFormat("matrix rank"));
        }
        if shape[0] > MAX_INPUT_FEATURES || shape[1] > MAX_OUTPUT_FEATURES {
            return Err(Error::LimitExceeded("matrix dimensions"));
        }
        let experts = shape.get(2).copied().unwrap_or(1);
        if experts > MAX_EXPERTS {
            return Err(Error::LimitExceeded("matrix experts"));
        }
        let inputs = usize::try_from(shape[0]).map_err(|_| Error::ArithmeticOverflow)?;
        let outputs = usize::try_from(shape[1]).map_err(|_| Error::ArithmeticOverflow)?;
        let experts = usize::try_from(experts).map_err(|_| Error::ArithmeticOverflow)?;
        Ok(Self {
            tensor,
            inputs,
            outputs,
            experts,
        })
    }

    pub fn inputs(self) -> usize {
        self.inputs
    }
    pub fn outputs(self) -> usize {
        self.outputs
    }
    pub fn experts(self) -> usize {
        self.experts
    }

    /// Run one dense matrix or one expert. Cancellation is observed at tile
    /// boundaries. Caller owns request input/output and its privacy lifecycle.
    pub fn apply(
        self,
        expert: usize,
        input: &[f32],
        output: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if input.len() != self.inputs || output.len() != self.outputs || expert >= self.experts {
            return Err(Error::InvalidBufferLength);
        }
        if !input.iter().all(|x| x.is_finite()) {
            return Err(Error::InvalidFormat("non-finite matrix input"));
        }
        let row_bytes = usize::try_from(self.tensor.info().row_bytes())
            .map_err(|_| Error::ArithmeticOverflow)?;
        if row_bytes > TILE_READ_BYTES {
            return Err(Error::LimitExceeded("matrix row scratch"));
        }
        let rows_per_tile = TILE_ROWS.min(TILE_READ_BYTES / row_bytes);
        let mut packed = vec![0u8; rows_per_tile * row_bytes];
        let mut decoded = vec![0f32; self.inputs];
        let expert_start = expert
            .checked_mul(self.outputs)
            .ok_or(Error::ArithmeticOverflow)?;
        for first in (0..self.outputs).step_by(rows_per_tile) {
            if cancel.load(Ordering::Relaxed) {
                output.fill(0.0);
                return Err(Error::InvalidFormat("matrix execution cancelled"));
            }
            let rows = rows_per_tile.min(self.outputs - first);
            let start = expert_start
                .checked_add(first)
                .and_then(|row| row.checked_mul(row_bytes))
                .ok_or(Error::ArithmeticOverflow)?;
            self.tensor.read_range(
                u64::try_from(start).map_err(|_| Error::ArithmeticOverflow)?,
                &mut packed[..rows * row_bytes],
            )?;
            for row in 0..rows {
                self.tensor.info().dtype().decode(
                    &packed[row * row_bytes..(row + 1) * row_bytes],
                    &mut decoded,
                )?;
                if !decoded.iter().all(|x| x.is_finite()) {
                    output.fill(0.0);
                    return Err(Error::InvalidFormat("non-finite matrix weights"));
                }
                let sum = decoded.iter().zip(input).map(|(w, x)| w * x).sum::<f32>();
                if !sum.is_finite() {
                    output.fill(0.0);
                    return Err(Error::InvalidFormat("non-finite matrix output"));
                }
                output[first + row] = sum;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GgufFile, GgufLimits};
    use std::io::Write;

    fn fixture(experts: usize) -> (std::path::PathBuf, GgufFile) {
        let path = std::env::temp_dir().join(format!(
            "kuno-matrix-fixture-{}-{experts}.gguf",
            std::process::id()
        ));
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(b"w");
        bytes.extend((if experts == 1 { 2u32 } else { 3u32 }).to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        bytes.extend(2u64.to_le_bytes());
        if experts != 1 {
            bytes.extend((experts as u64).to_le_bytes());
        }
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        for e in 0..experts {
            for value in [1f32, 2., 3., 4., 5., 6.] {
                bytes.extend((value + e as f32 * 10.).to_le_bytes());
            }
        }
        std::fs::File::create(&path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
        let file = GgufFile::open(&path, GgufLimits::default()).unwrap();
        (path, file)
    }

    #[test]
    fn nonsquare_matrix_and_expert_orientation() {
        let (path, file) = fixture(2);
        let matrix = Matrix::new(file.tensor("w").unwrap()).unwrap();
        let mut output = [0f32; 2];
        let cancel = AtomicBool::new(false);
        matrix
            .apply(0, &[2., 3., 4.], &mut output, &cancel)
            .unwrap();
        assert_eq!(output, [20., 47.]);
        matrix
            .apply(1, &[2., 3., 4.], &mut output, &cancel)
            .unwrap();
        assert_eq!(output, [110., 137.]);
        assert!(
            matrix
                .apply(2, &[2., 3., 4.], &mut output, &cancel)
                .is_err()
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cancelled_matrix_clears_output_without_reading() {
        let (path, file) = fixture(1);
        let matrix = Matrix::new(file.tensor("w").unwrap()).unwrap();
        let mut output = [99f32; 2];
        assert!(
            matrix
                .apply(0, &[2., 3., 4.], &mut output, &AtomicBool::new(true))
                .is_err()
        );
        assert_eq!(output, [0., 0.]);
        std::fs::remove_file(path).unwrap();
    }
}
