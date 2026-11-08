//! Bounded erased host reference for the alternate Q6/two-Q8 arithmetic.
//! Natural-order scalar Q6 bytes are checked against pinned independent C;
//! f64 integer products are independent of shader lane packing/reductions.
use kuno_text::{Error, Result};
use zeroize::{Zeroize, Zeroizing};

pub(crate) const RECORD_BYTES: usize = 210 + 256 * 4;
pub(crate) const FIXTURE: &[u8] = include_bytes!("q6_integer_fixtures/q6_k.bin");
const MAX_COLUMNS: usize = 16384;

struct Block {
    first: [i8; 32],
    second: [i8; 32],
    d: f32,
    rd: f32,
}
impl Zeroize for Block {
    fn zeroize(&mut self) {
        self.first.zeroize();
        self.second.zeroize();
        self.d.zeroize();
        self.rd.zeroize();
    }
}
pub(crate) struct Planes {
    blocks: Zeroizing<Vec<Block>>,
}
fn reject() -> Error {
    Error::InvalidFormat("fixed Q6 integer reference")
}
fn code(x: f32, d: f32) -> i8 {
    if d == 0. {
        0
    } else {
        (x / d).clamp(-127., 127.).round_ties_even() as i8
    }
}
impl Planes {
    pub(crate) fn new(input: &[f32]) -> Result<Self> {
        if input.is_empty()
            || input.len() > MAX_COLUMNS
            || !input.len().is_multiple_of(256)
            || input.iter().any(|x| !x.is_finite() || x.abs() == f32::MAX)
        {
            return Err(reject());
        }
        let mut blocks = Zeroizing::new(Vec::with_capacity(input.len() / 32));
        for values in input.chunks_exact(32) {
            let mut block = Zeroizing::new(Block {
                first: [0; 32],
                second: [0; 32],
                d: 0.,
                rd: 0.,
            });
            block.d = values.iter().map(|x| x.abs()).fold(0., f32::max) / 127.;
            let mut residual = Zeroizing::new([0.; 32]);
            for i in 0..32 {
                block.first[i] = code(values[i], block.d);
                let reconstructed = f32::from(block.first[i]) * block.d;
                if !reconstructed.is_finite() {
                    return Err(reject());
                }
                residual[i] = values[i] - reconstructed;
                if !residual[i].is_finite() {
                    return Err(reject());
                }
            }
            block.rd = residual.iter().map(|x| x.abs()).fold(0., f32::max) / 127.;
            for i in 0..32 {
                block.second[i] = code(residual[i], block.rd);
            }
            blocks.push(Block {
                first: block.first,
                second: block.second,
                d: block.d,
                rd: block.rd,
            });
        }
        Ok(Self { blocks })
    }
    pub(crate) fn columns(&self) -> usize {
        self.blocks.len() * 32
    }
    /// Exactly the shader's logical packed-word layout, without page padding.
    pub(crate) fn words(&self) -> Zeroizing<Vec<u32>> {
        let packed = self.columns() / 4;
        let mut words = Zeroizing::new(vec![0; packed * 2 + self.blocks.len() * 2]);
        for (i, block) in self.blocks.iter().enumerate() {
            for word in 0..8 {
                let a: [u8; 4] = std::array::from_fn(|j| block.first[word * 4 + j] as u8);
                let b: [u8; 4] = std::array::from_fn(|j| block.second[word * 4 + j] as u8);
                words[i * 8 + word] = u32::from_le_bytes(a);
                words[packed + i * 8 + word] = u32::from_le_bytes(b);
            }
            words[packed * 2 + i] = block.d.to_bits();
            words[packed * 2 + self.blocks.len() + i] = block.rd.to_bits();
        }
        words
    }
    #[cfg(test)]
    fn value(&self, column: usize) -> f64 {
        let block = &self.blocks[column / 32];
        let lane = column % 32;
        f64::from(block.d) * f64::from(block.first[lane])
            + f64::from(block.rd) * f64::from(block.second[lane])
    }
}

/// Original Q6 natural-order 16-value subscale, not the shader's packed words.
pub(crate) fn unpack16(wire: &[u8], columns: usize, group: usize) -> Result<([i8; 16], f32)> {
    if columns == 0
        || columns > MAX_COLUMNS
        || !columns.is_multiple_of(256)
        || wire.len() != columns / 256 * 210
        || group >= columns / 16
    {
        return Err(reject());
    }
    let base = group / 16 * 210;
    let sub = group % 16;
    let half = sub / 8;
    let quarter = sub % 8 / 2;
    let side = sub % 2;
    let mut codes = [0; 16];
    for (lane, out) in codes.iter_mut().enumerate() {
        let low = wire[base + half * 64 + quarter % 2 * 32 + side * 16 + lane];
        let high = wire[base + 128 + half * 32 + side * 16 + lane];
        let nibble = if quarter < 2 { low & 15 } else { low >> 4 };
        *out = (nibble | ((high >> (quarter * 2) & 3) << 4)) as i8 - 32;
    }
    let d = kuno_text::quant::f16_to_f32(u16::from_le_bytes([wire[base + 208], wire[base + 209]]));
    let factor = d * f32::from(wire[base + 192 + sub] as i8);
    if !factor.is_finite() {
        return Err(reject());
    }
    Ok((codes, factor))
}
pub(crate) fn project(wire: &[u8], planes: &Planes) -> Result<f32> {
    let mut sum = 0f64;
    for group in 0..planes.columns() / 16 {
        let (weights, factor) = unpack16(wire, planes.columns(), group)?;
        let block = &planes.blocks[group / 2];
        let start = group % 2 * 16;
        let mut a = 0i64;
        let mut b = 0i64;
        for (lane, &weight) in weights.iter().enumerate() {
            a += i64::from(weight) * i64::from(block.first[start + lane]);
            b += i64::from(weight) * i64::from(block.second[start + lane]);
        }
        sum += f64::from(factor) * (f64::from(block.d) * a as f64 + f64::from(block.rd) * b as f64);
    }
    let value = sum as f32;
    if !value.is_finite() {
        return Err(reject());
    }
    Ok(value)
}
/// Alternative scalar decoded-value dot, used only for arithmetic validation.
#[cfg(test)]
pub(crate) fn scalar_project(wire: &[u8], planes: &Planes) -> Result<f32> {
    let mut sum = 0f64;
    for group in 0..planes.columns() / 16 {
        let (weights, factor) = unpack16(wire, planes.columns(), group)?;
        for (lane, weight) in weights.into_iter().enumerate() {
            sum += f64::from(factor) * f64::from(weight) * planes.value(group * 16 + lane);
        }
    }
    let value = sum as f32;
    if !value.is_finite() {
        return Err(reject());
    }
    Ok(value)
}
pub(crate) fn public_input(columns: usize, variant: usize) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..columns)
            .map(|i| ((i * 17 + variant * 11) % 257) as f32 / 32. - 4.)
            .collect(),
    )
}
/// Public synthetic rows drawn only from the first sixteen ordinary C blocks.
pub(crate) fn public_rows(rows: usize, columns: usize) -> Result<Vec<u8>> {
    if rows == 0
        || rows > 16384
        || columns == 0
        || columns > MAX_COLUMNS
        || !columns.is_multiple_of(256)
        || rows * (columns / 256 * 210) > 64 * 1024 * 1024
    {
        return Err(reject());
    }
    let mut out = Vec::with_capacity(rows * columns / 256 * 210);
    for row in 0..rows {
        for block in 0..columns / 256 {
            let start = ((row * 3 + block * 5) % 16) * RECORD_BYTES;
            out.extend_from_slice(&FIXTURE[start..start + 210]);
        }
    }
    Ok(out)
}
/// Validate the complete finite reference inventory before public GPU work.
pub(crate) fn validate_fixture() -> Result<usize> {
    if FIXTURE.len() != 28 * RECORD_BYTES {
        return Err(reject());
    }
    let mut count = 0;
    for record in FIXTURE.chunks_exact(RECORD_BYTES).take(25) {
        for group in 0..16 {
            let (codes, factor) = unpack16(&record[..210], 256, group)?;
            for (lane, code) in codes.into_iter().enumerate() {
                let at = 210 + (group * 16 + lane) * 4;
                let c = u32::from_le_bytes(record[at..at + 4].try_into().unwrap());
                if (factor * f32::from(code)).to_bits() != c {
                    return Err(reject());
                }
                count += 1;
            }
        }
    }
    for record in FIXTURE.chunks_exact(RECORD_BYTES).skip(25) {
        if unpack16(&record[..210], 256, 0).is_ok() {
            return Err(reject());
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_original_q6_code_scale_quarter_and_half_matches_independent_c() {
        assert_eq!(validate_fixture().unwrap(), 6400);
        assert!(unpack16(&FIXTURE[..209], 256, 0).is_err());
        assert!(unpack16(&FIXTURE[..210], 256, 16).is_err());
    }
    #[test]
    fn ties_zero_underflow_and_nonfinite_preflight_are_explicit() {
        for (x, expected) in [
            (0.5, 0),
            (1.5, 2),
            (2.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (-2.5, -2),
        ] {
            assert_eq!(code(x, 1.), expected);
        }
        let zeros = Planes::new(&[0.; 256]).unwrap();
        assert!(zeros.words().iter().all(|x| *x == 0));
        let tiny = Planes::new(&[f32::from_bits(1); 256]).unwrap();
        assert!(tiny.words().iter().all(|x| *x == 0));
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            -f32::MAX,
        ] {
            let mut input = [0.; 256];
            input[255] = value;
            assert!(Planes::new(&input).is_err());
        }
        assert!(Planes::new(&[0.; 255]).is_err());
        assert!(Planes::new(&[]).is_err());
    }
    #[test]
    fn independent_integer_and_scalar_decoded_products_agree() {
        for columns in [256, 512, 2560, 16384] {
            for variant in 0..3 {
                let input = public_input(columns, variant);
                let planes = Planes::new(&input).unwrap();
                assert_eq!(planes.words().len(), columns / 2 + columns / 16);
                let rows = public_rows(7, columns).unwrap();
                for wire in rows.chunks_exact(columns / 256 * 210) {
                    let a = project(wire, &planes).unwrap();
                    let b = scalar_project(wire, &planes).unwrap();
                    assert!((a - b).abs() <= 2e-6 * (1. + b.abs()));
                }
            }
        }
    }
}
