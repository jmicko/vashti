//! GGML-compatible little-endian packing, decoded entirely in safe Rust.
//!
//! Layout and scalar equations were checked against ggml revision
//! `4bf5f6000653b7881d00963cd6ddb665ccd62a8d`, `ggml-common.h` and
//! `ggml-quants.c`. IQ2_S/IQ3_S/Q5_K/Q2_0 and the nonlinear IQ formats use
//! the pinned llama.cpp `b11232` sources recorded in `quant_iq.rs`,
//! `quant_daslab.rs` and their independent public fixture receipts.
//! See ../NOTICE for the MIT provenance notice.

use crate::{Error, Result};

#[path = "quant_iq.rs"]
mod iq;

#[path = "quant_daslab.rs"]
mod daslab;

/// Original GGML magnitude grids used by IQ2_S and IQ3_S device decoders.
pub use daslab::{IQ2_S_GRID, IQ3_S_GRID};

/// Immutable GGML IQ3_XXS magnitudes, packed as four little-endian bytes.
/// Native device compilers may embed this public table as a shader constant.
pub use iq::IQ3_XXS_GRID;
/// Immutable signed nonlinear values shared by GGML IQ4_NL and IQ4_XS.
pub use iq::IQ4_NL_VALUES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorType {
    F32,
    F16,
    Q5_0,
    Q8_0,
    Q4K,
    Q6K,
    Iq3Xxs,
    Iq4Nl,
    Iq4Xs,
    Bf16,
    Q5K,
    Q2_0,
    Iq2S,
    Iq3S,
    Q2K,
    Q3K,
}

impl TensorType {
    pub fn from_gguf_id(id: u32) -> Result<Self> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            6 => Ok(Self::Q5_0),
            8 => Ok(Self::Q8_0),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            18 => Ok(Self::Iq3Xxs),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            30 => Ok(Self::Bf16),
            42 => Ok(Self::Q2_0),
            _ => Err(Error::UnsupportedTensorType(id)),
        }
    }

    pub const fn gguf_id(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q5_0 => 6,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Q2_0 => 42,
            Self::Q6K => 14,
            Self::Iq3Xxs => 18,
            Self::Iq4Nl => 20,
            Self::Iq4Xs => 23,
            Self::Bf16 => 30,
        }
    }

    pub const fn block_elements(self) -> u64 {
        match self {
            Self::F32 | Self::F16 | Self::Bf16 => 1,
            Self::Q5_0 | Self::Q8_0 | Self::Iq4Nl => 32,
            Self::Q2_0 => 64,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Iq3Xxs
            | Self::Iq4Xs
            | Self::Iq2S
            | Self::Iq3S => 256,
        }
    }

    pub const fn block_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::Q5_0 => 22,
            Self::Q8_0 => 34,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q2_0 => 18,
            Self::Iq2S => 82,
            Self::Iq3S => 110,
            Self::Q6K => 210,
            Self::Iq3Xxs => 98,
            Self::Iq4Nl => 18,
            Self::Iq4Xs => 136,
        }
    }

    /// Size of a contiguous row. Quantization blocks cannot cross row boundaries.
    pub fn row_bytes(self, elements: u64) -> Result<u64> {
        if elements == 0 || !elements.is_multiple_of(self.block_elements()) {
            return Err(Error::InvalidFormat(
                "row does not contain whole quantization blocks",
            ));
        }
        (elements / self.block_elements())
            .checked_mul(self.block_bytes())
            .ok_or(Error::ArithmeticOverflow)
    }

    /// Decode exact packed blocks into a caller-owned output buffer.
    /// Lengths are validated before writing anything. No allocation is performed.
    pub fn decode(self, packed: &[u8], output: &mut [f32]) -> Result<()> {
        let count = u64::try_from(output.len()).map_err(|_| Error::ArithmeticOverflow)?;
        let expected = self.row_bytes(count)?;
        if u64::try_from(packed.len()).map_err(|_| Error::ArithmeticOverflow)? != expected {
            return Err(Error::InvalidBufferLength);
        }
        let elements = self.block_elements() as usize;
        let bytes = self.block_bytes() as usize;
        for (block, out) in packed
            .chunks_exact(bytes)
            .zip(output.chunks_exact_mut(elements))
        {
            match self {
                Self::F32 => out[0] = f32::from_le_bytes(block.try_into().unwrap()),
                Self::F16 => out[0] = f16_to_f32(u16::from_le_bytes(block.try_into().unwrap())),
                Self::Bf16 => {
                    out[0] = f32::from_bits(
                        u32::from(u16::from_le_bytes(block.try_into().unwrap())) << 16,
                    )
                }
                Self::Q5_0 => decode_q5_0(block, out),
                Self::Q8_0 => {
                    let scale = half_at(block, 0);
                    for (target, &value) in out.iter_mut().zip(&block[2..]) {
                        *target = scale * f32::from(value as i8);
                    }
                }
                Self::Q2K => decode_q2k(block, out),
                Self::Q3K => decode_q3k(block, out),
                Self::Q4K => decode_q4k(block, out),
                Self::Q5K => daslab::decode_q5_k(block, out),
                Self::Q2_0 => daslab::decode_q2_0(block, out),
                Self::Iq2S => daslab::decode_iq2_s(block, out),
                Self::Iq3S => daslab::decode_iq3_s(block, out),
                Self::Q6K => decode_q6k(block, out),
                Self::Iq3Xxs => iq::decode_iq3_xxs(block, out),
                Self::Iq4Nl => iq::decode_iq4_nl(block, out),
                Self::Iq4Xs => iq::decode_iq4_xs(block, out),
            }
        }
        Ok(())
    }
}

/// Exact IEEE binary16 conversion, including subnormals, signed zero and NaNs.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 31);
    let mut mantissa = u32::from(bits & 1023);
    let value = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let mut exponent = 113_u32;
            while mantissa & 1024 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (exponent << 23) | ((mantissa & 1023) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(value)
}

fn half_at(bytes: &[u8], offset: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn decode_q5_0(block: &[u8], out: &mut [f32]) {
    let scale = half_at(block, 0);
    let high = u32::from_le_bytes(block[2..6].try_into().unwrap());
    for index in 0..16 {
        let low = block[6 + index];
        let first = i32::from((low & 15) | (((high >> index) as u8 & 1) << 4)) - 16;
        let second = i32::from((low >> 4) | (((high >> (index + 16)) as u8 & 1) << 4)) - 16;
        out[index] = scale * first as f32;
        out[index + 16] = scale * second as f32;
    }
}

fn scale_min_q4k(scales: &[u8], index: usize) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        (
            (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

/// GGML `block_q2_K`: 16 scale/min bytes (4 bits each), 64 bytes of 2-bit values, then the
/// f16 super-block scale and minimum (`dequantize_row_q2_K`).
fn decode_q2k(block: &[u8], out: &mut [f32]) {
    let (scales, packed) = (&block[..16], &block[16..80]);
    let (d, dmin) = (half_at(block, 80), half_at(block, 82));
    let mut is = 0;
    for half in 0..2 {
        let q = &packed[half * 32..][..32];
        for j in 0..4 {
            let shift = 2 * j;
            for part in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * f32::from(sc & 15), dmin * f32::from(sc >> 4));
                for l in 0..16 {
                    let value = (q[part * 16 + l] >> shift) & 3;
                    out[half * 128 + j * 32 + part * 16 + l] = dl * f32::from(value) - ml;
                }
            }
        }
    }
}

/// GGML `block_q3_K`: a 32-byte high-bit mask, 64 bytes of 2-bit values, 12 bytes of 6-bit
/// scales, then the f16 super-block scale (`dequantize_row_q3_K`).
fn decode_q3k(block: &[u8], out: &mut [f32]) {
    let (hmask, packed, raw) = (&block[..32], &block[32..96], &block[96..108]);
    let d = half_at(block, 108);
    let word = |i: usize| u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
    let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let (a0, a1, tmp) = (word(0), word(1), word(2));
    let aux = [
        (a0 & k2) | ((tmp & k1) << 4),
        (a1 & k2) | (((tmp >> 2) & k1) << 4),
        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4),
        ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4),
    ];
    let scales: Vec<i8> = aux.iter().flat_map(|w| w.to_le_bytes()).map(|b| b as i8).collect();
    let mut is = 0;
    let mut m = 1u8;
    for half in 0..2 {
        let q = &packed[half * 32..][..32];
        for j in 0..4 {
            let shift = 2 * j;
            for part in 0..2 {
                let dl = d * f32::from(scales[is] - 32);
                is += 1;
                for l in 0..16 {
                    let low = i32::from((q[part * 16 + l] >> shift) & 3);
                    let high = if hmask[part * 16 + l] & m != 0 { 0 } else { 4 };
                    out[half * 128 + j * 32 + part * 16 + l] = dl * (low - high) as f32;
                }
            }
            m <<= 1;
        }
    }
}

fn decode_q4k(block: &[u8], out: &mut [f32]) {
    let scale = half_at(block, 0);
    let minimum = half_at(block, 2);
    let scales = &block[4..16];
    let packed = &block[16..];
    for group in 0..4 {
        let (scale1, min1) = scale_min_q4k(scales, group * 2);
        let (scale2, min2) = scale_min_q4k(scales, group * 2 + 1);
        let factor1 = scale * f32::from(scale1);
        let factor2 = scale * f32::from(scale2);
        let bias1 = minimum * f32::from(min1);
        let bias2 = minimum * f32::from(min2);
        for index in 0..32 {
            let q = packed[group * 32 + index];
            out[group * 64 + index] = factor1 * f32::from(q & 15) - bias1;
            out[group * 64 + index + 32] = factor2 * f32::from(q >> 4) - bias2;
        }
    }
}

fn decode_q6k(block: &[u8], out: &mut [f32]) {
    let low = &block[..128];
    let high = &block[128..192];
    let scales = &block[192..208];
    let scale = half_at(block, 208);
    for half in 0..2 {
        for index in 0..32 {
            let low1 = low[half * 64 + index];
            let low2 = low[half * 64 + index + 32];
            let high = high[half * 32 + index];
            let quants = [
                (low1 & 15) | ((high & 3) << 4),
                (low2 & 15) | (((high >> 2) & 3) << 4),
                (low1 >> 4) | (((high >> 4) & 3) << 4),
                (low2 >> 4) | ((high >> 6) << 4),
            ];
            for (quarter, quant) in quants.into_iter().enumerate() {
                let subscale = scales[half * 8 + quarter * 2 + index / 16] as i8;
                out[half * 128 + quarter * 32 + index] =
                    scale * f32::from(subscale) * f32::from(quant as i16 - 32);
            }
        }
    }
}
