//! Original DASLab mixed quantization layouts, without activation quantization.
//! Equations/codebooks are adapted from llama.cpp b11232 GGML (MIT); see
//! crates/kuno-text/NOTICE and its independent C fixture provenance.
use super::{binary16, byte_at, u16_at};
use cubecl::prelude::*;

#[cube]
fn iq2_grid_word(index: usize, half: usize) -> u32 {
    // WGSL has no u64 arithmetic. Keep the original eight-byte grid as two
    // little-endian words; no runtime table upload or expanded weight buffer.
    let grid = Array::<u32>::from_data(comptime!(
        kuno_text::quant::IQ2_S_GRID
            .iter()
            .flat_map(|word| [*word as u32, (*word >> 32) as u32])
            .collect::<Vec<_>>()
    ));
    grid[index * 2 + half]
}

#[cube]
fn iq3_grid_word(index: usize) -> u32 {
    let grid = Array::<u32>::from_data(comptime!(kuno_text::quant::IQ3_S_GRID.to_vec()));
    grid[index]
}

#[cube]
pub(super) fn iq2_s(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 256) * 82;
    let index = column % 256;
    let group = index / 32;
    let octet = (index % 32) / 8;
    let lane = index % 8;
    let low = byte_at(words, base + 2 + group * 4 + octet);
    let high = (byte_at(words, base + 66 + group) >> ((2 * octet) as u32)) & 3;
    let grid = iq2_grid_word((low | (high << 8)) as usize, lane / 4);
    let quant = (grid >> (((lane % 4) * 8) as u32)) & 255;
    let scales = byte_at(words, base + 74 + group);
    let mut scale = scales & 15;
    if octet >= 2 {
        scale = scales >> 4;
    }
    // Preserve the scalar C's two factor multiplications and final sign.
    let factor = binary16(u16_at(words, base)) * (0.5 + scale as f32) * 0.25;
    let mut value = factor * quant as f32;
    let signs = byte_at(words, base + 34 + group * 4 + octet);
    if ((signs >> (lane as u32)) & 1) != 0 {
        value *= -1.0;
    }
    value
}

#[cube]
pub(super) fn iq3_s(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 256) * 110;
    let index = column % 256;
    let group = index / 32;
    let octet = (index % 32) / 8;
    let lane = index % 8;
    let quartet = octet * 2 + lane / 4;
    let low = byte_at(words, base + 2 + group * 8 + quartet);
    let high = (byte_at(words, base + 66 + group) >> (quartet as u32)) & 1;
    let grid = iq3_grid_word((low | (high << 8)) as usize);
    let quant = (grid >> (((lane % 4) * 8) as u32)) & 255;
    let scales = byte_at(words, base + 106 + group / 2);
    let mut scale = scales & 15;
    if (group & 1) != 0 {
        scale = scales >> 4;
    }
    let factor = binary16(u16_at(words, base)) * (1 + 2 * scale) as f32;
    let mut value = factor * quant as f32;
    let signs = byte_at(words, base + 74 + group * 4 + octet);
    if ((signs >> (lane as u32)) & 1) != 0 {
        value *= -1.0;
    }
    value
}

// Staged subgroup factors retain the original scalar multiply order. The
// original decoders above remain unchanged for every unselected entrypoint.
#[cube]
pub(super) fn iq2_s_factor(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 256) * 82;
    let index = column % 256;
    let group = index / 32;
    let scales = byte_at(words, base + 74 + group);
    let mut scale = scales & 15;
    if index % 32 >= 16 {
        scale = scales >> 4;
    }
    binary16(u16_at(words, base)) * (0.5 + scale as f32) * 0.25
}

#[cube]
pub(super) fn iq2_s_value_with_factor(
    words: &Array<u32>,
    row: usize,
    column: usize,
    factor: f32,
) -> f32 {
    let base = row + (column / 256) * 82;
    let index = column % 256;
    let group = index / 32;
    let octet = (index % 32) / 8;
    let lane = index % 8;
    let low = byte_at(words, base + 2 + group * 4 + octet);
    let high = (byte_at(words, base + 66 + group) >> ((2 * octet) as u32)) & 3;
    let grid = iq2_grid_word((low | (high << 8)) as usize, lane / 4);
    let quant = (grid >> (((lane % 4) * 8) as u32)) & 255;
    let mut value = factor * quant as f32;
    let signs = byte_at(words, base + 34 + group * 4 + octet);
    if ((signs >> (lane as u32)) & 1) != 0 {
        value *= -1.0;
    }
    value
}

#[cube]
pub(super) fn iq3_s_factor(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 256) * 110;
    let group = (column % 256) / 32;
    let scales = byte_at(words, base + 106 + group / 2);
    let mut scale = scales & 15;
    if (group & 1) != 0 {
        scale = scales >> 4;
    }
    binary16(u16_at(words, base)) * (1 + 2 * scale) as f32
}

#[cube]
pub(super) fn iq3_s_value_with_factor(
    words: &Array<u32>,
    row: usize,
    column: usize,
    factor: f32,
) -> f32 {
    let base = row + (column / 256) * 110;
    let index = column % 256;
    let group = index / 32;
    let octet = (index % 32) / 8;
    let lane = index % 8;
    let quartet = octet * 2 + lane / 4;
    let low = byte_at(words, base + 2 + group * 8 + quartet);
    let high = (byte_at(words, base + 66 + group) >> (quartet as u32)) & 1;
    let grid = iq3_grid_word((low | (high << 8)) as usize);
    let quant = (grid >> (((lane % 4) * 8) as u32)) & 255;
    let mut value = factor * quant as f32;
    let signs = byte_at(words, base + 74 + group * 4 + octet);
    if ((signs >> (lane as u32)) & 1) != 0 {
        value *= -1.0;
    }
    value
}

#[cube]
pub(super) fn q2_0(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 64) * 18;
    let index = column % 64;
    let packed = byte_at(words, base + 2 + index / 4);
    let code = ((packed >> (((index % 4) * 2) as u32)) & 3) as i32 - 1;
    binary16(u16_at(words, base)) * code as f32
}

#[cube]
pub(super) fn q5_k(words: &Array<u32>, row: usize, column: usize) -> f32 {
    let base = row + (column / 256) * 176;
    let index = column % 256;
    let group = index / 32;
    let mut scale = byte_at(words, base + 4 + group) & 63;
    let mut minimum = byte_at(words, base + 8 + group) & 63;
    if group >= 4 {
        scale =
            (byte_at(words, base + 8 + group) & 15) | ((byte_at(words, base + group) >> 6) << 4);
        minimum = (byte_at(words, base + 8 + group) >> 4)
            | ((byte_at(words, base + 4 + group) >> 6) << 4);
    }
    let packed = byte_at(words, base + 48 + (index / 64) * 32 + index % 32);
    let mut code = packed & 15;
    if index % 64 >= 32 {
        code = packed >> 4;
    }
    let high = (byte_at(words, base + 16 + index % 32) >> (group as u32)) & 1;
    code |= high << 4;
    let factor = binary16(u16_at(words, base)) * scale as f32;
    let bias = binary16(u16_at(words, base + 2)) * minimum as f32;
    factor * code as f32 - bias
}
