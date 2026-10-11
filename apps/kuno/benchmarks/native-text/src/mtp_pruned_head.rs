//! Explicit public English Q8 draft-head arithmetic, not a model/controller.
//! The full original head remains resident; only selected original rows execute.
//! All logits/candidate/token private owners remain full sized. The parent owns
//! the immutable public map source, its full page/stream/alias proof and erasure.
//! Omitting draft rows changes proposal coverage, never actual target verification.
#![allow(dead_code)] // Parent selection/ownership is a separate integration gate.

use super::super::{Q8_0, binary16, byte_at, signed_byte, u16_at};
use super::{PHYSICAL_LANES, ROWS_PER_CUBE, dot, lane, local_row, reduce_original64};
use crate::mtp_resident_kernels::{D, GREEDY_CHUNK, GREEDY_WORDS, ROW_BYTES, VOCAB};
use cubecl::prelude::*;
use kuno_text::{Error, Result};

pub const ENGLISH_ROWS: usize = 40_525;
pub const ENGLISH_BYTES: usize = ENGLISH_ROWS * 4;
pub const ENGLISH_OWNED_BYTES: usize = 196_608;
pub const ENGLISH_SHA256: &str = "369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25";
pub const STRATA_REVISION: &str = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf";
pub const TILE_ROWS: usize = 16_384;
pub const SEGMENTS: usize = VOCAB.div_ceil(TILE_ROWS);
pub const ENGLISH_GREEDY_GROUPS: usize = ENGLISH_ROWS.div_ceil(GREEDY_CHUNK);
const SEGMENT_ROWS: [usize; SEGMENTS] = [
    16_384, 16_384, 2_777, 1_958, 1_503, 996, 0, 0, 0, 31, 44, 33, 26, 32, 34, 323,
];
pub const ENGLISH_MAP: &[u8; ENGLISH_BYTES] =
    include_bytes!("../fixtures/strata-english-vocab/draft_vocab_en.bin");

fn reject() -> Error {
    Error::InvalidFormat("MTP pinned public English vocabulary")
}

/// Zero-allocation public proof. Never accepts a request-selected vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EnglishVocabulary(());
impl EnglishVocabulary {
    pub(crate) fn embedded() -> Result<Self> {
        Self::validate_exact(ENGLISH_MAP)
    }
    pub(crate) fn validate_exact(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != ENGLISH_BYTES {
            return Err(reject());
        }
        let mut previous = None;
        for encoded in bytes.chunks_exact(4) {
            let id = u32::from_le_bytes(encoded.try_into().map_err(|_| reject())?);
            if id as usize >= VOCAB || previous.is_some_and(|old| old >= id) {
                return Err(reject());
            }
            previous = Some(id);
        }
        // Byte equality to the reviewed compiled public artifact, not merely
        // shape/sortedness. The protected runner pins the complete executable.
        if bytes != ENGLISH_MAP {
            return Err(reject());
        }
        Ok(Self(()))
    }
    pub(crate) fn bytes(self) -> &'static [u8] {
        ENGLISH_MAP
    }
    pub(crate) fn id(self, index: usize) -> Result<u32> {
        if index >= ENGLISH_ROWS {
            return Err(reject());
        }
        let first = index * 4;
        Ok(u32::from_le_bytes(
            ENGLISH_MAP[first..first + 4]
                .try_into()
                .map_err(|_| reject())?,
        ))
    }
    pub(crate) fn segment(self, index: usize) -> Result<HeadSegment> {
        if index >= SEGMENTS {
            return Err(reject());
        }
        let source_first = index * TILE_ROWS;
        let source_rows = (VOCAB - source_first).min(TILE_ROWS);
        // Public fixed intervals are proved against every artifact ID by the
        // source tests. Do not rescan the entire map for each draft launch.
        let subset_first = SEGMENT_ROWS[..index].iter().sum();
        Ok(HeadSegment {
            source_first,
            source_rows,
            subset_first,
            subset_rows: SEGMENT_ROWS[index],
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HeadSegment {
    pub(crate) source_first: usize,
    pub(crate) source_rows: usize,
    pub(crate) subset_first: usize,
    pub(crate) subset_rows: usize,
}
impl HeadSegment {
    pub(crate) fn validate(self) -> Result<()> {
        if !self.source_first.is_multiple_of(TILE_ROWS)
            || EnglishVocabulary(()).segment(self.source_first / TILE_ROWS)? != self
        {
            return Err(reject());
        }
        Ok(())
    }
    /// Include every physical output cell in the final original finite sweep.
    pub(crate) fn cube_count(self, check_tail: bool, output_words: usize) -> Result<CubeCount> {
        self.validate()?;
        if self.subset_rows == 0 || output_words < VOCAB || output_words > u32::MAX as usize {
            return Err(reject());
        }
        let rows = self.subset_rows.div_ceil(ROWS_PER_CUBE);
        let cubes = if check_tail {
            rows.max(output_words.div_ceil(ROWS_PER_CUBE * PHYSICAL_LANES))
        } else {
            rows
        };
        Ok(CubeCount::Static(cubes as u32, 1, 1))
    }
}

#[cube]
#[allow(clippy::collapsible_if)]
fn original_row(
    ids: &Array<u32>,
    index: usize,
    source_first: usize,
    #[comptime] source_rows: usize,
) -> usize {
    let mut result = 248_320_usize;
    if ENGLISH_ROWS <= ids.len() && index < ENGLISH_ROWS && source_first < VOCAB {
        if source_rows <= VOCAB - source_first {
            let id = ids[index] as usize;
            let mut valid = id < VOCAB && id >= source_first;
            if index > 0 {
                valid = valid && ids[index - 1] < id as u32;
            }
            if index + 1 < ENGLISH_ROWS {
                valid = valid && (id as u32) < ids[index + 1];
            }
            if valid {
                if id - source_first < source_rows {
                    result = id - source_first;
                }
            }
        }
    }
    result
}

#[cube]
#[allow(clippy::collapsible_if, clippy::manual_is_multiple_of)]
fn packed_extent(packed: &Array<u32>, first: usize, #[comptime] rows: usize) -> bool {
    let mut valid = false;
    if first % 4 == 0 && first / 4 <= packed.len() {
        valid = rows * ROW_BYTES / 4 <= packed.len() - first / 4;
    }
    valid
}

#[cube]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord
)]
fn head_finish(
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    value: f32,
    row: usize,
    subset_first: usize,
    count: usize,
    active: bool,
    selected: bool,
    #[comptime] check_tail: bool,
) {
    if lane() == 0 && selected {
        if active {
            output[subset_first + row] = value;
            if !(value.abs() <= f32::MAX) {
                fault[0].fetch_or(1);
            }
        } else {
            fault[0].fetch_or(1);
        }
    }
    if comptime!(check_tail) {
        let index = ABSOLUTE_POS;
        if index >= ENGLISH_ROWS && index < output.len() {
            if !(output[index].abs() <= f32::MAX) {
                fault[0].fetch_or(1);
            }
        }
    }
    if lane() == 0 && count != 1 {
        fault[0].fetch_or(1);
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn packed_q8_english_head_fault(
    packed: &Array<u32>,
    ids: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    source_first: usize,
    subset_first: usize,
    #[comptime] source_rows: usize,
    #[comptime] subset_rows: usize,
    #[comptime] check_tail: bool,
) {
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + local_row();
    let selected = row < subset_rows;
    let mut source = 248_320_usize;
    if selected && subset_first <= ENGLISH_ROWS && row < ENGLISH_ROWS - subset_first {
        source = original_row(ids, subset_first + row, source_first, source_rows);
    }
    let mut active = count == 1 && selected && source < source_rows;
    active = active && packed_extent(packed, packed_first, source_rows);
    if input_first <= input.len() {
        active = active && D <= input.len() - input_first;
    } else {
        active = false;
    }
    active = active && VOCAB <= output.len();
    let value = dot(
        packed,
        input,
        packed_first + source * ROW_BYTES,
        input_first,
        active,
        D,
        Q8_0,
    );
    head_finish(
        output,
        fault,
        value,
        row,
        subset_first,
        count,
        active,
        selected,
        check_tail,
    );
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
pub fn packed_q8_english_head_shared_scales_fault(
    packed: &Array<u32>,
    ids: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    source_first: usize,
    subset_first: usize,
    #[comptime] source_rows: usize,
    #[comptime] subset_rows: usize,
    #[comptime] check_tail: bool,
) {
    let blocks = comptime!(D / 32);
    let mut scales = SharedMemory::<f32>::new(comptime!(ROWS_PER_CUBE * D / 32));
    #[unroll]
    for slice in 0..comptime!((ROWS_PER_CUBE * D / 32).div_ceil(ROWS_PER_CUBE * PHYSICAL_LANES)) {
        let index = UNIT_POS_X as usize + slice * ROWS_PER_CUBE * PHYSICAL_LANES;
        if index < ROWS_PER_CUBE * blocks {
            let row = CUBE_POS_X as usize * ROWS_PER_CUBE + index / blocks;
            let mut source = 248_320_usize;
            if row < subset_rows
                && subset_first <= ENGLISH_ROWS
                && row < ENGLISH_ROWS - subset_first
            {
                source = original_row(ids, subset_first + row, source_first, source_rows);
            }
            let mut scale = 0.0_f32;
            if count == 1
                && source < source_rows
                && packed_extent(packed, packed_first, source_rows)
            {
                let base = packed_first + source * ROW_BYTES + (index % blocks) * 34;
                scale = binary16(u16_at(packed, base));
            }
            scales[index] = scale;
        }
    }
    sync_cube();
    let slot = local_row();
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + slot;
    let selected = row < subset_rows;
    let mut source = 248_320_usize;
    if selected && subset_first <= ENGLISH_ROWS && row < ENGLISH_ROWS - subset_first {
        source = original_row(ids, subset_first + row, source_first, source_rows);
    }
    let mut active = count == 1 && selected && source < source_rows;
    active = active && packed_extent(packed, packed_first, source_rows);
    if input_first <= input.len() {
        active = active && D <= input.len() - input_first;
    } else {
        active = false;
    }
    active = active && VOCAB <= output.len();
    let logical_lane = lane();
    let mut low = 0.0_f32;
    let mut high = 0.0_f32;
    if active {
        for tile in 0..comptime!(D.div_ceil(64)) {
            let base = packed_first + source * ROW_BYTES + tile * 68;
            let col = tile * 64 + logical_lane;
            if col < D {
                let quant = signed_byte(byte_at(packed, base + 2 + logical_lane));
                low += (scales[slot * blocks + tile * 2] * quant as f32) * input[input_first + col];
            }
            if col + 32 < D {
                let quant = signed_byte(byte_at(packed, base + 36 + logical_lane));
                high += (scales[slot * blocks + tile * 2 + 1] * quant as f32)
                    * input[input_first + col + 32];
            }
        }
    }
    let value = reduce_original64(low, high);
    head_finish(
        output,
        fault,
        value,
        row,
        subset_first,
        count,
        active,
        selected,
        check_tail,
    );
}

#[cube]
#[allow(clippy::manual_range_contains)]
fn finite(value: f32) -> bool {
    value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32
}

/// Compact logits, full original IDs. Never scan omitted rows/cleared zeros.
#[cube(launch, launch_unchecked)]
pub fn english_greedy_chunks(
    logits: &Array<f32>,
    ids: &Array<u32>,
    candidates: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let group = CUBE_POS_X as usize;
    let lane = UNIT_POS_X as usize;
    let mut best = 0.0_f32;
    best -= 3.4028235e38_f32;
    let mut id = 4294967295_u32;
    let mut good = 1_u32;
    if logits.len() < ENGLISH_ROWS || ids.len() < ENGLISH_ROWS || candidates.len() < GREEDY_WORDS {
        good = 0;
    } else {
        for tile in 0..GREEDY_CHUNK / 64 {
            let index = group * GREEDY_CHUNK + tile * 64 + lane;
            if index < ENGLISH_ROWS {
                let v = logits[index];
                let token = ids[index];
                if !finite(v) || token >= VOCAB as u32 {
                    good = 0;
                } else if v > best || v == best && token < id {
                    best = v;
                    id = token;
                }
            }
        }
    }
    let mut values = SharedMemory::<f32>::new(64_usize);
    let mut ids = SharedMemory::<u32>::new(64_usize);
    let mut flags = SharedMemory::<u32>::new(64_usize);
    values[lane] = best;
    ids[lane] = id;
    flags[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            let v = values[lane + stride];
            let j = ids[lane + stride];
            if v > values[lane] || v == values[lane] && j < ids[lane] {
                values[lane] = v;
                ids[lane] = j;
            }
            flags[lane] &= flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 && group < ENGLISH_GREEDY_GROUPS {
        if candidates.len() >= GREEDY_WORDS {
            candidates[group * 3] = ids[0];
            candidates[group * 3 + 1] = values[0].to_bits();
            candidates[group * 3 + 2] = flags[0];
        }
        if flags[0] != 1 {
            fault[0].fetch_or(1);
        }
    }
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::len_zero)] // Cube Array has no is_empty operation.
pub fn english_greedy_finish(
    candidates: &Array<u32>,
    token: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let lane = UNIT_POS_X as usize;
    let mut value = 0.0_f32;
    value -= 3.4028235e38_f32;
    let mut id = 4294967295_u32;
    let mut good = 1_u32;
    if candidates.len() < GREEDY_WORDS || token.len() < 1 {
        good = 0;
    } else if lane < ENGLISH_GREEDY_GROUPS {
        id = candidates[lane * 3];
        value = f32::reinterpret(candidates[lane * 3 + 1]);
        if candidates[lane * 3 + 2] != 1 || id >= VOCAB as u32 || !finite(value) {
            good = 0;
        }
    }
    let mut values = SharedMemory::<f32>::new(64_usize);
    let mut ids = SharedMemory::<u32>::new(64_usize);
    let mut flags = SharedMemory::<u32>::new(64_usize);
    values[lane] = value;
    ids[lane] = id;
    flags[lane] = good;
    sync_cube();
    let mut stride = 32_usize;
    while stride > 0 {
        if lane < stride {
            let v = values[lane + stride];
            let j = ids[lane + stride];
            if v > values[lane] || v == values[lane] && j < ids[lane] {
                values[lane] = v;
                ids[lane] = j;
            }
            flags[lane] &= flags[lane + stride];
        }
        sync_cube();
        stride /= 2;
    }
    if lane == 0 {
        if flags[0] == 1 && fault[0].load() == 0 {
            token[0] = ids[0];
        } else {
            fault[0].fetch_or(1);
        }
    }
}

#[cfg(test)]
#[path = "mtp_pruned_head_tests.rs"]
mod tests;
