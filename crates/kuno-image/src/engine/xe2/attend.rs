//! Attention for Xe2 as one machine-code kernel: scores, softmax weights and mixing in a
//! single pass over the keys, with nothing stored in between.
//!
//! One hardware thread owns 16 queries of one head (128 channels) and walks the keys 16 at a
//! time. Queries are the 16 lanes of every register, so everything per query is lane-wise.
//! Per step it
//!   1. multiplies the 16 keys with its queries (half-float `dpas`, f32 scores, one register
//!      per key),
//!   2. turns each key's scores into weights `2^((score - ref) * c)`, sums them, and packs
//!      them as half floats,
//!   3. multiplies the 16 values with the weights into f32 accumulators (one register per
//!      channel).
//!
//! The caller divides the mixed values by the weight sum, so `ref` (one per query) only has
//! to keep weights in half-float range. It starts as the first key's score; when a later
//! score exceeds it by more than 2^11, a slow path raises it and rescales what has been
//! accumulated. The exponents replace the scores in their registers, which leaves a load
//! buffer free while weights are computed and lends the scores' and the weights' own
//! registers as further buffers when they are idle: four loads are in flight during
//! scoring, two during the weights and five during mixing.
//!
//! Operands are streams, so every load is one contiguous 512-byte run:
//!   Q   `[head][query tile][channel chunk 0..8][channel pair 0..8][query 0..16][2 halves]`
//!   KV  `[head][key chunk]` of `CHUNK_BYTES`: keys `[channel chunk 0..8][key 0..16][16
//!       halves]`, then values `[channel 0..128][key 0..16]` halves.
//!   OUT `[head][query tile][channel 0..128][query 0..16]` f32 (transposed), and
//!   SUM `[head][query tile][query 0..16]` f32.
//!
//! Keys past the live count are never used; the KV buffer needs `SLACK` bytes after its
//! last chunk because loads run one chunk ahead.
use super::{Ty::*, *};

/// Queries per thread and keys per step.
pub const TILE: usize = 16;
pub const CHANNELS: usize = 128;
/// Bytes of one query tile in the Q stream and of one key chunk in the KV stream.
pub const TILE_BYTES: usize = TILE * CHANNELS * 2;
pub const CHUNK_BYTES: usize = 2 * TILE * CHANNELS * 2;
/// Offset of the values inside a key chunk.
pub const VALUES: usize = TILE * CHANNELS * 2;
/// Bytes of one query tile in OUT and in SUM.
pub const OUT_BYTES: usize = TILE * CHANNELS * 4;
pub const SUM_BYTES: usize = TILE * 4;
pub const SLACK: usize = CHUNK_BYTES;

const LOAD_64X64: u32 = 0x0288_F780;
const STORE_64X64: u32 = 0x0200_F784;
const STORE_64X8: u32 = 0x0200_C784;
/// Weights may grow to 2^LIMIT before `ref` is raised (half floats end at 2^16).
const LIMIT: f32 = 11.0;

#[derive(Clone, Copy)]
pub struct Shape {
    /// Padded tokens (a multiple of 16): queries per head, and the key chunks' stride.
    pub padded: u32,
    /// Live keys.
    pub tokens: u32,
    /// Scores are multiplied by this before the softmax.
    pub scale: f32,
}
impl Shape {
    /// None when the offsets leave 32 bits.
    pub fn new(padded: usize, tokens: usize, heads: usize, scale: f32) -> Option<Shape> {
        let fits = padded.is_multiple_of(TILE)
            && tokens > 0
            && tokens <= padded
            && heads * (padded / TILE) * CHUNK_BYTES < 1 << 32;
        fits.then_some(Shape {
            padded: padded as u32,
            tokens: tokens as u32,
            scale,
        })
    }
    /// Thread groups: query tiles, heads.
    pub fn groups(&self, heads: usize) -> [u32; 3] {
        [self.padded / TILE as u32, heads as u32, 1]
    }
}

/// Byte offsets inside one head's streams.
pub fn query_at(query: usize, channel: usize) -> usize {
    (query / TILE) * TILE_BYTES
        + (channel / 16) * 512
        + (channel % 16 / 2) * 64
        + (query % TILE) * 4
        + (channel % 2) * 2
}
pub fn key_at(key: usize, channel: usize) -> usize {
    (key / TILE) * CHUNK_BYTES + (channel / 16) * 512 + (key % TILE) * 32 + (channel % 16) * 2
}
pub fn value_at(key: usize, channel: usize) -> usize {
    (key / TILE) * CHUNK_BYTES + VALUES + channel * 32 + (key % TILE) * 2
}
pub fn out_at(query: usize, channel: usize) -> usize {
    (query / TILE) * OUT_BYTES + channel * 64 + (query % TILE) * 4
}

// Register plan (256 registers).
const KV_AT: u32 = 3;
const OUT_AT: u32 = 4;
const SUM_AT: u32 = 5;
const REF: u32 = 6;
const CONST: u32 = 7;
const INT: u32 = 8;
const MAX: u32 = 9;
const SUM: u32 = 10;
const WIDE: u32 = 11;
const ROUND: u32 = 12;
const POWER: u32 = 13;
const STEP: u32 = 14;
const HEADER: u32 = 15;
/// Queries, eight registers per channel chunk.
fn q(j: u32) -> u32 {
    16 + j * 8
}
/// Accumulators, eight channels (registers) per block.
fn out(b: u32) -> u32 {
    80 + b * 8
}
/// Scores of keys `8h..8h + 8`.
fn s(h: u32) -> u32 {
    208 + h * 8
}
/// Weights: register `g` holds keys `2g` (low halves) and `2g + 1` (high halves).
const P: u32 = 224;
/// Load buffers as (first register, scoreboard token). Three are buffers only; the weights'
/// registers serve as a fourth while scores are computed, and the scores' registers as a
/// fifth and sixth while values are mixed.
fn slot(i: u32) -> (u32, u16) {
    match i {
        0..=2 => (232 + i * 8, i as u16),
        3 => (P, 21),
        _ => (s(i - 4), 18 + i as u16),
    }
}
/// Scratch while weights are computed: the third load buffer.
const N: [u32; 4] = [248, 249, 250, 251];
const PP: [u32; 4] = [252, 253, 254, 255];
// Scoreboard tokens: 0..3 and 21..24 for loads, then one per score register group and per
// accumulator block.
fn score(h: u32) -> u16 {
    3 + h as u16
}
fn mix(b: u32) -> u16 {
    5 + b as u16
}
const LAST_MIX: u16 = 20;
// Lanes of INT: 0 counts steps, 1..3 are scratch.
const CHUNK: u32 = 4;
const START: u32 = 5;
// Lanes of CONST.
const C: u32 = 0;
const A1: u32 = 1;
const A2: u32 = 2;
const A3: u32 = 3;
const A4: u32 = 4;
const ONE: u32 = 5;

/// One step of `2^f` for the fraction `frac` (|f| <= 0.5) into `to`: a fourth-degree
/// Taylor polynomial evaluated in four multiply-adds, 3e-5 relative error.
fn power(k: &mut Asm, part: u32, to: Reg, frac: Reg, w: Swsb) {
    let c = |lane: u32| r(CONST, lane, F).s();
    match part {
        0 => k.mad(16, to, c(A3), frac, c(A4), w),
        1 => k.mad(16, to, c(A2), to, frac, w),
        2 => k.mad(16, to, c(A1), to, frac, w),
        _ => k.mad(16, to, c(ONE), to, frac, w),
    }
}
fn fetch(k: &mut Asm, to: u32, offset: u32) {
    if skip(1) {
        return;
    }
    let (register, token) = slot(to);
    k.send(
        SFID_UGM,
        r(register, 0, Ud),
        r(KV_AT, 0, Ud),
        null(Ud),
        0,
        offset << 12,
        LOAD_64X64,
        false,
        Swsb::after_set(1, token),
    );
}

/// Timing experiments only: bit 0 skips the weights, 1 the key and value loads, 2 the
/// score products, 3 the mixing products.
#[cfg(test)]
static SKIP: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
fn skip(bit: u32) -> bool {
    #[cfg(test)]
    return SKIP.load(std::sync::atomic::Ordering::Relaxed) >> bit & 1 == 1;
    #[cfg(not(test))]
    {
        let _ = bit;
        false
    }
}

/// One key chunk with `live` live keys, starting with its first three key loads requested
/// and ending with the next chunk's requested.
fn step(k: &mut Asm, live: u32) {
    let w = Swsb::after(1);
    let (exp, init, slow, weights) = (k.label(), k.label(), k.label(), k.label());
    let row = |key: u32| s(key / 8) + key % 8;
    let keys: Vec<u32> = (0..live).collect();

    // Scores: eight channel chunks, each against both key groups. The weights' registers
    // are a load buffer once the previous step's mixing has read them.
    k.sync(w);
    k.sync(Swsb::src(LAST_MIX));
    fetch(k, 3, 3 * 512);
    for j in 0..8 {
        let (buffer, token) = slot(j % 4);
        k.sync(Swsb::dst(token));
        for h in 0..2 {
            let acc = if j == 0 {
                null(F)
            } else {
                k.sync(Swsb::dst(score(h)));
                r(s(h), 0, F)
            };
            if skip(2) {
                continue;
            }
            k.dpas(
                8,
                8,
                16,
                r(s(h), 0, F),
                acc,
                r(q(j), 0, Hf),
                r(buffer + 4 * h, 0, Hf),
                Swsb::set(score(h)),
            );
        }
        // Refill the buffer once both products have read it: a later key chunk, or the
        // first values. The third buffer and the weights' registers are needed next.
        k.sync(Swsb::src(score(1)));
        match j {
            0..=3 => fetch(k, j, (j + 4) * 512),
            4 | 5 => fetch(k, j - 4, VALUES as u32 + (j - 4) * 512),
            _ => {}
        }
    }
    k.sync(Swsb::dst(score(0)));
    k.sync(Swsb::dst(score(1)));
    k.cmp(Cond::Eq, r(INT, 0, D), imm(0), w);
    k.jmpi(init, true, w);

    // Exponents, in place of the scores: t = score * c - ref, and the largest per query.
    // Rows (keys) go in groups of up to four, one instruction kind at a time. Integer and
    // float instructions run in separate pipes: the first instruction of a group that
    // reads the other pipe's results waits for everything before it, which covers the
    // rest of its group.
    k.place(exp);
    let first = |i: usize| if i == 0 { w } else { Swsb::NONE };
    for group in keys.chunks(4) {
        for (i, &key) in group.iter().enumerate() {
            k.mad(
                16,
                r(row(key), 0, F),
                r(REF, 0, F).minus(),
                r(row(key), 0, F),
                r(CONST, C, F).s(),
                first(i),
            );
        }
        for &key in group {
            if key == 0 {
                k.mov(16, r(MAX, 0, F), r(row(key), 0, F), w);
            } else {
                k.sel(
                    16,
                    Cond::Ge,
                    r(MAX, 0, F),
                    r(MAX, 0, F),
                    r(row(key), 0, F),
                    w,
                );
            }
        }
    }
    // Largest t of the step per query, floored at zero. Non-negative floats order like
    // their bits, so the reduction across queries can run on integers.
    k.sel(16, Cond::Ge, r(MAX, 0, F), r(MAX, 0, F), immf(0.0), w);
    k.sel(8, Cond::Ge, r(WIDE, 0, D), r(MAX, 0, D), r(MAX, 8, D), w);
    for half in [4, 2, 1] {
        k.sel(
            half,
            Cond::Ge,
            r(WIDE, 0, D),
            r(WIDE, 0, D),
            r(WIDE, half, D),
            w,
        );
    }
    k.cmp(Cond::Gt, r(WIDE, 0, D), imm(LIMIT.to_bits() as i32), w);
    if !skip(2) {
        k.jmpi(slow, true, w);
    }

    // Weights: 2^t = 2^round(t) * 2^fraction, packed as half floats.
    k.place(weights);
    k.sync(w);
    if live < 16 || skip(0) {
        for reg in P..P + 8 {
            k.mov(16, r(reg, 0, D), imm(0), Swsb::NONE);
        }
    }
    let weighted = if skip(0) { &[][..] } else { &keys[..] };
    for group in weighted.chunks(4) {
        let back = Swsb::after(group.len() as u16);
        let each = |k: &mut Asm, f: &dyn Fn(&mut Asm, usize, u32)| {
            for (i, key) in group.iter().enumerate() {
                f(k, i, *key);
            }
        };
        let (n, pp) = (|i: usize| r(N[i], 0, F), |i: usize| r(PP[i], 0, F));
        let t = |key: u32| r(row(key), 0, F);
        each(k, &|k, i, key| {
            k.sel(16, Cond::Ge, t(key), t(key), immf(-126.0), first(i))
        });
        each(k, &|k, i, key| k.rnde(16, n(i), t(key), back));
        each(k, &|k, i, key| {
            k.add(16, t(key), t(key), n(i).minus(), back)
        });
        for part in 0..4 {
            each(k, &|k, i, key| power(k, part, pp(i), t(key), back));
        }
        each(k, &|k, i, _| k.mov(16, r(N[i], 0, D), n(i), back));
        each(k, &|k, i, _| {
            k.shl(16, r(N[i], 0, D), r(N[i], 0, D), immw(23), first(i))
        });
        each(k, &|k, i, _| {
            k.add(16, r(PP[i], 0, D), r(PP[i], 0, D), r(N[i], 0, D), back)
        });
        // Even keys convert straight into the low halves; odd keys go through their own
        // row, because a float destination must keep its lanes in place.
        each(k, &|k, i, key| {
            let to = if key % 2 == 0 {
                r(P + key / 2, 0, Hf)
            } else {
                r(row(key), 0, Hf)
            };
            k.mov(16, to.step2(), pp(i), first(i));
        });
        each(k, &|k, i, key| {
            if key == 0 {
                k.mov(16, r(STEP, 0, F), pp(i), w);
            } else {
                k.add(16, r(STEP, 0, F), r(STEP, 0, F), pp(i), w);
            }
        });
        each(k, &|k, _, key| {
            if key % 2 == 1 {
                k.mov(
                    16,
                    r(P + key / 2, 1, Uw).step2(),
                    r(row(key), 0, Uw).step2(),
                    w,
                );
            }
        });
        k.sync(w);
    }

    // Mixing: eight pairs of channel blocks. The first two were requested during scoring;
    // the scratch buffer and the scores' registers take the next three.
    k.add(16, r(SUM, 0, F), r(SUM, 0, F), r(STEP, 0, F), w);
    k.sync(w);
    for m in 2..5 {
        fetch(k, [2, 4, 5][m as usize - 2], VALUES as u32 + m * 512);
    }
    for m in 0..8 {
        let (buffer, token) = slot([0, 1, 2, 4, 5, 0, 1, 2][m as usize]);
        k.sync(Swsb::dst(token));
        for e in 0..2 {
            let b = 2 * m + e;
            k.sync(Swsb::dst(mix(b)));
            if skip(3) {
                continue;
            }
            k.dpas(
                8,
                8,
                16,
                r(out(b), 0, F),
                r(out(b), 0, F),
                r(P, 0, Hf),
                r(buffer + 4 * e, 0, Hf),
                Swsb::set(mix(b)),
            );
        }
        // Refill with the last values, then with the next chunk's first keys.
        k.sync(Swsb::src(mix(2 * m + 1)));
        match m {
            0..=2 => fetch(k, m, VALUES as u32 + (m + 5) * 512),
            5..=7 => fetch(k, m - 5, CHUNK_BYTES as u32 + (m - 5) * 512),
            _ => {}
        }
    }
    for i in 0..3 {
        k.sync(Swsb::src(slot(i).1));
    }
    k.add(
        1,
        r(KV_AT, 0, Q),
        r(KV_AT, 0, Q),
        imm(CHUNK_BYTES as i32),
        w,
    );
    k.add(1, r(INT, 0, D), r(INT, 0, D), imm(1), w);
    let done = k.label();
    k.jmpi(done, false, w);

    // First step: every query's reference is its score against the first key.
    k.place(init);
    k.mul(16, r(REF, 0, F), r(row(0), 0, F), r(CONST, C, F).s(), w);
    k.jmpi(exp, false, w);

    // A weight would leave half-float range: raise every query's reference by its largest
    // t of this step, lower the step's exponents by the same, and scale what has been
    // accumulated by 2^-(the raise).
    k.place(slow);
    for token in mix(0)..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }
    let (t, n, pp) = (r(WIDE, 0, F), r(ROUND, 0, F), r(POWER, 0, F));
    k.add(16, r(REF, 0, F), r(REF, 0, F), r(MAX, 0, F), w);
    for &key in &keys {
        k.add(
            16,
            r(row(key), 0, F),
            r(row(key), 0, F),
            r(MAX, 0, F).minus(),
            w,
        );
    }
    k.sel(16, Cond::Ge, t, r(MAX, 0, F).minus(), immf(-126.0), w);
    k.rnde(16, n, t, w);
    k.add(16, t, t, n.minus(), w);
    for part in 0..4 {
        power(k, part, pp, t, w);
    }
    k.mov(16, r(ROUND, 0, D), n, w);
    k.shl(16, r(ROUND, 0, D), r(ROUND, 0, D), immw(23), w);
    k.add(16, r(POWER, 0, D), r(POWER, 0, D), r(ROUND, 0, D), w);
    for reg in (out(0)..out(16)).chain([SUM]) {
        k.mul(16, r(reg, 0, F), r(reg, 0, F), pp, w);
    }
    k.jmpi(weights, false, w);
    k.place(done);
}

pub fn kernel(shape: &Shape) -> Vec<u8> {
    let Shape {
        padded,
        tokens,
        scale,
    } = *shape;
    let tiles = padded / TILE as u32;
    let w = Swsb::after(1);
    let int = |lane: u32| r(INT, lane, D);
    let mut k = Asm::new();

    // r0: thread header (query tile in dword 1, head in dword 6); r1: Q, KV, OUT, SUM.
    k.mov(16, r(HEADER, 0, F), r(0, 0, F), Swsb::NONE);
    let ln2 = std::f32::consts::LN_2;
    let consts = [
        (C, scale * std::f32::consts::LOG2_E),
        (A1, ln2),
        (A2, ln2 * ln2 / 2.0),
        (A3, ln2 * ln2 * ln2 / 6.0),
        (A4, ln2 * ln2 * ln2 * ln2 / 24.0),
        (ONE, 1.0),
    ];
    for (lane, value) in consts {
        k.mov(1, r(CONST, lane, F), immf(value), Swsb::NONE);
    }
    // Addresses of this thread's streams: per head, plus per query tile.
    let sizes = [TILE_BYTES, CHUNK_BYTES, OUT_BYTES, SUM_BYTES];
    for (i, size) in sizes.into_iter().enumerate() {
        let i = i as u32;
        k.mov(1, int(1), imm((tiles * size as u32) as i32), Swsb::NONE);
        k.mul(1, int(2), int(1), r(0, 12, Uw), w);
        if i != 1 {
            k.shl(1, int(3), r(0, 1, D), immw(size.trailing_zeros() as i16), w);
            k.add(1, int(2), int(2), int(3), w);
        }
        k.mov(1, r(WIDE, 0, Uq), r(INT, 2, Ud), w);
        k.add(1, r(2 + i, 0, Q), r(1, i, Q), r(WIDE, 0, Q), w);
    }
    // Tile `i` starts at key chunk `i` and wraps around: threads start together, and a thread
    // that reads what another read a step ago finds it cached, where threads in step all
    // wait for the same memory. Tiles past the last full chunk start at the first.
    let (full, rest) = (tokens / TILE as u32, tokens % TILE as u32);
    k.mov(1, int(START), imm(0), Swsb::NONE);
    if full > 0 {
        let first = k.label();
        k.cmp(Cond::Ge, r(0, 1, D), imm(full as i32), w);
        k.jmpi(first, true, w);
        k.mov(1, int(START), r(0, 1, D), w);
        k.shl(
            1,
            int(2),
            r(0, 1, D),
            immw(CHUNK_BYTES.trailing_zeros() as i16),
            w,
        );
        k.mov(1, r(WIDE, 0, Uq), r(INT, 2, Ud), w);
        k.add(1, r(KV_AT, 0, Q), r(KV_AT, 0, Q), r(WIDE, 0, Q), w);
        k.place(first);
    }
    k.mov(1, int(CHUNK), int(START), w);
    // The queries; the same tokens later belong to the accumulator blocks, so every block
    // token has a predecessor to wait for (the second eight get a throwaway load).
    for j in 0..8 {
        k.send(
            SFID_UGM,
            r(q(j), 0, Ud),
            r(2, 0, Ud),
            null(Ud),
            0,
            (j * 512) << 12,
            LOAD_64X64,
            false,
            Swsb::after_set(1, mix(j)),
        );
    }
    for b in 8..16 {
        k.send(
            SFID_UGM,
            r(slot(0).0, 0, Ud),
            r(KV_AT, 0, Ud),
            null(Ud),
            0,
            0,
            LOAD_64X64,
            false,
            Swsb::after_set(1, mix(b)),
        );
    }
    for reg in (out(0)..out(16)).chain([SUM]) {
        k.mov(16, r(reg, 0, D), imm(0), Swsb::NONE);
    }
    k.mov(1, int(0), imm(0), Swsb::NONE);
    for token in mix(0)..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }
    for j in 0..3 {
        fetch(&mut k, j, j * 512);
    }

    // Chunks from the start to the last, then from the first up to the start.
    if full > 0 {
        let top = k.label();
        k.place(top);
        step(&mut k, TILE as u32);
        k.add(1, int(CHUNK), int(CHUNK), imm(1), w);
        k.cmp(Cond::Lt, int(CHUNK), imm(full as i32), w);
        k.jmpi(top, true, w);
    }
    if rest > 0 {
        step(&mut k, rest);
    }
    if full > 0 {
        let (top, end) = (k.label(), k.label());
        k.cmp(Cond::Eq, int(START), imm(0), w);
        k.jmpi(end, true, w);
        // Back to the first chunk; the loads requested past the last one are replaced.
        let walked = (full + u32::from(rest > 0)) * CHUNK_BYTES as u32;
        k.add(1, r(KV_AT, 0, Q), r(KV_AT, 0, Q), imm(-(walked as i32)), w);
        k.mov(1, int(CHUNK), imm(0), w);
        for j in 0..3 {
            k.sync(Swsb::dst(slot(j).1));
        }
        k.sync(w);
        for j in 0..3 {
            fetch(&mut k, j, j * 512);
        }
        k.place(top);
        step(&mut k, TILE as u32);
        k.add(1, int(CHUNK), int(CHUNK), imm(1), w);
        k.cmp(Cond::Lt, int(CHUNK), int(START), w);
        k.jmpi(top, true, w);
        k.place(end);
    }

    // Store. Accumulator registers are in memory order: channel, then query.
    for token in 0..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }
    k.sync(w);
    for b in 0..16 {
        k.send(
            SFID_UGM,
            null(Ud),
            r(OUT_AT, 0, Ud),
            r(out(b), 0, Ud),
            8,
            (b * 512) << 12,
            STORE_64X64,
            false,
            Swsb::after_set(1, b as u16),
        );
    }
    k.send(
        SFID_UGM,
        null(Ud),
        r(SUM_AT, 0, Ud),
        r(SUM, 0, Ud),
        1,
        0,
        STORE_64X8,
        false,
        Swsb::after_set(1, 16),
    );
    for token in 0..=16 {
        k.sync(Swsb::dst(token));
    }
    k.end(r(HEADER, 0, F));
    k.bytes()
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            super::{
                gpu::{Buffer, Gpu, Select, bytes_of, floats},
                safetensors::{narrow, widen},
            },
            level_zero::LevelZero,
            zebin,
        },
        *,
    };

    fn noise(count: usize, seed: u64, spread: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * spread
            })
            .collect()
    }

    /// Builds without tripping the encoder's checks; `KUNO_XE2_DUMP` names a file to write
    /// the code to, for a look through an outside disassembler.
    #[test]
    fn kernel_builds() {
        let code = kernel(&Shape::new(256, 100, 3, 0.088).unwrap());
        assert!(code.len().is_multiple_of(16) && code.len() > 16 * 500);
        if let Ok(path) = std::env::var("KUNO_XE2_DUMP") {
            std::fs::write(path, &code).unwrap();
        }
    }

    /// The kernel against a plain softmax on the CPU, on random half-float operands: mild
    /// scores, sharp scores (which force the reference to be raised), and a clip-sized run
    /// that is timed and spot-checked.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn attention_matches_cpu() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let scale = 1.0 / (CHANNELS as f32).sqrt();
        for (heads, padded, tokens, spread, checks) in [
            (3usize, 256usize, 100usize, 1.0f32, 768usize),
            (2, 256, 250, 6.0, 512),
            (56, 8704, 8611, 1.5, 48),
        ] {
            let half = |values: Vec<f32>| -> Vec<f32> {
                values.into_iter().map(|v| widen(narrow(v))).collect()
            };
            let queries = half(noise(heads * padded * CHANNELS, 1, spread));
            let keys = half(noise(heads * padded * CHANNELS, 2, spread));
            let values = half(noise(heads * padded * CHANNELS, 3, 1.0));
            let tiles = padded / TILE;
            let mut q_stream = vec![0u16; heads * tiles * TILE_BYTES / 2];
            let mut kv_stream = vec![0u16; (heads * tiles * CHUNK_BYTES + SLACK) / 2];
            for head in 0..heads {
                let (q0, kv0) = (head * tiles * TILE_BYTES, head * tiles * CHUNK_BYTES);
                for token in 0..padded {
                    let from = (head * padded + token) * CHANNELS;
                    // Dead keys and values are noise too: the kernel must not use them.
                    for channel in 0..CHANNELS {
                        q_stream[(q0 + query_at(token, channel)) / 2] =
                            narrow(queries[from + channel]);
                        kv_stream[(kv0 + key_at(token, channel)) / 2] =
                            narrow(keys[from + channel]);
                        kv_stream[(kv0 + value_at(token, channel)) / 2] =
                            narrow(values[from + channel]);
                    }
                }
            }
            let (out_size, sum_size) = (heads * tiles * OUT_BYTES, heads * tiles * SUM_BYTES);
            let buffers = [
                gpu.shared((q_stream.len() * 2) as u64).unwrap(),
                gpu.shared((kv_stream.len() * 2) as u64).unwrap(),
                gpu.shared(out_size as u64).unwrap(),
                gpu.shared(sum_size as u64).unwrap(),
            ];
            gpu.upload(&buffers[0], 0, bytes_of(&q_stream)).unwrap();
            gpu.upload(&buffers[1], 0, bytes_of(&kv_stream)).unwrap();
            let import = |buffer: &Buffer| {
                fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                    .unwrap()
            };
            let memory: Vec<_> = buffers.iter().map(import).collect();
            let shape = Shape::new(padded, tokens, heads, scale).unwrap();
            let code = kernel(&shape);
            let native = zebin::write(&zebin::Kernel {
                name: "attend",
                code: &code,
                grf: 256,
                pointers: 4,
            });
            let attend = fast.kernel(&native, "attend", 4).unwrap();
            let arguments: Vec<_> = memory.iter().collect();
            let mut best = f64::MAX;
            for _ in 0..4 {
                let started = std::time::Instant::now();
                fast.launch(&attend, &arguments, shape.groups(heads))
                    .unwrap();
                best = best.min(started.elapsed().as_secs_f64() * 1e3);
            }
            let mixed = floats(&gpu.download(&buffers[2], 0, out_size).unwrap());
            let sums = floats(&gpu.download(&buffers[3], 0, sum_size).unwrap());
            let (mut error, mut power, mut worst) = (0f64, 0f64, 0f64);
            for check in 0..checks {
                let (head, query) = if checks == heads * padded {
                    (check / padded, check % padded)
                } else {
                    ((check * 7) % heads, (check * 2731) % padded)
                };
                let q = &queries[(head * padded + query) * CHANNELS..][..CHANNELS];
                let logits: Vec<f64> = (0..tokens)
                    .map(|key| {
                        let k = &keys[(head * padded + key) * CHANNELS..][..CHANNELS];
                        f64::from(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale)
                    })
                    .collect();
                let top = logits.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
                let total: f64 = weights.iter().sum();
                let sum = f64::from(sums[head * padded + query]);
                for channel in 0..CHANNELS {
                    let want: f64 = (0..tokens)
                        .map(|key| {
                            weights[key]
                                * f64::from(values[(head * padded + key) * CHANNELS + channel])
                        })
                        .sum::<f64>()
                        / total;
                    let got =
                        f64::from(mixed[(head * tiles * OUT_BYTES + out_at(query, channel)) / 4])
                            / sum;
                    error += (got - want).powi(2);
                    power += want.powi(2);
                    worst = worst.max((got - want).abs());
                }
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!(
                "{heads} heads, {tokens} of {padded} tokens, spread {spread}: {best:.2} ms, {db:.1} dB, worst {worst:.2e}, {} instructions",
                code.len() / 16
            );
            assert!(db > 50.0);
            if heads > 8 {
                // Timing only: without the weights (1), the loads (2), the score products
                // (4), the mixing products (8).
                for mask in [1u32, 3, 13, 15] {
                    SKIP.store(mask, std::sync::atomic::Ordering::Relaxed);
                    let code = kernel(&shape);
                    SKIP.store(0, std::sync::atomic::Ordering::Relaxed);
                    let native = zebin::write(&zebin::Kernel {
                        name: "attend",
                        code: &code,
                        grf: 256,
                        pointers: 4,
                    });
                    let bare = fast.kernel(&native, "attend", 4).unwrap();
                    let mut best = f64::MAX;
                    for _ in 0..4 {
                        let started = std::time::Instant::now();
                        fast.launch(&bare, &arguments, shape.groups(heads)).unwrap();
                        best = best.min(started.elapsed().as_secs_f64() * 1e3);
                    }
                    println!("  skipping {mask:04b}: {best:.2} ms");
                }
            }
        }
    }
}
