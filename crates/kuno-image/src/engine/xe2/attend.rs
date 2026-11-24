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
//!
//! With `Shape::int8` queries and keys are signed bytes (SageAttention-style: scores from an
//! integer `dpas`, half as many key bytes to load), each with its own scale; values and
//! mixing stay half floats. Streams keep their sizes and places:
//!   Q   `[head][query tile]`: `[channel chunk 0..4][channel quad 0..8][query 0..16][4
//!       bytes]`, then at `SCALES` the queries' scales `[query 0..16]` f32.
//!   KV  `[head][key chunk]`: keys `[channel chunk 0..4][key 0..16][32 bytes]`, at `SCALES`
//!       the keys' scales `[key 0..16]` f32, and the values at `VALUES` as before.
//! A score is then `(integer product) * query scale * key scale`. Half the key bytes and
//! half the query registers leave room for three more load buffers, so every value block is
//! requested a phase before it is mixed. With loads no longer the limit, the arithmetic
//! between products is what remains: conversions go four at a time, the largest exponent
//! is kept as four running maxima (also in the half-float form: exact, so the same result),
//! the weights come from the math pipe's `2^x` instead of the polynomial, and their sum is
//! kept as four running sums.
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
/// Offset of the scales in an int8 query tile and in an int8 key chunk.
pub const SCALES: usize = 2048;

const LOAD_64X64: u32 = 0x0288_F780;
const STORE_64X64: u32 = 0x0200_F784;
const STORE_64X8: u32 = 0x0200_C784;
const LOAD_64X8: u32 = 0x0218_C780;
/// Weights may grow to 2^LIMIT before `ref` is raised (half floats end at 2^16).
const LIMIT: f32 = 11.0;

#[derive(Clone, Copy)]
pub struct Shape {
    /// Padded keys (a multiple of 16): the key chunks' stride per head.
    pub padded: u32,
    /// Padded queries (a multiple of 16) per head in Q, OUT and SUM; `padded` unless
    /// queries and keys differ (`with_queries`).
    pub queries: u32,
    /// Live keys.
    pub tokens: u32,
    /// Scores are multiplied by this before the softmax.
    pub scale: f32,
    /// Scaled scores are known to stay within about eight of zero (unit-length queries and
    /// keys), so `ref` starts at zero instead of at the first key's score. Weights then
    /// keep full half-float precision whichever key comes first.
    pub bounded: bool,
    /// Only channels 0..64 of every head are in use (the rest of the streams is zero and
    /// stays unread; the upper half of the output is not written).
    pub narrow: bool,
    /// Hardware threads per group (`Kernel::set_group([16 * group, 1, 1])`): consecutive query
    /// tiles of one head, which share an Xe core and so its cache.
    pub group: u32,
    /// Queries and keys as signed bytes with scales (see the module notes). Not with
    /// `narrow`.
    pub int8: bool,
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
            queries: padded as u32,
            tokens: tokens as u32,
            scale,
            bounded: false,
            narrow: false,
            group: 1,
            int8: false,
        })
    }
    /// `queries` padded queries per head, attending to the same keys. None unless a
    /// multiple of 16 that keeps the offsets within 32 bits.
    pub fn with_queries(self, queries: usize, heads: usize) -> Option<Shape> {
        let fits = queries > 0
            && queries.is_multiple_of(TILE)
            && heads * (queries / TILE) * OUT_BYTES < 1 << 32;
        fits.then_some(Shape {
            queries: queries as u32,
            ..self
        })
    }
    /// Thread groups: query tiles, heads.
    pub fn groups(&self, heads: usize) -> [u32; 3] {
        [
            (self.queries / TILE as u32).div_ceil(self.group),
            heads as u32,
            1,
        ]
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
/// Byte offsets of int8 queries and keys inside one head's streams.
pub fn query8_at(query: usize, channel: usize) -> usize {
    (query / TILE) * TILE_BYTES
        + (channel / 32) * 512
        + (channel % 32 / 4) * 64
        + (query % TILE) * 4
        + channel % 4
}
pub fn key8_at(key: usize, channel: usize) -> usize {
    (key / TILE) * CHUNK_BYTES + (channel / 32) * 512 + (key % TILE) * 32 + channel % 32
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
/// Running maxima of the exponents while they are computed (MAX, then registers that are
/// free until the weights and the slow path).
const PARTS: [u32; 4] = [MAX, ROUND, POWER, STEP];
/// int8 only: running sums of the weights (STEP, then registers free once the maxima are
/// used).
const SUMS: [u32; 4] = [STEP, ROUND, POWER, MAX];
/// int8 only: the chunk's key scales, and the queries' scales times `c`.
const KS: u32 = 48;
const COMB: u32 = 49;
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
/// fifth and sixth while values are mixed. int8 queries leave room for three more (6..8),
/// which hold the next chunk's first values from the moment they are mixed.
fn slot(i: u32) -> (u32, u16) {
    match i {
        0..=2 => (232 + i * 8, i as u16),
        3 => (P, 21),
        4 | 5 => (s(i - 4), 18 + i as u16),
        _ => (50 + (i - 6) * 8, 20 + i as u16),
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
const KS_TOKEN: u16 = 25;
// Lanes of INT: 0 counts steps, 1..3 are scratch.
const CHUNK: u32 = 4;
const START: u32 = 5;
const TILE_ID: u32 = 6;
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
fn step(k: &mut Asm, live: u32, bounded: bool, narrow: bool, int8: bool) {
    // Channel chunks of 16 mixed in pairs of blocks, and of 16 (32 as bytes) scored.
    let chunks = if narrow { 4 } else { 8 };
    let scored = if int8 { 4 } else { chunks };
    let (sum, operand) = if int8 { (D, B) } else { (F, Hf) };
    // Scaled scores: c per query (int8: times the query's scale) as a vector or scalar.
    let c = if int8 {
        r(COMB, 0, F)
    } else {
        r(CONST, C, F).s()
    };
    let w = Swsb::after(1);
    let (exp, init, slow, weights) = (k.label(), k.label(), k.label(), k.label());
    let row = |key: u32| s(key / 8) + key % 8;
    let keys: Vec<u32> = (0..live).collect();

    // Scores: eight channel chunks, each against both key groups. The weights' registers
    // are a load buffer once the previous step's mixing has read them.
    k.sync(w);
    k.sync(Swsb::src(mix(2 * chunks - 1)));
    fetch(k, 3, 3 * 512);
    if int8 {
        k.send(
            SFID_UGM,
            r(KS, 0, Ud),
            r(KV_AT, 0, Ud),
            null(Ud),
            0,
            (SCALES as u32) << 12,
            LOAD_64X8,
            false,
            Swsb::after_set(1, KS_TOKEN),
        );
    }
    for j in 0..scored {
        let (buffer, token) = slot(j % 4);
        k.sync(Swsb::dst(token));
        for h in 0..2 {
            let acc = if j == 0 {
                null(sum)
            } else {
                k.sync(Swsb::dst(score(h)));
                r(s(h), 0, sum)
            };
            if skip(2) {
                continue;
            }
            k.dpas(
                8,
                8,
                16,
                r(s(h), 0, sum),
                acc,
                r(q(j), 0, operand),
                r(buffer + 4 * h, 0, operand),
                Swsb::set(score(h)),
            );
        }
        // Refill the buffer once both products have read it: a later key chunk, or the
        // first values. The third buffer and the weights' registers are needed next.
        k.sync(Swsb::src(score(1)));
        match (narrow, j) {
            (_, 0 | 1) if int8 => fetch(k, j, VALUES as u32 + (j + 3) * 512),
            _ if int8 => {}
            (false, 0..=3) => fetch(k, j, (j + 4) * 512),
            (false, 4 | 5) => fetch(k, j - 4, VALUES as u32 + (j - 4) * 512),
            (true, 0 | 1) => fetch(k, j, VALUES as u32 + j * 512),
            _ => {}
        }
    }
    k.sync(Swsb::dst(score(0)));
    k.sync(Swsb::dst(score(1)));
    if int8 {
        // Integer products to floats, times each key's scale.
        k.sync(Swsb::dst(KS_TOKEN));
        for group in keys.chunks(4) {
            let back = Swsb::after(group.len() as u16);
            for (i, &key) in group.iter().enumerate() {
                let wait = if i == 0 { w } else { Swsb::NONE };
                k.mov(16, r(row(key), 0, F), r(row(key), 0, D), wait);
            }
            for &key in group {
                k.mul(
                    16,
                    r(row(key), 0, F),
                    r(row(key), 0, F),
                    r(KS, key, F).s(),
                    back,
                );
            }
        }
    }
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
                c,
                first(i),
            );
        }
        // Four running maxima (key % 4), each waiting only for its own row's exponent.
        let back = Swsb::after(group.len() as u16);
        for (i, &key) in group.iter().enumerate() {
            let part = r(PARTS[i], 0, F);
            if key < 4 {
                k.mov(16, part, r(row(key), 0, F), back);
            } else {
                k.sel(16, Cond::Ge, part, part, r(row(key), 0, F), back);
            }
        }
    }
    // Maxima are exact, so combining them in any order gives the same largest value.
    for &part in &PARTS[1..keys.len().min(4)] {
        k.sel(16, Cond::Ge, r(MAX, 0, F), r(MAX, 0, F), r(part, 0, F), w);
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
        if int8 {
            // The math pipe's 2^t (its own pipe: the group after waits for it as a whole).
            each(k, &|k, i, key| {
                k.math(16, Math::Exp, pp(i), t(key), first(i))
            });
        } else {
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
        }
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
            if int8 {
                // Four running sums (key % 4); the first waits for the conversions above.
                let part = r(SUMS[i], 0, F);
                if key < 4 {
                    k.mov(16, part, pp(i), first(i));
                } else {
                    k.add(16, part, part, pp(i), first(i));
                }
            } else if key == 0 {
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
    if int8 {
        for &part in &SUMS[1..weighted.len().clamp(1, 4)] {
            k.add(16, r(STEP, 0, F), r(STEP, 0, F), r(part, 0, F), w);
        }
    }

    // Mixing: eight pairs of channel blocks. The first two were requested during scoring;
    // the scratch buffer and the scores' registers take the next three.
    k.add(16, r(SUM, 0, F), r(SUM, 0, F), r(STEP, 0, F), w);
    k.sync(w);
    // int8: values 0..2 wait in buffers 6..8 and 3, 4 arrived during scoring; 5..7 go
    // where the weights' scratch and the scores were.
    for m in 2..chunks.min(5) {
        let at = if int8 { m + 3 } else { m };
        fetch(k, [2, 4, 5][m as usize - 2], VALUES as u32 + at * 512);
    }
    let order = if int8 {
        [6, 7, 8, 0, 1, 2, 4, 5]
    } else {
        [0, 1, 2, 4, 5, 0, 1, 2]
    };
    for m in 0..chunks {
        let (buffer, token) = slot(order[m as usize]);
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
        let next = CHUNK_BYTES as u32;
        if int8 {
            // The next chunk's first values, then its first keys.
            match m {
                0..=2 => fetch(k, 6 + m, next + VALUES as u32 + m * 512),
                3..=5 => fetch(k, m - 3, next + (m - 3) * 512),
                _ => {}
            }
            continue;
        }
        match (narrow, m) {
            (false, 0..=2) => fetch(k, m, VALUES as u32 + (m + 5) * 512),
            (false, 5..=7) => fetch(k, m - 5, CHUNK_BYTES as u32 + (m - 5) * 512),
            (true, 1..=3) => fetch(k, m - 1, CHUNK_BYTES as u32 + (m - 1) * 512),
            _ => {}
        }
    }
    for i in if int8 {
        &[0, 1, 2, 6, 7, 8][..]
    } else {
        &[0, 1, 2][..]
    } {
        k.sync(Swsb::src(slot(*i).1));
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

    // First step: every query's reference is its score against the first key, or zero.
    k.place(init);
    if bounded {
        k.mov(16, r(REF, 0, F), immf(0.0), w);
    } else {
        k.mul(16, r(REF, 0, F), r(row(0), 0, F), c, w);
    }
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
        queries,
        tokens,
        scale,
        bounded,
        narrow,
        group,
        int8,
    } = *shape;
    assert!(!(int8 && narrow), "int8 attention needs full heads");
    let (tiles, query_tiles) = (padded / TILE as u32, queries / TILE as u32);
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
    // The query tile: group x times the group's threads plus the thread's index in its group
    // (byte 8 of the header); threads past the last tile leave at once.
    let exit = k.label();
    if group > 1 {
        k.and(1, int(1), r(0, 2, D), imm(0xff), Swsb::NONE);
        k.mul(1, int(TILE_ID), r(0, 1, D), imm(group as i32), Swsb::NONE);
        k.add(1, int(TILE_ID), int(TILE_ID), int(1), w);
        k.cmp(Cond::Ge, int(TILE_ID), imm(query_tiles as i32), w);
        k.jmpi(exit, true, w);
    } else {
        k.mov(1, int(TILE_ID), r(0, 1, D), Swsb::NONE);
    }
    // Addresses of this thread's streams: per head, plus per query tile.
    let sizes = [TILE_BYTES, CHUNK_BYTES, OUT_BYTES, SUM_BYTES];
    for (i, size) in sizes.into_iter().enumerate() {
        let i = i as u32;
        let per_head = if i == 1 { tiles } else { query_tiles };
        k.mov(1, int(1), imm((per_head * size as u32) as i32), Swsb::NONE);
        k.mul(1, int(2), int(1), r(0, 12, Uw), w);
        if i != 1 {
            k.shl(
                1,
                int(3),
                int(TILE_ID),
                immw(size.trailing_zeros() as i16),
                w,
            );
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
        k.cmp(Cond::Ge, int(TILE_ID), imm(full as i32), w);
        k.jmpi(first, true, w);
        k.mov(1, int(START), int(TILE_ID), w);
        k.shl(
            1,
            int(2),
            int(TILE_ID),
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
        let (to, offset, descriptor) = match j {
            _ if !int8 || j < 4 => (q(j), j * 512, LOAD_64X64),
            4 => (COMB, SCALES as u32, LOAD_64X8),
            _ => (slot(0).0, 0, LOAD_64X64),
        };
        k.send(
            SFID_UGM,
            r(to, 0, Ud),
            r(2, 0, Ud),
            null(Ud),
            0,
            offset << 12,
            descriptor,
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
    if int8 {
        k.mul(16, r(COMB, 0, F), r(COMB, 0, F), r(CONST, C, F).s(), w);
    }
    let primed: &[u32] = if int8 {
        &[0, 1, 2, 6, 7, 8]
    } else {
        &[0, 1, 2]
    };
    let prime = |k: &mut Asm| {
        for &j in primed {
            let offset = if j < 3 {
                j * 512
            } else {
                VALUES as u32 + (j - 6) * 512
            };
            fetch(k, j, offset);
        }
    };
    prime(&mut k);

    // Chunks from the start to the last, then from the first up to the start.
    if full > 0 {
        let top = k.label();
        k.place(top);
        step(&mut k, TILE as u32, bounded, narrow, int8);
        k.add(1, int(CHUNK), int(CHUNK), imm(1), w);
        k.cmp(Cond::Lt, int(CHUNK), imm(full as i32), w);
        k.jmpi(top, true, w);
    }
    if rest > 0 {
        step(&mut k, rest, bounded, narrow, int8);
    }
    if full > 0 {
        let (top, end) = (k.label(), k.label());
        k.cmp(Cond::Eq, int(START), imm(0), w);
        k.jmpi(end, true, w);
        // Back to the first chunk; the loads requested past the last one are replaced.
        let walked = (full + u32::from(rest > 0)) * CHUNK_BYTES as u32;
        k.add(1, r(KV_AT, 0, Q), r(KV_AT, 0, Q), imm(-(walked as i32)), w);
        k.mov(1, int(CHUNK), imm(0), w);
        for &j in primed {
            k.sync(Swsb::dst(slot(j).1));
        }
        k.sync(w);
        prime(&mut k);
        k.place(top);
        step(&mut k, TILE as u32, bounded, narrow, int8);
        k.add(1, int(CHUNK), int(CHUNK), imm(1), w);
        k.cmp(Cond::Lt, int(CHUNK), int(START), w);
        k.jmpi(top, true, w);
        k.place(end);
    }

    // Store. Accumulator registers are in memory order: channel, then query.
    for token in (0..=LAST_MIX).chain(if int8 { 26..29 } else { 0..0 }) {
        k.sync(Swsb::dst(token));
    }
    k.sync(w);
    for b in 0..if narrow { 8 } else { 16 } {
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
    k.place(exit);
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

    /// int8 queries and keys (per-token scales, as the Qwen glue makes them) against the
    /// half-float kernel and an exact softmax, both timed, on Qwen-like shapes.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn int8_attention_against_half() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let scale = 1.0 / (CHANNELS as f32).sqrt();
        for (heads, padded, tokens, spread, checks) in [
            (3usize, 256usize, 100usize, 1.0f32, 768usize),
            (2, 256, 250, 6.0, 512),
            (24, 8224, 8214, 1.5, 48),
            (24, 8224, 8214, 3.0, 48),
        ] {
            let half = |values: Vec<f32>| -> Vec<f32> {
                values.into_iter().map(|v| widen(narrow(v))).collect()
            };
            let queries = half(noise(heads * padded * CHANNELS, 1, spread));
            let keys = half(noise(heads * padded * CHANNELS, 2, spread));
            let values = half(noise(heads * padded * CHANNELS, 3, 1.0));
            let tiles = padded / TILE;
            let quantize = |row: &[f32]| -> (Vec<i8>, f32) {
                let top = row.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
                let s = top / 127.0;
                (
                    row.iter()
                        .map(|v| (v / s).round_ties_even() as i8)
                        .collect(),
                    s,
                )
            };
            let mut streams = Vec::new();
            for int8 in [false, true] {
                let mut q_stream = vec![0u8; heads * tiles * TILE_BYTES];
                let mut kv_stream = vec![0u8; heads * tiles * CHUNK_BYTES + SLACK];
                for head in 0..heads {
                    let (q0, kv0) = (head * tiles * TILE_BYTES, head * tiles * CHUNK_BYTES);
                    for token in 0..padded {
                        let from = (head * padded + token) * CHANNELS;
                        let (qr, kr) = (
                            &queries[from..from + CHANNELS],
                            &keys[from..from + CHANNELS],
                        );
                        for channel in 0..CHANNELS {
                            let v = narrow(values[from + channel]).to_le_bytes();
                            let at = kv0 + value_at(token, channel);
                            kv_stream[at..at + 2].copy_from_slice(&v);
                        }
                        if int8 {
                            let ((qi, qs), (ki, ks)) = (quantize(qr), quantize(kr));
                            for channel in 0..CHANNELS {
                                q_stream[q0 + query8_at(token, channel)] = qi[channel] as u8;
                                kv_stream[kv0 + key8_at(token, channel)] = ki[channel] as u8;
                            }
                            let at = q0 + (token / TILE) * TILE_BYTES + SCALES + token % TILE * 4;
                            q_stream[at..at + 4].copy_from_slice(&qs.to_le_bytes());
                            let at = kv0 + (token / TILE) * CHUNK_BYTES + SCALES + token % TILE * 4;
                            kv_stream[at..at + 4].copy_from_slice(&ks.to_le_bytes());
                        } else {
                            for channel in 0..CHANNELS {
                                let at = q0 + query_at(token, channel);
                                q_stream[at..at + 2]
                                    .copy_from_slice(&narrow(qr[channel]).to_le_bytes());
                                let at = kv0 + key_at(token, channel);
                                kv_stream[at..at + 2]
                                    .copy_from_slice(&narrow(kr[channel]).to_le_bytes());
                            }
                        }
                    }
                }
                streams.push((q_stream, kv_stream));
            }
            let (out_size, sum_size) = (heads * tiles * OUT_BYTES, heads * tiles * SUM_BYTES);
            for (int8, (q_stream, kv_stream)) in [false, true].into_iter().zip(&streams) {
                let buffers = [
                    gpu.shared(q_stream.len() as u64).unwrap(),
                    gpu.shared(kv_stream.len() as u64).unwrap(),
                    gpu.shared(out_size as u64).unwrap(),
                    gpu.shared(sum_size as u64).unwrap(),
                ];
                gpu.upload(&buffers[0], 0, q_stream).unwrap();
                gpu.upload(&buffers[1], 0, kv_stream).unwrap();
                let sizes = [q_stream.len(), kv_stream.len(), out_size, sum_size];
                let memory: Vec<_> = buffers
                    .iter()
                    .zip(sizes)
                    .map(|(b, size)| fast.import(&gpu.export(b).unwrap(), size as u64).unwrap())
                    .collect();
                let group = if heads > 8 { 8 } else { 1 };
                let shape = Shape {
                    int8,
                    group,
                    ..Shape::new(padded, tokens, heads, scale).unwrap()
                };
                let code = kernel(&shape);
                let native = zebin::write(&zebin::Kernel {
                    name: "attend",
                    code: &code,
                    grf: 256,
                    pointers: 4,
                });
                let attend = fast.kernel(&native, "attend", 4).unwrap();
                attend.set_group([16 * group, 1, 1]).unwrap();
                let arguments: Vec<_> = memory.iter().collect();
                let mut best = f64::MAX;
                for _ in 0..6 {
                    let started = std::time::Instant::now();
                    fast.launch(&attend, &arguments, shape.groups(heads))
                        .unwrap();
                    best = best.min(started.elapsed().as_secs_f64() * 1e3);
                }
                let mixed = floats(&gpu.download(&buffers[2], 0, out_size).unwrap());
                let sums = floats(&gpu.download(&buffers[3], 0, sum_size).unwrap());
                let (mut error, mut power) = (0f64, 0f64);
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
                            q.iter().zip(k).map(|(a, b)| f64::from(a * b)).sum::<f64>()
                                * f64::from(scale)
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
                        let got = f64::from(
                            mixed[(head * tiles * OUT_BYTES + out_at(query, channel)) / 4],
                        ) / sum;
                        error += (got - want).powi(2);
                        power += want.powi(2);
                    }
                }
                let db = 10.0 * (power / error.max(1e-30)).log10();
                let digest = mixed.iter().fold(0u64, |h, v| {
                    (h ^ u64::from(v.to_bits())).wrapping_mul(0x100_0000_01b3)
                });
                println!(
                    "{} {heads} heads, {tokens} of {padded}, spread {spread}: {best:.2} ms, {db:.1} dB, output {digest:016x}",
                    if int8 { "int8" } else { "half" }
                );
                assert!(db > if int8 { 25.0 } else { 50.0 });
            }
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
            (32, 8240, 8239, 1.5, 48),
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
            let reference = mixed.clone();
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
                for group in [2u32, 4, 8, 16] {
                    let shape = Shape { group, ..shape };
                    let code = kernel(&shape);
                    let native = zebin::write(&zebin::Kernel {
                        name: "attend",
                        code: &code,
                        grf: 256,
                        pointers: 4,
                    });
                    let bare = fast.kernel(&native, "attend", 4).unwrap();
                    bare.set_group([16 * group, 1, 1]).unwrap();
                    let mut best = f64::MAX;
                    for _ in 0..4 {
                        let started = std::time::Instant::now();
                        fast.launch(&bare, &arguments, shape.groups(heads)).unwrap();
                        best = best.min(started.elapsed().as_secs_f64() * 1e3);
                    }
                    let mixed = floats(&gpu.download(&buffers[2], 0, out_size).unwrap());
                    let same = mixed == reference;
                    println!("  {group} threads per group: {best:.2} ms, same output {same}");
                }
                // Timing only: without the weights (1), the loads (2), the score products
                // (4), the mixing products (8).
                for mask in [4u32, 8, 1, 5, 3, 13, 15] {
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
