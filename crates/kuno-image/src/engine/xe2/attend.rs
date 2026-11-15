//! Attention for Xe2 as one machine-code kernel: scores, softmax weights and mixing in a
//! single pass over the keys, with nothing stored in between.
//!
//! One hardware thread owns 16 queries of one head (128 channels) and walks the keys 16 at a
//! time. Per step it
//!   1. multiplies its queries with the 16 keys (half-float `dpas`, f32 scores in registers),
//!   2. turns the scores into half-float weights `2^((score - ref) * c)`,
//!   3. multiplies the weights with the 16 values into f32 accumulators, one of which sums
//!      the weights themselves.
//!
//! The caller divides the mixed values by the weight sum, so `ref` only has to keep weights
//! in half-float range. Each query starts from its first key's score; when a later score
//! exceeds `ref` by more than 2^11, a slow path raises `ref` and rescales the accumulators.
//!
//! Operands are streams laid out so that every load is one contiguous 256- or 512-byte run:
//!   Q   `[head][query tile][channel chunk 0..8][query group 0..2][8 queries x 16 halves]`
//!   KV  `[head][key chunk]` of `CHUNK_BYTES`: eight key fragments, then nine value fragments.
//!       A key fragment (512 B) is `[channel pair 0..8][key 0..16][2 halves]`; a value
//!       fragment is `[key pair 0..8][channel 0..16][2 halves]`. The ninth value fragment is
//!       1.0 in channel 0 for live keys, so its accumulator is the weight sum.
//!   OUT `[head][query][128]` f32, and SUM `[head][query][8]` f32 with the sum in slot 0.
//! Keys past the live count inside the last live chunk must repeat a live key and have zero
//! values (and zero in the ninth fragment); whole dead chunks are never read.
use super::{Ty::*, *};

/// Queries per thread and keys per step.
pub const TILE: usize = 16;
pub const CHANNELS: usize = 128;
/// Bytes of one query tile in the Q stream and of one key chunk in the KV stream.
pub const TILE_BYTES: usize = TILE * CHANNELS * 2;
pub const CHUNK_BYTES: usize = 17 * 512;
/// Offset of the value fragments inside a key chunk.
pub const VALUES: usize = 8 * 512;

const LOAD_32X64: u32 = 0x0248_F580;
const LOAD_64X64: u32 = 0x0288_F780;
const STORE_64X8: u32 = 0x0200_C784;
const STORE_32X8: u32 = 0x0200_C584;
/// Weights may grow to 2^LIMIT before `ref` is raised (half floats end at 2^16).
const LIMIT: f32 = 11.0;

#[derive(Clone, Copy)]
pub struct Shape {
    /// Padded tokens (a multiple of 16): queries per head, and the key chunks' stride.
    pub padded: u32,
    /// Key chunks holding live keys.
    pub chunks: u32,
    /// Scores are multiplied by this before the softmax.
    pub scale: f32,
}
impl Shape {
    /// None when the offsets leave 32 bits.
    pub fn new(padded: usize, tokens: usize, heads: usize, scale: f32) -> Option<Shape> {
        let fits = padded.is_multiple_of(TILE)
            && tokens > 0
            && tokens <= padded
            && heads * padded * (CHUNK_BYTES / TILE).max(CHANNELS * 4) < 1 << 32;
        fits.then(|| Shape {
            padded: padded as u32,
            chunks: tokens.div_ceil(TILE) as u32,
            scale,
        })
    }
    /// Thread groups: query tiles, heads.
    pub fn groups(&self, heads: usize) -> [u32; 3] {
        [self.padded / TILE as u32, heads as u32, 1]
    }
}

/// Byte offset in the Q stream of the 16 halves `channel / 16` of `query` (head-relative).
pub fn query_at(query: usize, channel: usize) -> usize {
    (((query / TILE) * 8 + channel / 16) * 2 + query % TILE / 8) * 256
        + (query % 8) * 32
        + (channel % 16) * 2
}
/// Byte offset in the KV stream of a key's channel (head-relative).
pub fn key_at(key: usize, channel: usize) -> usize {
    (key / TILE) * CHUNK_BYTES
        + (channel / 16) * 512
        + (channel % 16 / 2) * 64
        + (key % TILE) * 4
        + (channel % 2) * 2
}
/// Byte offset in the KV stream of a value's channel; channel 128 is the weight-sum marker.
pub fn value_at(key: usize, channel: usize) -> usize {
    (key / TILE) * CHUNK_BYTES
        + VALUES
        + (channel / 16) * 512
        + (key % TILE / 2) * 64
        + (channel % 16) * 4
        + (key % 2) * 2
}

// Register plan (256 registers).
const HEADER: u32 = 15;
const REF: u32 = 6;
const CONST: u32 = 7;
const INT: u32 = 8;
const MAX: u32 = 9;
const WIDE: u32 = 10;
const T: [u32; 4] = [248, 249, 250, 251];
const N: [u32; 4] = [252, 253, 254, 255];
const PP: [u32; 4] = [11, 12, 13, 14];
const KV: u32 = 248;
fn q(g: u32, j: u32) -> u32 {
    16 + (g * 8 + j) * 4
}
fn out(g: u32, j: u32) -> u32 {
    80 + (g * 9 + j) * 8
}
fn s(g: u32) -> u32 {
    224 + g * 8
}
fn p(g: u32) -> u32 {
    240 + g * 4
}
// Scoreboard tokens.
const LOAD: u16 = 0;
fn score(g: u32) -> u16 {
    1 + g as u16
}
fn mix(g: u32, j: u32) -> u16 {
    3 + (g * 9 + j) as u16
}
const LAST_MIX: u16 = 20;
// Lanes of CONST.
const C: u32 = 0;
const A1: u32 = 1;
const A2: u32 = 2;
const A3: u32 = 3;
const A4: u32 = 4;
const ONE: u32 = 5;

/// `2^f` for the fraction in `frac` (|f| <= 0.5) into `to`, both `exec` lanes wide:
/// a fourth-degree Taylor polynomial, 3e-5 relative error.
fn power(k: &mut Asm, exec: u32, to: Reg, frac: Reg, w: Swsb) {
    let c = |lane: u32| r(CONST, lane, F).s();
    k.mad(exec, to, c(A3), frac, c(A4), w);
    k.mad(exec, to, c(A2), to, frac, w);
    k.mad(exec, to, c(A1), to, frac, w);
    k.mad(exec, to, c(ONE), to, frac, w);
}

pub fn kernel(shape: &Shape) -> Vec<u8> {
    let Shape {
        padded,
        chunks,
        scale,
    } = *shape;
    let tiles = padded / TILE as u32;
    let (w, w4) = (Swsb::after(1), Swsb::after(4));
    let int = |lane: u32| r(INT, lane, D);
    let row = |n: u32| s(n / 8) + n % 8;
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
    // Byte offsets of this thread's streams: per head, plus per query tile.
    let per_head = [
        tiles * TILE_BYTES as u32,
        tiles * CHUNK_BYTES as u32,
        padded * CHANNELS as u32 * 4,
        padded * 32,
    ];
    let per_tile = [
        TILE_BYTES as u32,
        0,
        (TILE * CHANNELS * 4) as u32,
        TILE as u32 * 32,
    ];
    for (i, (head, tile)) in per_head.into_iter().zip(per_tile).enumerate() {
        let i = i as u32;
        k.mov(1, int(1), imm(head as i32), Swsb::NONE);
        k.mul(1, int(2), int(1), r(0, 12, Uw), w);
        if tile > 0 {
            k.shl(1, int(3), r(0, 1, D), immw(tile.trailing_zeros() as i16), w);
            k.add(1, int(2), int(2), int(3), w);
        }
        k.mov(1, r(WIDE, 0, Uq), r(INT, 2, Ud), w);
        k.add(1, r(2 + i, 0, Q), r(1, i, Q), r(WIDE, 0, Q), w);
    }
    for g in 0..2 {
        for j in 0..8 {
            k.send(
                SFID_UGM,
                r(q(g, j), 0, Ud),
                r(2, 0, Ud),
                null(Ud),
                0,
                ((j * 2 + g) * 256) << 12,
                LOAD_32X64,
                false,
                Swsb::after_set(1, mix(g, j)),
            );
        }
    }
    // Touch the remaining mixing tokens, so later waits on them always have a predecessor.
    for token in [mix(0, 8), mix(1, 8)] {
        k.send(
            SFID_UGM,
            r(KV, 0, Ud),
            r(3, 0, Ud),
            null(Ud),
            0,
            0,
            LOAD_64X64,
            false,
            Swsb::after_set(1, token),
        );
    }
    for g in 0..2 {
        for reg in out(g, 0)..out(g, 9) {
            k.mov(16, r(reg, 0, D), imm(0), Swsb::NONE);
        }
    }
    k.mov(1, int(0), imm(0), Swsb::NONE);
    for token in mix(0, 0)..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }

    let (top, exp, init, slow) = (k.label(), k.label(), k.label(), k.label());
    // Scores of 16 queries against the chunk's 16 keys, 16 channels per fragment.
    k.place(top);
    for j in 0..8 {
        // The fragment buffer is free once its last reader has read it.
        k.sync(Swsb::src(if j == 0 { LAST_MIX } else { score(1) }));
        k.send(
            SFID_UGM,
            r(KV, 0, Ud),
            r(3, 0, Ud),
            null(Ud),
            0,
            (j * 512) << 12,
            LOAD_64X64,
            false,
            Swsb::after_set(1, LOAD),
        );
        k.sync(Swsb::dst(LOAD));
        for g in 0..2 {
            let acc = if j == 0 {
                null(F)
            } else {
                k.sync(Swsb::dst(score(g)));
                r(s(g), 0, F)
            };
            k.dpas(
                8,
                8,
                16,
                r(s(g), 0, F),
                acc,
                r(KV, 0, Hf),
                r(q(g, j), 0, Hf),
                Swsb::set(score(g)),
            );
        }
    }
    k.sync(Swsb::dst(score(0)));
    k.sync(Swsb::dst(score(1)));
    k.cmp(Cond::Eq, int(0), imm(0), w);
    k.jmpi(init, true, w);

    // Weights: t = score * c - ref, 2^t = 2^round(t) * 2^fraction, as half floats. Four
    // rows at a time, so each instruction depends on the one four back. Distances count
    // instructions of the producer's pipe (integer and float instructions run in separate
    // pipes), so where a group reads what the other pipe's group wrote, row i's producer is
    // 4 - i back.
    let across = |i: usize| Swsb::after(4 - i as u16);
    k.place(exp);
    k.sync(w);
    for quad in 0..4 {
        let rows: Vec<u32> = (0..4).map(|i| quad * 4 + i).collect();
        let each = |k: &mut Asm, f: &dyn Fn(&mut Asm, usize, u32)| {
            for (i, n) in rows.iter().enumerate() {
                f(k, i, *n);
            }
        };
        let (t, n, pp) = (
            |i: usize| r(T[i], 0, F),
            |i: usize| r(N[i], 0, F),
            |i: usize| r(PP[i], 0, F),
        );
        each(&mut k, &|k, i, n| {
            k.mad(
                16,
                t(i),
                r(REF, n, F).s().minus(),
                r(row(n), 0, F),
                r(CONST, C, F).s(),
                across(i),
            )
        });
        each(&mut k, &|k, i, n| {
            if n == 0 {
                k.mov(16, r(MAX, 0, F), t(i), w);
            } else {
                k.sel(16, Cond::Ge, r(MAX, 0, F), r(MAX, 0, F), t(i), w);
            }
        });
        each(&mut k, &|k, i, _| {
            k.sel(16, Cond::Ge, t(i), t(i), immf(-126.0), w4)
        });
        each(&mut k, &|k, i, _| k.rnde(16, n(i), t(i), w4));
        each(&mut k, &|k, i, _| k.add(16, t(i), t(i), n(i).minus(), w4));
        for step in 0..4 {
            each(&mut k, &|k, i, _| {
                let c = |lane: u32| r(CONST, lane, F).s();
                match step {
                    0 => k.mad(16, pp(i), c(A3), t(i), c(A4), w4),
                    1 => k.mad(16, pp(i), c(A2), pp(i), t(i), w4),
                    2 => k.mad(16, pp(i), c(A1), pp(i), t(i), w4),
                    _ => k.mad(16, pp(i), c(ONE), pp(i), t(i), w4),
                }
            });
        }
        each(&mut k, &|k, i, _| k.mov(16, r(N[i], 0, D), n(i), w4));
        each(&mut k, &|k, i, _| {
            k.shl(16, r(N[i], 0, D), r(N[i], 0, D), immw(23), across(i))
        });
        each(&mut k, &|k, i, _| {
            k.add(16, r(PP[i], 0, D), r(PP[i], 0, D), r(N[i], 0, D), w4)
        });
        each(&mut k, &|k, i, _| {
            k.mov(16, r(T[i], 0, Hf).step2(), pp(i), across(i))
        });
        each(&mut k, &|k, i, n| {
            k.mov(
                16,
                r(p(n / 8) + n % 8 / 2, (n % 2) * 16, Uw),
                r(T[i], 0, Uw).step2(),
                across(i),
            )
        });
    }
    // Largest t of the step: non-negative floats order like their bits, so the reduction
    // across lanes can run on integers.
    k.sel(16, Cond::Ge, r(MAX, 0, F), r(MAX, 0, F), immf(0.0), w);
    for half in [8, 4, 2, 1] {
        k.sel(
            half,
            Cond::Ge,
            r(MAX, 0, D),
            r(MAX, 0, D),
            r(MAX, half, D),
            w,
        );
    }
    k.cmp(Cond::Gt, r(MAX, 0, D), imm(LIMIT.to_bits() as i32), w);
    k.jmpi(slow, true, w);

    // Mixing: nine value fragments against the weights.
    k.sync(w);
    for j in 0..9 {
        if j > 0 {
            k.sync(Swsb::src(mix(1, j - 1)));
        }
        k.send(
            SFID_UGM,
            r(KV, 0, Ud),
            r(3, 0, Ud),
            null(Ud),
            0,
            (VALUES as u32 + j * 512) << 12,
            LOAD_64X64,
            false,
            Swsb::after_set(1, LOAD),
        );
        k.sync(Swsb::dst(LOAD));
        for g in 0..2 {
            k.sync(Swsb::dst(mix(g, j)));
            k.dpas(
                8,
                8,
                16,
                r(out(g, j), 0, F),
                r(out(g, j), 0, F),
                r(KV, 0, Hf),
                r(p(g), 0, Hf),
                Swsb::set(mix(g, j)),
            );
        }
    }
    k.add(1, r(3, 0, Q), r(3, 0, Q), imm(CHUNK_BYTES as i32), w);
    k.add(1, int(0), int(0), imm(1), w);
    k.cmp(Cond::Lt, int(0), imm(chunks as i32), w);
    k.jmpi(top, true, w);

    // Store: one address register per query (the score and weight registers are free now).
    for token in mix(0, 0)..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }
    let (to_out, to_sum) = (|n: u32| 224 + n, |n: u32| 240 + n);
    for (first, pointer, step) in [(224, 4, CHANNELS as i32 * 4), (240, 5, 32)] {
        k.mov(1, r(first, 0, Q), r(pointer, 0, Q), w);
        for n in 1..16 {
            k.add(1, r(first + n, 0, Q), r(first + n - 1, 0, Q), imm(step), w);
        }
    }
    let mut stores = 0u32;
    let mut store = |k: &mut Asm, address: u32, data: u32, offset: u32, desc: u32| {
        let id = (stores % 32) as u16;
        if stores >= 32 {
            k.sync(Swsb::dst(id));
        }
        k.send(
            SFID_UGM,
            null(Ud),
            r(address, 0, Ud),
            r(data, 0, Ud),
            1,
            offset << 12,
            desc,
            false,
            Swsb::after_set(1, id),
        );
        stores += 1;
    };
    for n in 0..16 {
        for j in 0..8 {
            store(&mut k, to_out(n), out(n / 8, j) + n % 8, j * 64, STORE_64X8);
        }
        store(&mut k, to_sum(n), out(n / 8, 8) + n % 8, 0, STORE_32X8);
    }
    for id in 0..32 {
        k.sync(Swsb::dst(id));
    }
    k.end(r(HEADER, 0, F));

    // First step: every query's reference is its first key's score.
    k.place(init);
    for n in 0..16 {
        k.mul(1, r(REF, n, F), r(row(n), 0, F), r(CONST, C, F), w);
    }
    k.jmpi(exp, false, w);

    // A weight left half-float range: raise each query's reference to its largest score of
    // this step (if higher) and scale what it has accumulated by 2^-(the raise).
    k.place(slow);
    for token in mix(0, 0)..=LAST_MIX {
        k.sync(Swsb::dst(token));
    }
    let scalar = |lane: u32| r(MAX, lane, F);
    for n in 0..16 {
        k.mad(
            16,
            r(WIDE, 0, F),
            r(REF, n, F).s().minus(),
            r(row(n), 0, F),
            r(CONST, C, F).s(),
            w,
        );
        k.sel(16, Cond::Ge, r(WIDE, 0, F), r(WIDE, 0, F), immf(0.0), w);
        for half in [8, 4, 2, 1] {
            k.sel(
                half,
                Cond::Ge,
                r(WIDE, 0, D),
                r(WIDE, 0, D),
                r(WIDE, half, D),
                w,
            );
        }
        k.add(1, r(REF, n, F), r(REF, n, F), r(WIDE, 0, F), w);
        k.sel(
            1,
            Cond::Ge,
            scalar(0),
            r(WIDE, 0, F).minus(),
            immf(-126.0),
            w,
        );
        k.rnde(1, scalar(1), scalar(0), w);
        k.add(1, scalar(0), scalar(0), scalar(1).minus(), w);
        power(&mut k, 1, scalar(2), scalar(0), w);
        k.mov(1, r(MAX, 1, D), scalar(1), w);
        k.shl(1, r(MAX, 1, D), r(MAX, 1, D), immw(23), w);
        k.add(1, r(MAX, 2, D), r(MAX, 2, D), r(MAX, 1, D), w);
        for j in 0..9 {
            let acc = r(out(n / 8, j) + n % 8, 0, F);
            k.mul(16, acc, acc, scalar(2).s(), w);
        }
    }
    k.jmpi(exp, false, w);
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
            let mut kv_stream = vec![0u16; heads * tiles * CHUNK_BYTES / 2];
            let one = narrow(1.0);
            for head in 0..heads {
                let (q0, kv0) = (head * tiles * TILE_BYTES, head * tiles * CHUNK_BYTES);
                for token in 0..padded {
                    let from = (head * padded + token) * CHANNELS;
                    // Dead keys repeat the last live key and carry no value.
                    let key_from = (head * padded + token.min(tokens - 1)) * CHANNELS;
                    let live = token < tokens;
                    for channel in 0..CHANNELS {
                        q_stream[(q0 + query_at(token, channel)) / 2] =
                            narrow(queries[from + channel]);
                        kv_stream[(kv0 + key_at(token, channel)) / 2] =
                            narrow(keys[key_from + channel]);
                        if live {
                            kv_stream[(kv0 + value_at(token, channel)) / 2] =
                                narrow(values[from + channel]);
                        }
                    }
                    if live {
                        kv_stream[(kv0 + value_at(token, CHANNELS)) / 2] = one;
                    }
                }
            }
            let (out_size, sum_size) = (heads * padded * CHANNELS * 4, heads * padded * 32);
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
                let row = head * padded + query;
                let sum: f64 = sums[row * 8..row * 8 + 8]
                    .iter()
                    .map(|v| f64::from(*v))
                    .sum();
                for channel in 0..CHANNELS {
                    let want: f64 = (0..tokens)
                        .map(|key| {
                            weights[key]
                                * f64::from(values[(head * padded + key) * CHANNELS + channel])
                        })
                        .sum::<f64>()
                        / total;
                    let got = f64::from(mixed[row * CHANNELS + channel]) / sum;
                    error += (got - want).powi(2);
                    power += want.powi(2);
                    worst = worst.max((got - want).abs());
                }
            }
            let mut shown = 0;
            for row in 0..heads * padded {
                let sum: f32 = sums[row * 8..row * 8 + 8].iter().sum();
                let bad = !sum.is_finite() || sum <= 0.0 || !mixed[row * CHANNELS].is_finite();
                if bad && shown < 12 {
                    shown += 1;
                    println!(
                        "  head {} query {}: sums {:?} mixed {:?}",
                        row / padded,
                        row % padded,
                        &sums[row * 8..row * 8 + 2],
                        &mixed[row * CHANNELS..row * CHANNELS + 3]
                    );
                }
            }
            let db = 10.0 * (power / error.max(1e-30)).log10();
            println!(
                "{heads} heads, {tokens} of {padded} tokens, spread {spread}: {best:.2} ms, {db:.1} dB, worst {worst:.2e}, {} instructions",
                code.len() / 16
            );
            assert!(db > 50.0);
        }
    }
}
