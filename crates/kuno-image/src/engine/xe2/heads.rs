//! The video decoder's query/key/value split as a kernel, feeding the attention kernel
//! (`attend`) straight from the projection's sums.
//!
//! One hardware thread owns 16 tokens of one head, one token per lane. Gather loads (one
//! address per lane) bring in four channels at a time, so every register holds one channel
//! of the 16 tokens, which is the layout the attention kernel's queries and values have.
//! Per side (query, key, value) the thread adds the bias; queries and keys are then
//! brought to unit RMS over their 64 channels and turned by the rotary table (pair `i` is
//! channels `i` and `i + 24`; the last 16 channels are not rotated). Everything is written
//! as half floats: queries and values as whole registers, keys through scatter stores of
//! four channels per token, because the kernel wants each key's channels side by side.
//!
//! The attention kernel's heads are 128 channels wide; channels 64 and up are never
//! written and stay zero.
//!
//! Arguments: the sums `[token][head][query 64, key 64, value 64]` f32, the layer's vectors
//! (f32, bias offset in values), the query stream, and the key/value stream, which also
//! holds the rotary table `ROTARY_BYTES` per token tile at `rotary_first` (see
//! `rotary_table`). Padded tokens are computed like any other.
use super::{Ty::*, attend, *};

pub const CHANNELS: u32 = 64;
const PAIRS: u32 = 24;
/// Bytes of one token's sums, and of one head's part of them.
const ROW_BYTES: u32 = 3 * 2048 * 4;
const HEAD_BYTES: u32 = 3 * CHANNELS * 4;
/// The rotary table of one token tile: 24 cosines then 24 sines, each for 16 tokens.
pub const ROTARY_BYTES: usize = 2 * PAIRS as usize * 64;

const GATHER_32X4: u32 = 0x0440_3580; // four dwords at each lane's address
const SCATTER_64: u32 = 0x0400_0784; // each 64-bit lane to its address
const LOAD_64X64: u32 = 0x0288_F780;
const LOAD_32X64: u32 = 0x0248_F580;
const STORE_64X64: u32 = 0x0200_F784;

#[derive(Clone, Copy)]
pub struct Shape {
    /// Padded tokens (a multiple of 16).
    pub padded: u32,
    /// First value of the projection's bias in the vectors.
    pub bias_first: u32,
    /// Byte offset of the rotary table in the key/value buffer.
    pub rotary_first: u32,
}
impl Shape {
    /// Thread groups: token tiles, heads.
    pub fn groups(&self, heads: usize) -> [u32; 3] {
        [self.padded / 16, heads as u32, 1]
    }
}

/// The table the kernel reads: per tile of 16 tokens, each pair's cosines for the 16 tokens,
/// then each pair's sines. `turns` is `[token][pair]` (cosine, sine); tokens past its end do
/// not turn.
pub fn rotary_table(turns: &[[f32; 2]], padded: usize) -> Vec<f32> {
    let pairs = PAIRS as usize;
    let mut table = Vec::with_capacity(padded * 2 * pairs);
    for tile in 0..padded / 16 {
        for part in 0..2 {
            for pair in 0..pairs {
                for lane in 0..16 {
                    let turn = turns.get((tile * 16 + lane) * pairs + pair);
                    table.push(turn.map_or([1.0, 0.0][part], |turn| turn[part]));
                }
            }
        }
    }
    table
}

// Register plan (256 registers).
const Q_AT: u32 = 2;
const KV_AT: u32 = 3;
const ROWS: u32 = 4; // and 5: each lane's row of sums
const TABLE_AT: u32 = 6;
const BIAS_AT: u32 = 7;
const KEYS: u32 = 8; // and 9: each lane's key in the stream
const INT: u32 = 10;
const WIDE: u32 = 11;
const FIRST: u32 = 12;
const SCALE: u32 = 13;
const SCRATCH: u32 = 14;
const HEADER: u32 = 240;
fn x(c: u32) -> u32 {
    16 + c
}
const COS: u32 = 80;
const SIN: u32 = 104;
const BIAS: u32 = 128;
const T1: u32 = 140;
const T2: u32 = 144;
const OUT: u32 = 148;
// Scoreboard tokens.
const TABLES: u16 = 4;
const STORE: u16 = 12;
const SCATTER: u16 = 16;

pub fn kernel(shape: &Shape) -> Vec<u8> {
    let tiles = shape.padded / 16;
    let w = Swsb::after(1);
    let first = |i: u32| if i == 0 { w } else { Swsb::NONE };
    let back = Swsb::after(4);
    let int = |lane: u32| r(INT, lane, D);
    let (tile, head) = (r(0, 2, Uw), r(0, 12, Uw));
    let mut k = Asm::new();

    // r0: thread header (tile in dword 1, head in dword 6); r1: the four pointers.
    k.mov(16, r(HEADER, 0, F), r(0, 0, F), Swsb::NONE);
    k.keep_subnormals();
    k.mov(1, int(0), imm((16 * ROW_BYTES) as i32), Swsb::NONE);
    k.mul(1, int(0), int(0), tile, w);
    k.mov(1, int(1), imm(HEAD_BYTES as i32), Swsb::NONE);
    k.mul(1, int(1), int(1), head, w);
    k.add(1, int(0), int(0), int(1), w); // the first token's sums for this head
    k.add(1, int(1), int(1), imm((shape.bias_first * 4) as i32), w); // the head's bias
    k.mov(1, int(2), imm(tiles as i32), Swsb::NONE);
    k.mul(1, int(2), int(2), head, w);
    k.add(1, int(2), int(2), r(0, 1, D), w);
    k.shl(
        1,
        int(3),
        int(2),
        immw(attend::CHUNK_BYTES.trailing_zeros() as i16),
        w,
    );
    k.shl(
        1,
        int(2),
        int(2),
        immw(attend::TILE_BYTES.trailing_zeros() as i16),
        w,
    );
    k.mov(1, int(4), imm(ROTARY_BYTES as i32), Swsb::NONE);
    k.mul(1, int(4), int(4), tile, w);
    k.add(1, int(4), int(4), imm(shape.rotary_first as i32), w);
    for i in 0..5 {
        k.mov(1, r(WIDE, i, Uq), r(INT, i, Ud), w);
    }
    let wide = |i: u32| r(WIDE, i, Q);
    k.add(1, r(FIRST, 0, Q), r(1, 0, Q), wide(0), w);
    k.add(1, r(BIAS_AT, 0, Q), r(1, 1, Q), wide(1), w);
    k.add(1, r(Q_AT, 0, Q), r(1, 2, Q), wide(2), w);
    k.add(1, r(KV_AT, 0, Q), r(1, 3, Q), wide(3), w);
    k.add(1, r(TABLE_AT, 0, Q), r(1, 3, Q), wide(4), w);
    k.lane_addresses(ROWS, r(FIRST, 0, Q), ROW_BYTES as i32, SCRATCH);
    k.lane_addresses(KEYS, r(KV_AT, 0, Q), 32, SCRATCH);

    let fetch = |k: &mut Asm, to: u32, at: u32, offset: u32, desc: u32, token: u16| {
        k.send(
            SFID_UGM,
            r(to, 0, Ud),
            r(at, 0, Ud),
            null(Ud),
            0,
            offset << 12,
            desc,
            false,
            Swsb::after_set(1, token),
        );
    };
    for i in 0..6 {
        fetch(
            &mut k,
            COS + 8 * i,
            TABLE_AT,
            i * 512,
            LOAD_64X64,
            TABLES + i as u16,
        );
    }
    fetch(&mut k, BIAS, BIAS_AT, 0, LOAD_64X64, TABLES + 6);
    fetch(&mut k, BIAS + 8, BIAS_AT, 512, LOAD_32X64, TABLES + 7);

    for side in 0..3 {
        // The stores of the side before read the registers the next side reuses.
        for token in STORE..SCATTER + 4 {
            k.sync(Swsb::dst(token));
        }
        for g in 0..CHANNELS / 4 {
            let token = g as u16 % 4;
            if g >= 4 {
                k.sync(Swsb::dst(token));
            }
            k.send16(
                SFID_UGM,
                r(x(4 * g), 0, Ud),
                r(ROWS, 0, Ud),
                null(Ud),
                0,
                ((side * CHANNELS + 4 * g) * 4) << 12,
                GATHER_32X4,
                Swsb::after_set(1, token),
            );
        }
        for token in (0..4).chain(TABLES..TABLES + 8) {
            k.sync(Swsb::dst(token));
        }
        for c in 0..CHANNELS {
            let at = side * CHANNELS + c;
            let bias = r(BIAS + at / 16, at % 16, F).s();
            k.add(16, r(x(c), 0, F), r(x(c), 0, F), bias, first(c));
        }
        if side < 2 {
            // Unit RMS: the sum of squares per token, in four accumulators.
            for a in 0..4 {
                k.mov(16, r(T2 + a, 0, D), imm(0), Swsb::NONE);
            }
            for c in 0..CHANNELS {
                let acc = r(T2 + c % 4, 0, F);
                let wait = if c < 4 { first(c) } else { back };
                k.mad(16, acc, acc, r(x(c), 0, F), r(x(c), 0, F), wait);
            }
            k.add(
                16,
                r(T2, 0, F),
                r(T2, 0, F),
                r(T2 + 1, 0, F),
                Swsb::after(3),
            );
            k.add(
                16,
                r(T2 + 2, 0, F),
                r(T2 + 2, 0, F),
                r(T2 + 3, 0, F),
                Swsb::after(2),
            );
            k.add(16, r(T2, 0, F), r(T2, 0, F), r(T2 + 2, 0, F), w);
            k.mul(
                16,
                r(SCALE, 0, F),
                r(T2, 0, F),
                immf(1.0 / CHANNELS as f32),
                w,
            );
            k.add(16, r(SCALE, 0, F), r(SCALE, 0, F), immf(1e-5), w);
            k.math(16, Math::Rsq, r(SCALE, 0, F), r(SCALE, 0, F), w);
            // The turn: a' = cos a - sin b, b' = sin a + cos b.
            for group in 0..PAIRS / 4 {
                let pair = |i: u32| group * 4 + i;
                let (a, b) = (
                    |i: u32| r(x(pair(i)), 0, F),
                    |i: u32| r(x(pair(i) + PAIRS), 0, F),
                );
                let (cos, sin) = (
                    |i: u32| r(COS + pair(i), 0, F),
                    |i: u32| r(SIN + pair(i), 0, F),
                );
                for i in 0..4 {
                    k.mul(16, r(T1 + i, 0, F), cos(i), a(i), first(i));
                }
                for i in 0..4 {
                    k.mul(16, r(T2 + i, 0, F), sin(i), a(i), Swsb::NONE);
                }
                for i in 0..4 {
                    k.mad(16, a(i), r(T1 + i, 0, F), sin(i).minus(), b(i), back);
                }
                for i in 0..4 {
                    k.mad(16, b(i), r(T2 + i, 0, F), cos(i), b(i), back);
                }
            }
            for c in 0..CHANNELS {
                k.mul(16, r(x(c), 0, F), r(x(c), 0, F), r(SCALE, 0, F), first(c));
            }
        }
        // Half floats in place: each lane's low half.
        for c in 0..CHANNELS {
            k.mov(16, r(x(c), 0, Hf).step2(), r(x(c), 0, F), first(c));
        }
        if side == 1 {
            // Four channels of each key side by side, as one 64-bit lane.
            for group in 0..CHANNELS / 4 {
                let (packed, token) = (OUT + 2 * (group % 4), SCATTER + group as u16 % 4);
                k.sync(Swsb::dst(token));
                for i in 0..4 {
                    let half = r(x(group * 4 + i), 0, Uw).step2();
                    k.mov(16, r(packed, i, Uw).step4(), half, first(i));
                }
                let c = group * 4;
                k.send16(
                    SFID_UGM,
                    null(Ud),
                    r(KEYS, 0, Ud),
                    r(packed, 0, Ud),
                    2,
                    ((c / 16) * 512 + (c % 16) * 2) << 12,
                    SCATTER_64,
                    Swsb::after_set(1, token),
                );
            }
            continue;
        }
        // Queries: a register per channel pair, each lane both channels of its token.
        // Values: a register per channel pair, each half one channel's 16 tokens.
        for chunk in 0..CHANNELS / 16 {
            let token = STORE + chunk as u16;
            k.sync(Swsb::dst(token));
            k.sync(Swsb::dst(STORE + (chunk as u16 + 3) % 4));
            for p in 0..8 {
                let c = chunk * 16 + 2 * p;
                let (even, odd) = if side == 0 {
                    (r(OUT + p, 0, Uw).step2(), r(OUT + p, 1, Uw).step2())
                } else {
                    (r(OUT + p, 0, Uw), r(OUT + p, 16, Uw))
                };
                k.mov(16, even, r(x(c), 0, Uw).step2(), first(p));
                k.mov(16, odd, r(x(c + 1), 0, Uw).step2(), Swsb::NONE);
            }
            let (at, offset) = if side == 0 {
                (Q_AT, chunk * 512)
            } else {
                (KV_AT, attend::VALUES as u32 + chunk * 512)
            };
            k.send(
                SFID_UGM,
                null(Ud),
                r(at, 0, Ud),
                r(OUT, 0, Ud),
                8,
                offset << 12,
                STORE_64X64,
                false,
                Swsb::after_set(1, token),
            );
        }
    }
    for token in STORE..SCATTER + 4 {
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
                gpu::{Buffer, Gpu, Select, bytes_of},
                safetensors::{narrow, widen},
            },
            level_zero::LevelZero,
            load,
        },
        *,
    };

    /// The kernel against the same arithmetic on the CPU, for random sums.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn split_matches_cpu() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let (padded, heads, channels) = (1920usize, 32usize, CHANNELS as usize);
        let tiles = padded / 16;
        let mut state = 3u32;
        let mut random = |count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / (1 << 22) as f32 - 2.0
                })
                .collect()
        };
        let sums = random(padded * heads * 3 * channels);
        let vectors = random(32 + heads * 3 * channels);
        let turns: Vec<[f32; 2]> = random(1797 * PAIRS as usize)
            .into_iter()
            .map(|angle| [angle.cos(), angle.sin()])
            .collect();

        let stream = heads * tiles * attend::CHUNK_BYTES + attend::SLACK;
        let mut want_q = vec![0u16; heads * tiles * attend::TILE_BYTES / 2];
        let mut want_kv = vec![0u16; stream / 2];
        for token in 0..padded {
            for head in 0..heads {
                let at = (token * heads + head) * 3 * channels;
                let bias = 32 + head * 3 * channels;
                let (q_at, kv_at) = (
                    head * tiles * attend::TILE_BYTES,
                    head * tiles * attend::CHUNK_BYTES,
                );
                for side in 0..3 {
                    let mut row: Vec<f32> = (0..channels)
                        .map(|c| {
                            sums[at + side * channels + c] + vectors[bias + side * channels + c]
                        })
                        .collect();
                    if side < 2 {
                        let squares: f32 = row.iter().map(|v| v * v).sum();
                        let scale = 1.0 / (squares / channels as f32 + 1e-5).sqrt();
                        for pair in 0..PAIRS as usize {
                            let [cos, sin] = turns
                                .get(token * PAIRS as usize + pair)
                                .copied()
                                .unwrap_or([1.0, 0.0]);
                            let (a, b) = (row[pair], row[pair + PAIRS as usize]);
                            row[pair] = cos * a - sin * b;
                            row[pair + PAIRS as usize] = sin * a + cos * b;
                        }
                        row.iter_mut().for_each(|v| *v *= scale);
                    }
                    for (c, value) in row.into_iter().enumerate() {
                        let half = narrow(value);
                        match side {
                            0 => want_q[(q_at + attend::query_at(token, c)) / 2] = half,
                            1 => want_kv[(kv_at + attend::key_at(token, c)) / 2] = half,
                            _ => want_kv[(kv_at + attend::value_at(token, c)) / 2] = half,
                        }
                    }
                }
            }
        }

        let table = rotary_table(&turns, padded);
        let filled = |bytes: &[u8], size: usize| {
            let buffer = gpu.shared(size as u64).unwrap();
            gpu.upload(&buffer, 0, bytes).unwrap();
            buffer
        };
        let g = filled(bytes_of(&sums), sums.len() * 4);
        let v = filled(bytes_of(&vectors), vectors.len() * 4);
        let q = gpu.shared(want_q.len() as u64 * 2).unwrap();
        let kv = gpu.shared((stream + table.len() * 4) as u64).unwrap();
        gpu.upload(&kv, stream as u64, bytes_of(&table)).unwrap();
        let import = |buffer: &Buffer| {
            fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                .unwrap()
        };
        let memory = [import(&g), import(&v), import(&q), import(&kv)];
        let memory: Vec<_> = memory.iter().collect();
        let shape = Shape {
            padded: padded as u32,
            bias_first: 32,
            rotary_first: stream as u32,
        };
        let kernel = load(&fast, &kernel(&shape), 4).unwrap();
        let mut ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            fast.launch(&kernel, &memory, shape.groups(heads)).unwrap();
            ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let read = |buffer: &Buffer, halves: usize| -> Vec<u16> {
            gpu.download(buffer, 0, halves * 2)
                .unwrap()
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect()
        };
        let decibels = |want: &[u16], got: &[u16]| {
            let (mut signal, mut error) = (0f64, 0f64);
            for (w, g) in want.iter().zip(got) {
                let (w, g) = (f64::from(widen(*w)), f64::from(widen(*g)));
                signal += w * w;
                error += (w - g) * (w - g);
            }
            10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10()
        };
        let (q_db, kv_db) = (
            decibels(&want_q, &read(&q, want_q.len())),
            decibels(&want_kv, &read(&kv, want_kv.len())),
        );
        println!(
            "split of {padded} tokens x {heads} heads: {ms:.3} ms; queries {q_db:.1} dB, keys and values {kv_db:.1} dB from the CPU"
        );
        assert!(q_db > 60.0 && kv_db > 60.0);
    }
}
