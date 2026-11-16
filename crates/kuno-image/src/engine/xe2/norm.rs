//! The video decoder's residual update and RMS normalization as one kernel, so that nothing
//! but machine code runs between two matrix products.
//!
//! One hardware thread owns one token row of 2048 f32 values and keeps it in registers:
//!   1. `x += scale * update + shift` (optional), stored back to the stream,
//!   2. `h = x * weight / sqrt(mean(x^2) + 1e-5)` (optional), written as half floats in the
//!      matrix product's token layout (`int8::token_word`).
//!
//! Arguments: the stream `[token][2048]` f32, the update `[token][2048]` f32, the layer's
//! vectors (f32; offsets in values) and the half-float output. Threads are launched as
//! (32 tokens of a tile, tiles); rows from `tokens` up skip the update, so they stay zero
//! and normalize to zeros.
use super::{Ty::*, *};

/// Values per token row.
pub const ROW: u32 = 2048;
const LOAD_64X64: u32 = 0x0288_F780; // 64 qwords
const STORE_64X64: u32 = 0x0200_F784;
const STORE_32X8: u32 = 0x0200_C584; // 8 dwords: 16 halves

#[derive(Clone, Copy)]
pub struct Shape {
    /// Live tokens.
    pub tokens: u32,
    /// First values of `scale` and `shift` in the vectors.
    pub update: Option<(u32, u32)>,
    /// First value of the norm weight in the vectors.
    pub norm: Option<u32>,
}
impl Shape {
    /// Thread groups for `padded` tokens (a multiple of 32).
    pub fn groups(padded: usize) -> [u32; 3] {
        [32, (padded / 32) as u32, 1]
    }
}

// Register plan (256 registers): the row in 16..144, three load buffers of eight registers
// per slot and two slots, then scratch.
const X_AT: u32 = 2;
const U_AT: u32 = 3;
const V_AT: u32 = 4;
const H_AT: u32 = 5;
const INT: u32 = 6;
const WIDE: u32 = 7;
const SCALE: u32 = 8;
const HALF: u32 = 9;
const HEADER: u32 = 240;
const CHUNKS: u32 = ROW / 128;
fn x(i: u32) -> u32 {
    16 + i
}
/// Load buffer `n` of six, with its scoreboard token.
fn buffer(n: u32) -> (u32, u16) {
    (144 + n * 8, 8 + n as u16)
}
const T: u32 = 192;
const P: u32 = 200;
const ACC: u32 = 208;
const X_STORE: u16 = 14;
const H_STORE: u16 = 22;

fn load(k: &mut Asm, to: u32, at: u32, offset: u32, token: u16) {
    k.send(
        SFID_UGM,
        r(to, 0, Ud),
        r(at, 0, Ud),
        null(Ud),
        0,
        offset << 12,
        LOAD_64X64,
        false,
        Swsb::after_set(1, token),
    );
}

pub fn kernel(shape: &Shape) -> Vec<u8> {
    let w = Swsb::after(1);
    let first = |i: u32| if i == 0 { w } else { Swsb::NONE };
    let mut k = Asm::new();

    // r0: thread header (group ids in dwords 1 and 6); r1: the four pointers.
    k.mov(16, r(HEADER, 0, F), r(0, 0, F), Swsb::NONE);
    k.keep_subnormals();
    k.shl(1, r(INT, 0, D), r(0, 6, D), immw(5), Swsb::NONE);
    k.add(1, r(INT, 0, D), r(INT, 0, D), r(0, 1, D), w); // token
    k.shl(1, r(INT, 1, D), r(INT, 0, D), immw(13), w); // its row in the stream
    k.shl(1, r(INT, 2, D), r(0, 6, D), immw(17), Swsb::NONE); // its tile in the output
    k.shl(1, r(INT, 3, D), r(0, 1, D), immw(5), Swsb::NONE);
    k.add(1, r(INT, 2, D), r(INT, 2, D), r(INT, 3, D), w);
    for i in 1..3 {
        k.mov(1, r(WIDE, i, Uq), r(INT, i, Ud), w);
    }
    k.add(1, r(X_AT, 0, Q), r(1, 0, Q), r(WIDE, 1, Q), w);
    k.add(1, r(U_AT, 0, Q), r(1, 1, Q), r(WIDE, 1, Q), w);
    k.mov(1, r(V_AT, 0, Q), r(1, 2, Q), Swsb::NONE);
    k.add(1, r(H_AT, 0, Q), r(1, 3, Q), r(WIDE, 2, Q), w);

    for c in 0..CHUNKS {
        if c >= 8 {
            k.sync(Swsb::dst(c as u16 % 8));
        }
        load(&mut k, x(c * 8), X_AT, c * 512, c as u16 % 8);
    }
    for token in 0..8 {
        k.sync(Swsb::dst(token));
    }

    if let Some((scale, shift)) = shape.update {
        let skip = k.label();
        k.cmp(Cond::Ge, r(INT, 0, D), imm(shape.tokens as i32), w);
        k.jmpi(skip, true, w);
        // Chunk c's update, scale and shift use buffers 3 * (c % 2)..; two chunks are in
        // flight.
        let request = |k: &mut Asm, c: u32| {
            for (n, (at, first)) in [(U_AT, 0), (V_AT, scale * 4), (V_AT, shift * 4)]
                .into_iter()
                .enumerate()
            {
                let (to, token) = buffer(3 * (c % 2) + n as u32);
                load(k, to, at, first + c * 512, token);
            }
        };
        request(&mut k, 0);
        for c in 0..CHUNKS {
            if c + 1 < CHUNKS {
                request(&mut k, c + 1);
            }
            let from = |n: u32| buffer(3 * (c % 2) + n);
            for n in 0..3 {
                k.sync(Swsb::dst(from(n).1));
            }
            for group in 0..2 {
                let row = |i: u32| r(x(c * 8 + group * 4 + i), 0, F);
                let part = |n: u32, i: u32| r(from(n).0 + group * 4 + i, 0, F);
                for i in 0..4 {
                    k.mad(16, row(i), row(i), part(1, i), part(0, i), first(i));
                }
                for i in 0..4 {
                    k.add(16, row(i), row(i), part(2, i), Swsb::after(4));
                }
            }
            let token = X_STORE + c as u16 % 8;
            if c >= 8 {
                k.sync(Swsb::dst(token));
            }
            k.send(
                SFID_UGM,
                null(Ud),
                r(X_AT, 0, Ud),
                r(x(c * 8), 0, Ud),
                8,
                (c * 512) << 12,
                STORE_64X64,
                false,
                Swsb::after_set(1, token),
            );
        }
        k.place(skip);
    }

    if let Some(weight) = shape.norm {
        // Sum of squares in four accumulators, then across them and across lanes. Float
        // sources must keep their lanes, so the upper lanes move down as integers first.
        for a in 0..4 {
            k.mov(16, r(ACC + a, 0, D), imm(0), Swsb::NONE);
        }
        for i in 0..ROW / 16 {
            let acc = r(ACC + i % 4, 0, F);
            let wait = if i < 4 { first(i) } else { Swsb::after(4) };
            k.mad(16, acc, acc, r(x(i), 0, F), r(x(i), 0, F), wait);
        }
        k.add(
            16,
            r(ACC, 0, F),
            r(ACC, 0, F),
            r(ACC + 1, 0, F),
            Swsb::after(3),
        );
        k.add(
            16,
            r(ACC + 2, 0, F),
            r(ACC + 2, 0, F),
            r(ACC + 3, 0, F),
            Swsb::after(2),
        );
        k.add(16, r(ACC, 0, F), r(ACC, 0, F), r(ACC + 2, 0, F), w);
        for lanes in [8, 4, 2, 1] {
            k.mov(lanes, r(ACC + 1, 0, Ud), r(ACC, lanes, Ud), w);
            k.add(lanes, r(ACC, 0, F), r(ACC, 0, F), r(ACC + 1, 0, F), w);
        }
        k.mul(1, r(SCALE, 0, F), r(ACC, 0, F), immf(1.0 / ROW as f32), w);
        k.add(1, r(SCALE, 0, F), r(SCALE, 0, F), immf(1e-5), w);
        k.math(1, Math::Rsq, r(SCALE, 0, F), r(SCALE, 0, F), None, w);

        // Weight chunk c uses buffer c % 4; three are in flight.
        let request = |k: &mut Asm, c: u32| {
            let (to, token) = buffer(c % 4);
            load(k, to, V_AT, weight * 4 + c * 512, token);
        };
        request(&mut k, 0);
        request(&mut k, 1);
        let mut stores = 0u32;
        for c in 0..CHUNKS {
            if c + 2 < CHUNKS {
                request(&mut k, c + 2);
            }
            let (from, token) = buffer(c % 4);
            k.sync(Swsb::dst(token));
            for group in 0..2 {
                let t = |i: u32| r(T + i, 0, F);
                let step = |i: u32| c * 8 + group * 4 + i;
                for i in 0..4 {
                    let weight = r(from + group * 4 + i, 0, F);
                    k.mul(16, t(i), r(x(step(i)), 0, F), weight, first(i));
                }
                for i in 0..4 {
                    k.mul(16, t(i), t(i), r(SCALE, 0, F).s(), Swsb::after(4));
                }
                // Sixteen halves at stride two, then packed into the low half of a register.
                for i in 0..4 {
                    k.mov(16, r(HALF + i, 0, Hf).step2(), t(i), Swsb::after(4));
                }
                let slot = |i: u32| (stores + i) % 8;
                for i in 0..4 {
                    if stores + i >= 8 {
                        k.sync(Swsb::dst(H_STORE + slot(i) as u16));
                    }
                }
                for i in 0..4 {
                    k.mov(
                        16,
                        r(P + slot(i), 0, Uw),
                        r(HALF + i, 0, Uw).step2(),
                        first(i),
                    );
                }
                for i in 0..4 {
                    k.send(
                        SFID_UGM,
                        null(Ud),
                        r(H_AT, 0, Ud),
                        r(P + slot(i), 0, Ud),
                        1,
                        (step(i) * 1024) << 12,
                        STORE_32X8,
                        false,
                        Swsb::after_set(1, H_STORE + slot(i) as u16),
                    );
                }
                stores += 4;
            }
        }
        for token in H_STORE..H_STORE + 8 {
            k.sync(Swsb::dst(token));
        }
    }
    if shape.update.is_some() {
        for token in X_STORE..X_STORE + 8 {
            k.sync(Swsb::dst(token));
        }
    }
    k.end(r(HEADER, 0, F));
    k.bytes()
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            super::{
                gpu::{Buffer, Gpu, Select, Step, bytes_of, floats, shaders},
                safetensors::widen,
            },
            int8,
            level_zero::LevelZero,
            load,
        },
        *,
    };

    fn decibels(want: &[f32], got: &[f32]) -> f64 {
        let (mut signal, mut error) = (0f64, 0f64);
        for (w, g) in want.iter().zip(got) {
            signal += f64::from(*w).powi(2);
            error += (f64::from(*w) - f64::from(*g)).powi(2);
        }
        10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10()
    }
    fn halves(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|b| widen(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    }

    /// The kernel against the residual and norm shaders on the same random rows.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn update_and_norm_match_the_shaders() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let residual = gpu.kernel(shaders::RESIDUAL, 3, 3, None).unwrap();
        let norm = gpu.kernel(shaders::NORM, 3, 5, None).unwrap();
        let (tokens, padded, row) = (1797usize, 1920usize, ROW as usize);
        let mut state = 5u32;
        let mut random = |count: usize, live: usize| -> Vec<f32> {
            (0..count)
                .map(|i| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    if i < live {
                        (state >> 8) as f32 / (1 << 22) as f32 - 2.0
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let start = random(padded * row, tokens * row);
        let update = random(padded * row, padded * row);
        // Bias, scale, weight, then the shift the kernel takes in place of the bias.
        let mut vectors = random(3 * row, 3 * row);
        let shift: Vec<f32> = (0..row).map(|i| vectors[i] * vectors[row + i]).collect();
        vectors.extend(shift);
        let (stream_bytes, half_bytes) = ((padded * row * 4) as u64, (padded * row * 2) as u64);
        let filled = |values: &[f32]| {
            let buffer = gpu.shared(values.len() as u64 * 4).unwrap();
            gpu.upload(&buffer, 0, bytes_of(values)).unwrap();
            buffer
        };
        let (u, v) = (filled(&update), filled(&vectors));
        let (plain, plain_half) = (filled(&start), gpu.buffer(half_bytes).unwrap());
        let n = tokens as u32;
        let (first, second) = (vec![&plain, &u, &v], vec![&plain, &v, &plain_half]);
        let steps = [
            Step {
                kernel: &residual,
                buffers: &first,
                constants: &[n, 0, row as u32],
                groups: [(row / 4 / 64) as u32, padded as u32, 1],
            },
            Step {
                kernel: &norm,
                buffers: &second,
                constants: &[n, 2 * row as u32, 0, 0, 1],
                groups: [padded as u32, 1, 1],
            },
        ];
        gpu.pass(&steps).unwrap();
        let want_stream = floats(&gpu.download(&plain, 0, stream_bytes as usize).unwrap());
        let want_half = halves(&gpu.download(&plain_half, 0, half_bytes as usize).unwrap());
        let mut shader_ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            gpu.pass(&steps).unwrap();
            shader_ms = shader_ms.min(started.elapsed().as_secs_f64() * 1e3);
        }

        let (x, h) = (filled(&start), gpu.shared(half_bytes).unwrap());
        let import = |buffer: &Buffer| {
            fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                .unwrap()
        };
        let memory = [import(&x), import(&u), import(&v), import(&h)];
        let memory: Vec<_> = memory.iter().collect();
        let shape = Shape {
            tokens: n,
            update: Some((row as u32, 3 * row as u32)),
            norm: Some(2 * row as u32),
        };
        let kernel = load(&fast, &kernel(&shape), 4).unwrap();
        let groups = Shape::groups(padded);
        fast.launch(&kernel, &memory, groups).unwrap();
        let got_stream = floats(&gpu.download(&x, 0, stream_bytes as usize).unwrap());
        let got_half = halves(&gpu.download(&h, 0, half_bytes as usize).unwrap());
        let mut ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            fast.launch(&kernel, &memory, groups).unwrap();
            ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let (stream_db, half_db) = (
            decibels(&want_stream, &got_stream),
            decibels(&want_half, &got_half),
        );
        println!(
            "update and norm of {padded} rows: {ms:.3} ms, shaders {shader_ms:.3} ms; stream {stream_db:.1} dB, halves {half_db:.1} dB apart"
        );
        let _ = int8::SLACK;
        assert!(stream_db > 120.0 && half_db > 80.0);
    }
}
