//! The video decoder's gated activation between its feed-forward products as a kernel:
//!   h[token][i] = silu(g[token][i] + b[i]) * (g[token][8192 + i] + b[8192 + i])
//! read from the first product's f32 sums and written as half floats in the second
//! product's token layout (`int8::token_word`).
//!
//! One hardware thread owns one token and walks its 8192 outputs 128 at a time. Two sets
//! of load buffers alternate, so the next chunk's four loads are in flight while the
//! current one is computed. `silu(x) = x / (1 + 2^(-x log2 e))` with the hardware's
//! power-of-two and reciprocal.
//!
//! Arguments: the sums `[token][16384]` f32, the layer's vectors (f32, bias offset in
//! values) and the half-float output. The sums need `SLACK` bytes after their last row
//! because loads run one chunk ahead. Threads are launched as (32 tokens of a tile, tiles);
//! padded tokens are computed like any other.
use super::{Ty::*, *};

/// Outputs per token.
pub const ROW: u32 = 8192;
pub const SLACK: usize = 1 << 16;
const LOAD_64X64: u32 = 0x0288_F780; // 64 qwords
const STORE_32X8: u32 = 0x0200_C584; // 8 dwords: 16 halves
const CHUNK: u32 = 512;
const CHUNKS: u32 = ROW / 128;

/// Thread groups for `padded` tokens (a multiple of 32).
pub fn groups(padded: usize) -> [u32; 3] {
    [32, (padded / 32) as u32, 1]
}

const G_AT: u32 = 2;
const V_AT: u32 = 3;
const H_AT: u32 = 4;
const INT: u32 = 5;
const WIDE: u32 = 6;
const HEADER: u32 = 240;
/// Buffer `part` (gate, value, gate bias, value bias) of set `q`, with its token.
fn buffer(q: u32, part: u32) -> (u32, u16) {
    (16 + (q * 4 + part) * 8, (q * 4 + part) as u16)
}
const T: u32 = 80;
const HALF: u32 = 84;
const P: u32 = 88;
const STORE: u16 = 8;

pub fn kernel(bias_first: u32) -> Vec<u8> {
    let w = Swsb::after(1);
    let first = |i: u32| if i == 0 { w } else { Swsb::NONE };
    let back = Swsb::after(4);
    let mut k = Asm::new();

    // r0: thread header (group ids in dwords 1 and 6); r1: the three pointers.
    k.mov(16, r(HEADER, 0, F), r(0, 0, F), Swsb::NONE);
    k.keep_subnormals();
    k.shl(1, r(INT, 0, D), r(0, 6, D), immw(5), Swsb::NONE);
    k.add(1, r(INT, 0, D), r(INT, 0, D), r(0, 1, D), w); // token
    k.shl(1, r(INT, 1, D), r(INT, 0, D), immw(16), w); // its row of sums
    k.shl(1, r(INT, 2, D), r(0, 6, D), immw(19), Swsb::NONE); // its tile in the output
    k.shl(1, r(INT, 3, D), r(0, 1, D), immw(5), Swsb::NONE);
    k.add(1, r(INT, 2, D), r(INT, 2, D), r(INT, 3, D), w);
    for i in 1..3 {
        k.mov(1, r(WIDE, i, Uq), r(INT, i, Ud), w);
    }
    k.add(1, r(G_AT, 0, Q), r(1, 0, Q), r(WIDE, 1, Q), w);
    k.add(
        1,
        r(V_AT, 0, Q),
        r(1, 1, Q),
        imm((bias_first * 4) as i32),
        Swsb::NONE,
    );
    k.add(1, r(H_AT, 0, Q), r(1, 2, Q), r(WIDE, 2, Q), w);
    k.mov(1, r(INT, 4, D), imm(0), Swsb::NONE);

    // The chunk `ahead` chunks past the pointers, into set `q`.
    let request = |k: &mut Asm, q: u32, ahead: u32| {
        for (part, (at, offset)) in [(G_AT, 0), (G_AT, ROW * 4), (V_AT, 0), (V_AT, ROW * 4)]
            .into_iter()
            .enumerate()
        {
            let (to, token) = buffer(q, part as u32);
            k.send(
                SFID_UGM,
                r(to, 0, Ud),
                r(at, 0, Ud),
                null(Ud),
                0,
                (offset + ahead * CHUNK) << 12,
                LOAD_64X64,
                false,
                Swsb::after_set(1, token),
            );
        }
    };
    // Chunk `chunk` of the two the pointers cover, from set `q`.
    let compute = |k: &mut Asm, q: u32, chunk: u32| {
        for part in 0..4 {
            k.sync(Swsb::dst(buffer(q, part).1));
        }
        for group in 0..2 {
            let part = |n: u32, i: u32| r(buffer(q, n).0 + group * 4 + i, 0, F);
            let (gate, value) = (|i: u32| part(0, i), |i: u32| part(1, i));
            let t = |i: u32| r(T + i, 0, F);
            for i in 0..4 {
                k.add(16, gate(i), gate(i), part(2, i), first(i));
            }
            for i in 0..4 {
                k.add(16, value(i), value(i), part(3, i), Swsb::NONE);
            }
            for i in 0..4 {
                k.mul(
                    16,
                    t(i),
                    gate(i),
                    immf(-std::f32::consts::LOG2_E),
                    Swsb::after(7),
                );
            }
            // Math functions run in a pipe of their own, so each of these groups waits for
            // the one before it as a whole.
            for i in 0..4 {
                k.math(16, Math::Exp, t(i), t(i), None, first(i));
            }
            for i in 0..4 {
                k.add(16, t(i), t(i), immf(1.0), first(i));
            }
            for i in 0..4 {
                k.math(16, Math::Inv, t(i), t(i), None, first(i));
            }
            for i in 0..4 {
                k.mul(16, t(i), t(i), gate(i), first(i));
            }
            for i in 0..4 {
                k.mul(16, t(i), t(i), value(i), back);
            }
            // Sixteen halves at stride two, then packed into the low half of a register.
            for i in 0..4 {
                k.mov(16, r(HALF + i, 0, Hf).step2(), t(i), back);
            }
            let slot = |i: u32| group * 4 + i;
            for i in 0..4 {
                k.sync(Swsb::dst(STORE + slot(i) as u16));
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
                    ((chunk * 8 + slot(i)) * 1024) << 12,
                    STORE_32X8,
                    false,
                    Swsb::after_set(1, STORE + slot(i) as u16),
                );
            }
        }
    };

    request(&mut k, 0, 0);
    let top = k.label();
    k.place(top);
    request(&mut k, 1, 1);
    compute(&mut k, 0, 0);
    // The pointers move once every send that addresses through them has read them.
    for part in 0..4 {
        k.sync(Swsb::src(buffer(1, part).1));
    }
    k.add(
        1,
        r(G_AT, 0, Q),
        r(G_AT, 0, Q),
        imm(2 * CHUNK as i32),
        Swsb::NONE,
    );
    k.add(
        1,
        r(V_AT, 0, Q),
        r(V_AT, 0, Q),
        imm(2 * CHUNK as i32),
        Swsb::NONE,
    );
    request(&mut k, 0, 0);
    compute(&mut k, 1, 1);
    for token in STORE..STORE + 8 {
        k.sync(Swsb::src(token));
    }
    k.add(1, r(H_AT, 0, Q), r(H_AT, 0, Q), imm(16 * 1024), Swsb::NONE);
    k.add(1, r(INT, 4, D), r(INT, 4, D), imm(2), Swsb::NONE);
    k.cmp(Cond::Lt, r(INT, 4, D), imm(CHUNKS as i32), w);
    k.jmpi(top, true, w);

    for token in 0..STORE + 8 {
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
                gpu::{Buffer, Gpu, Select, Step, bytes_of, shaders},
                safetensors::widen,
            },
            level_zero::LevelZero,
            load,
        },
        *,
    };

    /// The kernel against the gate shader on the same random sums.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn gate_matches_the_shader() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let gate = gpu.kernel(shaders::GATE, 3, 3, None).unwrap();
        let (tokens, row) = (1920usize, ROW as usize);
        let mut state = 9u32;
        // Mostly within -8..8, with a few large values of either sign.
        let mut random = |count: usize| -> Vec<f32> {
            (0..count)
                .map(|i| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    let value = (state >> 8) as f32 / (1 << 20) as f32 - 8.0;
                    if i % 1000 == 0 { value * 40.0 } else { value }
                })
                .collect()
        };
        let sums = random(tokens * row * 2);
        let vectors = random(row * 2 + 64);
        let half_bytes = (tokens * row * 2) as u64;
        let g = gpu.shared((sums.len() * 4 + SLACK) as u64).unwrap();
        gpu.upload(&g, 0, bytes_of(&sums)).unwrap();
        let v = gpu.shared(vectors.len() as u64 * 4).unwrap();
        gpu.upload(&v, 0, bytes_of(&vectors)).unwrap();
        let (plain, h) = (
            gpu.buffer(half_bytes).unwrap(),
            gpu.shared(half_bytes).unwrap(),
        );
        let buffers = vec![&g, &v, &plain];
        let steps = [Step {
            kernel: &gate,
            buffers: &buffers,
            constants: &[tokens as u32, 64, 1],
            groups: [(row / 4 / 64) as u32, tokens as u32, 1],
        }];
        let mut shader_ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            gpu.pass(&steps).unwrap();
            shader_ms = shader_ms.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let import = |buffer: &Buffer| {
            fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                .unwrap()
        };
        let memory = [import(&g), import(&v), import(&h)];
        let memory: Vec<_> = memory.iter().collect();
        let kernel = load(&fast, &kernel(64), 3).unwrap();
        let mut ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            fast.launch(&kernel, &memory, groups(tokens)).unwrap();
            ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let read = |buffer: &Buffer| -> Vec<f32> {
            gpu.download(buffer, 0, half_bytes as usize)
                .unwrap()
                .chunks_exact(2)
                .map(|b| widen(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        };
        let (want, got) = (read(&plain), read(&h));
        let (mut signal, mut error, mut worst) = (0f64, 0f64, 0f64);
        let mut odd = 0;
        for (at, (w, g)) in want.iter().zip(&got).enumerate() {
            if w == g {
                signal += f64::from(*w).powi(2);
                continue;
            }
            if !w.is_finite() || !g.is_finite() {
                odd += 1;
                if odd <= 8 {
                    // Stream position back to token and column.
                    let (tile, rest) = (at / (512 * 512), at % (512 * 512));
                    let (step, rest) = (rest / 512, rest % 512);
                    let (token, column) = (tile * 32 + rest / 16, step * 16 + rest % 16);
                    let (gate, value) = (
                        sums[token * 2 * row + column] + vectors[64 + column],
                        sums[token * 2 * row + row + column] + vectors[64 + row + column],
                    );
                    println!("shader {w} kernel {g} from gate {gate} value {value}");
                }
                continue;
            }
            signal += f64::from(*w).powi(2);
            error += (f64::from(*w) - f64::from(*g)).powi(2);
            worst = worst.max((f64::from(*w) - f64::from(*g)).abs() / f64::from(w.abs()).max(1.0));
        }
        println!("{odd} values not finite on one side only");
        let db = 10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10();
        println!(
            "gate of {tokens} rows: {ms:.3} ms, shader {shader_ms:.3} ms; {db:.1} dB apart, worst relative difference {worst:.2e}"
        );
        assert!(db > 80.0 && odd == 0);
    }
}
