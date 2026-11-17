//! The video decoder's gather of attention output into token rows, as a kernel: the
//! attention kernel's mixed values (`attend`), divided by each query's weight sum, written
//! as half floats in the next matrix product's token layout (`int8::token_word`).
//!
//! One hardware thread owns 16 tokens of one head, one token per lane, so every register is
//! one channel of the 16 tokens. The product wants each token's channels side by side, so
//! each channel goes out through one scatter store (one address per lane).
//!
//! Arguments: the mixed values and the weight sums as the attention kernel writes them,
//! and the half-float output for rows of `heads x 64` channels. Only the lower 64 channels
//! of the kernel's 128 are read. Padded tokens are computed like any other.
use super::{Ty::*, attend, *};

const CHANNELS: u32 = 64;
const LOAD_64X64: u32 = 0x0288_F780;
const LOAD_32X16: u32 = 0x0210_D580;
const SCATTER_16: u32 = 0x0400_0B84; // the low half of each lane to its address

/// Thread groups for `padded` tokens (a multiple of 32): token tiles, heads.
pub fn groups(padded: usize, heads: usize) -> [u32; 3] {
    [(padded / 16) as u32, heads as u32, 1]
}

const MIXED_AT: u32 = 2;
const SUMS_AT: u32 = 3;
const ROWS: u32 = 4; // and 5: each lane's token in the output
const FIRST: u32 = 6;
const INT: u32 = 7;
const WIDE: u32 = 8;
const SUM: u32 = 9;
const SCRATCH: u32 = 10;
const HEADER: u32 = 240;
fn x(c: u32) -> u32 {
    16 + c
}
const SUM_TOKEN: u16 = 8;
const SCATTER: u16 = 9;

/// `padded` tokens and `heads` heads of 64 channels: the output's rows are `heads * 64`
/// halves, which must be 4096 bytes (the product's 128 steps).
pub fn kernel(padded: u32, heads: u32) -> Vec<u8> {
    assert!(heads * CHANNELS * 2 == 4096 && padded.is_multiple_of(32));
    let tiles = padded / 16;
    let w = Swsb::after(1);
    let first = |i: u32| if i == 0 { w } else { Swsb::NONE };
    let int = |lane: u32| r(INT, lane, D);
    let mut k = Asm::new();

    // r0: thread header (tile in dword 1, head in dword 6); r1: the three pointers.
    k.mov(16, r(HEADER, 0, F), r(0, 0, F), Swsb::NONE);
    k.keep_subnormals();
    k.mov(1, int(0), imm(tiles as i32), Swsb::NONE);
    k.mul(1, int(0), int(0), r(0, 12, Uw), w);
    k.add(1, int(0), int(0), r(0, 1, D), w);
    k.shl(
        1,
        int(1),
        int(0),
        immw(attend::SUM_BYTES.trailing_zeros() as i16),
        w,
    );
    k.shl(
        1,
        int(0),
        int(0),
        immw(attend::OUT_BYTES.trailing_zeros() as i16),
        w,
    );
    // The output: 32 tokens share a tile of 128 steps of 1024 bytes; a token's part of a
    // step is 32 bytes, and a head is four steps.
    k.shr(1, int(2), r(0, 1, D), immw(1), Swsb::NONE);
    k.shl(1, int(2), int(2), immw(17), w);
    k.and(1, int(3), r(0, 1, D), imm(1), Swsb::NONE);
    k.shl(1, int(3), int(3), immw(9), w);
    k.add(1, int(2), int(2), int(3), w);
    k.shl(1, int(3), r(0, 6, D), immw(12), Swsb::NONE);
    k.add(1, int(2), int(2), int(3), w);
    for i in 0..3 {
        k.mov(1, r(WIDE, i, Uq), r(INT, i, Ud), w);
    }
    k.add(1, r(MIXED_AT, 0, Q), r(1, 0, Q), r(WIDE, 0, Q), w);
    k.add(1, r(SUMS_AT, 0, Q), r(1, 1, Q), r(WIDE, 1, Q), w);
    k.add(1, r(FIRST, 0, Q), r(1, 2, Q), r(WIDE, 2, Q), w);
    k.lane_addresses(ROWS, r(FIRST, 0, Q), 32, SCRATCH);

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
    for i in 0..CHANNELS / 8 {
        fetch(&mut k, x(8 * i), MIXED_AT, i * 512, LOAD_64X64, i as u16);
    }
    fetch(&mut k, SUM, SUMS_AT, 0, LOAD_32X16, SUM_TOKEN);
    for token in 0..=SUM_TOKEN {
        k.sync(Swsb::dst(token));
    }
    k.math(16, Math::Inv, r(SUM, 0, F), r(SUM, 0, F), w);
    let mut stores = 0u16;
    for group in 0..CHANNELS / 4 {
        let row = |i: u32| r(x(group * 4 + i), 0, F);
        for i in 0..4 {
            k.mul(16, row(i), row(i), r(SUM, 0, F), first(i));
        }
        // Half floats in place: each lane's low half.
        for i in 0..4 {
            k.mov(
                16,
                r(x(group * 4 + i), 0, Hf).step2(),
                row(i),
                Swsb::after(4),
            );
        }
        for i in 0..4 {
            let (c, token) = (group * 4 + i, SCATTER + stores % 16);
            if stores >= 16 {
                k.sync(Swsb::dst(token));
            }
            k.send16(
                SFID_UGM,
                null(Ud),
                r(ROWS, 0, Ud),
                r(x(c), 0, Ud),
                1,
                ((c / 16) * 1024 + (c % 16) * 2) << 12,
                SCATTER_16,
                Swsb::after_set(1, token),
            );
            stores += 1;
        }
    }
    for token in SCATTER..SCATTER + 16 {
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
            int8,
            level_zero::LevelZero,
            load,
        },
        *,
    };

    /// The kernel against the same division on the CPU, for random values and sums.
    #[test]
    #[ignore = "needs the Intel GPU with Level Zero"]
    fn merge_matches_cpu() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let fast = LevelZero::open().unwrap();
        let (padded, heads, channels) = (1920usize, 32usize, CHANNELS as usize);
        let tiles = padded / 16;
        let mut state = 11u32;
        let mut random = |count: usize| -> Vec<f32> {
            (0..count)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 8) as f32 / (1 << 22) as f32 + 0.5
                })
                .collect()
        };
        let mixed = random(heads * tiles * attend::OUT_BYTES / 4);
        let sums = random(heads * tiles * attend::SUM_BYTES / 4);
        let row = heads * channels * 2;
        let mut want = vec![0u16; padded * row / 2];
        for token in 0..padded {
            for head in 0..heads {
                let tile = head * tiles + token / 16;
                let sum = sums[tile * 16 + token % 16];
                for c in 0..channels {
                    let value =
                        mixed[(tile * attend::OUT_BYTES + attend::out_at(token % 16, c)) / 4];
                    let byte = (head * channels + c) * 2;
                    let at = int8::token_word(token, byte / 16, row) * 16 + byte % 16;
                    want[at / 2] = narrow(value * (1.0 / sum));
                }
            }
        }
        let filled = |values: &[f32]| {
            let buffer = gpu.shared(values.len() as u64 * 4).unwrap();
            gpu.upload(&buffer, 0, bytes_of(values)).unwrap();
            buffer
        };
        let (o, s) = (filled(&mixed), filled(&sums));
        let h = gpu.shared((padded * row) as u64).unwrap();
        let import = |buffer: &Buffer| {
            fast.import(&gpu.export(buffer).unwrap(), buffer.bytes())
                .unwrap()
        };
        let memory = [import(&o), import(&s), import(&h)];
        let memory: Vec<_> = memory.iter().collect();
        let kernel = load(&fast, &kernel(padded as u32, heads as u32), 3).unwrap();
        let mut ms = f64::MAX;
        for _ in 0..8 {
            let started = std::time::Instant::now();
            fast.launch(&kernel, &memory, groups(padded, heads))
                .unwrap();
            ms = ms.min(started.elapsed().as_secs_f64() * 1e3);
        }
        let got = gpu.download(&h, 0, padded * row).unwrap();
        let (mut signal, mut error) = (0f64, 0f64);
        for (w, g) in want.iter().zip(got.chunks_exact(2)) {
            let (w, g) = (
                f64::from(widen(*w)),
                f64::from(widen(u16::from_le_bytes([g[0], g[1]]))),
            );
            signal += w * w;
            error += (w - g) * (w - g);
        }
        let db = 10.0 * (signal / error.max(f64::MIN_POSITIVE)).log10();
        println!("merge of {padded} tokens x {heads} heads: {ms:.3} ms; {db:.1} dB from the CPU");
        assert!(db > 60.0);
    }
}
