//! Int8 matrix product kernel for Xe2, written instruction by instruction.
//!
//! One hardware thread computes one tile: `8·MA` tokens × `16·NB` outputs, accumulated over
//! `ks` steps of 32 inputs. Operands are pre-arranged streams so that every load is one
//! contiguous cache-line run:
//!   A `[token tile][step][MA fragments × 256 B]`  (row r of a fragment = 32 inputs of token r)
//!   B `[output tile][step][NB fragments × 512 B]` (register g, lane n = inputs 4g.. of output n)
//!   C `[token][output]` int32, row-major, `c_row` bytes from one token to the next
//! Two register sets alternate, so the loads for the next step overlap the multiplies of the
//! current one. Loads and multiplies run out of order; each carries a scoreboard token and
//! every reader waits on the token explicitly.
use rayon::prelude::*;

use super::{Ty::*, *};

/// Fragments per tile: `MA` of 8 tokens and `NB` of 16 outputs. With two operand sets this
/// fills the 256-register file exactly.
pub const MA: u32 = 4;
pub const NB: u32 = 4;
/// Tokens and outputs of one tile; callers pad both operands to whole tiles.
pub const TOKENS: usize = 8 * MA as usize;
pub const OUTPUTS: usize = 16 * NB as usize;
/// Bytes a kernel may read past the end of either operand stream (its loads run one
/// iteration ahead); operand buffers carry this much padding.
pub const SLACK: usize = 1 << 16;
/// Token tiles per dispatch block: every output tile passes over one block of tokens before
/// the next block starts. Measured best for all four layer shapes of the DiT (see the engine
/// document); much smaller re-reads the weights too often, much larger leaves the cache.
const BLOCK_TILES: u32 = 100;

const LOAD_32X64: u32 = 0x0248_F580; // 64 dwords at a 64-bit address
const LOAD_64X64: u32 = 0x0288_F780; // 64 qwords
const STORE_64X8: u32 = 0x0200_C784; // 8 qwords: one accumulator row

#[derive(Clone, Copy)]
pub struct Shape {
    pub ma: u32,
    pub nb: u32,
    /// Steps of 32 inputs (even).
    pub ks: u32,
    /// Token tiles in the whole product; threads dispatched beyond it exit at once.
    pub tx: u32,
    /// Bytes between consecutive token rows of the output.
    pub c_row: u32,
    /// Token tiles per dispatch block (group count x); group z selects the block.
    pub xc: u32,
}

impl Shape {
    /// The product of `tokens` rows of `columns` bytes with `rows` weight rows, written to
    /// `[token][row]`. None when the shape does not tile or its offsets leave 32 bits.
    pub fn linear(tokens: usize, rows: usize, columns: usize) -> Option<Shape> {
        let fits = tokens.is_multiple_of(TOKENS)
            && rows.is_multiple_of(OUTPUTS)
            && columns.is_multiple_of(64)
            && tokens > 0
            && rows > 0
            && columns > 0
            && (tokens * columns)
                .max(rows * columns)
                .max(tokens * rows * 4)
                < 1 << 32;
        fits.then(|| Shape {
            ma: MA,
            nb: NB,
            ks: (columns / 32) as u32,
            tx: (tokens / TOKENS) as u32,
            xc: BLOCK_TILES,
            c_row: (rows * 4) as u32,
        })
    }
    /// Thread groups to launch for `rows` weight rows.
    pub fn groups(&self, rows: usize) -> [u32; 3] {
        [
            self.xc,
            (rows / (16 * self.nb as usize)) as u32,
            self.tx.div_ceil(self.xc),
        ]
    }
}

/// Rearranges row-major weights `[rows][columns]` into the kernel's B stream.
pub fn weight_stream(weight: &[u8], rows: usize, columns: usize, to: &mut [u8]) {
    let tile = OUTPUTS * columns;
    assert!(rows.is_multiple_of(OUTPUTS) && columns.is_multiple_of(32));
    assert!(weight.len() == rows * columns && to.len() >= weight.len());
    to[..weight.len()]
        .par_chunks_mut(tile)
        .zip(weight.par_chunks(tile))
        .for_each(|(to, from)| {
            // Output n of the tile is lane n % 16 of fragment n / 16; each lane holds 4 inputs.
            for (n, row) in from.chunks_exact(columns).enumerate() {
                let lane = (n / 16) * 512 + (n % 16) * 4;
                for (step, inputs) in row.chunks_exact(32).enumerate() {
                    let at = step * NB as usize * 512 + lane;
                    for (g, four) in inputs.chunks_exact(4).enumerate() {
                        to[at + g * 64..at + g * 64 + 4].copy_from_slice(four);
                    }
                }
            }
        });
    to[weight.len()..].fill(0);
}

/// Position of 16-byte word `word` of token `token` in the kernel's A stream, in words
/// (the quantizer shader writes this layout directly).
pub fn token_word(token: usize, word: usize, columns: usize) -> usize {
    (((token / TOKENS) * (columns / 32) + word / 2) * TOKENS + token % TOKENS) * 2 + word % 2
}

pub fn kernel(s: &Shape) -> Vec<u8> {
    let Shape {
        ma,
        nb,
        ks,
        tx,
        xc,
        c_row,
    } = *s;
    let (a_step, b_step) = ((ma * 256) as i32, (nb * 512) as i32);
    let set = 4 * ma + 8 * nb;
    let (nl, nacc) = (ma + nb, ma * nb);
    let a = |q: u32, i: u32| 16 + q * set + 4 * i;
    let b = |q: u32, j: u32| 16 + q * set + 4 * ma + 8 * j;
    let c = |n: u32| 16 + 2 * set + 8 * n;
    assert!(c(nacc - 1) + 8 <= 240 && ks % 2 == 0 && 2 * nl + nacc <= 32);
    assert!(
        8 * ma <= 2 * set,
        "output row addresses reuse the operand registers"
    );
    let ltok = |q: u32, t: u32| (q * nl + t) as u16;
    let dtok = |n: u32| (2 * nl + n) as u16;
    let (ar, br) = ([2, 9], [3, 10]); // stream pointers of each register set
    let w = Swsb::after(1);
    let mut k = Asm::new();

    // r0: thread header (group ids in dwords 1, 6, 7); r1: the three stream pointers.
    k.mov(16, r(240, 0, F), r(0, 0, F), Swsb::NONE);
    k.mov(1, r(6, 0, D), imm(a_step * ks as i32), Swsb::NONE);
    k.mov(1, r(6, 1, D), imm(b_step * ks as i32), Swsb::NONE);
    k.mov(1, r(6, 2, D), imm((8 * ma * c_row) as i32), Swsb::NONE);
    k.mov(1, r(6, 3, D), imm((64 * nb) as i32), Swsb::NONE);
    k.mov(1, r(6, 4, D), imm(xc as i32), Swsb::NONE);
    k.mul(1, r(5, 4, D), r(6, 4, D), r(0, 14, Uw), w); // token tile = z·xc + x
    k.add(1, r(5, 4, D), r(5, 4, D), r(0, 1, D), w);
    let exit = k.label();
    k.cmp(Cond::Ge, r(5, 4, D), imm(tx as i32), w);
    k.jmpi(exit, true, w);
    k.mul(1, r(5, 0, D), r(6, 0, D), r(5, 8, Uw), w); // A offset
    k.mul(1, r(5, 1, D), r(6, 1, D), r(0, 12, Uw), w); // B offset
    k.mul(1, r(5, 2, D), r(6, 2, D), r(5, 8, Uw), w); // C offset
    k.mul(1, r(5, 3, D), r(6, 3, D), r(0, 12, Uw), w);
    k.add(1, r(5, 2, D), r(5, 2, D), r(5, 3, D), w);
    for i in 0..3 {
        k.mov(1, r(7, i, Uq), r(5, i, Ud), w);
    }
    for i in 0..3 {
        k.add(1, r(2 + i, 0, Q), r(1, i, Q), r(7, i, Q), w);
    }
    k.add(1, r(9, 0, Q), r(2, 0, Q), imm(-a_step), w);
    k.add(1, r(10, 0, Q), r(3, 0, Q), imm(-b_step), w);
    k.mov(1, r(8, 0, D), imm(0), Swsb::NONE);
    for n in 0..nacc {
        for row in 0..8 {
            k.mov(16, r(c(n) + row, 0, D), imm(0), Swsb::NONE);
        }
    }

    let load = |k: &mut Asm, q: u32| {
        for i in 0..ma {
            k.send(
                SFID_UGM,
                r(a(q, i), 0, Ud),
                r(ar[q as usize], 0, Ud),
                null(Ud),
                0,
                (256 * i) << 12,
                LOAD_32X64,
                false,
                Swsb::after_set(1, ltok(q, i)),
            );
        }
        for j in 0..nb {
            k.send(
                SFID_UGM,
                r(b(q, j), 0, Ud),
                r(br[q as usize], 0, Ud),
                null(Ud),
                0,
                (512 * j) << 12,
                LOAD_64X64,
                false,
                Swsb::after_set(1, ltok(q, ma + j)),
            );
        }
    };
    // Sources are read in program order, so the last multiply having read its sources means
    // all of them have, and the set can be overwritten.
    let advance = |k: &mut Asm, q: u32| {
        k.sync(Swsb::src(dtok(nacc - 1)));
        k.add(
            1,
            r(ar[q as usize], 0, Q),
            r(ar[q as usize], 0, Q),
            imm(2 * a_step),
            Swsb::NONE,
        );
        k.add(
            1,
            r(br[q as usize], 0, Q),
            r(br[q as usize], 0, Q),
            imm(2 * b_step),
            Swsb::NONE,
        );
        load(k, q);
    };
    let multiply = |k: &mut Asm, q: u32, first: bool| {
        for t in 0..nl {
            k.sync(Swsb::dst(ltok(q, t)));
        }
        for n in 0..nacc {
            if !first {
                k.sync(Swsb::dst(dtok(n)));
            }
            k.dpas(
                8,
                8,
                16,
                r(c(n), 0, D),
                r(c(n), 0, D),
                r(b(q, n % nb), 0, B),
                r(a(q, n / nb), 0, B),
                Swsb::set(dtok(n)),
            );
        }
    };

    load(&mut k, 0);
    // First half-iteration peeled, so every later multiply has a predecessor to wait for.
    k.add(1, r(9, 0, Q), r(9, 0, Q), imm(2 * a_step), w);
    k.add(1, r(10, 0, Q), r(10, 0, Q), imm(2 * b_step), w);
    load(&mut k, 1);
    multiply(&mut k, 0, true);
    let (top, entry) = (k.label(), k.label());
    k.jmpi(entry, false, Swsb::NONE);
    k.place(top);
    advance(&mut k, 1);
    multiply(&mut k, 0, false);
    k.place(entry);
    advance(&mut k, 0);
    multiply(&mut k, 1, false);
    k.add(1, r(8, 0, D), r(8, 0, D), imm(2), Swsb::NONE);
    k.cmp(Cond::Lt, r(8, 0, D), imm(ks as i32), w);
    k.jmpi(top, true, w);

    for t in 0..nl {
        k.sync(Swsb::dst(ltok(0, t)));
    }
    for n in 0..nacc {
        k.sync(Swsb::dst(dtok(n)));
    }
    // Row-major output: one address register per token (the operand registers are free now),
    // one 64-byte store per accumulator row. Tokens are reused once their store has finished.
    let addr = |t: u32| 16 + t;
    k.mov(1, r(addr(0), 0, Q), r(4, 0, Q), Swsb::NONE);
    for t in 1..8 * ma {
        k.add(
            1,
            r(addr(t), 0, Q),
            r(addr(t - 1), 0, Q),
            imm(c_row as i32),
            w,
        );
    }
    let mut stores = 0u32;
    for n in 0..nacc {
        for row in 0..8 {
            let id = (stores % 32) as u16;
            if stores >= 32 {
                k.sync(Swsb::dst(id));
            }
            k.send(
                SFID_UGM,
                null(Ud),
                r(addr(n / nb * 8 + row), 0, Ud),
                r(c(n) + row, 0, Ud),
                1,
                (64 * (n % nb)) << 12,
                STORE_64X8,
                false,
                Swsb::after_set(1, id),
            );
            stores += 1;
        }
    }
    for id in 0..32.min(stores) {
        k.sync(Swsb::dst(id as u16));
    }
    k.place(exit);
    k.end(r(240, 0, F));
    k.bytes()
}
