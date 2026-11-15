//! Machine code for Intel Xe2 (Battlemage) written by the engine itself, loaded through Level
//! Zero. The vendor libraries get their speed from generated machine code rather than from a
//! compiler (see the engine document), so the hot kernels are encoded here instruction by
//! instruction.
//!
//! This file is the encoder for the instruction forms those kernels use. Every instruction is
//! 128 bits, uncompacted, with masking disabled. Field positions follow Mesa's Xe2 tables, and
//! the tests hold encodings produced by Mesa's assembler.
pub mod int8;
pub mod level_zero;
pub mod zebin;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Ub = 0,
    Uw = 1,
    Ud = 2,
    Uq = 3,
    B = 4,
    W = 5,
    D = 6,
    Q = 7,
    Hf = 9,
    F = 10,
}

impl Ty {
    fn bytes(self) -> u32 {
        1 << (self as u32 & 3)
    }
    /// Three-bit form used by three-source instructions.
    fn short(self) -> u64 {
        match self {
            Ty::Hf => 1,
            Ty::F => 2,
            t => t as u64,
        }
    }
    fn float(self) -> bool {
        matches!(self, Ty::Hf | Ty::F)
    }
}

/// A register operand: `nr`, byte offset inside the register, element type.
#[derive(Clone, Copy)]
pub struct Reg {
    pub nr: u8,
    pub off: u8,
    pub ty: Ty,
    pub grf: bool,
}

/// General register `r<nr>.<sub>` with `sub` counted in elements of `ty`.
pub fn r(nr: u32, sub: u32, ty: Ty) -> Reg {
    assert!(nr < 256 && sub * ty.bytes() < 64);
    Reg {
        nr: nr as u8,
        off: (sub * ty.bytes()) as u8,
        ty,
        grf: true,
    }
}
pub fn null(ty: Ty) -> Reg {
    Reg {
        nr: 0,
        off: 0,
        ty,
        grf: false,
    }
}

#[derive(Clone, Copy)]
pub enum Src {
    Reg(Reg),
    Imm(i32, Ty),
}
impl From<Reg> for Src {
    fn from(v: Reg) -> Src {
        Src::Reg(v)
    }
}
pub fn imm(v: i32) -> Src {
    Src::Imm(v, Ty::D)
}

/// Dependency annotation (bits 17..8). The hardware does no dependency tracking of its own.
#[derive(Clone, Copy)]
pub struct Swsb(u16);
impl Swsb {
    pub const NONE: Swsb = Swsb(0);
    /// Wait for the instruction `dist` back, whatever pipe it ran in (`{A@dist}`).
    pub fn after(dist: u16) -> Swsb {
        assert!(dist < 8);
        Swsb(8 | dist)
    }
    /// Out-of-order instruction allocating token `id` (`{$id}`).
    pub fn set(id: u16) -> Swsb {
        assert!(id < 32);
        Swsb(0xc0 | id)
    }
    /// Send allocating token `id` after waiting for the instruction `dist` back (`{A@dist,$id}`).
    pub fn after_set(dist: u16, id: u16) -> Swsb {
        assert!(dist < 8 && id < 32);
        Swsb(1 << 8 | dist << 5 | id)
    }
    /// Wait until the token's instruction has written its destination (`{$id.dst}`).
    pub fn dst(id: u16) -> Swsb {
        assert!(id < 32);
        Swsb(0x80 | id)
    }
    /// Wait until the token's instruction has read its sources (`{$id.src}`).
    pub fn src(id: u16) -> Swsb {
        assert!(id < 32);
        Swsb(0xa0 | id)
    }
}

#[derive(Clone, Copy)]
pub enum Cond {
    Eq = 1,
    Ne = 2,
    Gt = 3,
    Ge = 4,
    Lt = 5,
    Le = 6,
}

pub const SFID_GATEWAY: u64 = 3;
pub const SFID_UGM: u64 = 15;

#[derive(Clone, Copy)]
pub struct Label(usize);

#[derive(Default)]
pub struct Asm {
    code: Vec<[u64; 2]>,
    labels: Vec<Option<usize>>,
    jumps: Vec<(usize, usize)>,
}

struct Inst([u64; 2]);
impl Inst {
    fn new(opcode: u64, exec: u32, swsb: Swsb) -> Inst {
        let mut i = Inst([0, 0]);
        i.set(6, 0, opcode);
        i.set(17, 8, swsb.0 as u64);
        i.set(20, 18, exec.trailing_zeros() as u64);
        i.set(31, 31, 1); // no write mask
        i
    }
    fn set(&mut self, hi: u32, lo: u32, v: u64) {
        assert!(
            hi / 64 == lo / 64 && v >> (hi - lo + 1) == 0,
            "field {hi}..{lo} value {v:#x}"
        );
        self.0[(lo / 64) as usize] |= v << (lo % 64);
    }
    /// Fourteen-bit register field: file, sub-register in words, number.
    fn operand(&mut self, lo: u32, reg: Reg, subreg: bool) {
        self.set(lo, lo, reg.grf as u64);
        if subreg {
            self.set(lo + 5, lo + 1, reg.off as u64 >> 1);
        }
        self.set(lo + 13, lo + 6, reg.nr as u64);
    }
    fn dst(&mut self, d: Reg) {
        self.set(39, 36, d.ty as u64);
        self.set(49, 48, 1);
        self.operand(50, d, true);
        self.set(33, 33, d.off as u64 & 1);
    }
    /// Scalar sources replicate one element; vector sources read consecutive elements.
    fn src0(&mut self, s: Src, exec: u32) {
        match s {
            Src::Imm(v, ty) => {
                self.set(43, 40, ty as u64);
                self.set(46, 46, 1);
                self.set(127, 96, v as u32 as u64);
            }
            Src::Reg(reg) => {
                self.set(43, 40, reg.ty as u64);
                self.operand(66, reg, true);
                self.set(87, 87, reg.off as u64 & 1);
                if exec > 1 {
                    self.set(86, 84, 1);
                }
            }
        }
    }
    fn src1(&mut self, s: Src, exec: u32) {
        match s {
            Src::Imm(v, ty) => {
                self.set(91, 88, ty as u64);
                self.set(47, 47, 1);
                self.set(127, 96, v as u32 as u64);
            }
            Src::Reg(reg) => {
                assert!(reg.off & 1 == 0);
                self.set(91, 88, reg.ty as u64);
                self.operand(98, reg, true);
                if exec > 1 {
                    self.set(118, 116, 1);
                }
            }
        }
    }
}

impl Asm {
    pub fn new() -> Asm {
        Asm::default()
    }
    fn push(&mut self, i: Inst) {
        self.code.push(i.0);
    }

    fn two(&mut self, opcode: u64, exec: u32, d: Reg, a: Src, b: Src, swsb: Swsb) -> Inst {
        assert!(!(matches!(a, Src::Imm(..)) && matches!(b, Src::Imm(..))));
        let mut i = Inst::new(opcode, exec, swsb);
        i.dst(d);
        i.src0(a, exec);
        i.src1(b, exec);
        i
    }
    pub fn mov(&mut self, exec: u32, d: Reg, a: impl Into<Src>, swsb: Swsb) {
        let mut i = Inst::new(97, exec, swsb);
        i.dst(d);
        i.src0(a.into(), exec);
        self.push(i);
    }
    pub fn add(&mut self, exec: u32, d: Reg, a: impl Into<Src>, b: impl Into<Src>, swsb: Swsb) {
        let i = self.two(64, exec, d, a.into(), b.into(), swsb);
        self.push(i);
    }
    pub fn mul(&mut self, exec: u32, d: Reg, a: impl Into<Src>, b: impl Into<Src>, swsb: Swsb) {
        let i = self.two(65, exec, d, a.into(), b.into(), swsb);
        self.push(i);
    }
    /// Compare into flag f0.0.
    pub fn cmp(&mut self, cond: Cond, a: impl Into<Src>, b: impl Into<Src>, swsb: Swsb) {
        let a = a.into();
        let ty = match a {
            Src::Reg(reg) => reg.ty,
            Src::Imm(_, ty) => ty,
        };
        let mut i = self.two(112, 1, null(ty), a, b.into(), swsb);
        i.set(95, 92, cond as u64);
        self.push(i);
    }
    /// Wait for a dependency without doing anything else.
    pub fn sync(&mut self, swsb: Swsb) {
        let mut i = Inst::new(1, 1, swsb);
        i.set(43, 40, Ty::Ud as u64);
        i.set(86, 84, 1);
        self.push(i);
    }
    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() - 1)
    }
    pub fn place(&mut self, l: Label) {
        assert!(self.labels[l.0].is_none());
        self.labels[l.0] = Some(self.code.len());
    }
    /// Jump; with `if_flag` only when f0.0 is set.
    pub fn jmpi(&mut self, to: Label, if_flag: bool, swsb: Swsb) {
        let mut i = Inst::new(32, 1, swsb);
        i.set(27, 26, if_flag as u64);
        i.set(46, 46, 1);
        self.jumps.push((self.code.len(), to.0));
        self.push(i);
    }
    /// Message to a shared function. `dst`/`payload`/`extra` name the first register of each
    /// span; `extra_len` is the length of the second payload in registers.
    #[allow(clippy::too_many_arguments)]
    pub fn send(
        &mut self,
        sfid: u64,
        dst: Reg,
        payload: Reg,
        extra: Reg,
        extra_len: u64,
        ex_desc: u32,
        desc: u32,
        eot: bool,
        swsb: Swsb,
    ) {
        let (d, x) = (desc as u64, ex_desc as u64);
        assert!(
            x & 0x7ff == 0,
            "low descriptor bits carry the extra payload length"
        );
        let mut i = Inst::new(49, 1, swsb);
        i.set(34, 34, eot as u64);
        i.set(95, 92, sfid);
        i.operand(50, dst, false);
        i.operand(66, payload, false);
        i.operand(98, extra, false);
        i.set(103, 99, extra_len);
        i.set(123, 122, d >> 30);
        i.set(71, 67, d >> 25 & 0x1f);
        i.set(55, 51, d >> 20 & 0x1f);
        i.set(121, 113, d >> 11 & 0x1ff);
        i.set(91, 81, d & 0x7ff);
        i.set(127, 124, x >> 28);
        i.set(97, 96, x >> 26 & 3);
        i.set(65, 64, x >> 24 & 3);
        i.set(47, 35, x >> 11 & 0x1fff);
        self.push(i);
    }
    /// Systolic dot-product accumulate: `dst = acc + Σ b·a` over `depth` k-groups and `rows` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn dpas(
        &mut self,
        depth: u32,
        rows: u32,
        exec: u32,
        dst: Reg,
        acc: Reg,
        b: Reg,
        a: Reg,
        swsb: Swsb,
    ) {
        let sdepth = match depth {
            8 => 3,
            4 => 2,
            2 => 1,
            _ => panic!("depth"),
        };
        let mut i = Inst::new(89, exec, swsb);
        i.set(45, 43, rows as u64 - 1);
        i.set(49, 48, sdepth);
        i.set(39, 39, dst.ty.float() as u64);
        i.set(38, 36, dst.ty.short());
        i.set(42, 40, acc.ty.short());
        i.set(90, 88, b.ty.short());
        i.set(82, 80, a.ty.short());
        i.operand(50, dst, true);
        i.operand(66, acc, true);
        i.operand(98, b, true);
        i.operand(114, a, true);
        self.push(i);
    }
    /// End the thread, handing the saved thread header back to the gateway.
    pub fn end(&mut self, header: Reg) {
        self.send(
            SFID_GATEWAY,
            null(Ty::Ud),
            header,
            null(Ty::Ud),
            0,
            0,
            0x0200_0010,
            true,
            Swsb::after(1),
        );
    }

    pub fn bytes(mut self) -> Vec<u8> {
        for &(at, label) in &self.jumps {
            let to = self.labels[label].expect("label never placed");
            let delta = (to as i64 - at as i64) * 16;
            self.code[at][1] |= (delta as i32 as u32 as u64) << 32;
        }
        self.code
            .iter()
            .flat_map(|w| [w[0].to_le_bytes(), w[1].to_le_bytes()])
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Ty::*, *};

    fn hex(text: &str) -> Vec<u8> {
        text.split_whitespace()
            .map(|b| u8::from_str_radix(b, 16).unwrap())
            .collect()
    }
    fn one(build: impl FnOnce(&mut Asm)) -> Vec<u8> {
        let mut k = Asm::new();
        build(&mut k);
        k.bytes()
    }

    /// Each expected encoding is Mesa's assembler output for the same instruction.
    #[test]
    fn encodings_match_the_reference_assembler() {
        let w = Swsb::after(1);
        assert_eq!(
            one(|k| k.mov(16, r(240, 0, F), r(0, 0, F), Swsb::NONE)),
            hex("61 00 10 80 a0 0a 05 f0 04 00 10 00 00 00 00 00")
        );
        assert_eq!(
            one(|k| k.mov(1, r(6, 0, D), imm(172_032), Swsb::NONE)),
            hex("61 00 00 80 60 46 05 06 00 00 00 00 00 a0 02 00")
        );
        assert_eq!(
            one(|k| k.mov(1, r(7, 0, Uq), r(5, 0, Ud), w)),
            hex("61 09 00 80 30 02 05 07 04 05 00 00 00 00 00 00")
        );
        assert_eq!(
            one(|k| k.mul(1, r(5, 4, D), r(6, 4, D), r(0, 14, Uw), w)),
            hex("41 09 00 80 60 06 45 05 44 06 00 01 74 00 00 00")
        );
        assert_eq!(
            one(|k| k.add(1, r(2, 0, Q), r(1, 0, Q), r(7, 0, Q), w)),
            hex("40 09 00 80 70 07 05 02 04 01 00 07 04 07 00 00")
        );
        assert_eq!(
            one(|k| k.cmp(Cond::Lt, r(8, 0, D), imm(168), w)),
            hex("70 09 00 80 60 86 01 00 04 08 00 56 a8 00 00 00")
        );
        assert_eq!(
            one(|k| k.cmp(Cond::Ge, r(5, 4, D), imm(272), w)),
            hex("70 09 00 80 60 86 01 00 44 05 00 46 10 01 00 00")
        );
        assert_eq!(
            one(|k| k.sync(Swsb::dst(0))),
            hex("01 80 00 80 00 02 00 00 00 00 10 00 00 00 00 00")
        );
        assert_eq!(
            one(|k| k.send(
                SFID_UGM,
                r(16, 0, Ud),
                r(2, 0, Ud),
                null(Ud),
                0,
                0,
                0x0248_F580,
                false,
                Swsb::after_set(1, 0)
            )),
            hex("31 20 01 80 00 00 24 10 0c 02 00 fb 00 00 3c 02")
        );
        assert_eq!(
            one(|k| k.send(
                SFID_UGM,
                null(Ud),
                r(4, 0, Ud),
                r(112, 0, Ud),
                1,
                0,
                0x0200_C784,
                false,
                Swsb::after_set(1, 3)
            )),
            hex("31 23 01 80 00 00 00 00 0c 04 08 ff 0c 70 30 00")
        );
        assert_eq!(
            one(|k| k.dpas(
                8,
                8,
                16,
                r(112, 0, D),
                r(112, 0, D),
                r(32, 0, B),
                r(16, 0, B),
                Swsb::set(16)
            )),
            hex("59 d0 10 80 60 3e 07 70 04 70 04 04 04 20 04 10")
        );
    }

    #[test]
    fn jumps_are_byte_distances_from_the_jump() {
        let mut k = Asm::new();
        let (top, end) = (k.label(), k.label());
        k.place(top);
        k.jmpi(end, false, Swsb::NONE);
        k.sync(Swsb::dst(1));
        k.jmpi(top, true, Swsb::after(1));
        k.place(end);
        let code = k.bytes();
        assert_eq!(
            code[..16],
            hex("20 00 00 80 00 40 00 00 00 00 00 00 30 00 00 00")[..]
        );
        assert_eq!(
            code[32..],
            hex("20 09 00 84 00 40 00 00 00 00 00 00 e0 ff ff ff")[..]
        );
    }
}
