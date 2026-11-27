//! A second GPU for the next tier of experts (the RTX 3060 beside the B70): per MoE block the
//! host hands it the router's choice and the block's input, it runs its experts while the
//! CPU runs the rest, and the host adds its sum to the CPU's.
use kuno_image::engine::gpu::{Buffer, Gpu, Kernel, Mapped, Passing, Step, bytes_of};

use crate::Result;
use crate::experts::{self, DevicePool};
use crate::gpu::filled;

pub struct Second {
    pub gpu: Gpu,
    lanes: u32,
    mve: Kernel,
    glu: Kernel,
    copy: Kernel,
    /// The input brought into device memory.
    local: Buffer,
    tables: Buffer,
    /// ids [used], weights [used], x [n] (the route layout).
    input: Mapped,
    ge: Buffer,
    gh: Buffer,
    out: Mapped,
    used: usize,
    n: usize,
    ff: usize,
}

impl Second {
    pub fn new(gpu: Gpu, used: usize, n: usize, ff: usize) -> Result<Self> {
        let lanes = gpu.subgroup().unwrap_or(32).clamp(8, 128);
        let mut t: Vec<u32> = Vec::with_capacity(2560);
        for g in kuno_text::quant::IQ2_S_GRID {
            t.push(g as u32);
            t.push((g >> 32) as u32);
        }
        t.extend(kuno_text::quant::IQ3_S_GRID);
        Ok(Self {
            mve: gpu.kernel(include_bytes!("../shaders/mve.spv"), 7, 6, Some(lanes))?,
            glu: gpu.kernel(include_bytes!("../shaders/glu.spv"), 2, 3, Some(lanes))?,
            copy: gpu.kernel(include_bytes!("../shaders/copy.spv"), 2, 1, Some(lanes))?,
            local: gpu.buffer(((2 * used + n) * 4) as u64)?,
            tables: filled(&gpu, bytes_of(&t))?,
            input: gpu.mapped(((2 * used + n) * 4) as u64)?,
            ge: gpu.buffer((used * 2 * ff * 4) as u64)?,
            gh: gpu.buffer((used * ff * 4) as u64)?,
            out: gpu.mapped((n * 4) as u64)?,
            lanes,
            used,
            n,
            ff,
            gpu,
        })
    }

    /// Starts this device's experts among the routed ones (`route`: ids, weights, x as
    /// floats); `finish` adds their sum into `sum`.
    pub fn begin(
        &self,
        pool: &DevicePool,
        gate_kind: kuno_text::TensorType,
        down_kind: kuno_text::TensorType,
        route: &[f32],
        limit: f32,
    ) -> Result<Passing> {
        self.gpu.write_mapped(&self.input, 0, route)?;
        let (used, n, ff) = (self.used as u32, self.n as u32, self.ff as u32);
        let rpg = 128 / self.lanes;
        let x_off = (2 * self.used) as u32;
        let c0 = [0, experts::kind(gate_kind)?, 2 * ff, n, used, x_off];
        let cg = [ff, limit.to_bits(), used];
        let c1 = [1, experts::kind(down_kind)?, n, ff, used, 0];
        let total = (2 * self.used + self.n) as u32;
        let cc = [total];
        let bc = [self.input.buffer(), &self.local];
        let input = &self.local;
        let b0 = [
            &pool.gu_w,
            &pool.gu_s,
            input,
            &self.ge,
            input,
            &pool.slots,
            &self.tables,
        ];
        let bg = [&self.ge, &self.gh];
        let b1 = [
            &pool.down_w,
            &pool.down_s,
            &self.gh,
            self.out.buffer(),
            input,
            &pool.slots,
            &self.tables,
        ];
        let steps = [
            Step {
                kernel: &self.copy,
                buffers: &bc,
                constants: &cc,
                groups: [total.div_ceil(256), 1, 1],
            },
            Step {
                kernel: &self.mve,
                buffers: &b0,
                constants: &c0,
                groups: [(2 * ff).div_ceil(rpg), used, 1],
            },
            Step {
                kernel: &self.glu,
                buffers: &bg,
                constants: &cg,
                groups: [(used * ff).div_ceil(256), 1, 1],
            },
            Step {
                kernel: &self.mve,
                buffers: &b1,
                constants: &c1,
                groups: [n.div_ceil(rpg), 1, 1],
            },
        ];
        // SAFETY: every buffer is this device's or the pool's, both alive until `finish`.
        unsafe { self.gpu.pass_begin(&steps, &[false, false, false, false]) }
    }

    pub fn finish(&self, passing: Passing, sum: &mut [f32]) -> Result<()> {
        passing.wait()?;
        let mut v = vec![0f32; self.n];
        self.gpu.read_mapped(&self.out, 0, &mut v)?;
        for (s, x) in sum.iter_mut().zip(&v) {
            *s += x;
        }
        Ok(())
    }
}
