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
    /// ids [used], weights [used], x [n] (the route layout); for a small batch every
    /// token's ids and weights, then every token's x.
    input: Mapped,
    /// A small batch's slot map (per expert its record in a `Tier`, or none), and its copy.
    map_in: Mapped,
    map: Buffer,
    ge: Buffer,
    gh: Buffer,
    out: Mapped,
    used: usize,
    n: usize,
    ff: usize,
    experts: usize,
}

/// A block's experts held on the second device for small batches: whole records (gate|up
/// then down, weights then scales, as the host keeps them), copies of host experts (the
/// host keeps its own, so a held expert can still run anywhere).
pub struct Tier {
    pub records: Buffer,
    /// Per expert its record here, or u32::MAX.
    pub held: Vec<u32>,
}

impl Second {
    /// For `tokens` tokens at once at most (`experts` per block).
    pub fn new(
        gpu: Gpu,
        used: usize,
        n: usize,
        ff: usize,
        experts: usize,
        tokens: usize,
    ) -> Result<Self> {
        let t = tokens.max(1);
        let lanes = gpu.subgroup().unwrap_or(32).clamp(8, 128);
        let mut grid: Vec<u32> = Vec::with_capacity(2560);
        for g in kuno_text::quant::IQ2_S_GRID {
            grid.push(g as u32);
            grid.push((g >> 32) as u32);
        }
        grid.extend(kuno_text::quant::IQ3_S_GRID);
        Ok(Self {
            mve: gpu.kernel(include_bytes!("../shaders/mve.spv"), 7, 10, Some(lanes))?,
            glu: gpu.kernel(include_bytes!("../shaders/glu.spv"), 2, 3, Some(lanes))?,
            copy: gpu.kernel(include_bytes!("../shaders/copy.spv"), 2, 1, Some(lanes))?,
            local: gpu.buffer((t * (2 * used + n) * 4) as u64)?,
            tables: filled(&gpu, bytes_of(&grid))?,
            input: gpu.mapped((t * (2 * used + n) * 4) as u64)?,
            map_in: gpu.mapped((experts * 4) as u64)?,
            map: gpu.buffer((experts * 4) as u64)?,
            ge: gpu.buffer((t * used * 2 * ff * 4) as u64)?,
            gh: gpu.buffer((t * used * ff * 4) as u64)?,
            out: gpu.mapped((t * n * 4) as u64)?,
            experts,
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
        let c0 = [
            0,
            experts::kind(gate_kind)?,
            2 * ff,
            n,
            used,
            x_off,
            0,
            0,
            0,
            0,
        ];
        let cg = [ff, limit.to_bits(), used];
        let c1 = [1, experts::kind(down_kind)?, n, ff, used, 0, 0, 0, 0, 0];
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

impl Second {
    /// A tier of `ids`' records (`record(e)`: expert e's record bytes), uploaded and checked.
    pub fn tier<'a>(&self, ids: &[usize], record: impl Fn(usize) -> &'a [u8]) -> Result<Tier> {
        let mut held = vec![u32::MAX; self.experts];
        let mut bytes = Vec::new();
        for (k, &e) in ids.iter().enumerate() {
            held[e] = k as u32;
            bytes.extend_from_slice(record(e));
        }
        Ok(Tier {
            records: filled(&self.gpu, &bytes)?,
            held,
        })
    }

    /// Starts a small batch's experts held in `tier` (`map`: per expert its record there for
    /// those to run here, else u32::MAX; `choice`: every token's ids and weights, `xs` every
    /// token's input; `st` the records' words of gu_w, gu_s, down_w, down_s); `finish_small`
    /// gives every token's sum.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_small(
        &self,
        tier: &Tier,
        kinds: (u32, u32),
        st: [u32; 4],
        map: &[u32],
        choice: &[f32],
        xs: &[f32],
        t: usize,
        limit: f32,
    ) -> Result<Passing> {
        if choice.len() != t * 2 * self.used || xs.len() != t * self.n || map.len() != self.experts
        {
            return Err("model_state_invalid");
        }
        self.gpu.write_mapped(&self.input, 0, choice)?;
        self.gpu
            .write_mapped(&self.input, (choice.len() * 4) as u64, xs)?;
        self.gpu.write_mapped(
            &self.map_in,
            0,
            &kuno_image::engine::gpu::floats(bytes_of(map)),
        )?;
        let (used, n, ff, tu) = (self.used as u32, self.n as u32, self.ff as u32, t as u32);
        let rpg = 128 / self.lanes;
        let rec: u32 = st.iter().sum();
        let c0 = [
            0,
            kinds.0,
            2 * ff,
            n,
            used,
            tu * 2 * used,
            rec,
            0,
            rec,
            st[0],
        ];
        let cg = [ff, limit.to_bits(), used * tu];
        let c1 = [
            1,
            kinds.1,
            n,
            ff,
            used,
            0,
            rec,
            st[0] + st[1],
            rec,
            st[0] + st[1] + st[2],
        ];
        let total = (t * (2 * self.used + self.n)) as u32;
        let cc = [total];
        let cm = [self.experts as u32];
        let bc = [self.input.buffer(), &self.local];
        let bm = [self.map_in.buffer(), &self.map];
        let r = &tier.records;
        let b0 = [
            r,
            r,
            &self.local,
            &self.ge,
            &self.local,
            &self.map,
            &self.tables,
        ];
        let bg = [&self.ge, &self.gh];
        let b1 = [
            r,
            r,
            &self.gh,
            self.out.buffer(),
            &self.local,
            &self.map,
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
                kernel: &self.copy,
                buffers: &bm,
                constants: &cm,
                groups: [(self.experts as u32).div_ceil(256), 1, 1],
            },
            Step {
                kernel: &self.mve,
                buffers: &b0,
                constants: &c0,
                groups: [(2 * ff).div_ceil(rpg), used, tu],
            },
            Step {
                kernel: &self.glu,
                buffers: &bg,
                constants: &cg,
                groups: [(tu * used * ff).div_ceil(256), 1, 1],
            },
            Step {
                kernel: &self.mve,
                buffers: &b1,
                constants: &c1,
                groups: [n.div_ceil(rpg), 1, tu],
            },
        ];
        // SAFETY: every buffer is this device's or the tier's, both alive until the pass is
        // waited for (`finish_small`).
        unsafe {
            self.gpu
                .pass_begin(&steps, &[true, false, false, false, false])
        }
    }

    /// Waits for `begin_small`'s pass; adds token i's sum into `sums[i * n..]`.
    pub fn finish_small(&self, passing: Passing, sums: &mut [f32]) -> Result<()> {
        passing.wait()?;
        let mut v = vec![0f32; sums.len()];
        self.gpu.read_mapped(&self.out, 0, &mut v)?;
        for (s, x) in sums.iter_mut().zip(&v) {
            *s += x;
        }
        Ok(())
    }
}
