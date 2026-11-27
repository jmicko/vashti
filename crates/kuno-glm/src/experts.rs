//! Where the routed experts live: a bank of slots per MoE block on the GPU (`mve.comp`), the
//! rest in host memory for the CPU (`cpu`). As on the Qwen engine, the banks adapt: every
//! block scores its experts by recent use (a count decaying with a half-life of
//! `HALF_LIFE` tokens, in memory only: nothing about the requests is kept or written),
//! and an expert the CPU keeps running is traded onto the device for the coldest resident
//! one not in use, the victim copied back to host memory (so memory use stays flat). The
//! copies run on the device while the CPU computes. Banks start with experts in index order.
use std::sync::Mutex;

use kuno_image::engine::gpu::{Buffer, Gpu, Mapped};
use kuno_text::TensorType;
use rayon::prelude::*;

use crate::gpu::filled;
use crate::{Model, Result, cpu};

/// Tokens for a use's weight to halve.
pub const HALF_LIFE: f32 = 64.0;

/// One MoE block's experts.
pub struct BlockExperts {
    pub gate_kind: TensorType,
    pub down_kind: TensorType,
    /// Per expert: its slot on the main device, if any.
    pub slot: Vec<Option<u32>>,
    /// Device pools (when any expert is on the device): gate|up rows and down rows.
    pub device: Option<DevicePool>,
    /// The same on the second device (fixed).
    pub slot2: Vec<Option<u32>>,
    pub device2: Option<DevicePool>,
    /// Host copies in the device's layout, in host-visible chunks the device copies to and
    /// from directly: per expert not on the device its host slot; one spare slot.
    pub arena: Vec<Mapped>,
    pub per_chunk: usize,
    pub host_slot: Vec<Option<u32>>,
    pub spare: u32,
    /// Words per expert of gu_w, gu_s, down_w, down_s (gu_s, down_s 0 for IQ4_XS).
    pub stride: [usize; 4],
    pub gate_row: usize,
    pub down_row: usize,
    /// Per main-device slot, the expert in it.
    pub resident: Vec<Option<usize>>,
    /// Recent use per expert: (score, token it was last brought up to date).
    pub uses: Mutex<Vec<(f32, u64)>>,
}

pub struct DevicePool {
    pub gu_w: Buffer,
    pub gu_s: Buffer,
    pub down_w: Buffer,
    pub down_s: Buffer,
    /// slot per expert (0xffffffff: on the CPU), for the kernel.
    pub slots: Buffer,
    /// Words per slot in each of gu_w, gu_s, down_w, down_s (gu_s/down_s 0 for IQ4_XS).
    pub stride: [usize; 4],
    pub capacity: usize,
}

/// The kernel's kind for an expert type.
pub fn kind(t: TensorType) -> Result<u32> {
    match t {
        TensorType::Iq2S => Ok(0),
        TensorType::Iq3S => Ok(1),
        TensorType::Iq4Xs => Ok(2),
        _ => Err("expert_type_unsupported"),
    }
}

fn block_bytes(t: TensorType) -> usize {
    match t {
        TensorType::Iq2S => 82,
        TensorType::Iq3S => 110,
        _ => 136,
    }
}

/// Raw rows as the kernel's words and f16 block scales.
fn repack(t: TensorType, raw: &[u8], w: &mut Vec<u8>, s: &mut Vec<u8>) {
    match t {
        TensorType::Iq2S | TensorType::Iq3S => {
            for b in raw.chunks_exact(block_bytes(t)) {
                s.extend_from_slice(&b[..2]);
                w.extend_from_slice(&b[2..]);
            }
        }
        _ => w.extend_from_slice(raw),
    }
}

/// The inverse of `repack`.
fn unpack(t: TensorType, w: &[u8], s: &[u8], raw: &mut Vec<u8>) {
    match t {
        TensorType::Iq2S | TensorType::Iq3S => {
            let body = block_bytes(t) - 2;
            for (b, d) in w.chunks_exact(body).zip(s.chunks_exact(2)) {
                raw.extend_from_slice(d);
                raw.extend_from_slice(b);
            }
        }
        _ => raw.extend_from_slice(w),
    }
}

/// An expert in the device's layout: gu_w, gu_s, down_w, down_s bytes.
pub fn device_layout(
    gk: TensorType,
    dk: TensorType,
    raw: &[u8],
    ff: usize,
    gr: usize,
) -> [Vec<u8>; 4] {
    let (mut gw, mut gs, mut dw, mut ds) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    repack(gk, &raw[..2 * ff * gr], &mut gw, &mut gs);
    repack(dk, &raw[2 * ff * gr..], &mut dw, &mut ds);
    [gw, gs, dw, ds]
}

/// An expert back from the device's layout to raw rows.
pub fn host_layout(gk: TensorType, dk: TensorType, parts: [&[u8]; 4]) -> Vec<u8> {
    let mut raw = Vec::new();
    unpack(gk, parts[0], parts[1], &mut raw);
    unpack(dk, parts[2], parts[3], &mut raw);
    raw
}

/// A block's expert `e` as raw rows from the file.
fn read_expert(model: &Model, il: usize, e: usize) -> Result<Vec<u8>> {
    let hp = &model.hp;
    let (n, ff) = (hp.embd, hp.ff_expert);
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (_, gr, _) = model.layout(&name("gate"))?;
    let (_, dr, _) = model.layout(&name("down"))?;
    let mut v = vec![0u8; 2 * ff * gr + n * dr];
    let (g, rest) = v.split_at_mut(ff * gr);
    let (u, d) = rest.split_at_mut(ff * gr);
    model.read_rows(&name("gate"), e * ff, g)?;
    model.read_rows(&name("up"), e * ff, u)?;
    model.read_rows(&name("down"), e * n, d)?;
    Ok(v)
}

/// One block's bank of `capacity` slots on `gpu`, the first holding `ids`.
fn pool(
    model: &Model,
    gpu: &Gpu,
    il: usize,
    capacity: usize,
    ids: &[usize],
    slot: &mut [Option<u32>],
) -> Result<Option<DevicePool>> {
    if capacity == 0 {
        return Ok(None);
    }
    let hp = &model.hp;
    let ff = hp.ff_expert;
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (gk, gr, _) = model.layout(&name("gate"))?;
    let (dk, _, _) = model.layout(&name("down"))?;
    let mut parts: [Vec<u8>; 4] = Default::default();
    let mut stride = [0usize; 4];
    for (k, &e) in ids.iter().enumerate() {
        slot[e] = Some(k as u32);
        let raw = read_expert(model, il, e)?;
        let p = device_layout(gk, dk, &raw, ff, gr);
        for i in 0..4 {
            stride[i] = p[i].len() / 4;
            parts[i].extend_from_slice(&p[i]);
        }
    }
    if ids.is_empty() {
        return Err("expert_bank_empty");
    }
    // The empty slots, zeroed.
    for (part, st) in parts.iter_mut().zip(&stride) {
        part.resize(capacity * st * 4, 0);
    }
    let slots: Vec<u32> = slot.iter().map(|s| s.unwrap_or(u32::MAX)).collect();
    Ok(Some(DevicePool {
        gu_w: filled(gpu, &parts[0])?,
        gu_s: filled(gpu, &parts[1])?,
        down_w: filled(gpu, &parts[2])?,
        down_s: filled(gpu, &parts[3])?,
        slots: filled(gpu, kuno_image::engine::gpu::bytes_of(&slots))?,
        stride,
        capacity,
    }))
}

/// Places every MoE block's experts: the same number of slots per block filling
/// `device_bytes` on `gpu`, then the second device's budget there (fixed), the rest in memory.
pub fn place(
    model: &Model,
    gpu: &Gpu,
    device_bytes: u64,
    second: Option<(&Gpu, u64)>,
) -> Result<Vec<Option<BlockExperts>>> {
    let hp = &model.hp;
    let (n, ff) = (hp.embd, hp.ff_expert);
    let name = |il: usize, p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let mut per_slot = 0u64;
    for il in hp.dense_lead..hp.layers {
        let (_, gr, _) = model.layout(&name(il, "gate"))?;
        let (_, dr, _) = model.layout(&name(il, "down"))?;
        per_slot += (2 * ff * gr + n * dr) as u64;
    }
    let slots1 = ((device_bytes / per_slot.max(1)) as usize).min(hp.experts);
    let slots2 = ((second.map_or(0, |s| s.1) / per_slot.max(1)) as usize).min(hp.experts - slots1);
    let mut out = Vec::with_capacity(hp.layers);
    for il in 0..hp.layers {
        if il < hp.dense_lead {
            out.push(None);
            continue;
        }
        let (gk, gr, _) = model.layout(&name(il, "gate"))?;
        let (dk, dr, _) = model.layout(&name(il, "down"))?;
        cpu::check(gk)?;
        cpu::check(dk)?;
        let mut slot = vec![None; hp.experts];
        let mut slot2 = vec![None; hp.experts];
        let ids1: Vec<usize> = (0..slots1).collect();
        let ids2: Vec<usize> = (slots1..slots1 + slots2).collect();
        let device = pool(model, gpu, il, slots1, &ids1, &mut slot)?;
        let device2 = match second {
            Some((g, _)) => pool(model, g, il, slots2, &ids2, &mut slot2)?,
            None => None,
        };
        // The rest into host-visible chunks (plus one spare slot), read in parallel.
        let stride = {
            let raw = read_expert(model, il, 0)?;
            device_layout(gk, dk, &raw, ff, gr).map(|p| p.len() / 4)
        };
        let slot_bytes = stride.iter().sum::<usize>() * 4;
        let on_host: Vec<usize> = if std::env::var_os("KUNO_GLM_NO_HOST").is_some() {
            Vec::new() // device profiling (with KUNO_GLM_SKIP_EXPERTS)
        } else {
            (0..hp.experts)
                .filter(|e| slot[*e].is_none() && slot2[*e].is_none())
                .collect()
        };
        let per_chunk = ((1usize << 30) / slot_bytes).max(1);
        let total = on_host.len() + 1;
        let mut arena = Vec::new();
        for c in 0..total.div_ceil(per_chunk) {
            let n = per_chunk.min(total - c * per_chunk);
            arena.push(gpu.mapped((n * slot_bytes) as u64)?);
        }
        let mut host_slot = vec![None; hp.experts];
        for (h, &e) in on_host.iter().enumerate() {
            host_slot[e] = Some(h as u32);
        }
        for (c, chunk) in arena.iter().enumerate() {
            // SAFETY: no pass uses the chunk yet; this is its only slice.
            let bytes = unsafe { chunk.bytes_mut() };
            bytes
                .par_chunks_exact_mut(slot_bytes)
                .enumerate()
                .try_for_each(|(k, out)| -> Result<()> {
                    let Some(&e) = on_host.get(c * per_chunk + k) else {
                        return Ok(());
                    };
                    let raw = read_expert(model, il, e)?;
                    let mut at = 0;
                    for part in device_layout(gk, dk, &raw, ff, gr) {
                        out[at..at + part.len()].copy_from_slice(&part);
                        at += part.len();
                    }
                    Ok(())
                })?;
        }
        let mut resident = vec![None; slots1];
        for (k, e) in ids1.iter().enumerate() {
            resident[k] = Some(*e);
        }
        out.push(Some(BlockExperts {
            gate_kind: gk,
            down_kind: dk,
            slot,
            device,
            slot2,
            device2,
            arena,
            per_chunk,
            host_slot,
            spare: on_host.len() as u32,
            stride,
            gate_row: gr,
            down_row: dr,
            resident,
            uses: Mutex::new(vec![(0.0, 0); hp.experts]),
        }));
    }
    eprintln!(
        "experts: {slots1} slots per block on the device ({:.1} GB), {slots2} on the second",
        (slots1 as u64 * per_slot) as f64 / 1e9
    );
    Ok(out)
}

/// A planned trade: expert `up` into device slot `slot` of block `block`, `down` (if any)
/// out into the spare host slot `spare`.
#[derive(Clone, Copy, Debug)]
pub struct Trade {
    pub block: usize,
    pub up: usize,
    pub down: Option<usize>,
    pub slot: usize,
    pub spare: u32,
}

impl BlockExperts {
    /// Counts this token's choice and plans at most one trade: the CPU-run expert with the
    /// highest score up, over the coldest resident not chosen now (or an empty slot), when
    /// it is used clearly more (by `margin`).
    pub fn observe(&self, block: usize, ids: &[usize], token: u64, margin: f32) -> Option<Trade> {
        let half = std::env::var("KUNO_GLM_HALF_LIFE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(HALF_LIFE);
        let decay = 0.5f32.powf(1.0 / half);
        let mut uses = self.uses.lock().ok()?;
        let at = |u: &mut (f32, u64)| {
            if u.1 < token {
                u.0 *= decay.powi((token - u.1) as i32);
                u.1 = token;
            }
            u.0
        };
        for &e in ids {
            at(&mut uses[e]);
            uses[e].0 += 1.0;
        }
        self.device.as_ref()?;
        let up = ids
            .iter()
            .copied()
            .filter(|e| self.host_slot[*e].is_some())
            .max_by(|a, b| uses[*a].0.total_cmp(&uses[*b].0).then(b.cmp(a)))?;
        let mut best: Option<(usize, Option<usize>, f32)> = None;
        for (k, r) in self.resident.iter().enumerate() {
            match r {
                None => {
                    best = Some((k, None, f32::MIN));
                    break;
                }
                Some(e) if !ids.contains(e) => {
                    let sc = at(&mut uses[*e]);
                    if best.is_none_or(|b| sc < b.2) {
                        best = Some((k, Some(*e), sc));
                    }
                }
                _ => {}
            }
        }
        let (slot, down, cold) = best?;
        if down.is_some() && uses[up].0 < cold + margin {
            return None;
        }
        Some(Trade {
            block,
            up,
            down,
            slot,
            spare: self.spare,
        })
    }

    /// Settles a completed trade's tables: the victim in the spare host slot, the promoted
    /// expert's old host slot the new spare.
    pub fn settle(&mut self, t: &Trade) {
        let old = self.host_slot[t.up].take();
        if let Some(d) = t.down {
            self.slot[d] = None;
            self.host_slot[d] = Some(t.spare);
            if let Some(o) = old {
                self.spare = o;
            }
        }
        self.slot[t.up] = Some(t.slot as u32);
        self.resident[t.slot] = Some(t.up);
    }

    /// Host slot `h`: its chunk and first word there.
    pub fn host_at(&self, h: u32) -> (&Mapped, usize) {
        let words: usize = self.stride.iter().sum();
        let h = h as usize;
        (
            &self.arena[h / self.per_chunk],
            (h % self.per_chunk) * words,
        )
    }

    /// The slot map after `t` (for the device).
    pub fn slots_after(&self, t: &Trade) -> Vec<u32> {
        let mut s: Vec<u32> = self.slot.iter().map(|s| s.unwrap_or(u32::MAX)).collect();
        if let Some(d) = t.down {
            s[d] = u32::MAX;
        }
        s[t.up] = t.slot as u32;
        s
    }

    /// The CPU's experts among `ids` (weights `w`) for input `x`: their weighted sum in the
    /// chosen order; zero when all are on the device.
    pub fn cpu_sum(&self, ids: &[usize], w: &[f32], x: &[f32], ff: usize, limit: f32) -> Vec<f32> {
        let n = x.len();
        let mine: Vec<(usize, f32)> = ids
            .iter()
            .zip(w)
            .filter(|(e, _)| self.host_slot[**e].is_some())
            .map(|(e, w)| (*e, *w))
            .collect();
        let mut sum = vec![0f32; n];
        if mine.is_empty() {
            return sum;
        }
        // Each expert's four parts in its host slot.
        let parts: Vec<[&[u8]; 4]> = mine
            .iter()
            .map(|(e, _)| {
                let (chunk, at) = self.host_at(self.host_slot[*e].unwrap());
                // SAFETY: device passes write only the spare host slot, never a resident
                // expert's (see `Trade`), so this slot is not written while it is read.
                let bytes = unsafe { chunk.bytes() };
                let mut out: [&[u8]; 4] = [&[]; 4];
                let mut o = at * 4;
                for (k, st) in self.stride.iter().enumerate() {
                    out[k] = &bytes[o..o + st * 4];
                    o += st * 4;
                }
                out
            })
            .collect();
        let q = cpu::Q8K::new(x);
        let (gb, gd) = (self.stride[0] * 4 / (2 * ff), self.stride[1] * 4 / (2 * ff));
        let (db, dd) = (self.stride[2] * 4 / n, self.stride[3] * 4 / n);
        // Gate and up rows of every expert here, 64 rows per task.
        let mut gu = vec![0f32; mine.len() * 2 * ff];
        gu.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (k, r) = (row0 / (2 * ff), row0 % (2 * ff));
            let p = &parts[k];
            let ds = if gd == 0 {
                &[][..]
            } else {
                &p[1][r * gd..][..out.len() * gd]
            };
            cpu::matvec(
                self.gate_kind,
                &p[0][r * gb..][..out.len() * gb],
                ds,
                gb,
                &q,
                out,
            );
        });
        let hidden: Vec<cpu::Q8K> = gu
            .chunks_exact(2 * ff)
            .map(|g| {
                let h: Vec<f32> = (0..ff)
                    .map(|i| {
                        let (a, b) = if limit > 1e-6 {
                            (g[i].min(limit), g[ff + i].clamp(-limit, limit))
                        } else {
                            (g[i], g[ff + i])
                        };
                        a / (1.0 + (-a).exp()) * b
                    })
                    .collect();
                cpu::Q8K::new(&h)
            })
            .collect();
        let mut d = vec![0f32; mine.len() * n];
        d.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (k, r) = (row0 / n, row0 % n);
            let p = &parts[k];
            let ds = if dd == 0 {
                &[][..]
            } else {
                &p[3][r * dd..][..out.len() * dd]
            };
            cpu::matvec(
                self.down_kind,
                &p[2][r * db..][..out.len() * db],
                ds,
                db,
                &hidden[k],
                out,
            );
        });
        for (k, (_, w)) in mine.iter().enumerate() {
            for (s, v) in sum.iter_mut().zip(&d[k * n..][..n]) {
                *s += w * v;
            }
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kuno_image::engine::gpu::{Select, Step, bytes_of};

    /// The device layout and back is byte-exact (a demoted expert's host copy comes from it).
    #[test]
    fn device_layout_round_trips() {
        let Some(dir) = std::env::var_os("KUNO_GLM_MODEL") else {
            return;
        };
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        shards.sort();
        let model = Model::open(&shards).unwrap();
        for il in [11usize, 20, 44] {
            let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
            let (gk, gr, _) = model.layout(&name("gate")).unwrap();
            let (dk, _, _) = model.layout(&name("down")).unwrap();
            let raw = read_expert(&model, il, 7).unwrap();
            let p = device_layout(gk, dk, &raw, model.hp.ff_expert, gr);
            let back = host_layout(gk, dk, [&p[0], &p[1], &p[2], &p[3]]);
            assert!(back == raw, "block {il}");
        }
    }

    /// mve against the decoded weights for blocks with each expert type (IQ2_S/IQ3_S in
    /// block 20, IQ3_S/IQ4_XS in block 11): two experts on the device among three routed.
    #[test]
    #[ignore = "needs a GPU and the model"]
    fn device_experts_match_decoded_weights() {
        let Some(dir) = std::env::var_os("KUNO_GLM_MODEL") else {
            return;
        };
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        shards.sort();
        let model = Model::open(&shards).unwrap();
        let hp = &model.hp;
        let (n, ff) = (hp.embd, hp.ff_expert);
        let pci = std::env::var("KUNO_GLM_GPU").unwrap_or_else(|_| "0000:2d:00.0".into());
        let gpu = Gpu::open(Select::Pci(pci)).unwrap();
        let k = crate::gpu::Kernels::new(&gpu).unwrap();
        let lanes = Some(k.lanes);
        let mve = gpu
            .kernel(include_bytes!("../shaders/mve.spv"), 7, 6, lanes)
            .unwrap();
        let mut t: Vec<u32> = Vec::new();
        for g in kuno_text::quant::IQ2_S_GRID {
            t.push(g as u32);
            t.push((g >> 32) as u32);
        }
        t.extend(kuno_text::quant::IQ3_S_GRID);
        let tables = filled(&gpu, bytes_of(&t)).unwrap();
        for il in [20usize, 11] {
            let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
            let (gk, gr, _) = model.layout(&name("gate")).unwrap();
            let (dk, dr, _) = model.layout(&name("down")).unwrap();
            let device = [5usize, 77];
            let routed = [77usize, 3, 5]; // 3 is not on the device
            let weights = [0.7f32, 0.2, 0.4];
            let (mut gw, mut gs, mut dw, mut ds) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut slots = vec![u32::MAX; hp.experts];
            let mut raw = Vec::new();
            for (s, &e) in device.iter().enumerate() {
                slots[e] = s as u32;
                for part in ["gate", "up"] {
                    raw.resize(ff * gr, 0);
                    model.read_rows(&name(part), e * ff, &mut raw).unwrap();
                    repack(gk, &raw, &mut gw, &mut gs);
                }
                raw.resize(n * dr, 0);
                model.read_rows(&name("down"), e * n, &mut raw).unwrap();
                repack(dk, &raw, &mut dw, &mut ds);
            }
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 7919) % 251) as f32 / 61.0 - 2.0)
                .collect();
            let h: Vec<f32> = (0..routed.len() * ff)
                .map(|i| ((i * 104729) % 241) as f32 / 97.0 - 1.2)
                .collect();
            let mut route: Vec<u32> = routed.iter().map(|e| *e as u32).collect();
            route.extend(weights.iter().map(|w| w.to_bits()));
            let b = |v: &[u8]| filled(&gpu, v).unwrap();
            let (guw, gus, dww, dss) = (b(&gw), b(&gs), b(&dw), b(&ds));
            let (slotb, routeb) = (b(bytes_of(&slots)), b(bytes_of(&route)));
            let (xb, hb) = (b(bytes_of(&x)), b(bytes_of(&h)));
            let ge = gpu.buffer((routed.len() * 2 * ff * 4) as u64).unwrap();
            let gd = gpu.buffer((n * 4) as u64).unwrap();
            let rpg = 128 / k.lanes;
            let used = routed.len() as u32;
            let c0 = [0, kind(gk).unwrap(), (2 * ff) as u32, n as u32, used, 0];
            let c1 = [1, kind(dk).unwrap(), n as u32, ff as u32, used, 0];
            let b0 = [&guw, &gus, &xb, &ge, &routeb, &slotb, &tables];
            let b1 = [&dww, &dss, &hb, &gd, &routeb, &slotb, &tables];
            let steps = [
                Step {
                    kernel: &mve,
                    buffers: &b0,
                    constants: &c0,
                    groups: [((2 * ff) as u32).div_ceil(rpg), used, 1],
                },
                Step {
                    kernel: &mve,
                    buffers: &b1,
                    constants: &c1,
                    groups: [(n as u32).div_ceil(rpg), 1, 1],
                },
            ];
            gpu.pass(&steps).unwrap();
            let read = |buf: &Buffer, count: usize| {
                let stage = gpu.readback((count * 4) as u64).unwrap();
                let mut v = vec![0f32; count];
                gpu.read_floats(buf, 0, &stage, &mut v).unwrap();
                v
            };
            let got_gu = read(&ge, routed.len() * 2 * ff);
            let got_d = read(&gd, n);
            let mut want_d = vec![0f32; n];
            let mut worst = 0f32;
            let mut scale = 0f32;
            for (kk, &e) in routed.iter().enumerate() {
                if slots[e] == u32::MAX {
                    continue;
                }
                let g = model.rows(&name("gate"), e * ff, ff).unwrap().matvec(&x);
                let u = model.rows(&name("up"), e * ff, ff).unwrap().matvec(&x);
                for (i, v) in g.iter().chain(&u).enumerate() {
                    worst = worst.max((got_gu[kk * 2 * ff + i] - v).abs());
                    scale = scale.max(v.abs());
                }
                let d = model
                    .rows(&name("down"), e * n, n)
                    .unwrap()
                    .matvec(&h[kk * ff..][..ff]);
                for (o, v) in want_d.iter_mut().zip(&d) {
                    *o += weights[kk] * v;
                }
            }
            let dscale = want_d.iter().map(|v| v.abs()).fold(0f32, f32::max);
            let dworst = got_d
                .iter()
                .zip(&want_d)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!(
                "block {il} {gk:?}/{dk:?}: gate|up worst {worst:.2e} of {scale:.2e}; down worst {dworst:.2e} of {dscale:.2e}"
            );
            assert!(
                worst <= 1e-4 * scale && dworst <= 1e-4 * dscale,
                "block {il}"
            );
        }
    }
}
