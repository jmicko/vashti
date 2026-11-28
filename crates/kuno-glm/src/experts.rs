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

/// HALF_LIFE, or KUNO_GLM_HALF_LIFE.
fn half_life() -> f32 {
    std::env::var("KUNO_GLM_HALF_LIFE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(HALF_LIFE)
}

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

/// One part of `raw` rows of type `t` in the device's layout, written to `w` (bodies) and
/// `s` (f16 scales; empty for IQ4_XS).
fn repack_into(t: TensorType, raw: &[u8], w: &mut [u8], s: &mut [u8]) {
    match t {
        TensorType::Iq2S | TensorType::Iq3S => {
            let bb = block_bytes(t);
            for ((b, wo), so) in raw
                .chunks_exact(bb)
                .zip(w.chunks_exact_mut(bb - 2))
                .zip(s.chunks_exact_mut(2))
            {
                so.copy_from_slice(&b[..2]);
                wo.copy_from_slice(&b[2..]);
            }
        }
        _ => w.copy_from_slice(raw),
    }
}

/// Part `k` (gu_w, gu_s, down_w, down_s) of an expert's device layout into `out` (its
/// exact size).
fn layout_part_into(
    gk: TensorType,
    dk: TensorType,
    raw: &[u8],
    ff: usize,
    gr: usize,
    k: usize,
    out: &mut [u8],
) {
    let (t, r) = if k < 2 {
        (gk, &raw[..2 * ff * gr])
    } else {
        (dk, &raw[2 * ff * gr..])
    };
    match t {
        TensorType::Iq2S | TensorType::Iq3S => {
            let bb = block_bytes(t);
            if k.is_multiple_of(2) {
                for (b, o) in r.chunks_exact(bb).zip(out.chunks_exact_mut(bb - 2)) {
                    o.copy_from_slice(&b[2..]);
                }
            } else {
                for (b, o) in r.chunks_exact(bb).zip(out.chunks_exact_mut(2)) {
                    o.copy_from_slice(&b[..2]);
                }
            }
        }
        _ if k.is_multiple_of(2) => out.copy_from_slice(r),
        _ => {}
    }
}

/// An expert's whole device layout (gu_w, gu_s, down_w, down_s) into `out`.
fn layout_into(
    gk: TensorType,
    dk: TensorType,
    raw: &[u8],
    ff: usize,
    gr: usize,
    stride: &[usize; 4],
    out: &mut [u8],
) {
    let (g, d) = raw.split_at(2 * ff * gr);
    let (gw, rest) = out.split_at_mut(stride[0] * 4);
    let (gs, rest) = rest.split_at_mut(stride[1] * 4);
    let (dw, ds) = rest.split_at_mut(stride[2] * 4);
    repack_into(gk, g, gw, gs);
    repack_into(dk, d, dw, &mut ds[..stride[3] * 4]);
}

/// Bytes per expert of block `il` (raw rows; the device layout is the same size).
pub fn slot_bytes(model: &Model, il: usize) -> Result<usize> {
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (_, gr, _) = model.layout(&name("gate"))?;
    let (_, dr, _) = model.layout(&name("down"))?;
    Ok(2 * model.hp.ff_expert * gr + model.hp.embd * dr)
}

/// Host-visible chunks (about 1 GiB each) for `count` experts of `slot_bytes` and a spare.
pub fn host_arena(gpu: &Gpu, slot_bytes: usize, count: usize) -> Result<Vec<Mapped>> {
    let per_chunk = ((1usize << 30) / slot_bytes).max(1);
    let total = count + 1;
    (0..total.div_ceil(per_chunk))
        .map(|c| {
            let n = per_chunk.min(total - c * per_chunk);
            gpu.mapped((n * slot_bytes) as u64)
        })
        .collect()
}

/// Block `il`'s experts read (as `read_block_threads`) while its host chunks for `count`
/// experts are allocated (allocation zeroes them: as slow as a third of the reading).
pub fn prefetch_block(
    model: &Model,
    gpu: &Gpu,
    il: usize,
    count: usize,
) -> Result<(Vec<Vec<u8>>, Vec<Mapped>)> {
    let bytes = slot_bytes(model, il)?;
    std::thread::scope(|sc| {
        let t = std::time::Instant::now();
        let alloc = sc.spawn(move || {
            let a = host_arena(gpu, bytes, count);
            (a, t.elapsed().as_secs_f64())
        });
        let raws = read_block_threads(model, il, 8)?;
        let read_s = t.elapsed().as_secs_f64();
        let (arena, alloc_s) = alloc.join().map_err(|_| "gpu_memory")?;
        let arena = arena?;
        if std::env::var_os("KUNO_GLM_TIMING").is_some() {
            eprintln!("block {il} read in {read_s:.2} s, its chunks allocated in {alloc_s:.2} s");
        }
        Ok((raws, arena))
    })
}

/// Block `il`'s experts read by `threads` threads of their own (beside rayon's, which the
/// placement of the previous block keeps busy).
pub fn read_block_threads(model: &Model, il: usize, threads: usize) -> Result<Vec<Vec<u8>>> {
    let e = model.hp.experts;
    let reader = model.reader()?;
    let mut out: Vec<Vec<u8>> = vec![Vec::new(); e];
    std::thread::scope(|sc| -> Result<()> {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                let reader = &reader;
                sc.spawn(move || -> Result<Vec<(usize, Vec<u8>)>> {
                    (i..e)
                        .step_by(threads)
                        .map(|x| read_expert(model, reader, il, x).map(|r| (x, r)))
                        .collect()
                })
            })
            .collect();
        for h in handles {
            for (x, r) in h.join().map_err(|_| "model_read_failed")?? {
                out[x] = r;
            }
        }
        Ok(())
    })?;
    Ok(out)
}

/// An expert back from the device's layout to raw rows.
pub fn host_layout(gk: TensorType, dk: TensorType, parts: [&[u8]; 4]) -> Vec<u8> {
    let mut raw = Vec::new();
    unpack(gk, parts[0], parts[1], &mut raw);
    unpack(dk, parts[2], parts[3], &mut raw);
    raw
}

/// A block's expert `e` as raw rows from the file.
pub fn read_expert(model: &Model, reader: &Reader, il: usize, e: usize) -> Result<Vec<u8>> {
    let hp = &model.hp;
    let (n, ff) = (hp.embd, hp.ff_expert);
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (_, gr, _) = model.layout(&name("gate"))?;
    let (_, dr, _) = model.layout(&name("down"))?;
    let mut v = vec![0u8; 2 * ff * gr + n * dr];
    let (g, rest) = v.split_at_mut(ff * gr);
    let (u, d) = rest.split_at_mut(ff * gr);
    model.read_rows_with(reader, &name("gate"), e * ff, g)?;
    model.read_rows_with(reader, &name("up"), e * ff, u)?;
    model.read_rows_with(reader, &name("down"), e * n, d)?;
    Ok(v)
}

pub type Reader<'m> = kuno_text::gguf::PositionalReader<'m>;

/// One block's bank of `capacity` slots on `gpu`, the first holding `ids`.
fn pool(
    model: &Model,
    get: &Raw,
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
    if ids.is_empty() {
        return Err("expert_bank_empty");
    }
    if gpu.vendor() == 0x8086 {
        // Each part written straight into its upload, in parallel (no check: see `filled`).
        let stride = device_layout(gk, dk, &get(ids[0])?, ff, gr).map(|p| p.len() / 4);
        let mut bufs = Vec::with_capacity(4);
        for (k, &st) in stride.iter().enumerate() {
            let bytes = (capacity * st * 4).max(4);
            bufs.push(gpu.filled(bytes as u64, |out| {
                out.fill(0);
                if st == 0 {
                    return Ok(());
                }
                out.par_chunks_mut(st * 4)
                    .zip(ids.par_iter())
                    .try_for_each(|(o, &e)| -> Result<()> {
                        layout_part_into(gk, dk, &get(e)?, ff, gr, k, o);
                        Ok(())
                    })
                    .map_err(|_| "model_read_failed")
            })?);
        }
        for (k, &e) in ids.iter().enumerate() {
            slot[e] = Some(k as u32);
        }
        let slots: Vec<u32> = slot.iter().map(|s| s.unwrap_or(u32::MAX)).collect();
        let mut b = bufs.into_iter();
        return Ok(Some(DevicePool {
            gu_w: b.next().ok_or("gpu_memory")?,
            gu_s: b.next().ok_or("gpu_memory")?,
            down_w: b.next().ok_or("gpu_memory")?,
            down_s: b.next().ok_or("gpu_memory")?,
            slots: filled(gpu, kuno_image::engine::gpu::bytes_of(&slots))?,
            stride,
            capacity,
        }));
    }
    let mut parts: [Vec<u8>; 4] = Default::default();
    let mut stride = [0usize; 4];
    // Read in parallel (the device's copies are made from them in order).
    let layouts: Vec<[Vec<u8>; 4]> = ids
        .par_iter()
        .map(|&e| -> Result<[Vec<u8>; 4]> {
            let raw = get(e)?;
            Ok(device_layout(gk, dk, &raw, ff, gr))
        })
        .collect::<Result<_>>()?;
    for ((k, &e), p) in ids.iter().enumerate().zip(layouts) {
        slot[e] = Some(k as u32);
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

/// A block's expert `e` as raw rows (read now, or read before).
pub type Raw<'a> = dyn Fn(usize) -> Result<std::borrow::Cow<'a, [u8]>> + Sync + 'a;

/// Slots per block on `gpu` within `device_bytes` (the same for every block), and on the
/// second device within its budget.
pub fn plan(model: &Model, device_bytes: u64, second: Option<u64>) -> Result<(usize, usize, u64)> {
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
    let slots2 = ((second.unwrap_or(0) / per_slot.max(1)) as usize).min(hp.experts - slots1);
    Ok((slots1, slots2, per_slot))
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
    let (slots1, slots2, per_slot) = plan(model, device_bytes, second.map(|s| s.1))?;
    let reader = model.reader()?;
    let mut out = Vec::with_capacity(hp.layers);
    for il in 0..hp.layers {
        if il < hp.dense_lead {
            out.push(None);
            continue;
        }
        let ids1: Vec<usize> = (0..slots1).collect();
        let ids2: Vec<usize> = (slots1..slots1 + slots2).collect();
        let get = |e: usize| read_expert(model, &reader, il, e).map(std::borrow::Cow::Owned);
        out.push(Some(place_block(
            model,
            gpu,
            il,
            &ids1,
            second.map(|s| (s.0, &ids2[..])),
            &get,
            None,
        )?));
    }
    eprintln!(
        "experts: {slots1} slots per block on the device ({:.1} GB), {slots2} on the second",
        (slots1 as u64 * per_slot) as f64 / 1e9
    );
    Ok(out)
}

/// Reads block `il`'s experts in parallel (raw rows, for `place_block`).
pub fn read_block(model: &Model, reader: &Reader, il: usize) -> Result<Vec<Vec<u8>>> {
    (0..model.hp.experts)
        .into_par_iter()
        .map(|e| read_expert(model, reader, il, e))
        .collect()
}

/// Places block `il`'s experts: `ids1` in the device's bank (in slot order), `ids2` on the
/// second device, the rest in host memory; `get` gives each expert's raw rows.
pub fn place_block(
    model: &Model,
    gpu: &Gpu,
    il: usize,
    ids1: &[usize],
    second: Option<(&Gpu, &[usize])>,
    get: &Raw,
    arena: Option<Vec<Mapped>>,
) -> Result<BlockExperts> {
    let hp = &model.hp;
    let ff = hp.ff_expert;
    let name = |il: usize, p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let started = std::time::Instant::now();
    let slots1 = ids1.len();
    {
        let (gk, gr, _) = model.layout(&name(il, "gate"))?;
        let (dk, dr, _) = model.layout(&name(il, "down"))?;
        cpu::check(gk)?;
        cpu::check(dk)?;
        let mut slot = vec![None; hp.experts];
        let mut slot2 = vec![None; hp.experts];
        let t = std::time::Instant::now();
        let device = pool(model, get, gpu, il, slots1, ids1, &mut slot)?;
        let bank_s = t.elapsed().as_secs_f64();
        let device2 = match second {
            Some((g, ids2)) => pool(model, get, g, il, ids2.len(), ids2, &mut slot2)?,
            None => None,
        };
        // The rest into host-visible chunks (plus one spare slot), read in parallel.
        let stride = {
            let raw = get(0)?;
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
        let arena = match arena {
            Some(a) if a.len() == (on_host.len() + 1).div_ceil(per_chunk) => a,
            _ => host_arena(gpu, slot_bytes, on_host.len())?,
        };
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
                    layout_into(gk, dk, &get(e)?, ff, gr, &stride, out);
                    Ok(())
                })?;
        }
        eprintln!(
            "block {il}'s experts placed ({:.1} s; bank {:.1} s)",
            started.elapsed().as_secs_f64(),
            bank_s
        );
        let mut resident = vec![None; slots1];
        for (k, e) in ids1.iter().enumerate() {
            resident[k] = Some(*e);
        }
        Ok(BlockExperts {
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
        })
    }
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
        let decay = 0.5f32.powf(1.0 / half_life());
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

    /// Counts a batch's choices: `counts` per expert over `tokens` tokens, the last at
    /// `token` (each use weighted by its mean decay over the batch).
    pub fn add_batch(&self, counts: &[u32], token: u64, tokens: usize) {
        let decay = 0.5f32.powf(1.0 / half_life());
        let Ok(mut uses) = self.uses.lock() else {
            return;
        };
        let t = tokens.max(1) as f32;
        let mean = (1.0 - decay.powf(t)) / ((1.0 - decay) * t);
        for (u, &c) in uses.iter_mut().zip(counts) {
            if u.1 < token {
                u.0 *= decay.powi((token - u.1) as i32);
                u.1 = token;
            }
            u.0 += c as f32 * mean;
        }
    }

    /// The best trade by score alone (between tokens: no choice in use to protect): the
    /// hottest host expert over the coldest resident (or an empty slot), when used clearly
    /// more (by `margin`).
    pub fn best_trade(&self, block: usize, margin: f32) -> Option<Trade> {
        self.device.as_ref()?;
        let uses = self.uses.lock().ok()?;
        let up = (0..uses.len())
            .filter(|e| self.host_slot[*e].is_some())
            .max_by(|a, b| uses[*a].0.total_cmp(&uses[*b].0).then(b.cmp(a)))?;
        let mut best: Option<(usize, Option<usize>, f32)> = None;
        for (k, r) in self.resident.iter().enumerate() {
            let sc = r.map_or(f32::MIN, |e| uses[e].0);
            if best.is_none_or(|b| sc < b.2) {
                best = Some((k, *r, sc));
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

    /// `cpu_sum` for several tokens at once (`choices`: each token's ids and weights, `xs`
    /// its input): each expert's rows are read once for every token that chose it. The
    /// sums are `cpu_sum`'s exactly (the same products, added in the same order).
    pub fn cpu_sums(
        &self,
        choices: &[(&[usize], &[f32])],
        xs: &[&[f32]],
        ff: usize,
        limit: f32,
    ) -> Vec<Vec<f32>> {
        let n = xs.first().map_or(0, |x| x.len());
        // (token, expert, weight) for every expert here, in each token's chosen order.
        let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
        for (t, (ids, w)) in choices.iter().enumerate() {
            for (e, wk) in ids.iter().zip(w.iter()) {
                if self.host_slot[*e].is_some() {
                    pairs.push((t, *e, *wk));
                }
            }
        }
        let mut sums = vec![vec![0f32; n]; choices.len()];
        if pairs.is_empty() {
            return sums;
        }
        let mut unique: Vec<usize> = pairs.iter().map(|p| p.1).collect();
        unique.sort_unstable();
        unique.dedup();
        let users: Vec<Vec<usize>> = unique
            .iter()
            .map(|e| (0..pairs.len()).filter(|&i| pairs[i].1 == *e).collect())
            .collect();
        let parts: Vec<[&[u8]; 4]> = unique
            .iter()
            .map(|e| {
                let (chunk, at) = self.host_at(self.host_slot[*e].unwrap());
                // SAFETY: as in `cpu_sum`.
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
        let qs: Vec<cpu::Q8K> = xs.iter().map(|x| cpu::Q8K::new(x)).collect();
        let (gb, gd) = (self.stride[0] * 4 / (2 * ff), self.stride[1] * 4 / (2 * ff));
        let (db, dd) = (self.stride[2] * 4 / n, self.stride[3] * 4 / n);
        // Tasks of 64 rows of one expert for each of its users: an expert's down projection
        // starts when its own gate and up rows are done (no barrier between all experts' two
        // halves, which kept the bus idle at each block's turn).
        let (gchunks, dchunks) = ((2 * ff).div_ceil(64), n.div_ceil(64));
        struct Work {
            gu: Mutex<Vec<f32>>,
            left: std::sync::atomic::AtomicUsize,
            hidden: std::sync::OnceLock<Vec<cpu::Q8K>>,
            down: Mutex<Vec<f32>>,
        }
        let work: Vec<Work> = users
            .iter()
            .map(|u| Work {
                gu: Mutex::new(vec![0f32; u.len() * 2 * ff]),
                left: std::sync::atomic::AtomicUsize::new(gchunks),
                hidden: std::sync::OnceLock::new(),
                down: Mutex::new(vec![0f32; u.len() * n]),
            })
            .collect();
        let down_task = |u: usize, c: usize| {
            let (r, len) = (c * 64, 64.min(n - c * 64));
            let p = &parts[u];
            let rows = &p[2][r * db..][..len * db];
            let ds = if dd == 0 {
                &[][..]
            } else {
                &p[3][r * dd..][..len * dd]
            };
            let hidden = work[u].hidden.get().expect("gate and up first");
            let mut out = vec![0f32; len];
            let mut outs = Vec::with_capacity(users[u].len());
            for h in hidden {
                cpu::matvec(self.down_kind, rows, ds, db, h, &mut out);
                outs.push(out.clone());
            }
            let mut d = work[u].down.lock().unwrap_or_else(|e| e.into_inner());
            for (j, o) in outs.iter().enumerate() {
                d[j * n + r..][..len].copy_from_slice(o);
            }
        };
        rayon::scope(|sc| {
            for u in 0..unique.len() {
                for c in 0..gchunks {
                    let (work, parts, users, pairs, qs, down_task) =
                        (&work, &parts, &users, &pairs, &qs, &down_task);
                    sc.spawn(move |sc| {
                        let (r, len) = (c * 64, 64.min(2 * ff - c * 64));
                        let p = &parts[u];
                        let rows = &p[0][r * gb..][..len * gb];
                        let ds = if gd == 0 {
                            &[][..]
                        } else {
                            &p[1][r * gd..][..len * gd]
                        };
                        let mut out = vec![0f32; users[u].len() * len];
                        for (j, &pi) in users[u].iter().enumerate() {
                            cpu::matvec(
                                self.gate_kind,
                                rows,
                                ds,
                                gb,
                                &qs[pairs[pi].0],
                                &mut out[j * len..][..len],
                            );
                        }
                        {
                            let mut g = work[u].gu.lock().unwrap_or_else(|e| e.into_inner());
                            for j in 0..users[u].len() {
                                g[j * 2 * ff + r..][..len].copy_from_slice(&out[j * len..][..len]);
                            }
                        }
                        let ordering = std::sync::atomic::Ordering::AcqRel;
                        if work[u].left.fetch_sub(1, ordering) == 1 {
                            // The last of this expert's gate and up rows: its hidden inputs,
                            // then its down projection's tasks.
                            let g = work[u].gu.lock().unwrap_or_else(|e| e.into_inner());
                            let hidden: Vec<cpu::Q8K> = g
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
                            drop(g);
                            let _ = work[u].hidden.set(hidden);
                            for c in 0..dchunks {
                                sc.spawn(move |_| down_task(u, c));
                            }
                        }
                    });
                }
            }
        });
        // Added in each token's chosen order (as `cpu_sum` adds them).
        let downs: Vec<Vec<f32>> = work
            .into_iter()
            .map(|w| w.down.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        let mut at = vec![(0usize, 0usize); pairs.len()];
        for (u, us) in users.iter().enumerate() {
            for (j, &pi) in us.iter().enumerate() {
                at[pi] = (u, j);
            }
        }
        for (pi, &(t, _, wk)) in pairs.iter().enumerate() {
            let (u, j) = at[pi];
            for (s, v) in sums[t].iter_mut().zip(&downs[u][j * n..][..n]) {
                *s += wk * v;
            }
        }
        sums
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
            let raw = read_expert(&model, &model.reader().unwrap(), il, 7).unwrap();
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
