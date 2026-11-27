//! Where the routed experts live: the most used on the GPU (per-block pools, `mve.comp`),
//! the rest in host memory for the CPU (`cpu`). Placement ranks every (block, expert) by how
//! often it was chosen (a counts file from earlier runs; without one, by expert index) and
//! fills the device's budget hottest first.
use kuno_image::engine::gpu::{Buffer, Gpu};
use kuno_text::TensorType;
use rayon::prelude::*;

use crate::gpu::filled;
use crate::{Model, Result, cpu};

/// One MoE block's experts.
pub struct BlockExperts {
    pub gate_kind: TensorType,
    pub down_kind: TensorType,
    /// Per expert: its slot on the main device, if any.
    pub slot: Vec<Option<u32>>,
    /// Device pools (when any expert is on the device): gate|up rows and down rows.
    pub device: Option<DevicePool>,
    /// The same on the second device.
    pub slot2: Vec<Option<u32>>,
    pub device2: Option<DevicePool>,
    /// Host copies, per expert not on the device: gate rows, up rows, down rows (raw).
    pub host: Vec<Option<Vec<u8>>>,
    pub gate_row: usize,
    pub down_row: usize,
}

pub struct DevicePool {
    pub gu_w: Buffer,
    pub gu_s: Buffer,
    pub down_w: Buffer,
    pub down_s: Buffer,
    /// slot per expert (0xffffffff: on the CPU), for the kernel.
    pub slots: Buffer,
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

/// Raw rows as the kernel's words and f16 block scales.
fn repack(t: TensorType, raw: &[u8], w: &mut Vec<u8>, s: &mut Vec<u8>) {
    match t {
        TensorType::Iq2S | TensorType::Iq3S => {
            let bb = if t == TensorType::Iq2S { 82 } else { 110 };
            for b in raw.chunks_exact(bb) {
                s.extend_from_slice(&b[..2]);
                w.extend_from_slice(&b[2..]);
            }
        }
        _ => w.extend_from_slice(raw),
    }
}

/// Reads counts written by gpu_check (`KUNO_GLM_COUNTS`): one line per block.
pub fn read_counts(path: &std::path::Path) -> Option<Vec<Vec<u32>>> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .map(|l| {
                l.split_whitespace()
                    .filter_map(|v| v.parse().ok())
                    .collect()
            })
            .collect(),
    )
}

/// One block's experts `ids` as a pool on `gpu`, with the slot map.
fn pool(
    model: &Model,
    gpu: &Gpu,
    il: usize,
    ids: &[usize],
    slot: &mut [Option<u32>],
) -> Result<Option<DevicePool>> {
    if ids.is_empty() {
        return Ok(None);
    }
    let hp = &model.hp;
    let (n, ff) = (hp.embd, hp.ff_expert);
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (gk, gr, _) = model.layout(&name("gate"))?;
    let (dk, dr, _) = model.layout(&name("down"))?;
    let (mut gw, mut gs, mut dw, mut ds) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut raw = Vec::new();
    for (k, &e) in ids.iter().enumerate() {
        slot[e] = Some(k as u32);
        for part in ["gate", "up"] {
            raw.resize(ff * gr, 0);
            model.read_rows(&name(part), e * ff, &mut raw)?;
            repack(gk, &raw, &mut gw, &mut gs);
        }
        raw.resize(n * dr, 0);
        model.read_rows(&name("down"), e * n, &mut raw)?;
        repack(dk, &raw, &mut dw, &mut ds);
    }
    let slots: Vec<u32> = slot.iter().map(|s| s.unwrap_or(u32::MAX)).collect();
    Ok(Some(DevicePool {
        gu_w: filled(gpu, &gw)?,
        gu_s: filled(gpu, &gs)?,
        down_w: filled(gpu, &dw)?,
        down_s: filled(gpu, &ds)?,
        slots: filled(gpu, kuno_image::engine::gpu::bytes_of(&slots))?,
    }))
}

/// Places every MoE block's experts: up to `device_bytes` on `gpu` (the hottest), then up
/// to the second device's budget there, the rest in memory.
pub fn place(
    model: &Model,
    gpu: &Gpu,
    device_bytes: u64,
    second: Option<(&Gpu, u64)>,
    counts: Option<&[Vec<u32>]>,
) -> Result<Vec<Option<BlockExperts>>> {
    let hp = &model.hp;
    let (n, ff) = (hp.embd, hp.ff_expert);
    let name = |il: usize, p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let mut ranking = Vec::new();
    let mut sizes = vec![0u64; hp.layers];
    for (il, size) in sizes.iter_mut().enumerate().skip(hp.dense_lead) {
        let (_, gr, _) = model.layout(&name(il, "gate"))?;
        let (_, dr, _) = model.layout(&name(il, "down"))?;
        *size = (2 * ff * gr + n * dr) as u64;
        for e in 0..hp.experts {
            let c = counts
                .and_then(|c| c.get(il))
                .and_then(|r| r.get(e))
                .copied()
                .unwrap_or(0);
            ranking.push((c, il, e));
        }
    }
    ranking.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    // 0: memory, 1: the main device, 2: the second.
    let mut on_device = vec![vec![0u8; hp.experts]; hp.layers];
    let (mut used, mut used2) = (0u64, 0u64);
    let budget2 = second.map_or(0, |s| s.1);
    for &(_, il, e) in &ranking {
        if used + sizes[il] <= device_bytes {
            used += sizes[il];
            on_device[il][e] = 1;
        } else if used2 + sizes[il] <= budget2 {
            used2 += sizes[il];
            on_device[il][e] = 2;
        }
    }
    let mut out = Vec::with_capacity(hp.layers);
    for (il, on_device) in on_device.iter().enumerate() {
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
        let ids1: Vec<usize> = (0..hp.experts).filter(|e| on_device[*e] == 1).collect();
        let ids2: Vec<usize> = (0..hp.experts).filter(|e| on_device[*e] == 2).collect();
        let device = pool(model, gpu, il, &ids1, &mut slot)?;
        let device2 = match second {
            Some((g, _)) => pool(model, g, il, &ids2, &mut slot2)?,
            None => None,
        };
        // The rest into memory, read in parallel.
        let host: Vec<Option<Vec<u8>>> = (0..hp.experts)
            .into_par_iter()
            .map(|e| -> Result<Option<Vec<u8>>> {
                // KUNO_GLM_NO_HOST (device profiling, with KUNO_GLM_SKIP_EXPERTS): none in memory.
                if slot[e].is_some()
                    || slot2[e].is_some()
                    || std::env::var_os("KUNO_GLM_NO_HOST").is_some()
                {
                    return Ok(None);
                }
                let mut v = vec![0u8; 2 * ff * gr + n * dr];
                let (g, rest) = v.split_at_mut(ff * gr);
                let (u, d) = rest.split_at_mut(ff * gr);
                model.read_rows(&name(il, "gate"), e * ff, g)?;
                model.read_rows(&name(il, "up"), e * ff, u)?;
                model.read_rows(&name(il, "down"), e * n, d)?;
                Ok(Some(v))
            })
            .collect::<Result<_>>()?;
        eprintln!(
            "block {il}: {} experts on the device, {} on the second, {} in memory",
            ids1.len(),
            ids2.len(),
            hp.experts - ids1.len() - ids2.len()
        );
        out.push(Some(BlockExperts {
            gate_kind: gk,
            down_kind: dk,
            slot,
            device,
            slot2,
            device2,
            host,
            gate_row: gr,
            down_row: dr,
        }));
    }
    eprintln!(
        "experts: {:.1} GB on the device, {:.1} GB on the second",
        used as f64 / 1e9,
        used2 as f64 / 1e9
    );
    Ok(out)
}

impl BlockExperts {
    /// The CPU's experts among `ids` (weights `w`) for input `x`: their weighted sum in the
    /// chosen order; zero when all are on the device.
    pub fn cpu_sum(&self, ids: &[usize], w: &[f32], x: &[f32], ff: usize, limit: f32) -> Vec<f32> {
        let n = x.len();
        let mine: Vec<(usize, f32)> = ids
            .iter()
            .zip(w)
            .filter(|(e, _)| self.slot[**e].is_none() && self.slot2[**e].is_none())
            .map(|(e, w)| (*e, *w))
            .collect();
        let mut sum = vec![0f32; n];
        if mine.is_empty() {
            return sum;
        }
        let q = cpu::Q8K::new(x);
        let (gr, dr) = (self.gate_row, self.down_row);
        // Gate and up rows of every expert here, 64 rows per task.
        let mut gu = vec![0f32; mine.len() * 2 * ff];
        gu.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (k, r) = (row0 / (2 * ff), row0 % (2 * ff));
            let bytes = self.host[mine[k].0].as_ref().unwrap();
            cpu::matvec(
                self.gate_kind,
                &bytes[r * gr..][..out.len() * gr],
                gr,
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
            let bytes = self.host[mine[k].0].as_ref().unwrap();
            let down = &bytes[2 * ff * gr..];
            cpu::matvec(
                self.down_kind,
                &down[r * dr..][..out.len() * dr],
                dr,
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
