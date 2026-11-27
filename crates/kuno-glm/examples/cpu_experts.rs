//! `cpu_experts <shard dir> [block]`: one MoE block's routed experts on the CPU. All 288 of
//! its experts are read into memory (2.6 GB, far past the caches), then 8 random experts per
//! token run gate and up, SwiGLU and down, as the decoder will; prints the time per block and
//! the weight bytes read per second beside a plain parallel read of the same memory.
use std::time::Instant;

use kuno_glm::{Model, cpu};
use rayon::prelude::*;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let block: usize = args.next().map_or(20, |b| b.parse().unwrap());
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let model = Model::open(&shards).unwrap();
    let hp = &model.hp;
    let (n, f) = (hp.embd, hp.ff_expert);
    let t = Instant::now();
    let name = |p: &str| format!("blk.{block}.ffn_{p}_exps.weight");
    let gate = model.matrix(&name("gate")).unwrap();
    let up = model.matrix(&name("up")).unwrap();
    let down = model.matrix(&name("down")).unwrap();
    let total = gate.bytes.len() + up.bytes.len() + down.bytes.len();
    eprintln!(
        "block {block}: {:?}/{:?}/{:?}, {:.2} GB read in {:.1} s; {} threads",
        gate.dtype,
        up.dtype,
        down.dtype,
        total as f64 / 1e9,
        t.elapsed().as_secs_f64(),
        rayon::current_num_threads()
    );

    // A plain parallel read: the memory's bandwidth for this layout.
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        let sum: u64 = [&gate.bytes, &up.bytes, &down.bytes]
            .par_iter()
            .flat_map(|b| b.par_chunks(1 << 20))
            .map(|c| {
                c.chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                    .fold(0u64, u64::wrapping_add)
            })
            .reduce(|| 0, u64::wrapping_add);
        std::hint::black_box(sum);
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("plain read: {:.1} GB/s", total as f64 / best / 1e9);

    let x: Vec<f32> = (0..n)
        .map(|i| ((i * 7919) % 255) as f32 / 37.0 - 3.4)
        .collect();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut times = Vec::new();
    let (gr, dr) = (gate.row_bytes, down.row_bytes);
    let expert_bytes = f * (gate.row_bytes + up.row_bytes) + n * down.row_bytes;
    for _ in 0..60 {
        let mut chosen = Vec::new();
        while chosen.len() < hp.experts_used {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let e = (seed % hp.experts as u64) as usize;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let t = Instant::now();
        let q = cpu::Q8K::new(&x);
        // Gate and up rows of every chosen expert, 64 rows per task.
        let mut gu = vec![0f32; chosen.len() * 2 * f];
        gu.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (e, rest) = (row0 / (2 * f), row0 % (2 * f));
            let (m, r) = if rest < f {
                (&gate, rest)
            } else {
                (&up, rest - f)
            };
            let first = chosen[e] * f + r;
            cpu::matvec(
                m.dtype,
                &m.bytes[first * gr..][..out.len() * gr],
                gr,
                &q,
                out,
            );
        });
        let hidden: Vec<cpu::Q8K> = gu
            .chunks_exact(2 * f)
            .map(|g| {
                let h: Vec<f32> = (0..f)
                    .map(|i| {
                        let (a, b) = (g[i].min(10.0), g[f + i].clamp(-10.0, 10.0));
                        a / (1.0 + (-a).exp()) * b
                    })
                    .collect();
                cpu::Q8K::new(&h)
            })
            .collect();
        let mut d = vec![0f32; chosen.len() * n];
        d.par_chunks_mut(64).enumerate().for_each(|(task, out)| {
            let row0 = task * 64;
            let (e, r) = (row0 / n, row0 % n);
            let first = chosen[e] * n + r;
            cpu::matvec(
                down.dtype,
                &down.bytes[first * dr..][..out.len() * dr],
                dr,
                &hidden[e],
                out,
            );
        });
        std::hint::black_box(&d);
        times.push(t.elapsed().as_secs_f64());
    }
    times.drain(..10);
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let bytes = expert_bytes * hp.experts_used;
    println!(
        "8 experts: median {:.2} ms (p10 {:.2}, p90 {:.2}), {:.1} MB, {:.1} GB/s",
        median * 1e3,
        times[times.len() / 10] * 1e3,
        times[times.len() * 9 / 10] * 1e3,
        bytes as f64 / 1e6,
        bytes as f64 / median / 1e9
    );
}
