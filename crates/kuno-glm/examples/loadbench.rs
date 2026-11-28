//! `loadbench <shard dir> [pci]`: where load time goes (drop the page cache first): one
//! block's experts read in parallel, their device layout, and one block's dense weights
//! read, repacked and uploaded.
use kuno_glm::Model;
use kuno_glm::experts::{device_layout, read_expert};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let pci = args.next().unwrap_or_else(|| "0000:2d:00.0".into());
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let model = Model::open(&shards).unwrap();
    let reader = model.reader().unwrap();
    let il = 20;
    let name = |p: &str| format!("blk.{il}.ffn_{p}_exps.weight");
    let (gk, gr, _) = model.layout(&name("gate")).unwrap();
    let (dk, _, _) = model.layout(&name("down")).unwrap();
    let t = Instant::now();
    let raws: Vec<Vec<u8>> = (0..288)
        .into_par_iter()
        .map(|e| read_expert(&model, &reader, il, e).unwrap())
        .collect();
    let s = t.elapsed().as_secs_f64();
    let bytes: usize = raws.iter().map(|r| r.len()).sum();
    println!(
        "experts read: {:.2} s, {:.2} GB/s",
        s,
        bytes as f64 / s / 1e9
    );
    let t = Instant::now();
    let parts: Vec<_> = raws
        .par_iter()
        .map(|r| device_layout(gk, dk, r, model.hp.ff_expert, gr))
        .collect();
    println!(
        "experts layout: {:.2} s ({})",
        t.elapsed().as_secs_f64(),
        parts.len()
    );
    let gpu =
        kuno_image::engine::gpu::Gpu::open(kuno_image::engine::gpu::Select::Pci(pci)).unwrap();
    for round in 0..2 {
        let t = Instant::now();
        let m = gpu.mapped(1 << 30).unwrap();
        let alloc = t.elapsed().as_secs_f64();
        let t = Instant::now();
        // SAFETY: no pass uses it.
        let b = unsafe { m.bytes_mut() };
        b.par_chunks_mut(1 << 20).for_each(|c| c.fill(1));
        let first = t.elapsed().as_secs_f64();
        let t = Instant::now();
        b.par_chunks_mut(1 << 20).for_each(|c| c.fill(2));
        println!(
            "mapped 1 GiB (round {round}): alloc {:.3} s, first write {:.3} s, second {:.3} s",
            alloc,
            first,
            t.elapsed().as_secs_f64()
        );
    }
    let m0 = Instant::now();
    let mut mats = Vec::new();
    for n in [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_output",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
        "ffn_gate_inp",
        "hc_attn_fn",
        "hc_ffn_fn",
    ] {
        mats.push(
            model
                .matrix_with(&reader, &format!("blk.21.{n}.weight"))
                .unwrap(),
        );
    }
    let rd = m0.elapsed().as_secs_f64();
    let bytes: usize = mats.iter().map(|m| m.bytes.len()).sum();
    let t = Instant::now();
    let mut ups = Vec::new();
    for m in &mats {
        ups.push(kuno_glm::gpu::Mat::upload(&gpu, m).unwrap());
    }
    println!(
        "dense block: read {:.2} s ({:.2} GB/s), repack+upload+verify {:.2} s ({:.0} MB)",
        rd,
        bytes as f64 / rd / 1e9,
        t.elapsed().as_secs_f64(),
        bytes as f64 / 1e6
    );
    let m = &mats[0];
    let t = Instant::now();
    let _ = kuno_glm::gpu::repack_q6k(&m.bytes, m.rows, m.cols);
    println!(
        "q6k repack of {:.0} MB: {:.2} s",
        m.bytes.len() as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let _b = kuno_glm::gpu::filled(&gpu, &m.bytes).unwrap();
    println!(
        "filled (verified upload) of {:.0} MB: {:.2} s",
        m.bytes.len() as f64 / 1e6,
        t.elapsed().as_secs_f64()
    );
}
