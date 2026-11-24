//! `cargo run --release -p kuno-glm --example reference -- <shard dir> <token ids, comma-separated>`:
//! the exact CPU pass over the tokens; prints each block's stream-mean statistics for the last
//! token and the ten most likely next tokens (ids and logits; no text).
use kuno_glm::{Model, reference::Reference};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let ids: Vec<u32> = args
        .next()
        .expect("token ids")
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let started = std::time::Instant::now();
    let model = Model::open(&shards).unwrap();
    let mut r = Reference::new(&model).unwrap();
    eprintln!("loaded in {:.1} s; {:?}", started.elapsed().as_secs_f64(), model.hp.dsa);
    let mut logits = None;
    for (i, id) in ids.iter().enumerate() {
        let started = std::time::Instant::now();
        logits = r.step(*id, i + 1 == ids.len()).unwrap();
        eprintln!("token {i}: {:.1} s", started.elapsed().as_secs_f64());
    }
    for (il, t) in r.trace.iter().enumerate() {
        let sum: f64 = t.iter().map(|v| f64::from(*v)).sum();
        let sq: f64 = t.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        println!("block {il}: sum {sum:.5} rms {:.5} first {:?}", (sq / t.len() as f64).sqrt(), &t[..4]);
    }
    let logits = logits.unwrap();
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|a, b| logits[*b].partial_cmp(&logits[*a]).unwrap());
    for &i in &order[..10] {
        println!("top {i} {:.4}", logits[i]);
    }
}
