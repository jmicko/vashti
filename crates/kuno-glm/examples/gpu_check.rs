//! `gpu_check <shard dir> <tokens json> [count] [pci]`: the GPU engine against the CPU
//! reference over the first `count` tokens: per block, the largest difference of the streams'
//! mean relative to its largest magnitude; and both next-token choices (ids only).
use kuno_glm::{Model, engine::Engine, reference::Reference};

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("shard directory"));
    let json = std::fs::read_to_string(args.next().expect("tokens json")).unwrap();
    let count: usize = args.next().map_or(3, |c| c.parse().unwrap());
    let pci = args.next().unwrap_or_else(|| "0000:2d:00.0".into());
    let ids: Vec<u32> = json
        .split('[')
        .nth(1)
        .unwrap()
        .split(']')
        .next()
        .unwrap()
        .split(',')
        .map(|t| t.trim().parse().unwrap())
        .collect();
    let mut shards: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
        .collect();
    shards.sort();
    let model = Model::open(&shards).unwrap();
    let t = std::time::Instant::now();
    let gpu =
        kuno_image::engine::gpu::Gpu::open(kuno_image::engine::gpu::Select::Pci(pci)).unwrap();
    let mut engine = Engine::new(&model, gpu, 2048).unwrap();
    eprintln!("engine ready in {:.1} s", t.elapsed().as_secs_f64());
    if std::env::var_os("KUNO_GLM_PROBE").is_some() {
        for (name, v) in engine.probe(ids[0]).unwrap() {
            let bad = v.iter().filter(|x| !x.is_finite()).count();
            let max = v
                .iter()
                .filter(|x| x.is_finite())
                .map(|x| x.abs())
                .fold(0f32, f32::max);
            println!(
                "{name}: {} values, {bad} non-finite, max |v| {max:.3e}, first {:?}",
                v.len(),
                &v[..4.min(v.len())]
            );
        }
        return;
    }
    engine.trace = Some(Vec::new());
    let mut reference = Reference::new(&model).unwrap();
    for (i, &id) in ids.iter().take(count).enumerate() {
        let t = std::time::Instant::now();
        let next = engine.step(id).unwrap();
        let gpu_s = t.elapsed().as_secs_f64();
        let logits = reference.step(id, true).unwrap().unwrap();
        let best =
            logits.iter().enumerate().fold(
                (0, f32::MIN),
                |b, (i, v)| if *v > b.1 { (i, *v) } else { b },
            );
        let got = engine.trace.as_ref().unwrap();
        let mut worst = (0usize, 0f32);
        for (b, (g, w)) in got.iter().zip(&reference.trace).enumerate() {
            let scale = w.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1e-9);
            let d = g
                .iter()
                .zip(w)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
                / scale;
            if d > worst.1 {
                worst = (b, d);
            }
            let nans = g.iter().filter(|v| !v.is_finite()).count();
            if i == 0 || b + 1 == got.len() || nans > 0 {
                println!(
                    "token {i} block {b}: {d:.2e}, {nans} non-finite, gpu {:?} ref {:?}",
                    &g[..3],
                    &w[..3]
                );
            }
            if nans > 0 && i == 0 {
                break;
            }
        }
        println!(
            "token {i}: gpu next {next}, reference next {} (logit {:.3}); worst block {} {:.2e}; gpu {:.2} s",
            best.0, best.1, worst.0, worst.1, gpu_s
        );
    }
}
