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
    // KUNO_GLM_SECOND: a second GPU (PCI address) for the next tier of experts.
    let second = std::env::var("KUNO_GLM_SECOND").ok().map(|p| {
        kuno_image::engine::gpu::Gpu::open(kuno_image::engine::gpu::Select::Pci(p)).unwrap()
    });
    let ctx = std::env::var("KUNO_GLM_CTX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);
    let mut engine = Engine::new(&model, gpu, second, ctx).unwrap();
    eprintln!("engine ready in {:.1} s", t.elapsed().as_secs_f64());
    if let Some(extra) = std::env::var("KUNO_GLM_BENCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        // The engine alone: the prompt, then `extra` greedy tokens (ids only).
        let mut next = 0;
        let mut times = Vec::new();
        let mut out = Vec::new();
        for (i, &id) in ids.iter().chain(std::iter::repeat_n(&0, extra)).enumerate() {
            let id = if i < ids.len() { id } else { next };
            let t = std::time::Instant::now();
            next = engine.step(id).unwrap();
            times.push(t.elapsed().as_secs_f64());
            if i + 1 >= ids.len() {
                out.push(next);
            }
        }
        let generated = &times[ids.len()..];
        if let Ok(path) = std::env::var("KUNO_GLM_COUNTS") {
            // Expert choice counts per block, one line each (no token content).
            let c = engine.counts.lock().unwrap();
            let text: String = c
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                        + "\n"
                })
                .collect();
            std::fs::write(path, text).unwrap();
        }
        if let Ok(p) = engine.profile.lock() {
            let tokens = times.len() as f64;
            let total: f64 = p.values().map(|v| v.0).sum();
            for (label, (ns, count)) in p.iter() {
                println!(
                    "{label:32} {:7.3} ms/token  {:5.1} us each  x{}",
                    ns / tokens / 1e6,
                    ns / *count as f64 / 1e3,
                    *count as f64 / tokens
                );
            }
            if total > 0.0 {
                println!("device total {:.2} ms/token", total / tokens / 1e6);
            }
        }
        println!("generated ids: {:?}", &out[..out.len().min(12)]);
        let mean = generated.iter().sum::<f64>() / generated.len().max(1) as f64;
        println!(
            "prompt {} tokens {:.2} s; {} generated: {:.1} ms each ({:.2} tokens/s)",
            ids.len(),
            times[..ids.len()].iter().sum::<f64>(),
            generated.len(),
            mean * 1e3,
            1.0 / mean
        );
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
            "token {i}: gpu next {next} (reference logit {:.3}), reference next {} (logit {:.3}); worst block {} {:.2e}; gpu {:.2} s",
            logits[next as usize], best.0, best.1, worst.0, worst.1, gpu_s
        );
    }
}
