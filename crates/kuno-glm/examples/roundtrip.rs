//! `roundtrip [pci]`: how long one host -> GPU -> host exchange takes (a tiny pass, the host
//! writing its input and reading its output), waiting on the fence or spinning on the result.
use std::time::Instant;

use kuno_image::engine::gpu::{Gpu, Select, Step};

fn main() {
    let pci = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "0000:2d:00.0".into());
    let gpu = Gpu::open(Select::Pci(pci)).unwrap();
    println!("{}", gpu.name());
    let k = gpu
        .kernel(include_bytes!("../shaders/bump.spv"), 2, 1, None)
        .unwrap();
    let x = gpu.mapped(4096 * 4).unwrap();
    let y = gpu.mapped(4096 * 4).unwrap();
    let n = 4096u32;
    let constants = [n];
    let buffers = [x.buffer(), y.buffer()];
    let step = [Step {
        kernel: &k,
        buffers: &buffers,
        constants: &constants,
        groups: [n / 64, 1, 1],
    }];
    let mut out = [0f32; 1];
    for spin in [false, true] {
        let mut times = Vec::new();
        let mut submit = Vec::new();
        for i in 0..3000 {
            let v = i as f32;
            gpu.write_mapped(&x, 0, &[v]).unwrap();
            let t = Instant::now();
            // SAFETY: x and y outlive the pass, which is waited for below.
            let passing = unsafe { gpu.pass_begin(&step, &[false]).unwrap() };
            submit.push(t.elapsed().as_secs_f64());
            if spin {
                loop {
                    gpu.read_mapped(&y, 0, &mut out).unwrap();
                    if out[0] == v + 1.0 {
                        break;
                    }
                    std::hint::spin_loop();
                }
                times.push(t.elapsed().as_secs_f64());
                passing.wait().unwrap();
            } else {
                passing.wait().unwrap();
                gpu.read_mapped(&y, 0, &mut out).unwrap();
                times.push(t.elapsed().as_secs_f64());
                assert_eq!(out[0], v + 1.0);
            }
        }
        times.drain(..200);
        submit.drain(..200);
        submit.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  pass_begin returns after {:.1} us (median)",
            submit[submit.len() / 2] * 1e6
        );
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |f: f64| times[(times.len() as f64 * f) as usize] * 1e6;
        println!(
            "{}: median {:.1} us, p10 {:.1}, p90 {:.1}, p99 {:.1}",
            if spin { "spin on result" } else { "fence wait" },
            q(0.5),
            q(0.1),
            q(0.9),
            q(0.99)
        );
    }
}
