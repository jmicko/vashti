//! `hostmem [pci] [GB]`: how much host-visible (mapped) memory the device gives in 1 GiB
//! buffers, and how fast the CPU reads it (16 threads) against ordinary memory.
use kuno_image::engine::gpu::{Gpu, Select};
use rayon::prelude::*;

fn read_gbs(b: &[u8]) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        let s: u64 = b
            .par_chunks(1 << 20)
            .map(|c| {
                c.chunks_exact(8)
                    .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
                    .fold(0, u64::wrapping_add)
            })
            .reduce(|| 0, u64::wrapping_add);
        std::hint::black_box(s);
        best = best.min(t.elapsed().as_secs_f64());
    }
    b.len() as f64 / best / 1e9
}

fn main() {
    let mut a = std::env::args().skip(1);
    let pci = a.next().unwrap_or_else(|| "0000:2d:00.0".into());
    let want: usize = a.next().map_or(85, |v| v.parse().unwrap());
    let gpu = Gpu::open(Select::Pci(pci)).unwrap();
    let t = std::time::Instant::now();
    let mut bufs = Vec::new();
    for i in 0..want {
        match gpu.mapped(1 << 30) {
            Ok(b) => bufs.push(b),
            Err(e) => {
                println!("stopped at {i} GiB: {e}");
                break;
            }
        }
    }
    println!(
        "{} GiB mapped in {:.1} s",
        bufs.len(),
        t.elapsed().as_secs_f64()
    );
    // SAFETY: no pass uses these buffers.
    let first = unsafe { bufs[0].bytes_mut() };
    first.iter_mut().enumerate().for_each(|(i, v)| *v = i as u8);
    println!(
        "mapped read: {:.1} GB/s",
        read_gbs(unsafe { bufs[0].bytes() })
    );
    let plain: Vec<u8> = (0..1usize << 30).map(|i| i as u8).collect();
    println!("plain read: {:.1} GB/s", read_gbs(&plain));
}
