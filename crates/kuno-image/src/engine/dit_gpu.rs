//! MiniMax H3 diffusion transformer on the GPU (Vulkan). Built kernel by kernel; every piece
//! is checked against the CPU reference in `dit.rs` on identical inputs, because whole-network
//! outputs only agree loosely (see the engine document).
use super::gpu::{Buffer, Gpu, Kernel, Step, shaders};

/// One dispatch with its arguments owned, so a pass can be assembled as a list.
pub struct Call<'a> {
    kernel: &'a Kernel,
    buffers: Vec<&'a Buffer>,
    constants: Vec<u32>,
    groups: [u32; 3],
}
/// Runs the calls in order in one submission and waits for it.
pub fn run(gpu: &Gpu, calls: &[Call]) -> Result<(), &'static str> {
    let steps: Vec<Step> = calls
        .iter()
        .map(|call| Step {
            kernel: call.kernel,
            buffers: &call.buffers,
            constants: &call.constants,
            groups: call.groups,
        })
        .collect();
    gpu.pass(&steps)
}

/// Kernels of one DiT evaluation.
pub struct Kernels {
    rotate8: Kernel,
}
impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self, &'static str> {
        Ok(Self {
            rotate8: gpu.kernel(shaders::ROTATE8, 3, 2, Some(16))?,
        })
    }
    /// Quantizes `rows` f32 rows of `width` values in `input` to INT8 rows in `bytes` and one
    /// scale per row in `scales`.
    pub fn quantize<'a>(
        &'a self,
        input: &'a Buffer,
        bytes: &'a Buffer,
        scales: &'a Buffer,
        rows: usize,
        width: usize,
    ) -> Call<'a> {
        Call {
            kernel: &self.rotate8,
            buffers: vec![input, bytes, scales],
            constants: vec![width as u32, (width / 256).div_ceil(16) as u32],
            groups: [rows as u32, 1, 1],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            dit,
            gpu::{Select, bytes_of, floats},
        },
        *,
    };

    fn noise(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    #[test]
    #[ignore = "needs the Intel GPU"]
    fn quantizer_matches_reference_bit_for_bit() {
        let gpu = Gpu::open(Select::Vendor(0x8086)).unwrap();
        let kernels = Kernels::new(&gpu).unwrap();
        for width in [5376usize, 7168, 14336] {
            let rows = 37;
            let mut x = noise(rows * width, width as u64);
            // Uneven magnitudes, a zero row and exact rounding ties.
            for (i, v) in x.iter_mut().enumerate() {
                *v *= 1.0 + (i % 97) as f32;
            }
            x[3 * width..4 * width].fill(0.0);
            for (i, v) in x[5 * width..6 * width].iter_mut().enumerate() {
                *v = (i % 5) as f32 * 0.5;
            }
            let (want, want_scales) = dit::quantize(&x, width);
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            gpu.upload(&input, 0, bytes_of(&x)).unwrap();
            run(
                &gpu,
                &[kernels.quantize(&input, &bytes, &scales, rows, width)],
            )
            .unwrap();
            let got = gpu.download(&bytes, 0, rows * width).unwrap();
            let got_scales = floats(&gpu.download(&scales, 0, rows * 4).unwrap());
            assert_eq!(got_scales, want_scales, "width {width}");
            let differ = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| **a as i8 != **b)
                .count();
            assert_eq!(differ, 0, "width {width}");

            let rows = 8704;
            let input = gpu.buffer((rows * width) as u64 * 4).unwrap();
            let bytes = gpu.buffer((rows * width) as u64).unwrap();
            let scales = gpu.buffer(rows as u64 * 4).unwrap();
            let call = [kernels.quantize(&input, &bytes, &scales, rows, width)];
            run(&gpu, &call).unwrap();
            let started = std::time::Instant::now();
            for _ in 0..4 {
                run(&gpu, &call).unwrap();
            }
            println!(
                "{rows} rows of {width}: {:.2} ms",
                started.elapsed().as_secs_f64() * 250.0
            );
        }
    }
}
