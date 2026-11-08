//! Fixed-public projections against independently decoded original GGUF rows.
use super::{
    gpu::{Gpu, Result},
    graph,
};
use kuno_text::GgufFile;
use wgpu::util::DeviceExt;
use zeroize::Zeroizing;

pub fn run(file: &GgufFile) -> Result<serde_json::Value> {
    graph::validate(file)?;
    let mut gpu = Gpu::new()?;
    gpu.state(16);
    let mut cases = Vec::new();
    for name in [
        "blk.0.ffn_gate_exps.weight",
        "output.weight",
        "blk.0.ffn_gate_inp.weight",
    ] {
        let view = file.tensor(name).ok_or("Public probe weight missing")?;
        let info = view.info();
        let cols = info.row_len() as usize;
        let rows = 32usize;
        for first_row in [0u64, 127] {
            let mut packed = vec![0; rows * info.row_bytes() as usize];
            view.read_range(first_row * info.row_bytes(), &mut packed)?;
            let weight = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("original-public-probe-weight-rows"),
                    contents: &packed,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let mut decoded = vec![vec![0f32; cols]; rows];
            for (i, row) in decoded.iter_mut().enumerate() {
                view.decode_row(first_row + i as u64, row)?;
            }
            for vector in 0..3 {
                let input = Zeroizing::new(
                    (0..cols)
                        .map(|i| match vector {
                            0 => ((i * 17 % 257) as i32 - 128) as f32 / 64.0,
                            1 => 1.0,
                            _ => {
                                if i % 2 == 0 {
                                    1.0
                                } else {
                                    -1.0
                                }
                            }
                        })
                        .collect::<Vec<_>>(),
                );
                let reference = Zeroizing::new(
                    decoded
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(input.iter())
                                .map(|(&w, &x)| w as f64 * x as f64)
                                .sum::<f64>() as f32
                        })
                        .collect::<Vec<_>>(),
                );
                let bytes = Zeroizing::new(
                    input
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                );
                gpu.queue.write_buffer(&gpu.arena, 0, &bytes);
                for (block, vector_dot) in [(false, false), (true, false), (false, true)] {
                    let code = graph::matrix_shader(
                        info.dtype().gguf_id(),
                        &format!(
                            "const ROWS:u32={rows}u;const COLUMNS:u32={cols}u;const ROW_BYTES:u32={}u;const ROUTED:bool=false;const RANKED_INPUT:bool=false;const BLOCK_DOT:bool={block};",
                            info.row_bytes()
                        ),
                        vector_dot,
                    );
                    let pipeline = gpu.pipeline(
                        &format!(
                            "public-probe-{}-{block}-{vector_dot}",
                            info.dtype().gguf_id()
                        ),
                        code,
                    );
                    let step = gpu.step(
                        pipeline,
                        Some(&weight),
                        None,
                        None,
                        [0, 4096, 0, 0, 0, 0, 0, 0],
                        [
                            (rows as u32).div_ceil(128 / gpu.info.subgroup_max_size),
                            1,
                            1,
                        ],
                    );
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&step.pipeline);
                        pass.set_bind_group(0, &step.bindings, &[]);
                        pass.dispatch_workgroups(step.grid[0], 1, 1);
                    }
                    gpu.queue.submit([encoder.finish()]);
                    let result = gpu.public_read(&gpu.arena, 4096 * 4, rows as u64 * 4)?;
                    let values = Zeroizing::new(
                        result
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                            .collect::<Vec<_>>(),
                    );
                    let mut max_absolute = 0f32;
                    let mut max_tolerance_fraction = 0f32;
                    for (&actual, &expected) in values.iter().zip(reference.iter()) {
                        let error = (actual - expected).abs();
                        let tolerance = 5e-4 + 5e-5 * expected.abs();
                        if !actual.is_finite() || error > tolerance {
                            return Err(
                                "Original public projection numeric tolerance failed".into()
                            );
                        }
                        max_absolute = max_absolute.max(error);
                        max_tolerance_fraction = max_tolerance_fraction.max(error / tolerance);
                    }
                    cases.push(serde_json::json!({"tensor":name,"first_row":first_row,"rows":rows,"public_vector":vector,"packed_block_dot":block,"vector_packed_dot":vector_dot,"maximum_absolute_error":max_absolute,"maximum_tolerance_fraction":max_tolerance_fraction}));
                }
            }
        }
    }
    let mut routing_cases = Vec::new();
    for case in 0..3 {
        let scores = Zeroizing::new(
            (0..256usize)
                .map(|i| match case {
                    0 => ((i * 17 % 257) as i32 - 128) as f32 / 64.0,
                    1 => 1.0,
                    _ => (i % 13) as f32 / 16.0,
                })
                .collect::<Vec<_>>(),
        );
        let mut ids = (0u32..256).collect::<Vec<_>>();
        ids.sort_by(|&a, &b| {
            scores[b as usize]
                .total_cmp(&scores[a as usize])
                .then(a.cmp(&b))
        });
        ids.truncate(8);
        let maximum = scores[ids[0] as usize];
        let mut weights = ids
            .iter()
            .map(|&i| ((scores[i as usize] - maximum) as f64).exp() as f32)
            .collect::<Vec<_>>();
        let sum = weights.iter().sum::<f32>();
        for w in &mut weights {
            *w /= sum;
        }
        let bytes = Zeroizing::new(
            scores
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        gpu.queue.write_buffer(&gpu.arena, 0, &bytes);
        let mut previous: Option<Zeroizing<Vec<u8>>> = None;
        for parallel in [false, true] {
            let pipeline = gpu.pipeline(
                &format!("public-router-{parallel}"),
                graph::shader(
                    0,
                    super::kernels::ORNITH_ROUTER,
                    if parallel {
                        include_str!("router_parallel.wgsl")
                    } else {
                        include_str!("router.wgsl")
                    },
                ),
            );
            let step = gpu.step(pipeline, None, None, None, [0; 8], [1, 1, 1]);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&step.pipeline);
                pass.set_bind_group(0, &step.bindings, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            gpu.queue.submit([encoder.finish()]);
            let actual = gpu.public_read(&gpu.routing, 0, 64)?;
            let values = actual
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            if values[..8] != ids
                || values[8..]
                    .iter()
                    .zip(&weights)
                    .any(|(&bits, &expected)| (f32::from_bits(bits) - expected).abs() > 2e-6)
            {
                return Err("Fixed public GPU router disagrees with independent ranking".into());
            }
            if previous
                .as_ref()
                .is_some_and(|old| old.as_slice() != actual.as_slice())
            {
                return Err("Parallel GPU router differs from scalar selections/weights".into());
            }
            previous = Some(actual);
            routing_cases.push(serde_json::json!({"public_score_case":case,"parallel":parallel,"exact_scalar_agreement":parallel}));
        }
    }
    gpu.clear()?;
    Ok(
        serde_json::json!({"fixed_public_projection_probe":true,"reference":"Rust original GGUF decoded rows with f64 accumulation","absolute_tolerance":5e-4,"relative_tolerance":5e-5,"cases":cases,"routing_cases":routing_cases,"normal_private_clear_completed":true}),
    )
}
