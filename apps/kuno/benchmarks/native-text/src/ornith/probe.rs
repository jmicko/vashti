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
                for block in [false, true] {
                    let code = graph::shader(
                        info.dtype().gguf_id(),
                        &format!(
                            "const ROWS:u32={rows}u;const COLUMNS:u32={cols}u;const ROW_BYTES:u32={}u;const ROUTED:bool=false;const RANKED_INPUT:bool=false;const BLOCK_DOT:bool={block};",
                            info.row_bytes()
                        ),
                        include_str!("matrix.wgsl"),
                    );
                    let pipeline = gpu.pipeline(
                        &format!("public-probe-{}-{block}", info.dtype().gguf_id()),
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
                    cases.push(serde_json::json!({"tensor":name,"first_row":first_row,"rows":rows,"public_vector":vector,"packed_block_dot":block,"maximum_absolute_error":max_absolute,"maximum_tolerance_fraction":max_tolerance_fraction}));
                }
            }
        }
    }
    gpu.clear()?;
    Ok(
        serde_json::json!({"fixed_public_projection_probe":true,"reference":"Rust original GGUF decoded rows with f64 accumulation","absolute_tolerance":5e-4,"relative_tolerance":5e-5,"cases":cases,"normal_private_clear_completed":true}),
    )
}
