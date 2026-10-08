//! Port acceptance on original Flash-Next weight slabs; NOT whole-model speed.
#[allow(dead_code)] // Shared GPU owner; this operator lab uses a subset of it.
#[path = "ornith/gpu.rs"]
mod gpu;
#[path = "ornith/kernels.rs"]
mod kernels;
use gpu::{Gpu, Result, Step};
use kuno_text::{GgufFile, GgufLimits, MetadataValue};
use std::time::Instant;
use wgpu::util::DeviceExt;
use zeroize::Zeroizing;

fn dispatch(gpu: &Gpu, step: &Step, repeats: usize) -> Result<u128> {
    let start = Instant::now();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&step.pipeline);
        pass.set_bind_group(0, &step.bindings, &[]);
        for _ in 0..repeats {
            pass.dispatch_workgroups(step.grid[0], step.grid[1], step.grid[2]);
        }
    }
    gpu.queue.submit([encoder.finish()]);
    gpu.wait()?;
    Ok(start.elapsed().as_nanos())
}

fn guard() -> Result<()> {
    if std::env::var("KUNO_LAB_NEXT_PUBLIC_KERNELS").as_deref() != Ok("1") {
        return Err("Fixed public Next port opt-in missing".into());
    }
    let limits = std::fs::read_to_string("/proc/self/limits")?;
    let fields = limits
        .lines()
        .find(|l| l.starts_with("Max core file size"))
        .ok_or("Core limit unavailable")?
        .split_whitespace()
        .collect::<Vec<_>>();
    if fields.get(4) != Some(&"0") || fields.get(5) != Some(&"0") {
        return Err("Core dumps must be disabled".into());
    }
    let cgroup = std::fs::read_to_string("/proc/self/cgroup")?;
    let rel = cgroup
        .lines()
        .find_map(|s| s.strip_prefix("0::"))
        .ok_or("Cgroupv2 required")?;
    if std::fs::read_to_string(format!("/sys/fs/cgroup{rel}/memory.swap.max"))?.trim() != "0" {
        return Err("Public lab swap guard required".into());
    }
    Ok(())
}

fn main() -> Result<()> {
    guard()?;
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3
        || !args[1]
            .ends_with("sha256-89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75")
    {
        return Err("Usage: flash-kernel-lab ORIGINAL_NEXT_BLOB FRESH_PUBLIC_RECEIPT".into());
    }
    let file = GgufFile::open(
        &args[1],
        GgufLimits {
            max_tensor_read_bytes: 512 * 1024 * 1024,
            ..Default::default()
        },
    )?;
    if file.file_len() != 119150722112
        || file
            .metadata()
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("qwen4exp")
    {
        return Err("Inspected original Next artifact mismatch".into());
    }
    let mut gpu = Gpu::new()?;
    if gpu.info.subgroup_min_size < 16 || gpu.info.subgroup_max_size > 32 {
        return Err("Inspected B70 subgroup bounds required".into());
    }
    gpu.state(16);
    let mut matrices = Vec::new();
    for (name, rows, cols, dtype) in [
        ("blk.0.ffn_gate_exps.weight", 640u32, 2560u32, 12u32),
        ("blk.0.ffn_up_exps.weight", 640, 2560, 12),
        ("blk.0.ffn_down_exps.weight", 2560, 640, 8),
        ("blk.6.ffn_down_exps.weight", 2560, 640, 6),
        ("blk.0.ffn_gate_inp.weight", 512, 2560, 0),
        ("output.weight", 248320, 2560, 14),
    ] {
        let view = file
            .tensor(name)
            .ok_or("Original Next port weight missing")?;
        let info = view.info();
        if info.row_len() != cols as u64
            || info.row_count() < rows as u64
            || info.dtype().gguf_id() != dtype
            || rows as u64 * info.row_bytes() > 512 * 1024 * 1024
        {
            return Err("Original Next projection geometry mismatch".into());
        }
        let mut packed = vec![0; rows as usize * info.row_bytes() as usize];
        view.read_range(0, &mut packed)?;
        let logical_bytes = packed.len();
        packed.resize(packed.len().next_multiple_of(4), 0);
        let weight = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("original-next-public-weight-slab"),
                contents: &packed,
                usage: wgpu::BufferUsages::STORAGE,
            });
        drop(packed);
        let input = Zeroizing::new(
            (0..cols as usize)
                .map(|i| ((i * 17 % 257) as i32 - 128) as f32 / 64.0)
                .collect::<Vec<_>>(),
        );
        let bytes = Zeroizing::new(
            input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        gpu.queue.write_buffer(&gpu.arena, 0, &bytes);
        let mut reference = Vec::new();
        let mut decoded = vec![0f32; cols as usize];
        for sample in 0u32..32 {
            let row = sample * (rows - 1) / 31;
            view.decode_row(row as u64, &mut decoded)?;
            let expected = decoded
                .iter()
                .zip(input.iter())
                .map(|(&w, &x)| w as f64 * x as f64)
                .sum::<f64>() as f32;
            reference.push((row, expected));
        }
        let mut variants = Vec::new();
        for vector in [false, true] {
            let constants = format!(
                "const ROWS:u32={rows}u;const COLUMNS:u32={cols}u;const ROW_BYTES:u32={}u;const ROUTED:bool=false;const RANKED_INPUT:bool=false;const BLOCK_DOT:bool=false;",
                info.row_bytes()
            );
            let pipeline = gpu.pipeline(
                &format!("next-{name}-{vector}"),
                kernels::matrix_shader(dtype, &constants, vector),
            );
            let step = gpu.step(
                pipeline,
                Some(&weight),
                None,
                None,
                [0, 4096, 0, 0, 0, 0, 0, 0],
                [rows.div_ceil(128 / gpu.info.subgroup_max_size), 1, 1],
            );
            dispatch(&gpu, &step, 3)?;
            let result = gpu.public_read(&gpu.arena, 4096 * 4, rows as u64 * 4)?;
            if result
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            {
                return Err("Next public projection non-finite".into());
            }
            let mut max_error = 0f32;
            for &(row, expected) in &reference {
                let ix = row as usize * 4;
                let actual = f32::from_le_bytes(result[ix..ix + 4].try_into().unwrap());
                let error = (actual - expected).abs();
                if error > 5e-4 + 5e-5 * expected.abs() {
                    return Err("Original Next projection tolerance failed".into());
                }
                max_error = max_error.max(error);
            }
            let mut samples = Vec::new();
            for _ in 0..3 {
                samples.push(dispatch(&gpu, &step, 20)?);
            }
            variants.push(serde_json::json!({"vector":vector,"maximum_absolute_error":max_error,"timed_dispatches_per_sample":20,"wall_nanoseconds":samples}));
        }
        matrices.push(serde_json::json!({"tensor":name,"rows":rows,"columns":cols,"dtype":dtype,"original_weight_bytes":logical_bytes,"independent_reference_rows":32,"variants":variants}));
    }
    let mut routes = Vec::new();
    let pipeline = gpu.pipeline(
        "next-cached-subgroup-router",
        kernels::shader(
            0,
            kernels::NEXT_ROUTER,
            include_str!("ornith/router_parallel.wgsl"),
        ),
    );
    let step = gpu.step(pipeline, None, None, None, [0; 8], [1, 1, 1]);
    for case in 0..6usize {
        let scores = Zeroizing::new(
            (0..512)
                .map(|i| match case {
                    0 => ((i * 71 % 509) as f32 - 253.0) / 31.0,
                    1 => 0.0,
                    2 => {
                        if i % 31 == 0 {
                            7.0
                        } else {
                            -9.0
                        }
                    }
                    3 => ((i % 17) as f32 - 8.0) * 10000.0,
                    4 => -(i as f32) / 37.0,
                    _ => {
                        if i == 17 {
                            1e20
                        } else {
                            -1e20
                        }
                    }
                })
                .collect::<Vec<_>>(),
        );
        let expected = kuno_text::ops::router_topk(&scores, 10, true)?;
        gpu.queue.write_buffer(
            &gpu.arena,
            0,
            &Zeroizing::new(
                scores
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        );
        let sentinel = vec![0x43; 4096];
        gpu.queue.write_buffer(&gpu.routing, 0, &sentinel);
        dispatch(&gpu, &step, 1)?;
        let result = gpu.public_read(&gpu.routing, 0, 4096)?;
        if result[80..] != sentinel[80..] {
            return Err("Next router changed owning interval".into());
        }
        let values = result[..80]
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (rank, &(id, expected_weight)) in expected.iter().enumerate() {
            let weight = f32::from_bits(values[rank * 2 + 1]);
            if values[rank * 2] != id as u32
                || !weight.is_finite()
                || (weight - expected_weight).abs() > 2e-7 + 2e-6 * expected_weight.abs()
            {
                return Err("Next cached GPU top10 disagreement".into());
            }
        }
        routes.push(serde_json::json!({"fixed_public_case":case,"top10_ids_match":true,"weights_match_tolerance":true,"untouched_owner_interval":true}));
    }
    gpu.clear()?;
    file.verify_file_unchanged()?;
    let receipt = serde_json::json!({"fixed_public_operator_port":true,"model_blob":"89059c880c2f000b24520919d4ba2edd2a8101327c7f8ab33b2280e01addda75","backend":"native-rust-wgpu-vulkan","trusted_shader_bounds":gpu.trusted_bounds,"matrices":matrices,"routing":routes,"whole_next_model_connected":false,"normal_private_clear_completed":true,"swap_disabled":true,"core_dumps_disabled":true});
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer_pretty(output, &receipt)?;
    println!(
        "public_next_kernel_port matrices=6 router_cases=6 numeric_checks_passed=true whole_model=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_next_packed_types_and_both_router_layouts_validate() {
        for dtype in [0, 1, 6, 8, 12, 14, 30] {
            for vector in [false, true] {
                let code = kernels::matrix_shader(
                    dtype,
                    "const ROWS:u32=32u;const COLUMNS:u32=2560u;const ROW_BYTES:u32=2100u;const ROUTED:bool=false;const RANKED_INPUT:bool=false;const BLOCK_DOT:bool=false;",
                    vector,
                );
                check(&code);
            }
        }
        for constants in [kernels::ORNITH_ROUTER, kernels::NEXT_ROUTER] {
            check(&kernels::shader(
                0,
                constants,
                include_str!("ornith/router_parallel.wgsl"),
            ));
        }
    }
    fn check(code: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(code)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}
