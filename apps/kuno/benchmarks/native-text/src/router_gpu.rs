//! Explicit fixed count1 top10. Original logits/IDs, f32 selected softmax.
//! Reuses the caller's guarded owning packet; no new private allocation.
use cubecl::prelude::*;

pub const EXPERTS: usize = 512;
pub const RANKS: usize = 10;
pub const FIRST: usize = 2560 + EXPERTS + 1;
pub const END: usize = FIRST + RANKS * 2;

#[cube(launch)]
#[allow(clippy::collapsible_if)]
pub fn top10(router: &Array<u32>, packet: &mut Array<u32>, router_first: usize) {
    let lane = UNIT_POS as usize;
    let mut best = SharedMemory::<f32>::new(64_usize);
    let mut ids = SharedMemory::<u32>::new(64_usize);
    let mut picked = SharedMemory::<u32>::new(RANKS);
    let mut scores = SharedMemory::<f32>::new(RANKS);
    for rank in 0_usize..RANKS {
        let mut value = 0.0_f32;
        value -= 3.402_823_5e38_f32;
        let mut id = 512_u32;
        for tile in 0_usize..8_usize {
            let candidate = lane + tile * 64;
            let mut used = false;
            for previous in 0_usize..rank {
                if picked[previous] == candidate as u32 {
                    used = true;
                }
            }
            let score = f32::reinterpret(router[router_first + candidate]);
            if !used && (score > value || (score == value && candidate < id as usize)) {
                value = score;
                id = candidate as u32;
            }
        }
        best[lane] = value;
        ids[lane] = id;
        sync_cube();
        let mut stride = 32_usize;
        while stride > 0 {
            if lane < stride {
                let other = best[lane + stride];
                let other_id = ids[lane + stride];
                if other > best[lane] || (other == best[lane] && other_id < ids[lane]) {
                    best[lane] = other;
                    ids[lane] = other_id;
                }
            }
            sync_cube();
            stride /= 2;
        }
        if lane == 0 {
            picked[rank] = ids[0];
            scores[rank] = best[0];
        }
        sync_cube();
    }
    if lane == 0 {
        let maximum = scores[0];
        let mut denominator = 0.0_f32;
        for rank in 0_usize..RANKS {
            scores[rank] = (scores[rank] - maximum).exp();
            denominator += scores[rank];
        }
        for rank in 0_usize..RANKS {
            packet[FIRST + rank * 2] = picked[rank];
            packet[FIRST + rank * 2 + 1] = (scores[rank] / denominator).to_bits();
        }
    }
}

/// Fixed public numeric checks only. Caller retains the original transient
/// stream and performs its mandatory audited cleanup before model construction.
pub fn probe(
    client: &cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
) -> Result<(), Box<dyn std::error::Error>> {
    use cubecl::wgpu::WgpuRuntime;
    for case in 0..6_usize {
        let mut logits = [0.0_f32; EXPERTS];
        for (id, value) in logits.iter_mut().enumerate() {
            *value = match case {
                0 => ((id * 71 % 509) as f32 - 253.) / 31.,
                1 => 0.,
                2 => {
                    if id % 31 == 0 {
                        7.
                    } else {
                        -9.
                    }
                }
                3 => ((id % 17) as f32 - 8.) * 10000.,
                4 => -(id as f32) / 37.,
                _ => {
                    if id == 17 {
                        1e20
                    } else {
                        -1e20
                    }
                }
            };
        }
        let expected = kuno_text::ops::router_topk(&logits, RANKS, true)?;
        let first = 3;
        let mut input = vec![0_u32; 8192];
        for (id, value) in logits.iter().enumerate() {
            input[first + id] = value.to_bits();
        }
        let mut initial = vec![0_u32; 8192];
        initial[FIRST - 1] = 0x43434343;
        initial[END] = 0x51515151;
        let router = client.create_from_slice(u32::as_bytes(&input));
        let packet = client.create_from_slice(u32::as_bytes(&initial));
        // SAFETY: both direct buffers above contain8192 u32 words; one64-lane
        // cube, checked first+512 and [FIRST,END) are inside these full owners.
        unsafe {
            top10::launch::<WgpuRuntime>(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(router, 8192),
                ArrayArg::from_raw_parts(packet.clone(), 8192),
                first,
            );
        }
        let bytes = client.read_one(packet)?;
        let actual = u32::from_bytes(&bytes);
        if actual.len() != initial.len()
            || actual[..FIRST] != initial[..FIRST]
            || actual[END..] != initial[END..]
        {
            return Err("fixed GPU router changed owning interval".into());
        }
        let mut sum = 0.;
        for (rank, &(id, weight)) in expected.iter().enumerate() {
            let got = f32::from_bits(actual[FIRST + rank * 2 + 1]);
            if actual[FIRST + rank * 2] != id as u32
                || !got.is_finite()
                || got < 0.
                || (got - weight).abs() > 2e-7 + 2e-6 * weight.abs()
            {
                return Err("fixed GPU router numeric check failed".into());
            }
            sum += got;
        }
        if (sum - 1.0).abs() > 2e-6 {
            return Err("fixed GPU router normalization check failed".into());
        }
    }
    println!("rust_gpu_router cases=6 ranks=10 numeric_and_owning_checks=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shader_has_fixed_top10_trees_and_valid_unconditional_barriers() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let mut b = KernelBuilder::default();
        let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        b.scope.register_type::<u32>(uint);
        b.scope.register_type::<usize>(uint);
        b.scope
            .register_type::<f32>(StorageType::Scalar(ElemType::Float(FloatKind::F32)));
        b.scope
            .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
        let router = b.input_array(Type::new(uint)).into();
        let packet = b.output_array(Type::new(uint)).into();
        let first = b.scalar(uint).into();
        top10::expand(&mut b.scope, router, packet, first);
        let source = WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(64))
                        .kernel_name("fixed_gpu_top10"),
                ),
                &WgpuCompilationOptions::default(),
                ExecutionMode::Checked,
                uint,
            )
            .unwrap()
            .to_string();
        let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert!(source.contains("workgroupBarrier"));
        assert!(source.contains("exp("));
        assert!(source.contains("3073"));
    }
}
