//! Bounded original-rank count3 merge; CPU readiness never fakes GPU GU flags.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::{Error, Result};
const H: usize = 2560;
const OWNERS: usize = 256;
const ROUTES: usize = 30;
const MAX_ROUTES: usize = 80;
const VERSION: u32 = 1;
fn reject() -> Error {
    Error::InvalidFormat("public CPU overlap rank merge rejected")
}
/// Private owner packet is minted only after joined CPU output and exact masks.
/// This does not itself authenticate source origin; the source guard does.
pub(super) fn packet(masks: [u16; 3], cpu_joined: bool) -> Result<[u32; OWNERS]> {
    if masks.iter().any(|m| *m >= 1024) || masks.iter().any(|m| *m != 0) && !cpu_joined {
        return Err(reject());
    }
    let mut words = [0; OWNERS];
    words[..4].copy_from_slice(&[3, H as u32, 10, VERSION]);
    for route in 0..ROUTES {
        let cpu = masks[route / 10] & (1 << (route % 10)) != 0;
        words[4 + route] = u32::from(cpu);
        words[4 + ROUTES + route] = u32::from(cpu && cpu_joined);
    }
    Ok(words)
}
#[allow(clippy::eq_op, clippy::collapsible_if)] // Same original portable finite predicate; guarded reads.
#[cube(launch)]
pub(super) fn original_rank_merge(
    gpu: &Array<f32>,
    cpu: &Array<f32>,
    prob: &Array<f32>,
    owners: &Array<u32>,
    status: &Array<u32>,
    out: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let feature = CUBE_POS_X as usize * 64 + UNIT_POS_X as usize;
    let member = CUBE_POS_Y as usize;
    if feature < H && member < 3 {
        let mut ready =
            owners[0] == 3 && owners[1] == H as u32 && owners[2] == 10 && owners[3] == VERSION;
        let mut sum = 0.0f32;
        for rank in 0..10 {
            let route = member * 10 + rank;
            let owner = owners[4 + route];
            let cpu_ready = owners[4 + ROUTES + route];
            let mut part = 0.0f32;
            if owner == 0 {
                if cpu_ready != 0 || status[route] != 1 || status[MAX_ROUTES + route] != 1 {
                    ready = false;
                }
                part = gpu[route * H + feature];
            } else if owner == 1 {
                if cpu_ready != 1 || status[route] != 0 || status[MAX_ROUTES + route] != 0 {
                    ready = false;
                }
                part = cpu[route * H + feature];
            } else {
                ready = false;
            }
            let p = prob[route];
            // Evaluate even when p==0: non-finite partials must never hide.
            if p != p || p.abs() > f32::MAX || part != part || part.abs() > f32::MAX {
                ready = false;
            }
            sum += p * part;
        }
        if sum != sum || sum.abs() > f32::MAX {
            ready = false;
        }
        for i in 4 + 2 * ROUTES..OWNERS {
            if owners[i] != 0 {
                ready = false;
            }
        }
        if ready {
            out[member * H + feature] = sum;
        } else {
            out[member * H + feature] = 0.0;
            fault[0].fetch_or(1);
        }
    }
}
/// All-GPU validation uses this same merge but never changes original readiness.
/// Parent validates every typed/full physical owner before private fill; this
/// final launch repeats logical shape, no-alias and same-stream checks.
pub(super) fn launch(arena: &PrivateArena, ids: [BufferId; 7]) -> Result<()> {
    let expected = [8 * 10 * H, 3 * 10 * H, 80, OWNERS, 4800, 8 * H, 1];
    let mut handles = Vec::with_capacity(7);
    for (i, id) in ids.iter().copied().enumerate() {
        if ids[..i].contains(&id) || u::words(arena, id)? != expected[i] {
            return Err(reject());
        }
        let h = arena.handle(id).map_err(u::fault)?.clone();
        let n = arena.layout(id).map_err(u::fault)?.owned_bytes as u64;
        if h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != n
            || h.size_in_used() != n
            || handles
                .first()
                .is_some_and(|old: &cubecl::server::Handle| old.stream != h.stream)
        {
            return Err(reject());
        }
        handles.push(h);
    }
    let actual = u::resources(arena.client().map_err(u::fault)?, handles)?;
    if actual.len() != 7 {
        return Err(reject());
    }
    for (i, r) in actual.iter().enumerate() {
        let r = r.resource();
        let n = arena.layout(ids[i]).map_err(u::fault)?.owned_bytes as u64;
        if r.offset != 0
            || r.size != n
            || r.buffer.size() != n
            || actual[..i]
                .iter()
                .any(|old| old.resource().buffer == r.buffer)
        {
            return Err(reject());
        }
    }
    original_rank_merge::launch::<WgpuRuntime>(
        arena.client().map_err(u::fault)?,
        CubeCount::Static(40, 3, 1),
        CubeDim::new_1d(64),
        u::arg(arena, ids[0])?,
        u::arg(arena, ids[1])?,
        u::arg(arena, ids[2])?,
        u::arg(arena, ids[3])?,
        u::arg(arena, ids[4])?,
        u::arg(arena, ids[5])?,
        u::arg(arena, ids[6])?,
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_ready_packet_does_not_mint_gpu_flags() {
        let p = packet([529; 3], true).unwrap();
        assert_eq!(&p[..4], &[3, 2560, 10, 1]);
        for i in 0..30 {
            assert_eq!(p[4 + i], u32::from([0, 4, 9].contains(&(i % 10))));
            assert_eq!(p[34 + i], p[4 + i]);
        }
        assert!(p[64..].iter().all(|x| *x == 0));
        assert!(packet([529; 3], false).is_err());
        assert!(packet([1024, 0, 0], true).is_err());
        assert!(packet([0; 3], false).is_ok());
    }
}
#[cfg(test)]
mod shader_tests {
    use super::*;
    #[test]
    fn actual_merge_shader_compiles_checked_and_direct_with_seven_storage_bindings() {
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        use cubecl::{
            compute::KernelBuilder,
            ir::{ElemType, FloatKind, StorageType, Type, UIntKind},
            server::ExecutionMode,
            wgpu::WgslCompiler,
        };
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
            let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
            let atom = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
            let mut b = KernelBuilder::default();
            b.scope.register_type::<f32>(f);
            b.scope.register_type::<u32>(u);
            b.scope.register_type::<usize>(u);
            b.scope
                .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
            let gpu = b.input_array(Type::new(f)).into();
            let cpu = b.input_array(Type::new(f)).into();
            let prob = b.input_array(Type::new(f)).into();
            let owners = b.input_array(Type::new(u)).into();
            let status = b.input_array(Type::new(u)).into();
            let out = b.output_array(Type::new(f)).into();
            let fault = b.output_array(Type::new(atom)).into();
            original_rank_merge::expand(&mut b.scope, gpu, cpu, prob, owners, status, out, fault);
            let source = WgslCompiler::default()
                .compile(
                    b.build(
                        KernelSettings::default()
                            .cube_dim(CubeDim::new_1d(64))
                            .kernel_name("public_original_rank_merge"),
                    ),
                    &WgpuCompilationOptions::default(),
                    mode,
                    u,
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
            assert!(source.contains("atomicOr"));
            assert_eq!(
                module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| matches!(v.space, wgpu::naga::AddressSpace::Storage { .. }))
                    .count(),
                8
            );
        }
    }
}
