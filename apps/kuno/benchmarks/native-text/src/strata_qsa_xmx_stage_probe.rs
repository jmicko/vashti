//! Fixed-public GPU control of the actual production hi/lo stage and CMMA.
//! Only aggregate differences are reported; no private inputs or timing path.
use super::*;
use crate::{gr_batch_probe, private_arena::PrivateArena};
use zeroize::Zeroizing;

#[cube(launch_unchecked)]
fn stage_identity(input: &Array<f32>, output: &mut Array<f32>, fault: &mut Array<Atomic<u32>>) {
    let lane = usize::cast_from(UNIT_POS);
    let base = usize::cast_from(CUBE_POS_X) * 128;
    let mut hi = SharedMemory::<f16>::new(128usize);
    let mut lo = SharedMemory::<f16>::new(128usize);
    let mut identity = SharedMemory::<f16>::new(256usize);
    let mut result = SharedMemory::<f32>::new(128usize);
    let mut i = lane;
    while i < 128 {
        let x = input[base + i];
        if !(x.abs() <= f32::MAX) {
            fault[0].fetch_or(1);
        }
        stage_split(&mut hi, &mut lo, i, x);
        i += 16;
    }
    i = lane;
    while i < 256 {
        let mut x = f32::cast_from(0u32);
        if i / 16 == i % 16 {
            x = f32::cast_from(1u32);
        }
        identity[i] = f16::cast_from(x);
        i += 16;
    }
    sync_cube();
    let a = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    let b = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let c = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    cmma::load(&b, &identity.to_slice(), 16);
    cmma::load(&a, &hi.to_slice(), 16);
    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);
    cmma::load(&a, &lo.to_slice(), 16);
    cmma::execute::<f16, f16, f32, f32>(&a, &b, &c, &c);
    cmma::store(
        &mut result.to_slice_mut(),
        &c,
        16,
        cmma::MatrixLayout::RowMajor,
    );
    sync_cube();
    i = lane;
    while i < 128 {
        output[(base + i) * 3] = f32::cast_from(hi[i]);
        output[(base + i) * 3 + 1] = f32::cast_from(lo[i]);
        output[(base + i) * 3 + 2] = result[i];
        i += 16;
    }
}

pub(crate) fn definition() -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(f32::as_type(&b.scope));
    let out = b.output_array(f32::as_type(&b.scope));
    let f = b.output_array(
        cubecl::ir::StorageType::Atomic(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32))
            .into(),
    );
    stage_identity::expand(&mut b.scope, input.into(), out.into(), f.into());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(16)))
}
pub(crate) fn run(
    arena: &mut PrivateArena,
) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error>> {
    if crate::strata_prompt::active() {
        return Err("public XMX stage must not fabricate a prompt scope".into());
    }
    const N: usize = 1024;
    let input = Zeroizing::new(
        (0..N)
            .map(|i| {
                let x = if i < 512 {
                    (((i * 17 % 43) as f32 - 21.0) / 31.0 + ((i / 256 % 7) as f32 - 3.0) / 127.0)
                        * 16384.0
                } else {
                    ((i * 29 % 997) as f32 / 997.0) * 16384.0
                };
                x.to_bits()
            })
            .collect::<Vec<_>>(),
    );
    let source = arena.upload_u32(&input)?;
    let out = arena.allocate_zeroed(N * 3)?;
    let fault = arena.allocate_zeroed(1)?;
    let ih = crate::resident_util::handle(arena, source)?;
    let oh = crate::resident_util::handle(arena, out)?;
    let fh = crate::resident_util::handle(arena, fault)?;
    if ih.size_in_used() != N as u64 * 4
        || oh.size_in_used() != N as u64 * 12
        || fh.size_in_used() != 4
        || [&ih, &oh, &fh]
            .into_iter()
            .any(|v| !arena.contains_private_view(v).unwrap_or(false))
    {
        return Err("public XMX stage original logical owners".into());
    }
    // SAFETY: only the compiled-public data above, authentic separate owners,
    // complete128-element groups and the original configured native device.
    unsafe {
        stage_identity::launch_unchecked::<WgpuRuntime>(
            arena.client()?,
            CubeCount::Static((N / 128) as u32, 1, 1),
            CubeDim::new_1d(16),
            crate::resident_util::raw_arg(ih, N)?,
            crate::resident_util::raw_arg(oh, N * 3)?,
            crate::resident_util::raw_arg(fh, 1)?,
        );
    }
    cubecl::future::block_on(arena.client()?.sync())?;
    let actual = gr_batch_probe::read_native_public(arena, out)?;
    let faults = gr_batch_probe::read_native_public(arena, fault)?;
    let mut mismatches = [0usize; 3];
    let mut max_abs = [0.0f64; 3];
    let mut nonfinite = 0;
    for (i, bits) in input.iter().enumerate() {
        let x = f32::from_bits(*bits);
        let hi = f16::from_f32(x).to_f32();
        let lo = f16::from_f32(x - hi).to_f32();
        for (j, reference) in [hi, lo, hi + lo].into_iter().enumerate() {
            let a = f32::from_bits(actual[i * 3 + j]);
            if a.to_bits() != reference.to_bits() {
                mismatches[j] += 1;
            }
            if a.is_finite() {
                max_abs[j] = max_abs[j].max((f64::from(a) - f64::from(reference)).abs());
            } else {
                nonfinite += 1;
            }
        }
    }
    let padding_zero = actual[N * 3..].iter().all(|&w| w == 0) && faults.iter().all(|&w| w == 0);
    Ok(
        serde_json::json!({"passed":mismatches==[0;3]&&padding_zero&&nonfinite==0,"compiled_public_fixture_only":true,"content_saved":false,"actual_shared_production_stage_helper":true,"actual_cooperative_identity_product":true,"original_half_rte_host_reference":true,"elements":N,"high_mismatches":mismatches[0],"low_mismatches":mismatches[1],"identity_product_mismatches":mismatches[2],"max_abs":max_abs,"nonfinite":nonfinite,"full_original_parent_padding_and_fault_zero":padding_zero}),
    )
}
