//! Deep K64 routed prefix, original publication helpers and authentic tail.
use super::*;
pub(crate) const OPERAND_BYTES: usize = 65536;
pub(crate) const LOGICAL_BYTES: usize = OPERAND_BYTES + 16384;
pub(crate) fn geometry(shared: usize, explicit: bool) -> bool {
    shared >= OPERAND_BYTES && explicit
}
pub(crate) fn selected(client: &ComputeClient<WgpuRuntime>) -> bool {
    let shared = client.properties().hardware.max_shared_memory_size;
    shared >= OPERAND_BYTES && geometry(shared, client.native_explicit_shared_layout())
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_prefix(
    client: &ComputeClient<WgpuRuntime>,
    args: [ArrayArg<WgpuRuntime>; 6],
    grid: (usize, usize),
    words: [u32; 7],
    down: bool,
    direct: bool,
) {
    let [ia, wa, oa, ma, sa, fa] = args;
    let [
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        prefix,
    ] = words;
    // SAFETY: the caller has retained the complete ORIGINAL source/owner/map/
    // geometry/CMMA/grid preflight and preconverted the authentic original tail.
    unsafe {
        if down {
            expert_wide_down::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                member_base,
                prefix,
            );
        } else if direct {
            expert_wide_gu_direct::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                member_base,
                prefix,
            );
        } else {
            expert_wide_gu::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(grid.0 as u32, grid.1 as u32, 1),
                CubeDim::new_1d(THREADS),
                ia,
                wa,
                oa,
                ma,
                sa,
                fa,
                columns,
                rows,
                count,
                weight_first,
                group,
                member_base,
                prefix,
            );
        }
    }
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn stage<F: Float>(
    input: &Array<Vector<F, Const<4>>>,
    weight: &Array<Vector<F, Const<4>>>,
    a: &mut SharedMemory<Vector<F, Const<4>>>,
    b: &mut SharedMemory<Vector<F, Const<4>>>,
    columns: usize,
    weight_first: usize,
    token_base: usize,
    row_base: usize,
    k: usize,
    buffer: usize,
) {
    // Full128x128x64 stages only. Two genuine vec4s per job preserve original
    // FP16/BF16 bits; every one of the2048 vectors per operand is written once.
    #[unroll]
    for round in 0..2usize {
        let job = usize::cast_from(UNIT_POS) + round * comptime!(STAGE_THREADS);
        let target = buffer * 2048 + (job % 8 / 2) * 512 + (job / 8) * 4 + (job % 2) * 2;
        let offset = k + (job % 8) * 8;
        let row = job / 8;
        let source_a = ((token_base + row) * columns + offset) / 4;
        let source_b = (weight_first + (row_base + row) * columns + offset) / 4;
        a[target] = input[source_a];
        a[target + 1] = input[source_a + 1];
        b[target] = weight[source_b];
        b[target + 1] = weight[source_b + 1];
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn wide(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    member_base: u32,
    piece_members: u32,
    #[comptime] down: bool,
    #[comptime] direct_gu: bool,
) {
    let columns = usize::cast_from(columns);
    let rows = usize::cast_from(rows);
    let count = usize::cast_from(count);
    let weight_first = usize::cast_from(weight_first);
    let group = usize::cast_from(group);
    let member_base = usize::cast_from(member_base);
    let piece_members = usize::cast_from(piece_members);
    // Every lane reads the same immutable header at identical scalar indices.
    // Make that existing uniform result visible to the compiler's liveness
    // analysis, so the later publication storage can reuse the freed operands.
    // This vote preserves both valid and invalid results and uses no SLM.
    if !plane_all(header_valid(map, count, group))
        || member_base != 0
        || piece_members != count / 128 * 128
        || columns != comptime!(if down { 640 } else { 2560 })
        || rows != comptime!(if down { 2560 } else { 1280 })
    {
        if UNIT_POS == 0 {
            fault[0].fetch_or(128);
        }
        terminate!();
    }
    // Two typed K64 stages per operand are32KiB each. Their real final
    // reader barrier and explicit free precede the16KiB publication lifetime.
    let mut a = SharedMemory::<Vector<f16, Const<4>>>::new(4096usize);
    let mut b = SharedMemory::<Vector<f16, Const<4>>>::new(4096usize);
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let a1 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::A,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b0 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    // SAFETY: every full current stage is initialized and synchronized before use.
    let b1 = unsafe {
        cmma::Matrix::<f16>::uninitialized(
            cmma::MatrixIdent::B,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
        )
    };
    let c0 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c1 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c2 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c3 = cmma::Matrix::<f32>::from_value(
        cmma::MatrixIdent::Accumulator,
        8usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0.0,
    );
    let c4 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c0
    };
    let c5 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c1
    };
    let c6 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c2
    };
    let c7 = if comptime!(M_FRAGMENTS == 4) {
        cmma::Matrix::<f32>::from_value(
            cmma::MatrixIdent::Accumulator,
            8usize,
            16usize,
            16usize,
            cmma::MatrixLayout::Undefined,
            0.0,
        )
    } else {
        c3
    };
    let subgroup = usize::cast_from(PLANE_POS);
    let token_base = usize::cast_from(CUBE_POS_Y) * 128;
    let row_base = usize::cast_from(CUBE_POS_X) * 128;
    stage::<f16>(
        input,
        weight,
        &mut a,
        &mut b,
        columns,
        weight_first,
        token_base,
        row_base,
        0usize,
        0usize,
    );
    sync_cube();
    let mut k = 0usize;
    let mut current = 0usize;
    while k < columns {
        let next = 1usize - current;
        let mut next_a_lo0 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_a_hi0 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_a_lo1 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_a_hi1 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_lo0 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_hi0 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_lo1 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        let mut next_b_hi1 = Vector::<f16, Const<4>>::new(f16::new(0.0));
        // Both actual future job rounds stay in registers across all four
        // original chronological K16 groups. They enter only the next SLM bank.
        if comptime!(cfg!(feature = "strata-wide-register-stage")) {
            if k + 64 < columns {
                let job0 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let job1 = job0 + 512;
                let row0 = job0 / 8;
                let off0 = k + 64 + (job0 % 8) * 8;
                let aa0 = ((token_base + row0) * columns + off0) / 4;
                let bb0 = (weight_first + (row_base + row0) * columns + off0) / 4;
                next_a_lo0 = input[aa0];
                next_a_hi0 = input[aa0 + 1];
                next_b_lo0 = weight[bb0];
                next_b_hi0 = weight[bb0 + 1];
                let row1 = job1 / 8;
                let off1 = k + 64 + (job1 % 8) * 8;
                let aa1 = ((token_base + row1) * columns + off1) / 4;
                let bb1 = (weight_first + (row_base + row1) * columns + off1) / 4;
                next_a_lo1 = input[aa1];
                next_a_hi1 = input[aa1 + 1];
                next_b_lo1 = weight[bb1];
                next_b_hi1 = weight[bb1 + 1];
            }
        } else if k + 64 < columns {
            stage::<f16>(
                input,
                weight,
                &mut a,
                &mut b,
                columns,
                weight_first,
                token_base,
                row_base,
                k + 64,
                next,
            );
        }
        #[unroll]
        for step in 0..4usize {
            let af = current * 2048 + step * 512 + (subgroup / 4) * 64;
            let bf = current * 2048 + step * 512 + (subgroup % 4) * 128;
            cmma::load(&a0, &a.to_slice().slice(af, a.len()), 16);
            cmma::load(&a1, &a.to_slice().slice(af + 32, a.len()), 16);
            cmma::load(&b0, &b.to_slice().slice(bf, b.len()), 16);
            cmma::load(&b1, &b.to_slice().slice(bf + 64, b.len()), 16);
            cmma::execute::<f16, f16, f32, f32>(&a0, &b0, &c0, &c0);
            cmma::execute::<f16, f16, f32, f32>(&a0, &b1, &c1, &c1);
            cmma::execute::<f16, f16, f32, f32>(&a1, &b0, &c2, &c2);
            cmma::execute::<f16, f16, f32, f32>(&a1, &b1, &c3, &c3);
        }
        if comptime!(cfg!(feature = "strata-wide-register-stage")) {
            if k + 64 < columns {
                let job0 = usize::cast_from(PLANE_POS) * 16 + usize::cast_from(UNIT_POS_PLANE);
                let job1 = job0 + 512;
                let at0 = next * 2048 + (job0 % 8 / 2) * 512 + (job0 / 8) * 4 + (job0 % 2) * 2;
                a[at0] = next_a_lo0;
                a[at0 + 1] = next_a_hi0;
                b[at0] = next_b_lo0;
                b[at0 + 1] = next_b_hi0;
                let at1 = next * 2048 + (job1 % 8 / 2) * 512 + (job1 / 8) * 4 + (job1 % 2) * 2;
                a[at1] = next_a_lo1;
                a[at1 + 1] = next_a_hi1;
                b[at1] = next_b_lo1;
                b[at1 + 1] = next_b_hi1;
            }
        }
        // Original uniform handoff ends current readers and publishes next.
        sync_cube();
        current = next;
        k += 64;
    }
    // SAFETY: every lane passed the final whole-workgroup handoff; no operand
    // pointer is used by the original publication helpers below.
    unsafe {
        a.free();
        b.free();
    }
    let mut scratch = SharedMemory::<f32>::new(comptime!(SUBGROUPS * 128));
    if comptime!(direct_gu) {
        publish_direct_gu(
            &c0,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            0usize,
            0usize,
        );
        publish_direct_gu(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            0usize,
            1usize,
        );
        publish_direct_gu(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            1usize,
            0usize,
        );
        publish_direct_gu(
            &c3,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            1usize,
            1usize,
        );
    } else {
        publish_routed(
            &c0,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            0usize,
            0usize,
        );
        publish_routed(
            &c1,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            0usize,
            1usize,
        );
        publish_routed(
            &c2,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            1usize,
            0usize,
        );
        publish_routed(
            &c3,
            &mut scratch,
            output,
            map,
            status,
            fault,
            token_base,
            row_base,
            group,
            down,
            1usize,
            1usize,
        );
        if comptime!(M_FRAGMENTS == 4) {
            publish_routed(
                &c4,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                2usize,
                0usize,
            );
            publish_routed(
                &c5,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                2usize,
                1usize,
            );
            publish_routed(
                &c6,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                3usize,
                0usize,
            );
            publish_routed(
                &c7,
                &mut scratch,
                output,
                map,
                status,
                fault,
                token_base,
                row_base,
                group,
                down,
                3usize,
                1usize,
            );
        }
    }
}

#[cube(launch_unchecked)]
pub fn expert_wide_gu(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        false,
        false,
    );
}

// Literal role is authenticated by the actual generated entry identity.
#[cube(launch_unchecked)]
pub fn expert_wide_gu_direct(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        false,
        true,
    );
}

// Literal role is authenticated by the actual generated entry identity.
#[cube(launch_unchecked)]
pub fn expert_wide_down(
    input: &Array<Vector<f16, Const<4>>>,
    weight: &Array<Vector<f16, Const<4>>>,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    columns: u32,
    rows: u32,
    count: u32,
    weight_first: u32,
    group: u32,
    member_base: u32,
    piece_members: u32,
) {
    wide(
        input,
        weight,
        output,
        map,
        status,
        fault,
        columns,
        rows,
        count,
        weight_first,
        group,
        member_base,
        piece_members,
        true,
        false,
    );
}

// Literal role is authenticated by the actual generated entry identity.

fn definition(down: bool, direct_gu: bool) -> KernelDefinition {
    let mut b = KernelBuilder::default();
    cubecl::ir::AddressType::U32.register(&mut b.scope);
    let input = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let weight = b.input_array(Vector::<f16, Const<4>>::as_type(&b.scope));
    let output = b.output_array(f32::as_type(&b.scope));
    let map = b.input_array(u32::as_type(&b.scope));
    let status = b.output_array(u32::as_type(&b.scope));
    let fault = b.output_array(
        cubecl::ir::StorageType::Atomic(ElemType::UInt(cubecl::ir::UIntKind::U32)).into(),
    );
    let a: [cubecl::ir::ManagedVariable; 7] =
        std::array::from_fn(|_| b.scalar(u32::as_type(&b.scope).storage_type()));
    if direct_gu {
        expert_wide_gu_direct::expand(
            &mut b.scope,
            input.into(),
            weight.into(),
            output.into(),
            map.into(),
            status.into(),
            fault.into(),
            a[0].clone().into(),
            a[1].clone().into(),
            a[2].clone().into(),
            a[3].clone().into(),
            a[4].clone().into(),
            a[5].clone().into(),
            a[6].clone().into(),
        );
    } else if down {
        expert_wide_down::expand(
            &mut b.scope,
            input.into(),
            weight.into(),
            output.into(),
            map.into(),
            status.into(),
            fault.into(),
            a[0].clone().into(),
            a[1].clone().into(),
            a[2].clone().into(),
            a[3].clone().into(),
            a[4].clone().into(),
            a[5].clone().into(),
            a[6].clone().into(),
        );
    } else {
        expert_wide_gu::expand(
            &mut b.scope,
            input.into(),
            weight.into(),
            output.into(),
            map.into(),
            status.into(),
            fault.into(),
            a[0].clone().into(),
            a[1].clone().into(),
            a[2].clone().into(),
            a[3].clone().into(),
            a[4].clone().into(),
            a[5].clone().into(),
            a[6].clone().into(),
        );
    }

    assert!(b.scope.pop_errors().is_empty());
    b.build(KernelSettings::default().cube_dim(CubeDim::new_1d(THREADS)))
}

fn compile(
    down: bool,
    direct: bool,
    mask: u8,
) -> std::result::Result<Vec<u32>, Box<dyn std::error::Error>> {
    use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
    use cubecl::{Compiler, WgpuCompilationOptions};
    let mut options = WgpuCompilationOptions {
        supports_vulkan: true,
        ..Default::default()
    };
    options.vulkan.max_spirv_version = (1, 6);
    options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
    options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
    options.vulkan.supports_explicit_smem = mask & 4 != 0;
    let mut compiler=AutoCompiler::SpirV(Default::default());
    let AutoRepresentation::SpirV(kernel) = Compiler::compile(
        &mut compiler,
        definition(down, direct),
        &options,
        ExecutionMode::Unchecked,
        cubecl::ir::StorageType::Scalar(ElemType::UInt(cubecl::ir::UIntKind::U32)),
    )?
    else {
        return Err("deep wide expert compiler representation".into());
    };
    if kernel.shared_size
        != if mask & 4 != 0 {
            OPERAND_BYTES
        } else {
            LOGICAL_BYTES
        }
    {
        #[cfg(test)]
        if let AutoCompiler::SpirV(c)=&compiler {
            eprintln!("deep_public_shared_allocations {:?}",c.shared_liveness.allocations);
            for node in c.opt.node_ids() {
                let live=c.shared_liveness.at_block(node);
                if !live.is_empty() {
                    eprintln!("deep_public_shared_block node={node:?} uniform={} live={live:?}",c.uniformity.is_block_uniform(node));
                }
            }
        }
        return Err(format!("deep wide expert actual physical/logical shared layout: actual={} expected={} mask={mask} down={down} direct={direct}",
            kernel.shared_size,if mask&4!=0 {OPERAND_BYTES}else{LOGICAL_BYTES}).into());
    }
    let words = wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)?;
    let words =
        wgpu::hal::vulkan::kuno_cooperative_subgroup_shared_barriers(&words)?.unwrap_or(words);
    let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)?.unwrap_or(words);
    if wgpu::hal::vulkan::kuno_cooperative_subgroup(&words) != Some(16) {
        return Err("deep wide expert original subgroup".into());
    }
    let (mut at, mut products, mut loads, mut stores, mut barriers, mut faults, mut local) =
        (5, 0, 0, 0, 0, 0, false);
    while at < words.len() {
        let n = (words[at] >> 16) as usize;
        if n == 0 || at.checked_add(n).is_none_or(|end| end > words.len()) {
            return Err("deep wide expert instruction extent".into());
        }
        let i = &words[at..at + n];
        match i[0] & 65535 {
            4459 => products += 1,
            4457 => loads += 1,
            4458 => stores += 1,
            224 => barriers += 1,
            241 => faults += 1,
            16 if i.len() == 6 && i[2] == 17 => {
                local = i[3..] == [512, 1, 1];
            }
            _ => {}
        }
        at += n;
    }
    if products != 16
        || loads != 16
        || stores != if direct { 8 } else { 4 }
        || barriers != 10
        || faults == 0
        || !local
    {
        return Err(
            "deep wide expert actual chronological products/owners/publication/geometry".into(),
        );
    }
    super::super::wide_pipeline::validate_program_expected(
        &words,
        16,
        "strata_deep_expert_register_stage_accepted",
    )?;
    Ok(words)
}

pub(crate) fn validate_sources() -> std::result::Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    for mask in 0..8 {
        for (down, direct) in [(false, false), (false, true), (true, false)] {
            let words = compile(down, direct, mask)?;
            let mut validator = std::process::Command::new("spirv-val");
            validator.args(["--target-env", "vulkan1.3"]);
            if mask & 1 != 0 {
                validator.arg("--uniform-buffer-standard-layout");
            }
            let mut command = validator
                .arg("-")
                .stdin(std::process::Stdio::piped())
                .spawn()?;
            command
                .stdin
                .take()
                .ok_or("deep wide expert validator stdin")?
                .write_all(
                    &words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
            if !command.wait()?.success() {
                return Err("deep wide expert SPIRV validation".into());
            }
            println!(
                "strata_deep_expert_shader_accepted down={down} direct={direct} layout_mask={mask}"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "strata_matrix_native_expert_deep_tests.rs"]
mod tests;
