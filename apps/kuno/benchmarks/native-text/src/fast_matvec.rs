//! Explicit count1, four-row packed projection. Original64-lane accumulation
//! and tree, with two logical lanes per physical lane and subgroup shuffles.
//! Requires enabled subgroup support,16..32 subgroup bounds and128 cube lanes.
//! Original MAX8 owners, spans, finite faults and erasure remain with the caller.
use super::{Q4_K, binary16, byte_at, packed_value, signed_byte, u16_at};
use cubecl::prelude::*;

#[path = "mtp_pruned_head.rs"]
pub(crate) mod pruned_head;

pub const ROWS_PER_CUBE: usize = 4;
pub const PHYSICAL_LANES: usize = 32;

#[cube]
fn byte(word: u32, index: usize) -> u32 {
    (word >> ((index * 8) as u32)) & 255
}
#[cube]
fn scale(a: u32, c: u32, group: usize) -> u32 {
    let index = group & 3;
    if group < 4 {
        byte(a, index) & 63
    } else {
        (byte(c, index) & 15) | ((byte(a, index) >> 6) << 4)
    }
}
#[cube]
fn minimum(b: u32, c: u32, group: usize) -> u32 {
    let index = group & 3;
    if group < 4 {
        byte(b, index) & 63
    } else {
        (byte(c, index) >> 4) | ((byte(b, index) >> 6) << 4)
    }
}

/// Logical lane numbering comes from subgroup ID/size, not an assumed mapping
/// of subgroup lanes to local_invocation_id. All128 physical invocations own
/// exactly one of four32-lane row partitions on supported subgroup sizes.
#[cube]
pub fn local_row() -> usize {
    PLANE_POS as usize / (PHYSICAL_LANES / PLANE_DIM as usize)
}
#[cube]
pub fn lane() -> usize {
    (PLANE_POS as usize % (PHYSICAL_LANES / PLANE_DIM as usize)) * PLANE_DIM as usize
        + UNIT_POS_PLANE as usize
}

#[cube]
#[allow(clippy::collapsible_if)]
pub fn reduce_original64(low: f32, high: f32) -> f32 {
    let logical_lane = lane();
    let row = local_row();
    // Exactly the original stride32 add, followed by stride16,8,4,2,1.
    let mut value = low + high;
    let mut shared = SharedMemory::<f32>::new(ROWS_PER_CUBE * PHYSICAL_LANES);
    if PLANE_DIM == 16 {
        // Only16-wide subgroups need the other half of the original32-lane
        // partition.32-wide subgroups reduce it directly with shuffles, so
        // shared stores and the workgroup barrier contribute no arithmetic.
        shared[row * PHYSICAL_LANES + logical_lane] = value;
        sync_cube();
        if logical_lane < 16 {
            value += shared[row * PHYSICAL_LANES + logical_lane + 16];
        }
    }
    // Subgroup collectives are outside row/active predication. No wave-size
    // assumption is hidden in the shuffle distance or logical reduction order.
    let mut stride = PLANE_DIM as usize / 2;
    while stride > 0 {
        let other = plane_shuffle_down(value, stride as u32);
        if (UNIT_POS_PLANE as usize) < stride {
            value += other;
        }
        stride /= 2;
    }
    value
}

#[cube]
pub fn dot(
    packed: &Array<u32>,
    input: &Array<f32>,
    row_offset: usize,
    input_first: usize,
    active: bool,
    #[comptime] columns: usize,
    #[comptime] dtype: u32,
) -> f32 {
    let logical_lane = lane();
    let mut low = 0.0_f32;
    let mut high = 0.0_f32;
    if active {
        if comptime!(dtype == Q4_K) {
            // Caller proves original Q4 rows and packed bases are word aligned.
            // One byte supplies the two halves of the original64-lane tile.
            for block in 0..comptime!(columns / 256) {
                let base = row_offset / 4 + block * 36;
                let header = packed[base];
                let d = binary16(header & 65535);
                let dmin = binary16(header >> 16);
                let a = packed[base + 1];
                let b = packed[base + 2];
                let c = packed[base + 3];
                #[unroll]
                for pair in 0..4_usize {
                    let q = byte(
                        packed[base + 4 + pair * 8 + logical_lane / 4],
                        logical_lane & 3,
                    );
                    let lower = (d * scale(a, c, pair * 2) as f32) * (q & 15) as f32
                        - dmin * minimum(b, c, pair * 2) as f32;
                    let upper = (d * scale(a, c, pair * 2 + 1) as f32) * (q >> 4) as f32
                        - dmin * minimum(b, c, pair * 2 + 1) as f32;
                    let col = block * 256 + pair * 64 + logical_lane;
                    low += lower * input[input_first + col];
                    high += upper * input[input_first + col + 32];
                }
            }
        } else {
            for tile in 0..comptime!(columns.div_ceil(64)) {
                let col = tile * 64 + logical_lane;
                if col < columns {
                    low += packed_value(packed, row_offset, col, dtype) * input[input_first + col];
                }
                if col + 32 < columns {
                    high += packed_value(packed, row_offset, col + 32, dtype)
                        * input[input_first + col + 32];
                }
            }
        }
    }
    reduce_original64(low, high)
}

/// Head-only experiment: one public Q8 half conversion per subgroup and
/// block. Each physical lane still owns the same two logical accumulators.
/// Broadcasts occur in converged control flow even for inactive tail rows.
/// The byte helpers preserve tightly packed, only two-byte-aligned34B blocks.
#[cube]
#[allow(clippy::collapsible_if)]
pub fn dot_q8_head_broadcast(
    packed: &Array<u32>,
    input: &Array<f32>,
    row_offset: usize,
    input_first: usize,
    active: bool,
    #[comptime] columns: usize,
) -> f32 {
    let logical_lane = lane();
    let mut low = 0.0_f32;
    let mut high = 0.0_f32;
    for tile in 0..comptime!(columns.div_ceil(64)) {
        let base = row_offset + tile * 68;
        let mut lower_scale = 0.0_f32;
        let mut upper_scale = 0.0_f32;
        if UNIT_POS_PLANE == 0 {
            if active {
                if tile * 64 < columns {
                    lower_scale = binary16(u16_at(packed, base));
                }
                if tile * 64 + 32 < columns {
                    upper_scale = binary16(u16_at(packed, base + 34));
                }
            }
        }
        let lower_scale = plane_broadcast(lower_scale, 0_u32);
        let upper_scale = plane_broadcast(upper_scale, 0_u32);
        if active {
            let col = tile * 64 + logical_lane;
            if col < columns {
                let quant = signed_byte(byte_at(packed, base + 2 + logical_lane));
                low += (lower_scale * quant as f32) * input[input_first + col];
            }
            if col + 32 < columns {
                let quant = signed_byte(byte_at(packed, base + 36 + logical_lane));
                high += (upper_scale * quant as f32) * input[input_first + col + 32];
            }
        }
    }
    reduce_original64(low, high)
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn packed_single(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    output_first: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] dtype: u32,
    #[comptime] row_bytes: usize,
) {
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + local_row();
    let active = row < rows && count == 1;
    let value = dot(
        packed,
        input,
        packed_first + row * row_bytes,
        input_first,
        active,
        columns,
        dtype,
    );
    if lane() == 0 && active {
        output[output_first + row] = value;
    }
}

pub fn grid(rows: usize) -> CubeCount {
    CubeCount::Static(rows.div_ceil(ROWS_PER_CUBE) as u32, 1, 1)
}
pub fn cube() -> CubeDim {
    CubeDim::new_1d((ROWS_PER_CUBE * PHYSICAL_LANES) as u32)
}

#[cube]
#[allow(clippy::collapsible_if)]
pub fn active_output(index: usize, first: usize, #[comptime] rows: usize) -> bool {
    let mut active = false;
    if index >= first {
        if index - first < rows {
            active = true;
        }
    }
    active
}

/// Every produced value folds its own finite decision. The last public row
/// tile ALSO inspects every other logical output value once, preserving the
/// old full-MAX8 finite check without racing other active tile writes. Only
/// the entire active matrix interval is excluded, never a caller-sliced tail.
#[cube(launch, launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord
)] // Cube comparison must reject NaN without a host-only partial_cmp.
pub fn packed_single_fault(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    output_first: usize,
    matrix_first: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] dtype: u32,
    #[comptime] row_bytes: usize,
    #[comptime] matrix_rows: usize,
    #[comptime] check_tail: bool,
) {
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + local_row();
    let active = row < rows && count == 1;
    let value = dot(
        packed,
        input,
        packed_first + row * row_bytes,
        input_first,
        active,
        columns,
        dtype,
    );
    if lane() == 0 && active {
        output[output_first + row] = value;
        if !(value.abs() <= f32::MAX) {
            fault[0].fetch_or(1);
        }
    }
    if comptime!(check_tail) {
        let index = ABSOLUTE_POS;
        if index < output.len() {
            if !active_output(index, matrix_first, matrix_rows) {
                if !(output[index].abs() <= f32::MAX) {
                    fault[0].fetch_or(1);
                }
            }
        }
    }
    if lane() == 0 && count != 1 {
        fault[0].fetch_or(1);
    }
}

/// Explicit original-Q8 head entrypoint. Runtime spans, complete output/tail
/// finite checks and the original four-row reduction remain unchanged.
#[cube(launch, launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord
)]
pub fn packed_q8_head_broadcast_fault(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    output_first: usize,
    matrix_first: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] row_bytes: usize,
    #[comptime] matrix_rows: usize,
    #[comptime] check_tail: bool,
) {
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + local_row();
    let active = row < rows && count == 1;
    let value = dot_q8_head_broadcast(
        packed,
        input,
        packed_first + row * row_bytes,
        input_first,
        active,
        columns,
    );
    if lane() == 0 && active {
        output[output_first + row] = value;
        if !(value.abs() <= f32::MAX) {
            fault[0].fetch_or(1);
        }
    }
    if comptime!(check_tail) {
        let index = ABSOLUTE_POS;
        if index < output.len() {
            if !active_output(index, matrix_first, matrix_rows) {
                if !(output[index].abs() <= f32::MAX) {
                    fault[0].fetch_or(1);
                }
            }
        }
    }
    if lane() == 0 && count != 1 {
        fault[0].fetch_or(1);
    }
}

/// Separate head experiment: cooperatively decode the original public Q8
/// scales once per four-row workgroup. Every bounded slot is initialized even
/// for inactive tail cubes; packed reads use the actual source row/count guard.
/// The uniform fence precedes all scale reads and the unchanged original tree.
#[cube(launch, launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord
)]
pub fn packed_q8_head_shared_scales_fault(
    packed: &Array<u32>,
    input: &Array<f32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    packed_first: usize,
    input_first: usize,
    output_first: usize,
    matrix_first: usize,
    #[comptime] rows: usize,
    #[comptime] columns: usize,
    #[comptime] row_bytes: usize,
    #[comptime] matrix_rows: usize,
    #[comptime] check_tail: bool,
) {
    let blocks = comptime!(columns / 32);
    let mut scales = SharedMemory::<f32>::new(comptime!(ROWS_PER_CUBE * columns / 32));
    #[unroll]
    for slice in
        0..comptime!((ROWS_PER_CUBE * columns / 32).div_ceil(ROWS_PER_CUBE * PHYSICAL_LANES))
    {
        let index = UNIT_POS_X as usize + slice * ROWS_PER_CUBE * PHYSICAL_LANES;
        if index < ROWS_PER_CUBE * blocks {
            let source_row = CUBE_POS_X as usize * ROWS_PER_CUBE + index / blocks;
            let mut scale = 0.0_f32;
            if source_row < rows && count == 1 {
                let base = packed_first + source_row * row_bytes + (index % blocks) * 34;
                scale = binary16(u16_at(packed, base));
            }
            scales[index] = scale;
        }
    }
    sync_cube();
    let slot = local_row();
    let row = CUBE_POS_X as usize * ROWS_PER_CUBE + slot;
    let active = row < rows && count == 1;
    let logical_lane = lane();
    let mut low = 0.0_f32;
    let mut high = 0.0_f32;
    if active {
        for tile in 0..comptime!(columns.div_ceil(64)) {
            let base = packed_first + row * row_bytes + tile * 68;
            let col = tile * 64 + logical_lane;
            if col < columns {
                let quant = signed_byte(byte_at(packed, base + 2 + logical_lane));
                low += (scales[slot * blocks + tile * 2] * quant as f32) * input[input_first + col];
            }
            if col + 32 < columns {
                let quant = signed_byte(byte_at(packed, base + 36 + logical_lane));
                high += (scales[slot * blocks + tile * 2 + 1] * quant as f32)
                    * input[input_first + col + 32];
            }
        }
    }
    let value = reduce_original64(low, high);
    if lane() == 0 && active {
        output[output_first + row] = value;
        if !(value.abs() <= f32::MAX) {
            fault[0].fetch_or(1);
        }
    }
    if comptime!(check_tail) {
        let index = ABSOLUTE_POS;
        if index < output.len() {
            if !active_output(index, matrix_first, matrix_rows) {
                if !(output[index].abs() <= f32::MAX) {
                    fault[0].fetch_or(1);
                }
            }
        }
    }
    if lane() == 0 && count != 1 {
        fault[0].fetch_or(1);
    }
}

pub fn fault_grid(rows: usize, output_words: usize, check_tail: bool) -> CubeCount {
    let cubes = rows.div_ceil(ROWS_PER_CUBE);
    let cubes = if check_tail {
        cubes.max(output_words.div_ceil(ROWS_PER_CUBE * PHYSICAL_LANES))
    } else {
        cubes
    };
    CubeCount::Static(cubes as u32, 1, 1)
}

#[cfg(test)]
#[path = "q8_head_broadcast_tests.rs"]
mod q8_head_broadcast_tests;

#[cfg(test)]
#[path = "q8_head_shared_scale_tests.rs"]
mod q8_head_shared_scale_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_matvec::{
        BF16, F16, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, Q5_0, Q5_K, Q6_K, Q8_0,
    };

    pub(super) fn assert_shared_reduction_is_only_for16(module: &wgpu::naga::Module) {
        use wgpu::naga::{AddressSpace, BinaryOperator, Expression, Literal, Statement};
        fn subgroup_size(
            module: &wgpu::naga::Module,
            f: &wgpu::naga::Function,
            expression: wgpu::naga::Handle<Expression>,
        ) -> bool {
            match f.expressions[expression] {
                Expression::Load { pointer } | Expression::As { expr: pointer, .. } => {
                    subgroup_size(module, f, pointer)
                }
                Expression::FunctionArgument(index) => {
                    f.arguments[index as usize].binding
                        == Some(wgpu::naga::Binding::BuiltIn(
                            wgpu::naga::BuiltIn::SubgroupSize,
                        ))
                }
                Expression::GlobalVariable(index) => {
                    module.global_variables[index].name.as_deref() == Some("subgroup_size")
                }
                _ => false,
            }
        }
        fn sixteen(f: &wgpu::naga::Function, expression: wgpu::naga::Handle<Expression>) -> bool {
            match f.expressions[expression] {
                Expression::Literal(Literal::U32(16) | Literal::I32(16)) => true,
                Expression::As { expr, .. } => sixteen(f, expr),
                _ => false,
            }
        }
        fn is16(
            module: &wgpu::naga::Module,
            f: &wgpu::naga::Function,
            expression: wgpu::naga::Handle<Expression>,
        ) -> bool {
            if let Expression::Binary {
                op: BinaryOperator::Equal,
                left,
                right,
            } = f.expressions[expression]
            {
                (subgroup_size(module, f, left) && sixteen(f, right))
                    || (subgroup_size(module, f, right) && sixteen(f, left))
            } else {
                false
            }
        }
        fn workgroup_pointer(
            module: &wgpu::naga::Module,
            f: &wgpu::naga::Function,
            expression: wgpu::naga::Handle<Expression>,
        ) -> bool {
            match f.expressions[expression] {
                Expression::Access { base, .. } | Expression::AccessIndex { base, .. } => {
                    workgroup_pointer(module, f, base)
                }
                Expression::GlobalVariable(index) => {
                    module.global_variables[index].space == AddressSpace::WorkGroup
                }
                _ => false,
            }
        }
        fn visit(
            module: &wgpu::naga::Module,
            f: &wgpu::naga::Function,
            block: &wgpu::naga::Block,
            inside16: bool,
            counts: &mut [usize; 3],
        ) {
            for statement in block {
                match statement {
                    Statement::Store { pointer, .. } if workgroup_pointer(module, f, *pointer) => {
                        assert!(
                            inside16,
                            "32-wide path must not store shared reduction data"
                        );
                        counts[0] += 1;
                    }
                    Statement::ControlBarrier(_) => {
                        assert!(
                            inside16,
                            "32-wide path must not execute a workgroup barrier"
                        );
                        counts[1] += 1;
                    }
                    Statement::SubgroupGather {
                        mode: wgpu::naga::GatherMode::ShuffleDown(_),
                        ..
                    } => {
                        assert!(
                            !inside16,
                            "original shuffle tree remains shared by both widths"
                        );
                        counts[2] += 1;
                    }
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    } => {
                        visit(
                            module,
                            f,
                            accept,
                            inside16 || is16(module, f, *condition),
                            counts,
                        );
                        visit(module, f, reject, inside16, counts);
                    }
                    Statement::Loop {
                        body, continuing, ..
                    } => {
                        visit(module, f, body, inside16, counts);
                        visit(module, f, continuing, inside16, counts);
                    }
                    Statement::Block(block) => visit(module, f, block, inside16, counts),
                    Statement::Switch { cases, .. } => {
                        for case in cases {
                            visit(module, f, &case.body, inside16, counts);
                        }
                    }
                    _ => {}
                }
            }
        }
        let entry = &module.entry_points[0].function;
        let mut counts = [0; 3];
        visit(module, entry, &entry.body, false, &mut counts);
        assert_eq!(counts, [1, 1, 1]);
    }
    #[test]
    fn all_original_formats_lower_to_valid_subgroup_wgsl_with_runtime_spans() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [
            cubecl::server::ExecutionMode::Checked,
            cubecl::server::ExecutionMode::Unchecked,
        ] {
            for dtype in [
                0, F16, BF16, Q4_K, Q5_0, Q8_0, Q6_K, IQ3_XXS, IQ4_NL, IQ4_XS, IQ2_S, IQ3_S, Q5_K,
                Q2_0,
            ] {
                let uint = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                let float = StorageType::Scalar(ElemType::Float(FloatKind::F32));
                let mut b = KernelBuilder::default();
                b.scope.register_type::<u32>(uint);
                b.scope.register_type::<usize>(uint);
                b.scope.register_type::<f32>(float);
                b.scope
                    .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
                b.scope
                    .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                let packed = b.input_array(Type::new(uint)).into();
                let input = b.input_array(Type::new(float)).into();
                let output = b.output_array(Type::new(float)).into();
                let count = b.scalar(uint).into();
                let first = b.scalar(uint).into();
                let x = b.scalar(uint).into();
                let y = b.scalar(uint).into();
                packed_single::expand(
                    &mut b.scope,
                    packed,
                    input,
                    output,
                    count,
                    first,
                    x,
                    y,
                    7,
                    if dtype == F16 { 4 } else { 256 },
                    dtype,
                    kuno_text::TensorType::from_gguf_id(dtype)
                        .unwrap()
                        .row_bytes(if dtype == F16 { 4 } else { 256 })
                        .unwrap() as usize,
                );
                let source = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(cube())
                                .kernel_name("fixed_public_subgroup"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        uint,
                    )
                    .unwrap()
                    .to_string();
                let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                if dtype == F16 {
                    assert!(source.contains("* u32(8)"));
                }
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::SUBGROUP,
                )
                .validate(&module)
                .unwrap();
                assert_shared_reduction_is_only_for16(&module);
                assert!(
                    source.contains("subgroupShuffleDown") && source.contains("workgroupBarrier")
                );
                for check_tail in [false, true] {
                    let mut b = KernelBuilder::default();
                    b.scope.register_type::<u32>(uint);
                    b.scope.register_type::<usize>(uint);
                    b.scope.register_type::<f32>(float);
                    b.scope
                        .register_type::<i32>(StorageType::Scalar(ElemType::Int(IntKind::I32)));
                    b.scope
                        .register_type::<bool>(StorageType::Scalar(ElemType::Bool));
                    let packed = b.input_array(Type::new(uint)).into();
                    let input = b.input_array(Type::new(float)).into();
                    let output = b.output_array(Type::new(float)).into();
                    let fault = b
                        .output_array(Type::new(StorageType::Atomic(ElemType::UInt(
                            UIntKind::U32,
                        ))))
                        .into();
                    let count = b.scalar(uint).into();
                    let first = b.scalar(uint).into();
                    let x = b.scalar(uint).into();
                    let y = b.scalar(uint).into();
                    let mf = b.scalar(uint).into();
                    packed_single_fault::expand(
                        &mut b.scope,
                        packed,
                        input,
                        output,
                        fault,
                        count,
                        first,
                        x,
                        y,
                        mf,
                        7,
                        256,
                        dtype,
                        kuno_text::TensorType::from_gguf_id(dtype)
                            .unwrap()
                            .row_bytes(256)
                            .unwrap() as usize,
                        11,
                        check_tail,
                    );
                    let source = WgslCompiler::default()
                        .compile(
                            b.build(
                                KernelSettings::default()
                                    .cube_dim(cube())
                                    .kernel_name("fixed_public_fused_subgroup"),
                            ),
                            &WgpuCompilationOptions::default(),
                            mode,
                            uint,
                        )
                        .unwrap()
                        .to_string();
                    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::SUBGROUP,
                    )
                    .validate(&module)
                    .unwrap();
                    assert_shared_reduction_is_only_for16(&module);
                    assert!(source.contains("atomicOr") && source.contains("subgroupShuffleDown"));
                }
            }
        }
    }
    #[test]
    fn fused_tail_coverage_excludes_entire_active_matrix_and_never_future_tile_writes() {
        for words in [17usize, 1024, 8 * 152064] {
            for first in [0usize, 3, 16] {
                let rows = 7;
                if first + rows > words {
                    continue;
                }
                let CubeCount::Static(x, 1, 1) = fault_grid(rows, words, true) else {
                    panic!()
                };
                assert!((x as usize) * 128 >= words);
                for index in 0..words {
                    let protected = index >= first && index - first < rows;
                    let expected = (first..first + rows).contains(&index);
                    assert_eq!(protected, expected);
                }
                let CubeCount::Static(x, 1, 1) = fault_grid(rows, words, false) else {
                    panic!()
                };
                assert_eq!(x as usize, rows.div_ceil(4));
            }
        }
    }
    #[test]
    fn subgroup_partition_covers_each_logical_lane_once_and_preserves_original_tree() {
        for width in [16, 32] {
            let mut seen = [[false; 32]; 4];
            for plane in 0..128 / width {
                for position in 0..width {
                    let row = plane / (32 / width);
                    let lane = (plane % (32 / width)) * width + position;
                    assert!(!seen[row][lane]);
                    seen[row][lane] = true;
                }
            }
            assert!(seen.iter().flatten().all(|v| *v));
            for seed in 0..101 {
                let mut original = [0f32; 64];
                for (i, value) in original.iter_mut().enumerate() {
                    // Fixed finite mixed scales/cancellation, not a monotone
                    // sum whose rounding happens to hide a changed tree.
                    let magnitude = ((i * 13 + seed * 31) % 57) as f32 / 23. - 1.;
                    *value = magnitude * 2.0_f32.powi(((i * 7 + seed) % 53) as i32 - 26);
                }
                let mut candidate = [0f32; 32];
                for i in 0..32 {
                    candidate[i] = original[i] + original[i + 32];
                }
                if width == 16 {
                    // Snapshot represents the complete shared stores +fence;
                    // only the lower16 subgroup adds its sibling half.
                    let shared = candidate;
                    for i in 0..16 {
                        candidate[i] += shared[i + 16];
                    }
                }
                let mut stride = width / 2;
                while stride > 0 {
                    // Native subgroup shuffle samples all lanes before the
                    // original predicate adds: model both16-wide subgroups,
                    // including the discarded high subgroup, independently.
                    let previous = candidate;
                    for subgroup in 0..32 / width {
                        for i in 0..stride {
                            let lane = subgroup * width + i;
                            candidate[lane] += previous[lane + stride];
                        }
                    }
                    stride /= 2;
                }
                for stride in [32, 16, 8, 4, 2, 1] {
                    for i in 0..stride {
                        original[i] += original[i + stride];
                    }
                }
                assert_eq!(candidate[0].to_bits(), original[0].to_bits());
            }
        }
    }
}
