//! Explicit count1, four-row packed projection. Original64-lane accumulation
//! and tree, with two logical lanes per physical lane and subgroup shuffles.
//! Requires enabled subgroup support,16..32 subgroup bounds and128 cube lanes.
//! Original MAX8 owners, spans, finite faults and erasure remain with the caller.
use super::{Q4_K, binary16, packed_value};
use cubecl::prelude::*;

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
    shared[row * PHYSICAL_LANES + logical_lane] = value;
    sync_cube();
    if PLANE_DIM == 16 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_matvec::{BF16, F16, Q5_0, Q6_K, Q8_0};
    #[test]
    fn all_original_formats_lower_to_valid_subgroup_wgsl_with_runtime_spans() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, IntKind, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for dtype in [0, F16, BF16, Q4_K, Q5_0, Q8_0, Q6_K] {
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
                    cubecl::server::ExecutionMode::Unchecked,
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
            assert!(source.contains("subgroupShuffleDown") && source.contains("workgroupBarrier"));
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
            for seed in 0..37 {
                let mut original = [0f32; 64];
                for (i, value) in original.iter_mut().enumerate() {
                    *value = ((i * 13 + seed * 31) % 57) as f32 / 23. - 1.;
                }
                let mut candidate = [0f32; 32];
                for i in 0..32 {
                    candidate[i] = original[i] + original[i + 32];
                }
                for stride in [16, 8, 4, 2, 1] {
                    for i in 0..stride {
                        candidate[i] += candidate[i + stride];
                    }
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
