//! Opt-in CubeK0.2 dense projections, with original physical data and guards.
//! CubeK is MIT OR Apache-2.0; the public batch/global/stage/tile implementation
//! is reused unchanged. Only this adapter's logical topology is normalized.
use cubecl::{
    ir::{
        AddressType, Branch, Builtin, CoopMma, ElemType, FloatKind, Operation, OperationReflect,
        Scope, Synchronization, Variable, VariableKind,
    },
    prelude::*,
    server::Handle,
    wgpu::WgpuRuntime,
};
use cubek_matmul::{
    components::{
        batch::{BatchMatmul, BatchMatmulFamily},
        stage::PartitionBuffering,
        tile::TileMatmulKind,
    },
    definition::{
        Blueprint, CubeMapping, MatmulElems, MatmulGlobalElems, MatmulProblem, MatmulVectorSizes,
        TilingBlueprint, TilingScheme, cube_mapping_launch,
    },
    launch::{
        ConcreteInputsFactory, ConcreteOutputFactory, MatmulArgs, TensorArgs, TensorInputs,
        TensorOutput,
    },
    routines::{BlueprintStrategy, DeviceSettings, Routine},
};
use cubek_std::{InputBinding, PartitionSize, StageSize, TileSize};
use half::{bf16, f16};
use kuno_text::{Error, Result};
use std::marker::PhantomData;

#[cfg(not(feature = "strata-cubek-tilewise"))]
type Algorithm = cubek_matmul::routines::double_buffering::CyclicDoubleBufferingAlgorithm;
#[cfg(feature = "strata-cubek-tilewise")]
type Algorithm = cubek_matmul::routines::double_buffering::TilewiseDoubleBufferingAlgorithm;
type Family = <Algorithm as Routine<()>>::BatchMatmul;
type LibraryConfig = <Family as BatchMatmulFamily<()>>::Config;
type Precision<F> = (
    (F, Const<4>, F, Const<4>, F, Const<4>),
    (F, Const<4>, F, Const<4>, F, Const<4>),
    (f32, Const<4>, f32, Const<4>, f32, Const<4>),
);
type Product<F> = <Family as BatchMatmulFamily<()>>::Matmul<Precision<F>>;
type Args = TensorArgs<()>;
type Inputs<F> = TensorInputs<Vector<F, Const<4>>, Vector<F, Const<4>>, Vector<f32, Const<4>>>;
type Output = TensorOutput<Vector<f32, Const<4>>>;
const SHARED_CAP: usize = 32768;
#[cfg(any(feature = "strata-cubek-m64", test))]
const M64_SHARED_CAP: usize = 34816;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatrixPlan {
    m_tiles: u32,
    required_shared: usize,
}
fn matrix_plan(count: usize, available_shared: usize) -> MatrixPlan {
    #[cfg(all(feature = "strata-cubek-m64", feature = "strata-cubek-tilewise"))]
    if count >= 256 && available_shared >= M64_SHARED_CAP {
        return MatrixPlan {
            m_tiles: 8,
            required_shared: M64_SHARED_CAP,
        };
    }
    let _ = (count, available_shared);
    MatrixPlan {
        m_tiles: 4,
        required_shared: SHARED_CAP,
    }
}
fn matrix_partition_buffering() -> PartitionBuffering {
    if cfg!(all(
        feature = "strata-cubek-balanced",
        feature = "strata-cubek-tilewise"
    )) {
        PartitionBuffering::Double
    } else {
        PartitionBuffering::Single
    }
}
fn matrix_scheme(plan: MatrixPlan) -> Result<TilingScheme> {
    let balanced = cfg!(all(
        feature = "strata-cubek-balanced",
        feature = "strata-cubek-tilewise"
    ));
    let (partition_m, partition_n, stage_m, stage_n) = if balanced {
        (plan.m_tiles / 2, 2, 2, 2)
    } else {
        (plan.m_tiles, 1, 1, 4)
    };
    TilingScheme::builder()
        .with_tile_size(TileSize::new(8, 16, 16))
        .with_partition_size(PartitionSize::new(partition_m, partition_n, 4))
        .with_stage_size(StageSize::new(stage_m, stage_n, 1))
        .build()
        .map_err(|_| Error::InvalidFormat("CubeK fixed matrix tiling"))
}

const WIDE_SHARED_CAP: usize = 36864;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DensePlan {
    Original(MatrixPlan),
    Wide,
}
impl DensePlan {
    fn required_shared(self) -> usize {
        match self {
            Self::Original(plan) => plan.required_shared,
            Self::Wide => WIDE_SHARED_CAP,
        }
    }
    fn cube_dim(self) -> CubeDim {
        CubeDim::new_2d(16, if self == Self::Wide { 8 } else { 4 })
    }
    fn scheme(self) -> Result<TilingScheme> {
        match self {
            Self::Original(plan) => matrix_scheme(plan),
            Self::Wide => TilingScheme::builder()
                .with_tile_size(TileSize::new(8, 16, 16))
                .with_partition_size(PartitionSize::new(4, 4, 2))
                .with_stage_size(StageSize::new(4, 2, 1))
                .build()
                .map_err(|_| Error::InvalidFormat("CubeK fixed wide dense tiling")),
        }
    }
    fn buffering(self) -> PartitionBuffering {
        match self {
            Self::Original(_) => matrix_partition_buffering(),
            Self::Wide => PartitionBuffering::Double,
        }
    }
}
fn dense_plan(
    count: usize,
    available_shared: usize,
    max_units: u32,
    max_dim: (u32, u32, u32),
) -> DensePlan {
    if cfg!(all(
        feature = "strata-cubek-wide",
        feature = "strata-cubek-tilewise"
    )) && count >= 512
        && available_shared >= WIDE_SHARED_CAP
        && max_units >= 128
        && max_dim.0 >= 16
        && max_dim.1 >= 8
        && max_dim.2 >= 1
    {
        DensePlan::Wide
    } else {
        DensePlan::Original(matrix_plan(count, available_shared))
    }
}

#[cfg(feature = "strata-cubek-wide")]
pub(super) fn public_dense_uses_wide(client: &ComputeClient<WgpuRuntime>, count: usize) -> bool {
    let hw = &client.properties().hardware;
    dense_plan(
        count,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
    ) == DensePlan::Wide
}

#[path = "strata_cubek_expert.rs"]
mod routed_expert;
#[allow(unused_imports)]
pub(super) use routed_expert::try_expert;
#[path = "strata_cubek_expert_probe.rs"]
mod routed_expert_probe;
#[allow(unused_imports)]
pub(crate) use routed_expert_probe::cases as public_expert_numeric;

/// The original library's entry calls these same public families. Fixed types
/// avoid introducing flex32/TF32, type inference, autotuning, or split-K.
#[cube(launch_unchecked)]
fn library_dense<F: Float>(
    inputs: &Inputs<F>,
    output: &mut Output,
    cube_mapping: CubeMapping,
    #[comptime] blueprint: TilingBlueprint,
    #[comptime] config: LibraryConfig,
) {
    let mut state =
        Args::init_state::<Vector<F, Const<4>>, Vector<F, Const<4>>, Vector<f32, Const<4>>>(
            inputs,
            output,
            (),
            blueprint.lhs_global_layout_config(),
            blueprint.rhs_global_layout_config(),
            blueprint.out_global_layout_config(),
        );
    Product::<F>::execute::<Args>(&mut state, cube_mapping, config);
    normalize_logical_topology();
}

// This is a compile-time CubeCL expansion hook, not a shader operation or a
// runtime text/SPIR-V patch. It runs after the library has expanded into this
// entry's IR, before normal optimization, validation and native IEEE controls.
fn normalize_logical_topology() {
    cubecl::unexpanded!()
}
mod normalize_logical_topology {
    pub fn expand(scope: &mut cubecl::ir::Scope) {
        if let Err(reason) = super::normalize_scope(scope) {
            scope.push_error(reason);
        }
    }
}

fn normalize_variable(value: &mut Variable) -> std::result::Result<(), &'static str> {
    if let VariableKind::Builtin(builtin) = &mut value.kind {
        *builtin = match *builtin {
            Builtin::UnitPosX => Builtin::UnitPosPlane,
            Builtin::UnitPosY => Builtin::PlanePos,
            Builtin::UnitPos
            | Builtin::UnitPosZ
            | Builtin::AbsolutePos
            | Builtin::AbsolutePosX
            | Builtin::AbsolutePosY
            | Builtin::AbsolutePosZ => {
                return Err("CubeK unsupported invocation topology");
            }
            other => other,
        };
    }
    Ok(())
}

fn normalize_scope(scope: &mut Scope) -> std::result::Result<(), &'static str> {
    for value in &mut scope.locals {
        normalize_variable(value)?;
    }
    for (variable, values) in &mut scope.const_arrays {
        normalize_variable(variable)?;
        for value in values {
            normalize_variable(value)?;
        }
    }
    for instruction in &mut scope.instructions {
        if !instruction.modes.fp_math_mode.is_empty() {
            return Err("CubeK relaxed arithmetic mode rejected");
        }
        if let Some(out) = &mut instruction.out {
            normalize_variable(out)?;
        }
        normalize_operation(&mut instruction.operation)?;
    }
    Ok(())
}

fn normalize_operation(op: &mut Operation) -> std::result::Result<(), &'static str> {
    if let Some(mut args) = op.args() {
        for arg in &mut args {
            normalize_variable(arg)?;
        }
        *op = Operation::from_code_and_args(op.op_code(), &args)
            .ok_or("CubeK operation reflection rejected")?;
        return Ok(());
    }
    match op {
        Operation::Branch(branch) => match branch {
            Branch::If(v) => {
                normalize_variable(&mut v.cond)?;
                normalize_scope(&mut v.scope)?;
            }
            Branch::IfElse(v) => {
                normalize_variable(&mut v.cond)?;
                normalize_scope(&mut v.scope_if)?;
                normalize_scope(&mut v.scope_else)?;
            }
            Branch::RangeLoop(v) => {
                for variable in [&mut v.i, &mut v.start, &mut v.end] {
                    normalize_variable(variable)?;
                }
                if let Some(step) = &mut v.step {
                    normalize_variable(step)?;
                }
                normalize_scope(&mut v.scope)?;
            }
            Branch::Loop(v) => normalize_scope(&mut v.scope)?,
            Branch::Switch(v) => {
                normalize_variable(&mut v.value)?;
                normalize_scope(&mut v.scope_default)?;
                for (value, scope) in &mut v.cases {
                    normalize_variable(value)?;
                    normalize_scope(scope)?;
                }
            }
            Branch::Return | Branch::Break | Branch::Unreachable => {}
        },
        Operation::CoopMma(CoopMma::Load {
            value,
            stride,
            offset,
            ..
        }) => {
            for variable in [value, stride, offset] {
                normalize_variable(variable)?;
            }
        }
        Operation::CoopMma(CoopMma::Store {
            mat,
            stride,
            offset,
            ..
        }) => {
            for variable in [mat, stride, offset] {
                normalize_variable(variable)?;
            }
        }
        Operation::CoopMma(CoopMma::Execute {
            mat_a,
            mat_b,
            mat_c,
        }) => {
            for variable in [mat_a, mat_b, mat_c] {
                normalize_variable(variable)?;
            }
        }
        Operation::Synchronization(
            Synchronization::SyncCube | Synchronization::SyncPlane | Synchronization::SyncStorage,
        ) => {}
        Operation::NonSemantic(
            cubecl::ir::NonSemantic::Comment { .. }
            | cubecl::ir::NonSemantic::EnterDebugScope
            | cubecl::ir::NonSemantic::ExitDebugScope,
        ) => {}
        // The selected synchronous CMMA family needs no CUDA/TMA barriers,
        // non-semantic printf or unreflected operator. Reject rather than guess.
        _ => return Err("CubeK unreviewed operation rejected"),
    }
    Ok(())
}

#[cube(launch_unchecked)]
fn finite_output(
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    rows: usize,
    first: usize,
    stride: usize,
) {
    let index = usize::cast_from(ABSOLUTE_POS);
    if index < count * rows {
        let at = first + index / rows * stride + index % rows;
        let value = output[at];
        let valid = value.abs() <= f32::MAX;
        if !valid {
            output[at] = 0.0;
            fault[0].fetch_or(1);
        }
    }
}

fn eligible(
    count: usize,
    columns: usize,
    rows: usize,
    first: usize,
    out_first: usize,
    stride: usize,
) -> bool {
    // Keep the complete existing small/unaligned path. The library's double
    // buffering requires at least two K64 stages; actual model K is much larger.
    (32..=4096).contains(&count)
        && rows >= 64
        && rows.is_multiple_of(16)
        && columns >= 128
        && columns.is_multiple_of(128)
        && first.is_multiple_of(4)
        && out_first.is_multiple_of(4)
        && stride >= rows
        && stride.is_multiple_of(4)
}

fn checked_view(handle: Handle, first: usize, elements: usize, bytes: usize) -> Result<Handle> {
    let offset = first.checked_mul(bytes).ok_or(Error::ArithmeticOverflow)?;
    let end = first
        .checked_add(elements)
        .and_then(|n| n.checked_mul(bytes))
        .ok_or(Error::ArithmeticOverflow)?;
    let used = handle
        .size()
        .checked_sub(handle.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(handle.offset_end.unwrap_or(0)))
        .ok_or(Error::InvalidBufferLength)?;
    if end as u64 > used
        || !handle
            .offset_start
            .unwrap_or(0)
            .is_multiple_of((4 * bytes) as u64)
        || !(offset as u64).is_multiple_of((4 * bytes) as u64)
    {
        return Err(Error::InvalidBufferLength);
    }
    // Real narrowed view of the original handle; the caller retains its full
    // allocation owner/lease and the original native reset/erasure scope.
    Ok(handle.offset_start(offset as u64))
}

fn tensor(handle: Handle, shape: [usize; 2], strides: [usize; 2]) -> TensorBinding<WgpuRuntime> {
    TensorBinding {
        handle: handle.binding(),
        shape: shape.into(),
        strides: strides.into(),
        runtime: PhantomData,
    }
}

fn finite_grid(elements: usize, limits: (u32, u32, u32)) -> Result<CubeCount> {
    if elements == 0 || limits.0 == 0 || limits.1 == 0 || limits.2 == 0 {
        return Err(Error::InvalidBufferLength);
    }
    let blocks = elements.div_ceil(128);
    let y = blocks.div_ceil(limits.0 as usize);
    if y > limits.1 as usize {
        return Err(Error::LimitExceeded("CubeK finite grid"));
    }
    let x = blocks.div_ceil(y);
    let covered = x
        .checked_mul(y)
        .and_then(|n| n.checked_mul(128))
        .ok_or(Error::ArithmeticOverflow)?;
    if covered > u32::MAX as usize {
        return Err(Error::LimitExceeded("CubeK finite index"));
    }
    Ok(CubeCount::Static(x as u32, y as u32, 1))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_dense(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 4],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
    bfloat: bool,
) -> Result<bool> {
    if !eligible(count, columns, rows, weight_first, out_first, out_stride) {
        return Ok(false);
    }
    let hw = &client.properties().hardware;
    if hw.max_units_per_cube < 128
        || hw.max_cube_dim.0 < 128
        || hw.max_cube_dim.1 < 4
        || hw.max_cube_dim.2 < 1
        || hw.max_shared_memory_size < SHARED_CAP
    {
        return Ok(false);
    }
    let [input, weight, output, fault] = h;
    // Native handle offsets must be descriptor-aligned, in addition to vec4
    // alignment. Do not introduce unaligned descriptor slices for a packed view.
    let alignment = client.properties().memory.alignment;
    let weight_offset = weight_first
        .checked_mul(2)
        .ok_or(Error::ArithmeticOverflow)? as u64;
    let output_offset = out_first.checked_mul(4).ok_or(Error::ArithmeticOverflow)? as u64;
    if alignment == 0
        || !weight_offset.is_multiple_of(alignment)
        || !output_offset.is_multiple_of(alignment)
        || [&input, &weight, &output]
            .iter()
            .any(|h| !h.offset_start.unwrap_or(0).is_multiple_of(alignment))
    {
        return Ok(false);
    }
    if bfloat {
        launch::<bf16>(
            client,
            input,
            weight,
            output,
            fault,
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
        )?;
    } else {
        launch::<f16>(
            client,
            input,
            weight,
            output,
            fault,
            count,
            columns,
            rows,
            weight_first,
            out_first,
            out_stride,
        )?;
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn launch<F: Float>(
    client: &ComputeClient<WgpuRuntime>,
    input: Handle,
    weight: Handle,
    output: Handle,
    fault: Handle,
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    out_first: usize,
    out_stride: usize,
) -> Result<()> {
    let lhs_elements = count
        .checked_mul(columns)
        .ok_or(Error::ArithmeticOverflow)?;
    let rhs_elements = rows.checked_mul(columns).ok_or(Error::ArithmeticOverflow)?;
    let out_elements = (count - 1)
        .checked_mul(out_stride)
        .and_then(|n| n.checked_add(rows))
        .ok_or(Error::ArithmeticOverflow)?;
    let guard_grid = finite_grid(
        count.checked_mul(rows).ok_or(Error::ArithmeticOverflow)?,
        client.properties().hardware.max_cube_count,
    )?;
    let lhs_handle = checked_view(input, 0, lhs_elements, 2)?;
    let rhs_handle = checked_view(weight, weight_first, rhs_elements, 2)?;
    let out_handle = checked_view(output.clone(), out_first, out_elements, 4)?;
    if !output.size_in_used().is_multiple_of(4) || fault.size_in_used() < 4 {
        return Err(Error::InvalidBufferLength);
    }
    let elem = F::as_type_native_unchecked().storage_type();
    let f32_ty = ElemType::Float(FloatKind::F32).into();
    if !client
        .properties()
        .features
        .matmul
        .cmma
        .contains(&cubecl::ir::features::MmaConfig {
            a_type: elem,
            b_type: elem,
            cd_type: f32_ty,
            m: 8,
            n: 16,
            k: 16,
        })
    {
        return Err(Error::InvalidFormat("CubeK original matrix capability"));
    }
    let lhs = InputBinding::new(tensor(lhs_handle, [count, columns], [columns, 1]), elem);
    // The physical original image remains [output feature][K]. This logical
    // K×N view is column-major, not a weight transpose or an F32 conversion.
    let rhs = InputBinding::new(tensor(rhs_handle, [columns, rows], [1, columns]), elem);
    let out = tensor(out_handle, [count, rows], [out_stride, 1]);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: elem,
        rhs: elem,
        out: f32_ty,
    });
    let problem = MatmulProblem::from_shapes_and_strides(
        lhs.data().shape.clone(),
        rhs.data().shape.clone(),
        out.shape.clone(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
        dtypes.as_global_elems(),
        AddressType::U32,
        None,
        None,
    )
    .map_err(|_| Error::InvalidFormat("CubeK original tensor layout"))?;
    let vector_sizes = MatmulVectorSizes {
        lhs: 4,
        rhs: 4,
        out: 4,
    };
    let settings = DeviceSettings {
        client: client.clone(),
        plane_dim: 16,
        vector_sizes,
        max_cube_count: client.properties().hardware.max_cube_count,
    };
    // Every complete blueprint keeps true SIMD16 and ordered K16 products.
    // Wide dense uses K32 paired stages and eight real subgroups; original
    // M32/M64 use K64 and four. Every choice checks actual device resources.
    let hw = &client.properties().hardware;
    let plan = dense_plan(
        count,
        hw.max_shared_memory_size,
        hw.max_units_per_cube,
        hw.max_cube_dim,
    );
    if hw.max_shared_memory_size < plan.required_shared() {
        return Err(Error::InvalidFormat(
            "CubeK actual matrix shared capability",
        ));
    }
    let scheme = plan.scheme()?;
    let blueprint = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, 16, &problem)
        .partition_buffering(plan.buffering())
        .build();
    let expanded = <Algorithm as Routine<()>>::expand_blueprint(
        &problem,
        &settings,
        &BlueprintStrategy::Forced(blueprint),
    )
    .map_err(|_| Error::InvalidFormat("CubeK fixed matrix blueprint"))?;
    let info = <Algorithm as Routine<()>>::prepare(&problem, &settings, expanded)
        .map_err(|_| Error::InvalidFormat("CubeK fixed matrix resources"))?;
    if info.cube_dim != plan.cube_dim() || info.dtypes != dtypes {
        return Err(Error::InvalidFormat(
            "CubeK fixed precision or geometry changed",
        ));
    }
    let config = <Family as BatchMatmulFamily<()>>::expand_config(
        client.properties(),
        &info.blueprint,
        &info.dtypes,
        &info.vector_sizes,
    )
    .map_err(|_| Error::InvalidFormat("CubeK complete matrix pipeline"))?;
    let inputs = <Inputs<F> as ConcreteInputsFactory<Algorithm>>::create(
        lhs,
        rhs,
        &info.blueprint,
        &problem,
        &info.vector_sizes,
        &info.dtypes,
    );
    let result = <Output as ConcreteOutputFactory<Algorithm>>::create(
        out,
        &info.blueprint,
        &problem,
        &info.vector_sizes,
        &info.dtypes,
    );
    // All dimensions, physical view bounds/strides, real capabilities and
    // complete library configuration were checked before this original launch.
    unsafe {
        library_dense::launch_unchecked::<F, WgpuRuntime>(
            client,
            info.cube_count_plan.resolve(),
            info.cube_dim,
            inputs,
            result,
            cube_mapping_launch(&info.cube_count_plan),
            info.blueprint,
            config,
        );
        finite_output::launch_unchecked::<WgpuRuntime>(
            client,
            guard_grid,
            CubeDim::new_1d(128),
            ArrayArg::from_raw_parts(output.clone(), output.size_in_used() as usize / 4),
            ArrayArg::from_raw_parts(fault, 1),
            count,
            rows,
            out_first,
            out_stride,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::Instruction;
    pub(super) fn properties() -> cubecl::ir::DeviceProperties {
        use cubecl::ir::{
            HardwareProperties, MemoryDeviceProperties, UIntKind,
            features::{Features, MmaConfig, TypeUsage},
        };
        let mut features = Features::default();
        for kind in [FloatKind::F16, FloatKind::BF16, FloatKind::F32] {
            features
                .types
                .storage
                .insert(ElemType::Float(kind).into(), TypeUsage::all());
        }
        features
            .types
            .storage
            .insert(ElemType::UInt(UIntKind::U32).into(), TypeUsage::all());
        features.types.address.insert(AddressType::U32);
        for kind in [FloatKind::F16, FloatKind::BF16] {
            let ty = ElemType::Float(kind).into();
            features.matmul.cmma.insert(MmaConfig {
                a_type: ty,
                b_type: ty,
                cd_type: ElemType::Float(FloatKind::F32).into(),
                m: 8,
                n: 16,
                k: 16,
            });
        }
        // Offline compiler fixtures only, never passed to a device/request.
        cubecl::ir::DeviceProperties::new(
            features,
            MemoryDeviceProperties {
                max_page_size: 1 << 30,
                alignment: 256,
            },
            HardwareProperties {
                load_width: 128,
                plane_size_min: 16,
                plane_size_max: 32,
                max_bindings: 32,
                max_shared_memory_size: 65536,
                max_cube_count: (65535, 65535, 65535),
                max_units_per_cube: 1024,
                max_cube_dim: (1024, 1024, 64),
                num_streaming_multiprocessors: None,
                num_cpu_cores: None,
                num_tensor_cores: None,
                min_tensor_cores_dim: Some(8),
                max_vector_size: 4,
            },
            cubecl_common::profile::TimingMethod::System,
        )
    }
    fn definition<F: Float>(count: usize) -> KernelDefinition {
        definition_partition::<F>(count, 4)
    }
    pub(super) fn definition_partition<F: Float>(count: usize, m_tiles: u32) -> KernelDefinition {
        definition_plan::<F>(
            count,
            1280,
            DensePlan::Original(MatrixPlan {
                m_tiles,
                required_shared: if m_tiles == 8 { 34816 } else { SHARED_CAP },
            }),
        )
    }
    fn definition_plan<F: Float>(count: usize, rows: usize, plan: DensePlan) -> KernelDefinition {
        let props = properties();
        let elem = F::as_type_native_unchecked().storage_type();
        let f32_ty = ElemType::Float(FloatKind::F32).into();
        let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: elem,
            rhs: elem,
            out: f32_ty,
        });
        let shape = |bytes| Handle::new(cubecl_common::stream_id::StreamId::current(), bytes);
        let lhs = InputBinding::new(
            tensor(shape((count * 2560 * 2) as u64), [count, 2560], [2560, 1]),
            elem,
        );
        let rhs = InputBinding::new(
            tensor(shape((rows * 2560 * 2) as u64), [2560, rows], [1, 2560]),
            elem,
        );
        let out = tensor(shape((count * rows * 4) as u64), [count, rows], [rows, 1]);
        let problem = MatmulProblem::from_shapes_and_strides(
            lhs.data().shape.clone(),
            rhs.data().shape.clone(),
            out.shape.clone(),
            lhs.data().strides.clone(),
            rhs.data().strides.clone(),
            out.strides.clone(),
            dtypes.as_global_elems(),
            AddressType::U32,
            None,
            None,
        )
        .unwrap();
        let scheme = plan.scheme().unwrap();
        let block_m = scheme.elements_per_stage_along_m() as usize;
        let block_n = scheme.elements_per_stage_along_n() as usize;
        let blueprint = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, 16, &problem)
            .partition_buffering(plan.buffering())
            .build();
        let vectors = MatmulVectorSizes {
            lhs: 4,
            rhs: 4,
            out: 4,
        };
        let config =
            <Family as BatchMatmulFamily<()>>::expand_config(&props, &blueprint, &dtypes, &vectors)
                .unwrap();
        assert_eq!(
            <Family as BatchMatmulFamily<()>>::cubedim_resource(&blueprint, &dtypes, &vectors)
                .unwrap()
                .num_planes(16)
                .unwrap_or_else(|e| panic!("{e}")),
            plan.cube_dim().y
        );
        let input = <Inputs<F> as ConcreteInputsFactory<Algorithm>>::create(
            lhs, rhs, &blueprint, &problem, &vectors, &dtypes,
        );
        let output = <Output as ConcreteOutputFactory<Algorithm>>::create(
            out, &blueprint, &problem, &vectors, &dtypes,
        );
        let settings = KernelSettings::default().cube_dim(plan.cube_dim());
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        launcher.with_scope(|scope| scope.device_properties(&props));
        let input = <Inputs<F> as LaunchArg>::register(input, &mut launcher);
        let output = <Output as LaunchArg>::register(output, &mut launcher);
        let plan = cubek_std::cube_count::CubeCountPlan::new_from_problem(
            cubek_std::cube_count::Count3d {
                x: count.div_ceil(block_m) as u32,
                y: rows.div_ceil(block_n) as u32,
                z: 1,
            },
        );
        let mapping =
            <CubeMapping as LaunchArg>::register(cube_mapping_launch(&plan), &mut launcher);
        let mut builder = KernelBuilder::default();
        builder.scope.device_properties(&props);
        AddressType::U32.register(&mut builder.scope);
        let input = <Inputs<F> as LaunchArg>::expand(&input, &mut builder);
        let output = <Output as LaunchArg>::expand_output(&output, &mut builder);
        let mapping = <CubeMapping as LaunchArg>::expand(&mapping, &mut builder);
        library_dense::expand::<F>(
            &mut builder.scope,
            input,
            output,
            mapping,
            blueprint,
            config,
        );
        let errors = builder.scope.pop_errors();
        assert!(errors.is_empty(), "{errors:?}");
        builder.build(settings)
    }
    #[test]
    fn m64_choice_is_explicit_high_count_and_real_per_workgroup_cap_only() {
        for count in [32, 33, 128, 255, 256, 257, 4096] {
            for shared in [32768, 34815, 34816, 49152] {
                let actual = matrix_plan(count, shared);
                let selected = cfg!(all(
                    feature = "strata-cubek-m64",
                    feature = "strata-cubek-tilewise"
                )) && count >= 256
                    && shared >= M64_SHARED_CAP;
                assert_eq!(actual.m_tiles, if selected { 8 } else { 4 });
                assert_eq!(
                    actual.required_shared,
                    if selected { M64_SHARED_CAP } else { SHARED_CAP }
                );
                assert!(actual.required_shared <= shared);
                let scheme = matrix_scheme(actual).unwrap();
                assert_eq!(
                    matrix_partition_buffering(),
                    if cfg!(all(
                        feature = "strata-cubek-balanced",
                        feature = "strata-cubek-tilewise"
                    )) {
                        PartitionBuffering::Double
                    } else {
                        PartitionBuffering::Single
                    }
                );
                assert_eq!(
                    scheme.partitions_per_stage_along_m() * scheme.partitions_per_stage_along_n(),
                    4
                );
                assert_eq!(scheme.elements_per_stage_along_m(), actual.m_tiles * 8);
                assert_eq!(scheme.elements_per_stage_along_n(), 64);
                assert_eq!(scheme.elements_per_stage_along_k(), 64);
            }
        }
    }
    #[test]
    fn candidate_wide_requires_explicit_feature_count_and_true_eight_plane_limits() {
        for count in [32, 255, 256, 511, 512, 513, 4096] {
            for shared in [32768, 34816, 36863, 36864, 49152] {
                for (units, dims) in [
                    (127, (16, 8, 1)),
                    (128, (15, 8, 1)),
                    (128, (16, 7, 1)),
                    (128, (16, 8, 0)),
                    (128, (16, 8, 1)),
                    (1024, (1024, 1024, 64)),
                ] {
                    let plan = dense_plan(count, shared, units, dims);
                    let selected = cfg!(all(
                        feature = "strata-cubek-wide",
                        feature = "strata-cubek-tilewise"
                    )) && count >= 512
                        && shared >= WIDE_SHARED_CAP
                        && units >= 128
                        && dims.0 >= 16
                        && dims.1 >= 8
                        && dims.2 >= 1;
                    assert_eq!(plan == DensePlan::Wide, selected);
                    if selected {
                        let scheme = plan.scheme().unwrap();
                        assert_eq!(scheme.elements_per_stage_along_m(), 128);
                        assert_eq!(scheme.elements_per_stage_along_n(), 128);
                        assert_eq!(scheme.elements_per_stage_along_k(), 32);
                        assert_eq!(scheme.partitions_per_stage_along_m(), 4);
                        assert_eq!(scheme.partitions_per_stage_along_n(), 2);
                        assert_eq!(plan.cube_dim(), CubeDim::new_2d(16, 8));
                        assert_eq!(plan.buffering(), PartitionBuffering::Double);
                        assert_eq!(plan.required_shared(), WIDE_SHARED_CAP);
                    } else {
                        assert_eq!(plan, DensePlan::Original(matrix_plan(count, shared)));
                    }
                }
            }
        }
    }
    #[cfg(feature = "strata-cubek-wide")]
    fn check_wide_modules(masks: std::ops::Range<u32>) {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        for mask in masks {
            for count in [512, 513] {
                let plan = dense_plan(count, 49152, 1024, (1024, 1024, 64));
                assert_eq!(plan, DensePlan::Wide);
                for (name, definition) in [
                    ("fp16", definition_plan::<f16>(count, 128, plan)),
                    ("bf16", definition_plan::<bf16>(count, 128, plan)),
                ] {
                    let mut options = WgpuCompilationOptions {
                        supports_vulkan: true,
                        ..Default::default()
                    };
                    options.vulkan.max_spirv_version = (1, 6);
                    options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
                    options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
                    options.vulkan.supports_explicit_smem = mask & 4 != 0;
                    let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                        &mut AutoCompiler::SpirV(Default::default()),
                        definition,
                        &options,
                        ExecutionMode::Unchecked,
                        StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                    )
                    .unwrap() else {
                        panic!("actual SPIR-V required")
                    };
                    assert_eq!(kernel.shared_size, WIDE_SHARED_CAP);
                    let words =
                        wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)
                            .unwrap();
                    let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
                        .unwrap()
                        .unwrap_or(words);
                    assert_eq!(
                        wgpu::hal::vulkan::kuno_cooperative_subgroup(&words),
                        Some(16)
                    );
                    let mut at = 5;
                    let mut local = None;
                    let mut builtin = Vec::new();
                    let mut products = 0;
                    while at < words.len() {
                        let n = (words[at] >> 16) as usize;
                        let op = words[at] & 65535;
                        let args = &words[at + 1..at + n];
                        if op == 16 && args.get(1) == Some(&17) {
                            local = Some(args[2..5].to_vec());
                        }
                        if op == 71 && args.get(1) == Some(&11) {
                            builtin.push(args[2]);
                        }
                        if op == 4459 {
                            products += 1;
                        }
                        assert!(
                            op != 71 || args.get(1) != Some(&40),
                            "relaxed FPFastMathMode"
                        );
                        at += n;
                    }
                    assert_eq!(local, Some(vec![16, 8, 1]));
                    assert!(builtin.contains(&40) && builtin.contains(&41));
                    assert!(!builtin.iter().any(|b| matches!(b, 27 | 28 | 29)));
                    assert_eq!(products, 128);
                    let mut cmd = std::process::Command::new("spirv-val");
                    cmd.args(["--target-env", "vulkan1.3"]);
                    if mask & 1 != 0 {
                        cmd.arg("--uniform-buffer-standard-layout");
                    }
                    let mut child = cmd
                        .arg("-")
                        .stdin(std::process::Stdio::piped())
                        .spawn()
                        .unwrap();
                    child
                        .stdin
                        .take()
                        .unwrap()
                        .write_all(
                            &words
                                .iter()
                                .flat_map(|v| v.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )
                        .unwrap();
                    assert!(child.wait().unwrap().success(), "{name}/{count}/{mask}");
                    println!(
                        "compiled_public_cubek_wide name={name} count={count} mask={mask} shared_bytes={} products={products}",
                        kernel.shared_size
                    );
                }
            }
        }
    }
    #[cfg(feature = "strata-cubek-wide")]
    #[test]
    fn candidate_wide_actual_first_layouts() {
        check_wide_modules(0..4);
    }
    #[cfg(feature = "strata-cubek-wide")]
    #[test]
    fn candidate_wide_actual_second_layouts() {
        check_wide_modules(4..8);
    }
    #[test]
    fn admission_keeps_original_small_and_unaligned_fallbacks() {
        assert!(eligible(32, 2560, 1280, 0, 0, 1280));
        assert!(eligible(4096, 10240, 2560, 64, 64, 2560));
        for shape in [
            (31, 2560, 1280, 0, 0, 1280),
            (32, 16, 1280, 0, 0, 1280),
            (32, 2561, 1280, 0, 0, 1280),
            (32, 2560, 65, 0, 0, 1280),
            (32, 2560, 1280, 1, 0, 1280),
            (32, 2560, 1280, 0, 1, 1280),
            (32, 2560, 1280, 0, 0, 1279),
            (4097, 2560, 1280, 0, 0, 1280),
        ] {
            assert!(!eligible(
                shape.0, shape.1, shape.2, shape.3, shape.4, shape.5
            ));
        }
    }
    #[test]
    fn actual_ir_hook_is_mapping_independent_and_rejects_unreviewed_topologies() {
        let ty: cubecl::ir::Type = cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32).into();
        let mut scope = Scope::root(false);
        let out: Variable = scope.create_local(ty).into();
        scope.instructions.push(Instruction::new(
            Operation::Copy(Variable::builtin(Builtin::UnitPosX, ty.storage_type())),
            out,
        ));
        scope.instructions.push(Instruction::new(
            Operation::Copy(Variable::builtin(Builtin::UnitPosY, ty.storage_type())),
            out,
        ));
        normalize_scope(&mut scope).unwrap();
        for (i, expected) in [Builtin::UnitPosPlane, Builtin::PlanePos]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                scope.instructions[i].operation.args().unwrap()[0].kind,
                VariableKind::Builtin(expected)
            );
        }
        for builtin in [
            Builtin::UnitPos,
            Builtin::UnitPosZ,
            Builtin::AbsolutePos,
            Builtin::AbsolutePosX,
            Builtin::AbsolutePosY,
            Builtin::AbsolutePosZ,
        ] {
            let mut value = Variable::builtin(builtin, ty.storage_type());
            assert!(normalize_variable(&mut value).is_err());
        }
        let mut nested = Scope::root(false);
        nested.instructions.push(Instruction::new(
            Operation::Copy(Variable::builtin(Builtin::UnitPosY, ty.storage_type())),
            out,
        ));
        let mut outer = Scope::root(false);
        outer
            .instructions
            .push(Instruction::no_out(Operation::Branch(Branch::If(
                Box::new(cubecl::ir::If {
                    cond: Variable::constant(cubecl::ir::ConstantValue::Bool(true), ElemType::Bool),
                    scope: nested,
                }),
            ))));
        normalize_scope(&mut outer).unwrap();
        let Operation::Branch(Branch::If(branch)) = &outer.instructions[0].operation else {
            panic!("if preserved")
        };
        assert_eq!(
            branch.scope.instructions[0].operation.args().unwrap()[0].kind,
            VariableKind::Builtin(Builtin::PlanePos)
        );
        outer.instructions[0].modes.fp_math_mode |= cubecl::ir::FastMath::NotNaN;
        assert!(normalize_scope(&mut outer).is_err());
        let mut outer = Scope::root(false);
        outer
            .instructions
            .push(Instruction::no_out(Operation::NonSemantic(
                cubecl::ir::NonSemantic::Print {
                    format_string: "rejected static diagnostic".into(),
                    args: vec![],
                },
            )));
        assert!(normalize_scope(&mut outer).is_err());
    }
    #[test]
    fn authentic_views_keep_owner_end_and_reject_overflow_truncation_or_false_alignment() {
        let new = |bytes| Handle::new(cubecl_common::stream_id::StreamId::current(), bytes);
        let owner = new(4096).offset_start(256).offset_end(64);
        let view = checked_view(owner.clone(), 128, 256, 2).unwrap();
        assert_eq!(view.size(), owner.size());
        assert_eq!(view.offset_start, Some(512));
        assert_eq!(view.offset_end, owner.offset_end);
        assert_eq!(view.size_in_used(), owner.size_in_used() - 256);
        for (handle, first, elements, bytes) in [
            (new(512), 0, 257, 2),
            (new(512), 1, 128, 2),
            (new(512).offset_start(1), 0, 128, 2),
            (new(512), usize::MAX, 2, 2),
            (new(512), 0, usize::MAX, 4),
            (new(4).offset_start(8), 0, 1, 2),
        ] {
            assert!(checked_view(handle, first, elements, bytes).is_err());
        }
    }
    #[test]
    fn actual_guard_grid_spreads_large_outputs_without_exceeding_real_limits_or_index_width() {
        for elements in [1, 128, 129, 4096 * 2560, 4096 * 10240, 4096 * 16384] {
            let CubeCount::Static(x, y, z) = finite_grid(elements, (65535, 65535, 65535)).unwrap()
            else {
                panic!("static")
            };
            assert!(x > 0 && x <= 65535 && y > 0 && y <= 65535 && z == 1);
            let span = x as usize * y as usize * 128;
            assert!(
                span >= elements && span < elements + 128 * y as usize && span <= u32::MAX as usize
            );
            // ABSOLUTE_POS has contiguous128-wide workgroup intervals;
            // (groupY*groupsX+groupX)*128+lane is bijective over this span.
            for (gy, gx) in [(0, 0), (y - 1, x - 1)] {
                let first = (gy as usize * x as usize + gx as usize) * 128;
                assert!(first + 127 < span);
            }
        }
        for (n, limits) in [
            (0, (65535, 65535, 65535)),
            (129, (1, 1, 1)),
            (1, (0, 1, 1)),
            (1, (1, 0, 1)),
            (1, (1, 1, 0)),
            (u32::MAX as usize, (65535, 65535, 65535)),
        ] {
            assert!(finite_grid(n, limits).is_err());
        }
    }
    #[test]
    fn actual_library_compiler_preserves_geometry_types_no_local_id_and_ieee_controls() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        let mut observed_shared = std::collections::BTreeSet::new();
        for mask in 0..8 {
            for (name, definition) in [
                ("fp16-full", definition::<f16>(32)),
                ("fp16-tail", definition::<f16>(33)),
                ("bf16-full", definition::<bf16>(32)),
                ("bf16-tail", definition::<bf16>(33)),
            ] {
                let mut options = WgpuCompilationOptions {
                    supports_vulkan: true,
                    ..Default::default()
                };
                options.vulkan.max_spirv_version = (1, 6);
                options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
                options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
                options.vulkan.supports_explicit_smem = mask & 4 != 0;
                let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                    &mut AutoCompiler::SpirV(Default::default()),
                    definition,
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap() else {
                    panic!("actual SPIR-V required")
                };
                observed_shared.insert(kernel.shared_size);
                assert!(
                    kernel.shared_size <= SHARED_CAP,
                    "actual {}",
                    kernel.shared_size
                );
                let words =
                    wgpu::hal::vulkan::kuno_native_workgroup_layout(&kernel.assembled_module)
                        .unwrap();
                let words = wgpu::hal::vulkan::kuno_cooperative_fp16_preserve(&words)
                    .unwrap()
                    .unwrap_or(words);
                assert_eq!(
                    wgpu::hal::vulkan::kuno_cooperative_subgroup(&words),
                    Some(16)
                );
                let mut at = 5;
                let mut local = None;
                let mut builtin = Vec::new();
                let mut products = 0;
                while at < words.len() {
                    let n = (words[at] >> 16) as usize;
                    let op = words[at] & 65535;
                    let args = &words[at + 1..at + n];
                    if op == 16 && args.get(1) == Some(&17) {
                        local = Some(args[2..5].to_vec());
                    }
                    if op == 71 && args.get(1) == Some(&11) {
                        builtin.push(args[2]);
                    }
                    if op == 4459 {
                        products += 1;
                    }
                    assert!(
                        op != 71 || args.get(1) != Some(&40),
                        "relaxed FPFastMathMode"
                    );
                    at += n;
                }
                assert_eq!(local, Some(vec![16, 4, 1]));
                assert!(builtin.contains(&40) && builtin.contains(&41));
                assert!(
                    !builtin.iter().any(|b| matches!(b, 27 | 28 | 29)),
                    "physical local/absolute ID remains"
                );
                assert!(products >= 4);
                let mut cmd = std::process::Command::new("spirv-val");
                cmd.args(["--target-env", "vulkan1.3"]);
                if mask & 1 != 0 {
                    cmd.arg("--uniform-buffer-standard-layout");
                }
                let mut child = cmd
                    .arg("-")
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .unwrap();
                child
                    .stdin
                    .take()
                    .unwrap()
                    .write_all(
                        &words
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                assert!(child.wait().unwrap().success(), "{name} mask{mask}");
            }
        }
        println!(
            "compiled_public_cubek_dense tilewise={} modules=32 observed_shared_bytes={observed_shared:?}",
            cfg!(feature = "strata-cubek-tilewise")
        );
    }
}
