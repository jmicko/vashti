//! Genuine routed CubeK output: original GU/PART/map/status/fault owners only.
use super::*;
use crate::strata_prompt::groups as abi;
use cubecl::std::tensor::{
    View, ViewExpand, ViewOperationsExpand, ViewOperationsMutExpand, launch::ViewArg,
};
use cubek_matmul::{
    components::global::memory::{GlobalLayout, GlobalLayoutLaunch},
    launch::BatchedCoords,
};

type Four = Vector<f32, Const<4>>;
type HalfFour = Vector<f16, Const<4>>;
type ExpertInputs = (View<HalfFour, BatchedCoords>, View<HalfFour, BatchedCoords>);
#[derive(Clone)]
struct ExpertArgs;
#[cube]
impl MatmulArgs for ExpertArgs {
    type Input<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive> =
        (View<L, BatchedCoords>, View<R, BatchedCoords>);
    type Output<E: CubePrimitive> = View<E, BatchedCoords, ReadWrite>;
    type Config = ();
    type State<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive> = (
        (View<L, BatchedCoords>, View<R, BatchedCoords>),
        View<E, BatchedCoords, ReadWrite>,
    );
    fn init_state<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        input: &Self::Input<L, R, E>,
        output: &mut Self::Output<E>,
        _config: (),
        #[comptime] _lhs_layout: cubek_matmul::components::global::memory::GlobalLayoutConfig,
        #[comptime] _rhs_layout: cubek_matmul::components::global::memory::GlobalLayoutConfig,
        #[comptime] _out_layout: cubek_matmul::components::global::memory::GlobalLayoutConfig,
    ) -> Self::State<L, R, E> {
        (*input, *output)
    }
    fn view_lhs<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        state: &Self::State<L, R, E>,
    ) -> View<L, BatchedCoords> {
        state.0.0
    }
    fn view_rhs<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        state: &Self::State<L, R, E>,
    ) -> View<R, BatchedCoords> {
        state.0.1
    }
    fn view_acc<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
    ) -> ComptimeOption<View<E, BatchedCoords>> {
        ComptimeOption::new_None()
    }
    fn view_out<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        state: &mut Self::State<L, R, E>,
    ) -> View<E, BatchedCoords, ReadWrite> {
        state.1
    }
    fn batch_lhs<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
        batch: usize,
    ) -> usize {
        batch
    }
    fn batch_rhs<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
        batch: usize,
    ) -> usize {
        batch
    }
    fn batch_acc<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
        batch: usize,
    ) -> usize {
        batch
    }
    fn batch_out<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
        batch: usize,
    ) -> usize {
        batch
    }
    fn runtime_config<L: CubePrimitive, R: CubePrimitive, E: CubePrimitive>(
        _state: &Self::State<L, R, E>,
    ) {
    }
}

#[cube]
fn output_bounds(batch: usize, member: u32, row: u32, count: usize, rows: usize) -> bool {
    batch == 0 && usize::cast_from(member) < count && usize::cast_from(row) < rows
}

#[allow(clippy::manual_range_contains)]
#[cube]
fn store_routed(
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    batch: usize,
    member: u32,
    row: u32,
    value: Four,
    count: usize,
    rows: usize,
    group: usize,
    #[comptime] down: bool,
) {
    if output_bounds(batch, member, row, count, rows) {
        let total = usize::cast_from(map[0]);
        let mut header = false;
        if total >= 1
            && total <= comptime!(abi::ROUTES / 10)
            && group < 512
            && group < usize::cast_from(map[3])
        {
            let offset = usize::cast_from(map[1028 + group]);
            header = map[1] == 0
                && usize::cast_from(map[2]) == total
                && usize::cast_from(map[3]) <= 512
                && usize::cast_from(map[4 + group]) < 512
                && usize::cast_from(map[516 + group]) == count
                && count <= total
                && offset <= total * 10
                && count <= total * 10 - offset;
        }
        if header {
            let route = usize::cast_from(
                map[1540 + usize::cast_from(map[1028 + group]) + usize::cast_from(member)],
            );
            if route < total * 10 {
                let mut base = usize::cast_from(member) * 1280;
                if comptime!(down) {
                    base = route * 2560;
                }
                #[unroll]
                for i in 0..4usize {
                    let feature = usize::cast_from(row) + i;
                    if feature < rows {
                        let v = value[i];
                        let good = v.abs() <= f32::MAX;
                        if good {
                            output[base + feature] = v;
                        } else {
                            output[base + feature] = 0.0;
                            fault[0].fetch_or(128);
                        }
                        if comptime!(down) && feature == 0 {
                            let mut valid = 0u32;
                            if good && status[route] == 1 {
                                valid = 1;
                            }
                            status[comptime!(abi::ROUTES) + route] = valid;
                        }
                    }
                }
            } else {
                fault[0].fetch_or(128);
            }
        } else {
            fault[0].fetch_or(128);
        }
    }
}

#[derive(Clone)]
struct RoutedExpand<E: CubePrimitive> {
    output: NativeExpand<Array<f32>>,
    map: NativeExpand<Array<u32>>,
    status: NativeExpand<Array<u32>>,
    fault: NativeExpand<Array<Atomic<u32>>>,
    count: NativeExpand<usize>,
    rows: NativeExpand<usize>,
    group: NativeExpand<usize>,
    down: bool,
    _element: PhantomData<E>,
}
impl<E: CubePrimitive> VectorizedExpand for RoutedExpand<E> {
    fn vector_size(&self) -> usize {
        4
    }
}
impl<E: CubePrimitive> ViewOperationsExpand<E, BatchedCoords> for RoutedExpand<E> {
    // This output is deliberately write-only. A future library family that
    // attempts an accumulator read, raw slice or TMA operation must fail at
    // compile time instead of manufacturing a contiguous routed output view.
    fn __expand_read_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
    ) -> E::ExpandType {
        panic!("CubeK routed output cannot be read");
    }
    fn __expand_read_checked_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
    ) -> E::ExpandType {
        panic!("CubeK routed output cannot be read");
    }
    fn __expand_read_masked_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
        _value: E::ExpandType,
    ) -> E::ExpandType {
        panic!("CubeK routed output cannot be read");
    }
    fn __expand_read_unchecked_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
    ) -> E::ExpandType {
        panic!("CubeK routed output cannot be read");
    }
    fn __expand_to_linear_slice_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
        _size: <BatchedCoords as CubeType>::ExpandType,
    ) -> SliceExpand<E, ReadOnly> {
        panic!("CubeK routed output has no contiguous slice");
    }
    fn __expand_tensor_map_load_method(
        &self,
        _scope: &mut Scope,
        _barrier: cubecl::prelude::barrier::BarrierExpand,
        _shared_memory: SliceExpand<E, ReadWrite>,
        _pos: <BatchedCoords as CubeType>::ExpandType,
    ) {
        panic!("CubeK routed output cannot use TMA");
    }
    fn __expand_shape_method(&self, scope: &mut Scope) -> <BatchedCoords as CubeType>::ExpandType {
        (
            1usize.into(),
            u32::__expand_cast_from(scope, self.count.clone()),
            u32::__expand_cast_from(scope, self.rows.clone()),
        )
    }
    fn __expand_is_in_bounds_method(
        &self,
        scope: &mut Scope,
        pos: <BatchedCoords as CubeType>::ExpandType,
    ) -> NativeExpand<bool> {
        output_bounds::expand(
            scope,
            pos.0,
            pos.1,
            pos.2,
            self.count.clone(),
            self.rows.clone(),
        )
    }
}
impl<E: CubePrimitive> ViewOperationsMutExpand<E, BatchedCoords> for RoutedExpand<E> {
    fn __expand_to_linear_slice_mut_method(
        &self,
        _scope: &mut Scope,
        _pos: <BatchedCoords as CubeType>::ExpandType,
        _size: <BatchedCoords as CubeType>::ExpandType,
    ) -> SliceExpand<E, ReadWrite> {
        panic!("CubeK routed output has no contiguous slice");
    }
    fn __expand_tensor_map_store_method(
        &self,
        _scope: &mut Scope,
        _shared_memory: SliceExpand<E, ReadOnly>,
        _pos: <BatchedCoords as CubeType>::ExpandType,
    ) {
        panic!("CubeK routed output cannot use TMA");
    }
    fn __expand_write_method(
        &self,
        scope: &mut Scope,
        pos: <BatchedCoords as CubeType>::ExpandType,
        value: E::ExpandType,
    ) {
        self.__expand_write_checked_method(scope, pos, value);
    }
    fn __expand_write_checked_method(
        &self,
        scope: &mut Scope,
        pos: <BatchedCoords as CubeType>::ExpandType,
        value: E::ExpandType,
    ) {
        // Validate the actual computed SSA type before adapting the Rust marker.
        // No scalar pointer/vector length or matrix result type is reinterpreted.
        if E::as_type(scope) != Four::as_type(scope) || value.expand.ty != Four::as_type(scope) {
            scope.push_error("CubeK expert result must be authentic F32vec4");
            return;
        }
        store_routed::expand(
            scope,
            self.output.clone(),
            self.map.clone(),
            self.status.clone(),
            self.fault.clone(),
            pos.0,
            pos.1,
            pos.2,
            NativeExpand::<Four>::new(value.expand),
            self.count.clone(),
            self.rows.clone(),
            self.group.clone(),
            self.down,
        );
    }
}
#[allow(clippy::too_many_arguments)]
fn routed_output(
    _output: &mut Array<f32>,
    _map: &Array<u32>,
    _status: &mut Array<u32>,
    _fault: &mut Array<Atomic<u32>>,
    _count: usize,
    _rows: usize,
    _group: usize,
    _down: bool,
) -> View<Four, BatchedCoords, ReadWrite> {
    cubecl::unexpanded!()
}
mod routed_output {
    use super::*;
    #[allow(clippy::too_many_arguments)]
    pub fn expand(
        _scope: &mut Scope,
        output: NativeExpand<Array<f32>>,
        map: NativeExpand<Array<u32>>,
        status: NativeExpand<Array<u32>>,
        fault: NativeExpand<Array<Atomic<u32>>>,
        count: NativeExpand<usize>,
        rows: NativeExpand<usize>,
        group: NativeExpand<usize>,
        down: bool,
    ) -> ViewExpand<Four, BatchedCoords, ReadWrite> {
        ViewExpand::new_mut(RoutedExpand {
            output,
            map,
            status,
            fault,
            count,
            rows,
            group,
            down,
            _element: PhantomData,
        })
    }
}
#[cube(launch_unchecked)]
fn library_expert(
    inputs: &ExpertInputs,
    output: &mut Array<f32>,
    map: &Array<u32>,
    status: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    rows: usize,
    group: usize,
    cube_mapping: CubeMapping,
    #[comptime] down: bool,
    #[comptime] blueprint: TilingBlueprint,
    #[comptime] config: LibraryConfig,
) {
    let mut view = routed_output(output, map, status, fault, count, rows, group, down);
    let mut state = ExpertArgs::init_state::<HalfFour, HalfFour, Four>(
        inputs,
        &mut view,
        (),
        blueprint.lhs_global_layout_config(),
        blueprint.rhs_global_layout_config(),
        blueprint.out_global_layout_config(),
    );
    Product::<f16>::execute::<ExpertArgs>(&mut state, cube_mapping, config);
    normalize_logical_topology();
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_expert(
    client: &ComputeClient<WgpuRuntime>,
    h: [Handle; 6],
    count: usize,
    columns: usize,
    rows: usize,
    weight_first: usize,
    group: usize,
    down: bool,
) -> Result<bool> {
    if !eligible(count, columns, rows, weight_first, 0, rows) {
        return Ok(false);
    }
    let expected = if down { (640, 2560) } else { (2560, 1280) };
    if (columns, rows) != expected || group >= 512 {
        return Err(Error::InvalidBufferLength);
    }
    let hw = &client.properties().hardware;
    if hw.max_units_per_cube < 64
        || hw.max_cube_dim.0 < 16
        || hw.max_cube_dim.1 < 4
        || hw.max_cube_dim.2 < 1
        || hw.max_shared_memory_size < SHARED_CAP
    {
        return Ok(false);
    }
    let align = client.properties().memory.alignment;
    if align == 0
        || !(weight_first
            .checked_mul(2)
            .ok_or(Error::ArithmeticOverflow)? as u64)
            .is_multiple_of(align)
        || h.iter()
            .any(|v| !v.offset_start.unwrap_or(0).is_multiple_of(align))
    {
        return Ok(false);
    }
    let [input, weight, output, map, status, fault] = h;
    if map.size_in_used() != (abi::MAP_WORDS * 4) as u64
        || status.size_in_used() != (abi::STATUS_WORDS * 4) as u64
        || output.size_in_used() != if down { abi::PART_BYTES } else { abi::GU_BYTES }
        || fault.size_in_used() < 4
        || [&input, &weight, &output, &status, &fault]
            .iter()
            .any(|v| v.stream != map.stream)
    {
        return Err(Error::InvalidBufferLength);
    }
    let input = checked_view(
        input,
        0,
        count
            .checked_mul(columns)
            .ok_or(Error::ArithmeticOverflow)?,
        2,
    )?;
    let weight = checked_view(
        weight,
        weight_first,
        rows.checked_mul(columns).ok_or(Error::ArithmeticOverflow)?,
        2,
    )?;
    let elem = ElemType::Float(FloatKind::F16).into();
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
        return Err(Error::InvalidFormat(
            "CubeK original expert matrix capability",
        ));
    }
    let lhs = tensor(input, [count, columns], [columns, 1]);
    let rhs = tensor(weight, [columns, rows], [1, columns]);
    let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
        lhs: elem,
        rhs: elem,
        out: f32_ty,
    });
    let problem = MatmulProblem::from_parameters(
        count,
        rows,
        columns,
        Vec::<usize>::new().into(),
        Vec::<usize>::new().into(),
        cubek_std::MatrixLayout::RowMajor,
        cubek_std::MatrixLayout::ColMajor,
        cubek_std::MatrixLayout::RowMajor,
        None,
        None,
        dtypes.as_global_elems(),
        AddressType::U32,
    );
    let vectors = MatmulVectorSizes {
        lhs: 4,
        rhs: 4,
        out: 4,
    };
    let settings = DeviceSettings {
        client: client.clone(),
        plane_dim: 16,
        vector_sizes: vectors,
        max_cube_count: hw.max_cube_count,
    };
    let plan = matrix_plan(count, hw.max_shared_memory_size);
    if hw.max_shared_memory_size < plan.required_shared {
        return Err(Error::InvalidFormat(
            "CubeK actual expert shared capability",
        ));
    }
    let scheme = matrix_scheme(plan)?;
    let blueprint = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, 16, &problem)
        .partition_buffering(matrix_partition_buffering())
        .build();
    let expanded = <Algorithm as Routine<()>>::expand_blueprint(
        &problem,
        &settings,
        &BlueprintStrategy::Forced(blueprint),
    )
    .map_err(|_| Error::InvalidFormat("CubeK expert blueprint"))?;
    let info = <Algorithm as Routine<()>>::prepare(&problem, &settings, expanded)
        .map_err(|_| Error::InvalidFormat("CubeK expert capabilities"))?;
    if info.cube_dim != CubeDim::new_2d(16, 4) || info.dtypes != dtypes {
        return Err(Error::InvalidFormat("CubeK expert precision/geometry"));
    }
    let config = <Family as BatchMatmulFamily<()>>::expand_config(
        client.properties(),
        &info.blueprint,
        &info.dtypes,
        &vectors,
    )
    .map_err(|_| Error::InvalidFormat("CubeK expert pipeline"))?;
    let lhs_layout =
        GlobalLayoutLaunch::from_handle(&lhs, 4, info.blueprint.lhs_global_layout_config());
    let rhs_layout =
        GlobalLayoutLaunch::from_handle(&rhs, 4, info.blueprint.rhs_global_layout_config());
    let inputs = (
        ViewArg::new_tensor::<GlobalLayout>(lhs.into_tensor_arg(), lhs_layout),
        ViewArg::new_tensor::<GlobalLayout>(rhs.into_tensor_arg(), rhs_layout),
    );
    unsafe {
        library_expert::launch_unchecked::<WgpuRuntime>(
            client,
            info.cube_count_plan.resolve(),
            info.cube_dim,
            inputs,
            ArrayArg::from_raw_parts(output.clone(), output.size_in_used() as usize / 4),
            ArrayArg::from_raw_parts(map, abi::MAP_WORDS),
            ArrayArg::from_raw_parts(status, abi::STATUS_WORDS),
            ArrayArg::from_raw_parts(fault, 1),
            count,
            rows,
            group,
            cube_mapping_launch(&info.cube_count_plan),
            down,
            info.blueprint,
            config,
        );
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(count: usize, down: bool) -> KernelDefinition {
        definition_partition(count, down, 4)
    }
    fn definition_partition(count: usize, down: bool, m_tiles: u32) -> KernelDefinition {
        let props = super::super::tests::properties();
        let (columns, rows) = if down { (640, 2560) } else { (2560, 1280) };
        let elem = ElemType::Float(FloatKind::F16).into();
        let dtypes = MatmulElems::from_globals(&MatmulGlobalElems {
            lhs: elem,
            rhs: elem,
            out: ElemType::Float(FloatKind::F32).into(),
        });
        let problem = MatmulProblem::from_parameters(
            count,
            rows,
            columns,
            Vec::<usize>::new().into(),
            Vec::<usize>::new().into(),
            cubek_std::MatrixLayout::RowMajor,
            cubek_std::MatrixLayout::ColMajor,
            cubek_std::MatrixLayout::RowMajor,
            None,
            None,
            dtypes.as_global_elems(),
            AddressType::U32,
        );
        let scheme = matrix_scheme(MatrixPlan {
            m_tiles,
            required_shared: if m_tiles == 8 { 34816 } else { SHARED_CAP },
        })
        .unwrap();
        let blueprint = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, 16, &problem)
            .partition_buffering(matrix_partition_buffering())
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
            4
        );
        // Metadata-only compiler fixtures. No client/device/request is created.
        let owner = |bytes| Handle::new(cubecl_common::stream_id::StreamId::current(), bytes);
        let lhs = tensor(
            owner((count * columns * 2) as u64),
            [count, columns],
            [columns, 1],
        );
        let rhs = tensor(
            owner((columns * rows * 2) as u64),
            [columns, rows],
            [1, columns],
        );
        let input = (
            ViewArg::new_tensor::<GlobalLayout>(
                lhs.clone().into_tensor_arg(),
                GlobalLayoutLaunch::from_handle(&lhs, 4, blueprint.lhs_global_layout_config()),
            ),
            ViewArg::new_tensor::<GlobalLayout>(
                rhs.clone().into_tensor_arg(),
                GlobalLayoutLaunch::from_handle(&rhs, 4, blueprint.rhs_global_layout_config()),
            ),
        );
        let settings = KernelSettings::default().cube_dim(CubeDim::new_2d(16, 4));
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(settings.clone());
        launcher.with_scope(|scope| scope.device_properties(&props));
        let input = <ExpertInputs as LaunchArg>::register(input, &mut launcher);
        let bytes = if down { abi::PART_BYTES } else { abi::GU_BYTES };
        let mut register = |bytes| {
            <Array<f32> as LaunchArg>::register(
                unsafe { ArrayArg::from_raw_parts(owner(bytes), bytes as usize / 4) },
                &mut launcher,
            )
        };
        let output = register(bytes);
        let map = <Array<u32> as LaunchArg>::register(
            unsafe { ArrayArg::from_raw_parts(owner((abi::MAP_WORDS * 4) as u64), abi::MAP_WORDS) },
            &mut launcher,
        );
        let status = <Array<u32> as LaunchArg>::register(
            unsafe {
                ArrayArg::from_raw_parts(owner((abi::STATUS_WORDS * 4) as u64), abi::STATUS_WORDS)
            },
            &mut launcher,
        );
        let fault = <Array<Atomic<u32>> as LaunchArg>::register(
            unsafe { ArrayArg::from_raw_parts(owner(32768), 1) },
            &mut launcher,
        );
        let count_arg = <usize as LaunchArg>::register(count, &mut launcher);
        let rows_arg = <usize as LaunchArg>::register(rows, &mut launcher);
        let group = <usize as LaunchArg>::register(0usize, &mut launcher);
        let plan = cubek_std::cube_count::CubeCountPlan::new_from_problem(
            cubek_std::cube_count::Count3d {
                x: count.div_ceil(m_tiles as usize * 8) as u32,
                y: rows.div_ceil(64) as u32,
                z: 1,
            },
        );
        let mapping =
            <CubeMapping as LaunchArg>::register(cube_mapping_launch(&plan), &mut launcher);
        let mut b = KernelBuilder::default();
        b.scope.device_properties(&props);
        AddressType::U32.register(&mut b.scope);
        let input = <ExpertInputs as LaunchArg>::expand(&input, &mut b);
        let output = <Array<f32> as LaunchArg>::expand_output(&output, &mut b);
        let map = <Array<u32> as LaunchArg>::expand(&map, &mut b);
        let status = <Array<u32> as LaunchArg>::expand_output(&status, &mut b);
        let fault = <Array<Atomic<u32>> as LaunchArg>::expand_output(&fault, &mut b);
        let count = <usize as LaunchArg>::expand(&count_arg, &mut b);
        let rows = <usize as LaunchArg>::expand(&rows_arg, &mut b);
        let group = <usize as LaunchArg>::expand(&group, &mut b);
        let mapping = <CubeMapping as LaunchArg>::expand(&mapping, &mut b);
        library_expert::expand(
            &mut b.scope,
            input,
            output,
            map,
            status,
            fault,
            count,
            rows,
            group,
            mapping,
            down,
            blueprint,
            config,
        );
        let errors = b.scope.pop_errors();
        assert!(errors.is_empty(), "{errors:?}");
        b.build(settings)
    }

    #[test]
    fn routed_actual_library_compiler_keeps_six_owners_subgroup_topology_and_ieee() {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        let mut observed_shared = std::collections::BTreeSet::new();
        for mask in 0..8 {
            for (count, down) in [(32, false), (33, false), (32, true), (33, true)] {
                let mut options = WgpuCompilationOptions {
                    supports_vulkan: true,
                    ..Default::default()
                };
                options.vulkan.max_spirv_version = (1, 6);
                options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
                options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
                options.vulkan.supports_explicit_smem = mask & 4 != 0;
                let definition = definition(count, down);
                assert_eq!(definition.buffers.len(), 6, "original matrix owners");
                assert!(definition.tensor_maps.is_empty());
                let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                    &mut AutoCompiler::SpirV(Default::default()),
                    definition,
                    &options,
                    ExecutionMode::Unchecked,
                    StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                )
                .unwrap() else {
                    panic!("real SPIR-V required")
                };
                observed_shared.insert(kernel.shared_size);
                assert!(kernel.shared_size <= SHARED_CAP);
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
                let mut builtin = Vec::new();
                let mut product = 0;
                let mut storage_bindings = Vec::new();
                let mut local = None;
                let mut storage_owners = Vec::new();
                let mut descriptor_by_id = std::collections::BTreeMap::new();
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
                    if op == 71 && args.get(1) == Some(&33) {
                        storage_bindings.push(args[2]);
                        descriptor_by_id.insert(args[0], args[2]);
                    }
                    if op == 4459 {
                        product += 1;
                    }
                    if op == 59 && args.get(2) == Some(&12) {
                        storage_owners.push(args[1]);
                    }
                    assert!(
                        op != 71 || args.get(1) != Some(&40),
                        "relaxed FPFastMathMode"
                    );
                    at += n;
                }
                assert_eq!(local, Some(vec![16, 4, 1]));
                assert!(builtin.contains(&40) && builtin.contains(&41));
                assert!(!builtin.iter().any(|b| matches!(b, 27 | 28 | 29)));
                assert!(product >= 4);
                let mut actual_storage: Vec<_> = storage_owners
                    .iter()
                    .map(|id| *descriptor_by_id.get(id).expect("real storage descriptor"))
                    .collect();
                actual_storage.sort_unstable();
                assert!(
                    actual_storage == vec![0, 1, 2, 3, 4, 5]
                        || actual_storage == vec![0, 1, 2, 3, 4, 5, 6],
                    "only six original owners plus actual CubeCL metadata at binding6"
                );
                // Six original storage bindings plus original metadata/scalar
                // uniforms; no seventh matrix scratch/result owner.
                storage_bindings.sort_unstable();
                storage_bindings.dedup();
                assert!(storage_bindings.starts_with(&[0, 1, 2, 3, 4, 5]));
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
                assert!(
                    child.wait().unwrap().success(),
                    "count{count} down{down} mask{mask}"
                );
            }
        }
        println!(
            "compiled_public_cubek_expert tilewise={} modules=32 observed_shared_bytes={observed_shared:?}",
            cfg!(feature = "strata-cubek-tilewise")
        );
    }
    /// Real selected opt-in M64 blueprint, compiled without constructing a GPU.
    #[cfg(feature = "strata-cubek-m64")]
    fn check_m64_modules(masks: std::ops::Range<usize>) {
        use cubecl::ir::{StorageType, UIntKind};
        use cubecl::wgpu::{AutoCompiler, AutoRepresentation};
        use cubecl::{Compiler, WgpuCompilationOptions};
        use std::io::Write;
        for mask in masks {
            for count in [256, 257] {
                let plan = matrix_plan(count, 49152);
                assert_eq!(plan.m_tiles, 8);
                assert_eq!(plan.required_shared, M64_SHARED_CAP);
                let m_tiles = plan.m_tiles;
                for (name, definition) in [
                    (
                        "dense-fp16",
                        super::super::tests::definition_partition::<f16>(count, m_tiles),
                    ),
                    (
                        "dense-bf16",
                        super::super::tests::definition_partition::<bf16>(count, m_tiles),
                    ),
                    ("expert-gu", definition_partition(count, false, m_tiles)),
                    ("expert-down", definition_partition(count, true, m_tiles)),
                ] {
                    let mut options = WgpuCompilationOptions {
                        supports_vulkan: true,
                        ..Default::default()
                    };
                    options.vulkan.max_spirv_version = (1, 6);
                    options.vulkan.supports_uniform_standard_layout = mask & 1 != 0;
                    options.vulkan.supports_uniform_unsized_array = mask & 2 != 0;
                    options.vulkan.supports_explicit_smem = mask & 4 != 0;
                    assert_eq!(
                        definition.buffers.len(),
                        if name.starts_with("dense") { 3 } else { 6 }
                    );
                    let AutoRepresentation::SpirV(kernel) = Compiler::compile(
                        &mut AutoCompiler::SpirV(Default::default()),
                        definition,
                        &options,
                        ExecutionMode::Unchecked,
                        StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                    )
                    .unwrap() else {
                        panic!("actual SPIR-V")
                    };
                    assert_eq!(
                        kernel.shared_size, M64_SHARED_CAP,
                        "actual M64 shared envelope"
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
                    let mut products = 0;
                    let mut local = None;
                    let mut builtin = Vec::new();
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
                    assert_eq!(products, 128);
                    assert!(builtin.contains(&40) && builtin.contains(&41));
                    assert!(!builtin.iter().any(|b| matches!(b, 27 | 28 | 29)));
                    println!(
                        "compiled_public_cubek_blueprint name={name} m_tiles={m_tiles} balanced={} count={count} mask={mask} shared_bytes={} products={products}",
                        cfg!(feature = "strata-cubek-balanced"),
                        kernel.shared_size
                    );
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
                    assert!(child.wait().unwrap().success());
                }
            }
        }
    }
    #[cfg(feature = "strata-cubek-m64")]
    #[test]
    fn candidate_m64_actual_first_layouts() {
        check_m64_modules(0..4);
    }
    #[cfg(feature = "strata-cubek-m64")]
    #[test]
    fn candidate_m64_actual_second_layouts() {
        check_m64_modules(4..8);
    }
}
