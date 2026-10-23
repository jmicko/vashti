//! Genuine launcher/grammar metadata controls. Handles are CPU-only metadata,
//! never device/source authority, request scopes or GPU correctness evidence.
use super::*;
use cubecl::{
    compute::KernelLauncher,
    prelude::{AddressType, Array, ArrayArg, Atomic, Const, KernelSettings, LaunchArg, Vector},
};
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::server::{CubeCount, native_expert_grammar::NativeStrataExpertGrammar};
use half::f16;
struct Entry {
    name: &'static str,
    release: bool,
    grid: CubeCount,
    args: KernelArguments,
}
fn arguments(h: Vec<Handle>, words: &[u32; 48], matrix: bool) -> KernelArguments {
    if !matrix {
        return KernelArguments::new()
            .with_buffers(h.into_iter().map(Handle::binding).collect())
            .with_info(MetadataBindingInfo::custom(
                words
                    .chunks_exact(2)
                    .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                    .collect(),
            ));
    }
    let mut launch = KernelLauncher::<WgpuRuntime>::new(
        KernelSettings::default().address_type(AddressType::U32),
    );
    for (i, handle) in h.into_iter().enumerate() {
        let words = handle.size_in_used() as usize / if i < 2 { 2 } else { 4 };
        let arg = unsafe { ArrayArg::from_raw_parts(handle, words) };
        match i {
            0 | 1 => {
                <Array<Vector<f16, Const<4>>> as LaunchArg>::register::<WgpuRuntime>(
                    arg,
                    &mut launch,
                );
            }
            2 => {
                <Array<f32> as LaunchArg>::register::<WgpuRuntime>(arg, &mut launch);
            }
            3 | 4 => {
                <Array<u32> as LaunchArg>::register::<WgpuRuntime>(arg, &mut launch);
            }
            _ => {
                <Array<Atomic<u32>> as LaunchArg>::register::<WgpuRuntime>(arg, &mut launch);
            }
        }
    }
    for &word in words {
        <u32 as LaunchArg>::register::<WgpuRuntime>(word, &mut launch);
    }
    launch.into_bindings()
}
fn program() -> Vec<Entry> {
    let stream = StreamId::current();
    let owner = |n| Handle::new(stream, n);
    let parent = owner(storage::WORKING_BYTES);
    let x = view(&parent, 0, X_BYTES).unwrap();
    let h = view(&parent, X_BYTES, H_BYTES).unwrap();
    let image = owner(storage::WORKING_BYTES);
    let raw = owner(allocation::RAW_RING_BYTES);
    let input = owner(storage::WORKING_BYTES);
    let gu = owner(groups::GU_BYTES);
    let part = owner(groups::PART_BYTES);
    let map = owner((groups::MAP_WORDS * 4) as u64);
    let status = owner((groups::STATUS_WORDS * 4) as u64);
    let fault = owner(4);
    let gate = owner(400);
    let up = owner(400);
    let down = owner(400);
    let table = owner(32768);
    let addresses = owner(32768);
    let mut w = [0u32; 48];
    w[0] = 32;
    w[1] = 2560;
    w[2] = 1280;
    w[6] = 100;
    w[7] = 100;
    w[8] = 300;
    w[9] = 2;
    w[10] = 17;
    w[16..20].copy_from_slice(&[0, 3, 0, 8]);
    w[20..24].copy_from_slice(&[1, 17, 8, 8]);
    let mut entries = Vec::new();
    let mut push = |name, release, x, y, owners: Vec<Handle>, words: &[u32; 48], matrix| {
        entries.push(Entry {
            name,
            release,
            grid: CubeCount::Static(x, y, 2),
            args: arguments(owners, words, matrix),
        });
    };
    push(
        CohortOp::Gather.name(),
        false,
        85,
        1,
        vec![map.clone(), input, x.clone(), fault.clone()],
        &w,
        false,
    );
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = 1;
        w[5] = if d { 2 } else { 0 };
        let mut owners = if d {
            vec![down.clone()]
        } else {
            vec![gate.clone(), up.clone()]
        };
        owners.extend([
            map.clone(),
            raw.clone(),
            image.clone(),
            fault.clone(),
            table.clone(),
            addresses.clone(),
        ]);
        let mut args = arguments(owners, &w, false);
        args.read_dependencies.push(gate.clone().binding());
        entries.push(Entry {
            name: CohortOp::Image {
                dtype: if d { q::Q2_0 } else { q::IQ3_S },
                down: d,
            }
            .original_name(),
            release: d,
            grid: CubeCount::Static(if d { 200 } else { 400 }, 1, 2),
            args,
        });
    }
    // GU is1280*2560/32/256=400, DOWN2560*640/32/256=200.
    let matrix_name = core::any::type_name::<
        crate::strata_matrix::cohorts::expert_cohort::ExpertCohort<WgpuRuntime>,
    >();
    for d in [false, true] {
        w[1] = if d { 640 } else { 2560 };
        w[2] = if d { 2560 } else { 1280 };
        w[3] = 0;
        w[5] = if d { 2 } else { 0 };
        for tile in [8, 16] {
            w[11] = tile;
            let owners = vec![
                if d { h.clone() } else { x.clone() },
                image.clone(),
                if d { part.clone() } else { gu.clone() },
                map.clone(),
                status.clone(),
                fault.clone(),
            ];
            entries.push(Entry {
                name: matrix_name,
                release: false,
                grid: CubeCount::Static(w[2] / 64, if tile == 8 { 1 } else { 2 }, 2),
                args: arguments(owners, &w, true),
            });
        }
        if !d {
            w[1] = 2560;
            w[2] = 1280;
            w[5] = 0;
            w[11] = 0;
            entries.push(Entry {
                name: CohortOp::Activate.name(),
                release: false,
                grid: CubeCount::Static((17u32 * 320).div_ceil(256), 1, 2),
                args: arguments(
                    vec![
                        map.clone(),
                        gu.clone(),
                        h.clone(),
                        status.clone(),
                        fault.clone(),
                    ],
                    &w,
                    false,
                ),
            });
        }
    }
    entries
}
#[test]
fn real_registered_cohort_info_and_all_actual_owners_pass_distinct_complete_grammar() {
    let mut grammar = NativeStrataExpertGrammar::default();
    for e in program() {
        grammar.launch(e.name, e.release, &e.grid, &e.args).unwrap();
    }
    let r = grammar.finish().unwrap();
    assert_eq!(
        (
            r.launches,
            r.logical_phases,
            r.matrix_pieces,
            r.cohort_groups,
            r.cohort_launches,
            r.cohort_batches
        ),
        (8, 12, 4, 2, 8, 1)
    );
}
#[test]
fn cohort_actual_grammar_rejects_phase_owner_grid_info_and_overflow_mutations() {
    for mutation in 0..16 {
        let mut p = program();
        match mutation {
            0 => p[0].release = true,
            1 => p[2].release = false,
            2 => p[1].name = "StrataPromptExpertF16Image",
            3 => p[3].grid = CubeCount::Static(20, 2, 2),
            4 => {
                p[0].args.info.data[0] = u64::MAX;
            }
            5 => {
                p[0].args.info.data[2] = u64::MAX;
            }
            6 => {
                p[0].args.info.data[3] = u64::MAX;
            }
            7 => {
                p[0].args.info.data[9] = u64::MAX;
            }
            8 => {
                p[4].args.info.data[10] ^= 1;
            }
            9 => {
                p[1].args.buffers.swap(2, 3);
            }
            10 => {
                p[5].args.buffers[2] = p[0].args.buffers[2].clone();
            }
            11 => {
                p[6].args.buffers[4] = p[0].args.buffers[3].clone();
            }
            12 => {
                p[3].args.info.data[24] ^= 1;
            }
            13 => {
                p[0].args.info.data.push(0);
            }
            14 => {
                p[0].args.buffers[2] = Handle::new(StreamId::current(), H_BYTES).binding();
            }
            _ => {
                p[0].args.info.data[8] = 0;
            }
        }
        let mut grammar = NativeStrataExpertGrammar::default();
        let accepted = p
            .iter()
            .all(|e| grammar.launch(e.name, e.release, &e.grid, &e.args).is_ok())
            && grammar.finish().is_ok();
        assert!(!accepted, "mutation {mutation}");
    }
}
#[test]
fn cohort_incomplete_grammar_never_accepts_or_reports_finished_logical_work() {
    for keep in 0..8 {
        let mut grammar = NativeStrataExpertGrammar::default();
        for e in program().into_iter().take(keep) {
            grammar.launch(e.name, e.release, &e.grid, &e.args).unwrap();
        }
        assert!(grammar.finish().is_err());
    }
}

#[test]
fn actual_unit_end_factory_requires_the_complete_cohort_and_forwards_original_task() {
    use cubecl::ir::{ElemType, StorageType, UIntKind};
    use cubecl_runtime::{
        compiler::{CompilationError, Compiler, CubeTask},
        id::KernelId,
        kernel::{CompiledKernel, KernelMetadata},
        server::{CubeDim, ExecutionMode, native_expert_batch::NativeStrataExpertUnitEnd},
    };
    // CPU metadata/forwarding fixture only. Its deterministic compilation result
    // is checked through the real production wrapper; it grants no GPU proof.
    struct OriginalTask(&'static str, bool);
    impl KernelMetadata for OriginalTask {
        fn name(&self)->&'static str {self.0}
        fn id(&self)->KernelId {KernelId::new::<Self>()}
        fn address_type(&self)->StorageType {StorageType::Scalar(ElemType::UInt(UIntKind::U32))}
        fn native_release_after(&self)->bool {self.1}
    }
    impl CubeTask<AutoCompiler> for OriginalTask {
        fn compile(&self,_compiler:&mut AutoCompiler,_options:&<AutoCompiler as Compiler>::CompilationOptions,
            mode:ExecutionMode,address:StorageType)->std::result::Result<CompiledKernel<AutoCompiler>,CompilationError> {
            assert!(matches!(mode,ExecutionMode::Unchecked));
            assert_eq!(address,self.address_type());
            Ok(CompiledKernel {entrypoint_name:"original_entry".into(),debug_name:Some(self.0),
                source:"original compilation forwarded unchanged".into(),repr:None,
                cube_dim:CubeDim::new_1d(256),debug_info:None})
        }
    }
    let total=program().len();
    for stop in 0..total {
        let mut grammar=NativeStrataExpertGrammar::default();
        for (i,e) in program().into_iter().enumerate() {
            if i==stop {
                let original=OriginalTask(e.name,e.release);
                let id=original.id();let address=original.address_type();
                let result=NativeStrataExpertUnitEnd::validate(Box::new(original),&mut grammar,&e.grid,&e.args);
                if stop+1!=total {assert!(result.is_err(),"early physical launch {stop}");}
                else {
                    let task=result.unwrap();
                    assert_eq!(task.name(),e.name);assert_eq!(task.id(),id);
                    assert_eq!(task.address_type(),address);assert!(task.native_release_after());
                    assert_ne!(task.id().native_release_boundary(true),id.native_release_boundary(false));
                    let mut compiler=AutoCompiler::SpirV(Default::default());
                    let actual=task.compile(&mut compiler,&Default::default(),ExecutionMode::Unchecked,address).unwrap();
                    assert_eq!(actual.entrypoint_name,"original_entry");assert_eq!(actual.debug_name,Some(e.name));
                    assert_eq!(actual.source,"original compilation forwarded unchanged");
                    assert!(actual.repr.is_none());assert_eq!(actual.cube_dim,CubeDim::new_1d(256));
                    assert_eq!(grammar.finish().unwrap().cohort_groups,2);
                }
                break;
            }
            grammar.launch(e.name,e.release,&e.grid,&e.args).unwrap();
        }
    }
}
