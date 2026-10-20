//! Actual collector host-inventory controls; no device/allocation authority or
//! GPU execution is constructed or claimed by these synthetic Handle owners.
use super::*;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::{
    memory_management::ManagedMemoryId,
    public_external::PublicHostCopy,
    server::{
        Binding, NativeStrataExpertBatch, NativeStrataExpertCommand,
        native_expert_batch::{
            NATIVE_EXPERT_BATCH_BYTES, NATIVE_EXPERT_BATCH_COMMANDS,
            NATIVE_EXPERT_BATCH_HOST_BYTES, NativeStrataExpertCopies,
        },
    },
};
type Kernel = Box<dyn CubeTask<AutoCompiler>>;

fn registered_matrix(h: [Handle; 6], params: &[u32], vector: bool) -> KernelArguments {
    use cubecl::{
        compute::KernelLauncher,
        prelude::{AddressType, Array, ArrayArg, Atomic, Const, KernelSettings, LaunchArg, Vector},
    };
    use half::f16;
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(
        KernelSettings::default().address_type(AddressType::U32),
    );
    let [input, weight, output, map, status, fault] = h;
    // Exact original ArrayArg constructors and real LaunchArg registration.
    // Handles describe CPU metadata only; no client/device is constructed.
    for handle in [input, weight] {
        let len = handle.size_in_used() as usize / if vector { 8 } else { 2 };
        let arg = unsafe { ArrayArg::from_raw_parts(handle, len) };
        if vector {
            <Array<Vector<f16, Const<4>>> as LaunchArg>::register::<WgpuRuntime>(
                arg,
                &mut launcher,
            );
        } else {
            <Array<f16> as LaunchArg>::register::<WgpuRuntime>(arg, &mut launcher);
        }
    }
    let f32_len = output.size_in_used() as usize / 4;
    <Array<f32> as LaunchArg>::register::<WgpuRuntime>(
        unsafe { ArrayArg::from_raw_parts(output, f32_len) },
        &mut launcher,
    );
    for handle in [map, status] {
        let len = handle.size_in_used() as usize / 4;
        <Array<u32> as LaunchArg>::register::<WgpuRuntime>(
            unsafe { ArrayArg::from_raw_parts(handle, len) },
            &mut launcher,
        );
    }
    <Array<Atomic<u32>> as LaunchArg>::register::<WgpuRuntime>(
        unsafe { ArrayArg::from_raw_parts(fault, 1) },
        &mut launcher,
    );
    for &param in params {
        if params.len() == 5 {
            <usize as LaunchArg>::register::<WgpuRuntime>(param as usize, &mut launcher);
        } else {
            <u32 as LaunchArg>::register::<WgpuRuntime>(param, &mut launcher);
        }
    }
    launcher.into_bindings()
}

#[test]
fn real_array_and_scalar_registration_matches_the_complete_logical_grammar() {
    use cubecl_runtime::server::native_expert_grammar::NativeStrataExpertGrammar;
    let stream = StreamId::current();
    let h = [
        80 * 1024 * 1024,
        80 * 1024 * 1024,
        groups::GU_BYTES,
        groups::MAP_WORDS as u64 * 4,
        groups::STATUS_WORDS as u64 * 4,
        4,
        groups::PART_BYTES,
    ]
    .map(|bytes| {
        Handle::new(stream, bytes + 512)
            .offset_start(256)
            .offset_end(256)
    });
    for mode_and_class in 0..8 {
        let mode = mode_and_class % 4;
        let resident = mode_and_class >= 4;
        let wide = mode == 3;
        let vector = mode != 0;
        let members = if mode == 2 { 17 } else { 257 };
        let mut grammar = NativeStrataExpertGrammar::default();
        let mut params = [0u32; 16];
        params[..3].copy_from_slice(&[members, 2560, 1280]);
        let gather = KernelArguments::new()
            .with_buffers(
                [h[3].clone(), h[0].clone(), h[0].clone(), h[5].clone()]
                    .into_iter()
                    .map(Handle::binding)
                    .collect(),
            )
            .with_info(MetadataBindingInfo::custom(
                params
                    .chunks_exact(2)
                    .map(|v| u64::from(v[0]) | (u64::from(v[1]) << 32))
                    .collect(),
            ));
        grammar
            .launch(
                Op::Gather.name(),
                false,
                &CubeCount::Static(members * 5, 1, 1),
                &gather,
            )
            .unwrap();
        for down in [false, true] {
            let dtype = if down { q::IQ4_NL } else { q::IQ3_S };
            let op = if resident {
                Op::ResidentDequant { dtype, down }
            } else {
                Op::Dequant { dtype, down }
            };
            grammar
                .launch(
                    op.name(),
                    op.native_release_after(),
                    &CubeCount::Static(1, 1, 1),
                    &KernelArguments::new(),
                )
                .unwrap();
        }
        for down in [false, true] {
            let columns = if down { 640 } else { 2560 };
            let rows = if down { 2560 } else { 1280 };
            let original = [
                h[0].clone(),
                h[1].clone(),
                h[if down { 6 } else { 2 }].clone(),
                h[3].clone(),
                h[4].clone(),
                h[5].clone(),
            ];
            let fields = if wide {
                vec![columns, rows, members, 0, 0, 0, 256]
            } else {
                vec![columns, rows, members, 0, 0]
            };
            let args = registered_matrix(original.clone(), &fields, vector);
            assert_eq!(args.info.data.len(), if wide { 10 } else { 9 });
            let words = args
                .info
                .data
                .iter()
                .flat_map(|n| [*n as u32, (*n >> 32) as u32])
                .collect::<Vec<_>>();
            let first = if wide { 8 } else { 6 };
            assert_eq!(
                u64::from(words[first]),
                original[0].size_in_used() / if vector { 8 } else { 2 }
            );
            assert_eq!(
                u64::from(words[first + 6]),
                original[0].size_in_used() / if vector { 32 } else { 2 }
            );
            let name = if wide {
                if down {
                    "probe::native_expert_wide::expert_wide_down::ExpertWideDown<Runtime>"
                } else {
                    "probe::native_expert_wide::expert_wide_gu::ExpertWideGu<Runtime>"
                }
            } else {
                crate::strata_matrix::expert_metadata_test_names()[mode]
            };
            let tile = if mode == 0 {
                8
            } else if mode == 1 {
                32
            } else {
                16
            };
            grammar
                .launch(
                    name,
                    false,
                    &if wide {
                        CubeCount::Static(rows / 128, 2, 1)
                    } else {
                        CubeCount::Static(
                            rows / if mode == 0 { 16 } else { 64 },
                            members.div_ceil(tile),
                            1,
                        )
                    },
                    &args,
                )
                .unwrap();
            if wide {
                let mut tail = original.clone();
                tail[0] = tail[0]
                    .clone()
                    .offset_start(256 * u64::from(columns) * 2)
                    .offset_end(
                        original[0].size_in_used() - u64::from(members) * u64::from(columns) * 2,
                    );
                let args = registered_matrix(tail, &[columns, rows, members, 0, 0, 256, 1], true);
                let name = if down {
                    "probe::native_expert_wide::expert_tail_down::ExpertTailDown<Runtime>"
                } else {
                    "probe::native_expert_wide::expert_tail_gu::ExpertTailGu<Runtime>"
                };
                grammar
                    .launch(name, false, &CubeCount::Static(rows / 64, 1, 1), &args)
                    .unwrap();
            }
            if !down {
                grammar
                    .launch(
                        Op::Activate.name(),
                        false,
                        &CubeCount::Static(1, 1, 1),
                        &KernelArguments::new(),
                    )
                    .unwrap();
            }
        }
        let receipt = grammar.finish().unwrap();
        assert_eq!(receipt.logical_phases, 6);
        assert_eq!(receipt.launches, if wide { 8 } else { 6 });
    }
}

fn command(
    handles: &[Handle],
    buffers: usize,
    capacity: usize,
    info_bytes: usize,
    reads: bool,
) -> NativeStrataExpertCommand<Kernel> {
    let mut bindings = Vec::with_capacity(capacity);
    for i in 0..buffers {
        bindings.push(handles[i % handles.len()].clone().binding());
    }
    let args = KernelArguments::new()
        .with_buffers(bindings)
        .with_read_dependencies(if reads {
            handles.iter().cloned().map(Handle::binding).collect()
        } else {
            Vec::new()
        })
        .with_info(MetadataBindingInfo::custom(vec![0; info_bytes / 8]));
    NativeStrataExpertCommand::Launch {
        kernel: Some(Box::new(Op::Gather)),
        count: CubeCount::Static(1, 1, 1),
        args,
        mode: ExecutionMode::Unchecked,
    }
}

#[test]
fn maximum_route_conserving_shadow_inventory_fits_the_original_byte_reservations() {
    let rows = groups::ROUTE_CAPACITY;
    assert_eq!(rows, 4096);
    let mut raw = vec![0u32; groups::LOGICAL + rows * 10];
    raw[..4].copy_from_slice(&[rows as u32, 0, rows as u32, 0]);
    let mut rank = vec![0usize; rows];
    let mut cursor = 0;
    for group in 0..512 {
        let members = if group < 158 { 257 } else { 1 };
        for _ in 0..members {
            let row = cursor % rows;
            let route = row * 10 + rank[row];
            rank[row] += 1;
            raw[groups::LOGICAL + route] = group as u32;
            raw[1540 + route] = group as u32;
            cursor += 1;
        }
    }
    assert_eq!(cursor, rows * 10);
    assert!(rank.iter().all(|v| *v == 10));
    let map = groups::group(&raw, rows).unwrap();
    assert_eq!(groups::validate(&map, rows).unwrap(), 512);
    assert_eq!(
        (0..512).map(|g| map[516 + g] as usize).sum::<usize>(),
        rows * 10
    );
    assert_eq!((0..512).filter(|&g| map[516 + g] >= 257).count(), 158);
    // Valid loan mode may retain43 GPU+12 HOST+30 shadow source views. The
    // original exact-size collection retains the full85-view list per DQ.
    let handles = (0..85)
        .map(|_| Handle::new(StreamId::current(), 512))
        .collect::<Vec<_>>();
    let destination = Handle::new(StreamId::current(), 512);
    let mut batch = NativeStrataExpertBatch::<Kernel>::new();
    let mut launches = 0;
    for group in 0..512 {
        let mut copies = Vec::with_capacity(4);
        for role in 0..3 {
            copies.push(PublicHostCopy {
                source: handles[role].clone().binding(),
                source_first: 0,
                destination: destination.clone().binding(),
                destination_first: role as u64 * 64,
                bytes: 64,
            });
        }
        batch
            .push(NativeStrataExpertCommand::HostCopies(
                NativeStrataExpertCopies(copies),
            ))
            .unwrap();
        // Original live Vec capacities, not merely their smaller lengths.
        for (buffers, capacity, info, reads) in [
            (4, 4, 64, false),
            (8, 8, 64, true),
            (7, 8, 64, true),
            (6, 8, if group < 158 { 80 } else { 72 }, false),
            (5, 5, 64, false),
            (6, 8, if group < 158 { 80 } else { 72 }, false),
        ] {
            batch
                .push(command(&handles, buffers, capacity, info, reads))
                .unwrap();
            launches += 1;
        }
        if group < 158 {
            for _ in 0..2 {
                batch.push(command(&handles, 6, 8, 80, false)).unwrap();
                launches += 1;
            }
        }
    }
    assert_eq!(launches, 512 * 6 + 158 * 2);
    assert_eq!(launches + 512 * 3, 4924);
    assert_eq!(NATIVE_EXPERT_BATCH_COMMANDS, 5632);
    let bytes = batch.metadata_bytes().unwrap();
    assert!(bytes <= NATIVE_EXPERT_BATCH_BYTES);
    type Key = (ManagedMemoryId, Option<u64>, Option<u64>, StreamId, u64);
    let backend = crate::strata_read_bundle_controls::host_bytes::<
        Binding,
        cubecl::wgpu::WgpuResource,
        Key,
        MetadataBindingInfo,
        std::sync::Arc<wgpu::ComputePipeline>,
    >(bytes, batch.commands.len())
    .unwrap();
    let worst = crate::strata_read_bundle_controls::host_bytes::<
        Binding,
        cubecl::wgpu::WgpuResource,
        Key,
        MetadataBindingInfo,
        std::sync::Arc<wgpu::ComputePipeline>,
    >(NATIVE_EXPERT_BATCH_BYTES, NATIVE_EXPERT_BATCH_COMMANDS)
    .unwrap();
    assert!(worst + super::miss_ring::HOST_BYTES <= NATIVE_EXPERT_BATCH_HOST_BYTES);
    assert!(backend <= NATIVE_EXPERT_BATCH_HOST_BYTES);
    assert!(backend + super::miss_ring::HOST_BYTES <= NATIVE_EXPERT_BATCH_HOST_BYTES);
    eprintln!(
        "native_expert_public_inventory binding_bytes={} command_bytes={} tape_nodes={} source_views={} actual_commands={} actual_nodes={} inventory_bytes={} inventory_headroom={} backend_charge={} parent_host_bytes={}",
        size_of::<Binding>(),
        size_of::<NativeStrataExpertCommand<Kernel>>(),
        NATIVE_EXPERT_BATCH_COMMANDS,
        handles.len(),
        batch.commands.len(),
        launches + 512 * 3,
        bytes,
        NATIVE_EXPERT_BATCH_BYTES - bytes,
        backend,
        NATIVE_EXPERT_BATCH_HOST_BYTES
    );
    // Complete actual fixed tape remains bounded. Minimal extra commands test
    // the independent node limit, never shader acceptance or valid semantics.
    for _ in launches + 512 * 3..NATIVE_EXPERT_BATCH_COMMANDS {
        batch.push(command(&handles, 1, 1, 8, false)).unwrap();
    }
    assert!(batch.push(command(&handles, 1, 1, 8, false)).is_err());
    let mut oversized = command(&handles, 1, 1, 8, false);
    if let NativeStrataExpertCommand::Launch { args, .. } = &mut oversized {
        args.read_dependencies =
            Vec::with_capacity(NATIVE_EXPERT_BATCH_BYTES / size_of::<Binding>());
    }
    assert!(NativeStrataExpertBatch::new().push(oversized).is_err());
}
