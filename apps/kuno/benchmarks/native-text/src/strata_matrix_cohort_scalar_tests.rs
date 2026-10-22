//! Actual old generated registration versus the production bulk registration.
//! Handles here are metadata only; these controls create no client/device,
//! execution authority, request, source lease or GPU correctness evidence.
use super::*;
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::server::{Binding, KernelArguments};

fn arguments(handles: [Handle; 6]) -> [ArrayArg<WgpuRuntime>; 6] {
    let mut index = 0;
    handles.map(|handle| {
        let unit = if index < 2 { 2 } else { 4 };
        index += 1;
        let length = handle.size_in_used() as usize / unit;
        // Same original scalar-half and scalar-u32 lengths; actual registration
        // supplies vectorization metadata, never a hand-written expected ABI.
        unsafe { ArrayArg::from_raw_parts(handle, length) }
    })
}
fn generated_arguments(handles: [Handle; 6], words: &[u32; 48]) -> KernelArguments {
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(
        KernelSettings::default()
            .cube_dim(CubeDim::new_1d(64))
            .address_type(AddressType::U32),
    );
    // Independent copy of the ORIGINAL generated registration sequence, using
    // actual CubeCL LaunchArg methods. Do not replace this with bulk helpers.
    for (index, arg) in arguments(handles).into_iter().enumerate() {
        match index {
            0 | 1 => {
                <Array<Vector<f16, Const<4>>> as LaunchArg>::register(arg, &mut launcher);
            }
            2 => {
                <Array<f32> as LaunchArg>::register(arg, &mut launcher);
            }
            3 | 4 => {
                <Array<u32> as LaunchArg>::register(arg, &mut launcher);
            }
            _ => {
                <Array<Atomic<u32>> as LaunchArg>::register(arg, &mut launcher);
            }
        }
    }
    for &word in words {
        <u32 as LaunchArg>::register(word, &mut launcher);
    }
    launcher.into_bindings()
}
fn bulk_arguments(handles: [Handle; 6], words: &[u32; 48]) -> KernelArguments {
    let mut launcher = KernelLauncher::<WgpuRuntime>::new(matrix_settings());
    let compilation = register_arrays(&mut launcher, arguments(handles));
    assert!(compilation.iter().all(|arg| arg.inplace.is_none()));
    register_words(&mut launcher, words);
    launcher.into_bindings()
}
fn same_binding(actual: &Binding, original: &Binding) {
    assert_eq!(
        actual.memory.allocation_id(),
        original.memory.allocation_id()
    );
    assert_eq!(actual.offset_start, original.offset_start);
    assert_eq!(actual.offset_end, original.offset_end);
    assert_eq!(actual.stream, original.stream);
    assert_eq!(actual.size, original.size);
}
fn fixture(down: bool, offset: u64) -> [Handle; 6] {
    let stream = StreamId::current();
    [
        if down { 5 } else { 20 } * 1024 * 1024,
        80 * 1024 * 1024,
        if down {
            4096 * 10 * 2560 * 4
        } else {
            4096 * 1280 * 4
        },
        65536 * 4,
        131072 * 4,
        4,
    ]
    .map(|bytes| {
        if offset == 0 {
            Handle::new(stream, bytes)
        } else {
            Handle::new(stream, bytes + 2 * offset)
                .offset_start(offset)
                .offset_end(offset)
        }
    })
}
#[test]
fn bulk_words_match_original_generated_info_and_actual_owner_views_for_every_family() {
    assert_eq!(
        matrix_settings(),
        KernelSettings::default()
            .cube_dim(CubeDim::new_1d(64))
            .address_type(AddressType::U32)
    );
    for down in [false, true] {
        for tile in [8u32, 16] {
            for mixed in [0u32, 1] {
                for offset in [0, 256] {
                    let handles = fixture(down, offset);
                    let mut words = zeroize::Zeroizing::new([0u32; 48]);
                    words[0] = 4096;
                    words[1] = if down { 640 } else { 2560 };
                    words[2] = if down { 2560 } else { 1280 };
                    words[4] = 400;
                    words[5] = if down { 2 } else { 0 };
                    words[6] = 100;
                    words[7] = 100;
                    words[8] = 300;
                    words[9] = 8;
                    words[10] = 31;
                    words[11] = tile;
                    words[12] = mixed;
                    let mut prefix = 0;
                    for (index, members) in [1u32, 3, 7, 8, 9, 16, 17, 31].into_iter().enumerate() {
                        words[16 + index * 4..20 + index * 4].copy_from_slice(&[
                            400 + index as u32,
                            members,
                            prefix,
                            index as u32,
                        ]);
                        prefix += members.div_ceil(if members <= 8 { 8 } else { 16 })
                            * if members <= 8 { 8 } else { 16 };
                    }
                    crate::strata_prompt::experts::cohorts::validate_words(&words).unwrap();
                    let original = generated_arguments(handles.clone(), &words);
                    let bulk = bulk_arguments(handles.clone(), &words);
                    assert_eq!(bulk.info.data, original.info.data);
                    assert_eq!(bulk.info.data.len(), 30);
                    assert_eq!(
                        bulk.info.dynamic_metadata_offset,
                        original.info.dynamic_metadata_offset
                    );
                    assert_eq!(bulk.info.dynamic_metadata_offset, 30);
                    assert!(bulk.info.native_parameters.is_empty());
                    assert!(original.info.native_parameters.is_empty());
                    assert!(
                        bulk.read_dependencies.is_empty() && original.read_dependencies.is_empty()
                    );
                    assert!(bulk.tensor_maps.is_empty() && original.tensor_maps.is_empty());
                    assert_eq!(bulk.buffers.len(), 6);
                    for ((actual, prior), owner) in
                        bulk.buffers.iter().zip(&original.buffers).zip(handles)
                    {
                        same_binding(actual, prior);
                        same_binding(actual, &owner.binding());
                    }
                    let unpacked: Vec<u32> = bulk
                        .info
                        .data
                        .iter()
                        .flat_map(|word| {
                            let b = word.to_ne_bytes();
                            [
                                u32::from_ne_bytes(b[..4].try_into().unwrap()),
                                u32::from_ne_bytes(b[4..].try_into().unwrap()),
                            ]
                        })
                        .collect();
                    assert_eq!(&unpacked[..48], words.as_slice());
                    let lengths: Vec<u32> = bulk
                        .buffers
                        .iter()
                        .enumerate()
                        .map(|(index, owner)| {
                            (owner.size_in_used() / if index < 2 { 8 } else { 4 }) as u32
                        })
                        .collect();
                    assert_eq!(&unpacked[48..54], lengths.as_slice());
                    assert_eq!(&unpacked[54..60], lengths.as_slice());
                }
            }
        }
    }
}
#[test]
fn abandoned_bulk_scalar_registration_releases_original_tls_lease_on_unwind() {
    // Existing real InfoLease/ScalarBuilder guards own full-capacity wiping.
    // This control exercises those guards through the NEW bulk entry on panic,
    // then verifies a clean independent registration is still byte-equivalent.
    let interrupted = std::panic::catch_unwind(|| {
        let mut launcher = KernelLauncher::<WgpuRuntime>::new(matrix_settings());
        let words = zeroize::Zeroizing::new([0xa13579bdu32; 48]);
        register_words(&mut launcher, &words);
        panic!("fixed public bulk scalar registration interruption");
    });
    assert!(interrupted.is_err());
    let words = zeroize::Zeroizing::new([0u32; 48]);
    let handles = fixture(false, 256);
    let original = generated_arguments(handles.clone(), &words);
    let bulk = bulk_arguments(handles, &words);
    assert_eq!(bulk.info.data, original.info.data);
    assert!(bulk.info.data[..24].iter().all(|word| *word == 0));
}
