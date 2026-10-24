//! Fixed-public host controls for the actual native grammar and original views.
use super::*;
use crate::server::Handle;
use alloc::{vec, vec::Vec};

const ORIGINAL: &str = "probe::strata_matrix::native::expert_macro::ExpertMacro<Runtime>";
const WIDE_GU: &str =
    "probe::strata_matrix::native_expert_wide::expert_wide_gu::ExpertWideGu<Runtime>";
const WIDE_DOWN: &str =
    "probe::strata_matrix::native_expert_wide::expert_wide_down::ExpertWideDown<Runtime>";
const TAIL_GU: &str =
    "probe::strata_matrix::native_expert_wide::expert_tail_gu::ExpertTailGu<Runtime>";
const TAIL_DOWN: &str =
    "probe::strata_matrix::native_expert_wide::expert_tail_down::ExpertTailDown<Runtime>";
fn packed(words: &[u32]) -> Vec<u64> {
    words
        .chunks_exact(2)
        .map(|v| {
            let mut bytes = [0; 8];
            bytes[..4].copy_from_slice(&v[0].to_ne_bytes());
            bytes[4..].copy_from_slice(&v[1].to_ne_bytes());
            u64::from_ne_bytes(bytes)
        })
        .collect()
}
fn set_word(info: &mut MetadataBindingInfo, index: usize, value: u32) {
    let mut bytes = info.data[index / 2].to_ne_bytes();
    let offset = index % 2 * 4;
    bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
    info.data[index / 2] = u64::from_ne_bytes(bytes);
}
fn owners() -> [Handle; 7] {
    [
        80 * 1024 * 1024,
        80 * 1024 * 1024,
        GU_BYTES,
        MAP_BYTES,
        STATUS_BYTES,
        4,
        PART_BYTES,
    ]
    .map(|bytes| {
        Handle::new(StreamId::current(), bytes + 512)
            .offset_start(256)
            .offset_end(256)
    })
}
fn matrix_args(
    h: &[Handle; 7],
    members: u32,
    down: bool,
    wide: bool,
    tail: bool,
) -> KernelArguments {
    let columns = if down { 640 } else { 2560 };
    let rows = if down { 2560 } else { 1280 };
    let prefix = members / 128 * 128;
    let base = if tail { prefix } else { 0 };
    let piece = if tail {
        members - prefix
    } else if wide {
        prefix
    } else {
        members
    };
    let input = if tail {
        h[0].clone()
            .offset_start(u64::from(base) * columns * 2)
            .offset_end(h[0].size_in_used() - u64::from(members) * columns * 2)
    } else {
        h[0].clone()
    };
    let handles = [
        input,
        h[1].clone(),
        h[if down { 6 } else { 2 }].clone(),
        h[3].clone(),
        h[4].clone(),
        h[5].clone(),
    ];
    let mut words = vec![columns as u32, rows, members, 0, 0];
    if wide {
        words.extend([base, piece, 0]);
    } else {
        words.push(0);
    }
    let lengths = handles
        .iter()
        .enumerate()
        .map(|(i, h)| (h.size_in_used() / if i < 2 { 8 } else { 4 }) as u32)
        .collect::<Vec<_>>();
    words.extend(&lengths);
    words.extend(
        lengths
            .iter()
            .enumerate()
            .map(|(i, n)| if i < 2 { n / 4 } else { *n }),
    );
    let info = MetadataBindingInfo::new(packed(&words), words.len() / 2);
    KernelArguments::new()
        .with_buffers(handles.into_iter().map(Handle::binding).collect())
        .with_info(info)
}
fn begin(h: &[Handle; 7], members: u32) -> NativeStrataExpertGrammar {
    begin_images(h, members, false)
}
fn begin_gather(h: &[Handle; 7], members: u32) -> NativeStrataExpertGrammar {
    let mut grammar = NativeStrataExpertGrammar::default();
    let mut words = [0; 16];
    words[..3].copy_from_slice(&[members, 2560, 1280]);
    let gather = KernelArguments::new()
        .with_buffers(
            [h[3].clone(), h[0].clone(), h[0].clone(), h[5].clone()]
                .into_iter()
                .map(Handle::binding)
                .collect(),
        )
        .with_info(MetadataBindingInfo::custom(packed(&words)));
    grammar
        .launch(
            "StrataPromptExpertGather",
            false,
            &CubeCount::Static(members * 5, 1, 1),
            &gather,
        )
        .unwrap();
    grammar
}
fn begin_images(h: &[Handle; 7], members: u32, resident: bool) -> NativeStrataExpertGrammar {
    let mut grammar = begin_gather(h, members);
    for down in [false, true] {
        grammar
            .launch(
                if resident {
                    if down {
                        "StrataPromptExpertResidentDownImage"
                    } else {
                        "StrataPromptExpertResidentGuImage"
                    }
                } else {
                    "StrataPromptExpertF16Image"
                },
                down && !resident,
                &CubeCount::Static(1, 1, 1),
                &KernelArguments::new(),
            )
            .unwrap();
    }
    grammar
}
fn product(
    grammar: &mut NativeStrataExpertGrammar,
    h: &[Handle; 7],
    members: u32,
    down: bool,
    wide: bool,
) {
    let rows = if down { 2560 } else { 1280 };
    let prefix = members / 128 * 128;
    let args = matrix_args(h, members, down, wide, false);
    grammar
        .launch(
            if wide {
                if down { WIDE_DOWN } else { WIDE_GU }
            } else {
                ORIGINAL
            },
            false,
            &if wide {
                CubeCount::Static(rows / 128, prefix / 128, 1)
            } else {
                CubeCount::Static(rows / 64, members.div_ceil(32), 1)
            },
            &args,
        )
        .unwrap();
    if wide && prefix < members {
        let remain = members - prefix;
        let tile = if remain <= 8 {
            8
        } else if remain < 32 {
            16
        } else {
            32
        };
        grammar
            .launch(
                if down { TAIL_DOWN } else { TAIL_GU },
                false,
                &CubeCount::Static(rows / 64, remain.div_ceil(tile), 1),
                &matrix_args(h, members, down, true, true),
            )
            .unwrap();
    }
}
fn activate(grammar: &mut NativeStrataExpertGrammar) {
    grammar
        .launch(
            "StrataPromptExpertF16Swiglu",
            false,
            &CubeCount::Static(1, 1, 1),
            &KernelArguments::new(),
        )
        .unwrap();
}

#[test]
fn resident_pairs_keep_all_original_phases_and_reject_mixed_or_reversed_roles() {
    let h = owners();
    for members in [1, 32, 257, 384, 4096] {
        let mut grammar = begin_images(&h, members, true);
        product(&mut grammar, &h, members, false, members >= 256);
        activate(&mut grammar);
        product(&mut grammar, &h, members, true, members >= 256);
        let report = grammar.finish().unwrap();
        assert_eq!(report.logical_phases, 6);
    }
    for (first, second) in [
        (
            "StrataPromptExpertResidentGuImage",
            "StrataPromptExpertF16Image",
        ),
        (
            "StrataPromptExpertF16Image",
            "StrataPromptExpertResidentDownImage",
        ),
    ] {
        let mut grammar = begin_gather(&h, 257);
        grammar
            .launch(
                first,
                false,
                &CubeCount::Static(1, 1, 1),
                &KernelArguments::new(),
            )
            .unwrap();
        assert!(
            grammar
                .launch(
                    second,
                    second == "StrataPromptExpertF16Image",
                    &CubeCount::Static(1, 1, 1),
                    &KernelArguments::new()
                )
                .is_err()
        );
    }
    for (name, phase, release) in [
        ("StrataPromptExpertResidentGuImage", 2, false),
        ("StrataPromptExpertResidentDownImage", 1, false),
        ("StrataPromptExpertResidentDownImage", 2, true),
        ("StrataPromptExpertF16Image", 2, false),
    ] {
        assert!(!native_strata_expert_operator(name, release, phase));
    }
}

#[test]
fn complete_original_or_exact_wide_tail_phases_report_real_launch_counts() {
    for members in [256, 257, 287, 288, 383, 384, 4096] {
        for gu_wide in [false, true] {
            for down_wide in [false, true] {
                let h = owners();
                let mut grammar = begin(&h, members);
                product(&mut grammar, &h, members, false, gu_wide);
                activate(&mut grammar);
                product(&mut grammar, &h, members, true, down_wide);
                let report = grammar.finish().unwrap();
                let prefixes = usize::from(gu_wide) + usize::from(down_wide);
                let tails = if members % 128 == 0 { 0 } else { prefixes };
                assert_eq!(report.logical_phases, 6);
                assert_eq!(report.launches, 6 + tails);
                assert_eq!(report.matrix_pieces, 2 + tails);
                assert_eq!(report.wide_prefixes, prefixes);
                assert_eq!(report.wide_blocks, prefixes * (members / 128) as usize);
                assert_eq!(report.tail_pieces, tails);
            }
        }
    }
}

#[cfg(feature = "kuno-expert-cohorts")]
#[test]
fn terminal_unit_metadata_requires_actual_original_down_or_complete_wide_tail() {
    use alloc::boxed::Box;
    use crate::{
        compiler::{CompilationError, Compiler, CubeTask},
        id::KernelId,
        kernel::{CompiledKernel, KernelDefinition, KernelMetadata},
        server::{CubeDim, ExecutionMode},
    };
    use super::super::native_expert_batch::NativeStrataExpertUnitEnd;
    use cubecl_ir::{ElemType, StorageType, UIntKind};
    #[derive(Clone,Debug)]
    struct FixtureCompiler;
    impl Compiler for FixtureCompiler {
        type Representation=alloc::string::String;
        type CompilationOptions=u8;
        fn compile(&mut self,_kernel:KernelDefinition,_options:&u8,_mode:ExecutionMode,_address:StorageType)
            ->Result<Self::Representation,CompilationError> {panic!("fixture task must be forwarded directly")}
        fn elem_size(&self,_elem:ElemType)->usize {4}
        fn extension(&self)->&'static str {"fixture"}
    }
    struct OriginalTask(&'static str);
    impl KernelMetadata for OriginalTask {
        fn name(&self)->&'static str {self.0}
        fn id(&self)->KernelId {KernelId::new::<Self>()}
        fn address_type(&self)->StorageType {StorageType::Scalar(ElemType::UInt(UIntKind::U32))}
    }
    impl CubeTask<FixtureCompiler> for OriginalTask {
        fn compile(&self,_compiler:&mut FixtureCompiler,options:&u8,mode:ExecutionMode,address:StorageType)
            ->Result<CompiledKernel<FixtureCompiler>,CompilationError> {
            assert_eq!(*options,17);assert!(matches!(mode,ExecutionMode::Unchecked));
            assert_eq!(address,self.address_type());
            Ok(CompiledKernel {entrypoint_name:"original_down".into(),debug_name:Some(self.0),
                source:"exact original down compilation".into(),repr:Some("original representation".into()),
                cube_dim:CubeDim::new_1d(256),debug_info:None})
        }
    }
    for members in [1,32,256,257,287,288,383,384,4096] {
        for wide in [false,true] {
            if wide && members<MIN_WIDE_MEMBERS {continue;}
            let h=owners();let mut grammar=begin(&h,members);
            product(&mut grammar,&h,members,false,wide);
            activate(&mut grammar);
            let prefix=members/128*128;
            let args=matrix_args(&h,members,true,wide,false);
            let grid=if wide {CubeCount::Static(2560/128,prefix/128,1)}
                else {CubeCount::Static(2560/64,members.div_ceil(32),1)};
            let name=if wide {WIDE_DOWN}else{ORIGINAL};
            let first=NativeStrataExpertUnitEnd::validate(Box::new(OriginalTask(name)),&mut grammar,&grid,&args);
            let (task,name)=if wide && prefix<members {
                // A genuine wide prefix advances its original pending-tail
                // state, but must never itself acquire terminal metadata.
                assert!(first.is_err());
                let tail=members-prefix;
                let tile=if tail<=8 {8}else if tail<32 {16}else{32};
                let task=NativeStrataExpertUnitEnd::validate(Box::new(OriginalTask(TAIL_DOWN)),&mut grammar,
                    &CubeCount::Static(2560/64,tail.div_ceil(tile),1),&matrix_args(&h,members,true,true,true)).unwrap();
                (task,TAIL_DOWN)
            } else {(first.unwrap(),name)};
            assert_eq!(task.name(),name);assert_eq!(task.id(),OriginalTask(name).id());
            assert!(task.native_release_after());assert_eq!(grammar.finish().unwrap().logical_phases,6);
            let actual=task.compile(&mut FixtureCompiler,&17,ExecutionMode::Unchecked,task.address_type()).unwrap();
            assert_eq!(actual.entrypoint_name,"original_down");assert_eq!(actual.debug_name,Some(name));
            assert_eq!(actual.source,"exact original down compilation");
            assert_eq!(actual.repr.as_deref(),Some("original representation"));
        }
    }
}
#[test]
fn missing_split_tail_cannot_be_counted_or_interleaved_or_replaced() {
    let h = owners();
    let pending = || {
        let mut grammar = begin(&h, 257);
        grammar
            .launch(
                WIDE_GU,
                false,
                &CubeCount::Static(10, 2, 1),
                &matrix_args(&h, 257, false, true, false),
            )
            .unwrap();
        grammar
    };
    assert!(pending().finish().is_err());
    assert!(pending().host_copies().is_err());
    for name in [
        WIDE_GU,
        TAIL_DOWN,
        ORIGINAL,
        "StrataPromptExpertF16Swiglu",
        "StrataPromptExpertF16Image",
    ] {
        assert!(
            pending()
                .launch(
                    name,
                    false,
                    &CubeCount::Static(20, 1, 1),
                    &matrix_args(&h, 257, false, true, true)
                )
                .is_err()
        );
    }
}
#[test]
fn scalar_schema_coverage_grid_and_every_original_tail_owner_are_checked() {
    let h = owners();
    for index in 0..7 {
        let mut grammar = begin(&h, 287);
        grammar
            .launch(
                WIDE_GU,
                false,
                &CubeCount::Static(10, 2, 1),
                &matrix_args(&h, 287, false, true, false),
            )
            .unwrap();
        let mut tail = matrix_args(&h, 287, false, true, true);
        let old = word(&tail.info, index).unwrap();
        set_word(&mut tail.info, index, old + 1);
        assert!(
            grammar
                .launch(TAIL_GU, false, &CubeCount::Static(20, 2, 1), &tail)
                .is_err()
        );
    }
    for index in 0..6 {
        let mut grammar = begin(&h, 287);
        grammar
            .launch(
                WIDE_GU,
                false,
                &CubeCount::Static(10, 2, 1),
                &matrix_args(&h, 287, false, true, false),
            )
            .unwrap();
        let mut tail = matrix_args(&h, 287, false, true, true);
        let view = &tail.buffers[index];
        tail.buffers[index] = Handle::new(view.stream, view.size)
            .offset_start(view.offset_start.unwrap_or(0))
            .offset_end(view.offset_end.unwrap_or(0))
            .binding();
        assert!(
            grammar
                .launch(TAIL_GU, false, &CubeCount::Static(20, 2, 1), &tail)
                .is_err()
        );
    }
    // Same allocation, same size and metadata: moving both ends equally is
    // still not the authentic prefix-relative remaining member interval.
    let mut grammar = begin(&h, 287);
    grammar
        .launch(
            WIDE_GU,
            false,
            &CubeCount::Static(10, 2, 1),
            &matrix_args(&h, 287, false, true, false),
        )
        .unwrap();
    let mut tail = matrix_args(&h, 287, false, true, true);
    tail.buffers[0].offset_start = Some(tail.buffers[0].offset_start.unwrap() + 8);
    tail.buffers[0].offset_end = Some(tail.buffers[0].offset_end.unwrap() - 8);
    assert!(
        grammar
            .launch(TAIL_GU, false, &CubeCount::Static(20, 2, 1), &tail)
            .is_err()
    );
    for index in [7, 8, 14] {
        let mut grammar = begin(&h, 287);
        let mut prefix = matrix_args(&h, 287, false, true, false);
        let old = word(&prefix.info, index).unwrap();
        set_word(&mut prefix.info, index, old + 1);
        assert!(
            grammar
                .launch(WIDE_GU, false, &CubeCount::Static(10, 2, 1), &prefix)
                .is_err()
        );
    }
    for count in [
        CubeCount::Static(10, 1, 1),
        CubeCount::Static(20, 2, 1),
        CubeCount::Static(10, 2, 2),
    ] {
        assert!(
            begin(&h, 287)
                .launch(
                    WIDE_GU,
                    false,
                    &count,
                    &matrix_args(&h, 287, false, true, false)
                )
                .is_err()
        );
    }
    let mut wrong = matrix_args(&h, 287, false, false, false);
    set_word(&mut wrong.info, 2, 256);
    assert!(
        begin(&h, 287)
            .launch(ORIGINAL, false, &CubeCount::Static(20, 8, 1), &wrong)
            .is_err()
    );
    for name in [
        WIDE_DOWN,
        "probe::native_expert_wide::expert_wide_gu::ExpertWideGuFake<Runtime>",
    ] {
        assert!(
            begin(&h, 287)
                .launch(
                    name,
                    false,
                    &CubeCount::Static(10, 2, 1),
                    &matrix_args(&h, 287, false, true, false)
                )
                .is_err()
        );
    }
}

const DIRECT_GU: &str =
    "probe::strata_matrix::native_expert_wide::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>";
fn direct_gu(h: &[Handle; 7], members: u32, tail: bool) -> NativeStrataExpertGrammar {
    let mut g = begin_images(h, members, true);
    g.launch(
        DIRECT_GU,
        false,
        &CubeCount::Static(10, members / 128, 1),
        &matrix_args(h, members, false, true, false),
    )
    .unwrap();
    if members % 128 != 0 && tail {
        let remain = members % 128;
        let tile = if remain <= 8 {
            8
        } else if remain < 32 {
            16
        } else {
            32
        };
        g.launch(
            TAIL_GU,
            false,
            &CubeCount::Static(20, remain.div_ceil(tile), 1),
            &matrix_args(h, members, false, true, true),
        )
        .unwrap();
    }
    g
}
fn checker(h: &[Handle; 7], members: u32) -> KernelArguments {
    let mut words = [0u32; 16];
    words[..3].copy_from_slice(&[members, 2560, 1280]);
    KernelArguments::new()
        .with_buffers(
            [
                h[3].clone(),
                h[2].clone(),
                h[0].clone(),
                h[4].clone(),
                h[5].clone(),
            ]
            .into_iter()
            .map(Handle::binding)
            .collect(),
        )
        .with_info(MetadataBindingInfo::custom(packed(&words)))
}
#[test]
fn direct_gu_requires_exact_checker_and_original_tail_before_any_complete_batch() {
    for members in [256, 257, 264, 383, 384, 513, 4096] {
        let h = owners();
        let mut g = direct_gu(&h, members, true);
        assert!(g.finish().is_err());
        g.launch(
            "StrataPromptExpertF16SwigluCheckGu",
            false,
            &CubeCount::Static((members * 5).div_ceil(4), 1, 1),
            &checker(&h, members),
        )
        .unwrap();
        product(&mut g, &h, members, true, true);
        let r = g.finish().unwrap();
        assert_eq!(r.direct_gu_prefixes, 1);
        assert_eq!(r.gu_checks, 1);
        assert_eq!(r.logical_phases, 6);
    }
    let h = owners();
    let mut g = direct_gu(&h, 257, false);
    assert!(
        g.launch(
            "StrataPromptExpertF16SwigluCheckGu",
            false,
            &CubeCount::Static(322, 1, 1),
            &checker(&h, 257)
        )
        .is_err()
    );
}
#[test]
fn direct_gu_rejects_plain_checker_owner_info_grid_release_and_order_substitution() {
    let h = owners();
    let mut g = direct_gu(&h, 257, true);
    assert!(
        g.launch(
            "StrataPromptExpertF16Swiglu",
            false,
            &CubeCount::Static(322, 1, 1),
            &checker(&h, 257)
        )
        .is_err()
    );
    for i in 0..5 {
        let mut g = direct_gu(&h, 257, true);
        let mut a = checker(&h, 257);
        a.buffers[i] = Handle::new(StreamId::current(), a.buffers[i].size).binding();
        assert!(
            g.launch(
                "StrataPromptExpertF16SwigluCheckGu",
                false,
                &CubeCount::Static(322, 1, 1),
                &a
            )
            .is_err()
        );
    }
    for index in [0, 1, 2, 4, 5] {
        let mut g = direct_gu(&h, 257, true);
        let mut a = checker(&h, 257);
        set_word(&mut a.info, index, 99);
        assert!(
            g.launch(
                "StrataPromptExpertF16SwigluCheckGu",
                false,
                &CubeCount::Static(322, 1, 1),
                &a
            )
            .is_err()
        );
    }
    for (release, grid) in [
        (true, CubeCount::Static(322, 1, 1)),
        (false, CubeCount::Static(321, 1, 1)),
    ] {
        let mut g = direct_gu(&h, 257, true);
        assert!(
            g.launch(
                "StrataPromptExpertF16SwigluCheckGu",
                release,
                &grid,
                &checker(&h, 257)
            )
            .is_err()
        );
    }
    let mut g = begin_images(&h, 257, true);
    assert!(
        g.launch(
            "StrataPromptExpertF16SwigluCheckGu",
            false,
            &CubeCount::Static(322, 1, 1),
            &checker(&h, 257)
        )
        .is_err()
    );
    let mut g = direct_gu(&h, 256, true);
    g.launch(
        "StrataPromptExpertF16SwigluCheckGu",
        false,
        &CubeCount::Static(320, 1, 1),
        &checker(&h, 256),
    )
    .unwrap();
    assert!(
        g.launch(
            "StrataPromptExpertF16SwigluCheckGu",
            false,
            &CubeCount::Static(320, 1, 1),
            &checker(&h, 256)
        )
        .is_err()
    );
}
#[test]
fn validating_activation_also_accepts_original_checked_small_group_without_changing_six_phases() {
    for members in [1, 7, 8, 16, 32, 255] {
        let h = owners();
        let mut g = begin_images(&h, members, true);
        product(&mut g, &h, members, false, false);
        g.launch(
            "StrataPromptExpertF16SwigluCheckGu",
            false,
            &CubeCount::Static((members * 5).div_ceil(4), 1, 1),
            &checker(&h, members),
        )
        .unwrap();
        product(&mut g, &h, members, true, false);
        let r = g.finish().unwrap();
        assert_eq!(r.direct_gu_prefixes, 0);
        assert_eq!(r.gu_checks, 1);
        assert_eq!(r.logical_phases, 6);
    }
}

#[cfg(feature = "kuno-native-expert-medium")]
#[test]
fn medium_exact_prefix_and_real_tail_keep_original_six_phase_grammar() {
    for members in [128, 129, 135, 136, 137, 159, 160, 191, 192, 255] {
        let h = owners();
        let mut grammar = begin(&h, members);
        product(&mut grammar, &h, members, false, true);
        activate(&mut grammar);
        product(&mut grammar, &h, members, true, true);
        let report = grammar.finish().unwrap();
        let tails = if members == 128 { 0 } else { 2 };
        assert_eq!(report.logical_phases, 6);
        assert_eq!(report.wide_prefixes, 2);
        assert_eq!(report.wide_blocks, 2);
        assert_eq!(report.tail_pieces, tails);
        assert_eq!(report.launches, 6 + tails);
    }
}
#[cfg(feature = "kuno-native-expert-medium")]
#[test]
fn medium_prefix_declines_incomplete_or_foreign_original_geometry() {
    for members in [1, 32, 127, 128, 129, 255] {
        for bad in 0..3 {
            let h = owners();
            let mut grammar = begin(&h, members);
            let mut args = matrix_args(&h, members, false, true, false);
            let rows = 1280;
            let mut grid = CubeCount::Static(rows / 128, members / 128, 1);
            match bad {
                0 => {
                    set_word(&mut args.info, 0, 0);
                }
                1 => {
                    grid = CubeCount::Static(rows / 128, 0, 1);
                }
                _ => {
                    args.buffers[0] = owners()[0].clone().binding();
                }
            }
            assert!(grammar.launch(WIDE_GU, false, &grid, &args).is_err());
        }
    }
}

const MACRO64: &str = "probe::strata_matrix::native::expert_macro64::ExpertMacro64<Runtime>";
#[test]
fn genuine_macro64_geometry_keeps_all_six_phases_and_original_owners() {
    for members in [64u32,65,127,128,129,255] {
        let h=owners();let mut grammar=begin(&h,members);
        for down in [false,true] {
            if down {activate(&mut grammar);}
            grammar.launch(MACRO64,false,
                &CubeCount::Static(if down{40}else{20},members.div_ceil(64),1),
                &matrix_args(&h,members,down,false,false)).unwrap();
        }
        let counts=grammar.finish().unwrap();
        assert_eq!((counts.launches,counts.logical_phases,counts.matrix_pieces),(6,6,2));
        assert_eq!((counts.wide_prefixes,counts.tail_pieces),(0,0));
    }
}
#[test]
fn macro64_rejects_original_grid_small_or_wide_members_and_changed_owner() {
    for members in [1u32,32,63,256,257,4096] {
        let h=owners();let mut grammar=begin(&h,members);
        assert!(grammar.launch(MACRO64,false,&CubeCount::Static(20,members.div_ceil(64),1),
            &matrix_args(&h,members,false,false,false)).is_err());
    }
    for bad in 0..4 {
        let h=owners();let mut grammar=begin(&h,129);
        let mut args=matrix_args(&h,129,false,false,false);
        if bad==0 {args.buffers[0]=owners()[0].clone().binding();}
        let grid=match bad {1=>CubeCount::Static(20,129u32.div_ceil(32),1),
            2=>CubeCount::Static(19,3,1),3=>CubeCount::Static(20,3,2),_=>CubeCount::Static(20,3,1)};
        assert!(grammar.launch(MACRO64,false,&grid,&args).is_err());
    }
}

#[cfg(feature="kuno-deep-matrix")]
fn deep_product(g:&mut NativeStrataExpertGrammar,h:&[Handle;7],members:u32,down:bool) {
    let name=if down {
        "probe::strata_matrix::native_expert_wide::deep::expert_wide_down::ExpertWideDown<Runtime>"
    }else{"probe::strata_matrix::native_expert_wide::deep::expert_wide_gu::ExpertWideGu<Runtime>"};
    let rows=if down{2560}else{1280};
    g.launch(name,false,&CubeCount::Static(rows/128,members/128,1),
        &matrix_args(h,members,down,true,false)).unwrap();
    if members%128!=0 {
        let remain=members%128;let tile=if remain<=8{8}else if remain<32{16}else{32};
        g.launch(if down{TAIL_DOWN}else{TAIL_GU},false,
            &CubeCount::Static(rows/64,remain.div_ceil(tile),1),
            &matrix_args(h,members,down,true,true)).unwrap();
    }
}
#[cfg(feature="kuno-deep-matrix")]
#[test]
fn deep_wide_namespace_keeps_all_original_role_owner_grid_and_tail_constraints() {
    for members in [256,257,287,4096] {
        let h=owners();let mut g=begin(&h,members);
        deep_product(&mut g,&h,members,false);activate(&mut g);deep_product(&mut g,&h,members,true);
        let r=g.finish().unwrap();assert_eq!(r.logical_phases,6);assert_eq!(r.wide_prefixes,2);
    }
    let h=owners();let args=matrix_args(&h,256,false,true,false);
    for name in [
        "probe::native_expert_wide::deep::deep::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::native_expert_wide::deep::expert_wide_gu::ExpertWideGuFake<Runtime>",
        "probe::native_expert_wide::deeper::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::native_expert_wide::deep::expert_wide_down::ExpertWideDown<Runtime>",
        "probe::native_expert_wide::deep::expert_tail_gu::ExpertTailGu<Runtime>",
    ] {assert!(begin(&h,256).launch(name,false,&CubeCount::Static(10,2,1),&args).is_err());}
    let name="probe::native_expert_wide::deep::expert_wide_gu::ExpertWideGu<Runtime>";
    assert!(begin(&h,256).launch(name,true,&CubeCount::Static(10,2,1),&args).is_err());
    assert!(begin(&h,256).launch(name,false,&CubeCount::Static(20,2,1),&args).is_err());
    let mut wrong=matrix_args(&h,256,false,true,false);
    wrong.buffers[3]=Handle::new(StreamId::current(),MAP_BYTES).binding();
    assert!(begin(&h,256).launch(name,false,&CubeCount::Static(10,2,1),&wrong).is_err());
}
#[cfg(feature="kuno-deep-matrix")]
#[test]
fn deep_direct_gu_still_requires_original_checker_before_down_and_complete_batch() {
    let name="probe::native_expert_wide::deep::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>";
    let h=owners();let mut g=begin_images(&h,256,true);
    g.launch(name,false,&CubeCount::Static(10,2,1),&matrix_args(&h,256,false,true,false)).unwrap();
    assert!(g.finish().is_err());
    let mut g=begin_images(&h,256,true);
    g.launch(name,false,&CubeCount::Static(10,2,1),&matrix_args(&h,256,false,true,false)).unwrap();
    assert!(g.launch("StrataPromptExpertF16Swiglu",false,&CubeCount::Static(320,1,1),&checker(&h,256)).is_err());
    let mut g=begin_images(&h,256,true);
    g.launch(name,false,&CubeCount::Static(10,2,1),&matrix_args(&h,256,false,true,false)).unwrap();
    g.launch("StrataPromptExpertF16SwigluCheckGu",false,&CubeCount::Static(320,1,1),&checker(&h,256)).unwrap();
    deep_product(&mut g,&h,256,true);let r=g.finish().unwrap();
    assert_eq!(r.logical_phases,6);assert_eq!(r.gu_checks,1);assert_eq!(r.direct_gu_prefixes,1);
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn ordered_full_down_preserves_original_matrix_roles_owners_and_dispatch() {
    for (members, module, ty, tile) in [
        (3u32, "expert_narrow", "ExpertNarrow", 8u32),
        (17, "expert_narrow", "ExpertNarrow", 16),
        (32, "expert_macro", "ExpertMacro", 32),
        (65, "expert_macro64", "ExpertMacro64", 64),
        (255, "expert_macro64", "ExpertMacro64", 64),
    ] {
        let name = alloc::format!("probe::strata_matrix::native::ordered::{module}::{ty}<Runtime>");
        let h = owners();
        let mut g = begin(&h, members);
        product(&mut g, &h, members, false, false);
        activate(&mut g);
        g.launch(&name, false, &CubeCount::Static(40, members.div_ceil(tile), 1),
            &matrix_args(&h, members, true, false, false)).unwrap();
        assert_eq!(g.finish().unwrap().logical_phases, 6);
        for bad in 0..4 {
            let mut g = begin(&h, members);
            product(&mut g, &h, members, false, false);
            activate(&mut g);
            let mut args = matrix_args(&h, members, true, false, false);
            if bad == 0 { args.buffers[3] = owners()[3].clone().binding(); }
            if bad == 1 { set_word(&mut args.info, 0, 2560); }
            let grid = CubeCount::Static(if bad == 2 { 20 } else { 40 }, members.div_ceil(tile), 1);
            assert!(g.launch(&name, bad == 3, &grid, &args).is_err());
        }
        // A valid new DOWN name is never an alias for a GU product.
        assert!(begin(&h, members).launch(&name, false,
            &CubeCount::Static(20, members.div_ceil(tile), 1),
            &matrix_args(&h, members, false, false, false)).is_err());
    }
    for name in [
        "probe::strata_matrix::native::ordered::ordered::expert_macro::ExpertMacro<Runtime>",
        "probe::strata_matrix::native::ordered::expert_macro::ExpertMacroFake<Runtime>",
        "probe::strata_matrix::native::ordered::other::expert_macro::ExpertMacro<Runtime>",
    ] { assert!(!native_strata_expert_operator(name, false, 5)); }
}

#[cfg(feature = "kuno-expert-ordered-output")]
#[test]
fn ordered_wide_down_and_tail_keep_one_layout_and_all_original_owner_checks() {
    let wide = "probe::strata_matrix::native_expert_wide::ordered::expert_wide_down::ExpertWideDown<Runtime>";
    let tail = "probe::strata_matrix::native_expert_wide::ordered::expert_tail_down::ExpertTailDown<Runtime>";
    for members in [256u32, 257, 287, 4096] {
        let h = owners();
        let mut g = begin(&h, members);
        product(&mut g, &h, members, false, true);
        activate(&mut g);
        g.launch(wide, false, &CubeCount::Static(20, members / 128, 1),
            &matrix_args(&h, members, true, true, false)).unwrap();
        if members % 128 != 0 {
            let rest = members % 128;
            let tile = if rest <= 8 { 8 } else if rest < 32 { 16 } else { 32 };
            g.launch(tail, false, &CubeCount::Static(40, rest.div_ceil(tile), 1),
                &matrix_args(&h, members, true, true, true)).unwrap();
        }
        assert_eq!(g.finish().unwrap().logical_phases, 6);
    }
    for (first, second) in [(wide, TAIL_DOWN), (WIDE_DOWN, tail)] {
        let h = owners(); let mut g = begin(&h, 257);
        product(&mut g, &h, 257, false, true); activate(&mut g);
        g.launch(first, false, &CubeCount::Static(20, 2, 1),
            &matrix_args(&h, 257, true, true, false)).unwrap();
        assert!(g.launch(second, false, &CubeCount::Static(40, 1, 1),
            &matrix_args(&h, 257, true, true, true)).is_err());
    }
    for name in [
        "probe::native_expert_wide::ordered::expert_wide_gu::ExpertWideGu<Runtime>",
        "probe::native_expert_wide::ordered::expert_wide_gu_direct::ExpertWideGuDirect<Runtime>",
        "probe::native_expert_wide::ordered::ordered::expert_wide_down::ExpertWideDown<Runtime>",
        "probe::native_expert_wide::ordered::expert_wide_down::ExpertWideDownFake<Runtime>",
    ] { assert!(!is_wide(name, false) && !is_wide(name, true) && !is_direct_gu(name)); }
}

#[cfg(feature = "kuno-wide-shared-reuse")]
#[path = "native_expert_shared_reuse_tests.rs"]
mod shared_reuse;

#[cfg(feature="kuno-dequant-direct8")]
#[path="native_expert_direct8_tests.rs"]
mod direct8;

#[cfg(all(feature="kuno-small-direct-publish",feature="kuno-expert-ordered-output"))]
#[path="native_expert_small_direct_tests.rs"]
mod small_direct;

#[cfg(feature="kuno-global-small-products")]
#[path="native_expert_global_small_tests.rs"]
mod global_small;

#[cfg(feature="kuno-small-full-walk")]
#[path="native_expert_full_walk_tests.rs"]
mod full_walk;

#[cfg(feature="kuno-padded-global")]
#[path="native_expert_padded_global_tests.rs"]
mod padded_global;
