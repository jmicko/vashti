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
    for release in [false, true] {
        grammar
            .launch(
                "StrataPromptExpertF16Image",
                release,
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
