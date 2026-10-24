//! Real GU and DOWN input producers initialize complete original narrow tiles.
//! No padded row has a route, output or status. All owning views stay complete.
use super::*;

pub(super) fn source(mut source: String, activate: bool) -> String {
    let (needle, output, words) = if activate {
        ("if !valid() {return;}let i=id.x;", "hidden", "320u")
    } else {
        (
            "if !valid() || p.r6>1u {return;}let i=id.x;",
            "packed",
            "(p.columns/2u)",
        )
    };
    assert_eq!(source.matches(needle).count(), 1);
    let limit=if cfg!(feature="strata-padded-global") {256}else{32};
    let code = format!(
        r#"{needle}
    // STRATA_NARROW_GLOBAL_PADDED_PRODUCER
    let real=map[516u+p.group];
    if real<{limit}u {{
        let tile=select(select(16u,8u,real<=8u),32u,real>=32u);let padded=((real+tile-1u)/tile)*tile;
        let row_words={words};let first=real*row_words;let end=padded*row_words;
        if end>arrayLength(&{output}) {{atomicOr(&fault[0],128u);return;}}
        // Original grid/real count stays authoritative, including tiny groups.
        let stride=((first+255u)/256u)*256u;
        for(var at=first+i;at<end;at+=stride) {{{output}[at]=0u;}}
    }}
"#
    );
    source = source.replace(needle, &code);
    source
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature="strata-padded-global")]
    #[test]
    fn actual_padded32_names_bind_both_producers_with_bounded_medium_initialization() {
        for(op,name)in [(Op::Gather,"StrataPromptExpertGatherPad32"),(Op::ActivateCheckGu,"StrataPromptExpertF16SwigluCheckGuPad32")] {
            assert_eq!(op.name(),name);let s=op.source();assert!(s.contains("if real<256u"));
            assert!(s.contains("32u,real>=32u"));assert!(s.contains("if end>arrayLength("));
            assert!(s.contains("at<end;at+=stride"));
        }
    }
    #[test]
    fn actual_both_producers_only_add_disjoint_bounded_padding_and_keep_old_grid() {
        for op in [Op::Gather, Op::Activate, Op::ActivateCheckGu] {
            let original = op.source_unpadded();
            let actual = source(original.clone(), op != Op::Gather);
            let module = wgpu::naga::front::wgsl::parse_str(&actual).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let begin = actual
                .find("\n    // STRATA_NARROW_GLOBAL_PADDED_PRODUCER")
                .unwrap();
            let end = actual[begin..].find("\n    }\n").unwrap() + begin + "\n    }\n".len();
            let mut restored = actual.clone();
            restored.replace_range(begin..end, "");
            assert_eq!(restored, original);
            assert!(actual.contains("if end>arrayLength("));
            assert!(actual.contains("let real=map[516u+p.group]"));
            assert_eq!(actual.matches("@workgroup_size(256)").count(), 1);
            assert_eq!(actual.matches("var<uniform> p:Params;").count(), 1);
        }
    }
    #[test]
    fn original_grids_initialize_each_extra_gu_and_down_word_once_without_touching_real_rows() {
        for real in 1usize..=4096 {
            for row_words in [1280usize, 320] {
                let original_grid = (real * row_words).div_ceil(256);
                let tile = if real <= 8 {8} else if real < 32 {16} else {32};
                let padded = if real < if cfg!(feature="strata-padded-global") {256} else {32} {
                    real.div_ceil(tile) * tile
                } else {
                    real
                };
                assert!(padded <= 4096);
                let first = real * row_words;
                let end = padded * row_words;
                let mut visits = vec![0u8; end - first];
                // Large cases have no padding; they execute no new loop.
                if first < end {
                    for tid in 0..original_grid * 256 {
                        for at in (first + tid..end).step_by(original_grid * 256) {
                            visits[at - first] += 1;
                        }
                    }
                }
                assert!(visits.iter().all(|&n| n == 1));
                assert_eq!(
                    super::super::member_dispatches(real, 4096).unwrap()
                        [usize::from(row_words == 320)],
                    original_grid
                );
            }
        }
    }
}
