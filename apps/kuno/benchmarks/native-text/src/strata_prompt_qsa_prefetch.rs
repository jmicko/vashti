//! Bounded prompt-only memory lookahead, preserving the selected scalar math.
//! Source order is not a hardware scheduling claim: inspect actual native ISA
//! before claiming that ANV keeps these independent loads ahead of their uses.

const TAG: &str = "// STRATA_QSA_PREFETCH: bounded raw-pair/Q-head/four-cell PV lookahead.\n";
const PAIR: &str = "for(var pair=warp;pair<64u;pair+=16u) {";
const PV: &str = "for(var c=0u;c<here;c++) {";
const Q: &str = "let qa=sq[off/4u];let qb=sq[off/4u+1u];\n";
const PROBABILITY: &str = "var probability=0.0;\nif lane<12u {probability=sp[lane*65u+c];}\n";

fn end_block(s: &str, start: usize) -> usize {
    assert_eq!(s.as_bytes()[start], b'{');
    let mut depth = 0usize;
    for (offset, byte) in s.as_bytes()[start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).expect("prefetch braces");
                if depth == 0 {
                    return start + offset + 1;
                }
            }
            _ => {}
        }
    }
    panic!("prefetch complete original block required")
}
fn once(s: &mut String, old: &str, new: &str) {
    assert_eq!(s.matches(old).count(), 1, "prefetch source drift: {old}");
    *s = s.replacen(old, new, 1);
}
fn block<'a>(s: &'a str, header: &str) -> &'a str {
    assert_eq!(s.matches(header).count(), 1, "prefetch block drift");
    let start = s.find(header).unwrap();
    &s[start..end_block(s, start + header.len() - 1)]
}

/// Save ONLY immutable raw words for two cells, not another decoded K image.
/// The original current-cell body still performs every causal/fault predicate.
/// A masked or future cell never supplies a cache address, even speculatively.
fn raw_pair_load(pair: &str) -> String {
    let mut s = format!("{{let prefetch_pair={pair};\n");
    for cell in 0..2 {
        s.push_str(&format!(
            "prefetch_token{cell}=CONTEXT_CAPACITY;\n\
             prefetch_word{cell}_0=0u;prefetch_word{cell}_1=0u;prefetch_scale{cell}=0u;\n\
             let prefetch_cell{cell}=prefetch_pair+{}u;\n\
             if prefetch_cell{cell}<here {{\n\
             prefetch_token{cell}=subgroupBroadcastFirst(row[prefetch_cell{cell}]);\n\
             if prefetch_token{cell}<p.start+query_row+1u {{\n\
             let prefetch_koff=kvh*256u+lane*8u;\n\
             prefetch_word{cell}_0=cache[prefetch_token{cell}*CACHE_WORDS+prefetch_koff/4u];\n\
             prefetch_word{cell}_1=cache[prefetch_token{cell}*CACHE_WORDS+prefetch_koff/4u+1u];\n\
             prefetch_scale{cell}=cache[prefetch_token{cell}*CACHE_WORDS+256u+prefetch_koff/128u];\n\
             }}}}\n",
            cell * 8
        ));
    }
    s.push_str("}\n");
    s
}
fn pair_pipeline(old: &str) -> String {
    assert!(old.starts_with(PAIR));
    let first_head = "{let h=0u;let off=h*256u+lane*8u;\n";
    assert_eq!(old.matches(first_head).count(), 1);
    let at = old.find(first_head).unwrap();
    let mut prefix = old[PAIR.len()..at].to_owned();
    for cell in 0..2 {
        once(
            &mut prefix,
            &format!("token{cell}=subgroupBroadcastFirst(row[c{cell}]);"),
            &format!("token{cell}=prefetch_token{cell};"),
        );
        for word in 0..2 {
            let suffix = if word == 0 { "" } else { "+1u" };
            once(
                &mut prefix,
                &format!("let kword{word}=cache[token{cell}*CACHE_WORDS+koff/4u{suffix}];"),
                &format!("let kword{word}=prefetch_word{cell}_{word};"),
            );
        }
        once(
            &mut prefix,
            &format!("unpack2x16float(cache[token{cell}*CACHE_WORDS+256u+koff/128u])"),
            &format!("unpack2x16float(prefetch_scale{cell})"),
        );
    }
    // A single carried head plus current aliases gives at most TWO live Q
    // heads. It is not a96-value all-query register cache. Issue head0 reads
    // before current K decoding, and next-head reads before the current dots.
    let mut s = String::new();
    for cell in 0..2 {
        s.push_str(&format!(
            "var prefetch_token{cell}=CONTEXT_CAPACITY;\n\
             var prefetch_word{cell}_0=0u;var prefetch_word{cell}_1=0u;var prefetch_scale{cell}=0u;\n"
        ));
    }
    s.push_str(&raw_pair_load("warp"));
    s.push_str(PAIR);
    s.push_str("\nvar prefetch_qa=sq[lane*2u];var prefetch_qb=sq[lane*2u+1u];\n");
    s.push_str(&prefix);
    // Current decoded K values already hold every raw operand before these
    // carry words are overwritten. Each next pair is fetched exactly once.
    s.push_str("if pair+16u<64u {\n");
    s.push_str(&raw_pair_load("pair+16u"));
    s.push_str("}\n");
    let mut cursor = at;
    for head in 0..12 {
        let header = format!("{{let h={head}u;let off=h*256u+lane*8u;\n");
        assert!(old[cursor..].starts_with(&header), "prefetch head order");
        let end = end_block(old, cursor);
        let mut body = old[cursor..end].to_owned();
        let mut loads = String::from("let qa=prefetch_qa;let qb=prefetch_qb;\n");
        if head + 1 < 12 {
            loads.push_str(&format!(
                "prefetch_qa=sq[{}u+lane*2u];prefetch_qb=sq[{}u+lane*2u+1u];\n",
                (head + 1) * 64,
                (head + 1) * 64
            ));
        }
        once(&mut body, Q, &loads);
        s.push_str(&body);
        s.push_str(&old[end..][..old[end..].len() - old[end..].trim_start().len()]);
        cursor = end + old[end..].len() - old[end..].trim_start().len();
    }
    assert_eq!(&old[cursor..], "}", "prefetch pair closing drift");
    s.push('}');
    s
}

fn pv_pipeline(old: &str) -> String {
    assert!(old.starts_with(PV));
    let body = old.strip_prefix(PV).unwrap().strip_suffix('}').unwrap();
    assert!(body.contains("if row[c]>=p.start+query_row+1u {continue;}"));
    assert_eq!(body.matches(PROBABILITY).count(), 1);
    let mut s = String::from("for(var prefetch_base=0u;prefetch_base<here;prefetch_base+=4u) {\n");
    for cell in 0..4 {
        s.push_str(&format!(
            "let prefetch_c{cell}=prefetch_base+{cell}u;\n\
             var prefetch_live{cell}=false;var prefetch_code{cell}=0u;var prefetch_vscale{cell}=0u;\n\
             var prefetch_probability{cell}=0.0;\n\
             if prefetch_c{cell}<here {{\n\
             let prefetch_vtoken=row[prefetch_c{cell}];\n\
             if prefetch_vtoken<p.start+query_row+1u {{\n\
             prefetch_live{cell}=true;let prefetch_voff=512u+kvh*256u+tid;\n\
             prefetch_code{cell}=cache[prefetch_vtoken*CACHE_WORDS+prefetch_voff/4u];\n\
             prefetch_vscale{cell}=cache[prefetch_vtoken*CACHE_WORDS+256u+prefetch_voff/128u];\n\
             if lane<12u {{prefetch_probability{cell}=sp[lane*65u+prefetch_c{cell}];}}\n\
             }}}}\n"
        ));
    }
    // Keep CELL-major order, including all twelve original FMAs per cell.
    // Invalid cells skip arithmetic exactly; multiplying by zero instead would
    // incorrectly change poisoned/infinite cases and signed-zero behavior.
    for cell in 0..4 {
        let mut current = body.to_owned();
        once(
            &mut current,
            "if row[c]>=p.start+query_row+1u {continue;}",
            "",
        );
        once(
            &mut current,
            "let v=kv(row[c],512u+kvh*256u+tid);",
            &format!(
                "let v=q8_value(prefetch_code{cell},prefetch_vscale{cell},512u+kvh*256u+tid);"
            ),
        );
        once(
            &mut current,
            PROBABILITY,
            &format!("let probability=prefetch_probability{cell};\n"),
        );
        assert!(!current.contains("row[c]") && !current.contains("+c]"));
        s.push_str(&format!("if prefetch_live{cell} {{{current}\n}}\n"));
    }
    s.push('}');
    s
}

fn replacements(s: &str) -> [(String, String); 2] {
    for marker in [
        "STRATA_ROUND_PRODUCTS",
        "STRATA_QK_EXPLICIT_FMA",
        "STRATA_QK_UNIFORM_METADATA",
        "STRATA_QK_COMPACT",
    ] {
        assert!(s.contains(marker), "prefetch complete source prerequisite");
    }
    assert!(!s.contains("STRATA_QK_PACKED_CACHE"));
    assert!(!s.contains("STRATA_QSA_PREFETCH"));
    assert!(s.contains("var<workgroup> sq:array<vec4<f32>,768>;"));
    assert!(s.contains("var<workgroup> sp:array<f32,780>;"));
    assert!(s.contains("fn kv(token:u32,offset:u32)->f32 {return q8_value(cache["));
    let pair = block(s, PAIR).to_owned();
    let pv = block(s, PV).to_owned();
    let new_pair = pair_pipeline(&pair);
    let new_pv = pv_pipeline(&pv);
    [(pair, new_pair), (pv, new_pv)]
}
pub(super) fn source(mut s: String) -> String {
    for (old, new) in replacements(&s) {
        once(&mut s, &old, &new);
    }
    s.insert_str(0, TAG);
    s
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_local_fault, prompt_paired, prompt_pv_gather,
        prompt_specialize, prompt_uniform,
    };
    use super::*;
    fn original(k: Kernel) -> String {
        prompt_compact::source(prompt_uniform::source(prompt_fma::source(
            prompt_pv_gather::source(prompt_paired::source(prompt_specialize::source(
                k.scalar_source(),
            ))),
        )))
    }
    #[test]
    fn complete_inverse_preserves_all_source_outside_bounded_memory_phases() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let new = source(old.clone());
            let mut restored = new.strip_prefix(TAG).unwrap().to_owned();
            for (before, after) in replacements(&old).into_iter().rev() {
                once(&mut restored, &after, &before);
            }
            assert_eq!(restored, old);
            assert_eq!(
                new.matches("workgroupBarrier();").count(),
                old.matches("workgroupBarrier();").count()
            );
            assert_eq!(
                new.matches("var<workgroup>").count(),
                old.matches("var<workgroup>").count()
            );
            assert!(!new.contains("var probability=0.0;"));
            assert_eq!(new.matches("if !finite(v)").count(), 4);
            assert_eq!(new.matches("if !finite(k0_").count(), 8);
            assert_eq!(new.matches("if !finite(k1_").count(), 8);
            for h in 0..12 {
                let fma = format!("accum{h}=fma(probability{h},v,accum{h});");
                assert_eq!(old.matches(&fma).count(), 1);
                assert_eq!(new.matches(&fma).count(), 4);
            }
        }
    }
    #[test]
    fn raw_pair_visits_each_original_cell_once_and_no_guarded_tail_or_future_read() {
        for here in 0..=64 {
            for bad in 0..=64 {
                let mut visited = [0u8; 64];
                for warp in 0..8 {
                    let mut prefetched = [warp, warp + 8];
                    for pair in (warp..64).step_by(16) {
                        assert_eq!(prefetched, [pair, pair + 8]);
                        for cell in prefetched {
                            if cell < here && cell != bad {
                                visited[cell] += 1;
                            }
                        }
                        if pair + 16 < 64 {
                            prefetched = [pair + 16, pair + 24];
                        }
                    }
                }
                for (cell, &count) in visited.iter().enumerate() {
                    assert_eq!(count, u8::from(cell < here && cell != bad));
                }
            }
        }
        let loads = raw_pair_load("warp");
        for cell in 0..2 {
            let guard = loads
                .find(&format!("if prefetch_token{cell}<p.start+query_row+1u {{"))
                .unwrap();
            for word in [
                format!("prefetch_word{cell}_0=cache["),
                format!("prefetch_word{cell}_1=cache["),
                format!("prefetch_scale{cell}=cache["),
            ] {
                assert!(loads.find(&word).unwrap() > guard);
            }
        }
    }
    #[test]
    fn four_cell_visits_preserve_cell_major_fma_order_for_every_tail_and_mask() {
        for here in 0..=64 {
            for invalid in 0..=64 {
                let expected = (0..here)
                    .filter(|&c| c != invalid)
                    .flat_map(|c| (0..12).map(move |h| (c, h)))
                    .collect::<Vec<_>>();
                let actual = (0..here)
                    .step_by(4)
                    .flat_map(|base| (0..4).map(move |c| base + c))
                    .filter(|&c| c < here && c != invalid)
                    .flat_map(|c| (0..12).map(move |h| (c, h)))
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
        }
        let old = original(Kernel::PromptAttentionFault);
        let new = source(old);
        let predecode = new.find("if prefetch_live0 {").unwrap();
        for cell in 0..4 {
            assert!(new.find(&format!("prefetch_code{cell}=cache[")).unwrap() < predecode);
            assert!(
                new.find(&format!("prefetch_probability{cell}=sp["))
                    .unwrap()
                    < predecode
            );
            assert!(new.contains(&format!("if prefetch_c{cell}<here {{\nlet prefetch_vtoken=row[prefetch_c{cell}];\nif prefetch_vtoken<p.start+query_row+1u {{")));
        }
    }
    #[test]
    fn two_head_query_lookahead_keeps_every_original_dot_reduction_and_status_body() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let old = block(&old, PAIR);
            let new = pair_pipeline(old);
            for head in 0..12 {
                let header = format!("{{let h={head}u;let off=h*256u+lane*8u;\n");
                let old_at = old.find(&header).unwrap();
                let before = &old[old_at..end_block(old, old_at)];
                let new_at = new.find(&header).unwrap();
                let after = &new[new_at..end_block(&new, new_at)];
                let mut expected = String::from("let qa=prefetch_qa;let qb=prefetch_qb;\n");
                if head < 11 {
                    expected.push_str(&format!(
                        "prefetch_qa=sq[{}u+lane*2u];prefetch_qb=sq[{}u+lane*2u+1u];\n",
                        (head + 1) * 64,
                        (head + 1) * 64
                    ));
                }
                assert_eq!(after.replacen(&expected, Q, 1), before);
                for lane in 0..32 {
                    for component in 0..8 {
                        let vector = head * 64 + lane * 2 + component / 4;
                        assert_eq!(
                            vector * 4 + component % 4,
                            head * 256 + lane * 8 + component
                        );
                        assert!(vector < 768);
                    }
                }
            }
            assert!(
                new.find("var prefetch_qa=sq[").unwrap()
                    < new.find("let kword0=prefetch_word0_0;").unwrap()
            );
            assert!(new.find("if pair+16u<64u").unwrap() < new.find("{let h=0u;").unwrap());
        }
    }
    fn op_count(words: &[u32], op: u32) -> usize {
        let mut at = 5;
        let mut count = 0;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            assert!(n > 0 && at + n <= words.len());
            count += usize::from(words[at] & 65535 == op);
            at += n;
        }
        count
    }
    #[test]
    fn actual_modules_keep_complete_abi_shared_geometry_barriers_and_precise_products() {
        use wgpu::naga as n;
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let new = source(old.clone());
            for local in [false, true] {
                if local && k != Kernel::PromptAttentionFault {
                    continue;
                }
                let finish = |s| {
                    if local {
                        prompt_local_fault::source(k, s)
                    } else {
                        s
                    }
                };
                let (before, ow) = prompt_specialize::tests::compiled(&finish(old.clone()));
                let (after, nw) = prompt_specialize::tests::compiled(&finish(new.clone()));
                assert_eq!(after.entry_points[0].workgroup_size, [256, 1, 1]);
                let globals = |m: &n::Module| {
                    m.global_variables
                        .iter()
                        .map(|(_, v)| {
                            (
                                v.binding.clone(),
                                v.space,
                                format!("{:?}", m.types[v.ty].inner),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(globals(&before), globals(&after));
                assert_eq!(op_count(&ow, 224), op_count(&nw, 224));
                let fmas = |words: &[u32]| {
                    let mut at = 5;
                    let mut count = 0;
                    while at < words.len() {
                        let n = (words[at] >> 16) as usize;
                        count +=
                            usize::from(words[at] & 65535 == 12 && n >= 5 && words[at + 4] == 50);
                        at += n;
                    }
                    count
                };
                // Only the PV body is emitted four times. QK remains168
                // static FMAs, while every live cell still executes12 PV FMAs.
                assert_eq!(fmas(&ow), 180);
                assert_eq!(fmas(&nw), 216);
                assert_eq!(op_count(&ow, 345), 142);
                assert_eq!(op_count(&nw, 345), 178);
                assert_eq!(op_count(&ow, 338), 26);
                assert_eq!(op_count(&nw, 338), 28);
                let mut products = std::collections::BTreeSet::new();
                let mut precise = std::collections::BTreeSet::new();
                let mut at = 5;
                while at < nw.len() {
                    let count = (nw[at] >> 16) as usize;
                    match nw[at] & 65535 {
                        133 => {
                            products.insert(nw[at + 2]);
                        }
                        71 if count == 3 && nw[at + 2] == 42 => {
                            precise.insert(nw[at + 1]);
                        }
                        _ => {}
                    }
                    at += count;
                }
                assert!(!products.is_empty());
                assert_eq!(products, precise);
                assert_eq!(new.matches("dot=fma(").count(), 168);
                assert_eq!(new.matches("accum").count() > 48, true);
            }
        }
    }
    #[test]
    fn explicit_hook_preserves_every_nonprompt_shader_and_existing_scalar_oracle() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let new = source(old.clone());
            let finish = |s| {
                if k == Kernel::PromptAttentionFault {
                    prompt_local_fault::source(k, s)
                } else {
                    s
                }
            };
            #[cfg(feature = "strata-qsa-native-xor")]
            let new = super::super::prompt_native_xor::source(new);
            #[cfg(feature = "strata-qsa-scale-finite")]
            let new = super::super::prompt_scale_finite::source(new);
            #[cfg(feature = "strata-qsa-query-planes")]
            let new = super::super::prompt_query_planes::source(new);
            #[cfg(feature = "strata-qsa-head-scatter")]
            let new = super::super::prompt_head_scatter::source(new);
            #[cfg(feature = "strata-qsa-head-frontier")]
            let new = super::super::prompt_head_scatter::frontier_source(new);
            #[cfg(feature = "strata-qsa-prefetch")]
            assert_eq!(k.source(), finish(new));
            #[cfg(not(feature = "strata-qsa-prefetch"))]
            {
                let _ = new;
                let _ = finish;
            }
            assert!(!k.scalar_source().contains("STRATA_QSA_PREFETCH"));
        }
        for k in [
            Kernel::Attention,
            Kernel::MtpAttention,
            Kernel::Merge,
            Kernel::MtpMerge,
        ] {
            assert_eq!(k.source(), k.scalar_source());
        }
    }
    #[test]
    fn missing_prerequisites_or_changed_guard_source_is_rejected() {
        let s = original(Kernel::PromptAttentionFault);
        assert!(std::panic::catch_unwind(|| source(source(s.clone()))).is_err());
        for (old, new) in [
            ("STRATA_QK_COMPACT", "changed"),
            (
                "if row[c]>=p.start+query_row+1u {continue;}",
                "if false {continue;}",
            ),
            (Q, "changed_query_load();\n"),
            ("token0=subgroupBroadcastFirst(row[c0]);", "token0=0u;"),
        ] {
            assert!(std::panic::catch_unwind(|| source(s.replacen(old, new, 1))).is_err());
        }
    }
}
