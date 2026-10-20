//! Opt-in prompt QK cell pairing. No subgroup, owner or precision substitution.
//!
//! The selected SYCL reference uses the same eight-product dot and XOR tree.
//! Its separate experimental pre75 NC=2 path demonstrates query reuse across
//! two cells; this module applies only that reuse to our existing SIMD32 path.
//! The causal selection, score/stat ownership, softmax, chronological PV FMAs
//! and closing fault/status code remain the original generated source.

pub(super) fn source(mut original: String) -> String {
    assert!(original.contains("// STRATA_ROUND_PRODUCTS:"));
    assert!(original.contains("var<workgroup> sq:array<vec4<f32>,768>;"));
    assert!(original.contains("let query_row=t+p.reserved;let n=cells(query_row,p.start);"));
    let begin = "for(var c=warp;c<64u;c+=8u) {";
    let ending = "\n    }\n    workgroupBarrier();\n    for(var h=warp;h<12u;h+=8u) {";
    assert_eq!(original.matches(begin).count(), 1, "paired QK source drift");
    assert_eq!(
        original.matches(ending).count(),
        1,
        "paired QK closing drift"
    );
    let first = original.find(begin).unwrap();
    let last = first + original[first..].find(ending).unwrap() + "\n    }".len();
    let status = original[first..last].contains("status[");

    let mut paired = String::from("for(var pair=warp;pair<64u;pair+=16u) {\n");
    for cell in 0..2 {
        paired.push_str(&format!(
            "let c{cell}=pair+{}u;var token{cell}=CONTEXT_CAPACITY;var valid{cell}=false;\n\
             if c{cell}<here {{token{cell}=row[c{cell}];\n\
             if token{cell}<p.start+query_row+1u {{valid{cell}=true;}}\n\
             else {{atomicOr(&broken,1u);}}}}\n\
             if !valid{cell} && lane<12u {{sp[lane*64u+c{cell}]=-3.4028234663852886e38;}}\n",
            cell * 8
        ));
        for component in 0..8 {
            paired.push_str(&format!("var k{cell}_{component}=0.0;\n"));
        }
        paired.push_str(&format!(
            "if valid{cell} {{\n\
             let koff=kvh*256u+lane*8u;\n\
             let kword0=cache[token{cell}*CACHE_WORDS+koff/4u];\n\
             let kword1=cache[token{cell}*CACHE_WORDS+koff/4u+1u];\n\
             let kscales=unpack2x16float(cache[token{cell}*CACHE_WORDS+256u+koff/128u]);\n\
             let kscale=select(kscales.x,kscales.y,((koff/64u)&1u)!=0u);\n"
        ));
        for component in 0..8 {
            paired.push_str(&format!(
                "k{cell}_{component}=f32(signed_code(kword{},{}u))*kscale;\n\
                 if !finite(k{cell}_{component}) {{atomicOr(&broken,1u);}}\n",
                component / 4,
                component % 4
            ));
        }
        paired.push_str("}\n");
    }
    // Cell validity is identical across each logical32-lane subgroup: both
    // indices, row IDs and causal bounds are subgroup-uniform. Hoist it once
    // for the common full-pair case; the guarded body below remains the exact
    // original paired tail/fault path, including each status publication.
    paired.push_str("if valid0 && valid1 {\n");
    paired.push_str(&heads(status, false));
    paired.push_str("} else {\n");
    paired.push_str(&heads(status, true));
    paired.push_str("}\n}");
    original.replace_range(first..last, &paired);
    original
}

fn heads(status: bool, guarded: bool) -> String {
    let mut body = String::new();
    for head in 0..12 {
        body.push_str(&format!(
            "{{let h={head}u;let off=h*256u+lane*8u;\n\
             let qa=sq[off/4u];let qb=sq[off/4u+1u];\n"
        ));
        for cell in 0..2 {
            if guarded {
                body.push_str(&format!("if valid{cell} {{var dot=k{cell}_0*qa.x;\n"));
            } else {
                body.push_str(&format!("{{var dot=k{cell}_0*qa.x;\n"));
            }
            for component in 1..8 {
                let vector = if component < 4 { "qa" } else { "qb" };
                let field = ["x", "y", "z", "w"][component % 4];
                body.push_str(&format!("dot+=k{cell}_{component}*{vector}.{field};\n"));
            }
            for distance in [16, 8, 4, 2, 1] {
                body.push_str(&format!("dot+=subgroupShuffleXor(dot,{distance}u);\n"));
            }
            body.push_str(&format!(
                "if lane==0u {{let score=dot/16.0;sp[h*64u+c{cell}]=score;\n"
            ));
            if status {
                body.push_str(&format!(
                    "status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c{cell}]=select(0u,1u,finite(score));\n"
                ));
            }
            body.push_str("if !finite(score) {atomicOr(&broken,1u);}\n}\n}\n");
        }
        body.push_str("}\n");
    }
    body
}

#[cfg(test)]
mod tests {
    use super::super::{Kernel, prompt_specialize};

    fn base(kernel: Kernel) -> String {
        prompt_specialize::source(kernel.scalar_source())
    }

    #[test]
    fn actual_paired_source_lowers_without_new_shared_state_or_precision_change() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let (original, old_words) = prompt_specialize::tests::compiled(&base(kernel));
            let (paired, new_words) =
                prompt_specialize::tests::compiled(&super::source(base(kernel)));
            assert_eq!(original.entry_points[0].workgroup_size, [256, 1, 1]);
            assert_eq!(paired.entry_points[0].workgroup_size, [256, 1, 1]);
            let shared = |m: &wgpu::naga::Module| {
                m.global_variables
                    .iter()
                    .filter(|(_, v)| v.space == wgpu::naga::AddressSpace::WorkGroup)
                    .count()
            };
            assert_eq!(shared(&original), 4);
            assert_eq!(shared(&paired), 4);
            let count = |words: &[u32], opcode: u32| {
                let mut at = 5;
                let mut found = 0;
                while at < words.len() {
                    let size = (words[at] >> 16) as usize;
                    assert!(size != 0 && at + size <= words.len());
                    found += usize::from(words[at] & 65535 == opcode);
                    at += size;
                }
                found
            };
            assert_eq!(count(&old_words, 224), 4); // OpControlBarrier.
            assert_eq!(count(&new_words, 224), 4);
            assert_eq!(count(&old_words, 246), count(&new_words, 246)); // OpLoopMerge.
            let fmas = |words: &[u32]| {
                let mut at = 5;
                let mut found = 0;
                while at < words.len() {
                    let size = (words[at] >> 16) as usize;
                    found +=
                        usize::from(words[at] & 65535 == 12 && size >= 5 && words[at + 4] == 50);
                    at += size;
                }
                found
            };
            assert_eq!(fmas(&old_words), 12);
            assert_eq!(fmas(&new_words), 12);
            let mut products = std::collections::BTreeSet::new();
            let mut precise = std::collections::BTreeSet::new();
            let mut at = 5;
            while at < new_words.len() {
                let size = (new_words[at] >> 16) as usize;
                match new_words[at] & 65535 {
                    133 => {
                        products.insert(new_words[at + 2]);
                    }
                    71 if size == 3 && new_words[at + 2] == 42 => {
                        precise.insert(new_words[at + 1]);
                    }
                    _ => {}
                }
                at += size;
            }
            assert!(!products.is_empty());
            assert_eq!(products, precise);
        }
    }

    #[test]
    fn all_cells_and_short_tails_have_one_original_owner() {
        for here in 0..=64 {
            let mut visits = [0; 64];
            let mut live = [0; 64];
            for warp in 0..8 {
                for pair in (warp..64).step_by(16) {
                    for cell in [pair, pair + 8] {
                        assert!(cell < 64);
                        visits[cell] += 1;
                        if cell < here {
                            live[cell] += 1;
                        }
                    }
                }
            }
            assert!(visits.into_iter().all(|count| count == 1));
            for (cell, count) in live.into_iter().enumerate() {
                assert_eq!(count, usize::from(cell < here));
            }
        }
    }

    #[test]
    fn original_scope_owners_barriers_softmax_pv_and_closing_are_unchanged() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = base(kernel);
            let new = super::source(old.clone());
            let first = old.find("for(var c=warp;c<64u;c+=8u) {").unwrap();
            assert_eq!(&old[..first], &new[..first]);
            let tail = "\n    workgroupBarrier();\n    for(var h=warp;h<12u;h+=8u) {";
            assert_eq!(
                old.split_once(tail).unwrap().1,
                new.split_once(tail).unwrap().1
            );
            assert_eq!(old.matches("workgroupBarrier();").count(), 4);
            assert_eq!(new.matches("workgroupBarrier();").count(), 4);
            assert_eq!(
                old.matches("var<workgroup>").count(),
                new.matches("var<workgroup>").count()
            );
            assert_eq!(
                new.matches("let qa=sq[off/4u];let qb=sq[off/4u+1u];")
                    .count(),
                24
            );
            assert_eq!(new.matches("let score=dot/16.0;").count(), 48);
            assert_eq!(new.matches("dot+=subgroupShuffleXor(dot,16u);").count(), 48);
            assert_eq!(new.matches("if !finite(score)").count(), 48);
            let full = super::heads(kernel == Kernel::PromptAttention, false);
            let fallback = super::heads(kernel == Kernel::PromptAttention, true);
            assert!(!full.contains("if valid"));
            assert_eq!(fallback.matches("if valid0 {").count(), 12);
            assert_eq!(fallback.matches("if valid1 {").count(), 12);
            assert!(new.contains(&format!(
                "if valid0 && valid1 {{\n{full}}} else {{\n{fallback}}}"
            )));
            assert_eq!(new.matches("if valid0 && valid1 {").count(), 1);
            for cell in 0..2 {
                assert!(new.contains(&format!("if c{cell}<here {{token{cell}=row[c{cell}];")));
                assert!(new.contains(&format!(
                    "if token{cell}<p.start+query_row+1u {{valid{cell}=true;}}"
                )));
                assert_eq!(new.matches(&format!("if !finite(k{cell}_")).count(), 8);
                for head in 0..12 {
                    assert_eq!(
                        new.matches(&format!(
                            "accum{head}=fma(sp[{head}u*64u+c],v,accum{head});"
                        ))
                        .count(),
                        1
                    );
                }
            }
            if kernel == Kernel::PromptAttentionFault {
                assert!(!new.contains("status["));
                assert!(new.contains("if tid==0u{atomicOr(&fault[0],1u);}"));
            } else {
                assert_eq!(new.matches("select(0u,1u,finite(score))").count(), 48);
            }
            #[cfg(feature = "strata-qsa-paired")]
            assert_eq!(kernel.source(), new);
            #[cfg(not(feature = "strata-qsa-paired"))]
            assert_eq!(kernel.source(), old);
        }
    }

    #[test]
    fn every_dot_retains_its_eight_products_and_original_xor_order() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let new = super::source(base(kernel));
            for cell in 0..2 {
                let mut dot = format!("var dot=k{cell}_0*qa.x;\n");
                for component in 1..8 {
                    let vector = if component < 4 { "qa" } else { "qb" };
                    let field = ["x", "y", "z", "w"][component % 4];
                    dot.push_str(&format!("dot+=k{cell}_{component}*{vector}.{field};\n"));
                }
                for distance in [16, 8, 4, 2, 1] {
                    dot.push_str(&format!("dot+=subgroupShuffleXor(dot,{distance}u);\n"));
                }
                assert_eq!(new.matches(&dot).count(), 24);
            }
        }
    }
}
