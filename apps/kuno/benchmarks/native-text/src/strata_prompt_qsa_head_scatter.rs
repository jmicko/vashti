//! Ordered reduce-scatter of twelve original head sums, in Rust-generated WGSL.
//! MIT Strata fb58e0's experimental reduce12 motivates the exchange topology.
//! Unlike its commutative spelling, order every addition exactly as lane zero.
const TAG: &str = "// STRATA_QSA_HEAD_SCATTER: exact lane-zero ordered twelve-head reductions.\n";
const FIRST: &str = "{let h=0u;let off=h*256u+lane*8u;\n";
const END: &str = "}\n    workgroupBarrier();\n    for(var h=warp;h<12u;h+=8u) {";
fn dots(cell: usize) -> String {
    let mut s = format!("if valid{cell} {{var dot=k{cell}_0*qa.x;\n");
    for j in 1..8 {
        let v = if j < 4 { "qa" } else { "qb" };
        let c = ["x", "y", "z", "w"][j % 4];
        s.push_str(&format!("dot=fma(k{cell}_{j},{v}.{c},dot);\n"));
    }
    s
}
fn old_finish(cell: usize, status: bool) -> String {
    let mut s = String::new();
    for d in [16, 8, 4, 2, 1] {
        s.push_str(&format!("dot+=subgroupShuffleXor(dot,{d}u);\n"));
    }
    s.push_str(&format!("let score=subgroupBroadcastFirst(dot/16.0);\nif !finite(score) {{atomicOr(&broken,1u);}}\nif lane==0u {{sp[h*65u+c{cell}]=score;\n"));
    if status {
        s.push_str(&format!("status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c{cell}]=select(0u,1u,finite(score));\n"));
    }
    s.push_str("}\n}\n");
    s
}
fn declarations() -> String {
    let mut s = String::new();
    for c in 0..2 {
        for h in 0..12 {
            s.push_str(&format!("var scatter{c}_{h}=0.0;\n"));
        }
    }
    s
}
fn reduction(cell: usize, status: bool) -> String {
    let mut s = format!("if valid{cell} {{\n");
    let mut names: Vec<String> = (0..16)
        .map(|h| {
            if h < 12 {
                format!("scatter{cell}_{h}")
            } else {
                "0.0".into()
            }
        })
        .collect();
    for (stage, distance) in [16, 8, 4, 2].into_iter().enumerate() {
        let n = names.len() / 2;
        for i in 0..n {
            // Peer chooses the opposite logical head BEFORE XOR exchanges.
            // Reorder its pair AFTER exchange to preserve lane-zero operand order,
            // including signed zero and nonfinite payloads, not just finite sums.
            s.push_str(&format!("let hi{stage}_{i}=(lane&{distance}u)!=0u;\nlet own{stage}_{i}=select({}, {}, hi{stage}_{i});\nlet peer{stage}_{i}=subgroupShuffleXor(select({}, {}, hi{stage}_{i}),{distance}u);\nlet r{stage}_{i}=select(own{stage}_{i},peer{stage}_{i},hi{stage}_{i})+select(peer{stage}_{i},own{stage}_{i},hi{stage}_{i});\n",names[i],names[i+n],names[i+n],names[i]));
        }
        names = (0..n).map(|i| format!("r{stage}_{i}")).collect();
    }
    s.push_str("let hi4=(lane&1u)!=0u;\nlet peer4=subgroupShuffleXor(r3_0,1u);\nlet ordered=select(r3_0,peer4,hi4)+select(peer4,r3_0,hi4);\nlet h=lane/2u;\nif h<12u {let score=ordered/16.0;\nif !finite(score) {atomicOr(&broken,1u);}\nif (lane&1u)==0u {\n");
    s.push_str(&format!("sp[h*65u+c{cell}]=score;\n"));
    if status {
        s.push_str(&format!("status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c{cell}]=select(0u,1u,finite(score));\n"));
    }
    s.push_str("}\n}\n}\n");
    s
}
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QSA_QUERY_PLANES"));
    assert!(!s.contains("STRATA_QSA_HEAD_SCATTER"));
    assert_eq!(s.matches(FIRST).count(), 1);
    assert_eq!(s.matches(END).count(), 1);
    let status = s.contains("status[");
    // Each original body appears once per head and cell, in chronological order.
    for c in 0..2 {
        let old = format!("{}{}", dots(c), old_finish(c, status));
        assert_eq!(s.matches(&old).count(), 12, "ordered head source drift");
        for h in 0..12 {
            let new = format!("{}scatter{c}_{h}=dot;\n}}\n", dots(c));
            s = s.replacen(&old, &new, 1);
        }
    }
    s = s.replacen(FIRST, &format!("{}{FIRST}", declarations()), 1);
    s = s.replacen(
        END,
        &format!("{}{}{END}", reduction(0, status), reduction(1, status)),
        1,
    );
    s.insert_str(0, TAG);
    s
}
#[cfg(test)]
fn restore(s: String) -> String {
    let mut s = s.strip_prefix(TAG).unwrap().to_owned();
    let status = s.contains("status[");
    for c in 0..2 {
        let r = reduction(c, status);
        assert_eq!(s.matches(&r).count(), 1);
        s = s.replacen(&r, "", 1);
        for h in 0..12 {
            let new = format!("{}scatter{c}_{h}=dot;\n}}\n", dots(c));
            assert_eq!(s.matches(&new).count(), 1);
            s = s.replacen(&new, &format!("{}{}", dots(c), old_finish(c, status)), 1);
        }
    }
    let d = declarations();
    assert_eq!(s.matches(&d).count(), 1);
    s = s.replacen(&d, "", 1);
    s
}
#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_local_fault, prompt_native_xor, prompt_paired,
        prompt_prefetch, prompt_pv_gather, prompt_query_planes, prompt_scale_finite,
        prompt_specialize, prompt_uniform,
    };
    use super::*;
    pub(super) fn before(k: Kernel) -> String {
        let s = prompt_specialize::source(k.scalar_source());
        let s = prompt_paired::source(s);
        let s = prompt_pv_gather::source(s);
        let s = prompt_fma::source(s);
        let s = prompt_uniform::source(s);
        let s = prompt_compact::source(s);
        let s = prompt_prefetch::source(s);
        let s = prompt_native_xor::source(s);
        let s = prompt_scale_finite::source(s);
        prompt_query_planes::source(s)
    }
    fn scatter<T: Clone>(p: &[Vec<T>], add: impl Fn(&T, &T) -> T) -> Vec<T> {
        let mut r = p.to_vec();
        for d in [16, 8, 4, 2] {
            let n = r.len() / 2;
            let mut next = Vec::new();
            for h in 0..n {
                let own: Vec<_> = (0..32)
                    .map(|l| r[if l & d != 0 { h + n } else { h }][l].clone())
                    .collect();
                let peer: Vec<_> = (0..32)
                    .map(|l| r[if (l ^ d) & d != 0 { h } else { h + n }][l ^ d].clone())
                    .collect();
                next.push(
                    (0..32)
                        .map(|l| {
                            if l & d != 0 {
                                add(&peer[l], &own[l])
                            } else {
                                add(&own[l], &peer[l])
                            }
                        })
                        .collect(),
                );
            }
            r = next;
        }
        (0..32)
            .map(|l| {
                if l & 1 != 0 {
                    add(&r[0][l ^ 1], &r[0][l])
                } else {
                    add(&r[0][l], &r[0][l ^ 1])
                }
            })
            .collect()
    }
    fn original<T: Clone>(p: &[T], add: impl Fn(&T, &T) -> T) -> T {
        let mut v = p.to_vec();
        for d in [16, 8, 4, 2, 1] {
            let old = v.clone();
            for l in 0..32 {
                v[l] = add(&old[l], &old[l ^ d]);
            }
        }
        v[0].clone()
    }
    #[test]
    fn every_head_and_lane_has_the_exact_original_ordered_expression_tree() {
        let p: Vec<Vec<String>> = (0..16)
            .map(|h| (0..32).map(|l| format!("h{h}l{l}")).collect())
            .collect();
        let plus = |a: &String, b: &String| format!("({a}+{b})");
        let got = scatter(&p, plus);
        for h in 0..16 {
            for l in [h * 2, h * 2 + 1] {
                assert_eq!(got[l], original(&p[h], plus));
            }
        }
    }
    #[test]
    fn ieee_bits_and_all_per_head_faults_follow_original_lane_zero() {
        let bits = [
            0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff,
            0x7f800000, 0xff800000, 0x7fc12345, 0xffc56789,
        ];
        for seed in 0..96 {
            let mut p = vec![vec![0.0f32; 32]; 16];
            for h in 0..12 {
                for l in 0..32 {
                    p[h][l] = f32::from_bits(bits[(h * 17 + l * 3 + seed) % bits.len()]);
                }
            }
            let got = scatter(&p, |a, b| *a + *b);
            for h in 0..12 {
                let want = original(&p[h], |a, b| *a + *b);
                for l in [h * 2, h * 2 + 1] {
                    assert_eq!(got[l].to_bits(), want.to_bits());
                }
            }
            for prior in [0u32, 1, 128, 129] {
                let old = (0..12).fold(prior, |v, h| {
                    v | u32::from(!(original(&p[h], |a, b| *a + *b) / 16.0).is_finite())
                });
                let new = (0..24).fold(prior, |v, l| v | u32::from(!(got[l] / 16.0).is_finite()));
                assert_eq!(old, new);
            }
        }
    }
    #[test]
    fn whole_source_inverse_preserves_all_original_loads_fmas_guards_and_closing() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            assert_eq!(restore(new.clone()), old);
            assert!(std::panic::catch_unwind(|| source(new)).is_err());
            assert!(
                std::panic::catch_unwind(|| source(old.replace(
                    "dot+=subgroupShuffleXor(dot,16u);",
                    "dot+=subgroupShuffleXor(dot,8u);"
                )))
                .is_err()
            );
            assert_eq!(
                old.matches("workgroupBarrier();").count(),
                source(old).matches("workgroupBarrier();").count()
            );
        }
    }
    #[test]
    fn actual_spirv_keeps_abi_math_and_geometry_with_42_instead_of_130_exchanges() {
        fn count(w: &[u32], op: u32) -> usize {
            let mut at = 5;
            let mut n = 0;
            while at < w.len() {
                let k = (w[at] >> 16) as usize;
                assert!(k > 0 && at + k <= w.len());
                n += usize::from(w[at] & 65535 == op);
                at += k;
            }
            n
        }
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            let finish = |s| {
                if k == Kernel::PromptAttentionFault {
                    prompt_local_fault::source(k, s)
                } else {
                    s
                }
            };
            let (a, aw) = prompt_specialize::tests::compiled(&finish(old));
            let (b, bw) = prompt_specialize::tests::compiled(&finish(new.clone()));
            let globals = |m: &wgpu::naga::Module| {
                m.global_variables
                    .iter()
                    .map(|(_, g)| {
                        (
                            g.binding.clone(),
                            g.space,
                            format!("{:?}", m.types[g.ty].inner),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(globals(&a), globals(&b));
            assert_eq!(a.entry_points[0].workgroup_size, [256, 1, 1]);
            assert_eq!(
                a.entry_points[0].workgroup_size,
                b.entry_points[0].workgroup_size
            );
            for op in [12, 133, 224, 246, 345] {
                assert_eq!(count(&aw, op), count(&bw, op), "opcode{op}");
            }
            assert_eq!(count(&aw, 338), count(&bw, 338) + 24);
            assert_eq!(count(&aw, 346) + count(&aw, 366), 130);
            assert_eq!(count(&bw, 346) + count(&bw, 366), 42);
            #[cfg(feature = "strata-qsa-head-frontier")]
            let new = super::frontier_source(new);
            #[cfg(feature = "strata-qsa-head-scatter")]
            assert_eq!(k.source(), finish(new));
        }
    }
}

#[cfg(any(test, feature = "strata-qsa-head-frontier"))]
#[path = "strata_prompt_qsa_head_frontier.rs"]
mod frontier;
#[cfg(any(test, feature = "strata-qsa-head-frontier"))]
pub(super) fn frontier_source(s: String) -> String {
    frontier::source(s)
}
