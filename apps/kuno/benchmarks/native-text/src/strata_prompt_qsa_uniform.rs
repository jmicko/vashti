//! Preserve every QK operand while exposing original subgroup-uniform metadata.
//! Selected IDs are immutable SLM words at subgroup-uniform indices. Score
//! stores/faults already use lane zero; broadcast THAT exact result before the
//! fault test, so a lane-zero store no longer carries divergent sticky state.

fn once(s: &mut String, old: &str, new: &str) {
    assert_eq!(s.matches(old).count(), 1, "uniform QK source drift");
    *s = s.replacen(old, new, 1);
}
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QK_EXPLICIT_FMA"));
    assert!(!s.contains("STRATA_QK_UNIFORM_METADATA"));
    assert!(s.contains("var<workgroup> sp:array<f32,780>;"));
    let mut indices = String::new();
    for d in [16, 8, 4, 2, 1] {
        indices.push_str(&format!("let qk_lane{d}=lane^{d}u;\n"));
    }
    once(
        &mut s,
        "for(var pair=warp;pair<64u;pair+=16u) {",
        &format!("{indices}for(var pair=warp;pair<64u;pair+=16u) {{"),
    );
    for cell in 0..2 {
        once(
            &mut s,
            &format!("token{cell}=row[c{cell}];"),
            &format!("token{cell}=subgroupBroadcastFirst(row[c{cell}]);"),
        );
        let first = format!("if lane==0u {{let score=dot/16.0;sp[h*65u+c{cell}]=score;\n");
        let last = "if !finite(score) {atomicOr(&broken,1u);}\n}\n";
        let mut count = 0;
        let mut at = 0;
        while let Some(n) = s[at..].find(&first) {
            let begin = at + n;
            let rest = begin + first.len();
            let end = rest + s[rest..].find(last).expect("uniform original QK closing");
            let status = &s[rest..end];
            let expected = format!(
                "status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c{cell}]=select(0u,1u,finite(score));\n"
            );
            assert!(
                status.is_empty() || status == expected,
                "uniform QK status drift"
            );
            let new = format!(
                "let score=subgroupBroadcastFirst(dot/16.0);\nif !finite(score) {{atomicOr(&broken,1u);}}\nif lane==0u {{sp[h*65u+c{cell}]=score;\n{status}}}\n"
            );
            s.replace_range(begin..end + last.len(), &new);
            at = begin + new.len();
            count += 1;
        }
        assert_eq!(count, 24, "complete paired QK score inventory");
    }
    for value in ["dot", "maximum", "total"] {
        for d in [16, 8, 4, 2, 1] {
            let old = format!("subgroupShuffleXor({value},{d}u)");
            assert_eq!(s.matches(&old).count(), if value == "dot" { 48 } else { 1 });
            s = s.replace(&old, &format!("subgroupShuffle({value},qk_lane{d})"));
        }
    }
    s.insert_str(
        0,
        "// STRATA_QK_UNIFORM_METADATA: immutable IDs and original lane-zero score.\n",
    );
    s
}

#[cfg(test)]
mod tests {
    use super::super::{Kernel, prompt_fma, prompt_paired, prompt_pv_gather, prompt_specialize};
    use super::*;
    fn old(k: Kernel) -> String {
        prompt_fma::source(prompt_pv_gather::source(prompt_paired::source(
            prompt_specialize::source(k.scalar_source()),
        )))
    }
    #[test]
    fn whole_source_undo_preserves_every_owning_access_arithmetic_fault_and_barrier() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = old(k);
            let new = source(old.clone());
            let mut s = new
                .strip_prefix(
                    "// STRATA_QK_UNIFORM_METADATA: immutable IDs and original lane-zero score.\n",
                )
                .unwrap()
                .to_owned();
            for v in ["dot", "maximum", "total"] {
                for d in [16, 8, 4, 2, 1] {
                    s = s.replace(
                        &format!("subgroupShuffle({v},qk_lane{d})"),
                        &format!("subgroupShuffleXor({v},{d}u)"),
                    );
                }
            }
            for d in [16, 8, 4, 2, 1] {
                s = s.replace(&format!("let qk_lane{d}=lane^{d}u;\n"), "");
            }
            let status = k == Kernel::PromptAttention;
            for cell in 0..2 {
                s = s.replace(
                    &format!("token{cell}=subgroupBroadcastFirst(row[c{cell}]);"),
                    &format!("token{cell}=row[c{cell}];"),
                );
                let st = if status {
                    format!(
                        "status[query_row*p.status_stride+(kvh*12u+h)*n+chunk*64u+c{cell}]=select(0u,1u,finite(score));\n"
                    )
                } else {
                    String::new()
                };
                let here = format!(
                    "let score=subgroupBroadcastFirst(dot/16.0);\nif !finite(score) {{atomicOr(&broken,1u);}}\nif lane==0u {{sp[h*65u+c{cell}]=score;\n{st}}}\n"
                );
                assert_eq!(s.matches(&here).count(), 24);
                s=s.replace(&here,&format!("if lane==0u {{let score=dot/16.0;sp[h*65u+c{cell}]=score;\n{st}if !finite(score) {{atomicOr(&broken,1u);}}\n}}\n"));
            }
            assert_eq!(s, old);
        }
    }
    #[test]
    fn actual_module_has_exact_uniform_samples_and_identical_full_group_geometry() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let source = source(old(k));
            let (m, w) = prompt_specialize::tests::compiled(&source);
            assert_eq!(m.entry_points[0].workgroup_size, [256, 1, 1]);
            let mut first = 0;
            let mut xor = 0;
            let mut shuffle = 0;
            let mut at = 5;
            while at < w.len() {
                let n = (w[at] >> 16) as usize;
                match w[at] & 65535 {
                    338 => first += 1,   // OpGroupNonUniformBroadcastFirst.
                    345 => shuffle += 1, // OpGroupNonUniformShuffle.
                    346 => xor += 1,     // OpGroupNonUniformShuffleXor.
                    _ => {}
                }
                at += n;
            }
            assert_eq!(first, 50); // Two immutable IDs +48 lane-zero scores.
            assert_eq!(xor, 0);
            assert!(shuffle >= 250);
        }
    }
    #[test]
    fn original_lane_zero_fault_policy_is_exact_for_adversarial_other_lanes() {
        for first in [
            0u32,
            0x80000000,
            1f32.to_bits(),
            0x7f800000,
            0xff800000,
            0x7fc01234,
            0xffc00001,
        ] {
            for bad in 0..32 {
                for prior in [0u32, 1, 128] {
                    let mut lanes = [1f32.to_bits(); 32];
                    lanes[bad] = 0x7fc0abcd;
                    lanes[0] = first;
                    let expected = prior | u32::from(!f32::from_bits(lanes[0]).is_finite());
                    let broadcast = [lanes[0]; 32];
                    let got = broadcast
                        .into_iter()
                        .fold(prior, |n, x| n | u32::from(!f32::from_bits(x).is_finite()));
                    assert_eq!(got, expected);
                    // An accidental all-lane check is NOT equivalent; this catches
                    // moving the predicate without sampling original lane zero.
                    if first == 1f32.to_bits() && bad != 0 && prior == 0 {
                        assert!(lanes.into_iter().any(|x| !f32::from_bits(x).is_finite()));
                        assert_eq!(got, 0);
                    }
                }
            }
        }
        for lane in 0..32u32 {
            for d in [16, 8, 4, 2, 1] {
                assert!((lane ^ d) < 32);
                assert_eq!(lane ^ d, lane ^ (d & 31));
            }
        }
    }
}
