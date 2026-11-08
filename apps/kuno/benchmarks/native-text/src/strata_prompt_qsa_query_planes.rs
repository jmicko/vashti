//! Split the unchanged two query vec4s into two contiguous lane planes.
//! Both original vectors keep every bit. Only the12KiB SLM permutation changes.
const TAG: &str = "// STRATA_QSA_QUERY_PLANES: disjoint contiguous32-lane vec4 planes.\n";
const STORE: &str = "sq[(vector/64u)*64u+(vector%2u)*32u+(vector%64u)/2u]";
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QSA_PREFETCH") && s.contains("STRATA_QSA_SCALE_FINITE"));
    assert!(s.contains("var<workgroup> sq:array<vec4<f32>,768>;"));
    assert!(!s.contains("STRATA_QSA_QUERY_PLANES"));
    assert_eq!(s.matches("sq[").count(), 27);
    assert_eq!(s.matches("sq[vector]=").count(), 3);
    s = s.replace("sq[vector]", STORE);
    for (old, new) in replacements() {
        assert_eq!(
            s.matches(&old).count(),
            1,
            "query-plane source drift: {old}"
        );
        s = s.replacen(&old, &new, 1);
    }
    assert_eq!(s.matches("sq[").count(), 27);
    s.insert_str(0, TAG);
    s
}
fn replacements() -> Vec<(String, String)> {
    let mut r = vec![
        (
            "prefetch_qa=sq[lane*2u]".into(),
            "prefetch_qa=sq[lane]".into(),
        ),
        (
            "prefetch_qb=sq[lane*2u+1u]".into(),
            "prefetch_qb=sq[32u+lane]".into(),
        ),
    ];
    for h in 1..12 {
        r.push((
            format!("prefetch_qa=sq[{}u+lane*2u]", h * 64),
            format!("prefetch_qa=sq[{}u+lane]", h * 64),
        ));
        r.push((
            format!("prefetch_qb=sq[{}u+lane*2u+1u]", h * 64),
            format!("prefetch_qb=sq[{}u+lane]", h * 64 + 32),
        ));
    }
    r
}
#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_local_fault, prompt_native_xor, prompt_paired,
        prompt_prefetch, prompt_pv_gather, prompt_scale_finite, prompt_specialize, prompt_uniform,
    };
    use super::*;
    fn before(k: Kernel) -> String {
        prompt_scale_finite::source(prompt_native_xor::source(prompt_prefetch::source(
            prompt_compact::source(prompt_uniform::source(prompt_fma::source(
                prompt_pv_gather::source(prompt_paired::source(prompt_specialize::source(
                    k.scalar_source(),
                ))),
            ))),
        )))
    }
    #[test]
    fn exact_inverse_keeps_every_original_load_arithmetic_guard_and_barrier() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            let mut inverse = new.strip_prefix(TAG).unwrap().replace(STORE, "sq[vector]");
            for (a, b) in replacements() {
                assert_eq!(inverse.matches(&b).count(), 1);
                inverse = inverse.replacen(&b, &a, 1);
            }
            assert_eq!(inverse, old);
            assert!(std::panic::catch_unwind(|| source(new.clone())).is_err());
            assert!(
                std::panic::catch_unwind(|| source(old.replace("sq[vector]=", "sq[vector+1u]=")))
                    .is_err()
            );
            assert!(
                std::panic::catch_unwind(|| source(
                    old.replace("prefetch_qb=sq[lane*2u+1u]", "prefetch_qb=sq[lane*2u]")
                ))
                .is_err()
            );
        }
    }
    #[test]
    fn all_768_vector_writes_are_disjoint_and_all_3072_query_bits_are_identical() {
        let mut slm = [[0u32; 4]; 768];
        let mut writes = [0u32; 768];
        let bits = |v: usize, c: usize| {
            [0x80000000, 0x7f800000, 0x7fc12345, 0xffc12345][c]
                ^ (v as u32).wrapping_mul(0x9e3779b9)
        };
        for tid in 0..256 {
            for step in 0..3 {
                let v = tid + step * 256;
                let to = (v / 64) * 64 + (v % 2) * 32 + (v % 64) / 2;
                writes[to] += 1;
                for c in 0..4 {
                    slm[to][c] = bits(v, c);
                }
            }
        }
        assert!(writes.iter().all(|&n| n == 1));
        for h in 0..12 {
            for lane in 0..32 {
                for half in 0..2 {
                    for c in 0..4 {
                        assert_eq!(
                            slm[h * 64 + half * 32 + lane][c],
                            bits(h * 64 + lane * 2 + half, c)
                        );
                    }
                }
            }
        }
        // This is an address-stride proof, not a hardware bank/timing claim.
        for lane in 0..31 {
            assert_eq!(((lane + 1) * 16) - (lane * 16), 16);
        }
    }
    #[test]
    fn actual_modules_keep_full_abi_geometry_memory_counts_and_selected_composition() {
        fn ops(w: &[u32], op: u32) -> usize {
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
            let finish = |s| {
                if k == Kernel::PromptAttentionFault {
                    prompt_local_fault::source(k, s)
                } else {
                    s
                }
            };
            let old = finish(before(k));
            let new = finish(source(before(k)));
            let (a, aw) = prompt_specialize::tests::compiled(&old);
            let (b, bw) = prompt_specialize::tests::compiled(&new);
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
            for op in [61, 62, 133, 224, 246, 338, 345, 346, 366] {
                assert_eq!(ops(&aw, op), ops(&bw, op), "opcode{op}");
            }
            #[cfg(feature = "strata-qsa-query-planes")]
            {
                #[cfg(feature = "strata-qsa-head-scatter")]
                let expected=super::super::prompt_head_scatter::source(source(before(k)));
                #[cfg(feature="strata-qsa-head-frontier")]
                let expected=super::super::prompt_head_scatter::frontier_source(expected);
                #[cfg(feature="strata-qsa-head-scatter")]
                let expected=finish(expected);
                #[cfg(not(feature = "strata-qsa-head-scatter"))]
                let expected=new.clone();
                assert_eq!(k.source(), expected);
            }
        }
    }
}
