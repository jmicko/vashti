//! Keep uniform metadata and full prefetch, but preserve constant XOR as XOR.
//! General lane lookup made Mesa emit indirect register addressing for the
//! original butterfly. No arithmetic, participant, tree or fault-policy change.
const TAG: &str = "// STRATA_QSA_NATIVE_XOR: original constant butterfly operations.\n";
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QSA_PREFETCH"));
    assert!(s.contains("STRATA_QK_UNIFORM_METADATA"));
    assert!(!s.contains("STRATA_QSA_NATIVE_XOR"));
    for value in ["dot", "maximum", "total"] {
        for d in [16, 8, 4, 2, 1] {
            assert_eq!(s.matches(&format!("let qk_lane{d}=lane^{d}u;")).count(), 1);
            let old = format!("subgroupShuffle({value},qk_lane{d})");
            assert_eq!(s.matches(&old).count(), if value == "dot" { 24 } else { 1 });
            s = s.replace(&old, &format!("subgroupShuffleXor({value},{d}u)"));
        }
    }
    s.insert_str(0, TAG);
    s
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_paired, prompt_prefetch, prompt_pv_gather,
        prompt_specialize, prompt_uniform,
    };
    use super::*;
    fn before(k: Kernel) -> String {
        prompt_prefetch::source(prompt_compact::source(prompt_uniform::source(
            prompt_fma::source(prompt_pv_gather::source(prompt_paired::source(
                prompt_specialize::source(k.scalar_source()),
            ))),
        )))
    }
    #[test]
    fn exact_inverse_preserves_all_memory_arithmetic_faults_and_uniform_metadata() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            let mut restored = new.strip_prefix(TAG).unwrap().to_owned();
            for value in ["dot", "maximum", "total"] {
                for d in [16, 8, 4, 2, 1] {
                    restored = restored.replace(
                        &format!("subgroupShuffleXor({value},{d}u)"),
                        &format!("subgroupShuffle({value},qk_lane{d})"),
                    );
                }
            }
            assert_eq!(restored, old);
        }
    }
    fn count(w: &[u32], op: u32) -> usize {
        let mut at = 5;
        let mut n = 0;
        while at < w.len() {
            let step = (w[at] >> 16) as usize;
            assert!(step > 0 && at + step <= w.len());
            n += usize::from(w[at] & 65535 == op);
            at += step;
        }
        n
    }
    #[test]
    fn actual_spirv_preserves_abi_geometry_barriers_and_only_changes_130_exact_shuffles() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            let (a, aw) = prompt_specialize::tests::compiled(&old);
            let (b, bw) = prompt_specialize::tests::compiled(&new);
            assert_eq!(
                a.entry_points[0].workgroup_size,
                b.entry_points[0].workgroup_size
            );
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
            assert_eq!(count(&aw, 224), count(&bw, 224));
            assert_eq!(count(&aw, 345), 178);
            assert_eq!(count(&bw, 345), 48);
            assert_eq!(count(&aw, 346), 0);
            assert_eq!(
                count(&bw, 346),
                if cfg!(feature = "strata-subgroup-quad") {
                    78
                } else {
                    130
                }
            );
            assert_eq!(
                count(&bw, 366),
                if cfg!(feature = "strata-subgroup-quad") {
                    52
                } else {
                    0
                }
            );
            assert_eq!(count(&aw, 338), count(&bw, 338));
            assert_eq!(count(&aw, 133), count(&bw, 133));
            // The exact inverse above retains all FMAs and precise product tags.
            assert_eq!(new.matches("dot=fma(").count(), 168);
        }
    }
    #[test]
    fn source_drift_and_duplicate_hook_are_rejected_before_generation() {
        let old = before(Kernel::PromptAttentionFault);
        assert!(std::panic::catch_unwind(|| source(source(old.clone()))).is_err());
        assert!(
            std::panic::catch_unwind(|| source(
                old.replace("qk_lane16=lane^16u", "qk_lane16=lane^8u")
            ))
            .is_err()
        );
    }
}
