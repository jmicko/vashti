//! Explicit QK contraction as observed in the pinned Strata INT8 AOT image.
//! Keep each lane's first product, seven left-to-right accumulation steps,
//! complete XOR tree, dequantization rounding and chronological PV FMAs.
//! Legacy generators and the independent scalar numerical reference stay intact.

pub(super) fn source(mut source: String) -> String {
    assert!(source.contains("STRATA_ROUND_PRODUCTS"));
    assert!(!source.contains("STRATA_QK_EXPLICIT_FMA"));
    let paired = source.contains("var dot=k0_0*qa.x;");
    let prefixes: &[&str] = if paired { &["k0_", "k1_"] } else { &["k"] };
    let per_operand = if paired { 24 } else { 12 };
    for prefix in prefixes {
        assert_eq!(
            source.matches(&format!("var dot={prefix}0*qa.x;")).count(),
            per_operand
        );
        for j in 1..8 {
            let vector = if j < 4 { "qa" } else { "qb" };
            let component = ["x", "y", "z", "w"][j % 4];
            let old = format!("dot+={prefix}{j}*{vector}.{component};");
            assert_eq!(
                source.matches(&old).count(),
                per_operand,
                "actual QK FMA generator drift"
            );
            source = source.replace(
                &old,
                &format!("dot=fma({prefix}{j},{vector}.{component},dot);"),
            );
        }
    }
    source.insert_str(
        0,
        "// STRATA_QK_EXPLICIT_FMA: pinned INT8 AOT chronological QK contraction.\n",
    );
    source
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_local_fault, prompt_paired, prompt_pv_gather, prompt_specialize,
    };
    use super::*;
    fn versions(k: Kernel) -> Vec<String> {
        let base = prompt_specialize::source(k.scalar_source());
        let pair = prompt_paired::source(base.clone());
        vec![
            base.clone(),
            pair.clone(),
            prompt_pv_gather::source(base),
            prompt_pv_gather::source(pair),
        ]
    }
    #[test]
    fn only_the_seven_chronological_qk_accumulations_change_for_every_existing_variant() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            for old in versions(k) {
                let new = source(old.clone());
                let mut restored=new.strip_prefix("// STRATA_QK_EXPLICIT_FMA: pinned INT8 AOT chronological QK contraction.\n").unwrap().to_owned();
                for prefix in ["k", "k0_", "k1_"] {
                    for j in 1..8 {
                        let v = if j < 4 { "qa" } else { "qb" };
                        let c = ["x", "y", "z", "w"][j % 4];
                        restored = restored.replace(
                            &format!("dot=fma({prefix}{j},{v}.{c},dot);"),
                            &format!("dot+={prefix}{j}*{v}.{c};"),
                        );
                    }
                }
                assert_eq!(restored, old);
                // Fault reduction transformation is still independent of the
                // arithmetic contraction and preserves every original predicate.
                if k == Kernel::PromptAttentionFault {
                    let local = prompt_local_fault::source(k, new.clone());
                    assert_eq!(
                        new.matches("!finite(").count(),
                        local.matches("!finite(").count()
                    );
                }
            }
        }
    }
    #[test]
    fn actual_lowered_fmas_replace_only_qk_products_and_keep_all_remaining_products_precise() {
        fn counts(w: &[u32]) -> (usize, usize, usize) {
            let mut fma = 0;
            let mut mul = 0;
            let mut precise = 0;
            let mut at = 5;
            while at < w.len() {
                let n = (w[at] >> 16) as usize;
                assert!(n > 0 && at + n <= w.len());
                match w[at] & 65535 {
                    12 if n >= 5 && w[at + 4] == 50 => fma += 1,
                    133 => mul += 1,
                    71 if n == 3 && w[at + 2] == 42 => precise += 1,
                    _ => {}
                }
                at += n;
            }
            (fma, mul, precise)
        }
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            for old in versions(k) {
                let new = source(old.clone());
                let (_, before) = prompt_specialize::tests::compiled(&old);
                let (module, after) = prompt_specialize::tests::compiled(&new);
                assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
                let a = counts(&before);
                let b = counts(&after);
                let dots = old.matches("var dot=").count();
                assert_eq!(b.0, a.0 + 7 * dots);
                assert_eq!(a.1, b.1 + 7 * dots);
                assert_eq!(b.1, b.2); // Retain original NoContraction on every FMul.
                assert_eq!(
                    old.matches("subgroupShuffleXor(").count(),
                    new.matches("subgroupShuffleXor(").count()
                );
                assert_eq!(
                    old.matches("workgroupBarrier();").count(),
                    new.matches("workgroupBarrier();").count()
                );
            }
        }
    }
}
