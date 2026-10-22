//! Exact selected Strata default schedule: one cell, ordinary shared PV reads,
//! constant XOR reductions, original eight-product QK and chronological PV FMAs.
//! All previous generators remain independently tested. No reference replacement.
use super::{Kernel, prompt_fma, prompt_local_fault, prompt_specialize};
pub(super) fn source(k: Kernel) -> String {
    assert!(matches!(
        k,
        Kernel::PromptAttention | Kernel::PromptAttentionFault
    ));
    let s = prompt_fma::source(prompt_specialize::source(k.scalar_source()));
    if k == Kernel::PromptAttentionFault {
        prompt_local_fault::source(k, s)
    } else {
        s
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_default_schedule_keeps_original_owners_geometry_barriers_and_exact_fma_inventory() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let s = source(k);
            let (m, w) = prompt_specialize::tests::compiled(&s);
            let base = k.scalar_source();
            // Compare the SAME pre-existing closing policy. Local-fault mode
            // already removed the workgroup atomic before this schedule work.
            let base = if k == Kernel::PromptAttentionFault {
                prompt_local_fault::source(k, base)
            } else {
                base
            };
            let (base, _) = prompt_specialize::tests::compiled(&base);
            let globals = |m: &wgpu::naga::Module| {
                m.global_variables
                    .iter()
                    .map(|(_, g)| (g.binding.clone(), g.space))
                    .collect::<Vec<_>>()
            };
            assert_eq!(globals(&m), globals(&base));
            assert_eq!(m.entry_points[0].workgroup_size, [256, 1, 1]);
            assert!(s.contains("for(var c=warp;c<64u;c+=8u)"));
            assert!(!s.contains("var pair="));
            assert!(!s.contains("subgroupBroadcast("));
            let mut at = 5;
            let mut fma = 0;
            let mut xor = 0;
            let mut barrier = 0;
            while at < w.len() {
                let n = (w[at] >> 16) as usize;
                assert!(n > 0);
                let op = w[at] & 65535;
                fma += usize::from(op == 12 && n >= 5 && w[at + 4] == 50);
                xor += usize::from(op == 346);
                barrier += usize::from(op == 224);
                at += n;
            }
            assert_eq!(fma, 96);
            assert_eq!(xor, 70);
            assert_eq!(
                barrier,
                if k == Kernel::PromptAttentionFault {
                    3
                } else {
                    4
                }
            );
            assert_eq!(s.matches("accum").count() > 12, true);
            #[cfg(feature = "strata-qsa-reference-schedule")]
            assert_eq!(k.source(), s);
        }
    }
    #[test]
    fn cell_inventory_and_chronological_products_match_complete_original_pairs() {
        for here in 0..=64usize {
            for warp in 0..8 {
                let one = (warp..64)
                    .step_by(8)
                    .filter(|&c| c < here)
                    .collect::<Vec<_>>();
                let pair = (warp..64)
                    .step_by(16)
                    .flat_map(|c| [c, c + 8])
                    .filter(|&c| c < here)
                    .collect::<Vec<_>>();
                assert_eq!(one, pair);
                for c in one {
                    for h in 0..12 {
                        let order = (0..8).map(|j| (c, h, j)).collect::<Vec<_>>();
                        assert_eq!(order.len(), 8);
                    }
                }
            }
        }
        assert_eq!((96 - 12) * 8, (180 - 12) * 4); // Same672 QK FMAs per subgroup chunk.
        assert_eq!(12 * 64, 768); // Same chronological PV contribution inventory.
    }
    #[test]
    fn selected_schedule_is_original_specialization_plus_explicit_fma_and_original_fault_closing() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let original = prompt_specialize::source(k.scalar_source());
            let fma = prompt_fma::source(original.clone());
            let expected = if k == Kernel::PromptAttentionFault {
                prompt_local_fault::source(k, fma)
            } else {
                fma
            };
            assert_eq!(source(k), expected);
            assert!(!source(k).contains("STRATA_QK_COMPACT"));
            assert!(!source(k).contains("STRATA_QK_PACKED_CACHE"));
            assert!(source(k).contains("var<workgroup> sp:array<f32,768>;"));
        }
        for k in [
            Kernel::Attention,
            Kernel::MtpAttention,
            Kernel::MtpMerge,
            Kernel::Merge,
        ] {
            assert_eq!(k.source(), k.scalar_source());
        }
    }
}
