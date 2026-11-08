//! Equivalent Q8 finite predicate: every signed-byte * FP16-scale product is
//! finite iff its scale is finite. Preserve every product and sticky fault.
const TAG: &str = "// STRATA_QSA_SCALE_FINITE: exact bounded Q8 finite predicate.\n";
pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QSA_NATIVE_XOR"));
    assert_eq!(
        s.matches("fn finite(x:f32)->bool {return abs(x)<=3.4028234663852886e38;}")
            .count(),
        1
    );
    assert!(!s.contains("STRATA_QSA_SCALE_FINITE"));
    assert_eq!(s.matches("fn signed_code(bits:u32,index:u32)->i32 {return (bitcast<i32>((bits>>((index&3u)*8u))<<24u))>>24u;}").count(),1);
    let scale = "let kscale=select(kscales.x,kscales.y,((koff/64u)&1u)!=0u);\n";
    assert_eq!(s.matches(scale).count(), 2);
    for cell in 0..2 {
        assert_eq!(
            s.matches(&format!(
                "let kscales=unpack2x16float(prefetch_scale{cell});"
            ))
            .count(),
            1
        );
        for component in 0..8 {
            let assignment = format!(
                "k{cell}_{component}=f32(signed_code(kword{},{}u))*kscale;\n",
                component / 4,
                component % 4
            );
            assert_eq!(s.matches(&assignment).count(), 1);
            let old = format!("if !finite(k{cell}_{component}) {{atomicOr(&broken,1u);}}\n");
            assert_eq!(s.matches(&old).count(), 1);
            s = s.replacen(&old, "", 1);
        }
    }
    s = s.replace(
        scale,
        &format!("{scale}if !finite(kscale) {{atomicOr(&broken,1u);}}\n"),
    );
    s.insert_str(0, TAG);
    s
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_compact, prompt_fma, prompt_local_fault, prompt_native_xor, prompt_paired,
        prompt_prefetch, prompt_pv_gather, prompt_specialize, prompt_uniform,
    };
    use super::*;
    fn before(k: Kernel) -> String {
        prompt_native_xor::source(prompt_prefetch::source(prompt_compact::source(
            prompt_uniform::source(prompt_fma::source(prompt_pv_gather::source(
                prompt_paired::source(prompt_specialize::source(k.scalar_source())),
            ))),
        )))
    }
    #[test]
    fn exhaustive_every_half_scale_and_signed_byte_has_the_identical_finite_predicate() {
        for bits in 0..=u16::MAX {
            let scale = half::f16::from_bits(bits).to_f32();
            for code in i8::MIN..=i8::MAX {
                let product = f32::from(code) * scale;
                assert_eq!(
                    product.is_finite(),
                    scale.is_finite(),
                    "half={bits:04x} code={code}"
                );
                if scale.is_finite() {
                    assert!(product.abs() <= 128.0 * 65504.0);
                }
            }
        }
    }
    #[test]
    fn exact_inverse_preserves_all_products_accesses_and_original_sticky_fault_policy() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = before(k);
            let new = source(old.clone());
            let mut restored = new
                .strip_prefix(TAG)
                .unwrap()
                .replace("if !finite(kscale) {atomicOr(&broken,1u);}\n", "");
            for cell in 0..2 {
                for component in 0..8 {
                    let assignment = format!(
                        "k{cell}_{component}=f32(signed_code(kword{},{}u))*kscale;\n",
                        component / 4,
                        component % 4
                    );
                    assert_eq!(restored.matches(&assignment).count(), 1);
                    restored = restored.replacen(&assignment,
                        &format!("{assignment}if !finite(k{cell}_{component}) {{atomicOr(&broken,1u);}}\n"), 1);
                }
            }
            assert!(
                std::panic::catch_unwind(|| source(
                    old.replace("abs(x)<=3.4028234663852886e38", "abs(x)<=1.0")
                ))
                .is_err()
            );
            assert_eq!(restored, old);
            assert_eq!(new.matches("if !finite(kscale)").count(), 2);
            assert_eq!(new.matches("*kscale;").count(), 16);
            assert!(std::panic::catch_unwind(|| source(new.clone())).is_err());
            assert!(std::panic::catch_unwind(|| source(old.replace("*kscale;", "*2.0;"))).is_err());
        }
    }
    #[test]
    fn actual_modules_keep_complete_abi_barriers_and_arithmetic_and_selected_hook() {
        fn ops(w: &[u32], op: u32) -> usize {
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
            assert_eq!(
                a.entry_points[0].workgroup_size,
                b.entry_points[0].workgroup_size
            );
            for op in [224, 133, 345, 346, 338] {
                assert_eq!(ops(&aw, op), ops(&bw, op));
            }
            #[cfg(feature = "strata-qsa-query-planes")]
            let new = super::super::prompt_query_planes::source(new);
            #[cfg(feature = "strata-qsa-head-scatter")]
            let new = super::super::prompt_head_scatter::source(new);
            #[cfg(feature = "strata-qsa-head-frontier")]
            let new = super::super::prompt_head_scatter::frontier_source(new);
            #[cfg(feature = "strata-qsa-scale-finite")]
            assert_eq!(k.source(), finish(new));
        }
    }
}
