//! Coalesce identical sticky fault writes between the original whole-group fences.
//! Every finite predicate and floating operation remains; good data does no atomic.
const TAG: &str =
    "// STRATA_GDN_COALESCED_FAULTS: same32 predicates, one sticky write per phase.\n";
fn predicate(r: usize) -> String {
    format!("if !finite(state{r}) {{atomicOr(&broken,1u);}}\n")
}
fn mask(r: usize, phase: &str) -> String {
    format!("state_{phase}_bad |= select(1u,0u,finite(state{r}));\n")
}
pub(super) fn source(mut s: String) -> String {
    assert!(!s.contains("STRATA_GDN_COALESCED_FAULTS"));
    assert_eq!(s.matches("atomicOr(&broken,1u)").count(), 64);
    assert_eq!(s.matches("workgroupBarrier();").count(), 6);
    for r in 0..32 {
        let old = predicate(r);
        assert_eq!(s.matches(&old).count(), 2);
        s = s.replacen(&old, &mask(r, "initial"), 1);
        s = s.replacen(&old, &mask(r, "step"), 1);
    }
    for (needle, phase) in [("var state0=old[", "initial"), ("state0=fma(", "step")] {
        assert_eq!(s.matches(needle).count(), 1);
        s = s.replacen(needle, &format!("var state_{phase}_bad=0u;\n{needle}"), 1);
        let end = mask(31, phase);
        assert_eq!(s.matches(&end).count(), 1);
        s = s.replacen(
            &end,
            &format!("{end}if state_{phase}_bad!=0u {{atomicOr(&broken,1u);}}\n"),
            1,
        );
    }
    assert_eq!(s.matches("atomicOr(&broken,1u)").count(), 2);
    s.insert_str(0, TAG);
    s
}
#[cfg(test)]
pub(super) fn restore(s: String) -> String {
    let mut s = s.strip_prefix(TAG).unwrap().to_owned();
    for phase in ["initial", "step"] {
        s = s.replace(&format!("var state_{phase}_bad=0u;\n"), "");
        s = s.replace(
            &format!("if state_{phase}_bad!=0u {{atomicOr(&broken,1u);}}\n"),
            "",
        );
        for r in 0..32 {
            let old = mask(r, phase);
            assert_eq!(s.matches(&old).count(), 1);
            s = s.replacen(&old, &predicate(r), 1);
        }
    }
    s
}
#[cfg(test)]
mod tests {
    use super::super::{Op, specialize};
    use super::*;
    fn before(op: Op) -> String {
        specialize::source(op.scalar_source())
    }
    #[test]
    fn exact_inverse_preserves_all_float_math_memory_closing_and_whole_group_barriers() {
        for op in [Op::Recurrence, Op::RecurrenceFault] {
            let old = before(op);
            let new = source(old.clone());
            assert_eq!(restore(new.clone()), old);
            assert_eq!(new.matches("workgroupBarrier();").count(), 6);
            for phase in ["initial", "step"] {
                let start = new.find(&format!("var state_{phase}_bad=0u;")).unwrap();
                let end = new.find(&format!("if state_{phase}_bad!=0u")).unwrap();
                assert!(!new[start..end].contains("atomicLoad(&broken)"));
                assert!(!new[start..end].contains("workgroupBarrier()"));
            }
        }
    }
    #[test]
    fn all_state_positions_and_ieee_classes_preserve_the_original_sticky_fault_bit() {
        let patterns = [
            0, 0x80000000, 1, 0x007fffff, 0x3f800000, 0x7f7fffff, 0x7f800000, 0xff800000,
            0x7fc12345, 0xffc12345,
        ];
        for prior in [0u32, 1, 128, 129, u32::MAX] {
            for bad in 0..32 {
                for bits in patterns {
                    let mut state = [0.5f32; 32];
                    state[bad] = f32::from_bits(bits);
                    state[(bad + 17) % 32] = f32::from_bits(bits.rotate_left(8));
                    let original = state
                        .iter()
                        .fold(prior, |b, x| b | u32::from(!x.is_finite()));
                    let phase = state
                        .iter()
                        .fold(0u32, |b, x| b | u32::from(!x.is_finite()));
                    assert_eq!(original, prior | phase);
                }
            }
        }
    }
    fn compiled(source: &str, entry: &str) -> (wgpu::naga::Module, Vec<u32>) {
        use wgpu::naga as n;
        let m = n::front::wgsl::parse_str(source).unwrap();
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&m)
        .unwrap();
        let options = n::back::spv::Options {
            lang_version: (1, 3),
            zero_initialize_workgroup_memory: n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
            bounds_check_policies: n::proc::BoundsCheckPolicies {
                index: n::proc::BoundsCheckPolicy::Unchecked,
                buffer: n::proc::BoundsCheckPolicy::Unchecked,
                image_load: n::proc::BoundsCheckPolicy::Unchecked,
                binding_array: n::proc::BoundsCheckPolicy::Unchecked,
            },
            ..Default::default()
        };
        let words = n::back::spv::write_vec(
            &m,
            &info,
            &options,
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: entry.into(),
            }),
        )
        .unwrap();
        (m, words)
    }
    #[test]
    fn actual_spirv_keeps_complete_abi_math_and_simd16_with_only_two_local_fault_writes() {
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
        for op in [Op::Recurrence, Op::RecurrenceFault] {
            let (selected, entry) = op.compilation();
            let old = before(op).replace("fn main(", &format!("fn {entry}("));
            let new = source(before(op)).replace("fn main(", &format!("fn {entry}("));
            let (a, aw) = compiled(&old, entry);
            let (b, bw) = compiled(&new, entry);
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
            assert_eq!(a.entry_points[0].workgroup_size, [128, 1, 1]);
            assert_eq!(
                a.entry_points[0].workgroup_size,
                b.entry_points[0].workgroup_size
            );
            for code in [12, 133, 224, 227, 228, 246] {
                assert_eq!(count(&aw, code), count(&bw, code), "op{code}");
            }
            assert_eq!(count(&aw, 241), count(&bw, 241) + 62);
            let subgroup = if cfg!(feature = "strata-gdn-subgroup16") {
                Some(16)
            } else {
                None
            };
            assert_eq!(wgpu::hal::vulkan::kuno_gdn_subgroup16(&aw), subgroup);
            assert_eq!(wgpu::hal::vulkan::kuno_gdn_subgroup16(&bw), subgroup);
            #[cfg(feature = "strata-gdn-coalesce-faults")]
            assert_eq!(selected, new);
        }
    }
    #[test]
    fn exact_selected_hook_keeps_other_stages_and_rejects_missing_or_duplicated_predicates() {
        for op in [Op::Recurrence, Op::RecurrenceFault] {
            let old = before(op);
            let new = source(old.clone());
            assert!(std::panic::catch_unwind(|| source(new)).is_err());
            assert!(
                std::panic::catch_unwind(|| source(
                    old.replace("if !finite(state31)", "if !finite(state30)")
                ))
                .is_err()
            );
            #[cfg(feature = "strata-gdn-coalesce-faults")]
            assert_eq!(op.source(), source(before(op)));
        }
        for op in [
            Op::Conv,
            Op::ConvFault,
            Op::History,
            Op::L2,
            Op::L2Fault,
            Op::Split,
            Op::Norm,
            Op::NormFault,
            Op::Retain,
            Op::Compatibility,
        ] {
            assert_eq!(op.source(), op.scalar_source());
        }
    }
}
