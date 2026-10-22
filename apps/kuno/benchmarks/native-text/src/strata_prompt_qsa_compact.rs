//! Keep the complete paired QK arithmetic in one uniformly guarded body.
//! The old common-pair specialization duplicates that entire body in native ISA.
use std::ops::Range;

const START: &str = "if valid0 && valid1 {";
const TAG: &str = "// STRATA_QK_COMPACT: identical paired guarded body, no duplicated fast body.\n";
struct Parts {
    branch: Range<usize>,
    fast: Range<usize>,
    guarded: Range<usize>,
}
fn close(s: &str, open: usize) -> usize {
    assert_eq!(s.as_bytes()[open], b'{');
    let mut depth = 0usize;
    for (offset, c) in s.as_bytes()[open..].iter().enumerate() {
        match c {
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1).expect("compact QK closing brace");
                if depth == 0 {
                    return open + offset;
                }
            }
            _ => {}
        }
    }
    panic!("compact QK complete branch required")
}
fn parts(s: &str) -> Parts {
    assert_eq!(
        s.matches(START).count(),
        1,
        "compact paired QK branch drift"
    );
    let first = s.find(START).unwrap();
    let open = first + START.len() - 1;
    let middle = close(s, open);
    assert!(s[middle..].starts_with("} else {\n"));
    let guarded_open = middle + "} else ".len();
    let end = close(s, guarded_open);
    Parts {
        branch: first..end + 1,
        fast: open + 1..middle,
        guarded: guarded_open + 1..end,
    }
}

pub(super) fn source(mut s: String) -> String {
    assert!(s.contains("STRATA_QK_UNIFORM_METADATA"));
    assert!(s.contains("STRATA_QK_EXPLICIT_FMA"));
    assert!(!s.contains("STRATA_QK_COMPACT"));
    let p = parts(&s);
    let guarded = s[p.guarded.clone()].to_owned();
    let mut full_pair = guarded.clone();
    for cell in 0..2 {
        let guard = format!("if valid{cell} {{var dot=");
        assert_eq!(full_pair.matches(&guard).count(), 12);
        full_pair = full_pair.replace(&guard, "{var dot=");
    }
    // The original full-pair body differs ONLY by these uniformly true guards.
    // Reject any future producer drift rather than guessing equivalent math.
    assert_eq!(full_pair, s[p.fast.clone()]);
    s.replace_range(p.branch, &guarded);
    s.insert_str(0, TAG);
    s
}

#[cfg(test)]
mod tests {
    use super::super::{
        Kernel, prompt_fma, prompt_local_fault, prompt_paired, prompt_pv_gather, prompt_specialize,
        prompt_uniform,
    };
    use super::*;

    fn original(k: Kernel) -> String {
        prompt_uniform::source(prompt_fma::source(prompt_pv_gather::source(
            prompt_paired::source(prompt_specialize::source(k.scalar_source())),
        )))
    }
    #[test]
    fn complete_source_and_all_guarded_arithmetic_are_exact_outside_duplicate_branch() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = original(k);
            let p = parts(&old);
            let new = source(old.clone());
            let mut restored = new.strip_prefix(TAG).unwrap().to_owned();
            assert_eq!(
                &restored[p.branch.start..p.branch.start + p.guarded.len()],
                &old[p.guarded.clone()]
            );
            restored.replace_range(
                p.branch.start..p.branch.start + p.guarded.len(),
                &old[p.branch.clone()],
            );
            assert_eq!(restored, old);
            assert_eq!(new.matches("if valid0 {var dot=").count(), 12);
            assert_eq!(new.matches("if valid1 {var dot=").count(), 12);
            assert!(!new.contains(START));
            let mut drift = old.clone();
            drift.insert_str(p.fast.start, "broken_source();");
            assert!(std::panic::catch_unwind(|| source(drift)).is_err());
        }
    }
    fn counts(words: &[u32]) -> [usize; 4] {
        let mut counts = [0; 4];
        let mut at = 5;
        while at < words.len() {
            let n = (words[at] >> 16) as usize;
            assert!(n != 0 && at + n <= words.len());
            let op = words[at] & 65535;
            counts[0] += usize::from(op == 338); // Original lane-zero samples.
            counts[1] += usize::from(op == 345); // Original indexed shuffle tree.
            counts[2] += usize::from(op == 12 && n >= 5 && words[at + 4] == 50); // Fma.
            counts[3] += usize::from(op == 224); // Original workgroup barriers.
            at += n;
        }
        counts
    }
    #[test]
    fn actual_compiled_pipeline_halves_only_duplicated_qk_and_keeps_geometry_and_barriers() {
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
            #[cfg(feature = "strata-qsa-prefetch")]
            let prefetched = super::super::prompt_prefetch::source(new.clone());
            #[cfg(feature = "strata-qsa-native-xor")]
            let prefetched = super::super::prompt_native_xor::source(prefetched);
            #[cfg(feature = "strata-qsa-scale-finite")]
            let prefetched = super::super::prompt_scale_finite::source(prefetched);
            #[cfg(feature = "strata-qsa-prefetch")]
            let prefetched = finish(prefetched);
            let old = finish(old);
            let new = finish(new);
            let (before, old_words) = prompt_specialize::tests::compiled(&old);
            let (after, new_words) = prompt_specialize::tests::compiled(&new);
            assert_eq!(before.entry_points[0].workgroup_size, [256, 1, 1]);
            assert_eq!(
                after.entry_points[0].workgroup_size,
                before.entry_points[0].workgroup_size
            );
            let a = counts(&old_words);
            let b = counts(&new_words);
            assert_eq!(&a[..3], &[50, 262, 348]);
            assert_eq!(&b[..3], &[26, 142, 180]);
            assert_eq!(a[3], b[3]);
            let shared = |m: &wgpu::naga::Module| {
                m.global_variables
                    .iter()
                    .filter(|(_, v)| v.space == wgpu::naga::AddressSpace::WorkGroup)
                    .count()
            };
            assert_eq!(shared(&before), shared(&after));
            #[cfg(all(
                feature = "strata-qsa-compact",
                not(feature = "strata-qsa-packed-cache")
            ))]
            {
                #[cfg(feature = "strata-qsa-prefetch")]
                let new = prefetched;
                assert_eq!(k.source(), new);
            }
            // The packed-cache suite separately checks the complete selected
            // composition. Keep this original compact-only inverse unchanged.
        }
    }
}
