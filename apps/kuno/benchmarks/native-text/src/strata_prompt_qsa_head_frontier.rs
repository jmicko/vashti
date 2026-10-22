//! Schedule the complete ordered scatter tree with a bounded live frontier.
//! Every node, operand order, source read and original dot FMA remains.
use super::{END, FIRST, TAG, declarations, dots, reduction};
const FRONTIER: &str =
    "// STRATA_QSA_HEAD_FRONTIER: chronological dots within a depth-first ordered tree.\n";
fn order() -> Vec<usize> {
    fn walk(stage: i32, i: usize, out: &mut Vec<usize>) {
        if stage < 0 {
            if i < 12 {
                out.push(i);
            }
            return;
        }
        walk(stage - 1, i, out);
        walk(stage - 1, i + (8usize >> stage), out);
    }
    let mut v = Vec::new();
    walk(3, 0, &mut v);
    assert_eq!(v.len(), 12);
    v
}
fn head(h: usize, next: Option<usize>) -> String {
    let mut s =
        format!("{{let h={h}u;let off=h*256u+lane*8u;\nlet qa=prefetch_qa;let qb=prefetch_qb;\n");
    if let Some(n) = next {
        s.push_str(&format!(
            "prefetch_qa=sq[{}u+lane];prefetch_qb=sq[{}u+lane];\n",
            n * 64,
            n * 64 + 32
        ));
    }
    for c in 0..2 {
        s.push_str(&format!("{}scatter{c}_{h}=dot;\n}}\n", dots(c)));
    }
    s.push_str("}\n");
    s
}
fn old_region(status: bool) -> String {
    let mut s = declarations();
    for h in 0..12 {
        s.push_str(&head(h, if h + 1 < 12 { Some(h + 1) } else { None }));
    }
    for c in 0..2 {
        s.push_str(&reduction(c, status));
    }
    s
}
fn scheduled_region(status: bool) -> String {
    fn node(
        stage: i32,
        i: usize,
        s: &mut String,
        heads: &[usize],
        cursor: &mut usize,
    ) -> [String; 2] {
        if stage < 0 {
            if i >= 12 {
                return ["0.0".into(), "0.0".into()];
            }
            assert_eq!(heads[*cursor], i);
            *cursor += 1;
            s.push_str(&head(i, heads.get(*cursor).copied()));
            return [format!("scatter0_{i}"), format!("scatter1_{i}")];
        }
        let left = node(stage - 1, i, s, heads, cursor);
        let right = node(stage - 1, i + (8usize >> stage), s, heads, cursor);
        let d = 16usize >> stage;
        let names = [
            format!("frontier0_r{stage}_{i}"),
            format!("frontier1_r{stage}_{i}"),
        ];
        for c in 0..2 {
            // These are register-only exchanges. Invalid cells' initialized
            // zeros cannot cause a load/store/fault; their final score remains
            // behind the unchanged original uniform valid-cell predicate.
            let n = &names[c];
            s.push_str(&format!("let {n}_hi=(lane&{d}u)!=0u;\nlet {n}_own=select({}, {}, {n}_hi);\nlet {n}_peer=subgroupShuffleXor(select({}, {}, {n}_hi),{d}u);\nlet {n}=select({n}_own,{n}_peer,{n}_hi)+select({n}_peer,{n}_own,{n}_hi);\n",left[c],right[c],right[c],left[c]));
        }
        names
    }
    let mut s = declarations();
    let mut cursor = 0;
    let names = node(3, 0, &mut s, &order(), &mut cursor);
    assert_eq!(cursor, 12);
    for c in 0..2 {
        let old = reduction(c, status);
        let tail = &old[old.find("let hi4=").unwrap()..];
        s.push_str(&format!(
            "if valid{c} {{\n{}",
            tail.replace("r3_0", &names[c])
        ));
    }
    s
}
pub(super) fn source(mut s: String) -> String {
    assert!(s.starts_with(TAG));
    assert!(!s.contains("STRATA_QSA_HEAD_FRONTIER"));
    assert_eq!(s.matches(FIRST).count(), 1);
    assert_eq!(s.matches(END).count(), 1);
    let status = s.contains("status[");
    let old = old_region(status);
    assert_eq!(
        s.matches(&old).count(),
        1,
        "complete ordered head/source layout drift"
    );
    s = s.replacen(&old, &scheduled_region(status), 1);
    s.insert_str(0, FRONTIER);
    s
}
#[cfg(test)]
fn restore(s: String) -> String {
    let s = s.strip_prefix(FRONTIER).unwrap();
    let status = s.contains("status[");
    let new = scheduled_region(status);
    assert_eq!(s.matches(&new).count(), 1);
    s.replacen(&new, &old_region(status), 1)
}
#[cfg(test)]
mod tests {
    use super::super::super::{Kernel, prompt_local_fault, prompt_specialize};
    use super::super::tests::before;
    use super::*;
    fn base(k: Kernel) -> String {
        super::super::source(before(k))
    }
    #[test]
    fn exact_full_source_inverse_preserves_all_memory_math_and_fault_owners() {
        for k in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            let old = base(k);
            let new = source(old.clone());
            assert_eq!(restore(new.clone()), old);
            assert!(std::panic::catch_unwind(|| source(new)).is_err());
            assert!(
                std::panic::catch_unwind(|| source(
                    old.replace("scatter0_11=dot;", "scatter0_10=dot;")
                ))
                .is_err()
            );
        }
    }
    #[test]
    fn twelve_immutable_query_heads_keep_all_24_addresses_with_bounded_tree_frontier() {
        let heads = order();
        assert_eq!(heads, vec![0, 8, 4, 2, 10, 6, 1, 9, 5, 3, 11, 7]);
        let mut sorted = heads.clone();
        sorted.sort();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        let selected = scheduled_region(false);
        for (j, &h) in heads.iter().enumerate() {
            assert_eq!(
                selected
                    .matches(&head(h, heads.get(j + 1).copied()))
                    .count(),
                1
            );
        }
        // First head0 was already prefetched by the unchanged outer pipeline.
        for h in 1..12 {
            for q in [h * 64, h * 64 + 32] {
                assert_eq!(selected.matches(&format!("=sq[{q}u+lane]")).count(), 1);
            }
        }
        fn live(stage: i32, i: usize) -> (usize, usize) {
            if stage < 0 {
                return if i < 12 { (1, 1) } else { (0, 0) };
            }
            let (a, ar) = live(stage - 1, i);
            let (b, br) = live(stage - 1, i + (8usize >> stage));
            (a.max(ar + b).max(ar + br), (ar + br).min(1))
        }
        // At most four nonzero subtree values per cell before each combine,
        // versus twelve whole-head sums. Compiler ISA must still prove spills0.
        assert_eq!(live(3, 0), (4, 1));
    }
    #[test]
    fn all_15_ordered_nodes_and_original_dot_members_survive_scheduling() {
        let s = scheduled_region(false);
        for c in 0..2 {
            for stage in 0..4 {
                for i in 0..(8usize >> stage) {
                    assert_eq!(
                        s.matches(&format!(
                            "let frontier{c}_r{stage}_{i}=select(frontier{c}_r{stage}_{i}_own,frontier{c}_r{stage}_{i}_peer,frontier{c}_r{stage}_{i}_hi)+select(frontier{c}_r{stage}_{i}_peer,frontier{c}_r{stage}_{i}_own,frontier{c}_r{stage}_{i}_hi);"
                        ))
                        .count(),
                        1
                    );
                }
            }
        }
        assert_eq!(s.matches("dot=fma(").count(), 168);
        for c in 0..2 {
            for h in 0..12 {
                assert_eq!(s.matches(&format!("scatter{c}_{h}=dot;")).count(), 1);
            }
        }
        assert_eq!(s.matches("if valid0 {").count(), 13);
        assert_eq!(s.matches("if valid1 {").count(), 13);
        assert!(!s.contains("array<"));
        assert!(!s.contains("workgroupBarrier"));
    }
    #[test]
    fn actual_spirv_keeps_complete_abi_168_fmas_42_exchanges_and_selected_composition() {
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
            let old = base(k);
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
            for op in [12, 61, 62, 133, 224, 246, 338, 345, 346, 366] {
                assert_eq!(count(&aw, op), count(&bw, op), "op{op}");
            }
            #[cfg(feature = "strata-qsa-head-frontier")]
            assert_eq!(k.source(), finish(new));
        }
    }
}
