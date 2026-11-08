//! Fault-only prompt QSA aggregation. All arithmetic and finite predicates stay
//! in place. Each invocation publishes its own sticky failure to the SAME real
//! global fault word; the original fenced request acceptance reads that owner.
//! Status-producing attention, scalar numerical references and decode/MTP are
//! deliberately unchanged: their per-head status contract needs a reduction.

use super::Kernel;

fn once(source: &mut String, old: &str, new: &str) {
    assert_eq!(
        source.matches(old).count(),
        1,
        "local QSA fault source drift"
    );
    *source = source.replacen(old, new, 1);
}

pub(super) fn source(kernel: Kernel, mut source: String) -> String {
    assert!(source.contains("var<storage,read_write> fault:array<atomic<u32>>;"));
    assert!(!source.contains("status["));
    match kernel {
        Kernel::PromptAttentionFault => {
            once(&mut source, "var<workgroup> broken:atomic<u32>;", "");
            once(
                &mut source,
                "if tid==0u {atomicStore(&broken,0u);}",
                "var local_broken=false;",
            );
            // Last barrier was exclusively the fault reduction. The query,
            // score/softmax and probability barriers remain at their exact
            // original positions. No other invocation reads partial stores.
            once(
                &mut source,
                "workgroupBarrier();\n    if atomicLoad(&broken)!=0u {\n        if tid==0u{atomicOr(&fault[0],1u);}\n    }",
                "if local_broken {atomicOr(&fault[0],1u);}",
            );
            assert!(source.matches("atomicOr(&broken,1u);").count() >= 4);
            source = source.replace("atomicOr(&broken,1u);", "local_broken=true;");
            assert!(!source.contains("&broken"));
        }
        Kernel::PromptMergeFault => {
            once(&mut source, "var<workgroup> good:array<u32,8>;", "");
            once(
                &mut source,
                "for(var distance=16u;distance>0u;distance/=2u) {ok&=subgroupShuffleXor(ok,distance);}\n    if lane==0u {good[warp]=ok;}workgroupBarrier();\n    if d==0u {var valid=1u;for(var i=0u;i<8u;i++) {valid&=good[i];}if valid!=1u{atomicOr(&fault[0],1u);}}",
                "if ok!=1u {atomicOr(&fault[0],1u);}",
            );
        }
        _ => panic!("local QSA fault kernel identity"),
    }
    source
}

#[cfg(test)]
mod tests {
    use super::super::{prompt_paired, prompt_pv_gather, prompt_specialize};
    use super::*;

    fn attention_versions() -> Vec<String> {
        let base = Kernel::PromptAttentionFault.scalar_source();
        let expanded = prompt_specialize::source(base.clone());
        let paired = prompt_paired::source(expanded.clone());
        vec![
            base,
            expanded.clone(),
            paired.clone(),
            prompt_pv_gather::source(expanded),
            prompt_pv_gather::source(paired),
        ]
    }

    #[test]
    fn entire_arithmetic_and_every_fault_predicate_are_unchanged() {
        for old in attention_versions() {
            let new = source(Kernel::PromptAttentionFault, old.clone());
            assert_eq!(old.matches("workgroupBarrier();").count(), 4);
            assert_eq!(new.matches("workgroupBarrier();").count(), 3);
            assert_eq!(
                old.matches("!finite(").count(),
                new.matches("!finite(").count()
            );
            assert_eq!(
                old.matches("atomicOr(&broken,1u);").count(),
                new.matches("local_broken=true;").count()
            );
            assert_eq!(new.matches("var local_broken=false;").count(), 1);
            let restored = new.replace("local_broken=true;", "atomicOr(&broken,1u);")
                .replace("var local_broken=false;", "if tid==0u {atomicStore(&broken,0u);}")
                .replace("if local_broken {atomicOr(&fault[0],1u);}", "workgroupBarrier();\n    if atomicLoad(&broken)!=0u {\n        if tid==0u{atomicOr(&fault[0],1u);}\n    }")
                .replacen("\n\nfn kv(", "\nvar<workgroup> broken:atomic<u32>;\nfn kv(", 1);
            assert_eq!(restored, old);
        }
        let old = Kernel::PromptMergeFault.scalar_source();
        let new = source(Kernel::PromptMergeFault, old.clone());
        // Compare the whole arithmetic prefix, including every output store;
        // only the trailing aggregation is replaced in this merge kernel.
        let prefix = old
            .split_once("    for(var distance=16u;distance>0u;distance/=2u)")
            .unwrap()
            .0;
        assert_eq!(
            new.split_once("    if ok!=1u {atomicOr(&fault[0],1u);}")
                .unwrap()
                .0,
            prefix.replace("var<workgroup> good:array<u32,8>;", "")
        );
        // OR is monotone. All single-lane failures, multiple failures and an
        // already-set preceding phase fault have exactly the same global bit.
        for existing in [0u32, 1, 128] {
            for bad_lane in 0..=257 {
                let mut global = existing;
                let mut shared = 0;
                for lane in 0..256 {
                    let bad = lane == bad_lane || bad_lane == 256;
                    if bad {
                        global |= 1;
                        shared |= 1;
                    }
                }
                assert_eq!(global, existing | shared);
            }
        }
    }

    #[test]
    fn actual_selected_modules_have_only_original_arithmetic_barriers_and_global_fault_atomics() {
        use wgpu::naga as n;
        let mut sources: Vec<_> = attention_versions()
            .into_iter()
            .map(|s| (Kernel::PromptAttentionFault, s, 3))
            .collect();
        sources.push((
            Kernel::PromptMergeFault,
            Kernel::PromptMergeFault.scalar_source(),
            0,
        ));
        for (kernel, old, barriers) in sources {
            let new = source(kernel, old.clone());
            assert_eq!(new.matches("workgroupBarrier();").count(), barriers);
            assert_eq!(
                old.matches("!finite(").count(),
                new.matches("!finite(").count()
            );
            assert!(!new.contains("subgroupShuffleXor(ok,"));
            let (module, words) = prompt_specialize::tests::compiled(&new);
            assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
            assert!(
                !module
                    .global_variables
                    .iter()
                    .any(|(_, v)| v.space == n::AddressSpace::WorkGroup
                        && matches!(module.types[v.ty].inner, n::TypeInner::Atomic(_)))
            );
            let mut count = 0;
            let mut at = 5;
            while at < words.len() {
                let size = (words[at] >> 16) as usize;
                assert!(size > 0 && at + size <= words.len());
                if words[at] & 65535 == 224 {
                    count += 1;
                }
                at += size;
            }
            assert_eq!(count, barriers);
        }
    }
}
