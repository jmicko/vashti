//! Opt-in probability gather for the existing prompt QSA PV loop.
//!
//! Keep the original logical32-lane subgroup and chronological F32 FMAs. One
//! active lane per head reads its own probability; constant subgroup broadcasts
//! carry those exact bits to every output dimension. The padded65-float head
//! stride avoids the original power-of-two address pattern for that gather.
//! All global owners, QK/softmax arithmetic, barriers and fault closing survive.

pub(super) fn source(mut original: String) -> String {
    assert!(original.contains("// STRATA_ROUND_PRODUCTS:"));
    let declaration = "var<workgroup> sp:array<f32,768>;";
    assert_eq!(
        original.matches(declaration).count(),
        1,
        "PV shared source drift"
    );
    original = original.replacen(declaration, "var<workgroup> sp:array<f32,780>;", 1);

    // Rewrite only actual sp subscripts, not64-cell grids, query/cache offsets
    // or global output/status strides. Every original probability subscript is
    // head*64+cell; reject generator drift rather than guessing an index.
    let mut at = 0;
    let mut subscripts = 0;
    while let Some(relative) = original[at..].find("sp[") {
        let first = at + relative + "sp[".len();
        let last = first
            + original[first..]
                .find(']')
                .expect("PV subscript closing drift");
        let index = &original[first..last];
        assert_eq!(
            index.matches("*64u+").count(),
            1,
            "PV head stride source drift"
        );
        let padded = index.replacen("*64u+", "*65u+", 1);
        original.replace_range(first..last, &padded);
        at = first + padded.len() + 1;
        subscripts += 1;
    }
    assert!(subscripts >= 20, "PV probability inventory drift");

    let mut scalar = String::new();
    let mut gathered = String::from(
        "var probability=0.0;\n\
         if lane<12u {probability=sp[lane*65u+c];}\n",
    );
    for head in 0..12 {
        scalar.push_str(&format!(
            "accum{head}=fma(sp[{head}u*65u+c],v,accum{head});\n"
        ));
        gathered.push_str(&format!(
            "let probability{head}=subgroupBroadcast(probability,{head}u);\n\
             accum{head}=fma(probability{head},v,accum{head});\n"
        ));
    }
    assert_eq!(
        original.matches(&scalar).count(),
        1,
        "PV ordered FMA source drift"
    );
    original.replacen(&scalar, &gathered, 1)
}

#[cfg(test)]
mod tests {
    use super::super::{Kernel, prompt_paired, prompt_specialize};

    fn versions(kernel: Kernel) -> [String; 2] {
        let base = prompt_specialize::source(kernel.scalar_source());
        [base.clone(), prompt_paired::source(base)]
    }

    #[test]
    fn all_shared_cells_are_exact_and_all_twelve_padding_words_are_unread() {
        for here in 0..=64 {
            let mut original = [[0u32; 64]; 12];
            let mut padded = [f32::NAN.to_bits(); 780];
            let mut written = [false; 780];
            for head in 0..12 {
                for cell in 0..here {
                    let bits = match (head, cell) {
                        (2, 3) => f32::NAN.to_bits(),
                        (4, 5) => (-0.0f32).to_bits(),
                        _ => ((head * 1024 + cell) as f32 / 31.0).to_bits(),
                    };
                    original[head][cell] = bits;
                    assert!(!written[head * 65 + cell]);
                    written[head * 65 + cell] = true;
                    padded[head * 65 + cell] = bits;
                }
            }
            for cell in 0..here {
                let mut lane_values = [0u32; 32];
                for lane in 0..12 {
                    lane_values[lane] = padded[lane * 65 + cell];
                }
                for output_lane in 0..32 {
                    for head in 0..12 {
                        assert_eq!(lane_values[head], original[head][cell]);
                        // A neighboring lane is a wrong head for this public
                        // distinct-bit fixture, rather than an alternate oracle.
                        if here > 1 && head < 11 && cell != 3 && cell != 5 {
                            assert_ne!(lane_values[(head + 1) % 12], original[head][cell]);
                        }
                    }
                    assert!(output_lane < 32);
                }
            }
            for head in 0..12 {
                assert!(!written[head * 65 + 64]);
                assert_eq!(padded[head * 65 + 64], f32::NAN.to_bits());
            }
        }
    }

    #[test]
    fn actual_source_only_changes_shared_stride_and_ordered_probability_reads() {
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            for original in versions(kernel) {
                let gathered = super::source(original.clone());
                assert_eq!(original.matches("workgroupBarrier();").count(), 4);
                assert_eq!(gathered.matches("workgroupBarrier();").count(), 4);
                assert_eq!(gathered.matches("probability=sp[lane*65u+c];").count(), 1);
                assert_eq!(
                    gathered.matches("subgroupBroadcast(probability,").count(),
                    12
                );
                for head in 0..12 {
                    assert_eq!(
                        gathered
                            .matches(&format!("subgroupBroadcast(probability,{head}u)"))
                            .count(),
                        1
                    );
                    assert_eq!(
                        gathered
                            .matches(&format!(
                                "accum{head}=fma(probability{head},v,accum{head});"
                            ))
                            .count(),
                        1
                    );
                }
                // Reverse only the specified memory/gather transformations.
                // This compares the complete QK/softmax, cache guards, global
                // stores, PV order, NaN handling and closing, including tails.
                let mut restored = gathered.replacen(
                    "var<workgroup> sp:array<f32,780>;",
                    "var<workgroup> sp:array<f32,768>;",
                    1,
                );
                let mut scalar = String::new();
                let mut group = String::from(
                    "var probability=0.0;\nif lane<12u {probability=sp[lane*65u+c];}\n",
                );
                for head in 0..12 {
                    scalar.push_str(&format!(
                        "accum{head}=fma(sp[{head}u*65u+c],v,accum{head});\n"
                    ));
                    group.push_str(&format!("let probability{head}=subgroupBroadcast(probability,{head}u);\naccum{head}=fma(probability{head},v,accum{head});\n"));
                }
                assert_eq!(restored.matches(&group).count(), 1);
                restored = restored.replacen(&group, &scalar, 1);
                restored = restored.replace("*65u+", "*64u+");
                assert_eq!(restored, original);
            }
            #[cfg(feature = "strata-qsa-pv-gather")]
            {
                let base = prompt_specialize::source(kernel.scalar_source());
                #[cfg(feature = "strata-qsa-paired")]
                let base = prompt_paired::source(base);
                assert_eq!(kernel.source(), super::source(base));
            }
        }
    }

    #[test]
    fn actual_naga_spirv_uses_twelve_constant_subgroup_broadcasts_and_exact_shared_bytes() {
        use wgpu::naga as n;
        for kernel in [Kernel::PromptAttention, Kernel::PromptAttentionFault] {
            for source in versions(kernel) {
                let (module, words) = prompt_specialize::tests::compiled(&super::source(source));
                assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
                let mut shared_bytes = 0;
                for (_, value) in module
                    .global_variables
                    .iter()
                    .filter(|(_, v)| v.space == n::AddressSpace::WorkGroup)
                {
                    shared_bytes += match module.types[value.ty].inner {
                        n::TypeInner::Array {
                            size: n::ArraySize::Constant(size),
                            stride,
                            ..
                        } => size.get() * stride,
                        n::TypeInner::Atomic(_) => 4,
                        _ => panic!("unpriced actual QSA shared type"),
                    };
                }
                assert_eq!(shared_bytes, 15620 + 48);
                let mut constants = std::collections::BTreeMap::new();
                let mut broadcasts = Vec::new();
                let mut products = std::collections::BTreeSet::new();
                let mut precise = std::collections::BTreeSet::new();
                let mut barriers = 0;
                let mut fmas = 0;
                let mut at = 5;
                while at < words.len() {
                    let size = (words[at] >> 16) as usize;
                    assert!(size != 0 && at + size <= words.len());
                    match words[at] & 65535 {
                        43 if size == 4 => {
                            constants.insert(words[at + 2], words[at + 3]);
                        }
                        // Naga27 lowers WGSL subgroupBroadcast to the original
                        // OpGroupNonUniformShuffle for Vulkan1.1 portability
                        // (back/spv/subgroup.rs). Require its genuine constant
                        // subgroup scope and all twelve exact source lanes.
                        345 => {
                            assert_eq!(size, 6);
                            broadcasts.push((words[at + 3], words[at + 5]));
                        }
                        133 => {
                            products.insert(words[at + 2]);
                        }
                        71 if size == 3 && words[at + 2] == 42 => {
                            precise.insert(words[at + 1]);
                        }
                        224 => barriers += 1,
                        12 if size >= 5 && words[at + 4] == 50 => fmas += 1,
                        _ => {}
                    }
                    at += size;
                }
                assert_eq!(broadcasts.len(), 12);
                assert_eq!(
                    broadcasts
                        .iter()
                        .map(|(_, index)| constants[index])
                        .collect::<Vec<_>>(),
                    (0..12).collect::<Vec<_>>()
                );
                assert!(broadcasts.iter().all(|(scope, _)| constants[scope] == 3));
                assert!(!products.is_empty());
                assert_eq!(products, precise);
                assert_eq!(barriers, 4);
                assert_eq!(fmas, 12);
            }
        }
    }
}
