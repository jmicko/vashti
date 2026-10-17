//! MIT Strata fb58e0 iq_kernels.dp.cpp q8_1_store and fused SwiGLU producer.
//! A workgroup owns one original padded frame, preserving real readiness and
//! clearing dead GU storage. Subgroups keep the original32-value XOR order.
#[derive(Clone, Copy)]
pub(super) struct Layout {
    pub words: usize,
    pub header: usize,
    pub frame: usize,
    pub blocks: usize,
    pub columns: usize,
    pub rows_per_input: usize,
}
pub(super) const GU: Layout = Layout {
    words: 16384,
    header: 64,
    frame: 1440,
    blocks: 80,
    columns: 2560,
    rows_per_input: 1,
};
pub(super) const DOWN: Layout = Layout {
    words: 32768,
    header: 128,
    frame: 360,
    blocks: 20,
    columns: 640,
    rows_per_input: 10,
};
impl Layout {
    pub fn frames(self, count: usize) -> kuno_text::Result<usize> {
        if !(1..=8).contains(&count) {
            return Err(kuno_text::Error::InvalidBufferLength);
        }
        Ok(count * self.rows_per_input)
    }
}
pub(super) fn source(down: bool) -> String {
    let all = include_str!("strata_expert_q8.wgsl");
    let section = |name: &str| {
        all.split_once(&format!("// {name}_BEGIN\n"))
            .unwrap()
            .1
            .split_once(&format!("// {name}_END"))
            .unwrap()
            .0
    };
    let l = if down { DOWN } else { GU };
    let mut s = format!(
        "const TOTAL_WORDS:u32={}u;const HEADER_WORDS:u32={}u;const FRAME_WORDS:u32={}u;const BLOCKS:u32={}u;const COLUMNS:u32={}u;const ROWS_PER_INPUT:u32={}u;\n",
        l.words, l.header, l.frame, l.blocks, l.columns, l.rows_per_input
    );
    if down {
        s.push_str("// STRATA_ROUND_PRODUCTS: native fused SwiGLU before Q8 sum.\n");
    }
    s.push_str(section("COMMON"));
    s.push_str(section(if down { "DOWN" } else { "GU" }));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_layout_clears_each_full_owner_without_cross_frame_writers_for_every_window() {
        assert_eq!((GU.words * 4, DOWN.words * 4), (65536, 131072));
        for l in [GU, DOWN] {
            assert_eq!(l.frame, l.blocks * 18);
            assert_eq!(l.columns, l.blocks * 32);
            assert!(l.header >= l.frames(8).unwrap());
            for invalid in [0, 9, usize::MAX] {
                assert!(l.frames(invalid).is_err());
            }
            for count in 1..=8 {
                let frames = l.frames(count).unwrap();
                let mut writers = vec![None; l.words];
                let mut visit = |range: std::ops::Range<usize>, owner: usize| {
                    for word in range {
                        assert!(writers[word].replace(owner).is_none());
                    }
                };
                for frame in 0..frames {
                    visit(frame..frame + 1, frame);
                    let start = l.header + frame * l.frame;
                    visit(start..start + l.frame, frame);
                    // Actual prepared words read by native row_entries: first
                    // packed plane and original half-rounded activation scales.
                    assert!(start + l.blocks * 16 + l.blocks <= start + l.frame);
                }
                visit(frames..l.header, 0);
                visit(l.header + frames * l.frame..l.words, 0);
                assert!(writers.iter().all(Option::is_some));
                let mut columns = vec![0u8; l.columns];
                for step in 0..l.blocks.div_ceil(8) {
                    for sg in 0..8 {
                        let b = step * 8 + sg;
                        if b < l.blocks {
                            for lane in 0..32 {
                                columns[b * 32 + lane] += 1;
                            }
                        }
                    }
                }
                assert!(columns.iter().all(|&v| v == 1));
            }
        }
    }
    #[test]
    fn native_postpass_pins_products_without_changing_explicit_fma_or_original_interfaces() {
        use wgpu::naga as n;
        let source = "@group(0) @binding(0) var<storage,read_write> x:array<f32>; @compute @workgroup_size(32) fn main(){let p=x[0]*x[1];x[2]=fma(x[0],x[1],p);}";
        let module = n::front::wgsl::parse_str(source).unwrap();
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let words = n::back::spv::write_vec(
            &module,
            &info,
            &n::back::spv::Options::default(),
            Some(&n::back::spv::PipelineOptions {
                shader_stage: n::ShaderStage::Compute,
                entry_point: "main".into(),
            }),
        )
        .unwrap();
        fn instructions(words: &[u32]) -> Vec<&[u32]> {
            let mut at = 5;
            let mut result = Vec::new();
            while at < words.len() {
                let len = (words[at] >> 16) as usize;
                assert!(len > 0 && at + len <= words.len());
                result.push(&words[at..at + len]);
                at += len;
            }
            result
        }
        let before = instructions(&words);
        // Independent SPIR-V grammar: FMul=133; OpDecorate=71;
        // NoContraction=42. The vendor implementation uses rspirv's enum.
        let products: std::collections::BTreeSet<_> = before
            .iter()
            .filter(|i| i[0] & 65535 == 133)
            .map(|i| i[2])
            .collect();
        assert!(!products.is_empty());
        let pinned = cubecl::wgpu::preserve_native_product_rounding(&words).unwrap();
        let after = instructions(&pinned);
        let decorated: std::collections::BTreeSet<_> = after
            .iter()
            .filter(|i| i[0] & 65535 == 71 && i.len() == 3 && i[2] == 42)
            .map(|i| i[1])
            .collect();
        assert_eq!(decorated, products);
        assert_eq!(words[..5], pinned[..5]); // IDs and bound unchanged.
        assert_eq!(
            before
                .into_iter()
                .filter(|i| i[0] & 65535 != 71)
                .collect::<Vec<_>>(),
            after
                .into_iter()
                .filter(|i| i[0] & 65535 != 71)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            cubecl::wgpu::preserve_native_product_rounding(&pinned).unwrap(),
            pinned
        );
        assert!(cubecl::wgpu::preserve_native_product_rounding(&[]).is_err());
        assert!(cubecl::wgpu::preserve_native_product_rounding(&[0; 5]).is_err());
    }
    #[test]
    fn actual_producers_lower_with_original_plane_and_consumed_gu_stamp_owners() {
        use wgpu::naga as n;
        for down in [false, true] {
            let source = source(down);
            let module = n::front::wgsl::parse_str(&source).unwrap();
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let uses = info.get_entry_point(0);
            assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
            // A zeroing lane and packing lane can differ. Real STORAGE
            // barriers must order both phase transitions, beyond SLM barriers.
            assert!(module.entry_points[0].function.body.iter().filter(|s|matches!(s,n::Statement::ControlBarrier(flags) if flags.contains(n::Barrier::STORAGE))).count()>=2);
            let mut owners = std::collections::BTreeMap::new();
            for (id, g) in module.global_variables.iter() {
                if let Some(binding) = &g.binding {
                    owners.insert(
                        binding.binding,
                        (
                            uses[id].contains(n::valid::GlobalUse::READ),
                            uses[id].contains(n::valid::GlobalUse::WRITE),
                        ),
                    );
                }
            }
            if down {
                assert_eq!(owners.len(), 4);
                assert_eq!(owners[&0], (true, true)); // Authentic PART: both GU inputs consumed/erased.
                assert_eq!(owners[&1], (true, true)); // Consume original GU stamps, clear UP before DOWN.
                assert!(owners[&2].1); // Same complete DOWN planes, no intermediate ACT device copy.
                assert!(
                    module
                        .global_variables
                        .iter()
                        .any(|(_, g)| g.name.as_deref() == Some("activated")
                            && g.space == n::AddressSpace::WorkGroup)
                );
            } else {
                assert_eq!(owners.len(), 3);
                assert_eq!(owners[&0], (true, false));
                assert!(owners[&1].1);
            }
            n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options::default(),
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
    }
}
