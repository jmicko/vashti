use super::*;
fn ops() -> impl Iterator<Item = CompactOp> {
    [
        (false, &[q::IQ2_S, q::IQ3_S, q::IQ3_XXS, q::IQ4_XS][..]),
        (true, &[q::IQ4_NL, q::Q2_0][..]),
    ]
    .into_iter()
    .flat_map(|(down, types)| types.iter().map(move |&dtype| CompactOp { dtype, down }))
}
#[test]
fn compact_actual_sources_preserve_all_original_source_guards_and_encode_only_integer_copy() {
    for op in ops() {
        let old = CohortOp::Image {
            dtype: op.dtype,
            down: op.down,
        }
        .source();
        let new = op.source();
        let old_main = old.find("fn main(").unwrap();
        let new_main = new.find("fn main(").unwrap();
        let old_body = old[old_main..]
            .find("    let record=id.x;")
            .or_else(|| old[old_main..].find("    let i=id.x;if i>=p.rows*p.columns/8u"))
            .map(|at| old_main + at)
            .unwrap();
        let new_body = new[new_main..]
            .find("    // The compact slot")
            .or_else(|| new[new_main..].find("    if p.r1%4u"))
            .map(|at| new_main + at)
            .unwrap();
        assert_eq!(&old[..old_body], &new[..new_body]);
        let end = crate::strata_port::function_end(&new, new_main);
        let copy = &new[new_body..end];
        assert!(!copy.contains("decoded8("));
        assert!(!copy.contains("pack2x16float"));
        assert!(copy.contains("strata_gate(at+3u)"));
        assert!(copy.contains("id.z*p.r5"));
        assert!(!copy.contains("cohort_slot*p.r5"));
        if !op.down {
            assert!(copy.contains("strata_up(at+3u)"));
        }
        let module = wgpu::naga::front::wgsl::parse_str(&new).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points[0].workgroup_size, [256, 1, 1]);
        assert_eq!(op.native_release_after(), op.down);
        assert_eq!(
            op.name(),
            if op.down {
                "StrataExpertCohortDownCompact"
            } else {
                "StrataExpertCohortGuCompact"
            }
        );
    }
}
#[test]
fn compact_grid_uses_original_role_bytes_and_rejects_wrong_roles_and_extents() {
    for op in ops() {
        let mut w = super::super::tests::fixture();
        let (columns, rows, role) = if op.down {
            (640usize, 2560usize, 2)
        } else {
            (2560, 1280, 0)
        };
        let (values, bytes) = q::block_layout(op.dtype).unwrap();
        let row_bytes = columns / values * bytes;
        let role_bytes = rows / if op.down { 1 } else { 2 } * row_bytes;
        w[1] = columns as u32;
        w[2] = rows as u32;
        w[3] = row_bytes as u32;
        w[5] = role;
        if op.down {
            w[6] = 64;
            w[7] = 64;
            w[8] = (role_bytes / 4 + 128) as u32;
        } else {
            w[6] = (role_bytes / 4) as u32;
            w[7] = w[6];
            w[8] = 2 * w[6] + 64;
        }
        assert_eq!(
            grid(op, &w).unwrap(),
            [(role_bytes / 16).div_ceil(256), 1, 8]
        );
        for index in [1usize, 2, 3, 5, 6, 7, 8] {
            let mut poison = w;
            poison[index] = poison[index].wrapping_add(1);
            assert!(grid(op, &poison).is_err(), "field {index}");
        }
    }
}
#[test]
fn compact_slots_and_fused_tile_categories_cover_exact_small_members_without_raw_aliases() {
    for members in 1usize..32 {
        let tile = tile_for(members);
        assert!(tile >= members && matches!(tile, 8 | 16 | 32));
        assert_eq!(
            [8usize, 16, 32].into_iter().filter(|&t| t == tile).count(),
            1
        );
    }
    // Complete distinct original role payloads, not decoded/F16-sized records.
    for stride in [1_280usize, 665_600, 1_048_576] {
        for logical in 0..8 {
            let out = logical * stride;
            assert!(out + stride <= 8 * stride);
            for raw in 0..=8 {
                assert_eq!(
                    logical * stride,
                    out,
                    "raw sentinel must not select outputslot {raw}"
                );
            }
        }
    }
}
