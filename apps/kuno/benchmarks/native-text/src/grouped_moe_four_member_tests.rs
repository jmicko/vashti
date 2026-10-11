use super::*;
use std::hash::{Hash, Hasher};

fn frozen() -> Geometry {
    Geometry::new(
        [
            Projection::new(IQ3_XXS, 2560, 640, 512).unwrap(),
            Projection::new(IQ3_XXS, 2560, 640, 512).unwrap(),
            Projection::new(IQ4_NL, 640, 2560, 512).unwrap(),
        ],
        8,
        144,
    )
    .unwrap()
    .with_frozen_host_sources()
    .unwrap()
    .with_inactive_partition_exit()
    .unwrap()
}
fn hash(value: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

#[test]
fn four_member_policy_preserves_full_owning_capacity_and_rejects_conflicts_both_orders() {
    let original = frozen();
    let selected = original.with_four_members().unwrap();
    assert_eq!(original.member_limit(), 3);
    assert_eq!(selected.member_limit(), 4);
    assert_eq!(original.member_capacity(), 8);
    assert_eq!(selected.member_capacity(), 8);
    assert_eq!(selected.status_words(), original.status_words());
    for down in [false, true] {
        let dimensions = |grid| match grid {
            CubeCount::Static(x, y, z) => (x, y, z),
            _ => panic!("expected original static geometry"),
        };
        assert_eq!(
            dimensions(selected.grid(40, down).unwrap()),
            dimensions(original.grid(40, down).unwrap())
        );
    }
    for count in 1..=4 {
        selected.validate_count(count).unwrap();
    }
    assert!(original.validate_count(4).is_err());
    for count in [0, 5, 8, usize::MAX] {
        assert!(selected.validate_count(count).is_err());
    }
    let plain = Geometry::new(original.projections(), 8, 144).unwrap();
    assert_eq!(plain.member_limit(), 8);
    plain.validate_count(8).unwrap();
    assert!(plain.with_four_members().is_err());
    assert!(selected.with_four_members().is_err());
    for policy in [
        Geometry::with_private_tables,
        Geometry::with_workgroup_tables,
        Geometry::with_shared_down_scales,
        Geometry::with_shared_gate_up_factors,
        Geometry::with_two_row_gate_up,
        Geometry::with_four_row_down,
        Geometry::with_three_member_arithmetic,
    ] {
        assert!(policy(selected).is_err());
        assert!(policy(original).unwrap().with_four_members().is_err());
    }
}

#[test]
fn unselected_geometry_hash_remains_the_original_derived_payload() {
    #[derive(Hash)]
    struct Original {
        projections: [Projection; 3],
        hidden: usize,
        ff: usize,
        tile: usize,
        capacity: usize,
        gate_policy: GatePolicy,
        reduction_policy: ReductionPolicy,
        secondary_partition: bool,
        frozen_host_sources: bool,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        three_member_arithmetic: bool,
    }
    let g = frozen();
    let old = Original {
        projections: g.projections,
        hidden: g.hidden,
        ff: g.ff,
        tile: g.tile,
        capacity: g.capacity,
        gate_policy: g.gate_policy,
        reduction_policy: g.reduction_policy,
        secondary_partition: g.secondary_partition,
        frozen_host_sources: g.frozen_host_sources,
        inactive_partition_exit: g.inactive_partition_exit,
        private_tables: g.private_tables,
        workgroup_tables: g.workgroup_tables,
        shared_down_scales: g.shared_down_scales,
        shared_gate_up_factors: g.shared_gate_up_factors,
        four_row_down: g.four_row_down,
        two_row_gate_up: g.two_row_gate_up,
        three_member_arithmetic: g.three_member_arithmetic,
    };
    assert_eq!(hash(&old), hash(&g));
    assert_ne!(hash(&g), hash(&g.with_four_members().unwrap()));
}
