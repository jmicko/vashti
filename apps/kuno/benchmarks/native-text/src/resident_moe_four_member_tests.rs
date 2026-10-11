//! Actual optional setter structure, without allocating any model/GPU owners.
fn actual_frozen_four_scope() -> super::FrozenFourChunkScope {
    super::FrozenFourChunkScope {
        source_four: true,
        workspace_four: true,
        target_four_reserved: true,
        target_four_active: true,
        target_rows_reserved: true,
        frozen_router_reserved: true,
        frozen_router_active: true,
        router_four_reserved: true,
        banks_frozen: true,
        source_identity_matches: true,
        gu_integer_sources_and_owners: true,
        down_integer_sources_and_owners: true,
        original_geometry: true,
        ordinary_modes_absent: true,
    }
}

#[test]
fn actual_frozen_four_chunk_requires_each_source_owner_and_active_permission() {
    let scope = actual_frozen_four_scope();
    scope.validate(4, 1).unwrap();
    let missing: [fn(&mut super::FrozenFourChunkScope); 14] = [
        |s| s.source_four = false,
        |s| s.workspace_four = false,
        |s| s.target_four_reserved = false,
        |s| s.target_four_active = false,
        |s| s.target_rows_reserved = false,
        |s| s.frozen_router_reserved = false,
        |s| s.frozen_router_active = false,
        |s| s.router_four_reserved = false,
        |s| s.banks_frozen = false,
        |s| s.source_identity_matches = false,
        |s| s.gu_integer_sources_and_owners = false,
        |s| s.down_integer_sources_and_owners = false,
        |s| s.original_geometry = false,
        |s| s.ordinary_modes_absent = false,
    ];
    for remove in missing {
        let mut invalid = scope;
        remove(&mut invalid);
        assert!(invalid.validate(4, 1).is_err());
    }
}

#[test]
fn frozen_four_permission_never_widens_the_original_demand_count_or_wave_guard() {
    let scope = actual_frozen_four_scope();
    for count in [0, 1, 2, 3, 5, 8, usize::MAX] {
        assert!(scope.validate(count, 1).is_err());
    }
    for waves in [0, 2, usize::MAX] {
        assert!(scope.validate(4, waves).is_err());
    }
    // Original default/demand behavior stays authoritative for count1..3.
    for count in 1..=3 {
        super::public_existing_count(true, count, 1).unwrap();
    }
    assert!(super::public_existing_count(true, 4, 1).is_err());
    super::public_existing_count(false, 4, 1).unwrap();
}

#[test]
fn four_member_workspace_selection_uses_real_permission_and_original_completed_boundary() {
    let source = include_str!("resident_moe_batch.rs");
    assert!(source.contains("public_four_members: false"));
    let body = source
        .split_once("pub(crate) fn set_public_four_members_policy(")
        .unwrap()
        .1
        .split_once("pub(crate) fn public_existing_fill_enabled")
        .unwrap()
        .0;
    assert!(
        body.find("validate_public_warm_boundary").unwrap()
            < body.find("self.public_four_members = true").unwrap()
    );
    for guard in [
        "!executor.target_four_rows_reserved",
        "!executor.target_frozen_gu_integer_reserved",
        "!executor.target_frozen_down_integer_reserved",
        "self.frozen_gu_integer.is_none()",
        "self.frozen_down_integer.is_none()",
        "self.frozen_sources.is_some()",
        "self.batch_router_active()",
        "!self.four_member_router_reserved()",
        "!self.public_existing_fill",
        "self.layout.capacity != 144",
        "self.layout.geometries.len() != 48",
        "self.public_four_members && !enabled",
    ] {
        assert!(body.contains(guard), "{guard}");
    }
    assert!(body.contains("for &original in &self.layout.geometries"));
    assert!(body.contains("and_then(k::Geometry::with_four_members)"));
    assert!(!body.contains("allocate_zeroed"));
    assert!(source.contains("if w.public_four_members_policy() { 4 } else { 3 }"));
}

#[test]
fn four_member_workspace_full_router_layout_uses_real_constructor_reservation() {
    let source = include_str!("resident_moe_batch.rs");
    let body = source
        .split_once("fn validate_ids(&self, arena:")
        .unwrap()
        .1
        .split_once("if let Some(ids) = &self.frozen_staging")
        .unwrap()
        .0;
    assert!(body.contains("if self.four_member_router_reserved()"));
    assert!(body.contains("crate::resident_batch_router_gpu::four::AUDIT_WORDS"));
    assert!(body.contains("crate::resident_batch_router_gpu::AUDIT_WORDS"));
    assert!(!body.contains("if self.public_four_members"));
    let old = crate::resident_batch_router_gpu::AUDIT_WORDS;
    let four = crate::resident_batch_router_gpu::four::AUDIT_WORDS;
    assert_eq!((old, four), (448128, 448224));
    let old = crate::private_arena::BufferLayout::new(old).unwrap();
    let four = crate::private_arena::BufferLayout::new(four).unwrap();
    assert_eq!(old.owned_words, four.owned_words);
    assert_ne!(old, four);
}

#[test]
fn four_member_pending_acceptance_uses_selected_bound_and_actual_source_geometry() {
    let source = include_str!("resident_moe_batch.rs");
    let body = source
        .split_once("if let Some(proof) = &p.frozen_sources {")
        .unwrap()
        .1
        .split_once("if self.frozen_sources.is_some()")
        .unwrap()
        .0;
    let normalized = body.split_whitespace().collect::<Vec<_>>().join(" ");
    assert!(normalized.contains("if self.public_four_members_policy() { 4 } else { 3 }"));
    assert!(!normalized.contains("(1..=3).contains(&s.stamp.count)"));
    assert!(body.contains("g.validate_count(s.stamp.count).map_err(err)?"));
    assert!(
        body.find("proof.validate(*p.layer, bank)?").unwrap()
            < body.find("g.validate_count(s.stamp.count)").unwrap()
    );
    assert!(
        body.find("g.validate_count(s.stamp.count)").unwrap()
            < body.find("source.physical_with_router(").unwrap()
    );
}
