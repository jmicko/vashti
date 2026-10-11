//! Actual optional setter structure, without allocating any model/GPU owners.
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
