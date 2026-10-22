//! Actual structural/one-of-owner/lifetime helpers, with metadata-only handles.
//! No client, physical buffer, copying, device completion or erasure is faked.
//! The real factory still independently attests full backend Vulkan resources.
use super::*;
use cubecl_common::stream_id::StreamId;

const RAW: u64 = crate::strata_prompt::allocation::RAW_RING_BYTES;

#[test]
fn raw_bank_enrollment_requires_distinct_whole_exact_owners_and_preserves_single_root() {
    let stream = StreamId::current();
    let primary = Handle::new(stream, RAW);
    let secondary = Handle::new(stream, RAW);
    assert!(validate_prompt_raw_bank_shapes(&primary, None, stream, false).is_ok());
    // The old one-root API retains its genuine larger cache-root envelope.
    let larger = Handle::new(stream, RAW * 2);
    assert!(validate_prompt_raw_bank_shapes(&larger, None, stream, false).is_ok());
    assert_eq!(
        validate_prompt_raw_bank_shapes(&primary, Some(&secondary), stream, false).is_ok(),
        cfg!(feature = "strata-expert-cohorts")
    );
    for invalid in [
        primary.clone(),
        Handle::new(stream, RAW - 256),
        Handle::new(stream, RAW + 256),
        Handle::new(stream, RAW * 2).offset_end(RAW),
        Handle::new(stream, RAW * 2).offset_start(RAW),
        Handle::new(
            StreamId {
                value: stream.value.wrapping_add(1),
            },
            RAW,
        ),
    ] {
        assert!(validate_prompt_raw_bank_shapes(&primary, Some(&invalid), stream, false).is_err());
    }
    assert!(validate_prompt_raw_bank_shapes(&larger, Some(&secondary), stream, false).is_err());
    assert!(validate_prompt_raw_bank_shapes(&primary, Some(&secondary), stream, true).is_err());
    assert!(
        validate_prompt_raw_bank_shapes(
            &primary,
            None,
            StreamId {
                value: stream.value.wrapping_add(1)
            },
            false,
        )
        .is_err()
    );
}

#[test]
fn raw_bank_copy_authority_is_exactly_one_of_current_enrolled_whole_roots() {
    let stream = StreamId::current();
    let primary = Handle::new(stream, RAW);
    let secondary = Handle::new(stream, RAW);
    let bank = |destination: &Handle| {
        enrolled_prompt_raw_bank(Some(&primary), Some(&secondary), destination, false)
            .map(|root| root.clone().binding().memory.allocation_id())
    };
    assert_eq!(
        bank(&primary).unwrap(),
        primary.clone().binding().memory.allocation_id()
    );
    assert_eq!(
        bank(&secondary).unwrap(),
        secondary.clone().binding().memory.allocation_id()
    );
    let mut foreign_stream = secondary.clone();
    foreign_stream.stream = StreamId {
        value: stream.value.wrapping_add(1),
    };
    for invalid in [
        Handle::new(stream, RAW), // equal extent is not enrollment authority
        secondary.clone().offset_start(256),
        secondary.clone().offset_end(256),
        foreign_stream,
    ] {
        assert!(bank(&invalid).is_err());
    }
    // An old root cannot be selected after a different set is enrolled.
    let replacement = Handle::new(stream, RAW);
    assert!(enrolled_prompt_raw_bank(Some(&replacement), None, &primary, false).is_err());
    assert!(enrolled_prompt_raw_bank(None, Some(&secondary), &secondary, false).is_err());
    assert!(enrolled_prompt_raw_bank(Some(&primary), None, &secondary, false).is_err());
}

#[test]
fn raw_bank_loans_cannot_carry_secondary_enrollment_or_copy_authority() {
    let stream = StreamId::current();
    let primary = Handle::new(stream, RAW * 2);
    let secondary = Handle::new(stream, RAW);
    assert!(single_prompt_raw_bank_only(None).is_ok());
    assert!(single_prompt_raw_bank_only(Some(&secondary)).is_err());
    assert!(enrolled_prompt_raw_bank(Some(&primary), None, &primary, true).is_ok());
    for destination in [&primary, &secondary] {
        assert!(
            enrolled_prompt_raw_bank(Some(&primary), Some(&secondary), destination, true,).is_err()
        );
    }
}

#[test]
fn raw_bank_joined_close_releases_both_and_unknown_completion_retains_both() {
    let stream = StreamId::current();
    let primary = Handle::new(stream, RAW);
    let secondary = Handle::new(stream, RAW);
    // Real can_mut allows the ordinary arena+caller pair of handles. Keep
    // that original pair alive so Sources' additional retained owner is the
    // third reference whose release/quarantine is observed below.
    let _arena_primary = primary.clone();
    let _arena_secondary = secondary.clone();
    let mut attached_primary = Some(primary.clone());
    let mut attached_secondary = Some(secondary.clone());
    assert!(!primary.can_mut() && !secondary.can_mut());
    release_prompt_raw_banks(&mut attached_primary, &mut attached_secondary, false);
    assert!(attached_primary.is_none() && attached_secondary.is_none());
    assert!(primary.can_mut() && secondary.can_mut());

    attached_primary = Some(primary.clone());
    attached_secondary = Some(secondary.clone());
    release_prompt_raw_banks(&mut attached_primary, &mut attached_secondary, true);
    assert!(attached_primary.is_none() && attached_secondary.is_none());
    // These two fixed metadata-only references deliberately remain quarantined,
    // like the real source destructor. There is no GPU allocation or payload.
    assert!(!primary.can_mut() && !secondary.can_mut());
}
