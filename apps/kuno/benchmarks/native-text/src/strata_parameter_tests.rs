//! Run the actual vendor branch-layout tests without building a second GPU
//! runtime workspace. Only its error-construction boundary is substituted.
#![allow(dead_code)]
mod transient_io {
    pub(crate) fn fault(reason: &'static str) -> cubecl_runtime::server::ServerError {
        cubecl_runtime::server::ServerError::Generic {
            reason: reason.into(),
            backtrace: Default::default(),
        }
    }
}
#[path = "../vendor/cubecl-wgpu/src/compute/strata_parameters.rs"]
mod parameters;

#[test]
fn independent_gdn_and_gpu_ple_roles_name_four_distinct_original_inventories() {
    use cubecl_runtime::server::NativeStrataBranch;
    for rows in 1..=8 {
        let keys: Vec<_> = (0..4)
            .map(|role| NativeStrataBranch::target_window_with_state(rows, 4096, role).unwrap())
            .collect();
        for (role, key) in keys.iter().enumerate() {
            assert_eq!(key.target_geometry(), Some((rows, 4096, role as u8)));
            assert!(!keys[..role].contains(key));
        }
        assert!(NativeStrataBranch::target_window_with_state(rows, 4096, 4).is_err());
    }
}

#[test]
fn context32768_controls_preserve_original_keys_and_check_readable_vs_boundary() {
    use cubecl_runtime::server::{
        NativeStrataBranch as B, NativeStrataInvocation as I, NativeStrataValue as V,
    };
    for rows in 1u8..=8 {
        let start = 32768 - u32::from(rows);
        for role in 0..4 {
            let key = B::target_window_with_state(rows, 32768, role).unwrap();
            assert_eq!(key.target_geometry(), Some((rows, 32768, role)));
            let input = I::target(start, 1).unwrap();
            input.validate_for(key).unwrap();
            assert!(
                I::target(start + 1, 1).is_err()
                    || I::target(start + 1, 1).unwrap().validate_for(key).is_err()
            );
            assert!(
                input
                    .validate_for(B::target_window_with_state(rows, 4096, role).unwrap())
                    .is_err()
            );
            if rows > 1 {
                I::commit(start, u32::from(rows))
                    .unwrap()
                    .validate_for(B::target_commit(rows, 32768, role).unwrap())
                    .unwrap();
            }
        }
        for proposals in 1u8..=7 {
            let key = B::mtp_round_for_context(rows, proposals, 32768).unwrap();
            let origin = 32768 - u32::from(rows + proposals - 1);
            I::mtp(origin).unwrap().validate_for(key).unwrap();
            assert!(
                I::mtp(origin + 1).is_err()
                    || I::mtp(origin + 1).unwrap().validate_for(key).is_err()
            );
            assert_ne!(key, B::mtp_round(rows, proposals).unwrap());
        }
    }
    let last = I::mtp(32767).unwrap();
    assert_eq!(last.value(V::Position { offset: 0 }).unwrap(), 32767);
    assert_eq!(last.value(V::Boundary { offset: 1 }).unwrap(), 32768);
    assert!(last.value(V::Position { offset: 1 }).is_err());
    assert!(last.value(V::Boundary { offset: 2 }).is_err());
    for step in 1..=6 {
        last.validate_for(B::mtp_step_for_context(step, 32768).unwrap())
            .unwrap();
    }
    assert!(I::mtp(32768).is_err());
    assert!(B::target_window_for_context(1, 32769).is_err());
}
