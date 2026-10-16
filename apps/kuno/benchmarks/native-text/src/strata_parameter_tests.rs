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
