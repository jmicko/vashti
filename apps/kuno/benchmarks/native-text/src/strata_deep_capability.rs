//! Compiled-public diagnostics of actual deep-matrix device capabilities.
//! These gates never replace a producer's full geometry/owner/CMMA checks.
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};

pub(crate) const REQUIRED_SHARED_BYTES: usize = 65536;
pub(crate) const DEEP_DENSE_CASES: usize = 21;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Capability {
    pub(crate) max_shared_memory_bytes: usize,
    pub(crate) actual_explicit_shared_layout: bool,
}

impl Capability {
    pub(crate) fn actual(client: &ComputeClient<WgpuRuntime>) -> Self {
        Self {
            max_shared_memory_bytes: client.properties().hardware.max_shared_memory_size,
            // Query the actual compiler/server even on devices below64KiB.
            // The observation is distinct from the capacity selection below.
            actual_explicit_shared_layout: client.native_explicit_shared_layout(),
        }
    }

    pub(crate) fn dense_selected(self) -> bool {
        self.max_shared_memory_bytes >= REQUIRED_SHARED_BYTES
    }

    pub(crate) fn routed_selected(self) -> bool {
        self.dense_selected() && self.actual_explicit_shared_layout
    }

    pub(crate) fn report(self) -> serde_json::Value {
        serde_json::json!({
            "compiled": true,
            "selection_scope": "actual-device-capability-gate; fixture geometry and actual dispatch are checked separately",
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
            "required_shared_memory_bytes": REQUIRED_SHARED_BYTES,
            "actual_explicit_shared_layout": self.actual_explicit_shared_layout,
            // Dense has no publication scratch and does not require aliasing.
            "deep_dense_selected": self.dense_selected(),
            "deep_macro_selected": self.routed_selected(),
            "deep_wide_selected": self.routed_selected(),
            "dense_requires_explicit_shared_layout": false,
            "routed_requires_explicit_shared_layout": true,
            "original_gpu_controls_retained": true,
        })
    }

    pub(crate) fn unavailable_dense(self) -> Option<serde_json::Value> {
        (!self.dense_selected()).then(|| {
            serde_json::json!({
                "family": "dense-deep-k64-finite-paired",
                "status": "unavailable",
                "reason": "insufficient_shared_memory",
                "expected_case_count": DEEP_DENSE_CASES,
                "executed_case_count": 0,
                "max_shared_memory_bytes": self.max_shared_memory_bytes,
                "required_shared_memory_bytes": REQUIRED_SHARED_BYTES,
                "gpu_math_exercised": false,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_observation_policy_keeps48k_and64k_boundaries_distinct() {
        for max in [0, 32768, 49152, 65535, 65536, 81920] {
            for explicit in [false, true] {
                let observed = Capability {
                    max_shared_memory_bytes: max,
                    actual_explicit_shared_layout: explicit,
                };
                assert_eq!(observed.dense_selected(), max >= 65536);
                assert_eq!(observed.routed_selected(), max >= 65536 && explicit);
                assert_eq!(observed.report()["max_shared_memory_bytes"], max);
                assert_eq!(observed.report()["actual_explicit_shared_layout"], explicit);
            }
        }
    }

    #[test]
    fn unsupported_gpu_cases_are_unavailable_and_never_reported_as_passed() {
        let observed = Capability {
            max_shared_memory_bytes: 49152,
            actual_explicit_shared_layout: true,
        };
        let report = observed.report();
        for key in [
            "deep_dense_selected",
            "deep_macro_selected",
            "deep_wide_selected",
        ] {
            assert_eq!(report[key], false);
        }
        let unavailable = observed.unavailable_dense().unwrap();
        assert_eq!(unavailable["status"], "unavailable");
        assert_eq!(unavailable["expected_case_count"], 21);
        assert_eq!(unavailable["executed_case_count"], 0);
        assert_eq!(unavailable["gpu_math_exercised"], false);
        assert!(unavailable.get("passed").is_none());
        assert!(unavailable.get("mismatches").is_none());
    }

    #[test]
    fn dense_without_aliasing_remains_available_when_routed_aliasing_is_absent() {
        let observed = Capability {
            max_shared_memory_bytes: 65536,
            actual_explicit_shared_layout: false,
        };
        assert!(observed.dense_selected());
        assert!(!observed.routed_selected());
        assert!(observed.unavailable_dense().is_none());
    }
}
