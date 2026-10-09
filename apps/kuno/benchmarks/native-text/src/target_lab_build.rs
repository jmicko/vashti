//! Dispatch-only build admission for three compiled-public DAS target fixtures.
//! No CPU arithmetic feature may accompany this comparison. Other laboratory
//! modes retain their existing build/dispatch policies.
use std::ffi::OsStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Build {
    Threaded,
    Inline,
    Rejected,
}
impl Build {
    const fn features(direct: bool, cpu: bool) -> Self {
        if cpu {
            Self::Rejected
        } else if direct {
            Self::Inline
        } else {
            Self::Threaded
        }
    }
    pub(crate) const fn compiled() -> Self {
        Self::features(
            cfg!(feature = "direct-device"),
            cfg!(any(
                feature = "cpu-fma",
                feature = "cpu-buffered-weights",
                feature = "cpu-work-stealing",
                feature = "cpu-q8-two-pass"
            )),
        )
    }
    pub(crate) fn selected(self, raw: Option<&OsStr>) -> Result<bool, &'static str> {
        match (self, raw) {
            (Self::Threaded, None) => Ok(false),
            (Self::Inline, Some(value)) if value == "inline" => Ok(true),
            _ => {
                Err("public target dispatch requires default or explicit direct-device-only build")
            }
        }
    }
}

/// A dedicated fixed-public stream, never a process-wide cache policy.
pub(crate) fn request_reuse(
    raw: Option<&OsStr>,
    inline: bool,
    chunk_io: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "request" && inline && chunk_io => Ok(true),
        _ => Err("public target reuse requires canonical request, direct dispatch and chunk I/O"),
    }
}

pub(crate) fn live_member_reduction(
    raw: Option<&OsStr>,
    prepared: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "live" && !prepared => Ok(true),
        _ => Err("public target live reductions require original F32 experts and canonical live"),
    }
}

pub(crate) fn host_io_snapshot() -> [u64; 6] {
    let (setup, flush, map) = cubecl::wgpu::transient_scoped_read_ns();
    let (wait_ns, waits) = cubecl::wgpu::transient_gpu_submission_wait_stats();
    [
        setup,
        flush,
        map,
        cubecl::wgpu::transient_inline_readback_batches(),
        wait_ns,
        waits,
    ]
}
pub(crate) fn host_io_delta(
    before: [u64; 6],
    after: [u64; 6],
) -> Result<serde_json::Value, &'static str> {
    let mut delta = [0; 6];
    for (out, (after, before)) in delta.iter_mut().zip(after.into_iter().zip(before)) {
        if before == u64::MAX || after == u64::MAX {
            return Err("public host I/O counter saturated");
        }
        *out = after
            .checked_sub(before)
            .ok_or("public host I/O counter regressed")?;
    }
    Ok(
        serde_json::json!({"provenance":"process-global whole public dual-device timed interval",
        "scoped_read_setup_ns":delta[0],"scoped_read_flush_ns":delta[1],
        "scoped_read_map_ns":delta[2],"inline_readback_batches":delta[3],
        "gpu_submission_wait_ns":delta[4],"gpu_submission_waits":delta[5]}),
    )
}

pub(crate) fn reuse_record(
    r: crate::resident_model::batch::public_target_request_io::Receipt,
) -> serde_json::Value {
    serde_json::json!({
        "scope_sequence":r.scope,"scope_closed":r.scope_closed,
        "full_io_audit":r.full_io_audit,"target_stream_configured":r.target_stream_configured,
        "metadata_reserved_bytes":r.metadata_bytes,"binding_capacity":r.binding_capacity,
        "runtime_limit_bytes":r.runtime_limit,"request_checkpoints":r.request_checkpoints,
        "request_io_reuses":r.request_io_reuses,"binding_stats":r.binding_stats,
        "info_stats":r.info_stats,"counter_provenance":"admission-local request checkpoints; checked process-global binding/INFO/request-reuse deltas",
        "closing_audit_ns":r.closing_audit_ns,
    })
}

pub(crate) fn actual_inline() -> bool {
    cubecl_common::device_handle::DeviceHandle::<
        <cubecl::wgpu::WgpuRuntime as cubecl::prelude::Runtime>::Server,
    >::is_blocking()
}
fn matching(selected: bool, actual: bool) -> Result<(), &'static str> {
    if selected != actual {
        return Err("public target dispatch does not match actual device handle");
    }
    Ok(())
}
pub(crate) fn validate_runtime(selected: bool) -> Result<(), &'static str> {
    matching(selected, actual_inline())?;
    match (Build::compiled(), selected) {
        (Build::Threaded, false) | (Build::Inline, true) => Ok(()),
        _ => Err("public target runtime includes unsupported arithmetic features"),
    }
}
pub(crate) fn name(inline: bool) -> &'static str {
    if inline { "inline" } else { "threaded" }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_build_flag_matrix_rejects_every_cpu_feature_combination() {
        for direct in [false, true] {
            // Four independent CPU flags. Any combination is forbidden.
            for cpu_bits in 0..16 {
                let build = Build::features(direct, cpu_bits != 0);
                for raw in [
                    None,
                    Some("inline"),
                    Some("threaded"),
                    Some(""),
                    Some("inline "),
                    Some("1"),
                ] {
                    let accepted = build.selected(raw.map(OsStr::new));
                    assert_eq!(
                        accepted.is_ok(),
                        cpu_bits == 0
                            && ((direct && raw == Some("inline")) || (!direct && raw.is_none()))
                    );
                }
            }
        }
    }
    #[test]
    fn request_reuse_is_canonical_and_requires_its_actual_inline_chunk_dependencies() {
        for inline in [false, true] {
            for chunk in [false, true] {
                assert_eq!(request_reuse(None, inline, chunk), Ok(false));
                for raw in ["request", "", "auto", "1", "request ", "chunk"] {
                    assert_eq!(
                        request_reuse(Some(OsStr::new(raw)), inline, chunk).is_ok(),
                        raw == "request" && inline && chunk,
                    );
                }
            }
        }
    }
    #[test]
    fn live_member_selection_keeps_original_default_and_rejects_prepared_math() {
        for prepared in [false, true] {
            assert_eq!(live_member_reduction(None, prepared), Ok(false));
            for raw in ["live", "", "auto", "1", "live "] {
                assert_eq!(
                    live_member_reduction(Some(OsStr::new(raw)), prepared).is_ok(),
                    raw == "live" && !prepared
                );
            }
        }
    }
    #[test]
    fn host_io_deltas_reject_unusable_counter_samples() {
        assert!(host_io_delta([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]).is_ok());
        for n in 0..6 {
            let mut regressed = [1; 6];
            regressed[n] = 0;
            assert!(host_io_delta([1; 6], regressed).is_err());
            let mut saturated = [1; 6];
            saturated[n] = u64::MAX;
            assert!(host_io_delta([1; 6], saturated).is_err());
            assert!(host_io_delta(saturated, saturated).is_err());
        }
    }
    #[test]
    fn actual_runtime_must_match_selected_dispatch_before_execution() {
        for selected in [false, true] {
            for actual in [false, true] {
                assert_eq!(matching(selected, actual).is_ok(), selected == actual);
            }
        }
        assert_eq!(actual_inline(), cfg!(feature = "direct-device"));
        assert_eq!(
            validate_runtime(actual_inline()).is_ok(),
            Build::compiled() != Build::Rejected
        );
        assert_eq!(
            Build::compiled(),
            Build::features(
                cfg!(feature = "direct-device"),
                cfg!(any(
                    feature = "cpu-fma",
                    feature = "cpu-buffered-weights",
                    feature = "cpu-work-stealing",
                    feature = "cpu-q8-two-pass"
                ))
            )
        );
    }
}
