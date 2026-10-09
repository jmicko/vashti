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
