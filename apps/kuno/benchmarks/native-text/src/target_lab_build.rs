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

/// No count-based or production selection: the controllers pass their exact
/// validated public frozen Verify3 scope, and reject arbitrary values.
pub(crate) fn three_gdn_streaming(
    raw: Option<&OsStr>,
    public_frozen_scope: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "three-streaming-public" && public_frozen_scope => Ok(true),
        _ => Err("three-token GDN requires canonical original public frozen Verify3 request scope"),
    }
}

/// The immutable GU arithmetic candidate is admitted only by the two compiled
/// public controllers after every original source and request prerequisite.
pub(crate) fn frozen_gu_integer(
    raw: Option<&OsStr>,
    public_frozen_scope: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "integer-public" && public_frozen_scope => Ok(true),
        _ => Err("integer GU requires canonical original public frozen Verify3 scope"),
    }
}

pub(crate) fn frozen_gu_stride_four(
    raw: Option<&OsStr>,
    integer: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "four-public" && integer => Ok(true),
        _ => Err("four-stride GU requires canonical selected public integer math"),
    }
}

pub(crate) fn frozen_gu_stride_four_record(
    selected: bool,
    actual: bool,
    integer_actual: bool,
    integer_dispatches: u64,
    stride_dispatches: u64,
) -> Result<serde_json::Value, &'static str> {
    if selected != actual
        || if selected {
            !integer_actual
                || integer_dispatches == 0
                || !integer_dispatches.is_multiple_of(240)
                || stride_dispatches != integer_dispatches
        } else {
            stride_dispatches != 0
        }
    {
        return Err("four-stride GU actual joined dispatch receipt rejected");
    }
    Ok(serde_json::json!({"gu_dispatches":stride_dispatches}))
}

/// These are completed source-specific counts, never requested settings or
/// generic projection dispatches. Every accepted target batch visits 48 layers.
pub(crate) fn frozen_gu_integer_record(
    selected: bool,
    actual: bool,
    preparations: u64,
    dispatches: u64,
    audits: u64,
) -> Result<serde_json::Value, &'static str> {
    if selected != actual
        || if selected {
            preparations == 0
                || !preparations.is_multiple_of(48)
                || preparations.checked_mul(5) != Some(dispatches)
                || preparations != audits
        } else {
            preparations != 0 || dispatches != 0 || audits != 0
        }
    {
        return Err("integer GU actual joined dispatch receipt rejected");
    }
    Ok(serde_json::json!({"preparations":preparations,
        "gu_dispatches":dispatches,"full_owner_audits":audits}))
}

/// The warm replay's first original-reference token consumes exactly one real
/// count1 batch after immutable source attachment at cursor0.
pub(crate) fn frozen_gu_integer_initial_record(
    selected: bool,
    actual: bool,
    preparations: u64,
    dispatches: u64,
    audits: u64,
) -> Result<serde_json::Value, &'static str> {
    let record = frozen_gu_integer_record(selected, actual, preparations, dispatches, audits)?;
    if selected && (preparations != 48 || dispatches != 240 || audits != 48) {
        return Err("integer GU original initial count1 receipt rejected");
    }
    Ok(record)
}

/// Initial numeric acceptance fixes every optional owner policy. Other public
/// controllers retain their existing close validation and exact owner proofs.
pub(crate) fn validate_frozen_gu_integer_close(
    selected: bool,
    main_buffers: u64,
    main_bytes: u64,
) -> Result<(), &'static str> {
    if selected && (main_buffers != 1381 || main_bytes != 828_932_096) {
        return Err("integer GU complete MAIN owner closure rejected");
    }
    Ok(())
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
    fn frozen_gu_stride_four_controller_flag_requires_canonical_integer_dependency() {
        for integer in [false, true] {
            assert_eq!(frozen_gu_stride_four(None, integer), Ok(false));
            for raw in ["four-public", "four", "true", "", "four-public "] {
                assert_eq!(
                    frozen_gu_stride_four(Some(OsStr::new(raw)), integer).is_ok(),
                    integer && raw == "four-public"
                );
            }
        }
    }
    #[test]
    fn frozen_gu_stride_four_controller_receipt_requires_actual_complete_integer_dispatches() {
        assert_eq!(
            frozen_gu_stride_four_record(false, false, false, 0, 0).unwrap(),
            serde_json::json!({"gu_dispatches":0})
        );
        assert!(frozen_gu_stride_four_record(false, false, true, 240, 0).is_ok());
        for dispatches in [240, 480, 17760] {
            assert_eq!(
                frozen_gu_stride_four_record(true, true, true, dispatches, dispatches).unwrap(),
                serde_json::json!({"gu_dispatches":dispatches})
            );
        }
        for (selected, actual, integer, dispatches, stride) in [
            (true, false, true, 240, 240),
            (false, true, true, 240, 240),
            (true, true, false, 240, 240),
            (true, true, true, 0, 0),
            (true, true, true, 239, 239),
            (true, true, true, 240, 239),
            (true, true, true, 240, 241),
            (false, false, true, 240, 1),
        ] {
            assert!(
                frozen_gu_stride_four_record(selected, actual, integer, dispatches, stride)
                    .is_err()
            );
        }
    }
    #[test]
    fn frozen_gu_integer_controller_flag_is_canonical_and_default_off() {
        for scope in [false, true] {
            assert_eq!(frozen_gu_integer(None, scope), Ok(false));
            for value in ["integer-public", "integer", "true", "", "integer-public "] {
                assert_eq!(
                    frozen_gu_integer(Some(OsStr::new(value)), scope).is_ok(),
                    scope && value == "integer-public"
                );
            }
        }
    }
    #[test]
    fn frozen_gu_integer_controller_receipts_require_actual_all_layer_joined_evidence() {
        assert_eq!(
            frozen_gu_integer_record(false, false, 0, 0, 0).unwrap(),
            serde_json::json!({"preparations":0,"gu_dispatches":0,"full_owner_audits":0})
        );
        for batches in [1, 9, 74] {
            let preparations = batches * 48;
            assert_eq!(
                frozen_gu_integer_record(true, true, preparations, preparations * 5, preparations)
                    .unwrap(),
                serde_json::json!({"preparations":preparations,
                    "gu_dispatches":preparations*5,"full_owner_audits":preparations})
            );
        }
        for (selected, actual, preparations, dispatches, audits) in [
            (true, false, 48, 240, 48),
            (false, true, 48, 240, 48),
            (true, true, 0, 0, 0),
            (true, true, 1, 5, 1),
            (true, true, 48, 239, 48),
            (true, true, 48, 240, 47),
            (true, true, u64::MAX / 48 * 48, 0, 0),
            (false, false, 48, 240, 48),
            (false, false, 0, 1, 0),
            (false, false, 0, 0, 1),
        ] {
            assert!(
                frozen_gu_integer_record(selected, actual, preparations, dispatches, audits)
                    .is_err()
            );
        }
    }
    #[test]
    fn frozen_gu_integer_controller_close_requires_complete_extra_owner() {
        assert!(validate_frozen_gu_integer_close(true, 1381, 828_932_096).is_ok());
        for (buffers, bytes) in [
            (1380, 828_866_560),
            (1380, 828_932_096),
            (1381, 828_866_560),
            (1382, 828_932_096),
            (1381, 828_932_095),
            (1381, 828_932_097),
        ] {
            assert!(validate_frozen_gu_integer_close(true, buffers, bytes).is_err());
            assert!(validate_frozen_gu_integer_close(false, buffers, bytes).is_ok());
        }
    }
    #[test]
    fn frozen_gu_integer_controller_initial_count1_requires_exact_first_batch() {
        assert!(frozen_gu_integer_initial_record(false, false, 0, 0, 0).is_ok());
        assert!(frozen_gu_integer_initial_record(true, true, 48, 240, 48).is_ok());
        for (actual, preparations, dispatches, audits) in [
            (false, 48, 240, 48),
            (true, 0, 0, 0),
            (true, 96, 480, 96),
            (true, 48, 240, 47),
        ] {
            assert!(
                frozen_gu_integer_initial_record(true, actual, preparations, dispatches, audits)
                    .is_err()
            );
        }
    }
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
