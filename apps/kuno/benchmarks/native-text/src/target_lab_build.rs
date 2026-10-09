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
