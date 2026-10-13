//! Explicit draft profile for the two existing fixed-public dual-device labs.
//!
//! Selection never depends on a filename. The smaller sidecar borrows only
//! embedding/head weights from the actual already-bound target. It does not
//! change the target model, private owners, runtime or conditioning schedule.

use crate::resident_mtp::{self, CloseReceipt, ResidentMtp};
use cubecl::{prelude::ComputeClient, wgpu::WgpuRuntime};
use kuno_text::{Error, GgufFile, Result, model::BoundGraph, mtp::MtpGraph};
use std::{ffi::OsStr, sync::Arc, sync::atomic::AtomicBool};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Profile {
    SelfContainedQ8,
    StrataQ2,
}

impl Profile {
    pub(crate) fn parse(value: Option<&OsStr>) -> std::result::Result<Self, &'static str> {
        match value {
            None => Ok(Self::SelfContainedQ8),
            Some(value) if value == "strata-q2" => Ok(Self::StrataQ2),
            _ => Err("MTP profile requires absent Q8 default or canonical strata-q2"),
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::SelfContainedQ8 => "self-contained-q8",
            Self::StrataQ2 => "strata-q2",
        }
    }

    pub(crate) fn public_bytes(self) -> u64 {
        match self {
            Self::SelfContainedQ8 => resident_mtp::PUBLIC_BYTES,
            Self::StrataQ2 if crate::strata_port::enabled() => {
                resident_mtp::strata_runtime_public_bytes()
            }
            Self::StrataQ2 => resident_mtp::STRATA_PUBLIC_BYTES,
        }
    }

    pub(crate) fn parse_public_english(
        self,
        value: Option<&OsStr>,
        canonical_public_scope: bool,
        broadcast: bool,
    ) -> std::result::Result<bool, &'static str> {
        match value {
            None => Ok(false),
            Some(value)
                if value == "english-public"
                    && (self == Self::SelfContainedQ8 || crate::strata_port::enabled())
                    && canonical_public_scope
                    && !broadcast =>
            {
                Ok(true)
            }
            _ => Err("English draft vocabulary requires canonical public Q8 fused scope"),
        }
    }

    pub(crate) fn public_bytes_with_english(self, english: bool) -> Result<u64> {
        if english {
            if self != Self::SelfContainedQ8 && !crate::strata_port::enabled() {
                return Err(Error::InvalidFormat("English vocabulary is Q8 only"));
            }
            Ok(self.public_bytes() + resident_mtp::ENGLISH_VOCAB_OWNED_BYTES)
        } else {
            Ok(self.public_bytes())
        }
    }

    pub(crate) fn validate_close_with_english(
        self,
        receipt: &CloseReceipt,
        english: bool,
    ) -> Result<()> {
        if !english {
            return self.validate_close(receipt);
        }
        if receipt.public_bytes != self.public_bytes_with_english(true)?
            || receipt.private.buffers != 42
            || receipt.private.bytes != 11_698_176
            || receipt.main_reserved != 0
        {
            return Err(Error::InvalidFormat(
                "English MTP complete owner receipt rejected",
            ));
        }
        Ok(())
    }

    fn validate_artifact(self, bytes: u64, tensors: usize) -> Result<()> {
        let expected = match self {
            Self::SelfContainedQ8 => (4_137_429_120, 34),
            Self::StrataQ2 => (889_017_216, 31),
        };
        if (bytes, tensors) != expected {
            return Err(Error::InvalidFormat(
                "selected original MTP artifact rejected",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_sidecar(self, sidecar: &GgufFile) -> Result<()> {
        self.validate_artifact(sidecar.file_len(), sidecar.tensors().len())
    }

    pub(crate) fn bind<'a>(
        self,
        sidecar: &'a GgufFile,
        target: BoundGraph<'a, 'a>,
    ) -> Result<MtpGraph<'a>> {
        self.validate_sidecar(sidecar)?;
        match self {
            Self::SelfContainedQ8 => MtpGraph::bind(sidecar),
            Self::StrataQ2 => MtpGraph::bind_strata_q2(sidecar, target),
        }
    }

    pub(crate) fn resident<'g, 'f>(
        self,
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
        fused_sweeps: bool,
    ) -> Result<ResidentMtp<'g, 'f>> {
        match (self, fused_sweeps) {
            (Self::SelfContainedQ8, false) => ResidentMtp::new(graph, client, limits, retired),
            (Self::SelfContainedQ8, true) => {
                ResidentMtp::new_with_fused_sweeps(graph, client, limits, retired)
            }
            (Self::StrataQ2, false) => ResidentMtp::new_strata_q2(graph, client, limits, retired),
            (Self::StrataQ2, true) => {
                ResidentMtp::new_strata_q2_with_fused_sweeps(graph, client, limits, retired)
            }
        }
    }

    /// The selected public plan must be the one actually released. Successful
    /// workspace close already proves full private owning erasure; physical
    /// GPU-memory recovery remains the protected runner's separate check.
    /// this also binds the numeric receipt to its immutable draft profile.
    pub(crate) fn validate_close(self, receipt: &CloseReceipt) -> Result<()> {
        if receipt.public_bytes != self.public_bytes()
            || receipt.private.buffers != 42
            || receipt.private.bytes != 11_698_176
            || receipt.main_reserved != 0
        {
            return Err(Error::InvalidFormat(
                "selected MTP complete owner receipt rejected",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_arena::OwnedZeroAudit;

    #[test]
    fn exact_opt_in_has_no_alias_auto_or_filename_selection() {
        assert_eq!(Profile::parse(None), Ok(Profile::SelfContainedQ8));
        assert_eq!(
            Profile::parse(Some(OsStr::new("strata-q2"))),
            Ok(Profile::StrataQ2)
        );
        for malformed in [
            "",
            "q8",
            "q2",
            "auto",
            "strata-Q2",
            "strata-q2 ",
            "strata-q2\n",
            "mtp-q2_0.gguf",
        ] {
            assert!(Profile::parse(Some(OsStr::new(malformed))).is_err());
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            assert!(Profile::parse(Some(OsStr::from_bytes(b"strata-q2\xff"))).is_err());
        }
    }

    #[test]
    fn english_vocabulary_is_an_independent_bounded_public_q8_opt_in() {
        for profile in [Profile::SelfContainedQ8, Profile::StrataQ2] {
            for scope in [false, true] {
                for broadcast in [false, true] {
                    assert_eq!(
                        profile.parse_public_english(None, scope, broadcast),
                        Ok(false)
                    );
                    assert_eq!(
                        profile
                            .parse_public_english(
                                Some(OsStr::new("english-public")),
                                scope,
                                broadcast
                            )
                            .is_ok(),
                        profile == Profile::SelfContainedQ8 && scope && !broadcast,
                    );
                }
            }
        }
        for raw in [
            "",
            "english",
            "auto",
            "full",
            "ENGLISH-PUBLIC",
            "english-public ",
            "english-public\n",
        ] {
            assert!(
                Profile::SelfContainedQ8
                    .parse_public_english(Some(OsStr::new(raw)), true, false)
                    .is_err()
            );
        }
    }

    #[test]
    fn english_public_map_charge_and_full_private_close_are_independent() {
        assert_eq!(resident_mtp::ENGLISH_VOCAB_OWNED_BYTES, 262_144);
        assert_eq!(
            resident_mtp::ENGLISH_PUBLIC_BYTES,
            resident_mtp::PUBLIC_BYTES + 262_144
        );
        let full = CloseReceipt {
            private: OwnedZeroAudit {
                buffers: 42,
                bytes: 11_698_176,
            },
            public_bytes: resident_mtp::PUBLIC_BYTES,
            main_reserved: 0,
        };
        let english = CloseReceipt {
            public_bytes: resident_mtp::ENGLISH_PUBLIC_BYTES,
            ..full
        };
        Profile::SelfContainedQ8
            .validate_close_with_english(&full, false)
            .unwrap();
        Profile::SelfContainedQ8
            .validate_close_with_english(&english, true)
            .unwrap();
        assert!(
            Profile::SelfContainedQ8
                .validate_close_with_english(&full, true)
                .is_err()
        );
        assert!(
            Profile::SelfContainedQ8
                .validate_close_with_english(&english, false)
                .is_err()
        );
        assert!(Profile::StrataQ2.public_bytes_with_english(true).is_err());
        assert!(
            Profile::StrataQ2
                .validate_close_with_english(&english, true)
                .is_err()
        );
        for bad in [
            CloseReceipt {
                private: OwnedZeroAudit {
                    buffers: 41,
                    ..english.private
                },
                ..english
            },
            CloseReceipt {
                private: OwnedZeroAudit {
                    bytes: 11_698_175,
                    ..english.private
                },
                ..english
            },
            CloseReceipt {
                main_reserved: 65_536,
                ..english
            },
            CloseReceipt {
                public_bytes: english.public_bytes - 1,
                ..english
            },
        ] {
            assert!(
                Profile::SelfContainedQ8
                    .validate_close_with_english(&bad, true)
                    .is_err()
            );
        }
    }

    #[test]
    fn sidecar_extents_are_profile_specific_and_cannot_cross_admit() {
        for (profile, bytes, tensors) in [
            (Profile::SelfContainedQ8, 4_137_429_120, 34),
            (Profile::StrataQ2, 889_017_216, 31),
        ] {
            profile.validate_artifact(bytes, tensors).unwrap();
            for (bad_bytes, bad_tensors) in [
                (bytes - 1, tensors),
                (bytes + 1, tensors),
                (bytes, tensors - 1),
                (bytes, tensors + 1),
            ] {
                assert!(profile.validate_artifact(bad_bytes, bad_tensors).is_err());
            }
        }
        assert!(
            Profile::SelfContainedQ8
                .validate_artifact(889_017_216, 31)
                .is_err()
        );
        assert!(
            Profile::StrataQ2
                .validate_artifact(4_137_429_120, 34)
                .is_err()
        );
    }

    #[test]
    fn selected_public_plan_keeps_private_runtime_host_and_source_bounds() {
        assert_eq!(Profile::SelfContainedQ8.public_bytes(), 5_011_668_992);
        assert_eq!(Profile::StrataQ2.public_bytes(), 2_539_257_856);
        assert_eq!(resident_mtp::MAIN_BYTES, 6 * 1024 * 1024 * 1024);
        assert_eq!(resident_mtp::PRIVATE_BYTES, 16 * 1024 * 1024);
        assert_eq!(resident_mtp::RUNTIME_BYTES, 64 * 1024 * 1024);
        assert_eq!(resident_mtp::HOST_BYTES, 16 * 1024 * 1024);
        assert_eq!(resident_mtp::PUBLIC_SOURCE_BYTES, 128 * 1024 * 1024);
        for profile in [Profile::SelfContainedQ8, Profile::StrataQ2] {
            let receipt = CloseReceipt {
                private: OwnedZeroAudit {
                    buffers: 42,
                    bytes: 11_698_176,
                },
                public_bytes: profile.public_bytes(),
                main_reserved: 0,
            };
            profile.validate_close(&receipt).unwrap();
            let alternate = match profile {
                Profile::SelfContainedQ8 => Profile::StrataQ2,
                Profile::StrataQ2 => Profile::SelfContainedQ8,
            };
            assert!(alternate.validate_close(&receipt).is_err());
            for bad in [
                CloseReceipt {
                    private: OwnedZeroAudit {
                        buffers: 41,
                        ..receipt.private
                    },
                    ..receipt
                },
                CloseReceipt {
                    private: OwnedZeroAudit {
                        bytes: 11_698_175,
                        ..receipt.private
                    },
                    ..receipt
                },
                CloseReceipt {
                    main_reserved: 32768,
                    ..receipt
                },
            ] {
                assert!(profile.validate_close(&bad).is_err());
            }
        }
    }
}
