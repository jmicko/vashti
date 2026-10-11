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

/// Explicit fixed-public draft placement. Default retains the RTX3060.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MtpDevice {
    Nvidia,
    Intel,
}
impl MtpDevice {
    pub(crate) fn parse(raw: Option<&OsStr>, scope: bool) -> Result<Self, &'static str> {
        match raw {
            None => Ok(Self::Nvidia),
            Some(raw) if raw == "intel-public" && scope => Ok(Self::Intel),
            _ => Err("Intel MTP requires canonical fixed public original target scope"),
        }
    }
    pub(crate) fn vendor(self) -> u32 {
        match self {
            Self::Nvidia => 0x10de,
            Self::Intel => 0x8086,
        }
    }
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia-default",
            Self::Intel => "intel-public",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntelMtpMemory {
    pub(crate) total: u64,
    pub(crate) used: u64,
    pub(crate) device: u32,
}
impl IntelMtpMemory {
    /// The protected runner supplies a fresh Xe MEM_REGIONS query while other
    /// inference is paused. Privilege-masked zero never means a free card.
    pub(crate) fn parse(
        selected: MtpDevice,
        total: Option<&OsStr>,
        used: Option<&OsStr>,
        device: Option<&OsStr>,
    ) -> Result<Option<Self>, &'static str> {
        if selected == MtpDevice::Nvidia {
            if total.is_some() || used.is_some() || device.is_some() {
                return Err("default Nvidia MTP rejects Intel memory metadata");
            }
            return Ok(None);
        }
        fn decimal(raw: Option<&OsStr>) -> Result<u64, &'static str> {
            let raw = raw
                .and_then(OsStr::to_str)
                .ok_or("Intel MTP memory metadata missing")?;
            if raw.is_empty()
                || !raw.bytes().all(|byte| byte.is_ascii_digit())
                || raw.len() > 1 && raw.starts_with('0')
            {
                return Err("Intel MTP memory metadata requires canonical decimal");
            }
            raw.parse()
                .map_err(|_| "Intel MTP memory metadata overflow")
        }
        let total = decimal(total)?;
        let used = decimal(used)?;
        let device = u32::try_from(decimal(device)?).map_err(|_| "Intel MTP device ID overflow")?;
        if total == 0 || used == 0 || used >= total || device == 0 {
            return Err("Intel MTP memory query is empty, masked or inconsistent");
        }
        Ok(Some(Self {
            total,
            used,
            device,
        }))
    }
    pub(crate) fn admit(
        self,
        target_device: u32,
        target_main: u64,
        draft_public: u64,
        banks: u64,
        target_source_pages: u64,
        target_baseline: u64,
    ) -> Result<serde_json::Value, &'static str> {
        if self.used == 0
            || self.used >= self.total
            || self.device == 0
            || target_device != self.device
            || banks != crate::daslab_profile::bank_bytes_144().physical
            || !matches!(target_main, 828_866_560 | 828_932_096)
            || !matches!(
                draft_public,
                crate::resident_mtp::PUBLIC_BYTES | crate::resident_mtp::STRATA_PUBLIC_BYTES
            )
            || target_source_pages != 30 * 1024 * 1024
            || target_baseline != 0
        {
            return Err("Intel MTP combined plan requires exact original fresh full144 owners");
        }
        let components = [
            banks,
            8 * 1024 * 1024 * 1024,
            target_main,
            25_165_824,
            crate::executor::RESIDENT_IO_MAX,
            crate::resident_model::batch::TARGET_ROWS_RUNTIME_LIMIT,
            target_source_pages,
            draft_public,
            crate::resident_mtp::PRIVATE_BYTES as u64,
            crate::resident_mtp::RUNTIME_BYTES,
            crate::resident_mtp::PUBLIC_SOURCE_BYTES as u64,
            512 * 1024 * 1024,
            target_baseline,
        ];
        let total = components.into_iter().try_fold(0u64, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or("Intel MTP combined plan overflow")
        })?;
        if total > self.total - self.used {
            return Err("Intel MTP combined full144 target and draft exceed available VRAM");
        }
        Ok(serde_json::json!({"source":"protected-xe-memory-regions",
            "physical_total_bytes":self.total,"physical_used_bytes":self.used,
            "physical_available_bytes":self.total-self.used,"physical_device_id":self.device,
            "target_slots":144,"target_bank_bytes":banks,"target_static_cache_bytes":8u64*1024*1024*1024,
            "target_main_bytes":target_main,"target_vector_bytes":25_165_824,
            "target_io_bytes":crate::executor::RESIDENT_IO_MAX,
            "target_runtime_io_bytes":crate::resident_model::batch::TARGET_ROWS_RUNTIME_LIMIT,
            "target_source_page_bytes":target_source_pages,"target_baseline_bytes":target_baseline,
            "draft_public_bytes":draft_public,"draft_private_cap_bytes":crate::resident_mtp::PRIVATE_BYTES,
            "draft_runtime_io_bytes":crate::resident_mtp::RUNTIME_BYTES,
            "draft_source_cap_bytes":crate::resident_mtp::PUBLIC_SOURCE_BYTES,
            "shared_driver_margin_bytes":512u64*1024*1024,
            "combined_reserved_bytes":total,"remaining_beyond_margin_bytes":self.total-self.used-total}))
    }
}

pub(crate) fn intel_mtp_preflight(
    proof: IntelMtpMemory,
    target: &wgpu::Device,
    client: &cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
    graph: kuno_text::model::BoundGraph<'_, '_>,
    integer: bool,
    profile: crate::mtp_lab_profile::Profile,
) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error>> {
    let info = target.adapter_info();
    if info.vendor != 0x8086
        || info.device_type != wgpu::DeviceType::DiscreteGpu
        || info.backend != wgpu::Backend::Vulkan
    {
        return Err("Intel MTP requires the actual discrete Vulkan target".into());
    }
    let usage = client.memory_usage()?;
    if usage.bytes_in_use != 0 || usage.bytes_reserved != 0 || usage.number_allocs != 0 {
        return Err("Intel MTP combined admission requires a fresh target runtime".into());
    }
    let banks = crate::expert_banks::preflight_residency_with_profile(
        graph,
        144,
        8 * 1024 * 1024 * 1024,
        31 * 1024 * 1024 * 1024,
        crate::expert_banks::OriginalPackedProfile::DaslabIq3,
    )? as u64;
    let mut source_pages = 0;
    for layer in 0..48 {
        let mut layer_pages = 0u64;
        for suffix in [
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        ] {
            let tensor = graph.tensor(&format!("blk.{layer}.{suffix}"))?;
            let bytes = tensor
                .info()
                .row_bytes()
                .checked_mul(tensor.info().dimensions()[1])
                .ok_or("Intel MTP source geometry overflow")?;
            layer_pages = layer_pages
                .checked_add(
                    crate::main_admission::page_bytes(bytes)?
                        .checked_mul(10)
                        .ok_or("Intel MTP source page count overflow")?,
                )
                .ok_or("Intel MTP source sum overflow")?;
        }
        source_pages = source_pages.max(layer_pages);
    }
    proof
        .admit(
            info.device,
            if integer { 828_932_096 } else { 828_866_560 },
            profile.public_bytes(),
            banks,
            source_pages,
            usage.bytes_reserved,
        )
        .map_err(Into::into)
}

/// Read only live, borrowed Vulkan device identities. Wgpu's local core IDs
/// cannot prove device independence across separately created Instances.
pub(crate) fn mtp_distinct_vulkan_devices(
    target: &wgpu::Device,
    draft: &wgpu::Device,
) -> Result<bool, &'static str> {
    // SAFETY: read borrowed identities only while both Wgpu/HAL guards remain
    // alive. No raw object is destroyed, retained, mutated or submitted through
    // these handles. Device ownership and submission remain with Wgpu.
    let target = unsafe { target.as_hal::<wgpu::hal::api::Vulkan>() };
    let draft = unsafe { draft.as_hal::<wgpu::hal::api::Vulkan>() };
    validate_mtp_vulkan_handles(
        target.as_ref().map(|device| device.raw_device().handle()),
        draft.as_ref().map(|device| device.raw_device().handle()),
    )
}
fn validate_mtp_vulkan_handles(
    target: Option<ash::vk::Device>,
    draft: Option<ash::vk::Device>,
) -> Result<bool, &'static str> {
    match (target, draft) {
        (Some(target), Some(draft))
            if target != ash::vk::Device::null()
                && draft != ash::vk::Device::null()
                && target != draft =>
        {
            Ok(true)
        }
        _ => Err("MTP requires two live distinct Vulkan logical devices"),
    }
}

pub(crate) fn mtp_device_receipt(
    selected: MtpDevice,
    target: Option<&wgpu::AdapterInfo>,
    draft: &wgpu::AdapterInfo,
    distinct_device: Option<bool>,
    distinct_runtime: bool,
    draft_runtime_id: u32,
) -> Result<serde_json::Value, &'static str> {
    let same = target.map(|target| {
        target.vendor == draft.vendor
            && target.device == draft.device
            && target.backend == draft.backend
            && target.device_pci_bus_id == draft.device_pci_bus_id
    });
    if draft.vendor != selected.vendor()
        || draft.device_type != wgpu::DeviceType::DiscreteGpu
        || draft.backend != wgpu::Backend::Vulkan
        || distinct_device == Some(false)
        || distinct_device.is_some() != target.is_some()
        || !distinct_runtime
        || draft_runtime_id > u16::MAX as u32
        || selected == MtpDevice::Intel && (same != Some(true) || target != Some(draft))
        || selected == MtpDevice::Nvidia && same == Some(true)
    {
        return Err("MTP actual physical device and independent runtime identity rejected");
    }
    Ok(
        serde_json::json!({"requested":selected.name(),"actual_vendor_id":draft.vendor,
        "actual_device_id":draft.device,"actual_pci_bus_id":draft.device_pci_bus_id,"same_physical_as_target":same,
        "actual_target_vendor_id":target.map(|info|info.vendor),
        "actual_target_device_id":target.map(|info|info.device),
        "actual_target_pci_bus_id":target.map(|info|&info.device_pci_bus_id),
        "distinct_wgpu_device":distinct_device,"distinct_runtime":distinct_runtime,
        "logical_device_identity":if distinct_device.is_some(){"borrowed-vulkan-device-handles"}else{"unverified-no-parent"},
        "actual_draft_runtime_id":draft_runtime_id,"draft_request_io_shared":false,
        "draft_private_arena_shared":false}),
    )
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

/// Original DOWN arithmetic has no dependency on the selected MTP profile or
/// GU math. Its compiled controller supplies the full original target scope.
pub(crate) fn frozen_down_stride_four(
    raw: Option<&OsStr>,
    original_four_row_scope: bool,
) -> Result<bool, &'static str> {
    match raw {
        None => Ok(false),
        Some(value) if value == "four-public" && original_four_row_scope => Ok(true),
        _ => Err("four-stride DOWN requires canonical original public frozen four-row scope"),
    }
}

pub(crate) fn frozen_down_stride_four_record(
    selected: bool,
    actual: bool,
    layers: u64,
    down_dispatches: u64,
    source_dispatches: u64,
) -> Result<serde_json::Value, &'static str> {
    if selected != actual
        || if selected {
            layers == 0
                || !layers.is_multiple_of(48)
                || layers.checked_mul(5) != Some(down_dispatches)
                || layers.checked_mul(10) != Some(source_dispatches)
        } else {
            layers != 0 || down_dispatches != 0
        }
    {
        return Err("four-stride DOWN actual all-layer joined dispatch receipt rejected");
    }
    Ok(
        serde_json::json!({"layers":layers,"down_dispatches":down_dispatches,
        "source_dispatches":source_dispatches}),
    )
}

pub(crate) fn frozen_down_stride_four_initial_record(
    selected: bool,
    actual: bool,
    layers: u64,
    down_dispatches: u64,
    source_dispatches: u64,
) -> Result<serde_json::Value, &'static str> {
    let record = frozen_down_stride_four_record(
        selected,
        actual,
        layers,
        down_dispatches,
        source_dispatches,
    )?;
    if selected && (layers != 48 || down_dispatches != 240 || source_dispatches != 480) {
        return Err("four-stride DOWN original initial count1 receipt rejected");
    }
    Ok(record)
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
pub(crate) fn validate_public_target_only_close(
    integer: bool,
    main_buffers: u64,
    main_bytes: u64,
) -> Result<(), &'static str> {
    let expected = if integer {
        (1381, 828_932_096)
    } else {
        (1380, 828_866_560)
    };
    if (main_buffers, main_bytes) != expected {
        return Err("target-only complete original target owner closing extent rejected");
    }
    Ok(())
}

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
    fn intel_mtp_selection_and_memory_transport_are_explicit_and_strict() {
        for scope in [false, true] {
            assert_eq!(MtpDevice::parse(None, scope), Ok(MtpDevice::Nvidia));
            for raw in [
                "intel-public",
                "intel",
                "nvidia",
                "auto",
                "",
                "intel-public ",
            ] {
                assert_eq!(
                    MtpDevice::parse(Some(OsStr::new(raw)), scope).is_ok(),
                    scope && raw == "intel-public"
                );
            }
        }
        let total = Some(OsStr::new("34242297856"));
        let used = Some(OsStr::new("50688000"));
        let device = Some(OsStr::new("4660"));
        assert_eq!(
            IntelMtpMemory::parse(MtpDevice::Intel, total, used, device).unwrap(),
            Some(IntelMtpMemory {
                total: 34242297856,
                used: 50688000,
                device: 4660
            })
        );
        assert_eq!(
            IntelMtpMemory::parse(MtpDevice::Nvidia, None, None, None),
            Ok(None)
        );
        for (total, used, device) in [
            (total, used, device),
            (total, None, None),
            (None, used, None),
            (None, None, device),
        ] {
            assert!(IntelMtpMemory::parse(MtpDevice::Nvidia, total, used, device).is_err());
        }
        for (total, used, device) in [
            (None, used, device),
            (total, None, device),
            (total, used, None),
            (Some(OsStr::new("0")), used, device),
            (total, Some(OsStr::new("0")), device),
            (total, total, device),
            (used, total, device),
            (total, used, Some(OsStr::new("0"))),
            (Some(OsStr::new("034242297856")), used, device),
            (total, Some(OsStr::new("+50688000")), device),
            (total, Some(OsStr::new("50688000 ")), device),
            (total, used, Some(OsStr::new("4294967296"))),
            (Some(OsStr::new("18446744073709551616")), used, device),
        ] {
            assert!(IntelMtpMemory::parse(MtpDevice::Intel, total, used, device).is_err());
        }
    }
    #[test]
    fn intel_mtp_combined_admission_charges_complete_full144_target_and_each_draft_owner() {
        let banks = crate::daslab_profile::bank_bytes_144().physical;
        let proof = IntelMtpMemory {
            total: 34_242_297_856,
            used: 50_688_000,
            device: 4660,
        };
        for (draft, expected) in [
            (crate::resident_mtp::PUBLIC_BYTES, 34_166_833_152),
            (crate::resident_mtp::STRATA_PUBLIC_BYTES, 31_694_422_016),
        ] {
            let got = proof
                .admit(4660, 828_932_096, draft, banks, 30 * 1024 * 1024, 0)
                .unwrap();
            assert_eq!(got["combined_reserved_bytes"], expected);
            assert_eq!(
                got["remaining_beyond_margin_bytes"],
                proof.total - proof.used - expected
            );
            assert_eq!(got["shared_driver_margin_bytes"], 512u64 * 1024 * 1024);
            let exact = IntelMtpMemory {
                total: expected + proof.used,
                ..proof
            };
            assert!(
                exact
                    .admit(4660, 828_932_096, draft, banks, 30 * 1024 * 1024, 0)
                    .is_ok()
            );
            let short = IntelMtpMemory {
                total: exact.total - 1,
                ..proof
            };
            assert!(
                short
                    .admit(4660, 828_932_096, draft, banks, 30 * 1024 * 1024, 0)
                    .is_err()
            );
        }
        assert_eq!(
            proof
                .admit(
                    4660,
                    828_866_560,
                    crate::resident_mtp::PUBLIC_BYTES,
                    banks,
                    30 * 1024 * 1024,
                    0
                )
                .unwrap()["combined_reserved_bytes"],
            34_166_767_616u64
        );
        for (device, main, draft, bank, source, baseline) in [
            (
                4661,
                828_932_096,
                crate::resident_mtp::PUBLIC_BYTES,
                banks,
                30 * 1024 * 1024,
                0,
            ),
            (
                4660,
                828_932_095,
                crate::resident_mtp::PUBLIC_BYTES,
                banks,
                30 * 1024 * 1024,
                0,
            ),
            (
                4660,
                828_932_096,
                crate::resident_mtp::PUBLIC_BYTES - 1,
                banks,
                30 * 1024 * 1024,
                0,
            ),
            (
                4660,
                828_932_096,
                crate::resident_mtp::PUBLIC_BYTES,
                banks - 1,
                30 * 1024 * 1024,
                0,
            ),
            (
                4660,
                828_932_096,
                crate::resident_mtp::PUBLIC_BYTES,
                banks,
                30 * 1024 * 1024 - 1,
                0,
            ),
            (
                4660,
                828_932_096,
                crate::resident_mtp::PUBLIC_BYTES,
                banks,
                30 * 1024 * 1024,
                32768,
            ),
        ] {
            assert!(
                proof
                    .admit(device, main, draft, bank, source, baseline)
                    .is_err()
            );
        }
        for used in [0, proof.total, u64::MAX] {
            assert!(
                IntelMtpMemory { used, ..proof }
                    .admit(
                        4660,
                        828_932_096,
                        crate::resident_mtp::PUBLIC_BYTES,
                        banks,
                        30 * 1024 * 1024,
                        0
                    )
                    .is_err()
            );
        }
    }
    fn mtp_adapter(vendor: u32, device: u32) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "compiled public adapter fixture".into(),
            vendor,
            device,
            device_type: wgpu::DeviceType::DiscreteGpu,
            device_pci_bus_id: "0000:01:00.0".into(),
            driver: "public fixture".into(),
            driver_info: "public fixture".into(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 16,
            subgroup_max_size: 32,
            transient_saves_memory: false,
        }
    }
    #[test]
    fn intel_mtp_vulkan_identity_requires_two_present_nonnull_distinct_live_handles() {
        use ash::vk::Handle;
        let first = ash::vk::Device::from_raw(1);
        let second = ash::vk::Device::from_raw(2);
        assert_eq!(
            validate_mtp_vulkan_handles(Some(first), Some(second)),
            Ok(true)
        );
        assert_eq!(
            validate_mtp_vulkan_handles(Some(second), Some(first)),
            Ok(true)
        );
        for (target, draft) in [
            (None, None),
            (None, Some(second)),
            (Some(first), None),
            (Some(ash::vk::Device::null()), Some(second)),
            (Some(first), Some(ash::vk::Device::null())),
            (Some(ash::vk::Device::null()), Some(ash::vk::Device::null())),
            (Some(first), Some(first)),
        ] {
            assert!(validate_mtp_vulkan_handles(target, draft).is_err());
        }
    }
    #[test]
    fn intel_mtp_actual_device_receipt_requires_distinct_runtime_on_same_verified_physical_gpu() {
        let target = mtp_adapter(0x8086, 4660);
        let nvidia = mtp_adapter(0x10de, 123);
        let intel = target.clone();
        let actual =
            mtp_device_receipt(MtpDevice::Intel, Some(&target), &intel, Some(true), true, 1)
                .unwrap();
        assert_eq!(actual["same_physical_as_target"], true);
        assert_eq!(actual["actual_target_device_id"], 4660);
        let unknown = mtp_device_receipt(MtpDevice::Nvidia, None, &nvidia, None, true, 1).unwrap();
        assert!(unknown["same_physical_as_target"].is_null());
        assert!(unknown["actual_target_device_id"].is_null());
        assert!(unknown["distinct_wgpu_device"].is_null());
        assert_eq!(unknown["logical_device_identity"], "unverified-no-parent");
        assert_eq!(
            actual["logical_device_identity"],
            "borrowed-vulkan-device-handles"
        );
        assert!(mtp_device_receipt(MtpDevice::Nvidia, None, &nvidia, Some(true), true, 1).is_err());
        assert!(
            mtp_device_receipt(MtpDevice::Intel, Some(&target), &intel, Some(true), true, 1)
                .is_ok()
        );
        assert!(
            mtp_device_receipt(
                MtpDevice::Nvidia,
                Some(&target),
                &nvidia,
                Some(true),
                true,
                1
            )
            .is_ok()
        );
        for (device, actual) in [(MtpDevice::Intel, &intel), (MtpDevice::Nvidia, &nvidia)] {
            for (distinct_device, distinct_runtime, id) in [
                (Some(false), true, 1),
                (None, true, 1),
                (Some(true), false, 1),
                (Some(true), true, 65536),
            ] {
                assert!(
                    mtp_device_receipt(
                        device,
                        Some(&target),
                        actual,
                        distinct_device,
                        distinct_runtime,
                        id
                    )
                    .is_err()
                );
            }
        }
        assert!(mtp_device_receipt(MtpDevice::Intel, None, &intel, Some(true), true, 1).is_err());
        assert!(
            mtp_device_receipt(
                MtpDevice::Intel,
                Some(&target),
                &nvidia,
                Some(true),
                true,
                1
            )
            .is_err()
        );
        assert!(
            mtp_device_receipt(
                MtpDevice::Nvidia,
                Some(&target),
                &intel,
                Some(true),
                true,
                1
            )
            .is_err()
        );
        let mut changed = intel.clone();
        changed.device_pci_bus_id = "0000:02:00.0".into();
        assert!(
            mtp_device_receipt(
                MtpDevice::Intel,
                Some(&target),
                &changed,
                Some(true),
                true,
                1
            )
            .is_err()
        );
        changed = intel.clone();
        changed.device += 1;
        assert!(
            mtp_device_receipt(
                MtpDevice::Intel,
                Some(&target),
                &changed,
                Some(true),
                true,
                1
            )
            .is_err()
        );
        changed = intel;
        changed.device_type = wgpu::DeviceType::IntegratedGpu;
        assert!(
            mtp_device_receipt(
                MtpDevice::Intel,
                Some(&target),
                &changed,
                Some(true),
                true,
                1
            )
            .is_err()
        );
    }
    #[test]
    fn target_only_close_requires_exact_original_complete_owner_extent() {
        for (integer, buffers, bytes) in [(false, 1380, 828_866_560), (true, 1381, 828_932_096)] {
            assert!(validate_public_target_only_close(integer, buffers, bytes).is_ok());
            for (buffers, bytes) in [
                (buffers - 1, bytes),
                (buffers + 1, bytes),
                (buffers, bytes - 1),
                (buffers, bytes + 1),
            ] {
                assert!(validate_public_target_only_close(integer, buffers, bytes).is_err());
            }
            assert!(validate_public_target_only_close(!integer, buffers, bytes).is_err());
        }
    }
    #[test]
    fn frozen_down_stride_four_controller_flag_is_canonical_and_default_off() {
        for scope in [false, true] {
            assert_eq!(frozen_down_stride_four(None, scope), Ok(false));
            for raw in ["four-public", "four", "true", "", "four-public "] {
                assert_eq!(
                    frozen_down_stride_four(Some(OsStr::new(raw)), scope).is_ok(),
                    scope && raw == "four-public"
                );
            }
        }
    }
    #[test]
    fn frozen_down_stride_four_controller_receipt_requires_complete_actual_layers() {
        assert!(frozen_down_stride_four_record(false, false, 0, 0, 0).is_ok());
        assert!(frozen_down_stride_four_record(false, false, 0, 0, 480).is_ok());
        for layers in [48, 96, 144, 48 * 200] {
            let record =
                frozen_down_stride_four_record(true, true, layers, layers * 5, layers * 10)
                    .unwrap();
            assert_eq!(record["layers"], layers);
            assert_eq!(record["down_dispatches"], layers * 5);
            assert_eq!(record["source_dispatches"], layers * 10);
        }
        for (selected, actual, layers, down, source) in [
            (true, false, 48, 240, 480),
            (false, true, 48, 240, 480),
            (true, true, 0, 0, 0),
            (true, true, 1, 5, 10),
            (true, true, 47, 235, 470),
            (true, true, 48, 239, 480),
            (true, true, 48, 241, 480),
            (true, true, 48, 240, 479),
            (true, true, 48, 240, 481),
            (true, true, u64::MAX - 15, u64::MAX, u64::MAX),
            (false, false, 1, 0, 480),
            (false, false, 0, 5, 480),
        ] {
            assert!(
                frozen_down_stride_four_record(selected, actual, layers, down, source).is_err()
            );
        }
    }
    #[test]
    fn frozen_down_stride_four_controller_initial_count1_requires_exact_first_batch() {
        assert!(frozen_down_stride_four_initial_record(false, false, 0, 0, 0).is_ok());
        assert!(frozen_down_stride_four_initial_record(true, true, 48, 240, 480).is_ok());
        for (actual, layers, down, source) in [
            (false, 48, 240, 480),
            (true, 0, 0, 0),
            (true, 1, 5, 10),
            (true, 96, 480, 960),
            (true, 48, 240, 479),
            (true, 48, 239, 480),
        ] {
            assert!(
                frozen_down_stride_four_initial_record(true, actual, layers, down, source).is_err()
            );
        }
    }
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
