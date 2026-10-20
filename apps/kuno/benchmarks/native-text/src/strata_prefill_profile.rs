//! Fenced coarse timing for compiled public requests only. The private worker
//! never enables this switch; no request values or contents enter the receipt.
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    time::Instant,
};
static ENABLED: AtomicBool = AtomicBool::new(false);
static REQUEST: AtomicU64 = AtomicU64::new(0);
thread_local! { static TOTALS: RefCell<BTreeMap<&'static str, (u64,u64,u64)>> = const { RefCell::new(BTreeMap::new()) }; }
pub(crate) fn enable_compiled_public() {
    ENABLED.store(true, Ordering::Release);
}
pub(crate) fn begin() {
    if ENABLED.load(Ordering::Acquire) {
        REQUEST.fetch_add(1, Ordering::Relaxed);
    }
    TOTALS.with(|v| v.borrow_mut().clear());
}

// PUBLIC_PREFILL_NATIVE_SAMPLE_BEGIN
const NATIVE_CLASSES: [&str; 7] = [
    "dequant",
    "matrix",
    "gather",
    "activate",
    "ordered_copy",
    "ordered_fill",
    "other",
];
#[derive(Clone, Copy)]
struct NativeSelection {
    authorized: bool,
    request: u64,
    start: usize,
    count: usize,
    layer: usize,
}
impl NativeSelection {
    fn selects(self) -> bool {
        self.authorized
            && self.request == 3
            && self.start == 0
            && self.count == 4096
            && self.layer == 24
    }
}
fn native_class(name: &str) -> usize {
    match name {
        "StrataPromptExpertF16Image" => 0,
        "StrataPromptExpertGemm" => 1,
        "StrataPromptExpertGather" => 2,
        "StrataPromptExpertF16Swiglu" => 3,
        "OrderedCopy" => 4,
        "OrderedFill" => 5,
        _ if name.contains("::expert::Expert")
            || name.contains("::expert_macro::ExpertMacro")
            || name.contains("::expert_narrow::ExpertNarrow") =>
        {
            1
        }
        _ => 6,
    }
}
#[cfg(test)]
mod native_sample_tests {
    use super::*;
    #[test]
    fn exactly_one_compiled_public_expert_interval_can_be_selected() {
        for authorized in [false, true] {
            for request in 0..5 {
                for start in [0, 4096] {
                    for count in [35, 149, 4095, 4096] {
                        for layer in 0..48 {
                            assert_eq!(
                                NativeSelection {
                                    authorized,
                                    request,
                                    start,
                                    count,
                                    layer
                                }
                                .selects(),
                                authorized
                                    && request == 3
                                    && start == 0
                                    && count == 4096
                                    && layer == 24
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn reports_have_static_classes_including_real_generated_matrix_names() {
        for (name, class) in [
            ("StrataPromptExpertF16Image", 0),
            ("StrataPromptExpertGemm", 1),
            ("kuno_native_text::strata_matrix::native::expert::Expert", 1),
            (
                "kuno_native_text::strata_matrix::native::expert_macro::ExpertMacro",
                1,
            ),
            (
                "kuno_native_text::strata_matrix::native::expert_narrow::ExpertNarrow",
                1,
            ),
            ("StrataPromptExpertGather", 2),
            ("StrataPromptExpertF16Swiglu", 3),
            ("OrderedCopy", 4),
            ("OrderedFill", 5),
            ("UnlabelledNativeOperator", 6),
        ] {
            assert_eq!(native_class(name), class);
            assert_eq!(
                NATIVE_CLASSES[class]
                    .as_bytes()
                    .iter()
                    .filter(|&&b| b == b':')
                    .count(),
                0
            );
        }
    }
}
// PUBLIC_PREFILL_NATIVE_SAMPLE_END

fn native_totals() -> [(u64, u64); NATIVE_CLASSES.len()] {
    let mut classes = [(0u64, 0u64); NATIVE_CLASSES.len()];
    for (name, calls, ns) in cubecl::wgpu::native_strata_operator_profile() {
        let total = &mut classes[native_class(&name)];
        total.0 = total.0.saturating_add(calls);
        total.1 = total.1.saturating_add(ns);
    }
    classes
}
/// One fixed-public expert interval only. Keep this guard alive through the
/// existing coarse Timer's real completion fence, not just enqueue_prompt().
/// Drop on error disables sampling without publishing incomplete timestamps.
pub(crate) struct NativeExpertLayer(Option<[(u64, u64); NATIVE_CLASSES.len()]>);
impl NativeExpertLayer {
    pub(crate) fn new(start: usize, count: usize, layer: usize) -> Self {
        if !ENABLED.load(Ordering::Acquire) {
            return Self(None);
        }
        let selected = NativeSelection {
            authorized: std::env::var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY").as_deref() == Ok("1")
                && std::env::var("KUNO_STRATA_NATIVE_PROFILE").as_deref() == Ok("prefill-layer"),
            request: REQUEST.load(Ordering::Relaxed),
            start,
            count,
            layer,
        }
        .selects();
        if !selected {
            return Self(None);
        }
        let before = native_totals();
        cubecl::wgpu::set_native_strata_profile_active(true);
        Self(Some(before))
    }
    pub(crate) fn finish(mut self) {
        if let Some(before) = self.0.take() {
            cubecl::wgpu::set_native_strata_profile_active(false);
            let after = native_totals();
            let classes: BTreeMap<_, _> = NATIVE_CLASSES
                .into_iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        name,
                        [
                            after[i].0.saturating_sub(before[i].0),
                            after[i].1.saturating_sub(before[i].1),
                        ],
                    )
                })
                .collect();
            let timestamp_calls = classes.values().map(|v| v[0]).sum::<u64>();
            // Counts/ns only, never source labels, experts, routes or tokens.
            // Zero timestamp_calls honestly means the existing graph threshold
            // did not sample this interval; no cap or fence is relaxed.
            eprintln!(
                "public_prefill_native_gpu_profile {}",
                serde_json::json!({"sampled_layers":1,"request_index":3,"start":0,"count":4096,"layer":24,"timestamp_calls":timestamp_calls,"classes":classes})
            );
        }
    }
}
impl Drop for NativeExpertLayer {
    fn drop(&mut self) {
        if self.0.is_some() {
            cubecl::wgpu::set_native_strata_profile_active(false);
        }
    }
}
pub(crate) fn report() {
    if ENABLED.load(Ordering::Acquire) {
        TOTALS.with(|v| {
            eprintln!(
                "public_prefill_profile {}",
                serde_json::to_string(&*v.borrow()).unwrap()
            )
        });
    }
}
/// Aggregate public-fixture resource counts only. Private inference never
/// enables this collector; source identities, offsets and route values are
/// deliberately absent. Counts use tuple slot0, with both time slots zero.
pub(crate) fn count(name: &'static str, amount: u64) {
    if ENABLED.load(Ordering::Acquire) {
        TOTALS.with(|totals| {
            let mut totals = totals.borrow_mut();
            let total = totals.entry(name).or_default();
            total.0 = total.0.saturating_add(amount);
        });
    }
}

/// Public-fixture aggregate geometry only, after the caller validates the map.
/// Never retain expert identities or enable this collector for private calls.
pub(crate) fn expert_member_geometry(map: &[u32], groups: usize) {
    if !ENABLED.load(Ordering::Acquire) {
        return;
    }
    let names = [
        ("expert_groups_rows_1_7", "expert_members_rows_1_7"),
        ("expert_groups_rows_8_15", "expert_members_rows_8_15"),
        ("expert_groups_rows_16_31", "expert_members_rows_16_31"),
        ("expert_groups_rows_32_63", "expert_members_rows_32_63"),
        ("expert_groups_rows_64_127", "expert_members_rows_64_127"),
        ("expert_groups_rows_128_255", "expert_members_rows_128_255"),
        (
            "expert_groups_rows_256_plus",
            "expert_members_rows_256_plus",
        ),
    ];
    let mut group_counts = [0u64; 7];
    let mut member_counts = [0u64; 7];
    let mut padded_members = 0u64;
    for &members in &map[516..516 + groups] {
        let bin = match members {
            0..=7 => 0,
            8..=15 => 1,
            16..=31 => 2,
            32..=63 => 3,
            64..=127 => 4,
            128..=255 => 5,
            _ => 6,
        };
        group_counts[bin] += 1;
        member_counts[bin] += u64::from(members);
        let tile = match members {
            0..=8 => 8,
            9..=31 => 16,
            _ => 32,
        };
        padded_members += u64::from(members.div_ceil(tile) * tile);
    }
    for (i, (group_name, member_name)) in names.into_iter().enumerate() {
        count(group_name, group_counts[i]);
        count(member_name, member_counts[i]);
    }
    // Actual fixed expert geometry: GU (2560,1280), DOWN (640,2560).
    // This is an arithmetic-work estimate, never a measured hardware FLOP rate.
    count(
        "expert_useful_matrix_flops",
        member_counts.iter().sum::<u64>() * 9_830_400,
    );
    count(
        "expert_tile_padded_matrix_flops",
        padded_members * 9_830_400,
    );
}
fn native_cpu_counts() -> [u64; 2] {
    let (_, _, capture, wait) = cubecl::wgpu::native_strata_graph_execution_stats();
    [capture, wait]
}
pub(crate) struct Timer(Option<(ComputeClient<WgpuRuntime>, Instant, [u64; 2])>);
impl Timer {
    pub(crate) fn new(client: &ComputeClient<WgpuRuntime>) -> kuno_text::Result<Self> {
        if !ENABLED.load(Ordering::Acquire) {
            return Ok(Self(None));
        }
        cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
        Ok(Self(Some((
            client.clone(),
            Instant::now(),
            native_cpu_counts(),
        ))))
    }
    pub(crate) fn mark(&mut self, name: &'static str) -> kuno_text::Result<()> {
        if let Some((client, start, native_before)) = &mut self.0 {
            let enqueue_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            let native_after = native_cpu_counts();
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(ns);
                t.2 = t.2.saturating_add(enqueue_ns);
                if name == "experts" {
                    // The enqueue slice ends BEFORE sync records native Vulkan
                    // descriptors/commands. Report that CPU cost separately;
                    // it must not be mistaken for GPU kernel time. The public
                    // protected unit is the sole process/device workload.
                    for (index, name) in ["expert_native_capture_cpu", "expert_native_wait_cpu"]
                        .into_iter()
                        .enumerate()
                    {
                        let delta = native_after[index].saturating_sub(native_before[index]);
                        let total = v.entry(name).or_default();
                        total.0 += 1;
                        total.1 = total.1.saturating_add(delta);
                        total.2 = total.2.saturating_add(delta);
                    }
                }
            });
            *native_before = native_after;
            *start = Instant::now();
        }
        Ok(())
    }
}

/// CPU-side scheduling slices; no additional device fence or changed overlap.
pub(crate) struct CpuTimer(Option<Instant>);
impl CpuTimer {
    pub(crate) fn new() -> Self {
        Self(ENABLED.load(Ordering::Acquire).then(Instant::now))
    }
    pub(crate) fn mark(&mut self, name: &'static str) {
        if let Some(start) = &mut self.0 {
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(ns);
                t.2 = t.2.saturating_add(ns);
            });
            *start = Instant::now();
        }
    }
}
