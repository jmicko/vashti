//! Fenced coarse timing for compiled public requests only. The private worker
//! never enables this switch; no request values or contents enter the receipt.
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};
static ENABLED: AtomicBool = AtomicBool::new(false);
thread_local! { static TOTALS: RefCell<BTreeMap<&'static str, (u64,u64,u64)>> = const { RefCell::new(BTreeMap::new()) }; }
pub(crate) fn enable_compiled_public() {
    ENABLED.store(true, Ordering::Release);
}
pub(crate) fn begin() {
    TOTALS.with(|v| v.borrow_mut().clear());
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
        let tile = if members < 32 { 8 } else { 32 };
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
pub(crate) struct Timer(Option<(ComputeClient<WgpuRuntime>, Instant)>);
impl Timer {
    pub(crate) fn new(client: &ComputeClient<WgpuRuntime>) -> kuno_text::Result<Self> {
        if !ENABLED.load(Ordering::Acquire) {
            return Ok(Self(None));
        }
        cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
        Ok(Self(Some((client.clone(), Instant::now()))))
    }
    pub(crate) fn mark(&mut self, name: &'static str) -> kuno_text::Result<()> {
        if let Some((client, start)) = &mut self.0 {
            let enqueue_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            cubecl::future::block_on(client.sync()).map_err(crate::resident_util::fault)?;
            let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            TOTALS.with(|v| {
                let mut v = v.borrow_mut();
                let t = v.entry(name).or_default();
                t.0 += 1;
                t.1 = t.1.saturating_add(ns);
                t.2 = t.2.saturating_add(enqueue_ns);
            });
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
