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
