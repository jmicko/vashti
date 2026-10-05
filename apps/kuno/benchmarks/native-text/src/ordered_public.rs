//! Optional fixed-public expert source uploads, on the configured resident stream.
//!
//! A full main page is retained BEFORE a moved write can fail or unwind. It is
//! pending ordered input, not initialized/accepted data. Release requires the
//! NEXT completed router (including cache-hit layers) or final token read,
//! dropped read leases, then checked runtime sanitation. No polling/cleanup in
//! the upload path and no ordinary public-write fallback. Main admission charges
//! these pages separately from the unchanged 64 MiB runtime staging cap.
use crate::main_admission::page_bytes;
use kuno_text::{Error, Result};

pub(crate) const SOURCE_LIMIT: u64 = 40 * 1024 * 1024;
pub(crate) const WINDOW_INFO: u64 = 512 * 2 * 512;
pub(crate) const WINDOW_PRIVATE_IO: u64 = 2 * 1024 * 1024;
pub(crate) const WINDOW_RUNTIME: u64 = SOURCE_LIMIT + WINDOW_INFO + WINDOW_PRIVATE_IO;

pub(crate) fn configured(raw: Option<&str>, transient: bool, batch: bool) -> Result<bool> {
    match raw {
        None | Some("0") => Ok(false),
        Some("1") if transient && !batch => Ok(true),
        Some("1") => Err(Error::InvalidFormat(
            "ordered public uploads require owned resident I/O and legacy expert fill",
        )),
        Some(_) => Err(Error::InvalidFormat(
            "KUNO_LAB_ORDERED_PUBLIC_UPLOAD must be 0 or 1",
        )),
    }
}

/// Reject another packing/geometry rather than silently expanding the proof.
pub(crate) fn validate_parts(parts: [u64; 3]) -> Result<()> {
    if parts[0] != 921_600 || parts[1] != 921_600 || ![1_126_400, 1_740_800].contains(&parts[2]) {
        return Err(Error::InvalidFormat("ordered public expert packing bound"));
    }
    let sum = parts.into_iter().try_fold(0_u64, |sum, n| {
        sum.checked_add(page_bytes(n).map_err(|_| Error::InvalidBufferLength)?)
            .ok_or(Error::ArithmeticOverflow)
    })?;
    if sum.checked_mul(10) != Some(SOURCE_LIMIT) {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}

pub(crate) fn validate_geometry(g: kuno_text::model::GraphGeometry) -> Result<()> {
    if g.layers != 48
        || g.residual.streams != 4
        || g.residual.width != 2560
        || g.low_rank != 320
        || g.vocabulary != 248_320
        || g.experts != 512
        || g.experts_used != 10
        || g.expert_hidden != 640
        || g.max_context > 4096
        || g.gdn.head_dim != 128
        || g.gdn.key_heads != 16
        || g.gdn.value_heads != 48
        || g.gdn_conv_channels != 10_240
        || g.gdn_conv_kernel != 4
        || g.qsa.query_heads != 24
        || g.qsa.kv_heads != 2
        || g.qsa.head_dim != 256
        || g.qsa.indexer_heads != 4
        || g.qsa.indexer_dim != 128
        || g.qsa.token_budget > 2048
        || g.qsa.block_size != 4
        || g.ple.heads != 16
        || g.ple.head_width != 160
        || g.ple.conv_channels != 10_240
    {
        return Err(Error::InvalidFormat("ordered public fixed geometry bound"));
    }
    Ok(())
}

/// Generic owner policy is exercised by host fault tests; production H=Handle.
pub(crate) struct PendingSources<H> {
    owners: Vec<H>,
    bytes: u64,
    peak: u64,
}
impl<H: Clone> PendingSources<H> {
    pub(crate) fn new() -> Self {
        Self {
            owners: Vec::new(),
            bytes: 0,
            peak: 0,
        }
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    pub(crate) fn peak(&self) -> u64 {
        self.peak
    }
    pub(crate) fn enqueue(
        &mut self,
        data: Vec<u8>,
        allocate: impl FnOnce(usize) -> Result<H>,
        write: impl FnOnce(H, Vec<u8>) -> Result<()>,
    ) -> Result<H> {
        let span = u64::try_from(data.len()).map_err(|_| Error::ArithmeticOverflow)?;
        if page_bytes(span).map_err(|_| Error::InvalidBufferLength)? != span {
            return Err(Error::InvalidBufferLength);
        }
        let next = self
            .bytes
            .checked_add(span)
            .ok_or(Error::ArithmeticOverflow)?;
        if next > SOURCE_LIMIT || self.owners.len() >= 30 {
            return Err(Error::LimitExceeded("ordered public source pages"));
        }
        // Reserve before main allocation. No fallible host operation can lose
        // the only destination owner between allocation and retention.
        self.owners
            .try_reserve(1)
            .map_err(|_| Error::LimitExceeded("ordered public source owners"))?;
        let owner = allocate(data.len())?;
        self.owners.push(owner.clone());
        self.bytes = next;
        self.peak = self.peak.max(next);
        write(owner.clone(), data)?;
        Ok(owner)
    }
    /// Caller already has an ordinary completed read fence. Cleanup failure
    /// preserves every owner. This method never samples/grants MAIN credit.
    pub(crate) fn after_completed_read(
        &mut self,
        sanitize: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        sanitize()?;
        self.owners.clear();
        self.bytes = 0;
        Ok(())
    }
}
impl<H> Drop for PendingSources<H> {
    fn drop(&mut self) {
        if !self.owners.is_empty() {
            // Failed explicit teardown has no accepted completion. Keep the
            // bounded public owners alive until process recovery; dropping a
            // logical owner is not a fence or permission to recycle pages.
            std::mem::forget(std::mem::take(&mut self.owners));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;
    fn failure<T>() -> Result<T> {
        Err(Error::InvalidFormat("public fault fixture"))
    }
    #[test]
    fn explicit_secure_legacy_policy_only() {
        for transient in [false, true] {
            for batch in [false, true] {
                assert!(!configured(None, transient, batch).unwrap());
                assert!(!configured(Some("0"), transient, batch).unwrap());
                assert_eq!(
                    configured(Some("1"), transient, batch).is_ok(),
                    transient && !batch
                );
                assert!(configured(Some("2"), transient, batch).is_err());
            }
        }
    }
    #[test]
    fn full_three_by_ten_pages_fit_unchanged_runtime_cap() {
        for down in [1_126_400, 1_740_800] {
            validate_parts([921_600, 921_600, down]).unwrap();
        }
        assert_eq!(WINDOW_RUNTIME, 44_564_480);
        const { assert!(WINDOW_RUNTIME < crate::executor::RESIDENT_RUNTIME_IO_MAX) };
        assert!(validate_parts([921_600, 921_600, 2_097_153]).is_err());
        let mut p = PendingSources::new();
        for _ in 0..10 {
            for n in [1 << 20, 1 << 20, 2 << 20] {
                p.enqueue(vec![0; n], Ok, |_, _| Ok(())).unwrap();
            }
        }
        assert_eq!(p.bytes(), SOURCE_LIMIT);
        let allocated = Cell::new(false);
        assert!(
            p.enqueue(
                vec![0; 32_768],
                |_| {
                    allocated.set(true);
                    Ok(0)
                },
                |_, _| Ok(())
            )
            .is_err()
        );
        assert!(!allocated.get());
        p.after_completed_read(|| Ok(())).unwrap();
        assert_eq!(p.bytes(), 0);
        assert_eq!(p.peak(), SOURCE_LIMIT);
    }
    #[test]
    fn moved_padding_and_owner_survive_failed_write_and_unwind() {
        let alive = Rc::new(Cell::new(0));
        #[derive(Clone)]
        struct Owner(Rc<Cell<usize>>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        for panic in [false, true] {
            let mut p = PendingSources::new();
            let mut data = vec![0; 32_768];
            data[..4].copy_from_slice(&[1, 2, 3, 4]);
            let ptr = data.as_ptr();
            let drops = alive.get();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                p.enqueue(
                    data,
                    |_| Ok(Owner(alive.clone())),
                    |_, moved| {
                        assert_eq!(moved.as_ptr(), ptr);
                        assert_eq!(&moved[..4], &[1, 2, 3, 4]);
                        assert!(moved[4..].iter().all(|&x| x == 0));
                        if panic {
                            panic!("public upload unwind");
                        }
                        failure()
                    },
                )
            }));
            assert!(result.is_err() || result.unwrap().is_err());
            assert_eq!(p.bytes(), 32_768);
            assert_eq!(p.owners.len(), 1);
            assert!(alive.get() > drops); // Temporary clones drop, retained owner remains.
            let retained_drops = alive.get();
            assert!(p.after_completed_read(failure).is_err());
            assert_eq!(p.bytes(), 32_768);
            assert_eq!(alive.get(), retained_drops);
            p.after_completed_read(|| Ok(())).unwrap();
            assert!(alive.get() > retained_drops);
        }
    }
    #[test]
    fn lease_completion_cleanup_and_release_order() {
        let mut p = PendingSources::new();
        let events = Rc::new(RefCell::new(Vec::new()));
        #[derive(Clone)]
        struct Owner(Rc<RefCell<Vec<&'static str>>>);
        impl Drop for Owner {
            fn drop(&mut self) {
                self.0.borrow_mut().push("owner release");
            }
        }
        let _returned = p
            .enqueue(
                vec![0; 32_768],
                |_| Ok(Owner(events.clone())),
                |_, _| Ok(()),
            )
            .unwrap();
        events.borrow_mut().clear();
        events.borrow_mut().push("completed read and lease drop");
        p.after_completed_read(|| {
            events.borrow_mut().push("checked sanitize");
            Ok(())
        })
        .unwrap();
        events.borrow_mut().push("main telemetry and credit");
        assert_eq!(
            *events.borrow(),
            [
                "completed read and lease drop",
                "checked sanitize",
                "owner release",
                "main telemetry and credit"
            ]
        );
    }
    #[test]
    fn malformed_page_never_allocates() {
        let mut p: PendingSources<usize> = PendingSources::new();
        for n in [0, 1, 32_769] {
            assert!(
                p.enqueue(
                    vec![0; n],
                    |_| panic!("invalid page allocated"),
                    |_, _| Ok(())
                )
                .is_err()
            );
        }
        assert_eq!(p.bytes(), 0);
    }
    #[test]
    fn generated_copy_metadata_bound_matches_actual_packer() {
        for address in [cubecl::ir::AddressType::U32, cubecl::ir::AddressType::U64] {
            let mut info = cubecl::InfoBuilder::default();
            info.metadata.register_array(524_288, 281_600, address);
            info.metadata
                .register_array(28_160_000, 28_160_000, address);
            info.scalars.push(123_u64);
            let packed = info.finish(address);
            assert!(packed.data.len() * 8 <= 40);
            assert_eq!(
                30 * 2 * (packed.data.len() * 8).max(256).next_power_of_two(),
                15_360
            );
        }
        // Conservative resident raw-Array argument inventory: no tensor
        // stride/shape metadata. Worst address width and padded u64 scalars.
        let address = cubecl::ir::AddressType::U64;
        let mut info = cubecl::InfoBuilder::default();
        for _ in 0..8 {
            info.metadata.register_array(1_048_576, 1_048_576, address);
        }
        for _ in 0..20 {
            info.scalars.push(1_u64);
        }
        assert!(info.finish(address).data.len() * 8 <= 512);
    }
}
