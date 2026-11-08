// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
//! Serialized same-thread guard for transient INFO TLS. KernelLauncher owns
//! this lease, so cancellation/panic/abandonment wipe the retained builder.
use crate::InfoBuilder;
use core::cell::RefCell;
std::thread_local! {
    pub(super) static INFO:RefCell<InfoBuilder>=RefCell::new(InfoBuilder::default());
    static INFO_BUSY:core::cell::Cell<bool>=const {core::cell::Cell::new(false)};
}
pub(super) struct InfoLease {
    _same_thread: core::marker::PhantomData<alloc::rc::Rc<()>>,
}
impl InfoLease {
    pub(super) fn acquire() -> Self {
        assert!(
            !INFO_BUSY.with(|busy| busy.replace(true)),
            "nested TLS launchers unsupported"
        );
        // Establish ownership BEFORE clearing/borrowing TLS so a panic also
        // releases the busy marker through this lease's destructor.
        let lease = Self {
            _same_thread: core::marker::PhantomData,
        };
        INFO.with_borrow_mut(|info| info.clear_private());
        lease
    }
}
impl Drop for InfoLease {
    fn drop(&mut self) {
        // A with_info borrow unwinds before its owning launcher; no active
        // RefCell borrow remains here on normal/cancel/panic paths. try_with
        // handles TLS destruction, where InfoBuilder's members already wipe.
        let wiped = INFO
            .try_with(|info| match info.try_borrow_mut() {
                Ok(mut info) => {
                    info.clear_private();
                    true
                }
                // Impossible for an ordinary owning launcher's unwind: retain the
                // busy marker on violated ownership instead of double-panicking or
                // permitting a new launcher to reuse uncleaned private TLS.
                Err(_) => false,
            })
            .unwrap_or(true); // TLS destruction already drops/wipes members.
        if wiped {
            let _ = INFO_BUSY.try_with(|busy| busy.set(false));
        }
    }
}
#[cfg(all(test, feature = "std"))]
mod private_tests {
    use super::*;
    use cubecl_ir::AddressType;
    #[test]
    fn violated_tls_ownership_rejects_reuse_without_double_panic() {
        // Separate thread: the deliberately violated TLS owner is retired until
        // thread teardown, whose InfoBuilder members independently wipe.
        std::thread::spawn(|| {
            INFO.with(|info| {
                let _borrow = info.borrow();
                assert!(std::panic::catch_unwind(InfoLease::acquire).is_err());
            });
            assert!(INFO_BUSY.with(|busy| busy.get()));
            assert!(std::panic::catch_unwind(InfoLease::acquire).is_err());
        })
        .join()
        .unwrap();
    }
    #[test]
    fn tls_lease_wipes_abandoned_builder_and_releases_owner_during_panic() {
        {
            let _lease = InfoLease::acquire();
            INFO.with_borrow_mut(|i| {
                i.scalars.push(0x23456789u32);
                i.metadata.register_array(777, 512, AddressType::U32);
            });
        }
        assert!(!INFO_BUSY.with(|c| c.get()));
        INFO.with_borrow(|i| {
            assert_eq!(i.scalars.len_aligned(), 0);
            assert_eq!(i.metadata.static_len(AddressType::U32), 0);
        });
        assert!(
            std::panic::catch_unwind(|| {
                let _lease = InfoLease::acquire();
                INFO.with_borrow_mut(|i| {
                    i.scalars.push(0x76543210u32);
                    panic!("synthetic registration panic");
                });
            })
            .is_err()
        );
        assert!(!INFO_BUSY.with(|c| c.get()));
        INFO.with_borrow(|i| assert_eq!(i.scalars.len_aligned(), 0));
        let lease = InfoLease::acquire();
        drop(lease);
    }
}
