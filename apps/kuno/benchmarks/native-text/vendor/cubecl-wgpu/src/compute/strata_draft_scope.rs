//! Original verified-R scope bookkeeping. No completion or numerical authority.
//! Graph pins the actual allocation and performs the real signal, wait and join.
use super::strata_schedule::Access;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Identity {
    pub generation: u64,
    pub request: u64,
}
struct Live {
    identity: Identity,
    source: Access,
    ready: bool,
    active: bool,
}
#[derive(Default)]
pub(super) struct Book {
    next: u64,
    live: Option<Live>,
}
impl Book {
    pub fn prepare(&mut self, request: u64, source: Access) -> Result<Identity, &'static str> {
        end(&source)?;
        if request == 0 || source.write || self.live.is_some() {
            return Err("native draft scope already owned or invalid");
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or("native draft scope exhausted")?;
        let identity = Identity {
            generation: self.next,
            request,
        };
        self.live = Some(Live {
            identity,
            source,
            ready: false,
            active: false,
        });
        Ok(identity)
    }
    fn original(&self, identity: Identity) -> Result<&Live, &'static str> {
        self.live
            .as_ref()
            .filter(|s| s.identity == identity)
            .ok_or("native draft original scope changed")
    }
    /// Called only after Graph submitted BOTH the original main signal and
    /// independent-lane wait. A partial submission leaves the source retained.
    pub fn dependency_submitted(&mut self, identity: Identity) -> Result<(), &'static str> {
        let s = self.original(identity)?;
        if s.ready || s.active {
            return Err("native draft dependency repeated");
        }
        self.live.as_mut().unwrap().ready = true;
        Ok(())
    }
    pub fn enter(&mut self, identity: Identity) -> Result<(), &'static str> {
        let s = self.original(identity)?;
        if !s.ready || s.active {
            return Err("native draft dependency absent or scope active");
        }
        self.live.as_mut().unwrap().active = true;
        Ok(())
    }
    pub fn check_active(&self, identity: Identity) -> Result<(), &'static str> {
        if !self.original(identity)?.active {
            return Err("native draft scope not active");
        }
        Ok(())
    }
    /// All unkeyed front/scratch work stays on the draft lane, including the
    /// interval between named first and continuation branches. A named target
    /// may never silently run inside an active MTP scope.
    pub fn lane(&self, named: Option<usize>) -> Result<usize, &'static str> {
        if named.is_some_and(|lane| lane > 1) {
            return Err("native draft lane extent");
        }
        if self.live.as_ref().is_some_and(|s| s.active) {
            if named == Some(0) {
                return Err("native target branch inside draft scope");
            }
            Ok(1)
        } else {
            if self.live.is_some() && named == Some(1) {
                return Err("native draft branch before scope entry");
            }
            Ok(named.unwrap_or(0))
        }
    }
    pub fn active_request(&self) -> Option<u64> {
        self.live
            .as_ref()
            .filter(|s| s.active)
            .map(|s| s.identity.request)
    }
    /// Full native ranges influence enqueue admission from PREPARATION, not
    /// just active draft execution. Outside this lifetime Graph still performs
    /// its original unconditional submission join and validates every actual
    /// resource under the real capture guard before any command can execute.
    pub fn needs_access_inventory(&self) -> bool {
        self.live.is_some()
    }
    /// Exact original owner interval, not an assumed whole VkBuffer. Shared
    /// readers and disjoint writes are legal; source writes/reset are forbidden
    /// from preparation through actual scope completion/terminal drain.
    pub fn check_accesses(&self, incoming: &[Access]) -> Result<(), &'static str> {
        let mut conflict = false;
        for access in incoming {
            let last = end(access)?;
            if let Some(s) = &self.live {
                conflict |= access.write
                    && access.buffer == s.source.buffer
                    && access.first < s.source.first + s.source.bytes
                    && s.source.first < last;
            }
        }
        if conflict {
            return Err("native verified draft source write while retained");
        }
        Ok(())
    }
    pub fn protected(&self) -> Option<&Access> {
        self.live.as_ref().map(|s| &s.source)
    }
    /// Graph caller has joined actual original draft work and readiness fence.
    pub fn finish_joined(&mut self, identity: Identity) -> Result<(), &'static str> {
        self.check_active(identity)?;
        self.live = None;
        Ok(())
    }
    /// Terminal owning cleanup only, after actual queue drains even on failure.
    /// This cannot publish state or make a failed inference successful.
    pub fn discard_after_drain(&mut self) {
        self.live = None;
    }
}
fn end(a: &Access) -> Result<u64, &'static str> {
    a.first
        .checked_add(a.bytes)
        .filter(|_| a.buffer != 0 && a.bytes != 0)
        .ok_or("native draft source/access extent")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source() -> Access {
        Access {
            buffer: 71,
            first: 4096,
            bytes: 2048,
            write: false,
        }
    }
    #[test]
    fn original_dependency_and_scope_cannot_be_forged_reentered_or_reused() {
        let mut b = Book::default();
        let id = b.prepare(9, source()).unwrap();
        assert!(b.prepare(9, source()).is_err());
        assert!(b.enter(id).is_err()); // No real dependency submission yet.
        for foreign in [
            Identity { request: 8, ..id },
            Identity {
                generation: 2,
                ..id
            },
        ] {
            assert!(b.dependency_submitted(foreign).is_err());
            assert!(b.enter(foreign).is_err());
        }
        assert_eq!(b.active_request(), None);
        b.dependency_submitted(id).unwrap(); // Synthetic host outcome only.
        assert!(b.dependency_submitted(id).is_err());
        b.enter(id).unwrap();
        assert_eq!(b.active_request(), Some(9));
        assert!(b.enter(id).is_err());
        b.check_active(id).unwrap();
        b.finish_joined(id).unwrap(); // Production must perform actual joins.
        assert_eq!(b.active_request(), None);
        assert!(b.finish_joined(id).is_err());
        let next = b.prepare(9, source()).unwrap();
        assert!(next.generation > id.generation);
        assert!(b.enter(id).is_err());
        b.discard_after_drain();
        assert!(b.enter(next).is_err());
        b.next = u64::MAX;
        assert!(b.prepare(9, source()).is_err());
    }
    #[test]
    fn unkeyed_staging_and_all_draft_branches_keep_the_original_lane() {
        let mut b = Book::default();
        let id = b.prepare(9, source()).unwrap();
        assert_eq!(b.lane(None), Ok(0));
        assert_eq!(b.lane(Some(0)), Ok(0)); // Original commit before entry.
        assert!(b.lane(Some(1)).is_err());
        b.dependency_submitted(id).unwrap();
        b.enter(id).unwrap();
        for _ in 0..8 {
            assert_eq!(b.lane(None), Ok(1));
            assert_eq!(b.lane(Some(1)), Ok(1));
            assert!(b.lane(Some(0)).is_err());
        }
        assert!(b.lane(Some(2)).is_err());
        b.finish_joined(id).unwrap();
        assert_eq!(b.lane(None), Ok(0));
    }
    #[test]
    fn source_is_read_only_from_prepare_through_drain_including_partial_signal() {
        let mut b = Book::default();
        assert!(!b.needs_access_inventory());
        let id = b.prepare(9, source()).unwrap();
        for phase in 0..3 {
            assert!(b.needs_access_inventory());
            let a = |first, bytes, write| Access {
                buffer: 71,
                first,
                bytes,
                write,
            };
            b.check_accesses(&[a(0, 4096, true), a(6144, 20, true), a(4096, 2048, false)])
                .unwrap();
            for (first, bytes) in [(0, 4097), (4095, 2), (4096, 1), (6143, 2), (0, 9000)] {
                assert!(b.check_accesses(&[a(first, bytes, true)]).is_err());
            }
            b.check_accesses(&[Access {
                buffer: 72,
                ..a(4096, 2048, true)
            }])
            .unwrap();
            assert!(
                b.check_accesses(&[a(4096, 1, true), a(u64::MAX, 1, false)])
                    .is_err()
            );
            if phase == 0 {
                b.dependency_submitted(id).unwrap();
            }
            if phase == 1 {
                b.enter(id).unwrap();
            }
        }
        b.finish_joined(id).unwrap();
        assert!(!b.needs_access_inventory());
        b.check_accesses(&[Access {
            write: true,
            ..source()
        }])
        .unwrap();
        assert!(
            b.check_accesses(&[Access {
                bytes: 0,
                ..source()
            }])
            .is_err()
        );
        assert!(std::mem::size_of::<Book>() < 256);
    }

    #[test]
    fn enqueue_inventory_is_required_until_the_actual_protected_owner_is_retired() {
        let mut book = Book::default();
        for request in 1..=16 {
            assert!(!book.needs_access_inventory());
            let original = book.prepare(request, source()).unwrap();
            assert!(book.needs_access_inventory());
            // Prepared but not active is deliberately NOT the fast path.
            assert!(book.active_request().is_none());
            assert!(
                book.check_accesses(&[Access {
                    write: true,
                    ..source()
                }])
                .is_err()
            );
            if request % 2 == 0 {
                book.dependency_submitted(original).unwrap();
                assert!(book.needs_access_inventory());
                book.enter(original).unwrap();
                assert!(book.needs_access_inventory());
                assert!(
                    book.finish_joined(Identity {
                        request: request + 1,
                        ..original
                    })
                    .is_err()
                );
                assert!(book.needs_access_inventory());
                book.finish_joined(original).unwrap();
            } else {
                // Only the existing terminal drain retires partial scopes.
                assert!(book.enter(original).is_err());
                assert!(book.needs_access_inventory());
                book.discard_after_drain();
            }
            assert!(!book.needs_access_inventory());
        }
    }
}
