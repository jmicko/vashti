//! Bounded native submission bookkeeping. This contains no output, device
//! receipt or completion authority; Graph must join the actual matching fence.
#[derive(Clone, Copy)]
pub(super) struct Flight {
    pub recording: u64,
    pub submission: u64,
    pub info_epoch: u64,
    pub compute_count: usize,
    pub replay: bool,
    pub profiling: bool,
    pub submitted: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Export {
    pub recording: u64,
    pub submission: u64,
    pub scope: u64,
    pub joined: bool,
}
#[derive(Default)]
pub(super) struct Book {
    lanes: [Option<Flight>; 2],
    exports: [Option<Export>; 2],
    next: u64,
}
impl Book {
    pub fn any(&self) -> bool {
        self.lanes.iter().any(Option::is_some)
    }
    pub fn live(&self, recording: u64) -> bool {
        self.lanes
            .iter()
            .flatten()
            .any(|f| f.recording == recording)
    }
    pub fn peek(&self, lane: usize) -> Result<Option<Flight>, &'static str> {
        self.lanes.get(lane).copied().ok_or("native flight lane")
    }
    pub fn begin(
        &mut self,
        lane: usize,
        recording: u64,
        compute_count: usize,
        replay: bool,
        profiling: bool,
    ) -> Result<(), &'static str> {
        if recording == 0
            || compute_count > 32768
            || self.live(recording)
            || self.peek(lane)?.is_some()
        {
            return Err("native flight already owned or invalid");
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or("native submission epoch exhausted")?;
        self.lanes[lane] = Some(Flight {
            recording,
            submission: self.next,
            info_epoch: 0,
            compute_count,
            replay,
            profiling,
            submitted: false,
        });
        Ok(())
    }
    pub fn header_wait(
        &mut self,
        lane: usize,
        recording: u64,
        epoch: u64,
    ) -> Result<(), &'static str> {
        let f = self
            .lanes
            .get_mut(lane)
            .ok_or("native header lane")?
            .as_mut()
            .ok_or("native header flight missing")?;
        if f.recording != recording || f.submitted || f.info_epoch != 0 || epoch == 0 {
            return Err("native header original flight mismatch");
        }
        f.info_epoch = epoch;
        Ok(())
    }
    pub fn submitted(&mut self, lane: usize, recording: u64) -> Result<(), &'static str> {
        let f = self
            .lanes
            .get_mut(lane)
            .ok_or("native flight lane")?
            .as_mut()
            .ok_or("native flight missing")?;
        if f.recording != recording || f.submitted {
            return Err("native flight submission changed");
        }
        f.submitted = true;
        Ok(())
    }
    /// Host bookkeeping only. Its sole production caller has waited on the
    /// exact lane's real fence and accepted the matching original recording.
    pub fn finish_joined(&mut self, lane: usize, recording: u64) -> Result<Flight, &'static str> {
        let f = self.peek(lane)?.ok_or("native flight missing")?;
        if !f.submitted || f.recording != recording {
            return Err("native flight has no matched joined submission");
        }
        if let Some(export) = &mut self.exports[lane]
            && export.submission == f.submission
        {
            if export.recording != recording {
                return Err("native exported recording changed");
            }
            export.joined = true;
        }
        self.lanes[lane] = None;
        Ok(f)
    }
    pub fn export_available(&self, lane: usize) -> Result<bool, &'static str> {
        self.exports
            .get(lane)
            .map(Option::is_none)
            .ok_or("native export lane")
    }
    /// Original Graph caller has submitted this exact recording on its lane.
    /// No completion/fault/state authority follows from exporting its identity.
    pub fn expose(&mut self, lane: usize, scope: u64) -> Result<u64, &'static str> {
        let flight = self
            .peek(lane)?
            .ok_or("native export has no original flight")?;
        if !flight.submitted || scope == 0 || self.exports[lane].is_some() {
            return Err("native export incomplete or already owned");
        }
        self.exports[lane] = Some(Export {
            recording: flight.recording,
            submission: flight.submission,
            scope,
            joined: false,
        });
        Ok(flight.submission)
    }
    pub fn exported(&self, submission: u64, scope: u64) -> Result<(usize, Export), &'static str> {
        self.exports
            .iter()
            .enumerate()
            .find_map(|(lane, e)| {
                e.filter(|e| e.submission == submission && e.scope == scope)
                    .map(|e| (lane, e))
            })
            .ok_or("native exported submission missing or foreign")
    }
    /// Graph has joined the original fence, possibly at an ordinary dependent
    /// drain. Acknowledge only that retained original completion metadata.
    pub fn acknowledge(&mut self, submission: u64, scope: u64) -> Result<(), &'static str> {
        let (lane, export) = self.exported(submission, scope)?;
        if !export.joined {
            return Err("native exported submission not joined");
        }
        self.exports[lane] = None;
        Ok(())
    }
    /// Explicit original cleanup only, after every actual math flight joined.
    /// Keep the monotonic epoch so an old token cannot name later work.
    pub fn clear_joined_exports(&mut self) -> Result<(), &'static str> {
        if self.any() || self.exports.iter().flatten().any(|e| !e.joined) {
            return Err("native export cleanup before original join");
        }
        self.exports = [None; 2];
        Ok(())
    }
}
const _: () = assert!(4 * size_of::<Book>() <= 1024);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exported_submission_distinguishes_replays_and_retains_original_join_until_ack() {
        let mut b = Book::default();
        b.begin(0, 31, 133, true, false).unwrap();
        assert!(b.expose(0, 5).is_err());
        assert!(b.clear_joined_exports().is_err());
        b.submitted(0, 31).unwrap();
        let old = b.expose(0, 5).unwrap();
        assert!(!b.export_available(0).unwrap());
        assert!(!b.exported(old, 5).unwrap().1.joined);
        assert!(b.acknowledge(old, 5).is_err());
        assert!(b.exported(old, 6).is_err());
        // Host bookkeeping only; production Graph must join the real fence.
        b.finish_joined(0, 31).unwrap();
        assert!(b.exported(old, 5).unwrap().1.joined);
        b.begin(0, 31, 133, true, false).unwrap();
        b.submitted(0, 31).unwrap();
        assert_ne!(b.peek(0).unwrap().unwrap().submission, old);
        assert!(b.expose(0, 5).is_err());
        b.acknowledge(old, 5).unwrap();
        assert!(b.live(31));
        let new = b.expose(0, 5).unwrap();
        assert_ne!(new, old);
        assert!(b.acknowledge(old, 5).is_err());
        assert!(b.exported(old, 5).is_err());
        assert!(!b.exported(new, 5).unwrap().1.joined);
        b.finish_joined(0, 31).unwrap();
        b.acknowledge(new, 5).unwrap();
        assert!(b.acknowledge(new, 5).is_err());
    }
    #[test]
    fn exported_cleanup_requires_original_completion_and_never_resets_epoch() {
        let mut b = Book::default();
        b.begin(0, 31, 133, false, false).unwrap();
        b.submitted(0, 31).unwrap();
        let a = b.expose(0, 5).unwrap();
        b.begin(1, 47, 90, true, false).unwrap();
        b.submitted(1, 47).unwrap();
        let c = b.expose(1, 5).unwrap();
        b.finish_joined(1, 47).unwrap();
        assert!(b.clear_joined_exports().is_err());
        assert!(!b.exported(a, 5).unwrap().1.joined);
        assert!(b.exported(c, 5).unwrap().1.joined);
        b.finish_joined(0, 31).unwrap();
        b.clear_joined_exports().unwrap();
        assert!(b.exported(a, 5).is_err() && b.exported(c, 5).is_err());
        b.begin(0, 31, 133, true, false).unwrap();
        b.submitted(0, 31).unwrap();
        let next = b.expose(0, 6).unwrap();
        assert!(next > c && c > a);
        assert!(b.exported(next, 5).is_err());
        b.finish_joined(0, 31).unwrap();
        b.acknowledge(next, 6).unwrap();
        b.next = u64::MAX;
        assert!(b.begin(0, 31, 133, true, false).is_err());
        assert!(!b.any() && b.next == u64::MAX);
    }
    #[test]
    fn both_lanes_retain_distinct_original_recording_and_operation_metadata() {
        let mut b = Book::default();
        b.begin(0, 31, 800, true, false).unwrap();
        b.begin(1, 47, 90, false, true).unwrap();
        b.header_wait(0, 31, 7).unwrap();
        b.header_wait(1, 47, 9).unwrap();
        assert!(b.header_wait(0, 31, 8).is_err());
        assert!(b.any() && b.live(31) && b.live(47));
        let a = b.peek(0).unwrap().unwrap();
        let c = b.peek(1).unwrap().unwrap();
        assert_eq!((a.info_epoch, c.info_epoch), (7, 9));
        assert_eq!(
            (a.recording, a.compute_count, a.replay, a.profiling),
            (31, 800, true, false)
        );
        assert_eq!(
            (c.recording, c.compute_count, c.replay, c.profiling),
            (47, 90, false, true)
        );
    }
    #[test]
    fn duplicate_or_foreign_submission_never_releases_or_overwrites_original_lane() {
        let mut b = Book::default();
        b.begin(0, 31, 800, true, false).unwrap();
        assert!(b.begin(0, 47, 2, false, false).is_err());
        assert!(b.begin(1, 31, 2, false, false).is_err());
        assert!(b.begin(2, 47, 2, false, false).is_err());
        assert!(b.header_wait(0, 47, 7).is_err());
        assert!(b.header_wait(0, 31, 0).is_err());
        assert!(b.submitted(0, 47).is_err());
        assert!(b.submitted(1, 31).is_err());
        assert!(b.finish_joined(0, 31).is_err()); // partial never becomes completed
        assert!(b.live(31) && !b.peek(0).unwrap().unwrap().submitted);
        b.submitted(0, 31).unwrap();
        assert!(b.header_wait(0, 31, 7).is_err());
        assert!(b.submitted(0, 31).is_err());
        assert!(b.finish_joined(0, 47).is_err());
        assert!(b.live(31));
    }
    #[test]
    fn joined_host_bookkeeping_removes_only_the_exact_original_slot() {
        let mut b = Book::default();
        b.begin(0, 31, 0, false, false).unwrap(); // real transfer-only graph
        b.begin(1, 47, 90, true, false).unwrap();
        b.submitted(0, 31).unwrap();
        b.submitted(1, 47).unwrap();
        let f = b.finish_joined(1, 47).unwrap();
        assert_eq!((f.recording, f.compute_count), (47, 90));
        assert!(b.live(31) && !b.live(47) && b.any());
        assert!(b.finish_joined(1, 47).is_err());
        assert!(b.begin(1, 0, 1, false, false).is_err());
        assert!(b.begin(1, 48, 32769, false, false).is_err());
        b.finish_joined(0, 31).unwrap();
        assert!(!b.any());
    }
}
