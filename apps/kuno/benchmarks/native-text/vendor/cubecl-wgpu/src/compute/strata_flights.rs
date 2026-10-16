//! Bounded native submission bookkeeping. This contains no output, device
//! receipt or completion authority; Graph must join the actual matching fence.
#[derive(Clone, Copy)]
pub(super) struct Flight {
    pub recording: u64,
    pub compute_count: usize,
    pub replay: bool,
    pub profiling: bool,
    pub submitted: bool,
}
#[derive(Default)]
pub(super) struct Book([Option<Flight>; 2]);
impl Book {
    pub fn any(&self) -> bool {
        self.0.iter().any(Option::is_some)
    }
    pub fn live(&self, recording: u64) -> bool {
        self.0.iter().flatten().any(|f| f.recording == recording)
    }
    pub fn peek(&self, lane: usize) -> Result<Option<Flight>, &'static str> {
        self.0.get(lane).copied().ok_or("native flight lane")
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
        self.0[lane] = Some(Flight {
            recording,
            compute_count,
            replay,
            profiling,
            submitted: false,
        });
        Ok(())
    }
    pub fn submitted(&mut self, lane: usize, recording: u64) -> Result<(), &'static str> {
        let f = self
            .0
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
        self.0[lane] = None;
        Ok(f)
    }
}
const _: () = assert!(4 * size_of::<Book>() <= 1024);

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_lanes_retain_distinct_original_recording_and_operation_metadata() {
        let mut b = Book::default();
        b.begin(0, 31, 800, true, false).unwrap();
        b.begin(1, 47, 90, false, true).unwrap();
        assert!(b.any() && b.live(31) && b.live(47));
        let a = b.peek(0).unwrap().unwrap();
        let c = b.peek(1).unwrap().unwrap();
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
        assert!(b.submitted(0, 47).is_err());
        assert!(b.submitted(1, 31).is_err());
        assert!(b.finish_joined(0, 31).is_err()); // partial never becomes completed
        assert!(b.live(31) && !b.peek(0).unwrap().unwrap().submitted);
        b.submitted(0, 31).unwrap();
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
