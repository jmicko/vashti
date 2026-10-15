//! Request-owned native branch layouts. Only shapes and original owner indices
//! live here; parameter bytes stay in the transient I/O owners and their audit.
//! First execution seals a branch's submission sequence. Re-execution must use
//! the same record lengths, page assignments and segment count before replay.
use super::transient_io::fault;
use cubecl_runtime::server::{NativeStrataBranch, ServerError};

const MAX_BRANCHES: usize = 128;
const MAX_SEGMENTS: usize = 256;
const MAX_RECORDS: usize = 32768;
const MAX_PAGES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordingKey {
    pub branch: NativeStrataBranch,
    pub nonce: u64,
    pub segment: usize,
}
#[derive(Default)]
struct Segment {
    pages: Vec<usize>,
    records: Vec<u64>,
    sealed: bool,
}
struct Branch {
    key: NativeStrataBranch,
    nonce: u64,
    segments: Vec<Segment>,
    sealed: bool,
}
#[derive(Clone, Copy)]
struct Active {
    branch: usize,
    segment: usize,
    records: usize,
    pages: usize,
}
#[derive(Default)]
pub(crate) struct Parameters {
    branches: Vec<Branch>,
    active: Option<Active>,
}
impl Parameters {
    pub(crate) fn active(&self) -> bool {
        self.active.is_some()
    }
    pub(crate) fn metadata_bytes(&self) -> usize {
        self.branches.capacity() * size_of::<Branch>()
            + self
                .branches
                .iter()
                .map(|b| {
                    b.segments.capacity() * size_of::<Segment>()
                        + b.segments
                            .iter()
                            .map(|s| {
                                s.pages.capacity() * size_of::<usize>()
                                    + s.records.capacity() * size_of::<u64>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    pub(crate) fn begin(&mut self, key: NativeStrataBranch, nonce: u64) -> Result<(), ServerError> {
        if self.active.is_some() || nonce == 0 {
            return Err(fault("native branch already active"));
        }
        let branch = if let Some(i) = self
            .branches
            .iter()
            .position(|b| b.key == key && b.nonce == nonce)
        {
            i
        } else {
            if self.branches.len() >= MAX_BRANCHES {
                return Err(fault("native branch capacity"));
            }
            self.branches
                .try_reserve_exact(1)
                .map_err(|_| fault("native branch metadata allocation"))?;
            self.branches.push(Branch {
                key,
                nonce,
                segments: Vec::new(),
                sealed: false,
            });
            self.branches.len() - 1
        };
        self.active = Some(Active {
            branch,
            segment: 0,
            records: 0,
            pages: 0,
        });
        Ok(())
    }
    fn segment(&mut self) -> Result<(&mut Segment, Active), ServerError> {
        let a = self.active.ok_or_else(|| fault("native branch inactive"))?;
        let b = &mut self.branches[a.branch];
        if a.segment == b.segments.len() {
            if b.sealed || a.segment >= MAX_SEGMENTS {
                return Err(fault("native branch submission shape"));
            }
            b.segments
                .try_reserve_exact(1)
                .map_err(|_| fault("native branch segment allocation"))?;
            b.segments.push(Segment::default());
        }
        Ok((&mut b.segments[a.segment], a))
    }
    pub(crate) fn record(&mut self, bytes: u64) -> Result<(), ServerError> {
        let (s, a) = self.segment()?;
        if s.sealed {
            if s.records.get(a.records) != Some(&bytes) {
                return Err(fault("native parameter record shape"));
            }
        } else {
            if s.records.len() >= MAX_RECORDS {
                return Err(fault("native parameter record capacity"));
            }
            s.records
                .try_reserve_exact(1)
                .map_err(|_| fault("native parameter record allocation"))?;
            s.records.push(bytes);
        }
        self.active.as_mut().unwrap().records += 1;
        Ok(())
    }
    pub(crate) fn page(&mut self) -> Result<Option<usize>, ServerError> {
        let (s, a) = self.segment()?;
        let page = s.pages.get(a.pages).copied();
        if page.is_none() && s.sealed {
            return Err(fault("native parameter page shape"));
        }
        Ok(page)
    }
    pub(crate) fn use_page(&mut self, owner: usize) -> Result<(), ServerError> {
        let total = self
            .branches
            .iter()
            .flat_map(|b| &b.segments)
            .map(|s| s.pages.len())
            .sum::<usize>();
        let (s, a) = self.segment()?;
        if s.sealed {
            if s.pages.get(a.pages) != Some(&owner) {
                return Err(fault("native parameter owner changed"));
            }
        } else {
            if total >= MAX_PAGES {
                return Err(fault("native branch parameter page capacity"));
            }
            s.pages
                .try_reserve_exact(1)
                .map_err(|_| fault("native parameter owner allocation"))?;
            s.pages.push(owner);
        }
        self.active.as_mut().unwrap().pages += 1;
        Ok(())
    }
    pub(crate) fn recording(&mut self) -> Result<Option<RecordingKey>, ServerError> {
        if self.active.is_none() {
            return Ok(None);
        }
        let (s, a) = self.segment()?;
        if a.records != s.records.len() || a.pages != s.pages.len() {
            return Err(fault("native branch incomplete parameter layout"));
        }
        let b = &self.branches[a.branch];
        Ok(Some(RecordingKey {
            branch: b.key,
            nonce: b.nonce,
            segment: a.segment,
        }))
    }
    /// Seal only after the real submission has accepted its complete inventory.
    /// Completion and reuse still require the original queue fence separately.
    pub(crate) fn submitted(&mut self) -> Result<(), ServerError> {
        if self.active.is_none() {
            return Ok(());
        }
        let (s, a) = self.segment()?;
        if a.records != s.records.len() || a.pages != s.pages.len() {
            return Err(fault("native branch incomplete parameter layout"));
        }
        s.sealed = true;
        let a = self.active.as_mut().unwrap();
        a.segment += 1;
        a.records = 0;
        a.pages = 0;
        Ok(())
    }
    pub(crate) fn end(&mut self, key: NativeStrataBranch, nonce: u64) -> Result<(), ServerError> {
        let a = self.active.ok_or_else(|| fault("native branch inactive"))?;
        let b = &mut self.branches[a.branch];
        if b.key != key
            || b.nonce != nonce
            || a.records != 0
            || a.pages != 0
            || a.segment == 0
            || a.segment != b.segments.len()
        {
            return Err(fault("native branch completion shape"));
        }
        b.sealed = true;
        self.active = None;
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        // These are content-free indices, never ownership or erasure receipts.
        self.active = None;
        self.branches.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn segment(p: &mut Parameters, owner: usize) -> RecordingKey {
        p.record(48).unwrap();
        let previous = p.page().unwrap();
        assert!(previous.is_none() || previous == Some(owner));
        p.use_page(owner).unwrap();
        p.record(32).unwrap();
        let key = p.recording().unwrap().unwrap();
        p.submitted().unwrap();
        key
    }
    #[test]
    fn branch_keeps_original_pages_across_different_shapes_and_replay() {
        let a = NativeStrataBranch::mtp_round(1, 1).unwrap();
        let b = NativeStrataBranch::mtp_round(2, 1).unwrap();
        let mut p = Parameters::default();
        for (key, owner) in [(a, 17), (b, 31), (a, 17), (b, 31)] {
            p.begin(key, 5).unwrap();
            let k = segment(&mut p, owner);
            assert_eq!(k.branch, key);
            assert_eq!(k.nonce, 5);
            p.end(key, 5).unwrap();
        }
        assert_eq!(p.branches.len(), 2);
    }
    #[test]
    fn replay_rejects_short_extra_and_changed_layouts() {
        let key = NativeStrataBranch::target_window(8).unwrap();
        let mut p = Parameters::default();
        p.begin(key, 3).unwrap();
        segment(&mut p, 7);
        p.end(key, 3).unwrap();
        p.begin(key, 3).unwrap();
        assert!(p.record(64).is_err());
        p.record(48).unwrap();
        assert!(p.use_page(9).is_err());
        p.use_page(7).unwrap();
        assert!(p.submitted().is_err());
        p.record(32).unwrap();
        assert!(p.record(32).is_err());
        p.submitted().unwrap();
        assert!(p.recording().is_err());
        p.end(key, 3).unwrap();
    }
    #[test]
    fn segmented_branches_preserve_count_nonce_and_cleanup() {
        let key = NativeStrataBranch::mtp_round(4, 3).unwrap();
        let mut p = Parameters::default();
        p.begin(key, 6).unwrap();
        segment(&mut p, 1);
        assert_eq!(p.recording().unwrap().unwrap().segment, 1);
        segment(&mut p, 2);
        p.end(key, 6).unwrap();
        p.begin(key, 6).unwrap();
        segment(&mut p, 1);
        assert!(p.end(key, 6).is_err());
        segment(&mut p, 2);
        assert!(p.end(key, 7).is_err());
        p.end(key, 6).unwrap();
        p.clear();
        p.begin(key, 7).unwrap();
        assert_eq!(p.recording().unwrap().unwrap().nonce, 7);
        assert_eq!(p.page().unwrap(), None);
    }
}
