//! Request-owned native branch layouts. Only shapes and original owner indices
//! live here; parameter bytes stay in the transient I/O owners and their audit.
//! First execution seals a branch's submission sequence. Re-execution must use
//! the same record lengths, page assignments and segment count before replay.
use super::transient_io::fault;
use cubecl_runtime::server::{
    NativeStrataBranch, NativeStrataInvocation, NativeStrataParameter, ServerError,
};

const MAX_BRANCHES: usize = 128;
const MAX_SEGMENTS: usize = 256;
const MAX_RECORDS: usize = 32768;
const MAX_PAGES: usize = 4096;
const MAX_TEMPLATE_BYTES: usize = 2 * 1024 * 1024;
const MAX_PROGRAM_METADATA: usize = 4 * 1024 * 1024;
const NONE: u32 = u32::MAX;

#[derive(Default)]
struct Template(Vec<u8>);
impl Drop for Template {
    fn drop(&mut self) {
        let p = self.0.as_mut_ptr();
        for i in 0..self.0.capacity() {
            // SAFETY: exclusively owned byte allocation, including spare capacity.
            unsafe { p.add(i).write_volatile(0) };
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
    }
}
struct Record {
    bytes: u64,
    template: u32,
    first_field: u32,
    fields: u32,
    owner: u32,
    offset: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RecordingKey {
    pub branch: NativeStrataBranch,
    pub nonce: u64,
    pub segment: usize,
}
#[derive(Default)]
struct Segment {
    pages: Vec<usize>,
    records: Vec<Record>,
    template: Template,
    fields: Vec<NativeStrataParameter>,
    sealed: bool,
}
struct Branch {
    key: NativeStrataBranch,
    nonce: u64,
    typed: bool,
    segments: Vec<Segment>,
    sealed: bool,
}
#[derive(Clone, Copy)]
struct Active {
    branch: usize,
    segment: usize,
    records: usize,
    pages: usize,
    invocation: Option<NativeStrataInvocation>,
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
                                    + s.records.capacity() * size_of::<Record>()
                                    + s.template.0.capacity()
                                    + s.fields.capacity() * size_of::<NativeStrataParameter>()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>()
    }
    #[cfg(test)]
    pub(crate) fn begin(&mut self, key: NativeStrataBranch, nonce: u64) -> Result<(), ServerError> {
        self.begin_invocation(key, nonce, None)
    }
    pub(crate) fn begin_invocation(
        &mut self,
        key: NativeStrataBranch,
        nonce: u64,
        invocation: Option<NativeStrataInvocation>,
    ) -> Result<(), ServerError> {
        if let Some(value) = invocation {
            value.validate_for(key).map_err(fault)?;
        }
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
                typed: invocation.is_some(),
                segments: Vec::new(),
                sealed: false,
            });
            self.branches.len() - 1
        };
        if self.branches[branch].typed != invocation.is_some() {
            return Err(fault("native invocation schema mode changed"));
        }
        self.active = Some(Active {
            branch,
            segment: 0,
            records: 0,
            pages: 0,
            invocation,
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
            if s.records.get(a.records).map(|r| r.bytes) != Some(bytes) {
                return Err(fault("native parameter record shape"));
            }
        } else {
            if s.records.len() >= MAX_RECORDS {
                return Err(fault("native parameter record capacity"));
            }
            s.records
                .try_reserve_exact(1)
                .map_err(|_| fault("native parameter record allocation"))?;
            s.records.push(Record {
                bytes,
                template: NONE,
                first_field: 0,
                fields: 0,
                owner: NONE,
                offset: 0,
            });
        }
        self.active.as_mut().unwrap().records += 1;
        Ok(())
    }
    /// The schema is declared by the actual kernel launcher. No value matching
    /// is used to guess dynamic fields. Templates erase those fields at capture.
    pub(crate) fn record_info(
        &mut self,
        data: &[u8],
        fields: &[NativeStrataParameter],
    ) -> Result<(), ServerError> {
        if data.is_empty()
            || !data.len().is_multiple_of(4)
            || fields.len() > 8
            || fields.iter().enumerate().any(|(i, f)| {
                f.word as usize * 4 + 4 > data.len()
                    || fields[..i].iter().any(|old| old.word == f.word)
            })
        {
            return Err(fault("native INFO field schema"));
        }
        let active = self.active.ok_or_else(|| fault("native branch inactive"))?;
        let Some(invocation) = active.invocation else {
            return self.record(data.len() as u64);
        };
        for field in fields {
            let at = field.word as usize * 4;
            let observed = u32::from_le_bytes(data[at..at + 4].try_into().unwrap());
            if observed != invocation.value(field.value).map_err(fault)? {
                return Err(fault("native INFO invocation value mismatch"));
            }
        }
        let sealed = self.branches[active.branch]
            .segments
            .get(active.segment)
            .is_some_and(|s| s.sealed);
        if !sealed {
            let template_bytes: usize = self
                .branches
                .iter()
                .flat_map(|b| &b.segments)
                .map(|s| s.template.0.len())
                .sum();
            if template_bytes
                .checked_add(data.len())
                .is_none_or(|n| n > MAX_TEMPLATE_BYTES)
                || self
                    .metadata_bytes()
                    .checked_add(data.len() + size_of::<Record>() + std::mem::size_of_val(fields))
                    .is_none_or(|n| n > MAX_PROGRAM_METADATA)
            {
                return Err(fault("native invocation template budget"));
            }
        }
        let (s, a) = self.segment()?;
        if s.sealed {
            let r = s
                .records
                .get(a.records)
                .ok_or_else(|| fault("native INFO record shape"))?;
            if r.bytes != data.len() as u64
                || r.template == NONE
                || s.fields
                    .get(r.first_field as usize..(r.first_field + r.fields) as usize)
                    != Some(fields)
            {
                return Err(fault("native INFO schema changed"));
            }
            let template = &s.template.0[r.template as usize..r.template as usize + data.len()];
            for (at, (&old, &new)) in template.iter().zip(data).enumerate() {
                if !fields.iter().any(|f| at / 4 == f.word as usize) && old != new {
                    return Err(fault("native INFO static value changed"));
                }
            }
        } else {
            if s.records.len() >= MAX_RECORDS {
                return Err(fault("native parameter record capacity"));
            }
            s.records
                .try_reserve_exact(1)
                .map_err(|_| fault("native record allocation"))?;
            s.template
                .0
                .try_reserve_exact(data.len())
                .map_err(|_| fault("native template allocation"))?;
            s.fields
                .try_reserve_exact(fields.len())
                .map_err(|_| fault("native field allocation"))?;
            let first = s.template.0.len();
            let first_field = s.fields.len();
            s.template.0.extend_from_slice(data);
            for field in fields {
                let at = first + field.word as usize * 4;
                s.template.0[at..at + 4].fill(0);
            }
            s.fields.extend_from_slice(fields);
            s.records.push(Record {
                bytes: data.len() as u64,
                template: first as u32,
                first_field: first_field as u32,
                fields: fields.len() as u32,
                owner: NONE,
                offset: 0,
            });
        }
        self.active.as_mut().unwrap().records += 1;
        Ok(())
    }
    /// Bind the schema to its original uniform owner and offset. Replays cannot
    /// move a record into a different physical page or claim a sliced new owner.
    pub(crate) fn bind_current(&mut self, owner: usize, offset: u64) -> Result<(), ServerError> {
        let owner = u32::try_from(owner).map_err(|_| fault("native INFO owner index"))?;
        let offset = u32::try_from(offset).map_err(|_| fault("native INFO page offset"))?;
        let (s, a) = self.segment()?;
        let r = s
            .records
            .get_mut(
                a.records
                    .checked_sub(1)
                    .ok_or_else(|| fault("native INFO record missing"))?,
            )
            .ok_or_else(|| fault("native INFO record missing"))?;
        if s.sealed {
            if r.owner != owner || r.offset != offset {
                return Err(fault("native INFO record binding changed"));
            }
        } else {
            r.owner = owner;
            r.offset = offset;
        }
        Ok(())
    }
    /// Emit actual GPU INFO bytes from the captured template and fresh controls.
    /// The caller lends a mapped slice of the already tracked upload source.
    pub(crate) fn write_current(
        &self,
        output_len: usize,
        mut write: impl FnMut(usize, &[u8]),
    ) -> Result<(), ServerError> {
        let a = self.active.ok_or_else(|| fault("native branch inactive"))?;
        let Some(invocation) = a.invocation else {
            return Ok(());
        };
        let s = &self.branches[a.branch].segments[a.segment];
        let r = s
            .records
            .get(
                a.records
                    .checked_sub(1)
                    .ok_or_else(|| fault("native INFO record missing"))?,
            )
            .ok_or_else(|| fault("native INFO record missing"))?;
        if r.bytes != output_len as u64 || r.template == NONE {
            return Err(fault("native INFO render shape"));
        }
        write(
            0,
            &s.template.0[r.template as usize..r.template as usize + output_len],
        );
        for field in &s.fields[r.first_field as usize..(r.first_field + r.fields) as usize] {
            let at = field.word as usize * 4;
            write(
                at,
                &invocation.value(field.value).map_err(fault)?.to_le_bytes(),
            );
        }
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
        if a.records != s.records.len()
            || a.pages != s.pages.len()
            || s.records
                .iter()
                .any(|r| r.template != NONE && r.owner == NONE)
        {
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
        if a.records != s.records.len()
            || a.pages != s.pages.len()
            || s.records
                .iter()
                .any(|r| r.template != NONE && r.owner == NONE)
        {
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
        if let Some(value) = self.active.as_mut().and_then(|a| a.invocation.as_mut()) {
            value.wipe_private();
        }
        self.active = None;
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        // These are content-free indices, never ownership or erasure receipts.
        if let Some(value) = self.active.as_mut().and_then(|a| a.invocation.as_mut()) {
            value.wipe_private();
        }
        self.active = None;
        self.branches.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn payload(position: u32, epoch: u32, constant: u32) -> Vec<u8> {
        [4u32, position, epoch, constant]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect()
    }
    fn fields() -> [NativeStrataParameter; 2] {
        use cubecl_runtime::server::NativeStrataValue;
        [
            NativeStrataParameter {
                word: 1,
                value: NativeStrataValue::Position { offset: 0 },
            },
            NativeStrataParameter {
                word: 2,
                value: NativeStrataValue::RouteEpoch,
            },
        ]
    }
    fn typed_segment(p: &mut Parameters, data: &[u8]) -> RecordingKey {
        p.record_info(data, &fields()).unwrap();
        assert!(p.page().unwrap().is_none_or(|id| id == 17));
        p.use_page(17).unwrap();
        p.bind_current(17, 32).unwrap();
        let mut emitted = vec![0u8; data.len()];
        p.write_current(emitted.len(), |at, bytes| {
            emitted[at..at + bytes.len()].copy_from_slice(bytes);
        })
        .unwrap();
        assert_eq!(emitted, data);
        let key = p.recording().unwrap().unwrap();
        p.submitted().unwrap();
        key
    }
    #[test]
    fn typed_controls_refresh_original_bound_records_without_value_keys() {
        let key = NativeStrataBranch::target_window_for_context(4, 256).unwrap();
        let mut p = Parameters::default();
        let mut first = None;
        for (position, epoch) in [(11, 7), (31, 8), (240, 999)] {
            p.begin_invocation(
                key,
                5,
                Some(NativeStrataInvocation::target(position, epoch).unwrap()),
            )
            .unwrap();
            let actual = typed_segment(&mut p, &payload(position, epoch, 123));
            assert!(first.is_none_or(|old| old == actual));
            first = Some(actual);
            // Only immutable data remains: dynamic control words are erased
            // in the retained CPU template, and never enter recording keys.
            assert_eq!(&p.branches[0].segments[0].template.0[4..12], &[0; 8]);
            p.end(key, 5).unwrap();
            assert!(p.active.is_none());
        }
        assert_eq!(p.branches.len(), 1);
    }
    #[test]
    fn typed_layout_rejects_stale_values_changed_constants_schema_and_owner() {
        let key = NativeStrataBranch::target_window(4).unwrap();
        let mut p = Parameters::default();
        p.begin_invocation(key, 3, Some(NativeStrataInvocation::target(11, 7).unwrap()))
            .unwrap();
        typed_segment(&mut p, &payload(11, 7, 123));
        p.end(key, 3).unwrap();
        p.begin_invocation(key, 3, Some(NativeStrataInvocation::target(31, 8).unwrap()))
            .unwrap();
        assert!(p.record_info(&payload(11, 7, 123), &fields()).is_err());
        assert!(p.record_info(&payload(31, 8, 124), &fields()).is_err());
        assert!(p.record_info(&payload(31, 8, 123), &fields()[..1]).is_err());
        p.record_info(&payload(31, 8, 123), &fields()).unwrap();
        p.use_page(17).unwrap();
        assert!(p.bind_current(18, 32).is_err());
        assert!(p.bind_current(17, 36).is_err());
        p.bind_current(17, 32).unwrap();
        p.submitted().unwrap();
        p.end(key, 3).unwrap();
        assert!(p.begin(key, 3).is_err());
        p.clear();
        assert!(p.branches.is_empty());
    }
    #[test]
    fn typed_invocation_bounds_and_schema_fail_before_parameter_retention() {
        use cubecl_runtime::server::NativeStrataValue;
        assert!(NativeStrataInvocation::target(0, 0).is_err());
        assert!(NativeStrataInvocation::target(4096, 1).is_err());
        let value = NativeStrataInvocation::target(1, 1).unwrap();
        assert!(
            value
                .value(NativeStrataValue::Position { offset: -2 })
                .is_err()
        );
        let mut p = Parameters::default();
        let key = NativeStrataBranch::target_window_for_context(4, 32).unwrap();
        assert!(
            p.begin_invocation(key, 1, Some(NativeStrataInvocation::target(30, 1).unwrap()))
                .is_err()
        );
        assert!(
            p.begin_invocation(NativeStrataBranch::mtp_step(1).unwrap(), 1, Some(value))
                .is_err()
        );
        p.begin_invocation(key, 1, Some(value)).unwrap();
        assert!(p.record_info(&[0; 3], &[]).is_err());
        assert!(
            p.record_info(&payload(1, 1, 123), &[fields()[0], fields()[0]])
                .is_err()
        );
        assert!(
            p.record_info(
                &payload(1, 1, 123),
                &[NativeStrataParameter {
                    word: 4,
                    value: NativeStrataValue::RouteEpoch
                }]
            )
            .is_err()
        );
        assert!(
            p.record_info(&vec![0; MAX_TEMPLATE_BYTES + 4], &[])
                .is_err()
        );
        assert!(p.branches[0].segments.is_empty());
        assert_eq!(p.active.unwrap().records, 0);
    }
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
