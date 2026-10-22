//! Selection evidence for a genuine, completely validated expert cohort batch.
//! This never grants execution, source, reuse, replay or completion authority.
use super::transient_io::{NativeExpertBatchScope, NativeExpertCohortStamp, fault};
use cubecl_core::server::ServerError;
use cubecl_runtime::server::native_expert_grammar::NativeStrataExpertGrammar;

struct Pending {
    stamp: NativeExpertCohortStamp,
    graph: u64,
    first: usize,
    launches: usize,
}
struct Closed {
    stamp: NativeExpertCohortStamp,
    graph: u64,
    first: usize,
    end: usize,
    launches: usize,
}
/// Fixed-size scratch: retaining the latest fully sealed batch is enough to
/// prove that the current append-only recording contains authentic cohorts.
/// A later batch cannot inherit its predecessor's pending/partial evidence.
#[derive(Default)]
pub(super) struct Book {
    pending: Option<Pending>,
    closed: Option<Closed>,
}
pub(super) struct Witness {
    graph: u64,
    nodes: usize,
    compute: usize,
}
pub(super) struct RecordingAdmission {
    graph: u64,
    id: u64,
    family: u32,
    nodes: usize,
    compute: usize,
}
impl RecordingAdmission {
    pub(super) fn matches(
        &self,
        graph: u64,
        id: u64,
        family: u32,
        nodes: usize,
        compute: usize,
    ) -> bool {
        self.graph == graph
            && self.id == id
            && self.family == family
            && self.nodes == nodes
            && self.compute == compute
            && id != 0
            && compute > 0
            && compute <= nodes
            && nodes <= 32768
    }
}
impl Witness {
    pub(super) fn recording(&self, id: u64, family: u32) -> RecordingAdmission {
        RecordingAdmission {
            graph: self.graph,
            id,
            family,
            nodes: self.nodes,
            compute: self.compute,
        }
    }
    pub(super) fn rolling(
        &self,
        lane: usize,
        named: bool,
        profiling: bool,
        nodes: usize,
        compute: usize,
    ) -> bool {
        lane == 0
            && !named
            && !profiling
            && nodes == self.nodes
            && compute == self.compute
            && compute > 0
            && compute <= nodes
            && nodes <= 32768
    }
}
impl Book {
    pub(super) const fn metadata_bytes() -> usize {
        core::mem::size_of::<Self>()
            + core::mem::size_of::<Witness>()
            + core::mem::size_of::<Option<RecordingAdmission>>()
    }
    /// Only the original grammar can provide these counts. The caller must
    /// first validate this exact scope against its real active TransientIo.
    pub(super) fn begin(
        &mut self,
        scope: &NativeExpertBatchScope,
        grammar: &NativeStrataExpertGrammar,
        graph: u64,
        first: usize,
    ) -> Result<(), ServerError> {
        let counts = grammar.finish()?;
        if counts.cohort_groups == 0 {
            if counts.cohort_launches != 0 || counts.cohort_batches != 0 {
                return Err(fault("native cohort empty grammar evidence"));
            }
            return Ok(());
        }
        let stamp = scope.cohort_stamp();
        if self.pending.is_some()
            || graph == 0
            || first > 32768
            || !stamp.live()
            || self.closed.as_ref().is_some_and(|old| {
                !old.stamp.same_owner_epoch(&stamp) || old.graph != graph || old.end > first
            })
            || counts.cohort_batches == 0
            || counts.cohort_groups > 512
            || counts.cohort_batches > counts.cohort_groups
            || counts.cohort_groups > counts.cohort_batches * 8
            || counts.logical_phases == 0
            || counts.logical_phases % 6 != 0
            || counts.logical_phases / 6 > 512
            || counts.cohort_groups > counts.logical_phases / 6
            || counts.cohort_launches < counts.cohort_batches * 6
            || counts.cohort_launches > counts.cohort_batches * 8
            || counts.launches < counts.cohort_launches
            || counts.launches > 5632
        {
            return Err(fault("native cohort complete grammar/scope population"));
        }
        self.pending = Some(Pending {
            stamp,
            graph,
            first,
            launches: counts.launches,
        });
        Ok(())
    }
    pub(super) fn pending_first(&self) -> Option<usize> {
        self.pending.as_ref().map(|p| p.first)
    }
    pub(super) fn finish(
        &mut self,
        scope: &NativeExpertBatchScope,
        graph: u64,
        end: usize,
        actual_compute: usize,
        success: bool,
    ) -> Result<(), ServerError> {
        if !success {
            self.clear();
            return Ok(());
        }
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        if !p.stamp.matches(scope)
            || p.graph != graph
            || end <= p.first
            || end > 32768
            || actual_compute != p.launches
            || actual_compute > end - p.first
        {
            self.clear();
            return Err(fault(
                "native cohort actual scoped enqueue population changed",
            ));
        }
        self.closed = Some(Closed {
            stamp: p.stamp.clone(),
            graph,
            first: p.first,
            end,
            launches: p.launches,
        });
        Ok(())
    }
    pub(super) fn closed_range(&self) -> Option<core::ops::Range<usize>> {
        self.closed.as_ref().map(|c| c.first..c.end)
    }
    pub(super) fn witness(
        &self,
        graph: u64,
        nodes: usize,
        compute: usize,
        actual_range_compute: usize,
    ) -> Result<Option<Witness>, ServerError> {
        if self.pending.is_some() {
            return Err(fault("native cohort cannot submit incomplete batch"));
        }
        let Some(c) = &self.closed else {
            return Ok(None);
        };
        if !c.stamp.live()
            || c.graph != graph
            || c.first >= c.end
            || c.end > nodes
            || nodes > 32768
            || compute < c.launches
            || compute > nodes
            || actual_range_compute != c.launches
        {
            return Err(fault("native cohort current graph population changed"));
        }
        Ok(Some(Witness {
            graph,
            nodes,
            compute,
        }))
    }
    pub(super) fn clear(&mut self) {
        self.pending = None;
        self.closed = None;
    }
}
fn wipe(word: &mut usize) {
    unsafe {
        core::ptr::write_volatile(word, 0);
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}
impl Pending {
    fn erase(&mut self) {
        unsafe {
            core::ptr::write_volatile(&mut self.graph, 0);
        }
        wipe(&mut self.first);
        wipe(&mut self.launches);
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.erase();
    }
}
impl Closed {
    fn erase(&mut self) {
        unsafe {
            core::ptr::write_volatile(&mut self.graph, 0);
        }
        wipe(&mut self.first);
        wipe(&mut self.end);
        wipe(&mut self.launches);
    }
}
impl Drop for Closed {
    fn drop(&mut self) {
        self.erase();
    }
}
impl Witness {
    fn erase(&mut self) {
        unsafe {
            core::ptr::write_volatile(&mut self.graph, 0);
        }
        wipe(&mut self.nodes);
        wipe(&mut self.compute);
    }
}
impl Drop for Witness {
    fn drop(&mut self) {
        self.erase();
    }
}
impl Drop for RecordingAdmission {
    fn drop(&mut self) {
        unsafe {
            core::ptr::write_volatile(&mut self.graph, 0);
            core::ptr::write_volatile(&mut self.id, 0);
            core::ptr::write_volatile(&mut self.family, 0);
        }
        wipe(&mut self.nodes);
        wipe(&mut self.compute);
    }
}

#[cfg(test)]
mod tests {
    use super::super::transient_io::cohort_scope_test_pair;
    use super::*;
    fn pending(scope: &NativeExpertBatchScope) -> Book {
        // Only exercise the private span state machine here. Production begin
        // additionally requires genuine grammar.finish and active IO admission.
        Book {
            pending: Some(Pending {
                stamp: scope.cohort_stamp(),
                graph: 17,
                first: 3,
                launches: 6,
            }),
            closed: None,
        }
    }
    #[test]
    fn empty_and_incomplete_grammar_never_supply_selection_evidence() {
        let (_, scope) = cohort_scope_test_pair();
        let mut book = Book::default();
        assert!(book.witness(17, 3, 2, 0).unwrap().is_none());
        assert!(
            book.begin(&scope, &NativeStrataExpertGrammar::default(), 17, 3)
                .is_err()
        );
        assert!(pending(&scope).witness(17, 12, 8, 6).is_err());
    }
    #[test]
    fn exact_scope_population_seals_and_binds_each_current_snapshot() {
        let (_, scope) = cohort_scope_test_pair();
        let mut book = pending(&scope);
        book.finish(&scope, 17, 12, 6, true).unwrap();
        assert_eq!(book.closed_range(), Some(3..12));
        // Original append-only commands may follow the authenticated batch;
        // the temporary witness binds their ACTUAL complete population too.
        let witness = book.witness(17, 15, 9, 6).unwrap().unwrap();
        assert!(witness.rolling(0, false, false, 15, 9));
        for (lane, named, profile, nodes, compute) in [
            (1, false, false, 15, 9),
            (2, false, false, 15, 9),
            (0, true, false, 15, 9),
            (0, false, true, 15, 9),
            (0, false, false, 16, 9),
            (0, false, false, 15, 10),
        ] {
            assert!(!witness.rolling(lane, named, profile, nodes, compute));
        }
        assert!(book.witness(18, 15, 9, 6).is_err());
        assert!(book.witness(17, 11, 9, 6).is_err());
        assert!(book.witness(17, 15, 9, 5).is_err());
        assert!(book.witness(17, 15, 9, 7).is_err());
        assert!(book.witness(17, 15, 5, 6).is_err());
        assert!(book.witness(17, 32769, 9, 6).is_err());
        let retained = witness.recording(42, 0);
        assert!(retained.matches(17, 42, 0, 15, 9));
        for (graph, id, family, nodes, compute) in [
            (18, 42, 0, 15, 9),
            (17, 43, 0, 15, 9),
            (17, 42, 1, 15, 9),
            (17, 42, 0, 16, 9),
            (17, 42, 0, 15, 10),
        ] {
            assert!(!retained.matches(graph, id, family, nodes, compute));
        }
        book.clear();
        assert!(book.closed_range().is_none());
        assert!(book.witness(17, 15, 9, 0).unwrap().is_none());
    }
    #[test]
    fn stale_foreign_partial_and_failed_batches_clear_all_evidence() {
        let (old, current) = cohort_scope_test_pair();
        for (graph, end, compute) in [
            (18, 12, 6),
            (17, 3, 6),
            (17, 12, 5),
            (17, 12, 7),
            (17, 32769, 6),
        ] {
            let mut book = pending(&current);
            assert!(book.finish(&current, graph, end, compute, true).is_err());
            assert!(book.pending_first().is_none() && book.closed_range().is_none());
        }
        let (_, foreign) = cohort_scope_test_pair();
        for wrong in [&old, &foreign] {
            let mut book = pending(&current);
            assert!(book.finish(wrong, 17, 12, 6, true).is_err());
            assert!(book.closed_range().is_none());
        }
        let mut book = pending(&current);
        book.finish(&current, 17, 12, 6, true).unwrap();
        book.finish(&current, 17, 12, 6, false).unwrap();
        assert!(book.closed_range().is_none());
    }
    #[test]
    fn numeric_evidence_is_erased_in_place_and_reservation_is_bounded() {
        let (_, scope) = cohort_scope_test_pair();
        let mut p = Pending {
            stamp: scope.cohort_stamp(),
            graph: 17,
            first: 3,
            launches: 6,
        };
        p.erase();
        assert_eq!((p.graph, p.first, p.launches), (0, 0, 0));
        let mut w = Witness {
            graph: 17,
            nodes: 12,
            compute: 6,
        };
        w.erase();
        assert_eq!((w.nodes, w.compute), (0, 0));
        assert!(Book::metadata_bytes() <= 1024);
    }
}
