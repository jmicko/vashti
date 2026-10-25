//! Selection evidence for a genuine, completely validated expert cohort batch.
//! This never grants execution, source, reuse, replay or completion authority.
use super::transient_io::{NativeExpertBatchScope, NativeExpertCohortStamp, fault};
use cubecl_core::server::ServerError;
use cubecl_runtime::server::native_expert_grammar::{NativeStrataExpertGrammar, NativeStrataExpertGrammarCounts};

// Original cohorts retain six to eight dispatches. Quant cohorts admit ten;
// distinct large cohorts admit twelve with all original prefix/tail families.
fn cohort_launch_population(c: &NativeStrataExpertGrammarCounts) -> bool {
    if c.large_cohort_groups>512 || c.large_cohort_batches>512 || c.large_cohort_launches>512*12
        || (!cfg!(feature="kuno-large-cohorts") && (c.large_cohort_groups!=0 || c.large_cohort_batches!=0 || c.large_cohort_launches!=0))
        || c.large_cohort_groups<c.large_cohort_batches
        || c.large_cohort_groups>8*c.large_cohort_batches
        || c.large_cohort_launches<6*c.large_cohort_batches
        || c.large_cohort_launches>12*c.large_cohort_batches
        || c.large_cohort_launches%2!=0 {return false;}
    if c.regular_cohort_groups>512 || c.regular_cohort_batches>512 || c.regular_cohort_launches>512*6
        || (!cfg!(feature="kuno-regular-cohorts") && (c.regular_cohort_groups!=0 || c.regular_cohort_batches!=0 || c.regular_cohort_launches!=0))
        || c.regular_cohort_groups<c.regular_cohort_batches
        || c.regular_cohort_groups>8*c.regular_cohort_batches
        || c.regular_cohort_launches!=6*c.regular_cohort_batches
        || c.regular_cohort_groups.checked_add(c.quant_cohort_groups).and_then(|n|n.checked_add(c.large_cohort_groups)).is_none_or(|n|n>c.cohort_groups)
        || c.regular_cohort_batches.checked_add(c.quant_cohort_batches).and_then(|n|n.checked_add(c.large_cohort_batches)).is_none_or(|n|n>c.cohort_batches)
        || c.regular_cohort_launches.checked_add(c.quant_cohort_launches).and_then(|n|n.checked_add(c.large_cohort_launches)).is_none_or(|n|n>c.cohort_launches) {
        return false;
    }
    if c.quant_cohort_groups > c.cohort_groups
        || c.quant_cohort_batches > c.cohort_batches
        || c.quant_cohort_launches > c.cohort_launches
        || (!cfg!(feature = "kuno-small-quantized-matrix") && c.quant_cohort_batches != 0)
        || c.quant_cohort_groups < c.quant_cohort_batches
        || c.quant_cohort_groups > c.quant_cohort_batches * 8
        || c.quant_cohort_launches < c.quant_cohort_batches * 6
        || c.quant_cohort_launches > c.quant_cohort_batches * 10 {
        return false;
    }
    let old_batches = c.cohort_batches - c.quant_cohort_batches - c.regular_cohort_batches - c.large_cohort_batches;
    let old_groups = c.cohort_groups - c.quant_cohort_groups - c.regular_cohort_groups - c.large_cohort_groups;
    let old_launches = c.cohort_launches - c.quant_cohort_launches - c.regular_cohort_launches - c.large_cohort_launches;
    old_groups >= old_batches && old_groups <= old_batches * 8
        && old_launches >= old_batches * 6 && old_launches <= old_batches * 8
}

/// Existing compiled-public fixed-layer profile only. These aggregate labels
/// identify actual scoped grammar, recording and original driver-pool events.
pub const NATIVE_COHORT_SELECTION_LABELS: [&str; 25] = [
    "grammar_admissions",
    "admitted_cohort_groups",
    "admitted_cohort_batches",
    "admitted_cohort_launches",
    "admitted_physical_launches",
    "witnessed_recordings",
    "witnessed_nodes",
    "witnessed_compute",
    "selected_rolling_recordings",
    "selected_rolling_nodes",
    "selected_rolling_compute",
    "selected_rolling_below1024",
    "submitted_rolling_recordings",
    "submitted_rolling_nodes",
    "submitted_rolling_compute",
    "submitted_rolling_below1024",
    "retired_pool_recordings",
    "retired_pool_compute",
    "adopted_pool_recordings",
    "adopted_pool_compute",
    "adopted_pool_below1024",
    "reused_recording_count",
    "admitted_large_cohort_groups",
    "admitted_large_cohort_batches",
    "admitted_large_cohort_launches",
];
static PUBLIC_TOTALS: [core::sync::atomic::AtomicU64; 25] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 25];
pub fn native_strata_cohort_selection_stats() -> [u64; 25] {
    core::array::from_fn(|i| PUBLIC_TOTALS[i].load(core::sync::atomic::Ordering::Relaxed))
}

// The complete grammar is the sole source of admission populations. Preserve
// existing aggregate indices and append the actual large subset separately.
fn admission_credits(c: &NativeStrataExpertGrammarCounts) -> [(usize,u64);8] {
    [(0,1),(1,c.cohort_groups as u64),(2,c.cohort_batches as u64),
        (3,c.cohort_launches as u64),(4,c.launches as u64),
        (22,c.large_cohort_groups as u64),(23,c.large_cohort_batches as u64),
        (24,c.large_cohort_launches as u64)]
}
/// Never collect private request shapes. The getter remains readable after the
/// original public sampler closes, so its existing collector can compute deltas.
pub(super) fn credit(values: &[(usize, u64)]) {
    if !super::strata_graph::cohort_public_profile_active() {
        return;
    }
    for &(index, amount) in values {
        PUBLIC_TOTALS[index].fetch_add(amount, core::sync::atomic::Ordering::Relaxed);
    }
}

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
            + core::mem::size_of::<Option<Witness>>()
            + core::mem::size_of::<Option<RecordingAdmission>>()
    }
    /// Only the original grammar can provide these counts. The caller must
    /// first validate this exact scope against its real active TransientIo.
    pub(super) fn begin(
        &mut self,
        stamp: NativeExpertCohortStamp,
        grammar: &NativeStrataExpertGrammar,
        graph: u64,
        first: usize,
    ) -> Result<(), ServerError> {
        let counts = grammar.finish()?;
        if counts.cohort_groups == 0 {
            if counts.cohort_launches != 0 || counts.cohort_batches != 0
                || counts.quant_cohort_groups != 0 || counts.quant_cohort_launches != 0
                || counts.quant_cohort_batches != 0 {
                return Err(fault("native cohort empty grammar evidence"));
            }
            if counts.regular_cohort_groups!=0 || counts.regular_cohort_batches!=0 || counts.regular_cohort_launches!=0 {
                return Err(fault("native cohort empty regular grammar evidence"));
            }
            if counts.large_cohort_groups!=0 || counts.large_cohort_batches!=0 || counts.large_cohort_launches!=0 {return Err(fault("native cohort empty large grammar evidence"));}
            return Ok(());
        }
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
            || !cohort_launch_population(&counts)
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
        credit(&admission_credits(&counts));
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

#[cfg(all(test, feature = "kuno-small-quantized-matrix"))]
mod quant_population_tests {
    use super::*;
    #[test]
    fn packed_ten_launch_bound_never_expands_the_original_eight_launch_partition() {
        let mut c = NativeStrataExpertGrammarCounts {
            cohort_groups: 2, cohort_batches: 1, cohort_launches: 8,
            ..Default::default()
        };
        assert!(cohort_launch_population(&c));
        c.cohort_launches = 9;
        assert!(!cohort_launch_population(&c));
        c.cohort_groups = 3; c.cohort_launches = 10;
        c.quant_cohort_groups = 3; c.quant_cohort_batches = 1; c.quant_cohort_launches = 10;
        assert!(cohort_launch_population(&c));
        for bad in [5,11] {
            c.cohort_launches = bad; c.quant_cohort_launches = bad;
            assert!(!cohort_launch_population(&c));
        }
        // Mixed old/new tapes keep two independent closed populations.
        c.cohort_groups = 5; c.cohort_batches = 2; c.cohort_launches = 18;
        c.quant_cohort_launches = 10;
        assert!(cohort_launch_population(&c));
        c.cohort_launches = 19;
        assert!(!cohort_launch_population(&c));
        c.cohort_launches = 18; c.quant_cohort_launches = 9;
        assert!(!cohort_launch_population(&c));
        c.quant_cohort_launches = 10; c.quant_cohort_groups = 5;
        assert!(!cohort_launch_population(&c));
    }
}

#[cfg(test)]
mod tests {
    use super::super::transient_io::{cohort_scope_test_pair, cohort_stamp_test};
    use super::*;
    /// Actual closed grammar and actual Handle/Binding views. This fixture is
    /// metadata only: it never creates device, request or execution authority.
    fn completed_cohort_grammar() -> NativeStrataExpertGrammar {
        use cubecl_common::stream_id::StreamId;
        use cubecl_runtime::server::{CubeCount, Handle, KernelArguments, MetadataBindingInfo};
        fn arguments(h: Vec<Handle>, w: &[u32; 48], matrix: bool) -> KernelArguments {
            let mut words = w.to_vec();
            if matrix {
                let lengths: Vec<u32> = h
                    .iter()
                    .enumerate()
                    .map(|(i, h)| (h.size_in_used() / if i < 2 { 8 } else { 4 }) as u32)
                    .collect();
                words.extend(&lengths);
                words.extend(&lengths);
            }
            let packed = words
                .chunks_exact(2)
                .map(|v| {
                    let mut bytes = [0; 8];
                    bytes[..4].copy_from_slice(&v[0].to_ne_bytes());
                    bytes[4..].copy_from_slice(&v[1].to_ne_bytes());
                    u64::from_ne_bytes(bytes)
                })
                .collect();
            KernelArguments::new()
                .with_buffers(h.into_iter().map(Handle::binding).collect())
                .with_info(MetadataBindingInfo::new(
                    packed,
                    if matrix { 30 } else { 0 },
                ))
        }
        let owner = |n| Handle::new(StreamId::current(), n);
        let working = 80 * 1024 * 1024;
        let parent = owner(working);
        let x = parent.clone().offset_end(working - 20 * 1024 * 1024);
        let h = parent
            .offset_start(20 * 1024 * 1024)
            .offset_end(55 * 1024 * 1024);
        let image = owner(working);
        let raw = owner(32 * 1024 * 1024);
        let map = owner(65536 * 4);
        let input = owner(working);
        let gu = owner(4096 * 1280 * 4);
        let part = owner(4096 * 10 * 2560 * 4);
        let status = owner(131072 * 4);
        let fault = owner(4);
        let gate = owner(400);
        let up = owner(400);
        let down = owner(400);
        let table = owner(32768);
        let addresses = owner(32768);
        let mut w = [0; 48];
        w[0] = 32;
        w[1] = 2560;
        w[2] = 1280;
        w[6] = 100;
        w[7] = 100;
        w[8] = 300;
        w[9] = 1;
        w[10] = 3;
        w[16..20].copy_from_slice(&[0, 3, 0, 8]);
        let mut grammar = NativeStrataExpertGrammar::default();
        grammar
            .launch(
                "StrataExpertCohortGather",
                false,
                &CubeCount::Static(15, 1, 1),
                &arguments(
                    vec![map.clone(), input, x.clone(), fault.clone()],
                    &w,
                    false,
                ),
            )
            .unwrap();
        for down_role in [false, true] {
            w[1] = if down_role { 640 } else { 2560 };
            w[2] = if down_role { 2560 } else { 1280 };
            w[3] = 1;
            w[5] = if down_role { 2 } else { 0 };
            let mut owners = if down_role {
                vec![down.clone()]
            } else {
                vec![gate.clone(), up.clone()]
            };
            owners.extend([
                map.clone(),
                raw.clone(),
                image.clone(),
                fault.clone(),
                table.clone(),
                addresses.clone(),
            ]);
            let mut args = arguments(owners, &w, false);
            args.read_dependencies.push(gate.clone().binding());
            grammar
                .launch(
                    if down_role {
                        "StrataExpertCohortDownImage"
                    } else {
                        "StrataExpertCohortGuImage"
                    },
                    down_role,
                    &CubeCount::Static(if down_role { 200 } else { 400 }, 1, 1),
                    &args,
                )
                .unwrap();
        }
        for down_role in [false, true] {
            w[1] = if down_role { 640 } else { 2560 };
            w[2] = if down_role { 2560 } else { 1280 };
            w[3] = 0;
            w[5] = if down_role { 2 } else { 0 };
            w[11] = 8;
            grammar
                .launch(
                    "fixture::cohorts::expert_cohort::ExpertCohort<Runtime>",
                    false,
                    &CubeCount::Static(w[2] / 64, 1, 1),
                    &arguments(
                        vec![
                            if down_role { h.clone() } else { x.clone() },
                            image.clone(),
                            if down_role { part.clone() } else { gu.clone() },
                            map.clone(),
                            status.clone(),
                            fault.clone(),
                        ],
                        &w,
                        true,
                    ),
                )
                .unwrap();
            if !down_role {
                w[11] = 0;
                grammar
                    .launch(
                        "StrataExpertCohortActivate",
                        false,
                        &CubeCount::Static(4, 1, 1),
                        &arguments(
                            vec![
                                map.clone(),
                                gu.clone(),
                                h.clone(),
                                status.clone(),
                                fault.clone(),
                            ],
                            &w,
                            false,
                        ),
                    )
                    .unwrap();
            }
        }
        let counts = grammar.finish().unwrap();
        assert_eq!(
            (
                counts.launches,
                counts.logical_phases,
                counts.cohort_groups,
                counts.cohort_batches
            ),
            (6, 6, 1, 1)
        );
        grammar
    }
    fn pending(scope: &NativeExpertBatchScope) -> Book {
        // Only exercise the private span state machine here. Production begin
        // additionally requires genuine grammar.finish and active IO admission.
        Book {
            pending: Some(Pending {
                stamp: cohort_stamp_test(scope),
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
            book.begin(
                cohort_stamp_test(&scope),
                &NativeStrataExpertGrammar::default(),
                17,
                3
            )
            .is_err()
        );
        assert!(pending(&scope).witness(17, 12, 8, 6).is_err());
    }
    #[test]
    fn exact_scope_population_seals_and_binds_each_current_snapshot() {
        let (_, scope) = cohort_scope_test_pair();
        let grammar = completed_cohort_grammar();
        let mut book = Book::default();
        book.begin(cohort_stamp_test(&scope), &grammar, 17, 3)
            .unwrap();
        assert_eq!(book.pending_first(), Some(3));
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
            stamp: cohort_stamp_test(&scope),
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

#[cfg(test)]
mod regular_population_tests {
 use super::*;
 #[test]
 fn regular_population_is_a_separate_exact_six_launch_partition_with_no_old_or_quant_credit(){
  let mut c=NativeStrataExpertGrammarCounts{cohort_groups:8,cohort_batches:1,cohort_launches:6,regular_cohort_groups:8,regular_cohort_batches:1,regular_cohort_launches:6,..Default::default()};
  assert_eq!(cohort_launch_population(&c),cfg!(feature="kuno-regular-cohorts"));
  for groups in [0,9,usize::MAX]{c.regular_cohort_groups=groups;assert!(!cohort_launch_population(&c));}c.regular_cohort_groups=8;
  for launches in [0,5,7,usize::MAX]{c.regular_cohort_launches=launches;assert!(!cohort_launch_population(&c));}c.regular_cohort_launches=6;
  c.quant_cohort_groups=1;c.quant_cohort_batches=1;c.quant_cohort_launches=6;assert!(!cohort_launch_population(&c));
  c.quant_cohort_groups=0;c.quant_cohort_batches=0;c.quant_cohort_launches=0;
  c.regular_cohort_batches=usize::MAX;assert!(!cohort_launch_population(&c));
 }
}

#[cfg(all(test,feature="kuno-large-cohorts"))]
#[test]
fn large_cohort_population_is_distinct_bounded_and_does_not_count_regular_twice() {
    for launches in [6,8,10,12] {
        let mut c=NativeStrataExpertGrammarCounts{cohort_groups:8,cohort_batches:1,cohort_launches:launches,
            large_cohort_groups:8,large_cohort_batches:1,large_cohort_launches:launches,..Default::default()};
        assert!(cohort_launch_population(&c));
        assert_eq!(&admission_credits(&c)[5..],&[(22,8),(23,1),(24,launches as u64)]);
        c.regular_cohort_groups=8;c.regular_cohort_batches=1;c.regular_cohort_launches=6;
        assert!(!cohort_launch_population(&c));
        c.cohort_groups+=8;c.cohort_batches+=1;c.cohort_launches+=6;
        assert!(cohort_launch_population(&c));
    }
    for launches in [0,5,7,13,usize::MAX] {
        let c=NativeStrataExpertGrammarCounts{cohort_groups:8,cohort_batches:1,cohort_launches:launches,
            large_cohort_groups:8,large_cohort_batches:1,large_cohort_launches:launches,..Default::default()};
        assert!(!cohort_launch_population(&c));
    }
}
