//! Fixed-public, position0/count1 first-divergence diagnostics only. Each
//! boundary uses fresh, serialized actual scalar/native owners and normal full
//! acceptance. No state import, extra MAIN, math change or numeric tolerance.
//! Unarmed collectors do nothing. Parent owns CLI/protection/device evidence.
use super::*;
use crate::{executor::RESIDENT_RUNTIME_IO_MAX, resident_gr::batch::full_private};
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::{GgufFile, Qwen35Tokenizer, TokenizerLimits, model::Qwen4Exp};
use serde_json::{Value, json};
use std::mem::size_of;

const CHANNELS: usize = 10240;
const VALUE_WORDS: usize = 2560;
const MAX_BATCH: usize = 8;
const STAGE_MAP_MAX: usize = 524288;
const FAULT_MAP: usize = 32768;
const HOST_CAP: usize = 1024 * 1024;
const META_CAP: usize = 4096;
const HOST_BOUND: usize = 2 * CHANNELS * 4 + STAGE_MAP_MAX + META_CAP;
const EXTRA_AUDITED_RUNTIME: u64 = 2 * (STAGE_MAP_MAX + FAULT_MAP) as u64;

fn error() -> Error {
    Error::InvalidFormat("public numeric boundary fixture rejected")
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Boundary {
    Input,
    ProjectedKey,
    ProjectedValue,
    Injected,
}
impl Boundary {
    const ALL: [Self; 4] = [
        Self::Input,
        Self::ProjectedKey,
        Self::ProjectedValue,
        Self::Injected,
    ];
    fn width(self) -> usize {
        if self == Self::ProjectedValue {
            VALUE_WORDS
        } else {
            CHANNELS
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Input => "layer0_combined_ple_input",
            Self::ProjectedKey => "ple_projected_key",
            Self::ProjectedValue => "ple_projected_value",
            Self::Injected => "ple_injected_residual",
        }
    }
}
struct Snapshot {
    words: Zeroizing<Vec<u32>>,
    width: usize,
    #[cfg(test)]
    wipe_witness: Option<Arc<AtomicBool>>,
}
impl Snapshot {
    fn zero(width: usize) -> Result<Self> {
        if ![CHANNELS, VALUE_WORDS].contains(&width) {
            return Err(error());
        }
        // Guard exists before reserve/fill. Full actual capacity is initialized
        // and never grows; even inactive retained words are wiped on Drop.
        let mut words = Zeroizing::new(Vec::new());
        words.try_reserve_exact(CHANNELS).map_err(|_| error())?;
        if words.capacity() != CHANNELS {
            return Err(error());
        }
        words.resize(CHANNELS, 0);
        Ok(Self {
            words,
            width,
            #[cfg(test)]
            wipe_witness: None,
        })
    }
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        self.words.as_mut_slice().zeroize();
        #[cfg(test)]
        if let Some(witness) = &self.wipe_witness {
            witness.store(self.words.iter().all(|x| *x == 0), Ordering::Release);
        }
        self.words.zeroize();
        self.width.zeroize();
    }
}
struct Armed {
    boundary: Boundary,
    snapshot: Snapshot,
    captured: bool,
}
#[derive(Default)]
pub(super) struct Collector {
    armed: Option<Armed>,
}
impl Collector {
    pub(super) fn is_unarmed(&self) -> bool {
        self.armed.is_none()
    }
    fn arm(&mut self, boundary: Boundary) -> Result<()> {
        if self.armed.is_some() {
            return Err(error());
        }
        self.armed = Some(Armed {
            boundary,
            snapshot: Snapshot::zero(boundary.width())?,
            captured: false,
        });
        Ok(())
    }
    fn matching(&self, boundary: Boundary) -> bool {
        self.armed.as_ref().is_some_and(|a| a.boundary == boundary)
    }
    fn preflight(&self, logical: usize, start: usize, count: usize) -> Result<()> {
        let a = self.armed.as_ref().ok_or(error())?;
        let width = a.boundary.width();
        if a.captured || start != 0 || count != 1 || ![width, MAX_BATCH * width].contains(&logical)
        {
            return Err(error());
        }
        Ok(())
    }
    fn take(&mut self) -> Result<Snapshot> {
        if self.armed.as_ref().is_none_or(|a| !a.captured) {
            return Err(error());
        }
        Ok(self.armed.take().ok_or(error())?.snapshot)
    }
    /// Default path returns before any client/arena/lock action. The armed lab
    /// path completes normal inference maps before bounded audited cleanup;
    /// admission credits the same stream only after the callback has returned.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn observe(
        &mut self,
        boundary: Boundary,
        arena: &PrivateArena,
        admission: &ResidentAdmission,
        target: BufferId,
        fault: BufferId,
        logical: usize,
        start: usize,
        count: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !self.matching(boundary) {
            return Ok(());
        }
        let retired = admission.retired_context();
        let poison = admission.sequence_poison();
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.preflight(logical, start, count)?;
            health(admission, arena, cancel)?;
            let ids = Zeroizing::new([target, fault]);
            if ids[0] == ids[1] {
                return Err(error());
            }
            let stage = full_private(arena, ids[0], logical)?;
            let fault = full_private(arena, ids[1], 1)?;
            let client = arena.client().map_err(u::fault)?;
            let own = arena.layout(ids[0]).map_err(u::fault)?.owned_bytes;
            if own > STAGE_MAP_MAX
                || fault.size() != FAULT_MAP as u64
                || stage.stream != fault.stream
            {
                return Err(error());
            }
            // Two stack-held resources, no private dynamically grown metadata.
            let sr = client.get_resource(stage.clone()).map_err(u::fault)?;
            let fr = client.get_resource(fault.clone()).map_err(u::fault)?;
            let s = sr.resource();
            let f = fr.resource();
            if s.offset != 0
                || s.size != own as u64
                || s.buffer.size() != s.size
                || f.offset != 0
                || f.size != FAULT_MAP as u64
                || f.buffer.size() != f.size
                || s.buffer == f.buffer
            {
                return Err(error());
            }
            let armed = self.armed.as_mut().ok_or(error())?;
            admission.readback(|| {
                // No ledger callback/reentry here. Both raw leases drop before
                // audit, then admission performs its normal forced checkpoint.
                capture_packets(
                    || client.read_one(fault).map(Bytes).map_err(u::fault),
                    || client.read_one(stage).map(Bytes).map_err(u::fault),
                    &mut armed.snapshot,
                    own,
                    || {
                        if cancel.load(Ordering::Acquire)
                            || retired.load(Ordering::Acquire)
                            || poison.load(Ordering::Acquire)
                        {
                            return Err(error());
                        }
                        let receipt =
                            cubecl::future::block_on(client.sanitize_transient_io_audited())
                                .map_err(u::fault)?;
                        audit_receipt(&receipt)
                    },
                )
            })?;
            health(admission, arena, cancel)?;
            armed.captured = true;
            Ok(())
        }));
        match attempt {
            Ok(Ok(())) => Ok(()),
            _ => {
                // Never call admission.poison() while its ledger lock is held.
                poison.store(true, Ordering::Release);
                retired.store(true, Ordering::Release);
                Err(error())
            }
        }
    }
}
fn health(admission: &ResidentAdmission, arena: &PrivateArena, cancel: &AtomicBool) -> Result<()> {
    u::cancel(cancel)?;
    if !admission.transient_io()
        || admission.retired_context().load(Ordering::Acquire)
        || admission.sequence_poison().load(Ordering::Acquire)
    {
        return Err(error());
    }
    admission.check_client(arena.client().map_err(u::fault)?)
}
struct Bytes(cubecl::bytes::Bytes);
impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn capture_packets<L: AsRef<[u8]>>(
    fault: impl FnOnce() -> Result<L>,
    stage: impl FnOnce() -> Result<L>,
    snapshot: &mut Snapshot,
    owned: usize,
    audit: impl FnOnce() -> Result<()>,
) -> Result<()> {
    {
        let lease = fault()?;
        if lease.as_ref().len() != FAULT_MAP || lease.as_ref().iter().any(|x| *x != 0) {
            return Err(error());
        }
    }
    {
        let lease = stage()?;
        decode_stage(lease.as_ref(), snapshot, owned)?;
    }
    audit()
}
fn decode_stage(bytes: &[u8], snapshot: &mut Snapshot, owned: usize) -> Result<()> {
    if bytes.len() != owned
        || owned > STAGE_MAP_MAX
        || !owned.is_multiple_of(4)
        || owned < snapshot.width * 4
        || snapshot.words.len() != CHANNELS
        || snapshot.words.capacity() != CHANNELS
    {
        return Err(error());
    }
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let bits = u32::from_le_bytes(chunk.try_into().map_err(|_| error())?);
        if i < snapshot.width {
            if !f32::from_bits(bits).is_finite() {
                return Err(error());
            }
            snapshot.words[i] = bits;
        } else if bits != 0 {
            return Err(error());
        }
    }
    Ok(())
}
fn audit_receipt(r: &TransientIoReceipt) -> Result<()> {
    if r.upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .is_none_or(|n| n == 0 || n != r.inspected_bytes)
        || r.inspected_buffers == 0
    {
        return Err(error());
    }
    Ok(())
}
fn envelope(original: u64) -> Result<u64> {
    let combined = original
        .checked_add(EXTRA_AUDITED_RUNTIME)
        .ok_or(Error::ArithmeticOverflow)?;
    if combined > RESIDENT_RUNTIME_IO_MAX
        || HOST_BOUND > HOST_CAP
        || size_of::<Collector>() * 2 + size_of::<Snapshot>() * 2 + size_of::<Report>() * 4
            > META_CAP
    {
        return Err(error());
    }
    Ok(combined)
}
#[derive(Default)]
struct Difference {
    mismatches: usize,
    max_abs: f64,
}
fn compare(a: &Snapshot, b: &Snapshot) -> Result<Difference> {
    if a.width != b.width
        || a.words.len() != CHANNELS
        || b.words.len() != CHANNELS
        || a.words.capacity() != CHANNELS
        || b.words.capacity() != CHANNELS
        || a.words[a.width..]
            .iter()
            .chain(&b.words[b.width..])
            .any(|x| *x != 0)
    {
        return Err(error());
    }
    let mut difference = Difference::default();
    for (&a, &b) in a.words[..a.width].iter().zip(&b.words[..b.width]) {
        let x = f32::from_bits(a);
        let y = f32::from_bits(b);
        if !x.is_finite() || !y.is_finite() {
            return Err(error());
        }
        difference.mismatches += usize::from(a != b);
        difference.max_abs = difference.max_abs.max((f64::from(x) - f64::from(y)).abs());
    }
    Ok(difference)
}
struct Report {
    boundary: Boundary,
    difference: Difference,
    scalar: lifecycle_probe::Audit,
    native: lifecycle_probe::Audit,
}
impl Report {
    fn value(&self) -> Value {
        json!({"boundary":self.boundary.label(),"logical_values":self.boundary.width(),
            "scalar_owned_map_bytes":BufferLayout::new(self.boundary.width()).expect("fixed layout").owned_bytes,
            "native_owned_map_bytes":BufferLayout::new(MAX_BATCH*self.boundary.width()).expect("fixed layout").owned_bytes,
            "bit_mismatches":self.difference.mismatches,"max_absolute":self.difference.max_abs,
            "bit_equal":self.difference.mismatches==0,"scalar_cleanup":self.scalar.value(),
            "native_cleanup":self.native.value()})
    }
}
/// Typed entry only. Same configured actual client/stream policy as lifecycle
/// fixture; caller owns fixed-public CLI, device protection and receipt writes.
/// Successful return proves all eight explicit owner/executor cleanups, never
/// whole-model numeric acceptance or production/private-worker acceptance.
pub(crate) fn run(file: &GgufFile, config: lifecycle_probe::RunConfig) -> Result<Value> {
    let retired = config.retired.clone();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_inner(file, config))) {
        Ok(Ok(value)) => Ok(value),
        _ => {
            retired.store(true, Ordering::Release);
            Err(error())
        }
    }
}
fn run_inner(file: &GgufFile, config: lifecycle_probe::RunConfig) -> Result<Value> {
    let fixture = lifecycle_probe::Fixture::new(file, config)?;
    let model = Qwen4Exp::bind(file)?;
    let graph = model.bound_graph();
    let key = graph.tensor("blk.1.ple_key.weight")?.info();
    let value = graph.tensor("blk.1.ple_value.weight")?.info();
    if key.row_count() != CHANNELS as u64
        || key.row_len() != VALUE_WORDS as u64
        || value.row_count() != VALUE_WORDS as u64
        || value.row_len() != VALUE_WORDS as u64
        || graph.geometry().residual.width * graph.geometry().residual.streams != CHANNELS
    {
        return Err(error());
    }
    let original_runtime = fixture.envelope(graph)?;
    let combined = envelope(original_runtime)?; // Before private allocation/fill.
    let tokenizer = Qwen35Tokenizer::from_gguf(file, TokenizerLimits::default())?;
    let tokenized = Zeroizing::new(tokenizer.encode("!", true)?);
    if tokenized.len() != 1 || tokenized[0] >= 248320 {
        return Err(error());
    }
    let token = Zeroizing::new([tokenized[0]]);
    drop(tokenized);
    let mut reports: [Option<Report>; 4] = std::array::from_fn(|_| None);
    for (i, boundary) in Boundary::ALL.into_iter().enumerate() {
        let (reference, scalar) = fixture.owner(graph, 1, false, 1, |m, e, cancel| {
            m.numeric_boundary.arm(boundary)?;
            let logits = m.step(e, token[0], cancel)?;
            drop(logits); // Original guarded full-vocab host budget ends here.
            lifecycle_probe::accepted(m, 1)?;
            m.numeric_boundary.take()
        })?;
        let (candidate, native) = fixture.owner(graph, 1, true, 0, |m, e, cancel| {
            m.numeric_boundary.arm(boundary)?;
            let logits = m.prefill_chunk(e, &*token, batch::Purpose::LastLogits, cancel)?;
            drop(logits);
            lifecycle_probe::accepted(m, 1)?;
            m.numeric_boundary.take()
        })?;
        // Both actual owners are explicitly drained/audited/dropped by owner()
        // before any numeric acceptance/aggregate report, including mismatch.
        let difference = compare(&reference, &candidate)?;
        reports[i] = Some(Report {
            boundary,
            difference,
            scalar,
            native,
        });
        drop(reference);
        drop(candidate);
    }
    Ok(
        json!({"fixture":"public_native_numeric_boundaries_v1","position":0,"count":1,
        "pairs":4,"owners_serialized":8,"extra_main_bytes":0,"guarded_host_bound":HOST_BOUND,
        "guarded_host_cap":HOST_CAP,"extra_audited_runtime_bytes":EXTRA_AUDITED_RUNTIME,
        "original_runtime_bytes":original_runtime,"combined_runtime_bytes":combined,
        "runtime_hard_cap":RESIDENT_RUNTIME_IO_MAX,"normal_family_acceptance":true,
        "explicit_full_cleanup":true,"whole_model_numeric_accepted":false,"performance_accepted":false,
        "boundaries":reports.iter().map(|r|r.as_ref().expect("all pairs complete").value()).collect::<Vec<_>>() }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc};
    struct Lease {
        bytes: Vec<u8>,
        events: Rc<RefCell<Vec<&'static str>>>,
    }
    impl AsRef<[u8]> for Lease {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            self.bytes.zeroize();
            self.events.borrow_mut().push("drop");
        }
    }
    fn lease(bytes: Vec<u8>, e: &Rc<RefCell<Vec<&'static str>>>) -> Lease {
        e.borrow_mut().push("read");
        Lease {
            bytes,
            events: e.clone(),
        }
    }
    #[test]
    fn default_unarmed_one_shot_and_exact_packet() {
        let mut c = Collector::default();
        assert!(!c.matching(Boundary::ProjectedKey));
        assert!(c.take().is_err());
        c.arm(Boundary::ProjectedKey).unwrap();
        assert!(c.arm(Boundary::ProjectedKey).is_err());
        assert!(c.preflight(CHANNELS, 0, 1).is_ok());
        assert!(c.preflight(MAX_BATCH * CHANNELS, 0, 1).is_ok());
        for packet in [(VALUE_WORDS, 0, 1), (CHANNELS, 1, 1), (CHANNELS, 0, 3)] {
            assert!(c.preflight(packet.0, packet.1, packet.2).is_err());
        }
        c.armed.as_mut().unwrap().captured = true;
        assert!(c.preflight(CHANNELS, 0, 1).is_err());
        assert!(c.take().is_ok());
    }
    #[test]
    fn complete_maps_drop_before_audit_and_error_skips_stage() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut s = Snapshot::zero(VALUE_WORDS).unwrap();
        capture_packets(
            || Ok(lease(vec![0; FAULT_MAP], &events)),
            || Ok(lease(vec![0; FAULT_MAP], &events)),
            &mut s,
            FAULT_MAP,
            || {
                events.borrow_mut().push("audit");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(*events.borrow(), ["read", "drop", "read", "drop", "audit"]);
        events.borrow_mut().clear();
        let mut bad = vec![0; FAULT_MAP];
        bad[0] = 1;
        assert!(
            capture_packets(
                || Ok(lease(bad, &events)),
                || { panic!("must skip stage after fault") },
                &mut s,
                FAULT_MAP,
                || { panic!("must skip audit after fault") }
            )
            .is_err()
        );
        assert_eq!(*events.borrow(), ["read", "drop"]);
    }
    #[test]
    fn full_extent_finite_inactive_and_padding_are_required() {
        let mut s = Snapshot::zero(VALUE_WORDS).unwrap();
        let mut bytes = vec![0; 131072];
        bytes[..4].copy_from_slice(&(-0f32).to_bits().to_le_bytes());
        decode_stage(&bytes, &mut s, 131072).unwrap();
        assert_eq!(s.words[0], (-0f32).to_bits());
        assert!(decode_stage(&bytes[..32768], &mut s, 131072).is_err());
        for at in [VALUE_WORDS * 4, MAX_BATCH * VALUE_WORDS * 4, 131068] {
            bytes[at] = 1;
            assert!(decode_stage(&bytes, &mut s, 131072).is_err());
            bytes[at] = 0;
        }
        bytes[..4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(decode_stage(&bytes, &mut s, 131072).is_err());
    }
    #[test]
    fn failed_audit_cannot_publish_and_full_snapshot_wipes_on_unwind() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let witness = Arc::new(AtomicBool::new(false));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut s = Snapshot::zero(CHANNELS).unwrap();
            s.wipe_witness = Some(witness.clone());
            s.words.fill(0x12345678);
            assert!(
                capture_packets(
                    || Ok(lease(vec![0; FAULT_MAP], &events)),
                    || Ok(lease(vec![0; 65536], &events)),
                    &mut s,
                    65536,
                    || Err(error())
                )
                .is_err()
            );
            panic!("injected outer failure");
        }));
        assert!(result.is_err());
        assert!(witness.load(Ordering::Acquire));
    }
    #[test]
    fn bitwise_comparison_has_no_tolerance_and_budget_is_checked() {
        let mut a = Snapshot::zero(VALUE_WORDS).unwrap();
        let mut b = Snapshot::zero(VALUE_WORDS).unwrap();
        a.words[0] = 1f32.to_bits();
        b.words[0] = (1f32 + f32::EPSILON).to_bits();
        b.words[1] = (-0f32).to_bits();
        let d = compare(&a, &b).unwrap();
        assert_eq!(d.mismatches, 2);
        assert!(d.max_abs > 0.);
        assert_eq!(HOST_BOUND, 610304);
        assert_eq!(EXTRA_AUDITED_RUNTIME, 1114112);
        assert_eq!(envelope(64090112).unwrap(), 65204224);
        assert!(envelope(RESIDENT_RUNTIME_IO_MAX).is_err());
        assert!(envelope(u64::MAX).is_err());
    }
}
