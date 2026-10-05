//! Fixed-public actual whole-model lifecycle fixture. Source-only until the
//! parent runs protected devices; no default/request/CLI route is registered.
//! All owners and executors run serially, close explicitly, and are dropped
//! before the next original-model binding. No state import or fake cursor.
use super::{batch, lifecycle_hooks::Site, *};
use crate::executor::GpuExecutor;
use kuno_text::model::{Qwen4Exp, greedy};
use kuno_text::{GgufFile, Qwen35Tokenizer, TokenizerLimits};
use serde_json::{Value, json};
use std::mem::size_of;

const SCALAR_BYTES: u64 = 785_186_816;
const NATIVE_BYTES: u64 = 818_249_728;
const SCALAR_PAGES: u64 = 1286;
const NATIVE_PAGES: u64 = 1376;
const VOCAB: usize = 248_320;
const PROBE_HOST_CAP: usize = 4 * 1024 * 1024;
const PROBE_META_CAP: usize = 64 * 1024;
const MAX_REPORTS: usize = 32;
const LIVE_SOURCE_CAP: u64 = crate::executor::RESIDENT_IO_MAX;

fn error() -> Error {
    Error::InvalidFormat("public model lifecycle fixture rejected")
}

/// The caller must configure this physical stream ONCE with the existing
/// audited transient runtime policy before calling run. No device selection or
/// policy reset occurs here. Parent CLI owns protected limits and receipt I/O.
pub(crate) struct RunConfig {
    pub(crate) client: ComputeClient<WgpuRuntime>,
    pub(crate) limits: wgpu::Limits,
    pub(crate) vendor: u32,
    pub(crate) retired: Arc<AtomicBool>,
}
struct Fixture<'a> {
    config: RunConfig,
    file: &'a GgufFile,
    cache: usize,
    slots: usize,
    device: usize,
    native: batch::Config,
}
impl<'a> Fixture<'a> {
    fn new(file: &'a GgufFile, config: RunConfig) -> Result<Self> {
        if config.retired.load(Ordering::Acquire) {
            return Err(error());
        }
        let (cache, slots, device) = device_plan(config.vendor)?;
        check_settings()?;
        let native = batch::Config::new(&config.client, &config.limits, 8)?;
        Ok(Self {
            config,
            file,
            cache,
            slots,
            device,
            native,
        })
    }
    fn executor(&self) -> Result<GpuExecutor<'a>> {
        let mut e = GpuExecutor::new(
            self.file,
            self.config.client.clone(),
            self.cache,
            self.slots,
            self.device,
        )?;
        e.enable_transient_resident(self.config.retired.clone())?;
        Ok(e)
    }
    /// Actual metadata/limits/admission proof, before any private/bank pages.
    fn envelope(&self, graph: BoundGraph<'_, 'a>) -> Result<()> {
        graph.verify_unchanged()?;
        let requests = private_requests(graph)?;
        if requests.len() as u64 != SCALAR_PAGES || checked_sum(&requests)? != SCALAR_BYTES {
            return Err(error());
        }
        let e = self.executor()?;
        let plan = batch::Plan::new(
            graph,
            &self.config.client,
            &e,
            self.native.clone(),
            &requests,
        )?;
        plan.check_public(graph, PublicVectors::prewarm_page_bytes(graph)?)?;
        if requests.len() + plan.requests.len() != NATIVE_PAGES as usize
            || checked_sum(&plan.requests)? + SCALAR_BYTES != NATIVE_BYTES
            || 2 * VOCAB * size_of::<f32>() > PROBE_HOST_CAP
            || MAX_REPORTS * size_of::<CaseReport>() > PROBE_META_CAP
        {
            return Err(error());
        }
        crate::expert_banks::preflight_residency(graph, self.slots, self.cache, self.device)?;
        // This metadata-only executor owns no residency, bank or transient
        // allocations. Its ordinary public cache remains empty.
        drop(e);
        Ok(())
    }
    fn owner<T>(
        &self,
        graph: BoundGraph<'_, 'a>,
        context: usize,
        native: bool,
        scalar_calls: u64,
        operation: impl FnOnce(
            &mut ResidentModel<'_, 'a>,
            &mut GpuExecutor<'a>,
            &AtomicBool,
        ) -> Result<T>,
    ) -> Result<(T, Audit)> {
        let mut executor = self.executor()?;
        let mut model = match ResidentModel::new_for_lab_fixture(
            graph,
            self.config.client.clone(),
            context,
            &mut executor,
            true,
            native.then(|| self.native.clone()),
        ) {
            Ok(model) => model,
            Err(_) => {
                self.config.retired.store(true, Ordering::Release);
                let _ = executor.close_transient_resident();
                return Err(error());
            }
        };
        let expected = if native { NATIVE_BYTES } else { SCALAR_BYTES };
        let cancel = AtomicBool::new(false);
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if model.owned_bytes() as u64 != expected {
                return Err(error());
            }
            operation(&mut model, &mut executor, &cancel)
        }));
        // Unexpected failure is terminal physical retirement even if cleanup
        // later succeeds. Expected negative cases are accepted inside operation.
        if !matches!(&attempt, Ok(Ok(_))) {
            self.config.retired.store(true, Ordering::Release);
        }
        let close_model = model.close();
        let audit = model.public_owned_zero_audit_receipt();
        let close_runtime = executor.close_transient_resident();
        let clean = close_model.is_ok()
            && close_runtime.is_ok()
            && model.erased_after_failure()
            && executor.ordered_source_pending_bytes() == 0
            && !self.config.retired.load(Ordering::Acquire);
        if !clean {
            self.config.retired.store(true, Ordering::Release);
            return Err(error());
        }
        let audit = match audit
            .and_then(|(main, source)| validate_audit(main, source, native, scalar_calls).ok())
        {
            Some(audit) => audit,
            None => {
                self.config.retired.store(true, Ordering::Release);
                return Err(error());
            }
        };
        // Owners retain all real resource/ledger pins through explicit runtime
        // cleanup; then release before the next Fixture::owner invocation.
        drop(model);
        drop(executor);
        match attempt {
            Ok(Ok(value)) => Ok((value, audit)),
            _ => Err(error()),
        }
    }
}
fn checked_sum(pages: &[u64]) -> Result<u64> {
    pages.iter().try_fold(0u64, |n, p| {
        n.checked_add(*p).ok_or(Error::ArithmeticOverflow)
    })
}
fn device_plan(vendor: u32) -> Result<(usize, usize, usize)> {
    match vendor {
        0x8086 => Ok((8usize << 30, 64, 26usize << 30)),
        0x10de => Ok((6usize << 30, 16, 10usize << 30)),
        _ => Err(error()),
    }
}
fn check_settings() -> Result<()> {
    // Freeze the unchanged original scalar/cooperative reference path. These
    // optional experiments have independent budgets and are not this fixture.
    for key in [
        "KUNO_LAB_BATCH_EXPERTS",
        "KUNO_LAB_WORD_DOWN",
        "KUNO_LAB_BLOCK_MOE",
        "KUNO_LAB_SMALL_CPU",
        "KUNO_LAB_FUSED_MATRIX_FAULT",
        "KUNO_LAB_CPU_GDN",
        "KUNO_LAB_IO_TOKEN_CHECKPOINT",
        "KUNO_LAB_ORDERED_PUBLIC_UPLOAD",
        "KUNO_LAB_INFO_SLAB_BYTES",
    ] {
        if std::env::var_os(key).is_some_and(|v| v != "0") {
            return Err(error());
        }
    }
    for key in [
        "KUNO_LAB_BLOCK_KERNEL",
        "KUNO_LAB_EXPERT_SLOTS",
        "KUNO_LAB_STATIC_CACHE_BYTES",
    ] {
        if std::env::var_os(key).is_some() {
            return Err(error());
        }
    }
    Ok(())
}
#[derive(Clone, Copy)]
struct Audit {
    main_bytes: u64,
    main_buffers: u64,
    source_bytes: u64,
    source_buffers: u64,
}
impl Audit {
    fn value(self) -> Value {
        json!({"main_bytes":self.main_bytes,"main_buffers":self.main_buffers,
            "source_bytes":self.source_bytes,"source_buffers":self.source_buffers})
    }
}
fn validate_audit(
    main: crate::private_arena::OwnedZeroAudit,
    source: crate::private_arena::OwnedZeroAudit,
    native: bool,
    scalar_calls: u64,
) -> Result<Audit> {
    let (bytes, pages) = if native {
        (NATIVE_BYTES, NATIVE_PAGES)
    } else {
        (SCALAR_BYTES, SCALAR_PAGES)
    };
    // SOURCE counts are cumulative across resets. They may exceed the LIVE
    // 64MiB cap; bound them by the actual fixed scalar-call inventory instead.
    let cumulative = scalar_calls
        .checked_mul(LIVE_SOURCE_CAP)
        .ok_or(Error::ArithmeticOverflow)?;
    let minimum = source
        .buffers
        .checked_mul(32768)
        .ok_or(Error::ArithmeticOverflow)?;
    if main.bytes != bytes
        || main.buffers != pages
        || (source.bytes == 0) != (source.buffers == 0)
        || source.bytes < minimum
        || source.bytes > cumulative
        || source.buffers > cumulative / 32768
        || native && (scalar_calls != 0 || source.bytes != 0)
    {
        return Err(error());
    }
    Ok(Audit {
        main_bytes: main.bytes,
        main_buffers: main.buffers,
        source_bytes: source.bytes,
        source_buffers: source.buffers,
    })
}
#[derive(Default, Clone, Copy)]
struct Numeric {
    bit_mismatches: usize,
    max_abs: f64,
    max_relative: f64,
    diagnostic_outside: usize,
    greedy_equal: bool,
}
impl Numeric {
    fn value(self) -> Value {
        json!({"bit_mismatches":self.bit_mismatches,"max_absolute":self.max_abs,
            "max_relative":self.max_relative,"diagnostic_outside":self.diagnostic_outside,
            "diagnostic_bound_only":"1e-4+3e-4*abs(reference)","greedy_equal":self.greedy_equal,
            "bit_equal":self.bit_mismatches==0})
    }
}
fn compare(reference: &Zeroizing<Vec<f32>>, candidate: &Zeroizing<Vec<f32>>) -> Result<Numeric> {
    if reference.len() != VOCAB
        || candidate.len() != VOCAB
        || reference
            .capacity()
            .checked_add(candidate.capacity())
            .and_then(|n| n.checked_mul(size_of::<f32>()))
            .is_none_or(|n| n > PROBE_HOST_CAP)
    {
        return Err(error());
    }
    let mut n = Numeric {
        greedy_equal: greedy(reference)? == greedy(candidate)?,
        ..Numeric::default()
    };
    for (&a, &b) in reference.iter().zip(candidate.iter()) {
        if !a.is_finite() || !b.is_finite() {
            return Err(error());
        }
        n.bit_mismatches += usize::from(a.to_bits() != b.to_bits());
        let difference = (f64::from(a) - f64::from(b)).abs();
        n.max_abs = n.max_abs.max(difference);
        n.max_relative = n
            .max_relative
            .max(difference / f64::from(a).abs().max(1e-30));
        n.diagnostic_outside += usize::from(difference > 1e-4 + 3e-4 * f64::from(a).abs());
    }
    // Numeric mismatch alone is reported after cleanup for parent review; no
    // arbitrary whole-model tolerance is accepted and no relaxation is retried.
    Ok(n)
}
struct CaseReport {
    id: u32,
    context: usize,
    numeric: Option<Numeric>,
    audit: Audit,
    terminal: bool,
    hook_observed: bool,
}
impl CaseReport {
    fn value(&self) -> Value {
        json!({"case":self.id,"context":self.context,"numeric":self.numeric.map(Numeric::value),
            "audit":self.audit.value(),"terminal_cleanup":self.terminal,"hook_observed":self.hook_observed})
    }
}
fn accepted(model: &ResidentModel<'_, '_>, position: usize) -> Result<()> {
    if model.position != position
        || model.poisoned
        || model.ple_batch_pending.is_some()
        || !model.pending_qsa.is_empty()
        || !model.pending_moe.is_empty()
        || model.retired.load(Ordering::Acquire)
    {
        return Err(error());
    }
    Ok(())
}
fn validate_partition(tokens: usize, partition: &[usize]) -> Result<()> {
    if !(1..=8).contains(&tokens)
        || !(1..=8).contains(&partition.len())
        || partition.iter().any(|n| !(1..=8).contains(n))
    {
        return Err(error());
    }
    let sum = partition.iter().try_fold(0usize, |n, p| {
        n.checked_add(*p).ok_or(Error::ArithmeticOverflow)
    })?;
    if sum != tokens {
        return Err(error());
    }
    Ok(())
}
fn native_sequence<'f>(
    model: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    ids: &[u32],
    partition: &[usize],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    validate_partition(ids.len(), partition)?;
    let mut at = 0usize;
    let mut output = Zeroizing::new(Vec::new());
    for (i, &count) in partition.iter().enumerate() {
        let end = at.checked_add(count).ok_or(Error::ArithmeticOverflow)?;
        output = model.prefill_chunk(
            e,
            ids.get(at..end).ok_or(error())?,
            if i + 1 == partition.len() {
                batch::Purpose::LastLogits
            } else {
                batch::Purpose::Advance
            },
            cancel,
        )?;
        accepted(model, end)?;
        if i + 1 != partition.len() && !output.is_empty() {
            return Err(error());
        }
        at = end;
    }
    Ok(output)
}
fn scalar_sequence<'f>(
    model: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    ids: &[u32],
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    let mut output = Zeroizing::new(Vec::new());
    for (i, &token) in ids.iter().enumerate() {
        if i + 1 == ids.len() {
            output = model.step(e, token, cancel)?;
        } else {
            model.advance_without_logits(e, token, cancel)?;
        }
        accepted(model, i + 1)?;
    }
    Ok(output)
}
fn retries_rejected<'f>(
    model: &mut ResidentModel<'_, 'f>,
    e: &mut GpuExecutor<'f>,
    token: u32,
    cancel: &AtomicBool,
    position: usize,
) -> Result<()> {
    cancel.store(false, Ordering::Release);
    if !model.erased_after_failure()
        || model.position != position
        || !model.layers.is_empty()
        || !model.moe_layers.is_empty()
        || model.native_batch.is_some()
        || model.ple_batch_pending.is_some()
        || !model.pending_qsa.is_empty()
        || !model.pending_moe.is_empty()
        || model
            .prefill_chunk(e, &[token], batch::Purpose::LastLogits, cancel)
            .is_ok()
        || model.step(e, token, cancel).is_ok()
        || !model.erased_after_failure()
    {
        return Err(error());
    }
    Ok(())
}

/// Original fixed "!" tokenizer fixture only; receipts contain aggregate
/// numeric/lifecycle statistics, never prompt/logit vectors, route summaries or
/// arbitrary input. Parent must create_new its numeric receipt after this returns.
pub(crate) fn run(file: &GgufFile, config: RunConfig) -> Result<Value> {
    let retired = config.retired.clone();
    let attempt =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_inner(file, config)));
    match attempt {
        Ok(Ok(receipt)) => Ok(receipt),
        _ => {
            retired.store(true, Ordering::Release);
            Err(error())
        }
    }
}
fn run_inner(file: &GgufFile, config: RunConfig) -> Result<Value> {
    let f = Fixture::new(file, config)?;
    let model = Qwen4Exp::bind(file)?;
    let graph = model.bound_graph();
    f.envelope(graph)?;
    let tokenizer = Qwen35Tokenizer::from_gguf(file, TokenizerLimits::default())?;
    let tokenized = Zeroizing::new(tokenizer.encode("!", true)?);
    if tokenized.len() != 1 || tokenized[0] as usize >= VOCAB {
        return Err(error());
    }
    let mut ids = Zeroizing::new([0u32; 9]);
    ids.fill(tokenized[0]);
    let mut reports = Vec::with_capacity(MAX_REPORTS);
    if reports.capacity() * size_of::<CaseReport>() > PROBE_META_CAP {
        return Err(error());
    }
    let mut sequence_exact = true;
    let mut greedy_exact = true;
    for context in [4usize, 8] {
        let (reference, audit) = f.owner(graph, context, false, context as u64, |m, e, c| {
            let logits = scalar_sequence(m, e, &ids[..context], c)?;
            // Original scalar preflight exhaustion rejects WITHOUT claiming
            // automatic erasure. Preserve it, then explicitly close/audit.
            if m.step(e, ids[0], c).is_ok() || m.position != context {
                return Err(error());
            }
            m.close()?;
            retries_rejected(m, e, ids[0], c, context)?;
            Ok(logits)
        })?;
        reports.push(CaseReport {
            id: if context == 4 { 0 } else { 1 },
            context,
            numeric: None,
            audit,
            terminal: true,
            hook_observed: false,
        });
        let partitions: &[&[usize]] = if context == 4 {
            &[&[3, 1]]
        } else {
            &[&[3, 5], &[8]]
        };
        for partition in partitions {
            let (numeric, audit) = f.owner(graph, context, true, 0, |m, e, c| {
                let logits = native_sequence(m, e, &ids[..context], partition, c)?;
                let numeric = compare(&reference, &logits)?;
                if m.prefill_chunk(e, &ids[..1], batch::Purpose::LastLogits, c)
                    .is_ok()
                {
                    return Err(error());
                }
                retries_rejected(m, e, ids[0], c, context)?;
                Ok(numeric)
            })?;
            sequence_exact &= numeric.bit_mismatches == 0;
            greedy_exact &= numeric.greedy_equal;
            reports.push(CaseReport {
                id: match *partition {
                    [3, 1] => 2,
                    [3, 5] => 3,
                    _ => 4,
                },
                context,
                numeric: Some(numeric),
                audit,
                terminal: true,
                hook_observed: false,
            });
        }
        // Actual fresh recovery after the exhausted owner/executor was dropped.
        let (numeric, audit) = f.owner(graph, context, true, 0, |m, e, c| {
            compare(
                &reference,
                &native_sequence(m, e, &ids[..context], &[context], c)?,
            )
        })?;
        sequence_exact &= numeric.bit_mismatches == 0;
        greedy_exact &= numeric.greedy_equal;
        reports.push(CaseReport {
            id: if context == 4 { 5 } else { 6 },
            context,
            numeric: Some(numeric),
            audit,
            terminal: false,
            hook_observed: false,
        });
    }
    // Fresh actual accepted prefix before malformed/injected attempt. Retaining
    // its original one-token history is proved by the real owner, never seeded.
    let (reference, audit) = f.owner(graph, 4, false, 4, |m, e, c| {
        scalar_sequence(m, e, &ids[..4], c)
    })?;
    reports.push(CaseReport {
        id: 7,
        context: 4,
        numeric: None,
        audit,
        terminal: false,
        hook_observed: false,
    });
    for invalid in 0..3 {
        let (_, audit) = f.owner(graph, 4, true, 0, |m, e, c| {
            m.prefill_chunk(e, &ids[..1], batch::Purpose::Advance, c)?;
            accepted(m, 1)?;
            let bad = Zeroizing::new([VOCAB as u32]);
            let input = match invalid {
                0 => &ids[..0],
                1 => &ids[..9],
                _ => &bad[..],
            };
            if m.prefill_chunk(e, input, batch::Purpose::LastLogits, c)
                .is_ok()
            {
                return Err(error());
            }
            retries_rejected(m, e, ids[0], c, 1)
        })?;
        reports.push(CaseReport {
            id: 8 + invalid,
            context: 4,
            numeric: None,
            audit,
            terminal: true,
            hook_observed: false,
        });
        let (numeric, audit) = f.owner(graph, 4, true, 0, |m, e, c| {
            compare(&reference, &native_sequence(m, e, &ids[..4], &[1, 3], c)?)
        })?;
        sequence_exact &= numeric.bit_mismatches == 0;
        greedy_exact &= numeric.greedy_equal;
        reports.push(CaseReport {
            id: 11 + invalid,
            context: 4,
            numeric: Some(numeric),
            audit,
            terminal: false,
            hook_observed: false,
        });
    }
    for (index, site) in [
        Site::LateCombineError,
        Site::PleSecondProofError,
        Site::PleSecondProofPanic,
        Site::FirstGdnCommitCancel,
    ]
    .into_iter()
    .enumerate()
    {
        let (_, audit) = f.owner(graph, 4, true, 0, |m, e, c| {
            m.prefill_chunk(e, &ids[..1], batch::Purpose::Advance, c)?;
            accepted(m, 1)?;
            m.lifecycle_hooks.arm(site)?;
            if m.prefill_chunk(e, &ids[1..4], batch::Purpose::LastLogits, c)
                .is_ok()
                || !m.lifecycle_hooks.observed(site)
            {
                return Err(error());
            }
            retries_rejected(m, e, ids[0], c, 1)
        })?;
        reports.push(CaseReport {
            id: 14 + index as u32,
            context: 4,
            numeric: None,
            audit,
            terminal: true,
            hook_observed: true,
        });
        let (numeric, audit) = f.owner(graph, 4, true, 0, |m, e, c| {
            compare(&reference, &native_sequence(m, e, &ids[..4], &[1, 3], c)?)
        })?;
        sequence_exact &= numeric.bit_mismatches == 0;
        greedy_exact &= numeric.greedy_equal;
        reports.push(CaseReport {
            id: 18 + index as u32,
            context: 4,
            numeric: Some(numeric),
            audit,
            terminal: false,
            hook_observed: false,
        });
    }
    if reports.len() != 22
        || reports.capacity() > MAX_REPORTS
        || f.config.retired.load(Ordering::Acquire)
    {
        return Err(error());
    }
    // Only fixed-public aggregate reports survive. Private vectors drop guarded
    // before the parent can emit a durable numeric-only receipt.
    drop(reference);
    drop(tokenized);
    drop(ids);
    graph.verify_unchanged()?;
    Ok(
        json!({"fixture":"actual-native-model-lifecycle","cases":reports.iter().map(CaseReport::value).collect::<Vec<_>>(),
        "whole_model_numeric_accepted":sequence_exact && greedy_exact,
        "numeric_not_accepted":!sequence_exact || !greedy_exact,"all_greedy_equal":greedy_exact,
        "all_logits_bit_equal":sequence_exact,"lifecycle_accepted":true,"owned_io_cleanup_accepted":true,"context_retired":false,
        "extra_main_bytes":0,"native_main_bytes":NATIVE_BYTES,"native_main_pages":NATIVE_PAGES,
        "scalar_main_bytes":SCALAR_BYTES,"scalar_main_pages":SCALAR_PAGES,"probe_host_cap":PROBE_HOST_CAP,
        "probe_metadata_cap":PROBE_META_CAP,"model_host_cap":12*1024*1024,
        "independent_runtime_cap":crate::executor::RESIDENT_RUNTIME_IO_MAX,"live_source_cap":LIVE_SOURCE_CAP,
        "source_audits_cumulative":true,"scope":"fixed-public whole-model lifecycle; numeric mismatch needs parent review"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::private_arena::OwnedZeroAudit;
    #[test]
    fn cumulative_source_audit_is_not_a_live_residency_cap() {
        let main = OwnedZeroAudit {
            bytes: SCALAR_BYTES,
            buffers: SCALAR_PAGES,
        };
        let source = OwnedZeroAudit {
            bytes: 70_746_112,
            buffers: 1905,
        };
        assert!(validate_audit(main, source, false, 127).is_ok());
        assert!(validate_audit(main, source, false, 1).is_err());
        assert!(validate_audit(main, source, true, 127).is_err());
        assert!(validate_audit(main, source, false, u64::MAX).is_err());
        assert!(
            validate_audit(
                main,
                OwnedZeroAudit {
                    bytes: 32767,
                    buffers: 1
                },
                false,
                1
            )
            .is_err()
        );
    }
    #[test]
    fn exact_main_and_empty_native_source_are_independent_gates() {
        let main = OwnedZeroAudit {
            bytes: NATIVE_BYTES,
            buffers: NATIVE_PAGES,
        };
        assert!(validate_audit(main, OwnedZeroAudit::default(), true, 0).is_ok());
        for bad in [
            OwnedZeroAudit {
                bytes: NATIVE_BYTES,
                buffers: NATIVE_PAGES - 1,
            },
            OwnedZeroAudit {
                bytes: NATIVE_BYTES - 1,
                buffers: NATIVE_PAGES,
            },
        ] {
            assert!(validate_audit(bad, OwnedZeroAudit::default(), true, 0).is_err());
        }
        assert!(
            validate_audit(
                main,
                OwnedZeroAudit {
                    bytes: 32768,
                    buffers: 1
                },
                true,
                0
            )
            .is_err()
        );
    }
    #[test]
    fn numeric_report_never_relaxes_bit_or_greedy_acceptance() {
        let mut a = Zeroizing::new(vec![0.0; VOCAB]);
        a[9] = 1.0;
        let b = Zeroizing::new(a.to_vec());
        let n = compare(&a, &b).unwrap();
        assert_eq!(n.bit_mismatches, 0);
        assert!(n.greedy_equal);
        let mut b = b;
        b[10] = -0.0;
        b[9] = f32::from_bits(1.0f32.to_bits() + 1);
        let n = compare(&a, &b).unwrap();
        assert_eq!(n.bit_mismatches, 2);
        assert!(n.greedy_equal);
        assert_eq!(n.diagnostic_outside, 0);
        b[10] = 2.;
        let n = compare(&a, &b).unwrap();
        assert!(!n.greedy_equal);
        assert!(n.diagnostic_outside > 0);
        b[7] = f32::NAN;
        assert!(compare(&a, &b).is_err());
        b[7] = 0.;
        b.pop();
        assert!(compare(&a, &b).is_err());
    }
    #[test]
    fn actual_fixture_partition_inventory_rejects_before_enqueue() {
        for (tokens, partition) in [
            (4, &[3, 1][..]),
            (8, &[3, 5][..]),
            (8, &[8][..]),
            (4, &[1, 3][..]),
        ] {
            validate_partition(tokens, partition).unwrap();
        }
        for (tokens, partition) in [
            (0, &[][..]),
            (9, &[8, 1][..]),
            (4, &[4, 1][..]),
            (4, &[0, 4][..]),
            (4, &[usize::MAX][..]),
        ] {
            assert!(validate_partition(tokens, partition).is_err());
        }
    }
    #[test]
    fn fixture_envelope_is_separate_from_original_model_credit() {
        assert_eq!(NATIVE_BYTES - SCALAR_BYTES, 33_062_912);
        assert_eq!(NATIVE_PAGES - SCALAR_PAGES, 90);
        const {
            assert!(2 * VOCAB * size_of::<f32>() < PROBE_HOST_CAP);
        }
        assert!(MAX_REPORTS * size_of::<CaseReport>() < PROBE_META_CAP);
        assert_eq!(
            device_plan(0x8086).unwrap(),
            (8usize << 30, 64, 26usize << 30)
        );
        assert_eq!(
            device_plan(0x10de).unwrap(),
            (6usize << 30, 16, 10usize << 30)
        );
        assert!(device_plan(0).is_err());
        assert!(checked_sum(&[u64::MAX, 1]).is_err());
    }
}
