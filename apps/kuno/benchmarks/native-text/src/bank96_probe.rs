//! Fixed-public one-layer split-bank acceptance. No arbitrary routes/prompts.
//! Separate bounded operator admission never relaxes full-model admission.
//! Every residency mapping is obtained through ordinary ten-expert plans,
//! original fills, enqueue, completed statuses/owned-I/O erasure, then commit.
use crate::{
    expert_banks::{self, ExpertBanks},
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    resident_moe::ResidentMoe,
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::transient::TransientIoReceipt;
use kuno_text::cpu::Matrix;
use kuno_text::model::{BoundGraph, Qwen4Exp, RoutedWeights};
use kuno_text::{Error, GgufFile, GgufLimits, TensorType, TensorView};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MIB: u64 = 1024 * 1024;
const MAIN_LIMIT: u64 = 1024 * MIB; // Includes retained allocator baseline between dtypes.
const PRIVATE_LIMIT: usize = 2 * 1024 * 1024;
const IO_LIMIT: u64 = 64 * MIB;
const IDS: [usize; 10] = [0, 63, 64, 95, 1, 62, 65, 94, 31, 32];
const PROB: [f32; 10] = [0.05, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.1, 0.08];
#[derive(Clone, Copy, PartialEq, Eq)]
enum CancelAt {
    Never,
    FirstCopy,
    Enqueued,
}
struct Values {
    activations: Zeroizing<Vec<f32>>,
    partials: Zeroizing<Vec<f32>>,
    output: Zeroizing<Vec<f32>>,
}
struct Oracle {
    tree: Values,
    core: Zeroizing<Vec<f32>>,
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    client
        .memory_usage()
        .map(|v| v.bytes_reserved)
        .map_err(|_| "public bank probe memory telemetry".into())
}
fn observe(ledger: &mut MainAdmission, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    ledger
        .observe_after(sample(client)?)
        .map_err(|_| "public bank probe admission".into())
}
fn audit(client: &ComputeClient<WgpuRuntime>) -> Result<TransientIoReceipt> {
    let r = cubecl::future::block_on(client.sanitize_transient_io_audited())
        .map_err(|_| "public bank probe owned I/O cleanup")?;
    let full = r
        .upload_bytes
        .checked_add(r.uniform_bytes)
        .and_then(|n| n.checked_add(r.readback_bytes))
        .ok_or("public bank probe audit overflow")?;
    if full == 0 || r.inspected_bytes != full || r.inspected_buffers == 0 {
        return Err("public bank probe incomplete I/O audit".into());
    }
    Ok(r)
}
struct Scrub<'a>(&'a mut [u8]);
impl Drop for Scrub<'_> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn read_words(
    arena: &PrivateArena,
    id: BufferId,
    logical: usize,
    status: bool,
) -> Result<Zeroizing<Vec<f32>>> {
    let layout = arena.layout(id)?;
    if layout.logical_words != logical {
        return Err("public bank probe read extent".into());
    }
    let mut bytes = arena
        .client()?
        .read_one(arena.handle(id)?.clone())
        .map_err(|_| "public bank probe completed read")?;
    let b = Scrub(&mut bytes);
    if b.0.len() != layout.owned_bytes || b.0[logical * 4..].iter().any(|&v| v != 0) {
        return Err("public bank probe changed owning-span tail".into());
    }
    let mut values = Zeroizing::new(Vec::with_capacity(logical));
    for x in b.0[..logical * 4].chunks_exact(4) {
        let bits = u32::from_le_bytes(x.try_into().map_err(|_| "public bank probe word")?);
        if status {
            if bits != 1 {
                return Err("public bank probe status failure".into());
            }
        } else {
            let v = f32::from_bits(bits);
            if !v.is_finite() {
                return Err("public bank probe nonfinite output".into());
            }
            values.push(v);
        }
    }
    Ok(values) // Scrub and mapped readback lease drop before the caller sanitizes.
}
fn source_requests(weights: RoutedWeights<'_>, misses: usize) -> Result<Vec<u64>> {
    if misses > 10 {
        return Err("public bank probe source count".into());
    }
    let mut sizes = [0; 3];
    for (i, t) in [weights.gate, weights.up, weights.down]
        .into_iter()
        .enumerate()
    {
        sizes[i] = t
            .info()
            .row_bytes()
            .checked_mul(t.info().dimensions()[1])
            .ok_or("public bank probe source overflow")?;
        main_admission::page_bytes(sizes[i])?;
    }
    Ok((0..misses).flat_map(|_| sizes).collect())
}
#[allow(clippy::too_many_arguments)]
fn step(
    client: &ComputeClient<WgpuRuntime>,
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    ledger: &mut MainAdmission,
    banks: &mut ExpertBanks,
    moe: &mut ResidentMoe,
    input: BufferId,
    output: BufferId,
    ids: &[usize; 10],
    cancel_at: CancelAt,
) -> Result<(Values, TransientIoReceipt)> {
    let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        if ledger.is_poisoned() {
            return Err("public bank probe owner poisoned".into());
        }
        graph.verify_unchanged()?;
        let weights = moe.weights(graph)?;
        banks.check_client(client)?;
        banks.check_names(weights)?;
        let cancel = AtomicBool::new(false);
        let plan = banks.slots.plan(ids)?;
        observe(ledger, client)?;
        ledger.precharge(&source_requests(weights, plan.uploads.len())?)?;
        let mut copies = 0;
        for &(expert, slot) in &plan.uploads {
            for (bank, t) in [
                (banks.gate(), weights.gate),
                (banks.up(), weights.up),
                (banks.down(), weights.down),
            ] {
                expert_banks::fill(client, bank, t, expert, slot, &cancel)?;
                copies += 1;
                if cancel_at == CancelAt::FirstCopy && copies == 1 {
                    cancel.store(true, Ordering::Release);
                }
            }
        }
        let pending = moe.enqueue(
            graph,
            arena,
            banks,
            input,
            output,
            &plan.slots,
            &PROB,
            &cancel,
            true,
            false,
        )?;
        if cancel_at == CancelAt::Enqueued {
            cancel.store(true, Ordering::Release);
        }
        let status = pending.status();
        read_words(arena, status.buffer, status.words, true)?;
        let [(act, an), (part, pn)] = moe.fixed_public_probe_outputs();
        let values = Values {
            activations: read_words(arena, act, an, false)?,
            partials: read_words(arena, part, pn, false)?,
            output: read_words(arena, output, banks.hidden, false)?,
        };
        let io = audit(client)?;
        if cancel.load(Ordering::Acquire) {
            return Err("public bank probe cancelled".into());
        }
        ledger.completed_fence(sample(client)?)?;
        banks.slots.commit(plan)?;
        moe.accept_after_whole_token_acceptance(pending)?;
        Ok((values, io))
    }));
    match attempted {
        Ok(Ok(v)) => Ok(v),
        _ => {
            ledger.poison();
            moe.poison();
            Err("public bank probe operation rejected and owner poisoned".into())
        }
    }
}
fn dot64(w: &[f32], x: &[f32]) -> f32 {
    let mut lanes = [0.0; 64];
    for (lane, s) in lanes.iter_mut().enumerate() {
        for c in (lane..x.len()).step_by(64) {
            *s += w[c] * x[c];
        }
    }
    let mut stride = 32;
    while stride > 0 {
        for lane in 0..stride {
            lanes[lane] += lanes[lane + stride];
        }
        stride /= 2;
    }
    lanes[0]
}
fn projection(t: TensorView<'_>, expert: usize, x: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
    let m = Matrix::new(t)?;
    if m.inputs() != x.len() || m.inputs() > 4096 || m.outputs() > 4096 || expert >= 512 {
        return Err("public bank probe CPU geometry".into());
    }
    let row = t.info().row_bytes() as usize;
    let count = row
        .checked_mul(m.outputs())
        .ok_or("public bank probe CPU overflow")?;
    let mut packed = vec![0; count]; // Original public weights, never persisted.
    t.read_range(
        expert
            .checked_mul(count)
            .ok_or("public bank probe CPU offset")? as u64,
        &mut packed,
    )?;
    let mut decoded = Zeroizing::new(vec![0.0; x.len()]);
    let mut y = Zeroizing::new(Vec::with_capacity(m.outputs()));
    for b in packed.chunks_exact(row) {
        t.info().dtype().decode(b, &mut decoded)?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err("public bank probe original weights nonfinite".into());
        }
        y.push(dot64(&decoded, x));
    }
    Ok(y)
}
fn core(t: TensorView<'_>, e: usize, x: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
    let m = Matrix::new(t)?;
    let mut y = Zeroizing::new(vec![0.0; m.outputs()]);
    m.apply(e, x, &mut y, &AtomicBool::new(false))?;
    Ok(y)
}
fn oracle(weights: RoutedWeights<'_>, x: &[f32]) -> Result<Oracle> {
    let ff = Matrix::new(weights.gate)?.outputs();
    let hidden = x.len();
    let mut act = Zeroizing::new(Vec::with_capacity(10 * ff));
    let mut parts = Zeroizing::new(Vec::with_capacity(10 * hidden));
    let mut core_out = Zeroizing::new(vec![0.0; hidden]);
    for (rank, &expert) in IDS.iter().enumerate() {
        let g = projection(weights.gate, expert, x)?;
        let u = projection(weights.up, expert, x)?;
        let a = Zeroizing::new(
            g.iter()
                .zip(&*u)
                .map(|(&g, &u)| kuno_text::ops::silu(g) * u)
                .collect::<Vec<_>>(),
        );
        let d = projection(weights.down, expert, &a)?;
        act.extend_from_slice(&a);
        parts.extend_from_slice(&d);
        let g = core(weights.gate, expert, x)?;
        let u = core(weights.up, expert, x)?;
        let a = Zeroizing::new(
            g.iter()
                .zip(&*u)
                .map(|(&g, &u)| kuno_text::ops::silu(g) * u)
                .collect::<Vec<_>>(),
        );
        let d = core(weights.down, expert, &a)?;
        for j in 0..hidden {
            core_out[j] += d[j] * PROB[rank];
        }
    }
    let mut output = Zeroizing::new(vec![0.0; hidden]);
    for rank in 0..10 {
        for j in 0..hidden {
            output[j] += parts[rank * hidden + j] * PROB[rank];
        }
    }
    Ok(Oracle {
        tree: Values {
            activations: act,
            partials: parts,
            output,
        },
        core: core_out,
    })
}
fn compare(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> Result<f32> {
    if a.len() != b.len() || a.is_empty() {
        return Err("public bank probe comparison length".into());
    }
    let mut max = 0.0f32;
    for (&a, &b) in a.iter().zip(b) {
        let error = (a - b).abs();
        if !a.is_finite() || !b.is_finite() || error > atol + rtol * b.abs() {
            return Err("public bank probe numeric comparison failed".into());
        }
        max = max.max(error);
    }
    Ok(max)
}
fn allocate_banks(
    client: &ComputeClient<WgpuRuntime>,
    ledger: &mut MainAdmission,
    weights: RoutedWeights<'_>,
    capacity: usize,
) -> Result<ExpertBanks> {
    observe(ledger, client)?;
    ledger.precharge(&ExpertBanks::allocation_requests(weights, capacity)?)?;
    let bank = ExpertBanks::new(client, weights, capacity)?;
    observe(ledger, client)?;
    Ok(bank)
}
fn prime_ids() -> Vec<[usize; 10]> {
    let mut v = (0..9)
        .map(|n| std::array::from_fn(|r| n * 10 + r))
        .collect::<Vec<_>>();
    v.push([90, 91, 92, 93, 94, 95, 0, 1, 2, 3]);
    v
}
fn refresh_first() -> Vec<[usize; 10]> {
    let mut v = (0..6)
        .map(|n| std::array::from_fn(|r| n * 10 + r))
        .collect::<Vec<_>>();
    v.push([60, 61, 62, 63, 0, 1, 2, 3, 4, 5]);
    v
}
fn public_weights_preserved(
    client: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
) -> Result<Vec<(cubecl::server::Handle, Vec<u8>)>> {
    let mut samples = Vec::new();
    for bank in [banks.gate(), banks.up(), banks.down()] {
        for segment in bank.segments() {
            let bytes = segment.words() * 4;
            for first in [0, bytes - 16] {
                let h = segment
                    .handle()
                    .clone()
                    .offset_start(first as u64)
                    .offset_end((bytes - first - 16) as u64);
                let actual = client
                    .read_one(h.clone())
                    .map_err(|_| "public bank probe public sentinel read")?;
                if actual.len() != 16 {
                    return Err("public bank probe public sentinel length".into());
                }
                samples.push((h, actual.to_vec()));
                drop(actual);
            }
        }
    }
    audit(client)?;
    Ok(samples)
}
fn expected_bank_pages(dtype: TensorType) -> Result<u64> {
    let down = match dtype {
        TensorType::Q5_0 => 1_126_400_u64,
        TensorType::Q8_0 => 1_740_800_u64,
        _ => return Err("public bank probe down dtype".into()),
    };
    let mut bytes = 0_u64;
    for count in [64, 32, 10] {
        for expert_bytes in [921_600_u64, 921_600_u64, down] {
            bytes = bytes
                .checked_add(main_admission::page_bytes(expert_bytes * count)?)
                .ok_or("public bank probe charge overflow")?;
        }
    }
    Ok(bytes)
}
fn one_case(
    client: &ComputeClient<WgpuRuntime>,
    arena: &mut PrivateArena,
    graph: BoundGraph<'_, '_>,
    layer: usize,
    dtype: TensorType,
) -> Result<serde_json::Value> {
    let weights = RoutedWeights {
        gate: graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        up: graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        down: graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    };
    let bank_bytes = ExpertBanks::allocation(weights, 96)? + ExpertBanks::allocation(weights, 10)?;
    if weights.down.info().dtype() != dtype || bank_bytes as u64 != expected_bank_pages(dtype)? {
        return Err("public bank probe bank envelope".into());
    }
    let baseline = sample(client)?;
    main_admission::check_capacity(
        MAIN_LIMIT,
        &[bank_bytes as u64, PRIVATE_LIMIT as u64, 40 * MIB, baseline],
    )?;
    let mut ledger = MainAdmission::new(MAIN_LIMIT, baseline)?;
    let mut split = allocate_banks(client, &mut ledger, weights, 96)?;
    let mut flat = allocate_banks(client, &mut ledger, weights, 10)?;
    ledger.precharge(&[PRIVATE_LIMIT as u64])?;
    let fixture = Zeroizing::new(
        (0..2560)
            .map(|i| ((i * 17 + i / 64 * 3) % 31) as f32 / 16.0 - 15.0 / 16.0)
            .collect::<Vec<_>>(),
    );
    let x = arena.upload_f32(&fixture)?;
    let y = arena.allocate_zeroed(2560)?;
    let old_y = arena.allocate_zeroed(2560)?;
    let mut moe = ResidentMoe::allocate(graph, layer, arena)?;
    let mut old = ResidentMoe::allocate(graph, layer, arena)?;
    observe(&mut ledger, client)?;
    for ids in prime_ids() {
        step(
            client,
            graph,
            arena,
            &mut ledger,
            &mut split,
            &mut moe,
            x,
            y,
            &ids,
            CancelAt::Never,
        )?;
    }
    let boundary = split.slots.plan(&IDS)?;
    if boundary.slots != IDS || !boundary.uploads.is_empty() {
        return Err("public bank probe actual cross-segment slots".into());
    }
    drop(boundary);
    let expected = oracle(weights, &fixture)?;
    let (got, io) = step(
        client,
        graph,
        arena,
        &mut ledger,
        &mut split,
        &mut moe,
        x,
        y,
        &IDS,
        CancelAt::Never,
    )?;
    let (legacy, _) = step(
        client,
        graph,
        arena,
        &mut ledger,
        &mut flat,
        &mut old,
        x,
        old_y,
        &IDS,
        CancelAt::Never,
    )?;
    let mut tree_errors = Vec::new();
    let mut legacy_errors = Vec::new();
    for (actual, want, previous) in [
        (
            &got.activations,
            &expected.tree.activations,
            &legacy.activations,
        ),
        (&got.partials, &expected.tree.partials, &legacy.partials),
        (&got.output, &expected.tree.output, &legacy.output),
    ] {
        tree_errors.push(compare(actual, want, 1e-4, 3e-4)?);
        legacy_errors.push(compare(actual, previous, 1e-4, 3e-4)?);
    }
    let core_error = compare(&got.output, &expected.core, 1e-3, 3e-3)?;
    let (cancel_at, cancelled_ids) = if dtype == TensorType::Q8_0 {
        // Refresh the first shard through accepted ordinary operations, so
        // the next missing expert really selects an LRU victim in shard two.
        for ids in refresh_first() {
            step(
                client,
                graph,
                arena,
                &mut ledger,
                &mut split,
                &mut moe,
                x,
                y,
                &ids,
                CancelAt::Never,
            )?;
        }
        let mut missing = IDS;
        missing[9] = 200;
        let plan = split.slots.plan(&missing)?;
        if plan.uploads.len() != 1 || plan.uploads[0].1 < 64 {
            return Err("public bank probe second-shard cancellation fixture".into());
        }
        drop(plan);
        (CancelAt::FirstCopy, missing)
    } else {
        (CancelAt::Enqueued, IDS)
    };
    if step(
        client,
        graph,
        arena,
        &mut ledger,
        &mut split,
        &mut moe,
        x,
        y,
        &cancelled_ids,
        cancel_at,
    )
    .is_ok()
        || !ledger.is_poisoned()
    {
        return Err("public bank probe cancellation accepted".into());
    }
    old.poison();
    if step(
        client,
        graph,
        arena,
        &mut ledger,
        &mut split,
        &mut moe,
        x,
        y,
        &IDS,
        CancelAt::Never,
    )
    .is_ok()
    {
        return Err("public bank probe poisoned retry accepted".into());
    }
    // Fixed fixture samples only; private cleanup must not erase public banks.
    let mut sentinels = public_weights_preserved(client, &split)?;
    sentinels.extend(public_weights_preserved(client, &flat)?);
    arena.reset()?;
    for (h, expected) in sentinels {
        let got = client
            .read_one(h)
            .map_err(|_| "public bank probe public preservation read")?;
        if &*got != expected.as_slice() {
            return Err("public bank probe private erasure changed public weights".into());
        }
        drop(got);
    }
    let after = audit(client)?;
    Ok(
        serde_json::json!({"down_dtype":dtype.gguf_id(),"layer":layer,"bank_bytes":bank_bytes,
        "all_ten_ranks":true,"cross_segment_slots":[0,63,64,95],"tree_max_abs_errors":tree_errors,
        "legacy_max_abs_errors":legacy_errors,"core_max_abs_error":core_error,
        "post_enqueue_cancel_poisoned":dtype==TensorType::Q5_0,"second_shard_partial_fill_cancel_poisoned":dtype==TensorType::Q8_0,"poisoned_retry_rejected":true,"public_weight_sentinels_preserved":true,
        "io_epoch":io.epoch,"io_inspected_bytes":io.inspected_bytes,"after_cleanup_io_inspected_bytes":after.inspected_bytes}),
    )
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    if !clean || retired {
        return Err("public bank probe cleanup or retired context".into());
    }
    attempt.map_err(|_| "public bank probe panicked")?
}
pub fn run(client: &ComputeClient<WgpuRuntime>) -> Result<()> {
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing original GGUF required")?;
    let file =
        GgufFile::open(path, GgufLimits::default()).map_err(|_| "public bank probe GGUF open")?;
    let model = Qwen4Exp::bind(&file).map_err(|_| "public bank probe original model bind")?;
    let graph = model.bound_graph();
    if expert_banks::preflight_residency(
        graph,
        96,
        6 * 1024 * 1024 * 1024,
        26 * 1024 * 1024 * 1024,
    )? != 18 * 1024 * 1024 * 1024
    {
        return Err("public bank probe full original graph geometry".into());
    }
    if !matches!(
        expert_banks::preflight_residency(
            graph,
            96,
            6 * 1024 * 1024 * 1024,
            10 * 1024 * 1024 * 1024
        ),
        Err(Error::LimitExceeded(_))
    ) {
        return Err("public bank probe NVIDIA full-model plan was not rejected".into());
    }
    let mut layers = Vec::new();
    for dtype in [TensorType::Q5_0, TensorType::Q8_0] {
        let layer = (0..48)
            .find(|&layer| {
                graph
                    .tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
                    .is_ok_and(|t| t.info().dtype() == dtype)
            })
            .ok_or("public bank probe both original down formats required")?;
        layers.push((layer, dtype));
    }
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), PRIVATE_LIMIT, IO_LIMIT, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut reports = Vec::new();
        for (layer, dtype) in layers {
            reports.push(one_case(client, &mut arena, graph, layer, dtype)?);
            client.memory_cleanup();
        }
        Ok(reports)
    }));
    let close = arena.close();
    let reports = finish(attempt, close.is_ok(), retired.load(Ordering::Acquire))?;
    let cleanup = arena
        .owned_zero_audit_receipt()
        .ok_or("public bank probe missing full main audit")?;
    file.verify_file_unchanged()
        .map_err(|_| "public bank probe original file changed")?;
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let r = serde_json::json!({"scope":"fixed-public-one-layer-segmented-MoE-operator-only","cases":reports,
            "full_model_96_on_nvidia_rejected":true,"normal_model_preflight_unchanged":true,"main_limit_bytes":MAIN_LIMIT,
            "runtime_io_limit_bytes":IO_LIMIT,"main_erased_buffers":cleanup.buffers,"main_erased_bytes":cleanup.bytes,
            "full_graph_or_performance_accepted":false,"context_retired":false});
        let mut out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&r)?.as_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{bank_slots::BankSlots, private_arena::BufferLayout, resident_moe::MoeLayout};

    fn prime() -> BankSlots {
        let mut slots = BankSlots::new(96, 512).unwrap();
        for ids in prime_ids() {
            let plan = slots.plan(&ids).unwrap();
            slots.commit(plan).unwrap();
        }
        slots
    }

    #[test]
    fn ordinary_accepted_plans_reach_every_actual_boundary_slot() {
        let mut slots = prime();
        let plan = slots.plan(&IDS).unwrap();
        assert_eq!(plan.slots, IDS);
        assert!(plan.uploads.is_empty());
        assert_eq!(plan.slots.iter().filter(|&&s| s < 64).count(), 6);
        assert_eq!(plan.slots.iter().filter(|&&s| s >= 64).count(), 4);
        slots.commit(plan).unwrap();
        let mut flat = BankSlots::new(10, 512).unwrap();
        let plan = flat.plan(&IDS).unwrap();
        assert_eq!(plan.slots, (0..10).collect::<Vec<_>>());
        assert_eq!(plan.uploads.len(), 10);
    }

    #[test]
    fn real_lru_refresh_selects_second_segment_and_abandon_grants_no_commit() {
        let mut slots = prime();
        let plan = slots.plan(&IDS).unwrap();
        slots.commit(plan).unwrap();
        for ids in refresh_first() {
            let plan = slots.plan(&ids).unwrap();
            assert!(plan.uploads.is_empty());
            slots.commit(plan).unwrap();
        }
        let mut missing = IDS;
        missing[9] = 200;
        let before = slots.plan(&missing).unwrap();
        assert_eq!(before.uploads, vec![(200, 66)]);
        let expected = before.slots.clone();
        drop(before); // A partial physical copy cannot be undone by this drop.
        let retry = slots.plan(&missing).unwrap();
        assert_eq!(retry.uploads, vec![(200, 66)]);
        assert_eq!(retry.slots, expected);
        let mut ledger = MainAdmission::new(MAIN_LIMIT, 0).unwrap();
        ledger.precharge(&[921600]).unwrap();
        let charged = ledger.upper_bytes();
        ledger.poison(); // Whole owner rejection; no physical retry is admitted.
        assert!(ledger.completed_fence(0).is_err());
        assert!(ledger.precharge(&[]).is_err());
        assert_eq!(ledger.upper_bytes(), charged);
    }

    #[test]
    fn one_layer_envelope_counts_each_source_page_and_private_span() {
        assert_eq!(expected_bank_pages(TensorType::Q5_0).unwrap(), 432 * MIB);
        assert_eq!(expected_bank_pages(TensorType::Q8_0).unwrap(), 448 * MIB);
        assert!(expected_bank_pages(TensorType::F32).is_err());
        let gate_page = main_admission::page_bytes(921600).unwrap();
        for down in [1126400, 1740800] {
            let down_page = main_admission::page_bytes(down).unwrap();
            assert_eq!(10 * (2 * gate_page + down_page), 40 * MIB);
        }
        let private = 2 * MoeLayout::new(2560, 640).unwrap().owned_bytes().unwrap()
            + 3 * BufferLayout::new(2560).unwrap().owned_bytes;
        assert_eq!(private, 622592);
        assert!(private <= PRIVATE_LIMIT);
        assert!(
            main_admission::check_capacity(
                MAIN_LIMIT,
                &[448 * MIB, 448 * MIB, PRIVATE_LIMIT as u64, 40 * MIB],
            )
            .is_ok()
        );
        assert!(
            main_admission::check_capacity(
                MAIN_LIMIT,
                &[448 * MIB, 600 * MIB, PRIVATE_LIMIT as u64, 40 * MIB],
            )
            .is_err()
        );
    }

    #[test]
    fn independent_dot_preserves_the_original_lane_tree() {
        let w = (0..2560)
            .map(|i| ((i * 7) % 13) as f32 - 6.0)
            .collect::<Vec<_>>();
        let x = (0..2560)
            .map(|i| ((i * 3) % 11) as f32 - 5.0)
            .collect::<Vec<_>>();
        assert_eq!(
            dot64(&w, &x),
            w.iter().zip(&x).map(|(&w, &x)| w * x).sum::<f32>()
        );
        let mut cancellation = vec![0.0; 128];
        cancellation[0] = 1e20;
        cancellation[1] = -1e20;
        cancellation[64] = 1.0;
        assert_eq!(dot64(&cancellation, &vec![1.0; 128]), 0.0);
        assert_eq!(cancellation.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn comparison_and_cleanup_gates_reject_incomplete_acceptance() {
        assert_eq!(compare(&[1.0, -2.0], &[1.0, -2.0], 0.0, 0.0).unwrap(), 0.0);
        for (a, b) in [
            (vec![], vec![]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![f32::NAN], vec![0.0]),
            (vec![f32::INFINITY], vec![f32::INFINITY]),
            (vec![1.0], vec![1.1]),
        ] {
            assert!(compare(&a, &b, 1e-4, 1e-4).is_err());
        }
        assert_eq!(finish(Ok(Ok(7)), true, false).unwrap(), 7);
        assert!(finish(Ok(Ok(7)), false, false).is_err());
        assert!(finish(Ok(Ok(7)), true, true).is_err());
        assert!(finish::<u32>(Ok(Err("rejected".into())), true, false).is_err());
        assert!(finish::<u32>(Err(Box::new("panic")), true, false).is_err());
        assert!((PROB.iter().map(|&p| f64::from(p)).sum::<f64>() - 1.0).abs() < 1e-5);
    }
}
