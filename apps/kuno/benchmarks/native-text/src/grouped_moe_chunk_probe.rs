//! Fixed-public original Q4/Q5/Q8 grouped routed-only acceptance. No prompts,
//! arbitrary routes, publication, shared expert, full FFN or model acceptance.
use crate::{
    expert_banks::{self, ExpertBanks},
    grouped_moe_chunk as c,
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    resident_moe::{MoeLayout, ResidentMoe},
    token_waves,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use kuno_text::{
    Error, GgufFile, GgufLimits, TensorType, TensorView,
    cpu::Matrix,
    model::{BoundGraph, Qwen4Exp, RoutedWeights},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::Zeroizing;
type Result<T> = c::Result<T>;
const PROB: [f32; 10] = [0.05, 0.07, 0.09, 0.1, 0.11, 0.12, 0.13, 0.15, 0.1, 0.08];
const BOUNDARY: [usize; 10] = [0, 63, 64, 95, 1, 62, 65, 94, 31, 32];
const MIB: u64 = 1024 * 1024;
struct Oracle {
    activations: Zeroizing<Vec<f32>>,
    partials: Zeroizing<Vec<f32>>,
    output: Zeroizing<Vec<f32>>,
    core: Zeroizing<Vec<f32>>,
}
#[derive(Clone, Copy)]
struct Case {
    count: usize,
    overlap: bool,
    tile: usize,
    capacity: usize,
    boundary: bool,
}
fn cases() -> Vec<Case> {
    let mut all = Vec::with_capacity(20);
    for count in [1, 3, 5, 8] {
        for overlap in [true, false] {
            for tile in [4, 8] {
                all.push(Case {
                    count,
                    overlap,
                    tile,
                    capacity: 16,
                    boundary: false,
                });
            }
        }
    }
    for capacity in [64, 96] {
        for overlap in [true, false] {
            all.push(Case {
                count: 8,
                overlap,
                tile: 8,
                capacity,
                boundary: capacity == 96 && overlap,
            });
        }
    }
    all
}
fn plan(case: Case) -> Result<token_waves::Plan> {
    let mut input = token_waves::Input::new(case.count, case.capacity, 512)
        .map_err(|_| "grouped fixed route geometry")?;
    for t in 0..case.count {
        for (r, &p) in PROB.iter().enumerate() {
            let e = if case.boundary {
                BOUNDARY[r]
            } else if case.overlap {
                r
            } else {
                t * 10 + r
            };
            // Original bits and ten ranks; neither sorting nor normalization.
            input
                .set(t, r, e as u32, p)
                .map_err(|_| "grouped fixed original route")?;
        }
    }
    for t in 0..case.count {
        let row = input.token(t).ok_or("grouped fixed source token")?;
        if row.len() != 10
            || row
                .iter()
                .zip(PROB)
                .any(|(r, p)| r.probability().to_bits() != p.to_bits())
        {
            return Err("grouped fixed source probabilities".into());
        }
    }
    Ok(token_waves::Plan::build(&input, &AtomicBool::new(false))
        .map_err(|_| "grouped fixed wave planning")?)
}
fn input(t: usize) -> Zeroizing<Vec<f32>> {
    Zeroizing::new(
        (0..2560)
            .map(|i| (((i * 17 + i / 64 * 3 + t * 11) % 31) as f32 - 15.) / 64.)
            .collect(),
    )
}
fn input_image(l: &c::Layout, count: usize, late: bool) -> Zeroizing<Vec<u32>> {
    let mut x = Zeroizing::new(vec![0x3f340000; l.words[c::INPUT]]);
    for t in 0..count {
        for (i, &v) in input(t).iter().enumerate() {
            x[l.spans.input_first() + t * l.spans.input_stride() + i] = v.to_bits();
        }
    }
    if late {
        x[l.spans.input_first() + (count - 1) * l.spans.input_stride() + 2559] = f32::NAN.to_bits();
    }
    x
}
fn host_bound() -> Result<usize> {
    // At most candidate BEFORE/ACTUAL/previous-wave pages +original eighty-rank tree/core
    // expectations +one original expert decoded working set +packed bytes +
    // scalar snapshots +guarded metadata. No per-case oracle cache persists.
    let parts = [
        3 * 1_671_168,
        80 * 640 * 4,
        80 * 2560 * 4,
        2 * 8 * 2560 * 4,
        2560 * 640 * 4,
        1_740_800,
        327_680,
        1024 * 1024,
    ];
    let sum = parts.into_iter().try_fold(0usize, |s, n| {
        s.checked_add(n).ok_or("grouped host bound overflow")
    })?;
    if sum > c::HOST_CAP {
        return Err("grouped host envelope".into());
    }
    Ok(sum)
}
fn expected_bank_bytes(dtype: TensorType, capacity: usize) -> Result<u64> {
    let down = match dtype {
        TensorType::Q5_0 => 1_126_400,
        TensorType::Q8_0 => 1_740_800,
        _ => return Err("grouped original down dtype".into()),
    };
    let counts = if capacity == 96 {
        vec![64, 32, 10]
    } else {
        vec![capacity, 10]
    };
    let mut sum = 0;
    for count in counts {
        for e in [921_600, 921_600, down] {
            sum += main_admission::page_bytes((e * count) as u64)?;
        }
    }
    Ok(sum)
}
fn preflight(
    client: &ComputeClient<WgpuRuntime>,
    w: RoutedWeights<'_>,
    dtype: TensorType,
    case: Case,
    limits: &wgpu::Limits,
) -> Result<MainAdmission> {
    let l = c::Layout::original(dtype, case.tile, case.capacity)?;
    l.validate_limits(limits)?;
    let banks = ExpertBanks::allocation(w, case.capacity)? + ExpertBanks::allocation(w, 10)?;
    if banks as u64 != expected_bank_bytes(dtype, case.capacity)?
        || l.page_bytes()? + MoeLayout::new(2560, 640)?.owned_bytes()? + 65536 != 1_998_848
    {
        return Err("grouped exact diagnostic envelope".into());
    }
    host_bound()?;
    let baseline = c::sample(client)?;
    main_admission::check_capacity(
        c::MAIN_CAP,
        &[baseline, banks as u64, c::PRIVATE_CAP as u64, 40 * MIB],
    )?;
    Ok(MainAdmission::new(c::MAIN_CAP, baseline)?)
}
struct Scalar {
    moe: ResidentMoe,
    input: BufferId,
    output: BufferId,
    retired: Arc<AtomicBool>,
}
struct ScalarValues {
    a: Zeroizing<Vec<u32>>,
    p: Zeroizing<Vec<u32>>,
    y: Zeroizing<Vec<u32>>,
}
impl Scalar {
    fn allocate(
        graph: BoundGraph<'_, '_>,
        layer: usize,
        arena: &mut PrivateArena,
        ledger: &mut MainAdmission,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        let b = MoeLayout::new(2560, 640)?.owned_bytes()?;
        ledger.precharge(&[b as u64, 32768, 32768])?;
        let input = arena.allocate_zeroed(2560)?;
        let output = arena.allocate_zeroed(2560)?;
        let moe = ResidentMoe::allocate(graph, layer, arena)?;
        ledger.observe_after(c::sample(arena.client()?)?)?;
        Ok(Self {
            moe,
            input,
            output,
            retired,
        })
    }
    fn step(
        &mut self,
        graph: BoundGraph<'_, '_>,
        arena: &PrivateArena,
        ledger: &mut MainAdmission,
        banks: &mut ExpertBanks,
        ids: &[usize],
        x: &[f32],
    ) -> Result<ScalarValues> {
        if ids.len() != 10 || x.len() != 2560 {
            return Err("grouped scalar input".into());
        }
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
            let cancel = AtomicBool::new(false);
            let bits = Zeroizing::new(x.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            c::write(arena, self.input, &bits, &self.retired)?;
            let w = self.moe.weights(graph)?;
            banks.check_client(arena.client()?)?;
            banks.check_names(w)?;
            let pending = banks.slots.plan(ids)?;
            banks.slots.validate_pending(&pending, ids)?;
            ledger.observe_before(c::sample(arena.client()?)?)?;
            ledger.precharge(&c::source_requests(w, pending.uploads.len())?)?;
            for &(e, s) in &pending.uploads {
                for (b, t) in [
                    (banks.gate(), w.gate),
                    (banks.up(), w.up),
                    (banks.down(), w.down),
                ] {
                    expert_banks::fill(arena.client()?, b, t, e, s, &cancel)?;
                }
            }
            let receipt = self.moe.enqueue(
                graph,
                arena,
                banks,
                self.input,
                self.output,
                &pending.slots,
                &PROB,
                &cancel,
                true,
                false,
            )?;
            let st = receipt.status();
            let status = c::read(arena, st.buffer, &self.retired)?;
            if status[..st.words].iter().any(|&x| x != 1)
                || status[st.words..].iter().any(|&x| x != 0)
            {
                return Err("grouped scalar status/tail".into());
            }
            let [(a, an), (p, pn)] = self.moe.fixed_public_probe_outputs();
            let values = ScalarValues {
                a: c::read(arena, a, &self.retired)?,
                p: c::read(arena, p, &self.retired)?,
                y: c::read(arena, self.output, &self.retired)?,
            };
            for (v, n) in [(&values.a, an), (&values.p, pn), (&values.y, 2560)] {
                if v[..n].iter().any(|&v| !f32::from_bits(v).is_finite())
                    || v[n..].iter().any(|&v| v != 0)
                {
                    return Err("grouped scalar full owning output".into());
                }
            }
            c::audit(arena.client()?, &self.retired)?;
            ledger.completed_fence(c::sample(arena.client()?)?)?;
            banks.slots.validate_pending(&pending, ids)?;
            banks.slots.commit(pending)?;
            self.moe.accept_after_whole_token_acceptance(receipt)?;
            Ok(values)
        }));
        match attempt {
            Ok(Ok(v)) => Ok(v),
            _ => {
                ledger.poison();
                self.moe.poison();
                Err("grouped scalar reference poisoned".into())
            }
        }
    }
}
fn prime(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    ledger: &mut MainAdmission,
    b: &mut ExpertBanks,
    s: &mut Scalar,
) -> Result<()> {
    for n in 0..9 {
        let ids: Vec<_> = (0..10).map(|r| n * 10 + r).collect();
        s.step(graph, arena, ledger, b, &ids, &input(0))?;
    }
    s.step(
        graph,
        arena,
        ledger,
        b,
        &[90, 91, 92, 93, 94, 95, 0, 1, 2, 3],
        &input(0),
    )?;
    let p = b.slots.plan(&BOUNDARY)?;
    b.slots.validate_pending(&p, &BOUNDARY)?;
    if p.slots != BOUNDARY || !p.uploads.is_empty() {
        return Err("grouped actual cross-segment priming".into());
    }
    Ok(())
}
fn tree(w: &[f32], x: &[f32]) -> f32 {
    let mut sums = Zeroizing::new([0.; 64]);
    for (l, s) in sums.iter_mut().enumerate() {
        for i in (l..x.len()).step_by(64) {
            *s += w[i] * x[i];
        }
    }
    let mut step = 32;
    while step > 0 {
        for l in 0..step {
            sums[l] += sums[l + step];
        }
        step /= 2;
    }
    sums[0]
}
fn projection(t: TensorView<'_>, e: usize, x: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
    let m = Matrix::new(t)?;
    if m.inputs() != x.len() || m.inputs() > 2560 || m.outputs() > 2560 || e >= 512 {
        return Err("grouped tree oracle geometry".into());
    }
    let row = t.info().row_bytes() as usize;
    let n = row
        .checked_mul(m.outputs())
        .ok_or("grouped oracle packed overflow")?;
    if n > 1_740_800 {
        return Err("grouped oracle one packed expert cap".into());
    }
    let mut packed = vec![0u8; n];
    t.read_range(
        e.checked_mul(n).ok_or("grouped oracle offset")? as u64,
        &mut packed,
    )?;
    let mut decoded = Zeroizing::new(vec![0.; x.len()]);
    let mut y = Zeroizing::new(Vec::with_capacity(m.outputs()));
    for r in packed.chunks_exact(row) {
        t.info().dtype().decode(r, &mut decoded)?;
        if decoded.iter().any(|v| !v.is_finite()) {
            return Err("grouped oracle original weight finite".into());
        }
        y.push(tree(&decoded, x));
    }
    Ok(y)
}
fn core(t: TensorView<'_>, e: usize, x: &[f32]) -> Result<Zeroizing<Vec<f32>>> {
    let m = Matrix::new(t)?;
    let mut y = Zeroizing::new(vec![0.; m.outputs()]);
    m.apply(e, x, &mut y, &AtomicBool::new(false))?;
    Ok(y)
}
fn oracle(w: RoutedWeights<'_>, plan: &token_waves::Plan) -> Result<Oracle> {
    host_bound()?;
    let count = plan.count();
    let mut a = Zeroizing::new(Vec::with_capacity(count * 10 * 640));
    let mut part = Zeroizing::new(Vec::with_capacity(count * 10 * 2560));
    let mut out = Zeroizing::new(vec![0.; count * 2560]);
    let mut co = Zeroizing::new(vec![0.; count * 2560]);
    for t in 0..count {
        let x = input(t);
        let row = plan.token(t).ok_or("grouped oracle token")?;
        for (r, route) in row.iter().enumerate() {
            let e = route.expert() as usize;
            let g = projection(w.gate, e, &x)?;
            let u = projection(w.up, e, &x)?;
            let act = Zeroizing::new(
                g.iter()
                    .zip(&*u)
                    .map(|(&g, &u)| kuno_text::ops::silu(g) * u)
                    .collect::<Vec<_>>(),
            );
            let d = projection(w.down, e, &act)?;
            a.extend_from_slice(&act);
            part.extend_from_slice(&d);
            let gc = core(w.gate, e, &x)?;
            let uc = core(w.up, e, &x)?;
            let ac = Zeroizing::new(
                gc.iter()
                    .zip(&*uc)
                    .map(|(&g, &u)| kuno_text::ops::silu(g) * u)
                    .collect::<Vec<_>>(),
            );
            let dc = core(w.down, e, &ac)?;
            for j in 0..2560 {
                out[t * 2560 + j] += route.probability() * part[(t * 10 + r) * 2560 + j];
                co[t * 2560 + j] += route.probability() * dc[j];
            }
        }
    }
    Ok(Oracle {
        activations: a,
        partials: part,
        output: out,
        core: co,
    })
}
fn compare(a: f32, b: f32, atol: f32, rtol: f32) -> Result<f32> {
    let e = (a - b).abs();
    if !a.is_finite() || !b.is_finite() || e > atol + rtol * b.abs() {
        return Err("grouped fixed numeric comparison".into());
    }
    Ok(e)
}
fn compare_oracle(
    l: &c::Layout,
    actual: &c::Snapshot,
    want: &Oracle,
    count: usize,
) -> Result<[f32; 4]> {
    let mut errors = [0f32; 4];
    for (field, first, stride, width, n, want, idx) in [
        (
            c::ACT,
            l.spans.act_first(),
            l.spans.act_stride(),
            640,
            count * 10,
            &want.activations,
            0,
        ),
        (
            c::PART,
            l.spans.partial_first(),
            l.spans.partial_stride(),
            2560,
            count * 10,
            &want.partials,
            1,
        ),
        (
            c::OUTPUT,
            l.spans.output_first(),
            l.spans.output_stride(),
            2560,
            count,
            &want.output,
            2,
        ),
    ] {
        for r in 0..n {
            for j in 0..width {
                errors[idx] = errors[idx].max(compare(
                    f32::from_bits(actual.0[field][first + r * stride + j]),
                    want[r * width + j],
                    1e-4,
                    3e-4,
                )?);
            }
        }
    }
    for t in 0..count {
        for j in 0..2560 {
            errors[3] = errors[3].max(compare(
                f32::from_bits(
                    actual.0[c::OUTPUT][l.spans.output_first() + t * l.spans.output_stride() + j],
                ),
                want.core[t * 2560 + j],
                1e-3,
                3e-3,
            )?);
        }
    }
    Ok(errors)
}
fn compare_scalar(
    l: &c::Layout,
    actual: &c::Snapshot,
    scalar: &ScalarValues,
    t: usize,
) -> Result<f32> {
    let mut max = 0f32;
    for (field, first, stride, width, ranks, values) in [
        (
            c::ACT,
            l.spans.act_first(),
            l.spans.act_stride(),
            640,
            10,
            &scalar.a,
        ),
        (
            c::PART,
            l.spans.partial_first(),
            l.spans.partial_stride(),
            2560,
            10,
            &scalar.p,
        ),
        (
            c::OUTPUT,
            l.spans.output_first(),
            l.spans.output_stride(),
            2560,
            1,
            &scalar.y,
        ),
    ] {
        for r in 0..ranks {
            for j in 0..width {
                let row = if field == c::OUTPUT { t } else { t * 10 + r };
                max = max.max(compare(
                    f32::from_bits(actual.0[field][first + row * stride + j]),
                    f32::from_bits(values[r * width + j]),
                    1e-4,
                    3e-4,
                )?);
            }
        }
    }
    Ok(max)
}
fn sentinels(
    client: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
    retired: &AtomicBool,
) -> Result<Vec<(Handle, Vec<u8>)>> {
    let mut samples = Vec::new();
    for b in [banks.gate(), banks.up(), banks.down()] {
        for s in b.segments() {
            let n = s.words() * 4;
            for first in [0, n - 16] {
                let h = s
                    .handle()
                    .clone()
                    .offset_start(first as u64)
                    .offset_end((n - first - 16) as u64);
                let v = client
                    .read_one(h.clone())
                    .map_err(|_| "grouped public sentinel read")?;
                if v.len() != 16 {
                    return Err("grouped public sentinel extent".into());
                }
                samples.push((h, v.to_vec()));
                drop(v);
            }
        }
    }
    c::audit(client, retired)?;
    Ok(samples)
}
fn cleanup(
    arena: &mut PrivateArena,
    client: &ComputeClient<WgpuRuntime>,
    retired: &AtomicBool,
    samples: Vec<(Handle, Vec<u8>)>,
) -> Result<()> {
    arena.reset()?;
    for (h, expected) in samples {
        let actual = client
            .read_one(h)
            .map_err(|_| "grouped preserved public sentinel read")?;
        if &*actual != expected.as_slice() {
            return Err("grouped private erasure changed public weights".into());
        }
        drop(actual);
    }
    c::audit(client, retired)?;
    if retired.load(Ordering::Acquire) {
        return Err("grouped physical context retired".into());
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn correct(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    arena: &mut PrivateArena,
    retired: &Arc<AtomicBool>,
    graph: BoundGraph<'_, '_>,
    layer: usize,
    dtype: TensorType,
    case: Case,
) -> Result<serde_json::Value> {
    let w = c::original_weights(graph, layer)?;
    let mut ledger = preflight(client, w, dtype, case, limits)?;
    let mut b = c::OriginalBank::allocate(graph, layer, client, &mut ledger, case.capacity)?;
    let mut flat = c::OriginalBank::allocate(graph, layer, client, &mut ledger, 10)?;
    let mut scalar = Scalar::allocate(graph, layer, arena, &mut ledger, retired.clone())?;
    if case.boundary {
        prime(graph, arena, &mut ledger, b.fixture_bank_mut(), &mut scalar)?;
    }
    let routes = plan(case)?;
    let l = c::Layout::original(dtype, case.tile, case.capacity)?;
    let x = input_image(&l, case.count, false);
    let want = oracle(w, &routes)?;
    let mut owner = c::Owner::allocate(
        graph,
        layer,
        arena,
        &mut ledger,
        l,
        &x,
        limits,
        b,
        retired.clone(),
    )?;
    if arena.charged_bytes() != 1_998_848 {
        return Err("grouped actual combined private page envelope".into());
    }
    let cancel = AtomicBool::new(false);
    let mut errors = [0.; 4];
    let actual = owner.execute(
        graph,
        arena,
        &mut ledger,
        &routes,
        &cancel,
        c::Failure::Never,
        |layout, v| {
            errors = compare_oracle(layout, v, &want, case.count)?;
            Ok(())
        },
    )?;
    if owner.phase() != c::Phase::AcceptedOperator {
        return Err("grouped operator phase".into());
    }
    owner.original_ledger_unchanged()?;
    let mut max_scalar = 0f32;
    for t in 0..case.count {
        let row = routes.token(t).ok_or("grouped scalar route token")?;
        let mut ids = Zeroizing::new([0usize; 10]);
        for (i, r) in row.iter().enumerate() {
            ids[i] = r.expert() as usize;
        }
        flat.validate(graph, layer, client)?;
        let values = scalar.step(
            graph,
            arena,
            &mut ledger,
            flat.fixture_bank_mut(),
            &ids[..],
            &input(t),
        )?;
        max_scalar = max_scalar.max(compare_scalar(owner.layout(), &actual, &values, t)?);
    }
    let mut public = sentinels(client, owner.banks(), retired)?;
    public.extend(sentinels(client, flat.banks(), retired)?);
    cleanup(arena, client, retired, public)?;
    owner.poison();
    scalar.moe.poison();
    drop(owner);
    drop(flat);
    client.memory_cleanup();
    Ok(
        serde_json::json!({"count":case.count,"overlap":case.overlap,"tile":case.tile,"capacity":case.capacity,
        "cross_segment_boundary":case.boundary,"bank_full_pages":expected_bank_bytes(dtype,case.capacity)?,"down_dtype":dtype.gguf_id(),"layer":layer,
        "tree_activation_partial_output_max_abs_errors":errors[..3],"core_output_max_abs_error":errors[3],"scalar_max_abs_error":max_scalar,
        "all_ten_ranks":true,"all_full_guards_and_inactive_tails":true,"original_bank_ledger_unchanged":true,"operator_only_no_publication":true}),
    )
}
#[derive(Clone, Copy)]
enum PhysicalFailure {
    Runtime(c::Failure),
    Alias,
    Stale,
}
#[allow(clippy::too_many_arguments)]
fn failure(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    arena: &mut PrivateArena,
    retired: &Arc<AtomicBool>,
    graph: BoundGraph<'_, '_>,
    layer: usize,
    dtype: TensorType,
    how: PhysicalFailure,
) -> Result<()> {
    let case = Case {
        count: 8,
        overlap: false,
        tile: 8,
        capacity: 16,
        boundary: false,
    };
    let w = c::original_weights(graph, layer)?;
    let mut ledger = preflight(client, w, dtype, case, limits)?;
    let b = c::OriginalBank::allocate(graph, layer, client, &mut ledger, 16)?;
    let routes = plan(case)?;
    let l = c::Layout::original(dtype, 8, 16)?;
    let late = matches!(how, PhysicalFailure::Runtime(c::Failure::LateInput));
    let x = input_image(&l, 8, late);
    let mut owner = c::Owner::allocate(
        graph,
        layer,
        arena,
        &mut ledger,
        l,
        &x,
        limits,
        b,
        retired.clone(),
    )?;
    if matches!(how, PhysicalFailure::Alias) {
        owner.fixed_public_alias_output_to_input()?;
    }
    if matches!(how, PhysicalFailure::Stale) {
        arena.reset()?;
    }
    let cancel = AtomicBool::new(false);
    let mode = match how {
        PhysicalFailure::Runtime(f) => f,
        _ => c::Failure::Never,
    };
    if owner
        .execute(graph, arena, &mut ledger, &routes, &cancel, mode, |_, _| {
            Ok(())
        })
        .is_ok()
        || owner.phase() != c::Phase::Poisoned
        || !ledger.is_poisoned()
    {
        return Err("grouped malformed/cancelled owner accepted".into());
    }
    owner.original_ledger_unchanged()?;
    if owner
        .execute(
            graph,
            arena,
            &mut ledger,
            &routes,
            &AtomicBool::new(false),
            c::Failure::Never,
            |_, _| Ok(()),
        )
        .is_ok()
    {
        return Err("grouped poisoned retry accepted".into());
    }
    let public = sentinels(client, owner.banks(), retired)?;
    cleanup(arena, client, retired, public)?;
    drop(owner);
    client.memory_cleanup();
    Ok(())
}
fn retire_failed_attempt<T>(attempt: &std::thread::Result<Result<T>>, retired: &AtomicBool) {
    if !matches!(attempt, Ok(Ok(_))) {
        retired.store(true, Ordering::Release);
    }
}
fn finish<T>(attempt: std::thread::Result<Result<T>>, clean: bool, retired: bool) -> Result<T> {
    if !clean || retired {
        return Err("grouped cleanup or retired context".into());
    }
    attempt.map_err(|_| "grouped fixed fixture panicked")?
}
pub fn run(client: &ComputeClient<WgpuRuntime>, limits: &wgpu::Limits) -> Result<()> {
    let host = host_bound()?;
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing original GGUF required")?;
    let file =
        GgufFile::open(path, GgufLimits::default()).map_err(|_| "grouped original GGUF open")?;
    let model = Qwen4Exp::bind(&file).map_err(|_| "grouped original graph bind")?;
    let graph = model.bound_graph();
    if expert_banks::preflight_residency(
        graph,
        96,
        6 * 1024 * 1024 * 1024,
        26 * 1024 * 1024 * 1024,
    )? != 18 * 1024 * 1024 * 1024
        || !matches!(
            expert_banks::preflight_residency(
                graph,
                96,
                6 * 1024 * 1024 * 1024,
                10 * 1024 * 1024 * 1024
            ),
            Err(Error::LimitExceeded(_))
        )
    {
        return Err("grouped normal full-model admission changed".into());
    }
    let mut layers = Vec::new();
    for dtype in [TensorType::Q5_0, TensorType::Q8_0] {
        let layer = (0..48)
            .find(|&i| {
                graph
                    .tensor(&format!("blk.{i}.ffn_down_exps.weight"))
                    .is_ok_and(|t| t.info().dtype() == dtype)
            })
            .ok_or("grouped both original down dtypes required")?;
        layers.push((layer, dtype));
    }
    let retired = Arc::new(AtomicBool::new(false));
    let mut arena =
        PrivateArena::new_transient(client.clone(), c::PRIVATE_CAP, c::IO_CAP, retired.clone())?;
    arena.enable_public_owned_zero_audit()?;
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<_> {
        let mut reports = Vec::new();
        let mut failures = 0;
        for (layer, dtype) in layers {
            for case in cases() {
                reports.push(correct(
                    client, limits, &mut arena, &retired, graph, layer, dtype, case,
                )?);
            }
            for how in [
                PhysicalFailure::Runtime(c::Failure::BeforePlan),
                PhysicalFailure::Runtime(c::Failure::FirstCopy),
                PhysicalFailure::Runtime(c::Failure::Enqueued),
                PhysicalFailure::Runtime(c::Failure::LateInput),
                PhysicalFailure::Alias,
                PhysicalFailure::Stale,
            ] {
                failure(
                    client, limits, &mut arena, &retired, graph, layer, dtype, how,
                )?;
                // Fresh independent owner must still pass after every rejected
                // fixture; no retry/resume of the partially changed bank.
                correct(
                    client,
                    limits,
                    &mut arena,
                    &retired,
                    graph,
                    layer,
                    dtype,
                    Case {
                        count: 1,
                        overlap: true,
                        tile: 8,
                        capacity: 16,
                        boundary: false,
                    },
                )?;
                failures += 1;
            }
        }
        Ok((reports, failures))
    }));
    retire_failed_attempt(&attempt, &retired);
    let closed = arena.close();
    let (reports, failures) = finish(attempt, closed.is_ok(), retired.load(Ordering::Acquire))?;
    let audit = arena
        .owned_zero_audit_receipt()
        .ok_or("grouped missing full MAIN cleanup audit")?;
    file.verify_file_unchanged()
        .map_err(|_| "grouped original file changed")?;
    let report = serde_json::json!({"scope":"fixed-public-grouped-routed-MoE-operator-only","cases":reports,"failure_cases":failures,
        "main_limit_bytes":c::MAIN_CAP,"private_limit_bytes":c::PRIVATE_CAP,"combined_private_pages":1_998_848,"io_cap_bytes":c::IO_CAP,"io_checkpoint_bound":c::IO_CHECKPOINT_BOUND,
        "host_scratch_upper_bound":host,"main_erased_bytes":audit.bytes,"main_erased_buffers":audit.buffers,
        "full_model_96_nvidia_rejected":true,"all_ten_original_ranks_and_probabilities":true,"no_original_ledger_publication":true,"shared_expert_or_FFN_or_full_graph_accepted":false,
        "performance_or_production_private_worker_accepted":false,"private_register_spill_driver_erasure_accepted":false,"context_retired":false});
    if let Some(path) = std::env::var_os("KUNO_PUBLIC_RECEIPT") {
        use std::io::Write;
        let mut out = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    }
    println!(
        "public_grouped_routed_acceptance cases=40 failures={} main_bytes={} main_buffers={}",
        failures, audit.bytes, audit.buffers
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_metadata_page_charges_and_guarded_host_envelopes_are_exact() {
        for (cap, q5, q8) in [
            (16, 112, 128),
            (64, 304, 320),
            (72, 304, 320),
            (96, 432, 448),
        ] {
            assert_eq!(
                expected_bank_bytes(TensorType::Q5_0, cap).unwrap(),
                q5 * MIB
            );
            assert_eq!(
                expected_bank_bytes(TensorType::Q8_0, cap).unwrap(),
                q8 * MIB
            );
        }
        assert_eq!(host_bound().unwrap(), 15_872_000);
        assert!(host_bound().unwrap() < c::HOST_CAP);
        assert_eq!(
            MoeLayout::new(2560, 640).unwrap().owned_bytes().unwrap() + 65536,
            327680
        );
        assert_eq!(
            c::Layout::original(TensorType::Q8_0, 8, 16)
                .unwrap()
                .page_bytes()
                .unwrap()
                + 327680,
            1_998_848
        );
    }
    #[test]
    fn fixed_matrix_cases_pin_actual_waves_and_boundary_slots_without_pruning() {
        let all = cases();
        assert_eq!(all.len(), 20);
        let mut boundary = 0;
        for case in all {
            let p = plan(case).unwrap();
            assert_eq!(p.count(), case.count);
            let want = if case.overlap {
                1
            } else if case.capacity == 16 {
                case.count
            } else if case.capacity == 64 {
                case.count.div_ceil(6)
            } else {
                1
            };
            assert_eq!(p.wave_count(), want);
            let mut used = Zeroizing::new([0u8; 80]);
            for i in 0..p.wave_count() {
                let wave = p.wave(i).unwrap();
                assert!(wave.protected_experts().len() <= case.capacity);
                for t in wave.tokens() {
                    for (r, route) in wave.token(t).unwrap().iter().enumerate() {
                        assert_eq!(route.probability().to_bits(), PROB[r].to_bits());
                        used[t * 10 + r] += 1;
                    }
                }
            }
            assert!(used[..case.count * 10].iter().all(|&n| n == 1));
            assert!(used[case.count * 10..].iter().all(|&n| n == 0));
            if case.boundary {
                boundary += 1;
                for e in [0, 63, 64, 95] {
                    assert!(p.wave(0).unwrap().protected_experts().contains(&e));
                }
            }
        }
        assert_eq!(boundary, 1);
    }
    #[test]
    fn independent_tree_is_original_schedule_and_numeric_comparator_rejects_nonfinite() {
        let w = (0..129)
            .map(|i| ((i * 17) % 31) as f32 / 13. - 1.)
            .collect::<Vec<_>>();
        let x = (0..129)
            .map(|i| ((i * 13) % 17) as f32 / 29. - 0.3)
            .collect::<Vec<_>>();
        let f64sum = w
            .iter()
            .zip(&x)
            .map(|(&w, &x)| f64::from(w) * f64::from(x))
            .sum::<f64>() as f32;
        compare(tree(&w, &x), f64sum, 1e-4, 3e-4).unwrap();
        assert!(compare(f32::NAN, 0., 1., 1.).is_err());
        assert!(compare(0., f32::INFINITY, 1., 1.).is_err());
        assert!(compare(1., 0., 1e-4, 1e-4).is_err());
        let l = c::Layout::original(TensorType::Q5_0, 4, 16).unwrap();
        let x = input_image(&l, 8, true);
        assert_eq!(
            x[l.spans.input_first() + 7 * l.spans.input_stride() + 2559],
            f32::NAN.to_bits()
        );
    }
    #[test]
    fn unexpected_error_and_panic_retire_before_cleanup_but_expected_negative_results_do_not() {
        let good: std::thread::Result<Result<()>> = Ok(Ok(()));
        let r = AtomicBool::new(false);
        retire_failed_attempt(&good, &r);
        assert!(!r.load(Ordering::Acquire));
        let error: std::thread::Result<Result<()>> =
            Ok(Err("fixed public comparator error".into()));
        retire_failed_attempt(&error, &r);
        assert!(r.load(Ordering::Acquire));
        let panic: std::thread::Result<Result<()>> = Err(Box::new(()));
        let r = AtomicBool::new(false);
        retire_failed_attempt(&panic, &r);
        assert!(r.load(Ordering::Acquire));
    }
}
