//! Fixed-public parity for the actual prompt-attention specialization against
//! its unchanged scalar shader. Genuine private arena owners only; no prompt
//! scope, model receipt, prediction, request data or timing is manufactured.
use super::*;
use crate::{gr_batch_probe, private_arena::BufferId};
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// QSA_ATTENTION_PARITY_ORACLE_BEGIN
const OWNER_ROWS: usize = 8;
const QUERY_WORDS: usize = 6144;
const SELECT_WORDS: usize = 2051;
const PARTIAL_ROW: usize = 202752;
const STATS_ROW: usize = 1584;
const ATTENTION_CHUNKS: usize = 33;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Control {
    Causal,
    FutureSelection,
    NonfiniteQuery,
}
impl Control {
    fn label(self) -> &'static str {
        match self {
            Self::Causal => "causal",
            Self::FutureSelection => "future_selection",
            Self::NonfiniteQuery => "nonfinite_query",
        }
    }
    fn fault(self) -> u32 {
        u32::from(self != Self::Causal)
    }
}

fn geometry(start: usize, count: usize, capacity: usize) -> bool {
    (1..=OWNER_ROWS).contains(&count)
        && (4..=32768).contains(&capacity)
        && capacity.is_multiple_of(4)
        && start.checked_add(count).is_some_and(|end| end <= capacity)
}

fn offset_geometry(start: usize, count: usize, capacity: usize, offset: usize) -> bool {
    geometry(start, count, capacity)
        && [0, 32, 64].contains(&offset)
        && start
            .checked_add(offset)
            .and_then(|v| v.checked_add(count))
            .is_some_and(|end| end <= capacity)
}

fn selected_rows(start: usize, count: usize) -> Vec<u32> {
    selected_rows_at(start, count, 0)
}
fn selected_rows_at(start: usize, count: usize, offset: usize) -> Vec<u32> {
    let mut selected = vec![u32::MAX; (offset + OWNER_ROWS) * SELECT_WORDS];
    for row in 0..count {
        let query_row = row + offset;
        let visible = start + query_row + 1;
        let width = visible.min(SELECT_WORDS);
        for i in 0..width {
            selected[query_row * SELECT_WORDS + i] = (visible - width + i) as u32;
        }
    }
    selected
}

/// Actual shader coordinates: [query][KV head][chunk][12 heads][256 or2].
/// Every unlaunched query/chunk and complete physical owner tail stays zero.
fn inactive_zero(words: &[u32], start: usize, count: usize, stats: bool) -> bool {
    let row_words = if stats { STATS_ROW } else { PARTIAL_ROW };
    let per_slot = if stats { 12 * 2 } else { 12 * 256 };
    if words.len() < OWNER_ROWS * row_words || !geometry(start, count, 32768) {
        return false;
    }
    words.iter().enumerate().all(|(i, &word)| {
        let row = i / row_words;
        if row >= count {
            return word == 0;
        }
        let chunk = (i % row_words / per_slot) % ATTENTION_CHUNKS;
        let active_chunks = (start + row + 1).min(SELECT_WORDS).div_ceil(64);
        chunk < active_chunks || word == 0
    })
}

fn fault_words_match(words: &[u32], control: Control) -> bool {
    words.first().copied() == Some(control.fault()) && words[1..].iter().all(|&word| word == 0)
}

#[derive(Default, Debug)]
struct Difference {
    mismatches: usize,
    max_abs: f64,
    max_relative: f64,
    max_ulp: u32,
    nonfinite_mismatches: usize,
}
fn numeric_difference(reference: &[u32], actual: &[u32]) -> Difference {
    let mut result = Difference {
        mismatches: reference.len().abs_diff(actual.len()),
        ..Default::default()
    };
    for (&expected, &observed) in reference.iter().zip(actual) {
        if expected == observed {
            continue;
        }
        result.mismatches += 1;
        let expected = f32::from_bits(expected);
        let observed = f32::from_bits(observed);
        if !expected.is_finite() || !observed.is_finite() {
            result.nonfinite_mismatches += 1;
            continue;
        }
        let error = (f64::from(expected) - f64::from(observed)).abs();
        result.max_abs = result.max_abs.max(error);
        // Diagnosis only, never an acceptance threshold. A finite floor keeps
        // the metadata finite when the scalar reference is exactly zero.
        result.max_relative = result
            .max_relative
            .max(error / f64::from(expected).abs().max(1e-30));
        let ordered = |value: f32| {
            let bits = if value == 0.0 { 0 } else { value.to_bits() };
            if bits & 0x80000000 != 0 {
                !bits
            } else {
                bits | 0x80000000
            }
        };
        result.max_ulp = result
            .max_ulp
            .max(ordered(expected).abs_diff(ordered(observed)));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn public_selection_has_exact_causal_rows_and_poisoned_unused_cells() {
        assert_eq!(QUERY_WORDS, 24 * 256);
        assert_eq!(Control::Causal.label(), "causal");
        for (start, count, capacity) in [
            (0, 1, 4096),
            (31, 3, 4096),
            (60, 4, 4096),
            (62, 8, 4096),
            (2047, 7, 4096),
            (4094, 8, 32768),
            (32760, 8, 32768),
        ] {
            assert!(geometry(start, count, capacity));
            let selected = selected_rows(start, count);
            for row in 0..OWNER_ROWS {
                if row >= count {
                    assert!(
                        selected[row * SELECT_WORDS..(row + 1) * SELECT_WORDS]
                            .iter()
                            .all(|&v| v == u32::MAX)
                    );
                    continue;
                }
                let visible = start + row + 1;
                let width = visible.min(SELECT_WORDS);
                let live = &selected[row * SELECT_WORDS..row * SELECT_WORDS + width];
                assert!(live.windows(2).all(|v| v[1] == v[0] + 1));
                assert_eq!(live.last().copied(), Some((visible - 1) as u32));
                assert!(
                    selected[row * SELECT_WORDS + width..(row + 1) * SELECT_WORDS]
                        .iter()
                        .all(|&v| v == u32::MAX)
                );
            }
        }
        for (start, count, capacity) in [
            (0, 0, 4096),
            (0, 9, 4096),
            (4095, 2, 4096),
            (usize::MAX, 1, 32768),
            (0, 1, 0),
            (0, 1, 4095),
        ] {
            assert!(!geometry(start, count, capacity));
        }
    }
    #[test]
    fn independent_whole_owner_checker_covers_both_kv_heads_chunks_and_physical_tails() {
        for stats in [false, true] {
            let per_slot = if stats { 12 * 2 } else { 12 * 256 };
            let row_words = if stats { STATS_ROW } else { PARTIAL_ROW };
            let mut words = vec![0; OWNER_ROWS * row_words + 11];
            for (start, count) in [(0, 1), (62, 3), (2047, 7), (32760, 8)] {
                words.fill(0);
                assert!(inactive_zero(&words, start, count, stats));
                for row in 0..count {
                    let chunks = (start + row + 1).min(SELECT_WORDS).div_ceil(64);
                    for kvh in 0..2 {
                        let live =
                            row * row_words + (kvh * ATTENTION_CHUNKS + chunks - 1) * per_slot;
                        words[live] = 1;
                        assert!(inactive_zero(&words, start, count, stats));
                        words[live] = 0;
                        if chunks < ATTENTION_CHUNKS {
                            let dead =
                                row * row_words + (kvh * ATTENTION_CHUNKS + chunks) * per_slot;
                            words[dead] = 1;
                            assert!(!inactive_zero(&words, start, count, stats));
                            words[dead] = 0;
                        }
                    }
                }
                if count < OWNER_ROWS {
                    words[count * row_words] = 1;
                    assert!(!inactive_zero(&words, start, count, stats));
                    words[count * row_words] = 0;
                }
                *words.last_mut().unwrap() = 1;
                assert!(!inactive_zero(&words, start, count, stats));
            }
        }
        assert!(fault_words_match(&[0, 0], Control::Causal));
        assert!(fault_words_match(&[1, 0], Control::FutureSelection));
        assert!(fault_words_match(&[1, 0], Control::NonfiniteQuery));
        assert!(!fault_words_match(&[1, 1], Control::NonfiniteQuery));
        assert!(!fault_words_match(&[0, 0], Control::NonfiniteQuery));
        assert!(!fault_words_match(&[], Control::Causal));
    }
    #[test]
    fn actual_query_offsets_and_numeric_diagnosis_do_not_relax_bitwise_gate() {
        for (start, count, offset) in [(17, 5, 0), (31, 3, 32), (1983, 8, 64)] {
            assert!(offset_geometry(start, count, 4096, offset));
            let selected = selected_rows_at(start, count, offset);
            assert_eq!(selected.len(), (offset + OWNER_ROWS) * SELECT_WORDS);
            assert!(
                selected[..offset * SELECT_WORDS]
                    .iter()
                    .all(|&v| v == u32::MAX)
            );
            for row in offset..offset + count {
                let visible = start + row + 1;
                let width = visible.min(SELECT_WORDS);
                assert_eq!(
                    selected[row * SELECT_WORDS + width - 1],
                    (visible - 1) as u32
                );
            }
            assert!(
                selected[(offset + count) * SELECT_WORDS..]
                    .iter()
                    .all(|&v| v == u32::MAX)
            );
        }
        for offset in [1, 31, 33, 63, 65, usize::MAX] {
            assert!(!offset_geometry(0, 1, 4096, offset));
        }
        assert!(!offset_geometry(32767, 1, 32768, 32));
        let same = numeric_difference(&[0, 1f32.to_bits()], &[0, 1f32.to_bits()]);
        assert_eq!(same.mismatches, 0);
        let changed = numeric_difference(
            &[1f32.to_bits(), (-2f32).to_bits(), 0, f32::NAN.to_bits()],
            &[
                1f32.to_bits() + 1,
                (-1f32).to_bits(),
                (-0f32).to_bits(),
                f32::INFINITY.to_bits(),
            ],
        );
        assert_eq!(changed.mismatches, 4);
        assert_eq!(changed.max_abs, 1.0);
        assert_eq!(changed.max_relative, 0.5);
        assert_eq!(changed.max_ulp, 8388608);
        assert_eq!(changed.nonfinite_mismatches, 1);
        assert_eq!(numeric_difference(&[0], &[0, 0]).mismatches, 1);
    }
}
// QSA_ATTENTION_PARITY_ORACLE_END

/// Distinct diagnostic shader identity: this can never reuse the specialized
/// production Kernel's cache entry, even though both bind the original owners.
struct ScalarAttention;
impl KernelMetadata for ScalarAttention {
    fn name(&self) -> &'static str {
        "StrataPublicOriginalScalarPromptAttention"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>().cube_dim(CubeDim::new_1d(256))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for ScalarAttention {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: Kernel::PromptAttentionFault.scalar_source(),
            repr: None,
            cube_dim: CubeDim::new_1d(256),
            debug_info: None,
        })
    }
}

fn scalar_launch(arena: &PrivateArena, owners: [Handle; 7], p: [u32; 12]) -> ProbeResult<()> {
    let mut args = KernelArguments::new();
    if owners.iter().any(|h| h.stream != owners[0].stream) {
        return Err("public QSA original scalar client identity".into());
    }
    for owner in owners {
        if !arena.contains_private_view(&owner)? {
            return Err("public QSA original scalar private owner".into());
        }
        args = args.with_buffer(owner.binding());
    }
    args = args.with_info(
        MetadataBindingInfo::custom(
            p.chunks_exact(2)
                .map(|v| v[0] as u64 | ((v[1] as u64) << 32))
                .collect(),
        )
        .with_native_parameter(
            1,
            cubecl_runtime::server::NativeStrataValue::PhasePosition { offset: 0 },
        ),
    );
    // SAFETY: only this fixed-public caller supplies the genuine full owners;
    // its geometry/lengths are checked before either scalar or candidate launch.
    // This grants no prompt scope, accepted state, model nonce or capture key.
    unsafe {
        arena.client()?.launch_unchecked(
            Box::new(ScalarAttention),
            CubeCount::Static(ATTENTION_CHUNKS as u32, 2, p[0]),
            args,
        );
    }
    Ok(())
}

fn read_owner(arena: &PrivateArena, id: BufferId) -> ProbeResult<Zeroizing<Vec<u32>>> {
    let result = gr_batch_probe::read_native_public(arena, id)?;
    if result.len() != arena.layout(id)?.owned_words {
        return Err("public QSA attention full owner read extent".into());
    }
    Ok(result)
}

fn check(
    arena: &mut PrivateArena,
    start: usize,
    count: usize,
    capacity: usize,
    control: Control,
    query_offset: usize,
    nondyadic: bool,
) -> ProbeResult<(serde_json::Value, bool)> {
    if !offset_geometry(start, count, capacity, query_offset)
        || !crate::strata_intel_core::enabled()
        || crate::strata_prompt::active()
    {
        return Err("public QSA attention fixed geometry".into());
    }
    let input_rows = query_offset + OWNER_ROWS;
    let effective_start = start + query_offset;
    let mut queries = Zeroizing::new(vec![0u32; input_rows * QUERY_WORDS]);
    for (i, value) in queries.iter_mut().enumerate() {
        let row = i / QUERY_WORDS;
        *value = if nondyadic {
            if !(query_offset..query_offset + count).contains(&row) {
                // A wrong query-row read must fault, not coincidentally use
                // another finite row of this deterministic public fixture.
                f32::NAN.to_bits()
            } else {
                ((((i * 17 + row * 3) % 43) as f32 - 21.0) / 31.0
                    + ((i / 256 % 7) as f32 - 3.0) / 127.0)
                    .to_bits()
            }
        } else {
            ((((i * 17 + row * 3) % 43) as f32 - 21.0) / 32.0).to_bits()
        };
    }
    if control == Control::NonfiniteQuery {
        queries[0] = f32::NAN.to_bits();
    }
    let mut selected = Zeroizing::new(if query_offset == 0 {
        selected_rows(start, count)
    } else {
        selected_rows_at(start, count, query_offset)
    });
    if control == Control::FutureSelection {
        selected[0] = (start + 1) as u32;
    }
    let mut cache = Zeroizing::new(vec![0u32; capacity * CACHE_WORDS]);
    for token in 0..capacity {
        for word in 0..256 {
            let mut codes = 0;
            for byte in 0..4 {
                let code = ((token * 7 + (word * 4 + byte) * 13) % 127) as i32 - 63;
                codes |= u32::from(code as i8 as u8) << (byte * 8);
            }
            cache[token * CACHE_WORDS + word] = codes;
        }
        for word in 0..8 {
            // Preserve the first nine fixtures exactly. The additional three
            // use finite non-power-of-two FP16 significands with nontrivial
            // F32 queries so rounding cannot hide behind power-of-two inputs.
            let scales = if nondyadic {
                [0x2113u32, 0x2487, 0x299d, 0x2c53]
            } else {
                [0x2000u32, 0x2400, 0x2800, 0x2c00]
            };
            cache[token * CACHE_WORDS + 256 + word] =
                scales[(token + word * 2) % 4] | (scales[(token + word * 2 + 1) % 4] << 16);
        }
    }
    let query = arena.upload_u32(&queries)?;
    let cache = arena.upload_u32(&cache)?;
    let selected = arena.upload_u32(&selected)?;
    let candidate = arena.allocate_zeroed(OWNER_ROWS * CACHE_WORDS)?;
    let reference = [
        arena.allocate_zeroed(OWNER_ROWS * PARTIAL_ROW)?,
        arena.allocate_zeroed(OWNER_ROWS * STATS_ROW)?,
        arena.allocate_zeroed(1)?,
    ];
    let actual = [
        arena.allocate_zeroed(OWNER_ROWS * PARTIAL_ROW)?,
        arena.allocate_zeroed(OWNER_ROWS * STATS_ROW)?,
        arena.allocate_zeroed(1)?,
    ];
    let owners = |output: [BufferId; 3]| -> ProbeResult<[Handle; 7]> {
        let ids = [
            query, cache, candidate, selected, output[0], output[1], output[2],
        ];
        let logical = [
            input_rows * QUERY_WORDS,
            capacity * CACHE_WORDS,
            OWNER_ROWS * CACHE_WORDS,
            input_rows * SELECT_WORDS,
            OWNER_ROWS * PARTIAL_ROW,
            OWNER_ROWS * STATS_ROW,
            1,
        ];
        for (index, (&id, expected)) in ids.iter().zip(logical).enumerate() {
            let layout = arena.layout(id)?;
            let owner = arena.handle(id)?;
            if layout.logical_words != expected
                || owner.size_in_used() != layout.owned_bytes as u64
                || ids[..index].contains(&id)
            {
                return Err("public QSA exact original role owner geometry".into());
            }
            for &previous in &ids[..index] {
                if crate::strata_prompt::storage::overlaps(arena.handle(previous)?, owner)? {
                    return Err("public QSA overlapping original role owners".into());
                }
            }
        }
        Ok([
            arena.handle(query)?.clone(),
            arena.handle(cache)?.clone(),
            arena.handle(candidate)?.clone(),
            arena.handle(selected)?.clone(),
            arena.handle(output[0])?.clone(),
            arena.handle(output[1])?.clone(),
            arena.handle(output[2])?.clone(),
        ])
    };
    let p = [
        count as u32,
        start as u32,
        capacity as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        query_offset as u32,
    ];
    scalar_launch(arena, owners(reference)?, p)?;
    // Actual production kernel selection and original launch validation.
    super::launch(
        arena,
        Kernel::PromptAttentionFault,
        [ATTENTION_CHUNKS, 2, count],
        owners(actual)?.to_vec(),
        p,
    )?;
    let reference_partial = read_owner(arena, reference[0])?;
    let reference_stats = read_owner(arena, reference[1])?;
    let reference_fault = read_owner(arena, reference[2])?;
    let actual_partial = read_owner(arena, actual[0])?;
    let actual_stats = read_owner(arena, actual[1])?;
    let actual_fault = read_owner(arena, actual[2])?;
    let partial_diff = numeric_difference(&reference_partial, &actual_partial);
    let stats_diff = numeric_difference(&reference_stats, &actual_stats);
    // Fixed-public aggregate numerical diagnosis only. This does not set any
    // tolerance or change the original bitwise gate, including signed zeros.
    println!(
        "public_device_error qsa_attention_math_numeric control={control:?} start={start} count={count} query_offset={query_offset} nondyadic={nondyadic} partial_mismatches={} partial_max_abs={} partial_max_relative={} partial_max_ulp={} partial_nonfinite_mismatches={} stats_mismatches={} stats_max_abs={} stats_max_relative={} stats_max_ulp={} stats_nonfinite_mismatches={}",
        partial_diff.mismatches,
        partial_diff.max_abs,
        partial_diff.max_relative,
        partial_diff.max_ulp,
        partial_diff.nonfinite_mismatches,
        stats_diff.mismatches,
        stats_diff.max_abs,
        stats_diff.max_relative,
        stats_diff.max_ulp,
        stats_diff.nonfinite_mismatches
    );
    if !inactive_zero(&actual_partial, effective_start, count, false)
        || !inactive_zero(&actual_stats, effective_start, count, true)
        || !inactive_zero(&reference_partial, effective_start, count, false)
        || !inactive_zero(&reference_stats, effective_start, count, true)
    {
        return Err("public QSA attention inactive row chunk or owner padding changed".into());
    }
    if !fault_words_match(&reference_fault, control)
        || !fault_words_match(&actual_fault, control)
        || reference_fault.as_slice() != actual_fault.as_slice()
    {
        return Err("public QSA attention original fault parity failed".into());
    }
    let bit_exact = reference_partial.as_slice() == actual_partial.as_slice()
        && reference_stats.as_slice() == actual_stats.as_slice();
    Ok((
        serde_json::json!({"family":"prompt-qsa-attention-scalar-parity","control":control.label(),"start":start,"count":count,"capacity":capacity,"query_offset":query_offset,"nondyadic_operands":nondyadic,"fault":control.fault(),"bit_exact_partial_and_stats":bit_exact,"full_owner_and_tail_checked":true,"actual_prompt_attention_fault_kernel":true,"unchanged_scalar_shader_reference":true,"unique_reference_kernel_identity":true,"original_host_launch_validation":true,"fabricated_prompt_scope":false}),
        bit_exact,
    ))
}

pub(crate) fn cases(arena: &mut PrivateArena) -> ProbeResult<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    let mut all_bit_exact = true;
    for (start, count, capacity, control, offset, nondyadic) in [
        (0, 1, 4096, Control::Causal, 0, false),
        (31, 3, 4096, Control::Causal, 0, false),
        (60, 4, 4096, Control::Causal, 0, false),
        (62, 8, 4096, Control::Causal, 0, false),
        (2047, 7, 4096, Control::Causal, 0, false),
        (4094, 8, 32768, Control::Causal, 0, false),
        (32760, 8, 32768, Control::Causal, 0, false),
        (62, 3, 4096, Control::FutureSelection, 0, false),
        (62, 3, 4096, Control::NonfiniteQuery, 0, false),
        (17, 5, 4096, Control::Causal, 0, true),
        (31, 3, 4096, Control::Causal, 32, true),
        (1983, 8, 4096, Control::Causal, 64, true),
    ] {
        let attempt = check(arena, start, count, capacity, control, offset, nondyadic);
        arena.reset()?;
        let (report, bit_exact) = attempt?;
        all_bit_exact &= bit_exact;
        reports.push(report);
    }
    if !all_bit_exact {
        return Err("public QSA attention original scalar bit parity failed".into());
    }
    Ok(reports)
}
