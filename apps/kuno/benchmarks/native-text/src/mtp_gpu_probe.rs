//! Fixed public Q3 target carry -> Q8 MTP projection parity, never a request API.
//!
//! Uses the existing ordinary, nonresident MatrixExecutor path unchanged.
//! Its logical activation clears are NOT a complete owning-span/I/O erasure
//! acceptance. CPU graph helpers also retain their existing public-only scope.
//! No draft throughput, speculative transaction or private-request claim follows.

use crate::executor::GpuExecutor;
use crate::expert_banks::OriginalPackedProfile;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::model::{MatrixExecutor, Qwen4Exp, greedy};
use kuno_text::mtp::{MtpGraph, MtpOutput};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{GgufFile, GgufLimits};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const TOKEN0: u32 = 0;
const VOCAB: usize = 248320;
const WIDE: usize = 10240;
const THREADS: usize = 8;
const CACHE_BYTES: usize = 2 * 1024 * 1024 * 1024;
const DEVICE_BYTES: usize = 4 * 1024 * 1024 * 1024;
const TILE_BYTES: usize = 64 * 1024 * 1024;
const TWO_POSITION_CACHE_BYTES: u64 = 1_252_130_816;
const ALL_PROJECTION_CACHE_BYTES: u64 = 4_347_527_168;
// Explicit full-graph tolerance; fixed before device execution. Greedies and
// fresh GPU replay remain exact gates. A failure never widens the tolerance.
const ABS_TOLERANCE: f32 = 0.006;
const REL_TOLERANCE: f32 = 0.0003;
const SHARDS: [&str; 3] = [
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
    "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
];

pub(crate) struct Inputs {
    directory: PathBuf,
    mtp: PathBuf,
    receipt: PathBuf,
}

pub(crate) fn default_build() -> bool {
    !cfg!(any(
        feature = "direct-device",
        feature = "cpu-fma",
        feature = "cpu-buffered-weights",
        feature = "cpu-work-stealing",
        feature = "cpu-q8-two-pass"
    ))
}

/// Run before device/model access. All Kuno experiment/input settings except
/// these four are rejected, including irrelevant flags explicitly set to zero.
pub(crate) fn registration(
    args: &[OsString],
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    default: bool,
) -> std::result::Result<Inputs, &'static str> {
    if !default
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "mtp-gpu-probe"
    {
        return Err("MTP probe requires default build, card and fixed public mode only");
    }
    let mut selected = BTreeMap::new();
    for (key, value) in environment {
        if !(key.as_encoded_bytes().starts_with(b"KUNO_LAB_")
            || key.as_encoded_bytes().starts_with(b"KUNO_TEST_")
            || key.as_encoded_bytes().starts_with(b"KUNO_PUBLIC_"))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_MTP_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_RECEIPT"
            )
        ) || value.is_empty()
            || selected.insert(key, value).is_some()
        {
            return Err("MTP probe rejects other input or laboratory settings");
        }
    }
    if selected.remove(OsStr::new("KUNO_LAB_MTP_PUBLIC_ONLY")) != Some(OsString::from("1")) {
        return Err("MTP probe requires its explicit public-fixture-only gate");
    }
    Ok(Inputs {
        directory: selected
            .remove(OsStr::new("KUNO_TEST_Q3_DIR"))
            .ok_or("MTP probe requires the verified original Q3 shard directory")?
            .into(),
        mtp: selected
            .remove(OsStr::new("KUNO_TEST_MTP"))
            .ok_or("MTP probe requires the verified original Q8 sidecar")?
            .into(),
        receipt: selected
            .remove(OsStr::new("KUNO_PUBLIC_RECEIPT"))
            .ok_or("MTP probe requires a new numeric receipt destination")?
            .into(),
    })
}

/// Selected process only: backend panics never print backend/source payloads.
/// This does not make a failed GPU context clean or recoverable in-process.
pub(crate) fn content_free_panic_hook() {
    std::panic::set_hook(Box::new(|_| eprintln!("public_mtp_gpu_failure panic=true")));
}

pub(crate) fn validate_limits(limits: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroups_per_dimension < 16384
        || limits.max_compute_workgroup_storage_size < 256
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_storage_buffer_binding_size < TILE_BYTES as u64
        || limits.max_buffer_size < TILE_BYTES as u64
    {
        return Err("MTP probe requires bounded original packed cooperative projection limits");
    }
    Ok(())
}

fn formats(file: &GgufFile) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    for tensor in file.tensors().values() {
        *counts.entry(tensor.dtype().gguf_id()).or_insert(0) += 1;
    }
    counts
}

fn require_artifact(
    file: &GgufFile,
    bytes: u64,
    tensors: usize,
    expected_formats: &[(u32, usize)],
) -> Result<()> {
    if file.file_len() != bytes
        || file.tensors().len() != tensors
        || formats(file) != expected_formats.iter().copied().collect()
    {
        return Err("MTP probe original artifact extent/formats rejected".into());
    }
    file.verify_file_unchanged()
        .map_err(|_| "MTP probe original artifact identity changed")?;
    Ok(())
}

fn matrix_cache_bytes(rows: u64, row_bytes: u64) -> Result<u64> {
    if rows == 0 || row_bytes == 0 || row_bytes > TILE_BYTES as u64 {
        return Err("MTP probe packed cache geometry rejected".into());
    }
    let tile_rows = (TILE_BYTES as u64 / row_bytes).min(16384);
    let mut first = 0;
    let mut bytes = 0u64;
    while first < rows {
        let count = tile_rows.min(rows - first);
        let logical = count
            .checked_mul(row_bytes)
            .ok_or("MTP probe cache overflow")?;
        let padded = logical.checked_add(3).ok_or("MTP probe cache overflow")? / 4 * 4;
        bytes = bytes
            .checked_add(padded.max(32768).next_power_of_two())
            .ok_or("MTP probe cache overflow")?;
        first += count;
    }
    Ok(bytes)
}

fn require_cache_geometry(file: &GgufFile) -> Result<()> {
    let mut dense = 0u64;
    let mut one_expert = 0u64;
    for tensor in file.tensors().values() {
        let dims = tensor.dimensions();
        if dims.len() < 2
            || tensor.name() == "token_embd.weight"
            || tensor.name().contains("indexer")
        {
            continue;
        }
        let bytes = matrix_cache_bytes(dims[1], tensor.row_bytes())?;
        if dims.len() == 3 {
            if dims[2] != 512 {
                return Err("MTP probe expert cache geometry rejected".into());
            }
            one_expert = one_expert
                .checked_add(bytes)
                .ok_or("MTP probe cache overflow")?;
        } else {
            dense = dense.checked_add(bytes).ok_or("MTP probe cache overflow")?;
        }
    }
    if dense + 20 * one_expert != TWO_POSITION_CACHE_BYTES
        || dense + 512 * one_expert != ALL_PROJECTION_CACHE_BYTES
        || TWO_POSITION_CACHE_BYTES > CACHE_BYTES as u64
    {
        return Err("MTP probe exact original packed cache charge rejected".into());
    }
    Ok(())
}

fn output_counts(output: &MtpOutput) -> Result<(usize, usize)> {
    if output.logits.len() != VOCAB
        || output.hidden.len() != WIDE
        || !output
            .logits
            .iter()
            .chain(output.hidden.iter())
            .all(|value| value.is_finite())
    {
        return Err("MTP probe incomplete/nonfinite output".into());
    }
    let counts = (
        output.logits.iter().filter(|v| v.abs() > 1e-6).count(),
        output.hidden.iter().filter(|v| v.abs() > 1e-6).count(),
    );
    if counts.0 == 0 || counts.1 == 0 {
        return Err("MTP probe zero output".into());
    }
    Ok(counts)
}

fn compare_values(reference: &[f32], actual: &[f32]) -> Result<f32> {
    if reference.is_empty() || reference.len() != actual.len() {
        return Err("MTP probe full-vector extent mismatch".into());
    }
    let mut max_error = 0f32;
    for (&r, &a) in reference.iter().zip(actual) {
        let error = (a - r).abs();
        if !r.is_finite() || !a.is_finite() || error > ABS_TOLERANCE + REL_TOLERANCE * r.abs() {
            return Err("MTP probe full-vector numerical mismatch".into());
        }
        max_error = max_error.max(error);
    }
    Ok(max_error)
}

fn bit_mismatches(reference: &[f32], replay: &[f32]) -> Result<usize> {
    if reference.len() != replay.len() {
        return Err("MTP probe replay extent mismatch".into());
    }
    Ok(reference
        .iter()
        .zip(replay)
        .filter(|(r, a)| r.to_bits() != a.to_bits())
        .count())
}

fn pair(
    graph: &MtpGraph<'_>,
    executor: &mut impl MatrixExecutor,
    token1: u32,
    hidden0: &[f32],
    cancel: &AtomicBool,
) -> Result<[MtpOutput; 2]> {
    let mut sequence = graph
        .sequence(2)
        .map_err(|_| "MTP probe sequence rejected")?;
    let zero = Zeroizing::new(vec![0.; WIDE]);
    let first = graph
        .step(executor, &mut sequence, TOKEN0, &zero, cancel)
        .map_err(|_| "MTP probe position0 execution rejected")?;
    // Position1 always consumes real target h0, never the draft's own h0.
    let second = graph
        .step(executor, &mut sequence, token1, hidden0, cancel)
        .map_err(|_| "MTP probe position1 execution rejected")?;
    if sequence.position() != 2 {
        return Err("MTP probe sequence position mismatch".into());
    }
    output_counts(&first)?;
    output_counts(&second)?;
    Ok([first, second])
}

fn gpu<'a>(file: &'a GgufFile, client: &ComputeClient<WgpuRuntime>) -> Result<GpuExecutor<'a>> {
    // The Q4 constructor's16 slots are unused: MTP never calls routed_moe.
    let mut executor = GpuExecutor::new(file, client.clone(), CACHE_BYTES, 16, DEVICE_BYTES)
        .map_err(|_| "MTP probe generic GPU executor rejected")?;
    executor.small_cpu_enabled = false;
    executor.block_kernel = false;
    Ok(executor)
}

fn gpu_counters(executor: &GpuExecutor<'_>) -> Result<serde_json::Value> {
    // 51 ordinary projections/position; the head's16 tiles add15 launches.
    if executor.calls != 132
        || executor.small_cpu_calls != 0
        || executor.moe_calls != 0
        || executor.gdn_calls != 0
        || executor.fast_dense_calls != 0
        || executor.shared_dense_calls != 0
        || executor.fused_expert_calls != 0
        || executor.gpu_router_calls != 0
        || executor.fast_dense_fault_calls != 0
        || executor.shared_overlap_calls != 0
        || executor.cpu_resident_expert_calls != 0
        || executor.expert_hits != 0
        || executor.expert_uploads != 0
        || executor.adaptive.is_some()
        || executor.secondary.is_some()
        || executor.secondary_frequency.is_some()
        || executor.small_cpu_enabled
        || executor.block_kernel
        || executor.block_moe
        || executor.fast_dense
        || executor.fast_dense_fault
        || executor.shared_kernels
        || executor.shared_gdn
        || executor.gpu_router
        || executor.fused_experts
        || executor.integer_experts
        || executor.word_down
        || executor.batch_experts
        || executor.ordered_public_upload
        || executor.uploaded_bytes == 0
        || executor.uploaded_bytes > TWO_POSITION_CACHE_BYTES
    {
        return Err("MTP probe generic projection counters rejected".into());
    }
    Ok(serde_json::json!({
        "projection_kernel_calls":executor.calls,
        "small_cpu_calls":executor.small_cpu_calls,
        "routed_bank_calls":executor.moe_calls,
        "uploaded_original_bytes":executor.uploaded_bytes,
        "cache_hits":executor.cache_hits
    }))
}

/// Public allocation release telemetry only, not an erasure/physical VRAM audit.
fn cleanup(client: &ComputeClient<WgpuRuntime>) -> Result<(u64, u64, u64, u64)> {
    cubecl::future::block_on(client.sync()).map_err(|_| "MTP probe GPU completion failed")?;
    client.memory_cleanup();
    cubecl::future::block_on(client.sync()).map_err(|_| "MTP probe GPU cleanup failed")?;
    let usage = client
        .memory_usage()
        .map_err(|_| "MTP probe GPU cleanup telemetry failed")?;
    Ok((
        usage.number_allocs,
        usage.bytes_in_use,
        usage.bytes_padding,
        usage.bytes_reserved,
    ))
}

fn released(before: (u64, u64, u64, u64), after: (u64, u64, u64, u64)) -> Result<()> {
    if after != before {
        return Err("MTP probe GPU allocation release incomplete".into());
    }
    Ok(())
}

pub(crate) fn run(client: &ComputeClient<WgpuRuntime>, vendor: u32, inputs: Inputs) -> Result<()> {
    let baseline = cleanup(client)?;
    if baseline != (0, 0, 0, 0) {
        return Err("MTP probe requires an empty main GPU allocator baseline".into());
    }
    // Full SHA verification remains the protected runner's provenance gate;
    // exact packed geometry/types/extent and pinned descriptor checks are local.
    let paths = SHARDS.map(|name| inputs.directory.join(name));
    let target_file = GgufFile::open_shards(&paths, GgufLimits::default())
        .map_err(|_| "MTP probe original Q3 shards rejected")?;
    require_artifact(
        &target_file,
        89_986_353_824,
        1224,
        &[
            (0, 557),
            (8, 502),
            (14, 1),
            (18, 94),
            (20, 44),
            (23, 2),
            (30, 24),
        ],
    )?;
    let target =
        Qwen4Exp::bind(&target_file).map_err(|_| "MTP probe Q3 target binding rejected")?;
    OriginalPackedProfile::UnslothQ3
        .validate_graph(target.bound_graph())
        .map_err(|_| "MTP probe original Q3 profile rejected")?;
    let mtp_file = GgufFile::open(&inputs.mtp, GgufLimits::default())
        .map_err(|_| "MTP probe original Q8 sidecar rejected")?;
    require_artifact(&mtp_file, 4_137_429_120, 34, &[(0, 11), (8, 21), (30, 2)])?;
    let draft = MtpGraph::bind(&mtp_file).map_err(|_| "MTP probe Q8 graph binding rejected")?;
    require_cache_geometry(&mtp_file)?;
    let cancel = AtomicBool::new(false);
    let mut cpu = ThreadedExecutor::new(THREADS).map_err(|_| "MTP probe CPU pool rejected")?;
    let mut target_sequence = target
        .sequence(1)
        .map_err(|_| "MTP probe target sequence rejected")?;
    let mut completed_layers = 0;
    let target0 = target
        .step_with_carry(&mut cpu, &mut target_sequence, TOKEN0, &cancel, |_| {
            completed_layers += 1;
        })
        .map_err(|_| "MTP probe real public target carry rejected")?;
    if completed_layers != 48
        || target0.binding_id() != target.bound_graph().binding_id()
        || target0.position() != 0
        || target_sequence.position() != 1
        || target0.logits().len() != VOCAB
        || target0.hidden().len() != WIDE
        || !target0
            .logits()
            .iter()
            .chain(target0.hidden())
            .all(|v| v.is_finite())
        || !target0.logits().iter().any(|v| v.abs() > 1e-6)
        || !target0.hidden().iter().any(|v| v.abs() > 1e-6)
    {
        return Err("MTP probe actual target carry extent/binding rejected".into());
    }
    let token1 = greedy(target0.logits()).map_err(|_| "MTP probe target greedy rejected")?;
    drop(target_sequence);
    let reference = pair(&draft, &mut cpu, token1, target0.hidden(), &cancel)?;
    drop(cpu);
    let mut executor = gpu(&mtp_file, client)?;
    let actual_result = pair(&draft, &mut executor, token1, target0.hidden(), &cancel);
    let counters_result = gpu_counters(&executor);
    drop(executor);
    let after_first = cleanup(client)?;
    released(baseline, after_first)?;
    let actual = actual_result?;
    let counters = counters_result?;
    let mut fresh = gpu(&mtp_file, client)?;
    let replay_result = pair(&draft, &mut fresh, token1, target0.hidden(), &cancel);
    let replay_counters_result = gpu_counters(&fresh);
    drop(fresh);
    let after_replay = cleanup(client)?;
    released(baseline, after_replay)?;
    let replay = replay_result?;
    let replay_counters = replay_counters_result?;
    let mut positions = Vec::with_capacity(2);
    for position in 0..2 {
        let counts = output_counts(&actual[position])?;
        let logits_error = compare_values(&reference[position].logits, &actual[position].logits)?;
        let hidden_error = compare_values(&reference[position].hidden, &actual[position].hidden)?;
        if greedy(&reference[position].logits).map_err(|_| "MTP probe CPU greedy rejected")?
            != greedy(&actual[position].logits).map_err(|_| "MTP probe GPU greedy rejected")?
        {
            return Err("MTP probe exact greedy parity rejected".into());
        }
        let logits_bits = bit_mismatches(&actual[position].logits, &replay[position].logits)?;
        let hidden_bits = bit_mismatches(&actual[position].hidden, &replay[position].hidden)?;
        if logits_bits != 0 || hidden_bits != 0 {
            return Err("MTP probe exact fresh GPU replay rejected".into());
        }
        positions.push(serde_json::json!({
            "position":position,"logit_values":VOCAB,"hidden_values":WIDE,
            "logits_nonzero":counts.0,"hidden_nonzero":counts.1,
            "max_logit_abs_error":logits_error,"max_hidden_abs_error":hidden_error,
            "greedy_matches":true,"replay_logit_bit_mismatches":logits_bits,
            "replay_hidden_bit_mismatches":hidden_bits
        }));
    }
    target_file
        .verify_file_unchanged()
        .map_err(|_| "MTP probe target source changed")?;
    mtp_file
        .verify_file_unchanged()
        .map_err(|_| "MTP probe sidecar source changed")?;
    let receipt = serde_json::json!({
        "scope":"fixed-public-target-carry-mtp-gpu-projection-parity",
        "target_model_bytes":89_986_353_824u64,"target_tensors":1224,
        "sidecar_bytes":4_137_429_120u64,"sidecar_tensors":34,
        "vendor":vendor,"target_positions":1,"cpu_draft_positions":2,
        "gpu_draft_positions":2,"fresh_gpu_draft_positions":2,
        "cpu_threads":THREADS,"cache_budget_bytes":CACHE_BYTES,"device_budget_bytes":DEVICE_BYTES,
        "two_position_cache_upper_bytes":TWO_POSITION_CACHE_BYTES,
        "all_executed_projection_cache_bytes":ALL_PROJECTION_CACHE_BYTES,
        "abs_tolerance":ABS_TOLERANCE,"rel_tolerance":REL_TOLERANCE,
        "positions":positions,"gpu":counters,"fresh_gpu":replay_counters,
        "allocator_baseline_allocations":baseline.0,"allocator_baseline_in_use_bytes":baseline.1,
        "allocator_baseline_padding_bytes":baseline.2,"allocator_baseline_reserved_bytes":baseline.3,
        "allocator_after_allocations":after_replay.0,"allocator_after_in_use_bytes":after_replay.1,
        "allocator_after_padding_bytes":after_replay.2,"allocator_after_reserved_bytes":after_replay.3,
        "main_allocator_cleanup_complete":true,"separate_staging_uniform_cleanup_audited":false,
        "whole_owning_erasure_accepted":false,"private_request_accepted":false,
        "fully_gpu_graph_accepted":false,"independent_mtp_numeric_oracle_accepted":false,
        "speculative_controller_accepted":false,"performance_accepted":false
    });
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(inputs.receipt)
        .map_err(|_| "MTP probe numeric receipt destination rejected")?;
    serde_json::to_writer_pretty(output, &receipt)
        .map_err(|_| "MTP probe numeric receipt write failed")?;
    println!(
        "public_mtp_gpu_complete target_positions=1 cpu_draft_positions=2 gpu_draft_positions=2 fresh_gpu_draft_positions=2 full_logits={VOCAB} full_hidden={WIDE} greedy_matches=true replay_bit_mismatches=0 main_allocator_cleanup=true whole_owning_erasure=false private_request=false performance=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Vec<OsString> {
        ["lab", "nvidia", "mtp-gpu-probe"]
            .map(OsString::from)
            .into()
    }
    fn environment() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_MTP_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_Q3_DIR", "/public/q3"),
            ("KUNO_TEST_MTP", "/public/mtp.gguf"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt.json"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .into()
    }
    #[test]
    fn registration_rejects_other_arguments_builds_and_all_irrelevant_settings() {
        assert!(registration(&args(), environment(), true).is_ok());
        assert!(registration(&args(), environment(), false).is_err());
        let mut extra = args();
        extra.push("arbitrary prompt".into());
        assert!(registration(&extra, environment(), true).is_err());
        for key in [
            "KUNO_LAB_TRANSIENT_IO",
            "KUNO_LAB_BLOCK_KERNEL",
            "KUNO_TEST_GGUF",
            "KUNO_PUBLIC_Q3_REFERENCE",
            "KUNO_LAB_FUTURE_EXPERIMENT",
        ] {
            let mut vars = environment();
            vars.push((key.into(), "0".into()));
            assert!(registration(&args(), vars, true).is_err());
        }
        for index in 0..4 {
            let mut vars = environment();
            vars.remove(index);
            assert!(registration(&args(), vars, true).is_err());
        }
        let mut vars = environment();
        vars[0].1 = "0".into();
        assert!(registration(&args(), vars, true).is_err());
    }
    #[test]
    fn full_vector_gate_checks_last_values_nonfinite_and_exact_replay_bits() {
        let values = [0.1, -2., 3.];
        assert_eq!(compare_values(&values, &values).unwrap(), 0.);
        assert!(compare_values(&values, &[0.1, -2., 3.1]).is_err());
        assert!(compare_values(&values, &[0.1, -2., f32::NAN]).is_err());
        assert!(compare_values(&values, &[0.1, -2.]).is_err());
        assert_eq!(bit_mismatches(&[0.], &[-0.]).unwrap(), 1);
        assert!(bit_mismatches(&values, &[0.1]).is_err());
    }
    #[test]
    fn allocation_release_cannot_hide_any_retained_page_or_allocation() {
        assert!(released((0, 0, 0, 0), (0, 0, 0, 0)).is_ok());
        for after in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)] {
            assert!(released((0, 0, 0, 0), after).is_err());
        }
        let head = matrix_cache_bytes(248320, 2720).unwrap();
        let expert = matrix_cache_bytes(640, 2720).unwrap();
        let first_two = head + 111_280_128 + 20 * 3 * expert;
        let all_experts = head + 111_280_128 + 512 * 3 * expert;
        assert_eq!(first_two, TWO_POSITION_CACHE_BYTES);
        assert_eq!(all_experts, ALL_PROJECTION_CACHE_BYTES);
        assert!(first_two < CACHE_BYTES as u64);
        assert!(all_experts > CACHE_BYTES as u64);
        assert_eq!(head, 1_015_021_568);
        assert_eq!(expert, 2_097_152);
        assert_eq!(matrix_cache_bytes(2560, 680).unwrap(), 2_097_152);
        assert!(matrix_cache_bytes(1, TILE_BYTES as u64 + 1).is_err());
    }
    #[test]
    fn all_output_values_and_required_cooperative_limits_are_bounded() {
        let mut output = MtpOutput {
            logits: Zeroizing::new(vec![1.; VOCAB]),
            hidden: Zeroizing::new(vec![1.; WIDE]),
        };
        assert_eq!(output_counts(&output).unwrap(), (VOCAB, WIDE));
        output.hidden[WIDE - 1] = f32::INFINITY;
        assert!(output_counts(&output).is_err());
        output.hidden[WIDE - 1] = 1.;
        output.logits.pop();
        assert!(output_counts(&output).is_err());
        let mut limits = wgpu::Limits::default();
        assert!(validate_limits(&limits).is_ok());
        limits.max_compute_workgroups_per_dimension = 16383;
        assert!(validate_limits(&limits).is_err());
    }
}
