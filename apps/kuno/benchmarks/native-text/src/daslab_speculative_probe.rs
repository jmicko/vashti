//! Fixed public greedy target/MTP on two original devices. No HTTP path.
//! Only real target-verified continuations enter the accepted-token count.
use crate::{
    daslab_carry_reference13 as reference, daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    resident_model::{
        ResidentModel,
        batch::{Config, LatestTargetRound},
    },
    resident_mtp,
};
use cubecl::{
    prelude::*,
    wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device},
};
use kuno_text::{
    GgufFile, GgufLimits,
    model::{Qwen4Exp, greedy},
    mtp::MtpGraph,
};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use zeroize::Zeroizing;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) struct Inputs {
    pub(crate) inline_dispatch: bool,
    pub(crate) fused_sweeps: bool,
    warm_replay: bool,
    pub(crate) verify3_dense: bool,
    pub(crate) verify3_fused: bool,
    pub(crate) prepared_experts: bool,
    public_existing_fill: bool,
    chunk_io: bool,
    request_reuse: bool,
    live_member_reduction: bool,
    device_router: bool,
    directory: PathBuf,
    mtp: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "daslab-speculative-probe" {
        return Err("speculative lab requires fixed public B70 target");
    }
    let mut values = BTreeMap::new();
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|p| key.as_encoded_bytes().starts_with(p))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_SPECULATIVE_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_DEVICE_DISPATCH"
                    | "KUNO_LAB_MTP_SWEEPS"
                    | "KUNO_LAB_WARM_REPLAY"
                    | "KUNO_LAB_TARGET_DENSE"
                    | "KUNO_LAB_TARGET_EXPERTS"
                    | "KUNO_LAB_TARGET_FILL"
                    | "KUNO_LAB_TARGET_IO"
                    | "KUNO_LAB_TARGET_REUSE"
                    | "KUNO_LAB_TARGET_MOE"
                    | "KUNO_LAB_TARGET_ROUTER"
            )
        ) || value.is_empty()
            || values.insert(key, value).is_some()
        {
            return Err("speculative lab rejects inherited experiments");
        }
    }
    if values.remove(OsStr::new("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY")) != Some("1".into())
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES")) != Some("8589934592".into())
    {
        return Err("speculative fixed public/cache gates missing");
    }
    let raw_dispatch = values.remove(OsStr::new("KUNO_LAB_DEVICE_DISPATCH"));
    let inline_dispatch = build.selected(raw_dispatch.as_deref())?;
    let prepared_experts = match values.remove(OsStr::new("KUNO_LAB_TARGET_EXPERTS")) {
        None => false,
        Some(value) if value == "prepared-verify3" => true,
        _ => return Err("target expert policy requires canonical prepared-verify3"),
    };
    let raw_moe = values.remove(OsStr::new("KUNO_LAB_TARGET_MOE"));
    let live_member_reduction =
        crate::target_lab_build::live_member_reduction(raw_moe.as_deref(), prepared_experts)?;
    let public_existing_fill = match values.remove(OsStr::new("KUNO_LAB_TARGET_FILL")) {
        None => false,
        Some(value) if value == "direct-public" => true,
        _ => return Err("target fill policy requires canonical direct-public"),
    };
    let chunk_io = match values.remove(OsStr::new("KUNO_LAB_TARGET_IO")) {
        None => false,
        Some(value) if value == "chunk" => true,
        _ => return Err("target I/O policy requires canonical chunk"),
    };
    let raw_reuse = values.remove(OsStr::new("KUNO_LAB_TARGET_REUSE"));
    let request_reuse =
        crate::target_lab_build::request_reuse(raw_reuse.as_deref(), inline_dispatch, chunk_io)?;
    let (verify3_dense, verify3_fused) = match values.remove(OsStr::new("KUNO_LAB_TARGET_DENSE")) {
        None => (false, false),
        Some(value) if value == "verify3" => (true, false),
        Some(value) if value == "verify3-fused" => (true, true),
        _ => return Err("target dense policy requires canonical verify3 or verify3-fused"),
    };
    let warm_replay = match values.remove(OsStr::new("KUNO_LAB_WARM_REPLAY")) {
        None => false,
        Some(value) if value == "once" => true,
        _ => return Err("warm replay requires canonical once"),
    };
    let device_router = match values.remove(OsStr::new("KUNO_LAB_TARGET_ROUTER")) {
        None => false,
        Some(v)
            if v == "device-warm-public" && warm_replay && request_reuse && !prepared_experts =>
        {
            true
        }
        _ => return Err("device router requires protected original warm request scope"),
    };
    let fused_sweeps = match values.remove(OsStr::new("KUNO_LAB_MTP_SWEEPS")) {
        None => false,
        Some(value) if value == "fused" => true,
        _ => return Err("MTP sweep policy requires canonical fused"),
    };
    let mut path = |name: &str| {
        values
            .remove(OsStr::new(name))
            .map(PathBuf::from)
            .ok_or("speculative public path missing")
    };
    Ok(Inputs {
        inline_dispatch,
        fused_sweeps,
        warm_replay,
        verify3_dense,
        verify3_fused,
        prepared_experts,
        public_existing_fill,
        chunk_io,
        request_reuse,
        live_member_reduction,
        device_router,
        directory: path("KUNO_TEST_Q3_DIR")?,
        mtp: path("KUNO_TEST_MTP")?,
        reference: path("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}

/// All rows are real target outputs. A first mismatch contributes only the
/// target correction; later provisional tokens are discarded.
fn accepted_prefix(inputs: &[u32], predicted: &[u32]) -> std::result::Result<usize, &'static str> {
    if !(1..=3).contains(&inputs.len())
        || predicted.len() != inputs.len()
        || inputs.iter().chain(predicted).any(|&t| t >= 248320)
    {
        return Err("speculative acceptance geometry");
    }
    for row in 1..inputs.len() {
        if inputs[row] != predicted[row - 1] {
            return Ok(row);
        }
    }
    Ok(inputs.len())
}

async fn draft_client(fused_sweeps: bool) -> Result<(ComputeClient<WgpuRuntime>, wgpu::Limits)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .filter(|a| {
            a.get_info().vendor == 0x10de
                && a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
        })
        .collect();
    if adapters.len() != 1 {
        return Err("speculative draft GPU ambiguous".into());
    }
    let adapter = adapters.remove(0);
    let info = adapter.get_info();
    let limits = adapter.limits();
    resident_mtp::validate_limits(&limits)?;
    if fused_sweeps {
        resident_mtp::validate_fused_sweep_limits(&limits)?;
    }
    if !adapter.features().contains(wgpu::Features::SUBGROUP)
        || info.subgroup_min_size < 16
        || info.subgroup_max_size > 32
    {
        return Err("speculative draft subgroups rejected".into());
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Kuno fixed public native MTP"),
            required_features: wgpu::Features::SUBGROUP,
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            ..Default::default()
        })
        .await
        .map_err(|_| "speculative draft device setup failed")?;
    let selected = init_device(
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: wgpu::Backend::Vulkan,
        },
        RuntimeOptions {
            tasks_max: 256,
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    Ok((WgpuRuntime::client(&selected), limits))
}

fn process_disk_read_bytes() -> Result<u64> {
    let fields = std::fs::read_to_string("/proc/self/io")?;
    fields
        .lines()
        .find_map(|line| line.strip_prefix("read_bytes: "))
        .ok_or("public fixture process I/O counter missing")?
        .trim()
        .parse()
        .map_err(Into::into)
}

fn fill_counter_record(c: crate::executor::batch_moe::PublicFillCounters) -> serde_json::Value {
    serde_json::json!({"first_touch_fills":c.first_touch_fills,"evictions":c.evictions,
        "direct_windows":c.direct_windows,"direct_components":c.direct_components,
        "direct_bytes":c.direct_bytes})
}

pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    input: Inputs,
) -> Result<()> {
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_speculative_failure panic=true")
    }));
    let expected = reference::Reference::load(&input.reference)?;
    let paths: Vec<_> = (1..=2)
        .map(|i| {
            input.directory.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.dtype().gguf_id())),
    )
    .map_err(|_| "original DAS inventory rejected")?;
    let sidecar = GgufFile::open(&input.mtp, GgufLimits::default())?;
    if sidecar.file_len() != 4_137_429_120 || sidecar.tensors().len() != 34 {
        return Err("original draft artifact rejected".into());
    }
    // SAFETY: exclusive protected runner verifies immutable static artifacts.
    unsafe {
        file.enable_mapped_weights()?;
        sidecar.enable_mapped_weights()?;
    }
    let graph = Qwen4Exp::bind(&file)?;
    let draft_graph = MtpGraph::bind(&sidecar)?;
    crate::resident_util::select_retained_metadata(true);
    crate::resident_util::select_compute_scratch_clear(true);
    if input.request_reuse {
        client.configure_public_target_request_io()?;
    } else {
        client.configure_transient_io(128 * 1024 * 1024)?;
    }
    let (draft_client, draft_limits) = cubecl::future::block_on(draft_client(input.fused_sweeps))?;
    draft_client.configure_transient_io(64 * 1024 * 1024)?;
    if draft_client.public_target_request_io_configured()? {
        return Err("target request configuration leaked to draft stream".into());
    }
    let target_retired = Arc::new(AtomicBool::new(false));
    let draft_retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let load_disk_before = process_disk_read_bytes()?;
    let loading = Instant::now();
    let mut draft = if input.fused_sweeps {
        resident_mtp::ResidentMtp::new_with_fused_sweeps(
            &draft_graph,
            draft_client,
            &draft_limits,
            draft_retired.clone(),
        )?
    } else {
        resident_mtp::ResidentMtp::new(
            &draft_graph,
            draft_client,
            &draft_limits,
            draft_retired.clone(),
        )?
    };
    let mut executor = GpuExecutor::new_with_profile(
        &file,
        client.clone(),
        8 * 1024 * 1024 * 1024,
        144,
        31 * 1024 * 1024 * 1024,
        OriginalPackedProfile::DaslabIq3,
    )?;
    executor.target_rows_reserved = true;
    executor.target_batch_router_reserved = input.device_router;
    executor.target_verify3_dense = input.verify3_dense;
    executor.target_verify3_fused = input.verify3_fused;
    executor.token_io_checkpoint = input.chunk_io;
    executor.enable_transient_resident(target_retired.clone())?;
    executor.fast_dense = true;
    executor.fast_dense_fault = true;
    executor.shared_kernels = true;
    executor.shared_gdn = true;
    let mut passes = Vec::with_capacity(if input.warm_replay { 2 } else { 1 });
    let mut warm_reset = None;
    let attempt = (|| -> Result<()> {
        // Keep one healthy admission. The optional one-shot reset retains its
        // ledger/epochs/public banks and audits every actual private owner.
        let mut config = Config::new(client, limits, 8)?
            .reserve_target_rows()?
            .reserve_latest_target_checkpoint()?;
        if input.prepared_experts {
            config = config.reserve_prepared_verify3_experts()?;
        }
        if input.request_reuse {
            config = config.reserve_public_target_request_io()?;
        }
        if input.live_member_reduction {
            config = config.with_live_member_reduction()?;
        }
        if input.public_existing_fill {
            config = config.with_public_existing_expert_fill()?;
        }
        if input.warm_replay {
            config = config.reserve_public_warm_replay()?;
        }
        if input.device_router {
            config = config.reserve_public_warm_batch_router()?;
        }
        let mut target = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            reference::CONTEXT,
            &mut executor,
            config,
            true,
        )?;
        let runs = (|| -> Result<()> {
            for replay in 0..if input.warm_replay { 2 } else { 1 } {
                if input.device_router && replay == 1 {
                    target.activate_public_warm_batch_router(&executor, &cancel)?;
                }
                let replay_disk_before = if replay == 0 {
                    load_disk_before
                } else {
                    process_disk_read_bytes()?
                };
                let replay_prefill = Instant::now();
                let pass = (|| -> Result<serde_json::Value> {
                    let initial = target.prefill_chunk_with_carry(&mut executor, &[0], &cancel)?;
                    if initial.start() != 0
                        || initial.count() != 1
                        || initial.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("speculative actual initial range rejected".into());
                    }
                    expected.verify(0, initial.logits(0)?, initial.hidden(0)?)?;
                    let mut confirmed = greedy(initial.logits(0)?)?;
                    let mut carry = Zeroizing::new(initial.hidden(0)?.to_vec());
                    drop(initial);
                    if target.completed_target_rows_cursor(&executor)? != 1 || draft.position() != 0
                    {
                        return Err("speculative initial real cursor rejected".into());
                    }
                    if target.native_live_member_reduction(&executor)?
                        != input.live_member_reduction
                    {
                        return Err("target initial live reduction geometry mismatch".into());
                    }
                    let load_ns = loading.elapsed().as_nanos() as u64;
                    let initial_prefill_ns = replay_prefill.elapsed().as_nanos() as u64;
                    let mut accepted_tokens = 0usize;
                    let mut proposed_tokens = 0usize;
                    let mut accepted_drafts = 0usize;
                    let mut rounds = Vec::new();
                    let before_traffic = draft.traffic();
                    let before_uploads = executor.expert_uploads;
                    let before_hits = executor.expert_hits;
                    let before_gpu_routers = executor.gpu_router_calls;
                    let before_bytes = executor.uploaded_bytes;
                    let before_dense = executor.target_verify3_dense_calls;
                    let before_fused = executor.target_verify3_fused_calls;
                    let load_disk_after = process_disk_read_bytes()?;
                    let before_weight_reads = executor.weight_read_ns;
                    let before_weight_uploads = executor.weight_upload_ns;
                    let before_phases = target.phase_ns.clone();
                    let (actual_direct, before_fills) =
                        target.native_public_fill_counters(&executor)?;
                    if actual_direct != input.public_existing_fill {
                        return Err("public fill frozen selection rejected".into());
                    }
                    let before_host_io = crate::target_lab_build::host_io_snapshot();
                    let start = Instant::now();
                    if input.request_reuse {
                        target.begin_public_target_request_io(&mut executor, &cancel)?;
                        if !target.public_target_request_io_active()? {
                            return Err("public target request scope did not activate".into());
                        }
                    }
                    while target.completed_target_rows_cursor(&executor)? < reference::POSITIONS {
                        let position = target.completed_target_rows_cursor(&executor)?;
                        let count = (reference::POSITIONS - position).min(3);
                        if confirmed != expected.input_token(position)?
                            || draft.position() != position - 1
                        {
                            return Err("speculative actual input/cell alignment rejected".into());
                        }
                        let round_start = Instant::now();
                        let mut inputs = Zeroizing::new(vec![confirmed]);
                        let drafting = Instant::now();
                        if count > 1 {
                            let output = draft.step_draft(confirmed, &carry, &cancel)?;
                            if output.start() != position - 1 {
                                return Err("speculative first draft cell rejected".into());
                            }
                            inputs.push(output.greedy());
                            drop(output);
                            if count > 2 {
                                let output = draft.follow_draft(&cancel)?;
                                if output.start() != position {
                                    return Err("speculative second draft cell rejected".into());
                                }
                                inputs.push(output.greedy());
                                drop(output);
                            }
                        }
                        let draft_ns = drafting.elapsed().as_nanos() as u64;
                        let verifying = Instant::now();
                        let mut round =
                            LatestTargetRound::begin(&mut target, &mut executor, &inputs, &cancel)?;
                        let rows = round.verify(&cancel)?;
                        if rows.start() != position
                            || rows.count() != count
                            || rows.binding_id() != graph.bound_graph().binding_id()
                        {
                            return Err("speculative verification identity rejected".into());
                        }
                        let mut predicted = Zeroizing::new(Vec::with_capacity(count));
                        for row in 0..count {
                            predicted.push(greedy(rows.logits(row)?)?);
                        }
                        let accepted = accepted_prefix(&inputs, &predicted)?;
                        let mut real_carries =
                            Zeroizing::new(Vec::with_capacity(accepted * reference::HIDDEN));
                        for row in 0..accepted {
                            expected.verify(
                                position + row,
                                rows.logits(row)?,
                                rows.hidden(row)?,
                            )?;
                            real_carries.extend_from_slice(rows.hidden(row)?);
                        }
                        let next_confirmed = predicted[accepted - 1];
                        drop(rows);
                        let verify_ns = verifying.elapsed().as_nanos() as u64;
                        let repair = Instant::now();
                        if let Some(recomputed) = round.finish(accepted, &cancel)? {
                            if recomputed.start() != position || recomputed.count() != accepted {
                                return Err("speculative repaired range rejected".into());
                            }
                            for row in 0..accepted {
                                expected.verify(
                                    position + row,
                                    recomputed.logits(row)?,
                                    recomputed.hidden(row)?,
                                )?;
                            }
                            drop(recomputed);
                        }
                        if target.completed_target_rows_cursor(&executor)? != position + accepted {
                            return Err("speculative repaired actual cursor rejected".into());
                        }
                        if count > 1 {
                            // First cell used real target carry. Keep that exact KV;
                            // replace only subsequent provisional cells with reality.
                            draft.rewind(position)?;
                            for row in 1..accepted {
                                draft.append_kv(
                                    inputs[row],
                                    &real_carries
                                        [(row - 1) * reference::HIDDEN..row * reference::HIDDEN],
                                    &cancel,
                                )?;
                            }
                        } else {
                            draft.append_kv(confirmed, &carry, &cancel)?;
                        }
                        if draft.position() != position + accepted - 1 {
                            return Err("speculative corrected draft cursor rejected".into());
                        }
                        carry.copy_from_slice(
                            &real_carries
                                [(accepted - 1) * reference::HIDDEN..accepted * reference::HIDDEN],
                        );
                        confirmed = next_confirmed;
                        accepted_tokens += accepted;
                        proposed_tokens += count - 1;
                        accepted_drafts += accepted - 1;
                        rounds.push(serde_json::json!({"target_start":position,"proposed_drafts":count-1,"accepted_drafts":accepted-1,"accepted_outputs":accepted,"draft_ns":draft_ns,"verify_ns_including_public_vector_checks":verify_ns,"repair_ns":repair.elapsed().as_nanos() as u64,"total_ns":round_start.elapsed().as_nanos() as u64,"target_completed_cursor":position+accepted,"draft_completed_cell":draft.position()}));
                    }
                    if target.native_live_member_reduction(&executor)?
                        != input.live_member_reduction
                    {
                        return Err("target completed live reduction geometry mismatch".into());
                    }
                    let request_close = Instant::now();
                    let mut request_scope_receipt = None;
                    if input.request_reuse {
                        // Full128MiB owning I/O audit is part of throughput,
                        // before any public warm reset or final model close.
                        request_scope_receipt = Some(crate::target_lab_build::reuse_record(
                            target.end_public_target_request_io(&mut executor, &cancel)?,
                        ));
                        if target.public_target_request_io_active()? {
                            return Err("public target request scope did not close".into());
                        }
                    }
                    let request_close_ns = if input.request_reuse {
                        request_close.elapsed().as_nanos() as u64
                    } else {
                        0
                    };
                    let elapsed_ns = start.elapsed().as_nanos() as u64;
                    let host_io = crate::target_lab_build::host_io_delta(
                        before_host_io,
                        crate::target_lab_build::host_io_snapshot(),
                    )?;
                    if accepted_tokens != 12 {
                        return Err("speculative actual accepted token count rejected".into());
                    }
                    if input.verify3_dense && executor.target_verify3_dense_calls == before_dense {
                        return Err("target Verify3 actual dispatch evidence missing".into());
                    }
                    if input.verify3_fused && executor.target_verify3_fused_calls == before_fused {
                        return Err("target fused Verify3 actual dispatch evidence missing".into());
                    }
                    let host_phase_ns: BTreeMap<_, _> = target
                        .phase_ns
                        .iter()
                        .map(|(name, value)| {
                            (*name, value - before_phases.get(name).copied().unwrap_or(0))
                        })
                        .collect();
                    let timed_physical_read_bytes = process_disk_read_bytes()?
                        .checked_sub(load_disk_after)
                        .ok_or("public process read counter regressed")?;
                    let after = draft.traffic();
                    let (actual_direct, after_fills) =
                        target.native_public_fill_counters(&executor)?;
                    if actual_direct != input.public_existing_fill {
                        return Err("public fill frozen selection changed".into());
                    }
                    Ok(
                        serde_json::json!({"replay":replay,"device_router":input.device_router && replay == 1,"device_router_scope":"conditional actual all-hit warm13; strict original IDs and full vectors, experimental F32 probabilities","request_scope_closed_and_audited":input.request_reuse,"request_scope":request_scope_receipt,"request_close_ns_inside_timing":request_close_ns,"host_io":host_io,"model_weights_warm":replay==1,"initial_prefill_ns":initial_prefill_ns,"gpu_router_calls":executor.gpu_router_calls-before_gpu_routers,"verify3_dense_calls":executor.target_verify3_dense_calls-before_dense,"verify3_fused_calls":executor.target_verify3_fused_calls-before_fused,"target_host_phase_ns":host_phase_ns,"load_and_initial_prefill_physical_read_bytes":load_disk_after.checked_sub(replay_disk_before).ok_or("public process read counter regressed")?,"timed_physical_read_bytes":timed_physical_read_bytes,"target_fill_counters_before":fill_counter_record(before_fills),"target_fill_counters_after":fill_counter_record(after_fills),"weight_read_ns":executor.weight_read_ns-before_weight_reads,"weight_upload_enqueue_ns":executor.weight_upload_ns-before_weight_uploads,"timing_scope":if input.device_router && replay == 1 {"conditional actual all-hit GPU routing; deferred original-route audit, public checks and actual repair included"} else {"original normal routing/demand uploads; public checks and actual repair included"},"load_and_initial_prefill_cumulative_ns":load_ns,"accepted_tokens":accepted_tokens,"elapsed_ns":elapsed_ns,"accepted_tokens_per_second":accepted_tokens as f64*1e9/elapsed_ns as f64,"proposed_drafts":proposed_tokens,"accepted_drafts":accepted_drafts,"rounds":rounds,"expert_uploads":executor.expert_uploads-before_uploads,"expert_hits":executor.expert_hits-before_hits,"uploaded_bytes":executor.uploaded_bytes-before_bytes,"draft_traffic":{"host_read_calls":after.host_read_calls-before_traffic.host_read_calls,"host_read_bytes":after.host_read_bytes-before_traffic.host_read_bytes,"explicit_dispatches":after.explicit_dispatches-before_traffic.explicit_dispatches,"private_upload_calls":after.private_upload_calls-before_traffic.private_upload_calls,"private_upload_bytes":after.private_upload_bytes-before_traffic.private_upload_bytes,"minimal_packet_reads":after.minimal_packet_reads-before_traffic.minimal_packet_reads,"followup_copy_calls":after.followup_copy_calls-before_traffic.followup_copy_calls}}),
                    )
                })()?;
                passes.push(pass);
                if input.warm_replay && replay == 0 {
                    // All guarded outputs and the old host carry are dropped.
                    // The reset performs complete actual private/source audits.
                    let receipt = target.reset_public_warm_replay(&mut executor, &cancel)?;
                    draft.rewind(0)?;
                    // The typed reset commits target cursor0; the existing
                    // completed-projection proof intentionally requires >0.
                    // Next pass proves start0/count1/cursor1 with real prefill.
                    if draft.position() != 0 {
                        return Err("public warm replay empty draft state rejected".into());
                    }
                    warm_reset = Some(
                        serde_json::json!({"main_buffers":receipt.main.buffers,"main_bytes":receipt.main.bytes,"source_buffers":receipt.source.buffers,"source_bytes":receipt.source.bytes,"admission_and_public_banks_retained":true,"transient_binding_info_caches_retained":false,"ledger_epoch_or_poison_reset":false}),
                    );
                }
            }
            Ok(())
        })();
        let close = target.close();
        let audit = target.public_owned_zero_audit_receipt();
        drop(target);
        close?;
        let (main, sources) = audit.ok_or("speculative target full owning erasure rejected")?;
        runs?;
        let record = passes
            .last_mut()
            .ok_or("public controller completion missing")?;
        record["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,"source_buffers":sources.buffers,"source_bytes":sources.bytes});
        Ok(())
    })();
    let target_close = executor.close_transient_resident();
    drop(executor);
    let draft_close = draft.close();
    drop(draft);
    target_close?;
    let draft_close = draft_close?;
    attempt?;
    if target_retired.load(Ordering::Acquire) || draft_retired.load(Ordering::Acquire) {
        return Err("speculative physical stream retired".into());
    }
    file.verify_file_unchanged()?;
    sidecar.verify_file_unchanged()?;
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    if client.public_target_request_io_configured()? != input.request_reuse {
        return Err("target request stream frozen configuration mismatch".into());
    }
    let value = serde_json::json!({"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"scope":"fixed-public-DASLab13-real-dual-GPU-greedy-speculation","target_card":"B70","draft_card":"RTX3060","verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"live_member_reduction":input.live_member_reduction,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"request_reuse_metadata_reserved_bytes":if input.request_reuse {4*1024*1024}else{0},"request_binding_capacity":if input.request_reuse {4096}else{0},"fused_mtp_sweeps":input.fused_sweeps,"warm_replay":input.warm_replay,"warm_reset":warm_reset,"compiled_token0":0,"seed":424242,"temperature":0,"oracle_forcing":false,"expert_route_forcing":false,"passes":passes,"draft_close":{"private_buffers":draft_close.private.buffers,"private_bytes":draft_close.private.bytes,"public_bytes":draft_close.public_bytes,"main_reserved":draft_close.main_reserved},"private_or_production_accepted":false});
    serde_json::to_writer_pretty(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.receipt)?,
        &value,
    )?;
    println!("public_speculative_complete all_accepted_vectors=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "/original"),
            ("KUNO_TEST_MTP", "/original/sidecar"),
            ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "/public/carry"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
    }
    #[test]
    fn device_router_requires_original_pure_inline_one_shot_warm_request() {
        let args = ["lab", "intel", "daslab-speculative-probe"].map(OsString::from);
        let mut values = env();
        for (key, value) in [
            ("KUNO_LAB_DEVICE_DISPATCH", "inline"),
            ("KUNO_LAB_TARGET_IO", "chunk"),
            ("KUNO_LAB_TARGET_REUSE", "request"),
            ("KUNO_LAB_WARM_REPLAY", "once"),
            ("KUNO_LAB_TARGET_ROUTER", "device-warm-public"),
        ] {
            values.push((key.into(), value.into()));
        }
        assert!(
            registration(
                &args,
                values.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .device_router
        );
        for missing in [
            "KUNO_LAB_DEVICE_DISPATCH",
            "KUNO_LAB_TARGET_IO",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_LAB_WARM_REPLAY",
        ] {
            let subset = values.iter().filter(|(k, _)| k != missing).cloned();
            assert!(registration(&args, subset, crate::target_lab_build::Build::Inline).is_err());
        }
        let mut integer = values.clone();
        integer.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(registration(&args, integer, crate::target_lab_build::Build::Inline).is_err());
        assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
    }

    #[test]
    fn real_greedy_acceptance_keeps_only_prefix_and_target_correction() {
        assert_eq!(accepted_prefix(&[7, 8, 9], &[8, 9, 10]), Ok(3));
        assert_eq!(accepted_prefix(&[7, 8, 9], &[8, 11, 12]), Ok(2));
        assert_eq!(accepted_prefix(&[7, 8, 9], &[11, 9, 12]), Ok(1));
        assert_eq!(accepted_prefix(&[7, 8], &[8, 9]), Ok(2));
        assert_eq!(accepted_prefix(&[7], &[8]), Ok(1));
        for (x, y) in [
            (vec![], vec![]),
            (vec![1, 2, 3, 4], vec![1, 2, 3, 4]),
            (vec![1, 2], vec![1]),
            (vec![1, 248320], vec![1, 2]),
        ] {
            assert!(accepted_prefix(&x, &y).is_err());
        }
    }
    #[test]
    fn registration_is_exclusive_default_public_dual_gpu() {
        let args: Vec<OsString> = ["binary", "intel", "daslab-speculative-probe"]
            .map(Into::into)
            .to_vec();
        let env = || {
            [
                ("KUNO_LAB_SPECULATIVE_PUBLIC_ONLY", "1"),
                ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
                ("KUNO_TEST_Q3_DIR", "/original"),
                ("KUNO_TEST_MTP", "/original/sidecar"),
                ("KUNO_PUBLIC_TARGET_CARRY13_REFERENCE", "/public/carry"),
                ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
            ]
            .map(|(k, v)| (k.into(), v.into()))
            .to_vec()
        };
        assert!(registration(&args, env(), crate::target_lab_build::Build::Threaded).is_ok());
        let mut live = env();
        live.push(("KUNO_LAB_TARGET_MOE".into(), "live".into()));
        assert!(
            registration(
                &args,
                live.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .unwrap()
            .live_member_reduction
        );
        live.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(registration(&args, live, crate::target_lab_build::Build::Threaded).is_err());
        for value in ["", "auto", "1", "live "] {
            let mut invalid = env();
            invalid.push(("KUNO_LAB_TARGET_MOE".into(), value.into()));
            assert!(
                registration(&args, invalid, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut reusable = env();
        reusable.extend([
            ("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()),
            ("KUNO_LAB_TARGET_IO".into(), "chunk".into()),
            ("KUNO_LAB_TARGET_REUSE".into(), "request".into()),
        ]);
        assert!(
            registration(
                &args,
                reusable.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .request_reuse
        );
        assert!(
            registration(
                &args,
                reusable.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        for key in ["KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO"] {
            let mut missing = reusable.clone();
            missing.retain(|(k, _)| k != key);
            assert!(registration(&args, missing, crate::target_lab_build::Build::Inline).is_err());
        }
        for raw in ["", "auto", "1", "request ", "chunk"] {
            let mut wrong = reusable.clone();
            for (k, v) in &mut wrong {
                if k == "KUNO_LAB_TARGET_REUSE" {
                    *v = raw.into();
                }
            }
            assert!(registration(&args, wrong, crate::target_lab_build::Build::Inline).is_err());
        }
        let mut inline = env();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert!(
            registration(
                &args,
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .inline_dispatch
        );
        assert!(
            registration(
                &args,
                inline.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        assert!(registration(&args, inline, crate::target_lab_build::Build::Rejected).is_err());
        assert!(registration(&args, env(), crate::target_lab_build::Build::Inline).is_err());
        for bad in ["", "auto", "threaded", "1", "inline "] {
            let mut value = env();
            value.push(("KUNO_LAB_DEVICE_DISPATCH".into(), bad.into()));
            assert!(registration(&args, value, crate::target_lab_build::Build::Inline).is_err());
        }

        let mut finite = env();
        finite.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3-fused".into()));
        let selected =
            registration(&args, finite, crate::target_lab_build::Build::Threaded).unwrap();
        assert!(selected.verify3_dense && selected.verify3_fused);
        let mut warm = env();
        warm.push(("KUNO_LAB_WARM_REPLAY".into(), "once".into()));
        assert!(
            registration(&args, warm, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .warm_replay
        );
        for invalid in ["", "auto", "0", "1", "once ", "twice"] {
            let mut values = env();
            values.push(("KUNO_LAB_WARM_REPLAY".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut fused = env();
        fused.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
        assert!(
            registration(&args, fused, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .fused_sweeps
        );
        for invalid in ["", "auto", "0", "1", "fused "] {
            let mut values = env();
            values.push(("KUNO_LAB_MTP_SWEEPS".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .verify3_dense
        );
        let mut direct = env();
        direct.push(("KUNO_LAB_TARGET_FILL".into(), "direct-public".into()));
        assert!(
            registration(&args, direct, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .public_existing_fill
        );
        for invalid in ["", "auto", "0", "1", "direct-public "] {
            let mut values = env();
            values.push(("KUNO_LAB_TARGET_FILL".into(), invalid.into()));
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .prepared_experts
        );
        for invalid in ["", "auto", "0", "1", "prepared-verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_IO".into(), "chunk".into()));
        assert!(
            registration(&args, candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .chunk_io
        );
        for invalid in ["", "auto", "0", "1", "chunk "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_IO".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        for invalid in ["", "auto", "0", "1", "verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_DENSE".into(), invalid.into()));
            assert!(
                registration(&args, candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        assert!(registration(&args, env(), crate::target_lab_build::Build::Rejected).is_err());
        for index in 0..6 {
            let mut values = env();
            values.remove(index);
            assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        }
        let mut values = env();
        values.push(("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY".into(), "1".into()));
        assert!(registration(&args, values, crate::target_lab_build::Build::Threaded).is_err());
        let mut wrong = args.clone();
        wrong[1] = "nvidia".into();
        assert!(registration(&wrong, env(), crate::target_lab_build::Build::Threaded).is_err());
    }
}
