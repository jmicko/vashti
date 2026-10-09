//! Fixed public target rows/carries, before any speculative controller.
//! Every supplied row commits. This is not accepted-prefix or MTP throughput.
use crate::{
    daslab_carry_reference as reference, daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    resident_model::{ResidentModel, batch::Config},
};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::model::Qwen4Exp;
use kuno_text::{GgufFile, GgufLimits};
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

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const CACHE: usize = 8 * 1024 * 1024 * 1024;
const DEVICE: usize = 31 * 1024 * 1024 * 1024;
const RUNTIME: u64 = 128 * 1024 * 1024;

pub(crate) struct Inputs {
    pub(crate) inline_dispatch: bool,
    pub(crate) verify3_dense: bool,
    pub(crate) verify3_fused: bool,
    pub(crate) prepared_experts: bool,
    public_existing_fill: bool,
    chunk_io: bool,
    request_reuse: bool,
    directory: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}

pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "daslab-target-rows-probe" {
        return Err("fixed target rows require B70 public probe");
    }
    let mut selected = BTreeMap::new();
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|prefix| key.as_encoded_bytes().starts_with(prefix))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_PUBLIC_TARGET_ROWS_REFERENCE"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_DEVICE_DISPATCH"
                    | "KUNO_LAB_TARGET_DENSE"
                    | "KUNO_LAB_TARGET_EXPERTS"
                    | "KUNO_LAB_TARGET_FILL"
                    | "KUNO_LAB_TARGET_IO"
                    | "KUNO_LAB_TARGET_REUSE"
            )
        ) || value.is_empty()
            || selected.insert(key, value).is_some()
        {
            return Err("target rows reject inherited experiments");
        }
    }
    if selected.remove(OsStr::new("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY")) != Some(OsString::from("1"))
    {
        return Err("fixed target rows require explicit public-only gate");
    }
    if selected.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES"))
        != Some(OsString::from("8589934592"))
    {
        return Err("target rows require explicit original 8GiB static cache");
    }
    let raw_dispatch = selected.remove(OsStr::new("KUNO_LAB_DEVICE_DISPATCH"));
    let inline_dispatch = build.selected(raw_dispatch.as_deref())?;
    let prepared_experts = match selected.remove(OsStr::new("KUNO_LAB_TARGET_EXPERTS")) {
        None => false,
        Some(value) if value == "prepared-verify3" => true,
        _ => return Err("target expert policy requires canonical prepared-verify3"),
    };
    let public_existing_fill = match selected.remove(OsStr::new("KUNO_LAB_TARGET_FILL")) {
        None => false,
        Some(value) if value == "direct-public" => true,
        _ => return Err("target fill policy requires canonical direct-public"),
    };
    let chunk_io = match selected.remove(OsStr::new("KUNO_LAB_TARGET_IO")) {
        None => false,
        Some(value) if value == "chunk" => true,
        _ => return Err("target I/O policy requires canonical chunk"),
    };
    let raw_reuse = selected.remove(OsStr::new("KUNO_LAB_TARGET_REUSE"));
    let request_reuse =
        crate::target_lab_build::request_reuse(raw_reuse.as_deref(), inline_dispatch, chunk_io)?;
    let (verify3_dense, verify3_fused) = match selected.remove(OsStr::new("KUNO_LAB_TARGET_DENSE"))
    {
        None => (false, false),
        Some(value) if value == "verify3" => (true, false),
        Some(value) if value == "verify3-fused" => (true, true),
        _ => return Err("target dense policy requires canonical verify3 or verify3-fused"),
    };
    Ok(Inputs {
        inline_dispatch,
        verify3_dense,
        verify3_fused,
        prepared_experts,
        public_existing_fill,
        chunk_io,
        request_reuse,
        directory: selected
            .remove(OsStr::new("KUNO_TEST_Q3_DIR"))
            .ok_or("original DAS shards required")?
            .into(),
        reference: selected
            .remove(OsStr::new("KUNO_PUBLIC_TARGET_ROWS_REFERENCE"))
            .ok_or("fixed public carry oracle required")?
            .into(),
        receipt: selected
            .remove(OsStr::new("KUNO_PUBLIC_RECEIPT"))
            .ok_or("numeric receipt required")?
            .into(),
    })
}

pub(crate) fn validate_limits(l: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 256
        || l.max_compute_workgroups_per_dimension < 16384
        || l.max_storage_buffers_per_shader_stage < 12
        || l.max_storage_buffer_binding_size < 64 * 1024 * 1024
        || l.max_buffer_size < 128 * 1024 * 1024
    {
        return Err("target row original kernels exceed available device limits");
    }
    Ok(())
}

pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    input: Inputs,
) -> Result<()> {
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_target_rows_failure panic=true")
    }));
    // Independent whole-file hashes belong to the protected launcher; strict
    // identity, split inventory and file immutability are still checked here.
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
        return Err("original DAS byte count rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(n, t)| (n.as_str(), t.dtype().gguf_id())),
    )
    .map_err(|_| "original DAS inventory rejected")?;
    // SAFETY: launcher owns the immutable original public artifact for this
    // entire exclusive session; no installer or in-place writer runs alongside.
    unsafe {
        file.enable_mapped_weights()?;
    }
    let model = Qwen4Exp::bind(&file)?;
    crate::resident_util::select_retained_metadata(true);
    crate::resident_util::select_compute_scratch_clear(true);
    let cancel = AtomicBool::new(false);
    // Configuration/retirement belong to this physical stream, across every
    // fresh logical sequence. Cleanup drains the same configuration.
    let retired = Arc::new(AtomicBool::new(false));
    if input.request_reuse {
        client.configure_public_target_request_io()?;
    } else {
        client.configure_transient_io(RUNTIME)?;
    }
    let mut reports = Vec::with_capacity(5);
    for partition in [&[1usize, 1, 1][..], &[1, 2], &[2, 1], &[3], &[3]] {
        let mut executor = GpuExecutor::new_with_profile(
            &file,
            client.clone(),
            CACHE,
            144,
            DEVICE,
            OriginalPackedProfile::DaslabIq3,
        )?;
        executor.target_rows_reserved = true;
        executor.target_verify3_dense = input.verify3_dense;
        executor.target_verify3_fused = input.verify3_fused;
        executor.token_io_checkpoint = input.chunk_io;
        executor.enable_transient_resident(retired.clone())?;
        executor.fast_dense = true;
        executor.fast_dense_fault = true;
        executor.shared_kernels = true;
        executor.shared_gdn = true;
        let mut config = Config::new(client, limits, 8)?.reserve_target_rows()?;
        if input.prepared_experts {
            config = config.reserve_prepared_verify3_experts()?;
        }
        if input.request_reuse {
            config = config.reserve_public_target_request_io()?;
        }
        if input.public_existing_fill {
            config = config.with_public_existing_expert_fill()?;
        }
        let mut resident = ResidentModel::new_with_native_batch(
            model.bound_graph(),
            client.clone(),
            reference::CONTEXT,
            &mut executor,
            config,
            true,
        )?;
        let owned_bytes = resident.owned_bytes();
        let attempt = (|| -> Result<serde_json::Value> {
            let mut start = 0usize;
            let mut checks = Vec::with_capacity(4);
            let mut logits_error = 0f64;
            let mut hidden_error = 0f64;
            let counts: Vec<usize> = if input.request_reuse {
                // Activate only after real token0 prefill/drop. The same five
                // partitions then cover all remaining three original rows.
                std::iter::once(1)
                    .chain(partition.iter().copied())
                    .collect()
            } else {
                partition
                    .iter()
                    .copied()
                    .chain(std::iter::once(1))
                    .collect()
            };
            for count in counts {
                let mut tokens = zeroize::Zeroizing::new(Vec::with_capacity(count));
                for row in start..start + count {
                    tokens.push(expected.input_token(row)?);
                }
                let begin = Instant::now();
                let rows = resident.prefill_chunk_with_carry(&mut executor, &tokens, &cancel)?;
                if rows.binding_id() != model.bound_graph().binding_id()
                    || rows.start() != start
                    || rows.count() != count
                {
                    return Err("target result identity/position rejected".into());
                }
                for row in 0..count {
                    if rows.position(row)? != start + row {
                        return Err("target row order rejected".into());
                    }
                    let report =
                        expected.verify(start + row, rows.logits(row)?, rows.hidden(row)?)?;
                    logits_error = logits_error.max(report.logits_max_abs_error);
                    hidden_error = hidden_error.max(report.hidden_max_abs_error);
                }
                let duration_ns = begin.elapsed().as_nanos() as u64;
                drop(rows); // No retained request payload across audit/close.
                let cursor = resident.completed_target_rows_cursor(&executor)?;
                expected.verify_completed_prefix(start, count, cursor)?;
                checks.push(serde_json::json!({"start":start,"count":count,"completed_cursor":cursor,
                    "whole_chunk_and_all_rows_accepted":true,"elapsed_ns_including_full_vector_checks":duration_ns}));
                start = cursor;
                if input.request_reuse && start == 1 {
                    resident.begin_public_target_request_io(&mut executor, &cancel)?;
                    if !resident.public_target_request_io_active()? {
                        return Err("target request scope did not activate".into());
                    }
                }
            }
            if start != 4 {
                return Err("target continuation incomplete".into());
            }
            let mut request_scope_receipt = None;
            if input.request_reuse {
                request_scope_receipt = Some(crate::target_lab_build::reuse_record(
                    resident.end_public_target_request_io(&mut executor, &cancel)?,
                ));
                if resident.public_target_request_io_active()? {
                    return Err("target request scope did not close".into());
                }
            }
            Ok(
                serde_json::json!({"partition":partition,"compiled_token0_prelude_before_partition":input.request_reuse,"request_scope_closed_and_audited":input.request_reuse,"request_scope":request_scope_receipt,"continuation_at_position3":!input.request_reuse,"complete_position3_checked":true,
                "logits_max_abs_error":logits_error,"hidden_max_abs_error":hidden_error,
                "checks":checks,"private_main_bytes":owned_bytes}),
            )
        })();
        let actual_verify3_dense_calls = executor.target_verify3_dense_calls;
        let actual_verify3_fused_calls = executor.target_verify3_fused_calls;
        let close = resident.close();
        let audit = resident.public_owned_zero_audit_receipt();
        drop(resident);
        let public_close = executor.close_transient_resident();
        drop(executor);
        close?;
        public_close?;
        if retired.load(Ordering::Acquire) {
            return Err("target context retired".into());
        }
        let (main, sources) = audit.ok_or("target full owner erasure not accepted")?;
        let mut report = attempt?;
        if input.verify3_dense && actual_verify3_dense_calls == 0 {
            return Err("target Verify3 actual dispatch evidence missing".into());
        }
        if input.verify3_fused && actual_verify3_fused_calls == 0 {
            return Err("target fused Verify3 actual dispatch evidence missing".into());
        }
        report["verify3_dense_calls"] = actual_verify3_dense_calls.into();
        report["verify3_fused_calls"] = actual_verify3_fused_calls.into();
        report["owner_audit"] = serde_json::json!({"main_bytes":main.bytes,"main_buffers":main.buffers,
            "source_bytes":sources.bytes,"source_buffers":sources.buffers});
        reports.push(report);
    }
    file.verify_file_unchanged()?;
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    if client.public_target_request_io_configured()? != input.request_reuse {
        return Err("target request stream frozen configuration mismatch".into());
    }
    let receipt = serde_json::json!({"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"scope":"fixed-public-DASLab-target-rows-and-carry-only",
        "model_repository":daslab_profile::REPOSITORY,"model_revision":daslab_profile::REVISION,
        "verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"request_reuse_metadata_reserved_bytes":if input.request_reuse {4*1024*1024}else{0},"request_binding_capacity":if input.request_reuse {4096}else{0},"public_token0":0,"seed":424242,"temperature":0,"complete_cpu_reference_rows":4,
        "logits_width":reference::VOCAB,"carry_width":reference::HIDDEN,"max_chunk":3,
        "absolute_tolerance":reference::ABS_TOLERANCE,"relative_tolerance":reference::REL_TOLERANCE,
        "carry_tap":"layer47 FFN HC residual before output_hc",
        "extra_fixture_host_owner_bound_bytes":16*1024*1024,"runtime_io_limit_bytes":RUNTIME,
        "all_supplied_rows_committed":true,"accepted_prefix_supported":false,
        "speculative_controller":false,"mtp_throughput_accepted":false,
        "private_or_production_accepted":false,"full_model_card":"B70","partitions":reports});
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(input.receipt)?;
    serde_json::to_writer_pretty(output, &receipt)?;
    println!(
        "public_target_rows_complete partitions=5 continuation=true owners_erased=true performance_accepted=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> Vec<OsString> {
        ["lab", "intel", "daslab-target-rows-probe"]
            .map(OsString::from)
            .to_vec()
    }
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_Q3_DIR", "weights"),
            ("KUNO_PUBLIC_TARGET_ROWS_REFERENCE", "fixed"),
            ("KUNO_PUBLIC_RECEIPT", "new"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
    }
    #[test]
    fn exclusive_fixed_public_registration() {
        assert!(registration(&args(), env(), crate::target_lab_build::Build::Threaded).is_ok());
        let mut reusable = env();
        reusable.extend([
            ("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()),
            ("KUNO_LAB_TARGET_IO".into(), "chunk".into()),
            ("KUNO_LAB_TARGET_REUSE".into(), "request".into()),
        ]);
        assert!(
            registration(
                &args(),
                reusable.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .request_reuse
        );
        assert!(
            registration(
                &args(),
                reusable.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        for key in ["KUNO_LAB_DEVICE_DISPATCH", "KUNO_LAB_TARGET_IO"] {
            let mut missing = reusable.clone();
            missing.retain(|(k, _)| k != key);
            assert!(
                registration(&args(), missing, crate::target_lab_build::Build::Inline).is_err()
            );
        }
        for raw in ["", "auto", "1", "request ", "chunk"] {
            let mut wrong = reusable.clone();
            for (k, v) in &mut wrong {
                if k == "KUNO_LAB_TARGET_REUSE" {
                    *v = raw.into();
                }
            }
            assert!(registration(&args(), wrong, crate::target_lab_build::Build::Inline).is_err());
        }
        let mut inline = env();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert!(
            registration(
                &args(),
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .inline_dispatch
        );
        assert!(
            registration(
                &args(),
                inline.clone(),
                crate::target_lab_build::Build::Threaded
            )
            .is_err()
        );
        assert!(registration(&args(), inline, crate::target_lab_build::Build::Rejected).is_err());
        assert!(registration(&args(), env(), crate::target_lab_build::Build::Inline).is_err());
        for bad in ["", "auto", "threaded", "1", "inline "] {
            let mut value = env();
            value.push(("KUNO_LAB_DEVICE_DISPATCH".into(), bad.into()));
            assert!(registration(&args(), value, crate::target_lab_build::Build::Inline).is_err());
        }

        let mut fused = env();
        fused.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3-fused".into()));
        let selected =
            registration(&args(), fused, crate::target_lab_build::Build::Threaded).unwrap();
        assert!(selected.verify3_dense && selected.verify3_fused);
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3".into()));
        assert!(
            registration(&args(), candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .verify3_dense
        );
        let mut direct = env();
        direct.push(("KUNO_LAB_TARGET_FILL".into(), "direct-public".into()));
        assert!(
            registration(&args(), direct, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .public_existing_fill
        );
        for invalid in ["", "auto", "0", "1", "direct-public "] {
            let mut values = env();
            values.push(("KUNO_LAB_TARGET_FILL".into(), invalid.into()));
            assert!(
                registration(&args(), values, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), "prepared-verify3".into()));
        assert!(
            registration(&args(), candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .prepared_experts
        );
        for invalid in ["", "auto", "0", "1", "prepared-verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_EXPERTS".into(), invalid.into()));
            assert!(
                registration(&args(), candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_IO".into(), "chunk".into()));
        assert!(
            registration(&args(), candidate, crate::target_lab_build::Build::Threaded)
                .unwrap()
                .chunk_io
        );
        for invalid in ["", "auto", "0", "1", "chunk "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_IO".into(), invalid.into()));
            assert!(
                registration(&args(), candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        for invalid in ["", "auto", "0", "1", "verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_DENSE".into(), invalid.into()));
            assert!(
                registration(&args(), candidate, crate::target_lab_build::Build::Threaded).is_err()
            );
        }
        assert!(registration(&args(), env(), crate::target_lab_build::Build::Rejected).is_err());
        let mut a = args();
        a[1] = "nvidia".into();
        assert!(registration(&a, env(), crate::target_lab_build::Build::Threaded).is_err());
        let mut e = env();
        e.push(("KUNO_LAB_Q3_INTEGER_MOE".into(), "0".into()));
        assert!(registration(&args(), e, crate::target_lab_build::Build::Threaded).is_err());
        let mut e = env();
        e[0].1 = "0".into();
        assert!(registration(&args(), e, crate::target_lab_build::Build::Threaded).is_err());
        let mut e = env();
        e.pop();
        assert!(registration(&args(), e, crate::target_lab_build::Build::Threaded).is_err());
    }
}
