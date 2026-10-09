//! Fixed-public actual wrong-draft rollback and original-prefix recompute.
//! No forced expert, private request, controller or throughput acceptance.
use crate::{
    daslab_carry_reference as reference, daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    resident_model::{
        ResidentModel,
        batch::{Config, LatestTargetRound},
    },
};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{GgufFile, GgufLimits, model::Qwen4Exp};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::Zeroizing;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(crate) struct Inputs {
    pub(crate) verify3_dense: bool,
    directory: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    default: bool,
) -> std::result::Result<Inputs, &'static str> {
    if !default || args.len() != 3 || args[1] != "intel" || args[2] != "daslab-target-repair-probe"
    {
        return Err("target repair requires default fixed public B70 fixture");
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
                "KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_PUBLIC_TARGET_ROWS_REFERENCE"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_TARGET_DENSE"
            )
        ) || value.is_empty()
            || values.insert(key, value).is_some()
        {
            return Err("target repair rejects inherited experiments");
        }
    }
    if values.remove(OsStr::new("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY")) != Some("1".into())
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES")) != Some("8589934592".into())
    {
        return Err("target repair fixed public/cache gates missing");
    }
    let verify3_dense = match values.remove(OsStr::new("KUNO_LAB_TARGET_DENSE")) {
        None => false,
        Some(value) if value == "verify3" => true,
        _ => return Err("target dense policy requires canonical verify3"),
    };
    let mut path = |name: &str| {
        values
            .remove(OsStr::new(name))
            .map(PathBuf::from)
            .ok_or("target repair public path missing")
    };
    Ok(Inputs {
        verify3_dense,
        directory: path("KUNO_TEST_Q3_DIR")?,
        reference: path("KUNO_PUBLIC_TARGET_ROWS_REFERENCE")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}
pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    input: Inputs,
) -> Result<()> {
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_target_repair_failure panic=true")
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
    // SAFETY: protected launcher owns the immutable original static shards.
    unsafe {
        file.enable_mapped_weights()?;
    }
    let graph = Qwen4Exp::bind(&file)?;
    crate::resident_util::select_retained_metadata(true);
    crate::resident_util::select_compute_scratch_clear(true);
    client.configure_transient_io(128 * 1024 * 1024)?;
    let retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let mut reports = Vec::with_capacity(3);
    for accepted in 1..=3 {
        let mut executor = GpuExecutor::new_with_profile(
            &file,
            client.clone(),
            8 * 1024 * 1024 * 1024,
            144,
            31 * 1024 * 1024 * 1024,
            OriginalPackedProfile::DaslabIq3,
        )?;
        executor.target_rows_reserved = true;
        executor.target_verify3_dense = input.verify3_dense;
        executor.enable_transient_resident(retired.clone())?;
        executor.fast_dense = true;
        executor.fast_dense_fault = true;
        executor.shared_kernels = true;
        executor.shared_gdn = true;
        let config = Config::new(client, limits, 8)?
            .reserve_target_rows()?
            .reserve_latest_target_checkpoint()?;
        let mut resident = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            reference::CONTEXT,
            &mut executor,
            config,
            true,
        )?;
        let attempt = (|| -> Result<serde_json::Value> {
            let mut tokens = Zeroizing::new(Vec::with_capacity(3));
            for row in 0..3 {
                tokens.push(if row < accepted {
                    expected.input_token(row)?
                } else {
                    17 + row as u32
                });
            }
            let mut round =
                LatestTargetRound::begin(&mut resident, &mut executor, &tokens, &cancel)?;
            let rows = round.verify(&cancel)?;
            if rows.binding_id() != graph.bound_graph().binding_id()
                || rows.start() != 0
                || rows.count() != 3
            {
                return Err("target proposed output identity rejected".into());
            }
            let mut max_logit = 0f64;
            let mut max_hidden = 0f64;
            for row in 0..accepted {
                if rows.position(row)? != row {
                    return Err("target proposed row order rejected".into());
                }
                let report = expected.verify(row, rows.logits(row)?, rows.hidden(row)?)?;
                max_logit = max_logit.max(report.logits_max_abs_error);
                max_hidden = max_hidden.max(report.hidden_max_abs_error);
            }
            drop(rows);
            let repaired = round.finish(accepted, &cancel)?;
            if let Some(rows) = repaired {
                if accepted == 3
                    || rows.binding_id() != graph.bound_graph().binding_id()
                    || rows.start() != 0
                    || rows.count() != accepted
                {
                    return Err("target repaired output identity rejected".into());
                }
                for row in 0..accepted {
                    let report = expected.verify(row, rows.logits(row)?, rows.hidden(row)?)?;
                    max_logit = max_logit.max(report.logits_max_abs_error);
                    max_hidden = max_hidden.max(report.hidden_max_abs_error);
                }
                drop(rows);
            } else if accepted != 3 {
                return Err("target rejected prefix was not recomputed".into());
            }
            if resident.completed_target_rows_cursor(&executor)? != accepted {
                return Err("target repaired actual cursor rejected".into());
            }
            for position in accepted..4 {
                let rows = resident.prefill_chunk_with_carry(
                    &mut executor,
                    &[expected.input_token(position)?],
                    &cancel,
                )?;
                if rows.binding_id() != graph.bound_graph().binding_id()
                    || rows.start() != position
                    || rows.count() != 1
                    || rows.position(0)? != position
                {
                    return Err("target continued identity rejected".into());
                }
                let report = expected.verify(position, rows.logits(0)?, rows.hidden(0)?)?;
                max_logit = max_logit.max(report.logits_max_abs_error);
                max_hidden = max_hidden.max(report.hidden_max_abs_error);
                drop(rows);
                if resident.completed_target_rows_cursor(&executor)? != position + 1 {
                    return Err("target continued actual cursor rejected".into());
                }
            }
            Ok(
                serde_json::json!({"accepted_input_prefix":accepted,"proposed_count":3,"wrong_future_inputs":3-accepted,"accepted_prefix_and_continuation_all_vectors":true,"actual_completed_cursor":4,"logits_max_abs_error":max_logit,"hidden_max_abs_error":max_hidden}),
            )
        })();
        let actual_verify3_dense_calls = executor.target_verify3_dense_calls;
        let close = resident.close();
        let audit = resident.public_owned_zero_audit_receipt();
        drop(resident);
        let source_close = executor.close_transient_resident();
        drop(executor);
        close?;
        source_close?;
        if retired.load(Ordering::Acquire) {
            return Err("target repair physical stream retired".into());
        }
        let (main, sources) = audit.ok_or("target repair full erasure not accepted")?;
        let mut report = attempt?;
        if input.verify3_dense && actual_verify3_dense_calls == 0 {
            return Err("target Verify3 actual dispatch evidence missing".into());
        }
        report["verify3_dense_calls"] = actual_verify3_dense_calls.into();
        report["owner_audit"] = serde_json::json!({"main_bytes":main.bytes,"main_buffers":main.buffers,"source_bytes":sources.bytes,"source_buffers":sources.buffers});
        reports.push(report);
    }
    file.verify_file_unchanged()?;
    let value = serde_json::json!({"scope":"fixed-public-DASLab-actual-target-prefix-repair-only","verify3_dense":input.verify3_dense,"public_token0":0,"seed":424242,"temperature":0,"max_chunk":3,"runtime_io_limit_bytes":128*1024*1024,"complete_cpu_reference_rows":4,"cases":reports,"speculative_controller":false,"accepted_speculative_tokens_per_second":null,"private_or_production_accepted":false});
    let out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(input.receipt)?;
    serde_json::to_writer_pretty(out, &value)?;
    println!(
        "public_target_repair_complete accepted_prefix=true continuation=true owners_erased=true performance_accepted=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exclusive_default_public_b70_registration() {
        let args: Vec<OsString> = ["binary", "intel", "daslab-target-repair-probe"]
            .map(Into::into)
            .to_vec();
        let env = || {
            [
                ("KUNO_LAB_TARGET_REPAIR_PUBLIC_ONLY", "1"),
                ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
                ("KUNO_TEST_Q3_DIR", "/original"),
                ("KUNO_PUBLIC_TARGET_ROWS_REFERENCE", "/public/carry"),
                ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
            ]
            .map(|(k, v)| (k.into(), v.into()))
            .to_vec()
        };
        assert!(registration(&args, env(), true).is_ok());
        let mut candidate = env();
        candidate.push(("KUNO_LAB_TARGET_DENSE".into(), "verify3".into()));
        assert!(registration(&args, candidate, true).unwrap().verify3_dense);
        for invalid in ["", "auto", "0", "1", "verify3 "] {
            let mut candidate = env();
            candidate.push(("KUNO_LAB_TARGET_DENSE".into(), invalid.into()));
            assert!(registration(&args, candidate, true).is_err());
        }
        assert!(registration(&args, env(), false).is_err());
        let mut wrong = args.clone();
        wrong[1] = "nvidia".into();
        assert!(registration(&wrong, env(), true).is_err());
        for index in 0..5 {
            let mut values = env();
            values.remove(index);
            assert!(registration(&args, values, true).is_err());
        }
        let mut values = env();
        values.push(("KUNO_LAB_TARGET_ROWS_PUBLIC_ONLY".into(), "1".into()));
        assert!(registration(&args, values, true).is_err());
    }
}
