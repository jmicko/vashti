//! Complete original Q8 MTP parity on an exclusive fixed-public stream.
//! No user input, accepted speculative output or throughput claim.
use crate::{daslab_carry_reference, mtp_public_reference, resident_mtp};
use cubecl::{prelude::*, wgpu::WgpuRuntime};
use kuno_text::{GgufFile, GgufLimits, mtp::MtpGraph};
use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};
use zeroize::Zeroizing;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn verify_minimal(
    expected: &mtp_public_reference::Row,
    output: &resident_mtp::DraftOutput,
    binding: u64,
    cursor: usize,
) -> Result<f64> {
    if output.binding() != binding
        || output.start() != expected.position()
        || cursor != expected.completed_cursor()
        || output.greedy() != expected.token()
        || output.hidden().len() != expected.hidden().len()
    {
        return Err("resident minimal MTP actual identity/greedy rejected".into());
    }
    let mut error = 0f64;
    for (&actual, &original) in output.hidden().iter().zip(expected.hidden()) {
        let delta = (actual as f64 - original as f64).abs();
        if !actual.is_finite() || delta > 0.006 + 0.0003 * (original as f64).abs() {
            return Err("resident minimal MTP complete hidden rejected".into());
        }
        error = error.max(delta);
    }
    Ok(error)
}
pub(crate) struct Inputs {
    mtp: PathBuf,
    carry: PathBuf,
    reference: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    default: bool,
) -> std::result::Result<Inputs, &'static str> {
    if !default
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "resident-mtp-probe"
    {
        return Err("resident MTP requires the default compiled public fixture");
    }
    let mut selected = BTreeMap::new();
    for (key, value) in environment {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|prefix| key.as_encoded_bytes().starts_with(prefix))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_DASLAB_CARRY"
                    | "KUNO_PUBLIC_MTP_REFERENCE"
                    | "KUNO_PUBLIC_RECEIPT"
            )
        ) || value.is_empty()
            || selected.insert(key, value).is_some()
        {
            return Err("resident MTP rejects inherited experiments");
        }
    }
    if selected.remove(OsStr::new("KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY")) != Some(OsString::from("1"))
    {
        return Err("resident MTP requires its public-only gate");
    }
    let mut path = |name: &str| -> std::result::Result<PathBuf, &'static str> {
        selected
            .remove(OsStr::new(name))
            .map(PathBuf::from)
            .ok_or("resident MTP fixed public path missing")
    };
    Ok(Inputs {
        mtp: path("KUNO_TEST_MTP")?,
        carry: path("KUNO_PUBLIC_DASLAB_CARRY")?,
        reference: path("KUNO_PUBLIC_MTP_REFERENCE")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}

pub(crate) fn run(
    client: &ComputeClient<WgpuRuntime>,
    limits: &wgpu::Limits,
    vendor: u32,
    input: Inputs,
) -> Result<()> {
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_resident_mtp_failure panic=true")
    }));
    let carry = daslab_carry_reference::Reference::load(&input.carry)?;
    let reference = mtp_public_reference::Reference::load(&input.reference, &carry)?;
    let file = GgufFile::open(&input.mtp, GgufLimits::default())?;
    if file.file_len() != 4_137_429_120 || file.tensors().len() != 34 {
        return Err("original Q8 MTP extent/inventory rejected".into());
    }
    // SAFETY: exclusive protected launcher pins the full original static
    // artifact; no installer or in-place writer runs during this session.
    unsafe {
        file.enable_mapped_weights()?;
    }
    let graph = MtpGraph::bind(&file)?;
    let cancel = AtomicBool::new(false);
    client.configure_transient_io(64 * 1024 * 1024)?;
    let retired = Arc::new(AtomicBool::new(false));
    let mut passes = Vec::with_capacity(2);
    for replay in 0..2 {
        let mut workspace =
            resident_mtp::ResidentMtp::new(&graph, client.clone(), limits, retired.clone())?;
        let attempt = (|| -> Result<serde_json::Value> {
            let binding = workspace.binding();
            let mut checks = Vec::with_capacity(9);
            let mut logits_error = 0f64;
            let mut hidden_error = 0f64;
            for position in 0..4 {
                let expected = reference.teacher_forced(position)?;
                // MTP cell c pairs targetR_c with tokenT_(c+1), at RoPE c.
                let preceding = carry.hidden(position)?;
                let output = workspace.step(expected.input_token(), preceding, &cancel)?;
                if output.binding() != binding || output.start() != position {
                    return Err("resident MTP output identity rejected".into());
                }
                let report = reference.verify(
                    mtp_public_reference::Conditioning::TeacherForced,
                    position,
                    expected.input_token(),
                    workspace.position(),
                    output.logits(),
                    output.hidden(),
                )?;
                logits_error = logits_error.max(report.logits_max_abs_error);
                hidden_error = hidden_error.max(report.hidden_max_abs_error);
                drop(output);
                checks.push(serde_json::json!({"teacher_forced":position,"cursor":workspace.position(),"all_vectors_accepted":true}));
            }
            workspace.rewind(0)?;
            let first = reference.teacher_forced(0)?;
            let output = workspace.step(first.input_token(), carry.hidden(0)?, &cancel)?;
            if output.binding() != binding || output.start() != 0 {
                return Err("resident MTP provisional identity rejected".into());
            }
            reference.verify(
                mtp_public_reference::Conditioning::ProvisionalTarget,
                0,
                first.input_token(),
                workspace.position(),
                output.logits(),
                output.hidden(),
            )?;
            let proposed = output.greedy();
            let provisional_carry = Zeroizing::new(output.hidden().to_vec());
            drop(output);
            let output = workspace.step(proposed, &provisional_carry, &cancel)?;
            if output.binding() != binding || output.start() != 1 {
                return Err("resident MTP second provisional identity rejected".into());
            }
            reference.verify(
                mtp_public_reference::Conditioning::ProvisionalDraft,
                1,
                proposed,
                workspace.position(),
                output.logits(),
                output.hidden(),
            )?;
            drop(output);
            drop(provisional_carry);
            checks.push(serde_json::json!({"provisional_rows":2,"cursor":workspace.position(),"all_vectors_accepted":true}));

            // Replace provisional KV with actual preceding target carries.
            workspace.rewind(0)?;
            workspace.append_kv(carry.token(0)?, carry.hidden(0)?, &cancel)?;
            if workspace.position() != 1 {
                return Err("resident MTP append cursor rejected".into());
            }
            for position in 1..4 {
                let expected = reference.teacher_forced(position)?;
                let output =
                    workspace.step(expected.input_token(), carry.hidden(position)?, &cancel)?;
                if output.binding() != binding || output.start() != position {
                    return Err("resident MTP repaired identity rejected".into());
                }
                reference.verify(
                    mtp_public_reference::Conditioning::TeacherForced,
                    position,
                    expected.input_token(),
                    workspace.position(),
                    output.logits(),
                    output.hidden(),
                )?;
                drop(output);
            }
            checks.push(serde_json::json!({"actual_target_KV_repair":true,"continued_cursor":workspace.position(),"all_vectors_accepted":true}));

            workspace.rewind(0)?;
            let before = workspace.traffic();
            let minimal_start = std::time::Instant::now();
            for position in 0..4 {
                let expected = reference.teacher_forced(position)?;
                let output = workspace.step_draft(
                    expected.input_token(),
                    carry.hidden(position)?,
                    &cancel,
                )?;
                hidden_error = hidden_error.max(verify_minimal(
                    expected,
                    &output,
                    binding,
                    workspace.position(),
                )?);
                drop(output);
            }
            let minimal_ns = minimal_start.elapsed().as_nanos() as u64;
            let after = workspace.traffic();
            if after.host_read_calls - before.host_read_calls != 4
                || after.host_read_bytes - before.host_read_bytes != 4 * 65536
                || after.private_upload_calls - before.private_upload_calls != 8
                || after.private_upload_bytes - before.private_upload_bytes != 4 * 98304
            {
                return Err("resident minimal MTP actual traffic rejected".into());
            }
            checks.push(serde_json::json!({"minimal_teacher_rows":4,"complete_hidden_and_real_greedy":true,"host_reads":after.host_read_calls-before.host_read_calls,"host_read_bytes":after.host_read_bytes-before.host_read_bytes,"elapsed_ns_including_public_checks":minimal_ns}));
            workspace.rewind(0)?;
            let expected = reference.provisional(0)?;
            let output = workspace.step_draft(expected.input_token(), carry.hidden(0)?, &cancel)?;
            verify_minimal(expected, &output, binding, workspace.position())?;
            drop(output);
            let before = workspace.traffic();
            let output = workspace.follow_draft(&cancel)?;
            verify_minimal(
                reference.provisional(1)?,
                &output,
                binding,
                workspace.position(),
            )?;
            drop(output);
            let after = workspace.traffic();
            if after.host_read_calls - before.host_read_calls != 1
                || after.host_read_bytes - before.host_read_bytes != 65536
                || after.private_upload_calls != before.private_upload_calls
                || after.private_upload_bytes != before.private_upload_bytes
                || after.followup_copy_calls - before.followup_copy_calls != 1
            {
                return Err("resident GPU follow MTP actual traffic rejected".into());
            }
            checks.push(serde_json::json!({"GPU_follow_draft":true,"complete_hidden_and_real_greedy":true,"host_read_bytes":65536,"private_upload_bytes":0,"actual_GPU_copy_dispatches":1,"actual_GPU_roles_copied":2}));
            Ok(
                serde_json::json!({"replay":replay,"checks":checks,"logits_max_abs_error":logits_error,"hidden_max_abs_error":hidden_error}),
            )
        })();
        let close = workspace.close();
        let result = attempt?;
        let close = close?;
        passes.push(serde_json::json!({"acceptance":result,"close":{
            "private_buffers":close.private.buffers,"private_bytes":close.private.bytes,
            "public_bytes":close.public_bytes,"main_reserved":close.main_reserved,
        }}));
    }
    file.verify_file_unchanged()?;
    let value = serde_json::json!({
        "scope":"fixed-public-original-Q8-resident-MTP-only", "vendor":vendor,
        "seed":424242, "temperature":0, "public_token0":0,
        "passes":passes, "target_model_on_this_card":false,
        "speculative_controller":false, "accepted_speculative_tokens_per_second":null,
        "private_or_production_accepted":false,
    });
    let mut receipt = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(input.receipt)?;
    use std::io::Write;
    receipt.write_all(serde_json::to_string_pretty(&value)?.as_bytes())?;
    println!(
        "public_resident_mtp_complete full_vectors=true rewind=true owners_erased=true performance_accepted=false"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_RESIDENT_MTP_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_MTP", "/original"),
            ("KUNO_PUBLIC_DASLAB_CARRY", "/public/carry"),
            ("KUNO_PUBLIC_MTP_REFERENCE", "/public/reference"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
    }
    #[test]
    fn registration_is_exclusive_to_fixed_default_both_card_fixture() {
        let args: Vec<OsString> = ["binary", "intel", "resident-mtp-probe"]
            .map(Into::into)
            .to_vec();
        assert!(registration(&args, env(), true).is_ok());
        let mut nvidia = args.clone();
        nvidia[1] = "nvidia".into();
        assert!(registration(&nvidia, env(), true).is_ok());
        assert!(registration(&args, env(), false).is_err());
        for index in 0..env().len() {
            let mut values = env();
            values.remove(index);
            assert!(registration(&args, values, true).is_err());
        }
        let mut values = env();
        values.push(("KUNO_LAB_NATIVE_DECODE".into(), "0".into()));
        assert!(registration(&args, values, true).is_err());
        let mut extra = args.clone();
        extra.push("extra".into());
        assert!(registration(&extra, env(), true).is_err());
    }
}
