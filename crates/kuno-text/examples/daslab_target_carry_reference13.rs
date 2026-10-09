//! Fixed public token0/greedy DASLab CPU logits+pre-output-HC reference.
//! Exactly thirteen sequential rows in context13; fixed original target math.
//! No prompt arguments, device engine, production persistence or binding IDs.
//! Full payload SHA256 checks belong to the protected original-artifact runner.

#[path = "../../../apps/kuno/benchmarks/native-text/src/daslab_carry_reference13.rs"]
pub mod daslab_carry_reference13;
#[path = "../../../apps/kuno/benchmarks/native-text/src/daslab_profile.rs"]
pub mod daslab_profile;

use daslab_carry_reference13::{CONTEXT, LegacyLogits, POSITIONS, Reference, Row};
use kuno_text::model::{Qwen4Exp, greedy};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{Error, GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

struct Inputs {
    directory: PathBuf,
    logits: PathBuf,
    output: PathBuf,
}
fn registration(
    args: &[OsString],
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    default_build: bool,
) -> Result<Inputs, &'static str> {
    if args.len() != 1 || !default_build {
        return Err("public DASLab carry writer requires default build and no prompt arguments");
    }
    let mut settings = BTreeMap::new();
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
                "KUNO_LAB_DASLAB_CARRY13_PUBLIC_ONLY"
                    | "KUNO_TEST_DASLAB_DIR"
                    | "KUNO_PUBLIC_DASLAB_REFERENCE"
                    | "KUNO_PUBLIC_DASLAB_CARRY13"
            )
        ) || value.is_empty()
            || settings.insert(key, value).is_some()
        {
            return Err("public DASLab carry writer rejects other inputs/settings");
        }
    }
    if settings.remove(OsStr::new("KUNO_LAB_DASLAB_CARRY13_PUBLIC_ONLY")) != Some("1".into()) {
        return Err("public DASLab carry writer requires explicit public-only gate");
    }
    Ok(Inputs {
        directory: settings
            .remove(OsStr::new("KUNO_TEST_DASLAB_DIR"))
            .ok_or("verified original DASLab directory required")?
            .into(),
        logits: settings
            .remove(OsStr::new("KUNO_PUBLIC_DASLAB_REFERENCE"))
            .ok_or("verified complete original32-row public CPU logits required")?
            .into(),
        output: settings
            .remove(OsStr::new("KUNO_PUBLIC_DASLAB_CARRY13"))
            .ok_or("fresh fixed public carry output required")?
            .into(),
    })
}

fn require_original(file: &GgufFile) -> kuno_text::Result<()> {
    if file.file_len() != daslab_carry_reference13::MODEL_BYTES
        || file.shard_count() != 2
        || file.version() != 3
        || file.tensors().len() != daslab_profile::TENSOR_COUNT
        || daslab_profile::REVISION != daslab_carry_reference13::REVISION
    {
        return Err(Error::InvalidFormat(
            "public DASLab complete original source",
        ));
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .values()
            .map(|info| (info.name(), info.dtype().gguf_id())),
    )
    .map_err(|_| Error::InvalidFormat("public DASLab exact original inventory"))?;
    file.verify_file_unchanged()
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let inputs = registration(
        &std::env::args_os().collect::<Vec<_>>(),
        std::env::vars_os(),
        !cfg!(any(
            feature = "cpu-q8-two-pass",
            feature = "cpu-work-stealing-lab"
        )),
    )?;
    daslab_carry_reference13::writer_fixture_peak_bytes()?;
    let legacy = LegacyLogits::load(&inputs.logits)?;
    let paths = [1, 2].map(|i| {
        inputs.directory.join(format!(
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
        ))
    });
    for (path, expected) in paths.iter().zip(daslab_carry_reference13::SHARD_BYTES) {
        if std::fs::metadata(path)?.len() != expected {
            return Err("public DASLab original shard byte count".into());
        }
    }
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    require_original(&file)?;
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    if tokenizer.encode("!", true)? != [0] {
        return Err("public DASLab token0 binding rejected".into());
    }
    let graph = Qwen4Exp::bind(&file)?;
    let mut sequence = graph.sequence(CONTEXT)?;
    let mut executor = ThreadedExecutor::new(16)?;
    let cancel = AtomicBool::new(false);
    let mut rows = Vec::new();
    rows.try_reserve_exact(POSITIONS)?;
    if rows.capacity() != POSITIONS {
        return Err("public DASLab exact row capacity".into());
    }
    let mut input = 0;
    for position in 0..POSITIONS {
        let mut completed_layers = 0;
        let output = graph.step_with_carry(&mut executor, &mut sequence, input, &cancel, |_| {
            completed_layers += 1
        })?;
        if completed_layers != 48
            || output.position() != position
            || output.binding_id() != graph.bound_graph().binding_id()
            || sequence.position() != position + 1
        {
            return Err("public DASLab complete original carry position".into());
        }
        legacy.verify_cpu_bits(position, output.logits())?;
        rows.push(Row::new(position, input, output.logits(), output.hidden())?);
        input = greedy(output.logits())?;
        println!(
            "public_daslab_carry13_position={position} full_cpu_logits_bitwise=true layers=48"
        );
    }
    require_original(&file)?;
    let reference = Reference::new(rows)?;
    sequence.reset();
    // No partial file is created before complete graph/source/CPU acceptance.
    reference.write_new(&inputs.output)?;
    println!(
        "public_daslab_carry13_reference13_complete=true positions={POSITIONS} bytes={} seed={} temperature=0 logits_values={} carry_values={} gpu_abs_tolerance={} gpu_rel_tolerance={} private_runtime_accepted=false",
        daslab_carry_reference13::FILE_BYTES,
        daslab_carry_reference13::SEED,
        daslab_carry_reference13::VOCAB,
        daslab_carry_reference13::HIDDEN,
        daslab_carry_reference13::ABS_TOLERANCE,
        daslab_carry_reference13::REL_TOLERANCE,
    );
    Ok(())
}
fn main() -> Result<(), &'static str> {
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_daslab_carry13_writer_failure panic=true")
    }));
    run().map_err(|_| "fixed public DASLab carry writer failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn environment() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_DASLAB_CARRY13_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_DASLAB_DIR", "/public-original"),
            ("KUNO_PUBLIC_DASLAB_REFERENCE", "/public-reference"),
            ("KUNO_PUBLIC_DASLAB_CARRY13", "/new-public-carry"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .into()
    }
    #[test]
    fn only_explicit_default_build_fixed_public_inputs_are_admitted() {
        let args = vec!["public-writer".into()];
        assert!(registration(&args, environment(), true).is_ok());
        assert!(registration(&args, environment(), false).is_err());
        assert!(registration(&[], environment(), true).is_err());
        assert!(
            registration(
                &["public-writer".into(), "prompt".into()],
                environment(),
                true
            )
            .is_err()
        );
        for index in 0..4 {
            let mut missing = environment();
            missing.remove(index);
            assert!(registration(&args, missing, true).is_err());
            let mut empty = environment();
            empty[index].1 = "".into();
            assert!(registration(&args, empty, true).is_err());
        }
        for key in ["KUNO_LAB_FAST_GPU", "KUNO_TEST_PROMPT", "KUNO_PUBLIC_OTHER"] {
            let mut inherited = environment();
            inherited.push((key.into(), "0".into()));
            assert!(registration(&args, inherited, true).is_err());
        }
        let mut disabled = environment();
        disabled[0].1 = "0".into();
        assert!(registration(&args, disabled, true).is_err());
        let mut duplicate = environment();
        duplicate.push(duplicate[0].clone());
        assert!(registration(&args, duplicate, true).is_err());
    }
}
