//! Fixed public DASLab IQ3_S reference. Compiled token0 only, no request input.
//! Distinct artifact/reference from the Unsloth diagnostic.
use kuno_text::model::{Qwen4Exp, greedy};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroizing;

fn public_positions(raw: Option<&str>) -> Result<usize, &'static str> {
    match raw {
        None | Some("4") => Ok(4),
        Some("32") => Ok(32),
        _ => Err("public DASLab positions must be4 or32"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let positions_raw = std::env::var_os("KUNO_LAB_DASLAB_PUBLIC_POSITIONS");
    let positions = public_positions(
        positions_raw
            .as_ref()
            .map(|raw| {
                raw.to_str()
                    .ok_or("public DASLab positions encoding rejected")
            })
            .transpose()?,
    )?;
    let dir = PathBuf::from(
        std::env::var_os("KUNO_TEST_DASLAB_DIR").ok_or("verified DASLab directory required")?,
    );
    let paths: Vec<_> = (1..=2)
        .map(|i| {
            dir.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("pinned main byte count".into());
    }
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    if tokenizer.encode("!", true)? != [0] {
        return Err("public token binding".into());
    }
    let graph = Qwen4Exp::bind(&file)?;
    let mut sequence = graph.sequence(positions)?;
    let mut executor = ThreadedExecutor::new(16)?;
    let cancel = AtomicBool::new(false);
    let output = std::env::var_os("KUNO_PUBLIC_DASLAB_LOGITS")
        .ok_or("fresh public reference output required")?;
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    out.write_all(b"KQ3R0001")?;
    let clock = Instant::now();
    let mut input = 0;
    let mut ids = Vec::new();
    for position in 0..positions as u32 {
        let logits =
            Zeroizing::new(graph.step(&mut executor, &mut sequence, input, &cancel, |_| {})?);
        if logits.len() != 248320 || !logits.iter().all(|v| v.is_finite()) {
            return Err("public full logits".into());
        }
        let token = greedy(&logits)?;
        for value in [position, input, token, 248320] {
            out.write_all(&value.to_le_bytes())?;
        }
        for value in logits.iter() {
            out.write_all(&value.to_le_bytes())?;
        }
        ids.push(token);
        input = token;
        println!(
            "public_daslab_reference_position={position} greedy={token} elapsed_ms={}",
            clock.elapsed().as_millis()
        );
    }
    out.flush()?;
    sequence.reset();
    println!("public_daslab_reference_ids={ids:?} completed=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_default4_or_explicit4_and32_are_public_positions() {
        assert_eq!(super::public_positions(None), Ok(4));
        assert_eq!(super::public_positions(Some("4")), Ok(4));
        assert_eq!(super::public_positions(Some("32")), Ok(32));
        for raw in [
            "", "0", "1", "3", "5", "8", "31", "33", "04", "032", "+32", "32 ",
        ] {
            assert!(super::public_positions(Some(raw)).is_err());
        }
    }
}
