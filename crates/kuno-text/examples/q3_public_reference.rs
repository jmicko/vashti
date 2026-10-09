//! Fixed public scalar/row-threaded quant reference, never a private request log.
use kuno_text::model::{Qwen4Exp, greedy};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroizing;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::var_os("KUNO_TEST_Q3_DIR").ok_or("verified Q3 directory required")?,
    );
    let paths: Vec<_> = (1..=3)
        .map(|i| {
            dir.join(format!(
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-{i:05}-of-00003.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 89_986_353_824 {
        return Err("pinned main byte count".into());
    }
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    if tokenizer.encode("!", true)? != [0] {
        return Err("public token binding".into());
    }
    let graph = Qwen4Exp::bind(&file)?;
    let mut sequence = graph.sequence(4)?;
    let mut executor = ThreadedExecutor::new(16)?;
    let cancel = AtomicBool::new(false);
    let output = std::env::var_os("KUNO_PUBLIC_Q3_LOGITS")
        .ok_or("fresh public reference output required")?;
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    out.write_all(b"KQ3R0001")?;
    let clock = Instant::now();
    let mut input = 0;
    let mut ids = Vec::new();
    for position in 0..4u32 {
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
            "public_q3_reference_position={position} greedy={token} elapsed_ms={}",
            clock.elapsed().as_millis()
        );
    }
    out.flush()?;
    sequence.reset();
    println!("public_q3_reference_ids={ids:?} completed=true");
    Ok(())
}
