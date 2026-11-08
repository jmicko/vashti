//! Synthetic correctness probe only. No HTTP integration or private input.
use kuno_text::model::{Qwen4Exp, ScalarExecutor, greedy};
use kuno_text::{GgufFile, GgufLimits};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing model path required")?;
    let token: u32 = std::env::args()
        .nth(1)
        .ok_or("public synthetic token ID required")?
        .parse()?;
    let start = Instant::now();
    let file = GgufFile::open(path, GgufLimits::default())?;
    let model = Qwen4Exp::bind(&file)?;
    println!("rust_model_bound_ms={}", start.elapsed().as_millis());
    let mut sequence = model.sequence(16)?;
    let mut executor = ScalarExecutor;
    let cancel = AtomicBool::new(false);
    let start = Instant::now();
    let mut logits = model.step(&mut executor, &mut sequence, token, &cancel, |l| {
        if l % 4 == 3 {
            println!(
                "layers_completed={} elapsed_ms={}",
                l + 1,
                start.elapsed().as_millis()
            );
        }
    })?;
    println!(
        "rust_step_ms={} greedy_token={} logits_count={}",
        start.elapsed().as_millis(),
        greedy(&logits)?,
        logits.len()
    );
    logits.fill(0.);
    sequence.reset();
    Ok(())
}
