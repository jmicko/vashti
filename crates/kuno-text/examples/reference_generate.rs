//! Operator-only fixed public fixtures. This is never a production request log.
use kuno_text::model::{Qwen4Exp, ScalarExecutor, greedy};
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zeroize::Zeroize;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = std::env::args().nth(1).unwrap_or("bang".into());
    let baseline: serde_json::Value = serde_json::from_str(include_str!(
        "../../../apps/kuno/benchmarks/text/qwen38-baseline.json"
    ))?;
    let (prompt, max_tokens) = match fixture.as_str() {
        "bang" => ("!", 4usize),
        "chat" => (
            "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            8,
        ),
        "rainwater" => (
            baseline["prompt"]
                .as_str()
                .ok_or("invalid public fixture")?,
            128,
        ),
        _ => return Err("unknown public fixture".into()),
    };
    let path = std::env::var_os("KUNO_TEST_GGUF").ok_or("existing model path required")?;
    let start = Instant::now();
    let file = GgufFile::open(path, GgufLimits::default())?;
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    assert_eq!(tokenizer.token_bytes(0)?, b"!");
    assert_eq!(tokenizer.token_bytes(49276)?, b"[](");
    assert_eq!(tokenizer.encode("[](", false)?, [49276]);
    let model = Qwen4Exp::bind(&file)?;
    let ids = tokenizer.encode(prompt, true)?;
    let mut sequence = model.sequence(ids.len() + max_tokens)?;
    let mut executor = ScalarExecutor;
    let cancel = AtomicBool::new(false);
    println!(
        "fixture={fixture} prompt_tokens={} seed=424242 temperature=0 rng=unused_greedy load_ms={}",
        ids.len(),
        start.elapsed().as_millis()
    );
    let start = Instant::now();
    let mut logits = Vec::new();
    for (n, &token) in ids.iter().enumerate() {
        logits.zeroize();
        logits = model.step(&mut executor, &mut sequence, token, &cancel, |_| {})?;
        println!(
            "prefill_tokens_complete={} elapsed_ms={}",
            n + 1,
            start.elapsed().as_millis()
        );
    }
    println!("prefill_ms={}", start.elapsed().as_millis());
    let start = Instant::now();
    let mut generated = Vec::new();
    let mut decoder = tokenizer.stream_decoder();
    for n in 0..max_tokens {
        let token = greedy(&logits)?;
        if token == model.eos_token() {
            break;
        }
        generated.push(token);
        let text = decoder.push_bytes(tokenizer.token_bytes(token)?)?;
        println!("synthetic_token={token} synthetic_text={text:?}");
        if n + 1 < max_tokens {
            logits.zeroize();
            logits = model.step(&mut executor, &mut sequence, token, &cancel, |_| {})?;
        }
    }
    println!(
        "generation_ms={} synthetic_output_tokens={generated:?}",
        start.elapsed().as_millis()
    );
    logits.zeroize();
    generated.zeroize();
    sequence.reset();
    decoder.clear();
    Ok(())
}
