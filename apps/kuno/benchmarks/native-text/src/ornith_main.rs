//! Fixed-public fully resident Ornith laboratory. Not a deployed inference API.
#[path = "ornith/gpu.rs"]
mod gpu;
#[path = "ornith/graph.rs"]
mod graph;
#[path = "ornith/kernels.rs"]
mod kernels;
#[path = "ornith/probe.rs"]
mod probe;
use gpu::Result;
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits};
use std::time::Instant;
use zeroize::Zeroizing;

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("Usage: ornith-resident-lab ORIGINAL_BLOB PUBLIC_RECEIPT 8|128|512|probe|profile|teacher".into());
    }
    if !args[1].ends_with("sha256-6540411d9e4ca638983e905959188e1be74db7ff62151c69d71a6f2748584f68")
    {
        return Err(
            "Only the inspected original Ornith blob is admitted by this fixed-public lab".into(),
        );
    }
    let count = match args[3].as_str() {
        "8" => 8,
        "128" => 128,
        "512" => 512,
        "tokenize-public-response" => 0,
        "probe" => 1,
        "profile" => 2,
        "teacher" => 3,
        _ => return Err("Unknown fixed public mode".into()),
    };
    if std::env::var("KUNO_LAB_ORNITH_PUBLIC").as_deref() != Ok("1") {
        return Err("Fixed-public lab opt-in missing".into());
    }
    let limits = std::fs::read_to_string("/proc/self/limits")?;
    let core = limits
        .lines()
        .find(|line| line.starts_with("Max core file size"))
        .ok_or("Core limit unavailable")?;
    let fields = core.split_whitespace().collect::<Vec<_>>();
    if fields.get(4) != Some(&"0") || fields.get(5) != Some(&"0") {
        return Err("Core dumps must be disabled".into());
    }
    let cgroup = std::fs::read_to_string("/proc/self/cgroup")?;
    let rel = cgroup
        .lines()
        .find_map(|s| s.strip_prefix("0::"))
        .ok_or("Cgroupv2 required")?;
    let swap = std::fs::read_to_string(format!("/sys/fs/cgroup{rel}/memory.swap.max"))?;
    if swap.trim() != "0" {
        return Err("Private lab memory must not enter disk swap".into());
    }
    let start = Instant::now();
    let file = GgufFile::open(
        &args[1],
        GgufLimits {
            max_tensor_read_bytes: 512 * 1024 * 1024,
            ..Default::default()
        },
    )?;
    if file.file_len() != 21166757664 {
        return Err("Original public blob extent mismatch".into());
    }
    // SAFETY: fixed inspected content-addressed Ollama inode, no concurrent model
    // mutations; original descriptor and all consumers live through graph close.
    unsafe {
        file.enable_mapped_weights()?;
    }
    if count == 1 {
        let receipt = probe::run(&file)?;
        file.verify_file_unchanged()?;
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?;
        serde_json::to_writer_pretty(output, &receipt)?;
        println!("public_original_projection_probes_passed");
        return Ok(());
    }
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    if count == 0 {
        use std::io::Read;
        let mut bytes = Zeroizing::new(Vec::new());
        std::io::stdin().take(65537).read_to_end(&mut bytes)?;
        if bytes.len() > 65536 {
            return Err("Fixed public response exceeds lab tokenization bound".into());
        }
        let text = std::str::from_utf8(&bytes)?;
        let ids = Zeroizing::new(tokenizer.encode(text, false)?);
        serde_json::to_writer(std::io::stdout().lock(), &*ids)?;
        return Ok(());
    }
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../text/ornith35-baseline.json"))?;
    let alternate = match std::env::var("KUNO_LAB_ORNITH_ALTERNATE_PUBLIC") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => return Err("Fixed public alternate selection rejected".into()),
    };
    if count == 3 && alternate {
        return Err("Teacher sequence requires original public prefix".into());
    }
    let prompt = fixture[if alternate {
        "alternate_prompt"
    } else {
        "prompt"
    }]
    .as_str()
    .ok_or("Public fixture missing")?;
    let tokens = Zeroizing::new(tokenizer.encode(prompt, true)?);
    let model = graph::Graph::new(&file)?;
    let total_load = start.elapsed().as_millis();
    println!(
        "public_ornith_resident_loaded bytes={} private_bytes={} dispatches={} load_ms={}",
        model.weight_bytes,
        model.gpu.private_bytes(),
        model.steps(),
        total_load
    );
    let mut samples = Vec::new();
    if count == 3 {
        let reference: Vec<u32> =
            serde_json::from_str(include_str!("../../text/ornith35-public-ids.json"))?;
        if reference.len() != 128 || reference.iter().any(|&id| id >= 248320) {
            return Err("Fixed public teacher sequence rejected".into());
        }
        model.gpu.clear()?;
        for (pos, &token) in tokens.iter().enumerate() {
            model.token(token, pos as u32, pos + 1 == tokens.len())?;
        }
        let mut observations = Vec::new();
        for (i, &expected) in reference.iter().enumerate() {
            let native = model.next()?;
            let mut metrics = model.public_expected_rank(expected)?;
            if metrics["native_id"].as_u64() != Some(native as u64) {
                return Err("Public CPU/GPU final selection mismatch".into());
            }
            metrics["output_position"] = i.into();
            observations.push(metrics);
            if i + 1 < reference.len() {
                model.token(expected, (tokens.len() + i) as u32, true)?;
            }
        }
        model.gpu.clear()?;
        file.verify_file_unchanged()?;
        let matches = observations
            .iter()
            .filter(|m| m["expected_rank"] == 1)
            .count();
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?;
        serde_json::to_writer_pretty(
            output,
            &serde_json::json!({"fixed_public_teacher_forcing":true,"fixture":"ornith35-baseline.json","reference_ids":"ornith35-public-ids.json","top1_matches":matches,"positions":observations,"normal_private_clear_completed":true}),
        )?;
        println!("public_teacher_forcing_completed top1_matches={matches}/128");
        return Ok(());
    }
    if count == 2 {
        model.gpu.clear()?;
        for (pos, &token) in tokens.iter().enumerate() {
            model.token(token, pos as u32, pos + 1 == tokens.len())?;
        }
        let first = model.next()?;
        let receipt = model.profile_token(first, tokens.len() as u32)?;
        model.gpu.clear()?;
        file.verify_file_unchanged()?;
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[2])?;
        serde_json::to_writer_pretty(output, &receipt)?;
        println!("public_ornith_operator_profile_completed");
        return Ok(());
    }
    for phase in ["first_sequence", "warm_weights_fresh_sequence"] {
        model.gpu.clear()?;
        let prefill = Instant::now();
        for (pos, &token) in tokens.iter().enumerate() {
            model.token(token, pos as u32, pos + 1 == tokens.len())?;
        }
        let mut generated = Zeroizing::new(Vec::with_capacity(count));
        generated.push(model.next()?);
        let prefill_ms = prefill.elapsed().as_millis();
        let generation = Instant::now();
        for step in 1..count {
            if count == 512 && generated.last() == Some(&tokenizer.eos_token_id()) {
                break;
            }
            model.token(generated[step - 1], (tokens.len() + step - 1) as u32, true)?;
            generated.push(model.next()?);
        }
        let decode_ns = generation.elapsed().as_nanos();
        let decode_steps = generated.len() - 1;
        let rate = decode_steps as f64 / (decode_ns as f64 / 1e9);
        println!(
            "public_ornith_sample phase={phase} prompt_tokens={} decode_steps={} prefill_ms={} decode_ns={} tokens_per_second={:.4}",
            tokens.len(),
            decode_steps,
            prefill_ms,
            decode_ns,
            rate
        );
        samples.push(serde_json::json!({"phase":phase,"prompt_tokens":tokens.len(),"prefill_millis":prefill_ms,
            "decode_steps":decode_steps,"output_cap":count,"stopped_on_eos":generated.last()==Some(&tokenizer.eos_token_id()),"decode_nanoseconds":decode_ns,"decode_tokens_per_second":rate,
            "public_generated_ids":&*generated}));
    }
    model.gpu.clear()?;
    file.verify_file_unchanged()?;
    let receipt = serde_json::json!({"fixture":"ornith35-baseline.json","model_blob":fixture["expected_model_blob"],
        "seed":424242,"temperature":0,"alternate_public_prefix":alternate,"backend":"native-rust-wgpu-vulkan","gpu_vendor":model.gpu.info.vendor,
        "gpu_device":model.gpu.info.device,"subgroup_min":model.gpu.info.subgroup_min_size,"subgroup_max":model.gpu.info.subgroup_max_size,
        "public_weight_bytes":model.weight_bytes,"private_gpu_bytes":model.gpu.private_bytes(),"load_millis":total_load,"packed_block_dot":model.block_dot,
        "gpu_construction_millis":model.load_millis,"dispatches_per_full_token":model.steps(),"trusted_shader_bounds":model.gpu.trusted_bounds,"vector_packed_dot":model.vector_dot,"parallel_gpu_router":model.parallel_router,
        "normal_private_clear_completed":true,"swap_disabled":true,"core_dumps_disabled":true,"samples":samples});
    let output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer_pretty(output, &receipt)?;
    Ok(())
}
