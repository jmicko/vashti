//! Fixed compiled language generation: real tokenizer, normal routes, greedy
//! target verification and actual rejected-prefix state repair. No oracle
//! tokens, HTTP API or private request inputs. Full target packets are retained
//! for this first diagnostic; an independent CPU200 quality gate is not claimed.
#![allow(dead_code)] // Parent owns registration/protected physical acceptance.
use crate::{
    daslab_profile,
    executor::GpuExecutor,
    expert_banks::OriginalPackedProfile,
    mtp_lab_profile::Profile,
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
    GgufFile, GgufLimits, Qwen35Tokenizer, TokenizerLimits,
    model::{Qwen4Exp, greedy},
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
const VOCAB: usize = 248320;
const HIDDEN: usize = 10240;
pub(crate) const MAX_OUTPUTS: usize = 200;
pub(crate) const FIXTURE_HOST_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const TOKENIZER_HOST_BYTES: usize = 192 * 1024 * 1024;
const ABS: f32 = 0.006;
const REL: f32 = 0.0003;
const TEMPLATE_REVISION: &str = "fb58e0dbc8399662c0e47c76578c6e878b14f6cf";
const TEMPLATE_FILE: &str = "serve/chat_template.jinja";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fixture {
    FirstCode,
    WarmProse,
}
impl Fixture {
    fn parse(value: &OsStr) -> std::result::Result<Self, &'static str> {
        match value.to_str() {
            Some("first-code") => Ok(Self::FirstCode),
            Some("warm-prose") => Ok(Self::WarmProse),
            _ => Err("compiled language fixture rejected"),
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::FirstCode => "first-code",
            Self::WarmProse => "warm-prose",
        }
    }
    fn prompt(self) -> &'static str {
        match self {
            Self::FirstCode => {
                "Write a Python function that merges two sorted lists of integers into one sorted list. Explain its time and space complexity."
            }
            Self::WarmProse => {
                "Explain how a hash table works, including collisions, resizing, and average lookup complexity."
            }
        }
    }
    pub(crate) fn prompt_tokens(self) -> usize {
        match self {
            Self::FirstCode => 35,
            Self::WarmProse => 30,
        }
    }
    pub(crate) fn context(self) -> usize {
        self.prompt_tokens() + MAX_OUTPUTS
    }
    fn rendered(self) -> Zeroizing<String> {
        Zeroizing::new(format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            self.prompt()
        ))
    }
}
pub(crate) struct Inputs {
    pub(crate) inline_dispatch: bool,
    pub(crate) fused_sweeps: bool,
    draft_profile: Profile,
    pub(crate) verify3_dense: bool,
    pub(crate) verify3_fused: bool,
    pub(crate) prepared_experts: bool,
    public_existing_fill: bool,
    chunk_io: bool,
    request_reuse: bool,
    live_member_reduction: bool,
    pub(crate) fixture: Fixture,
    directory: PathBuf,
    mtp: PathBuf,
    receipt: PathBuf,
}
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    build: crate::target_lab_build::Build,
) -> std::result::Result<Inputs, &'static str> {
    if args.len() != 3 || args[1] != "intel" || args[2] != "daslab-language-probe" {
        return Err("language lab requires fixed public B70 target");
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
                "KUNO_LAB_LANGUAGE_PUBLIC_ONLY"
                    | "KUNO_LAB_LANGUAGE_FIXTURE"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_TEST_MTP"
                    | "KUNO_PUBLIC_RECEIPT"
                    | "KUNO_LAB_STATIC_CACHE_BYTES"
                    | "KUNO_LAB_DEVICE_DISPATCH"
                    | "KUNO_LAB_MTP_SWEEPS"
                    | "KUNO_LAB_MTP_PROFILE"
                    | "KUNO_LAB_TARGET_DENSE"
                    | "KUNO_LAB_TARGET_EXPERTS"
                    | "KUNO_LAB_TARGET_FILL"
                    | "KUNO_LAB_TARGET_IO"
                    | "KUNO_LAB_TARGET_REUSE"
                    | "KUNO_LAB_TARGET_MOE"
            )
        ) || value.is_empty()
            || values.insert(key, value).is_some()
        {
            return Err("language lab rejects inherited experiments");
        }
    }
    if values.remove(OsStr::new("KUNO_LAB_LANGUAGE_PUBLIC_ONLY")) != Some("1".into())
        || values.remove(OsStr::new("KUNO_LAB_STATIC_CACHE_BYTES")) != Some("8589934592".into())
    {
        return Err("compiled language/cache gates missing");
    }
    let raw = values.remove(OsStr::new("KUNO_LAB_DEVICE_DISPATCH"));
    let inline_dispatch = build.selected(raw.as_deref())?;
    let mut choice = |key: &str, canonical: &str| match values.remove(OsStr::new(key)) {
        None => Ok(false),
        Some(v) if v == canonical => Ok(true),
        _ => Err("language policy requires canonical selection"),
    };
    let prepared_experts = choice("KUNO_LAB_TARGET_EXPERTS", "prepared-verify3")?;
    let public_existing_fill = choice("KUNO_LAB_TARGET_FILL", "direct-public")?;
    let chunk_io = choice("KUNO_LAB_TARGET_IO", "chunk")?;
    let fused_sweeps = choice("KUNO_LAB_MTP_SWEEPS", "fused")?;
    let raw_profile = values.remove(OsStr::new("KUNO_LAB_MTP_PROFILE"));
    let draft_profile = Profile::parse(raw_profile.as_deref())?;
    let raw = values.remove(OsStr::new("KUNO_LAB_TARGET_MOE"));
    let live_member_reduction =
        crate::target_lab_build::live_member_reduction(raw.as_deref(), prepared_experts)?;
    let raw = values.remove(OsStr::new("KUNO_LAB_TARGET_REUSE"));
    let request_reuse =
        crate::target_lab_build::request_reuse(raw.as_deref(), inline_dispatch, chunk_io)?;
    let (verify3_dense, verify3_fused) = match values.remove(OsStr::new("KUNO_LAB_TARGET_DENSE")) {
        None => (false, false),
        Some(v) if v == "verify3" => (true, false),
        Some(v) if v == "verify3-fused" => (true, true),
        _ => return Err("language dense policy requires canonical selection"),
    };
    let fixture = Fixture::parse(
        values
            .remove(OsStr::new("KUNO_LAB_LANGUAGE_FIXTURE"))
            .as_deref()
            .ok_or("language fixture missing")?,
    )?;
    let mut path = |key: &str| {
        values
            .remove(OsStr::new(key))
            .map(PathBuf::from)
            .ok_or("language public path missing")
    };
    Ok(Inputs {
        inline_dispatch,
        fused_sweeps,
        draft_profile,
        verify3_dense,
        verify3_fused,
        prepared_experts,
        public_existing_fill,
        chunk_io,
        request_reuse,
        live_member_reduction,
        fixture,
        directory: path("KUNO_TEST_Q3_DIR")?,
        mtp: path("KUNO_TEST_MTP")?,
        receipt: path("KUNO_PUBLIC_RECEIPT")?,
    })
}
pub(crate) fn validate_limits(l: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    crate::daslab_target_rows_probe::validate_limits(l)
}

fn tokenize(
    file: &GgufFile,
    fixture: Fixture,
    eos: u32,
) -> Result<(Qwen35Tokenizer, Zeroizing<Vec<u32>>)> {
    let limits = TokenizerLimits {
        max_vocabulary: VOCAB,
        max_input_bytes: 1024,
        max_output_tokens: 64,
        max_piece_bytes: 1024,
        max_merge_steps: 8192,
        max_decoded_bytes: 4096,
        max_decode_chunk_bytes: 16384,
        max_allocation_bytes: TOKENIZER_HOST_BYTES,
        ..TokenizerLimits::default()
    };
    let tokenizer = Qwen35Tokenizer::from_gguf(file, limits)?;
    if tokenizer.vocabulary_len() != VOCAB || tokenizer.eos_token_id() != eos {
        return Err("language original tokenizer identity rejected".into());
    }
    let rendered = fixture.rendered();
    let ids = Zeroizing::new(tokenizer.encode(&rendered, true)?);
    let decoded = Zeroizing::new(tokenizer.decode(&ids)?);
    if ids.len() != fixture.prompt_tokens()
        || ids.capacity() > 64
        || decoded.as_str() != rendered.as_str()
        || ids.iter().any(|&t| t >= VOCAB as u32)
    {
        return Err("language compiled template encoding rejected".into());
    }
    Ok((tokenizer, ids))
}
/// Actual target predictions alone define the accepted prefix. EOS clips it
/// before restoring state, including any provisional suffix after terminalEOS.
fn accepted_prefix(
    inputs: &[u32],
    predicted: &[u32],
    eos: u32,
) -> std::result::Result<usize, &'static str> {
    if !(1..=3).contains(&inputs.len())
        || predicted.len() != inputs.len()
        || eos >= VOCAB as u32
        || inputs.iter().chain(predicted).any(|&t| t >= VOCAB as u32)
    {
        return Err("language acceptance geometry rejected");
    }
    let mut accepted = inputs.len();
    for row in 1..inputs.len() {
        if inputs[row] != predicted[row - 1] {
            accepted = row;
            break;
        }
    }
    if let Some(index) = predicted[..accepted].iter().position(|&t| t == eos) {
        accepted = index + 1;
    }
    Ok(accepted)
}
fn teacher_input(prompt: &[u32], position: usize, draft_cursor: usize) -> Result<Option<u32>> {
    if !matches!(prompt.len(), 30 | 35)
        || position >= prompt.len()
        || draft_cursor != position
        || prompt.iter().any(|&t| t >= VOCAB as u32)
    {
        return Err("language actual prompt teacher alignment rejected".into());
    }
    Ok(prompt.get(position + 1).copied())
}
fn repair_vector(expected: &[f32], actual: &[f32], width: usize) -> Result<f32> {
    if expected.len() != width || actual.len() != width {
        return Err("language repair complete vector shape rejected".into());
    }
    let mut maximum = 0f32;
    for (&r, &a) in expected.iter().zip(actual) {
        let difference = (a - r).abs();
        if !r.is_finite() || !a.is_finite() || difference > ABS + REL * r.abs() {
            return Err("language real repaired vector rejected".into());
        }
        maximum = maximum.max(difference);
    }
    Ok(maximum)
}
fn owned_fixture_bytes(
    vectors: &[&Zeroizing<Vec<f32>>],
    ids: &[&Zeroizing<Vec<u32>>],
) -> Result<usize> {
    let mut bytes = std::mem::size_of::<Statistics>() + 1024;
    for v in vectors {
        bytes = bytes
            .checked_add(v.capacity() * 4)
            .ok_or("language host extent overflow")?;
    }
    for v in ids {
        bytes = bytes
            .checked_add(v.capacity() * 4)
            .ok_or("language host extent overflow")?;
    }
    if bytes > FIXTURE_HOST_BYTES {
        return Err("language fixed host owner ceiling rejected".into());
    }
    Ok(bytes)
}
fn elapsed(start: Instant) -> Result<u64> {
    u64::try_from(start.elapsed().as_nanos()).map_err(Into::into)
}
#[derive(Default)]
struct Statistics {
    rounds: u64,
    proposed: u64,
    accepted_drafts: u64,
    continuations: u64,
    draft_ns: u64,
    verify_ns: u64,
    repair_ns: u64,
    repairs: u64,
    repair_logit_max: f32,
    repair_hidden_max: f32,
}
impl Statistics {
    fn add(
        &mut self,
        count: usize,
        accepted: usize,
        draft: u64,
        verify: u64,
        repair: u64,
        recomputed: bool,
    ) -> Result<()> {
        if !(1..=3).contains(&count) || accepted == 0 || accepted > count {
            return Err("language count geometry rejected".into());
        }
        fn add(n: &mut u64, v: u64) -> Result<()> {
            *n = n.checked_add(v).ok_or("language counter overflow")?;
            Ok(())
        }
        add(&mut self.rounds, 1)?;
        add(&mut self.proposed, (count - 1) as u64)?;
        add(&mut self.accepted_drafts, (accepted - 1) as u64)?;
        add(&mut self.continuations, accepted as u64)?;
        add(&mut self.draft_ns, draft)?;
        add(&mut self.verify_ns, verify)?;
        add(&mut self.repair_ns, repair)?;
        add(&mut self.repairs, recomputed as u64)?;
        Ok(())
    }
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
    validate_limits(limits)?;
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_language_failure panic=true")
    }));
    let paths: Vec<_> = (1..=2)
        .map(|i| {
            input.directory.join(format!(
                "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-{i:05}-of-00002.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    if file.file_len() != 83_617_662_656 {
        return Err("language original DAS extent rejected".into());
    }
    daslab_profile::validate_inventory(
        file.tensors()
            .iter()
            .map(|(name, t)| (name.as_str(), t.dtype().gguf_id())),
    )
    .map_err(|_| "language original DAS inventory rejected")?;
    let sidecar = GgufFile::open(&input.mtp, GgufLimits::default())?;
    input.draft_profile.validate_sidecar(&sidecar)?;
    // SAFETY: root's protected runner exclusively verifies immutable original
    // two-shard DAS and explicitly selected original MTP files before this mode.
    unsafe {
        file.enable_mapped_weights()?;
        sidecar.enable_mapped_weights()?;
    }
    let graph = Qwen4Exp::bind(&file)?;
    let draft_graph = input.draft_profile.bind(&sidecar, graph.bound_graph())?;
    let (tokenizer, prompt) = tokenize(&file, input.fixture, graph.eos_token())?;
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
        return Err("language target runtime leaked to draft stream".into());
    }
    let target_retired = Arc::new(AtomicBool::new(false));
    let draft_retired = Arc::new(AtomicBool::new(false));
    let cancel = AtomicBool::new(false);
    let whole = Instant::now();
    let load_disk_before = process_disk_read_bytes()?;
    let mut draft = input.draft_profile.resident(
        &draft_graph,
        draft_client,
        &draft_limits,
        draft_retired.clone(),
        input.fused_sweeps,
    )?;
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
    executor.target_verify3_fused = input.verify3_fused;
    executor.token_io_checkpoint = input.chunk_io;
    executor.enable_transient_resident(target_retired.clone())?;
    executor.fast_dense = true;
    executor.fast_dense_fault = true;
    executor.shared_kernels = true;
    executor.shared_gdn = true;
    let attempt = (|| -> Result<serde_json::Value> {
        let mut config = Config::new(client, limits, 8)?
            .reserve_target_rows()?
            .reserve_latest_target_checkpoint()?;
        if input.prepared_experts {
            config = config.reserve_prepared_verify3_experts()?;
        }
        if input.live_member_reduction {
            config = config.with_live_member_reduction()?;
        }
        if input.public_existing_fill {
            config = config.with_public_existing_expert_fill()?;
        }
        if input.request_reuse {
            config = config.reserve_public_language_request_io(prompt.len(), MAX_OUTPUTS)?;
        }
        let mut target = ResidentModel::new_with_native_batch(
            graph.bound_graph(),
            client.clone(),
            input.fixture.context(),
            &mut executor,
            config,
            true,
        )?;
        let runs = (|| -> Result<serde_json::Value> {
            let load_ns = elapsed(whole)?;
            let prefill = Instant::now();
            let mut carry = Zeroizing::new(vec![0f32; HIDDEN]);
            let mut confirmed = 0u32;
            // Teacher cells contain the real next prompt token and actual
            // current target hidden state. The final prompt carry waits for the
            // first generated token; no dummy initial cell is inserted.
            for start in (0..prompt.len()).step_by(3) {
                let count = (prompt.len() - start).min(3);
                let rows = target.prefill_chunk_with_carry(
                    &mut executor,
                    &prompt[start..start + count],
                    &cancel,
                )?;
                if rows.start() != start
                    || rows.count() != count
                    || rows.binding_id() != graph.bound_graph().binding_id()
                {
                    return Err("language prompt output identity rejected".into());
                }
                for row in 0..count {
                    let position = start + row;
                    if rows.position(row)? != position {
                        return Err("language prompt consumed position rejected".into());
                    }
                    if let Some(token) = teacher_input(&prompt, position, draft.position())? {
                        draft.append_kv(token, rows.hidden(row)?, &cancel)?;
                    } else {
                        confirmed = greedy(rows.logits(row)?)?;
                        carry.copy_from_slice(rows.hidden(row)?);
                    }
                }
                drop(rows);
                if target.completed_target_rows_cursor(&executor)? != start + count {
                    return Err("language prompt actual cursor rejected".into());
                }
            }
            if target.completed_target_rows_cursor(&executor)? != prompt.len()
                || draft.position() != prompt.len() - 1
            {
                return Err("language initial target/draft alignment rejected".into());
            }
            if target.native_live_member_reduction(&executor)? != input.live_member_reduction {
                return Err("language frozen live reduction rejected".into());
            }
            let prefill_ns = elapsed(prefill)?;
            let load_disk_after = process_disk_read_bytes()?;
            let before_uploads = executor.expert_uploads;
            let before_hits = executor.expert_hits;
            let before_bytes = executor.uploaded_bytes;
            let before_weight_reads = executor.weight_read_ns;
            let before_weight_uploads = executor.weight_upload_ns;
            let before_dense = executor.target_verify3_dense_calls;
            let before_fused = executor.target_verify3_fused_calls;
            let before_phases = target.phase_ns.clone();
            let before_traffic = draft.traffic();
            let before_host = crate::target_lab_build::host_io_snapshot();
            let (actual_direct, before_fills) = target.native_public_fill_counters(&executor)?;
            if actual_direct != input.public_existing_fill {
                return Err("language frozen direct fill rejected".into());
            }
            let start = Instant::now();
            if input.request_reuse {
                target.begin_public_language_request_io(&mut executor, &cancel)?;
                if !target.public_language_request_io_active()? {
                    return Err("language actual request scope rejected".into());
                }
            }
            let mut emitted = Zeroizing::new(Vec::with_capacity(MAX_OUTPUTS));
            emitted.push(confirmed);
            let mut decoder = tokenizer.stream_decoder();
            let mut nonempty_text = false;
            if confirmed != graph.eos_token() {
                let text = Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(confirmed)?)?);
                nonempty_text |= !text.is_empty();
            }
            let mut stats = Statistics::default();
            let mut fixture_peak = owned_fixture_bytes(&[&carry], &[&prompt, &emitted])?;
            while emitted.len() < MAX_OUTPUTS && confirmed != graph.eos_token() {
                let position = target.completed_target_rows_cursor(&executor)?;
                if position != prompt.len() + stats.continuations as usize
                    || draft.position() != position - 1
                {
                    return Err("language real target/draft cursor rejected".into());
                }
                let count = (MAX_OUTPUTS - emitted.len()).min(3);
                let mut inputs = Zeroizing::new(Vec::with_capacity(count));
                inputs.push(confirmed);
                let drafting = Instant::now();
                if count > 1 {
                    let output = draft.step_draft(confirmed, &carry, &cancel)?;
                    if output.start() != position - 1 {
                        return Err("language first actual draft stamp rejected".into());
                    }
                    inputs.push(output.greedy());
                    drop(output);
                    if count > 2 {
                        let output = draft.follow_draft(&cancel)?;
                        if output.start() != position {
                            return Err("language second actual draft stamp rejected".into());
                        }
                        inputs.push(output.greedy());
                        drop(output);
                    }
                }
                let draft_ns = elapsed(drafting)?;
                let verifying = Instant::now();
                let mut round =
                    LatestTargetRound::begin(&mut target, &mut executor, &inputs, &cancel)?;
                let rows = round.verify(&cancel)?;
                if rows.start() != position
                    || rows.count() != count
                    || rows.binding_id() != graph.bound_graph().binding_id()
                {
                    return Err("language actual verification identity rejected".into());
                }
                let mut predicted = Zeroizing::new(Vec::with_capacity(count));
                for row in 0..count {
                    if rows.position(row)? != position + row {
                        return Err("language consumed target position rejected".into());
                    }
                    predicted.push(greedy(rows.logits(row)?)?);
                }
                let accepted = accepted_prefix(&inputs, &predicted, graph.eos_token())?;
                let mut actual_carries = Zeroizing::new(Vec::with_capacity(accepted * HIDDEN));
                let mut saved_logits = Zeroizing::new(Vec::with_capacity(if accepted < count {
                    accepted * VOCAB
                } else {
                    0
                }));
                for row in 0..accepted {
                    actual_carries.extend_from_slice(rows.hidden(row)?);
                    if accepted < count {
                        saved_logits.extend_from_slice(rows.logits(row)?);
                    }
                }
                fixture_peak = fixture_peak.max(owned_fixture_bytes(
                    &[&carry, &actual_carries, &saved_logits],
                    &[&prompt, &emitted, &inputs, &predicted],
                )?);
                drop(rows);
                let verify_ns = elapsed(verifying)?;
                let repairing = Instant::now();
                let repaired = round.finish(accepted, &cancel)?;
                let recomputed = repaired.is_some();
                if recomputed != (accepted < count) {
                    return Err("language actual rejection repair evidence missing".into());
                }
                if let Some(recomputed) = repaired {
                    if recomputed.start() != position
                        || recomputed.count() != accepted
                        || recomputed.binding_id() != graph.bound_graph().binding_id()
                    {
                        return Err("language recomputed actual identity rejected".into());
                    }
                    for row in 0..accepted {
                        stats.repair_logit_max = stats.repair_logit_max.max(repair_vector(
                            &saved_logits[row * VOCAB..(row + 1) * VOCAB],
                            recomputed.logits(row)?,
                            VOCAB,
                        )?);
                        stats.repair_hidden_max = stats.repair_hidden_max.max(repair_vector(
                            &actual_carries[row * HIDDEN..(row + 1) * HIDDEN],
                            recomputed.hidden(row)?,
                            HIDDEN,
                        )?);
                        if greedy(recomputed.logits(row)?)? != predicted[row] {
                            return Err("language repair greedy changed".into());
                        }
                    }
                    drop(recomputed);
                }
                if target.completed_target_rows_cursor(&executor)? != position + accepted {
                    return Err("language repaired actual cursor rejected".into());
                }
                if count > 1 {
                    // Retain first real cell; replace only provisional cells
                    // with actual accepted target carry/token conditioning.
                    draft.rewind(position)?;
                    for row in 1..accepted {
                        draft.append_kv(
                            inputs[row],
                            &actual_carries[(row - 1) * HIDDEN..row * HIDDEN],
                            &cancel,
                        )?;
                    }
                } else {
                    draft.append_kv(confirmed, &carry, &cancel)?;
                }
                if draft.position() != position + accepted - 1 {
                    return Err("language repaired draft cell rejected".into());
                }
                carry.copy_from_slice(&actual_carries[(accepted - 1) * HIDDEN..accepted * HIDDEN]);
                confirmed = predicted[accepted - 1];
                for &token in &predicted[..accepted] {
                    emitted.push(token);
                    if token != graph.eos_token() {
                        let text =
                            Zeroizing::new(decoder.push_bytes(tokenizer.token_bytes(token)?)?);
                        nonempty_text |= !text.is_empty();
                    }
                }
                stats.add(
                    count,
                    accepted,
                    draft_ns,
                    verify_ns,
                    elapsed(repairing)?,
                    recomputed,
                )?;
            }
            let tail = Zeroizing::new(decoder.finish());
            nonempty_text |= !tail.is_empty();
            if emitted.len() != stats.continuations as usize + 1
                || emitted.len() > MAX_OUTPUTS
                || target.completed_target_rows_cursor(&executor)?
                    != prompt.len() + emitted.len() - 1
            {
                return Err("language actual emitted/continuation accounting rejected".into());
            }
            let request_close = Instant::now();
            let mut request_scope = None;
            if input.request_reuse {
                let r = target.end_public_language_request_io(&mut executor, &cancel)?;
                if !r.full_io_audit
                    || !r.scope_closed
                    || r.scope != 1
                    || target.public_language_request_io_active()?
                {
                    return Err("language complete request closing audit rejected".into());
                }
                request_scope = Some(
                    serde_json::json!({"scope_sequence":r.scope,"full_io_audit":r.full_io_audit,"scope_closed":r.scope_closed,"closing_audit_ns":r.closing_audit_ns}),
                );
            }
            let request_close_ns = if input.request_reuse {
                elapsed(request_close)?
            } else {
                0
            };
            let elapsed_ns = elapsed(start)?;
            let host_io = crate::target_lab_build::host_io_delta(
                before_host,
                crate::target_lab_build::host_io_snapshot(),
            )?;
            if stats.continuations > 0
                && (input.verify3_dense && executor.target_verify3_dense_calls == before_dense
                    || input.verify3_fused && executor.target_verify3_fused_calls == before_fused)
            {
                return Err("language actual selected projector evidence missing".into());
            }
            let phase_ns: BTreeMap<_, _> = target
                .phase_ns
                .iter()
                .map(|(n, v)| (*n, v - before_phases.get(n).copied().unwrap_or(0)))
                .collect();
            let after_traffic = draft.traffic();
            let (actual_direct, after_fills) = target.native_public_fill_counters(&executor)?;
            if actual_direct != input.public_existing_fill
                || target.native_live_member_reduction(&executor)? != input.live_member_reduction
            {
                return Err("language construction-fixed policy changed".into());
            }
            let physical_read_bytes = process_disk_read_bytes()?
                .checked_sub(load_disk_after)
                .ok_or("language process read counter regressed")?;
            Ok(
                serde_json::json!({"fixture":input.fixture.name(),"prompt_tokens":prompt.len(),"context_capacity":input.fixture.context(),"compiled_prompt_template_revision":TEMPLATE_REVISION,"compiled_prompt_template_file":TEMPLATE_FILE,"load_ns":load_ns,"prefill_ns_including_real_teacher_kv":prefill_ns,"prefill_first_output_count":1,"prefill_first_output_generation_pick_timed":false,"first_output_emission_timed":true,"emitted_tokens_including_eos":emitted.len(),"post_prefill_target_continuations":stats.continuations,"stop_reason":if confirmed==graph.eos_token(){"eos"}else{"limit"},"nonempty_text":nonempty_text,"elapsed_ns":elapsed_ns,"continuation_tokens_per_second":stats.continuations as f64*1e9/elapsed_ns as f64,"emitted_tokens_per_second":emitted.len()as f64*1e9/elapsed_ns as f64,"rounds":stats.rounds,"proposed_drafts":stats.proposed,"accepted_drafts":stats.accepted_drafts,"draft_ns":stats.draft_ns,"verify_ns_full_packets":stats.verify_ns,"repair_ns_including_vector_checks":stats.repair_ns,"actual_target_recomputations":stats.repairs,"repair_logit_max_abs":stats.repair_logit_max,"repair_hidden_max_abs":stats.repair_hidden_max,"repair_abs_tolerance":ABS,"repair_rel_tolerance":REL,"fixture_peak_host_bytes":fixture_peak,"request_scope":request_scope,"request_close_ns_inside_timing":request_close_ns,"host_io":host_io,"target_host_phase_ns":phase_ns,"verify3_dense_calls":executor.target_verify3_dense_calls-before_dense,"verify3_fused_calls":executor.target_verify3_fused_calls-before_fused,"load_and_prefill_physical_read_bytes":load_disk_after.checked_sub(load_disk_before).ok_or("language process read counter regressed")?,"generation_physical_read_bytes":physical_read_bytes,"expert_uploads":executor.expert_uploads-before_uploads,"expert_hits":executor.expert_hits-before_hits,"uploaded_bytes":executor.uploaded_bytes-before_bytes,"weight_read_ns":executor.weight_read_ns-before_weight_reads,"weight_upload_enqueue_ns":executor.weight_upload_ns-before_weight_uploads,"target_fill_counters_before":fill_counter_record(before_fills),"target_fill_counters_after":fill_counter_record(after_fills),"draft_traffic":{"host_read_calls":after_traffic.host_read_calls-before_traffic.host_read_calls,"host_read_bytes":after_traffic.host_read_bytes-before_traffic.host_read_bytes,"explicit_dispatches":after_traffic.explicit_dispatches-before_traffic.explicit_dispatches,"private_upload_calls":after_traffic.private_upload_calls-before_traffic.private_upload_calls,"private_upload_bytes":after_traffic.private_upload_bytes-before_traffic.private_upload_bytes,"minimal_packet_reads":after_traffic.minimal_packet_reads-before_traffic.minimal_packet_reads,"followup_copy_calls":after_traffic.followup_copy_calls-before_traffic.followup_copy_calls}}),
            )
        })();
        let close = target.close();
        let audit = target.public_owned_zero_audit_receipt();
        drop(target);
        close?;
        let (main, sources) = audit.ok_or("language target complete erasure rejected")?;
        let mut record = runs?;
        record["target_close"] = serde_json::json!({"main_buffers":main.buffers,"main_bytes":main.bytes,"source_buffers":sources.buffers,"source_bytes":sources.bytes});
        Ok(record)
    })();
    let target_close = executor.close_transient_resident();
    drop(executor);
    let draft_close = draft.close();
    drop(draft);
    target_close?;
    let draft_close = draft_close?;
    input.draft_profile.validate_close(&draft_close)?;
    let record = attempt?;
    if target_retired.load(Ordering::Acquire) || draft_retired.load(Ordering::Acquire) {
        return Err("language physical stream retired".into());
    }
    file.verify_file_unchanged()?;
    sidecar.verify_file_unchanged()?;
    crate::target_lab_build::validate_runtime(input.inline_dispatch)?;
    if client.public_target_request_io_configured()? != input.request_reuse {
        return Err("language target stream configuration changed".into());
    }
    let value = serde_json::json!({"scope":"compiled-public-language200-real-dual-device-greedy","target_card":"B70","draft_card":"RTX3060","draft_profile":input.draft_profile.name(),"draft_public_reserved_bytes":input.draft_profile.public_bytes(),"device_dispatch":crate::target_lab_build::name(input.inline_dispatch),"observed_inline_device_dispatch":crate::target_lab_build::actual_inline(),"temperature":0,"seed":424242,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0,"thinking":false,"max_outputs":MAX_OUTPUTS,"verify3_dense":input.verify3_dense,"verify3_logical_finite_fused":input.verify3_fused,"prepared_verify3_experts":input.prepared_experts,"live_member_reduction":input.live_member_reduction,"public_existing_fill":input.public_existing_fill,"chunk_io_checkpoint":input.chunk_io,"request_io_reuse":input.request_reuse,"fused_mtp_sweeps":input.fused_sweeps,"fixture_host_reserved_bytes":FIXTURE_HOST_BYTES,"tokenizer_host_cap_bytes":TOKENIZER_HOST_BYTES,"request_reuse_metadata_reserved_bytes":if input.request_reuse{4*1024*1024}else{0},"full_target_packets_retained":true,"greedy_target_verification":true,"repair_original_prefix_vector_checks":true,"independent_cpu200_reference_accepted":false,"forced_oracle_tokens":false,"forced_experts":false,"private_or_production_accepted":false,"record":record,"draft_close":{"private_buffers":draft_close.private.buffers,"private_bytes":draft_close.private.bytes,"public_bytes":draft_close.public_bytes,"main_reserved":draft_close.main_reserved}});
    serde_json::to_writer_pretty(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(input.receipt)?,
        &value,
    )?;
    println!("public_language_complete real_target_greedy=true owners_erased=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args() -> Vec<OsString> {
        ["binary", "intel", "daslab-language-probe"]
            .map(Into::into)
            .to_vec()
    }
    fn env() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "1"),
            ("KUNO_LAB_LANGUAGE_FIXTURE", "first-code"),
            ("KUNO_LAB_STATIC_CACHE_BYTES", "8589934592"),
            ("KUNO_TEST_Q3_DIR", "/original"),
            ("KUNO_TEST_MTP", "/original/sidecar"),
            ("KUNO_PUBLIC_RECEIPT", "/public/receipt"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec()
    }
    #[test]
    fn both_compiled_language_fixtures_require_explicit_bounded_draft_selection() {
        let threaded = crate::target_lab_build::Build::Threaded;
        assert_eq!(
            registration(&args(), env(), threaded)
                .unwrap()
                .draft_profile,
            Profile::SelfContainedQ8
        );
        let mut q2 = env();
        q2.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        for fixture in ["first-code", "warm-prose"] {
            let mut values = q2.clone();
            for (key, value) in &mut values {
                if key == "KUNO_LAB_LANGUAGE_FIXTURE" {
                    *value = fixture.into();
                }
            }
            let selected = registration(&args(), values, threaded).unwrap();
            assert_eq!(selected.draft_profile, Profile::StrataQ2);
            assert_eq!(selected.mtp, PathBuf::from("/original/sidecar"));
            assert_eq!(selected.draft_profile.public_bytes(), 2_539_257_856);
            assert_eq!(
                selected.fixture,
                Fixture::parse(OsStr::new(fixture)).unwrap()
            );
        }
        let mut fused = q2.clone();
        fused.push(("KUNO_LAB_MTP_SWEEPS".into(), "fused".into()));
        assert!(registration(&args(), fused, threaded).unwrap().fused_sweeps);
        let mut inline = q2.clone();
        inline.push(("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()));
        assert_eq!(
            registration(
                &args(),
                inline.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .draft_profile,
            Profile::StrataQ2
        );
        assert!(registration(&args(), inline, threaded).is_err());
        assert!(
            registration(
                &args(),
                q2.clone(),
                crate::target_lab_build::Build::Rejected
            )
            .is_err()
        );
        for key in [
            "KUNO_LAB_LANGUAGE_PUBLIC_ONLY",
            "KUNO_LAB_LANGUAGE_FIXTURE",
            "KUNO_LAB_STATIC_CACHE_BYTES",
            "KUNO_TEST_Q3_DIR",
            "KUNO_TEST_MTP",
            "KUNO_PUBLIC_RECEIPT",
        ] {
            let missing = q2.iter().filter(|(k, _)| k != key).cloned();
            assert!(registration(&args(), missing, threaded).is_err());
        }
        for raw in [
            "",
            "q8",
            "q2",
            "auto",
            "strata-Q2",
            "strata-q2 ",
            "strata-q2\n",
        ] {
            let mut malformed = env();
            malformed.push(("KUNO_LAB_MTP_PROFILE".into(), raw.into()));
            assert!(registration(&args(), malformed, threaded).is_err());
        }
        let mut repeated = q2.clone();
        repeated.push(("KUNO_LAB_MTP_PROFILE".into(), "strata-q2".into()));
        assert!(registration(&args(), repeated, threaded).is_err());
        for key in [
            "KUNO_LAB_LANGUAGE_PROMPT",
            "KUNO_LAB_LANGUAGE_MAX_OUTPUTS",
            "KUNO_LAB_WARM_REPLAY",
            "KUNO_TEST_OTHER",
            "KUNO_PUBLIC_MTP_REFERENCE",
        ] {
            let mut foreign = q2.clone();
            foreign.push((key.into(), "1".into()));
            assert!(registration(&args(), foreign, threaded).is_err());
        }
    }
    #[test]
    fn compiled_prompts_bind_exact_thinking_off_template_and_geometry() {
        for (f, p) in [(Fixture::FirstCode, 35), (Fixture::WarmProse, 30)] {
            assert_eq!(f.prompt_tokens(), p);
            assert_eq!(f.context(), p + 200);
            let r = f.rendered();
            assert_eq!(
                r.as_str(),
                format!(
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                    f.prompt()
                )
            );
            assert!(!r.contains("<|im_start|>system"));
            assert!(!r.contains("enable_thinking"));
            assert!(r.len() < 1024);
        }
        assert_eq!(TEMPLATE_REVISION.len(), 40);
        for bad in ["", "user prompt", "first-code ", "warm", "first-code\n"] {
            assert!(Fixture::parse(OsStr::new(bad)).is_err());
        }
    }
    #[test]
    fn actual_greedy_and_terminal_eos_clip_before_state_commit() {
        let e = 248046;
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 3, 4], e), Ok(3));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 4, 7], e), Ok(2));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[5, 3, 4], e), Ok(1));
        assert_eq!(accepted_prefix(&[1, e, 3], &[e, 3, 4], e), Ok(1));
        assert_eq!(accepted_prefix(&[1, 2, e], &[2, e, 4], e), Ok(2));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[2, 3, e], e), Ok(3));
        assert_eq!(accepted_prefix(&[1, 2, 3], &[4, e, 7], e), Ok(1));
        assert_eq!(accepted_prefix(&[1], &[e], e), Ok(1));
        for (x, y) in [
            (vec![], vec![]),
            (vec![1, 2, 3, 4], vec![2, 3, 4, 5]),
            (vec![1], vec![2, 3]),
            (vec![1, VOCAB as u32], vec![2, 3]),
        ] {
            assert!(accepted_prefix(&x, &y, e).is_err());
        }
        assert!(accepted_prefix(&[1], &[2], VOCAB as u32).is_err());
    }
    #[test]
    fn prompt_teacher_cell_pairing_survives_chunk_edges_and_has_no_dummy_cell() {
        for p in [30, 35] {
            let prompt: Vec<_> = (7..p as u32 + 7).collect();
            for chunk in [1, 2, 3] {
                let mut draft = 0;
                for start in (0..p).step_by(chunk) {
                    for row in 0..chunk.min(p - start) {
                        let at = start + row;
                        let actual = teacher_input(&prompt, at, draft).unwrap();
                        if at + 1 < p {
                            assert_eq!(actual, Some(prompt[at + 1]));
                            draft += 1;
                        } else {
                            assert_eq!(actual, None);
                        }
                    }
                }
                assert_eq!(draft, p - 1);
            }
            assert!(teacher_input(&prompt, 0, 1).is_err());
            assert!(teacher_input(&prompt, p, 0).is_err());
            assert_eq!(teacher_input(&prompt, p - 1, p - 1).unwrap(), None);
        }
        assert!(teacher_input(&[0; 13], 0, 0).is_err());
        let mut wrong = [0; 30];
        wrong[29] = VOCAB as u32;
        assert!(teacher_input(&wrong, 0, 0).is_err());
    }
    #[test]
    fn repair_requires_complete_finite_vectors_and_frozen_whole_graph_gate() {
        assert_eq!(
            repair_vector(&[0.0, 1.0], &[0.005, 1.006], 2).unwrap(),
            0.006000042
        );
        assert!(repair_vector(&[0.0], &[0.0061], 1).is_err());
        assert!(repair_vector(&[1.0], &[1.007], 1).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(repair_vector(&[1.0], &[bad], 1).is_err());
            assert!(repair_vector(&[bad], &[1.0], 1).is_err());
        }
        assert!(repair_vector(&[0.0], &[0.0], HIDDEN).is_err());
        assert!(repair_vector(&[0.0], &[0.0, 0.0], 1).is_err());
    }
    #[test]
    fn transient_fixture_allocations_and_emitted_continuations_are_bounded() {
        let carry = Zeroizing::new(vec![0.0; HIDDEN]);
        let saved = Zeroizing::new(vec![0.0; 2 * VOCAB]);
        let hc = Zeroizing::new(vec![0.0; 3 * HIDDEN]);
        let ids = Zeroizing::new(Vec::<u32>::with_capacity(MAX_OUTPUTS));
        assert!(owned_fixture_bytes(&[&carry, &saved, &hc], &[&ids]).unwrap() < FIXTURE_HOST_BYTES);
        let huge = Zeroizing::new(vec![0.0; FIXTURE_HOST_BYTES / 4]);
        assert!(owned_fixture_bytes(&[&huge], &[]).is_err());
        let mut stats = Statistics::default();
        for _ in 0..66 {
            stats.add(3, 3, 1, 2, 3, false).unwrap();
        }
        stats.add(1, 1, 1, 2, 3, false).unwrap();
        assert_eq!(stats.continuations, 199);
        assert_eq!(stats.continuations + 1, MAX_OUTPUTS as u64);
        assert_eq!(stats.proposed, 132);
        assert_eq!(stats.accepted_drafts, 132);
        assert!(stats.add(4, 4, 0, 0, 0, false).is_err());
        stats.rounds = u64::MAX;
        assert!(stats.add(1, 1, 0, 0, 0, false).is_err());
    }
    #[test]
    fn registration_rejects_private_prompts_oracles_unbounded_outputs_and_foreign_experiments() {
        let threaded = crate::target_lab_build::Build::Threaded;
        assert_eq!(
            registration(&args(), env(), threaded).unwrap().fixture,
            Fixture::FirstCode
        );
        let mut prose = env();
        for (k, v) in &mut prose {
            if k == "KUNO_LAB_LANGUAGE_FIXTURE" {
                *v = "warm-prose".into();
            }
        }
        assert_eq!(
            registration(&args(), prose, threaded).unwrap().fixture,
            Fixture::WarmProse
        );
        for key in [
            "KUNO_PUBLIC_TARGET_CARRY13_REFERENCE",
            "KUNO_LAB_WARM_REPLAY",
            "KUNO_LAB_LANGUAGE_PROMPT",
            "KUNO_LAB_LANGUAGE_MAX_OUTPUTS",
            "KUNO_LAB_TEST",
            "KUNO_TEST_OTHER",
        ] {
            let mut e = env();
            e.push((key.into(), "0".into()));
            assert!(registration(&args(), e, threaded).is_err());
        }
        for bad in ["", "auto", "First-code", "first-code ", "private text"] {
            let mut e = env();
            for (k, v) in &mut e {
                if k == "KUNO_LAB_LANGUAGE_FIXTURE" {
                    *v = bad.into();
                }
            }
            assert!(registration(&args(), e, threaded).is_err());
        }
        let mut duplicate = env();
        duplicate.push(("KUNO_LAB_LANGUAGE_FIXTURE".into(), "first-code".into()));
        assert!(registration(&args(), duplicate, threaded).is_err());
        let mut selected = env();
        selected.extend([
            ("KUNO_LAB_DEVICE_DISPATCH".into(), "inline".into()),
            ("KUNO_LAB_TARGET_IO".into(), "chunk".into()),
            ("KUNO_LAB_TARGET_REUSE".into(), "request".into()),
        ]);
        assert!(
            registration(
                &args(),
                selected.clone(),
                crate::target_lab_build::Build::Inline
            )
            .unwrap()
            .request_reuse
        );
        assert!(registration(&args(), selected.clone(), threaded).is_err());
        selected.retain(|(k, _)| k != "KUNO_LAB_TARGET_IO");
        assert!(registration(&args(), selected, crate::target_lab_build::Build::Inline).is_err());
        assert!(registration(&args(), env(), crate::target_lab_build::Build::Rejected).is_err());
        let mut wrong = args();
        wrong[2] = "daslab-speculative-probe".into();
        assert!(registration(&wrong, env(), threaded).is_err());
    }
}
