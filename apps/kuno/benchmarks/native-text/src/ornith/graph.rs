//! Complete text-only qwen35moe token graph. No CPU experts or weight eviction.
use super::gpu::{Gpu, Result, Step, words};
use kuno_text::{GgufFile, MetadataValue, TensorType};
use std::collections::BTreeMap;
use std::time::Instant;

const WIDTH: u32 = 2048;
const NORM: u32 = 2048;
const MIX: u32 = 4096;
const GATE: u32 = 12288;
const ALPHA: u32 = 16384;
const BETA: u32 = 16416;
const CONV: u32 = 16448;
const CORE: u32 = 24640;
const ATTN: u32 = 28736;
const ROUTER: u32 = 30784;
const EX_GATE: u32 = 31040;
const EX_UP: u32 = 35136;
const EX_ACT: u32 = 39232;
const EX_DOWN: u32 = 43328;
const SH_GATE: u32 = 59712;
const SH_UP: u32 = 60224;
const SH_ACT: u32 = 60736;
const SH_DOWN: u32 = 61248;
const SH_SCALE: u32 = 63296;
const SCORES: u32 = 65536;
const LOGITS: u32 = 131072;
pub const CONTEXT: u32 = 4096;

pub fn validate(file: &GgufFile) -> Result<()> {
    let metadata = file.metadata();
    if metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        != Some("qwen35moe")
    {
        return Err("The resident lab only supports the inspected qwen35moe geometry".into());
    }
    for (suffix, expected) in [
        ("block_count", 40),
        ("embedding_length", 2048),
        ("expert_count", 256),
        ("expert_used_count", 8),
        ("expert_feed_forward_length", 512),
        ("expert_shared_feed_forward_length", 512),
        ("attention.head_count", 16),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("rope.dimension_count", 64),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 32),
        ("ssm.inner_size", 4096),
        ("full_attention_interval", 4),
    ] {
        if metadata
            .get(&format!("qwen35moe.{suffix}"))
            .and_then(MetadataValue::as_u64)
            != Some(expected)
        {
            return Err("Unsupported resident model dimensions".into());
        }
    }
    if metadata.keys().any(|k| {
        k.starts_with("qwen35moe.rope.scaling")
            || k == "qwen35moe.attention.scale"
            || k == "qwen35moe.expert_weights_scale"
    }) {
        return Err("Unsupported resident graph override".into());
    }
    let theta = metadata
        .get("qwen35moe.rope.freq_base")
        .and_then(MetadataValue::as_f64);
    let epsilon = metadata
        .get("qwen35moe.attention.layer_norm_rms_epsilon")
        .and_then(MetadataValue::as_f64);
    if theta != Some(10000000.0)
        || epsilon.is_none_or(|x| !x.is_finite() || (x - 1e-6).abs() > 1e-12)
    {
        return Err("Unsupported resident normalization".into());
    }
    if file.tensors().values().any(|t| {
        !matches!(
            t.dtype(),
            TensorType::F32 | TensorType::Q4K | TensorType::Q6K
        )
    }) {
        return Err("Unsupported resident packed weight type".into());
    }
    let check = |name: &str, dims: &[u64]| -> Result<()> {
        let t = file.tensor(name).ok_or("Missing resident model tensor")?;
        if t.info().dimensions() != dims {
            return Err("Resident tensor shape mismatch".into());
        }
        Ok(())
    };
    check("token_embd.weight", &[2048, 248320])?;
    check("output.weight", &[2048, 248320])?;
    check("output_norm.weight", &[2048])?;
    for layer in 0..40 {
        let prefix = format!("blk.{layer}.");
        for name in ["attn_norm.weight", "post_attention_norm.weight"] {
            check(&(prefix.clone() + name), &[2048])?;
        }
        check(&(prefix.clone() + "ffn_gate_inp.weight"), &[2048, 256])?;
        check(&(prefix.clone() + "ffn_gate_inp_shexp.weight"), &[2048])?;
        for suffix in ["gate", "up"] {
            check(
                &format!("{prefix}ffn_{suffix}_exps.weight"),
                &[2048, 512, 256],
            )?;
            check(&format!("{prefix}ffn_{suffix}_shexp.weight"), &[2048, 512])?;
        }
        check(
            &(prefix.clone() + "ffn_down_exps.weight"),
            &[512, 2048, 256],
        )?;
        check(&(prefix.clone() + "ffn_down_shexp.weight"), &[512, 2048])?;
        if layer % 4 == 3 {
            for (suffix, dims) in [
                ("attn_q.weight", vec![2048, 8192]),
                ("attn_k.weight", vec![2048, 512]),
                ("attn_v.weight", vec![2048, 512]),
                ("attn_output.weight", vec![4096, 2048]),
                ("attn_q_norm.weight", vec![256]),
                ("attn_k_norm.weight", vec![256]),
            ] {
                check(&(prefix.clone() + suffix), &dims)?;
            }
        } else {
            for (suffix, dims) in [
                ("attn_qkv.weight", vec![2048, 8192]),
                ("attn_gate.weight", vec![2048, 4096]),
                ("ssm_alpha.weight", vec![2048, 32]),
                ("ssm_beta.weight", vec![2048, 32]),
                ("ssm_a", vec![32]),
                ("ssm_dt.bias", vec![32]),
                ("ssm_norm.weight", vec![128]),
                ("ssm_conv1d.weight", vec![4, 8192]),
                ("ssm_out.weight", vec![4096, 2048]),
            ] {
                check(&(prefix.clone() + suffix), &dims)?;
            }
        }
    }
    Ok(())
}

pub(super) fn shader(dtype: u32, constants: &str, body: &str) -> String {
    format!(
        "const DTYPE:u32={dtype}u;\n{constants}\n{}\n{body}",
        include_str!("common.wgsl")
    )
}

pub(super) fn matrix_shader(dtype: u32, constants: &str, vector: bool) -> String {
    if vector {
        shader(dtype, constants, include_str!("matrix_vector.wgsl"))
            .replace("arena:array<f32>", "arena4:array<vec4<f32>>")
    } else {
        shader(dtype, constants, include_str!("matrix.wgsl"))
    }
}

struct Weight {
    buffer: wgpu::Buffer,
    dtype: u32,
    row_bytes: u32,
}

pub struct Graph {
    pub gpu: Gpu,
    body: Vec<Step>,
    head: Vec<Step>,
    _weights: BTreeMap<String, Weight>,
    pub weight_bytes: u64,
    pub load_millis: u128,
    pub block_dot: bool,
    pub vector_dot: bool,
    pub parallel_router: bool,
    subgroup_max: u32,
}

impl Graph {
    pub fn new(file: &GgufFile) -> Result<Self> {
        validate(file)?;
        let block_dot = match std::env::var("KUNO_LAB_ORNITH_BLOCK_DOT") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => return Err("Explicit resident packed-block selection rejected".into()),
        };
        let start = Instant::now();
        let vector_dot = match std::env::var("KUNO_LAB_ORNITH_VECTOR_DOT") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => return Err("Explicit resident vector-dot selection rejected".into()),
        };
        let parallel_router = match std::env::var("KUNO_LAB_ORNITH_PARALLEL_ROUTER") {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if value == "1" => true,
            _ => return Err("Explicit resident parallel routing selection rejected".into()),
        };
        let mut gpu = Gpu::new()?;
        if gpu.info.subgroup_min_size < 16 || gpu.info.subgroup_max_size > 32 {
            return Err("Resident lab requires inspected16..32 subgroup bounds".into());
        }
        gpu.state(16); // distinct mutable zero placeholder, never aliases public weights
        let reader = file.positional_reader()?;
        let mut weights = BTreeMap::new();
        let mut weight_bytes = 0;
        for (name, info) in file.tensors() {
            let view = file.tensor(name).ok_or("Resident weight disappeared")?;
            let size = info.byte_len().next_multiple_of(4);
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("public-resident-packed-weight"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut offset = 0;
            while offset < info.byte_len() {
                let bytes = (info.byte_len() - offset).min(32 * 1024 * 1024) as usize;
                let original = reader
                    .mapped_range(view, offset, bytes)?
                    .ok_or("Resident lab requires original read-only mapping")?;
                gpu.queue.write_buffer(&buffer, offset, original);
                gpu.queue.submit([]);
                gpu.wait()?;
                offset += bytes as u64;
            }
            weight_bytes += size;
            weights.insert(
                name.clone(),
                Weight {
                    buffer,
                    dtype: info.dtype().gguf_id(),
                    row_bytes: info.row_bytes() as u32,
                },
            );
        }
        if weight_bytes > 22 * 1024 * 1024 * 1024 {
            return Err("Resident public weight bound exceeded".into());
        }
        let subgroup_max = gpu.info.subgroup_max_size;
        let mut graph = Self {
            gpu,
            body: Vec::new(),
            head: Vec::new(),
            _weights: weights,
            weight_bytes,
            load_millis: 0,
            block_dot,
            vector_dot,
            parallel_router,
            subgroup_max,
        };
        graph.construct()?;
        graph.gpu.wait()?;
        graph.load_millis = start.elapsed().as_millis();
        Ok(graph)
    }

    #[allow(clippy::too_many_arguments)] // Fixed dispatch geometry and binding selection.
    fn op(
        &mut self,
        name: &str,
        source: &str,
        dtype: u32,
        constants: &str,
        first: Option<&str>,
        second: Option<&str>,
        state: Option<usize>,
        p: [u32; 8],
        grid: [u32; 3],
        head: bool,
    ) -> Result<()> {
        let code = if name.starts_with("matrix-") {
            matrix_shader(dtype, constants, self.vector_dot)
        } else {
            shader(dtype, constants, source)
        };
        let pipeline = self.gpu.pipeline(name, code);
        let first = first
            .map(|name| {
                self._weights
                    .get(name)
                    .ok_or("Missing first resident weight")
            })
            .transpose()?;
        let second = second
            .map(|name| {
                self._weights
                    .get(name)
                    .ok_or("Missing second resident weight")
            })
            .transpose()?;
        let mut step = self.gpu.step(
            pipeline,
            first.map(|w| &w.buffer),
            second.map(|w| &w.buffer),
            state,
            p,
            grid,
        );
        step.name = name.to_owned();
        if head {
            self.head.push(step);
        } else {
            self.body.push(step);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // Independent original rows/columns and rank layout.
    fn matrix(
        &mut self,
        name: &str,
        rows: u32,
        cols: u32,
        input: u32,
        output: u32,
        routed: bool,
        ranked: bool,
        head: bool,
    ) -> Result<()> {
        let w = self._weights.get(name).ok_or("Missing resident matrix")?;
        let dtype = w.dtype;
        let row_bytes = w.row_bytes;
        let constants = format!(
            "const ROWS:u32={rows}u;const COLUMNS:u32={cols}u;const ROW_BYTES:u32={row_bytes}u;const ROUTED:bool={routed};const RANKED_INPUT:bool={ranked};"
        );
        let constants = format!("{constants}const BLOCK_DOT:bool={};", self.block_dot);
        self.op(
            &format!(
                "matrix-{dtype}-{rows}-{cols}-{routed}-{ranked}-{}-{}",
                self.block_dot, self.vector_dot
            ),
            include_str!("matrix.wgsl"),
            dtype,
            &constants,
            Some(name),
            None,
            None,
            [input, output, 0, 0, 0, 0, 0, 0],
            [
                rows.div_ceil(128 / self.subgroup_max),
                if routed { 8 } else { 1 },
                1,
            ],
            head,
        )
    }

    fn point(
        &mut self,
        op: u32,
        weight: Option<&str>,
        state: Option<usize>,
        p: [u32; 8],
        groups: u32,
        head: bool,
    ) -> Result<()> {
        let (dtype, row) = weight
            .and_then(|name| self._weights.get(name))
            .map(|w| (w.dtype, w.row_bytes))
            .unwrap_or((0, 0));
        self.op(
            &format!("point-{op}-{dtype}-{row}"),
            include_str!("pointwise.wgsl"),
            dtype,
            &format!("const OP:u32={op}u;const ROW_BYTES:u32={row}u;"),
            weight,
            None,
            state,
            p,
            [groups, 1, 1],
            head,
        )
    }

    fn construct(&mut self) -> Result<()> {
        self.point(0, Some("token_embd.weight"), None, [0; 8], 8, false)?;
        for layer in 0..40 {
            let t = |suffix: &str| format!("blk.{layer}.{suffix}");
            self.point(
                1,
                Some(&t("attn_norm.weight")),
                None,
                [0, NORM, WIDTH, 0, 0, 0, 0, 0],
                1,
                false,
            )?;
            if layer % 4 != 3 {
                let state = self.gpu.state((24576 + 32 * 128 * 128) * 4);
                self.matrix(
                    &t("attn_qkv.weight"),
                    8192,
                    WIDTH,
                    NORM,
                    MIX,
                    false,
                    false,
                    false,
                )?;
                self.matrix(
                    &t("attn_gate.weight"),
                    4096,
                    WIDTH,
                    NORM,
                    GATE,
                    false,
                    false,
                    false,
                )?;
                self.matrix(
                    &t("ssm_alpha.weight"),
                    32,
                    WIDTH,
                    NORM,
                    ALPHA,
                    false,
                    false,
                    false,
                )?;
                self.matrix(
                    &t("ssm_beta.weight"),
                    32,
                    WIDTH,
                    NORM,
                    BETA,
                    false,
                    false,
                    false,
                )?;
                self.point(
                    2,
                    Some(&t("ssm_conv1d.weight")),
                    Some(state),
                    [MIX, CONV, 0, 0, 0, 0, 0, 0],
                    32,
                    false,
                )?;
                self.point(3, None, None, [CONV, 0, 0, 0, 0, 0, 0, 0], 32, false)?;
                self.op(
                    "gdn",
                    include_str!("gdn.wgsl"),
                    0,
                    "",
                    Some(&t("ssm_a")),
                    Some(&t("ssm_dt.bias")),
                    Some(state),
                    [CONV, CORE, ALPHA, BETA, 0, 0, 0, 0],
                    [32, 1, 1],
                    false,
                )?;
                self.point(
                    1,
                    Some(&t("ssm_norm.weight")),
                    None,
                    [CORE, CORE, 128, 0, 0, 1, GATE, 0],
                    32,
                    false,
                )?;
                self.matrix(
                    &t("ssm_out.weight"),
                    WIDTH,
                    4096,
                    CORE,
                    ATTN,
                    false,
                    false,
                    false,
                )?;
            } else {
                let state = self.gpu.state(4096 * 512 * 2 * 4);
                self.matrix(
                    &t("attn_q.weight"),
                    8192,
                    WIDTH,
                    NORM,
                    MIX,
                    false,
                    false,
                    false,
                )?;
                self.matrix(
                    &t("attn_k.weight"),
                    512,
                    WIDTH,
                    NORM,
                    CONV,
                    false,
                    false,
                    false,
                )?;
                self.matrix(
                    &t("attn_v.weight"),
                    512,
                    WIDTH,
                    NORM,
                    CONV + 512,
                    false,
                    false,
                    false,
                )?;
                self.op(
                    "qsa-norm-rope",
                    include_str!("qsa.wgsl"),
                    0,
                    "const OP:u32=0u;",
                    Some(&t("attn_q_norm.weight")),
                    None,
                    None,
                    [MIX, CORE, 0, 0, 0, 0, 0, 0],
                    [16, 1, 1],
                    false,
                )?;
                self.op(
                    "qsa-norm-rope",
                    include_str!("qsa.wgsl"),
                    0,
                    "const OP:u32=0u;",
                    Some(&t("attn_k_norm.weight")),
                    None,
                    Some(state),
                    [CONV, 0, 0, 1, CONV + 512, 0, 0, 0],
                    [2, 1, 1],
                    false,
                )?;
                self.op(
                    "qsa-scores",
                    include_str!("scores.wgsl"),
                    0,
                    "",
                    None,
                    None,
                    Some(state),
                    [CORE, SCORES, 0, 0, 0, 0, 0, 0],
                    [16, 0, 1],
                    false,
                )?;
                self.op(
                    "qsa-softmax-value",
                    include_str!("qsa.wgsl"),
                    0,
                    "const OP:u32=1u;",
                    None,
                    None,
                    Some(state),
                    [SCORES, CORE, MIX, 0, 0, 0, 0, 0],
                    [16, 1, 1],
                    false,
                )?;
                self.matrix(
                    &t("attn_output.weight"),
                    WIDTH,
                    4096,
                    CORE,
                    ATTN,
                    false,
                    false,
                    false,
                )?;
            }
            self.point(
                1,
                Some(&t("post_attention_norm.weight")),
                None,
                [0, NORM, WIDTH, 1, ATTN, 0, 0, 0],
                1,
                false,
            )?;
            self.matrix(
                &t("ffn_gate_inp.weight"),
                256,
                WIDTH,
                NORM,
                ROUTER,
                false,
                false,
                false,
            )?;
            self.op(
                if self.parallel_router {
                    "router-parallel"
                } else {
                    "router"
                },
                if self.parallel_router {
                    include_str!("router_parallel.wgsl")
                } else {
                    include_str!("router.wgsl")
                },
                0,
                "",
                None,
                None,
                None,
                [ROUTER, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1],
                false,
            )?;
            self.matrix(
                &t("ffn_gate_exps.weight"),
                512,
                WIDTH,
                NORM,
                EX_GATE,
                true,
                false,
                false,
            )?;
            self.matrix(
                &t("ffn_up_exps.weight"),
                512,
                WIDTH,
                NORM,
                EX_UP,
                true,
                false,
                false,
            )?;
            self.point(
                4,
                None,
                None,
                [EX_GATE, EX_ACT, 4096, EX_UP, 0, 0, 0, 0],
                16,
                false,
            )?;
            self.matrix(
                &t("ffn_down_exps.weight"),
                WIDTH,
                512,
                EX_ACT,
                EX_DOWN,
                true,
                true,
                false,
            )?;
            self.matrix(
                &t("ffn_gate_shexp.weight"),
                512,
                WIDTH,
                NORM,
                SH_GATE,
                false,
                false,
                false,
            )?;
            self.matrix(
                &t("ffn_up_shexp.weight"),
                512,
                WIDTH,
                NORM,
                SH_UP,
                false,
                false,
                false,
            )?;
            self.point(
                4,
                None,
                None,
                [SH_GATE, SH_ACT, 512, SH_UP, 0, 0, 0, 0],
                2,
                false,
            )?;
            self.matrix(
                &t("ffn_down_shexp.weight"),
                WIDTH,
                512,
                SH_ACT,
                SH_DOWN,
                false,
                false,
                false,
            )?;
            self.matrix(
                &t("ffn_gate_inp_shexp.weight"),
                1,
                WIDTH,
                NORM,
                SH_SCALE,
                false,
                false,
                false,
            )?;
            self.point(
                5,
                None,
                None,
                [EX_DOWN, SH_DOWN, SH_SCALE, 0, 0, 0, 0, 0],
                8,
                false,
            )?;
        }
        self.point(
            1,
            Some("output_norm.weight"),
            None,
            [0, NORM, WIDTH, 0, 0, 0, 0, 0],
            1,
            true,
        )?;
        self.matrix(
            "output.weight",
            248320,
            WIDTH,
            NORM,
            LOGITS,
            false,
            false,
            true,
        )?;
        self.op(
            "argmax-groups",
            include_str!("argmax.wgsl"),
            0,
            "const OP:u32=0u;",
            None,
            None,
            None,
            [LOGITS, 0, 0, 0, 0, 0, 0, 0],
            [61, 1, 1],
            true,
        )?;
        self.op(
            "argmax-final",
            include_str!("argmax.wgsl"),
            0,
            "const OP:u32=1u;",
            None,
            None,
            None,
            [0; 8],
            [1, 1, 1],
            true,
        )?;
        Ok(())
    }

    pub fn token(&self, token: u32, pos: u32, head: bool) -> Result<()> {
        if token >= 248320 || pos >= CONTEXT {
            return Err("Resident token or context limit exceeded".into());
        }
        self.gpu
            .queue
            .write_buffer(&self.gpu.control, 0, &words(&[token, pos, 0, 0]));
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for step in self
                .body
                .iter()
                .chain(if head { self.head.as_slice() } else { &[] })
            {
                pass.set_pipeline(&step.pipeline);
                pass.set_bind_group(0, &step.bindings, &[]);
                pass.dispatch_workgroups(
                    step.grid[0],
                    if step.grid[1] == 0 {
                        (pos + 1).div_ceil(64)
                    } else {
                        step.grid[1]
                    },
                    step.grid[2],
                );
            }
        }
        self.gpu.queue.submit([encoder.finish()]);
        Ok(())
    }

    pub fn next(&self) -> Result<u32> {
        let result = self.gpu.result()?;
        if result[0] >= 248320 || result[2] != 0 {
            return Err("Resident final output rejected".into());
        }
        Ok(result[0])
    }

    /// Numerical diagnosis on the compiled-in public reference sequence only.
    /// Never read logits from a real inference request through this lab path.
    pub fn public_expected_rank(&self, expected: u32) -> Result<serde_json::Value> {
        if expected >= 248320 {
            return Err("Public reference ID out of range".into());
        }
        let bytes = self
            .gpu
            .public_read(&self.gpu.arena, LOGITS as u64 * 4, 248320 * 4)?;
        let at = |id: usize| f32::from_le_bytes(bytes[id * 4..id * 4 + 4].try_into().unwrap());
        let expected_value = at(expected as usize);
        let mut rank = 1u32;
        let mut maximum = f32::NEG_INFINITY;
        let mut second = f32::NEG_INFINITY;
        let mut id = 0u32;
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes(chunk.try_into().unwrap());
            if !value.is_finite() {
                return Err("Public logits contain non-finite values".into());
            }
            if value > expected_value || (value == expected_value && i < expected as usize) {
                rank += 1;
            }
            if value > maximum {
                second = maximum;
                maximum = value;
                id = i as u32;
            } else if value > second {
                second = value;
            }
        }
        Ok(
            serde_json::json!({"expected_id":expected,"native_id":id,"expected_rank":rank,"expected_gap_from_maximum":maximum-expected_value,"native_first_second_margin":maximum-second}),
        )
    }

    pub fn steps(&self) -> usize {
        self.body.len() + self.head.len()
    }

    /// One fixed-public following token with per-dispatch GPU counters. These
    /// instrumented timings are diagnostics, never counter-free throughput.
    pub fn profile_token(&self, token: u32, pos: u32) -> Result<serde_json::Value> {
        if token >= 248320
            || pos >= CONTEXT
            || !self
                .gpu
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
        {
            return Err("Public timestamp profile not admitted".into());
        }
        let count = (self.steps() * 2) as u32;
        let queries = self.gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("public-ornith-operator-profile"),
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let resolved = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("public-timestamp-resolution"),
            size: count as u64 * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu
            .queue
            .write_buffer(&self.gpu.control, 0, &words(&[token, pos, 0, 0]));
        let start = Instant::now();
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            for (i, step) in self.body.iter().chain(self.head.iter()).enumerate() {
                pass.set_pipeline(&step.pipeline);
                pass.set_bind_group(0, &step.bindings, &[]);
                pass.write_timestamp(&queries, i as u32 * 2);
                pass.dispatch_workgroups(
                    step.grid[0],
                    if step.grid[1] == 0 {
                        (pos + 1).div_ceil(64)
                    } else {
                        step.grid[1]
                    },
                    step.grid[2],
                );
                pass.write_timestamp(&queries, i as u32 * 2 + 1);
            }
        }
        encoder.resolve_query_set(&queries, 0..count, &resolved, 0);
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.wait()?;
        let wall_ns = start.elapsed().as_nanos();
        let counters = self.gpu.public_read(&resolved, 0, count as u64 * 8)?;
        let ticks = counters
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let period = self.gpu.queue.get_timestamp_period() as f64;
        let mut groups: BTreeMap<String, (u32, f64)> = BTreeMap::new();
        for (i, step) in self.body.iter().chain(self.head.iter()).enumerate() {
            let entry = groups.entry(step.name.clone()).or_default();
            entry.0 += 1;
            entry.1 += ticks[i * 2 + 1].saturating_sub(ticks[i * 2]) as f64 * period;
        }
        let mut groups = groups.into_iter().map(|(name,(count,ns))| serde_json::json!({"operator":name,"dispatches":count,"gpu_nanoseconds":ns})).collect::<Vec<_>>();
        groups.sort_by(|a, b| {
            b["gpu_nanoseconds"]
                .as_f64()
                .unwrap()
                .total_cmp(&a["gpu_nanoseconds"].as_f64().unwrap())
        });
        Ok(
            serde_json::json!({"instrumented":true,"wall_nanoseconds":wall_ns,"gpu_span_nanoseconds":(ticks[ticks.len()-1]-ticks[0]) as f64*period,"operators":groups}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_resident_kernel_variants_parse_and_validate_offline() {
        let bodies = [
            include_str!("matrix.wgsl"),
            include_str!("pointwise.wgsl"),
            include_str!("gdn.wgsl"),
            include_str!("qsa.wgsl"),
            include_str!("scores.wgsl"),
            include_str!("router.wgsl"),
            include_str!("router_parallel.wgsl"),
            include_str!("argmax.wgsl"),
        ];
        for dtype in [0, 12, 14] {
            let code = matrix_shader(
                dtype,
                "const ROWS:u32=2048u;const COLUMNS:u32=2048u;const ROW_BYTES:u32=1152u;const ROUTED:bool=true;const RANKED_INPUT:bool=false;",
                true,
            );
            let module = wgpu::naga::front::wgsl::parse_str(&code)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            for body in bodies {
                let code = shader(
                    dtype,
                    "const ROWS:u32=2048u;const COLUMNS:u32=2048u;const ROW_BYTES:u32=1152u;const ROUTED:bool=true;const RANKED_INPUT:bool=false;const BLOCK_DOT:bool=true;const OP:u32=0u;",
                    body,
                );
                let module = wgpu::naga::front::wgsl::parse_str(&code)
                    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
            }
        }
    }
}
