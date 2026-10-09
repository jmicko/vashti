//! Experimental, sequential Rust reference for the borrowed Qwen4exp artifact.
//!
//! This is not exposed through Kuno's HTTP API. It deliberately starts with
//! exact operators and bounded read-only weights; performance, production
//! sampling, and worker privacy acceptance are separate milestones.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zeroize::{Zeroize, Zeroizing};

use crate::cpu::Matrix;
use crate::ops::{
    self, CausalConvState, GatedResidualShape, GdnShape, GdnState, PleHashConfig, PleTokenHistory,
    QsaShape,
};
use crate::{Error, GgufFile, MetadataValue, Result, TensorView};

const D: usize = 2560;
const HC: usize = 4;
const LR: usize = 320;
const LAYERS: usize = 48;
const VOCAB: usize = 248320;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const FF: usize = 640;
const EPS: f32 = 1e-6;
const MAX_REFERENCE_CONTEXT: usize = 4096;
static NEXT_BINDING_ID: AtomicU64 = AtomicU64::new(1);

/// Backend owns computation, never graph semantics or model routing. All tensors
/// retain their original mixed-format GGUF packing and expert row orientation.
pub trait MatrixExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>>;
    /// Optional full routed MoE fusion. The graph still owns routing, all
    /// selected experts and the separate shared expert. None means scalar
    /// projections; an error must never trigger a replay on another engine.
    fn routed_moe(
        &mut self,
        _weights: RoutedWeights<'_>,
        _input: &[f32],
        _selected: &[(usize, f32)],
        _cancel: &AtomicBool,
    ) -> Result<Option<Vec<f32>>> {
        Ok(None)
    }

    /// Backend override for the exact delta rule, with CPU-owned committed
    /// state. This avoids hidden backend state surviving Sequence reset/drop.
    #[allow(clippy::too_many_arguments)]
    fn delta_rule(
        &mut self,
        state: &mut GdnState,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        log_decay: &[f32],
        beta: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("graph cancelled"));
        }
        op(state.step(query, key, value, log_decay, beta))
    }
}

#[derive(Clone, Copy)]
pub struct RoutedWeights<'a> {
    pub gate: TensorView<'a>,
    pub up: TensorView<'a>,
    pub down: TensorView<'a>,
}

#[derive(Default)]
pub struct ScalarExecutor;
impl MatrixExecutor for ScalarExecutor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let matrix = Matrix::new(tensor)?;
        let mut output = vec![0.; matrix.outputs()];
        if let Err(error) = matrix.apply(expert, input, &mut output, cancel) {
            output.fill(0.);
            return Err(error);
        }
        Ok(output)
    }
}

fn op<T>(value: std::result::Result<T, ops::OpsError>) -> Result<T> {
    value.map_err(|_| Error::InvalidFormat("native graph operator"))
}

fn nonnegative(value: &MetadataValue) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|n| u64::try_from(n).ok()))
}
fn integer(file: &GgufFile, key: &str) -> Result<u64> {
    file.metadata()
        .get(key)
        .and_then(nonnegative)
        .ok_or(Error::InvalidFormat("graph integer metadata"))
}
fn integers(file: &GgufFile, key: &str) -> Result<Vec<u64>> {
    file.metadata()
        .get(key)
        .and_then(MetadataValue::as_array)
        .ok_or(Error::InvalidFormat("graph array metadata"))?
        .iter()
        .map(|value| nonnegative(value).ok_or(Error::InvalidFormat("graph array integer")))
        .collect()
}
fn checked_tensor<'a>(file: &'a GgufFile, name: &str, dims: &[u64]) -> Result<TensorView<'a>> {
    let tensor = file
        .tensor(name)
        .ok_or(Error::InvalidFormat("missing graph tensor"))?;
    if tensor.info().dimensions() != dims {
        return Err(Error::InvalidFormat("graph tensor geometry"));
    }
    Ok(tensor)
}
fn vector(tensor: TensorView<'_>) -> Result<Vec<f32>> {
    let len =
        usize::try_from(tensor.info().element_count()).map_err(|_| Error::ArithmeticOverflow)?;
    let size = usize::try_from(tensor.info().byte_len()).map_err(|_| Error::ArithmeticOverflow)?;
    if len > 65536 || size > 512 * 1024 {
        return Err(Error::LimitExceeded("graph static vector"));
    }
    let mut bytes = vec![0u8; size];
    let mut values = vec![0.; len];
    tensor.read_range(0, &mut bytes)?;
    tensor.info().dtype().decode(&bytes, &mut values)?;
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidFormat("non-finite static graph weights"));
    }
    Ok(values)
}

fn checked_ple_shape(config: &PleHashConfig) -> Result<()> {
    // ngram_size=3 and heads_per_ngram=8 imply exactly 16 rows of 160
    // features. A hash with another multiplier count must not be truncated by
    // the scalar graph's row/embedding zip or admitted as a fixed GPU plan.
    if config.history_len() != 2 {
        return Err(Error::InvalidFormat("unsupported graph PLE head geometry"));
    }
    Ok(())
}

/// Exact-geometry binding, rejected before execution if unsupported. Model
/// vectors are public static weights; no request-derived caches live here.
pub struct Qwen4Exp<'a> {
    file: &'a GgufFile,
    vectors: BTreeMap<String, Vec<f32>>,
    ple_hash: PleHashConfig,
    eos: u32,
    binding_id: u64,
    bound_tensors: BTreeSet<String>,
}

/// Opaque identity of one successful graph binding, not merely a file path.
/// A separate execution backend must bind its sequence and public weight cache
/// to this identity and a device-context identity. No sequence state is exposed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GraphBindingId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Gdn,
    Qsa,
}

/// Fixed geometry admitted by Qwen4Exp::bind. Lengths refer to logical f32
/// values, not packed bytes or an allocator's padded span.
#[derive(Clone, Copy, Debug)]
pub struct GraphGeometry {
    pub layers: usize,
    pub residual: GatedResidualShape,
    pub low_rank: usize,
    pub vocabulary: usize,
    pub experts: usize,
    pub experts_used: usize,
    pub expert_hidden: usize,
    pub epsilon: f32,
    pub max_context: usize,
    pub generation_eos: u32,
    pub gdn: GdnShape,
    pub gdn_conv_channels: usize,
    pub gdn_conv_kernel: usize,
    pub gdn_conv_dilation: usize,
    pub qsa: QsaShape,
    pub ple: PleGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleGeometry {
    pub layer: usize,
    pub head_width: usize,
    pub heads: usize,
    pub table_rows: u64,
    pub predecessor_tokens: usize,
    pub boundary_token: u32,
    pub conv_channels: usize,
    pub conv_kernel: usize,
    pub conv_dilation: usize,
}

/// Read-only plan borrowed from a successfully bound graph. There is no public
/// constructor: extra tensors in a GGUF are inaccessible through this view.
/// It exposes public model weights only, never mutable/request-derived state.
/// Tensor views preserve original GGUF packing and dimension-zero row order.
#[derive(Clone, Copy)]
pub struct BoundGraph<'model, 'file> {
    model: &'model Qwen4Exp<'file>,
}

impl<'model, 'file> BoundGraph<'model, 'file> {
    pub fn binding_id(self) -> GraphBindingId {
        GraphBindingId(self.model.binding_id)
    }
    pub fn geometry(self) -> GraphGeometry {
        GraphGeometry {
            layers: LAYERS,
            residual: GatedResidualShape {
                streams: HC,
                width: D,
            },
            low_rank: LR,
            vocabulary: VOCAB,
            experts: EXPERTS,
            experts_used: TOP_K,
            expert_hidden: FF,
            epsilon: EPS,
            max_context: MAX_REFERENCE_CONTEXT,
            generation_eos: self.model.eos,
            gdn: GdnShape::qwen38(),
            gdn_conv_channels: 10240,
            gdn_conv_kernel: 4,
            gdn_conv_dilation: 1,
            qsa: QsaShape::qwen38(),
            ple: PleGeometry {
                layer: 1,
                head_width: 160,
                heads: 16,
                table_rows: 320001536,
                predecessor_tokens: self.model.ple_hash.history_len(),
                boundary_token: 248044,
                conv_channels: HC * D,
                conv_kernel: 4,
                conv_dilation: 3,
            },
        }
    }
    pub fn layer_kind(self, layer: usize) -> Result<AttentionKind> {
        if layer >= LAYERS {
            return Err(Error::OutOfBounds);
        }
        Ok(if layer % 4 == 3 {
            AttentionKind::Qsa
        } else {
            AttentionKind::Gdn
        })
    }
    pub fn tensor_names(self) -> impl Iterator<Item = &'model str> + 'model {
        self.model.bound_tensors.iter().map(String::as_str)
    }
    pub fn tensor(self, name: &str) -> Result<TensorView<'file>> {
        if !self.model.bound_tensors.contains(name) {
            return Err(Error::InvalidFormat("tensor not bound to graph"));
        }
        self.model.weight(name)
    }
    /// Gamma is already folded/direct. Convolution vectors retain their
    /// GGUF kernel-first row layout. Returned storage belongs to the model.
    pub fn static_vector(self, name: &str) -> Result<&'model [f32]> {
        self.model.values(name)
    }
    /// Immutable public vectors from the successfully admitted graph only.
    /// Allows a device owner to prewarm/account these weights before requests.
    pub fn static_vectors(self) -> impl Iterator<Item = (&'model str, &'model [f32])> + 'model {
        self.model
            .vectors
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice()))
    }
    /// Retain the existing scalar hash oracle for row selection. Each request
    /// creates its own PleTokenHistory; never share history in a public cache.
    pub fn ple_hash(self) -> &'model PleHashConfig {
        &self.model.ple_hash
    }
    /// Call before executing from cached weights. Tensor reads independently
    /// check source identity; a cache hit alone does not read the source file.
    pub fn verify_unchanged(self) -> Result<()> {
        self.model.file.verify_file_unchanged()
    }
}

// Only compiled public MTP admission tests use this miniature target binding.
// It admits real packed head/embedding views, but is not a full 48-layer target
// oracle or a way for production callers to forge BoundGraph.
#[cfg(test)]
pub(crate) fn mtp_test_target(file: &GgufFile) -> Result<Qwen4Exp<'_>> {
    let mut model = Qwen4Exp {
        file,
        vectors: BTreeMap::new(),
        ple_hash: PleHashConfig::qwen38(),
        eos: 248046,
        binding_id: NEXT_BINDING_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?,
        bound_tensors: BTreeSet::new(),
    };
    for name in ["token_embd.weight", "output.weight"] {
        model.matrix(name, &[D as u64, VOCAB as u64])?;
    }
    Ok(model)
}

impl<'a> Qwen4Exp<'a> {
    pub fn bind(file: &'a GgufFile) -> Result<Self> {
        if file
            .metadata()
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("qwen4exp")
        {
            return Err(Error::InvalidFormat("unsupported native architecture"));
        }
        if file.metadata().keys().any(|key| {
            key.starts_with("qwen4exp.rope.scaling")
                || key == "qwen4exp.attention.scale"
                || key == "qwen4exp.expert_weights_scale"
        }) || file
            .tensors()
            .keys()
            .any(|name| name.ends_with(".bias") && !name.ends_with(".ssm_dt.bias"))
        {
            return Err(Error::InvalidFormat("unsupported native graph override"));
        }
        for (suffix, expected) in [
            ("embedding_length", D),
            ("block_count", LAYERS),
            ("hyper_connection.count", HC),
            ("hyper_connection.low_rank", LR),
            ("expert_count", EXPERTS),
            ("expert_used_count", TOP_K),
            ("expert_feed_forward_length", FF),
            ("expert_shared_feed_forward_length", FF),
            ("full_attention_interval", 4),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("attention.indexer.head_count", 4),
            ("attention.indexer.key_length", 128),
            ("attention.indexer.top_k", 2048),
            ("ssm.group_count", 16),
            ("ssm.inner_size", 6144),
            ("ssm.state_size", 128),
            ("ssm.time_step_rank", 48),
            ("ssm.conv_kernel", 4),
            ("ple.conv_kernel", 4),
            ("rope.dimension_count", 64),
            ("embedding_length_per_layer_input", 160),
            ("ple.ngram_size", 3),
            ("ple.heads_per_ngram", 8),
            ("ple.eos_token_id", 248044),
        ] {
            if integer(file, &format!("qwen4exp.{suffix}"))? != expected as u64 {
                return Err(Error::InvalidFormat("unsupported native geometry"));
            }
        }
        let epsilon = file
            .metadata()
            .get("qwen4exp.attention.layer_norm_rms_epsilon")
            .and_then(MetadataValue::as_f64)
            .ok_or(Error::InvalidFormat("graph epsilon"))?;
        let theta = file
            .metadata()
            .get("qwen4exp.rope.freq_base")
            .and_then(MetadataValue::as_f64)
            .ok_or(Error::InvalidFormat("graph rope base"))?;
        if !epsilon.is_finite()
            || !theta.is_finite()
            || (epsilon - EPS as f64).abs() > 1e-12
            || theta != 10_000_000.
        {
            return Err(Error::InvalidFormat("unsupported graph normalization"));
        }
        if integers(file, "qwen4exp.ple.layers")? != [1]
            || integers(file, "qwen4exp.rope.dimension_sections")? != [11, 11, 10, 0]
            || integers(file, "qwen4exp.attention.compress_ratios")?
                != (0..LAYERS)
                    .map(|l| if l % 4 == 3 { 4 } else { 0 })
                    .collect::<Vec<_>>()
        {
            return Err(Error::InvalidFormat("unsupported graph layer arrangement"));
        }
        checked_tensor(file, "token_embd.weight", &[D as u64, VOCAB as u64])?;
        checked_tensor(file, "output.weight", &[D as u64, VOCAB as u64])?;
        let ple = checked_tensor(file, "per_layer_token_embd.weight", &[160, 320001536])?;
        let ple_hash = op(PleHashConfig::new(
            integers(file, "qwen4exp.ple.layer_multipliers")?,
            integers(file, "qwen4exp.ple.head_vocab_sizes")?,
            integers(file, "qwen4exp.ple.head_offsets")?,
            8,
            248044,
            VOCAB as u32,
            ple.info().dimensions()[1],
        ))?;
        checked_ple_shape(&ple_hash)?;
        let eos = u32::try_from(integer(file, "tokenizer.ggml.eos_token_id")?)
            .map_err(|_| Error::ArithmeticOverflow)?;
        if eos != 248046 {
            return Err(Error::InvalidFormat("unsupported generation EOS"));
        }
        let binding_id = NEXT_BINDING_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        let mut model = Self {
            file,
            vectors: BTreeMap::new(),
            ple_hash,
            eos,
            binding_id,
            bound_tensors: [
                "token_embd.weight",
                "output.weight",
                "per_layer_token_embd.weight",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        };
        for l in 0..LAYERS {
            let prefix = format!("blk.{l}.");
            for kind in ["attn", "ffn"] {
                let base = format!("{prefix}hc_{kind}");
                model.static_vector(&format!("{base}_norm.weight"), &[10240])?;
                model.matrix(&format!("{base}_down.weight"), &[10240, 320])?;
                model.matrix(&format!("{base}_up.weight"), &[320, 10240])?;
                model.matrix(&format!("{base}_inject.weight"), &[10240, 4])?;
            }
            model.matrix(&format!("{prefix}ffn_gate_inp.weight"), &[2560, 512])?;
            model.static_vector(&format!("{prefix}ffn_gate_inp_shexp.weight"), &[2560])?;
            for suffix in ["ffn_gate_exps", "ffn_up_exps"] {
                model.matrix(&format!("{prefix}{suffix}.weight"), &[2560, 640, 512])?;
            }
            model.matrix(&format!("{prefix}ffn_down_exps.weight"), &[640, 2560, 512])?;
            for suffix in ["ffn_gate_shexp", "ffn_up_shexp"] {
                model.matrix(&format!("{prefix}{suffix}.weight"), &[2560, 640])?;
            }
            model.matrix(&format!("{prefix}ffn_down_shexp.weight"), &[640, 2560])?;
            if l % 4 == 3 {
                for (suffix, dims) in [
                    ("attn_q.weight", [2560, 12288]),
                    ("attn_k.weight", [2560, 512]),
                    ("attn_v.weight", [2560, 512]),
                    ("attn_output.weight", [6144, 2560]),
                    ("indexer.q_proj.weight", [2560, 512]),
                    ("indexer.k_proj.weight", [2560, 128]),
                ] {
                    model.matrix(&format!("{prefix}{suffix}"), &dims)?;
                }
                for (suffix, width) in [
                    ("attn_q_norm.weight", 256),
                    ("attn_k_norm.weight", 256),
                    ("indexer.q_norm.weight", 128),
                    ("indexer.k_norm.weight", 128),
                ] {
                    model.static_vector(&format!("{prefix}{suffix}"), &[width])?;
                }
            } else {
                for (suffix, dims) in [
                    ("attn_qkv.weight", [2560, 10240]),
                    ("attn_gate.weight", [2560, 6144]),
                    ("ssm_alpha.weight", [2560, 48]),
                    ("ssm_beta.weight", [2560, 48]),
                    ("ssm_out.weight", [6144, 2560]),
                ] {
                    model.matrix(&format!("{prefix}{suffix}"), &dims)?;
                }
                for (suffix, dims) in [
                    ("ssm_conv1d.weight", vec![4, 10240]),
                    ("ssm_dt.bias", vec![48]),
                    ("ssm_a", vec![48]),
                    ("ssm_norm.weight", vec![128]),
                ] {
                    model.static_vector(&format!("{prefix}{suffix}"), &dims)?;
                }
            }
        }
        model.static_vector("output_hc_norm.weight", &[10240])?;
        model.matrix("output_hc_down.weight", &[10240, 320])?;
        model.matrix("output_hc_up.weight", &[320, 10240])?;
        for (suffix, dims) in [
            ("ple_key.weight", vec![2560, 10240]),
            ("ple_value.weight", vec![2560, 2560]),
        ] {
            model.matrix(&format!("blk.1.{suffix}"), &dims)?;
        }
        for (suffix, dims) in [
            ("ple_norm_key.weight", vec![10240]),
            ("ple_norm_query.weight", vec![10240]),
            ("ple_norm_conv.weight", vec![10240]),
            ("ple_conv1d.weight", vec![4, 10240]),
        ] {
            model.static_vector(&format!("blk.1.{suffix}"), &dims)?;
        }
        file.verify_file_unchanged()?;
        Ok(model)
    }
    fn matrix(&mut self, name: &str, dims: &[u64]) -> Result<()> {
        Matrix::new(checked_tensor(self.file, name, dims)?)?;
        self.bound_tensors.insert(name.into());
        Ok(())
    }
    fn static_vector(&mut self, name: &str, dims: &[u64]) -> Result<()> {
        let values = vector(checked_tensor(self.file, name, dims)?)?;
        self.vectors.insert(name.into(), values);
        self.bound_tensors.insert(name.into());
        Ok(())
    }
    fn weight(&self, name: &str) -> Result<TensorView<'a>> {
        self.file
            .tensor(name)
            .ok_or(Error::InvalidFormat("missing bound graph weights"))
    }
    fn values(&self, name: &str) -> Result<&[f32]> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or(Error::InvalidFormat("missing bound graph vector"))
    }
    fn project(
        &self,
        executor: &mut impl MatrixExecutor,
        name: &str,
        expert: usize,
        x: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("graph cancelled"));
        }
        let tensor = self.weight(name)?;
        let dims = tensor.info().dimensions();
        if x.len() != dims[0] as usize || expert >= dims.get(2).copied().unwrap_or(1) as usize {
            return Err(Error::InvalidBufferLength);
        }
        let mut output = executor.run(tensor, expert, x, cancel)?;
        if cancel.load(Ordering::Relaxed) {
            output.zeroize();
            return Err(Error::InvalidFormat("graph cancelled"));
        }
        if output.len() != dims[1] as usize || output.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidFormat("graph executor output"));
        }
        Ok(output)
    }
    pub fn eos_token(&self) -> u32 {
        self.eos
    }
    pub fn bound_graph(&self) -> BoundGraph<'_, 'a> {
        BoundGraph { model: self }
    }
    pub fn sequence(&self, context_limit: usize) -> Result<Sequence> {
        Sequence::new(context_limit, &self.ple_hash, self.binding_id)
    }
    fn gr_read(
        &self,
        executor: &mut impl MatrixExecutor,
        residual: &[f32],
        base: &str,
        inject: bool,
        cancel: &AtomicBool,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let shape = op(GatedResidualShape::new(HC, D))?;
        let norm = op(ops::rms_norm(
            residual,
            self.values(&format!("{base}_norm.weight"))?,
            D,
            EPS,
        ))?;
        let down = self.project(executor, &format!("{base}_down.weight"), 0, &norm, cancel)?;
        let low = op(shape.activate_down(&down))?;
        let up = self.project(executor, &format!("{base}_up.weight"), 0, &low, cancel)?;
        let mixed = op(shape.mix(&norm, &up))?;
        let injection = if inject {
            self.project(executor, &format!("{base}_inject.weight"), 0, &norm, cancel)?
        } else {
            Vec::new()
        };
        Ok((mixed, injection))
    }
    fn gdn(
        &self,
        executor: &mut impl MatrixExecutor,
        l: usize,
        x: &[f32],
        state: &mut GdnLayer,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let base = format!("blk.{l}.");
        let qkv = self.project(executor, &format!("{base}attn_qkv.weight"), 0, x, cancel)?;
        let z = self.project(executor, &format!("{base}attn_gate.weight"), 0, x, cancel)?;
        let alpha = self.project(executor, &format!("{base}ssm_alpha.weight"), 0, x, cancel)?;
        let beta = self
            .project(executor, &format!("{base}ssm_beta.weight"), 0, x, cancel)?
            .into_iter()
            .map(ops::sigmoid)
            .collect::<Vec<_>>();
        let mut u = op(state
            .conv
            .step(&qkv, self.values(&format!("{base}ssm_conv1d.weight"))?))?;
        u.iter_mut().for_each(|v| *v = ops::silu(*v));
        let mut q = op(ops::l2_norm_heads(&u[..2048], 128, EPS))?;
        q.iter_mut().for_each(|v| *v /= 128f32.sqrt());
        let k = op(ops::l2_norm_heads(&u[2048..4096], 128, EPS))?;
        let decay = op(ops::gdn_log_decay(
            &alpha,
            self.values(&format!("{base}ssm_dt.bias"))?,
            self.values(&format!("{base}ssm_a"))?,
        ))?;
        let out = executor.delta_rule(
            &mut state.recurrent,
            &q,
            &k,
            &u[4096..],
            &decay,
            &beta,
            cancel,
        )?;
        let gated = op(ops::gdn_output_gate(
            &out,
            &z,
            self.values(&format!("{base}ssm_norm.weight"))?,
            128,
            EPS,
        ))?;
        self.project(
            executor,
            &format!("{base}ssm_out.weight"),
            0,
            &gated,
            cancel,
        )
    }
    fn qsa(
        &self,
        executor: &mut impl MatrixExecutor,
        l: usize,
        x: &[f32],
        state: &mut QsaLayer,
        position: u32,
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let base = format!("blk.{l}.");
        let shape = QsaShape::qwen38();
        let qfull = self.project(executor, &format!("{base}attn_q.weight"), 0, x, cancel)?;
        let mut q = Vec::with_capacity(6144);
        let mut gate = Vec::with_capacity(6144);
        for head in qfull.chunks_exact(512) {
            q.extend_from_slice(&head[..256]);
            gate.extend_from_slice(&head[256..]);
        }
        q = op(ops::rms_norm(
            &q,
            self.values(&format!("{base}attn_q_norm.weight"))?,
            256,
            EPS,
        ))?;
        let kraw = self.project(executor, &format!("{base}attn_k.weight"), 0, x, cancel)?;
        let mut k = op(ops::rms_norm(
            &kraw,
            self.values(&format!("{base}attn_k_norm.weight"))?,
            256,
            EPS,
        ))?;
        let v = self.project(executor, &format!("{base}attn_v.weight"), 0, x, cancel)?;
        op(ops::rope_neox_heads(
            &mut q,
            256,
            64,
            position,
            shape.rope_theta,
        ))?;
        op(ops::rope_neox_heads(
            &mut k,
            256,
            64,
            position,
            shape.rope_theta,
        ))?;
        state.keys.extend_from_slice(&k);
        state.values.extend_from_slice(&v);
        let iqraw = self.project(
            executor,
            &format!("{base}indexer.q_proj.weight"),
            0,
            x,
            cancel,
        )?;
        let mut iq = op(ops::rms_norm(
            &iqraw,
            self.values(&format!("{base}indexer.q_norm.weight"))?,
            128,
            EPS,
        ))?;
        op(ops::rope_neox_heads(
            &mut iq,
            128,
            64,
            position,
            shape.rope_theta,
        ))?;
        let ik = self.project(
            executor,
            &format!("{base}indexer.k_proj.weight"),
            0,
            x,
            cancel,
        )?;
        state.indexer_keys.extend_from_slice(&ik);
        state.positions.push(position);
        let visible = state.positions.len();
        let scores = op(ops::qsa_block_scores(
            &state.indexer_keys,
            &iq,
            self.values(&format!("{base}indexer.k_norm.weight"))?,
            &state.positions,
            visible,
            shape,
        ))?;
        let selected = op(ops::qsa_select_blocks(
            &scores,
            visible,
            shape.block_size,
            shape.token_budget,
        ))?;
        let mut out = op(ops::qsa_attend(
            &q,
            &state.keys,
            &state.values,
            &selected,
            visible,
            shape,
        ))?;
        for (y, z) in out.iter_mut().zip(gate) {
            *y *= ops::sigmoid(z);
        }
        self.project(
            executor,
            &format!("{base}attn_output.weight"),
            0,
            &out,
            cancel,
        )
    }
    fn moe(
        &self,
        executor: &mut impl MatrixExecutor,
        l: usize,
        x: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let base = format!("blk.{l}.");
        let router = self.project(
            executor,
            &format!("{base}ffn_gate_inp.weight"),
            0,
            x,
            cancel,
        )?;
        let selected = op(ops::router_topk(&router, TOP_K, true))?;
        let weights = RoutedWeights {
            gate: self.weight(&format!("{base}ffn_gate_exps.weight"))?,
            up: self.weight(&format!("{base}ffn_up_exps.weight"))?,
            down: self.weight(&format!("{base}ffn_down_exps.weight"))?,
        };
        let fused = executor.routed_moe(weights, x, &selected, cancel)?;
        let has_fused = fused.is_some();
        let mut result = if let Some(output) = fused {
            output
        } else {
            vec![0.; D]
        };
        if result.len() != D
            || result.iter().any(|v| !v.is_finite())
            || cancel.load(Ordering::Relaxed)
        {
            result.zeroize();
            return Err(Error::InvalidFormat("routed MoE backend output"));
        }
        if !has_fused {
            for (expert, weight) in selected {
                let gate = self.project(
                    executor,
                    &format!("{base}ffn_gate_exps.weight"),
                    expert,
                    x,
                    cancel,
                )?;
                let up = self.project(
                    executor,
                    &format!("{base}ffn_up_exps.weight"),
                    expert,
                    x,
                    cancel,
                )?;
                let activation = gate
                    .iter()
                    .zip(up)
                    .map(|(g, u)| ops::silu(*g) * u)
                    .collect::<Vec<_>>();
                let down = self.project(
                    executor,
                    &format!("{base}ffn_down_exps.weight"),
                    expert,
                    &activation,
                    cancel,
                )?;
                for (y, e) in result.iter_mut().zip(down) {
                    *y += weight * e;
                }
            }
        }
        let gate = self.project(
            executor,
            &format!("{base}ffn_gate_shexp.weight"),
            0,
            x,
            cancel,
        )?;
        let up = self.project(
            executor,
            &format!("{base}ffn_up_shexp.weight"),
            0,
            x,
            cancel,
        )?;
        let activation = gate
            .iter()
            .zip(up)
            .map(|(g, u)| ops::silu(*g) * u)
            .collect::<Vec<_>>();
        let shared = self.project(
            executor,
            &format!("{base}ffn_down_shexp.weight"),
            0,
            &activation,
            cancel,
        )?;
        let scalar = self
            .values(&format!("{base}ffn_gate_inp_shexp.weight"))?
            .iter()
            .zip(x)
            .map(|(w, v)| w * v)
            .sum::<f32>();
        if !scalar.is_finite() {
            return Err(Error::InvalidFormat("non-finite shared expert gate"));
        }
        for (y, s) in result.iter_mut().zip(shared) {
            *y += ops::sigmoid(scalar) * s;
        }
        if result.iter().any(|x| !x.is_finite()) {
            return Err(Error::InvalidFormat("non-finite MoE output"));
        }
        Ok(result)
    }
    /// Sequential token reference. On failure, all partially advanced state is
    /// discarded and the sequence must be reset; it can never silently resume.
    pub fn step(
        &self,
        executor: &mut impl MatrixExecutor,
        sequence: &mut Sequence,
        token: u32,
        cancel: &AtomicBool,
        layer_complete: impl FnMut(usize),
    ) -> Result<Vec<f32>> {
        self.step_with_carry(executor, sequence, token, cancel, layer_complete)
            .map(TargetOutput::into_logits)
    }

    /// Public-fixture CPU reference with the target's unnormalized HC carry.
    /// The hidden output is the four contiguous 2560-wide residual streams
    /// after the final FFN combine and BEFORE the output HC RMS/mixer. Logits
    /// use that same residual and the original single head projection.
    ///
    /// A result at position p consumed token p, predicts token p+1, and carries
    /// the hidden input needed to condition MTP on token p+1. Preflight rejects
    /// without advancing state; execution failure clears and poisons it as in
    /// `step`. Output buffers are guarded, but inherited scalar intermediates
    /// have not received whole-stack private error-path erasure acceptance.
    pub fn step_with_carry(
        &self,
        executor: &mut impl MatrixExecutor,
        sequence: &mut Sequence,
        token: u32,
        cancel: &AtomicBool,
        mut layer_complete: impl FnMut(usize),
    ) -> Result<TargetOutput> {
        if sequence.binding_id != self.binding_id
            || sequence.poisoned
            || sequence.position >= sequence.context_limit
            || token as usize >= VOCAB
        {
            return Err(Error::InvalidFormat("invalid native sequence input"));
        }
        sequence.poisoned = true;
        let result = (|| {
            let mut emb = vec![0.; D];
            self.weight("token_embd.weight")?
                .decode_row(token as u64, &mut emb)?;
            let mut residual = emb.repeat(HC);
            let gr = op(GatedResidualShape::new(HC, D))?;
            for l in 0..LAYERS {
                if cancel.load(Ordering::Relaxed) {
                    return Err(Error::InvalidFormat("graph cancelled"));
                }
                if l == 1 {
                    let rows = op(self
                        .ple_hash
                        .rows_and_advance(token as i32, &mut sequence.ple_history))?;
                    let mut ple_emb = vec![0.; D];
                    for (row, out) in rows.into_iter().zip(ple_emb.chunks_exact_mut(160)) {
                        self.weight("per_layer_token_embd.weight")?
                            .decode_row(row as u64, out)?;
                    }
                    let key =
                        self.project(executor, "blk.1.ple_key.weight", 0, &ple_emb, cancel)?;
                    let value =
                        self.project(executor, "blk.1.ple_value.weight", 0, &ple_emb, cancel)?;
                    residual = op(ops::ple_inject_projected(
                        &key,
                        &value,
                        &residual,
                        self.values("blk.1.ple_norm_key.weight")?,
                        self.values("blk.1.ple_norm_query.weight")?,
                        self.values("blk.1.ple_norm_conv.weight")?,
                        self.values("blk.1.ple_conv1d.weight")?,
                        &mut sequence.ple_conv,
                        gr,
                        EPS,
                    ))?
                    .residual;
                }
                let (x, inject) = self.gr_read(
                    executor,
                    &residual,
                    &format!("blk.{l}.hc_attn"),
                    true,
                    cancel,
                )?;
                let attention = match &mut sequence.layers[l] {
                    LayerState::Gdn(state) => self.gdn(executor, l, &x, state, cancel)?,
                    LayerState::Qsa(state) => {
                        self.qsa(executor, l, &x, state, sequence.position as u32, cancel)?
                    }
                };
                residual = op(gr.combine(&residual, &attention, &inject))?;
                let (x, inject) = self.gr_read(
                    executor,
                    &residual,
                    &format!("blk.{l}.hc_ffn"),
                    true,
                    cancel,
                )?;
                let ffn = self.moe(executor, l, &x, cancel)?;
                residual = op(gr.combine(&residual, &ffn, &inject))?;
                layer_complete(l);
            }
            finish_target_output(residual, sequence.position, self.binding_id, |hidden| {
                let (mixed, _) = self.gr_read(executor, hidden, "output_hc", false, cancel)?;
                self.project(executor, "output.weight", 0, &mixed, cancel)
            })
        })();
        match result {
            Ok(output) => {
                if cancel.load(Ordering::Relaxed) {
                    sequence.clear();
                    sequence.poisoned = true;
                    return Err(Error::InvalidFormat("graph cancelled"));
                }
                sequence.position += 1;
                sequence.poisoned = false;
                Ok(output)
            }
            Err(error) => {
                sequence.clear();
                sequence.poisoned = true;
                Err(error)
            }
        }
    }
}

/// One accepted target position and its request-derived output allocations.
/// There is no Debug, serialization, mutable payload or persistence API. Both
/// buffers are wiped on drop, including retained capacity. This does not make
/// the inherited scalar graph a private-request engine.
pub struct TargetOutput {
    logits: Zeroizing<Vec<f32>>,
    hidden: Zeroizing<Vec<f32>>,
    position: usize,
    binding_id: u64,
}
impl TargetOutput {
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    /// Unnormalized 10240-wide HC residual before the final head mixer.
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }
    /// Zero-based consumed target position; the sequence has advanced to p+1.
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn binding_id(&self) -> GraphBindingId {
        GraphBindingId(self.binding_id)
    }
    // Preserve the original step return allocation without a vocabulary-sized
    // clone. Only that legacy adapter transfers logits out of the new guard;
    // its unused carry remains guarded and is wiped when this owner drops.
    fn into_logits(mut self) -> Vec<f32> {
        std::mem::take(&mut *self.logits)
    }
}
impl Drop for TargetOutput {
    fn drop(&mut self) {
        self.position.zeroize();
        self.binding_id.zeroize();
    }
}

fn finish_target_output(
    residual: Vec<f32>,
    position: usize,
    binding_id: u64,
    head: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
) -> Result<TargetOutput> {
    // Own the real pre-head residual before head work so it is wiped even if
    // projection rejects or unwinds. No normalized/mixed substitute or copy.
    let hidden = Zeroizing::new(residual);
    if hidden.len() != HC * D || hidden.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("invalid target HC carry"));
    }
    let logits = Zeroizing::new(head(&hidden)?);
    if logits.len() != VOCAB || logits.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("invalid target logits"));
    }
    Ok(TargetOutput {
        logits,
        hidden,
        position,
        binding_id,
    })
}

struct GdnLayer {
    conv: CausalConvState,
    recurrent: GdnState,
}
struct QsaLayer {
    keys: Vec<f32>,
    values: Vec<f32>,
    indexer_keys: Vec<f32>,
    positions: Vec<u32>,
}
impl QsaLayer {
    fn new(limit: usize) -> Self {
        Self {
            keys: Vec::with_capacity(limit * 512),
            values: Vec::with_capacity(limit * 512),
            indexer_keys: Vec::with_capacity(limit * 128),
            positions: Vec::with_capacity(limit),
        }
    }
    fn clear(&mut self) {
        for buffer in [&mut self.keys, &mut self.values, &mut self.indexer_keys] {
            buffer.resize(buffer.capacity(), 0.);
            buffer.as_mut_slice().zeroize();
            buffer.clear();
        }
        self.positions.resize(self.positions.capacity(), 0);
        self.positions.as_mut_slice().zeroize();
        self.positions.clear();
    }
}
impl Drop for QsaLayer {
    fn drop(&mut self) {
        self.clear();
    }
}
enum LayerState {
    Gdn(GdnLayer),
    Qsa(QsaLayer),
}

/// Per-request state; never stored in the public/static model binding.
pub struct Sequence {
    layers: Vec<LayerState>,
    ple_history: PleTokenHistory,
    ple_conv: CausalConvState,
    context_limit: usize,
    position: usize,
    poisoned: bool,
    binding_id: u64,
}
impl Sequence {
    fn new(context_limit: usize, config: &PleHashConfig, binding_id: u64) -> Result<Self> {
        if !(1..=MAX_REFERENCE_CONTEXT).contains(&context_limit) {
            return Err(Error::LimitExceeded("reference context"));
        }
        let mut layers = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            layers.push(if l % 4 == 3 {
                LayerState::Qsa(QsaLayer::new(context_limit))
            } else {
                LayerState::Gdn(GdnLayer {
                    conv: op(CausalConvState::new(10240, 4, 1))?,
                    recurrent: op(GdnState::new(GdnShape::qwen38()))?,
                })
            });
        }
        Ok(Self {
            layers,
            ple_history: PleTokenHistory::new(config),
            ple_conv: op(CausalConvState::new(10240, 4, 3))?,
            context_limit,
            position: 0,
            poisoned: false,
            binding_id,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn reset(&mut self) {
        self.clear();
        self.poisoned = false;
    }
    fn clear(&mut self) {
        for state in &mut self.layers {
            match state {
                LayerState::Gdn(state) => {
                    state.conv.reset();
                    state.recurrent.reset();
                }
                LayerState::Qsa(state) => state.clear(),
            }
        }
        self.ple_history.reset();
        self.ple_conv.reset();
        self.position = 0;
    }
}
impl Drop for Sequence {
    fn drop(&mut self) {
        self.clear();
    }
}

/// Greedy token selection with an explicit ascending-ID tie rule.
pub fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidFormat("invalid sampling logits"));
    }
    let mut best = 0;
    for id in 1..logits.len() {
        if logits[id] > logits[best] {
            best = id;
        }
    }
    u32::try_from(best).map_err(|_| Error::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    struct TinyFixture(PathBuf);
    impl TinyFixture {
        fn new() -> Self {
            fn string(bytes: &mut Vec<u8>, text: &str) {
                bytes.extend((text.len() as u64).to_le_bytes());
                bytes.extend(text.as_bytes());
            }
            let tensors: [(&str, &[u64], u64, &[f32]); 4] = [
                ("small_matrix.weight", &[2, 2], 0, &[1., 2., 3., 4.]),
                ("small_norm.weight", &[2], 32, &[2., 3.]),
                ("extra.weight", &[2], 64, &[7., 8.]),
                ("invalid_norm.weight", &[2], 96, &[1., f32::NAN]),
            ];
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3_u32.to_le_bytes());
            bytes.extend((tensors.len() as u64).to_le_bytes());
            bytes.extend(0_u64.to_le_bytes());
            for (name, dims, offset, _) in tensors {
                string(&mut bytes, name);
                bytes.extend((dims.len() as u32).to_le_bytes());
                for &d in dims {
                    bytes.extend(d.to_le_bytes());
                }
                bytes.extend(0_u32.to_le_bytes());
                bytes.extend(offset.to_le_bytes());
            }
            let data = bytes.len().next_multiple_of(32);
            bytes.resize(data + 104, 0);
            for (_, _, offset, values) in tensors {
                for (i, &value) in values.iter().enumerate() {
                    let start = data + offset as usize + 4 * i;
                    bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
            let path = std::env::temp_dir().join(format!(
                "kuno-bound-graph-{}-{}.gguf",
                std::process::id(),
                NEXT_BINDING_ID.fetch_add(1, Ordering::Relaxed)
            ));
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            f.write_all(&bytes).unwrap();
            Self(path)
        }
        fn open(&self) -> GgufFile {
            GgufFile::open(&self.0, crate::GgufLimits::default()).unwrap()
        }
    }
    impl Drop for TinyFixture {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }
    // Exercise the exact same private admission helpers that populate the
    // production registry, without downloading/reading a real model. This
    // miniature fixture deliberately does not claim full Qwen4Exp::bind parity.
    fn tiny_binding(file: &GgufFile, identity: u64) -> Qwen4Exp<'_> {
        Qwen4Exp {
            file,
            vectors: BTreeMap::new(),
            ple_hash: PleHashConfig::qwen38(),
            eos: 248046,
            binding_id: identity,
            bound_tensors: BTreeSet::new(),
        }
    }
    #[test]
    fn bound_view_only_exposes_successfully_admitted_tensors_and_vectors() {
        let fixture = TinyFixture::new();
        let file = fixture.open();
        assert!(file.tensor("extra.weight").is_some());
        let mut model = tiny_binding(&file, 1);
        model.matrix("small_matrix.weight", &[2, 2]).unwrap();
        model.static_vector("small_norm.weight", &[2]).unwrap();
        assert!(model.matrix("extra.weight", &[1, 2]).is_err());
        assert!(model.static_vector("invalid_norm.weight", &[2]).is_err());
        let graph = model.bound_graph();
        assert_eq!(
            graph.tensor_names().collect::<Vec<_>>(),
            ["small_matrix.weight", "small_norm.weight"]
        );
        assert!(matches!(
            graph.tensor("extra.weight"),
            Err(Error::InvalidFormat("tensor not bound to graph"))
        ));
        assert!(graph.tensor("invalid_norm.weight").is_err());
        assert!(graph.static_vector("invalid_norm.weight").is_err());
        assert!(graph.static_vector("small_matrix.weight").is_err());
        assert_eq!(graph.static_vector("small_norm.weight").unwrap(), [2., 3.]);
        assert_eq!(
            graph.static_vectors().collect::<Vec<_>>(),
            [("small_norm.weight", &[2., 3.][..])]
        );
        let tensor = graph.tensor("small_matrix.weight").unwrap();
        assert!(tensor.belongs_to(&file));
        assert_eq!(tensor.info().dimensions(), [2, 2]);
        let mut row = [0.; 2];
        tensor.decode_row(1, &mut row).unwrap();
        assert_eq!(row, [3., 4.]);
        graph.verify_unchanged().unwrap();
        // Opaque binding identity distinguishes two bindings over the same file.
        let other = tiny_binding(&file, 2);
        assert!(graph.binding_id() != other.bound_graph().binding_id());
    }
    #[test]
    fn bound_geometry_and_ple_hash_preserve_exact_graph_contract() {
        let fixture = TinyFixture::new();
        let file = fixture.open();
        let model = tiny_binding(&file, 3);
        let graph = model.bound_graph();
        let g = graph.geometry();
        assert_eq!(
            (g.layers, g.residual.streams, g.residual.width, g.low_rank),
            (48, 4, 2560, 320)
        );
        assert_eq!(
            (g.vocabulary, g.experts, g.experts_used, g.expert_hidden),
            (248320, 512, 10, 640)
        );
        assert_eq!((g.max_context, g.generation_eos), (4096, 248046));
        assert_eq!(g.epsilon, 1e-6);
        assert_eq!(g.gdn, GdnShape::qwen38());
        assert_eq!(
            (g.gdn_conv_channels, g.gdn_conv_kernel, g.gdn_conv_dilation),
            (10240, 4, 1)
        );
        assert_eq!(
            (g.qsa.query_heads, g.qsa.kv_heads, g.qsa.head_dim),
            (24, 2, 256)
        );
        assert_eq!(
            (g.qsa.block_size, g.qsa.token_budget, g.qsa.rotary_dims),
            (4, 2048, 64)
        );
        assert_eq!(g.qsa.rope_theta, 10_000_000.0);
        assert_eq!(
            (g.ple.layer, g.ple.head_width, g.ple.heads, g.ple.table_rows),
            (1, 160, 16, 320001536)
        );
        assert_eq!(
            (g.ple.predecessor_tokens, g.ple.boundary_token),
            (2, 248044)
        );
        assert_ne!(g.ple.boundary_token, g.generation_eos);
        assert_eq!(
            (g.ple.conv_channels, g.ple.conv_kernel, g.ple.conv_dilation),
            (10240, 4, 3)
        );
        for l in 0..g.layers {
            assert_eq!(
                graph.layer_kind(l).unwrap(),
                if l % 4 == 3 {
                    AttentionKind::Qsa
                } else {
                    AttentionKind::Gdn
                }
            );
        }
        assert!(matches!(graph.layer_kind(48), Err(Error::OutOfBounds)));
        let mut actual = PleTokenHistory::new(graph.ple_hash());
        let mut expected = PleTokenHistory::new(&PleHashConfig::qwen38());
        for token in [1, 32, 248044, 73] {
            assert_eq!(
                graph
                    .ple_hash()
                    .rows_and_advance(token, &mut actual)
                    .unwrap(),
                PleHashConfig::qwen38()
                    .rows_and_advance(token, &mut expected)
                    .unwrap()
            );
        }
    }
    #[test]
    fn cached_plan_can_detect_source_mutation_without_reading_cached_weights() {
        let fixture = TinyFixture::new();
        let file = fixture.open();
        let model = tiny_binding(&file, 4);
        model.bound_graph().verify_unchanged().unwrap();
        OpenOptions::new()
            .append(true)
            .open(&fixture.0)
            .unwrap()
            .write_all(&[0])
            .unwrap();
        assert!(matches!(
            model.bound_graph().verify_unchanged(),
            Err(Error::FileChanged)
        ));
    }
    #[test]
    fn graph_rejects_hash_head_geometry_inconsistent_with_fixed_embedding() {
        checked_ple_shape(&PleHashConfig::qwen38()).unwrap();
        let inconsistent =
            PleHashConfig::new(vec![1, 3], vec![1; 8], (0..8).collect(), 8, 0, 2, 8).unwrap();
        assert!(matches!(
            checked_ple_shape(&inconsistent),
            Err(Error::InvalidFormat("unsupported graph PLE head geometry"))
        ));
    }

    #[test]
    fn target_carry_taps_final_combine_before_per_stream_normalization_and_head_mix() {
        let original: Vec<_> = (0..HC)
            .flat_map(|stream| {
                (0..D).map(move |d| (stream + 1) as f32 * if d % 2 == 0 { 3. } else { 4. })
            })
            .collect();
        let block: Vec<_> = (0..D)
            .map(|d| if d % 2 == 0 { 1.25 } else { -0.75 })
            .collect();
        let inject = [-4., -2., 0., 4.];
        let shape = GatedResidualShape::new(HC, D).unwrap();
        let final_residual = shape.combine(&original, &block, &inject).unwrap();
        let mut expected_carry = vec![0.; HC * D];
        for stream in 0..HC {
            let weight = 2. / (1. + (-inject[stream] / HC as f32).exp());
            for d in 0..D {
                expected_carry[stream * D + d] = (f64::from(original[stream * D + d])
                    + f64::from(block[d]) * f64::from(weight))
                    as f32;
            }
        }
        for (&actual, &expected) in final_residual.iter().zip(&expected_carry) {
            assert!((actual - expected).abs() < 2e-6);
        }
        let carry_pointer = final_residual.as_ptr();
        let mut expected_mixed = [0.; 2];
        for stream in 0..HC {
            let even = f64::from(final_residual[stream * D]);
            let odd = f64::from(final_residual[stream * D + 1]);
            let inverse = 1. / (((even * even + odd * odd) / 2.) + f64::from(EPS)).sqrt();
            // Original group normalization rounds each result to F32 before
            // the HC head's zero up projection gives sigmoid(0)=1/2.
            expected_mixed[0] += f64::from((even * inverse) as f32) * 0.5 / HC as f64;
            expected_mixed[1] += f64::from((odd * inverse * 2.) as f32) * 0.5 / HC as f64;
        }
        let gamma: Vec<_> = (0..D).map(|d| if d % 2 == 0 { 1. } else { 2. }).collect();
        let mut head_calls = 0;
        let output = finish_target_output(final_residual, 7, 31, |hidden| {
            head_calls += 1;
            let normalized = ops::rms_norm(hidden, &gamma, D, EPS).unwrap();
            assert!(normalized != hidden);
            let mixed = shape.mix(&normalized, &vec![0.; HC * D]).unwrap();
            for d in 0..2 {
                assert_eq!(mixed[d].to_bits(), (expected_mixed[d] as f32).to_bits());
            }
            let mut logits = vec![0.; VOCAB];
            logits[..2].copy_from_slice(&mixed[..2]);
            Ok(logits)
        })
        .unwrap();
        assert_eq!(head_calls, 1);
        assert_eq!(output.hidden().as_ptr(), carry_pointer);
        assert_eq!(output.hidden().len(), 10240);
        assert_eq!(output.logits().len(), VOCAB);
        assert_eq!(output.position(), 7);
        assert!(output.binding_id() == GraphBindingId(31));
        for (&actual, &expected) in output.hidden().iter().zip(&expected_carry) {
            assert!((actual - expected).abs() < 2e-6);
        }
    }

    #[test]
    fn target_output_rejects_bad_extents_and_nonfinite_before_publication() {
        for hidden in [vec![0.; HC * D - 1], vec![f32::NAN; HC * D]] {
            let mut called = false;
            assert!(
                finish_target_output(hidden, 0, 1, |_| {
                    called = true;
                    Ok(vec![0.; VOCAB])
                })
                .is_err()
            );
            assert!(!called);
        }
        for logits in [vec![0.; VOCAB - 1], vec![f32::INFINITY; VOCAB]] {
            assert!(finish_target_output(vec![0.; HC * D], 0, 1, |_| Ok(logits)).is_err());
        }
        let mut logits = Vec::with_capacity(VOCAB + 8);
        logits.resize(VOCAB, 0.);
        logits[5] = -17.;
        let pointer = logits.as_ptr();
        let capacity = logits.capacity();
        let output = finish_target_output(vec![2.; HC * D], 0, 1, |_| Ok(logits)).unwrap();
        let legacy = output.into_logits();
        assert_eq!(legacy.as_ptr(), pointer);
        assert_eq!(legacy.capacity(), capacity);
        assert_eq!(legacy[5], -17.);
    }

    #[test]
    fn target_carry_preflight_keeps_state_and_execution_failure_requires_reset() {
        struct NoProjection;
        impl MatrixExecutor for NoProjection {
            fn run(
                &mut self,
                _: TensorView<'_>,
                _: usize,
                _: &[f32],
                _: &AtomicBool,
            ) -> Result<Vec<f32>> {
                panic!("unexpected projection in bounded preflight fixture")
            }
        }
        let fixture = TinyFixture::new();
        let file = fixture.open();
        let model = tiny_binding(&file, 73);
        // The preflight test owns only its tiny public PLE state; it never
        // claims miniature tensors implement the real 48-layer graph.
        let mut sequence = Sequence {
            layers: Vec::new(),
            ple_history: PleTokenHistory::new(&model.ple_hash),
            ple_conv: CausalConvState::new(10240, 4, 3).unwrap(),
            context_limit: 2,
            position: 1,
            poisoned: false,
            binding_id: 73,
        };
        let cancel = AtomicBool::new(false);
        let mut executor = NoProjection;
        assert!(
            model
                .step_with_carry(&mut executor, &mut sequence, VOCAB as u32, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(sequence.position(), 1);
        assert!(!sequence.poisoned);
        let foreign = tiny_binding(&file, 74);
        assert!(
            foreign
                .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(sequence.position(), 1);
        assert!(!sequence.poisoned);
        sequence.position = 2;
        assert!(
            model
                .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(sequence.position(), 2);
        assert!(!sequence.poisoned);
        sequence.position = 1;
        // Missing embedding is an executed source failure, so no carry or
        // position is returned and the exact existing clear/poison gate runs.
        assert!(
            model
                .step_with_carry(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        assert_eq!(sequence.position(), 0);
        assert!(sequence.poisoned);
        assert!(
            model
                .step(&mut executor, &mut sequence, 0, &cancel, |_| {})
                .is_err()
        );
        sequence.reset();
        assert_eq!(sequence.position(), 0);
        assert!(!sequence.poisoned);
    }
    #[test]
    fn greedy_ties_nonfinite_and_empty() {
        assert_eq!(greedy(&[1., 3., 3., 2.]).unwrap(), 1);
        assert!(greedy(&[]).is_err());
        assert!(greedy(&[0., f32::NAN]).is_err());
    }
    #[test]
    fn integer_metadata_never_rounds_through_float() {
        assert_eq!(
            nonnegative(&MetadataValue::Signed(23703573157769)),
            Some(23703573157769)
        );
        assert_eq!(
            nonnegative(&MetadataValue::Unsigned(u64::MAX)),
            Some(u64::MAX)
        );
        assert_eq!(nonnegative(&MetadataValue::Signed(-1)), None);
        assert_eq!(nonnegative(&MetadataValue::Float(1.)), None);
    }

    #[test]
    fn sequence_reset_and_context_bound() {
        assert!(Sequence::new(0, &PleHashConfig::qwen38(), 1).is_err());
        assert!(Sequence::new(4097, &PleHashConfig::qwen38(), 1).is_err());
        let mut seq = Sequence::new(1, &PleHashConfig::qwen38(), 1).unwrap();
        seq.position = 1;
        seq.poisoned = true;
        seq.reset();
        assert_eq!(seq.position(), 0);
        assert!(!seq.poisoned);
        assert!(
            seq.layers
                .iter()
                .filter_map(|l| if let LayerState::Gdn(s) = l {
                    Some(s)
                } else {
                    None
                })
                .all(|s| s.recurrent.values().iter().all(|x| *x == 0.))
        );
    }
}
