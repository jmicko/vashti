//! Strict, sequential Rust reference for the self-contained Qwen4Exp MTP head.
//!
//! This is an internal laboratory graph, not an HTTP model or speculative
//! controller. It takes the preceding *wide* target/draft residual and current
//! token. Draft outputs must be verified by the target; they are never responses.
//! Equations were independently translated from llama.cpp PR28243 revision
//! 6fcaa16f4b360649933a54d1f91ad40ed35c0e11, src/models/qwen4exp.cpp (MIT).
//! The pinned Unsloth Q8 sidecar has one dense-attention block at index48.
//! Its four stored indexer tensors are inspected but deliberately not executed.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use zeroize::{Zeroize, Zeroizing};

use crate::cpu::Matrix;
use crate::model::MatrixExecutor;
use crate::ops::{self, GatedResidualShape};
use crate::{Error, GgufFile, MetadataValue, Result, TensorType, TensorView};

const D: usize = 2560;
const HC: usize = 4;
const WIDE: usize = D * HC;
const VOCAB: usize = 248320;
const EPS: f32 = 1e-6;
const MAX_CONTEXT: usize = 2048;
static NEXT_BINDING: AtomicU64 = AtomicU64::new(1);
type Private = Zeroizing<Vec<f32>>;

fn op<T>(value: std::result::Result<T, ops::OpsError>) -> Result<T> {
    value.map_err(|_| Error::InvalidFormat("MTP graph operator"))
}

fn integer(file: &GgufFile, key: &str) -> Option<u64> {
    file.metadata().get(key).and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_i64().and_then(|n| u64::try_from(n).ok()))
    })
}

fn private(values: Vec<f32>) -> Private {
    Zeroizing::new(values)
}

/// Original sidecar weights only. No sequence, prompt, draft or cache is shared.
/// Admission is deliberately limited to the inspected self-contained Q8 head.
pub struct MtpGraph<'a> {
    file: &'a GgufFile,
    vectors: BTreeMap<String, Vec<f32>>,
    names: BTreeSet<String>,
    binding: u64,
}

impl<'a> MtpGraph<'a> {
    pub fn bind(file: &'a GgufFile) -> Result<Self> {
        if file
            .metadata()
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("qwen4exp")
        {
            return Err(Error::InvalidFormat("unsupported MTP architecture"));
        }
        for (suffix, value) in [
            ("block_count", 49),
            ("nextn_predict_layers", 1),
            ("embedding_length", D),
            ("hyper_connection.count", HC),
            ("hyper_connection.low_rank", 320),
            ("expert_count", 512),
            ("expert_used_count", 10),
            ("expert_feed_forward_length", 640),
            ("expert_shared_feed_forward_length", 640),
            ("attention.head_count", 24),
            ("attention.head_count_kv", 2),
            ("attention.key_length", 256),
            ("attention.value_length", 256),
            ("rope.dimension_count", 64),
        ] {
            if integer(file, &format!("qwen4exp.{suffix}")) != Some(value as u64) {
                return Err(Error::InvalidFormat("unsupported MTP geometry"));
            }
        }
        let epsilon = file
            .metadata()
            .get("qwen4exp.attention.layer_norm_rms_epsilon")
            .and_then(MetadataValue::as_f64);
        let theta = file
            .metadata()
            .get("qwen4exp.rope.freq_base")
            .and_then(MetadataValue::as_f64);
        if epsilon.is_none_or(|e| !e.is_finite() || (e - f64::from(EPS)).abs() > 1e-12)
            || theta != Some(10_000_000.0)
            || file.metadata().keys().any(|k| {
                k.starts_with("qwen4exp.rope.scaling")
                    || k == "qwen4exp.attention.scale"
                    || k == "qwen4exp.expert_weights_scale"
            })
        {
            return Err(Error::InvalidFormat(
                "unsupported MTP normalization override",
            ));
        }
        let sections = file
            .metadata()
            .get("qwen4exp.rope.dimension_sections")
            .and_then(MetadataValue::as_array)
            .ok_or(Error::InvalidFormat("missing MTP rope sections"))?;
        if sections.len() != 4
            || sections.iter().zip([11, 11, 10, 0]).any(|(a, b)| {
                a.as_u64()
                    .or_else(|| a.as_i64().and_then(|v| u64::try_from(v).ok()))
                    != Some(b)
            })
        {
            return Err(Error::InvalidFormat("unsupported MTP rope sections"));
        }
        let ratios = file
            .metadata()
            .get("qwen4exp.attention.compress_ratios")
            .and_then(MetadataValue::as_array)
            .ok_or(Error::InvalidFormat("missing MTP block arrangement"))?;
        if ratios.len() != 49
            || ratios.iter().enumerate().any(|(i, v)| {
                let expected = if i == 48 || i % 4 == 3 { 4 } else { 0 };
                v.as_u64()
                    .or_else(|| v.as_i64().and_then(|n| u64::try_from(n).ok()))
                    != Some(expected)
            })
        {
            return Err(Error::InvalidFormat("unsupported MTP block arrangement"));
        }
        let binding = NEXT_BINDING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        let mut graph = Self {
            file,
            vectors: BTreeMap::new(),
            names: BTreeSet::new(),
            binding,
        };
        use TensorType::{Bf16, F32, Q8_0};
        for (name, dims, dtype) in [
            ("output.weight", vec![D as u64, VOCAB as u64], Q8_0),
            ("token_embd.weight", vec![D as u64, VOCAB as u64], Q8_0),
            ("blk.48.attn_q.weight", vec![2560, 12288], Q8_0),
            ("blk.48.attn_k.weight", vec![2560, 512], Q8_0),
            ("blk.48.attn_v.weight", vec![2560, 512], Q8_0),
            ("blk.48.attn_output.weight", vec![6144, 2560], Q8_0),
            ("blk.48.attn_q_norm.weight", vec![256], F32),
            ("blk.48.attn_k_norm.weight", vec![256], F32),
            ("blk.48.ffn_gate_exps.weight", vec![2560, 640, 512], Q8_0),
            ("blk.48.ffn_up_exps.weight", vec![2560, 640, 512], Q8_0),
            ("blk.48.ffn_down_exps.weight", vec![640, 2560, 512], Q8_0),
            ("blk.48.ffn_gate_shexp.weight", vec![2560, 640], Q8_0),
            ("blk.48.ffn_up_shexp.weight", vec![2560, 640], Q8_0),
            ("blk.48.ffn_down_shexp.weight", vec![640, 2560], Q8_0),
            ("blk.48.ffn_gate_inp.weight", vec![2560, 512], F32),
            ("blk.48.ffn_gate_inp_shexp.weight", vec![2560], F32),
            ("blk.48.indexer.q_proj.weight", vec![2560, 512], Bf16),
            ("blk.48.indexer.k_proj.weight", vec![2560, 128], Bf16),
            ("blk.48.indexer.q_norm.weight", vec![128], F32),
            ("blk.48.indexer.k_norm.weight", vec![128], F32),
            ("blk.48.nextn.eh_proj.weight", vec![5120, 2560], Q8_0),
            ("blk.48.nextn.enorm.weight", vec![2560], F32),
            ("blk.48.nextn.hnorm.weight", vec![10240], F32),
            ("blk.48.nextn.hc_head_norm.weight", vec![10240], F32),
            ("blk.48.nextn.hc_head_down.weight", vec![10240, 320], Q8_0),
            ("blk.48.nextn.hc_head_up.weight", vec![320, 10240], Q8_0),
        ] {
            graph.admit(name, &dims, dtype)?;
        }
        for base in ["blk.48.hc_attn", "blk.48.hc_ffn"] {
            for (suffix, dims, dtype) in [
                ("norm", vec![10240], F32),
                ("down", vec![10240, 320], Q8_0),
                ("up", vec![320, 10240], Q8_0),
                ("inject", vec![10240, 4], Q8_0),
            ] {
                graph.admit(&format!("{base}_{suffix}.weight"), &dims, dtype)?;
            }
        }
        if graph.names.len() != 34 || file.tensors().keys().any(|n| !graph.names.contains(n)) {
            return Err(Error::InvalidFormat("unexpected MTP tensor"));
        }
        file.verify_file_unchanged()?;
        Ok(graph)
    }

    fn admit(&mut self, name: &str, dims: &[u64], dtype: TensorType) -> Result<()> {
        let tensor = self
            .file
            .tensor(name)
            .ok_or(Error::InvalidFormat("missing MTP tensor"))?;
        if tensor.info().dimensions() != dims || tensor.info().dtype() != dtype {
            return Err(Error::InvalidFormat("MTP tensor geometry or dtype"));
        }
        if dims.len() == 1 {
            let mut values = vec![0.; dims[0] as usize];
            tensor.decode_row(0, &mut values)?;
            if values.iter().any(|x| !x.is_finite()) {
                return Err(Error::InvalidFormat("non-finite MTP static vector"));
            }
            self.vectors.insert(name.into(), values);
        } else {
            Matrix::new(tensor)?;
        }
        self.names.insert(name.into());
        Ok(())
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.names.iter().map(String::as_str)
    }
    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>> {
        if !self.names.contains(name) {
            return Err(Error::InvalidFormat("unbound MTP tensor"));
        }
        self.file
            .tensor(name)
            .ok_or(Error::InvalidFormat("missing bound MTP tensor"))
    }
    fn vector(&self, name: &str) -> Result<&[f32]> {
        self.vectors
            .get(name)
            .map(Vec::as_slice)
            .ok_or(Error::InvalidFormat("unbound MTP vector"))
    }

    pub fn sequence(&self, context_limit: usize) -> Result<MtpSequence> {
        MtpSequence::new(context_limit, self.binding)
    }

    fn project(
        &self,
        executor: &mut impl MatrixExecutor,
        name: &str,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Private> {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("MTP cancelled"));
        }
        let tensor = self.tensor(name)?;
        let matrix = Matrix::new(tensor)?;
        if input.len() != matrix.inputs() || expert >= matrix.experts() {
            return Err(Error::InvalidBufferLength);
        }
        let result = private(executor.run(tensor, expert, input, cancel)?);
        if cancel.load(Ordering::Relaxed)
            || result.len() != matrix.outputs()
            || result.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidFormat("invalid MTP projection output"));
        }
        Ok(result)
    }

    fn mix(
        &self,
        executor: &mut impl MatrixExecutor,
        residual: &[f32],
        base: &str,
        inject: bool,
        cancel: &AtomicBool,
    ) -> Result<(Private, Private)> {
        let shape = op(GatedResidualShape::new(HC, D))?;
        let norm = private(op(ops::rms_norm(
            residual,
            self.vector(&format!("{base}_norm.weight"))?,
            D,
            EPS,
        ))?);
        let down = self.project(executor, &format!("{base}_down.weight"), 0, &norm, cancel)?;
        let low = private(op(shape.activate_down(&down))?);
        let up = self.project(executor, &format!("{base}_up.weight"), 0, &low, cancel)?;
        let mixed = private(op(shape.mix(&norm, &up))?);
        let injection = if inject {
            self.project(executor, &format!("{base}_inject.weight"), 0, &norm, cancel)?
        } else {
            private(Vec::new())
        };
        Ok((mixed, injection))
    }

    fn condition(
        &self,
        executor: &mut impl MatrixExecutor,
        token: u32,
        hidden: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Private> {
        let mut embedding = private(vec![0.; D]);
        self.tensor("token_embd.weight")?
            .decode_row(u64::from(token), &mut embedding)?;
        let e = private(op(ops::rms_norm(
            &embedding,
            self.vector("blk.48.nextn.enorm.weight")?,
            D,
            EPS,
        ))?);
        let h = private(op(ops::rms_norm(
            hidden,
            self.vector("blk.48.nextn.hnorm.weight")?,
            D,
            EPS,
        ))?);
        project_conditioned(&e, &h, |concat| {
            self.project(executor, "blk.48.nextn.eh_proj.weight", 0, concat, cancel)
        })
    }

    fn attention(
        &self,
        executor: &mut impl MatrixExecutor,
        input: &[f32],
        sequence: &mut MtpSequence,
        cancel: &AtomicBool,
    ) -> Result<Private> {
        let full = self.project(executor, "blk.48.attn_q.weight", 0, input, cancel)?;
        let (raw_query, gate) = split_query_gate(&full, 256)?;
        let mut query = private(op(ops::rms_norm(
            &raw_query,
            self.vector("blk.48.attn_q_norm.weight")?,
            256,
            EPS,
        ))?);
        let (mut key, value) = self.key_value(executor, input, cancel)?;
        op(ops::rope_neox_heads(
            &mut query,
            256,
            64,
            sequence.position as u32,
            10_000_000.0,
        ))?;
        append_key_value(sequence, &mut key, &value)?;
        // All visible positions are attended. The stored QSA indexer metadata
        // must not silently impose a 2048-token selection window here.
        let mut attended =
            dense_attention(&query, &sequence.keys, &sequence.values, 24, 2, 256, cancel)?;
        for (out, g) in attended.iter_mut().zip(gate.iter()) {
            *out *= ops::sigmoid(*g);
        }
        self.project(executor, "blk.48.attn_output.weight", 0, &attended, cancel)
    }

    fn key_value(
        &self,
        executor: &mut impl MatrixExecutor,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<(Private, Private)> {
        let raw_key = self.project(executor, "blk.48.attn_k.weight", 0, input, cancel)?;
        let key = private(op(ops::rms_norm(
            &raw_key,
            self.vector("blk.48.attn_k_norm.weight")?,
            256,
            EPS,
        ))?);
        let value = self.project(executor, "blk.48.attn_v.weight", 0, input, cancel)?;
        Ok((key, value))
    }

    fn expert(
        &self,
        executor: &mut impl MatrixExecutor,
        input: &[f32],
        expert: usize,
        shared: bool,
        cancel: &AtomicBool,
    ) -> Result<Private> {
        let suffix = if shared { "shexp" } else { "exps" };
        let gate = self.project(
            executor,
            &format!("blk.48.ffn_gate_{suffix}.weight"),
            expert,
            input,
            cancel,
        )?;
        let up = self.project(
            executor,
            &format!("blk.48.ffn_up_{suffix}.weight"),
            expert,
            input,
            cancel,
        )?;
        let mut activation = private(vec![0.; 640]);
        for ((out, g), u) in activation.iter_mut().zip(gate.iter()).zip(up.iter()) {
            *out = ops::silu(*g) * u;
        }
        self.project(
            executor,
            &format!("blk.48.ffn_down_{suffix}.weight"),
            expert,
            &activation,
            cancel,
        )
    }

    fn moe(
        &self,
        executor: &mut impl MatrixExecutor,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Private> {
        let logits = self.project(executor, "blk.48.ffn_gate_inp.weight", 0, input, cancel)?;
        let selected = op(ops::router_topk_guarded(&logits, 10, true))?;
        let mut output = private(vec![0.; D]);
        for &(expert, weight) in selected.as_slice() {
            let down = self.expert(executor, input, expert as usize, false, cancel)?;
            for (out, v) in output.iter_mut().zip(down.iter()) {
                *out += weight * v;
            }
        }
        let shared = self.expert(executor, input, 0, true, cancel)?;
        let scalar: f32 = self
            .vector("blk.48.ffn_gate_inp_shexp.weight")?
            .iter()
            .zip(input)
            .map(|(w, x)| w * x)
            .sum();
        if !scalar.is_finite() {
            return Err(Error::InvalidFormat("non-finite MTP shared gate"));
        }
        for (out, v) in output.iter_mut().zip(shared.iter()) {
            *out += ops::sigmoid(scalar) * v;
        }
        if output.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidFormat("non-finite MTP MoE"));
        }
        Ok(output)
    }

    /// Condition current token on the preceding wide target/draft residual.
    /// The caller must pair positions correctly during prefill/catch-up.
    /// Preflight rejection leaves the sequence unchanged. Once execution starts,
    /// failure clears and poisons it; explicit reset is then needed.
    /// Existing scalar operator helpers have not received a whole-stack private
    /// error-path erasure audit. This graph is for compiled public fixtures only.
    pub fn step(
        &self,
        executor: &mut impl MatrixExecutor,
        sequence: &mut MtpSequence,
        token: u32,
        preceding_hidden: &[f32],
        cancel: &AtomicBool,
    ) -> Result<MtpOutput> {
        self.validate_input(sequence, token, preceding_hidden)?;
        sequence.poisoned = true;
        let result = (|| {
            self.file.verify_file_unchanged()?;
            let gr = op(GatedResidualShape::new(HC, D))?;
            let mut residual = self.condition(executor, token, preceding_hidden, cancel)?;
            let (input, inject) = self.mix(executor, &residual, "blk.48.hc_attn", true, cancel)?;
            let attention = self.attention(executor, &input, sequence, cancel)?;
            residual = private(op(gr.combine(&residual, &attention, &inject))?);
            let (input, inject) = self.mix(executor, &residual, "blk.48.hc_ffn", true, cancel)?;
            let ffn = self.moe(executor, &input, cancel)?;
            residual = private(op(gr.combine(&residual, &ffn, &inject))?);
            let (mixed, _) =
                self.mix(executor, &residual, "blk.48.nextn.hc_head", false, cancel)?;
            let logits = self.project(executor, "output.weight", 0, &mixed, cancel)?;
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::InvalidFormat("MTP cancelled"));
            }
            Ok(MtpOutput {
                logits,
                hidden: residual,
            })
        })();
        sequence.complete(result)
    }

    /// Append one teacher-forced token's dense KV, using its preceding *real*
    /// target residual. This is cache preparation/catch-up, not a draft step:
    /// no query, attention readout, FFN, head or output logits/residual are
    /// produced. Only the sequence's K/V allocations retain derived values.
    ///
    /// Conditioning, HC-attention mixing and K/V math are shared with `step`.
    /// The unused HC injection is skipped. Success advances exactly one
    /// position. Preflight rejection leaves the sequence unchanged; any error
    /// or cancellation after execution begins clears the complete cache and
    /// poisons it until reset. There is no reusable half-committed prefix.
    /// This remains a compiled-public-fixture reference, without private-runtime
    /// erasure acceptance or speculative-controller/performance acceptance.
    pub fn append_kv(
        &self,
        executor: &mut impl MatrixExecutor,
        sequence: &mut MtpSequence,
        token: u32,
        preceding_hidden: &[f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.validate_input(sequence, token, preceding_hidden)?;
        sequence.poisoned = true;
        let result = (|| {
            self.file.verify_file_unchanged()?;
            let residual = self.condition(executor, token, preceding_hidden, cancel)?;
            let (input, _) = self.mix(executor, &residual, "blk.48.hc_attn", false, cancel)?;
            let (mut key, value) = self.key_value(executor, &input, cancel)?;
            append_key_value(sequence, &mut key, &value)?;
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::InvalidFormat("MTP cancelled"));
            }
            Ok(())
        })();
        sequence.complete(result)
    }

    fn validate_input(
        &self,
        sequence: &MtpSequence,
        token: u32,
        preceding_hidden: &[f32],
    ) -> Result<()> {
        if sequence.binding != self.binding
            || sequence.poisoned
            || sequence.position >= sequence.limit
            || token as usize >= VOCAB
            || preceding_hidden.len() != WIDE
            || preceding_hidden.iter().any(|v| !v.is_finite())
        {
            return Err(Error::InvalidFormat("invalid MTP sequence input"));
        }
        Ok(())
    }
}

/// Request-derived outputs: no Debug, serialization or persistence API.
/// Both allocations are wiped on drop, including their retained capacity.
pub struct MtpOutput {
    pub logits: Private,
    pub hidden: Private,
}

/// One private dense-attention cache, owned by a single graph binding.
pub struct MtpSequence {
    keys: Vec<f32>,
    values: Vec<f32>,
    position: usize,
    limit: usize,
    poisoned: bool,
    binding: u64,
}
impl MtpSequence {
    fn new(limit: usize, binding: u64) -> Result<Self> {
        if !(1..=MAX_CONTEXT).contains(&limit) {
            return Err(Error::LimitExceeded("MTP reference context"));
        }
        let mut keys = Vec::new();
        let mut values = Vec::new();
        keys.try_reserve_exact(limit * 512)
            .map_err(|_| Error::LimitExceeded("MTP key cache"))?;
        values
            .try_reserve_exact(limit * 512)
            .map_err(|_| Error::LimitExceeded("MTP value cache"))?;
        Ok(Self {
            keys,
            values,
            position: 0,
            limit,
            poisoned: false,
            binding,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    fn complete<T>(&mut self, result: Result<T>) -> Result<T> {
        match result {
            Ok(output) => {
                self.position += 1;
                self.poisoned = false;
                Ok(output)
            }
            Err(error) => {
                self.clear();
                self.poisoned = true;
                Err(error)
            }
        }
    }
    /// Repair a provisional draft suffix before replay with real target hidden
    /// states. This discards only dense KV; it does not roll back target GDN/PLE.
    pub fn rewind(&mut self, position: usize) -> Result<()> {
        if self.poisoned || position > self.position {
            return Err(Error::OutOfBounds);
        }
        let len = position * 512;
        wipe_suffix(&mut self.keys, len);
        wipe_suffix(&mut self.values, len);
        self.keys.truncate(len);
        self.values.truncate(len);
        self.position = position;
        Ok(())
    }
    pub fn reset(&mut self) {
        self.clear();
        self.poisoned = false;
    }
    fn clear(&mut self) {
        for b in [&mut self.keys, &mut self.values] {
            wipe_cache(b);
            b.clear();
        }
        self.position = 0;
    }
}
fn append_key_value(sequence: &mut MtpSequence, key: &mut [f32], value: &[f32]) -> Result<()> {
    op(ops::rope_neox_heads(
        key,
        256,
        64,
        sequence.position as u32,
        10_000_000.0,
    ))?;
    sequence.keys.extend_from_slice(key);
    sequence.values.extend_from_slice(value);
    Ok(())
}
fn wipe_cache(buffer: &mut Vec<f32>) {
    buffer.resize(buffer.capacity(), 0.);
    buffer.as_mut_slice().zeroize();
}
fn wipe_suffix(buffer: &mut [f32], start: usize) {
    buffer[start..].zeroize();
}
impl Drop for MtpSequence {
    fn drop(&mut self) {
        self.clear();
    }
}

fn project_conditioned(
    embedding: &[f32],
    hidden: &[f32],
    mut project: impl FnMut(&[f32]) -> Result<Private>,
) -> Result<Private> {
    if embedding.len() != D || hidden.len() != WIDE {
        return Err(Error::InvalidBufferLength);
    }
    let mut residual = private(vec![0.; WIDE]);
    let mut concat = private(vec![0.; 2 * D]);
    concat[..D].copy_from_slice(embedding);
    for (stream, target) in hidden.chunks_exact(D).zip(residual.chunks_exact_mut(D)) {
        concat[D..].copy_from_slice(stream);
        let projected = project(&concat)?;
        if projected.len() != D {
            return Err(Error::InvalidBufferLength);
        }
        target.copy_from_slice(&projected);
    }
    Ok(residual)
}

fn split_query_gate(full: &[f32], width: usize) -> Result<(Private, Private)> {
    if width == 0
        || full.is_empty()
        || !full.len().is_multiple_of(2 * width)
        || full.iter().any(|v| !v.is_finite())
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut query = private(vec![0.; full.len() / 2]);
    let mut gate = private(vec![0.; full.len() / 2]);
    for ((h, q), g) in full
        .chunks_exact(2 * width)
        .zip(query.chunks_exact_mut(width))
        .zip(gate.chunks_exact_mut(width))
    {
        q.copy_from_slice(&h[..width]);
        g.copy_from_slice(&h[width..]);
    }
    Ok((query, gate))
}

// Ordinary causal dense GQA, with guarded temporary allocations and no QSA
// selection. f64 reductions/f32 softmax round points match the scalar oracle.
#[allow(clippy::too_many_arguments)]
fn dense_attention(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    heads: usize,
    kv: usize,
    width: usize,
    cancel: &AtomicBool,
) -> Result<Private> {
    if heads == 0
        || kv == 0
        || width == 0
        || !heads.is_multiple_of(kv)
        || query.len() != heads * width
        || keys.is_empty()
        || keys.len() != values.len()
        || !keys.len().is_multiple_of(kv * width)
        || query
            .iter()
            .chain(keys)
            .chain(values)
            .any(|v| !v.is_finite())
    {
        return Err(Error::InvalidBufferLength);
    }
    let count = keys.len() / (kv * width);
    let mut output = private(vec![0.; heads * width]);
    let mut scores = private(vec![0.; count]);
    let mut probabilities = private(vec![0.; count]);
    for head in 0..heads {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("MTP cancelled"));
        }
        let kh = head / (heads / kv);
        for (t, score) in scores.iter_mut().enumerate() {
            let sum = (0..width)
                .map(|d| {
                    f64::from(query[head * width + d]) * f64::from(keys[(t * kv + kh) * width + d])
                })
                .sum::<f64>();
            *score = (sum / (width as f64).sqrt()) as f32;
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidFormat("non-finite MTP attention"));
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0_f64;
        for (p, s) in probabilities.iter_mut().zip(scores.iter()) {
            *p = (f64::from(*s) - f64::from(maximum)).exp() as f32;
            denominator += f64::from(*p);
        }
        for p in probabilities.iter_mut() {
            *p = (f64::from(*p) / denominator) as f32;
        }
        for d in 0..width {
            output[head * width + d] = (0..count)
                .map(|t| f64::from(probabilities[t]) * f64::from(values[(t * kv + kh) * width + d]))
                .sum::<f64>() as f32;
        }
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("non-finite MTP attention output"));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "mtp_kv_tests.rs"]
mod kv_tests;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditioning_preserves_all_four_streams_and_embedding_first() {
        let embedding = vec![2.; D];
        let hidden: Vec<_> = (0..HC).flat_map(|i| vec![(i + 3) as f32; D]).collect();
        let mut calls = 0;
        let out = project_conditioned(&embedding, &hidden, |input| {
            assert!(input[..D].iter().all(|v| *v == 2.));
            assert!(input[D..].iter().all(|v| *v == (calls + 3) as f32));
            calls += 1;
            Ok(private(
                input[..D]
                    .iter()
                    .zip(&input[D..])
                    .map(|(e, h)| e + h)
                    .collect(),
            ))
        })
        .unwrap();
        assert_eq!(calls, 4);
        for (i, stream) in out.chunks_exact(D).enumerate() {
            assert!(stream.iter().all(|v| *v == (i + 5) as f32));
        }
        assert!(
            project_conditioned(&embedding, &hidden, |_| Ok(private(vec![0.; D - 1]))).is_err()
        );
    }
    #[test]
    fn query_and_gate_are_interleaved_per_head() {
        let (q, g) = split_query_gate(&[1., 2., 11., 12., 3., 4., 13., 14.], 2).unwrap();
        assert_eq!(&*q, &[1., 2., 3., 4.]);
        assert_eq!(&*g, &[11., 12., 13., 14.]);
        assert!(split_query_gate(&[1.; 7], 2).is_err());
    }
    #[test]
    fn dense_gqa_matches_independent_public_scalar_oracle() {
        let query = [1., -2., 3., 1.];
        let keys = [1., 0., 0., 1., -1., 2.];
        let values = [2., 3., 5., 7., 11., 13.];
        let out =
            dense_attention(&query, &keys, &values, 2, 1, 2, &AtomicBool::new(false)).unwrap();
        // Independently formed dot products for each query; stable scalar softmax.
        for (h, dots) in [[1_f64, -2., -5.], [3., 1., -1.]].iter().enumerate() {
            let exps = dots.map(|d| (d / 2_f64.sqrt()).exp());
            let sum: f64 = exps.iter().sum();
            for d in 0..2 {
                let expected = (0..3)
                    .map(|t| exps[t] / sum * f64::from(values[t * 2 + d]))
                    .sum::<f64>() as f32;
                assert!((out[h * 2 + d] - expected).abs() < 2e-6);
            }
        }
    }
    #[test]
    fn dense_attention_keeps_every_visible_token_past_qsa_budget() {
        let q = [0.];
        let keys = vec![0.; 2050];
        let mut values = vec![0.; 2050];
        values[0] = 2050.;
        let out = dense_attention(&q, &keys, &values, 1, 1, 1, &AtomicBool::new(false)).unwrap();
        assert!((out[0] - 1.).abs() < 1e-6);
        assert!(dense_attention(&q, &keys, &values, 1, 1, 1, &AtomicBool::new(true)).is_err());
    }
    #[test]
    fn dense_gqa_groups_queries_contiguously_by_kv_head() {
        let out = dense_attention(
            &[0.; 4],
            &[0.; 4],
            &[3., 11., 7., 19.],
            4,
            2,
            1,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(&*out, &[5., 5., 15., 15.]);
    }
    #[test]
    fn draft_rewind_erases_provisional_suffix_and_reset_erases_capacity() {
        let mut s = MtpSequence::new(3, 1).unwrap();
        s.keys.resize(3 * 512, 7.);
        s.values.resize(3 * 512, 9.);
        s.position = 3;
        s.rewind(1).unwrap();
        assert_eq!(s.position(), 1);
        assert_eq!(s.keys.len(), 512);
        // Inspect the same suffix-wipe helper while the allocation is still
        // fully initialized; a resize after truncate would overwrite evidence.
        let mut keys = vec![7.; 3 * 512];
        let mut values = vec![9.; 3 * 512];
        wipe_suffix(&mut keys, 512);
        wipe_suffix(&mut values, 512);
        assert!(keys[..512].iter().all(|v| *v == 7.));
        assert!(values[..512].iter().all(|v| *v == 9.));
        assert!(keys[512..].iter().all(|v| v.to_bits() == 0));
        assert!(values[512..].iter().all(|v| v.to_bits() == 0));
        // Observe the same wiping helper before truncation: includes the
        // previously used suffix, not just the active visible cache prefix.
        wipe_cache(&mut s.keys);
        wipe_cache(&mut s.values);
        assert!(s.keys.iter().all(|v| v.to_bits() == 0));
        assert!(s.values.iter().all(|v| v.to_bits() == 0));
        s.reset();
        assert_eq!((s.position(), s.keys.len(), s.values.len()), (0, 0, 0));
        assert!(s.rewind(1).is_err());
    }
}
