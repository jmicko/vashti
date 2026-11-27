//! GLM 5.3 Flash (GGUF architecture `glm5-next`) for Kuno. `reference` is the exact CPU
//! forward pass (f32 arithmetic over the decoded GGUF weights), the yardstick for the GPU
//! engine; it follows llama.cpp's `glm5next` graph (PR 27754) operation for operation.
//!
//! The model: 45 blocks with four hyper-connection streams (mHC, Sinkhorn-mixed), each
//! block either a KDA linear-attention layer (gated delta rule with per-channel decay) or a
//! DSA layer (multi-head latent attention, no rotary, a lightning indexer choosing at most
//! 2048 earlier positions once the context is longer than that); the first three blocks have
//! a dense SwiGLU, the rest 288 routed experts (8 per token, sigmoid router with a selection
//! bias) and one shared expert.
use std::path::Path;

use kuno_text::{GgufFile, GgufLimits, MetadataValue, TensorType};

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod engine;
#[cfg(feature = "gpu")]
pub mod experts;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod reference;
#[cfg(feature = "gpu")]
pub mod second;

pub type Result<T> = std::result::Result<T, &'static str>;

/// The model's dimensions and constants from the GGUF metadata.
#[derive(Clone, Debug)]
pub struct Hparams {
    /// Trunk blocks (the NextN draft block excluded).
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub rms_eps: f32,
    /// Per block: true for DSA attention, false for KDA.
    pub dsa: Vec<bool>,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub qk_head: usize,
    pub v_head: usize,
    pub conv: usize,
    pub kda_head: usize,
    pub gate_lower_bound: f32,
    pub hc: usize,
    pub sinkhorn: usize,
    pub hc_eps: f32,
    pub ff: usize,
    pub experts: usize,
    pub experts_used: usize,
    pub ff_expert: usize,
    pub ff_shared: usize,
    pub dense_lead: usize,
    pub expert_scale: f32,
    pub expert_norm: bool,
    pub clamp_expert: Vec<f32>,
    pub clamp_shared: Vec<f32>,
    pub vocab: usize,
    /// The indexer's selection: `top_k` cells in pools of `kpool` (scoring only once the
    /// context is longer than `top_k + kpool - 1`).
    pub indexer_top_k: usize,
    pub indexer_kpool: usize,
}

pub struct Model {
    pub gguf: GgufFile,
    pub hp: Hparams,
}

fn meta<'a>(g: &'a GgufFile, key: &str) -> Result<&'a MetadataValue> {
    g.metadata().get(key).ok_or("model_metadata_missing")
}
fn uint(g: &GgufFile, key: &str) -> Result<usize> {
    meta(g, key)?
        .as_u64()
        .map(|v| v as usize)
        .ok_or("model_metadata_invalid")
}
fn float(g: &GgufFile, key: &str) -> Result<f32> {
    meta(g, key)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or("model_metadata_invalid")
}
fn floats(g: &GgufFile, key: &str) -> Result<Vec<f32>> {
    meta(g, key)?
        .as_array()
        .ok_or("model_metadata_invalid")?
        .iter()
        .map(|v| v.as_f64().map(|v| v as f32).ok_or("model_metadata_invalid"))
        .collect()
}

impl Model {
    /// Opens the shards of a `glm5-next` GGUF.
    pub fn open<P: AsRef<Path>>(shards: &[P]) -> Result<Self> {
        let limits = GgufLimits {
            max_tensor_read_bytes: 1 << 36,
            ..GgufLimits::default()
        };
        let gguf = GgufFile::open_shards(shards, limits).map_err(|e| {
            eprintln!("gguf: {e:?}");
            "model_file_invalid"
        })?;
        if meta(&gguf, "general.architecture")?.as_str() != Some("glm5-next") {
            return Err("model_file_invalid");
        }
        let a = |k: &str| format!("glm5-next.{k}");
        let g = &gguf;
        let blocks = uint(g, &a("block_count"))?;
        let layers = blocks - uint(g, &a("nextn_predict_layers")).unwrap_or(0);
        let dsa = meta(g, &a("attention.head_count_kv"))?
            .as_array()
            .ok_or("model_metadata_invalid")?
            .iter()
            .take(layers)
            .map(|v| v.as_i64().map(|v| v != 0).ok_or("model_metadata_invalid"))
            .collect::<Result<Vec<_>>>()?;
        let hp = Hparams {
            layers,
            embd: uint(g, &a("embedding_length"))?,
            heads: uint(g, &a("attention.head_count"))?,
            rms_eps: float(g, &a("attention.layer_norm_rms_epsilon"))?,
            dsa,
            q_lora: uint(g, &a("attention.q_lora_rank"))?,
            kv_lora: uint(g, &a("attention.kv_lora_rank"))?,
            qk_head: uint(g, &a("attention.key_length_mla"))?,
            v_head: uint(g, &a("attention.value_length_mla"))?,
            conv: uint(g, &a("ssm.conv_kernel"))?,
            kda_head: uint(g, &a("kda.head_dim"))?,
            gate_lower_bound: float(g, &a("kda.gate_lower_bound"))?,
            hc: uint(g, &a("hyper_connection.count"))?,
            sinkhorn: uint(g, &a("hyper_connection.sinkhorn_iterations"))?,
            hc_eps: float(g, &a("hyper_connection.epsilon"))?,
            ff: uint(g, &a("feed_forward_length"))?,
            experts: uint(g, &a("expert_count"))?,
            experts_used: uint(g, &a("expert_used_count"))?,
            ff_expert: uint(g, &a("expert_feed_forward_length"))?,
            ff_shared: uint(g, &a("expert_shared_feed_forward_length"))?,
            dense_lead: uint(g, &a("leading_dense_block_count"))?,
            expert_scale: float(g, &a("expert_weights_scale"))?,
            expert_norm: matches!(
                meta(g, &a("expert_weights_norm"))?,
                MetadataValue::Bool(true)
            ),
            clamp_expert: floats(g, &a("swiglu_clamp_exp"))?,
            clamp_shared: floats(g, &a("swiglu_clamp_shexp"))?,
            vocab: uint(g, &a("vocab_size"))?,
            indexer_top_k: uint(g, &a("attention.indexer.top_k"))?,
            indexer_kpool: uint(g, &a("attention.indexer.kpool"))?,
        };
        if uint(g, &a("expert_gating_func"))? != 2 || hp.hc != 4 {
            return Err("model_file_invalid");
        }
        Ok(Self { gguf, hp })
    }

    /// A weight's packed bytes and layout (`rows` rows of `cols` values).
    pub fn matrix(&self, name: &str) -> Result<Matrix> {
        let view = self.gguf.tensor(name).ok_or("model_tensor_missing")?;
        let info = view.info();
        let mut bytes = vec![0u8; info.byte_len() as usize];
        view.read_range(0, &mut bytes)
            .map_err(|_| "model_read_failed")?;
        Ok(Matrix {
            dtype: info.dtype(),
            rows: info.row_count() as usize,
            cols: info.row_len() as usize,
            row_bytes: info.row_bytes() as usize,
            bytes,
        })
    }
    /// Rows `first..first + count` of a weight (one expert of a stacked expert tensor).
    pub fn rows(&self, name: &str, first: usize, count: usize) -> Result<Matrix> {
        let view = self.gguf.tensor(name).ok_or("model_tensor_missing")?;
        let info = view.info();
        let row_bytes = info.row_bytes() as usize;
        let mut bytes = vec![0u8; row_bytes * count];
        view.read_range((first * row_bytes) as u64, &mut bytes)
            .map_err(|_| "model_read_failed")?;
        Ok(Matrix {
            dtype: info.dtype(),
            rows: count,
            cols: info.row_len() as usize,
            row_bytes,
            bytes,
        })
    }
    /// Rows `first..first + count` of a weight's packed bytes into `out` (its exact size).
    pub fn read_rows(&self, name: &str, first: usize, out: &mut [u8]) -> Result<()> {
        let view = self.gguf.tensor(name).ok_or("model_tensor_missing")?;
        let row_bytes = view.info().row_bytes() as usize;
        view.read_range((first * row_bytes) as u64, out)
            .map_err(|_| "model_read_failed")
    }
    /// A weight's type and packed bytes per row.
    pub fn layout(&self, name: &str) -> Result<(TensorType, usize, usize)> {
        let view = self.gguf.tensor(name).ok_or("model_tensor_missing")?;
        let info = view.info();
        Ok((
            info.dtype(),
            info.row_bytes() as usize,
            info.row_len() as usize,
        ))
    }
    /// A small weight decoded to f32 (norms, biases, scales).
    pub fn vector(&self, name: &str) -> Result<Vec<f32>> {
        self.matrix(name)?.decode_all()
    }
}

/// A packed GGUF weight: `rows` rows of `cols` values, one output per row.
pub struct Matrix {
    pub dtype: TensorType,
    pub rows: usize,
    pub cols: usize,
    pub row_bytes: usize,
    pub bytes: Vec<u8>,
}
impl Matrix {
    pub fn decode_row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        self.dtype
            .decode(&self.bytes[row * self.row_bytes..][..self.row_bytes], out)
            .map_err(|_| "model_decode_failed")
    }
    pub fn decode_all(&self) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.rows * self.cols];
        for (r, row) in out.chunks_exact_mut(self.cols.max(1)).enumerate() {
            self.decode_row(r, row)?;
        }
        Ok(out)
    }
    /// `out[r] = row r . x` for every row, the rows decoded in parallel.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        assert_eq!(x.len(), self.cols);
        (0..self.rows)
            .into_par_iter()
            .map_init(
                || vec![0f32; self.cols],
                |row, r| {
                    self.decode_row(r, row).expect("decode");
                    let mut sum = 0f64;
                    for (a, b) in row.iter().zip(x) {
                        sum += f64::from(*a) * f64::from(*b);
                    }
                    sum as f32
                },
            )
            .collect()
    }
}
