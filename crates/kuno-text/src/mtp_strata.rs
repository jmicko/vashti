//! Explicit original-packed Strata Q2 MTP GGUF profile.
//!
//! Inventory/layout: MIT Strata v0.1.41, fb58e0dbc8399662c0e47c76578c6e878b14f6cf,
//! tools/mtp_pack.py and tools/mtp_rt.py. The latter's runtime files are NOT
//! accepted here: their dense Q8 conversion and already folded Gemma norms
//! require a distinct profile. These equations use original GGUF BF16 dense
//! weights, raw Gemma F32 norms, and the existing Rust Q2_0 packed decoder.
//! The pinned publisher checkpoint is de4b8e4d43b917e7706784d8bb445c9af86a3540.
//! No upstream engine or Python code executes during inference.

use super::*;
use crate::model::GraphGeometry;

// Manifest-of-original-tensor hashes written by the pinned packer. This is a
// format/provenance discriminator, NOT authentication or a whole GGUF digest.
const SOURCE_MANIFEST_SHA256: &str =
    "4ef372235caf4a756be32c0c62856c251d24df3d785a0c7a9e8c7181e27a6e8c";

struct Spec {
    name: &'static str,
    role: Option<&'static str>,
    dimensions: &'static [u64],
    dtype: TensorType,
}
use TensorType::{Bf16, F32, Q2_0};

const SPECS: [Spec; 31] = [
    Spec {
        name: "mtp.fc_embedding.weight",
        role: None,
        dimensions: &[2560, 2560],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.fc_hidden.weight",
        role: None,
        dimensions: &[2560, 2560],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.hyper_connection_mixer.hc_norm.weight",
        role: Some("blk.48.nextn.hc_head_norm.weight"),
        dimensions: &[10240],
        dtype: F32,
    },
    Spec {
        name: "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        role: Some("blk.48.nextn.hc_head_down.weight"),
        dimensions: &[10240, 320],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        role: Some("blk.48.nextn.hc_head_up.weight"),
        dimensions: &[320, 10240],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
        role: Some("blk.48.hc_attn_inject.weight"),
        dimensions: &[10240, 4],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
        role: Some("blk.48.hc_attn_norm.weight"),
        dimensions: &[10240],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
        role: Some("blk.48.hc_attn_down.weight"),
        dimensions: &[10240, 320],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
        role: Some("blk.48.hc_attn_up.weight"),
        dimensions: &[320, 10240],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp.experts.down_proj",
        role: Some("blk.48.ffn_down_exps.weight"),
        dimensions: &[640, 2560, 512],
        dtype: Q2_0,
    },
    Spec {
        name: "mtp.layers.0.mlp.experts.gate_up_proj",
        role: Some("blk.48.ffn_gate_up_exps.weight"),
        dimensions: &[2560, 1280, 512],
        dtype: Q2_0,
    },
    Spec {
        name: "mtp.layers.0.mlp.gate.weight",
        role: Some("blk.48.ffn_gate_inp.weight"),
        dimensions: &[2560, 512],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        role: Some("blk.48.ffn_down_shexp.weight"),
        dimensions: &[640, 2560],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
        role: Some("blk.48.ffn_gate_shexp.weight"),
        dimensions: &[2560, 640],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp.shared_expert.up_proj.weight",
        role: Some("blk.48.ffn_up_shexp.weight"),
        dimensions: &[2560, 640],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp.shared_expert_gate.weight",
        role: Some("blk.48.ffn_gate_inp_shexp.weight"),
        dimensions: &[2560, 1],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
        role: Some("blk.48.hc_ffn_inject.weight"),
        dimensions: &[10240, 4],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
        role: Some("blk.48.hc_ffn_norm.weight"),
        dimensions: &[10240],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
        role: Some("blk.48.hc_ffn_down.weight"),
        dimensions: &[10240, 320],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
        role: Some("blk.48.hc_ffn_up.weight"),
        dimensions: &[320, 10240],
        dtype: Bf16,
    },
    // Stored selection indexer, inspected/pinned but not executed by dense GQA.
    Spec {
        name: "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
        role: None,
        dimensions: &[2560, 640],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
        role: None,
        dimensions: &[128],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
        role: None,
        dimensions: &[128],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.self_attn.k_norm.weight",
        role: Some("blk.48.attn_k_norm.weight"),
        dimensions: &[256],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.self_attn.k_proj.weight",
        role: Some("blk.48.attn_k.weight"),
        dimensions: &[2560, 512],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.self_attn.o_proj.weight",
        role: Some("blk.48.attn_output.weight"),
        dimensions: &[6144, 2560],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.self_attn.q_norm.weight",
        role: Some("blk.48.attn_q_norm.weight"),
        dimensions: &[256],
        dtype: F32,
    },
    Spec {
        name: "mtp.layers.0.self_attn.q_proj.weight",
        role: Some("blk.48.attn_q.weight"),
        dimensions: &[2560, 12288],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.layers.0.self_attn.v_proj.weight",
        role: Some("blk.48.attn_v.weight"),
        dimensions: &[2560, 512],
        dtype: Bf16,
    },
    Spec {
        name: "mtp.pre_fc_norm_embedding.weight",
        role: Some("blk.48.nextn.enorm.weight"),
        dimensions: &[2560],
        dtype: F32,
    },
    Spec {
        name: "mtp.pre_fc_norm_hidden.weight",
        role: Some("blk.48.nextn.hnorm.weight"),
        dimensions: &[10240],
        dtype: F32,
    },
];

pub(super) fn source_name(role: &str) -> Option<&'static str> {
    SPECS.iter().find(|s| s.role == Some(role)).map(|s| s.name)
}

fn validate_metadata(file: &GgufFile) -> Result<()> {
    let expected = [
        ("general.architecture", "qwen4exp-mtp"),
        (
            "strata.mtp.source",
            "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors",
        ),
        ("strata.mtp.source_sha256", SOURCE_MANIFEST_SHA256),
        ("strata.mtp.expert_format", "q2_0"),
        ("strata.mtp.expert_quantizer", "per-block MSE scale search"),
    ];
    if file.version() != 3
        || file.shard_count() != 1
        || file.metadata().len() != expected.len()
        || expected.iter().any(|(key, value)| {
            file.metadata().get(*key).and_then(MetadataValue::as_str) != Some(*value)
        })
    {
        return Err(Error::InvalidFormat(
            "unsupported Strata MTP provenance or format",
        ));
    }
    Ok(())
}

fn validate_target_geometry(g: GraphGeometry) -> Result<()> {
    if g.layers != 48
        || g.residual.streams != HC
        || g.residual.width != D
        || g.low_rank != 320
        || g.vocabulary != VOCAB
        || g.experts != 512
        || g.experts_used != 10
        || g.expert_hidden != 640
        || g.epsilon != EPS
        || g.qsa.query_heads != 24
        || g.qsa.kv_heads != 2
        || g.qsa.head_dim != 256
        || g.qsa.rotary_dims != 64
        || g.qsa.rope_theta != 10_000_000.0
        || g.max_context < MAX_CONTEXT
    {
        return Err(Error::InvalidFormat(
            "incompatible Strata MTP target geometry",
        ));
    }
    Ok(())
}

impl<'a> MtpGraph<'a> {
    /// Bind the inspected Strata Q2 GGUF and borrow the actual DASLab target's
    /// admitted IQ4_XS embedding / Q6_K full-vocabulary output head. There is no
    /// file-name detection, disk model copy, runtime-file conversion, vocabulary
    /// pruning, or alternate activation quantization. Both pinned descriptors
    /// remain guarded for cached execution. The target must outlive this graph.
    ///
    /// Cell c consumes target residual R(c) and token T(c+1), at RoPE position c,
    /// and predicts T(c+2). The caller supplies that schedule; no dummy cell is
    /// inserted. As with the original profile, this is a public-fixture reference
    /// rather than a private-runtime or speculative-controller acceptance claim.
    pub fn bind_strata_q2(file: &'a GgufFile, target: BoundGraph<'a, 'a>) -> Result<Self> {
        validate_metadata(file)?;
        validate_target_geometry(target.geometry())?;
        file.verify_file_unchanged()?;
        target.verify_unchanged()?;
        for (name, dtype) in [
            ("token_embd.weight", TensorType::Iq4Xs),
            ("output.weight", TensorType::Q6K),
        ] {
            let tensor = target.tensor(name)?;
            if tensor.info().dimensions() != [D as u64, VOCAB as u64]
                || tensor.info().dtype() != dtype
            {
                return Err(Error::InvalidFormat(
                    "incompatible Strata MTP target weights",
                ));
            }
            Matrix::new(tensor)?;
        }
        if file.tensors().len() != SPECS.len()
            || file
                .tensors()
                .keys()
                .any(|n| !SPECS.iter().any(|s| s.name == n))
        {
            return Err(Error::InvalidFormat("unexpected Strata MTP tensor"));
        }
        let binding = NEXT_BINDING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        let mut graph = Self {
            file,
            vectors: BTreeMap::new(),
            names: BTreeSet::new(),
            binding,
            profile: MtpProfile::StrataQ2,
            target: Some(target),
        };
        for s in &SPECS {
            graph.admit(s.name, s.dimensions, s.dtype)?;
            if s.dimensions.len() == 1 {
                let values = graph
                    .vectors
                    .get_mut(s.name)
                    .ok_or(Error::InvalidFormat("missing Strata MTP norm"))?;
                for value in values.iter_mut() {
                    *value += 1.;
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(Error::InvalidFormat("non-finite Strata MTP norm"));
                }
            }
        }
        graph
            .names
            .extend(["token_embd.weight".into(), "output.weight".into()]);
        graph.verify_unchanged()?;
        Ok(graph)
    }
}

/// Separate publisher FCs have distinct reduction/rounding from an EH concat.
/// Project normalized embedding once, then every normalized hidden branch once
/// and broadcast-add the embedding projection to each complete hidden branch.
pub(super) fn project_conditioned(
    embedding: &[f32],
    hidden: &[f32],
    mut project: impl FnMut(bool, &[f32]) -> Result<Private>,
) -> Result<Private> {
    if embedding.len() != D
        || hidden.len() != WIDE
        || embedding.iter().chain(hidden).any(|v| !v.is_finite())
    {
        return Err(Error::InvalidBufferLength);
    }
    let embedding = project(false, embedding)?;
    if embedding.len() != D || embedding.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidBufferLength);
    }
    let mut residual = private(vec![0.; WIDE]);
    for (source, destination) in hidden.chunks_exact(D).zip(residual.chunks_exact_mut(D)) {
        let projected = project(true, source)?;
        if projected.len() != D || projected.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidBufferLength);
        }
        for ((out, h), e) in destination
            .iter_mut()
            .zip(projected.iter())
            .zip(embedding.iter())
        {
            *out = h + e;
        }
    }
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("non-finite Strata MTP conditioning"));
    }
    Ok(residual)
}

pub(super) fn activate_gate_up(values: &[f32]) -> Result<Private> {
    if values.len() != 1280 || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidBufferLength);
    }
    let output = private(
        values[..640]
            .iter()
            .zip(&values[640..])
            .map(|(gate, up)| ops::silu(*gate) * up)
            .collect(),
    );
    if output.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidFormat("non-finite Strata MTP activation"));
    }
    Ok(output)
}

#[cfg(test)]
#[path = "mtp_strata_tests.rs"]
mod tests;
