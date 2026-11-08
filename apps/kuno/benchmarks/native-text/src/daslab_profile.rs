//! Exact raw-ID inventory of the pinned public DASLab IQ3_S original shards.
//!
//! This is an explicit profile, never model-name or quantization auto-detection.
//! It checks canonical tensor names, raw formats and completeness only: existing
//! GGUF/binder owners must still check split/source identity, dimensions, ranges,
//! budgets and completion. Header digests describe the inspected public prefixes;
//! whole-file integrity remains a separate original-shard SHA256 acceptance gate.

pub const REPOSITORY: &str = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF";
pub const REVISION: &str = "ed59f92082b1e93c0e96d60a8b11aab089b52f09";
pub const HEADER_SHA256: [&str; 2] = [
    "d17e0d2ac2dc46fba367316963e024a2ad653be065b77a30a6cc48e4c18098a6",
    "49f992d94de7d687ab0042f54d72fbabb4fec16e6860181f1b5bd91e46c89a96",
];
pub const TENSOR_COUNT: usize = 1224;
pub const LAYER_COUNT: usize = 48;
pub const EXPERT_COUNT: usize = 512;
pub const BANK_SLOTS: usize = if cfg!(feature = "strata-resident-240") {
    240
} else if cfg!(feature = "strata-resident-224") {
    224
} else if cfg!(feature = "strata-rust-port") {
    192
} else {
    144
};
pub const SEGMENT_SLOTS: usize = BANK_SLOTS / 2;
pub const HOST_PART: usize = if cfg!(feature = "strata-resident-224") {
    128
} else {
    144
};
pub const SOURCE_PARTS: [usize; 5] = [
    SEGMENT_SLOTS,
    SEGMENT_SLOTS,
    HOST_PART,
    HOST_PART,
    512 - BANK_SLOTS - 2 * HOST_PART,
];
pub const SOURCE_FIRST: [usize; 5] = [
    0,
    SEGMENT_SLOTS,
    BANK_SLOTS,
    BANK_SLOTS + HOST_PART,
    BANK_SLOTS + 2 * HOST_PART,
];
const ABSENT: u32 = u32::MAX;

const GLOBALS: &[(&str, u32)] = &[
    ("output.weight", 14),
    ("output_hc_down.weight", 30),
    ("output_hc_norm.weight", 0),
    ("output_hc_up.weight", 30),
    ("token_embd.weight", 23),
    ("per_layer_token_embd.weight", 20),
];

const DTYPES_0: [u32; LAYER_COUNT] = [
    12, 14, 23, ABSENT, 14, 12, 23, ABSENT, 23, 12, 14, ABSENT, 23, 12, 13, ABSENT, 23, 14, 23,
    ABSENT, 14, 14, 12, ABSENT, 14, 14, 14, ABSENT, 14, 14, 12, ABSENT, 13, 14, 23, ABSENT, 12, 12,
    13, ABSENT, 14, 12, 23, ABSENT, 12, 14, 13, ABSENT,
];
const DTYPES_1: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 23, ABSENT, ABSENT, ABSENT, 14, ABSENT,
    ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT,
    ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
];
const DTYPES_2: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT,
    ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT,
    ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0,
    ABSENT, ABSENT, ABSENT, 0, ABSENT, ABSENT, ABSENT, 0,
];
const DTYPES_3: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 12, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT,
    ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT,
    ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
];
const DTYPES_4: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 23, ABSENT, ABSENT, ABSENT, 23, ABSENT, ABSENT, ABSENT, 12, ABSENT,
    ABSENT, ABSENT, 23, ABSENT, ABSENT, ABSENT, 13, ABSENT, ABSENT, ABSENT, 13, ABSENT, ABSENT,
    ABSENT, 13, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
];
const DTYPES_5: [u32; LAYER_COUNT] = [
    14, 13, 14, ABSENT, 14, 14, 14, ABSENT, 14, 14, 13, ABSENT, 13, 12, 14, ABSENT, 14, 14, 14,
    ABSENT, 14, 14, 14, ABSENT, 14, 14, 13, ABSENT, 14, 13, 14, ABSENT, 14, 23, 13, ABSENT, 14, 14,
    13, ABSENT, 14, 12, 13, ABSENT, 13, 23, 13, ABSENT,
];
const DTYPES_6: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 13, ABSENT,
    ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 13, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT,
    ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
    ABSENT, ABSENT, ABSENT, 14, ABSENT, ABSENT, ABSENT, 14,
];
const DTYPES_7: [u32; LAYER_COUNT] = [
    20, 42, 20, 20, 20, 20, 20, 20, 42, 20, 42, 20, 42, 42, 20, 20, 20, 20, 20, 20, 42, 42, 20, 20,
    20, 20, 20, 20, 20, 20, 42, 20, 20, 20, 20, 20, 42, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
];
const DTYPES_8: [u32; LAYER_COUNT] = [
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
    20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 8,
];
const DTYPES_9: [u32; LAYER_COUNT] = [
    18, 22, 22, 22, 22, 22, 22, 18, 22, 22, 22, 22, 18, 22, 18, 22, 18, 21, 18, 22, 18, 21, 18, 22,
    22, 21, 22, 21, 21, 21, 18, 18, 22, 18, 22, 21, 22, 18, 18, 22, 21, 18, 18, 18, 21, 21, 18, 23,
];
const DTYPES_10: [u32; LAYER_COUNT] = [30; LAYER_COUNT];
const DTYPES_11: [u32; LAYER_COUNT] = [
    12, 23, 12, 23, 14, 23, 14, 23, 23, 12, 12, 12, 14, 12, 23, 14, 12, 23, 13, 14, 23, 12, 14, 13,
    23, 23, 14, 23, 14, 13, 23, 23, 14, 23, 12, 23, 14, 14, 12, 12, 13, 12, 14, 14, 12, 12, 12, 12,
];
const DTYPES_12: [u32; LAYER_COUNT] = [
    23, 12, 14, 12, 13, 14, 12, 23, 12, 23, 12, 14, 14, 23, 23, 23, 12, 12, 13, 14, 12, 23, 23, 13,
    13, 14, 14, 23, 14, 23, 12, 13, 12, 12, 13, 14, 14, 13, 23, 12, 23, 12, 23, 14, 12, 14, 14, 13,
];
const DTYPES_13: [u32; LAYER_COUNT] = [0; LAYER_COUNT];
const DTYPES_14: [u32; LAYER_COUNT] = [
    ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT,
    ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT,
    ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30,
    ABSENT, ABSENT, ABSENT, 30, ABSENT, ABSENT, ABSENT, 30,
];
const DTYPES_15: [u32; LAYER_COUNT] = [
    ABSENT, 1, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
];
const DTYPES_16: [u32; LAYER_COUNT] = [
    ABSENT, 30, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
];
const DTYPES_17: [u32; LAYER_COUNT] = [
    ABSENT, 0, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
    ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT, ABSENT,
];
const DTYPES_18: [u32; LAYER_COUNT] = [
    0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0,
    ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0, 0, 0, ABSENT, 0,
    0, 0, ABSENT,
];
const DTYPES_19: [u32; LAYER_COUNT] = [
    30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30,
    ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT, 30, 30,
    30, ABSENT, 30, 30, 30, ABSENT, 30, 30, 30, ABSENT,
];
const DTYPES_20: [u32; LAYER_COUNT] = [
    14, 14, 14, ABSENT, 14, 12, 14, ABSENT, 12, 14, 14, ABSENT, 14, 14, 12, ABSENT, 14, 14, 13,
    ABSENT, 14, 13, 14, ABSENT, 14, 14, 13, ABSENT, 14, 14, 14, ABSENT, 14, 14, 14, ABSENT, 14, 14,
    14, ABSENT, 14, 14, 14, ABSENT, 14, 14, 13, ABSENT,
];

const FAMILIES: &[(&str, &[u32; LAYER_COUNT])] = &[
    ("attn_gate.weight", &DTYPES_0),
    ("attn_k.weight", &DTYPES_1),
    ("attn_k_norm.weight", &DTYPES_2),
    ("attn_output.weight", &DTYPES_3),
    ("attn_q.weight", &DTYPES_4),
    ("attn_q_norm.weight", &DTYPES_2),
    ("attn_qkv.weight", &DTYPES_5),
    ("attn_v.weight", &DTYPES_6),
    ("ffn_down_exps.weight", &DTYPES_7),
    ("ffn_down_shexp.weight", &DTYPES_8),
    ("ffn_gate_exps.weight", &DTYPES_9),
    ("ffn_gate_inp.weight", &DTYPES_10),
    ("ffn_gate_inp_shexp.weight", &DTYPES_10),
    ("ffn_gate_shexp.weight", &DTYPES_11),
    ("ffn_up_exps.weight", &DTYPES_9),
    ("ffn_up_shexp.weight", &DTYPES_12),
    ("hc_attn_down.weight", &DTYPES_10),
    ("hc_attn_inject.weight", &DTYPES_10),
    ("hc_attn_norm.weight", &DTYPES_13),
    ("hc_attn_up.weight", &DTYPES_10),
    ("hc_ffn_down.weight", &DTYPES_10),
    ("hc_ffn_inject.weight", &DTYPES_10),
    ("hc_ffn_norm.weight", &DTYPES_13),
    ("hc_ffn_up.weight", &DTYPES_10),
    ("indexer.k_norm.weight", &DTYPES_2),
    ("indexer.k_proj.weight", &DTYPES_14),
    ("indexer.q_norm.weight", &DTYPES_2),
    ("indexer.q_proj.weight", &DTYPES_14),
    ("ple_conv1d.weight", &DTYPES_15),
    ("ple_key.weight", &DTYPES_16),
    ("ple_norm_conv.weight", &DTYPES_17),
    ("ple_norm_key.weight", &DTYPES_17),
    ("ple_norm_query.weight", &DTYPES_17),
    ("ple_value.weight", &DTYPES_16),
    ("ssm_a", &DTYPES_18),
    ("ssm_alpha.weight", &DTYPES_19),
    ("ssm_beta.weight", &DTYPES_19),
    ("ssm_conv1d.weight", &DTYPES_18),
    ("ssm_dt.bias", &DTYPES_18),
    ("ssm_norm.weight", &DTYPES_18),
    ("ssm_out.weight", &DTYPES_20),
];

fn lookup(name: &str) -> Option<(usize, u32)> {
    if let Some((index, (_, dtype))) = GLOBALS.iter().enumerate().find(|(_, (n, _))| *n == name) {
        return Some((index, *dtype));
    }
    let (layer, family) = name.strip_prefix("blk.")?.split_once('.')?;
    // Parsing alone would accept aliases such as 00 or +1. Only the inspected
    // canonical decimal spelling is an admitted inventory name.
    if layer.is_empty()
        || (layer.len() > 1 && layer.starts_with('0'))
        || !layer.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let layer: usize = layer.parse().ok()?;
    if layer >= LAYER_COUNT {
        return None;
    }
    let (family_index, (_, dtypes)) = FAMILIES
        .iter()
        .enumerate()
        .find(|(_, (n, _))| *n == family)?;
    let dtype = dtypes[layer];
    (dtype != ABSENT).then_some((GLOBALS.len() + family_index * LAYER_COUNT + layer, dtype))
}

/// Exact original on-disk GGML type ID for a canonical pinned tensor name.
pub fn expected_dtype(name: &str) -> Option<u32> {
    lookup(name).map(|(_, dtype)| dtype)
}

/// Original [gate, up, down] types for one complete routed expert layer.
pub fn routed_types(layer: usize) -> Option<[u32; 3]> {
    if layer >= LAYER_COUNT {
        return None;
    }
    let get = |name| {
        FAMILIES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, types)| types[layer])
    };
    Some([
        get("ffn_gate_exps.weight")?,
        get("ffn_up_exps.weight")?,
        get("ffn_down_exps.weight")?,
    ])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InventoryError {
    CountMismatch,
    UnknownName,
    WrongType,
    DuplicateName,
}

/// Bounded, order-independent validation of the entire 1224-name inventory.
/// No names or content are retained. Exactly one entry is required for every
/// expected name; count plus uniqueness rejects omission/extra substitutions.
pub fn validate_inventory<'a>(
    tensors: impl IntoIterator<Item = (&'a str, u32)>,
) -> Result<(), InventoryError> {
    const WORDS: usize = (GLOBALS.len() + FAMILIES.len() * LAYER_COUNT).div_ceil(64);
    let mut seen = [0_u64; WORDS];
    let mut count = 0;
    for (name, dtype) in tensors {
        if count == TENSOR_COUNT {
            return Err(InventoryError::CountMismatch);
        }
        let (index, expected) = lookup(name).ok_or(InventoryError::UnknownName)?;
        if dtype != expected {
            return Err(InventoryError::WrongType);
        }
        let mask = 1_u64 << (index % 64);
        if seen[index / 64] & mask != 0 {
            return Err(InventoryError::DuplicateName);
        }
        seen[index / 64] |= mask;
        count += 1;
    }
    if count != TENSOR_COUNT {
        return Err(InventoryError::CountMismatch);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BankBytes {
    pub logical: u64,
    /// Six separately rounded exclusive main-pool allocations per layer.
    pub physical: u64,
}

// One expert's actual 2560x640 or 640x2560 original packed tensor.
fn expert_bytes(dtype: u32) -> u64 {
    let (block, bytes) = match dtype {
        18 => (256, 98),  // IQ3_XXS
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        42 => (64, 18),   // Q2_0: half scale + 16 packed bytes, not TQ2_0
        _ => unreachable!("pinned routed profile has no other raw format"),
    };
    (2560 * 640 / block) * bytes
}

/// Metadata-only 144-slot capacity proof for the fixed 72+72 owner geometry.
/// This is not allocation telemetry, and grants no host or retirement credit.
pub fn bank_bytes_144() -> BankBytes {
    let mut result = BankBytes {
        logical: 0,
        physical: 0,
    };
    for layer in 0..LAYER_COUNT {
        for dtype in routed_types(layer).expect("all pinned layers have expert triples") {
            let expert = expert_bytes(dtype);
            result.logical += expert * 144;
            let segment = expert * 72;
            // Existing main_admission exclusive bins, independently per root.
            assert!(segment <= 128 * 1024 * 1024);
            result.physical += 2 * segment.max(32 * 1024).next_power_of_two();
        }
    }
    result
}

/// Selected port allocation geometry; exact persistent roots have no bin padding.
pub fn selected_bank_bytes() -> BankBytes {
    if !crate::strata_port::enabled() {
        return bank_bytes_144();
    }
    let logical = (0..LAYER_COUNT)
        .flat_map(|layer| routed_types(layer).unwrap())
        .map(|dtype| expert_bytes(dtype) * BANK_SLOTS as u64)
        .sum();
    BankBytes {
        logical,
        physical: logical,
    }
}

/// Shader shard plans and CPU source views use exactly the same fixed geometry.
pub(crate) fn source_geometry_shader(source: &str) -> String {
    if !crate::strata_port::enabled() {
        return source.into();
    }
    let mut source = source.to_owned();
    for (index, old) in [(1, 72), (2, 144), (3, 288), (4, 432)] {
        source = source.replace(
            &format!("if shard=={index}u {{first={old}u;}}"),
            &format!("if shard=={index}u {{first={}u;}}", SOURCE_FIRST[index]),
        );
    }
    source = source
        .replace(
            "if shard==2u {parts=144u;}",
            &format!("if shard==2u {{parts={HOST_PART}u;}}"),
        )
        .replace(
            "if shard==3u {parts=144u;}",
            &format!("if shard==3u {{parts={HOST_PART}u;}}"),
        );
    source
        .replace("var parts=72u;", &format!("var parts={}u;", SEGMENT_SLOTS))
        .replace(
            "if shard==4u {parts=80u;}",
            &format!("if shard==4u {{parts={}u;}}", SOURCE_PARTS[4]),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_header_inventory() -> Vec<(String, u32)> {
        let receipt: serde_json::Value = serde_json::from_str(include_str!(
            "../../text/receipts/daslab-iq3-artifact-20261009.json"
        ))
        .unwrap();
        assert_eq!(receipt["revision"], REVISION);
        assert_eq!(receipt["repository"], REPOSITORY);
        for (shard, digest) in HEADER_SHA256.iter().enumerate() {
            assert_eq!(receipt["shards"][shard]["header_sha256"], *digest);
        }
        let entries = receipt["tensor_inventory"].as_array().unwrap();
        assert_eq!(entries.len(), TENSOR_COUNT);
        assert_eq!(
            entries.iter().filter(|t| t["source_shard"] == 0).count(),
            1223
        );
        assert_eq!(entries.iter().filter(|t| t["source_shard"] == 1).count(), 1);
        entries
            .iter()
            .map(|t| {
                (
                    t["name"].as_str().unwrap().to_owned(),
                    u32::try_from(t["type_id"].as_u64().unwrap()).unwrap(),
                )
            })
            .collect()
    }

    fn validate(entries: &[(String, u32)]) -> Result<(), InventoryError> {
        validate_inventory(entries.iter().map(|(name, dtype)| (name.as_str(), *dtype)))
    }

    #[test]
    fn all_1224_actual_header_names_and_raw_types_match_without_order_assumptions() {
        let mut entries = fixed_header_inventory();
        assert_eq!(validate(&entries), Ok(()));
        for (name, dtype) in &entries {
            assert_eq!(expected_dtype(name), Some(*dtype), "{name}");
        }
        entries.reverse();
        assert_eq!(validate(&entries), Ok(()));
        let expected_count = GLOBALS.len()
            + FAMILIES
                .iter()
                .map(|(_, types)| types.iter().filter(|&&dtype| dtype != ABSENT).count())
                .sum::<usize>();
        assert_eq!(expected_count, TENSOR_COUNT);
    }

    #[test]
    fn missing_duplicate_unknown_wrong_type_and_extra_are_rejected() {
        let entries = fixed_header_inventory();
        assert_eq!(
            validate(&entries[..TENSOR_COUNT - 1]),
            Err(InventoryError::CountMismatch)
        );
        let mut changed = entries.clone();
        changed[1] = changed[0].clone();
        assert_eq!(validate(&changed), Err(InventoryError::DuplicateName));
        changed = entries.clone();
        changed[0].0 = "blk.48.ffn_down_exps.weight".into();
        assert_eq!(validate(&changed), Err(InventoryError::UnknownName));
        changed = entries.clone();
        changed[0].1 ^= 1;
        assert_eq!(validate(&changed), Err(InventoryError::WrongType));
        changed = entries.clone();
        changed.push(entries[0].clone());
        assert_eq!(validate(&changed), Err(InventoryError::CountMismatch));
    }

    #[test]
    fn canonical_layers_and_exact_family_presence_are_required() {
        for name in [
            "blk.00.ffn_down_exps.weight",
            "blk.+1.ffn_down_exps.weight",
            "blk.-1.ffn_down_exps.weight",
            "blk..ffn_down_exps.weight",
            "blk.48.ffn_down_exps.weight",
            "blk.0.attn_q.weight",
            "blk.0.ple_conv1d.weight",
            "blk.3.ssm_a",
            "blk.1.unknown.weight",
            "token_embd.weight.extra",
            "output.weight\0",
            "blk.999999999999999999999999.ffn_down_exps.weight",
        ] {
            assert_eq!(expected_dtype(name), None, "{name}");
        }
        assert_eq!(expected_dtype("blk.1.ple_conv1d.weight"), Some(1));
        assert_eq!(expected_dtype("blk.3.attn_q.weight"), Some(23));
        assert_eq!(expected_dtype("token_embd.weight"), Some(23));
        assert_eq!(expected_dtype("output.weight"), Some(14));
        assert_eq!(expected_dtype("per_layer_token_embd.weight"), Some(20));
    }

    #[test]
    fn every_routed_triple_matches_the_fixed_header_and_all_seven_groups() {
        let entries = fixed_header_inventory();
        let mut groups = std::collections::BTreeMap::<[u32; 3], usize>::new();
        for layer in 0..LAYER_COUNT {
            let types = routed_types(layer).unwrap();
            assert_eq!(types[0], types[1]);
            for (family, dtype) in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ]
            .into_iter()
            .zip(types)
            {
                let name = format!("blk.{layer}.{family}");
                assert!(entries.iter().any(|(n, t)| n == &name && *t == dtype));
            }
            *groups.entry(types).or_default() += 1;
        }
        assert_eq!(
            groups,
            std::collections::BTreeMap::from([
                ([18, 18, 20], 14),
                ([18, 18, 42], 3),
                ([21, 21, 20], 9),
                ([21, 21, 42], 1),
                ([22, 22, 20], 15),
                ([22, 22, 42], 5),
                ([23, 23, 20], 1),
            ])
        );
        assert_eq!(routed_types(LAYER_COUNT), None);
        assert_eq!(routed_types(usize::MAX), None);
    }

    #[test]
    fn separately_rounded_144_slot_bank_plan_matches_actual_tensor_bytes() {
        let receipt: serde_json::Value = serde_json::from_str(include_str!(
            "../../text/receipts/daslab-iq3-artifact-20261009.json"
        ))
        .unwrap();
        let mut logical = 0;
        let mut physical = 0;
        let mut selected_logical = 0;
        for t in receipt["tensor_inventory"].as_array().unwrap() {
            let name = t["name"].as_str().unwrap();
            if name.ends_with("ffn_gate_exps.weight")
                || name.ends_with("ffn_up_exps.weight")
                || name.ends_with("ffn_down_exps.weight")
            {
                let total = t["tensor_bytes"].as_u64().unwrap();
                assert_eq!(total % EXPERT_COUNT as u64, 0);
                let expert = total / EXPERT_COUNT as u64;
                assert_eq!(expert_bytes(t["type_id"].as_u64().unwrap() as u32), expert);
                // This is the independent historical144-slot receipt, even
                // when the executable selects a240-slot port configuration.
                logical += expert * 144;
                selected_logical += expert * BANK_SLOTS as u64;
                physical += 2 * (expert * 72).max(32768).next_power_of_two();
            }
        }
        assert_eq!(bank_bytes_144(), BankBytes { logical, physical });
        assert_eq!(logical, 14_144_716_800);
        assert_eq!(physical, 18_723_373_056);
        let selected = selected_bank_bytes();
        assert_eq!(selected.logical, selected_logical);
        assert_eq!(
            selected.physical,
            if crate::strata_port::enabled() {
                selected_logical
            } else {
                physical
            }
        );
    }
}
