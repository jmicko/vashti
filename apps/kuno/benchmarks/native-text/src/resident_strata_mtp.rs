//! Immutable original-GGUF role/page geometry for the explicit Strata draft.
//! Runtime .bin conversion, dense requantization and vocabulary pruning are
//! separate candidates. Original public files remain borrowed, never copied.
use super::*;

pub const SHARDS: usize = 4;
pub const EXPERTS_PER_SHARD: usize = 128;
pub const GU_NAME: &str = "blk.48.ffn_gate_up_exps.weight";
pub const DOWN_NAME: &str = "blk.48.ffn_down_exps.weight";
pub const FC_EMBEDDING: &str = "mtp.fc_embedding.weight";
pub const FC_HIDDEN: &str = "mtp.fc_hidden.weight";
pub const GU_ROW_BYTES: usize = k::D / 64 * 18;
pub const DOWN_ROW_BYTES: usize = k::FF / 64 * 18;
pub const GU_EXPERT_BYTES: usize = 2 * k::FF * GU_ROW_BYTES;
pub const DOWN_EXPERT_BYTES: usize = k::D * DOWN_ROW_BYTES;

const fn matrices() -> [Spec; 19] {
    let mut result = [MATRICES[0]; 19];
    let mut dest = 0;
    let mut i = 0;
    while i < MATRICES.len() {
        if i != 9 {
            // The Q8 concatenated EH has no counterpart in this graph.
            let mut spec = MATRICES[i];
            spec.dtype = if i == 0 {
                quant_matvec::Q6_K
            } else {
                quant_matvec::BF16
            };
            result[dest] = spec;
            dest += 1;
        }
        i += 1;
    }
    result[16] = Spec {
        name: FC_EMBEDDING,
        columns: k::D,
        rows: k::D,
        dtype: quant_matvec::BF16,
    };
    result[17] = Spec {
        name: FC_HIDDEN,
        columns: k::D,
        rows: k::D,
        dtype: quant_matvec::BF16,
    };
    result[18] = Spec {
        name: "blk.48.ffn_gate_inp_shexp.weight",
        columns: k::D,
        rows: 1,
        dtype: quant_matvec::BF16,
    };
    result
}
pub(super) const MATRICES_Q2: [Spec; 19] = matrices();
pub(super) const HEAD_UP_Q2: Spec = Spec {
    dtype: quant_matvec::BF16,
    ..HEAD_UP
};
pub(super) const EMBEDDING_Q2: Spec = Spec {
    dtype: quant_matvec::IQ4_XS,
    ..EMBEDDING
};
pub(super) const VECTORS_Q2: [(&str, usize); 7] = [
    VECTORS[0], VECTORS[1], VECTORS[3], VECTORS[4], VECTORS[5], VECTORS[6], VECTORS[7],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shard {
    pub first: usize,
}
impl Shard {
    pub fn new(index: usize) -> Result<Self> {
        if index >= SHARDS {
            return Err(Error::OutOfBounds);
        }
        Ok(Self {
            first: index * EXPERTS_PER_SHARD,
        })
    }
    pub fn bytes(self, gate_up: bool) -> Result<usize> {
        if !self.first.is_multiple_of(EXPERTS_PER_SHARD) || self.first >= k::EXPERTS {
            return Err(Error::OutOfBounds);
        }
        Ok(EXPERTS_PER_SHARD
            * if gate_up {
                GU_EXPERT_BYTES
            } else {
                DOWN_EXPERT_BYTES
            })
    }
    pub fn owned(self, gate_up: bool) -> Result<usize> {
        Ok(main_admission::page_bytes(self.bytes(gate_up)? as u64).map_err(error)? as usize)
    }
}
pub fn expert_owned_bytes() -> Result<u64> {
    (0..SHARDS).try_fold(0, |total, index| {
        let shard = Shard::new(index)?;
        Ok(total + shard.owned(true)? as u64 + shard.owned(false)? as u64)
    })
}
