//! Original-packed Strata Q2 arithmetic, reusing native64 lane dots exactly.
//! Parent owns source/fullspan/stream/noalias/route/fault/complete publication.
use crate::{
    mtp_resident_kernels as k,
    quant_matvec::{self as q, fast},
};
use cubecl::prelude::*;

#[cube]
#[allow(clippy::manual_range_contains)]
fn finite(value: f32) -> bool {
    value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32
}
#[cube]
#[allow(unused_assignments)]
fn sigmoid(value: f32) -> f32 {
    let mut out = 0.0_f32;
    if value >= 0.0 {
        out = 1.0 / (1.0 + (-value).exp());
    } else {
        let e = value.exp();
        out = e / (1.0 + e);
    }
    out
}
#[cube(launch, launch_unchecked)]
pub fn embedding_tile(
    weights: &Array<u32>,
    token: &Array<u32>,
    output: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
    #[comptime] rows: usize,
) {
    let column = ABSOLUTE_POS;
    let selected = token[0] as usize;
    if column < k::D {
        if selected >= k::VOCAB {
            fault[0].fetch_or(1);
        } else if selected >= first && selected - first < rows {
            let value = q::packed_value(weights, (selected - first) * 1360, column, q::IQ4_XS);
            if finite(value) {
                output[column] = value;
            } else {
                fault[0].fetch_or(1);
            }
        }
    }
}
/// The hidden FC has finished into residual; embedding FC scratch is separate.
/// One in-place resource avoids an aliased readonly/writable duplicate binding.
#[cube(launch, launch_unchecked)]
pub fn broadcast_embedding(
    embedding: &Array<f32>,
    residual: &mut Array<f32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let i = ABSOLUTE_POS;
    if i < k::WIDE {
        let h = residual[i];
        let e = embedding[i % k::D];
        let value = h + e;
        if finite(h) && finite(e) && finite(value) {
            residual[i] = value;
        } else {
            fault[0].fetch_or(1);
        }
    }
}
/// Fused GGUF expert rows are gate0..639 then up640..1279, not runtime
/// interleaved rows. Four128-expert gate/up shards precede all down shards.
#[cube(launch, launch_unchecked)]
pub fn gate_up(
    weights: &Array<u32>,
    input: &Array<f32>,
    packet: &Array<u32>,
    activation: &mut Array<f32>,
    readiness: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
) {
    let rank = CUBE_POS_Y as usize;
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let mut id = 512_usize;
    if rank < k::RANKS {
        id = packet[k::ROUTE_FIRST + rank * 2] as usize;
    }
    let mut local = 0_usize;
    let selected = rank < k::RANKS && id >= first && id - first < 128 && id < k::EXPERTS;
    if selected {
        local = id - first;
    }
    let valid = selected && row < k::FF && fault[0].load() == 0;
    let base = (local * 1280 + row) * 720;
    let gate = fast::dot(weights, input, base, 0, valid, k::D, q::Q2_0);
    let up = fast::dot(weights, input, base + 640 * 720, 0, valid, k::D, q::Q2_0);
    if fast::lane() == 0 && valid {
        let value = gate * sigmoid(gate) * up;
        if finite(value) {
            activation[rank * k::FF + row] = value;
            if row == 0 {
                readiness[rank] = 1;
            }
        } else {
            fault[0].fetch_or(1);
        }
    }
}
#[cube(launch, launch_unchecked)]
pub fn down(
    weights: &Array<u32>,
    activation: &Array<f32>,
    packet: &Array<u32>,
    partials: &mut Array<f32>,
    readiness: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    #[comptime] first: usize,
) {
    let rank = CUBE_POS_Y as usize;
    let row = CUBE_POS_X as usize * fast::ROWS_PER_CUBE + fast::local_row();
    let mut id = 512_usize;
    if rank < k::RANKS {
        id = packet[k::ROUTE_FIRST + rank * 2] as usize;
    }
    let mut local = 0_usize;
    let selected = rank < k::RANKS && id >= first && id - first < 128 && id < k::EXPERTS;
    if selected {
        local = id - first;
    }
    let mut valid = selected && row < k::D && fault[0].load() == 0;
    if rank < k::RANKS {
        valid = valid && readiness[rank] == 1;
    }
    let base = (local * k::D + row) * 180;
    let value = fast::dot(
        weights,
        activation,
        base,
        rank * k::FF,
        valid,
        k::FF,
        q::Q2_0,
    );
    if fast::lane() == 0 && valid {
        if finite(value) {
            partials[rank * k::D + row] = value;
            if row == 0 {
                readiness[k::RANKS + rank] = 1;
            }
        } else {
            fault[0].fetch_or(1);
        }
    }
}
