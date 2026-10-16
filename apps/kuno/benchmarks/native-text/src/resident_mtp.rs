//! Fixed-public, fully resident original-packed MTP laboratory workspace.
//!
//! No model alias, HTTP request, speculative acceptance controller or arbitrary
//! configuration is exposed here. The caller supplies a fresh exclusive-bin
//! client and an explicitly selected strict `MtpGraph`. Every expert is original and
//! resident. A step is provisional until the target verifies it. Position is
//! committed only after complete finite/fault/tail and guarded output proofs.
//! A failed queued step destroys the entire cache and poisons this sequence.
#![allow(dead_code)] // Parent fixed-public registration/physical gate is separate.

#[path = "strata_mtp_cjk.rs"]
pub(crate) mod cjk;
#[path = "resident_mtp_prompt.rs"]
mod prompt;
#[path = "resident_strata_mtp.rs"]
mod strata;
#[path = "strata_mtp_kernels.rs"]
mod strata_k;
#[path = "mtp_private_sweeps.rs"]
mod sweeps;

use crate::{
    main_admission::{self, MainAdmission},
    mtp_resident_kernels as k,
    private_arena::{BufferId, BufferLayout, OwnedZeroAudit, PrivateArena},
    quant_matvec::{self, fast, matmul},
    resident_util as u,
};
use cubecl::{prelude::*, server::Handle, wgpu::WgpuRuntime};
use fast::pruned_head as english;
use kuno_text::{
    Error, Result, TensorType,
    mtp::{MtpGraph, MtpProfile},
};
use std::{
    cell::Cell,
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};

const ROPE_CONTEXT_INCREMENT: u64 = if crate::strata_intel_core::enabled() {
    7 * 1024 * 1024
} else {
    0
};
pub const PUBLIC_BYTES: u64 = 5_011_668_992 + ROPE_CONTEXT_INCREMENT;
pub const ENGLISH_VOCAB_OWNED_BYTES: u64 = english::ENGLISH_OWNED_BYTES as u64;
pub const ENGLISH_PUBLIC_BYTES: u64 = PUBLIC_BYTES + ENGLISH_VOCAB_OWNED_BYTES;
// Original GGUF dense BF16 +borrowed IQ4_XS embedding/Q6_K full head. Exact
// independent startup page geometry is checked before any device allocation.
pub const STRATA_PUBLIC_BYTES: u64 = 2_539_257_856 + ROPE_CONTEXT_INCREMENT;
pub const MAIN_BYTES: u64 = 6 * 1024 * 1024 * 1024;
pub const PRIVATE_BYTES: usize = if crate::strata_intel_core::enabled() {
    128
} else {
    16
} * 1024
    * 1024;
pub const RUNTIME_BYTES: u64 = 64 * 1024 * 1024
    + if cfg!(feature = "strata-intel-core") {
        crate::strata_sampler::request::RUNTIME_BYTES
    } else {
        0
    };
pub const HOST_BYTES: usize = 16 * 1024 * 1024;
pub const PUBLIC_SOURCE_BYTES: usize = 128 * 1024 * 1024;
const READ_CHUNK: usize = 64 * 1024 * 1024;
const TILE_ROWS: usize = 16384;
const EPS: f32 = 1e-6;
// Original MTP conditioning is one global10240 RMS row. Ordinary HC mixers
// below remain four independent2560 RMS rows. vLLM0.30 nvidia/mtp.py211–216.
const HIDDEN_NORM_GROUPS: usize = 1;
const HIDDEN_NORM_WIDTH: usize = k::WIDE;
// Conservatively includes all per-owner clears/checks, projection/finite
// folds, sixteen embedding tiles and eight gate/up + eight down shards. The
// source ledger never credits a dropped owner before a completed fence.
const MAX_TOKEN_DISPATCHES: usize = 512;
const INFO_BYTES: u64 = MAX_TOKEN_DISPATCHES as u64 * 2048;
// Non-fused worst case: original40 clears/42 guards/rewind, then two
//39-clear/84-guard scalar cells. Original condition/HC/KV math uses46 calls.
const MAX_BATCHED_REPAIR_DISPATCHES: usize = 40 + 1 + N + 2 * (39 + 2 * N + 46);
const _: () = assert!(MAX_BATCHED_REPAIR_DISPATCHES <= MAX_TOKEN_DISPATCHES);
// Separate unused fused-only envelope. Every original owner is still checked;
// fused sweeps group those checks without changing their mathematical work.
const MAX_FUSED_FOUR_REPAIR_DISPATCHES: usize = 6 + 1 + 8 + 3 * (7 + 16 + 46);
const _: () = assert!(MAX_FUSED_FOUR_REPAIR_DISPATCHES <= MAX_TOKEN_DISPATCHES);
const PUBLIC_ENGLISH_IDS: u32 = u32::MAX;
static NEXT_BINDING: AtomicU64 = AtomicU64::new(1);
const MAX_LIVE_OUTPUTS: usize = 4;
const OUTPUT_HOST_BYTES: usize = (k::VOCAB + k::WIDE) * 4;
const DRAFT_TOKEN: usize = k::WIDE;
const DRAFT_FAULT: usize = k::WIDE + 1;
const DRAFT_WORDS: usize = k::WIDE + 2;
const DRAFT_PROBABILITY: usize = DRAFT_WORDS;
const DRAFT_NATIVE_WORDS: usize = DRAFT_WORDS + 1;

/// Unforgeable workspace-private authorization for the last completed full
/// forward. It never keys a shader/cache or crosses the public lab boundary.
struct LastForward {
    binding: u64,
    start: usize,
}
impl LastForward {
    fn valid(&self, binding: u64, position: usize) -> bool {
        self.binding == binding && position.checked_sub(1) == Some(self.start)
    }
}
impl Drop for LastForward {
    fn drop(&mut self) {
        self.binding.zeroize();
        self.start.zeroize();
    }
}

/// Completed KV front is deliberately a different capability from a completed
/// full forward; it cannot be passed to start_follow or publish a draft token.
struct LastFront(LastForward);
impl LastFront {
    fn valid(&self, binding: u64, position: usize) -> bool {
        self.0.valid(binding, position)
    }
}

enum NativeFrontInputs<'a> {
    Host(&'a [u32], &'a [f32]),
    Device(
        &'a [u32],
        &'a crate::resident_model::batch::target_rows::DeviceCarry<'a>,
    ),
}

/// Original TOKEN/CARRY owners staged before graph capture. One outstanding
/// preparation is allowed per MTP owner. Abandonment leaves that owner's flag
/// set; only consuming the exact receipt clears it. It never retires the GPU
/// context early or obstructs the original full-owner cleanup path.
struct PreparedNativeFront {
    binding: Zeroizing<u64>,
    ids: Zeroizing<[BufferId; 25]>,
    controls: Zeroizing<[usize; 4]>, // count, start, owner cursor, queued-input copies
    pending: Arc<AtomicBool>,
}
impl PreparedNativeFront {
    fn arm(
        binding: u64,
        ids: [BufferId; 25],
        count: usize,
        start: usize,
        cursor: usize,
        queued: bool,
        pending: &Arc<AtomicBool>,
    ) -> Result<Self> {
        if !(1..=8).contains(&count)
            || start.checked_add(count).is_none_or(|end| end > k::CONTEXT)
            || cursor > k::CONTEXT
            || queued && count != 1
        {
            return Err(Error::InvalidFormat("MTP native front preparation bounds"));
        }
        pending
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::InvalidFormat("MTP native front preparation pending"))?;
        Ok(Self {
            binding: Zeroizing::new(binding),
            ids: Zeroizing::new(ids),
            controls: Zeroizing::new([count, start, cursor, usize::from(queued)]),
            pending: pending.clone(),
        })
    }
    fn check(
        &self,
        binding: u64,
        ids: &[BufferId; 25],
        cursor: usize,
        pending: &Arc<AtomicBool>,
    ) -> Result<()> {
        if *self.binding != binding
            || &*self.ids != ids
            || self.controls[2] != cursor
            || !Arc::ptr_eq(&self.pending, pending)
            || !pending.load(Ordering::Acquire)
        {
            return Err(Error::InvalidFormat("MTP native front preparation changed"));
        }
        Ok(())
    }
    fn queued(self) -> Result<()> {
        self.pending
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::InvalidFormat("MTP native front preparation consumed"))?;
        Ok(())
    }
}
// Both caller/local receipt representations fit the existing2MiB host
// validation/stack margin; this allocates no private GPU owner or data mirror.
const _: () = {
    assert!(2 * size_of::<Option<PreparedNativeFront>>() <= 2048);
};

#[cfg(test)]
mod native_front_preparation_tests {
    use super::*;
    fn ids(epoch: u64) -> [BufferId; 25] {
        std::array::from_fn(|slot| BufferId::metadata_fixture(17, epoch, slot))
    }
    #[test]
    fn original_front_can_only_consume_its_exact_owner_cursor_and_roots() {
        let pending = Arc::new(AtomicBool::new(false));
        let roots = ids(1);
        let p = PreparedNativeFront::arm(91, roots, 4, 10, 13, false, &pending).unwrap();
        p.check(91, &roots, 13, &pending).unwrap();
        assert!(p.check(92, &roots, 13, &pending).is_err());
        assert!(p.check(91, &roots, 14, &pending).is_err());
        assert!(p.check(91, &ids(2), 13, &pending).is_err());
        assert!(
            p.check(91, &roots, 13, &Arc::new(AtomicBool::new(true)))
                .is_err()
        );
        assert!(PreparedNativeFront::arm(91, roots, 1, 14, 14, true, &pending).is_err());
        p.queued().unwrap();
        assert!(!pending.load(Ordering::Acquire));
        PreparedNativeFront::arm(91, roots, 1, 14, 14, true, &pending)
            .unwrap()
            .queued()
            .unwrap();
        assert!(!pending.load(Ordering::Acquire));
    }
    #[test]
    fn abandoning_or_panicking_with_front_preparation_keeps_owner_uncommittable() {
        for panic in [false, true] {
            let pending = Arc::new(AtomicBool::new(false));
            let p = PreparedNativeFront::arm(91, ids(1), 1, 10, 10, false, &pending).unwrap();
            if panic {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _receipt = p;
                    panic!("fixed public preparation abandonment");
                }));
                assert!(r.is_err());
            } else {
                drop(p);
            }
            assert!(pending.load(Ordering::Acquire));
            assert!(PreparedNativeFront::arm(91, ids(1), 1, 10, 10, false, &pending).is_err());
        }
    }
    #[test]
    fn front_shape_bounds_fail_before_arming_and_cover_teacher_and_selected_cursors() {
        let pending = Arc::new(AtomicBool::new(false));
        for (count, start, cursor, queued) in [
            (0, 0, 0, false),
            (9, 0, 0, false),
            (1, usize::MAX, 0, false),
            (8, k::CONTEXT - 7, k::CONTEXT - 1, false),
            (1, 0, k::CONTEXT + 1, false),
            (2, 0, 0, true),
        ] {
            assert!(
                PreparedNativeFront::arm(91, ids(1), count, start, cursor, queued, &pending)
                    .is_err()
            );
            assert!(!pending.load(Ordering::Acquire));
        }
        for count in 1..=8 {
            for start in [0, 10, k::CONTEXT - count] {
                for cursor in [start, start + count - 1] {
                    PreparedNativeFront::arm(91, ids(1), count, start, cursor, false, &pending)
                        .unwrap()
                        .queued()
                        .unwrap();
                }
            }
        }
        assert!(2 * size_of::<Option<PreparedNativeFront>>() <= 2048);
    }
}

struct AcceptedCacheRepair {
    rewind: bool,
    first_row: usize,
    end: usize,
}
impl AcceptedCacheRepair {
    fn new(
        old: usize,
        start: usize,
        completed_forward: bool,
        tokens: &[u32],
        preceding: &[f32],
        carries: &[f32],
    ) -> Result<Self> {
        Self::new_mode(
            old,
            start,
            completed_forward,
            tokens,
            preceding,
            carries,
            false,
        )
    }
    fn new_mode(
        old: usize,
        start: usize,
        completed_forward: bool,
        tokens: &[u32],
        preceding: &[f32],
        carries: &[f32],
        native_short_chain: bool,
    ) -> Result<Self> {
        if native_short_chain && !crate::strata_intel_core::enabled() {
            return Err(Error::InvalidFormat("native MTP repair mode"));
        }
        let count = tokens.len();
        let capacity = if crate::strata_intel_core::enabled() {
            8
        } else {
            3
        };
        if !(1..=capacity).contains(&count)
            || start == 0
            || old > k::CONTEXT
            || carries.len() != count * k::WIDE
            || carries.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidBufferLength);
        }
        let end = start
            .checked_add(count - 1)
            .filter(|&end| end <= k::CONTEXT)
            .ok_or(Error::OutOfBounds)?;
        let rewind = old >= start;
        if rewind {
            if !completed_forward
                || old - start > capacity - 2
                || !native_short_chain && count > old - start + 2
            {
                return Err(Error::InvalidFormat("MTP accepted repair draft boundary"));
            }
        } else if completed_forward || old != start - 1 || count != 1 {
            return Err(Error::InvalidFormat(
                "MTP accepted repair count-one boundary",
            ));
        }
        // Validate every supplied token/preceding pair before consuming the
        // old follow capability or clearing any owner. Last carry is validated
        // above even though it conditions the next round, not this cache.
        for (row, &token) in tokens.iter().enumerate() {
            let carry = if row == 0 {
                preceding
            } else {
                &carries[(row - 1) * k::WIDE..row * k::WIDE]
            };
            validate_input(token, carry, start + row - 1)?;
        }
        Ok(Self {
            rewind,
            first_row: usize::from(rewind),
            end,
        })
    }

    fn new_fused_four(
        old: usize,
        start: usize,
        completed_forward: bool,
        fused_sweeps: bool,
        tokens: &[u32],
        preceding: &[f32],
        carries: &[f32],
    ) -> Result<Self> {
        if !fused_sweeps || !completed_forward || start.checked_add(2) != Some(old) {
            return Err(Error::InvalidFormat("MTP fused four repair draft boundary"));
        }
        let count = tokens.len();
        if !(1..=4).contains(&count)
            || start == 0
            || old > k::CONTEXT
            || carries.len() != count * k::WIDE
            || carries.iter().any(|x| !x.is_finite())
        {
            return Err(Error::InvalidBufferLength);
        }
        let end = start
            .checked_add(count - 1)
            .filter(|&end| end <= k::CONTEXT)
            .ok_or(Error::OutOfBounds)?;
        for (row, &token) in tokens.iter().enumerate() {
            let carry = if row == 0 {
                preceding
            } else {
                &carries[(row - 1) * k::WIDE..row * k::WIDE]
            };
            validate_input(token, carry, start + row - 1)?;
        }
        Ok(Self {
            rewind: true,
            first_row: 1,
            end,
        })
    }
}

fn completed_batched_repair(mut traffic: Traffic, cells: usize) -> Result<Traffic> {
    traffic.batched_repair_calls = traffic
        .batched_repair_calls
        .checked_add(1)
        .ok_or(Error::ArithmeticOverflow)?;
    traffic.batched_repair_cells = traffic
        .batched_repair_cells
        .checked_add(cells as u64)
        .ok_or(Error::ArithmeticOverflow)?;
    traffic.batched_repair_fault_gates = traffic
        .batched_repair_fault_gates
        .checked_add(1)
        .ok_or(Error::ArithmeticOverflow)?;
    Ok(traffic)
}

struct OutputLease {
    live: Arc<AtomicUsize>,
}
impl OutputLease {
    fn acquire(live: &Arc<AtomicUsize>) -> Result<Self> {
        live.fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < MAX_LIVE_OUTPUTS).then_some(n + 1)
        })
        .map_err(|_| Error::LimitExceeded("MTP live guarded outputs"))?;
        Ok(Self { live: live.clone() })
    }
}
impl Drop for OutputLease {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
enum B {
    Token,
    Carry,
    Embedding,
    EmbeddingNorm,
    HiddenNorm,
    Condition,
    Residual0,
    Residual1,
    Residual2,
    Norm,
    LowRaw,
    Low,
    Up,
    Mixed,
    Injection,
    Status,
    RawQuery,
    SplitQuery,
    Query,
    Key,
    Value,
    KeyRotated,
    Keys,
    Values,
    Attention,
    Probability,
    Attended,
    Block,
    Router,
    Packet,
    Activation,
    Partials,
    Readiness,
    SharedGate,
    SharedUp,
    SharedAct,
    SharedDown,
    SharedScalar,
    Logits,
    Candidates,
    Greedy,
    Fault,
}
const N: usize = B::Fault as usize + 1;
const WORDS: [usize; N] = [
    1,
    k::WIDE,
    k::D,
    k::D,
    k::WIDE,
    k::CONDITION_WORDS,
    k::WIDE,
    k::WIDE,
    k::WIDE,
    k::WIDE,
    k::LOW_RANK,
    k::LOW_RANK,
    k::WIDE,
    k::D,
    k::HC,
    k::QUERY * 2,
    k::QUERY * 2,
    k::QUERY * 2,
    k::QUERY,
    k::KV,
    k::KV,
    k::KV,
    k::CACHE_STORAGE_WORDS,
    k::CACHE_STORAGE_WORDS,
    if crate::strata_intel_core::enabled() {
        24 * k::CONTEXT.div_ceil(64) * 256
    } else {
        k::ATTENTION_WORDS
    },
    if crate::strata_intel_core::enabled() {
        24 * k::CONTEXT.div_ceil(64) * 2
    } else {
        k::ATTENTION_WORDS
    },
    k::QUERY,
    k::D,
    k::EXPERTS,
    if k::CONTEXT > 8192 { k::CONTEXT } else { 8192 },
    k::ACTIVATION_WORDS,
    k::PARTIAL_WORDS,
    k::READINESS_WORDS,
    k::FF,
    k::FF,
    k::FF,
    k::D,
    1,
    k::VOCAB,
    k::GREEDY_WORDS,
    1,
    1,
];
fn integer_buffer(index: usize) -> bool {
    if crate::strata_intel_core::enabled()
        && [B::Keys as usize, B::Values as usize].contains(&index)
    {
        return true;
    }
    [
        B::Token,
        B::Status,
        B::Packet,
        B::Readiness,
        B::Candidates,
        B::Greedy,
        B::Fault,
    ]
    .iter()
    .any(|b| *b as usize == index)
}

fn sweep_role(index: usize) -> u32 {
    if index == B::Keys as usize || index == B::Values as usize {
        if crate::strata_intel_core::enabled() {
            sweeps::INTEGER
        } else {
            sweeps::CACHE
        }
    } else if integer_buffer(index) {
        sweeps::INTEGER
    } else {
        sweeps::FLOAT
    }
}
fn clear_roles(preserve_inputs: bool) -> ([usize; 40], usize) {
    const { assert!(N == 42 && B::Fault as usize == 41) };
    let mut roles = [0; 40];
    let mut count = 0;
    for index in 0..N {
        let cache = index == B::Keys as usize || index == B::Values as usize;
        let input = [B::Token, B::Carry, B::Fault]
            .iter()
            .any(|b| index == *b as usize);
        if !(cache || preserve_inputs && input) {
            roles[count] = index;
            count += 1;
        }
    }
    (roles, count)
}

#[derive(Clone, Copy)]
struct Spec {
    name: &'static str,
    columns: usize,
    rows: usize,
    dtype: u32,
}
const MATRICES: [Spec; 17] = [
    Spec {
        name: "output.weight",
        columns: k::D,
        rows: k::VOCAB,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.attn_q.weight",
        columns: k::D,
        rows: k::QUERY * 2,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.attn_k.weight",
        columns: k::D,
        rows: k::KV,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.attn_v.weight",
        columns: k::D,
        rows: k::KV,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.attn_output.weight",
        columns: k::QUERY,
        rows: k::D,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.ffn_gate_shexp.weight",
        columns: k::D,
        rows: k::FF,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.ffn_up_shexp.weight",
        columns: k::D,
        rows: k::FF,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.ffn_down_shexp.weight",
        columns: k::FF,
        rows: k::D,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.ffn_gate_inp.weight",
        columns: k::D,
        rows: k::EXPERTS,
        dtype: quant_matvec::F32,
    },
    Spec {
        name: "blk.48.nextn.eh_proj.weight",
        columns: k::D * 2,
        rows: k::D,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_attn_down.weight",
        columns: k::WIDE,
        rows: k::LOW_RANK,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_attn_up.weight",
        columns: k::LOW_RANK,
        rows: k::WIDE,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_attn_inject.weight",
        columns: k::WIDE,
        rows: k::HC,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_ffn_down.weight",
        columns: k::WIDE,
        rows: k::LOW_RANK,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_ffn_up.weight",
        columns: k::LOW_RANK,
        rows: k::WIDE,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.hc_ffn_inject.weight",
        columns: k::WIDE,
        rows: k::HC,
        dtype: quant_matvec::Q8_0,
    },
    Spec {
        name: "blk.48.nextn.hc_head_down.weight",
        columns: k::WIDE,
        rows: k::LOW_RANK,
        dtype: quant_matvec::Q8_0,
    },
];
// Kept separately so every role is a compile-time original public tensor name.
const HEAD_UP: Spec = Spec {
    name: "blk.48.nextn.hc_head_up.weight",
    columns: k::LOW_RANK,
    rows: k::WIDE,
    dtype: quant_matvec::Q8_0,
};
const EMBEDDING: Spec = Spec {
    name: "token_embd.weight",
    columns: k::D,
    rows: k::VOCAB,
    dtype: quant_matvec::Q8_0,
};
const VECTORS: [(&str, usize); 8] = [
    ("blk.48.attn_q_norm.weight", k::HD),
    ("blk.48.attn_k_norm.weight", k::HD),
    ("blk.48.ffn_gate_inp_shexp.weight", k::D),
    ("blk.48.nextn.enorm.weight", k::D),
    ("blk.48.nextn.hnorm.weight", k::WIDE),
    ("blk.48.nextn.hc_head_norm.weight", k::WIDE),
    ("blk.48.hc_attn_norm.weight", k::WIDE),
    ("blk.48.hc_ffn_norm.weight", k::WIDE),
];
const EXPERT_NAMES: [&str; 3] = [
    "blk.48.ffn_gate_exps.weight",
    "blk.48.ffn_up_exps.weight",
    "blk.48.ffn_down_exps.weight",
];

struct Page {
    handle: Handle,
    logical_bytes: usize,
    owned_bytes: usize,
    first: usize,
    rows: usize,
}
struct Public {
    pages: BTreeMap<&'static str, Vec<Page>>,
    sin: Page,
    cos: Page,
    english_ids: Option<Page>,
    cjk: Option<cjk::Head>,
    bytes: u64,
}
impl Public {
    fn pages(&self, name: &'static str) -> Result<&[Page]> {
        self.pages
            .get(name)
            .map(Vec::as_slice)
            .ok_or(Error::InvalidFormat("MTP resident public role"))
    }
    fn one(&self, name: &'static str) -> Result<&Page> {
        let pages = self.pages(name)?;
        if pages.len() != 1 {
            return Err(Error::InvalidBufferLength);
        }
        Ok(&pages[0])
    }
    fn handles(&self) -> Vec<Handle> {
        self.pages
            .values()
            .flatten()
            .map(|p| p.handle.clone())
            .chain([self.sin.handle.clone(), self.cos.handle.clone()])
            .chain(self.english_ids.iter().map(|p| p.handle.clone()))
            .chain(self.cjk.iter().flat_map(cjk::Head::handles))
            .collect()
    }
}

/// Guarded whole draft result. These values remain provisional until target
/// verification; no request-derived values implement Debug/Serialize/Clone.
pub struct Output {
    logits: Zeroizing<Vec<f32>>,
    hidden: Zeroizing<Vec<f32>>,
    greedy: Zeroizing<u32>,
    binding: u64,
    start: Zeroizing<usize>,
    // Last field: contents above are scrubbed BEFORE host credit is released.
    _lease: OutputLease,
}
impl Output {
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }
    pub fn greedy(&self) -> u32 {
        *self.greedy
    }
    pub fn binding(&self) -> u64 {
        self.binding
    }
    pub fn start(&self) -> usize {
        *self.start
    }
}
pub struct CloseReceipt {
    pub private: OwnedZeroAudit,
    /// Closed public addressable spans, including shared views. This is not
    /// physical release credit for roots still pinned by the target owner.
    pub public_bytes: u64,
    pub main_reserved: u64,
}

/// Content-free scheduling counters. Full-owner host read/upload bytes are
/// counted after the backend operation succeeds, including a later rejected
/// result. Construction is included; compare snapshots around each operation.
/// Batched-repair fields count only fully accepted joined cache repairs.
/// These counters do not synchronize the device or include close's separately
/// audited erasure readback. `explicit_dispatches` counts workspace dispatches,
/// not any backend-internal copy/sanitizer work.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Traffic {
    pub host_read_calls: u64,
    pub host_read_bytes: u64,
    pub private_upload_calls: u64,
    pub private_upload_bytes: u64,
    pub minimal_packet_reads: u64,
    pub followup_copy_calls: u64,
    pub draft_window_reads: u64,
    pub full_output_reads: u64,
    pub explicit_dispatches: u64,
    pub batched_repair_calls: u64,
    pub batched_repair_cells: u64,
    pub batched_repair_fault_gates: u64,
}

/// Provisional draft result without a full vocabulary host copy. The complete
/// vocabulary is selected by immutable construction policy. Token/probability
/// come from the real GPU head/sampler; the target verifies every proposal.
pub struct DraftOutput {
    probability: Option<Zeroizing<f32>>,
    hidden: Zeroizing<Vec<f32>>,
    greedy: Zeroizing<u32>,
    binding: u64,
    start: Zeroizing<usize>,
    _lease: OutputLease,
}
impl DraftOutput {
    pub fn probability(&self) -> Result<f32> {
        self.probability
            .as_deref()
            .copied()
            .ok_or(Error::InvalidFormat("MTP confidence unavailable"))
    }
    pub fn hidden(&self) -> &[f32] {
        &self.hidden
    }
    pub fn greedy(&self) -> u32 {
        *self.greedy
    }
    pub fn binding(&self) -> u64 {
        self.binding
    }
    pub fn start(&self) -> usize {
        *self.start
    }
}

#[path = "resident_mtp_device.rs"]
mod device_cache;
pub(crate) use device_cache::NativeRequestResetReceipt;
#[path = "resident_mtp_admission.rs"]
mod shared_admission;
use shared_admission::Ledger;
#[path = "strata_mtp_replay.rs"]
mod replay;
#[path = "resident_mtp_window.rs"]
mod window;

pub struct ResidentMtp<'g, 'f> {
    graph: &'g MtpGraph<'f>,
    arena: PrivateArena,
    ids: [BufferId; N],
    draft_window: Option<BufferId>,
    dense_planes: Option<BufferId>,
    native_front: Option<crate::strata_mtp_front::Workspace>,
    native_front_pending: Arc<AtomicBool>,
    native_replay: replay::Cache,
    client: ComputeClient<WgpuRuntime>,
    public: Option<Public>,
    shared_target_weights: Option<Arc<crate::strata_shared_weights::Weights>>,
    ledger: Ledger,
    retired: Arc<AtomicBool>,
    position: usize,
    poisoned: bool,
    closed: bool,
    dispatches: usize,
    binding: u64,
    live_outputs: Arc<AtomicUsize>,
    last_forward: Option<LastForward>,
    last_front: Option<LastFront>,
    traffic: Cell<Traffic>,
    fused_sweeps: bool,
    q8_head_broadcast: bool,
    q8_head_policy_locked: bool,
    q8_head_broadcast_completed: bool,
    q8_head_shared_scales: bool,
    q8_head_shared_scales_completed: bool,
    public_english_vocab: Option<english::EnglishVocabulary>,
    public_english_vocab_completed: bool,
}

fn validate_q8_head_broadcast_selection(
    profile: MtpProfile,
    closed: bool,
    poisoned: bool,
    position: usize,
    locked: bool,
) -> Result<()> {
    if profile != MtpProfile::SelfContainedQ8 || closed || poisoned || position != 0 || locked {
        return Err(Error::InvalidFormat(
            "MTP Q8 head selection requires unused Q8 workspace",
        ));
    }
    Ok(())
}

fn validate_q8_head_modes(broadcast: bool, shared_scales: bool) -> Result<()> {
    if broadcast && shared_scales {
        return Err(Error::InvalidFormat(
            "MTP Q8 head experiments are mutually exclusive",
        ));
    }
    Ok(())
}

fn error(_: impl std::fmt::Display) -> Error {
    Error::InvalidFormat("MTP resident owning operation")
}
fn sample(client: &ComputeClient<WgpuRuntime>) -> Result<u64> {
    Ok(client.memory_usage().map_err(error)?.bytes_reserved)
}
fn io_audit(client: &ComputeClient<WgpuRuntime>, retired: &AtomicBool) -> Result<()> {
    let receipt =
        cubecl::future::block_on(client.sanitize_transient_io_audited()).map_err(error)?;
    if receipt.inspected_bytes > RUNTIME_BYTES || retired.load(Ordering::Acquire) {
        return Err(Error::InvalidFormat("MTP resident I/O audit"));
    }
    Ok(())
}
struct Bytes(cubecl::bytes::Bytes);
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
struct Upload(Vec<u8>);
impl Drop for Upload {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
impl Upload {
    fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.0)
    }
}

/// Full span validator. All loads are guarded by actual array length; only the
/// sticky fault owner can be written. `live` describes committed KV prefix at
/// preflight and candidate prefix at acceptance. Logical integers are not
/// reinterpreted as floats. Every padding word remains exactly zero.
#[cube(launch)]
#[allow(clippy::manual_range_contains)]
fn private_guard(
    values: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    logical: usize,
    live: usize,
    #[comptime] floats: bool,
) {
    let i = ABSOLUTE_POS;
    if i < values.len() {
        let bits = values[i];
        if i >= logical || i >= live {
            if bits != 0 {
                fault[0].fetch_or(4);
            }
        } else if comptime!(floats) {
            let value = f32::reinterpret(bits);
            if !(value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32) {
                fault[0].fetch_or(1);
            }
        }
    }
}
/// Same complete public bytes/tail semantics as host validation. Packed half
/// scales may be finite subnormals/negative/zero; all exponent31 are rejected.
#[cube(launch)]
#[allow(clippy::manual_range_contains, clippy::collapsible_if)] // Keep compile-time dtype admission outside runtime bit predicates.
fn public_guard(
    values: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    logical_bytes: usize,
    #[comptime] dtype: u32,
) {
    let i = ABSOLUTE_POS;
    if i < values.len() {
        let word = values[i];
        if i * 4 >= logical_bytes {
            if word != 0 {
                fault[0].fetch_or(4);
            }
        } else if comptime!(dtype == PUBLIC_ENGLISH_IDS) {
            if logical_bytes != english::ENGLISH_BYTES || word >= k::VOCAB as u32 {
                fault[0].fetch_or(1);
            }
            if i > 0 {
                if values[i - 1] >= word {
                    fault[0].fetch_or(1);
                }
            }
        } else if comptime!(dtype == cjk::MAP_KIND) {
            if logical_bytes != cjk::MAP_BYTES || word >= k::VOCAB as u32 {
                fault[0].fetch_or(1);
            }
        } else if comptime!(dtype == cjk::INVERSE_KIND) {
            if logical_bytes != k::VOCAB * 4 || (word != u32::MAX && word >= cjk::ROWS as u32) {
                fault[0].fetch_or(1);
            }
        } else if comptime!(dtype == quant_matvec::F32) {
            let value = f32::reinterpret(word);
            if !(value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32) {
                fault[0].fetch_or(1);
            }
        } else if comptime!(dtype == quant_matvec::BF16) {
            if word & 32640 == 32640 || (word >> 16) & 32640 == 32640 {
                fault[0].fetch_or(1);
            }
        }
        if comptime!(dtype == quant_matvec::Q8_0) {
            let block = i * 34;
            if block + 34 <= logical_bytes {
                let index = block / 4;
                let shift = (block % 4) * 8;
                let half = (values[index] >> (shift as u32)) & 65535;
                if half & 31744 == 31744 {
                    fault[0].fetch_or(1);
                }
            }
        }
        if comptime!(
            dtype == quant_matvec::Q2_0
                || dtype == quant_matvec::IQ4_XS
                || dtype == quant_matvec::Q6_K
        ) {
            let block_bytes = comptime!(if dtype == quant_matvec::Q2_0 {
                18
            } else if dtype == quant_matvec::IQ4_XS {
                136
            } else {
                210
            });
            let scale_offset = comptime!(if dtype == quant_matvec::Q6_K { 208 } else { 0 });
            // Bound the block index BEFORE multiplication. Full owned page
            // sweeps also visit padding words, not only packed block count.
            if i < logical_bytes / block_bytes {
                let block = i * block_bytes;
                let at = block + scale_offset;
                let half = (values[at / 4] >> ((at % 4) * 8) as u32) & 65535;
                if half & 31744 == 31744 {
                    fault[0].fetch_or(1);
                }
            }
        }
    }
}

/// Same-stream copy of the actual last committed GPU carry/token. Complete
/// source tails/fault and finite/token decisions remain sticky. Destinations
/// are complete prevalidated owners, and their full padding is overwritten0.
/// Partial candidate stores on a fault are never a committed proposal.
#[cube(launch)]
#[allow(clippy::manual_range_contains, clippy::collapsible_if)]
fn copy_committed(
    hidden: &Array<u32>,
    greedy: &Array<u32>,
    carry: &mut Array<u32>,
    token: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
) {
    let i = ABSOLUTE_POS;
    // Keep the actual bounds branch outside the atomic load in generated GPU
    // code; a source Boolean combination need not lower with lazy evaluation.
    if i < fault.len() {
        if fault[i].load() != 0 {
            fault[0].fetch_or(4);
        }
    }
    if i < hidden.len() {
        let bits = hidden[i];
        if i < k::WIDE {
            let value = f32::reinterpret(bits);
            if value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32 {
                if fault[0].load() == 0 {
                    carry[i] = bits;
                }
            } else {
                fault[0].fetch_or(1);
            }
        } else if bits != 0 {
            fault[0].fetch_or(4);
        }
    }
    if i < carry.len() && i >= k::WIDE {
        carry[i] = 0;
    }
    if i < greedy.len() {
        let value = greedy[i];
        if i == 0 {
            if value < k::VOCAB as u32 {
                if fault[0].load() == 0 {
                    token[0] = value;
                }
            } else {
                fault[0].fetch_or(1);
            }
        } else if value != 0 {
            fault[0].fetch_or(4);
        }
    }
    if i < token.len() && i > 0 {
        token[i] = 0;
    }
}

/// RawQuery has completed every consumer by this phase. Caller clears its
/// COMPLETE64KiB owner, copies actual carry+greedy, then seals fault in a
/// separate same-stream dispatch AFTER every producer/check has completed.
#[cube(launch)]
#[allow(clippy::manual_range_contains)]
#[allow(clippy::collapsible_if)] // Keep comptime specialization outside device control flow.
fn pack_draft(
    hidden: &Array<u32>,
    greedy: &Array<u32>,
    packet: &mut Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    confidence: &Array<u32>,
    #[comptime] with_probability: bool,
) {
    let i = ABSOLUTE_POS;
    if with_probability {
        if i == DRAFT_PROBABILITY {
            let bits = confidence[64];
            let p = f32::reinterpret(bits);
            if p >= 0.0 && p <= 1.0 {
                packet[DRAFT_PROBABILITY] = bits;
            } else {
                fault[0].fetch_or(1);
            }
        }
    }
    if i < k::WIDE {
        let bits = hidden[i];
        let value = f32::reinterpret(bits);
        if value >= -3.4028235e38_f32 && value <= 3.4028235e38_f32 {
            packet[i] = bits;
        } else {
            fault[0].fetch_or(1);
        }
    }
    if i == k::WIDE {
        let token = greedy[0];
        if token < k::VOCAB as u32 {
            packet[DRAFT_TOKEN] = token;
        } else {
            fault[0].fetch_or(1);
        }
    }
}
#[cube(launch)]
fn seal_draft(packet: &mut Array<u32>, fault: &Array<Atomic<u32>>) {
    if UNIT_POS_X == 0 {
        packet[DRAFT_FAULT] = fault[0].load();
    }
}

fn grid(words: usize) -> Result<CubeCount> {
    let l = BufferLayout::new(words).map_err(error)?;
    Ok(CubeCount::Static(l.workgroups_x, l.workgroups_y, 1))
}
fn private_plan() -> Result<u64> {
    WORDS.into_iter().try_fold(
        window::EXTRA_BYTES
            + crate::strata_mtp_front::Workspace::bytes()?
            + if crate::strata_intel_core::enabled() {
                32768
            } else {
                0
            },
        |sum, w| {
            sum.checked_add(BufferLayout::new(w).map_err(error)?.owned_bytes as u64)
                .ok_or(Error::ArithmeticOverflow)
        },
    )
}
/// The same full owning geometry is checked at original arena close and by
/// the final profile validator. No reduced readback/view acquires this credit.
pub(crate) fn private_owned_geometry() -> Result<(u64, u64)> {
    Ok((
        N as u64
            + u64::from(crate::strata_port::enabled())
            + u64::from(crate::strata_intel_core::enabled())
            + if crate::strata_intel_core::enabled() {
                crate::strata_mtp_front::WORDS.len() as u64
            } else {
                0
            },
        private_plan()?,
    ))
}
fn tile_plan(spec: Spec) -> Result<Vec<(usize, usize, usize, usize)>> {
    let row_bytes = quant_matvec::packed_row_bytes(spec.columns, spec.dtype);
    (0..spec.rows)
        .step_by(TILE_ROWS)
        .map(|first| {
            let rows = (spec.rows - first).min(TILE_ROWS);
            let logical = rows
                .checked_mul(row_bytes)
                .ok_or(Error::ArithmeticOverflow)?;
            let owned = if crate::strata_intel_core::enabled()
                && crate::strata_shared_weights::Tensor::named(spec.name)
                    .is_some_and(|t| t.dtype() == spec.dtype)
            {
                (logical + crate::strata_shared_weights::ALIGNMENT as usize - 1)
                    & !(crate::strata_shared_weights::ALIGNMENT as usize - 1)
            } else {
                main_admission::page_bytes(logical as u64).map_err(error)? as usize
            };
            Ok((first, rows, logical, owned))
        })
        .collect()
}
fn public_plan() -> Result<u64> {
    public_plan_for(MtpProfile::SelfContainedQ8)
}
fn matrices_for(profile: MtpProfile) -> &'static [Spec] {
    match profile {
        MtpProfile::SelfContainedQ8 => &MATRICES,
        MtpProfile::StrataQ2 if crate::strata_port::enabled() => &strata::MATRICES_RUNTIME_Q8,
        MtpProfile::StrataQ2 => &strata::MATRICES_Q2,
    }
}
fn head_up_for(profile: MtpProfile) -> Spec {
    match profile {
        MtpProfile::SelfContainedQ8 => HEAD_UP,
        MtpProfile::StrataQ2 if crate::strata_intel_core::enabled() => strata::HEAD_UP_Q2,
        MtpProfile::StrataQ2 if crate::strata_port::enabled() => HEAD_UP,
        MtpProfile::StrataQ2 => strata::HEAD_UP_Q2,
    }
}
fn embedding_for(profile: MtpProfile) -> Spec {
    match profile {
        MtpProfile::SelfContainedQ8 => EMBEDDING,
        MtpProfile::StrataQ2 => strata::EMBEDDING_Q2,
    }
}
fn vectors_for(profile: MtpProfile) -> &'static [(&'static str, usize)] {
    match profile {
        MtpProfile::SelfContainedQ8 => &VECTORS,
        MtpProfile::StrataQ2 => &strata::VECTORS_Q2,
    }
}
fn public_bytes_for(profile: MtpProfile) -> u64 {
    match profile {
        MtpProfile::SelfContainedQ8 => PUBLIC_BYTES,
        MtpProfile::StrataQ2 if crate::strata_port::enabled() => strata_runtime_public_bytes(),
        MtpProfile::StrataQ2 => STRATA_PUBLIC_BYTES,
    }
}
pub(crate) fn strata_runtime_public_bytes() -> u64 {
    public_plan_for(MtpProfile::StrataQ2).expect("static Strata runtime public plan")
}
fn public_bytes_with_english(profile: MtpProfile, english: bool) -> Result<u64> {
    if english && profile != MtpProfile::SelfContainedQ8 && !crate::strata_port::enabled() {
        return Err(Error::InvalidFormat(
            "MTP English vocabulary requires original Q8 head",
        ));
    }
    Ok(public_bytes_for(profile)
        + if english {
            ENGLISH_VOCAB_OWNED_BYTES
        } else {
            0
        })
}
fn public_plan_with_english(profile: MtpProfile, english: bool) -> Result<u64> {
    public_bytes_with_english(profile, english)?;
    Ok(public_plan_for(profile)?
        + if english {
            main_admission::page_bytes(english::ENGLISH_BYTES as u64).map_err(error)?
        } else {
            0
        })
}
fn validate_full_output(english: bool) -> Result<()> {
    if english {
        return Err(Error::InvalidFormat(
            "MTP public English vocabulary supports draft output only",
        ));
    }
    Ok(())
}
fn validate_english_broadcast(english: bool, broadcast: bool) -> Result<()> {
    if english && broadcast {
        return Err(Error::InvalidFormat(
            "MTP public English vocabulary excludes broadcast",
        ));
    }
    Ok(())
}
fn completed_english_policy(
    selected: bool,
    completed: bool,
    closed: bool,
    poisoned: bool,
    retired: bool,
) -> Result<bool> {
    if closed || poisoned || retired {
        return Err(Error::InvalidFormat(
            "MTP completed English vocabulary unavailable",
        ));
    }
    Ok(selected && completed)
}
fn validate_english_readback(bytes: &[u8]) -> Result<()> {
    if bytes.len() != english::ENGLISH_OWNED_BYTES {
        return Err(Error::InvalidBufferLength);
    }
    english::EnglishVocabulary::validate_exact(&bytes[..english::ENGLISH_BYTES])?;
    if bytes[english::ENGLISH_BYTES..]
        .iter()
        .any(|&value| value != 0)
    {
        return Err(Error::InvalidFormat("MTP English public map padding"));
    }
    Ok(())
}
fn validate_english_owner(
    profile: MtpProfile,
    english: bool,
    public_bytes: u64,
    map: Option<(usize, usize, usize, usize, u64)>,
) -> Result<()> {
    let expected = english.then_some((
        english::ENGLISH_BYTES,
        english::ENGLISH_OWNED_BYTES,
        0,
        english::ENGLISH_ROWS,
        ENGLISH_VOCAB_OWNED_BYTES,
    ));
    if public_bytes != public_bytes_with_english(profile, english)? || map != expected {
        return Err(Error::InvalidFormat("MTP immutable English public owner"));
    }
    Ok(())
}
fn validate_profile(actual: MtpProfile, expected: MtpProfile, target_bound: bool) -> Result<()> {
    if actual != expected || expected == MtpProfile::StrataQ2 && !target_bound {
        return Err(Error::InvalidFormat(
            "MTP resident constructor profile mismatch",
        ));
    }
    Ok(())
}
fn public_plan_for(profile: MtpProfile) -> Result<u64> {
    let mut bytes = match profile {
        MtpProfile::SelfContainedQ8 => k::expert_owned_bytes()?,
        MtpProfile::StrataQ2 => strata::expert_owned_bytes()?,
    };
    for spec in matrices_for(profile)
        .iter()
        .copied()
        .chain([head_up_for(profile), embedding_for(profile)])
    {
        for (_, _, _, owned) in tile_plan(spec)? {
            bytes += owned as u64;
        }
    }
    for &(_, words) in vectors_for(profile) {
        bytes += main_admission::page_bytes((words * 4) as u64).map_err(error)?;
    }
    bytes += 2 * main_admission::page_bytes((crate::qsa_gpu::ROPE_TABLE_WORDS * 4) as u64)
        .map_err(error)?;
    if cjk::selected(profile) {
        bytes += cjk::public_bytes();
    }
    Ok(bytes)
}
fn validate_source(bytes: &[u8], dtype: u32) -> Result<()> {
    match dtype {
        quant_matvec::Q8_0 if bytes.len().is_multiple_of(34) => {
            if bytes
                .chunks_exact(34)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7c00 == 0x7c00)
            {
                return Err(Error::InvalidFormat("MTP nonfinite original Q8 source"));
            }
        }
        quant_matvec::F32 if bytes.len().is_multiple_of(4) => {
            if bytes
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes([b[0], b[1], b[2], b[3]]).is_finite())
            {
                return Err(Error::InvalidFormat("MTP nonfinite original F32 source"));
            }
        }
        quant_matvec::BF16 if bytes.len().is_multiple_of(2) => {
            if bytes
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(Error::InvalidFormat("MTP nonfinite original BF16 source"));
            }
        }
        quant_matvec::Q2_0 | quant_matvec::IQ4_XS | quant_matvec::Q6_K => {
            let (block, scale) = match dtype {
                quant_matvec::Q2_0 => (18, 0),
                quant_matvec::IQ4_XS => (136, 0),
                _ => (210, 208),
            };
            if !bytes.len().is_multiple_of(block) {
                return Err(Error::InvalidBufferLength);
            }
            if bytes
                .chunks_exact(block)
                .any(|b| u16::from_le_bytes([b[scale], b[scale + 1]]) & 0x7c00 == 0x7c00)
            {
                return Err(Error::InvalidFormat("MTP nonfinite original packed source"));
            }
        }
        _ => return Err(Error::InvalidBufferLength),
    }
    Ok(())
}

pub fn validate_limits(limits: &wgpu::Limits) -> Result<()> {
    k::validate_limits(limits)?;
    if limits.max_compute_workgroups_per_dimension < 65535 {
        return Err(Error::LimitExceeded("MTP complete public page grid"));
    }
    Ok(())
}
pub fn validate_fused_sweep_limits(limits: &wgpu::Limits) -> Result<()> {
    validate_limits(limits)?;
    // Seven actual storage owners plus CubeCL's length/launch INFO binding.
    if limits.max_storage_buffers_per_shader_stage < 8 {
        return Err(Error::LimitExceeded("MTP fused full-owner sweep bindings"));
    }
    Ok(())
}

/// Native Intel uses split attention, not the old QH*CONTEXT one-dimensional
/// scalar grid. Keep that old validator for its actual legacy consumers. The
/// native frontend/head still require the full common1024-thread/18-binding
/// core, plus the genuine complete-page guard and original public root bounds.
fn validate_native_strata_limits(limits: &wgpu::Limits) -> Result<()> {
    if !crate::strata_intel_core::enabled() {
        return Err(Error::InvalidFormat("native MTP limits require core"));
    }
    crate::strata_port::validate_limits(limits)?;
    if limits.max_compute_workgroups_per_dimension < 65535
        || limits.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || limits.max_buffer_size < 128 * 1024 * 1024
    {
        return Err(Error::LimitExceeded("native MTP complete page limits"));
    }
    Ok(())
}

impl<'g, 'f> ResidentMtp<'g, 'f> {
    /// Caller must initialize required_memory_configuration and enable physical
    /// SUBGROUP16..32 before entering this laboratory-only constructor. This
    /// workspace owns one fresh client/stream; sharing it with a target or
    /// another model is forbidden. Parent configures64MiB transient I/O ONCE
    /// and supplies its physical-context retirement flag, including on fresh
    /// workspace replay. No context/device/controller creation happens here.
    pub fn new(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::construct(
            graph,
            client,
            limits,
            retired,
            false,
            MtpProfile::SelfContainedQ8,
            false,
        )
    }
    /// Separate fixed-public performance candidate. The mode is immutable for
    /// this workspace, and ordinary `new` retains the accepted original path.
    /// Ownership, source admission and conservative512 INFO plan are identical.
    pub fn new_with_fused_sweeps(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::construct(
            graph,
            client,
            limits,
            retired,
            true,
            MtpProfile::SelfContainedQ8,
            false,
        )
    }
    /// Explicit immutable English/code proposal subset of the full original
    /// Q8 head. Full42 private owners and every original public weight page
    /// remain resident; only one pinned public ID owner is added. The full
    /// Output API and broadcast policy are rejected before any mutation.
    pub fn new_public_english_with_fused_sweeps(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::construct(
            graph,
            client,
            limits,
            retired,
            true,
            if crate::strata_port::enabled() {
                graph.profile()
            } else {
                MtpProfile::SelfContainedQ8
            },
            true,
        )
    }
    /// Explicit original Strata GGUF profile. Requires the already-bound paired
    /// target; no dense repacking, vocabulary subset or CPU expert fallback.
    pub fn new_strata_q2(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::construct(
            graph,
            client,
            limits,
            retired,
            false,
            MtpProfile::StrataQ2,
            false,
        )
    }
    pub fn new_strata_q2_with_fused_sweeps(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        Self::construct(
            graph,
            client,
            limits,
            retired,
            true,
            MtpProfile::StrataQ2,
            false,
        )
    }
    fn construct(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
        fused_sweeps: bool,
        expected_profile: MtpProfile,
        public_english: bool,
    ) -> Result<Self> {
        Self::construct_mode(
            graph,
            client,
            limits,
            retired,
            fused_sweeps,
            expected_profile,
            public_english,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)] // Explicit owning constructor dependencies.
    fn construct_mode(
        graph: &'g MtpGraph<'f>,
        client: ComputeClient<WgpuRuntime>,
        limits: &wgpu::Limits,
        retired: Arc<AtomicBool>,
        fused_sweeps: bool,
        expected_profile: MtpProfile,
        public_english: bool,
        shared: Option<crate::executor::ResidentAdmission>,
    ) -> Result<Self> {
        validate_profile(
            graph.profile(),
            expected_profile,
            graph.target_binding_id().is_some(),
        )?;
        if cjk::selected(expected_profile) && public_english {
            return Err(Error::InvalidFormat(
                "native CJK head excludes legacy Q8 vocabulary policy",
            ));
        }
        let public_bytes = public_bytes_with_english(expected_profile, public_english)?;
        let public_english_vocab = public_english
            .then(english::EnglishVocabulary::embedded)
            .transpose()?;
        graph.verify_unchanged()?;
        if crate::strata_intel_core::enabled() && expected_profile == MtpProfile::StrataQ2 {
            validate_native_strata_limits(limits)?;
        } else if fused_sweeps {
            validate_fused_sweep_limits(limits)?;
        } else {
            validate_limits(limits)?;
        }
        if retired.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("MTP physical context retired"));
        }
        let private = private_plan()?;
        if public_plan_with_english(expected_profile, public_english)? != public_bytes
            || private > PRIVATE_BYTES as u64
            || INFO_BYTES + 3 * 4 * 1024 * 1024 > RUNTIME_BYTES
        {
            return Err(Error::LimitExceeded("MTP fixed resident capacity plan"));
        }
        if shared.is_none() {
            main_admission::check_capacity(
                MAIN_BYTES,
                &[
                    public_bytes,
                    PRIVATE_BYTES as u64,
                    RUNTIME_BYTES,
                    PUBLIC_SOURCE_BYTES as u64,
                ],
            )
            .map_err(error)?;
        } else {
            // The consumed peer permit already priced this exact owner in
            // the target's shared plan. Runtime I/O is priced ONCE there.
            main_admission::check_capacity(
                Self::shared_capacity()?,
                &[
                    public_bytes
                        .checked_sub(crate::strata_shared_weights::addressable_bytes())
                        .ok_or(Error::ArithmeticOverflow)?,
                    PRIVATE_BYTES as u64,
                    PUBLIC_SOURCE_BYTES as u64,
                ],
            )
            .map_err(error)?;
        }
        let usage = client.memory_usage().map_err(error)?;
        if shared.is_none()
            && (usage.number_allocs != 0 || usage.bytes_in_use != 0 || usage.bytes_reserved != 0)
        {
            return Err(Error::InvalidFormat("MTP resident requires fresh client"));
        }
        if let Some(a) = &shared {
            a.check_client(&client)?;
        }
        let shared_target_weights = shared
            .as_ref()
            .map(|a| a.shared_target_weights())
            .transpose()?;

        let owned_client = client.clone();
        let mut arena =
            PrivateArena::new_transient_on_configured(client, PRIVATE_BYTES, retired.clone())
                .map_err(error)?;
        arena.enable_public_owned_zero_audit().map_err(error)?;
        let mut ledger = match shared {
            Some(a) => Ledger::Shared(a),
            None => Ledger::Exclusive(MainAdmission::new(MAIN_BYTES, 0).map_err(error)?),
        };
        ledger.observe_before(usage.bytes_reserved)?;
        let mut plan: Vec<u64> = WORDS.map(|n| (n * 4) as u64).to_vec();
        if crate::strata_port::enabled() {
            plan.push(16);
        }
        if crate::strata_intel_core::enabled() {
            plan.push(32768);
        }
        plan.extend(crate::strata_mtp_front::Workspace::requests());
        ledger.precharge(&plan).map_err(error)?;
        let mut allocated = Vec::with_capacity(N);
        for n in WORDS {
            allocated.push(arena.allocate_zeroed(n).map_err(error)?);
        }
        let ids: [BufferId; N] = allocated
            .try_into()
            .map_err(|_| Error::InvalidBufferLength)?;
        let draft_window = if crate::strata_port::enabled() {
            Some(arena.allocate_zeroed(window::PACKET_WORDS).map_err(error)?)
        } else {
            None
        };
        let dense_planes = if crate::strata_intel_core::enabled() {
            Some(arena.allocate_zeroed(8192).map_err(error)?)
        } else {
            None
        };
        let native_front = crate::strata_mtp_front::Workspace::allocate(&mut arena)?;
        if arena.charged_bytes() as u64 != private {
            return Err(Error::InvalidBufferLength);
        }
        ledger
            .observe_after(sample(arena.client().map_err(error)?)?)
            .map_err(error)?;
        let binding = NEXT_BINDING
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        let mut workspace = Self {
            graph,
            arena,
            ids,
            draft_window,
            dense_planes,
            native_front,
            native_front_pending: Arc::new(AtomicBool::new(false)),
            native_replay: replay::Cache::new()?,
            client: owned_client,
            public: None,
            shared_target_weights,
            ledger,
            retired,
            position: 0,
            poisoned: true,
            closed: false,
            dispatches: 0,
            binding,
            live_outputs: Arc::new(AtomicUsize::new(0)),
            last_forward: None,
            last_front: None,
            traffic: Cell::new(Traffic::default()),
            fused_sweeps,
            q8_head_broadcast: false,
            q8_head_policy_locked: false,
            q8_head_broadcast_completed: false,
            q8_head_shared_scales: false,
            q8_head_shared_scales_completed: false,
            public_english_vocab,
            public_english_vocab_completed: false,
        };
        let result = workspace.load_public();
        match result {
            Ok(public) => workspace.public = Some(public),
            Err(e) => {
                workspace.fail();
                return Err(e);
            }
        }
        workspace.physical()?;
        workspace.guard_private(0)?;
        workspace.complete_fault()?;
        workspace.verify_source()?;
        workspace.poisoned = false;
        Ok(workspace)
    }

    // Retain the same pinned client separately so completed arena close can be
    // followed by physical allocator recovery and sanitizer proof. `arena` is
    // !Send/!Sync and pins this workspace to the construction worker thread.
    fn client(&self) -> Result<&ComputeClient<WgpuRuntime>> {
        Ok(&self.client)
    }
    fn id(&self, b: B) -> BufferId {
        self.ids[b as usize]
    }
    fn a(&self, b: B) -> Result<ArrayArg<WgpuRuntime>> {
        u::arg(&self.arena, self.id(b))
    }
    fn full(&self, b: B) -> Result<ArrayArg<WgpuRuntime>> {
        let layout = self.arena.layout(self.id(b)).map_err(error)?;
        u::raw_arg(
            self.arena.handle(self.id(b)).map_err(error)?.clone(),
            layout.owned_words,
        )
    }
    fn prefix(&self, b: B, first: usize, words: usize) -> Result<ArrayArg<WgpuRuntime>> {
        if first.checked_add(words).ok_or(Error::ArithmeticOverflow)? > WORDS[b as usize] {
            return Err(Error::InvalidBufferLength);
        }
        let mut h = u::handle(&self.arena, self.id(b))?;
        h = h
            .offset_start((first * 4) as u64)
            .offset_end(((WORDS[b as usize] - first - words) * 4) as u64);
        u::raw_arg(h, words)
    }
    fn public_arg(page: &Page) -> Result<ArrayArg<WgpuRuntime>> {
        u::raw_arg(page.handle.clone(), page.logical_bytes / 4)
    }
    fn tick(&mut self, count: usize) -> Result<()> {
        self.dispatches = self
            .dispatches
            .checked_add(count)
            .ok_or(Error::ArithmeticOverflow)?;
        if self.dispatches > MAX_TOKEN_DISPATCHES {
            return Err(Error::LimitExceeded("MTP token INFO records"));
        }
        self.count_traffic(|traffic| {
            traffic.explicit_dispatches = traffic
                .explicit_dispatches
                .checked_add(count as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        Ok(())
    }
    fn count_traffic(&self, update: impl FnOnce(&mut Traffic) -> Result<()>) -> Result<()> {
        let mut next = self.traffic.get();
        update(&mut next)?;
        self.traffic.set(next);
        Ok(())
    }
    fn verify_source(&self) -> Result<()> {
        // This zero-length range still validates the pinned descriptor and all
        // source stat identities before/after; no model payload read occurs.
        self.graph.verify_unchanged()
    }
    fn physical(&self) -> Result<()> {
        let mut owners: Vec<Handle> = self
            .ids
            .iter()
            .map(|&id| self.arena.handle(id).cloned().map_err(error))
            .collect::<Result<_>>()?;
        if let Some(id) = self.draft_window {
            owners.push(self.arena.handle(id).cloned().map_err(error)?);
        }
        if let Some(id) = self.dense_planes {
            owners.push(self.arena.handle(id).cloned().map_err(error)?);
        }
        if let Some(front) = &self.native_front {
            owners.extend(front.handles(&self.arena)?);
        }
        if let Some(public) = &self.public {
            if public.cjk.is_some() != cjk::selected(self.graph.profile()) {
                return Err(Error::InvalidFormat("native CJK owner selection"));
            }
            if let Some(head) = &public.cjk {
                head.validate()?;
            }
            validate_english_owner(
                self.graph.profile(),
                self.public_english_vocab.is_some(),
                public.bytes,
                public.english_ids.as_ref().map(|page| {
                    (
                        page.logical_bytes,
                        page.owned_bytes,
                        page.first,
                        page.rows,
                        page.handle.size(),
                    )
                }),
            )?;
            owners.extend(public.handles());
        }
        crate::resident_gr::batch::physical(&self.arena, &owners)?;
        let stream = self.arena.handle(self.id(B::Fault)).map_err(error)?.stream;
        if owners.iter().any(|h| h.stream != stream) {
            return Err(Error::InvalidBufferLength);
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)] // Original tensor span + complete page geometry.
    fn load_page(
        &mut self,
        name: &'static str,
        offset: usize,
        logical: usize,
        owned: usize,
        first: usize,
        rows: usize,
        dtype: u32,
    ) -> Result<Page> {
        if logical == 0
            || !logical.is_multiple_of(4)
            || owned > PUBLIC_SOURCE_BYTES
            || (main_admission::page_bytes(logical as u64).map_err(error)? as usize != owned
                && !(crate::strata_intel_core::enabled()
                    && crate::strata_shared_weights::Tensor::named(name)
                        .is_some_and(|t| t.dtype() == dtype)
                    && owned == (logical + 4095) & !4095))
        {
            return Err(Error::InvalidBufferLength);
        }
        let mut source = Upload(vec![0; owned]);
        if source.0.capacity() > PUBLIC_SOURCE_BYTES {
            return Err(Error::LimitExceeded("MTP public source capacity"));
        }
        let tensor = self.graph.tensor(name)?;
        let expected = TensorType::from_gguf_id(dtype)?;
        if crate::strata_port::enabled()
            && self.graph.profile() == MtpProfile::StrataQ2
            && dtype == quant_matvec::Q8_0
            && tensor.info().dtype() != expected
        {
            // Strata runtime conversion. Borrow original GGUF rows, convert in
            // RAM, and upload once. No converted weight artifact is persisted.
            let columns = tensor.info().dimensions()[0] as usize;
            if !columns.is_multiple_of(32) || logical != rows * (columns / 32) * 34 {
                return Err(Error::InvalidBufferLength);
            }
            let mut packed = Zeroizing::new(vec![0u8; tensor.info().row_bytes() as usize]);
            let mut decoded = Zeroizing::new(vec![0f32; columns]);
            for row in 0..rows {
                tensor.read_row((first + row) as u64, &mut packed)?;
                tensor.info().dtype().decode(&packed, &mut decoded)?;
                for (block, values) in decoded.chunks_exact(32).enumerate() {
                    if values.iter().any(|x| !x.is_finite()) {
                        return Err(Error::InvalidFormat(
                            "Strata public runtime conversion nonfinite",
                        ));
                    }
                    let scale = values.iter().map(|x| x.abs()).fold(0f32, f32::max) / 127.;
                    let at = (row * (columns / 32) + block) * 34;
                    let half = half::f16::from_f32(scale);
                    if !half.is_finite() {
                        return Err(Error::InvalidFormat("Strata Q8 weight scale"));
                    }
                    source.0[at..at + 2].copy_from_slice(&half.to_bits().to_le_bytes());
                    for (i, &x) in values.iter().enumerate() {
                        source.0[at + 2 + i] = if scale == 0. {
                            0
                        } else {
                            (x / scale).round().clamp(-127., 127.) as i8 as u8
                        };
                    }
                }
            }
            validate_source(&source.0[..logical], dtype)?;
            return self.upload_page(source, logical, owned, first, rows, dtype);
        }
        if tensor.info().dtype() != expected {
            return Err(Error::InvalidBufferLength);
        }
        for first_chunk in (0..logical).step_by(READ_CHUNK) {
            let end = (first_chunk + READ_CHUNK).min(logical);
            tensor.read_range(
                (offset + first_chunk) as u64,
                &mut source.0[first_chunk..end],
            )?;
        }
        validate_source(&source.0[..logical], dtype)?;
        self.upload_page(source, logical, owned, first, rows, dtype)
    }
    fn upload_page(
        &mut self,
        mut source: Upload,
        logical: usize,
        owned: usize,
        first: usize,
        rows: usize,
        dtype: u32,
    ) -> Result<Page> {
        if source.0.len() != owned || source.0.capacity() > PUBLIC_SOURCE_BYTES {
            return Err(Error::InvalidBufferLength);
        }
        self.ledger
            .observe_before(sample(self.client()?)?)
            .map_err(error)?;
        // The bound accounts both the persistent full page and the sole public
        // upload source. Credit only follows completed device read + sanitizer.
        self.ledger
            .precharge(&[owned as u64, owned as u64])
            .map_err(error)?;
        let data = source.take();
        let handle = if crate::strata_intel_core::enabled()
            && main_admission::page_bytes(logical as u64).map_err(error)? as usize != owned
        {
            crate::expert_banks::exact_persistent::persistent_scope(
                |mode| {
                    unsafe { self.client()?.allocation_mode(mode) };
                    Ok(())
                },
                &self.retired,
                || Ok(self.client()?.create_public_owned(data)),
            )?
        } else {
            self.client()?.create_public_owned(data)
        };
        if handle.size() != owned as u64 {
            return Err(Error::InvalidBufferLength);
        }
        self.ledger
            .observe_after(sample(self.client()?)?)
            .map_err(error)?;
        let page = Page {
            handle,
            logical_bytes: logical,
            owned_bytes: owned,
            first,
            rows,
        };
        // Complete original public scales/F32 and exact zero padding are read
        // by device before permitting any private consumption of these bytes.
        public_guard::launch::<WgpuRuntime>(
            self.client()?,
            grid(owned / 4)?,
            CubeDim::new_1d(64),
            u::raw_arg(page.handle.clone(), owned / 4)?,
            self.a(B::Fault)?,
            logical,
            dtype,
        );
        self.complete_fault()?;
        Ok(page)
    }
    fn load_public_english_ids(&mut self, proof: english::EnglishVocabulary) -> Result<Page> {
        let owned = english::ENGLISH_OWNED_BYTES;
        if main_admission::page_bytes(english::ENGLISH_BYTES as u64).map_err(error)? as usize
            != owned
            || owned > PUBLIC_SOURCE_BYTES
            || owned > HOST_BYTES
        {
            return Err(Error::LimitExceeded("MTP English public map source plan"));
        }
        let mut source = Upload(vec![0; owned]);
        source.0[..english::ENGLISH_BYTES].copy_from_slice(proof.bytes());
        english::EnglishVocabulary::validate_exact(&source.0[..english::ENGLISH_BYTES])?;
        let page = self.upload_page(
            source,
            english::ENGLISH_BYTES,
            owned,
            0,
            english::ENGLISH_ROWS,
            PUBLIC_ENGLISH_IDS,
        )?;
        // Public construction readback proves the exact immutable uploaded ID
        // bytes as well as every padding byte. No private request row is read.
        let read = self
            .client()?
            .read_one(page.handle.clone())
            .map_err(error)?;
        let bytes = Bytes(read);
        self.count_traffic(|traffic| {
            traffic.host_read_calls = traffic
                .host_read_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            traffic.host_read_bytes = traffic
                .host_read_bytes
                .checked_add(owned as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        validate_english_readback(&bytes.0)?;
        drop(bytes);
        self.audit_io()?;
        self.ledger
            .completed_fence(sample(self.client()?)?)
            .map_err(error)?;
        Ok(page)
    }
    fn load_public(&mut self) -> Result<Public> {
        let mut pages = BTreeMap::new();
        let mut bytes = 0_u64;
        let profile = self.graph.profile();
        for spec in matrices_for(profile)
            .iter()
            .copied()
            .chain([head_up_for(profile), embedding_for(profile)])
        {
            let tensor = self.graph.tensor(spec.name)?;
            if tensor.info().dimensions() != [spec.columns as u64, spec.rows as u64]
                || (!crate::strata_port::enabled()
                    || profile != MtpProfile::StrataQ2
                    || spec.dtype != quant_matvec::Q8_0)
                    && tensor.info().byte_len() as usize
                        != spec.rows * quant_matvec::packed_row_bytes(spec.columns, spec.dtype)
            {
                return Err(Error::InvalidBufferLength);
            }
            let mut list = Vec::new();
            let row_bytes = tensor.info().row_bytes() as usize;
            if let Some(shared) = &self.shared_target_weights
                && let Some(role) = crate::strata_shared_weights::Tensor::named(spec.name)
            {
                if spec.dtype != role.dtype() {
                    return Err(Error::InvalidBufferLength);
                }
                for p in shared.pages(role) {
                    list.push(Page {
                        handle: p.handle.clone(),
                        logical_bytes: p.tile.logical,
                        owned_bytes: p.tile.owned,
                        first: p.tile.first,
                        rows: p.tile.rows,
                    });
                    bytes += p.tile.owned as u64;
                }
                pages.insert(spec.name, list);
                continue;
            }
            for (first, rows, logical, owned) in tile_plan(spec)? {
                list.push(self.load_page(
                    spec.name,
                    first * row_bytes,
                    logical,
                    owned,
                    first,
                    rows,
                    spec.dtype,
                )?);
                bytes += owned as u64;
            }
            pages.insert(spec.name, list);
        }
        if profile == MtpProfile::StrataQ2 {
            for (name, gate_up) in [(strata::GU_NAME, true), (strata::DOWN_NAME, false)] {
                let expert_bytes = if gate_up {
                    strata::GU_EXPERT_BYTES
                } else {
                    strata::DOWN_EXPERT_BYTES
                };
                let tensor = self.graph.tensor(name)?;
                if tensor.info().dtype() != TensorType::Q2_0
                    || tensor.info().byte_len() != (k::EXPERTS * expert_bytes) as u64
                {
                    return Err(Error::InvalidBufferLength);
                }
                let mut list = Vec::with_capacity(strata::SHARDS);
                for index in 0..strata::SHARDS {
                    let shard = strata::Shard::new(index)?;
                    let owned = shard.owned(gate_up)?;
                    list.push(self.load_page(
                        name,
                        shard.first * expert_bytes,
                        shard.bytes(gate_up)?,
                        owned,
                        shard.first,
                        strata::EXPERTS_PER_SHARD,
                        quant_matvec::Q2_0,
                    )?);
                    bytes += owned as u64;
                }
                pages.insert(name, list);
            }
        } else {
            for name in EXPERT_NAMES {
                if self.graph.tensor(name)?.info().byte_len()
                    != (k::EXPERTS * k::EXPERT_BYTES) as u64
                {
                    return Err(Error::InvalidBufferLength);
                }
                let mut list = Vec::with_capacity(k::SHARDS);
                for index in 0..k::SHARDS {
                    let shard = k::Shard::new(index)?;
                    let logical = shard.packed_bytes()?;
                    let owned = shard.owned_bytes()? as usize;
                    list.push(self.load_page(
                        name,
                        shard.first * k::EXPERT_BYTES,
                        logical,
                        owned,
                        shard.first,
                        shard.experts,
                        quant_matvec::Q8_0,
                    )?);
                    bytes += owned as u64;
                }
                pages.insert(name, list);
            }
        }
        for &(name, n) in vectors_for(profile) {
            let owned = main_admission::page_bytes((n * 4) as u64).map_err(error)? as usize;
            if profile == MtpProfile::StrataQ2 {
                let values = self.graph.static_vector(name)?;
                if values.len() != n {
                    return Err(Error::InvalidBufferLength);
                }
                let mut source = Upload(vec![0; owned]);
                for (word, value) in source.0.chunks_exact_mut(4).zip(values) {
                    word.copy_from_slice(&value.to_le_bytes());
                }
                validate_source(&source.0[..n * 4], quant_matvec::F32)?;
                self.verify_source()?;
                pages.insert(
                    name,
                    vec![self.upload_page(source, n * 4, owned, 0, 1, quant_matvec::F32)?],
                );
            } else {
                pages.insert(
                    name,
                    vec![self.load_page(name, 0, n * 4, owned, 0, 1, quant_matvec::F32)?],
                );
            }
            bytes += owned as u64;
        }
        let (sin, cos) = k::rope_tables();
        let mut upload_rope = |values: &[f32]| -> Result<Page> {
            let owned =
                main_admission::page_bytes((values.len() * 4) as u64).map_err(error)? as usize;
            let mut source = Upload(vec![0; owned]);
            for (word, value) in source.0.chunks_exact_mut(4).zip(values) {
                word.copy_from_slice(&value.to_le_bytes());
            }
            self.upload_page(source, values.len() * 4, owned, 0, 1, quant_matvec::F32)
        };
        let sin = upload_rope(sin)?;
        let cos = upload_rope(cos)?;
        bytes += (sin.owned_bytes + cos.owned_bytes) as u64;
        let english_ids = self
            .public_english_vocab
            .map(|proof| self.load_public_english_ids(proof))
            .transpose()?;
        if let Some(map) = &english_ids {
            bytes += map.owned_bytes as u64;
        }
        let cjk = if cjk::selected(profile) {
            bytes += cjk::public_bytes();
            Some(cjk::Head::load(self)?)
        } else {
            None
        };
        if bytes != public_bytes_with_english(profile, self.public_english_vocab.is_some())?
            || pages.len() != 30
        {
            return Err(Error::InvalidBufferLength);
        }
        Ok(Public {
            pages,
            sin,
            cos,
            english_ids,
            cjk,
            bytes,
        })
    }
    fn guard_private(&mut self, cache_position: usize) -> Result<()> {
        self.guard_private_with_phase(cache_position, None)
    }
    /// Explicit current-cell or after-forward boundary for captured math. The
    /// displacement is caller-declared, never guessed from an observed value.
    fn guard_private_at_native(&mut self, offset: i16) -> Result<()> {
        if !matches!(offset, 0 | 1) {
            return Err(Error::InvalidBufferLength);
        }
        let position = self
            .position
            .checked_add(offset as usize)
            .ok_or(Error::ArithmeticOverflow)?;
        self.guard_private_with_phase(position, Some(offset))
    }
    fn guard_private_with_phase(
        &mut self,
        cache_position: usize,
        phase: Option<i16>,
    ) -> Result<()> {
        if cache_position > k::CONTEXT {
            return Err(Error::OutOfBounds);
        }
        if self.fused_sweeps {
            self.guard_private_fused(cache_position)?;
            return self.guard_packed_cache(cache_position, phase);
        }
        for (index, &logical) in WORDS.iter().enumerate() {
            let b = self.ids[index];
            let layout = self.arena.layout(b).map_err(error)?;
            let live = if index == B::Keys as usize || index == B::Values as usize {
                if crate::strata_intel_core::enabled() {
                    WORDS[index]
                } else {
                    cache_position * k::KV
                }
            } else if index == B::Fault as usize {
                0
            } else {
                WORDS[index]
            };
            private_guard::launch::<WgpuRuntime>(
                self.client()?,
                grid(layout.owned_words)?,
                CubeDim::new_1d(64),
                u::raw_arg(
                    self.arena.handle(b).map_err(error)?.clone(),
                    layout.owned_words,
                )?,
                self.a(B::Fault)?,
                logical,
                live,
                !integer_buffer(index),
            );
        }
        self.tick(N)?;
        self.guard_packed_cache(cache_position, phase)
    }
    fn guard_packed_cache(&mut self, position: usize, phase: Option<i16>) -> Result<()> {
        if !crate::strata_intel_core::enabled() {
            return Ok(());
        }
        crate::strata_mtp_front::guard(
            &self.arena,
            [k::CACHE_STORAGE_WORDS.div_ceil(64), 1, 1],
            vec![
                u::handle(&self.arena, self.id(B::Keys))?,
                u::handle(&self.arena, self.id(B::Values))?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            position,
            phase,
        )?;
        self.tick(1)
    }
    fn rewind_cache(&mut self, old: usize, new: usize) -> Result<()> {
        if crate::strata_intel_core::enabled() {
            crate::strata_mtp_front::launch(
                &self.arena,
                crate::strata_mtp_front::Kernel::Rewind,
                [k::CACHE_STORAGE_WORDS.div_ceil(64), 1, 1],
                vec![
                    u::handle(&self.arena, self.id(B::Keys))?,
                    u::handle(&self.arena, self.id(B::Values))?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                ],
                [1, new as u32, 0, old as u32, 0, 0, 0, 0],
            )?;
        } else {
            k::rewind_kv::launch::<WgpuRuntime>(
                self.client()?,
                u::grid(k::CACHE_WORDS)?,
                CubeDim::new_1d(64),
                self.a(B::Keys)?,
                self.a(B::Values)?,
                self.a(B::Fault)?,
                old,
                new,
            );
        }
        self.tick(1)
    }
    fn validate_sweep_roles(&self, roles: &[usize]) -> Result<()> {
        if N != 42 || roles.is_empty() || roles.len() > N {
            return Err(Error::InvalidBufferLength);
        }
        // The original physical() before the stage still proves actual full
        // backing extents and every private/public pair disjoint. These checks
        // bind each batched full view to the exact original public role before
        // the FIRST sweep enqueue; no dummy or aliased padding input is used.
        for (i, &index) in roles.iter().enumerate() {
            if index >= N || roles[..i].contains(&index) {
                return Err(Error::InvalidBufferLength);
            }
            let expected = BufferLayout::new(WORDS[index]).map_err(error)?;
            if self.arena.layout(self.ids[index]).map_err(error)? != expected
                || self.arena.handle(self.ids[index]).map_err(error)?.size()
                    != expected.owned_bytes as u64
            {
                return Err(Error::InvalidBufferLength);
            }
        }
        Ok(())
    }
    fn sweep_arg(&self, index: usize) -> Result<ArrayArg<WgpuRuntime>> {
        let layout = self.arena.layout(self.ids[index]).map_err(error)?;
        u::raw_arg(
            self.arena.handle(self.ids[index]).map_err(error)?.clone(),
            layout.owned_words,
        )
    }
    fn sweep_grid(&self, roles: &[usize]) -> Result<CubeCount> {
        let mut words = 0;
        for &index in roles {
            words = words.max(
                self.arena
                    .layout(self.ids[index])
                    .map_err(error)?
                    .owned_words,
            );
        }
        grid(words)
    }
    fn guard_private_fused(&mut self, cache_position: usize) -> Result<()> {
        // Packed native caches have INTEGER roles; no fused-sweep predicate
        // reads this scalar. Keep its unused metadata constant. The separate
        // packed guard below still checks the actual current/closing boundary.
        let cache_position = if crate::strata_intel_core::enabled() {
            if sweep_role(B::Keys as usize) != sweeps::INTEGER
                || sweep_role(B::Values as usize) != sweeps::INTEGER
            {
                return Err(Error::InvalidFormat("MTP packed cache sweep roles changed"));
            }
            0
        } else {
            cache_position
        };
        // Fault is last in the original42-role registry. It gets a dedicated
        // atomic full scan AFTER these41 role guards, without u32/atomic alias.
        let roles: [usize; 41] = std::array::from_fn(|i| i);
        self.validate_sweep_roles(&roles)?;
        self.validate_sweep_roles(&[B::Fault as usize])?;
        for group in roles.chunks(6) {
            match *group {
                [a, b, c, d, e, f] => sweeps::guard6::launch::<WgpuRuntime>(
                    self.client()?,
                    self.sweep_grid(group)?,
                    CubeDim::new_1d(64),
                    self.sweep_arg(a)?,
                    self.sweep_arg(b)?,
                    self.sweep_arg(c)?,
                    self.sweep_arg(d)?,
                    self.sweep_arg(e)?,
                    self.sweep_arg(f)?,
                    self.full(B::Fault)?,
                    cache_position,
                    WORDS[a],
                    sweep_role(a),
                    WORDS[b],
                    sweep_role(b),
                    WORDS[c],
                    sweep_role(c),
                    WORDS[d],
                    sweep_role(d),
                    WORDS[e],
                    sweep_role(e),
                    WORDS[f],
                    sweep_role(f),
                ),
                [a, b, c, d, e] => sweeps::guard5::launch::<WgpuRuntime>(
                    self.client()?,
                    self.sweep_grid(group)?,
                    CubeDim::new_1d(64),
                    self.sweep_arg(a)?,
                    self.sweep_arg(b)?,
                    self.sweep_arg(c)?,
                    self.sweep_arg(d)?,
                    self.sweep_arg(e)?,
                    self.full(B::Fault)?,
                    cache_position,
                    WORDS[a],
                    sweep_role(a),
                    WORDS[b],
                    sweep_role(b),
                    WORDS[c],
                    sweep_role(c),
                    WORDS[d],
                    sweep_role(d),
                    WORDS[e],
                    sweep_role(e),
                ),
                _ => return Err(Error::InvalidBufferLength),
            }
            self.tick(1)?;
        }
        sweeps::guard_fault::launch::<WgpuRuntime>(
            self.client()?,
            self.sweep_grid(&[B::Fault as usize])?,
            CubeDim::new_1d(64),
            self.full(B::Fault)?,
        );
        self.tick(1)
    }
    fn check_fault(&self) -> Result<()> {
        let bits = self.read_words(B::Fault)?;
        if bits.iter().any(|&x| x != 0) {
            return Err(Error::InvalidFormat("MTP resident sticky fault"));
        }
        drop(bits);
        Ok(())
    }
    fn complete_fault(&mut self) -> Result<()> {
        self.check_fault()?;
        self.ledger
            .completed_fence(sample(self.client()?)?)
            .map_err(error)
    }
    fn read_words(&self, b: B) -> Result<Zeroizing<Vec<u32>>> {
        self.read_words_id(self.id(b))
    }
    fn read_words_id(&self, id: BufferId) -> Result<Zeroizing<Vec<u32>>> {
        let layout = self.arena.layout(id).map_err(error)?;
        // Simultaneous full Bytes + guarded conversion + largest retained
        // output guards remain inside the explicit16MiB host plan.
        if 2 * layout.owned_bytes + MAX_LIVE_OUTPUTS * OUTPUT_HOST_BYTES + 2 * 1024 * 1024
            > HOST_BYTES
        {
            return Err(Error::LimitExceeded("MTP guarded host capacity"));
        }
        let result = self
            .client()?
            .read_one(self.arena.handle(id).map_err(error)?.clone())
            .map_err(error)?;
        let bytes = Bytes(result);
        self.count_traffic(|traffic| {
            traffic.host_read_calls = traffic
                .host_read_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            traffic.host_read_bytes = traffic
                .host_read_bytes
                .checked_add(layout.owned_bytes as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        if bytes.0.len() != layout.owned_bytes {
            return Err(Error::InvalidBufferLength);
        }
        let mut words = Zeroizing::new(Vec::with_capacity(layout.owned_words));
        words.extend(
            bytes
                .0
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        drop(bytes);
        self.audit_io()?;
        if words.capacity() * 4 > layout.owned_bytes
            || words[layout.logical_words..].iter().any(|&x| x != 0)
        {
            return Err(Error::InvalidFormat("MTP full readback padding"));
        }
        Ok(words)
    }
    fn write(&self, b: B, values: &[u32]) -> Result<()> {
        let layout = self.arena.layout(self.id(b)).map_err(error)?;
        if values.len() != layout.logical_words {
            return Err(Error::InvalidBufferLength);
        }
        let mut upload = Upload(vec![0; layout.owned_bytes]);
        if upload.0.capacity() > HOST_BYTES {
            return Err(Error::LimitExceeded("MTP host private upload"));
        }
        for (out, word) in upload.0.chunks_exact_mut(4).zip(values) {
            out.copy_from_slice(&word.to_le_bytes());
        }
        self.client()?
            .write_private(
                self.arena.handle(self.id(b)).map_err(error)?.clone(),
                upload.take(),
            )
            .map_err(error)?;
        self.count_traffic(|traffic| {
            traffic.private_upload_calls = traffic
                .private_upload_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            traffic.private_upload_bytes = traffic
                .private_upload_bytes
                .checked_add(layout.owned_bytes as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })
    }
    fn clear_scratch(&mut self) -> Result<()> {
        self.clear_scratch_preserving_inputs(false)
    }
    fn clear_scratch_preserving_inputs(&mut self, preserve: bool) -> Result<()> {
        if self.fused_sweeps {
            return self.clear_scratch_fused(preserve);
        }
        let mut count = 0;
        for index in 0..N {
            if index != B::Keys as usize && index != B::Values as usize {
                if preserve
                    && [B::Carry, B::Token, B::Fault]
                        .iter()
                        .any(|b| *b as usize == index)
                {
                    continue;
                }
                u::zero(&self.arena, self.ids[index])?;
                count += 1;
            }
        }
        self.tick(count)
    }
    fn clear_scratch_fused(&mut self, preserve: bool) -> Result<()> {
        let (roles, count) = clear_roles(preserve);
        let roles = &roles[..count];
        self.validate_sweep_roles(roles)?;
        for group in roles.chunks(7) {
            match *group {
                [a, b, c, d, e, f, g] => sweeps::clear7::launch::<WgpuRuntime>(
                    self.client()?,
                    self.sweep_grid(group)?,
                    CubeDim::new_1d(64),
                    self.sweep_arg(a)?,
                    self.sweep_arg(b)?,
                    self.sweep_arg(c)?,
                    self.sweep_arg(d)?,
                    self.sweep_arg(e)?,
                    self.sweep_arg(f)?,
                    self.sweep_arg(g)?,
                ),
                [a, b, c, d, e] => sweeps::clear5::launch::<WgpuRuntime>(
                    self.client()?,
                    self.sweep_grid(group)?,
                    CubeDim::new_1d(64),
                    self.sweep_arg(a)?,
                    self.sweep_arg(b)?,
                    self.sweep_arg(c)?,
                    self.sweep_arg(d)?,
                    self.sweep_arg(e)?,
                ),
                [a, b] => sweeps::clear2::launch::<WgpuRuntime>(
                    self.client()?,
                    self.sweep_grid(group)?,
                    CubeDim::new_1d(64),
                    self.sweep_arg(a)?,
                    self.sweep_arg(b)?,
                ),
                _ => return Err(Error::InvalidBufferLength),
            }
            self.tick(1)?;
        }
        Ok(())
    }
    /// Original complete scratch clears with only Fault retained. Reuse the
    /// existing37-role clear and exact two-input clear; no new owner/shader.
    fn clear_repair_scratch(&mut self) -> Result<()> {
        self.clear_scratch_preserving_inputs(true)?;
        let roles = [B::Token as usize, B::Carry as usize];
        if self.fused_sweeps {
            self.validate_sweep_roles(&roles)?;
            sweeps::clear2::launch::<WgpuRuntime>(
                self.client()?,
                self.sweep_grid(&roles)?,
                CubeDim::new_1d(64),
                self.sweep_arg(roles[0])?,
                self.sweep_arg(roles[1])?,
            );
            self.tick(1)
        } else {
            for index in roles {
                u::zero(&self.arena, self.ids[index])?;
            }
            self.tick(roles.len())
        }
    }
    fn start(&mut self, token: u32, carry: &[f32]) -> Result<()> {
        self.start_at(token, carry, self.position)
    }
    /// Stage a real input at its eventual cell while validating cache contents
    /// at the current completed cursor. Accepted-front catch-up may rewind a
    /// full cache; that cursor is not itself another cell to execute.
    fn start_at(&mut self, token: u32, carry: &[f32], cell: usize) -> Result<()> {
        if self.closed || self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("MTP resident poisoned"));
        }
        validate_input(token, carry, cell)?;
        self.q8_head_policy_locked = true;
        self.verify_source()?;
        self.physical()?;
        self.ledger
            .observe_before(sample(self.client()?)?)
            .map_err(error)?;
        self.poisoned = true;
        self.dispatches = 0;
        self.clear_scratch()?;
        self.write(B::Token, &[token])?;
        let input = Zeroizing::new(carry.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        if input.capacity() * 4 > HOST_BYTES {
            return Err(Error::LimitExceeded("MTP host carry"));
        }
        self.write(B::Carry, &input)?;
        self.guard_private(self.position)
    }
    fn finite(&mut self, b: B) -> Result<()> {
        u::validate(&self.arena, self.id(b), self.id(B::Fault))?;
        self.tick(1)
    }
    fn status(&mut self, n: usize) -> Result<()> {
        u::status_batch(
            &self.arena,
            self.id(B::Status),
            self.id(B::Fault),
            1,
            0,
            n,
            n,
        )?;
        self.tick(1)
    }

    fn fail(&mut self) {
        self.last_forward.take();
        self.last_front.take();
        self.poisoned = true;
        self.position.zeroize();
        // Every private owner, including all candidate/committed KV, is erased
        // before the caller can reuse this sequence. Any error retires context;
        // the tracked arena retains owners for its full close/quarantine path.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            for &id in &self.ids {
                u::zero(&self.arena, id)?;
            }
            if let Some(id) = self.draft_window {
                u::zero(&self.arena, id)?;
            }
            if let Some(id) = self.dense_planes {
                u::zero(&self.arena, id)?;
            }
            if let Some(front) = &self.native_front {
                for &id in &*front.ids {
                    u::zero(&self.arena, id)?;
                }
            }
            cubecl::future::block_on(self.client()?.sync()).map_err(error)?;
            self.audit_io()
        }));
        if !matches!(result, Ok(Ok(()))) {
            self.retired.store(true, Ordering::Release);
            self.ledger.poison();
        }
    }
}

impl ResidentMtp<'_, '_> {
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn binding(&self) -> u64 {
        self.binding
    }
    pub fn traffic(&self) -> Traffic {
        self.traffic.get()
    }
    /// Select the head-only experiment before any draft or KV conditioning.
    /// Rewinding a used workspace does not reopen this immutable selection.
    #[allow(dead_code)] // Protected runner selection is integrated separately.
    pub fn set_q8_head_broadcast(&mut self, enabled: bool) -> Result<()> {
        validate_english_broadcast(self.public_english_vocab.is_some(), enabled)?;
        validate_q8_head_broadcast_selection(
            self.graph.profile(),
            self.closed,
            self.poisoned || self.retired.load(Ordering::Acquire),
            self.position,
            self.q8_head_policy_locked,
        )?;
        validate_q8_head_modes(enabled, self.q8_head_shared_scales)?;
        self.verify_source()?;
        self.physical()?;
        self.q8_head_broadcast = enabled;
        Ok(())
    }
    /// Actual completed full-forward policy, captured before workspace close.
    /// Merely selecting the experiment or appending KV cannot claim its use.
    #[allow(dead_code)] // Protected runner receipts are integrated separately.
    pub fn completed_q8_head_broadcast(&self) -> Result<bool> {
        if self.closed || self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat(
                "MTP completed Q8 head policy unavailable",
            ));
        }
        self.physical()?;
        Ok(self.q8_head_broadcast_completed)
    }
    /// Separate pre-run cooperative-scale choice, exclusive with broadcast.
    #[allow(dead_code)] // Protected runner selection is integrated separately.
    pub fn set_q8_head_shared_scales(&mut self, enabled: bool) -> Result<()> {
        validate_q8_head_broadcast_selection(
            self.graph.profile(),
            self.closed,
            self.poisoned || self.retired.load(Ordering::Acquire),
            self.position,
            self.q8_head_policy_locked,
        )?;
        validate_q8_head_modes(self.q8_head_broadcast, enabled)?;
        self.verify_source()?;
        self.physical()?;
        self.q8_head_shared_scales = enabled;
        Ok(())
    }
    /// Actual successful full-forward use, captured with live owners before close.
    #[allow(dead_code)] // Protected runner receipts are integrated separately.
    pub fn completed_q8_head_shared_scales(&self) -> Result<bool> {
        if self.closed || self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat(
                "MTP completed Q8 shared scale policy unavailable",
            ));
        }
        self.physical()?;
        Ok(self.q8_head_shared_scales_completed)
    }
    /// Actual successful draft-head execution, available only with live healthy
    /// owners. Construction/append/rewind alone never reports completed use.
    pub fn completed_public_english_vocab(&self) -> Result<bool> {
        let completed = completed_english_policy(
            self.public_english_vocab.is_some(),
            self.public_english_vocab_completed,
            self.closed,
            self.poisoned,
            self.retired.load(Ordering::Acquire),
        )?;
        self.verify_source()?;
        self.physical()?;
        Ok(completed)
    }
    /// Content-free immutable map identity after an accepted draft operation.
    pub fn public_english_vocab_receipt(&self) -> Result<(u32, &'static str)> {
        if !self.completed_public_english_vocab()? {
            return Err(Error::InvalidFormat(
                "MTP English vocabulary has no completed draft",
            ));
        }
        Ok((english::ENGLISH_ROWS as u32, english::ENGLISH_SHA256))
    }
    fn public(&self) -> Result<&Public> {
        self.public
            .as_ref()
            .ok_or(Error::InvalidFormat("MTP resident public owner absent"))
    }
    fn vector_arg(&self, name: &'static str) -> Result<ArrayArg<WgpuRuntime>> {
        Self::public_arg(self.public()?.one(name)?)
    }
    fn project(&mut self, name: &'static str, input: B, output: B) -> Result<()> {
        let spec = matrices_for(self.graph.profile())
            .iter()
            .copied()
            .chain([head_up_for(self.graph.profile())])
            .find(|s| s.name == name)
            .ok_or(Error::InvalidFormat("MTP projection role"))?;
        if WORDS[input as usize] != spec.columns
            || WORDS[output as usize] != spec.rows
            || input == output
        {
            return Err(Error::InvalidBufferLength);
        }
        if name == "output.weight" && self.public()?.cjk.is_some() {
            let planes = self.dense_planes.ok_or(Error::InvalidBufferLength)?;
            let calls = self
                .public()?
                .cjk
                .as_ref()
                .ok_or(Error::InvalidBufferLength)?
                .project(
                    self.client()?,
                    u::handle(&self.arena, self.id(input))?,
                    self.arena.handle(planes).cloned().map_err(error)?,
                    u::handle(&self.arena, self.id(output))?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                )?;
            self.tick(calls)?;
            return self.finite(output);
        }
        if self.public_english_vocab.is_some() && name == "output.weight" {
            return self.project_public_english(input, output);
        }
        if let Some(id) = self.dense_planes {
            let planes = self.arena.handle(id).cloned().map_err(error)?;
            if crate::strata_dense::supports_count(
                spec.dtype,
                spec.columns,
                planes.size_in_used(),
                1,
            ) {
                let client = self.client()?;
                let x = u::handle(&self.arena, self.id(input))?;
                let y = u::handle(&self.arena, self.id(output))?;
                let fault = u::handle(&self.arena, self.id(B::Fault))?;
                crate::strata_dense::prepare(
                    client,
                    spec.columns,
                    x,
                    planes.clone(),
                    fault.clone(),
                    &matmul::BatchSpans::new(1, 0, 0, spec.columns, 0, spec.rows),
                )?;
                let pages = self.public()?.pages(name)?;
                let row_bytes = quant_matvec::packed_row_bytes(spec.columns, spec.dtype);
                for page in pages {
                    if page.logical_bytes != page.rows * row_bytes
                        || page.first + page.rows > spec.rows
                    {
                        return Err(Error::InvalidBufferLength);
                    }
                    crate::strata_dense::project(
                        client,
                        spec.dtype,
                        page.rows,
                        spec.columns,
                        page.handle.clone(),
                        planes.clone(),
                        y.clone(),
                        fault.clone(),
                        page.first,
                        spec.rows,
                        1,
                    )?;
                }
                self.tick(pages.len() + 1)?;
                return self.finite(output);
            }
        }
        let pages = self.public()?.pages(name)?;
        let row_bytes = quant_matvec::packed_row_bytes(spec.columns, spec.dtype);
        for (index, page) in pages.iter().enumerate() {
            if page.logical_bytes != page.rows * row_bytes || page.first + page.rows > spec.rows {
                return Err(Error::InvalidBufferLength);
            }
            // SAFETY: exact original role geometry, complete disjoint owners and
            // source/finite/tail preflight precede all unchecked leaf launches.
            unsafe {
                if self.q8_head_shared_scales && name == "output.weight" {
                    validate_q8_head_modes(self.q8_head_broadcast, self.q8_head_shared_scales)?;
                    if self.graph.profile() != MtpProfile::SelfContainedQ8
                        || spec.dtype != quant_matvec::Q8_0
                        || spec.columns != k::D
                        || spec.rows != k::VOCAB
                    {
                        return Err(Error::InvalidFormat("MTP Q8 head shared scale geometry"));
                    }
                    fast::packed_q8_head_shared_scales_fault::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        fast::fault_grid(page.rows, spec.rows, index + 1 == pages.len()),
                        CubeDim::new_1d(128),
                        Self::public_arg(page)?,
                        self.a(input)?,
                        self.a(output)?,
                        self.a(B::Fault)?,
                        1,
                        0,
                        0,
                        page.first,
                        0,
                        page.rows,
                        spec.columns,
                        row_bytes,
                        spec.rows,
                        index + 1 == pages.len(),
                    );
                    continue;
                }
                if self.q8_head_broadcast && name == "output.weight" {
                    if self.graph.profile() != MtpProfile::SelfContainedQ8
                        || spec.dtype != quant_matvec::Q8_0
                        || spec.columns != k::D
                        || spec.rows != k::VOCAB
                    {
                        return Err(Error::InvalidFormat("MTP Q8 head broadcast geometry"));
                    }
                    fast::packed_q8_head_broadcast_fault::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        fast::fault_grid(page.rows, spec.rows, index + 1 == pages.len()),
                        CubeDim::new_1d(128),
                        Self::public_arg(page)?,
                        self.a(input)?,
                        self.a(output)?,
                        self.a(B::Fault)?,
                        1,
                        0,
                        0,
                        page.first,
                        0,
                        page.rows,
                        spec.columns,
                        row_bytes,
                        spec.rows,
                        index + 1 == pages.len(),
                    );
                    continue;
                }
                fast::packed_single_fault::launch_unchecked::<WgpuRuntime>(
                    self.client()?,
                    fast::fault_grid(page.rows, spec.rows, index + 1 == pages.len()),
                    CubeDim::new_1d(128),
                    Self::public_arg(page)?,
                    self.a(input)?,
                    self.a(output)?,
                    self.a(B::Fault)?,
                    1,
                    0,
                    0,
                    page.first,
                    0,
                    page.rows,
                    spec.columns,
                    spec.dtype,
                    row_bytes,
                    spec.rows,
                    index + 1 == pages.len(),
                );
            }
        }
        self.tick(pages.len())?;
        self.finite(output)
    }
    fn project_public_english(&mut self, input: B, output: B) -> Result<()> {
        let proof = self
            .public_english_vocab
            .ok_or(Error::InvalidFormat("MTP English vocabulary absent"))?;
        if (self.graph.profile() != MtpProfile::SelfContainedQ8 && !crate::strata_port::enabled())
            || input != B::Mixed
            || output != B::Logits
            || WORDS[input as usize] != k::D
            || WORDS[output as usize] != k::VOCAB
            || self.q8_head_broadcast
        {
            return Err(Error::InvalidFormat(
                "MTP indexed original Q8 head geometry",
            ));
        }
        if let Some(id) = self.dense_planes {
            let planes = self.arena.handle(id).cloned().map_err(error)?;
            let client = self.client()?;
            let x = u::handle(&self.arena, self.id(input))?;
            let y = u::handle(&self.arena, self.id(output))?;
            let fault = u::handle(&self.arena, self.id(B::Fault))?;
            crate::strata_dense::prepare(
                client,
                k::D,
                x,
                planes.clone(),
                fault.clone(),
                &matmul::BatchSpans::new(1, 0, 0, k::D, 0, k::VOCAB),
            )?;
            let map = self
                .public()?
                .english_ids
                .as_ref()
                .ok_or(Error::InvalidBufferLength)?;
            let pages = self.public()?.pages("output.weight")?;
            if pages.len() != english::SEGMENTS {
                return Err(Error::InvalidBufferLength);
            }
            let mut dispatches = 1;
            for (index, page) in pages.iter().enumerate() {
                let segment = proof.segment(index)?;
                segment.validate()?;
                if page.first != segment.source_first
                    || page.rows != segment.source_rows
                    || page.logical_bytes != segment.source_rows * k::ROW_BYTES
                {
                    return Err(Error::InvalidBufferLength);
                }
                if segment.subset_rows == 0 {
                    continue;
                }
                crate::strata_dense::project_indexed(
                    client,
                    segment.source_first,
                    segment.source_rows,
                    segment.subset_first,
                    segment.subset_rows,
                    k::D,
                    page.handle.clone(),
                    planes.clone(),
                    y.clone(),
                    fault.clone(),
                    map.handle.clone(),
                )?;
                dispatches += 1;
            }
            self.tick(dispatches)?;
            return self.finite(output);
        }
        let map = self
            .public()?
            .english_ids
            .as_ref()
            .ok_or(Error::InvalidFormat("MTP public English map owner absent"))?;
        let pages = self.public()?.pages("output.weight")?;
        if pages.len() != english::SEGMENTS {
            return Err(Error::InvalidBufferLength);
        }
        let owned_words = self
            .arena
            .layout(self.id(output))
            .map_err(error)?
            .owned_words;
        let mut dispatches = 0;
        for (index, page) in pages.iter().enumerate() {
            let segment = proof.segment(index)?;
            segment.validate()?;
            if page.first != segment.source_first
                || page.rows != segment.source_rows
                || page.logical_bytes != segment.source_rows * k::ROW_BYTES
            {
                return Err(Error::InvalidBufferLength);
            }
            if segment.subset_rows == 0 {
                continue;
            }
            let check_tail = index + 1 == pages.len();
            let grid = segment.cube_count(check_tail, owned_words)?;
            // SAFETY: exact immutable map readback and full original source/page,
            // input/output extents, no-alias/stream and full padding proof are
            // admitted before consumption. Row mapping precedes packed reads.
            unsafe {
                if self.q8_head_shared_scales {
                    english::packed_q8_english_head_shared_scales_fault::launch_unchecked::<
                        WgpuRuntime,
                    >(
                        self.client()?,
                        grid,
                        CubeDim::new_1d(128),
                        Self::public_arg(page)?,
                        Self::public_arg(map)?,
                        self.a(input)?,
                        self.full(output)?,
                        self.a(B::Fault)?,
                        1,
                        0,
                        0,
                        segment.source_first,
                        segment.subset_first,
                        segment.source_rows,
                        segment.subset_rows,
                        check_tail,
                    );
                } else {
                    english::packed_q8_english_head_fault::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        grid,
                        CubeDim::new_1d(128),
                        Self::public_arg(page)?,
                        Self::public_arg(map)?,
                        self.a(input)?,
                        self.full(output)?,
                        self.a(B::Fault)?,
                        1,
                        0,
                        0,
                        segment.source_first,
                        segment.subset_first,
                        segment.source_rows,
                        segment.subset_rows,
                        check_tail,
                    );
                }
            }
            dispatches += 1;
        }
        self.tick(dispatches)?;
        self.finite(output)
    }
    fn normalize(
        &mut self,
        input: B,
        output: B,
        gamma: &'static str,
        groups: usize,
        width: usize,
    ) -> Result<()> {
        if groups * width != WORDS[input as usize]
            || groups * width != WORDS[output as usize]
            || input == output
        {
            return Err(Error::InvalidBufferLength);
        }
        k::grouped_rms_norm::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(groups as u32, 1, 1),
            CubeDim::new_1d(64),
            self.a(input)?,
            self.vector_arg(gamma)?,
            self.a(output)?,
            self.prefix(B::Status, 0, groups)?,
            EPS,
            groups,
            width,
        );
        self.tick(1)?;
        self.status(groups)?;
        self.finite(output)
    }
    fn condition(&mut self) -> Result<()> {
        let pages = self.public()?.pages(EMBEDDING.name)?;
        for page in pages {
            if self.graph.profile() == MtpProfile::StrataQ2 {
                unsafe {
                    strata_k::embedding_tile::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        u::grid(k::D)?,
                        CubeDim::new_1d(64),
                        Self::public_arg(page)?,
                        self.a(B::Token)?,
                        self.a(B::Embedding)?,
                        self.a(B::Fault)?,
                        page.first,
                        page.rows,
                    );
                }
                continue;
            }
            unsafe {
                k::embedding_tile::launch_unchecked::<WgpuRuntime>(
                    self.client()?,
                    u::grid(k::D)?,
                    CubeDim::new_1d(64),
                    Self::public_arg(page)?,
                    self.a(B::Token)?,
                    self.a(B::Embedding)?,
                    self.a(B::Fault)?,
                    page.first,
                    page.rows,
                );
            }
        }
        self.tick(pages.len())?;
        self.finite(B::Embedding)?;
        self.normalize(
            B::Embedding,
            B::EmbeddingNorm,
            "blk.48.nextn.enorm.weight",
            1,
            k::D,
        )?;
        self.normalize(
            B::Carry,
            B::HiddenNorm,
            "blk.48.nextn.hnorm.weight",
            HIDDEN_NORM_GROUPS,
            HIDDEN_NORM_WIDTH,
        )?;
        if self.graph.profile() == MtpProfile::StrataQ2 {
            return self.condition_strata();
        }
        k::condition_concat::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::CONDITION_WORDS)?,
            CubeDim::new_1d(64),
            self.a(B::EmbeddingNorm)?,
            self.a(B::HiddenNorm)?,
            self.a(B::Condition)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        let page = self.public()?.one("blk.48.nextn.eh_proj.weight")?;
        let geometry =
            matmul::MatrixGeometry::new(quant_matvec::Q8_0, k::D, k::D * 2, 4).map_err(error)?;
        let spans = matmul::BatchSpans::new(k::HC, 0, 0, k::D * 2, 0, k::D);
        geometry
            .validate_spans(&spans, page.logical_bytes / 4, k::CONDITION_WORDS, k::WIDE)
            .map_err(error)?;
        unsafe {
            matmul::packed_matmul::launch_unchecked::<WgpuRuntime>(
                self.client()?,
                geometry.grid(),
                CubeDim::new_1d(64),
                Self::public_arg(page)?,
                self.a(B::Condition)?,
                self.a(B::Residual0)?,
                k::HC,
                0,
                0,
                k::D * 2,
                0,
                k::D,
                k::D,
                k::D * 2,
                quant_matvec::Q8_0,
                4,
            );
        }
        self.tick(1)?;
        self.finite(B::Residual0)
    }
    fn condition_strata(&mut self) -> Result<()> {
        // Original embedding is dead after its norm. Reuse its exact2560-word
        // owner for the embedding FC, distinct from the normalized input.
        self.project(strata::FC_EMBEDDING, B::EmbeddingNorm, B::Embedding)?;
        let page = self.public()?.one(strata::FC_HIDDEN)?;
        let dtype = if crate::strata_port::enabled() {
            quant_matvec::Q8_0
        } else {
            quant_matvec::BF16
        };
        let geometry = matmul::MatrixGeometry::new(dtype, k::D, k::D, 4).map_err(error)?;
        let spans = matmul::BatchSpans::new(k::HC, 0, 0, k::D, 0, k::D);
        geometry
            .validate_spans(&spans, page.logical_bytes / 4, k::WIDE, k::WIDE)
            .map_err(error)?;
        unsafe {
            matmul::packed_matmul::launch_unchecked::<WgpuRuntime>(
                self.client()?,
                geometry.grid(),
                CubeDim::new_1d(64),
                Self::public_arg(page)?,
                self.a(B::HiddenNorm)?,
                self.a(B::Residual0)?,
                k::HC,
                0,
                0,
                k::D,
                0,
                k::D,
                k::D,
                k::D,
                dtype,
                4,
            );
        }
        self.tick(1)?;
        self.finite(B::Residual0)?;
        strata_k::broadcast_embedding::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::WIDE)?,
            CubeDim::new_1d(64),
            self.a(B::Embedding)?,
            self.a(B::Residual0)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        self.finite(B::Residual0)
    }
    fn hc(&mut self, input: B, kind: usize, inject: bool) -> Result<()> {
        let names = match kind {
            0 => [
                "blk.48.hc_attn_norm.weight",
                "blk.48.hc_attn_down.weight",
                "blk.48.hc_attn_up.weight",
                "blk.48.hc_attn_inject.weight",
            ],
            1 => [
                "blk.48.hc_ffn_norm.weight",
                "blk.48.hc_ffn_down.weight",
                "blk.48.hc_ffn_up.weight",
                "blk.48.hc_ffn_inject.weight",
            ],
            2 if !inject => [
                "blk.48.nextn.hc_head_norm.weight",
                "blk.48.nextn.hc_head_down.weight",
                "blk.48.nextn.hc_head_up.weight",
                "",
            ],
            _ => return Err(Error::InvalidFormat("MTP HC role")),
        };
        self.normalize(input, B::Norm, names[0], k::HC, k::D)?;
        self.project(names[1], B::Norm, B::LowRaw)?;
        k::activate_down::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::LOW_RANK)?,
            CubeDim::new_1d(64),
            self.a(B::LowRaw)?,
            self.a(B::Low)?,
            self.prefix(B::Status, 0, k::LOW_RANK)?,
            k::HC,
            k::LOW_RANK,
        );
        self.tick(1)?;
        self.status(k::LOW_RANK)?;
        self.project(names[2], B::Low, B::Up)?;
        k::mix::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::D)?,
            CubeDim::new_1d(64),
            self.a(B::Norm)?,
            self.a(B::Up)?,
            self.a(B::Mixed)?,
            self.prefix(B::Status, 0, k::D)?,
            k::HC,
            k::D,
        );
        self.tick(1)?;
        self.status(k::D)?;
        self.finite(B::Mixed)?;
        if inject {
            self.project(names[3], B::Norm, B::Injection)?;
        }
        Ok(())
    }
    fn combine(&mut self, original: B, output: B) -> Result<()> {
        if original == output {
            return Err(Error::InvalidBufferLength);
        }
        k::combine::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::WIDE)?,
            CubeDim::new_1d(64),
            self.a(original)?,
            self.a(B::Block)?,
            self.a(B::Injection)?,
            self.a(output)?,
            self.prefix(B::Status, 0, k::WIDE)?,
            k::HC,
            k::D,
        );
        self.tick(1)?;
        self.status(k::WIDE)?;
        self.finite(output)
    }
    fn key_value(&mut self) -> Result<()> {
        self.key_value_at(self.position)
    }
    fn key_value_at(&mut self, position: usize) -> Result<()> {
        self.project("blk.48.attn_k.weight", B::Mixed, B::Key)?;
        self.project("blk.48.attn_v.weight", B::Mixed, B::Value)?;
        self.rope_at(
            B::Key,
            B::KeyRotated,
            "blk.48.attn_k_norm.weight",
            k::KVH,
            position,
        )?;
        k::append_kv::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::KV)?,
            CubeDim::new_1d(64),
            self.a(B::KeyRotated)?,
            self.a(B::Value)?,
            self.a(B::Keys)?,
            self.a(B::Values)?,
            self.a(B::Fault)?,
            position,
        );
        self.tick(1)
    }
    fn rope(&mut self, input: B, output: B, gamma: &'static str, groups: usize) -> Result<()> {
        self.rope_at(input, output, gamma, groups, self.position)
    }
    fn rope_at(
        &mut self,
        input: B,
        output: B,
        gamma: &'static str,
        groups: usize,
        position: usize,
    ) -> Result<()> {
        if groups * k::HD != WORDS[input as usize] || groups * k::HD != WORDS[output as usize] {
            return Err(Error::InvalidBufferLength);
        }
        let public = self.public()?;
        k::rms_rope::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(groups as u32, 1, 1),
            CubeDim::new_1d(64),
            self.a(input)?,
            self.vector_arg(gamma)?,
            Self::public_arg(&public.sin)?,
            Self::public_arg(&public.cos)?,
            self.a(B::Token)?,
            self.a(output)?,
            self.prefix(B::Status, 0, groups)?,
            groups,
            position as u32,
            EPS,
            k::HD,
            true,
            false,
        );
        self.tick(1)?;
        self.status(groups)?;
        self.finite(output)
    }
    fn attention(&mut self) -> Result<()> {
        if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
            return self.native_attention();
        }
        self.project("blk.48.attn_q.weight", B::Mixed, B::RawQuery)?;
        k::split_query_gate::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::QUERY * 2)?,
            CubeDim::new_1d(64),
            self.a(B::RawQuery)?,
            self.a(B::SplitQuery)?,
            self.a(B::Status)?,
        );
        self.tick(1)?;
        self.status(k::QUERY * 2)?;
        let public = self.public()?;
        k::rms_rope::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(k::QH as u32, 1, 1),
            CubeDim::new_1d(64),
            self.prefix(B::SplitQuery, 0, k::QUERY)?,
            self.vector_arg("blk.48.attn_q_norm.weight")?,
            Self::public_arg(&public.sin)?,
            Self::public_arg(&public.cos)?,
            self.a(B::Token)?,
            self.a(B::Query)?,
            self.prefix(B::Status, 0, k::QH)?,
            k::QH,
            self.position as u32,
            EPS,
            k::HD,
            true,
            false,
        );
        self.tick(1)?;
        self.status(k::QH)?;
        if !crate::strata_intel_core::enabled() || self.graph.profile() != MtpProfile::StrataQ2 {
            self.key_value()?;
        }
        let visible = k::validate_position(self.position)?;
        k::dense_logits::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static((k::QH * visible) as u32, 1, 1),
            CubeDim::new_1d(64),
            self.a(B::Query)?,
            self.a(B::Keys)?,
            self.a(B::Attention)?,
            self.a(B::Fault)?,
            visible,
        );
        self.tick(1)?;
        k::attention_softmax::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(k::QH as u32, 1, 1),
            CubeDim::new_1d(64),
            self.a(B::Attention)?,
            self.a(B::Probability)?,
            self.prefix(B::Status, 0, k::QH)?,
            visible,
        );
        self.tick(1)?;
        self.status(k::QH)?;
        k::dense_values_gated::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::QUERY)?,
            CubeDim::new_1d(64),
            self.a(B::Values)?,
            self.a(B::Probability)?,
            self.prefix(B::SplitQuery, k::QUERY, k::QUERY)?,
            self.a(B::Attended)?,
            self.a(B::Fault)?,
            visible,
        );
        self.tick(1)?;
        self.project("blk.48.attn_output.weight", B::Attended, B::Block)
    }
    fn native_attention(&mut self) -> Result<()> {
        use crate::{strata_mtp_front as f, strata_qsa as q};
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        let visible = k::validate_position(self.position)?;
        // Named native recordings keep a constant admitted-context grid. The
        // identity kernel guards i<=start; attention rejects chunk>=ceil(n/64)
        // before any load/barrier, and merge consumes only those logical chunks.
        // The scratch clear retains zero inactive/status semantics.
        let attention_grid = if cfg!(feature = "strata-intel-core") {
            k::CONTEXT.div_ceil(64)
        } else {
            visible.div_ceil(64)
        };
        self.native_front_project(
            "blk.48.attn_q.weight",
            self.id(B::Mixed),
            self.id(B::RawQuery),
            1,
            0,
            0,
            ids[f::PLANES],
            &AtomicBool::new(false),
            false,
        )?;
        k::split_query_gate::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::QUERY * 2)?,
            CubeDim::new_1d(64),
            self.a(B::RawQuery)?,
            self.a(B::SplitQuery)?,
            self.a(B::Status)?,
        );
        self.tick(1)?;
        self.status(k::QUERY * 2)?;
        let mut info = [
            1,
            self.position as u32,
            k::CONTEXT as u32,
            12288,
            6144,
            24,
            24,
            256,
            0,
            EPS.to_bits(),
            crate::qsa_gpu::THETA.powf(-2. / 64.).to_bits(),
            0,
        ];
        q::launch(
            &self.arena,
            q::Kernel::Norm,
            [24, 1, 1],
            vec![
                u::handle(&self.arena, self.id(B::SplitQuery))?,
                self.front_public("blk.48.attn_q_norm.weight")?,
                u::handle(&self.arena, ids[f::POS])?,
                u::handle(&self.arena, self.id(B::Query))?,
                u::handle(&self.arena, self.id(B::Status))?,
            ],
            info,
        )?;
        self.tick(1)?;
        self.status(k::QH)?;
        f::launch(
            &self.arena,
            f::Kernel::Identity,
            [attention_grid, 1, 1],
            vec![u::handle(&self.arena, self.id(B::Packet))?],
            [1, self.position as u32, 0, 0, 0, 0, 0, 0],
        )?;
        info[5] = k::ATTENTION_WORDS as u32;
        q::launch(
            &self.arena,
            q::Kernel::MtpAttention,
            [attention_grid, 2, 1],
            vec![
                u::handle(&self.arena, self.id(B::Query))?,
                u::handle(&self.arena, self.id(B::Keys))?,
                u::handle(&self.arena, self.id(B::Values))?,
                u::handle(&self.arena, self.id(B::Packet))?,
                u::handle(&self.arena, self.id(B::Attention))?,
                u::handle(&self.arena, self.id(B::Probability))?,
                u::handle(&self.arena, ids[f::STATUS])?,
            ],
            info,
        )?;
        u::status_batch(
            &self.arena,
            ids[f::STATUS],
            self.id(B::Fault),
            1,
            0,
            k::QH * visible,
            k::QH * visible,
        )?;
        info[5] = 24;
        q::launch(
            &self.arena,
            q::Kernel::MtpMerge,
            [24, 1, 1],
            vec![
                u::handle(&self.arena, self.id(B::Attention))?,
                u::handle(&self.arena, self.id(B::Probability))?,
                u::handle(&self.arena, self.id(B::Attended))?,
                u::handle(&self.arena, ids[f::STATUS])?,
            ],
            info,
        )?;
        u::status_batch(&self.arena, ids[f::STATUS], self.id(B::Fault), 1, 0, 24, 24)?;
        info[5] = 6144;
        q::launch(
            &self.arena,
            q::Kernel::Gate,
            [96, 1, 1],
            vec![
                u::handle(&self.arena, self.id(B::SplitQuery))?,
                u::handle(&self.arena, self.id(B::Attended))?,
                u::handle(&self.arena, ids[f::STATUS])?,
            ],
            info,
        )?;
        u::status_batch(
            &self.arena,
            ids[f::STATUS],
            self.id(B::Fault),
            1,
            0,
            6144,
            6144,
        )?;
        self.tick(7)?;
        self.project("blk.48.attn_output.weight", B::Attended, B::Block)
    }
    fn moe(&mut self) -> Result<()> {
        if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
            return self.native_moe();
        }
        self.project("blk.48.ffn_gate_inp.weight", B::Mixed, B::Router)?;
        crate::next_shared::router(
            self.client()?,
            u::handle(&self.arena, self.id(B::Router))?,
            u::handle(&self.arena, self.id(B::Packet))?,
            0,
        )?;
        self.tick(1)?;
        k::routes_preflight::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            self.a(B::Packet)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        if self.graph.profile() == MtpProfile::StrataQ2 {
            self.strata_experts()?;
        } else {
            for index in 0..k::SHARDS {
                let shard = k::Shard::new(index)?;
                let gate = &self.public()?.pages(EXPERT_NAMES[0])?[index];
                let up = &self.public()?.pages(EXPERT_NAMES[1])?[index];
                if gate.first != shard.first
                    || up.first != shard.first
                    || gate.rows != shard.experts
                    || up.rows != shard.experts
                {
                    return Err(Error::InvalidBufferLength);
                }
                unsafe {
                    k::q8_gate_up::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        CubeCount::Static((k::FF / 4) as u32, k::RANKS as u32, 1),
                        CubeDim::new_1d(128),
                        Self::public_arg(gate)?,
                        Self::public_arg(up)?,
                        self.a(B::Mixed)?,
                        self.a(B::Packet)?,
                        self.a(B::Activation)?,
                        self.a(B::Readiness)?,
                        self.a(B::Fault)?,
                        shard.first,
                        shard.experts,
                    );
                }
            }
            self.tick(k::SHARDS)?;
            for index in 0..k::SHARDS {
                let shard = k::Shard::new(index)?;
                let down = &self.public()?.pages(EXPERT_NAMES[2])?[index];
                if down.first != shard.first || down.rows != shard.experts {
                    return Err(Error::InvalidBufferLength);
                }
                unsafe {
                    k::q8_down::launch_unchecked::<WgpuRuntime>(
                        self.client()?,
                        CubeCount::Static((k::D / 4) as u32, k::RANKS as u32, 1),
                        CubeDim::new_1d(128),
                        Self::public_arg(down)?,
                        self.a(B::Activation)?,
                        self.a(B::Packet)?,
                        self.a(B::Partials)?,
                        self.a(B::Readiness)?,
                        self.a(B::Fault)?,
                        shard.first,
                        shard.experts,
                    );
                }
            }
            self.tick(k::SHARDS)?;
        }
        self.project("blk.48.ffn_gate_shexp.weight", B::Mixed, B::SharedGate)?;
        self.project("blk.48.ffn_up_shexp.weight", B::Mixed, B::SharedUp)?;
        k::shared_activate::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::FF)?,
            CubeDim::new_1d(64),
            self.a(B::SharedGate)?,
            self.a(B::SharedUp)?,
            self.a(B::SharedAct)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        self.project("blk.48.ffn_down_shexp.weight", B::SharedAct, B::SharedDown)?;
        if self.graph.profile() == MtpProfile::StrataQ2 {
            self.project(
                "blk.48.ffn_gate_inp_shexp.weight",
                B::Mixed,
                B::SharedScalar,
            )?;
        } else {
            k::shared_gate::launch::<WgpuRuntime>(
                self.client()?,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(64),
                self.vector_arg("blk.48.ffn_gate_inp_shexp.weight")?,
                self.a(B::Mixed)?,
                self.a(B::SharedScalar)?,
                self.a(B::Fault)?,
            );
            self.tick(1)?;
        }
        k::routed_shared_merge::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::D)?,
            CubeDim::new_1d(64),
            self.a(B::Partials)?,
            self.a(B::SharedDown)?,
            self.a(B::SharedScalar)?,
            self.a(B::Packet)?,
            self.a(B::Readiness)?,
            self.a(B::Block)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        self.finite(B::Block)
    }
    fn native_bf16_moe_dot(&mut self, name: &'static str, output: B, rows: usize) -> Result<()> {
        let spec = matrices_for(self.graph.profile())
            .iter()
            .find(|s| s.name == name)
            .ok_or(Error::InvalidFormat("native MTP BF16 MoE role"))?;
        let page = self.public()?.one(name)?;
        if spec.dtype != quant_matvec::BF16
            || spec.columns != k::D
            || spec.rows != rows
            || ![1, k::EXPERTS].contains(&rows)
            || page.first != 0
            || page.rows != rows
            || page.logical_bytes != rows * k::D * 2
            || WORDS[output as usize] < rows
        {
            return Err(Error::InvalidBufferLength);
        }
        crate::strata_mtp_moe::launch(
            &self.arena,
            crate::strata_mtp_moe::Kernel::RouterDot,
            rows,
            vec![
                self.front_public(name)?,
                u::handle(&self.arena, self.id(B::Mixed))?,
                u::handle(&self.arena, self.id(output))?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            [1, 0, rows as u32, 0, 0, 0, 0, 0],
        )?;
        self.tick(1)
    }
    fn native_moe(&mut self) -> Result<()> {
        use crate::{strata_mtp_front as f, strata_mtp_moe as m};
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        self.native_bf16_moe_dot("blk.48.ffn_gate_inp.weight", B::Router, k::EXPERTS)?;
        m::launch(
            &self.arena,
            m::Kernel::Route,
            1,
            vec![
                u::handle(&self.arena, self.id(B::Router))?,
                u::handle(&self.arena, self.id(B::Packet))?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )?;
        k::routes_preflight::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            self.a(B::Packet)?,
            self.a(B::Fault)?,
        );
        m::launch(
            &self.arena,
            m::Kernel::Prepare,
            10,
            vec![
                u::handle(&self.arena, self.id(B::Mixed))?,
                u::handle(&self.arena, ids[f::EXPERT_INPUT])?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )?;
        self.tick(3)?;
        // Count1 routes are unique, all512 MTP experts resident. The actual rank
        // is the native destination identity, not a substituted expert choice.
        for down in [false, true] {
            if down {
                m::launch(
                    &self.arena,
                    m::Kernel::ActivateQuant,
                    25,
                    vec![
                        u::handle(&self.arena, ids[f::EXPERT_GU])?,
                        u::handle(&self.arena, ids[f::EXPERT_HIDDEN])?,
                        u::handle(&self.arena, self.id(B::Fault))?,
                    ],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                )?;
                self.tick(1)?;
            }
            for index in 0..strata::SHARDS {
                let shard = strata::Shard::new(index)?;
                let page = &self.public()?.pages(if down {
                    strata::DOWN_NAME
                } else {
                    strata::GU_NAME
                })?[index];
                if page.first != shard.first
                    || page.rows != strata::EXPERTS_PER_SHARD
                    || page.logical_bytes != shard.bytes(!down)?
                    || page.owned_bytes != shard.owned(!down)?
                {
                    return Err(Error::InvalidBufferLength);
                }
                m::launch(
                    &self.arena,
                    if down { m::Kernel::Down } else { m::Kernel::Gu },
                    if down { 1600 } else { 800 },
                    vec![
                        page.handle
                            .clone()
                            .offset_end((page.owned_bytes - page.logical_bytes) as u64),
                        u::handle(
                            &self.arena,
                            ids[if down {
                                f::EXPERT_HIDDEN
                            } else {
                                f::EXPERT_INPUT
                            }],
                        )?,
                        u::handle(&self.arena, self.id(B::Packet))?,
                        u::handle(
                            &self.arena,
                            if down {
                                self.id(B::Partials)
                            } else {
                                ids[f::EXPERT_GU]
                            },
                        )?,
                        u::handle(&self.arena, self.id(B::Readiness))?,
                        u::handle(&self.arena, self.id(B::Fault))?,
                    ],
                    [1, shard.first as u32, 0, 0, 0, 0, 0, 0],
                )?;
            }
            self.tick(strata::SHARDS)?;
        }
        let cancel = AtomicBool::new(false);
        // Gate/up consume one prepared native dense input; routed S2 planes have
        // separate ownership and scales. Fused SwiGLU writes the actual down
        // Q8_1 planes, matching Strata's default fused shared producer.
        self.native_front_project(
            "blk.48.ffn_gate_shexp.weight",
            self.id(B::Mixed),
            self.id(B::SharedGate),
            1,
            0,
            0,
            ids[f::PLANES],
            &cancel,
            true,
        )?;
        self.native_front_project(
            "blk.48.ffn_up_shexp.weight",
            self.id(B::Mixed),
            self.id(B::SharedUp),
            1,
            0,
            0,
            ids[f::PLANES],
            &cancel,
            false,
        )?;
        m::launch(
            &self.arena,
            m::Kernel::SharedActivate,
            3,
            vec![
                u::handle(&self.arena, self.id(B::SharedGate))?,
                u::handle(&self.arena, self.id(B::SharedUp))?,
                u::handle(&self.arena, self.id(B::SharedAct))?,
                u::handle(&self.arena, ids[f::PLANES])?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )?;
        self.tick(1)?;
        self.native_front_project(
            "blk.48.ffn_down_shexp.weight",
            self.id(B::SharedAct),
            self.id(B::SharedDown),
            1,
            0,
            0,
            ids[f::PLANES],
            &cancel,
            false,
        )?;
        self.native_bf16_moe_dot("blk.48.ffn_gate_inp_shexp.weight", B::SharedScalar, 1)?;
        m::launch(
            &self.arena,
            m::Kernel::Combine,
            10,
            vec![
                u::handle(&self.arena, self.id(B::Partials))?,
                u::handle(&self.arena, self.id(B::SharedDown))?,
                u::handle(&self.arena, self.id(B::SharedScalar))?,
                u::handle(&self.arena, self.id(B::Packet))?,
                u::handle(&self.arena, self.id(B::Readiness))?,
                u::handle(&self.arena, self.id(B::Block))?,
                u::handle(&self.arena, self.id(B::Fault))?,
            ],
            [1, 0, 0, 0, 0, 0, 0, 0],
        )?;
        self.tick(1)
    }
    fn strata_experts(&mut self) -> Result<()> {
        // Original top10 ranks/probabilities are unchanged. Every actual expert
        // exists in one of four complete128-expert shards; no fallback/remap.
        for index in 0..strata::SHARDS {
            let shard = strata::Shard::new(index)?;
            let gate_up = &self.public()?.pages(strata::GU_NAME)?[index];
            if gate_up.first != shard.first
                || gate_up.rows != strata::EXPERTS_PER_SHARD
                || gate_up.logical_bytes != shard.bytes(true)?
                || gate_up.owned_bytes != shard.owned(true)?
            {
                return Err(Error::InvalidBufferLength);
            }
            unsafe {
                strata_k::gate_up::launch_unchecked::<WgpuRuntime>(
                    self.client()?,
                    CubeCount::Static((k::FF / 4) as u32, k::RANKS as u32, 1),
                    CubeDim::new_1d(128),
                    Self::public_arg(gate_up)?,
                    self.a(B::Mixed)?,
                    self.a(B::Packet)?,
                    self.a(B::Activation)?,
                    self.a(B::Readiness)?,
                    self.a(B::Fault)?,
                    shard.first,
                );
            }
        }
        self.tick(strata::SHARDS)?;
        for index in 0..strata::SHARDS {
            let shard = strata::Shard::new(index)?;
            let down = &self.public()?.pages(strata::DOWN_NAME)?[index];
            if down.first != shard.first
                || down.rows != strata::EXPERTS_PER_SHARD
                || down.logical_bytes != shard.bytes(false)?
                || down.owned_bytes != shard.owned(false)?
            {
                return Err(Error::InvalidBufferLength);
            }
            unsafe {
                strata_k::down::launch_unchecked::<WgpuRuntime>(
                    self.client()?,
                    CubeCount::Static((k::D / 4) as u32, k::RANKS as u32, 1),
                    CubeDim::new_1d(128),
                    Self::public_arg(down)?,
                    self.a(B::Activation)?,
                    self.a(B::Packet)?,
                    self.a(B::Partials)?,
                    self.a(B::Readiness)?,
                    self.a(B::Fault)?,
                    shard.first,
                );
            }
        }
        self.tick(strata::SHARDS)
    }
    fn finish_gpu(&mut self) -> Result<()> {
        self.project("output.weight", B::Mixed, B::Logits)?;
        if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
            if crate::strata_sampler::request::active() && self.public_english_vocab.is_some() {
                return Err(Error::InvalidFormat(
                    "native sampling requires original full MTP head",
                ));
            }
            let positions = self
                .native_front
                .as_ref()
                .ok_or(Error::InvalidBufferLength)?
                .ids[crate::strata_mtp_front::POS];
            if crate::strata_sampler::request::draft_pick(
                self.client()?,
                self.binding,
                self.position,
                u::handle(&self.arena, self.id(B::Logits))?,
                u::handle(&self.arena, positions)?,
                u::handle(&self.arena, self.id(B::Token))?,
                u::handle(&self.arena, self.id(B::Greedy))?,
                u::handle(&self.arena, self.id(B::Candidates))?,
                u::handle(&self.arena, self.id(B::Fault))?,
                self.public()?.cjk.as_ref().map(cjk::Head::maps),
            )? {
                self.tick(4)?;
                self.physical()?;
                self.guard_private_at_native(1)?;
                return Ok(());
            }
        }
        if self.native_confidence_supported() {
            // SAFETY: same original private head/token/fault roots as the
            // legacy selector; all grants survive this scope's fault fence.
            unsafe {
                crate::strata_mtp_probability::argmax(
                    self.client()?,
                    u::handle(&self.arena, self.id(B::Logits))?,
                    u::handle(&self.arena, self.id(B::Greedy))?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                    self.public()?.cjk.is_some(),
                )?;
            }
            self.tick(1)?;
            // SAFETY: cell start cleared Candidates (both normal/follow and
            // fused/scalar clear-role paths), and no front/rest/argmax consumer
            // writes it. Original full head/token/fault roots stay authenticated.
            // The COMPLETE Candidates page is lent to
            // disjoint scratch/probability views and survives the fault fence.
            unsafe {
                crate::strata_mtp_probability::launch(
                    self.client()?,
                    u::handle(&self.arena, self.id(B::Logits))?,
                    u::handle(&self.arena, self.id(B::Greedy))?,
                    u::handle(&self.arena, self.id(B::Candidates))?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                    self.public()?.cjk.is_some(),
                )?;
            }
            self.tick(1)?;
            if let Some(head) = &self.public()?.cjk {
                crate::strata_mtp_probability::map_cjk(
                    self.client()?,
                    head.maps()[0].clone(),
                    u::handle(&self.arena, self.id(B::Greedy))?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                )?;
                self.tick(1)?;
            }
            self.physical()?;
            self.guard_private_at_native(1)?;
            return Ok(());
        }
        if self.public_english_vocab.is_some() {
            let map = self
                .public()?
                .english_ids
                .as_ref()
                .ok_or(Error::InvalidFormat("MTP public English map owner absent"))?;
            english::english_greedy_chunks::launch::<WgpuRuntime>(
                self.client()?,
                CubeCount::Static(english::ENGLISH_GREEDY_GROUPS as u32, 1, 1),
                CubeDim::new_1d(64),
                self.a(B::Logits)?,
                Self::public_arg(map)?,
                self.a(B::Candidates)?,
                self.a(B::Fault)?,
            );
        } else {
            k::greedy_chunks::launch::<WgpuRuntime>(
                self.client()?,
                CubeCount::Static(k::GREEDY_GROUPS as u32, 1, 1),
                CubeDim::new_1d(64),
                self.a(B::Logits)?,
                self.a(B::Candidates)?,
                self.a(B::Fault)?,
            );
        }
        self.tick(1)?;
        if self.public_english_vocab.is_some() {
            english::english_greedy_finish::launch::<WgpuRuntime>(
                self.client()?,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(64),
                self.a(B::Candidates)?,
                self.a(B::Greedy)?,
                self.a(B::Fault)?,
            );
        } else {
            k::greedy_finish::launch::<WgpuRuntime>(
                self.client()?,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(64),
                self.a(B::Candidates)?,
                self.a(B::Greedy)?,
                self.a(B::Fault)?,
            );
        }
        self.tick(1)?;
        self.physical()?;
        self.guard_private_at_native(1)?;
        Ok(())
    }
    fn native_confidence_supported(&self) -> bool {
        crate::strata_intel_core::enabled()
            && self.graph.profile() == MtpProfile::StrataQ2
            && self.public_english_vocab.is_none()
    }
    fn output(&mut self, lease: OutputLease) -> Result<Output> {
        validate_full_output(
            self.public_english_vocab.is_some() || cjk::selected(self.graph.profile()),
        )?;
        self.finish_gpu()?;
        self.complete_fault()?;
        let logits = self.read_f32(B::Logits)?;
        let hidden = self.read_f32(B::Residual2)?;
        let greedy = self.read_words(B::Greedy)?;
        if greedy[0] as usize >= k::VOCAB || greedy[0] != greedy_host(&logits)? {
            return Err(Error::InvalidFormat("MTP full-vocabulary greedy proof"));
        }
        self.verify_source()?;
        self.ledger
            .completed_fence(sample(self.client()?)?)
            .map_err(error)?;
        self.count_traffic(|traffic| {
            traffic.full_output_reads = traffic
                .full_output_reads
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        Ok(Output {
            logits,
            hidden,
            greedy: Zeroizing::new(greedy[0]),
            binding: self.binding,
            start: Zeroizing::new(self.position),
            _lease: lease,
        })
    }
    fn draft_output(&mut self, lease: OutputLease) -> Result<DraftOutput> {
        self.finish_gpu()?;
        let with_probability = self.native_confidence_supported();
        let layout = self.arena.layout(self.id(B::RawQuery)).map_err(error)?;
        if WORDS[B::RawQuery as usize] < DRAFT_WORDS || layout.owned_bytes != 65536 {
            return Err(Error::InvalidBufferLength);
        }
        // Every RawQuery consumer ended at attention, before FFN/head/greedy.
        // Its complete owner is now a single wiping/fenced readback packet.
        u::zero(&self.arena, self.id(B::RawQuery))?;
        self.tick(1)?;
        pack_draft::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(if with_probability {
                DRAFT_NATIVE_WORDS
            } else {
                DRAFT_WORDS
            })?,
            CubeDim::new_1d(64),
            self.a(B::Residual2)?,
            self.a(B::Greedy)?,
            self.full(B::RawQuery)?,
            self.a(B::Fault)?,
            self.a(B::Candidates)?,
            with_probability,
        );
        self.tick(1)?;
        // Distinct dispatch is essential: no grid-wide barrier exists in the
        // producer, so sampling the fault there could precede anotherWG error.
        seal_draft::launch::<WgpuRuntime>(
            self.client()?,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(64),
            self.full(B::RawQuery)?,
            self.a(B::Fault)?,
        );
        self.tick(1)?;
        let packet = self.read_words(B::RawQuery)?;
        self.count_traffic(|traffic| {
            traffic.minimal_packet_reads = traffic
                .minimal_packet_reads
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        let (hidden, greedy, probability) =
            decode_draft_with_probability(&packet, with_probability)?;
        self.verify_source()?;
        self.ledger
            .completed_fence(sample(self.client()?)?)
            .map_err(error)?;
        Ok(DraftOutput {
            hidden,
            probability: probability.map(Zeroizing::new),
            greedy: Zeroizing::new(greedy),
            binding: self.binding,
            start: Zeroizing::new(self.position),
            _lease: lease,
        })
    }
    fn forward(&mut self) -> Result<()> {
        if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
            self.queue_native_front(1, None, self.position, &AtomicBool::new(false))?;
            return self.native_selected_rest(1);
        } else {
            self.condition()?;
            self.hc(B::Residual0, 0, true)?;
        }
        self.attention()?;
        self.combine(B::Residual0, B::Residual1)?;
        self.hc(B::Residual1, 1, true)?;
        self.moe()?;
        self.combine(B::Residual1, B::Residual2)?;
        self.hc(B::Residual2, 2, false)
    }
    /// capture_round's selected last accepted row becomes the sole rest input.
    /// The original full front roots stay owned; no offset view becomes an owner.
    fn native_selected_rest(&mut self, count: usize) -> Result<()> {
        use crate::strata_mtp_front as f;
        if !(1..=8).contains(&count) {
            return Err(Error::InvalidBufferLength);
        }
        if count > 1 {
            let ids = *self
                .native_front
                .as_ref()
                .ok_or(Error::InvalidBufferLength)?
                .ids;
            f::launch(
                &self.arena,
                f::Kernel::Select,
                [k::WIDE.div_ceil(64), 1, 1],
                vec![
                    u::handle(&self.arena, ids[f::RES])?,
                    u::handle(&self.arena, ids[f::INJ])?,
                    u::handle(&self.arena, ids[f::MIXED])?,
                    u::handle(&self.arena, self.id(B::Fault))?,
                ],
                [count as u32, 0, (count - 1) as u32, 0, 0, 0, 0, 0],
            )?;
            self.tick(1)?;
        }
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        for (source, dest, words) in [
            (crate::strata_mtp_front::RES, B::Residual0, k::WIDE),
            (crate::strata_mtp_front::MIXED, B::Mixed, k::D),
            (crate::strata_mtp_front::INJ, B::Injection, k::HC),
        ] {
            u::copy_words::launch::<WgpuRuntime>(
                self.client()?,
                u::grid(words)?,
                CubeDim::new_1d(64),
                u::arg(&self.arena, ids[source])?,
                self.a(dest)?,
            );
        }
        self.tick(3)?;
        if count > 1 {
            crate::strata_dense::prepare(
                self.client()?,
                k::D,
                u::handle(&self.arena, self.id(B::Mixed))?,
                u::handle(&self.arena, ids[f::PLANES])?,
                u::handle(&self.arena, self.id(B::Fault))?,
                &matmul::BatchSpans::new(1, 0, 0, k::D, 0, k::D),
            )?;
            self.tick(1)?;
        }
        self.native_rest_gr(1)?;
        self.moe()?;
        self.native_rest_gr(2)
    }
    fn native_rest_gr(&mut self, kind: usize) -> Result<()> {
        use crate::strata_mtp_front as f;
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        // Attention precedes the MLP read; its output remains in the original
        // Block owner while we copy it into the admitted wide batch view.
        if kind == 1 {
            self.native_attention()?;
        }
        u::copy_words::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::D)?,
            CubeDim::new_1d(64),
            self.a(B::Block)?,
            u::raw_arg(
                self.arena
                    .handle(ids[f::BLOCK])
                    .cloned()
                    .map_err(error)?
                    .offset_end(
                        (self.arena.layout(ids[f::BLOCK]).map_err(error)?.owned_bytes - k::D * 4)
                            as u64,
                    ),
                k::D,
            )?,
        );
        let (source, residual, names) = match kind {
            1 => (
                ids[f::RES],
                ids[f::RES1],
                [
                    "blk.48.hc_ffn_norm.weight",
                    "blk.48.hc_ffn_down.weight",
                    "blk.48.hc_ffn_up.weight",
                    "blk.48.hc_ffn_inject.weight",
                ],
            ),
            2 => (
                ids[f::RES1],
                ids[f::RES],
                [
                    "blk.48.nextn.hc_head_norm.weight",
                    "blk.48.nextn.hc_head_down.weight",
                    "blk.48.nextn.hc_head_up.weight",
                    "blk.48.hc_attn_inject.weight",
                ],
            ),
            _ => return Err(Error::InvalidBufferLength),
        };
        crate::strata_gr::read(
            crate::strata_gr::Read {
                arena: &self.arena,
                residual,
                previous: Some((source, ids[f::BLOCK])),
                gamma: self.front_public(names[0])?,
                buffers: crate::strata_gr::Buffers {
                    normalized: ids[f::NORM],
                    scales: ids[f::SCALES],
                    partial: ids[f::PART],
                    down: ids[f::DOWN],
                    low: ids[f::LOW],
                    injection: ids[f::INJ],
                    mixed: ids[f::MIXED],
                },
                fault: self.id(B::Fault),
                count: 1,
                first: 0,
                epsilon: EPS,
                inject: kind == 1,
            },
            [
                self.front_public(names[1])?,
                self.front_public(names[2])?,
                self.front_public(names[3])?,
            ],
        )?;
        u::copy_words::launch::<WgpuRuntime>(
            self.client()?,
            u::grid(k::D)?,
            CubeDim::new_1d(64),
            u::arg(&self.arena, ids[f::MIXED])?,
            self.a(B::Mixed)?,
        );
        if kind == 2 {
            u::copy_words::launch::<WgpuRuntime>(
                self.client()?,
                u::grid(k::WIDE)?,
                CubeDim::new_1d(64),
                u::arg(&self.arena, residual)?,
                self.a(B::Residual2)?,
            );
            self.tick(1)?;
            self.guard_native_front()?;
        }
        self.tick(6)
    }
    fn commit_forward(&mut self) -> Result<()> {
        if self.native_front_pending.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat(
                "MTP forward has unconsumed front preparation",
            ));
        }
        self.last_front.take();
        let start = self.position;
        self.position += 1;
        self.poisoned = false;
        self.q8_head_broadcast_completed = self.q8_head_broadcast;
        self.q8_head_shared_scales_completed = self.q8_head_shared_scales;
        self.public_english_vocab_completed = self.public_english_vocab.is_some();
        self.last_forward = Some(LastForward {
            binding: self.binding,
            start,
        });
        Ok(())
    }
    fn start_follow(&mut self, stamp: &LastForward) -> Result<()> {
        if self.closed
            || self.poisoned
            || self.retired.load(Ordering::Acquire)
            || !stamp.valid(self.binding, self.position)
        {
            return Err(Error::InvalidFormat("MTP committed followup source"));
        }
        k::validate_position(self.position)?;
        self.verify_source()?;
        self.physical()?;
        self.ledger
            .observe_before(sample(self.client()?)?)
            .map_err(error)?;
        self.poisoned = true;
        self.dispatches = 0;
        let source = self.arena.layout(self.id(B::Residual2)).map_err(error)?;
        let carry = self.arena.layout(self.id(B::Carry)).map_err(error)?;
        let greedy = self.arena.layout(self.id(B::Greedy)).map_err(error)?;
        let token = self.arena.layout(self.id(B::Token)).map_err(error)?;
        if source != carry
            || greedy != token
            || source.logical_words != k::WIDE
            || greedy.logical_words != 1
        {
            return Err(Error::InvalidBufferLength);
        }
        copy_committed::launch::<WgpuRuntime>(
            self.client()?,
            grid(source.owned_words)?,
            CubeDim::new_1d(64),
            self.full(B::Residual2)?,
            self.full(B::Greedy)?,
            self.full(B::Carry)?,
            self.full(B::Token)?,
            self.full(B::Fault)?,
        );
        self.tick(1)?;
        // Source copy precedes any old-scratch clear; preserve the actual new
        // Carry/Token AND stickyfault so copy/preflight faults cannot disappear.
        self.count_traffic(|traffic| {
            traffic.followup_copy_calls = traffic
                .followup_copy_calls
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?;
            Ok(())
        })?;
        self.clear_scratch_preserving_inputs(true)?;
        self.guard_private(self.position)
    }
    fn read_f32(&self, b: B) -> Result<Zeroizing<Vec<f32>>> {
        let bits = self.read_words(b)?;
        let mut values = Zeroizing::new(Vec::with_capacity(WORDS[b as usize]));
        values.extend(bits[..WORDS[b as usize]].iter().map(|&x| f32::from_bits(x)));
        if values.capacity() * 4 > HOST_BYTES || values.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidBufferLength);
        }
        Ok(values)
    }
    pub fn step(&mut self, token: u32, carry: &[f32], cancel: &AtomicBool) -> Result<Output> {
        validate_full_output(
            self.public_english_vocab.is_some() || cjk::selected(self.graph.profile()),
        )?;
        // Publisher alignment: cell c consumes T[c+1] with R[c], using RoPE
        // position c, to predict T[c+2]. The caller starts with this real pair,
        // never a dummy zero-conditioned cell. A provisional next cell can
        // use the preceding draft hidden only behind target verification.
        // Invalid preflight/cancellation must not destroy a previous valid KV
        // sequence. Once start() begins writes, any failure wipes and poisons.
        self.last_forward.take();
        self.last_front.take();
        u::cancel(cancel)?;
        if self.poisoned || self.closed {
            return Err(Error::InvalidFormat("MTP resident poisoned"));
        }
        validate_input(token, carry, self.position)?;
        let lease = OutputLease::acquire(&self.live_outputs)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Output> {
                self.start(token, carry)?;
                self.forward()?;
                u::cancel(cancel)?;
                self.output(lease)
            }));
        match result {
            Ok(Ok(output)) => {
                if let Err(e) = u::cancel(cancel) {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
                if let Err(e) = self.commit_forward() {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
                Ok(output)
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP resident queued failure"))
            }
        }
    }
    pub fn step_draft(
        &mut self,
        token: u32,
        carry: &[f32],
        cancel: &AtomicBool,
    ) -> Result<DraftOutput> {
        self.last_forward.take();
        self.last_front.take();
        u::cancel(cancel)?;
        if self.poisoned || self.closed {
            return Err(Error::InvalidFormat("MTP resident poisoned"));
        }
        validate_input(token, carry, self.position)?;
        let lease = OutputLease::acquire(&self.live_outputs)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<DraftOutput> {
                self.start(token, carry)?;
                self.forward()?;
                u::cancel(cancel)?;
                self.draft_output(lease)
            }));
        self.accept_draft(result, cancel)
    }
    /// Consume the last complete forward's actual GPU token/carry once. There
    /// is no host token/carry argument and no substitution of draft experts.
    /// Any append/rewind/explicit step/failure/close invalidates this authority.
    pub fn follow_draft(&mut self, cancel: &AtomicBool) -> Result<DraftOutput> {
        let stamp = self
            .last_forward
            .take()
            .ok_or(Error::InvalidFormat("MTP no committed followup"))?;
        u::cancel(cancel)?;
        if self.poisoned || self.closed || !stamp.valid(self.binding, self.position) {
            return Err(Error::InvalidFormat("MTP stale committed followup"));
        }
        k::validate_position(self.position)?;
        let lease = OutputLease::acquire(&self.live_outputs)?;
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<DraftOutput> {
                self.start_follow(&stamp)?;
                self.forward()?;
                u::cancel(cancel)?;
                self.draft_output(lease)
            }));
        self.accept_draft(result, cancel)
    }
    fn accept_draft(
        &mut self,
        result: std::thread::Result<Result<DraftOutput>>,
        cancel: &AtomicBool,
    ) -> Result<DraftOutput> {
        match result {
            Ok(Ok(output)) => {
                if let Err(e) = u::cancel(cancel) {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
                if let Err(e) = self.commit_forward() {
                    drop(output);
                    self.fail();
                    return Err(e);
                }
                Ok(output)
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP resident draft failure"))
            }
        }
    }
    /// Repair cache conditioning with the actual preceding target residual.
    /// No FFN, draft logits or controller acceptance occurs in this path.
    pub fn append_kv(&mut self, token: u32, carry: &[f32], cancel: &AtomicBool) -> Result<()> {
        if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
            return self.append_kv_group(&[token], carry, cancel);
        }
        self.last_forward.take();
        self.last_front.take();
        u::cancel(cancel)?;
        if self.poisoned || self.closed {
            return Err(Error::InvalidFormat("MTP resident poisoned"));
        }
        validate_input(token, carry, self.position)?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.start(token, carry)?;
            self.condition()?;
            self.hc(B::Residual0, 0, false)?;
            self.key_value()?;
            self.guard_private(self.position + 1)?;
            u::cancel(cancel)?;
            self.complete_fault()?;
            self.verify_source()?;
            Ok(())
        }));
        match result {
            Ok(Ok(())) => {
                if let Err(e) = u::cancel(cancel) {
                    self.fail();
                    return Err(e);
                }
                self.position += 1;
                self.poisoned = false;
                Ok(())
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP resident append failure"))
            }
        }
    }
    /// Native front-only prompt/catch-up graph. All rows queue before one fault
    /// fence; cursor publication follows the same owned state and erasure proof.
    pub(crate) fn append_kv_group(
        &mut self,
        tokens: &[u32],
        carries: &[f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if !crate::strata_intel_core::enabled()
            || self.graph.profile() != MtpProfile::StrataQ2
            || self.poisoned
            || self.closed
            || self.live_outputs.load(Ordering::Acquire) != 0
            || !(1..=8).contains(&tokens.len())
            || carries.len() != tokens.len() * k::WIDE
        {
            return Err(Error::InvalidBufferLength);
        }
        for (t, &token) in tokens.iter().enumerate() {
            validate_input(
                token,
                &carries[t * k::WIDE..(t + 1) * k::WIDE],
                self.position + t,
            )?;
        }
        u::cancel(cancel)?;
        self.verify_source()?;
        self.physical()?;
        self.last_forward.take();
        self.last_front.take();
        self.poisoned = true;
        self.dispatches = 0;
        let attempted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.clear_scratch()?;
            self.enqueue_native_front(tokens, carries, self.position, cancel)?;
            self.guard_private(self.position + tokens.len())?;
            self.complete_fault()?;
            self.verify_source()?;
            u::cancel(cancel)
        }));
        match attempted {
            Ok(Ok(())) => {
                self.position += tokens.len();
                self.poisoned = false;
                Ok(())
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("native MTP front failure"))
            }
        }
    }
    fn front_public(&self, name: &'static str) -> Result<Handle> {
        let p = self.public()?.one(name)?;
        Ok(p.handle
            .clone()
            .offset_end((p.owned_bytes - p.logical_bytes) as u64))
    }
    fn guard_native_front(&mut self) -> Result<()> {
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        for (i, &id) in ids.iter().enumerate() {
            let l = self.arena.layout(id).map_err(error)?;
            private_guard::launch::<WgpuRuntime>(
                self.client()?,
                grid(l.owned_words)?,
                CubeDim::new_1d(64),
                u::raw_arg(
                    self.arena.handle(id).cloned().map_err(error)?,
                    l.owned_words,
                )?,
                self.a(B::Fault)?,
                l.logical_words,
                l.logical_words,
                !matches!(i, 0 | 17 | 18 | 19 | 22 | 24),
            );
        }
        self.tick(ids.len())
    }
    fn enqueue_native_front(
        &mut self,
        tokens: &[u32],
        carries: &[f32],
        start: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self.queue_native_front(
            tokens.len(),
            Some(NativeFrontInputs::Host(tokens, carries)),
            start,
            cancel,
        )
    }
    fn upload_native_front(&self, id: BufferId, words: &[u32]) -> Result<()> {
        let l = self.arena.layout(id).map_err(error)?;
        if words.len() > l.logical_words {
            return Err(Error::InvalidBufferLength);
        }
        let mut upload = Upload(vec![0; l.owned_bytes]);
        for (dest, word) in upload.0.chunks_exact_mut(4).zip(words) {
            dest.copy_from_slice(&word.to_le_bytes());
        }
        self.client()?
            .write_private(
                self.arena.handle(id).cloned().map_err(error)?,
                upload.take(),
            )
            .map_err(error)
    }
    fn queue_native_front(
        &mut self,
        count: usize,
        inputs: Option<NativeFrontInputs<'_>>,
        start: usize,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let prepared = self.prepare_native_front(count, inputs, start, cancel)?;
        self.enqueue_prepared_native_front(prepared, cancel)
    }
    /// Full original source staging, with the receipt armed before any clear,
    /// upload or borrowed device copy. A graph may begin only after this phase.
    fn prepare_native_front(
        &mut self,
        count: usize,
        inputs: Option<NativeFrontInputs<'_>>,
        start: usize,
        cancel: &AtomicBool,
    ) -> Result<PreparedNativeFront> {
        u::cancel(cancel)?;
        use crate::strata_mtp_front as f;
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        let prepared = PreparedNativeFront::arm(
            self.binding,
            ids,
            count,
            start,
            self.position,
            inputs.is_none(),
            &self.native_front_pending,
        )?;
        if inputs.as_ref().is_some_and(|input| match input {
            NativeFrontInputs::Host(tokens, carries) => {
                tokens.len() != count || carries.len() != count * k::WIDE
            }
            NativeFrontInputs::Device(tokens, _) => tokens.len() != count,
        }) {
            return Err(Error::InvalidBufferLength);
        }
        let fault_id = self.id(B::Fault);
        for &id in &ids {
            u::zero(&self.arena, id)?;
        }
        if let Some(inputs) = &inputs {
            match inputs {
                NativeFrontInputs::Host(tokens, carries) => {
                    self.upload_native_front(ids[f::TOKEN], tokens)?;
                    let words =
                        Zeroizing::new(carries.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    self.upload_native_front(ids[f::CARRY], &words)?;
                }
                NativeFrontInputs::Device(tokens, source) => {
                    self.upload_native_front(ids[f::TOKEN], tokens)?;
                    source.copy_rows(&self.arena, ids[f::CARRY], fault_id, 0, count)?;
                    self.tick(1)?;
                }
            }
        } else {
            for (source, dest, words) in [(B::Token, f::TOKEN, 1), (B::Carry, f::CARRY, k::WIDE)] {
                u::copy_words::launch::<WgpuRuntime>(
                    self.client()?,
                    u::grid(words)?,
                    CubeDim::new_1d(64),
                    self.a(source)?,
                    u::raw_arg(
                        self.arena
                            .handle(ids[dest])
                            .cloned()
                            .map_err(error)?
                            .offset_end(
                                (self.arena.layout(ids[dest]).map_err(error)?.owned_bytes
                                    - words * 4) as u64,
                            ),
                        words,
                    )?,
                );
            }
        }
        u::cancel(cancel)?;
        Ok(prepared)
    }
    /// Consume the exact source plan and queue only its original front math.
    /// The later completed fault/output ledger still owns publication.
    fn enqueue_prepared_native_front(
        &mut self,
        prepared: PreparedNativeFront,
        cancel: &AtomicBool,
    ) -> Result<()> {
        use crate::strata_mtp_front as f;
        u::cancel(cancel)?;
        let ids = *self
            .native_front
            .as_ref()
            .ok_or(Error::InvalidBufferLength)?
            .ids;
        prepared.check(
            self.binding,
            &ids,
            self.position,
            &self.native_front_pending,
        )?;
        let count = prepared.controls[0];
        let start = prepared.controls[1];
        let queued_inputs = prepared.controls[3] != 0;
        let fault_id = self.id(B::Fault);
        let h = |i| u::handle(&self.arena, ids[i]);
        let info = [count as u32, start as u32, 0, 0, 0, 0, EPS.to_bits(), 0];
        let embedding_pages = self.public()?.pages(EMBEDDING.name)?.len();
        for page in self.public()?.pages(EMBEDDING.name)? {
            let mut p = info;
            p[2] = k::VOCAB as u32;
            p[3] = page.rows as u32;
            p[4] = page.first as u32;
            f::launch(
                &self.arena,
                f::Kernel::Embed,
                [40, count, 1],
                vec![
                    page.handle.clone(),
                    h(f::TOKEN)?,
                    h(f::EMBED)?,
                    u::handle(&self.arena, fault_id)?,
                ],
                p,
            )?;
        }
        for (input, out, gamma, columns) in [
            (f::EMBED, f::EN, "blk.48.nextn.enorm.weight", k::D),
            (f::CARRY, f::HN, "blk.48.nextn.hnorm.weight", k::WIDE),
        ] {
            let mut p = info;
            p[2] = columns as u32;
            f::launch(
                &self.arena,
                f::Kernel::Norm,
                [count, 1, 1],
                vec![
                    h(input)?,
                    self.front_public(gamma)?,
                    h(out)?,
                    u::handle(&self.arena, fault_id)?,
                ],
                p,
            )?;
        }
        self.native_front_project(
            strata::FC_EMBEDDING,
            ids[f::EN],
            ids[f::E2],
            count,
            0,
            0,
            ids[f::PLANES],
            cancel,
            true,
        )?;
        for first in (0..count * k::HC).step_by(8) {
            self.native_front_project(
                strata::FC_HIDDEN,
                ids[f::HN],
                ids[f::RES],
                (count * k::HC - first).min(8),
                first * k::D,
                first * k::D,
                ids[f::PLANES],
                cancel,
                true,
            )?;
        }
        let h = |i| u::handle(&self.arena, ids[i]);
        f::launch(
            &self.arena,
            f::Kernel::Broadcast,
            [160, count, 1],
            vec![h(f::E2)?, h(f::RES)?, u::handle(&self.arena, fault_id)?],
            info,
        )?;
        crate::strata_gr::read(
            crate::strata_gr::Read {
                arena: &self.arena,
                residual: ids[f::RES],
                previous: None,
                gamma: self.front_public("blk.48.hc_attn_norm.weight")?,
                buffers: crate::strata_gr::Buffers {
                    normalized: ids[f::NORM],
                    scales: ids[f::SCALES],
                    partial: ids[f::PART],
                    down: ids[f::DOWN],
                    low: ids[f::LOW],
                    injection: ids[f::INJ],
                    mixed: ids[f::MIXED],
                },
                fault: fault_id,
                count,
                first: 0,
                epsilon: EPS,
                inject: true,
            },
            [
                self.front_public("blk.48.hc_attn_down.weight")?,
                self.front_public("blk.48.hc_attn_up.weight")?,
                self.front_public("blk.48.hc_attn_inject.weight")?,
            ],
        )?;
        self.native_front_project(
            "blk.48.attn_k.weight",
            ids[f::MIXED],
            ids[f::KEY],
            count,
            0,
            0,
            ids[f::PLANES],
            cancel,
            true,
        )?;
        self.native_front_project(
            "blk.48.attn_v.weight",
            ids[f::MIXED],
            ids[f::VALUE],
            count,
            0,
            0,
            ids[f::PLANES],
            cancel,
            false,
        )?;
        let h = |i| u::handle(&self.arena, ids[i]);
        f::launch(
            &self.arena,
            f::Kernel::Positions,
            [1, 1, 1],
            vec![h(f::POS)?],
            info,
        )?;
        crate::strata_qsa::launch(
            &self.arena,
            crate::strata_qsa::Kernel::Norm,
            [k::KVH, count, 1],
            vec![
                h(f::KEY)?,
                self.front_public("blk.48.attn_k_norm.weight")?,
                h(f::POS)?,
                h(f::ROTATED)?,
                h(f::STATUS)?,
            ],
            [
                count as u32,
                start as u32,
                k::CONTEXT as u32,
                k::KV as u32,
                k::KV as u32,
                k::KVH as u32,
                k::KVH as u32,
                k::HD as u32,
                0,
                EPS.to_bits(),
                crate::qsa_gpu::THETA.powf(-2. / 64.).to_bits(),
                0,
            ],
        )?;
        u::status_batch(
            &self.arena,
            ids[f::STATUS],
            fault_id,
            count,
            0,
            k::KVH,
            k::KVH,
        )?;
        f::launch(
            &self.arena,
            f::Kernel::Append,
            [4, count, 1],
            vec![
                h(f::ROTATED)?,
                h(f::VALUE)?,
                u::handle(&self.arena, self.id(B::Keys))?,
                u::handle(&self.arena, self.id(B::Values))?,
                u::handle(&self.arena, fault_id)?,
            ],
            info,
        )?;
        self.tick(31 + embedding_pages + if queued_inputs { 2 } else { 0 })?;
        self.guard_native_front()?;
        u::cancel(cancel)?;
        prepared.queued()
    }
    #[allow(clippy::too_many_arguments)]
    fn native_front_project(
        &mut self,
        name: &'static str,
        input: BufferId,
        out: BufferId,
        count: usize,
        input_first: usize,
        output_first: usize,
        planes: BufferId,
        cancel: &AtomicBool,
        prepare: bool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        let spec = *matrices_for(self.graph.profile())
            .iter()
            .find(|s| s.name == name)
            .ok_or(Error::InvalidFormat("native MTP front projection"))?;
        if spec.dtype != quant_matvec::Q8_0 {
            return Err(Error::InvalidBufferLength);
        }
        let spans =
            matmul::BatchSpans::new(count, 0, input_first, spec.columns, output_first, spec.rows);
        if prepare {
            crate::strata_dense::prepare(
                self.client()?,
                spec.columns,
                u::handle(&self.arena, input)?,
                u::handle(&self.arena, planes)?,
                u::handle(&self.arena, self.id(B::Fault))?,
                &spans,
            )?;
        }
        for page in self.public()?.pages(name)? {
            crate::strata_dense::project(
                self.client()?,
                spec.dtype,
                page.rows,
                spec.columns,
                page.handle.clone(),
                u::handle(&self.arena, planes)?,
                u::handle(&self.arena, out)?,
                u::handle(&self.arena, self.id(B::Fault))?,
                output_first + page.first,
                spec.rows,
                count,
            )?;
        }
        self.tick(self.public()?.pages(name)?.len() + usize::from(prepare))?;
        Ok(())
    }
    /// Unused fused-only join after three actual completed drafts. Caller
    /// supplies the already accepted target prefix; this does not accept it.
    pub(crate) fn repair_accepted_cache_fused_four(
        &mut self,
        target_start: usize,
        accepted_tokens: &[u32],
        preceding_carry: &[f32],
        accepted_carries: &[f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        if !self.fused_sweeps
            || self.poisoned
            || self.closed
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
            || self.graph.profile() != MtpProfile::SelfContainedQ8
            || self.public_english_vocab.is_some()
        {
            return Err(Error::InvalidFormat("MTP fused four repair workspace"));
        }
        let plan = AcceptedCacheRepair::new_fused_four(
            self.position,
            target_start,
            self.last_forward
                .as_ref()
                .is_some_and(|stamp| stamp.valid(self.binding, self.position)),
            self.fused_sweeps,
            accepted_tokens,
            preceding_carry,
            accepted_carries,
        )?;
        self.repair_accepted_cache_plan(
            plan,
            target_start,
            accepted_tokens,
            preceding_carry,
            accepted_carries,
            cancel,
            MAX_FUSED_FOUR_REPAIR_DISPATCHES,
        )
    }
    /// Join original cache rewind/catch-up for one actually accepted target
    /// prefix. Caller retains target numerical authority; there is no proposal
    /// or prefix acceptance here. Cursor/ledger credit follows one full gate.
    pub fn repair_accepted_cache(
        &mut self,
        target_start: usize,
        accepted_tokens: &[u32],
        preceding_carry: &[f32],
        accepted_carries: &[f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        u::cancel(cancel)?;
        if self.poisoned
            || self.closed
            || self.retired.load(Ordering::Acquire)
            || self.live_outputs.load(Ordering::Acquire) != 0
            || !(self.graph.profile() == MtpProfile::SelfContainedQ8
                || crate::strata_intel_core::enabled()
                    && self.graph.profile() == MtpProfile::StrataQ2)
            || self.public_english_vocab.is_some()
        {
            return Err(Error::InvalidFormat("MTP accepted repair workspace"));
        }
        let plan = AcceptedCacheRepair::new_mode(
            self.position,
            target_start,
            self.last_forward
                .as_ref()
                .is_some_and(|stamp| stamp.valid(self.binding, self.position)),
            accepted_tokens,
            preceding_carry,
            accepted_carries,
            crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2,
        )?;
        self.repair_accepted_cache_plan(
            plan,
            target_start,
            accepted_tokens,
            preceding_carry,
            accepted_carries,
            cancel,
            MAX_TOKEN_DISPATCHES,
        )
    }
    // Both wrappers retain this original queue, catch/unwind and wiping order.
    // The limit is applied to actual tick accounting before any publication.
    #[allow(clippy::too_many_arguments)]
    fn repair_accepted_cache_plan(
        &mut self,
        plan: AcceptedCacheRepair,
        target_start: usize,
        accepted_tokens: &[u32],
        preceding_carry: &[f32],
        accepted_carries: &[f32],
        cancel: &AtomicBool,
        dispatch_limit: usize,
    ) -> Result<()> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.verify_source()?;
            self.physical()?;
            self.ledger
                .observe_before(sample(self.client()?)?)
                .map_err(error)?;
            self.last_forward.take();
            self.last_front.take();
            self.q8_head_policy_locked = true;
            self.poisoned = true;
            self.dispatches = 0;
            if plan.rewind {
                self.clear_scratch()?;
                self.rewind_cache(self.position, target_start)?;
                self.guard_private(target_start)?;
            }
            if crate::strata_intel_core::enabled() && self.graph.profile() == MtpProfile::StrataQ2 {
                if plan.first_row < accepted_tokens.len() {
                    let mut carries = Zeroizing::new(Vec::with_capacity(
                        (accepted_tokens.len() - plan.first_row) * k::WIDE,
                    ));
                    for row in plan.first_row..accepted_tokens.len() {
                        carries.extend_from_slice(if row == 0 {
                            preceding_carry
                        } else {
                            &accepted_carries[(row - 1) * k::WIDE..row * k::WIDE]
                        });
                    }
                    let position = target_start + plan.first_row - 1;
                    self.enqueue_native_front(
                        &accepted_tokens[plan.first_row..],
                        &carries,
                        position,
                        cancel,
                    )?;
                    self.guard_private(target_start + accepted_tokens.len() - 1)?;
                }
            } else {
                for row in plan.first_row..accepted_tokens.len() {
                    u::cancel(cancel)?;
                    self.verify_source()?;
                    self.physical()?;
                    self.ledger
                        .observe_before(sample(self.client()?)?)
                        .map_err(error)?;
                    if plan.rewind || row != 0 {
                        self.clear_repair_scratch()?;
                    } else {
                        self.clear_scratch()?;
                    }
                    let position = target_start + row - 1;
                    let carry = if row == 0 {
                        preceding_carry
                    } else {
                        &accepted_carries[(row - 1) * k::WIDE..row * k::WIDE]
                    };
                    self.write(B::Token, &[accepted_tokens[row]])?;
                    let input =
                        Zeroizing::new(carry.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                    if input.capacity() * 4 > HOST_BYTES {
                        return Err(Error::LimitExceeded("MTP host carry"));
                    }
                    self.write(B::Carry, &input)?;
                    drop(input);
                    self.guard_private(position)?;
                    self.condition()?;
                    self.hc(B::Residual0, 0, false)?;
                    self.key_value_at(position)?;
                    self.guard_private(position + 1)?;
                    self.verify_source()?;
                }
            }
            if self.dispatches > dispatch_limit {
                return Err(Error::LimitExceeded("MTP accepted repair INFO records"));
            }
            self.check_fault()?;
            self.verify_source()?;
            u::cancel(cancel)?;
            let completed = completed_batched_repair(
                self.traffic.get(),
                accepted_tokens.len() - plan.first_row,
            )?;
            // All fallible input/source/fault/cancellation work precedes this
            // sole completed ledger fence. No per-cell credit/reset occurs.
            self.ledger
                .completed_fence(sample(self.client()?)?)
                .map_err(error)?;
            self.traffic.set(completed);
            self.position = plan.end;
            self.poisoned = false;
            Ok(())
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP accepted repair failure"))
            }
        }
    }
    pub fn rewind(&mut self, new_position: usize) -> Result<()> {
        self.last_forward.take();
        self.last_front.take();
        if self.closed || self.poisoned || self.retired.load(Ordering::Acquire) {
            return Err(Error::InvalidFormat("MTP resident poisoned"));
        }
        k::validate_rewind(self.position, new_position)?;
        self.physical()?;
        self.verify_source()?;
        self.poisoned = true;
        self.dispatches = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            self.clear_scratch()?;
            self.rewind_cache(self.position, new_position)?;
            self.guard_private(new_position)?;
            self.complete_fault()?;
            Ok(())
        }));
        match result {
            Ok(Ok(())) => {
                self.position = new_position;
                self.poisoned = false;
                Ok(())
            }
            Ok(Err(e)) => {
                self.fail();
                Err(e)
            }
            Err(_) => {
                self.fail();
                Err(Error::InvalidFormat("MTP resident rewind failure"))
            }
        }
    }
    pub fn close(&mut self) -> Result<CloseReceipt> {
        self.last_forward.take();
        self.last_front.take();
        if self.closed {
            return Err(Error::InvalidFormat("MTP resident already closed"));
        }
        if self.live_outputs.load(Ordering::Acquire) != 0 {
            return Err(Error::InvalidFormat(
                "MTP close requires guarded outputs released",
            ));
        }
        self.poisoned = true;
        self.arena.close().map_err(error)?;
        let private = self
            .arena
            .owned_zero_audit_receipt()
            .ok_or(Error::InvalidFormat("MTP complete erasure receipt"))?;
        let (expected_buffers, expected_bytes) = private_owned_geometry()?;
        if private.bytes != expected_bytes
            || private.buffers != expected_buffers
            || private.buffers
                != N as u64
                    + u64::from(self.draft_window.is_some())
                    + u64::from(self.dense_planes.is_some())
                    + if self.native_front.is_some() {
                        crate::strata_mtp_front::WORDS.len() as u64
                    } else {
                        0
                    }
        {
            return Err(Error::InvalidBufferLength);
        }
        let public_bytes = self.public.take().map(|p| p.bytes).unwrap_or(0);
        self.shared_target_weights.take();
        // Arena close completed all queued work and inspected every complete
        // private owner. Public model handles are now released, not private data.
        self.client()?.memory_cleanup();
        cubecl::future::block_on(self.client()?.sync()).map_err(error)?;
        self.audit_io()?;
        self.client()?.memory_cleanup();
        cubecl::future::block_on(self.client()?.sync()).map_err(error)?;
        let usage = self.client()?.memory_usage().map_err(error)?;
        if self.ledger.shared().is_none()
            && (usage.number_allocs != 0 || usage.bytes_in_use != 0 || usage.bytes_reserved != 0)
            || self.retired.load(Ordering::Acquire)
        {
            self.retired.store(true, Ordering::Release);
            return Err(Error::InvalidFormat("MTP resident physical recovery"));
        }
        self.ledger
            .completed_fence(usage.bytes_reserved)
            .map_err(error)?;
        self.closed = true;
        self.position.zeroize();
        self.dispatches.zeroize();
        self.ids.zeroize();
        Ok(CloseReceipt {
            private,
            public_bytes,
            main_reserved: usage.bytes_reserved,
        })
    }
}
impl Drop for ResidentMtp<'_, '_> {
    fn drop(&mut self) {
        if !self.closed
            && !matches!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.close())),
                Ok(Ok(_))
            )
        {
            self.retired.store(true, Ordering::Release);
        }
        self.position.zeroize();
        self.dispatches.zeroize();
        self.ids.zeroize();
    }
}
fn greedy_host(values: &[f32]) -> Result<u32> {
    if values.len() != k::VOCAB || values.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidBufferLength);
    }
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    Ok(best as u32)
}

fn decode_draft(packet: &[u32]) -> Result<(Zeroizing<Vec<f32>>, u32)> {
    let (h, t, _) = decode_draft_with_probability(packet, false)?;
    Ok((h, t))
}
fn decode_draft_with_probability(
    packet: &[u32],
    with_probability: bool,
) -> Result<(Zeroizing<Vec<f32>>, u32, Option<f32>)> {
    let owned = BufferLayout::new(WORDS[B::RawQuery as usize])
        .map_err(error)?
        .owned_words;
    if packet.len() != owned
        || packet[DRAFT_TOKEN] as usize >= k::VOCAB
        || packet[DRAFT_FAULT] != 0
        || packet[if with_probability {
            DRAFT_NATIVE_WORDS
        } else {
            DRAFT_WORDS
        }..]
            .iter()
            .any(|&x| x != 0)
        || with_probability && !(0. ..=1.).contains(&f32::from_bits(packet[DRAFT_PROBABILITY]))
        || packet[..k::WIDE]
            .iter()
            .any(|&x| !f32::from_bits(x).is_finite())
    {
        return Err(Error::InvalidFormat("MTP draft full owning packet"));
    }
    let mut hidden = Zeroizing::new(Vec::with_capacity(k::WIDE));
    hidden.extend(packet[..k::WIDE].iter().map(|&x| f32::from_bits(x)));
    if hidden.capacity() != k::WIDE {
        return Err(Error::LimitExceeded("MTP draft host capacity"));
    }
    Ok((
        hidden,
        packet[DRAFT_TOKEN],
        with_probability.then(|| f32::from_bits(packet[DRAFT_PROBABILITY])),
    ))
}

#[cfg(test)]
#[path = "resident_mtp_english_tests.rs"]
mod english_tests;
#[cfg(test)]
#[path = "resident_strata_mtp_tests.rs"]
mod strata_tests;
#[cfg(test)]
#[path = "resident_mtp_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "resident_mtp_batched_repair_tests.rs"]
mod batched_repair_tests;

#[cfg(test)]
#[path = "resident_mtp_fused_four_repair_tests.rs"]
mod fused_four_repair_tests;

#[cfg(test)]
#[test]
fn q8_head_broadcast_selection_rejects_used_or_non_q8_workspaces() {
    assert!(
        validate_q8_head_broadcast_selection(MtpProfile::SelfContainedQ8, false, false, 0, false,)
            .is_ok()
    );
    for (profile, closed, poisoned, position, locked) in [
        (MtpProfile::StrataQ2, false, false, 0, false),
        (MtpProfile::SelfContainedQ8, true, false, 0, false),
        (MtpProfile::SelfContainedQ8, false, true, 0, false),
        (MtpProfile::SelfContainedQ8, false, false, 1, false),
        // A rewind to zero must not reopen policy selection.
        (MtpProfile::SelfContainedQ8, false, false, 0, true),
    ] {
        assert!(
            validate_q8_head_broadcast_selection(profile, closed, poisoned, position, locked,)
                .is_err()
        );
    }
}

#[cfg(test)]
#[test]
fn q8_head_shared_scales_selection_is_exclusive_with_broadcast() {
    assert!(validate_q8_head_modes(false, false).is_ok());
    assert!(validate_q8_head_modes(true, false).is_ok());
    assert!(validate_q8_head_modes(false, true).is_ok());
    assert!(validate_q8_head_modes(true, true).is_err());
}

fn validate_input(token: u32, carry: &[f32], position: usize) -> Result<()> {
    k::validate_position(position)?;
    if token as usize >= k::VOCAB || carry.len() != k::WIDE || carry.iter().any(|x| !x.is_finite())
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}

#[cfg(test)]
mod native_confidence_tests {
    use super::*;
    #[test]
    fn native_confidence_draft_packet_has_checked_optional_probability() {
        let owned = BufferLayout::new(WORDS[B::RawQuery as usize])
            .unwrap()
            .owned_words;
        let mut p = vec![0; owned];
        p[DRAFT_TOKEN] = 17;
        for value in [0., 0.1, 1.] {
            p[DRAFT_PROBABILITY] = f32::to_bits(value);
            let (_, token, prob) = decode_draft_with_probability(&p, true).unwrap();
            assert_eq!(token, 17);
            assert_eq!(prob, Some(value));
        }
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            p[DRAFT_PROBABILITY] = value.to_bits();
            assert!(decode_draft_with_probability(&p, true).is_err());
        }
        p[DRAFT_PROBABILITY] = 0.5f32.to_bits();
        assert!(decode_draft(&p).is_err());
        p[owned - 1] = 1;
        assert!(decode_draft_with_probability(&p, true).is_err());
    }
    #[test]
    fn native_confidence_suffix_repair_can_outgrow_short_confidence_chain() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        let preceding = vec![0.; k::WIDE];
        for drafted in 1..crate::strata_port::verify_capacity() {
            for accepted in 1..=crate::strata_port::verify_capacity() {
                let tokens = vec![3; accepted];
                let carries = vec![0.; accepted * k::WIDE];
                let plan = AcceptedCacheRepair::new_mode(
                    32 + drafted - 1,
                    32,
                    true,
                    &tokens,
                    &preceding,
                    &carries,
                    true,
                )
                .unwrap();
                assert_eq!(plan.first_row, 1);
                assert!(plan.rewind);
                assert_eq!(plan.end, 32 + accepted - 1);
            }
        }
        assert!(
            AcceptedCacheRepair::new(32, 32, true, &[1, 2, 3], &preceding, &vec![0.; 3 * k::WIDE])
                .is_err()
        );
    }
}

#[cfg(test)]
mod context32768_tests {
    use super::*;
    #[test]
    fn target_shared_and_draft_owned_pages_are_admitted_at_selected_context() {
        if !crate::strata_intel_core::enabled() {
            return;
        }
        assert_eq!(k::CONTEXT, 32768);
        assert_eq!(
            BufferLayout::new(WORDS[B::Keys as usize])
                .unwrap()
                .owned_bytes,
            32 * 1024 * 1024
        );
        assert_eq!(
            BufferLayout::new(WORDS[B::Values as usize])
                .unwrap()
                .owned_bytes,
            32 * 1024 * 1024
        );
        assert_eq!(
            BufferLayout::new(WORDS[B::Attention as usize])
                .unwrap()
                .owned_bytes,
            16 * 1024 * 1024
        );
        assert!(WORDS[B::Packet as usize] >= k::CONTEXT);
        assert!(private_plan().unwrap() <= PRIVATE_BYTES as u64);
        assert_eq!(
            public_plan_for(MtpProfile::StrataQ2).unwrap(),
            public_bytes_for(MtpProfile::StrataQ2)
        );
        assert_eq!(public_plan().unwrap(), PUBLIC_BYTES);
        assert!(k::validate_position(32767).is_ok());
        assert!(k::validate_position(32768).is_err());
    }
}
