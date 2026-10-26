//! Original live prompt inventory, shared by carving and physical root pricing.
//! Preserve256-byte Alloc::take gaps; fixed roots round only to native2MiB.
use super::{allocation, groups};
pub(crate) const WORKING_BYTES: u64 = (allocation::DEFAULT_ROWS * 4 * 2560 * 2) as u64;
const PERSISTENT: [usize; 18] = [
    5120, 320, 160, 10240, 4, 1280, 1280, 10240, 2560, 2560, 2560, 4, 512, 640, 640, 2560, 1, 2560,
];
pub(crate) const PLE: [usize; 10] = [2560, 10240, 2560, 10240, 4, 10240, 4, 4, 4, 4];
pub(crate) const GDN: [usize; 7] = [10240, 6144, 10240, 6144, 96, 96, 48];
pub(crate) const fn persistent(capacity: usize) -> [usize; 19] {
    let mut out = [0; 19];
    let mut i = 0;
    while i < 18 {
        out[i] = PERSISTENT[i] * capacity;
        i += 1;
    }
    out[18] = 9 * 10240;
    out
}
pub(crate) const fn qsa(capacity: usize) -> [usize; 20] {
    [
        12288 * capacity,
        512 * capacity,
        512 * capacity,
        512 * capacity,
        128 * capacity,
        6144 * capacity,
        512 * capacity,
        512 * capacity,
        crate::strata_qsa::CACHE_WORDS * capacity,
        6144 * capacity,
        crate::strata_qsa::PARTIAL_WORDS * 32,
        crate::strata_qsa::STATS_WORDS * 32,
        (crate::qsa_gpu::MAX_CONTEXT / 4) * 128,
        crate::qsa_gpu::MAX_CONTEXT / 4,
        24 * capacity,
        2 * capacity,
        4 * capacity,
        (crate::qsa_gpu::MAX_CONTEXT / 4) * crate::strata_qsa::prompt::SELECT_BATCH,
        (crate::qsa_gpu::MAX_CONTEXT / 4) * crate::strata_qsa::prompt::SELECT_BATCH,
        2051 * capacity,
    ]
}
const fn end(words: &[usize], scale: usize) -> u64 {
    let mut cursor = WORKING_BYTES;
    let mut i = 0;
    while i < words.len() {
        cursor = (cursor + (words[i] * scale * 4) as u64 + 511) & !255;
        i += 1;
    }
    cursor
}
const fn unrounded_roots_for(layer:bool) -> [u64;3] {
    let n = allocation::DEFAULT_ROWS;
    let mut phase = end(&PLE, n);
    let gdn = end(&GDN, n);
    if gdn > phase {
        phase = gdn;
    }
    let qsa = end(&qsa(n), 1);
    if qsa > phase {
        phase = qsa;
    }
    let moe = WORKING_BYTES
        + groups::GU_BYTES
        + groups::PART_BYTES
        + 4 * (groups::RAW_WORDS + groups::MAP_WORDS + groups::PROB_WORDS + groups::STATUS_WORDS)
            as u64
        + 6 * 256
        + if layer {groups::LAYER_X_BYTES + 256} else {0};
    if moe > phase {
        phase = moe;
    }
    [end(&persistent(n), 1), phase, allocation::RAW_RING_BYTES]
}
pub(crate) const fn unrounded_roots()->[u64;3]{unrounded_roots_for(cfg!(feature="strata-layer-gather"))}
const fn root_sizes_for(layer:bool) -> [u64;3] {
    let sizes=unrounded_roots_for(layer);
    let granule = crate::private_arena::NATIVE_GRANULE as u64;
    [
        sizes[0].div_ceil(granule) * granule,
        sizes[1].div_ceil(granule) * granule,
        sizes[2],
    ]
}

pub(crate) const fn root_sizes()->[u64;3]{root_sizes_for(cfg!(feature="strata-layer-gather"))}
pub(crate) const fn layer_extra_bytes()->u64 {if cfg!(feature="strata-layer-gather") {root_sizes_for(true)[1]-root_sizes_for(false)[1]}else{0}}
