//! Connected whole-chunk prompt QSA. Native full Q/K/V/index projections are
//! already in the scoped region. All other data/flags/selection are carved;
//! only authoritative cache/positions/pooled and fault come from the stage.
use super::*;

/// Four RMS gammas; authoritative cache, positions, pooled/dead, model fault.
/// This enqueues the full family and packed output, not a state acceptance.
pub(crate) fn run(
    arena: &PrivateArena,
    p: [u32; 12],
    gamma: [Handle; 4],
    owners: [Handle; 4],
) -> Result<()> {
    let count = p[0] as usize;
    let start = p[1] as usize;
    let capacity = p[2] as usize;
    if !(1..=crate::strata_prompt::groups::CAPACITY).contains(&count)
        || !(4..=crate::qsa_gpu::MAX_CONTEXT).contains(&capacity)
        || !capacity.is_multiple_of(4)
        || start.checked_add(count).is_none_or(|end| end > capacity)
        || p[11] != 0
        || !f32::from_bits(p[9]).is_finite()
        || f32::from_bits(p[9]) <= 0.
        || !f32::from_bits(p[10]).is_finite()
        || f32::from_bits(p[10]) <= 0.
    {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let blocks = capacity / 4;
    let [cache, positions, pooled, fault] = owners;
    for (h, n) in [&cache, &positions, &pooled, &fault].into_iter().zip([
        capacity * CACHE_WORDS,
        capacity,
        (blocks + 1) * 128,
        1,
    ]) {
        span(h, n)?;
    }
    for (h, n) in gamma.iter().zip([256, 256, 128, 128]) {
        span(h, n)?;
    }
    if fault.size_in_used() != 4 {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    let q = crate::strata_prompt::qsa_workspace()?;
    for (i, n) in [12288, 512, 512, 512, 128, 6144, 512, 512, CACHE_WORDS, 6144]
        .into_iter()
        .enumerate()
    {
        span(&q[i], count * n)?;
    }
    span(&q[10], 32 * PARTIAL_WORDS)?;
    span(&q[11], 32 * STATS_WORDS)?;
    for (i, n) in [
        (12, blocks * 128),
        (13, blocks),
        (14, count * 24),
        (15, count * 2),
        (16, count * 4),
        (17, count * blocks),
        (18, count * blocks),
        (19, count * 2051),
    ] {
        span(&q[i], n)?;
    }
    launch(
        arena,
        Op::Positions,
        [count.div_ceil(64), 1, 1],
        vec![positions.clone()],
        p,
    )?;
    launch(
        arena,
        Op::SplitFault,
        [12288 / 64, count, 1],
        vec![q[0].clone(), q[5].clone(), fault.clone()],
        p,
    )?;
    for (input, output, heads, width, g, flag) in [
        (5, 5, 24, 256, 0, 14),
        (1, 6, 2, 256, 1, 15),
        (3, 7, 4, 128, 2, 16),
    ] {
        let mut np = p;
        np[3] = (heads * width) as u32;
        np[4] = np[3];
        np[5] = heads as u32;
        np[6] = heads as u32;
        np[7] = width as u32;
        np[8] = 0;
        norm_rope(
            arena,
            np,
            [
                q[input].clone(),
                gamma[g].clone(),
                positions.clone(),
                q[output].clone(),
                q[flag].clone(),
            ],
        )?;
        join_flags(arena, p, q[flag].clone(), fault.clone(), 0, count, heads)?;
    }
    super::super::launch(
        arena,
        Kernel::PromptPackFault,
        [super::super::PACK_GROUPS, count, 1],
        vec![
            q[6].clone(),
            q[2].clone(),
            q[4].clone(),
            q[8].clone(),
            fault.clone(),
        ],
        p,
    )?;
    if start == 0 {
        selection::initialize(
            arena,
            [
                q[8].clone(),
                gamma[3].clone(),
                pooled.clone(),
                fault.clone(),
            ],
            capacity,
        )?;
    }
    super::super::launch(
        arena,
        Kernel::PromptAppendFault,
        [CACHE_WORDS.div_ceil(64), count, 1],
        vec![
            q[8].clone(),
            cache.clone(),
            positions.clone(),
            fault.clone(),
        ],
        p,
    )?;
    pool(arena, p, [cache.clone(), q[8].clone()])?;
    let mut np = p;
    np[3] = (blocks * 128) as u32;
    np[4] = np[3];
    np[5] = blocks as u32;
    np[6] = blocks as u32;
    np[7] = 128;
    np[8] = 1;
    launch(
        arena,
        Op::PoolNorm,
        [blocks, 1, 1],
        vec![
            q[12].clone(),
            gamma[3].clone(),
            pooled.clone(),
            q[13].clone(),
        ],
        np,
    )?;
    launch(
        arena,
        Op::PoolRope,
        [blocks, 1, 1],
        vec![positions, pooled.clone(), q[13].clone()],
        np,
    )?;
    join_flags(
        arena,
        p,
        q[13].clone(),
        fault.clone(),
        start / 4,
        1,
        (start + count) / 4 - start / 4,
    )?;
    let mut sp = p;
    sp[3] = (blocks * 128) as u32;
    sp[4] = blocks as u32;
    sp[5] = blocks as u32;
    super::super::launch(
        arena,
        Kernel::PromptScoreFault,
        [blocks.div_ceil(8), count, 1],
        vec![
            pooled.clone(),
            q[7].clone(),
            q[17].clone(),
            q[18].clone(),
            pooled,
            fault.clone(),
        ],
        sp,
    )?;
    selection::select(
        arena,
        [q[17].clone(), q[18].clone(), q[19].clone(), fault.clone()],
        start,
        count,
        blocks,
        2051,
    )?;
    for first in (0..count).step_by(32) {
        let rows = (count - first).min(32);
        let mut ap = p;
        ap[0] = rows as u32;
        ap[11] = first as u32;
        super::super::launch(
            arena,
            Kernel::PromptAttentionFault,
            [CHUNKS, 2, rows],
            vec![
                q[5].clone(),
                cache.clone(),
                q[8].clone(),
                q[19].clone(),
                q[10].clone(),
                q[11].clone(),
                fault.clone(),
            ],
            ap,
        )?;
        super::super::launch(
            arena,
            Kernel::PromptMergeFault,
            [24, rows, 1],
            vec![q[10].clone(), q[11].clone(), q[9].clone(), fault.clone()],
            ap,
        )?;
    }
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, count * 3072)?;
    launch(
        arena,
        Op::GateFault,
        [3072 / 64, count, 1],
        vec![q[0].clone(), q[9].clone(), packed, fault],
        p,
    )
}
fn join_flags(
    arena: &PrivateArena,
    mut p: [u32; 12],
    source: Handle,
    fault: Handle,
    first: usize,
    rows: usize,
    width: usize,
) -> Result<()> {
    if width == 0 {
        return Ok(());
    }
    span(&source, first + rows * width)?;
    span(&fault, 1)?;
    p[0] = rows as u32;
    p[3] = first as u32;
    p[4] = width as u32;
    p[5] = width as u32;
    launch(
        arena,
        Op::JoinFlags,
        [(rows * width).div_ceil(64), 1, 1],
        vec![source, fault],
        p,
    )
}
/// Transitional MAX8 verifier status views only. Full prompt users call run
/// directly and consume the original model fault; no per-value maps are needed.
pub(crate) fn compatibility(
    arena: &PrivateArena,
    mut p: [u32; 12],
    status: Handle,
    fault: Handle,
    width: usize,
    mode: u32,
) -> Result<()> {
    if !(1..=8).contains(&p[0]) || width == 0 || mode > 3 {
        return Err(crate::strata_prompt::invalid_buffer_length());
    }
    span(&status, p[0] as usize * width)?;
    span(&fault, 1)?;
    p[3] = width as u32;
    p[4] = mode;
    p[5] = width as u32;
    launch(
        arena,
        Op::Compatibility,
        [width.div_ceil(64), p[0] as usize, 1],
        vec![fault, status],
        p,
    )
}
