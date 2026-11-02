//! Connected whole-chunk prompt QSA. Native full Q/K/V/index projections are
//! already in the scoped region. All other data/flags/selection are carved;
//! only authoritative cache/positions/pooled and fault come from the stage.
use super::*;

#[derive(Clone, Copy)]
struct ScoreBatch {
    first: usize,
    rows: usize,
    start: usize,
}
fn score_batches(
    start: usize,
    count: usize,
    context: usize,
) -> Result<impl Iterator<Item = ScoreBatch>> {
    if count == 0
        || count > crate::strata_prompt::groups::CAPACITY
        || context > crate::qsa_gpu::MAX_CONTEXT
        || start.checked_add(count).is_none_or(|end| end > context)
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok((0..count)
        .step_by(super::SELECT_BATCH)
        .map(move |first| ScoreBatch {
            first,
            rows: (count - first).min(super::SELECT_BATCH),
            start: start + first,
        }))
}
fn score_query_view(root: &Handle, batch: ScoreBatch) -> Result<Handle> {
    let row = 512 * 4;
    let end = batch
        .first
        .checked_add(batch.rows)
        .and_then(|n| n.checked_mul(row))
        .ok_or(Error::ArithmeticOverflow)?;
    if batch.rows == 0
        || batch.rows > super::SELECT_BATCH
        || !batch.first.is_multiple_of(super::SELECT_BATCH)
        || end as u64 > root.size_in_used()
    {
        return Err(Error::InvalidBufferLength);
    }
    Ok(root
        .clone()
        .offset_start((batch.first * row) as u64)
        .offset_end(root.size_in_used() - end as u64))
}

/// Four RMS gammas; cache, positions, pooled/dead, fault, tail/snapshot/window.
/// This enqueues the full family and packed output, not a state acceptance.
pub(crate) fn run(
    arena: &PrivateArena,
    p: [u32; 12],
    gamma: [Handle; 4],
    owners: [Handle; 7],
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
    let [cache, positions, pooled, fault, tail, snapshot, window] = owners;
    for (h, n) in [&cache, &positions, &pooled, &fault].into_iter().zip([
        capacity * CACHE_WORDS,
        capacity,
        (blocks + 1) * 128,
        1,
    ]) {
        span(h, n)?;
    }
    for (h, n) in [&tail, &snapshot, &window].into_iter().zip([
        INDEX_TAIL_WORDS,
        INDEX_TAIL_WORDS,
        INDEX_WINDOW_WORDS,
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
        (17, super::SELECT_BATCH * blocks),
        (18, super::SELECT_BATCH * blocks),
        (19, count * 2051),
    ] {
        span(&q[i], n)?;
    }
    let mut family_timing = crate::strata_prefill_profile::Timer::for_public_arena(arena)?;
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
                q[4].clone(),
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
            q[4].clone(),
        ],
        p,
    )?;
    pool(arena, p, [tail.clone(), q[4].clone()])?;
    super::super::launch(
        arena,
        Kernel::PromptTail,
        [3, 1, 1],
        vec![q[4].clone(), tail, snapshot, window, fault.clone()],
        p,
    )?;
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
    family_timing.mark("qsa_family_producers")?;
    // Exactly the upstream256-query score/selection loop, including C-1's
    // last-query active grid clamp. Omitted score slots are never consumed:
    // selection validates its reach and reads only each query's causal blocks.
    // Both score/flag strides and selected owners stay at full capacity.
    // Only IQ is rebased
    // (2048-byte row alignment); the2051-word selected stride stays on its full
    // owner and is addressed by a row index, never an unaligned descriptor.
    for batch in score_batches(start, count, capacity)? {
        let ScoreBatch {
            first,
            rows,
            start: query_start,
        } = batch;
        let mut sp = p;
        sp[0] = rows as u32;
        sp[1] = query_start as u32;
        sp[3] = (blocks * 128) as u32;
        sp[4] = blocks as u32;
        sp[5] = blocks as u32;
        let active_blocks = selection::prompt_active_blocks(query_start, rows, blocks)?;
        let iq = score_query_view(&q[7], batch)?;
        super::super::launch(
            arena,
            Kernel::PromptScoreFault,
            [active_blocks.div_ceil(8), rows, 1],
            vec![
                pooled.clone(),
                iq,
                q[17].clone(),
                q[18].clone(),
                pooled.clone(),
                fault.clone(),
            ],
            sp,
        )?;
        selection::select_active_prefix(
            arena,
            [q[17].clone(), q[18].clone(), q[19].clone(), fault.clone()],
            query_start,
            rows,
            blocks,
            2051,
            first,
            active_blocks,
        )?;
    }
    family_timing.mark("qsa_family_score_select")?;
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
    family_timing.mark("qsa_family_attention_merge")?;
    let packed = crate::strata_prompt::activation_stage()?;
    span(&packed, count * 3072)?;
    launch(
        arena,
        Op::GateFault,
        [3072 / 128, count, 1],
        vec![q[0].clone(), q[9].clone(), packed, fault],
        p,
    )?;
    family_timing.mark("qsa_family_gate")
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

#[cfg(test)]
mod score_batch_tests {
    use super::*;
    use cubecl_common::stream_id::StreamId;
    #[test]
    fn actual_query_views_cover_all_rows_and_keep_global_selection_rows_aligned_logically() {
        let bytes = crate::strata_prompt::groups::CAPACITY * 512 * 4;
        let parent = Handle::new(StreamId { value: 711 }, bytes as u64 + 8192);
        let query = parent.clone().offset_start(4096).offset_end(4096);
        for count in [1, 8, 31, 32, 255, 256, 257, 2049, 4096] {
            for start in [0, 13, crate::qsa_gpu::MAX_CONTEXT - count] {
                let mut next = 0;
                for b in score_batches(start, count, crate::qsa_gpu::MAX_CONTEXT).unwrap() {
                    assert_eq!(b.first, next);
                    assert_eq!(b.start, start + next);
                    let active = selection::prompt_active_blocks(
                        b.start,
                        b.rows,
                        crate::qsa_gpu::MAX_CONTEXT / 4,
                    )
                    .unwrap();
                    assert!((b.start + b.rows).div_ceil(4) <= active);
                    assert!(active <= crate::qsa_gpu::MAX_CONTEXT / 4);
                    let view = score_query_view(&query, b).unwrap();
                    assert_eq!(view.size(), parent.size());
                    assert_eq!(
                        view.offset_start.unwrap(),
                        4096 + (b.first * 512 * 4) as u64
                    );
                    assert_eq!(view.size_in_used(), (b.rows * 512 * 4) as u64);
                    assert!(view.offset_start.unwrap().is_multiple_of(256));
                    //2051-word rows stay on the complete selected owner instead
                    // of rebasing an often unaligned Vulkan descriptor.
                    assert!(
                        (b.first + b.rows) * 2051 <= crate::strata_prompt::groups::CAPACITY * 2051
                    );
                    next += b.rows;
                }
                assert_eq!(next, count);
            }
        }
        assert!(score_batches(0, 0, 32768).is_err());
        assert!(score_batches(32767, 2, 32768).is_err());
        assert!(
            score_query_view(
                &query,
                ScoreBatch {
                    first: 1,
                    rows: 256,
                    start: 1
                }
            )
            .is_err()
        );
        assert!(
            score_query_view(
                &query,
                ScoreBatch {
                    first: crate::strata_prompt::groups::CAPACITY,
                    rows: 1,
                    start: crate::strata_prompt::groups::CAPACITY
                }
            )
            .is_err()
        );
    }
    #[test]
    fn every_score_selection_record_is_priced_at_full_prompt_capacity() {
        let n = crate::strata_prompt::groups::CAPACITY;
        let calls = 3 * score_batches(0, n, 32768).unwrap().count() + 2 * n.div_ceil(32) + 18;
        assert_eq!(super::super::INFO_BYTES, 12 * calls as u64 * 2048);
        assert_eq!(super::super::INFO_BYTES, 8_847_360);
    }
}
