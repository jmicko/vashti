//! Fixed-public active-prefix controls through the original Validate/Select
//! kernels and private arena. No prompt scope, state receipt or speed claim.
use super::{Op, launch, prompt_active_blocks};
use crate::private_arena::PrivateArena;
use zeroize::Zeroizing;
type ProbeResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// QSA_ACTIVE_PREFIX_ORACLE_BEGIN
const BLOCKS: usize = 8192;
const STRIDE: usize = 2051;
const ROWS: usize = 4;
const CONTEXT: usize = BLOCKS * 4;

fn public_score(row: usize, block: usize) -> f32 {
    if block.is_multiple_of(47) {
        -0.0
    } else {
        ((block * 17 + row * 11) % 31) as f32 * 0.125
    }
}

/// Independent sort-and-expand oracle: whole blocks compete by descending
/// score, stable ties by ascending block; partial blocks cost their actual
/// visible cells. Only the last threshold block may be cut to the cell budget.
/// Selected cells are finally returned in chronological order.
fn expected_row(start: usize, row: usize) -> Vec<u32> {
    let visible = start + row + 1;
    let width = visible.min(STRIDE);
    let mut blocks: Vec<_> = (0..visible.div_ceil(4)).collect();
    blocks.sort_by(|&a, &b| {
        public_score(row, b)
            .partial_cmp(&public_score(row, a))
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    let mut selected = Vec::with_capacity(STRIDE);
    for block in blocks {
        let count = (visible - block * 4).min(4);
        let take = count.min(width - selected.len());
        selected.extend((block * 4..block * 4 + take).map(|v| v as u32));
        if selected.len() == width {
            break;
        }
    }
    selected.sort_unstable();
    selected.resize(STRIDE, CONTEXT as u32);
    selected
}

fn selected_matches(actual: &[u32], start: usize, count: usize, positive: bool) -> bool {
    if actual.len() < ROWS * STRIDE || !(1..=ROWS).contains(&count) {
        return false;
    }
    for row in 0..count {
        if positive && actual[row * STRIDE..(row + 1) * STRIDE] != expected_row(start, row) {
            return false;
        }
    }
    // Undispatched rows and all real owned padding stay zero, including on
    // negative controls where no particular live-row publication is promised.
    actual[count * STRIDE..].iter().all(|&word| word == 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn independent_weighted_oracle_preserves_ties_partial_blocks_and_chronology() {
        assert_eq!(
            &expected_row(0, 0)[..4],
            &[0, CONTEXT as u32, CONTEXT as u32, CONTEXT as u32]
        );
        for (start, count) in [(0, 3), (2050, 3), (4094, 3), (32764, 4)] {
            for row in 0..count {
                let expected = expected_row(start, row);
                let visible = start + row + 1;
                let width = visible.min(STRIDE);
                assert_eq!(expected.len(), STRIDE);
                assert!(expected[..width].windows(2).all(|v| v[0] < v[1]));
                assert!(expected[..width].iter().all(|&v| (v as usize) < visible));
                assert!(expected[width..].iter().all(|&v| v == CONTEXT as u32));
                // A deliberately simple cell-level ranking must agree with
                // weighted block selection, including the partial last block.
                let mut cells: Vec<_> = (0..visible).collect();
                cells.sort_by(|&a, &b| {
                    public_score(row, b / 4)
                        .partial_cmp(&public_score(row, a / 4))
                        .unwrap()
                        .then(a.cmp(&b))
                });
                cells.truncate(width);
                cells.sort_unstable();
                assert_eq!(
                    expected[..width],
                    cells.into_iter().map(|v| v as u32).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn complete_selected_owner_checker_rejects_wrong_live_sentinel_and_padding() {
        for (start, count) in [(0, 3), (2050, 3), (4094, 3), (32764, 4)] {
            let mut words = vec![0; 16384];
            for row in 0..count {
                words[row * STRIDE..(row + 1) * STRIDE].copy_from_slice(&expected_row(start, row));
            }
            assert!(selected_matches(&words, start, count, true));
            words[0] ^= 1;
            assert!(!selected_matches(&words, start, count, true));
            words[0] ^= 1;
            if start == 0 {
                words[1] = 0;
                assert!(!selected_matches(&words, start, count, true));
                words[1] = CONTEXT as u32;
            }
            *words.last_mut().unwrap() = 1;
            assert!(!selected_matches(&words, start, count, true));
            assert!(!selected_matches(&words, start, count, false));
        }
    }
}
// QSA_ACTIVE_PREFIX_ORACLE_END

fn read_owner(
    arena: &PrivateArena,
    id: crate::private_arena::BufferId,
) -> ProbeResult<Zeroizing<Vec<u32>>> {
    let layout = arena.layout(id)?;
    let words = crate::gr_batch_probe::read_native_public(arena, id)?;
    if words.len() != layout.owned_words {
        return Err("public QSA actual full owner read extent".into());
    }
    Ok(words)
}

#[derive(Clone, Copy, Debug)]
enum Control {
    Causal,
    UndersizedActive,
    VisibleNan,
}
fn check(
    arena: &mut PrivateArena,
    start: usize,
    count: usize,
    control: Control,
) -> ProbeResult<serde_json::Value> {
    // All unused source rows/columns are deliberately poisoned, including the
    // full rows beyond count. They are original complete owners, never a small
    // forged allocation used to avoid validating the true context strides.
    let mut scores = Zeroizing::new(vec![f32::NAN; ROWS * BLOCKS]);
    let mut status = Zeroizing::new(vec![u32::MAX; ROWS * BLOCKS]);
    for row in 0..count {
        for block in 0..(start + row + 1).div_ceil(4) {
            scores[row * BLOCKS + block] = public_score(row, block);
            status[row * BLOCKS + block] = 1;
        }
    }
    if matches!(control, Control::VisibleNan) {
        scores[0] = f32::NAN;
    }
    let mut active = prompt_active_blocks(start, count, BLOCKS)?;
    if matches!(control, Control::UndersizedActive) {
        active = (start + count).div_ceil(4) - 1;
    }
    // Deliberate fixed-public NaNs must reach the actual shader. The ordinary
    // float uploader correctly rejects nonfinite data before GPU work; retain
    // that production guard and use the original raw-word upload for this
    // diagnostic's exact NaN/negative-zero bits instead.
    let score_bits = Zeroizing::new(scores.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
    let scores = arena.upload_u32(&score_bits)?;
    let status = arena.upload_u32(&status)?;
    let selected = arena.allocate_zeroed(ROWS * STRIDE)?;
    let fault = arena.allocate_zeroed(1)?;
    let owners = [
        arena.handle(scores)?.clone(),
        arena.handle(status)?.clone(),
        arena.handle(selected)?.clone(),
        arena.handle(fault)?.clone(),
    ];
    let p = [
        count as u32,
        start as u32,
        BLOCKS as u32,
        STRIDE as u32,
        BLOCKS as u32,
        BLOCKS as u32,
        active as u32,
        0,
    ];
    // Direct original primitives avoid pretending that a synthetic test owns a
    // live prompt scope, committed state, capture key or model receipt.
    launch(arena, Op::Validate, count, owners.clone(), p)?;
    launch(arena, Op::Select, count, owners, p)?;
    let actual = read_owner(arena, selected)?;
    let faults = read_owner(arena, fault)?;
    let positive = matches!(control, Control::Causal);
    let live_mismatches = if positive && actual.len() >= count * STRIDE {
        (0..count)
            .map(|row| {
                actual[row * STRIDE..(row + 1) * STRIDE]
                    .iter()
                    .zip(expected_row(start, row))
                    .filter(|(actual, expected)| **actual != *expected)
                    .count()
            })
            .sum::<usize>()
    } else {
        0
    };
    let selected_padding_nonzero = actual
        .get(count * STRIDE..)
        .map(|padding| padding.iter().filter(|&&word| word != 0).count())
        .unwrap_or(usize::MAX);
    let fault_padding_nonzero = faults
        .get(1..)
        .map(|padding| padding.iter().filter(|&&word| word != 0).count())
        .unwrap_or(usize::MAX);
    // Fixed-public fixture coordinates and aggregate numeric counts only.
    // Never print selected indices, source arrays, request text or pointers.
    println!(
        "public_device_error qsa_active_prefix_numeric control={control:?} start={start} count={count} active={active} expected_fault={} actual_fault={:?} live_mismatches={live_mismatches} selected_padding_nonzero={selected_padding_nonzero} fault_padding_nonzero={fault_padding_nonzero} selected_owner_words={} fault_owner_words={}",
        u32::from(!positive),
        faults.first().copied(),
        actual.len(),
        faults.len()
    );
    if faults.first().copied() != Some(u32::from(!positive))
        || faults[1..].iter().any(|&word| word != 0)
        || !selected_matches(&actual, start, count, positive)
    {
        return Err("public QSA active-prefix actual primitive parity failed".into());
    }
    Ok(
        serde_json::json!({"start":start,"count":count,"blocks":BLOCKS,"stride":STRIDE,
        "active_blocks":active,"causal_positive":positive,"undersized_active":matches!(control,Control::UndersizedActive),
        "visible_nan":matches!(control,Control::VisibleNan),"fault":faults[0],"full_owner_and_tail_checked":true,
        "future_scores_and_status_poisoned":true,"actual_validate_and_select_kernels":true}),
    )
}

pub(crate) fn cases(arena: &mut PrivateArena) -> ProbeResult<Vec<serde_json::Value>> {
    let mut reports = Vec::new();
    for (start, count, control) in [
        (0, 3, Control::Causal),
        (2050, 3, Control::Causal),
        (4094, 3, Control::Causal),
        (32764, 4, Control::Causal),
        (4094, 3, Control::UndersizedActive),
        (4094, 3, Control::VisibleNan),
    ] {
        let attempt = check(arena, start, count, control);
        // Genuine drain/full-owner erase is required even if validation fails.
        // The existing worker still owns final close and owned-zero audit.
        arena.reset()?;
        reports.push(attempt?);
    }
    Ok(reports)
}
