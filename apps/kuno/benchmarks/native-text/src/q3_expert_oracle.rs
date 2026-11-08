//! Numeric-only fixed-public diagnostic for the actual Q3 GPU expert rows.
//! Inputs and selected expert IDs stay with guarded caller owners. This module
//! does not execute a model, authorize publication, or accept model numerics.
use kuno_text::{Error, Result};
use zeroize::Zeroizing;

pub(super) const POSITION: u64 = 23;
pub(super) const HIDDEN: usize = 2560;
pub(super) const RANKS: usize = 10;
pub(super) const GPU_WORDS: usize = RANKS * HIDDEN;
// Both the raw prefix lease and copied f32 vector are admitted at the next
// power-of-two page, plus comparator/caller frame metadata. Original CPU jobs
// finish and release scratch before this readback is created.
pub(super) const READ_HOST: usize = 2 * 131072 + 4096;

#[derive(Default)]
pub(super) struct Report {
    pub ranks: usize,
    pub values: usize,
    pub different_bits: usize,
    pub max_abs_error: f32,
    pub max_scaled_error: f32,
    pub first_different_rank: Option<usize>,
    pub first_different_column: Option<usize>,
}

fn err() -> Error {
    Error::InvalidFormat("public Q3 actual expert oracle shape or values")
}

/// GPU partial rows are addressed by original rank, while CPU output rows use
/// compact selected-GPU order. Never compare compact ordinals as GPU ranks.
pub(super) fn compare(
    original: &[usize; RANKS],
    gpu_ids: &[usize],
    gpu: &[f32],
    cpu: &[f32],
) -> Result<Report> {
    if gpu_ids.is_empty()
        || gpu_ids.len() > RANKS
        || gpu.len() != GPU_WORDS
        || cpu.len() != gpu_ids.len() * HIDDEN
        || original
            .iter()
            .enumerate()
            .any(|(rank, &id)| id >= 512 || original[..rank].contains(&id))
        || gpu.iter().chain(cpu).any(|value| !value.is_finite())
    {
        return Err(err());
    }
    let mut ranks = Zeroizing::new([0usize; RANKS]);
    for (ordinal, id) in gpu_ids.iter().enumerate() {
        if gpu_ids[..ordinal].contains(id) {
            return Err(err());
        }
        ranks[ordinal] = original
            .iter()
            .position(|candidate| candidate == id)
            .ok_or_else(err)?;
    }
    let mut report = Report {
        ranks: gpu_ids.len(),
        values: cpu.len(),
        ..Report::default()
    };
    for ordinal in 0..gpu_ids.len() {
        let rank = ranks[ordinal];
        for column in 0..HIDDEN {
            let actual = gpu[rank * HIDDEN + column];
            let reference = cpu[ordinal * HIDDEN + column];
            let error = (actual - reference).abs();
            // Diagnostic scale only, using the existing expert probe equation.
            // No mismatch is hidden, repaired or accepted by this report.
            report.max_abs_error = report.max_abs_error.max(error);
            report.max_scaled_error = report
                .max_scaled_error
                .max(error / (1e-4 + 3e-4 * reference.abs()));
            if actual.to_bits() != reference.to_bits() {
                report.different_bits += 1;
                if report.first_different_rank.is_none() {
                    report.first_different_rank = Some(rank);
                    report.first_different_column = Some(column);
                }
            }
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    const IDS: [usize; RANKS] = [511, 71, 0, 143, 255, 72, 14, 1, 15, 2];
    fn fixture(selected: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let gpu: Vec<_> = (0..GPU_WORDS)
            .map(|index| (index / HIDDEN) as f32 + (index % HIDDEN) as f32 / 4096.)
            .collect();
        let cpu = selected
            .iter()
            .flat_map(|id| {
                let rank = IDS.iter().position(|candidate| candidate == id).unwrap();
                gpu[rank * HIDDEN..(rank + 1) * HIDDEN].iter().copied()
            })
            .collect();
        (gpu, cpu)
    }
    #[test]
    fn compact_noncontiguous_rows_compare_against_original_rank_and_last_column() {
        let selected = [IDS[9], IDS[2], IDS[5]];
        let (gpu, mut cpu) = fixture(&selected);
        let report = compare(&IDS, &selected, &gpu, &cpu).unwrap();
        assert_eq!(
            (report.ranks, report.values, report.different_bits),
            (3, 3 * HIDDEN, 0)
        );
        cpu[2 * HIDDEN - 1] += 0.125;
        let report = compare(&IDS, &selected, &gpu, &cpu).unwrap();
        assert_eq!(
            (
                report.different_bits,
                report.first_different_rank,
                report.first_different_column
            ),
            (1, Some(2), Some(HIDDEN - 1))
        );
        assert_eq!(report.max_abs_error, 0.125);
        assert!(report.max_scaled_error > 1.);
    }
    #[test]
    fn every_rank_is_observed_and_unselected_rows_are_not_false_comparisons() {
        let (mut gpu, cpu) = fixture(&IDS);
        for rank in 0..RANKS {
            gpu[rank * HIDDEN + HIDDEN - 1] += 0.03125;
        }
        let report = compare(&IDS, &IDS, &gpu, &cpu).unwrap();
        assert_eq!(report.different_bits, RANKS);
        let (mut gpu, cpu) = fixture(&[IDS[7]]);
        gpu[0] = 1_000_000.;
        assert_eq!(
            compare(&IDS, &[IDS[7]], &gpu, &cpu).unwrap().different_bits,
            0
        );
        gpu[7 * HIDDEN] = -0.;
        let mut cpu = cpu;
        cpu[0] = 0.;
        let report = compare(&IDS, &[IDS[7]], &gpu, &cpu).unwrap();
        assert_eq!(report.different_bits, 1);
        assert_eq!(report.max_abs_error, 0.);
    }
    #[test]
    fn malformed_owners_selection_and_nonfinite_values_are_rejected() {
        let (gpu, cpu) = fixture(&IDS);
        for (a, b) in [
            (&gpu[..GPU_WORDS - 1], &cpu[..]),
            (&gpu[..], &cpu[..GPU_WORDS - 1]),
        ] {
            assert!(compare(&IDS, &IDS, a, b).is_err());
        }
        assert!(compare(&IDS, &[], &gpu, &[]).is_err());
        assert!(compare(&IDS, &[511, 511], &gpu, &cpu[..2 * HIDDEN]).is_err());
        assert!(compare(&IDS, &[510], &gpu, &cpu[..HIDDEN]).is_err());
        let mut original = IDS;
        original[0] = original[1];
        assert!(compare(&original, &IDS, &gpu, &cpu).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = gpu.clone();
            bad[GPU_WORDS - 1] = value;
            assert!(compare(&IDS, &IDS, &bad, &cpu).is_err());
            let mut bad = cpu.clone();
            bad[GPU_WORDS - 1] = value;
            assert!(compare(&IDS, &IDS, &gpu, &bad).is_err());
        }
        assert_eq!(READ_HOST, 266240);
    }
}
