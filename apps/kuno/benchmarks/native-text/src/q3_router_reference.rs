//! Bounded numeric fixture for one approved public Q3 router position.
//! This module accepts no prompts/hidden states and is not an inference API.
//! The protected runner pins original artifact and executable hashes separately.

use kuno_text::ops;
use kuno_text::{Error, Result};
use std::io::{Read, Write};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

pub const POSITIONS: usize = 32;
pub const POSITION: usize = 23;
pub const LAYERS: usize = 48;
pub const EXPERTS: usize = 512;
pub const TOP_K: usize = 10;
pub const MODEL_BYTES: u64 = 89_986_353_824;
const MAGIC: &[u8; 8] = b"KQ3T0001";
const HEADER_FIELDS: [u32; 6] = [32, 23, 48, 512, 10, 24];
const HEADER_BYTES: usize = 8 + 6 * 4 + 8;
const RECORD_BYTES: usize = 4 + EXPERTS * 4 + TOP_K * 8 + 2 * 4;
pub const FILE_BYTES: usize = HEADER_BYTES + LAYERS * RECORD_BYTES;
pub const OWNER_HOST_CAP: usize = 128 * 1024;
pub const LOAD_HOST_CAP: usize = 256 * 1024;
const PARSE_SCRATCH_CAP: usize = 16 * 1024;
pub const COMPARE_SCRATCH_CAP: usize = 16 * 1024;

/// Numeric-only diagnostic for this fixed public fixture. Matching counts are
/// in 0..=10; probability differences compare equal selected rank positions.
/// No expert IDs, score vectors or hidden states are retained in this report.
pub struct Report {
    pub raw_gpu_valid: bool,
    pub cpu_membership_matches: usize,
    pub cpu_rank_matches: usize,
    pub ops_membership_matches: usize,
    pub ops_rank_matches: usize,
    pub max_score_difference: f64,
    pub max_probability_difference: f64,
    pub max_ops_probability_difference: f64,
    pub cpu_logit_margin: f32,
    pub gpu_logit_margin: f32,
    pub cpu_probability_margin: f32,
    pub gpu_probability_margin: f32,
    pub gpu_ops_logit_margin: f32,
}

fn err() -> Error {
    Error::InvalidFormat("fixed public Q3 router fixture rejected")
}

/// CPU graph's actual router result and normalized selected probabilities.
/// No Debug/Clone: these owners must never become arbitrary request diagnostics.
pub struct Layer {
    scores: Zeroizing<[f32; EXPERTS]>,
    selected: Zeroizing<[(u32, f32); TOP_K]>,
    logit_margin: f32,
    probability_margin: f32,
}
impl Layer {
    pub fn new(scores: &[f32], selected: &[(usize, f32)]) -> Result<Self> {
        if scores.len() != EXPERTS || selected.len() != TOP_K {
            return Err(err());
        }
        let canonical = ops::router_topk(scores, TOP_K, true).map_err(|_| err())?;
        if canonical
            .iter()
            .zip(selected)
            .any(|(&(id, p), &(actual_id, actual_p))| {
                id != actual_id || p.to_bits() != actual_p.to_bits()
            })
        {
            return Err(err());
        }
        // Selection ranks are based on rounded softmax probabilities, including
        // the original ascending-ID tie rule; sorting raw logits is different.
        let boundary = ops::router_topk(scores, TOP_K + 1, false).map_err(|_| err())?;
        let logit_margin = scores[boundary[TOP_K - 1].0] - scores[boundary[TOP_K].0];
        let probability_margin = boundary[TOP_K - 1].1 - boundary[TOP_K].1;
        if !logit_margin.is_finite() || !probability_margin.is_finite() || probability_margin < 0. {
            return Err(err());
        }
        let mut score_owner = Zeroizing::new([0.; EXPERTS]);
        score_owner.copy_from_slice(scores);
        let mut selected_owner = Zeroizing::new([(0, 0.); TOP_K]);
        for (destination, &(id, probability)) in selected_owner.iter_mut().zip(selected) {
            *destination = (u32::try_from(id).map_err(|_| err())?, probability);
        }
        Ok(Self {
            scores: score_owner,
            selected: selected_owner,
            logit_margin,
            probability_margin,
        })
    }
    pub fn scores(&self) -> &[f32; EXPERTS] {
        &self.scores
    }
    pub fn selected(&self) -> &[(u32, f32); TOP_K] {
        &self.selected
    }
    pub fn logit_margin(&self) -> f32 {
        self.logit_margin
    }
    pub fn probability_margin(&self) -> f32 {
        self.probability_margin
    }
    /// Diagnostic only: original GPU admission ranks raw logits, whereas the
    /// original CPU operation ranks rounded global softmax probabilities. Both
    /// comparisons are reported separately, so a rounded tie is not mistaken
    /// for a difference in router inputs or a new acceptance failure.
    pub fn compare(&self, actual_scores: &[f32], actual_selected: &[(u32, f32)]) -> Result<Report> {
        validate_raw_gpu(actual_scores, actual_selected)?;
        let actual_canonical =
            ops::router_topk_guarded(actual_scores, TOP_K, true).map_err(|_| err())?;
        let canonical = actual_canonical.as_slice();
        let matches = |expected: &[(u32, f32)]| {
            let membership = actual_selected
                .iter()
                .filter(|&&(id, _)| expected.iter().any(|&(other, _)| other == id))
                .count();
            let ranks = actual_selected
                .iter()
                .zip(expected)
                .filter(|&(&(id, _), &(other, _))| id == other)
                .count();
            (membership, ranks)
        };
        let (cpu_membership_matches, cpu_rank_matches) = matches(&*self.selected);
        let (ops_membership_matches, ops_rank_matches) = matches(canonical);
        let probability_difference = |expected: &[(u32, f32)]| {
            actual_selected
                .iter()
                .zip(expected)
                .map(|(&(_, actual), &(_, expected))| {
                    (f64::from(actual) - f64::from(expected)).abs()
                })
                .fold(0., f64::max)
        };
        let max_probability_difference = probability_difference(&*self.selected);
        let max_ops_probability_difference = probability_difference(canonical);
        // Release the bounded guarded-router scratch before the original
        // eleven-rank operation allocates its fixed-512 softmax/sort scratch.
        drop(actual_canonical);
        let boundary =
            Zeroizing::new(ops::router_topk(actual_scores, TOP_K + 1, false).map_err(|_| err())?);
        let boundary_owned = boundary
            .capacity()
            .checked_mul(std::mem::size_of::<(usize, f32)>())
            .ok_or(Error::ArithmeticOverflow)?;
        // softmax holds at most 512 f32s and the sort at most 512 usize IDs;
        // no source-sized matrix, expert weights or additional vectors exist.
        let scratch = EXPERTS
            .checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<usize>())
            .and_then(|n| n.checked_add(boundary_owned))
            .and_then(|n| n.checked_add(std::mem::size_of::<Report>()))
            .ok_or(Error::ArithmeticOverflow)?;
        if scratch > COMPARE_SCRATCH_CAP
            || ops::GUARDED_ROUTER_CAPACITY_BYTES + std::mem::size_of::<Report>()
                > COMPARE_SCRATCH_CAP
        {
            return Err(err());
        }
        let mut first_unselected = None;
        for (id, &score) in actual_scores.iter().enumerate() {
            if actual_selected
                .iter()
                .any(|&(chosen, _)| chosen as usize == id)
            {
                continue;
            }
            if first_unselected
                .is_none_or(|(old, previous)| score > previous || (score == previous && id < old))
            {
                first_unselected = Some((id, score));
            }
        }
        let gpu_logit_margin = actual_scores[actual_selected[TOP_K - 1].0 as usize]
            - first_unselected.ok_or_else(err)?.1;
        let gpu_ops_logit_margin =
            actual_scores[boundary[TOP_K - 1].0] - actual_scores[boundary[TOP_K].0];
        let gpu_probability_margin = boundary[TOP_K - 1].1 - boundary[TOP_K].1;
        if !gpu_logit_margin.is_finite()
            || !gpu_ops_logit_margin.is_finite()
            || !gpu_probability_margin.is_finite()
            || gpu_logit_margin < 0.
            || gpu_probability_margin < 0.
        {
            return Err(err());
        }
        Ok(Report {
            raw_gpu_valid: true,
            cpu_membership_matches,
            cpu_rank_matches,
            ops_membership_matches,
            ops_rank_matches,
            max_score_difference: actual_scores
                .iter()
                .zip(self.scores.iter())
                .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).abs())
                .fold(0., f64::max),
            max_probability_difference,
            max_ops_probability_difference,
            cpu_logit_margin: self.logit_margin,
            gpu_logit_margin,
            cpu_probability_margin: self.probability_margin,
            gpu_probability_margin,
            gpu_ops_logit_margin,
        })
    }
}

/// Same raw-logit, ascending-ID tie and probability checks as the original
/// Count1RouterInput validator. This does not enforce rounded CPU selection.
fn validate_raw_gpu(scores: &[f32], selected: &[(u32, f32)]) -> Result<()> {
    if scores.len() != EXPERTS
        || selected.len() != TOP_K
        || scores.iter().any(|value| !value.is_finite())
    {
        return Err(err());
    }
    let mut sum = 0.0_f32;
    for (rank, &(id, probability)) in selected.iter().enumerate() {
        let id = id as usize;
        if id >= EXPERTS
            || !probability.is_finite()
            || !(0. ..=1.).contains(&probability)
            || selected[..rank].iter().any(|&(old, _)| old as usize == id)
        {
            return Err(err());
        }
        if rank > 0 {
            let previous = selected[rank - 1].0 as usize;
            if scores[previous] < scores[id] || (scores[previous] == scores[id] && previous > id) {
                return Err(err());
            }
        }
        sum += probability;
    }
    if (sum - 1.).abs() > 2e-5 {
        return Err(err());
    }
    let last = selected[TOP_K - 1].0 as usize;
    for (id, &value) in scores.iter().enumerate() {
        if !selected.iter().any(|&(chosen, _)| chosen as usize == id)
            && (value > scores[last] || (value == scores[last] && id < last))
        {
            return Err(err());
        }
    }
    Ok(())
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.logit_margin.zeroize();
        self.probability_margin.zeroize();
    }
}

pub struct Reference {
    layers: Vec<Layer>,
}
impl Reference {
    pub fn new(layers: Vec<Layer>) -> Result<Self> {
        if layers.len() != LAYERS {
            return Err(err());
        }
        let out = Self { layers };
        if out.owned_bytes()? > OWNER_HOST_CAP {
            return Err(err());
        }
        Ok(out)
    }
    pub fn owned_bytes(&self) -> Result<usize> {
        self.layers
            .capacity()
            .checked_mul(std::mem::size_of::<Layer>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(Error::ArithmeticOverflow)
    }
    pub fn layer(&self, layer: usize) -> Result<&Layer> {
        self.layers.get(layer).ok_or_else(err)
    }
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        if file.metadata()?.len() != FILE_BYTES as u64 {
            return Err(err());
        }
        let mut bytes = Zeroizing::new(vec![0; FILE_BYTES]);
        file.read_exact(&mut bytes)?;
        if file.read(&mut [0; 1])? != 0 {
            return Err(err());
        }
        let out = Self::parse(&bytes)?;
        if out
            .owned_bytes()?
            .checked_add(bytes.capacity())
            .and_then(|n| n.checked_add(PARSE_SCRATCH_CAP))
            .is_none_or(|n| n > LOAD_HOST_CAP)
        {
            return Err(err());
        }
        Ok(out)
    }
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FILE_BYTES || &bytes[..8] != MAGIC {
            return Err(err());
        }
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        for (index, &field) in HEADER_FIELDS.iter().enumerate() {
            if word(8 + index * 4) != field {
                return Err(err());
            }
        }
        if u64::from_le_bytes(bytes[32..40].try_into().unwrap()) != MODEL_BYTES {
            return Err(err());
        }
        let mut layers = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            let mut cursor = HEADER_BYTES + layer * RECORD_BYTES;
            if word(cursor) != layer as u32 {
                return Err(err());
            }
            cursor += 4;
            let mut scores = Zeroizing::new([0.; EXPERTS]);
            for score in scores.iter_mut() {
                *score = f32::from_bits(word(cursor));
                cursor += 4;
            }
            let mut selected = Zeroizing::new([(0usize, 0f32); TOP_K]);
            for pair in selected.iter_mut() {
                *pair = (word(cursor) as usize, f32::from_bits(word(cursor + 4)));
                cursor += 8;
            }
            let record = Layer::new(&*scores, &*selected)?;
            if record.logit_margin.to_bits() != word(cursor)
                || record.probability_margin.to_bits() != word(cursor + 4)
            {
                return Err(err());
            }
            layers.push(record);
        }
        Self::new(layers)
    }
    fn encoded(&self) -> Zeroizing<Vec<u8>> {
        let mut bytes = Zeroizing::new(Vec::with_capacity(FILE_BYTES));
        bytes.extend_from_slice(MAGIC);
        for field in HEADER_FIELDS {
            bytes.extend_from_slice(&field.to_le_bytes());
        }
        bytes.extend_from_slice(&MODEL_BYTES.to_le_bytes());
        for (layer, record) in self.layers.iter().enumerate() {
            bytes.extend_from_slice(&(layer as u32).to_le_bytes());
            for score in record.scores.iter() {
                bytes.extend_from_slice(&score.to_le_bytes());
            }
            for &(id, probability) in record.selected.iter() {
                bytes.extend_from_slice(&id.to_le_bytes());
                bytes.extend_from_slice(&probability.to_le_bytes());
            }
            bytes.extend_from_slice(&record.logit_margin.to_le_bytes());
            bytes.extend_from_slice(&record.probability_margin.to_le_bytes());
        }
        bytes
    }
    /// Only the approved public CPU writer calls this; create_new prevents
    /// replacing an existing reference. No partial model-state file is written.
    pub fn write_new(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.encoded();
        if bytes.len() != FILE_BYTES {
            return Err(err());
        }
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Reference {
        Reference::new(
            (0..LAYERS)
                .map(|layer| {
                    let scores: Vec<_> = (0..EXPERTS)
                        .map(|expert| ((expert * 17 + layer) % EXPERTS) as f32 / 128.)
                        .collect();
                    let selected = ops::router_topk(&scores, TOP_K, true).unwrap();
                    Layer::new(&scores, &selected).unwrap()
                })
                .collect(),
        )
        .unwrap()
    }
    #[test]
    fn exact_fixture_roundtrip_preserves_all_scores_routes_and_boundary_margins() {
        let original = fixture();
        let bytes = original.encoded();
        assert_eq!(bytes.len(), FILE_BYTES);
        assert_eq!(FILE_BYTES, 102_760);
        assert!(original.owned_bytes().unwrap() <= OWNER_HOST_CAP);
        assert!(
            original.owned_bytes().unwrap() + bytes.capacity() + PARSE_SCRATCH_CAP <= LOAD_HOST_CAP
        );
        let restored = Reference::parse(&bytes).unwrap();
        for layer in 0..LAYERS {
            let a = original.layer(layer).unwrap();
            let b = restored.layer(layer).unwrap();
            assert_eq!(a.scores().map(f32::to_bits), b.scores().map(f32::to_bits));
            assert_eq!(
                a.selected().map(|(id, p)| (id, p.to_bits())),
                b.selected().map(|(id, p)| (id, p.to_bits())),
            );
            assert_eq!(a.logit_margin().to_bits(), b.logit_margin().to_bits());
            assert_eq!(
                a.probability_margin().to_bits(),
                b.probability_margin().to_bits()
            );
        }
        assert!(restored.layer(LAYERS).is_err());
    }
    #[test]
    fn strict_header_extent_and_every_layer_ordinal_are_required() {
        let bytes = fixture().encoded();
        for offset in [0, 8, 12, 16, 20, 24, 28, 32] {
            let mut bad = bytes.clone();
            bad[offset] ^= 1;
            assert!(Reference::parse(&bad).is_err());
        }
        for layer in 0..LAYERS {
            let mut bad = bytes.clone();
            bad[HEADER_BYTES + layer * RECORD_BYTES] ^= 1;
            assert!(Reference::parse(&bad).is_err());
        }
        assert!(Reference::parse(&bytes[..FILE_BYTES - 1]).is_err());
        let mut surplus = bytes.to_vec();
        surplus.push(0);
        assert!(Reference::parse(&surplus).is_err());
    }
    #[test]
    fn score_selected_id_probability_and_margin_corruption_are_rejected() {
        let bytes = fixture().encoded();
        let last = HEADER_BYTES + (LAYERS - 1) * RECORD_BYTES;
        for (offset, bits) in [
            (last + 4 + (EXPERTS - 1) * 4, f32::NAN.to_bits()),
            (last + 4 + (EXPERTS - 1) * 4, f32::INFINITY.to_bits()),
            (last + 4 + EXPERTS * 4, 512),
            (last + 4 + EXPERTS * 4 + 4, f32::NAN.to_bits()),
            (last + RECORD_BYTES - 8, f32::NAN.to_bits()),
            (last + RECORD_BYTES - 4, 1f32.to_bits()),
        ] {
            let mut bad = bytes.clone();
            bad[offset..offset + 4].copy_from_slice(&bits.to_le_bytes());
            assert!(Reference::parse(&bad).is_err());
        }
    }
    #[test]
    fn equal_softmax_probabilities_keep_original_ascending_id_ties() {
        let scores = [0.; EXPERTS];
        let selected = ops::router_topk(&scores, TOP_K, true).unwrap();
        let record = Layer::new(&scores, &selected).unwrap();
        assert_eq!(
            record.selected().map(|(id, _)| id),
            std::array::from_fn(|i| i as u32)
        );
        assert_eq!(record.logit_margin(), 0.);
        assert_eq!(record.probability_margin(), 0.);
        let mut wrong = selected;
        wrong.swap(0, 1);
        assert!(Layer::new(&scores, &wrong).is_err());
    }
    #[test]
    fn retained_vector_capacity_is_part_of_the_owner_bound() {
        let mut excessive = Vec::with_capacity(LAYERS * 3);
        excessive.extend(fixture().layers);
        assert!(Reference::new(excessive).is_err());
        assert!(Reference::new(Vec::new()).is_err());
    }
    fn cpu_layer(scores: &[f32; EXPERTS]) -> Layer {
        let selected = ops::router_topk(scores, TOP_K, true).unwrap();
        Layer::new(scores, &selected).unwrap()
    }
    fn gpu_from_cpu(scores: &[f32; EXPERTS]) -> Vec<(u32, f32)> {
        ops::router_topk(scores, TOP_K, true)
            .unwrap()
            .into_iter()
            .map(|(id, probability)| (id as u32, probability))
            .collect()
    }
    #[test]
    fn identical_routes_report_all_matches_and_zero_differences() {
        let scores = [0.; EXPERTS];
        let layer = cpu_layer(&scores);
        let report = layer.compare(&scores, layer.selected()).unwrap();
        assert!(report.raw_gpu_valid);
        assert_eq!(report.cpu_membership_matches, TOP_K);
        assert_eq!(report.cpu_rank_matches, TOP_K);
        assert_eq!(report.ops_membership_matches, TOP_K);
        assert_eq!(report.ops_rank_matches, TOP_K);
        assert_eq!(report.max_score_difference, 0.);
        assert_eq!(report.max_probability_difference, 0.);
        assert_eq!(report.max_ops_probability_difference, 0.);
        assert_eq!(report.cpu_logit_margin, 0.);
        assert_eq!(report.gpu_logit_margin, 0.);
        assert_eq!(report.cpu_probability_margin, 0.);
        assert_eq!(report.gpu_probability_margin, 0.);
        assert_eq!(report.gpu_ops_logit_margin, 0.);
        assert_eq!(COMPARE_SCRATCH_CAP, 16 * 1024);
        assert!(
            ops::GUARDED_ROUTER_CAPACITY_BYTES + std::mem::size_of::<Report>()
                <= COMPARE_SCRATCH_CAP
        );
    }
    #[test]
    fn rounded_softmax_cutoff_tie_is_distinguished_from_changed_inputs() {
        let mut scores = [-10.; EXPERTS];
        scores[..TOP_K - 1].fill(1.);
        scores[9] = 0.;
        scores[10] = 1e-8;
        let layer = cpu_layer(&scores);
        assert_eq!(layer.selected()[9].0, 9);
        assert_eq!(layer.probability_margin(), 0.);
        assert!(layer.logit_margin() < 0.);
        // Raw top-k legitimately picks the slightly larger score even though
        // the original CPU softmax has rounded these two probabilities equal.
        let mut selected = layer.selected().to_vec();
        selected[9].0 = 10;
        let report = layer.compare(&scores, &selected).unwrap();
        assert!(report.raw_gpu_valid);
        assert_eq!(report.cpu_membership_matches, TOP_K - 1);
        assert_eq!(report.cpu_rank_matches, TOP_K - 1);
        assert_eq!(report.ops_membership_matches, TOP_K - 1);
        assert_eq!(report.ops_rank_matches, TOP_K - 1);
        assert_eq!(report.max_score_difference, 0.);
        assert_eq!(report.max_probability_difference, 0.);
        assert_eq!(report.max_ops_probability_difference, 0.);
        assert!(report.gpu_logit_margin > 0.);
        assert!(report.gpu_ops_logit_margin < 0.);
        assert_eq!(report.gpu_probability_margin, 0.);
        // The rounded CPU route is not a valid raw-logit GPU route here.
        assert!(layer.compare(&scores, layer.selected()).is_err());
    }
    #[test]
    fn input_selection_divergence_and_rank_only_changes_have_distinct_counts() {
        let mut original = [-10.; EXPERTS];
        original[..9].fill(1.);
        original[9] = 0.5;
        original[10] = 0.25;
        let layer = cpu_layer(&original);
        let mut actual = original;
        actual[10] = 0.75;
        let report = layer.compare(&actual, &gpu_from_cpu(&actual)).unwrap();
        assert_eq!(report.cpu_membership_matches, 9);
        assert_eq!(report.cpu_rank_matches, 9);
        assert_eq!(report.ops_membership_matches, TOP_K);
        assert_eq!(report.ops_rank_matches, TOP_K);
        assert_eq!(report.max_score_difference, 0.5);
        assert!(report.max_probability_difference > 0.);
        assert_eq!(report.max_ops_probability_difference, 0.);
        assert_eq!(report.cpu_logit_margin, 0.25);
        assert_eq!(report.gpu_logit_margin, 0.25);
        assert!(report.gpu_probability_margin > 0.);
        actual = original;
        actual[1] = 1.125;
        let report = layer.compare(&actual, &gpu_from_cpu(&actual)).unwrap();
        assert_eq!(report.cpu_membership_matches, TOP_K);
        assert_eq!(report.cpu_rank_matches, TOP_K - 2);
        assert_eq!(report.ops_membership_matches, TOP_K);
        assert_eq!(report.ops_rank_matches, TOP_K);
    }
    #[test]
    fn compare_rejects_malformed_or_nonadmitted_raw_gpu_routes() {
        let scores = [0.; EXPERTS];
        let layer = cpu_layer(&scores);
        let selected = layer.selected();
        assert!(layer.compare(&scores[..EXPERTS - 1], selected).is_err());
        assert!(layer.compare(&scores, &selected[..TOP_K - 1]).is_err());
        for bad_score in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut bad = scores;
            bad[511] = bad_score;
            assert!(layer.compare(&bad, selected).is_err());
        }
        for (rank, id, probability) in [
            (0, 512, 0.1),
            (1, 0, 0.1),
            (9, 10, 0.1),
            (0, 0, -0.1),
            (0, 0, f32::NAN),
            (0, 0, f32::INFINITY),
            (0, 0, 0.2),
        ] {
            let mut bad = *selected;
            bad[rank] = (id, probability);
            assert!(layer.compare(&scores, &bad).is_err());
        }
        let mut bad = *selected;
        bad.swap(0, 1);
        assert!(layer.compare(&scores, &bad).is_err());
    }
}
