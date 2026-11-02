//! Bounded known-input prefix geometry. No GPU/state/commit authority lives
//! here; actual original model output and both completed cursors remain checked
//! by the controller. MAX8 is the separately reserved known-prefix shape;
//! ordinary Prefill/final-input/generation admission remains unchanged.
use kuno_text::{Error, Result};
use zeroize::Zeroizing;

pub(super) const PREFIX_LIMIT: usize = 64;
pub(super) const WINDOW_ROWS: usize = crate::strata_port::verify_capacity();
/// Prefixes up to this length use the 8-row verification windows; longer ones
/// take batched prefill. Measured on the B70 (warm): 14 tokens 0.45 vs 0.64 s,
/// 17 tokens 0.49 vs 0.53 s, 26 tokens 0.78 vs 0.73 s, 45 tokens 1.20 vs 0.85 s.
/// PREFIX_LIMIT stays the reserved capacity of the windowed path.
pub(super) const ROUTE_LIMIT: usize = 20;
const _: () = assert!(ROUTE_LIMIT <= PREFIX_LIMIT);
pub(super) struct Plan(Zeroizing<usize>);
impl Plan {
    pub(super) fn new(prompt: usize) -> Option<Self> {
        prompt
            .checked_sub(1)
            .filter(|&prefix| prefix <= ROUTE_LIMIT)?;
        Some(Self(Zeroizing::new(prompt)))
    }
    pub(super) fn prefix(&self) -> usize {
        *self.0 - 1
    }
    pub(super) fn chunks(&self) -> impl Iterator<Item = std::ops::Range<usize>> {
        (0..self.prefix())
            .step_by(WINDOW_ROWS)
            .map(|start| start..(start + WINDOW_ROWS).min(self.prefix()))
    }
    pub(super) fn inputs<'a>(
        &self,
        prompt: &'a [u32],
        chunk: std::ops::Range<usize>,
    ) -> Result<(&'a [u32], &'a [u32])> {
        if prompt.len() != *self.0
            || chunk.is_empty()
            || !chunk.start.is_multiple_of(WINDOW_ROWS)
            || chunk.end > self.prefix()
            || chunk
                .start
                .checked_add(WINDOW_ROWS)
                .is_none_or(|end| chunk.end != end.min(self.prefix()))
        {
            return Err(Error::InvalidFormat("native short prompt teacher window"));
        }
        Ok((
            &prompt[chunk.clone()],
            &prompt[chunk.start + 1..chunk.end + 1],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn known_short_prefix_covers_max_eight_tails_and_exact_threshold() {
        for prefix in [0usize, 1, 2, 3, 4, 7, 8, 9, 19, 20] {
            let prompt: Vec<u32> = (0..=prefix as u32).collect();
            let plan = Plan::new(prompt.len()).unwrap();
            assert_eq!(plan.prefix(), prefix);
            let mut seen = Vec::new();
            let mut teachers = Vec::new();
            for chunk in plan.chunks() {
                assert_eq!(chunk.start, seen.len());
                assert!((1..=WINDOW_ROWS).contains(&chunk.len()));
                let (inputs, next) = plan.inputs(&prompt, chunk).unwrap();
                seen.extend_from_slice(inputs);
                teachers.extend_from_slice(next);
            }
            assert_eq!(seen, prompt[..prefix]);
            assert_eq!(teachers, prompt[1..]);
            assert_eq!(&prompt[plan.prefix()..], &[prefix as u32]);
        }
        // The threshold counts FRESH PREFIX rows, excluding the final normal
        // count1 input; a 21-token prompt is the last windowed one.
        assert!(Plan::new(ROUTE_LIMIT + 1).is_some());
        assert!(Plan::new(ROUTE_LIMIT + 2).is_none());
        assert!(Plan::new(0).is_none());
        assert!(Plan::new(usize::MAX).is_none());
    }
    #[test]
    fn wrong_teacher_slices_and_window_boundaries_are_rejected() {
        for length in [1usize, 3, 4, 8, 9, 20, 21] {
            let plan = Plan::new(length).unwrap();
            let prompt = vec![7; length];
            for chunk in plan.chunks() {
                let mut wrong = prompt.clone();
                wrong.push(7);
                assert!(plan.inputs(&wrong, chunk.clone()).is_err());
                assert!(plan.inputs(&prompt[..length - 1], chunk).is_err());
            }
            for bad in [
                0..0,
                1..2,
                0..WINDOW_ROWS + 1,
                0..usize::MAX,
                usize::MAX..usize::MAX,
            ] {
                assert!(plan.inputs(&prompt, bad).is_err());
            }
        }
    }
}
