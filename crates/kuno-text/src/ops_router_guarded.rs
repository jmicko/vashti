//! Bounded, allocation-free router for the native MAX8 coordinator.
//!
//! This independently owns only one token's router scratch/result; its input
//! remains the caller's guarded borrowed logits. The existing scalar router
//! API is unchanged. Arithmetic/rounding/order match `super::router_topk`.
//! Clearing covers these API-visible fixed fields, including inactive slots;
//! it does not establish erasure of compiler registers or opaque host copies.

use super::{OpsError, checked_f32, finite};
use zeroize::{Zeroize, Zeroizing};

pub const GUARDED_ROUTER_MAX_EXPERTS: usize = 512;
pub const GUARDED_ROUTER_MAX_TOP_K: usize = 10;

struct Scratch {
    probabilities: [f32; GUARDED_ROUTER_MAX_EXPERTS],
    ids: [usize; GUARDED_ROUTER_MAX_EXPERTS],
    maximum: f64,
    sum: f64,
    denominator: f64,
}
impl Scratch {
    fn empty() -> Self {
        Self {
            probabilities: [0.0; GUARDED_ROUTER_MAX_EXPERTS],
            ids: [0; GUARDED_ROUTER_MAX_EXPERTS],
            maximum: 0.0,
            sum: 0.0,
            denominator: 0.0,
        }
    }
}
impl Zeroize for Scratch {
    fn zeroize(&mut self) {
        self.probabilities.zeroize();
        self.ids.zeroize();
        self.maximum.zeroize();
        self.sum.zeroize();
        self.denominator.zeroize();
        #[cfg(test)]
        witness(
            0,
            self.probabilities.iter().all(|x| x.to_bits() == 0)
                && self.ids.iter().all(|&x| x == 0)
                && self.maximum.to_bits() == 0
                && self.sum.to_bits() == 0
                && self.denominator.to_bits() == 0,
        );
    }
}

/// Borrowed guard permits safe inspection of the same backing fields after
/// drop in behavioral fixtures. Production guards exist before private fill.
struct ScopedWipe<'a, T: Zeroize>(&'a mut T);
impl<T: Zeroize> Drop for ScopedWipe<'_, T> {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

struct Selection {
    // u32 expert IDs are exact for the admitted <=512 experts and directly
    // match token_waves' packet format. Every rank, including zero, is kept.
    ranks: [(u32, f32); GUARDED_ROUTER_MAX_TOP_K],
    count: usize,
}
impl Zeroize for Selection {
    fn zeroize(&mut self) {
        self.ranks.zeroize();
        self.count.zeroize();
        #[cfg(test)]
        witness(
            1,
            self.count == 0
                && self
                    .ranks
                    .iter()
                    .all(|&(id, p)| id == 0 && p.to_bits() == 0),
        );
    }
}

/// Exact fixed-field capacity, independent of active logits/top-k count. No
/// private Vec or allocator growth exists in this helper. Caller logits and
/// implementation-dependent call-stack/sort temporaries are separate owners.
pub const GUARDED_ROUTER_CAPACITY_BYTES: usize =
    std::mem::size_of::<Scratch>() + std::mem::size_of::<Selection>();

/// Private top-k owner: no Clone/Debug/raw mutable export or serialization.
/// Drop clears every rank and the count, including the unused rank suffix.
pub struct GuardedRouterSelection {
    data: Zeroizing<Selection>,
}
impl GuardedRouterSelection {
    pub fn as_slice(&self) -> &[(u32, f32)] {
        &self.data.ranks[..self.data.count]
    }
}

/// Original f64 softmax with f32 exponential/probability round points, original
/// total_cmp + ascending expert-ID ties, then original selected-sum clamp and
/// f32 normalized result. No expert pruning, activation quantization or extra
/// renormalization. Only bounded native geometry is newly constrained here.
pub fn router_topk_guarded(
    logits: &[f32],
    k: usize,
    normalize: bool,
) -> Result<GuardedRouterSelection, OpsError> {
    if logits.is_empty()
        || logits.len() > GUARDED_ROUTER_MAX_EXPERTS
        || k == 0
        || k > logits.len()
        || k > GUARDED_ROUTER_MAX_TOP_K
    {
        return Err(OpsError::InvalidShape("guarded router top k"));
    }
    finite(logits, "softmax logits")?;
    let mut backing = Scratch::empty();
    let scratch = ScopedWipe(&mut backing);
    let mut result = GuardedRouterSelection {
        data: Zeroizing::new(Selection {
            ranks: [(0, 0.0); GUARDED_ROUTER_MAX_TOP_K],
            count: 0,
        }),
    };
    // Maximum is computed in f32 exactly as the scalar oracle, then widened.
    scratch.0.maximum = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    for (i, &logit) in logits.iter().enumerate() {
        scratch.0.probabilities[i] = (f64::from(logit) - scratch.0.maximum).exp() as f32;
        scratch.0.sum += f64::from(scratch.0.probabilities[i]);
        scratch.0.ids[i] = i;
    }
    for probability in &mut scratch.0.probabilities[..logits.len()] {
        *probability = checked_f32(
            f64::from(*probability) / scratch.0.sum,
            "softmax probability",
        )?;
    }
    scratch.0.ids[..logits.len()].sort_unstable_by(|&a, &b| {
        scratch.0.probabilities[b]
            .total_cmp(&scratch.0.probabilities[a])
            .then(a.cmp(&b))
    });
    scratch.0.denominator = if normalize {
        scratch.0.ids[..k]
            .iter()
            .map(|&i| f64::from(scratch.0.probabilities[i]))
            .sum::<f64>()
            .max(2_f64.powi(-14))
    } else {
        1.0
    };
    for (rank, &id) in scratch.0.ids[..k].iter().enumerate() {
        result.data.ranks[rank] = (
            id as u32,
            (f64::from(scratch.0.probabilities[id]) / scratch.0.denominator) as f32,
        );
    }
    result.data.count = k;
    Ok(result)
}

#[cfg(test)]
thread_local! {
    static WITNESS: std::cell::Cell<([usize; 2], bool)> = const {
        std::cell::Cell::new(([0; 2], true))
    };
}
#[cfg(test)]
fn witness(kind: usize, clean: bool) {
    WITNESS.with(|state| {
        let (mut counts, prior) = state.get();
        counts[kind] += 1;
        state.set((counts, prior && clean));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deliberately separate reference: uses a Vec, explicit global probability
    // loops and stable ordering. Only synthetic public fixtures use this oracle.
    fn independent(logits: &[f32], k: usize, normalize: bool) -> Vec<(usize, f32)> {
        let mut max = f32::NEG_INFINITY;
        for &x in logits {
            max = max.max(x);
        }
        let mut p: Vec<f32> = logits
            .iter()
            .map(|&x| (f64::from(x) - f64::from(max)).exp() as f32)
            .collect();
        let mut total = 0_f64;
        for &x in &p {
            total += f64::from(x);
        }
        for x in &mut p {
            *x = (f64::from(*x) / total) as f32;
        }
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        let mut selected_sum = 0_f64;
        for &i in &ids[..k] {
            selected_sum += f64::from(p[i]);
        }
        let denominator = if normalize {
            selected_sum.max(2_f64.powi(-14))
        } else {
            1.0
        };
        ids[..k]
            .iter()
            .map(|&i| (i, (f64::from(p[i]) / denominator) as f32))
            .collect()
    }
    fn parity(logits: &[f32], k: usize) {
        for normalize in [false, true] {
            let got = router_topk_guarded(logits, k, normalize).unwrap();
            let old = super::super::router_topk(logits, k, normalize).unwrap();
            let reference = independent(logits, k, normalize);
            assert_eq!(got.as_slice().len(), k);
            for (rank, &(id, p)) in got.as_slice().iter().enumerate() {
                assert_eq!(id as usize, reference[rank].0);
                assert_eq!(p.to_bits(), reference[rank].1.to_bits());
                assert_eq!(id as usize, old[rank].0);
                assert_eq!(p.to_bits(), old[rank].1.to_bits());
            }
        }
    }

    #[test]
    fn independent_and_original_bit_parity_cover_rounding_ties_and_geometry() {
        for n in [1, 3, 5, 8, 10, 17, 127, 511, 512] {
            let fixtures = [
                vec![0.0; n],
                (0..n)
                    .map(|i| if i % 2 == 0 { -0.0 } else { 0.0 })
                    .collect(),
                (0..n)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
                    .collect(),
                (0..n)
                    .map(|i| f32::from_bits(0x3f800000 + (i % 5) as u32))
                    .collect(),
                (0..n)
                    .map(|i| if i % 3 == 0 { f32::MAX } else { -f32::MAX })
                    .collect(),
            ];
            for logits in fixtures {
                parity(&logits, n.min(GUARDED_ROUTER_MAX_TOP_K));
                parity(&logits, 1);
            }
        }
    }

    #[test]
    fn all_ten_ranks_include_zero_probability_and_exact_expert_ties() {
        let mut logits = [-f32::MAX; GUARDED_ROUTER_MAX_EXPERTS];
        logits[17] = f32::MAX;
        let selected = router_topk_guarded(&logits, 10, true).unwrap();
        assert_eq!(selected.as_slice()[0], (17, 1.0));
        assert_eq!(selected.as_slice()[1..].len(), 9);
        for (rank, &(id, p)) in selected.as_slice()[1..].iter().enumerate() {
            assert_eq!(id, rank as u32);
            assert_eq!(p.to_bits(), 0);
        }
        parity(&logits, 10);
    }

    #[test]
    fn invalid_bounds_and_nonfinite_inputs_return_only_static_errors() {
        for (logits, k) in [
            (&[][..], 1),
            (&[0.0][..], 0),
            (&[0.0][..], 2),
            (&[0.0; 11][..], 11),
            (&[0.0; 513][..], 10),
        ] {
            assert!(matches!(
                router_topk_guarded(logits, k, true),
                Err(OpsError::InvalidShape(_))
            ));
        }
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(matches!(
                router_topk_guarded(&[0.0, value], 1, true),
                Err(OpsError::NonFinite("softmax logits"))
            ));
        }
    }

    #[test]
    fn fixed_capacity_is_bounded_and_does_not_depend_on_active_count() {
        assert_eq!(
            std::mem::size_of::<Selection>(),
            80 + std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::size_of::<Scratch>(),
            512 * (4 + std::mem::size_of::<usize>()) + 24
        );
        assert_eq!(
            GUARDED_ROUTER_CAPACITY_BYTES,
            std::mem::size_of::<Scratch>() + std::mem::size_of::<Selection>()
        );
        const { assert!(GUARDED_ROUTER_CAPACITY_BYTES < 8192) };
        for n in [1, 3, 5, 8, 512] {
            let result = router_topk_guarded(&vec![0.0; n], n.min(10), true).unwrap();
            assert_eq!(
                std::mem::size_of_val(&result),
                std::mem::size_of::<Selection>()
            );
        }
    }

    fn fill_all(s: &mut Scratch) {
        s.probabilities.fill(-3.75);
        s.ids.fill(511);
        s.maximum = 10.0;
        s.sum = 7.0;
        s.denominator = 4.0;
    }
    fn clean(s: &Scratch) -> bool {
        s.probabilities.iter().all(|v| v.to_bits() == 0)
            && s.ids.iter().all(|&v| v == 0)
            && s.maximum.to_bits() == 0
            && s.sum.to_bits() == 0
            && s.denominator.to_bits() == 0
    }
    #[test]
    fn same_scratch_guard_clears_full_inactive_fields_on_success_error_and_unwind() {
        for outcome in 0..3 {
            let mut backing = Scratch::empty();
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let protected = ScopedWipe(&mut backing);
                fill_all(protected.0);
                match outcome {
                    0 => Ok(()),
                    1 => Err(OpsError::NonFinite("public injected failure")),
                    _ => panic!("public injected unwind"),
                }
            }));
            assert!(clean(&backing));
            match outcome {
                0 => assert!(attempt.unwrap().is_ok()),
                1 => assert!(attempt.unwrap().is_err()),
                _ => assert!(attempt.is_err()),
            }
        }
    }

    #[test]
    fn actual_result_drop_and_abandon_unwind_scrub_every_rank_and_count() {
        for unwind in [false, true] {
            WITNESS.with(|w| w.set(([0; 2], true)));
            let attempt = std::panic::catch_unwind(|| {
                let result = router_topk_guarded(&[1.0; 512], 10, true).unwrap();
                assert_eq!(result.as_slice().len(), 10);
                if unwind {
                    panic!("public abandon result");
                }
            });
            assert_eq!(attempt.is_err(), unwind);
            WITNESS.with(|w| assert_eq!(w.get(), ([1, 1], true)));
        }
        // Cover unused returned ranks as well as the active prefix.
        let mut selection = Selection {
            ranks: [(511, -0.25); 10],
            count: 1,
        };
        {
            let _guard = ScopedWipe(&mut selection);
        }
        assert_eq!(selection.count, 0);
        assert!(
            selection
                .ranks
                .iter()
                .all(|&(id, p)| id == 0 && p.to_bits() == 0)
        );
    }
}
