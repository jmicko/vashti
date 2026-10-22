//! Unique immutable program identities, never shader-byte or request identity.
use std::sync::atomic::{AtomicU64, Ordering};
pub(super) fn next(counter: &AtomicU64) -> Result<u64, &'static str> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            if current == 0 {
                None
            } else {
                current.checked_add(1)
            }
        })
        .map_err(|_| "native pipeline identity exhaustion")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identities_are_distinct_and_exhaustion_never_reuses_a_live_key() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(next(&counter), Ok(u64::MAX - 1));
        assert!(next(&counter).is_err());
        assert!(next(&counter).is_err());
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        let counter = AtomicU64::new(1);
        assert_eq!(next(&counter), Ok(1));
        assert_eq!(next(&counter), Ok(2));
    }
    #[test]
    fn reserved_zero_is_rejected_without_publishing_or_advancing_a_key() {
        let counter = AtomicU64::new(0);
        assert!(next(&counter).is_err());
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
