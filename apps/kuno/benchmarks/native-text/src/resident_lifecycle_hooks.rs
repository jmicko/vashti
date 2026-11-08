//! Fixed-public laboratory injection capabilities. Never armed by defaults,
//! environment variables, request payloads or the scalar sequence API.
use kuno_text::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Site {
    LateCombineError,
    PleSecondProofError,
    PleSecondProofPanic,
    FirstGdnCommitCancel,
}
#[derive(Default)]
pub(super) struct Hooks {
    armed: Option<Site>,
    observed: Option<Site>,
}
impl Hooks {
    pub(super) fn is_unarmed(&self) -> bool {
        self.armed.is_none() && self.observed.is_none()
    }
    pub(super) fn arm(&mut self, site: Site) -> Result<()> {
        if self.armed.is_some() || self.observed.is_some() {
            return Err(Error::InvalidFormat("public lifecycle hook already used"));
        }
        self.armed = Some(site);
        Ok(())
    }
    pub(super) fn observed(&self, site: Site) -> bool {
        self.observed == Some(site) && self.armed.is_none()
    }
    pub(super) fn late_combine(&mut self, layer: usize) -> Result<()> {
        if layer == 47 && self.armed == Some(Site::LateCombineError) {
            self.take(Site::LateCombineError);
            return Err(Error::InvalidFormat("public late combine injection"));
        }
        Ok(())
    }
    pub(super) fn before_second_proof(&mut self) -> Result<()> {
        match self.armed {
            Some(Site::PleSecondProofError) => {
                self.take(Site::PleSecondProofError);
                Err(Error::InvalidFormat("public PLE proof injection"))
            }
            Some(Site::PleSecondProofPanic) => {
                self.take(Site::PleSecondProofPanic);
                panic!("fixed public PLE proof panic");
            }
            _ => Ok(()),
        }
    }
    pub(super) fn after_gdn_commit(&mut self, layer: usize, cancel: &AtomicBool) {
        if layer == 0 && self.armed == Some(Site::FirstGdnCommitCancel) {
            self.take(Site::FirstGdnCommitCancel);
            cancel.store(true, Ordering::Release);
        }
    }
    fn take(&mut self, site: Site) {
        self.armed = None;
        self.observed = Some(site);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_hook_cannot_inject_and_fixed_sites_are_one_shot() {
        let cancel = AtomicBool::new(false);
        let mut h = Hooks::default();
        for layer in 0..48 {
            h.late_combine(layer).unwrap();
            h.after_gdn_commit(layer, &cancel);
        }
        h.before_second_proof().unwrap();
        assert!(!cancel.load(Ordering::Acquire));
        h.arm(Site::LateCombineError).unwrap();
        h.late_combine(46).unwrap();
        assert!(h.late_combine(47).is_err());
        assert!(h.observed(Site::LateCombineError));
        h.late_combine(47).unwrap();
        assert!(h.arm(Site::PleSecondProofError).is_err());
    }
    #[test]
    fn error_panic_and_cancel_mark_the_exact_consumed_site() {
        for site in [Site::PleSecondProofError, Site::PleSecondProofPanic] {
            let mut h = Hooks::default();
            h.arm(site).unwrap();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.before_second_proof()));
            assert!(matches!(result, Ok(Err(_))) || result.is_err());
            assert!(h.observed(site));
            h.before_second_proof().unwrap();
        }
        let mut h = Hooks::default();
        h.arm(Site::FirstGdnCommitCancel).unwrap();
        let cancel = AtomicBool::new(false);
        h.after_gdn_commit(1, &cancel);
        assert!(!cancel.load(Ordering::Acquire));
        h.after_gdn_commit(0, &cancel);
        assert!(cancel.load(Ordering::Acquire));
        assert!(h.observed(Site::FirstGdnCommitCancel));
    }
}
