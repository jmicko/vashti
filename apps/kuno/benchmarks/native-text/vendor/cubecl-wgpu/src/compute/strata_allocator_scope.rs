//! Original managed allocation pins for the precommit/draft interval. These
//! references prevent allocator reuse; they are not GPU completion authority.
use cubecl_runtime::memory_management::{ManagedMemoryBinding, ManagedMemoryId};
use std::collections::HashMap;

pub const HOST_BYTES: usize = 4 * 1024 * 1024;
#[derive(Default)]
pub(super) struct Pins {
    request: Option<u64>,
    owners: HashMap<ManagedMemoryId, ManagedMemoryBinding>,
}
impl core::fmt::Debug for Pins {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("OriginalAllocationPins")
    }
}
impl Pins {
    pub fn active(&self) -> bool {
        self.request.is_some()
    }
    pub fn begin(&mut self, request: u64) -> Result<(), &'static str> {
        if request == 0 || self.active() || !self.owners.is_empty() {
            return Err("native allocation scope already owned or invalid");
        }
        self.request = Some(request);
        Ok(())
    }
    pub fn retain(&mut self, owner: &ManagedMemoryBinding) -> Result<(), &'static str> {
        if !self.active() || self.owners.contains_key(&owner.allocation_id()) {
            return Ok(());
        }
        // Hash-table control/bucket overhead and both original handle types are
        // conservatively charged. Allocation failure leaves the scope retained.
        let entry = size_of::<(ManagedMemoryId, ManagedMemoryBinding)>() + 32;
        if (self.owners.len() + 1) * entry > HOST_BYTES / 2 {
            return Err("native allocation pin capacity");
        }
        self.owners
            .try_reserve(1)
            .map_err(|_| "native allocation pin metadata allocation")?;
        if self
            .owners
            .capacity()
            .checked_mul(entry)
            .is_none_or(|n| n > HOST_BYTES)
        {
            return Err("native allocation pin metadata capacity");
        }
        self.owners.insert(owner.allocation_id(), owner.clone());
        Ok(())
    }
    /// The stream calls this only AFTER actual global scope-end/terminal drains.
    pub fn release_joined(&mut self, request: u64) -> Result<(), &'static str> {
        if self.request != Some(request) {
            return Err("native allocation original scope mismatch");
        }
        self.owners.clear();
        self.request = None;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_runtime::memory_management::ManagedMemoryHandle;
    #[test]
    fn actual_descriptor_pins_prevent_free_and_deduplicate_views_until_original_release() {
        let mut p = Pins::default();
        let owner = ManagedMemoryHandle::new();
        let monitor = owner.clone();
        let binding = owner.binding();
        p.retain(&binding).unwrap();
        assert!(p.owners.is_empty());
        p.begin(19).unwrap();
        p.retain(&binding).unwrap();
        p.retain(&binding.clone()).unwrap();
        assert_eq!(p.owners.len(), 1);
        drop(binding);
        assert!(!monitor.is_free());
        assert!(p.begin(20).is_err());
        assert!(p.release_joined(20).is_err());
        assert!(!monitor.is_free());
        p.release_joined(19).unwrap();
        assert!(monitor.is_free());
        assert!(p.release_joined(19).is_err());
        assert!(p.begin(0).is_err());
    }
}
