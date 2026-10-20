//! Capture-local exact buffer identity lookup, protected by one real Wgpu
//! destruction lock per phase. Never persisted in a recording or across calls.
use super::{WgpuResource, fault};
use ash::vk;
use cubecl_core::server::ServerError;

pub(super) struct Keys {
    buffers: Vec<wgpu::Buffer>,
    bytes: usize,
    guard_budget: usize,
}
impl Keys {
    pub fn new(resources: &[WgpuResource], budget: usize) -> Result<Self, ServerError> {
        if resources.is_empty() || resources.len() > 65536 {
            return Err(fault("native capture root capacity"));
        }
        if !resources.windows(2).all(|w| w[0].buffer < w[1].buffer) {
            return Err(fault("native capture original root identity order"));
        }
        // Includes core's Arc + HAL pointer vectors, temporary real BufferIds,
        // all headers/rank data and alignment. Rechecked by both actual APIs.
        let guard_budget = resources
            .len()
            .checked_mul(32)
            .and_then(|n| n.checked_add(4096))
            .ok_or_else(|| fault("native capture guard metadata overflow"))?;
        let minimum = resources
            .len()
            .checked_mul(size_of::<wgpu::Buffer>())
            .and_then(|n| n.checked_add(guard_budget))
            .ok_or_else(|| fault("native capture root metadata overflow"))?;
        if minimum > budget {
            return Err(fault("native capture root metadata capacity"));
        }
        let mut buffers = Vec::new();
        buffers
            .try_reserve_exact(resources.len())
            .map_err(|_| fault("native capture root allocation"))?;
        let bytes = buffers
            .capacity()
            .checked_mul(size_of::<wgpu::Buffer>())
            .and_then(|n| n.checked_add(guard_budget))
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or_else(|| fault("native capture root metadata overflow"))?;
        if bytes > budget {
            return Err(fault("native capture actual root metadata capacity"));
        }
        buffers.extend(resources.iter().map(|r| r.buffer.clone()));
        Ok(Self {
            buffers,
            bytes,
            guard_budget,
        })
    }
    pub fn metadata_bytes(&self) -> usize {
        self.bytes
    }
    pub fn acquire(&self) -> Result<Guarded<'_>, ServerError> {
        // SAFETY: All actual original owners and views remain pinned. The
        // returned guard prevents destruction until released. No nested Wgpu
        // lookup/poll/drop/destruction is allowed in the guarded phase.
        let batch = unsafe {
            wgpu::Buffer::kuno_as_hal_batch::<wgpu::hal::api::Vulkan>(
                &self.buffers,
                self.guard_budget,
            )
        }
        .ok_or_else(|| fault("native capture Vulkan destruction guard"))?;
        if batch.metadata_bytes().is_none_or(|n| n > self.guard_budget) {
            return Err(fault("native capture actual guard metadata capacity"));
        }
        Ok(Guarded { keys: self, batch })
    }
}
pub(super) struct Guarded<'a> {
    keys: &'a Keys,
    batch: wgpu::wgc::KunoBufferHalBatch<wgpu::hal::api::Vulkan>,
}
impl Guarded<'_> {
    pub fn signature_view(
        &self,
        r: &WgpuResource,
    ) -> Result<super::capture_inventory::View, ServerError> {
        use ash::vk::Handle;
        if r.size == 0
            || r.offset
                .checked_add(r.size)
                .is_none_or(|end| end > r.buffer.size())
        {
            return Err(fault("native signature original full owner interval"));
        }
        Ok(super::capture_inventory::View {
            raw: self.get(r)?.as_raw(),
            first: r.offset,
            bytes: r.size,
        })
    }
    pub fn get(&self, resource: &WgpuResource) -> Result<vk::Buffer, ServerError> {
        let index = self
            .keys
            .buffers
            .binary_search(&resource.buffer)
            .map_err(|_| fault("native capture foreign root identity"))?;
        let raw = self
            .batch
            .get(index)
            .ok_or_else(|| fault("native capture missing actual guarded buffer"))?;
        // SAFETY: This exact original raw object remains guarded for all Vulkan
        // planning/encoding uses, not just the duration of this method.
        Ok(unsafe { raw.raw_buffer_handle() })
    }
}

/// Bound to the SAME real key owner, never an equal-byte/root surrogate.
pub(super) struct ScopedSnapshot<'k> {
    keys: &'k Keys,
    inventory: super::capture_inventory::Snapshot,
    readonly: Option<super::replay_signature::Readonly>,
    source_lists: u64,
    factored: u64,
}
pub(super) struct Captured<'a, 'k> {
    snapshot: &'a super::capture_inventory::Snapshot,
    signature: &'a super::replay_signature::Signature,
    readonly: Option<&'a super::replay_signature::Readonly>,
    _guard: &'a Guarded<'k>,
}
impl<'k> Guarded<'k> {
    fn inventory_root(&self, i: usize) -> Result<super::capture_inventory::Root, ServerError> {
        let raw = self
            .batch
            .get(i)
            .ok_or_else(|| fault("native capture actual guarded root missing"))?;
        let buffer = self
            .keys
            .buffers
            .get(i)
            .ok_or_else(|| fault("native capture actual key missing"))?;
        use ash::vk::Handle;
        Ok(super::capture_inventory::Root {
            raw: unsafe { raw.raw_buffer_handle() }.as_raw(),
            bytes: buffer.size(),
        })
    }
    pub fn snapshot(
        &self,
        signature: &super::replay_signature::Signature,
        nodes: usize,
        budget: usize,
    ) -> Result<ScopedSnapshot<'k>, ServerError> {
        let inventory = match signature {
            super::replay_signature::Signature::Flat(v) => {
                super::capture_inventory::Snapshot::build(
                    v,
                    nodes,
                    self.keys.buffers.len(),
                    |i| {
                        self.inventory_root(i)
                            .map_err(|_| "native capture actual root unavailable")
                    },
                    budget,
                )
            }
            super::replay_signature::Signature::Compact(c) => {
                if c.nodes() != nodes {
                    return Err(fault("native canonical original node count"));
                }
                super::capture_inventory::Snapshot::build_canonical(
                    c.identity(),
                    nodes,
                    c.views(),
                    self.keys.buffers.len(),
                    |i| {
                        self.inventory_root(i)
                            .map_err(|_| "native capture actual root unavailable")
                    },
                    budget,
                )
            }
        }
        .map_err(fault)?;
        let readonly = signature.compact().and_then(|c| {
            super::replay_signature::Readonly::build(
                c,
                |r| inventory.root_size(r),
                budget.saturating_sub(inventory.metadata_bytes()),
            )
            .ok()
        });
        let source_lists = signature.compact().map_or(0, |c| c.source_lists() as u64);
        let factored = match (&readonly, signature.compact()) {
            (Some(r), Some(c)) => r.factored_count(c).map_err(fault)?,
            _ => 0,
        };
        Ok(ScopedSnapshot {
            keys: self.keys,
            inventory,
            readonly,
            source_lists,
            factored,
        })
    }
}
impl<'k> ScopedSnapshot<'k> {
    pub fn metadata_bytes(&self) -> usize {
        self.inventory.metadata_bytes() + self.readonly.as_ref().map_or(0, |r| r.metadata_bytes())
    }
    pub fn stats(&self) -> [u64; 6] {
        let [a, b, c] = self.inventory.stats();
        [
            a,
            b,
            c,
            self.source_lists,
            self.readonly.as_ref().map_or(0, |r| r.roots().len() as u64),
            self.factored,
        ]
    }
    pub fn bind<'a>(
        &'a self,
        native: &'a Guarded<'k>,
        signature: &'a super::replay_signature::Signature,
    ) -> Result<Captured<'a, 'k>, ServerError> {
        if !std::ptr::eq(self.keys, native.keys)
            || native.keys.buffers.len() != self.inventory.roots()
        {
            return Err(fault("native capture actual root key owner changed"));
        }
        match signature {
            super::replay_signature::Signature::Flat(v) => self.inventory.check_signature(v),
            super::replay_signature::Signature::Compact(c) => {
                self.inventory.check_canonical(c.identity())
            }
        }
        .map_err(fault)?;
        for i in 0..self.inventory.roots() {
            self.inventory
                .check_root(i, native.inventory_root(i)?)
                .map_err(fault)?;
        }
        self.inventory.note_revalidated();
        Ok(Captured {
            snapshot: &self.inventory,
            signature,
            readonly: self.readonly.as_ref(),
            _guard: native,
        })
    }
}
impl Captured<'_, '_> {
    pub fn readonly_views(&self) -> &[super::capture_inventory::View] {
        self.readonly.map_or(&[], |r| r.roots())
    }
    pub fn readonly_contains(&self, raw: u64) -> bool {
        self.readonly.is_some_and(|r| r.contains(raw))
    }
    pub fn node(&self, index: usize) -> Result<super::replay_signature::Node<'_>, ServerError> {
        match self.signature {
            super::replay_signature::Signature::Flat(v) => Ok(super::replay_signature::Node::Flat(
                self.snapshot.node(v, index).map_err(fault)?,
            )),
            super::replay_signature::Signature::Compact(c) => Ok(c
                .node(index)
                .map_err(fault)?
                .attach(self.snapshot, self.readonly)),
        }
    }
}
