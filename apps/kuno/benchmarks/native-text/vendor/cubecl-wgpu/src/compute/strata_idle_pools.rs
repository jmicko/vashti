//! Content-free ORIGINAL device resources after completed recording retirement.
//! No buffer owner, plan, signature, INFO/source image or model value is retained.
use super::{Commands, Plan, Reencoding, Segment, command_reencode, plan_metadata_bytes};
use ash::vk;

pub(crate) struct IdlePools {
    device: ash::Device,
    graph: u64,
    family: u32,
    main: vk::CommandPool,
    copy: vk::CommandPool,
    shared: vk::CommandPool,
    timelines: [vk::Semaphore; 3],
    next_base: u64,
    header: vk::CommandBuffer,
    spares: [Vec<vk::CommandBuffer>; 3],
}
impl IdlePools {
    pub fn command_count(&self) -> usize {
        1 + self.spares.iter().map(Vec::len).sum::<usize>()
    }
    pub fn metadata_bytes(&self) -> Option<usize> {
        self.spares
            .iter()
            .try_fold(0usize, |n, v| n.checked_add(v.capacity()))?
            .checked_mul(size_of::<vk::CommandBuffer>())?
            .checked_add(size_of::<Self>())
    }
    fn matches(&self, device: &ash::Device, graph: u64, family: u32) -> bool {
        self.device.handle() == device.handle()
            && self.graph == graph
            && self.family == family
            && self.header != vk::CommandBuffer::null()
            && self.main != vk::CommandPool::null()
            && self.copy != vk::CommandPool::null()
    }
    pub fn prepare(
        &self,
        device: &ash::Device,
        graph: u64,
        family: u32,
        plan: &Plan,
        budget: usize,
    ) -> Option<Reencoding> {
        if !self.matches(device, graph, family)
            || plan.segments.iter().any(|s| s.copy && s.shared)
            || (plan.last_shared != 0 && self.shared == vk::CommandPool::null())
        {
            return None;
        }
        let p = command_reencode::prepare_capacity(
            &[0usize],
            |q| *q,
            &[self.header],
            &self.spares,
            plan.segments.iter().map(Segment::queue),
            plan_metadata_bytes(plan)?,
            size_of::<Commands>().checked_add(size_of::<Reencoding>())?,
            budget,
        )?;
        Some(Reencoding {
            header: self.header,
            capacity: Some(p),
        })
    }
    pub fn prepared_metadata_bytes(&self, plan: &Plan, p: &Reencoding) -> Option<usize> {
        if p.header != self.header {
            return None;
        }
        let c = p.capacity.as_ref()?;
        if c.header != self.header
            || c.commands.first().copied() != Some(self.header)
            || c.commands.len() != plan.segments.len()
        {
            return None;
        }
        c.metadata_bytes(
            plan_metadata_bytes(plan)?,
            size_of::<Commands>().checked_add(size_of::<Reencoding>())?,
        )
    }
    /// Consumes only already-reset original pools. New full source/plan/owner
    /// admission is the caller's original capture; no old execution is replayed.
    pub fn into_commands(
        mut self,
        plan: Plan,
        p: Reencoding,
        budget: usize,
    ) -> Result<(Commands, vk::CommandPool, vk::CommandBuffer), &'static str> {
        if self
            .prepared_metadata_bytes(&plan, &p)
            .is_none_or(|n| n > budget)
        {
            return Err("native idle pool original command inventory");
        }
        let p = p
            .capacity
            .ok_or("native idle pool prepared command inventory")?;
        let work = Commands {
            device: self.device.clone(),
            copy_pool: std::mem::replace(&mut self.copy, vk::CommandPool::null()),
            shared_pool: std::mem::replace(&mut self.shared, vk::CommandPool::null()),
            timelines: std::mem::replace(&mut self.timelines, [vk::Semaphore::null(); 3]),
            next_base: self.next_base,
            plan,
            commands: p.commands,
            spares: p.spares,
        };
        let main = std::mem::replace(&mut self.main, vk::CommandPool::null());
        Ok((work, main, self.header))
    }
}
impl Drop for IdlePools {
    fn drop(&mut self) {
        unsafe {
            for semaphore in self.timelines {
                if semaphore != vk::Semaphore::null() {
                    self.device.destroy_semaphore(semaphore, None);
                }
            }
            for pool in [self.main, self.copy, self.shared] {
                if pool != vk::CommandPool::null() {
                    self.device.destroy_command_pool(pool, None);
                }
            }
        }
    }
}
impl Commands {
    /// Original Graph must prove NONPENDING through its real flight book and
    /// release every source lease before dropping the retired recording owners.
    /// On None/error the original recording still owns all its resources.
    pub unsafe fn retire_completed_pools(
        &mut self,
        main: vk::CommandPool,
        graph: u64,
        family: u32,
        budget: usize,
    ) -> Result<Option<IdlePools>, &'static str> {
        if main == vk::CommandPool::null()
            || self.copy_pool == vk::CommandPool::null()
            || main == self.copy_pool
            || main == self.shared_pool
            || self.copy_pool == self.shared_pool
            || self.plan.segments.iter().any(|s| s.copy && s.shared)
        {
            return Err("native idle pool original retirement ownership");
        }
        let Some(p) = command_reencode::prepare_capacity(
            &self.plan.segments,
            Segment::queue,
            &self.commands,
            &self.spares,
            std::iter::once(0),
            0,
            size_of::<IdlePools>(),
            budget,
        ) else {
            return Ok(None);
        };
        for pool in [main, self.copy_pool, self.shared_pool] {
            if pool != vk::CommandPool::null() {
                unsafe {
                    self.device
                        .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
                }
                .map_err(|_| "native completed idle pool retirement reset")?;
            }
        }
        Ok(Some(IdlePools {
            device: self.device.clone(),
            graph,
            family,
            main,
            copy: std::mem::replace(&mut self.copy_pool, vk::CommandPool::null()),
            shared: std::mem::replace(&mut self.shared_pool, vk::CommandPool::null()),
            timelines: std::mem::replace(&mut self.timelines, [vk::Semaphore::null(); 3]),
            next_base: self.next_base,
            header: p.header,
            spares: p.spares,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Exhaustive field destructuring makes new cache fields require review.
    /// Only device identity and typed driver handle inventories are admissible.
    #[test]
    fn idle_pool_type_cannot_retain_private_values_plans_or_resource_owners() {
        trait DeviceMetadata {}
        impl DeviceMetadata for ash::Device {}
        impl DeviceMetadata for u64 {}
        impl DeviceMetadata for u32 {}
        impl DeviceMetadata for vk::CommandPool {}
        impl DeviceMetadata for vk::CommandBuffer {}
        impl DeviceMetadata for [vk::Semaphore; 3] {}
        impl DeviceMetadata for [Vec<vk::CommandBuffer>; 3] {}
        fn allowed<T: DeviceMetadata>(_: &T) {}
        fn fields(p: &IdlePools) {
            let IdlePools {
                device,
                graph,
                family,
                main,
                copy,
                shared,
                timelines,
                next_base,
                header,
                spares,
            } = p;
            allowed(device);
            allowed(graph);
            allowed(family);
            allowed(main);
            allowed(copy);
            allowed(shared);
            allowed(timelines);
            allowed(next_base);
            allowed(header);
            allowed(spares);
        }
        let _all_fields_are_content_free: fn(&IdlePools) = fields;
        assert!(size_of::<IdlePools>() < 16 * 1024);
    }
    #[test]
    fn actual_retirement_joins_and_source_leases_stay_ahead_of_owner_release() {
        let graph = include_str!("strata_graph.rs");
        let a = graph
            .find("pub(crate) fn clear_completed_recordings")
            .unwrap();
        let b = a + graph[a..].find("pub(crate) fn pending").unwrap();
        let clear = &graph[a..b];
        for check in [
            "self.flights.any()",
            "self.handoff.complete()",
            "clear_joined_exports()",
        ] {
            assert!(clear.find(check).unwrap() < clear.find("retire_recording").unwrap());
        }
        let a = graph.find("fn retire_recording(").unwrap();
        let b = a + graph[a..]
            .find("pub(crate) fn clear_completed_recordings")
            .unwrap();
        let retire = &graph[a..b];
        assert!(
            retire.find("self.flights.live").unwrap()
                < retire.find("retire_completed_pools").unwrap()
        );
        assert!(
            retire.find("r.ready = false").unwrap()
                < retire.find("retire_completed_pools").unwrap()
        );
        assert!(
            retire.find("r.pool = vk::CommandPool::null()").unwrap()
                < retire.find("self.recordings.remove").unwrap()
        );
        let a = graph.find("let signature = signature.finish()").unwrap();
        let source = &graph[a..];
        assert!(
            source.find("drop(native)").unwrap() < source.find("self.retire_recording").unwrap()
        );
        assert!(graph.contains("std::mem::forget(self.idle_pools.take())"));
    }
}
