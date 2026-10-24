//! Native-WGPU-only bulk admission of the original six expert operators.
use super::*;
use cubecl_runtime::memory_management::ManagedMemoryId;
use cubecl_runtime::server::native_expert_grammar::NativeStrataExpertGrammar;
use cubecl_runtime::server::{
    MetadataBindingInfo, NativeStrataExpertBatch, NativeStrataExpertBatchReceipt,
    NativeStrataExpertCommand,
};
use hashbrown::HashMap as ExactMap;

#[path = "strata_read_bundles.rs"]
mod read_bundles;
const MAX_BINDINGS: usize = 4096;
const MAX_PIPELINES: usize = 32;
type Key = (ManagedMemoryId, Option<u64>, Option<u64>, StreamId, u64);
fn key(binding: &Binding) -> Key {
    (
        binding.memory.allocation_id(),
        binding.offset_start,
        binding.offset_end,
        binding.stream,
        binding.size,
    )
}
fn fault(reason: &'static str) -> ServerError {
    super::super::transient_io::fault(reason)
}
impl WgpuServer {
    pub(super) fn native_expert_batch(
        &mut self,
        mut batch: NativeStrataExpertBatch<<Self as ComputeServer>::Kernel>,
        stream_id: StreamId,
    ) -> Result<NativeStrataExpertBatchReceipt, ServerError> {
        #[cfg(feature = "kuno-public-host-clock")]
        let mut clock = super::super::strata_graph::host_capture_profile::BatchTimer::new(
            super::super::strata_graph::host_profile_active(),
        );
        if self.backend != wgpu::Backend::Vulkan
            || !super::super::strata_graph::enabled()
            || self.public_external_stream != Some(stream_id)
            || batch.commands.is_empty()
        {
            return Err(fault("native expert batch selected backend required"));
        }
        // Original scheduling work must precede this exact ordered batch. This
        // drains tasks into the same graph, not a submission/completion fence.
        self.scheduler.enqueue_stream_tasks(stream_id);
        // Conservatively admit all simultaneously live original command,
        // binding/resource/hash tables, pipeline/INFO views and one packing page
        // before allocating any backend scratch. Actual capacities rechecked.
        let metadata = read_bundles::host_bytes::<
            Binding,
            WgpuResource,
            Key,
            MetadataBindingInfo,
            Arc<ComputePipeline>,
        >(batch.metadata_bytes()?, batch.commands.len())
        .ok_or_else(|| fault("native expert batch metadata overflow"))?;
        let scope = self
            .scheduler
            .stream(&stream_id)
            .begin_native_expert_batch(metadata)?;
        let mutation = self.scheduler.stream(&stream_id).private_mutation_guard();
        let scopes = super::super::transient_io::IoScopes::new(&self.device);
        #[cfg(feature = "kuno-public-host-clock")]
        clock.mark(0);
        let result = (|| {
            // Memory bindings AND resources coexist. No raw VkBuffer/resource
            // clone substitutes allocator ownership during deferred execution.
            let mut indices = ExactMap::<Key, usize>::new();
            let mut bindings = Vec::new();
            let mut resources = Vec::new();
            indices
                .try_reserve(MAX_BINDINGS)
                .map_err(|_| fault("native expert original binding index allocation"))?;
            bindings
                .try_reserve_exact(MAX_BINDINGS)
                .map_err(|_| fault("native expert original owner allocation"))?;
            resources
                .try_reserve_exact(MAX_BINDINGS)
                .map_err(|_| fault("native expert resource allocation"))?;
            if indices.capacity() > MAX_BINDINGS * 2
                || bindings.capacity() > MAX_BINDINGS * 2
                || resources.capacity() > MAX_BINDINGS * 2
            {
                return Err(fault("native expert binding table actual capacity"));
            }
            for command in &batch.commands {
                let mut admit = |binding: &Binding| -> Result<(), ServerError> {
                    if binding.stream != stream_id
                        || binding
                            .offset_start
                            .unwrap_or(0)
                            .checked_add(binding.offset_end.unwrap_or(0))
                            .is_none_or(|n| n >= binding.size)
                    {
                        return Err(fault("native expert original binding extent/stream"));
                    }
                    let key = key(binding);
                    if !indices.contains_key(&key) {
                        if bindings.len() >= MAX_BINDINGS {
                            return Err(fault("native expert original binding capacity"));
                        }
                        let resource = self
                            .scheduler
                            .stream(&stream_id)
                            .mem_manage
                            .get_resource(binding.clone())?;
                        if resource.size != binding.size_in_used() || resource.size == 0 {
                            return Err(fault("native expert original resolved extent"));
                        }
                        indices.insert(key, bindings.len());
                        bindings.push(binding.clone());
                        resources.push(resource);
                    }
                    Ok(())
                };
                match command {
                    NativeStrataExpertCommand::Launch { args, .. } => {
                        super::super::public_external::validate_stream(args, stream_id)
                            .map_err(fault)?;
                        for binding in args.buffers.iter().chain(&args.read_dependencies) {
                            admit(binding)?;
                        }
                    }
                    NativeStrataExpertCommand::HostCopies(copies) => {
                        cubecl_runtime::public_external::validate_host_copies(&copies.0, stream_id)
                            .map_err(fault)?;
                        for copy in &copies.0 {
                            admit(&copy.source)?;
                            admit(&copy.destination)?;
                        }
                    }
                }
            }
            #[cfg(feature = "kuno-public-host-clock")]
            clock.mark(1);
            // This same immutable bundle enters EACH original recording that
            // references it, including split/rolling and quarantine paths.
            let owners = Arc::new(bindings);
            let mut pipelines = Vec::new();
            pipelines
                .try_reserve_exact(batch.commands.len())
                .map_err(|_| fault("native expert pipeline table allocation"))?;
            let mut known = HashMap::new();
            known
                .try_reserve(MAX_PIPELINES)
                .map_err(|_| fault("native expert pipeline index allocation"))?;
            let mut records = Vec::<MetadataBindingInfo>::new();
            records
                .try_reserve_exact(batch.commands.len())
                .map_err(|_| fault("native expert INFO record allocation"))?;
            if pipelines.capacity() > batch.commands.len() * 2
                || records.capacity() > batch.commands.len() * 2
                || known.capacity() > MAX_PIPELINES * 2
            {
                return Err(fault("native expert actual pipeline/INFO capacity"));
            }
            let mut grammar = NativeStrataExpertGrammar::default();
            #[cfg(feature = "kuno-expert-cohorts")]
            let mut unit_releases = 0usize;
            for command in &mut batch.commands {
                match command {
                    NativeStrataExpertCommand::Launch {
                        kernel,
                        args,
                        mode,
                        count,
                        #[cfg(feature = "kuno-expert-cohorts")]
                        unit_end,
                    } => {
                        let kernel = kernel
                            .take()
                            .ok_or_else(|| fault("native expert original kernel missing"))?;
                        #[cfg(feature = "kuno-expert-cohorts")]
                        let kernel = if *unit_end {
                            // Validate the actual original task/INFO through
                            // its final DOWN before adding scheduling metadata.
                            // Original release=false matrix guards stay intact.
                            let terminal = cubecl_runtime::server::native_expert_batch::NativeStrataExpertUnitEnd::validate(
                                kernel, &mut grammar, count, args,
                            )?;
                            unit_releases += 1;
                            Box::new(terminal) as <Self as ComputeServer>::Kernel
                        } else {
                            grammar.launch(kernel.name(), kernel.native_release_after(), count, args)?;
                            kernel
                        };
                        #[cfg(not(feature = "kuno-expert-cohorts"))]
                        grammar.launch(kernel.name(), kernel.native_release_after(), count, args)?;
                        let mut id = kernel
                            .id()
                            .native_release_boundary(kernel.native_release_after());
                        id.mode(*mode);
                        id = id.public_external_read_mask(
                            super::super::public_external::external_mask(args).map_err(fault)?,
                        );
                        let pipeline = if let Some(p) = known.get(&id) {
                            Arc::clone(p)
                        } else {
                            if known.len() >= MAX_PIPELINES {
                                return Err(fault("native expert pipeline capacity"));
                            }
                            let p = self
                                .pipeline(kernel, args, *mode)
                                .map_err(ServerError::Launch)?;
                            known.insert(id, p.clone());
                            p
                        };
                        pipelines.push(Some(pipeline));
                        records.push(core::mem::take(&mut args.info));
                    }
                    NativeStrataExpertCommand::HostCopies(_) => {
                        grammar.host_copies()?;
                        pipelines.push(None);
                    }
                }
            }
            #[cfg(feature = "kuno-public-host-clock")]
            clock.mark(2);
            let logical = grammar.finish()?;
            #[cfg(feature="kuno-expert-cohorts")]
            self.scheduler.stream(&stream_id).admit_native_expert_cohorts(&scope,&grammar)?;
            let info = self
                .scheduler
                .stream(&stream_id)
                .native_expert_uniforms(&records, &scope)?;
            let mut read_cache = read_bundles::Cache::new().map_err(fault)?;
            let mut read_indices = read_bundles::Indices::default();
            read_indices
                .try_reserve_exact(MAX_BINDINGS)
                .map_err(|_| fault("native source index scratch allocation"))?;
            if read_indices.capacity() > MAX_BINDINGS {
                return Err(fault("native source index actual scratch capacity"));
            }
            #[cfg(feature = "kuno-public-host-clock")]
            clock.mark(3);
            let mut next_info = 0;
            let mut receipt = NativeStrataExpertBatchReceipt::default();
            for (index, command) in batch.commands.iter_mut().enumerate() {
                match command {
                    NativeStrataExpertCommand::Launch { count, args, .. } => {
                        let mut bound = Vec::with_capacity(args.buffers.len() + 1);
                        for binding in &args.buffers {
                            bound.push(resources[indices[&key(binding)]].clone());
                        }
                        bound.push(info[next_info].clone());
                        next_info += 1;
                        read_indices.clear();
                        read_indices.extend(
                            args.read_dependencies
                                .iter()
                                .map(|binding| indices[&key(binding)]),
                        );
                        let reads = read_cache
                            .resolve(&read_indices, |index| {
                                resources
                                    .get(index)
                                    .cloned()
                                    .ok_or("native source original resource missing")
                            })
                            .map_err(fault)?;
                        let unused = self
                            .scheduler
                            .stream(&stream_id)
                            .enqueue_native_expert_owned(
                                pipelines[index].as_ref().unwrap().clone(),
                                bound,
                                reads,
                                count,
                                owners.clone(),
                                &scope,
                            )?;
                        receipt.unused_descriptor_views += unused;
                        receipt.launches += 1;
                    }
                    NativeStrataExpertCommand::HostCopies(copies) => {
                        let mut actual = Vec::with_capacity(copies.0.len());
                        for copy in &copies.0 {
                            let source = resources[indices[&key(&copy.source)]].clone();
                            let target = resources[indices[&key(&copy.destination)]].clone();
                            actual.push((
                                ManagedResource::new(copy.source.memory.clone(), source),
                                copy.source_first,
                                ManagedResource::new(copy.destination.memory.clone(), target),
                                copy.destination_first,
                                copy.bytes,
                            ));
                        }
                        self.scheduler
                            .stream(&stream_id)
                            .stage_public_host_copies_owned(
                                &actual,
                                Some(owners.clone()),
                                Some(&scope),
                            )?;
                        receipt.copy_segments += actual.len();
                    }
                }
            }
            #[cfg(feature = "kuno-public-host-clock")]
            clock.mark(4);
            if next_info != info.len() || receipt.launches != logical.launches {
                return Err(fault("native expert original INFO inventory"));
            }
            receipt.cohort_groups=logical.cohort_groups;
            #[cfg(feature = "kuno-expert-cohorts")]
            { receipt.unit_releases = unit_releases; }
            receipt.cohort_launches=logical.cohort_launches;
            receipt.cohort_batches=logical.cohort_batches;
            receipt.quant_cohort_groups = logical.quant_cohort_groups;
            receipt.quant_cohort_launches = logical.quant_cohort_launches;
            receipt.quant_cohort_batches = logical.quant_cohort_batches;
            receipt.regular_cohort_groups=logical.regular_cohort_groups;
            receipt.regular_cohort_batches=logical.regular_cohort_batches;
            receipt.regular_cohort_launches=logical.regular_cohort_launches;
            receipt.logical_phases = logical.logical_phases;
            receipt.matrix_pieces = logical.matrix_pieces;
            receipt.wide_prefixes = logical.wide_prefixes;
            receipt.direct_gu_prefixes = logical.direct_gu_prefixes;
            receipt.gu_checks = logical.gu_checks;
            receipt.wide_blocks = logical.wide_blocks;
            receipt.tail_pieces = logical.tail_pieces;
            Ok(receipt)
        })();
        let finished = self.scheduler.stream(&stream_id).end_native_expert_batch(
            scope,
            scopes,
            result.is_ok(),
        );
        if result.is_ok() && finished.is_ok() {
            if let Some(mutation) = mutation {
                mutation.accept();
            }
        }
        #[cfg(feature = "kuno-public-host-clock")]
        clock.mark(5);
        finished?;
        result
    }
}
