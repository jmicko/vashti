//! Fixed PUBLIC source attachment. The parent owns complete bank preparation,
//! family transactions, private cleanup and hardware acceptance. This helper
//! never chooses experts, fills slots, resets an admission or retries failures.
//!
//! Keep PreparedPlan alive until Attachment closes. One checked target stream
//! owns all foreign registrations. Failed setup/close leaves exposed Mirror
//! backing retained by its fail-safe Drop; it grants no release/erasure receipt.
//! Source-set metadata is the SAME reservation already declared by the model,
//! not a second allocation/credit. Registry retired metadata remains live after
//! backing release until the caller drops the actual device/server.

use crate::{
    executor::{GpuExecutor, batch_moe},
    private_arena::PrivateArena,
    public_host_expert_mirror::{self as mirror, Mirror},
    public_host_expert_plan::{self as plan, Plan, Role},
    resident_util as u,
};
use batch_moe::frozen_sources::{self as sources, RegisteredPage, Sources};
use cubecl::{
    prelude::ComputeClient,
    wgpu::{WgpuResource, WgpuRuntime},
};
use cubecl_runtime::public_external::{
    MAX_METADATA_BYTES, PublicExternalBudget, PublicExternalDetachReceipt, PublicExternalHandle,
    PublicExternalHostSource, PublicExternalUsage,
};
use kuno_text::{Error, GgufFile, Result, model::BoundGraph};
pub(crate) use sources::ResidentBankAllocation;
use std::{
    mem::size_of,
    sync::{Arc, atomic::AtomicBool},
};

#[path = "public_cpu_host_borrow.rs"]
mod cpu_host_borrow;
pub(crate) use cpu_host_borrow::CpuLayerBorrow;
#[path = "public_host_cpu_overlap_probe.rs"]
pub(crate) mod overlap_probe;

pub(crate) const ATTACHMENT_HOST_BYTES: u64 = 256 * 1024;
pub(crate) const REGISTRY_HOST_BYTES: u64 = MAX_METADATA_BYTES as u64;
const HOST_CAP: u64 = 64 * 1024 * 1024 * 1024;
const ROLES: [Role; 3] = [Role::Gate, Role::Up, Role::Down];
fn rejected() -> Error {
    Error::InvalidFormat("public host expert attachment rejected")
}
fn backend_error() -> Error {
    Error::InvalidFormat("public host expert attachment backend failure")
}

/// Separate PUBLIC RAM constituents. No GPU MAIN/private erasure credit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttachmentBudget {
    pub physical_public_ram_envelope_bytes: u64,
    pub planner_mirror_metadata_bytes: u64,
    pub source_window_bytes: u64,
    pub registry_metadata_bytes: u64,
    /// Already included in the model's constructor host reservation.
    pub shared_model_source_host_bytes: u64,
    pub attachment_metadata_bytes: u64,
    pub total_host_reserved_bytes: u64,
    /// Added by the model before allocation, never charged again here.
    pub shared_model_runtime_increment_bytes: u64,
}
impl AttachmentBudget {
    fn new(physical: u64) -> Result<Self> {
        if physical < plan::LOGICAL_PUBLIC_RAM_BYTES {
            return Err(rejected());
        }
        let total = [
            plan::METADATA_HOST_CAP,
            plan::SOURCE_WINDOW_HOST_BYTES,
            REGISTRY_HOST_BYTES,
            sources::EXTRA_HOST_BYTES,
            ATTACHMENT_HOST_BYTES,
        ]
        .into_iter()
        .try_fold(physical, |sum, next| sum.checked_add(next))
        .filter(|&n| n <= HOST_CAP)
        .ok_or(Error::ArithmeticOverflow)?;
        Ok(Self {
            physical_public_ram_envelope_bytes: physical,
            planner_mirror_metadata_bytes: plan::METADATA_HOST_CAP,
            source_window_bytes: plan::SOURCE_WINDOW_HOST_BYTES,
            registry_metadata_bytes: REGISTRY_HOST_BYTES,
            shared_model_source_host_bytes: sources::EXTRA_HOST_BYTES,
            attachment_metadata_bytes: ATTACHMENT_HOST_BYTES,
            total_host_reserved_bytes: total,
            shared_model_runtime_increment_bytes: sources::EXTRA_INFO_BYTES_PER_CHUNK,
        })
    }
    /// Add this to a model budget that ALREADY contains shared source metadata.
    pub(crate) fn exclusive_host_increment_bytes(self) -> Result<u64> {
        self.total_host_reserved_bytes
            .checked_sub(self.shared_model_source_host_bytes)
            .ok_or(Error::ArithmeticOverflow)
    }
    fn mirror_reservation(self) -> Result<u64> {
        self.physical_public_ram_envelope_bytes
            .checked_add(self.planner_mirror_metadata_bytes)
            .and_then(|n| n.checked_add(self.source_window_bytes))
            .ok_or(Error::ArithmeticOverflow)
    }
    fn check_exact(self, reserved: u64) -> Result<()> {
        if reserved != self.total_host_reserved_bytes {
            return Err(rejected());
        }
        Ok(())
    }
}

pub(crate) struct PreparedPlan<'model, 'file> {
    plan: Box<Plan<'model, 'file>>,
    graph: BoundGraph<'model, 'file>,
    model: &'file GgufFile,
    budget: AttachmentBudget,
}
impl<'model, 'file> PreparedPlan<'model, 'file> {
    /// Parent's actual completed bank/family proof precedes this pure snapshot.
    /// The two-MiB plan reservation is a constituent of the final budget.
    pub(crate) fn new(
        graph: BoundGraph<'model, 'file>,
        model: &'file GgufFile,
        executor: &GpuExecutor<'file>,
        allocation_alignment: u64,
        plan_metadata_reserved: u64,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        Self::new_with_fill_publication(
            graph,
            model,
            executor,
            allocation_alignment,
            plan_metadata_reserved,
            cancel,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)] // Same original plan inputs plus opaque actual publication.
    pub(crate) fn new_with_fill_publication(
        graph: BoundGraph<'model, 'file>,
        model: &'file GgufFile,
        executor: &GpuExecutor<'file>,
        allocation_alignment: u64,
        plan_metadata_reserved: u64,
        cancel: &AtomicBool,
        publication: Option<crate::executor::BankFillPublication>,
    ) -> Result<Self> {
        u::cancel(cancel)?;
        if plan_metadata_reserved != plan::METADATA_HOST_CAP {
            return Err(rejected());
        }
        let banks = executor.public_host_completed_banks(graph)?;
        let plan = if let Some(publication) = publication.as_ref() {
            Plan::from_completed_banks_with_fill(
                graph,
                model,
                banks,
                allocation_alignment,
                Some(publication),
            )?
        } else {
            Plan::from_completed_banks(graph, model, banks, allocation_alignment)?
        };
        let budget = AttachmentBudget::new(plan.budget().physical_public_ram_envelope_bytes)?;
        u::cancel(cancel)?;
        Ok(Self {
            plan: Box::new(plan),
            graph,
            model,
            budget,
        })
    }
    pub(crate) fn budget(&self) -> AttachmentBudget {
        self.budget
    }

    /// # Safety
    /// Parent authenticates both full original shards, proves these completed
    /// actual bank bytes/device and healthy idle family boundary, and reserves
    /// the complete process/RAM budget BEFORE this call. Its model constructor
    /// already admits the stated source runtime/host constituents. This PUBLIC
    /// fixture has no private inputs. No alternate registry/raw queue/alias may
    /// consume or mutate these pages; no fallback, bank fill or recency commit
    /// is permitted after attachment. Retain this PreparedPlan until close.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        // SAFETY: forwards the same caller contract with original math.
        unsafe {
            self.attach_with_gu_math(
                device,
                client,
                executor,
                workspace,
                arena,
                exact_reserved_host,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_with_gu_math<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        // SAFETY: unchanged caller contract; the existing entry keeps the
        // original one-group-per-workgroup integer scheduling by default.
        unsafe {
            self.attach_with_gu_schedule(
                device,
                client,
                executor,
                workspace,
                arena,
                exact_reserved_host,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_with_gu_schedule<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        // SAFETY: same original caller contract; default DOWN schedule retained.
        unsafe {
            self.attach_with_schedules(
                device,
                client,
                executor,
                workspace,
                arena,
                exact_reserved_host,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_with_schedules<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        // Original entry retains its exact default arithmetic policy.
        unsafe {
            self.attach_with_down_math(
                device,
                client,
                executor,
                workspace,
                arena,
                exact_reserved_host,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                down_stride_four,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_with_down_math<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        down_integer: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        unsafe {
            self.attach_with_four_rows(
                device,
                client,
                executor,
                workspace,
                arena,
                exact_reserved_host,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                down_stride_four,
                down_integer,
                false,
                cancel,
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn attach_with_four_rows<'plan>(
        &'plan self,
        device: &wgpu::Device,
        client: &ComputeClient<WgpuRuntime>,
        executor: &mut GpuExecutor<'file>,
        workspace: &mut batch_moe::Workspace,
        arena: &PrivateArena,
        exact_reserved_host: u64,
        inactive_partition_exit: bool,
        private_tables: bool,
        workgroup_tables: bool,
        shared_down_scales: bool,
        shared_gate_up_factors: bool,
        four_row_down: bool,
        two_row_gate_up: bool,
        packed_host_staging: bool,
        three_members: bool,
        gu_integer: bool,
        gu_stride_four: bool,
        down_stride_four: bool,
        down_integer: bool,
        four_rows: bool,
        cancel: &AtomicBool,
    ) -> Result<Attachment<'plan, 'model, 'file>> {
        u::cancel(cancel)?;
        Sources::validate_down_integer_policy(
            down_integer,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            four_row_down,
            two_row_gate_up,
            packed_host_staging,
            three_members,
            down_stride_four,
        )?;
        Sources::validate_gu_schedule_policy(gu_integer, gu_stride_four)?;
        Sources::validate_down_stride_four_policy(
            down_stride_four,
            inactive_partition_exit,
            four_row_down,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            packed_host_staging,
            three_members,
            two_row_gate_up,
        )?;
        Sources::validate_kernel_policies(
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            four_row_down,
            two_row_gate_up,
        )?;
        Sources::validate_staging_policy(
            packed_host_staging,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
        )?;
        Sources::validate_three_member_policy(
            three_members,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            two_row_gate_up,
        )?;
        Sources::validate_gu_integer_policy(
            gu_integer,
            inactive_partition_exit,
            private_tables,
            workgroup_tables,
            shared_down_scales,
            shared_gate_up_factors,
            two_row_gate_up,
            packed_host_staging,
            three_members,
        )?;
        if down_integer != executor.target_frozen_down_integer_reserved
            || down_integer != workspace.frozen_down_integer_owner_id().is_some()
            || gu_integer != executor.target_frozen_gu_integer_reserved
            || gu_integer != workspace.frozen_gu_integer_owner_id().is_some()
            || (gu_integer || down_stride_four || down_integer)
                && !executor.target_frozen_host_router_reserved
        {
            return Err(rejected());
        }
        if three_members && !executor.target_frozen_host_router_reserved {
            return Err(rejected());
        }
        if workspace.frozen_staging_owner_ids().is_some() != packed_host_staging {
            return Err(rejected());
        }
        self.budget.check_exact(exact_reserved_host)?;
        executor.check_client(client)?;
        workspace.validate_public_warm_boundary(executor, self.graph, arena)?;
        // Borrow actual owners only in read-only scopes; none escapes into the
        // mutable workspace attachment. Freeze the client to that real stream.
        let handle = {
            let banks = executor.public_host_completed_banks(self.graph)?;
            self.plan.check_frozen(self.graph, self.model, banks)?;
            println!("public_frozen_host_source_freeze_complete");
            banks[0]
                .gate()
                .segments()
                .first()
                .ok_or_else(rejected)?
                .handle()
                .clone()
        };
        let actual = u::resources(client, vec![handle.clone()])?;
        if actual.len() != 1 {
            return Err(rejected());
        }
        check_device(device, actual[0].resource())?;
        drop(actual);
        let mut client = client.clone();
        // SAFETY: exact original owning stream, successfully resolved through
        // this same client before mutation; no new stream/queue is invented.
        unsafe {
            client.set_stream(handle.stream);
        }

        let mut owners = exact_vec::<Option<PublicExternalHandle>>()?;
        let mut registered = exact_vec::<RegisteredPage>()?;
        let mut receipts = exact_vec::<PublicExternalDetachReceipt<WgpuResource>>()?;
        let metadata_peak = attachment_metadata(owners.capacity(), receipts.capacity())?;
        let registry_budget = PublicExternalBudget {
            backing_bytes: self.budget.physical_public_ram_envelope_bytes,
            buffers: mirror::PAGES,
            metadata_bytes: MAX_METADATA_BYTES,
        }
        .validate()
        .map_err(Error::InvalidFormat)?;
        client
            .configure_public_external_host(registry_budget)
            .map_err(|_| backend_error())?;
        let empty = client
            .public_external_host_usage()
            .map_err(|_| backend_error())?;
        validate_registry(empty, 0, 0, 0)?;
        let mut mirror = {
            let banks = executor.public_host_completed_banks(self.graph)?;
            // SAFETY: caller establishes the constructor prerequisites above;
            // original source/bank freeze and exact device were checked here.
            unsafe {
                Mirror::new(
                    &self.plan,
                    self.graph,
                    self.model,
                    banks,
                    device,
                    self.budget.mirror_reservation()?,
                )
            }
            .map_err(Error::InvalidFormat)?
        };
        let ledger = mirror.ledger();
        println!(
            "public_frozen_host_materialization_complete pages={}",
            mirror::PAGES
        );
        for layer in 0..plan::LAYERS {
            for role in ROLES {
                for page_index in 0..3 {
                    u::cancel(cancel)?;
                    let witness = mirror
                        .page_resource(device, layer, role, page_index)
                        .map_err(Error::InvalidFormat)?;
                    let placement = witness.placement();
                    // SAFETY: sealed original page from the actual Mirror; its
                    // entire backing/padding was initialized and is retained.
                    let source = unsafe {
                        PublicExternalHostSource::from_initialized_public_host(
                            witness.resource().clone(),
                            witness.resource().size,
                            witness.backing_bytes(),
                            placement.property_flags,
                            placement.heap_flags,
                            witness.source_sha256(),
                        )
                    }
                    .map_err(Error::InvalidFormat)?;
                    let owner = client
                        .register_public_external_host(source)
                        .map_err(|_| backend_error())?;
                    let handle = owner.handle();
                    // Retain consuming authority before any further fallible
                    // checks. On error Mirror Drop still retains exposed RAM.
                    owners.push(Some(owner));
                    let page =
                        RegisteredPage::from_mirror(&client, &self.plan, &witness, handle.clone())?;
                    mirror
                        .registered(layer, role, page_index, handle.binding())
                        .map_err(Error::InvalidFormat)?;
                    registered.push(page);
                }
            }
        }
        if owners.len() != mirror::PAGES || registered.len() != mirror::PAGES {
            return Err(rejected());
        }
        let sources = if gu_integer
            || down_integer
            || three_members
            || packed_host_staging
            || two_row_gate_up
            || four_row_down
            || shared_gate_up_factors
            || shared_down_scales && (private_tables || workgroup_tables)
        {
            Sources::new_with_policy_and_four_rows(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
                inactive_partition_exit,
                private_tables,
                workgroup_tables,
                shared_down_scales,
                shared_gate_up_factors,
                four_row_down,
                two_row_gate_up,
                packed_host_staging,
                three_members,
                gu_integer,
                gu_stride_four,
                down_stride_four,
                down_integer,
                four_rows,
            )
        } else if workgroup_tables {
            Sources::new_with_workgroup_tables(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
            )
        } else if shared_down_scales {
            Sources::new_with_shared_down_scales(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
            )
        } else if private_tables {
            Sources::new_with_private_tables(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
            )
        } else if inactive_partition_exit {
            Sources::new_with_inactive_partition_exit(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
            )
        } else {
            Sources::new(
                executor,
                self.graph,
                &self.plan,
                registered,
                self.budget.shared_model_source_host_bytes,
            )
        }?;
        let registry = client
            .public_external_host_usage()
            .map_err(|_| backend_error())?;
        validate_registry(registry, mirror::PAGES, ledger.live_backing_bytes, 0)?;
        u::cancel(cancel)?;
        workspace.attach_frozen_host_sources(
            executor,
            self.graph,
            arena,
            sources.clone(),
            self.budget.shared_model_source_host_bytes,
        )?;
        // All fallible checks precede the actual attachment; only owned moves
        // follow. There is no post-publication allocation or retry.
        Ok(Attachment {
            mirror: Some(mirror),
            sources: Some(sources),
            owners,
            receipts: std::mem::take(&mut receipts),
            client,
            budget: self.budget,
            metadata_peak,
            registry,
            phase: Phase::Attached,
        })
    }
}

fn exact_vec<T>() -> Result<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(mirror::PAGES)
        .map_err(|_| Error::LimitExceeded("public host attachment metadata"))?;
    if out.capacity() != mirror::PAGES {
        return Err(rejected());
    }
    Ok(out)
}
fn attachment_metadata(owners: usize, receipts: usize) -> Result<u64> {
    if owners != mirror::PAGES || receipts != mirror::PAGES {
        return Err(rejected());
    }
    let bytes = owners
        .checked_mul(size_of::<Option<PublicExternalHandle>>())
        .and_then(|n| {
            receipts
                .checked_mul(size_of::<PublicExternalDetachReceipt<WgpuResource>>())
                .and_then(|r| n.checked_add(r))
        })
        .and_then(|n| n.checked_add(size_of::<Attachment<'_, '_, '_>>() * 2))
        .and_then(|n| n.checked_add(size_of::<PreparedPlan<'_, '_>>() * 2))
        // One exclusive scoped CPU layer borrow; fixed map headers remain in
        // the unchanged Mirror metadata constituent, not copied weight bytes.
        .and_then(|n| n.checked_add(size_of::<CpuLayerBorrow<'_, '_, '_, '_>>() * 2))
        // Fixed attachment/close scalar snapshots, never another GPU owner.
        .and_then(|n| n.checked_add(size_of::<ResidentBankAllocation>() * 2))
        .and_then(|n| n.checked_add(8192)) // temporary identity/resource validation
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes as u64 > ATTACHMENT_HOST_BYTES {
        return Err(rejected());
    }
    Ok(bytes as u64)
}
fn validate_registry(
    usage: PublicExternalUsage,
    registrations: usize,
    registered: u64,
    detached: u64,
) -> Result<()> {
    if usage.registration_count != registrations
        || usage.registered_backing_bytes != registered
        || usage.total_registered_bytes
            != registered
                .checked_add(detached)
                .ok_or(Error::ArithmeticOverflow)?
        || usage.total_detached_bytes != detached
        || usage.metadata_owned_bytes == 0
        || usage.metadata_owned_bytes > MAX_METADATA_BYTES
    {
        return Err(rejected());
    }
    Ok(())
}
fn check_device(device: &wgpu::Device, resource: &WgpuResource) -> Result<()> {
    if resource.size == 0
        || resource.offset != 0
        || resource.size > plan::MAX_BINDING_BYTES
        || device.limits().max_storage_buffer_binding_size < plan::MAX_BINDING_BYTES
        || device.limits().max_buffer_size < plan::MAX_BINDING_BYTES
        || unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }.is_none()
        || unsafe { resource.buffer.as_hal::<wgpu::hal::api::Vulkan>() }.is_none()
    {
        return Err(rejected());
    }
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("public-host-target-identity"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("public-host-target-identity"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &resource.buffer,
                offset: resource.offset,
                size: std::num::NonZeroU64::new(resource.size),
            }),
        }],
    });
    let rejected_device = cubecl_common::future::block_on(errors.pop()).is_some();
    drop(group);
    drop(layout);
    if rejected_device {
        return Err(rejected());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Attached,
    Detaching,
    Released,
}
fn require_detach(phase: Phase, source_count: usize) -> Result<()> {
    if phase != Phase::Attached || source_count != 1 {
        return Err(rejected());
    }
    Ok(())
}
pub(crate) struct Attachment<'plan, 'model, 'file> {
    mirror: Option<Mirror<'plan, 'model, 'file>>,
    sources: Option<Arc<Sources>>,
    owners: Vec<Option<PublicExternalHandle>>,
    receipts: Vec<PublicExternalDetachReceipt<WgpuResource>>,
    client: ComputeClient<WgpuRuntime>,
    budget: AttachmentBudget,
    metadata_peak: u64,
    registry: PublicExternalUsage,
    phase: Phase,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CloseReceipt {
    pub budget: AttachmentBudget,
    pub registered_pages: usize,
    pub mirror: mirror::Ledger,
    pub full_allocation_zero: bool,
    pub registry: PublicExternalUsage,
    pub attachment_metadata_peak_bytes: u64,
    pub detached_pages: usize,
    pub resident_bank_allocation: ResidentBankAllocation,
}
impl Attachment<'_, '_, '_> {
    /// # Safety
    /// Caller holds the completed, serialized target boundary, with no pending
    /// output lease or GPU consumer. Refresh router tables before new work.
    pub(crate) unsafe fn adapt_strata(&mut self, queue: &wgpu::Queue) -> Result<u64> {
        if self.phase != Phase::Attached || !crate::strata_port::enabled() {
            return Err(rejected());
        }
        let source = self.sources.as_ref().ok_or_else(rejected)?;
        // SAFETY: forwarded completed-boundary authority; mirror remains
        // attached, physically retained and unavailable to another borrower.
        unsafe {
            source.adapt_strata(
                &self.client,
                self.mirror.as_mut().ok_or_else(rejected)?,
                queue,
            )
        }
    }
    pub(crate) fn strata_cache_receipt(&self) -> Result<serde_json::Value> {
        self.sources
            .as_ref()
            .ok_or_else(rejected)?
            .strata_cache_receipt()
    }
    pub(crate) fn budget(&self) -> AttachmentBudget {
        self.budget
    }
    pub(crate) fn ledger(&self) -> Result<mirror::Ledger> {
        self.mirror
            .as_ref()
            .map(Mirror::ledger)
            .ok_or_else(rejected)
    }
    pub(crate) fn registry_at_attachment(&self) -> PublicExternalUsage {
        self.registry
    }
    /// Scalar ownership measurement only; gives no GPU join/free authority.
    pub(crate) fn resident_bank_allocation(&self) -> Result<ResidentBankAllocation> {
        if self.phase != Phase::Attached {
            return Err(rejected());
        }
        self.sources
            .as_ref()
            .ok_or_else(rejected)?
            .resident_bank_allocation()
    }
    /// # Safety
    /// Parent must first accept the ACTUAL complete original all-family/fault
    /// proof and explicit model private-zero close, then lexically drop its
    /// model/workspace and every pending consumer. Counts/Drop acknowledgements
    /// cannot substitute for that proof. No raw/untracked consumer or mutation
    /// is permitted. This is terminal; the frozen executor is never reused.
    pub(crate) unsafe fn close(mut self) -> Result<CloseReceipt> {
        let source = self.sources.as_ref().ok_or_else(rejected)?;
        require_detach(self.phase, Arc::strong_count(source))?;
        cubecl_common::future::block_on(self.client.sync()).map_err(|_| backend_error())?;
        let resident_bank_allocation = self.resident_bank_allocation()?;
        self.sources.take(); // all actual consumers joined; release our final pins
        self.phase = Phase::Detaching;
        let backing = self.ledger()?.live_backing_bytes;
        for owner in &mut self.owners {
            let receipt = self
                .client
                .detach_public_external_host(owner.take().ok_or_else(rejected)?)
                .map_err(|_| backend_error())?;
            self.receipts.push(receipt);
        }
        if self.receipts.len() != mirror::PAGES {
            return Err(rejected());
        }
        let registry = self
            .client
            .public_external_host_usage()
            .map_err(|_| backend_error())?;
        validate_registry(registry, mirror::PAGES, 0, backing)?;
        let receipts = std::mem::take(&mut self.receipts);
        let closed = self
            .mirror
            .take()
            .ok_or_else(rejected)?
            .close(receipts)
            .map_err(Error::InvalidFormat)?;
        self.phase = Phase::Released;
        Ok(CloseReceipt {
            budget: self.budget,
            registered_pages: self.registry.registration_count,
            mirror: closed.ledger,
            full_allocation_zero: closed.full_allocation_zero,
            registry,
            attachment_metadata_peak_bytes: self.metadata_peak,
            detached_pages: mirror::PAGES,
            resident_bank_allocation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn constituents_are_exact_and_source_charge_is_shared_once() {
        let budget = AttachmentBudget::new(plan::LOGICAL_PUBLIC_RAM_BYTES).unwrap();
        assert_eq!(
            budget.total_host_reserved_bytes,
            plan::LOGICAL_PUBLIC_RAM_BYTES + 6_561_792
        );
        assert_eq!(
            budget.exclusive_host_increment_bytes().unwrap(),
            plan::LOGICAL_PUBLIC_RAM_BYTES + 4_456_448
        );
        assert_eq!(budget.shared_model_source_host_bytes, 2_105_344);
        assert_eq!(budget.shared_model_runtime_increment_bytes, 589_824);
        budget
            .check_exact(budget.total_host_reserved_bytes)
            .unwrap();
        assert!(
            budget
                .check_exact(budget.total_host_reserved_bytes - 1)
                .is_err()
        );
        assert!(
            budget
                .check_exact(budget.total_host_reserved_bytes + 1)
                .is_err()
        );
        assert!(AttachmentBudget::new(u64::MAX).is_err());
        assert!(AttachmentBudget::new(plan::LOGICAL_PUBLIC_RAM_BYTES - 1).is_err());
    }
    #[test]
    fn actual_typed_vectors_are_precharged_without_capacity_growth() {
        let owners = exact_vec::<Option<PublicExternalHandle>>().unwrap();
        let receipts = exact_vec::<PublicExternalDetachReceipt<WgpuResource>>().unwrap();
        assert_eq!(owners.capacity(), mirror::PAGES);
        assert_eq!(receipts.capacity(), mirror::PAGES);
        assert!(
            attachment_metadata(owners.capacity(), receipts.capacity()).unwrap()
                <= ATTACHMENT_HOST_BYTES
        );
        assert!(attachment_metadata(mirror::PAGES + 1, mirror::PAGES).is_err());
        assert!(attachment_metadata(usize::MAX, mirror::PAGES).is_err());
    }
    #[test]
    fn retired_registry_metadata_is_not_backing_or_private_cleanup_credit() {
        let usage = PublicExternalUsage {
            registration_count: mirror::PAGES,
            registered_backing_bytes: 0,
            total_registered_bytes: 8192,
            total_detached_bytes: 8192,
            metadata_owned_bytes: 65536,
        };
        validate_registry(usage, mirror::PAGES, 0, 8192).unwrap();
        for wrong in [
            PublicExternalUsage {
                registered_backing_bytes: 1,
                ..usage
            },
            PublicExternalUsage {
                registration_count: mirror::PAGES - 1,
                ..usage
            },
            PublicExternalUsage {
                total_detached_bytes: 8191,
                ..usage
            },
            PublicExternalUsage {
                metadata_owned_bytes: MAX_METADATA_BYTES + 1,
                ..usage
            },
        ] {
            assert!(validate_registry(wrong, mirror::PAGES, 0, 8192).is_err());
        }
        assert!(validate_registry(usage, mirror::PAGES, u64::MAX, 8192).is_err());
    }
    #[test]
    fn pending_workspace_or_repeated_detach_never_grants_release() {
        require_detach(Phase::Attached, 1).unwrap();
        for count in [0, 2, 48, usize::MAX] {
            assert!(require_detach(Phase::Attached, count).is_err());
        }
        assert!(require_detach(Phase::Detaching, 1).is_err());
        assert!(require_detach(Phase::Released, 1).is_err());
    }
}
