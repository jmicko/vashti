// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use crate::{
    config::{TypeNameFormatLevel, type_name_format},
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    server::{
        CommunicationId, ComputeServer, CopyDescriptor, CubeCount, ExecutionMode, Handle, IoError,
        KernelArguments, MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutPolicy,
        MemoryLayoutStrategy, ProfileError, ReduceOperation, ServerCommunication, ServerError,
        ServerUtilities,
    },
    storage::{ComputeStorage, ManagedResource},
};
use alloc::{boxed::Box, format, sync::Arc, vec, vec::Vec};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::{AllocationProperty, Bytes},
    device::{Device, DeviceId},
    device_handle::DeviceHandle,
    future::DynFut,
    profile::ProfileDuration,
};
use cubecl_ir::{DeviceProperties, ElemType, VectorSize, features::Features};
use cubecl_zspace::Shape;

#[allow(unused)]
use cubecl_common::profile::TimingMethod;
use cubecl_common::stream_id::StreamId;

/// The `ComputeClient` is the entry point to require tasks from the `ComputeServer`.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    device: DeviceHandle<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            utilities: self.utilities.clone(),
            stream_id: self.stream_id,
        }
    }
}

impl<R: Runtime> ComputeClient<R> {
    /// Get the info of the current backend.
    pub fn info(&self) -> &<R::Server as ComputeServer>::Info {
        &self.utilities.info
    }

    /// Create a new client with a new server.
    pub fn init<D: Device>(device: &D, server: R::Server) -> Self {
        let utilities = server.utilities();
        let context = DeviceHandle::<R::Server>::insert(device.to_id(), server)
            .expect("Can't create a new client on an already registered server");

        Self {
            device: context,
            utilities,
            stream_id: None,
        }
    }

    /// Load the client for the given device.
    pub fn load<D: Device>(device: &D) -> Self {
        let context = DeviceHandle::<R::Server>::new(device.to_id());

        // This is safe because we now know the return type of [`DeviceHandle::utilities()`].
        let utilities = context
            .utilities()
            .downcast::<ServerUtilities<R::Server>>()
            .expect("Can downcast to `ServerUtilities`");

        Self {
            device: context,
            utilities,
            stream_id: None,
        }
    }

    fn stream_id(&self) -> StreamId {
        match self.stream_id {
            Some(val) => val,
            None => StreamId::current(),
        }
    }

    /// Set the stream in which the current client is operating on.
    ///
    /// # Safety
    ///
    /// This is highly unsafe and should probably only be used by the CubeCL/Burn projects for now.
    pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
        self.stream_id = Some(stream_id);
    }

    fn do_read(&self, descriptors: Vec<CopyDescriptor>) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.read(descriptors, stream_id))
            .unwrap()
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        let shapes = handles
            .iter()
            .map(|it| [it.size_in_used() as usize].into())
            .collect::<Vec<Shape>>();
        let descriptors = handles
            .into_iter()
            .zip(shapes)
            .map(|(handle, shape)| CopyDescriptor::new(handle.binding(), shape, [1].into(), 1))
            .collect();

        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    pub fn read_one(&self, handle: Handle) -> Result<Bytes, ServerError> {
        Ok(cubecl_common::reader::read_sync(self.read_async(vec![handle]))?.remove(0))
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails. Useful for tests.
    pub fn read_one_unchecked(&self, handle: Handle) -> Bytes {
        cubecl_common::reader::read_sync(self.read_async(vec![handle]))
            .unwrap()
            .remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_tensor_async(
        &self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send {
        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    ///
    /// The tensor must be in the same layout as created by the runtime, or more strict.
    /// Contiguous tensors are always fine, strided tensors are only ok if the stride is similar to
    /// the one created by the runtime (i.e. padded on only the last dimension). A way to check
    /// stride compatibility on the runtime will be added in the future.
    ///
    /// Also see [`ComputeClient::create_tensor`].
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_tensor_async(descriptors)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_tensor_async(
        &self,
        descriptor: CopyDescriptor,
    ) -> impl Future<Output = Result<Bytes, ServerError>> + Send {
        let fut = self.read_tensor_async(vec![descriptor]);

        async { Ok(fut.await?.remove(0)) }
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_unchecked_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        handle: Handle,
    ) -> Result<
        ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>,
        ServerError,
    > {
        let stream_id = self.stream_id();
        let binding = handle.binding();

        self.device
            .submit_blocking(move |state| state.get_resource(binding, stream_id))
            .unwrap()
    }

    /// Explicit laboratory PUBLIC host metadata/backing reservation. This
    /// configures no ordinary/private allocator and changes no environment.
    pub fn configure_public_external_host(
        &self,
        budget: crate::public_external::PublicExternalBudget,
    ) -> Result<(), ServerError> {
        budget.validate().map_err(|reason| ServerError::Generic {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        })?;
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.configure_public_external_host(budget, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// # Safety
    /// Sole authenticated original-model owner; all raw aliases/readers remain
    /// excluded until finish. Only original public model bytes may be written.
    pub unsafe fn begin_public_external_rebase(
        &self,
        bindings: Vec<crate::server::Binding>,
    ) -> Result<crate::public_external::PublicExternalRebase, ServerError> {
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.begin_public_external_rebase(bindings, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external rebase channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }
    /// # Safety
    /// Original public bytes fully initialized/flushed, all raw writes/maps
    /// complete and every external queue joined. No inference/private payload.
    pub unsafe fn finish_public_external_rebase(
        &self,
        token: crate::public_external::PublicExternalRebase,
    ) -> Result<(), ServerError> {
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.finish_public_external_rebase(token, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external rebase channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Extend a precharged public registry without resetting prior identities.
    /// This adds no model backing itself and never applies to private payloads.
    pub fn extend_public_external_host(
        &self,
        budget: crate::public_external::PublicExternalBudget,
    ) -> Result<(), ServerError> {
        budget.validate().map_err(|reason| ServerError::Generic {
            reason: reason.into(),
            backtrace: BackTrace::capture(),
        })?;
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.extend_public_external_host(budget, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external extension channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Register a source whose unsafe constructor already establishes the
    /// caller's immutable PUBLIC provenance and retained-backing obligations.
    pub fn register_public_external_host(
        &self,
        source: crate::public_external::PublicExternalHostSource<
            <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource,
        >,
    ) -> Result<crate::public_external::PublicExternalHandle, ServerError>
    where
        <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: 'static,
    {
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.register_public_external_host(source, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Consuming release authority. On ANY failure backing must remain alive;
    /// no retry, erased-pool credit, or ordinary allocator reset is implied.
    pub fn detach_public_external_host(
        &self,
        handle: crate::public_external::PublicExternalHandle,
    ) -> Result<
        crate::public_external::PublicExternalDetachReceipt<
            <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource,
        >,
        ServerError,
    >
    where
        <<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource: 'static,
    {
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.detach_public_external_host(handle, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external channel unavailable; retain backing".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Public host ownership numbers; excludes managed MAIN/runtime counts.
    pub fn public_external_host_usage(
        &self,
    ) -> Result<crate::public_external::PublicExternalUsage, ServerError> {
        let stream = self.stream_id();
        self.device
            .submit_blocking(move |server| server.public_external_host_usage(stream))
            .map_err(|_| ServerError::Generic {
                reason: "public external channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Inspect only already initialized, retained FULL same-stream allocations.
    /// Channel ordering completes prior initialization calls; queued consumers
    /// need not run for immutable allocation identity/extent to be inspected.
    /// This is NOT a readback, submission, completion, or memory-credit fence.
    pub fn retained_resource_metadata(
        &self,
        handles: Vec<Handle>,
    ) -> Result<
        Vec<ManagedResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource>>,
        ServerError,
    > {
        let stream_id = self.stream_id();
        let bindings: Vec<_> = handles.into_iter().map(Handle::binding).collect();
        crate::transient::retained_metadata_bindings(&bindings, stream_id)?;
        self.device
            .submit_blocking(move |server| server.retained_resource_metadata(bindings, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "retained metadata channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Queue registered immutable model weights onto retained same-stream
    /// staging owners through the checked backend. This is not completion.
    pub fn stage_public_host_copies(
        &self,
        copies: Vec<crate::public_external::PublicHostCopy>,
    ) -> Result<(), ServerError> {
        let stream = self.stream_id();
        crate::public_external::validate_host_copies(&copies, stream).map_err(|reason| {
            ServerError::Generic {
                reason: reason.into(),
                backtrace: BackTrace::capture(),
            }
        })?;
        self.device
            .submit_blocking(move |server| server.stage_public_host_copies(copies, stream))
            .map_err(|_| ServerError::Generic {
                reason: "public host copy channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Ordered full-owner scratch clearing, without shader/INFO allocations.
    /// Requires configured private I/O; the owning caller must retain these
    /// allocations through its normal completion and whole-owner audit.
    pub fn clear_private_owners(&self, handles: Vec<Handle>) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        let bindings: Vec<_> = handles.into_iter().map(Handle::binding).collect();
        crate::transient::clear_owner_bindings(&bindings, stream_id)?;
        self.device
            .submit_blocking(move |server| server.clear_private_owners(bindings, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private clear channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        slices: Vec<Vec<u8>>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(slices)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.memory.clone().binding(),
                        desc.shape,
                        alloc.strides.clone(),
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    fn do_create(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
        mut data: Vec<Bytes>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        self.staging(data.iter_mut(), true);

        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let descriptors = descriptors
            .into_iter()
            .zip(layouts.iter())
            .zip(data)
            .map(|((desc, layout), data)| {
                (
                    CopyDescriptor::new(
                        layout.memory.clone().binding(),
                        desc.shape,
                        layout.strides.clone(),
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect::<Vec<_>>();

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(descriptors, stream_id);
        });

        Ok(layouts)
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [`Self::create`] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle {
        let shape: Shape = [slice.len()].into();

        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// Move immutable public model bytes into the ordinary upload queue.
    ///
    /// This native-lab path must never receive prompts, activations, routing
    /// packets or other private data. It preserves ordinary resource layout,
    /// stream ordering and server writes; only redundant host copies disappear.
    /// Private uploads must continue through `write_private` and its controller.
    pub fn create_public_owned(&self, data: Vec<u8>) -> Handle {
        let stream_id = self.stream_id();
        let shape: Shape = [data.len()].into();
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape.clone(), 1);
        let (handle_base, mut layouts) =
            self.utilities.layout_policy.apply(stream_id, &[descriptor]);
        let layout = layouts.remove(0);
        let copy = CopyDescriptor::new(
            layout.memory.clone().binding(),
            shape,
            layout.strides.clone(),
            1,
        );
        let data = Bytes::from_elems(data);
        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(vec![(copy, data)], stream_id);
        });
        layout.memory
    }

    /// Move immutable original public model-weight bytes into existing views.
    ///
    /// This native-lab path uses the same ordinary PUBLIC upload policy as
    /// [`Self::create_public_owned`], without allocating or initializing MAIN
    /// resources. It must never receive prompts, activations, routing packets
    /// or other private data; private uploads retain [`Self::write_private`].
    /// Every view must be an initialized same-stream, four-byte-aligned span
    /// matching its payload. The caller proves original weight/slot identity,
    /// retains each full bank owner, precharges all sources and excludes live
    /// consumers until its existing audited owning completion. Success means
    /// scheduling only: backend errors and completion require that readback.
    pub fn write_public_owned_batch(
        &self,
        writes: Vec<(Handle, Vec<u8>)>,
    ) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        let descriptors = public_write_descriptors(writes, stream_id)?;
        self.device.submit(move |server| {
            server.write(descriptors, stream_id);
        });
        Ok(())
    }

    /// Enable bounded explicitly owned I/O for this serialized stream.
    /// Use create_private for ALL private uploads; legacy create/create_from_slice
    /// remain PUBLIC APIs and can contain unguarded host/opaque GPU copies.
    pub fn configure_transient_io(&self, budget_bytes: u64) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.configure_transient_io(budget_bytes, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Fixed public target laboratory stream: 128 MiB owned I/O, batched INFO,
    /// inline readback and an explicitly charged 4096-entry/4-MiB reuse scope.
    /// Requires the actual inline device handle. Ordinary and draft streams
    /// keep configure_transient_io; no process setting is changed here.
    pub fn configure_public_target_request_io(&self) -> Result<(), ServerError> {
        if !DeviceHandle::<R::Server>::is_blocking() {
            return Err(ServerError::Generic {
                reason: "public target I/O requires inline device".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.configure_public_target_request_io(stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "public target I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Construct the same bounded public target I/O with an explicit immutable
    /// regulation window. Invoke before target model construction; generic and
    /// draft streams cannot acquire it through configure_transient_io.
    pub fn configure_public_target_request_io_with_backlog(
        &self,
        backlog: crate::server::PublicTargetSubmissionBacklog,
    ) -> Result<(), ServerError> {
        if !DeviceHandle::<R::Server>::is_blocking() {
            return Err(ServerError::Generic {
                reason: "public target I/O requires inline device".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| {
                server.configure_public_target_request_io_with_backlog(backlog, stream_id)
            })
            .map_err(|_| ServerError::Generic {
                reason: "public target I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Actual configured policy only; this query does not grant device success
    /// or credit a selected window that never reaches a regulation boundary.
    pub fn public_target_submission_backlog(
        &self,
    ) -> Result<crate::server::PublicTargetSubmissionBacklog, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.public_target_submission_backlog(stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "public target I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Content-free validation of this stream's immutable target I/O policy.
    /// A retired stream fails; an ordinary or unconfigured stream returns false.
    pub fn public_target_request_io_configured(&self) -> Result<bool, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.public_target_request_io_configured(stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "public target I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Move a heap-only private allocation to an owned/wiping controller.
    /// This path does not call staging(), to_vec(), clone(), or file controllers.
    /// Full capacity is wiped even when layout/scheduling/backend calls fail.
    /// Caller owns wiping any preceding copies and the destination main span.
    pub fn create_private(&self, data: Vec<u8>) -> Result<Handle, ServerError> {
        let data = crate::transient::upload_bytes(data);
        if data.is_empty() {
            return Err(ServerError::Generic {
                reason: "empty private upload".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let stream_id = self.stream_id();
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, [data.len()].into(), 1);
        let (base, mut layouts) = self
            .utilities
            .layout_policy
            .apply(stream_id, &[descriptor.clone()]);
        let layout = layouts.remove(0);
        let copy = CopyDescriptor::new(
            layout.memory.clone().binding(),
            descriptor.shape,
            layout.strides,
            1,
        );
        let (size, memory) = (base.size(), base.memory);
        let result = layout.memory;
        self.device
            .submit_blocking(move |server| {
                server.initialize_memory(memory, size, stream_id);
                server.write_private(vec![(copy, data)], stream_id)?;
                Ok(result)
            })
            .map_err(|_| ServerError::Generic {
                reason: "private I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Move private bytes to a destination already retained by its arena owner.
    /// Requires the full unoffset handle span and the client's current stream.
    /// No new main allocation or host-content copy occurs here; on scheduling
    /// failure the caller still owns its original handle for full-span cleanup.
    pub fn write_private(&self, handle: Handle, data: Vec<u8>) -> Result<(), ServerError> {
        let data = crate::transient::upload_bytes(data);
        let stream_id = self.stream_id();
        let descriptor = crate::transient::private_write_descriptor(handle, stream_id, data.len())?;
        self.device
            .submit_blocking(move |server| {
                server.write_private(vec![(descriptor, data)], stream_id)
            })
            .map_err(|_| ServerError::Generic {
                reason: "private I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Protected lab cleanup only: map a FULL retained same-stream span with
    /// a checked bounded poll. Caller must have completed ordinary inference
    /// (or explicitly abandoned it for cleanup), retain the owning handle, and
    /// drop returned Bytes before sanitizing/auditing its owned readback stage.
    /// Ordinary read/read_async and their inference waits remain unchanged.
    pub fn read_private_cleanup(&self, handle: Handle) -> Result<Bytes, ServerError> {
        let len = usize::try_from(handle.size()).map_err(|_| ServerError::Generic {
            reason: "private cleanup read span overflow".into(),
            backtrace: BackTrace::capture(),
        })?;
        let stream_id = self.stream_id();
        let descriptor = crate::transient::private_write_descriptor(handle, stream_id, len)?;
        self.device
            .submit_blocking(move |server| server.read_private_cleanup(descriptor, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Drain and explicitly erase owned stream-local I/O. Outstanding readback
    /// futures/Bytes must be dropped first. Any erasure failure retires the
    /// private stream; caller must retire the entire shared device context.
    pub fn sanitize_transient_io(
        &self,
    ) -> DynFut<Result<crate::transient::TransientIoReceipt, ServerError>> {
        let stream_id = self.stream_id();
        match self
            .device
            .submit_blocking(move |server| server.sanitize_transient_io(stream_id))
        {
            Ok(future) => future,
            Err(_) => Box::pin(async {
                Err(ServerError::Generic {
                    reason: "private I/O channel unavailable".into(),
                    backtrace: BackTrace::capture(),
                })
            }),
        }
    }

    /// Protected lab fixture: same sanitizer, plus zero-only full-span
    /// inspection before release and erasure of its bounded audit staging.
    /// All readback futures/Bytes must be dropped; returns counts, never bytes.
    pub fn sanitize_transient_io_audited(
        &self,
    ) -> DynFut<Result<crate::transient::TransientIoReceipt, ServerError>> {
        let stream_id = self.stream_id();
        match self
            .device
            .submit_blocking(move |server| server.sanitize_transient_io_audited(stream_id))
        {
            Ok(future) => future,
            Err(_) => Box::pin(async {
                Err(ServerError::Generic {
                    reason: "private I/O channel unavailable".into(),
                    backtrace: BackTrace::capture(),
                })
            }),
        }
    }

    /// Begin an exclusive request-scoped pool on this exact client stream.
    /// No environment variable alone permits dirty reuse or cross-owner access.
    pub fn begin_transient_io_reuse(&self) -> Result<u64, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.begin_transient_io_reuse(stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O reuse channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Begin only the separately admitted fixed public-target reuse scope.
    /// Ordinary reuse cannot activate its larger metadata reservation.
    pub fn begin_public_target_request_io_reuse(&self) -> Result<u64, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.begin_public_target_request_io_reuse(stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "public target I/O reuse channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Checked completed fence. Buffers remain dirty, owned and fully charged;
    /// caller must still end the scope before handing the stream to another request.
    pub fn recycle_transient_io(&self, nonce: u64) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.recycle_transient_io(nonce, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O reuse channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Accept this request checkpoint without remapping, clearing or publishing
    /// buffers for reuse. The eventual recycle and full closing audit remain due.
    pub fn accept_transient_io_reuse(&self, nonce: u64) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.accept_transient_io_reuse(nonce, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O reuse channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// Synchronous full inspected erasure of every retained scope owner.
    pub fn end_transient_io_reuse(
        &self,
        nonce: u64,
    ) -> Result<crate::transient::TransientIoReceipt, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.end_transient_io_reuse(nonce, stream_id))
            .map_err(|_| ServerError::Generic {
                reason: "private I/O reuse channel unavailable".into(),
                backtrace: BackTrace::capture(),
            })?
    }

    /// todo: docs
    pub fn exclusive<'a, Re: Send + 'static, F: FnOnce() -> Re + Send + 'a>(
        &'a self,
        task: F,
    ) -> Result<Re, ServerError> {
        // We then launch the task.
        self.device
            .exclusive(task)
            .map_err(|err| ServerError::Generic {
                reason: format!("Communication channel with the server is down: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    /// dodo: Docs
    pub fn memory_persistent_allocation<
        'a,
        Re: Send,
        Input: Send,
        F: FnOnce(Input) -> Re + Send + 'a,
    >(
        &'a self,
        input: Input,
        task: F,
    ) -> Result<Re, ServerError> {
        let stream_id = StreamId::current();

        self.device.submit(move |server| {
            server.allocation_mode(MemoryAllocationMode::Persistent, stream_id);
        });

        // All tasks created on the same stream will have persistent memory.
        let output = task(input);

        self.device.submit(move |server| {
            server.allocation_mode(MemoryAllocationMode::Auto, stream_id);
        });

        Ok(output)
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle {
        let shape = [data.len()].into();

        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                shape,
                1,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
        .memory
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensor`] for better performance.
    pub fn create_tensor_from_slice(
        &self,
        slice: &[u8],
        shape: Shape,
        elem_size: usize,
    ) -> MemoryLayout {
        self.do_create_from_slices(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![slice.to_vec()],
        )
        .unwrap()
        .remove(0)
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    pub fn create_tensor(&self, bytes: Bytes, shape: Shape, elem_size: usize) -> MemoryLayout {
        self.do_create(
            vec![MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Optimized,
                shape,
                elem_size,
            )],
            vec![bytes],
        )
        .unwrap()
        .remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensors`] for better performance.
    pub fn create_tensors_from_slices(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, &[u8])>,
    ) -> Vec<MemoryLayout> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut descriptors_ = Vec::with_capacity(descriptors.len());
        for (a, b) in descriptors {
            data.push(b.to_vec());
            descriptors_.push(a);
        }

        self.do_create_from_slices(descriptors_, data).unwrap()
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(MemoryLayoutDescriptor, Bytes)>,
    ) -> Vec<MemoryLayout> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<MemoryLayoutDescriptor>,
    ) -> Result<Vec<MemoryLayout>, IoError> {
        let stream_id = self.stream_id();
        let (handle_base, layouts) = self.utilities.layout_policy.apply(stream_id, &descriptors);

        let (size, memory) = (handle_base.size(), handle_base.memory);
        self.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
        });

        Ok(layouts)
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        let shape: Shape = [size].into();
        let descriptor = MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, 1);
        self.do_empty(vec![descriptor]).unwrap().remove(0).memory
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensor(&self, shape: Shape, elem_size: usize) -> MemoryLayout {
        let descriptor =
            MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).unwrap().remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensors(&self, descriptors: Vec<MemoryLayoutDescriptor>) -> Vec<MemoryLayout> {
        self.do_empty(descriptors).unwrap()
    }

    /// Marks the given [Bytes] as being a staging buffer, maybe transferring it to pinned memory
    /// for faster data transfer with compute device.
    ///
    /// TODO: This blocks the compute queue, so it will drop the compute utilization.
    pub fn staging<'a, I>(&self, bytes: I, file_only: bool)
    where
        I: Iterator<Item = &'a mut Bytes>,
    {
        let has_staging = |b: &Bytes| match b.property() {
            AllocationProperty::Pinned => false,
            AllocationProperty::File => true,
            AllocationProperty::Native | AllocationProperty::Other => !file_only,
        };

        let mut to_be_updated = Vec::new();
        let sizes = bytes
            .filter_map(|b| match has_staging(b) {
                true => {
                    let len = b.len();
                    to_be_updated.push(b);
                    Some(len)
                }
                false => None,
            })
            .collect::<Vec<usize>>();

        if sizes.is_empty() {
            return;
        }

        let stream_id = self.stream_id();
        let sizes = sizes.to_vec();
        let stagings = self
            .device
            .submit_blocking(move |server| server.staging(&sizes, stream_id))
            .unwrap();

        let stagings = match stagings {
            Ok(val) => val,
            Err(_) => return,
        };

        to_be_updated
            .into_iter()
            .zip(stagings)
            .for_each(|(b, mut staging)| {
                b.copy_into(&mut staging);
                core::mem::swap(b, &mut staging);
            });
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst_server))
    )]
    pub fn to_client(&mut self, src: Handle, dst_server: &Self, dtype: ElemType) -> Handle {
        let shape = [src.size_in_used() as usize];
        let src_descriptor = src.copy_descriptor(shape.into(), [1].into(), 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server, dtype)
        } else {
            let alloc_desc = MemoryLayoutDescriptor::new(
                MemoryLayoutStrategy::Contiguous,
                src_descriptor.shape.clone(),
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
                .memory
        }
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, device_ids))
    )]
    pub fn ensure_init_collective(&mut self, device_ids: Vec<DeviceId>) {
        let comm_id = CommunicationId::from(device_ids.clone());
        let is_comms_init = self
            .utilities
            .initialized_comms
            .read()
            .unwrap()
            .contains(&comm_id);
        if !is_comms_init {
            self.device
                .submit(move |server| server.comm_init(device_ids).unwrap());
            let mut initialized_comms = self.utilities.initialized_comms.write().unwrap();
            initialized_comms.insert(comm_id);
            // Flush immediately so other devices aren't blocked waiting on this initialization.
            self.device.flush_queue();
        }
    }

    /// Wait on the communication stream.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn sync_collective(&self) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `sync_collective` with a blocking device handle");
        }
        let stream_id = self.stream_id();

        self.device.submit(move |server| {
            server.sync_collective(stream_id).unwrap();
        });

        // We don't actually need or want to sync the server here, but we need to make sure any
        // task enqueued on the communication channel is done.
        self.device.flush_queue();
    }

    /// Perform an `all_reduce` operation on the given devices.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst, dtype, device_ids, op))
    )]
    pub fn all_reduce(
        &mut self,
        src: Handle,
        dst: Handle,
        dtype: ElemType,
        device_ids: Vec<DeviceId>,
        op: ReduceOperation,
    ) {
        if DeviceHandle::<R::Server>::is_blocking() {
            panic!("Can't use `all_reduce` with a blocking device handle");
        }

        let stream_id = self.stream_id();
        let src = src.binding();
        let dst = dst.binding();

        self.ensure_init_collective(device_ids.clone());

        self.device.submit(move |server| {
            server
                .all_reduce(src, dst, dtype, stream_id, op, device_ids)
                .unwrap();
        });
    }

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src_descriptor, dst_server))
    )]
    pub fn to_client_tensor(
        &mut self,
        src_descriptor: CopyDescriptor,
        dst_server: &Self,
        dtype: ElemType,
    ) -> Handle {
        let stream_id_src = self.stream_id();
        let stream_id_dst = dst_server.stream_id();

        let device_id_src = self.device.device_id();
        let device_id_dst = dst_server.device.device_id();

        let mut dst_server = dst_server.clone();
        let handle = Handle::new(stream_id_dst, src_descriptor.handle.size_in_used());
        let handle_cloned = handle.clone();

        let device_ids = vec![device_id_src, device_id_dst];
        self.ensure_init_collective(device_ids.clone());
        dst_server.ensure_init_collective(device_ids);

        self.device.submit(move |server_src| {
            server_src
                .send(src_descriptor, dtype, stream_id_src, device_id_dst)
                .unwrap()
        });

        dst_server.device.submit(move |server_dst| {
            server_dst
                .recv(handle_cloned, dtype, stream_id_dst, device_id_src)
                .unwrap();
            server_dst.sync_collective(stream_id_dst).unwrap();
        });

        // `ServerCommunication::send` and`ServerCommunication::recv` are blocking: they each wait for the corresponding recv/send
        // call to be made. We flush the operations right away so that the neither server ends up in a deadlock.
        // The actual data transfer is still executed asynchronously on the communication stream.
        self.device.flush_queue();
        dst_server.device.flush_queue();

        handle
    }

    #[track_caller]
    #[cfg_attr(feature = "tracing", tracing::instrument(level="trace",
        skip(self, kernel, bindings),
        fields(
            kernel.name = %kernel.name(),
            kernel.id = %kernel.id(),
        )
    ))]
    unsafe fn launch_inner(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let level = self.utilities.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                let utilities = self.utilities.clone();
                self.device.submit(move |state| {
                    let name = kernel.name();
                    unsafe { state.launch(kernel, count, bindings, mode, stream_id) };

                    if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                        let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                        utilities.logger.register_execution(info);
                    }
                });
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let context = self.device.clone();
                let count_moved = count.clone();
                let (result, profile) = self
                    .profile(
                        move || {
                            context
                                .submit_blocking(move |state| unsafe {
                                    state.launch(kernel, count_moved, bindings, mode, stream_id)
                                })
                                .unwrap()
                        },
                        name,
                    )
                    .unwrap();
                let info = match level {
                    ProfileLevel::Full => {
                        format!("{name}: {kernel_id} CubeCount {count:?}")
                    }
                    _ => type_name_format(name, TypeNameFormatLevel::Balanced),
                };
                self.utilities.logger.register_profiled(info, profile);
                result
            }
        }
    }

    /// Launches the `kernel` with the given `bindings`.
    #[track_caller]
    pub fn launch(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
    ) {
        // SAFETY: Using checked execution mode.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Checked,
                self.stream_id(),
            )
        }
    }

    /// Launches the `kernel` with the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// To ensure this is safe, you must verify your kernel:
    /// - Has no out-of-bound reads and writes that can happen.
    /// - Has no infinite loops that might never terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
    ) {
        // SAFETY: Caller has to uphold kernel being safe.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                match self.utilities.check_mode {
                    crate::config::compilation::BoundsCheckMode::Enforce => ExecutionMode::Checked,
                    crate::config::compilation::BoundsCheckMode::Validate => {
                        ExecutionMode::Validate
                    }
                    crate::config::compilation::BoundsCheckMode::Auto => ExecutionMode::Unchecked,
                },
                self.stream_id(),
            )
        }
    }

    /// Pause or resume only an explicitly configured per-client profiler.
    /// No device work, private allocation, fence or ownership transition occurs.
    pub fn set_profiling_paused(&self, paused: bool) {
        self.utilities.logger.set_profiling_paused(paused);
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();

        self.device
            .submit_blocking(move |server| server.flush(stream_id))
            .unwrap()
    }

    /// Selected native Intel expert submission. This is NOT completion or
    /// erasure authority; normal sync/readback/cleanup still joins the work.
    pub fn submit_native_strata_experts(&self) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.submit_native_strata_experts(stream_id))
            .unwrap()
    }

    /// Begin a stable native shape on this exact serialized request stream.
    pub fn begin_native_strata_branch(
        &self,
        key: crate::server::NativeStrataBranch,
    ) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.begin_native_strata_branch(stream_id, key))
            .unwrap()
    }
    /// Join and seal the original branch. Private owners remain charged.
    pub fn end_native_strata_branch(
        &self,
        key: crate::server::NativeStrataBranch,
    ) -> Result<(), ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.end_native_strata_branch(stream_id, key))
            .unwrap()
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) -> DynFut<Result<(), ServerError>> {
        let stream_id = self.stream_id();

        let fut = self
            .device
            .submit_blocking(move |server| server.sync(stream_id))
            .unwrap();

        self.utilities.logger.profile_summary();

        fut
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties {
        &self.utilities.properties
    }

    /// Get the features supported by the compute server.
    pub fn features(&self) -> &Features {
        &self.utilities.properties.features
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> {
        Arc::get_mut(&mut self.utilities).map(|state| &mut state.properties)
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> Result<MemoryUsage, ServerError> {
        let stream_id = self.stream_id();
        self.device
            .submit_blocking(move |server| server.memory_usage(stream_id))
            .unwrap()
    }

    /// Get all devices of a specific type available to this runtime
    pub fn enumerate_devices(&self, type_id: u16) -> Vec<DeviceId> {
        R::enumerate_devices(type_id, self.info())
    }

    /// Get all devices available to this runtime
    pub fn enumerate_all_devices(&self) -> Vec<DeviceId> {
        R::enumerate_all_devices(self.info())
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count(&self, type_id: u16) -> usize {
        self.enumerate_devices(type_id).len()
    }

    /// Get the number of devices of a specific type available to this runtime
    pub fn device_count_total(&self) -> usize {
        self.enumerate_all_devices().len()
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.allocation_mode(mode, stream_id));
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        let stream_id = self.stream_id();
        self.device
            .submit(move |server| server.memory_cleanup(stream_id));
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O: Send + 'static>(
        &self,
        func: impl FnOnce() -> O + Send,
        #[allow(unused)] func_name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        // Get the outer caller. For execute() this points straight to the
        // cube kernel. For general profiling it points to whoever calls profile.
        #[cfg(feature = "profile-tracy")]
        let location = std::panic::Location::caller();

        // Make a CPU span. If the server has system profiling this is all you need.
        #[cfg(feature = "profile-tracy")]
        let _span = tracy_client::Client::running().unwrap().span_alloc(
            None,
            func_name,
            location.file(),
            location.line(),
            0,
        );

        let stream_id = self.stream_id();

        #[cfg(feature = "profile-tracy")]
        let gpu_span = if self.utilities.properties.timing_method == TimingMethod::Device {
            let gpu_span = self
                .utilities
                .gpu_client
                .span_alloc(func_name, "profile", location.file(), location.line())
                .unwrap();
            Some(gpu_span)
        } else {
            None
        };

        let device = self.device.clone();
        #[allow(unused_mut, reason = "Used in profile-tracy")]
        let mut result = self
            .device
            .exclusive(move || {
                // We first get mut access to the server to create a token.
                // Then we free to server, since it's going to be accessed in `func()`.
                let token =
                    match device.submit_blocking(move |server| server.start_profile(stream_id)) {
                        Ok(token) => match token {
                            Ok(token) => token,
                            Err(err) => return Err(err),
                        },
                        Err(err) => {
                            return Err(ServerError::Generic {
                                reason: alloc::format!(
                                    "Can't start profiling because of a call error: {err:?}"
                                ),
                                backtrace: BackTrace::capture(),
                            });
                        }
                    };

                // We execute `func()` which will recursibly access the server.
                let out = func();

                // Finally we get the result from the token.
                let result = device
                    .submit_blocking(move |server| {
                        let mut result = server.end_profile(stream_id, token);

                        match result {
                            Ok(result) => Ok((out, result)),
                            Err(err) => Err(err),
                        }
                    })
                    .unwrap();

                Ok(result)
            })
            .unwrap()
            .map_err(|err| ProfileError::Unknown {
                reason: alloc::format!("{err:?}"),
                backtrace: BackTrace::capture(),
            })?;

        #[cfg(feature = "profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.utilities.epoch_time;
            // Add in the work to upload the timestamp data.
            result = result.map(|(o, result)| {
                (
                    o,
                    ProfileDuration::new(
                        alloc::boxed::Box::pin(async move {
                            let ticks = result.resolve().await;
                            let start_duration =
                                ticks.start_duration_since(epoch).as_nanos() as i64;
                            let end_duration = ticks.end_duration_since(epoch).as_nanos() as i64;
                            gpu_span.upload_timestamp_start(start_duration);
                            gpu_span.upload_timestamp_end(end_duration);
                            ticks
                        }),
                        TimingMethod::Device,
                    ),
                )
            });
        }

        result
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "trace",
            skip(self, src_descriptor, alloc_descriptor, dst_server)
        )
    )]
    fn change_client_sync(
        &self,
        src_descriptor: CopyDescriptor,
        alloc_descriptor: MemoryLayoutDescriptor,
        dst_server: &Self,
    ) -> MemoryLayout {
        let shape = src_descriptor.shape.clone();
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        let read = self
            .device
            .submit_blocking(move |server| server.read(vec![src_descriptor], stream_id))
            .unwrap();

        let mut data = cubecl_common::future::block_on(read).unwrap();

        let (handle_base, mut layouts) = self
            .utilities
            .layout_policy
            .apply(stream_id, &[alloc_descriptor]);
        let alloc = layouts.remove(0);

        let desc_descriptor = CopyDescriptor {
            handle: handle_base.clone().binding(),
            shape,
            strides: alloc.strides.clone(),
            elem_size,
        };

        let (size, memory) = (handle_base.size(), handle_base.memory);
        dst_server.device.submit(move |server| {
            server.initialize_memory(memory, size, stream_id);
            server.write(vec![(desc_descriptor, data.remove(0))], stream_id)
        });

        alloc
    }

    /// Returns all vector sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_vector_sizes(
        &self,
        size: usize,
    ) -> impl Iterator<Item = VectorSize> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let size_bits = size * 8;
        let max = load_width / size_bits;
        let max = usize::min(self.properties().hardware.max_vector_size, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = max.trailing_zeros() + 1;

        (0..num_candidates).map(|i| 2usize.pow(i)).rev()
    }
}

fn public_write_descriptors(
    writes: Vec<(Handle, Vec<u8>)>,
    stream: StreamId,
) -> Result<Vec<(CopyDescriptor, Bytes)>, ServerError> {
    // Validate the entire batch before moving payloads or scheduling any write.
    // Avoid size_in_used(): forged offsets can underflow its subtraction.
    if writes.is_empty()
        || writes.iter().any(|(handle, data)| {
            let first = handle.offset_start.unwrap_or(0);
            let span = handle
                .size()
                .checked_sub(first)
                .and_then(|size| size.checked_sub(handle.offset_end.unwrap_or(0)));
            handle.stream != stream
                || data.is_empty()
                || !first.is_multiple_of(4)
                || !data.len().is_multiple_of(4)
                || u64::try_from(data.len()).ok() != span
        })
    {
        return Err(ServerError::Generic {
            reason: "public weight write requires exact aligned same-stream views".into(),
            backtrace: BackTrace::capture(),
        });
    }
    Ok(writes
        .into_iter()
        .map(|(handle, data)| {
            let descriptor = handle.copy_descriptor([data.len()].into(), [1].into(), 1);
            (descriptor, Bytes::from_elems(data))
        })
        .collect())
}

#[cfg(test)]
mod public_write_tests {
    use super::*;

    #[test]
    fn public_write_preserves_existing_views_and_moves_payloads() {
        let stream = StreamId { value: 37 };
        let owner = Handle::new(stream, 32768);
        let view = owner.clone().offset_start(16).offset_end(32768 - 24);
        let data = vec![1u8; 8];
        let pointer = data.as_ptr();
        let descriptors = public_write_descriptors(
            vec![(view, data), (owner.clone(), vec![2u8; 32768])],
            stream,
        )
        .unwrap();
        assert_eq!(descriptors.len(), 2);
        let (descriptor, bytes) = &descriptors[0];
        assert_eq!(
            descriptor.handle.memory.descriptor().id,
            owner.memory.descriptor().id
        );
        assert_eq!(descriptor.handle.size(), owner.size());
        assert_eq!(descriptor.handle.offset_start, Some(16));
        assert_eq!(descriptor.handle.offset_end, Some(32768 - 24));
        assert_eq!(descriptor.handle.size_in_used(), 8);
        assert_eq!(descriptor.shape.as_slice(), &[8]);
        assert_eq!(descriptor.strides.as_ref(), &[1]);
        assert_eq!(descriptor.elem_size, 1);
        assert_eq!(bytes.as_ptr(), pointer);
        assert_eq!(descriptors[1].0.handle.size_in_used(), 32768);
    }

    #[test]
    fn public_write_rejects_invalid_batches_before_scheduling() {
        let stream = StreamId { value: 37 };
        let owner = Handle::new(stream, 32);
        assert!(public_write_descriptors(Vec::new(), stream).is_err());
        for (view, length) in [
            (owner.clone(), 0),
            (owner.clone(), 28),
            (owner.clone().offset_start(1).offset_end(3), 28),
            (owner.clone().offset_start(4).offset_end(3), 25),
            (owner.clone().offset_start(u64::MAX), 4),
            (owner.clone().offset_end(u64::MAX), 4),
            (owner.clone().offset_start(20).offset_end(20), 4),
            (Handle::new(StreamId { value: 38 }, 32), 32),
        ] {
            assert!(
                public_write_descriptors(
                    vec![(owner.clone(), vec![1; 32]), (view, vec![2; length])],
                    stream,
                )
                .is_err()
            );
        }
    }
}
