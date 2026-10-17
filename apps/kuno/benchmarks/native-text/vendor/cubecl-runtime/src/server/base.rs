// Modified for Kuno's laboratory transient-I/O ownership/erasure patch (2026-10-05).
use super::Handle;
use crate::{
    client::ComputeClient,
    compiler::CompilationError,
    config::{CubeClRuntimeConfig, RuntimeConfig, compilation::BoundsCheckMode},
    kernel::KernelMetadata,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    server::Binding,
    storage::{ComputeStorage, ManagedResource},
    tma::{OobFill, TensorMapFormat, TensorMapInterleave, TensorMapPrefetch, TensorMapSwizzle},
};
use ahash::AHasher;
use alloc::boxed::Box;
#[cfg(feature = "profile-tracy")]
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::{
    fmt::Debug,
    hash::{Hash, Hasher},
};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::Bytes,
    device::{self, DeviceId},
    future::DynFut,
    profile::ProfileDuration,
    stream_id::StreamId,
    stub::RwLock,
};
use cubecl_ir::{DeviceProperties, ElemType, StorageType};
use cubecl_zspace::{Shape, Strides, metadata::Metadata};
use hashbrown::HashSet;
use thiserror::Error;

/// Immutable construction policy for the separately admitted public target
/// stream. Ordinary streams retain the original 512-task regulation window.
/// This does not change the scheduler/submission size or authorize completion.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PublicTargetSubmissionBacklog {
    /// Original regulator, with its historical zero-task submission cost.
    #[default]
    Default512,
    /// Explicit native target experiment; each actual submission costs at least one.
    Queued2048,
}
impl PublicTargetSubmissionBacklog {
    /// Accumulated task-cost threshold for the original previous-stamp wait.
    pub const fn task_window(self) -> usize {
        match self {
            Self::Default512 => 512,
            Self::Queued2048 => 2048,
        }
    }

    /// A submitted INFO-only command buffer spends one selected-policy slot.
    /// The original policy retains its historical zero-task accounting.
    pub const fn minimum_submission_cost(self) -> usize {
        match self {
            Self::Default512 => 0,
            Self::Queued2048 => 1,
        }
    }
}

/// Content-free shape/model namespace of a selected native Intel branch. The
/// first recording authenticates and retains its exact buffer inventory.
/// Positions, tokens, probabilities and sampling seeds are never keys.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct NativeStrataBranch {
    model_owner: u64,
    coupled_sampling: bool,
    family: u8,
    rows: u8,
    proposals: u8,
    context_capacity: u16,
    verify_capacity: u8,
    state_role: u8,
}
/// Maximum immutable capacity of the selected Intel target/draft ABI.
/// Default branch constructors retain their original4096-cell geometry.
pub const NATIVE_STRATA_MAX_CONTEXT: u16 = 32768;
impl NativeStrataBranch {
    /// Separate original coupled-sampler and greedy/argmax command inventories.
    pub fn with_coupled_sampling(mut self, coupled: bool) -> Self {
        self.coupled_sampling = coupled;
        self
    }
    /// Static MTP geometry: accepted rows, first proposal index, proposal count.
    pub fn mtp_geometry(self) -> Option<(u8, u8, u8)> {
        match self.family {
            2 => Some((self.rows, 0, self.proposals)),
            3 => Some((1, self.proposals, 1)),
            _ => None,
        }
    }
    /// Namespace recordings by the immutable model-instance identity. This is
    /// content-free and restores per-drafter graph ownership on a shared stream.
    pub fn with_model_owner(mut self, owner: u64) -> Result<Self, &'static str> {
        if owner == 0 {
            return Err("native model recording owner missing");
        }
        self.model_owner = owner;
        Ok(self)
    }
    /// Resolve a declared math phase from immutable branch geometry only.
    pub fn phase_offset(self, phase: NativeStrataPhase) -> Result<i16, &'static str> {
        match (self.family, phase) {
            (1 | 2 | 4, NativeStrataPhase::Front) => Ok(0),
            (2, NativeStrataPhase::Draft { index }) if index < self.proposals => {
                Ok(i16::from(self.rows - 1) + i16::from(index))
            }
            (3, NativeStrataPhase::Draft { index }) if index == self.proposals => Ok(0),
            _ => Err("native branch phase shape"),
        }
    }
    /// Static target geometry only; no request controls or access authority.
    pub fn target_geometry(self) -> Option<(u8, u16, u8)> {
        (self.family == 1).then_some((self.rows, self.context_capacity, self.state_role))
    }
    /// The admitted MAX8 target shape at the maximum 4096-cell context.
    pub fn target_window(rows: u8) -> Result<Self, &'static str> {
        Self::target_window_for_context(rows, 4096)
    }
    /// Verification-only target shape with its actual immutable context capacity.
    pub fn target_window_for_context(rows: u8, context: u16) -> Result<Self, &'static str> {
        Self::target_window_with_state(rows, context, 0)
    }
    /// Fixed original GDN (bit0) and PLE (bit1) old/next bindings, never a token position.
    pub fn target_window_with_state(
        rows: u8,
        context: u16,
        state_role: u8,
    ) -> Result<Self, &'static str> {
        if !(1..=8).contains(&rows)
            || !(1..=NATIVE_STRATA_MAX_CONTEXT).contains(&context)
            || state_role > 3
        {
            return Err("native target branch shape");
        }
        Ok(Self {
            model_owner: 0,
            coupled_sampling: false,
            family: 1,
            rows,
            proposals: 0,
            context_capacity: context,
            verify_capacity: 8,
            state_role,
        })
    }
    /// Retained target state commit. Accepted n is a runtime INFO value, never
    /// a graph key. Count1 uses its verified candidate without a commit graph.
    pub fn target_commit(rows: u8, context: u16, state_role: u8) -> Result<Self, &'static str> {
        if !(2..=8).contains(&rows) {
            return Err("native commit branch shape");
        }
        let mut key = Self::target_window_with_state(rows, context, state_role)?;
        key.family = 4;
        Ok(key)
    }
    /// Immutable proposed-window geometry of a commit, not accepted/request data.
    pub fn commit_geometry(self) -> Option<(u8, u16, u8)> {
        (self.family == 4).then_some((self.rows, self.context_capacity, self.state_role))
    }
    /// One accepted-prefix round, optionally followed by a zero-threshold chain.
    pub fn mtp_round(rows: u8, proposals: u8) -> Result<Self, &'static str> {
        Self::mtp_round_for_context(rows, proposals, 4096)
    }
    /// Immutable cache capacity, independent of current request position.
    pub fn mtp_round_for_context(
        rows: u8,
        proposals: u8,
        context: u16,
    ) -> Result<Self, &'static str> {
        if !(1..=NATIVE_STRATA_MAX_CONTEXT).contains(&context) {
            return Err("native MTP context shape");
        }
        if !(1..=8).contains(&rows) || !(1..=7).contains(&proposals) {
            return Err("native MTP round branch shape");
        }
        Ok(Self {
            model_owner: 0,
            coupled_sampling: false,
            family: 2,
            rows,
            proposals,
            context_capacity: context,
            verify_capacity: 8,
            state_role: 0,
        })
    }
    /// A subsequent confidence-observed draft cell (indices one through six).
    pub fn mtp_step(step: u8) -> Result<Self, &'static str> {
        Self::mtp_step_for_context(step, 4096)
    }
    /// Immutable cache capacity, independent of current request position.
    pub fn mtp_step_for_context(step: u8, context: u16) -> Result<Self, &'static str> {
        if !(1..=NATIVE_STRATA_MAX_CONTEXT).contains(&context) {
            return Err("native MTP context shape");
        }
        if !(1..=6).contains(&step) {
            return Err("native MTP step branch shape");
        }
        Ok(Self {
            model_owner: 0,
            coupled_sampling: false,
            family: 3,
            rows: 1,
            proposals: step,
            context_capacity: context,
            verify_capacity: 8,
            state_role: 0,
        })
    }
}

/// Symbolic source of one native INFO u32. Offsets are immutable graph shape,
/// never inferred by comparing request values or included in kernel identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeStrataValue {
    /// Current window's position plus a shape-defined displacement.
    Position {
        /// Constant displacement encoded by the kernel's fixed graph shape.
        offset: i16,
    },
    /// Window boundary, which may equal the admitted context end.
    Boundary {
        /// Constant displacement from the invocation origin.
        offset: i16,
    },
    /// Declared current math phase; resolved to a static window displacement
    /// before template capture. It never stores the actual current position.
    PhasePosition {
        /// Constant displacement from the shape-defined current phase.
        offset: i16,
    },
    /// Phase-relative boundary rather than a readable token cell.
    PhaseBoundary {
        /// Constant displacement from the shape-defined current phase.
        offset: i16,
    },
    /// Actual accepted input count of a target commit, bounded by its shape.
    AcceptedCount,
    /// Current expert-routing stamp, supplied outside the branch identity.
    RouteEpoch,
    /// Whether this target window contributes to the original expert-usage map.
    RouteUsage,
}
/// Static native math phase, not a request value or an access/completion grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeStrataPhase {
    /// Accepted-prefix front; also the sole target phase.
    Front,
    /// Selected rest or follow-up draft, with the original proposal index.
    Draft {
        /// Original proposal index, bounded by this branch's fixed geometry.
        index: u8,
    },
}
/// Typed location in one original INFO record; contains no invocation values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeStrataParameter {
    /// Little-endian u32 index within the original INFO record.
    pub word: u16,
    /// Symbolic control source; never the actual request value.
    pub value: NativeStrataValue,
}
/// Pending native submission identity. No Debug/Serialize, completion or
/// numerical/state authority. Only the original backend may acknowledge it.
#[derive(Clone, Copy)]
pub struct NativeStrataSubmission {
    graph: u64,
    submission: u64,
    scope: u64,
}
impl NativeStrataSubmission {
    /// Backend metadata constructor, not proof of submission/completion. The
    /// joining backend must validate the original retained scope/flight/fence.
    pub fn from_queue(graph: u64, submission: u64, scope: u64) -> Result<Self, &'static str> {
        if graph == 0 || submission == 0 || scope == 0 {
            return Err("native pending identity bounds");
        }
        Ok(Self {
            graph,
            submission,
            scope,
        })
    }
    /// For original backend validation only. Do not log private scope identities.
    pub fn original_identity(self) -> (u64, u64, u64) {
        (self.graph, self.submission, self.scope)
    }
}
/// Original verified-R dependency/scope identity. No content, completion,
/// model-ready or erasure authority; the original backend validates every use.
#[derive(Clone, Copy)]
pub struct NativeStrataDraftScope {
    graph: u64,
    generation: u64,
    request: u64,
}
impl NativeStrataDraftScope {
    /// Backend metadata only. A fabricated token cannot create a live scope.
    pub fn from_queue(graph: u64, generation: u64, request: u64) -> Result<Self, &'static str> {
        if graph == 0 || generation == 0 || request == 0 {
            return Err("native draft identity bounds");
        }
        Ok(Self {
            graph,
            generation,
            request,
        })
    }
    /// Original backend validation only; do not log private identities.
    pub fn original_identity(self) -> (u64, u64, u64) {
        (self.graph, self.generation, self.request)
    }
}
/// Current request's control values. Deliberately no Debug, Hash or Serialize.
/// The branch key and retained template contain no copy of these values.
#[derive(Clone, Copy)]
pub struct NativeStrataInvocation {
    kind: u32,
    position: u32,
    route_epoch: u32,
    route_usage: u32,
    accepted: u32,
}
impl NativeStrataInvocation {
    /// Validate fresh target controls before opening the retained shape.
    pub fn target(position: u32, route_epoch: u32) -> Result<Self, &'static str> {
        Self::target_with_usage(position, route_epoch, false)
    }
    /// Fresh target controls, including the actual verification usage policy.
    pub fn target_with_usage(
        position: u32,
        route_epoch: u32,
        route_usage: bool,
    ) -> Result<Self, &'static str> {
        if position >= u32::from(NATIVE_STRATA_MAX_CONTEXT) || route_epoch == 0 {
            return Err("native target invocation bounds");
        }
        Ok(Self {
            kind: 1,
            position,
            route_epoch,
            route_usage: u32::from(route_usage),
            accepted: 0,
        })
    }
    /// Fresh MTP cell origin: accepted-front start for a round, current cell
    /// for an independently observed step. Geometry comes from its branch key.
    pub fn mtp(position: u32) -> Result<Self, &'static str> {
        if position >= u32::from(NATIVE_STRATA_MAX_CONTEXT) {
            return Err("native MTP invocation bounds");
        }
        Ok(Self {
            kind: 2,
            position,
            route_epoch: 0,
            route_usage: 0,
            accepted: 0,
        })
    }
    /// Fresh target commit controls. The matching original transaction and
    /// device/fault authority stay with the engine, not this parameter packet.
    pub fn commit(position: u32, accepted: u32) -> Result<Self, &'static str> {
        if position >= u32::from(NATIVE_STRATA_MAX_CONTEXT) || !(1..=8).contains(&accepted) {
            return Err("native commit invocation bounds");
        }
        Ok(Self {
            kind: 3,
            position,
            route_epoch: 0,
            route_usage: 0,
            accepted,
        })
    }
    /// Check that the current target/draft window fits its immutable shape.
    pub fn validate_for(self, key: NativeStrataBranch) -> Result<(), &'static str> {
        let span = match (self.kind, key.family) {
            (1, 1) if self.route_epoch != 0 => u32::from(key.rows),
            (2, 2) => u32::from(key.rows) + u32::from(key.proposals) - 1,
            (2, 3) => 1,
            (3, 4) if (1..=u32::from(key.rows)).contains(&self.accepted) => u32::from(key.rows),
            _ => return Err("native invocation family shape"),
        };
        if self
            .position
            .checked_add(span)
            .is_none_or(|end| end > u32::from(key.context_capacity))
        {
            return Err("native invocation context extent");
        }
        Ok(())
    }
    /// Evaluate a declared INFO source with checked position arithmetic.
    pub fn value(self, source: NativeStrataValue) -> Result<u32, &'static str> {
        match source {
            NativeStrataValue::Position { offset } => {
                let value = self.position as i64 + offset as i64;
                if !(0..i64::from(NATIVE_STRATA_MAX_CONTEXT)).contains(&value) {
                    return Err("native position expression bounds");
                }
                Ok(value as u32)
            }
            NativeStrataValue::Boundary { offset } => {
                let value = self.position as i64 + offset as i64;
                if !(0..=i64::from(NATIVE_STRATA_MAX_CONTEXT)).contains(&value) {
                    return Err("native boundary expression bounds");
                }
                Ok(value as u32)
            }
            NativeStrataValue::AcceptedCount if self.kind == 3 => Ok(self.accepted),
            NativeStrataValue::RouteEpoch if self.kind == 1 => Ok(self.route_epoch),
            NativeStrataValue::RouteUsage if self.kind == 1 => Ok(self.route_usage),
            _ => Err("native control expression requires declared phase or target"),
        }
    }
    /// Clear the active retained control copy at close or retirement.
    pub fn wipe_private(&mut self) {
        // Control copies are short-lived stack values. The retained active copy
        // is cleared explicitly at branch close/retirement, not cached as a key.
        unsafe {
            core::ptr::write_volatile(&mut self.kind, 0);
            core::ptr::write_volatile(&mut self.position, 0);
            core::ptr::write_volatile(&mut self.route_epoch, 0);
            core::ptr::write_volatile(&mut self.route_usage, 0);
            core::ptr::write_volatile(&mut self.accepted, 0);
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
/// An error during profiling.
pub enum ProfileError {
    /// An unknown error happened during profiling
    #[error(
        "An unknown error happened during profiling\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}"
    )]
    Unknown {
        /// The caused of the error
        reason: String,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// No profiling was registered
    #[error("No profiling registered\nBacktrace:\n{backtrace}")]
    NotRegistered {
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Server(#[from] Box<ServerError>),
}

impl core::fmt::Debug for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Contains many different types that are useful for server implementations and compute clients.
pub struct ServerUtilities<Server: ComputeServer> {
    /// The time when `profile-tracy` is activated.
    #[cfg(feature = "profile-tracy")]
    pub epoch_time: web_time::Instant,
    /// The GPU client when `profile-tracy` is activated.
    #[cfg(feature = "profile-tracy")]
    pub gpu_client: tracy_client::GpuContext,
    /// Information shared between all servers.
    pub properties: DeviceProperties,
    /// Stable hash of the device properties
    pub properties_hash: u64,
    /// Information specific to the current server.
    pub info: Server::Info,
    /// The logger based on global cubecl configs.
    pub logger: Arc<ServerLogger>,
    /// How to create the allocation.
    pub layout_policy: Server::MemoryLayoutPolicy,
    /// How to enforce bounds checking on kernels.
    pub check_mode: BoundsCheckMode,
    /// A set containing the ids for which the inter-device communication has already been initialized.
    pub initialized_comms: RwLock<HashSet<CommunicationId>>,
}

/// Defines how the memory layout is determined.
pub trait MemoryLayoutPolicy: Send + Sync + 'static {
    /// Applies the memory layout policy to a list of descriptors.
    ///
    /// Returns a vector of `MemoryLayout`, one per descriptor, with layouts that share a
    /// single `Binding`.
    fn apply(
        &self,
        stream_id: StreamId,
        descriptors: &[MemoryLayoutDescriptor],
    ) -> (Handle, Vec<MemoryLayout>);
}

impl<Server: core::fmt::Debug> core::fmt::Debug for ServerUtilities<Server>
where
    Server: ComputeServer,
    Server::Info: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("ServerUtilities")
            .field("properties", &self.properties)
            .field("info", &self.info)
            .field("logger", &self.logger)
            .finish()
    }
}

impl<S: ComputeServer> ServerUtilities<S> {
    /// Creates a new server utilities.
    pub fn new(
        properties: DeviceProperties,
        logger: Arc<ServerLogger>,
        info: S::Info,
        allocator: S::MemoryLayoutPolicy,
    ) -> Self {
        // Start a tracy client if needed.
        #[cfg(feature = "profile-tracy")]
        let client = tracy_client::Client::start();

        Self {
            properties_hash: properties.checksum(),
            properties,
            logger,
            // Create the GPU client if needed.
            #[cfg(feature = "profile-tracy")]
            gpu_client: client
                .clone()
                .new_gpu_context(
                    Some(&format!("{info:?}")),
                    // In the future should ask the server what makes sense here. 'Invalid' atm is a generic stand-in (Tracy doesn't have CUDA/RocM atm anyway).
                    tracy_client::GpuContextType::Invalid,
                    0,   // Timestamps are manually aligned to this epoch so start at 0.
                    1.0, // Timestamps are manually converted to be nanoseconds so period is 1.
                )
                .unwrap(),
            #[cfg(feature = "profile-tracy")]
            epoch_time: web_time::Instant::now(),
            info,
            layout_policy: allocator,
            check_mode: CubeClRuntimeConfig::get().compilation.check_mode,
            initialized_comms: RwLock::new(HashSet::default()),
        }
    }
}

/// Kernel Launch Errors.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum LaunchError {
    /// The given kernel can't be compiled.
    #[error("A compilation error happened during launch\nCaused by:\n  {0}")]
    CompilationError(#[from] CompilationError),

    /// The server is out of memory.
    #[error(
        "An out-of-memory error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    OutOfMemory {
        /// The caused of the memory error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Too many resources were requested
    #[error("Too many resources were requested during launch\n{0}")]
    TooManyResources(#[from] ResourceLimitError),

    /// Unknown launch error.
    #[error(
        "An unknown error happened during launch\nCaused by:\n  {reason}\nBacktrace\n{backtrace}"
    )]
    Unknown {
        /// The caused of the unknown error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Can't launch because of an IO Error.
    #[error("An io error happened during launch\nCaused by:\n  {0}")]
    IoError(#[from] IoError),
}

/// Resource limit errors.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ResourceLimitError {
    /// Shared memory exceeds maximum
    #[error(
        "Too much shared memory requested.\nRequested {requested} bytes, maximum {max} bytes available.\nBacktrace\n{backtrace}"
    )]
    SharedMemory {
        /// Value requested
        requested: usize,
        /// Maximum value
        max: usize,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
    /// Total units exceeds maximum
    #[error(
        "Total unit count exceeds maximum.\nRequested {requested} units, max units is {max}.\nBacktrace\n{backtrace}"
    )]
    Units {
        /// Requested value
        requested: u32,
        /// Maximum value
        max: u32,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
    /// `CubeDim` exceeds maximum
    #[error(
        "Cube dim exceeds maximum bounds.\nRequested {requested:?}, max is {max:?}.\nBacktrace\n{backtrace}"
    )]
    CubeDim {
        /// Requested value
        requested: (u32, u32, u32),
        /// Maximum value
        max: (u32, u32, u32),
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

impl core::fmt::Debug for LaunchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

impl core::fmt::Debug for ResourceLimitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Error that can happen asynchronously while executing registered kernels.
#[derive(Error, Debug, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ServerError {
    /// A generic runtime error.
    #[error("An error happened during execution\nCaused by:\n  {reason}\nBacktrace:\n{backtrace}")]
    Generic {
        /// The details of the generic error.
        reason: String,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// A launch error happened during profiling
    #[error("A launch error happened during profiling\nCaused by:\n  {0}")]
    Launch(#[from] LaunchError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Profile(#[from] ProfileError),

    /// An execution error happened during profiling
    #[error("An execution error happened during profiling\nCaused by:\n  {0}")]
    Io(#[from] IoError),

    /// The server is an invalid state.
    #[error("The server is in an invalid state\nCaused by:\n  {errors:?}")]
    ServerUnhealthy {
        /// The details of the generic error.
        errors: Vec<Self>,
        /// The backtrace for this error.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },
}

/// How errors are handled in a stream when executing a task.
#[derive(Clone, Copy)]
pub struct StreamErrorMode {
    /// Whether the task still executes even if the stream is in error.
    pub ignore: bool,
    /// Whether the errors are flushed by the current task.
    pub flush: bool,
}

/// Enqueue boundaries within authentic target/draft work; no completion credit.
#[derive(Clone, Copy, Debug)]
pub enum NativeStrataSharedBoundary {
    /// Subsequent nodes use the opposite adopted compute queue.
    Begin,
    /// Return enqueue tagging to MAIN without waiting on the GPU.
    End,
    /// Release the routed segment before a later shared join is enqueued.
    ReleaseMain,
}

/// The compute server is responsible for handling resources and computations over resources.
/// Everything is mutable and accessed through the serialized ComputeClient.
pub trait ComputeServer:
    Send + core::fmt::Debug + ServerCommunication + device::DeviceService + 'static
where
    Self: Sized,
{
    /// The kernel type defines the computation algorithms.
    type Kernel: KernelMetadata;
    /// Information that can be retrieved for the runtime.
    type Info: Debug + Send + Sync;
    /// Manages how allocations are performed for a server.
    type MemoryLayoutPolicy: MemoryLayoutPolicy;
    /// The [storage](ComputeStorage) type defines how data is stored and accessed.
    type Storage: ComputeStorage;

    /// Initializes [memory](ManagedMemoryHandle) on the given [stream](StreamId) with the given size.
    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId);

    /// Reserves N [Bytes] of the provided sizes to be used as staging to load data.
    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Retrieve the server logger.
    fn logger(&self) -> Arc<ServerLogger>;

    /// Retrieve the server utilities.
    fn utilities(&self) -> Arc<ServerUtilities<Self>>;

    /// Given bindings, returns the owned resources as bytes.
    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>>;

    /// Writes the specified bytes into the buffers given
    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId);

    /// Wait for the completion of every task in the server.
    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>>;

    /// Reserve a separate, non-reused PUBLIC host-resource registry.
    fn configure_public_external_host(
        &mut self,
        _budget: crate::public_external::PublicExternalBudget,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Revoke selected original-model sources and join all prior readers.
    fn begin_public_external_rebase(
        &mut self,
        _bindings: Vec<Binding>,
        _stream: StreamId,
    ) -> Result<crate::public_external::PublicExternalRebase, ServerError> {
        Err(ServerError::Generic {
            reason: "public external rebase unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }
    /// Publish a completed authenticated model-only rebase on its owning stream.
    fn finish_public_external_rebase(
        &mut self,
        _token: crate::public_external::PublicExternalRebase,
        _stream: StreamId,
    ) -> Result<(), ServerError> {
        Err(ServerError::Generic {
            reason: "public external rebase unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }
    /// Extend an existing same-stream PUBLIC registry grant without resetting
    /// identities, counters, retirement or immutable resource ownership.
    fn extend_public_external_host(
        &mut self,
        _budget: crate::public_external::PublicExternalBudget,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Register caller-owned, immutable authenticated PUBLIC host backing.
    fn register_public_external_host(
        &mut self,
        _source: crate::public_external::PublicExternalHostSource<
            <Self::Storage as ComputeStorage>::Resource,
        >,
        _stream_id: StreamId,
    ) -> Result<crate::public_external::PublicExternalHandle, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Revoke lookup, join scheduled consumers, destroy safe clones, then join
    /// again. Only success permits the caller to release its backing.
    fn detach_public_external_host(
        &mut self,
        _handle: crate::public_external::PublicExternalHandle,
        _stream_id: StreamId,
    ) -> Result<
        crate::public_external::PublicExternalDetachReceipt<
            <Self::Storage as ComputeStorage>::Resource,
        >,
        ServerError,
    > {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Content-free ownership independent from managed MAIN/runtime pools.
    fn public_external_host_usage(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<crate::public_external::PublicExternalUsage, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Opt-in bounded private I/O for one serialized stream. Unsupported
    /// backends must reject it; never silently use ordinary upload staging.
    fn configure_transient_io(
        &mut self,
        _budget_bytes: u64,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Fixed, separately admitted public target I/O. Unsupported backends
    /// cannot substitute their ordinary transient configuration.
    fn configure_public_target_request_io(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Explicit target-only construction variant. Unsupported backends reject
    /// it; they cannot silently substitute the ordinary regulation policy.
    fn configure_public_target_request_io_with_backlog(
        &mut self,
        _backlog: PublicTargetSubmissionBacklog,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// The actual immutable policy of an already configured target stream.
    fn public_target_submission_backlog(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<PublicTargetSubmissionBacklog, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Content-free fixed-policy validation; unsupported backends reject it.
    fn public_target_request_io_configured(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<bool, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Queue owned private uploads without cloning their host controller.
    fn write_private(
        &mut self,
        _descriptors: Vec<(CopyDescriptor, Bytes)>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Protected native lab cleanup read of one retained full span. Ordinary
    /// inference reads are separate; unsupported backends must reject this API.
    fn read_private_cleanup(
        &mut self,
        _descriptor: CopyDescriptor,
        _stream_id: StreamId,
    ) -> Result<Bytes, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Checked stream-local I/O erasure. Main/public weights are excluded.
    /// Caller must resolve/drop readbacks and serialize all stream enqueues.
    fn sanitize_transient_io(
        &mut self,
        _stream_id: StreamId,
    ) -> DynFut<Result<crate::transient::TransientIoReceipt, ServerError>> {
        Box::pin(async {
            Err(IoError::UnsupportedIoOperation {
                backtrace: BackTrace::capture(),
            }
            .into())
        })
    }

    /// Protected opt-in lab audit: inspect every erased full span and erase
    /// the additional bounded audit staging. Return counts only, never bytes.
    fn sanitize_transient_io_audited(
        &mut self,
        _stream_id: StreamId,
    ) -> DynFut<Result<crate::transient::TransientIoReceipt, ServerError>> {
        Box::pin(async {
            Err(IoError::UnsupportedIoOperation {
                backtrace: BackTrace::capture(),
            }
            .into())
        })
    }

    /// Begin one exclusive request-owned transient I/O reuse scope. The
    /// returned nonzero nonce belongs only to this stream until checked end.
    fn begin_transient_io_reuse(&mut self, _stream_id: StreamId) -> Result<u64, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Dedicated activation for the fixed public target metadata reservation.
    fn begin_public_target_request_io_reuse(
        &mut self,
        _stream_id: StreamId,
    ) -> Result<u64, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Complete outstanding work and retain all dirty I/O owners for reuse by
    /// the SAME request. This grants no erasure receipt or ownership credit.
    fn recycle_transient_io(
        &mut self,
        _nonce: u64,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Accept completion and scoped GPU errors without preparing or publishing
    /// any additional reusable owner. All dirty spans remain fully charged.
    fn accept_transient_io_reuse(
        &mut self,
        _nonce: u64,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// End only after full two-pass erasure and CPU inspection of all retained
    /// owners. Failed cleanup retains the scope and retires the stream.
    fn end_transient_io_reuse(
        &mut self,
        _nonce: u64,
        _stream_id: StreamId,
    ) -> Result<crate::transient::TransientIoReceipt, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Given a resource handle, returns the storage resource.
    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError>;

    /// Original loader metadata on the owning native device. No readback,
    /// allocation, fence credit or ownership transfer; all source handles must
    /// remain retained through their actual native consumers.
    fn native_strata_source_addresses(
        &mut self,
        _bindings: Vec<Binding>,
        _stream_id: StreamId,
    ) -> Result<Vec<[u64; 2]>, ServerError> {
        Err(ServerError::Generic {
            reason: "native original source addresses unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }

    /// Bounded, same-stream FULL retained allocation metadata only. Does not
    /// execute pending tasks, map bytes, synchronize, allocate GPU memory,
    /// clean a pool, or grant completed-fence credit. Unsupported backends
    /// reject this API; ordinary get_resource keeps its existing semantics.
    fn retained_resource_metadata(
        &mut self,
        _bindings: Vec<Binding>,
        _stream_id: StreamId,
    ) -> Result<Vec<ManagedResource<<Self::Storage as ComputeStorage>::Resource>>, ServerError>
    {
        Err(ServerError::Generic {
            reason: "retained resource metadata unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }

    /// Queue immutable registered HOST weights through this sole owning stream.
    fn stage_public_host_copies(
        &mut self,
        _copies: Vec<crate::public_external::PublicHostCopy>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(ServerError::Generic {
            reason: "public host staging copies unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }

    /// Enqueue full retained private-owner clears on the configured stream.
    /// This is ordered work, not completed-erasure credit. The caller retains
    /// original full owners until its normal checked completion fence.
    fn clear_private_owners(
        &mut self,
        _bindings: Vec<Binding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        Err(ServerError::Generic {
            reason: "private owner clearing unsupported".into(),
            backtrace: BackTrace::capture(),
        })
    }

    /// Executes the `kernel` over the given memory `handles`.
    ///
    /// Kernels have mutable access to every resource they are given
    /// and are responsible of determining which should be read or written.
    ///
    /// # Safety
    ///
    /// When executing with mode [`ExecutionMode::Unchecked`], out-of-bound reads and writes can happen.
    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        kind: ExecutionMode,
        stream_id: StreamId,
    );

    /// Flush all outstanding tasks in the server.
    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError>;

    /// Submit selected native Intel expert work without a host completion wait.
    /// The backend must retain every command/buffer and join before any later
    /// stream mutation, readback, reuse or cleanup. Unsupported servers reject.
    fn submit_native_strata_experts(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Open a request-local stable native parameter/recording shape.
    fn begin_native_strata_branch(
        &mut self,
        _stream_id: StreamId,
        _key: NativeStrataBranch,
        _invocation: Option<NativeStrataInvocation>,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }
    /// Change the static math phase after enqueuing all preceding work on this
    /// stream. This changes no GPU data and grants no completion authority.
    fn set_native_strata_phase(
        &mut self,
        _stream_id: StreamId,
        _phase: NativeStrataPhase,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }
    /// Submit a previously sealed typed branch on the current request stream.
    /// False means no supported recording exists, before any replay mutation.
    /// True is returned only after fresh controls and the actual native fence
    /// complete. Neither result authorizes numerical acceptance or erasure.
    fn replay_native_strata_branch(
        &mut self,
        _stream_id: StreamId,
        _key: NativeStrataBranch,
        _invocation: NativeStrataInvocation,
    ) -> Result<bool, ServerError> {
        Ok(false)
    }
    /// Complete and seal that exact shape; this is not an erasure receipt.
    fn end_native_strata_branch(
        &mut self,
        _stream_id: StreamId,
        _key: NativeStrataBranch,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Submit a retained target commit without joining. None is a cache miss;
    /// Some names original queued owners, never numerical/state acceptance.
    fn replay_native_strata_commit_pending(
        &mut self,
        _stream_id: StreamId,
        _key: NativeStrataBranch,
        _invocation: NativeStrataInvocation,
    ) -> Result<Option<NativeStrataSubmission>, ServerError> {
        Ok(None)
    }
    /// Submit/seal the original cold commit, retaining its actual pending fence.
    fn end_native_strata_commit_pending(
        &mut self,
        _stream_id: StreamId,
        _key: NativeStrataBranch,
    ) -> Result<NativeStrataSubmission, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }
    /// Join exactly the original exported submission. This does not accept its
    /// global numerical fault, model state or owning erasure obligations.
    fn join_native_strata_submission(
        &mut self,
        _stream_id: StreamId,
        _submission: NativeStrataSubmission,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Change a target recording queue tag under its authentic request scope.
    fn native_strata_shared_boundary(
        &mut self,
        _stream_id: StreamId,
        _boundary: NativeStrataSharedBoundary,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// Pin the full original verified-R allocation and submit its real
    /// main-to-draft dependency BEFORE any accepted-prefix commit is queued.
    fn prepare_native_strata_draft_scope(
        &mut self,
        _stream_id: StreamId,
        _source: Binding,
    ) -> Result<NativeStrataDraftScope, ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }
    /// Enter the original dependency scope before ALL MTP staging/control work.
    fn enter_native_strata_draft_scope(
        &mut self,
        _stream_id: StreamId,
        _scope: NativeStrataDraftScope,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }
    /// Join original scope consumers before releasing source/returning to MAIN.
    /// This is not target state, fault or owning erasure acceptance.
    fn end_native_strata_draft_scope(
        &mut self,
        _stream_id: StreamId,
        _scope: NativeStrataDraftScope,
    ) -> Result<(), ServerError> {
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    /// The current memory usage of the server.
    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError>;

    /// Ask the server to release memory that it can release.
    fn memory_cleanup(&mut self, stream_id: StreamId);

    /// Enable collecting timestamps.
    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError>;

    /// Disable collecting timestamps.
    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError>;

    /// Update the memory mode of allocation in the server.
    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId);
}

/// An ID unique to any unordered combination of devices.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct CommunicationId {
    /// The ID as a `String`.
    pub id: u64,
}

impl From<Vec<DeviceId>> for CommunicationId {
    fn from(mut value: Vec<DeviceId>) -> Self {
        // Make sure that device ids are sorted so that any combination of the same devices uses the same communicator.
        value.sort();
        let mut hasher = AHasher::default();
        value.hash(&mut hasher);
        CommunicationId {
            id: hasher.finish(),
        }
    }
}

/// Different reduce operations.
pub enum ReduceOperation {
    /// Sum.
    Sum,
    /// Mean.
    Mean,
}

/// Defines functions for optimized data transfer between servers, supporting custom communication
/// mechanisms such as peer-to-peer communication or specialized implementations.
pub trait ServerCommunication {
    /// Indicates whether server-to-server communication is enabled for this implementation.
    const SERVER_COMM_ENABLED: bool;

    /// Ensure that all queued collective operations have been executed.
    ///
    /// # Arguments
    ///
    /// * `stream_id` - The [`StreamId`] of the stream waiting for the sync.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        todo!() // For backends other than cuda.
    }

    /// Initialize the communication between the devices in `device_ids`.
    ///
    /// # Arguments
    ///
    /// * `device_ids` - The IDs of the devices that need communication.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn comm_init(&mut self, device_ids: Vec<DeviceId>) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Performs an `all_reduce` operation on the input data and writes it to the output buffer.
    /// see <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce>
    ///
    /// # Arguments
    ///
    /// * `src` - The data to be reduced.
    /// * `dst` - Where to write the result.
    /// * `dtype` - The element type of the data being reduced
    /// * `stream_id` - The data's stream id.
    /// * `op` - The reduce's aggregation operation e.g. mean, sum, etc.
    /// * `device_ids` - The list of device ids from which to `all_reduce`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        dtype: ElemType,
        stream_id: StreamId,
        op: ReduceOperation,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Sends data from this server to a destination server.
    ///
    /// # Arguments
    ///
    /// * `desc` - A descriptor specifying the data to be sent, including shape, strides, and binding.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_dst` - ID of the device receiving the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn send(
        &mut self,
        desc: CopyDescriptor,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_dst: DeviceId,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }

    /// Receive data from another server.
    ///
    /// # Arguments
    ///
    /// * `handle` - The handle in which the received data is written.
    /// * `dtype` - The element type of the data being sent.
    /// * `stream_id` - The stream ID associated with the server's operation.
    /// * `device_id_src` - ID of the device sending the data.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing an `ServerError` if the operation fails.
    #[allow(unused_variables)]
    fn recv(
        &mut self,
        handle: Handle,
        dtype: ElemType,
        stream_id: StreamId,
        device_id_src: DeviceId,
    ) -> Result<(), ServerError> {
        unimplemented!()
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
/// Profiling identification so that the server can support recursive and overlapping profilings.
pub struct ProfilingToken {
    /// The token value.
    pub id: u64,
}

/// Type of allocation, either contiguous or optimized (row-aligned when possible)
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum MemoryLayoutStrategy {
    /// Contiguous layout, with no padding
    Contiguous,
    /// Optimized for access speed. In practice this means row-aligned with padding for runtimes
    /// that support it.
    Optimized,
}

/// Descriptor for a new tensor allocation
#[derive(new, Debug, Clone)]
pub struct MemoryLayoutDescriptor {
    /// Strategy used to create the memory layout.
    pub strategy: MemoryLayoutStrategy,
    /// Shape of the tensor
    pub shape: Shape,
    /// Size of each element in the tensor (used for conversion of shape to bytes)
    pub elem_size: usize,
}

impl MemoryLayoutDescriptor {
    /// Create an optimized allocation descriptor
    pub fn optimized(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Optimized, shape, elem_size)
    }

    /// Create a contiguous allocation descriptor
    pub fn contiguous(shape: Shape, elem_size: usize) -> Self {
        MemoryLayoutDescriptor::new(MemoryLayoutStrategy::Contiguous, shape, elem_size)
    }
}

/// An allocation with associated strides. Strides depend on tensor layout.
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// The handle for the memory resource
    pub memory: Handle,
    /// TODO: `Strides` should become `Layout`.
    ///
    /// The strides of the tensor
    pub strides: Strides,
}

impl MemoryLayout {
    /// Create a new memory layout.
    pub fn new(handle: Handle, strides: impl Into<Strides>) -> Self {
        MemoryLayout {
            memory: handle,
            strides: strides.into(),
        }
    }
}

/// A reason for an error.
#[derive(Default, Clone)]
pub struct Reason {
    inner: ReasonInner,
}

#[cfg(std_io)]
mod _reason_serde {
    use super::*;

    use alloc::string::ToString;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    impl Serialize for Reason {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: Serializer,
        {
            // Use the Display implementation (via to_string) to flatten the enum
            serializer.serialize_str(&self.to_string())
        }
    }

    impl<'de> Deserialize<'de> for Reason {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: Deserializer<'de>,
        {
            // Deserialize into a standard String first
            let s = String::deserialize(deserializer)?;

            // Wrap it in the Dynamic variant since we can't safely
            // reconstruct a 'static str from a runtime string.
            Ok(Reason {
                inner: ReasonInner::Dynamic(Arc::new(s)),
            })
        }
    }
}

#[derive(Default, Clone)]
enum ReasonInner {
    Static(&'static str),
    Dynamic(Arc<String>),
    #[default]
    NotProvided,
}

impl core::fmt::Display for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match &self.inner {
            ReasonInner::Static(content) => f.write_str(content),
            ReasonInner::Dynamic(content) => f.write_str(content),
            ReasonInner::NotProvided => f.write_str("No reason provided for the error"),
        }
    }
}

impl core::fmt::Debug for Reason {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Display::fmt(&self, f)
    }
}

impl From<&'static str> for Reason {
    fn from(value: &'static str) -> Self {
        Self {
            inner: ReasonInner::Static(value),
        }
    }
}

impl From<String> for Reason {
    fn from(value: String) -> Self {
        Self {
            inner: ReasonInner::Dynamic(Arc::new(value)),
        }
    }
}

/// Error returned from `create`/`read`/`write` functions. Due to async execution not all errors
/// are able to be caught, so some IO errors will still panic.
#[derive(Error, Clone)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum IoError {
    /// Buffer size exceeds the max available
    #[error("can't allocate buffer of size: {size}\n{backtrace}")]
    BufferTooBig {
        /// The size of the buffer in bytes.
        size: u64,
        /// The captured backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Strides aren't supported for this copy operation on this runtime
    #[error("the provided strides are not supported for this operation\n{backtrace}")]
    UnsupportedStrides {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Memory wasn't found in the memory pool
    #[error("couldn't find resource for that handle: {reason}\n{backtrace}")]
    NotFound {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
        /// The reason the handle is invalid.
        reason: Reason,
    },

    /// Handle wasn't found in the memory pool
    #[error("couldn't free the handle, since it is currently in used. \n{backtrace}")]
    FreeError {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Unknown error happened during execution
    #[error("Unknown error happened during execution\n{backtrace}")]
    Unknown {
        /// Details of the error
        description: String,
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// The current IO operation is not supported
    #[error("The current IO operation is not supported\n{backtrace}")]
    UnsupportedIoOperation {
        /// The backtrace.
        #[cfg_attr(std_io, serde(skip))]
        backtrace: BackTrace,
    },

    /// Can't perform the IO operation because of a runtime error.
    #[error("Can't perform the IO operation because of a runtime error: {0}")]
    Execution(#[from] Box<ServerError>),
}

impl core::fmt::Debug for IoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}

/// Arguments to execute a kernel.
#[derive(Debug, Default)]
pub struct KernelArguments {
    /// Buffer bindings
    pub buffers: Vec<Binding>,
    /// Native physical-pointer READ dependencies, retained and ordered without
    /// adding descriptors. Unsupported backends must reject nonempty lists.
    pub read_dependencies: Vec<Binding>,
    /// Packed scalars and metadata. First scalars sorted by type, then static metadata,
    /// then dynamic metadata.
    pub info: MetadataBindingInfo,
    /// Tensor map bindings
    pub tensor_maps: Vec<TensorMapBinding>,
}

impl core::fmt::Display for KernelArguments {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("KernelArguments")?;
        for b in self.buffers.iter() {
            f.write_fmt(format_args!("\n - buffer: {b:?}\n"))?;
        }

        Ok(())
    }
}

impl KernelArguments {
    /// Create a new bindings struct
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a buffer binding
    pub fn with_buffer(mut self, binding: Binding) -> Self {
        self.buffers.push(binding);
        self
    }

    /// Extend the buffers with `bindings`
    pub fn with_buffers(mut self, bindings: Vec<Binding>) -> Self {
        self.buffers.extend(bindings);
        self
    }

    /// Retain real BDA source handles without granting shader write authority.
    pub fn with_read_dependencies(mut self, bindings: Vec<Binding>) -> Self {
        self.read_dependencies.extend(bindings);
        self
    }

    /// Set the info to `info`
    pub fn with_info(mut self, info: MetadataBindingInfo) -> Self {
        self.info = info;
        self
    }

    /// Extend the tensor maps with `bindings`
    pub fn with_tensor_maps(mut self, bindings: Vec<TensorMapBinding>) -> Self {
        self.tensor_maps.extend(bindings);
        self
    }
}

/// Binding of a set of scalars of the same type to execute a kernel.
///
/// The [`ComputeServer`] is responsible to convert those info into actual [`Binding`] when launching
/// kernels.
#[derive(Default)]
pub struct MetadataBindingInfo {
    /// Scalar and metadata values
    pub data: Vec<u64>,
    /// Start of the dynamically sized portion of the metadata, relative to the entire info buffer
    pub dynamic_metadata_offset: usize,
    /// Optional native INFO schema. These are symbolic locations only.
    pub native_parameters: Vec<NativeStrataParameter>,
}

impl core::fmt::Debug for MetadataBindingInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetadataBindingInfo")
            .field("words", &self.data.len())
            .finish_non_exhaustive()
    }
}
impl Drop for MetadataBindingInfo {
    fn drop(&mut self) {
        self.wipe_private();
    }
}
impl MetadataBindingInfo {
    /// Erase every packed metadata capacity word before release/reuse. Drop
    /// also calls this on rejected or never-enqueued kernel arguments.
    pub fn wipe_private(&mut self) {
        let pointer = self.data.as_mut_ptr();
        for i in 0..self.data.capacity() {
            // SAFETY: exclusive Vec allocation; all u64 bit patterns valid,
            // including writable uninitialized spare capacity.
            unsafe { pointer.add(i).write_volatile(0) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

impl MetadataBindingInfo {
    /// Preserve the original constructor used by generic CubeCL metadata.
    pub fn new(data: Vec<u64>, dynamic_metadata_offset: usize) -> Self {
        Self {
            data,
            dynamic_metadata_offset,
            native_parameters: Vec::new(),
        }
    }
    /// Declare one u32 invocation field without storing its value in a key.
    pub fn with_native_parameter(mut self, word: u16, value: NativeStrataValue) -> Self {
        self.native_parameters
            .push(NativeStrataParameter { word, value });
        self
    }
    /// Create a new binding info for custom data, for externally compiled kernels.
    pub fn custom(data: Vec<u64>) -> Self {
        Self::new(data, 0)
    }
}

/// A binding with shape and stride info for non-contiguous reading
#[derive(new, Debug)]
pub struct CopyDescriptor {
    /// Binding for the memory resource
    pub handle: Binding,
    /// Shape of the resource
    pub shape: Shape,
    /// Strides of the resource
    pub strides: Strides,
    /// Size of each element in the resource
    pub elem_size: usize,
}

/// A tensor map used with TMA ops
#[derive(new, Debug)]
pub struct TensorMapBinding {
    /// The binding for the backing tensor
    pub binding: Binding,
    /// The tensormap metadata
    pub map: TensorMapMeta,
}

/// `TensorMap` metadata for the opaque proxy used in TMA copies
#[derive(Debug, Clone)]
pub struct TensorMapMeta {
    /// Tensormap format (tiled or im2col)
    pub format: TensorMapFormat,
    /// Metadata of the backing tensor
    pub metadata: Metadata,
    /// Element stride, usually 1 but may be 2 for complex tensors
    /// For im2col, this is equivalent to the kernel stride
    pub elem_stride: Strides,
    /// Interleave mode
    pub interleave: TensorMapInterleave,
    /// Swizzle mode
    pub swizzle: TensorMapSwizzle,
    /// Prefetch settings
    pub prefetch: TensorMapPrefetch,
    /// OOB fill value
    pub oob_fill: OobFill,
    /// Storage type
    pub storage_ty: StorageType,
}

/// Specifieds the number of cubes to be dispatched for a kernel.
///
/// This translates to eg. a grid for CUDA, or to `num_workgroups` for wgsl.
#[allow(clippy::large_enum_variant)]
pub enum CubeCount {
    /// Dispatch a known count of x, y, z cubes.
    Static(u32, u32, u32),
    /// Dispatch an amount based on the values in this buffer. The buffer should contain a u32 array [x, y, z].
    Dynamic(Binding),
}

/// Defines how to select cube count based on the number of cubes required.
pub enum CubeCountSelection {
    /// If the number of cubes is the same as required.
    Exact(CubeCount),
    /// If the number of cubes isn't the same as required.
    ///
    /// This can happen based on the hardware limit, requiring the kernel to perform OOB checks.
    Approx(CubeCount, u32),
}

impl CubeCountSelection {
    /// Creates a [`CubeCount`] while respecting the hardware limits.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, num_cubes: u32) -> Self {
        let cube_count = cube_count_spread(&client.properties().hardware.max_cube_count, num_cubes);

        let num_cubes_actual = cube_count[0] * cube_count[1] * cube_count[2];
        let cube_count = CubeCount::Static(cube_count[0], cube_count[1], cube_count[2]);

        match num_cubes_actual == num_cubes {
            true => CubeCountSelection::Exact(cube_count),
            false => CubeCountSelection::Approx(cube_count, num_cubes_actual),
        }
    }

    /// If some cubes will be idle.
    pub fn has_idle(&self) -> bool {
        matches!(self, Self::Approx(..))
    }

    /// Converts into [`CubeCount`].
    pub fn cube_count(self) -> CubeCount {
        match self {
            CubeCountSelection::Exact(cube_count) => cube_count,
            CubeCountSelection::Approx(cube_count, _) => cube_count,
        }
    }
}

impl From<CubeCountSelection> for CubeCount {
    fn from(value: CubeCountSelection) -> Self {
        value.cube_count()
    }
}

impl CubeCount {
    /// Create a new static cube count with the given x = y = z = 1.
    pub fn new_single() -> Self {
        CubeCount::Static(1, 1, 1)
    }

    /// Create a new static cube count with the given x, and y = z = 1.
    pub fn new_1d(x: u32) -> Self {
        CubeCount::Static(x, 1, 1)
    }

    /// Create a new static cube count with the given x and y, and z = 1.
    pub fn new_2d(x: u32, y: u32) -> Self {
        CubeCount::Static(x, y, 1)
    }

    /// Create a new static cube count with the given x, y and z.
    pub fn new_3d(x: u32, y: u32, z: u32) -> Self {
        CubeCount::Static(x, y, z)
    }

    /// Checks whether the cube count is definitely empty, i.e. has 0 dispatches.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Static(x, y, z) => *x == 0 || *y == 0 || *z == 0,
            Self::Dynamic(_) => false,
        }
    }
}

impl Debug for CubeCount {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CubeCount::Static(x, y, z) => f.write_fmt(format_args!("({x}, {y}, {z})")),
            CubeCount::Dynamic(_) => f.write_str("binding"),
        }
    }
}

impl Clone for CubeCount {
    fn clone(&self) -> Self {
        match self {
            Self::Static(x, y, z) => Self::Static(*x, *y, *z),
            Self::Dynamic(binding) => Self::Dynamic(binding.clone()),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
#[allow(missing_docs)]
/// The number of units across all 3 axis totalling to the number of working units in a cube.
pub struct CubeDim {
    /// The number of units in the x axis.
    pub x: u32,
    /// The number of units in the y axis.
    pub y: u32,
    /// The number of units in the z axis.
    pub z: u32,
}

impl CubeDim {
    /// Creates a new [`CubeDim`] based on the maximum number of tasks that can be parellalized by units, in other words,
    /// by the maximum number of working units.
    ///
    /// # Notes
    ///
    /// For complex problems, you probably want to have your own logic function to create the
    /// [`CubeDim`], but for simpler problems such as elemwise-operation, this is a great default.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, working_units: usize) -> Self {
        let properties = client.properties();
        let plane_size = properties.hardware.plane_size_max;
        let plane_count = Self::calculate_plane_count_per_cube(
            working_units as u32,
            plane_size,
            properties.hardware.num_cpu_cores,
        );

        // Make sure it respects the max units per cube (especially on wasm)
        let limit = properties.hardware.max_units_per_cube / plane_size;

        // Ensure at least 1 plane so CubeDim is always valid (num_elems() > 0).
        Self::new_2d(plane_size, u32::min(limit, plane_count).max(1))
    }

    fn calculate_plane_count_per_cube(
        working_units: u32,
        plane_dim: u32,
        num_cpu_cores: Option<u32>,
    ) -> u32 {
        match num_cpu_cores {
            Some(num_cores) => core::cmp::min(num_cores, working_units),
            None => {
                let plane_count_max = core::cmp::max(1, working_units / plane_dim);

                // Ensures `plane_count` is a power of 2.
                const NUM_PLANE_MAX: u32 = 8u32;
                const NUM_PLANE_MAX_LOG2: u32 = NUM_PLANE_MAX.ilog2();
                let plane_count_max_log2 =
                    core::cmp::min(NUM_PLANE_MAX_LOG2, u32::ilog2(plane_count_max));
                2u32.pow(plane_count_max_log2)
            }
        }
    }

    /// Create a new cube dim with x = y = z = 1.
    pub const fn new_single() -> Self {
        Self { x: 1, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x, and y = z = 1.
    pub const fn new_1d(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }

    /// Create a new cube dim with the given x and y, and z = 1.
    pub const fn new_2d(x: u32, y: u32) -> Self {
        Self { x, y, z: 1 }
    }

    /// Create a new cube dim with the given x, y and z.
    /// This is equivalent to the [new](CubeDim::new) function.
    pub const fn new_3d(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// Total numbers of units per cube
    pub const fn num_elems(&self) -> u32 {
        self.x * self.y * self.z
    }

    /// Whether this `CubeDim` can fully contain `other`
    pub const fn can_contain(&self, other: CubeDim) -> bool {
        self.x >= other.x && self.y >= other.y && self.z >= other.z
    }
}

impl From<(u32, u32, u32)> for CubeDim {
    fn from(value: (u32, u32, u32)) -> Self {
        CubeDim::new_3d(value.0, value.1, value.2)
    }
}

impl From<CubeDim> for (u32, u32, u32) {
    fn from(val: CubeDim) -> Self {
        (val.x, val.y, val.z)
    }
}

/// The kind of execution to be performed.
#[derive(Default, Hash, PartialEq, Eq, Clone, Debug, Copy)]
#[cfg_attr(std_io, derive(serde::Serialize, serde::Deserialize))]
pub enum ExecutionMode {
    /// Checked kernels are safe.
    #[default]
    Checked,
    /// Validate OOB and alert if OOB access occurs
    Validate,
    /// Unchecked kernels are unsafe.
    Unchecked,
}

fn cube_count_spread(max: &(u32, u32, u32), num_cubes: u32) -> [u32; 3] {
    let max_cube_counts = [max.0, max.1, max.2];
    let mut num_cubes = [num_cubes, 1, 1];
    let base = 2;

    let mut reduce_count = |i: usize| {
        if num_cubes[i] <= max_cube_counts[i] {
            return true;
        }

        loop {
            num_cubes[i] = num_cubes[i].div_ceil(base);
            num_cubes[i + 1] *= base;

            if num_cubes[i] <= max_cube_counts[i] {
                return false;
            }
        }
    };

    for i in 0..2 {
        if reduce_count(i) {
            break;
        }
    }

    num_cubes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test_log::test]
    fn safe_num_cubes_even() {
        let max = (32, 32, 32);
        let required = 2048;

        let actual = cube_count_spread(&max, required);
        let expected = [32, 32, 2];
        assert_eq!(actual, expected);
    }

    #[test_log::test]
    fn safe_num_cubes_odd() {
        let max = (48, 32, 16);
        let required = 3177;

        let actual = cube_count_spread(&max, required);
        let expected = [25, 32, 4];
        assert_eq!(actual, expected);
    }
}
