//! Bounded, transient collection of the ORIGINAL native expert commands.
//! No device handles, completion authority, model data or reusable request cache.
use super::{CubeCount, ExecutionMode, KernelArguments, ServerError};
use crate::public_external::PublicHostCopy;
use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;

// At most512 original groups, each with up to8 physical compute pieces (six
// logical phases, split GU/DOWN) and3 checked HOST copies. The actual8MiB tape
// inventory and24MiB parent reservation remain independent hard bounds.
pub const NATIVE_EXPERT_BATCH_COMMANDS: usize = 512 * (8 + 3);
pub const NATIVE_EXPERT_BATCH_BYTES: usize = 8 * 1024 * 1024;
/// Complete one-active-batch host reservation, including the original command
/// tape/arguments and simultaneously live native backend owner/INFO tables.
/// The application must reserve this before collecting the first payload.
pub const NATIVE_EXPERT_BATCH_HOST_BYTES: usize = 3 * NATIVE_EXPERT_BATCH_BYTES;

/// The closed original six-operator family admitted by the native backend.
/// HOST copies do not advance this compute-only index. Keep the actual final
/// streamed dequant release boundary; neither a missing release nor another
/// kernel may stand in for that producer. The separately named resident GU and
/// DOWN producers have identical math but no raw-slot release, like Strata's
/// original `slot >= 0` condition. Their names also bind the actual role.
/// Closed direct8 identities retain the original exact role/release contract.
/// No suffix, prefix, or namespace alias is admitted.
pub(crate) fn native_strata_expert_dequant_identity(name:&str)->&str {
    #[cfg(feature="kuno-dequant-direct8")]
    match name {
        "StrataPromptExpertF16Direct8Image"=>return "StrataPromptExpertF16Image",
        "StrataPromptExpertResidentGuDirect8Image"=>return "StrataPromptExpertResidentGuImage",
        "StrataPromptExpertResidentDownDirect8Image"=>return "StrataPromptExpertResidentDownImage",
        _=>{}
    }
    name
}

/// Genuine new small-matrix entries. Keep this closed before the legacy
/// substring classifier: disabled or nested new namespaces grant no authority.
pub(crate) fn native_strata_generated_type_suffix(s: &str) -> bool {
    if s.is_empty() { return true; }
    if !s.starts_with('<') || !s.ends_with('>') { return false; }
    let mut depth = 0usize;
    for (i, b) in s.bytes().enumerate() {
        if b == b'<' { depth += 1; }
        if b == b'>' {
            if depth == 0 { return false; }
            depth -= 1;
            if depth == 0 && i + 1 != s.len() { return false; }
        }
    }
    depth == 0
}
pub(crate) fn native_strata_small_direct_generated(name: &str, module: &str, ty: &str) -> bool {
    let allowed = cfg!(feature = "kuno-small-direct-publish") && matches!((module, ty),
        ("expert_macro", "ExpertMacro") | ("expert_macro64", "ExpertMacro64")
        | ("expert_narrow", "ExpertNarrow") | ("expert_tail_gu", "ExpertTailGu")
        | ("expert_tail_down", "ExpertTailDown"));
    allowed && name.split_once("::strata_matrix::native::small_direct::")
        .is_some_and(|(_, suffix)| suffix.strip_prefix(module)
            .and_then(|s| s.strip_prefix("::"))
            .and_then(|s| s.strip_prefix(ty))
            .is_some_and(native_strata_generated_type_suffix))
}
pub(crate) fn native_strata_small_direct_matrix(name: &str) -> bool {
    [("expert_macro", "ExpertMacro"), ("expert_macro64", "ExpertMacro64"),
        ("expert_narrow", "ExpertNarrow")].into_iter()
        .any(|(module, ty)| native_strata_small_direct_generated(name, module, ty))
}

pub fn native_strata_expert_operator(name: &str, release: bool, index: usize) -> bool {
    let name=native_strata_expert_dequant_identity(name);
    if name.contains("::small_direct::") {
        return !release && matches!(index % 6, 3 | 5)
            && (index % 6 != 5 || cfg!(feature = "kuno-expert-ordered-output"))
            && native_strata_small_direct_matrix(name);
    }
    // Ordered PART producers are genuine separate DOWN entries. Do not let
    // the original loose legacy-family check accidentally admit them for GU,
    // or accept arbitrary nesting under the new namespace.
    if name.contains("::ordered::") {
        #[cfg(feature = "kuno-expert-ordered-output")]
        if index % 6 == 5 && !release {
            if let Some(suffix) = name.split_once("::native::ordered::").map(|(_, s)| s) {
                #[cfg(feature = "kuno-deep-matrix")]
                if suffix.strip_prefix("expert_macro64_deep::ExpertMacro64Deep")
                    .is_some_and(|s| s.is_empty() || s.starts_with('<')) { return true; }
                return [
                    "expert::Expert", "expert_macro::ExpertMacro",
                    "expert_macro64::ExpertMacro64", "expert_narrow::ExpertNarrow",
                ].iter().any(|entry| suffix.strip_prefix(entry)
                    .is_some_and(|s| s.is_empty() || s.starts_with('<')));
            }
        }
        return false;
    }
    match index % 6 {
        0 => name == "StrataPromptExpertGather" && !release,
        1 => {
            matches!(
                name,
                "StrataPromptExpertF16Image" | "StrataPromptExpertResidentGuImage"
            ) && !release
        }
        2 => {
            (name == "StrataPromptExpertF16Image" && release)
                || (name == "StrataPromptExpertResidentDownImage" && !release)
        }
        4 => {
            matches!(
                name,
                "StrataPromptExpertF16Swiglu" | "StrataPromptExpertF16SwigluCheckGu"
            ) && !release
        }
        _ => {
            (name.contains("::expert::Expert")
                || name.contains("::expert_macro::ExpertMacro")
                || name.split_once("::expert_macro64::ExpertMacro64")
                    .is_some_and(|(_,suffix)| suffix.is_empty() || suffix.starts_with('<'))
                || name.contains("::expert_narrow::ExpertNarrow"))
                && !release
        }
    }
}

/// Actual backend enqueue counts, never a GPU completion/acceptance receipt.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeStrataExpertBatchReceipt {
    /// Actual completed-unit tasks given terminal release metadata by the
    /// original grammar, returned only after full successful batch enqueue.
    pub unit_releases: usize,
    pub cohort_groups:usize,
    pub cohort_launches:usize,
    pub cohort_batches:usize,
    pub launches: usize,
    pub copy_segments: usize,
    pub logical_phases: usize,
    pub matrix_pieces: usize,
    pub wide_prefixes: usize,
    pub wide_blocks: usize,
    pub tail_pieces: usize,
    pub direct_gu_prefixes: usize,
    pub gu_checks: usize,
    pub unused_descriptor_views: usize,
}

pub enum NativeStrataExpertCommand<K> {
    Launch {
        kernel: Option<K>,
        count: CubeCount,
        args: KernelArguments,
        mode: ExecutionMode,
        /// Scheduling intent only. The backend must validate the ORIGINAL
        /// grammar through this exact launch before deriving release metadata.
        #[cfg(feature = "kuno-expert-cohorts")]
        unit_end: bool,
    },
    HostCopies(NativeStrataExpertCopies),
}

pub struct NativeStrataExpertCopies(pub Vec<PublicHostCopy>);
impl Drop for NativeStrataExpertCopies {
    fn drop(&mut self) {
        self.0.clear();
        let bytes = self.0.capacity() * core::mem::size_of::<PublicHostCopy>();
        for i in 0..bytes {
            unsafe { (self.0.as_mut_ptr() as *mut u8).add(i).write_volatile(0) }
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

pub struct NativeStrataExpertBatch<K> {
    pub commands: Vec<NativeStrataExpertCommand<K>>,
    payload_bytes: usize,
    nodes: usize,
}

pub(crate) fn batch_fault(reason: &'static str) -> ServerError {
    ServerError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    }
}

impl<K> NativeStrataExpertBatch<K> {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            payload_bytes: 0,
            nodes: 0,
        }
    }
    pub fn metadata_bytes(&self) -> Result<usize, ServerError> {
        if self.commands.len() > NATIVE_EXPERT_BATCH_COMMANDS
            || self.commands.capacity() > NATIVE_EXPERT_BATCH_COMMANDS
        {
            return Err(batch_fault("native expert batch actual tape capacity"));
        }
        let mut nodes = 0usize;
        let mut bytes =
            self.commands.capacity() * core::mem::size_of::<NativeStrataExpertCommand<K>>();
        for command in &self.commands {
            let (count, payload) = Self::measure(command)?;
            nodes = nodes
                .checked_add(count)
                .ok_or_else(|| batch_fault("native expert batch command overflow"))?;
            bytes = bytes
                .checked_add(payload)
                .ok_or_else(|| batch_fault("native expert batch metadata overflow"))?;
        }
        if nodes > NATIVE_EXPERT_BATCH_COMMANDS || bytes > NATIVE_EXPERT_BATCH_BYTES {
            return Err(batch_fault("native expert batch actual inventory capacity"));
        }
        Ok(bytes)
    }
    fn measure(command: &NativeStrataExpertCommand<K>) -> Result<(usize, usize), ServerError> {
        let nodes = match command {
            NativeStrataExpertCommand::Launch {
                args,
                count,
                kernel,
                ..
            } => {
                if args.buffers.is_empty()
                    || args.buffers.len() > 32
                    || args.read_dependencies.len() > 4096
                    || kernel.is_none()
                    || !args.tensor_maps.is_empty()
                    || !matches!(count, CubeCount::Static(x,y,z) if *x>0 && *y>0 && *z>0)
                {
                    return Err(batch_fault("native expert batch original launch shape"));
                }
                1
            }
            NativeStrataExpertCommand::HostCopies(copies) => {
                if copies.0.is_empty() || copies.0.len() > 3 {
                    return Err(batch_fault("native expert batch original copies"));
                }
                copies.0.len()
            }
        };
        let bytes = match command {
            NativeStrataExpertCommand::Launch { args, .. } => args
                .buffers
                .capacity()
                .checked_add(args.read_dependencies.capacity())
                .and_then(|n| n.checked_mul(core::mem::size_of::<super::Binding>()))
                .and_then(|n| {
                    args.info
                        .data
                        .capacity()
                        .checked_mul(8)
                        .and_then(|b| n.checked_add(b))
                })
                .and_then(|n| {
                    args.info
                        .native_parameters
                        .capacity()
                        .checked_mul(core::mem::size_of::<super::NativeStrataParameter>())
                        .and_then(|b| n.checked_add(b))
                }),
            NativeStrataExpertCommand::HostCopies(copies) => copies
                .0
                .capacity()
                .checked_mul(core::mem::size_of::<PublicHostCopy>()),
        }
        .ok_or_else(|| batch_fault("native expert batch metadata overflow"))?;
        Ok((nodes, bytes))
    }
    pub fn push(&mut self, command: NativeStrataExpertCommand<K>) -> Result<(), ServerError> {
        let (nodes, bytes) = Self::measure(&command)?;
        if self
            .nodes
            .checked_add(nodes)
            .is_none_or(|n| n > NATIVE_EXPERT_BATCH_COMMANDS)
        {
            return Err(batch_fault("native expert batch command capacity"));
        }
        let payload = self
            .payload_bytes
            .checked_add(bytes)
            .ok_or_else(|| batch_fault("native expert batch metadata overflow"))?;
        if self.commands.capacity() == 0 {
            // Reserve the complete bounded tape BEFORE its first populated
            // insertion. No realloc may free an old private command image.
            let planned = NATIVE_EXPERT_BATCH_COMMANDS
                .checked_mul(core::mem::size_of::<NativeStrataExpertCommand<K>>())
                .and_then(|n| n.checked_add(payload))
                .ok_or_else(|| batch_fault("native expert batch metadata overflow"))?;
            if planned > NATIVE_EXPERT_BATCH_BYTES {
                return Err(batch_fault("native expert batch metadata capacity"));
            }
            self.commands
                .try_reserve_exact(NATIVE_EXPERT_BATCH_COMMANDS)
                .map_err(|_| batch_fault("native expert batch bounded allocation"))?;
        }
        let total = self
            .commands
            .capacity()
            .checked_mul(core::mem::size_of::<NativeStrataExpertCommand<K>>())
            .and_then(|n| n.checked_add(payload))
            .ok_or_else(|| batch_fault("native expert batch metadata overflow"))?;
        if total > NATIVE_EXPERT_BATCH_BYTES {
            return Err(batch_fault("native expert batch metadata capacity"));
        }
        self.payload_bytes = payload;
        self.nodes += nodes;
        self.commands.push(command);
        Ok(())
    }
}

impl<K> Default for NativeStrataExpertBatch<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K> Drop for NativeStrataExpertBatch<K> {
    fn drop(&mut self) {
        // Drop original binding and INFO owners first. INFO's own Drop wipes its
        // complete allocation. Then erase unused/old command storage as well.
        self.commands.clear();
        let bytes = self.commands.capacity() * core::mem::size_of::<NativeStrataExpertCommand<K>>();
        for i in 0..bytes {
            unsafe {
                (self.commands.as_mut_ptr() as *mut u8)
                    .add(i)
                    .write_volatile(0)
            };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(feature = "std")]
pub(crate) struct Capture<K> {
    pub stream: cubecl_common::stream_id::StreamId,
    pub batch: Option<NativeStrataExpertBatch<K>>,
    pub error: Option<ServerError>,
}
#[cfg(feature = "std")]
impl<K> Capture<K> {
    pub fn require_open(
        &self,
        stream: cubecl_common::stream_id::StreamId,
    ) -> Result<(), ServerError> {
        if stream != self.stream || self.error.is_some() || self.batch.is_none() {
            return Err(batch_fault("native expert batch sink sealed"));
        }
        Ok(())
    }
    pub fn push(
        &mut self,
        command: NativeStrataExpertCommand<K>,
        stream: cubecl_common::stream_id::StreamId,
    ) -> Result<(), ServerError> {
        if stream != self.stream {
            self.error = Some(batch_fault("native expert batch foreign original stream"));
            return Err(batch_fault("native expert batch foreign original stream"));
        }
        if self.error.is_some() || self.batch.is_none() {
            return Err(batch_fault("native expert batch sink sealed"));
        }
        if let Err(error) = self.batch.as_mut().unwrap().push(command) {
            self.error = Some(error);
            return Err(batch_fault("native expert batch collection failed"));
        }
        Ok(())
    }

    /// Mark only the actual last launch of this still-open ordered collector.
    /// This grants no phase/execution authority: the backend independently
    /// validates its exact original kernel, arguments and completed grammar.
    #[cfg(feature = "kuno-expert-cohorts")]
    pub fn mark_unit_end(
        &mut self,
        stream: cubecl_common::stream_id::StreamId,
    ) -> Result<(), ServerError> {
        let result = (|| {
            self.require_open(stream)?;
            match self.batch.as_mut().unwrap().commands.last_mut() {
                Some(NativeStrataExpertCommand::Launch { unit_end, kernel, .. })
                    if !*unit_end && kernel.is_some() => {
                        *unit_end = true;
                        Ok(())
                    }
                _ => Err(batch_fault("native expert unit end requires a new last launch")),
            }
        })();
        if let Err(error) = result {
            self.error = Some(error);
            return Err(batch_fault("native expert unit end collection failed"));
        }
        Ok(())
    }
}

/// A scheduling-only wrapper around the actual original compilation task.
/// Its only constructor advances the ORIGINAL grammar using that same task's
/// metadata and actual packed arguments, and requires a complete unit there.
/// No generated kernel/definition, arithmetic, binding or owner is replaced.
#[cfg(feature = "kuno-expert-cohorts")]
pub struct NativeStrataExpertUnitEnd<C: crate::compiler::Compiler> {
    original: alloc::boxed::Box<dyn crate::compiler::CubeTask<C>>,
}
#[cfg(feature = "kuno-expert-cohorts")]
impl<C: crate::compiler::Compiler> NativeStrataExpertUnitEnd<C> {
    /// Validate this original task at the current grammar position and require
    /// its completed six-phase unit before deriving the recording-only cut.
    pub fn validate(
        original: alloc::boxed::Box<dyn crate::compiler::CubeTask<C>>,
        grammar: &mut super::native_expert_grammar::NativeStrataExpertGrammar,
        count: &CubeCount,
        args: &KernelArguments,
    ) -> Result<Self, ServerError> {
        grammar.launch(original.name(), original.native_release_after(), count, args)?;
        grammar.finish()?;
        // The existing grammar only closes after the actual final DOWN matrix.
        // In particular its original streamed dequant release is never tagged.
        if original.native_release_after() {
            return Err(batch_fault("native expert unit end already releases"));
        }
        Ok(Self { original })
    }
}
#[cfg(feature = "kuno-expert-cohorts")]
impl<C: crate::compiler::Compiler> crate::kernel::KernelMetadata for NativeStrataExpertUnitEnd<C> {
    fn name(&self) -> &'static str { self.original.name() }
    fn id(&self) -> crate::id::KernelId { self.original.id() }
    fn address_type(&self) -> cubecl_ir::StorageType { self.original.address_type() }
    fn native_release_after(&self) -> bool { true }
}
#[cfg(feature = "kuno-expert-cohorts")]
impl<C: crate::compiler::Compiler> crate::compiler::CubeTask<C> for NativeStrataExpertUnitEnd<C> {
    fn compile(
        &self,
        compiler: &mut C,
        options: &C::CompilationOptions,
        mode: ExecutionMode,
        address_type: cubecl_ir::StorageType,
    ) -> Result<crate::kernel::CompiledKernel<C>, crate::compiler::CompilationError> {
        self.original.compile(compiler, options, mode, address_type)
    }
}

#[cfg(feature = "std")]
pub(crate) struct CaptureScope<K> {
    pub capture: alloc::sync::Arc<std::sync::Mutex<Capture<K>>>,
}
#[cfg(feature = "std")]
impl<K> CaptureScope<K> {
    pub fn new(stream: cubecl_common::stream_id::StreamId) -> Self {
        Self {
            capture: alloc::sync::Arc::new(std::sync::Mutex::new(Capture {
                stream,
                batch: Some(NativeStrataExpertBatch::new()),
                error: None,
            })),
        }
    }
    pub fn take(&self) -> Result<NativeStrataExpertBatch<K>, ServerError> {
        let mut capture = self.capture.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(error) = capture.error.take() {
            return Err(error);
        }
        capture
            .batch
            .take()
            .ok_or_else(|| batch_fault("native expert batch sink sealed"))
    }
}
#[cfg(feature = "std")]
impl<K> Drop for CaptureScope<K> {
    fn drop(&mut self) {
        self.capture
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .batch
            .take();
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::server::{Handle, MetadataBindingInfo};
    use cubecl_common::stream_id::StreamId;
    fn launch(handle: &Handle) -> NativeStrataExpertCommand<()> {
        NativeStrataExpertCommand::Launch {
            kernel: Some(()),
            count: CubeCount::Static(1, 1, 1),
            args: KernelArguments::new()
                .with_buffer(handle.clone().binding())
                .with_info(MetadataBindingInfo::custom(vec![0x1234, 0x5678])),
            mode: ExecutionMode::Unchecked,
            #[cfg(feature = "kuno-expert-cohorts")]
            unit_end: false,
        }
    }
    #[test]
    fn direct_selection_requires_real_open_scope_and_closes_on_take_error_drop_and_panic() {
        let stream = StreamId::current();
        let scope = CaptureScope::<()>::new(stream);
        let escaped = scope.capture.clone();
        assert!(escaped.lock().unwrap().require_open(stream).is_ok());
        assert!(
            escaped
                .lock()
                .unwrap()
                .require_open(StreamId {
                    value: stream.value.wrapping_add(1)
                })
                .is_err()
        );
        // A foreign query cannot itself mint authority or invalidate the owner.
        assert!(escaped.lock().unwrap().require_open(stream).is_ok());
        let batch = scope.take().unwrap();
        assert!(escaped.lock().unwrap().require_open(stream).is_err());
        drop(batch);
        drop(scope);
        assert!(escaped.lock().unwrap().require_open(stream).is_err());
        let scope = CaptureScope::<()>::new(stream);
        let escaped = scope.capture.clone();
        escaped.lock().unwrap().error = Some(batch_fault("fixed public failure"));
        assert!(escaped.lock().unwrap().require_open(stream).is_err());
        assert!(scope.take().is_err());
        drop(scope);
        assert!(escaped.lock().unwrap().require_open(stream).is_err());
        let scope = CaptureScope::<()>::new(stream);
        let escaped = scope.capture.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = scope;
            panic!("fixed public panic");
        }));
        assert!(result.is_err());
        assert!(escaped.lock().unwrap().require_open(stream).is_err());
    }
    #[test]
    fn escaped_sink_seals_and_releases_original_bindings_after_panic() {
        let stream = StreamId::current();
        let handle = Handle::new(stream, 64);
        let scope = CaptureScope::new(stream);
        let escaped = scope.capture.clone();
        escaped
            .lock()
            .unwrap()
            .push(launch(&handle), stream)
            .unwrap();
        assert!(!handle.memory.is_free());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope = scope;
            panic!("compiled public capture failure");
        }));
        assert!(result.is_err());
        assert!(handle.memory.is_free());
        assert!(escaped.lock().unwrap().batch.is_none());
        assert!(
            escaped
                .lock()
                .unwrap()
                .push(launch(&handle), stream)
                .is_err()
        );
        assert!(handle.memory.is_free());
    }
    #[test]
    fn original_stream_and_single_take_are_required_without_populated_growth() {
        let stream = StreamId::current();
        let handle = Handle::new(stream, 64);
        let scope = CaptureScope::new(stream);
        scope
            .capture
            .lock()
            .unwrap()
            .push(launch(&handle), stream)
            .unwrap();
        let mut batch = scope.take().unwrap();
        let pointer = batch.commands.as_ptr();
        for _ in 1..NATIVE_EXPERT_BATCH_COMMANDS {
            batch.push(launch(&handle)).unwrap();
        }
        assert_eq!(pointer, batch.commands.as_ptr());
        assert!(batch.push(launch(&handle)).is_err());
        assert!(scope.take().is_err());
        drop(batch);
        assert!(handle.memory.is_free());
        let foreign = CaptureScope::new(stream);
        assert!(
            foreign
                .capture
                .lock()
                .unwrap()
                .push(
                    launch(&handle),
                    StreamId {
                        value: stream.value.wrapping_add(1)
                    }
                )
                .is_err()
        );
        assert!(foreign.take().is_err());
        assert!(handle.memory.is_free());
    }
    #[test]
    fn backend_remeasures_publicly_mutated_command_inventory_before_admission() {
        let stream = StreamId::current();
        let handle = Handle::new(stream, 64);
        let mut batch = NativeStrataExpertBatch::new();
        // The backend must validate real fields/capacities, never trust the
        // collector's private cached counters when a safe caller bypasses push.
        let mut bad = launch(&handle);
        if let NativeStrataExpertCommand::Launch { args, .. } = &mut bad {
            args.buffers.clear();
        }
        batch.commands.push(bad);
        assert!(batch.metadata_bytes().is_err());
        batch.commands.clear();
        batch.commands.push(launch(&handle));
        if let NativeStrataExpertCommand::Launch { args, .. } = &mut batch.commands[0] {
            args.info
                .data
                .try_reserve_exact(NATIVE_EXPERT_BATCH_BYTES / 8 + 1)
                .unwrap();
        }
        assert!(batch.metadata_bytes().is_err());
    }

    #[cfg(feature = "kuno-expert-cohorts")]
    #[test]
    fn unit_end_marks_only_the_actual_last_launch_of_a_current_open_collector() {
        let stream = StreamId::current();
        let handle = Handle::new(stream, 64);
        let scope = CaptureScope::<()>::new(stream);
        assert!(scope.capture.lock().unwrap().mark_unit_end(stream).is_err());
        assert!(scope.take().is_err());
        drop(scope);

        let scope = CaptureScope::new(stream);
        scope.capture.lock().unwrap().push(launch(&handle), stream).unwrap();
        scope.capture.lock().unwrap().mark_unit_end(stream).unwrap();
        let batch = scope.take().unwrap();
        assert!(matches!(&batch.commands[0], NativeStrataExpertCommand::Launch { unit_end: true, .. }));
        assert!(scope.capture.lock().unwrap().mark_unit_end(stream).is_err());
        drop(batch);

        let scope = CaptureScope::new(stream);
        scope.capture.lock().unwrap().push(launch(&handle), stream).unwrap();
        scope.capture.lock().unwrap().mark_unit_end(stream).unwrap();
        assert!(scope.capture.lock().unwrap().mark_unit_end(stream).is_err());
        assert!(scope.take().is_err());
        drop(scope);

        let scope = CaptureScope::new(stream);
        scope.capture.lock().unwrap().push(launch(&handle), stream).unwrap();
        assert!(scope.capture.lock().unwrap().mark_unit_end(StreamId { value:stream.value.wrapping_add(1) }).is_err());
        assert!(scope.take().is_err());
        drop(scope);

        let scope = CaptureScope::new(stream);
        scope.capture.lock().unwrap().push(launch(&handle), stream).unwrap();
        scope.capture.lock().unwrap().push(NativeStrataExpertCommand::HostCopies(NativeStrataExpertCopies(vec![PublicHostCopy {
            source:handle.clone().binding(), source_first:0,
            destination:handle.clone().binding(), destination_first:0, bytes:16,
        }])),stream).unwrap();
        assert!(scope.capture.lock().unwrap().mark_unit_end(stream).is_err());
        assert!(scope.take().is_err());
        drop(scope);

        let scope = CaptureScope::new(stream);
        let escaped = scope.capture.clone();
        escaped.lock().unwrap().push(launch(&handle), stream).unwrap();
        escaped.lock().unwrap().mark_unit_end(stream).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _scope=scope;
            panic!("fixed public unit end panic");
        }));
        assert!(result.is_err());
        assert!(escaped.lock().unwrap().mark_unit_end(stream).is_err());
        // Every failed/successful/panic collector retains then drops original
        // owners normally; the tape Drop also wipes the new terminal bit.
        assert!(handle.memory.is_free());
    }
}
