//! Explicit, bounded packed Rust CPU experts for a source-only decode experiment.
//!
//! By default all seven original formats retain F32 activations and the GPU's
//! 64-lane dot schedule. Explicit pooled Q8 activation arithmetic is a separate
//! approximate experiment. Returns unweighted original-rank partials; routing, probability
//! multiplication, shared experts and whole-model acceptance remain the caller's.
//! No scalar executor/default, GPU integration or persistent request cache changes.
//! Owning host buffers wipe on error/unwind; scoped workers finish before their
//! borrowed request owners release. This does not claim register/OS-stack erasure.
use crate::gguf::PositionalReader;
use crate::{Error, Result, TensorType, TensorView};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};

pub const LANES: usize = 64;
pub const MAX_SELECTED: usize = 10;
pub const MAX_WORKERS: usize = 16;
pub const MAX_PUBLIC_PACKED_BYTES: usize = 40 * 1024 * 1024;
pub const MAX_PRIVATE_HOST_BYTES: usize = 1024 * 1024;
const MAX_COMPONENT_BYTES: usize = 8 * 1024 * 1024;
const MAX_HIDDEN: usize = 2560;
const MAX_FEED_FORWARD: usize = 1024;
const MAX_EXPERTS: usize = 512;
const WORKER_FIXED_PRIVATE_BYTES: usize = (256 + LANES) * size_of::<f32>() + 4096;
const MAX_WORKER_PRIVATE_BYTES: usize = 32 * 1024;
#[path = "cpu_quantized.rs"]
mod quantized;

/// ISA selection is explicit and never a response to inference/GPU failure.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Portable,
    Avx2,
    /// Explicit original-weight FMA/32-lane candidate. Floating reduction
    /// order differs; never selected automatically or used as a fallback.
    Avx2Fma,
}

/// Public hardware-only worker configuration; never holds request data.
/// Physical plans select one allowed logical CPU per actual physical core.
/// Portable plans are an explicit unpinned reference option, not a fallback.
pub struct WorkerPlan {
    count: usize,
    cores: [usize; MAX_WORKERS],
    pinned: bool,
}

impl WorkerPlan {
    pub fn portable(count: usize) -> Result<Self> {
        validate_workers(count)?;
        Ok(Self {
            count,
            cores: [0; MAX_WORKERS],
            pinned: false,
        })
    }

    pub fn physical(count: usize) -> Result<Self> {
        validate_workers(count)?;
        let cores = affinity::physical_cores(count)?;
        Ok(Self {
            count,
            cores,
            pinned: true,
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

fn validate_workers(count: usize) -> Result<()> {
    if (1..=MAX_WORKERS).contains(&count) {
        Ok(())
    } else {
        Err(Error::LimitExceeded("CPU expert worker count"))
    }
}

/// Immutable model-bound original triple. Public weight bytes are read only
/// for the selected experts at call time and are never converted/cached as a
/// complete model. Tensor identity is pinned by the positional reader.
pub struct PackedExperts<'r, 'm> {
    reader: &'r PositionalReader<'m>,
    tensors: [TensorView<'m>; 3],
    geometry: Geometry,
}

struct Geometry {
    hidden: usize,
    feed_forward: usize,
    experts: usize,
    kinds: [TensorType; 3],
    row_bytes: [usize; 3],
    packed_bytes: [usize; 3],
}

/// One original packed matrix/expert, tied to the actual inspected reader.
/// Intended for bounded expert/operator calibration, not a complete model head.
pub struct PackedProjection<'r, 'm> {
    reader: &'r PositionalReader<'m>,
    tensor: TensorView<'m>,
    columns: usize,
    rows: usize,
    experts: usize,
    row_bytes: usize,
    packed_bytes: usize,
}

impl<'r, 'm> PackedProjection<'r, 'm> {
    pub fn new(reader: &'r PositionalReader<'m>, tensor: TensorView<'m>) -> Result<Self> {
        if !reader.owns(tensor) {
            return Err(Error::InvalidFormat("CPU projection artifact ownership"));
        }
        let shape = tensor.info().dimensions();
        if !(2..=3).contains(&shape.len()) {
            return Err(Error::InvalidFormat("CPU projection original geometry"));
        }
        let columns = usize::try_from(shape[0]).map_err(|_| Error::ArithmeticOverflow)?;
        let rows = usize::try_from(shape[1]).map_err(|_| Error::ArithmeticOverflow)?;
        let experts = usize::try_from(shape.get(2).copied().unwrap_or(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        if columns == 0
            || columns > MAX_HIDDEN
            || rows == 0
            || rows > 4096
            || experts == 0
            || experts > MAX_EXPERTS
        {
            return Err(Error::LimitExceeded("CPU projection geometry"));
        }
        let row_bytes = usize::try_from(tensor.info().dtype().row_bytes(columns as u64)?)
            .map_err(|_| Error::ArithmeticOverflow)?;
        let packed_bytes = row_bytes
            .checked_mul(rows)
            .ok_or(Error::ArithmeticOverflow)?;
        if packed_bytes > MAX_COMPONENT_BYTES
            || tensor.info().row_bytes() != row_bytes as u64
            || tensor.info().byte_len()
                != (packed_bytes as u64)
                    .checked_mul(experts as u64)
                    .ok_or(Error::ArithmeticOverflow)?
        {
            return Err(Error::LimitExceeded("CPU projection packed capacity"));
        }
        reader.verify_file_unchanged()?;
        Ok(Self {
            reader,
            tensor,
            columns,
            rows,
            experts,
            row_bytes,
            packed_bytes,
        })
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }
}

/// Call-owned guarded output with a read-only logical view.
pub struct ProjectionOutput {
    rows: usize,
    values: Zeroizing<Vec<f32>>,
}
impl ProjectionOutput {
    pub fn values(&self) -> &[f32] {
        &self.values[..self.rows]
    }
    pub fn owned_bytes(&self) -> usize {
        self.values.capacity() * size_of::<f32>()
    }
}

impl<'r, 'm> PackedExperts<'r, 'm> {
    pub fn new(
        reader: &'r PositionalReader<'m>,
        gate: TensorView<'m>,
        up: TensorView<'m>,
        down: TensorView<'m>,
    ) -> Result<Self> {
        let tensors = [gate, up, down];
        if tensors.iter().any(|&tensor| !reader.owns(tensor)) {
            return Err(Error::InvalidFormat("CPU expert artifact ownership"));
        }
        let dims = tensors.map(|tensor| tensor.info().dimensions());
        if dims.iter().any(|shape| shape.len() != 3)
            || dims[1] != dims[0]
            || dims[2] != [dims[0][1], dims[0][0], dims[0][2]]
        {
            return Err(Error::InvalidFormat("CPU expert original geometry"));
        }
        let hidden = usize::try_from(dims[0][0]).map_err(|_| Error::ArithmeticOverflow)?;
        let feed_forward = usize::try_from(dims[0][1]).map_err(|_| Error::ArithmeticOverflow)?;
        let experts = usize::try_from(dims[0][2]).map_err(|_| Error::ArithmeticOverflow)?;
        if hidden == 0
            || hidden > MAX_HIDDEN
            || feed_forward == 0
            || feed_forward > MAX_FEED_FORWARD
            || experts == 0
            || experts > MAX_EXPERTS
        {
            return Err(Error::LimitExceeded("CPU expert geometry"));
        }
        let kinds = tensors.map(|tensor| tensor.info().dtype());
        let columns = [hidden, hidden, feed_forward];
        let rows = [feed_forward, feed_forward, hidden];
        let mut row_bytes = [0usize; 3];
        let mut packed_bytes = [0usize; 3];
        for part in 0..3 {
            row_bytes[part] = usize::try_from(kinds[part].row_bytes(columns[part] as u64)?)
                .map_err(|_| Error::ArithmeticOverflow)?;
            packed_bytes[part] = row_bytes[part]
                .checked_mul(rows[part])
                .ok_or(Error::ArithmeticOverflow)?;
            if packed_bytes[part] == 0
                || packed_bytes[part] > MAX_COMPONENT_BYTES
                || tensors[part].info().row_bytes() != row_bytes[part] as u64
                || tensors[part].info().byte_len()
                    != (packed_bytes[part] as u64)
                        .checked_mul(experts as u64)
                        .ok_or(Error::ArithmeticOverflow)?
            {
                return Err(Error::LimitExceeded("CPU expert packed component"));
            }
        }
        reader.verify_file_unchanged()?;
        Ok(Self {
            reader,
            tensors,
            geometry: Geometry {
                hidden,
                feed_forward,
                experts,
                kinds,
                row_bytes,
                packed_bytes,
            },
        })
    }

    pub fn hidden(&self) -> usize {
        self.geometry.hidden
    }
    pub fn feed_forward(&self) -> usize {
        self.geometry.feed_forward
    }
}

/// Fixed ten-rank logical capacity. Only requested ranks are active; the full
/// inactive suffix remains zero. No Clone, Debug, serialization or mutable view.
pub struct Partials {
    hidden: usize,
    count: usize,
    values: Zeroizing<Vec<f32>>,
}

impl Partials {
    pub fn ranks(&self) -> usize {
        self.count
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    pub fn active(&self) -> &[f32] {
        &self.values[..self.count * self.hidden]
    }
    pub fn full(&self) -> &[f32] {
        &self.values[..MAX_SELECTED * self.hidden]
    }
    pub fn owned_bytes(&self) -> usize {
        self.values.capacity() * size_of::<f32>()
    }
}
impl Drop for Partials {
    fn drop(&mut self) {
        self.hidden.zeroize();
        self.count.zeroize();
        // The nested Zeroizing erases initialized and spare allocation capacity.
    }
}

/// Stores only explicit backend and public worker topology. Every private
/// selection, input, activation, partial and worker scratch is call-local.
pub struct ExpertOperator {
    workers: WorkerPlan,
    backend: Backend,
    pool: Option<rayon::ThreadPool>,
    q8_activations: bool,
    q8_residual: bool,
    mapped_prefetch: bool,
    prefetch_calls: std::sync::atomic::AtomicU64,
}

// The full-capacity retained job owner erases these request-derived headers on
// every exit. Packed payloads themselves are immutable public model bytes.
enum PackedJob<'m> {
    Owned(Vec<u8>),
    Mapped(&'m [u8]),
}
impl AsRef<[u8]> for PackedJob<'_> {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Mapped(v) => v,
        }
    }
}

impl ExpertOperator {
    pub fn new(workers: WorkerPlan, backend: Backend) -> Result<Self> {
        if backend == Backend::Avx2 && !avx2_available() {
            return Err(Error::InvalidFormat("CPU expert AVX2 unavailable"));
        }
        if backend == Backend::Avx2Fma && !fma_available() {
            return Err(Error::InvalidFormat("CPU expert AVX2/F16C/FMA unavailable"));
        }
        Ok(Self {
            workers,
            backend,
            pool: None,
            q8_activations: false,
            q8_residual: false,
            mapped_prefetch: false,
            prefetch_calls: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Explicit reusable workers. Only public topology lives between calls.
    /// Broadcast joins all consumers; scratch and guarded row headers are
    /// call-owned. Opaque runtime jobs/stacks have the same exclusion as std
    /// thread internals; no forensic erasure claim is made for either backend.
    pub fn new_pooled(workers: WorkerPlan, backend: Backend) -> Result<Self> {
        let mut operator = Self::new(workers, backend)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(operator.workers.count)
            .build()
            .map_err(|_| Error::LimitExceeded("CPU expert pool spawn"))?;
        let setup = pool.broadcast(|context| {
            if operator.workers.pinned {
                affinity::pin(operator.workers.cores[context.index()])
            } else {
                Ok(())
            }
        });
        for result in setup {
            result?;
        }
        operator.pool = Some(pool);
        Ok(operator)
    }

    fn rows(
        &self,
        output: &mut [f32],
        width: usize,
        cancel: &AtomicBool,
        hook: &(impl Fn(usize) + Sync),
        run: impl Fn(usize, &mut Scratch) -> Result<f32> + Sync,
    ) -> Result<()> {
        match &self.pool {
            Some(pool) => pooled_rows(pool, output, width, cancel, hook, run),
            None => parallel_rows(&self.workers, output, width, cancel, hook, run),
        }
    }

    pub fn workers_pooled(&self) -> bool {
        self.pool.is_some()
    }

    /// Explicit approximate activation arithmetic; original weight bytes stay
    /// unchanged. No automatic selection/failure fallback, no original64 claim.
    pub fn with_q8_activations(mut self) -> Result<Self> {
        if self.backend != Backend::Avx2 || self.pool.is_none() {
            return Err(Error::InvalidFormat("CPU quantized requires pooled AVX2"));
        }
        self.q8_activations = true;
        Ok(self)
    }
    /// Explicit residual Q8 activations (three passes, or compiled two-pass lab), still approximate and
    /// never an automatic fallback. Original packed model bytes are unchanged.
    pub fn with_q8_residual(self) -> Result<Self> {
        let mut this = self.with_q8_activations()?;
        this.q8_residual = true;
        Ok(this)
    }
    /// Explicit actual-selection-only source read-ahead. Requires mapped
    /// immutable weights at every call; failures are not swallowed/fallbacks.
    pub fn with_mapped_prefetch(mut self) -> Self {
        self.mapped_prefetch = true;
        self
    }
    pub fn mapped_prefetch(&self) -> bool {
        self.mapped_prefetch
    }
    pub fn prefetch_calls(&self) -> u64 {
        self.prefetch_calls.load(Ordering::Relaxed)
    }
    pub fn q8_residual_passes(&self) -> usize {
        if self.q8_residual {
            quantized::residual_passes()
        } else {
            0
        }
    }
    pub fn q8_residual(&self) -> bool {
        self.q8_residual
    }
    pub fn q8_activations(&self) -> bool {
        self.q8_activations
    }

    pub fn worker_count(&self) -> usize {
        self.workers.count
    }
    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn project(
        &self,
        matrix: &PackedProjection<'_, '_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<ProjectionOutput> {
        if self.q8_activations {
            return Err(Error::InvalidFormat(
                "CPU quantized requires complete experts",
            ));
        }
        if expert >= matrix.experts || input.len() != matrix.columns {
            return Err(Error::InvalidBufferLength);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidFormat("CPU projection non-finite input"));
        }
        check_cancel(cancel)?;
        let mut private_input = zeroed(matrix.columns)?;
        let mut values = zeroed(matrix.rows)?;
        let private_bytes = checked_sum(&[
            bytes(&private_input)?,
            bytes(&values)?,
            MAX_WORKER_PRIVATE_BYTES * self.workers.count,
            4096,
            worker_handle_metadata_bytes()?,
        ])?;
        if private_bytes > MAX_PRIVATE_HOST_BYTES {
            return Err(Error::LimitExceeded("CPU projection private host capacity"));
        }
        private_input[..matrix.columns].copy_from_slice(input);
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(matrix.packed_bytes)
            .map_err(|_| Error::LimitExceeded("CPU projection packed allocation"))?;
        if packed.capacity() > MAX_COMPONENT_BYTES {
            return Err(Error::LimitExceeded("CPU projection packed owned capacity"));
        }
        packed.resize(matrix.packed_bytes, 0);
        let mut offset = Zeroizing::new(
            expert
                .checked_mul(matrix.packed_bytes)
                .ok_or(Error::ArithmeticOverflow)?,
        );
        matrix
            .reader
            .read_range(matrix.tensor, *offset as u64, &mut packed)?;
        offset.zeroize();
        self.rows(
            &mut values[..matrix.rows],
            matrix.columns,
            cancel,
            &|_| {},
            |index, scratch| {
                dot_row(
                    self.backend,
                    matrix.tensor.info().dtype(),
                    row(&packed, index, matrix.row_bytes)?,
                    &private_input[..matrix.columns],
                    scratch,
                    cancel,
                )
            },
        )?;
        check_cancel(cancel)?;
        matrix.reader.verify_file_unchanged()?;
        if values[matrix.rows..]
            .iter()
            .any(|value| value.to_bits() != 0)
        {
            return Err(Error::InvalidFormat("CPU projection inactive tail"));
        }
        Ok(ProjectionOutput {
            rows: matrix.rows,
            values,
        })
    }

    /// Every selected rank executes, including ranks with zero probability
    /// in the caller. The caller must preserve original IDs/probabilities and
    /// perform one accepted rank-order reduction after all execution owners.
    pub fn execute(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Partials> {
        self.execute_with(weights, selected, input, cancel, |_| {})
    }

    /// A caller with live guarded request owners may only subtract from the
    /// existing one-MiB private envelope. The actual CPU capacities are checked
    /// against this remainder before any request value is copied or read.
    /// This grants no scheduling, publication, fallback or timing authority.
    pub fn execute_with_private_limit(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
        private_limit: usize,
    ) -> Result<Partials> {
        if private_limit == 0 || private_limit > MAX_PRIVATE_HOST_BYTES {
            return Err(Error::LimitExceeded("CPU expert shared private remainder"));
        }
        self.execute_with_limit(weights, selected, input, cancel, private_limit, |_| {})
    }

    /// Explicit pooled CPU/GPU laboratory seam. The caller owns parent-work
    /// capacity and GPU authority. Every CPU consumer finishes outside admission
    /// locks even if parent work errors or panics; neither side can publish here.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_private_limit_alongside<R>(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
        private_limit: usize,
        parent: impl FnOnce() -> Result<R>,
    ) -> Result<(Partials, R)> {
        self.execute_alongside_with(
            weights,
            selected,
            input,
            cancel,
            private_limit,
            parent,
            |_| {},
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_alongside_with<R>(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
        private_limit: usize,
        parent: impl FnOnce() -> Result<R>,
        hook: impl Fn(usize) + Sync,
    ) -> Result<(Partials, R)> {
        const ORCHESTRATOR_PRIVATE: usize = 32 * 1024;
        if self.pool.is_none() || private_limit > MAX_PRIVATE_HOST_BYTES {
            return Err(Error::LimitExceeded("CPU expert concurrent public policy"));
        }
        let remainder = private_limit
            .checked_sub(ORCHESTRATOR_PRIVATE)
            .filter(|&n| n > 0)
            .ok_or(Error::LimitExceeded("CPU expert concurrent host remainder"))?;
        check_cancel(cancel)?;
        let hook = &hook;
        std::thread::scope(|scope| {
            let mut handles = retained_metadata::Owner::new(1)?;
            if handles.metadata_bytes()? > ORCHESTRATOR_PRIVATE {
                return Err(Error::LimitExceeded(
                    "CPU expert concurrent header capacity",
                ));
            }
            let task = std::thread::Builder::new()
                .spawn_scoped(scope, || {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        self.execute_with_limit(weights, selected, input, cancel, remainder, hook)
                    }))
                    .unwrap_or(Err(Error::InvalidFormat("CPU expert orchestrator panic")))
                })
                .map_err(|_| Error::LimitExceeded("CPU expert orchestrator spawn"))?;
            handles.push(task)?;
            // Parent remains on the original GPU/stream thread. Its returned
            // static error and CPU completion are both checked only after join.
            let parent_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(parent))
                .unwrap_or(Err(Error::InvalidFormat("CPU expert parent work panic")));
            let handle = handles
                .take_first()
                .ok_or(Error::InvalidFormat("CPU expert missing orchestrator"))?;
            let cpu_result = handle
                .join()
                .unwrap_or(Err(Error::InvalidFormat("CPU expert orchestrator panic")));
            drop(handles);
            let partials = cpu_result?;
            let parent_output = parent_result?;
            check_cancel(cancel)?;
            Ok((partials, parent_output))
        })
    }

    fn execute_with(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
        hook: impl Fn(usize) + Sync,
    ) -> Result<Partials> {
        self.execute_with_limit(
            weights,
            selected,
            input,
            cancel,
            MAX_PRIVATE_HOST_BYTES,
            hook,
        )
    }

    fn execute_with_limit(
        &self,
        weights: &PackedExperts<'_, '_>,
        selected: &[usize],
        input: &[f32],
        cancel: &AtomicBool,
        private_limit: usize,
        hook: impl Fn(usize) + Sync,
    ) -> Result<Partials> {
        let g = &weights.geometry;
        if self.q8_activations && g.kinds.iter().any(|&kind| !quantized::supported(kind)) {
            return Err(Error::InvalidFormat("CPU quantized format unsupported"));
        }
        if selected.is_empty() || selected.len() > MAX_SELECTED || input.len() != g.hidden {
            return Err(Error::InvalidBufferLength);
        }
        let mut ids = Zeroizing::new([0usize; MAX_SELECTED]);
        for (rank, &expert) in selected.iter().enumerate() {
            if expert >= g.experts || ids[..rank].contains(&expert) {
                return Err(Error::InvalidFormat("CPU expert original selection"));
            }
            ids[rank] = expert;
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidFormat("CPU expert non-finite input"));
        }
        check_cancel(cancel)?;
        let per_expert = checked_sum(&g.packed_bytes)?;
        let public_request = per_expert
            .checked_mul(selected.len())
            .ok_or(Error::ArithmeticOverflow)?;
        if public_request > MAX_PUBLIC_PACKED_BYTES {
            return Err(Error::LimitExceeded("CPU expert public packed capacity"));
        }
        let mut jobs = retained_metadata::Owner::<[PackedJob<'_>; 3]>::new(MAX_SELECTED)?;
        let mut private_input = zeroed(g.hidden)?;
        let mut activations = zeroed(MAX_SELECTED * g.feed_forward)?;
        let mut values = zeroed(MAX_SELECTED * g.hidden)?;
        let mut quantized_inputs = if self.q8_activations {
            Some(retained_metadata::Owner::<quantized::Input>::new(
                MAX_SELECTED + 1,
            )?)
        } else {
            None
        };
        let row_width = g.hidden.max(g.feed_forward);
        let worker_bytes = MAX_WORKER_PRIVATE_BYTES
            .checked_mul(self.workers.count)
            .ok_or(Error::ArithmeticOverflow)?;
        let private_bytes = checked_sum(&[
            bytes(&private_input)?,
            bytes(&activations)?,
            bytes(&values)?,
            worker_bytes,
            size_of::<[usize; MAX_SELECTED]>() + 4096,
            jobs.metadata_bytes()?,
            worker_handle_metadata_bytes()?,
            quantized_inputs
                .as_ref()
                .map_or(Ok(0), |owner| owner.metadata_bytes())?,
        ])?;
        if private_bytes > private_limit {
            return Err(Error::LimitExceeded("CPU expert private host capacity"));
        }
        // All actual allocation capacities have been admitted before private fill.
        private_input[..g.hidden].copy_from_slice(input);
        if let Some(owner) = &mut quantized_inputs {
            owner.push(quantized::Input::empty())?;
            if self.q8_residual {
                owner
                    .get_mut(0)?
                    .fill_refined(&private_input[..g.hidden], cancel)?;
            } else {
                owner.get_mut(0)?.fill(&private_input[..g.hidden], cancel)?;
            }
        }
        let mut public_owned = Zeroizing::new(0usize);
        for rank in 0..selected.len() {
            check_cancel(cancel)?;
            // Store empty public Vec headers directly in their guarded owner;
            // subsequent allocation/read failure cannot strand local job metadata.
            jobs.push(std::array::from_fn(|_| PackedJob::Owned(Vec::new())))?;
            for part in 0..3 {
                let offset = Zeroizing::new(
                    ids[rank]
                        .checked_mul(g.packed_bytes[part])
                        .ok_or(Error::ArithmeticOverflow)?,
                );
                if let Some(mapped) = weights.reader.mapped_range(
                    weights.tensors[part],
                    *offset as u64,
                    g.packed_bytes[part],
                )? {
                    if self.mapped_prefetch {
                        weights.reader.prefetch_range(
                            weights.tensors[part],
                            *offset as u64,
                            g.packed_bytes[part],
                        )?;
                        self.prefetch_calls
                            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                                n.checked_add(1)
                            })
                            .map_err(|_| Error::ArithmeticOverflow)?;
                    }
                    jobs.get_mut(rank)?[part] = PackedJob::Mapped(mapped);
                    continue;
                }
                if self.mapped_prefetch {
                    return Err(Error::InvalidFormat(
                        "CPU source prefetch requires mapped weights",
                    ));
                }
                let PackedJob::Owned(packed) = &mut jobs.get_mut(rank)?[part] else {
                    return Err(Error::InvalidFormat("CPU expert packed job state"));
                };
                packed
                    .try_reserve_exact(g.packed_bytes[part])
                    .map_err(|_| Error::LimitExceeded("CPU expert public packed allocation"))?;
                *public_owned = public_owned
                    .checked_add(packed.capacity())
                    .ok_or(Error::ArithmeticOverflow)?;
                if *public_owned > MAX_PUBLIC_PACKED_BYTES {
                    return Err(Error::LimitExceeded("CPU expert public owned capacity"));
                }
                packed.resize(g.packed_bytes[part], 0);
                weights
                    .reader
                    .read_range(weights.tensors[part], *offset as u64, packed)?;
            }
        }
        weights.reader.verify_file_unchanged()?;
        self.rows(
            &mut activations[..selected.len() * g.feed_forward],
            row_width,
            cancel,
            &hook,
            |index, scratch| {
                let rank = index / g.feed_forward;
                let feature = index % g.feed_forward;
                let gate = if let Some(owner) = &quantized_inputs {
                    quantized::dot(
                        g.kinds[0],
                        row(jobs.get(rank)?[0].as_ref(), feature, g.row_bytes[0])?,
                        owner.get(0)?,
                        cancel,
                    )?
                } else {
                    dot_row(
                        self.backend,
                        g.kinds[0],
                        row(jobs.get(rank)?[0].as_ref(), feature, g.row_bytes[0])?,
                        &private_input[..g.hidden],
                        scratch,
                        cancel,
                    )?
                };
                let up = if let Some(owner) = &quantized_inputs {
                    quantized::dot(
                        g.kinds[1],
                        row(jobs.get(rank)?[1].as_ref(), feature, g.row_bytes[1])?,
                        owner.get(0)?,
                        cancel,
                    )?
                } else {
                    dot_row(
                        self.backend,
                        g.kinds[1],
                        row(jobs.get(rank)?[1].as_ref(), feature, g.row_bytes[1])?,
                        &private_input[..g.hidden],
                        scratch,
                        cancel,
                    )?
                };
                finite(
                    crate::ops::silu(gate) * up,
                    "CPU expert non-finite activation",
                )
            },
        )?;
        if let Some(owner) = &mut quantized_inputs {
            for rank in 0..selected.len() {
                owner.push(quantized::Input::empty())?;
                let values = &activations[rank * g.feed_forward..(rank + 1) * g.feed_forward];
                if self.q8_residual {
                    owner.get_mut(rank + 1)?.fill_refined(values, cancel)?;
                } else {
                    owner.get_mut(rank + 1)?.fill(values, cancel)?;
                }
            }
        }
        self.rows(
            &mut values[..selected.len() * g.hidden],
            row_width,
            cancel,
            &hook,
            |index, scratch| {
                let rank = index / g.hidden;
                let feature = index % g.hidden;
                if let Some(owner) = &quantized_inputs {
                    quantized::dot(
                        g.kinds[2],
                        row(jobs.get(rank)?[2].as_ref(), feature, g.row_bytes[2])?,
                        owner.get(rank + 1)?,
                        cancel,
                    )
                } else {
                    dot_row(
                        self.backend,
                        g.kinds[2],
                        row(jobs.get(rank)?[2].as_ref(), feature, g.row_bytes[2])?,
                        &activations[rank * g.feed_forward..(rank + 1) * g.feed_forward],
                        scratch,
                        cancel,
                    )
                }
            },
        )?;
        check_cancel(cancel)?;
        weights.reader.verify_file_unchanged()?;
        if values[selected.len() * g.hidden..]
            .iter()
            .any(|value| value.to_bits() != 0)
            || activations[selected.len() * g.feed_forward..]
                .iter()
                .any(|value| value.to_bits() != 0)
        {
            return Err(Error::InvalidFormat("CPU expert inactive tail"));
        }
        Ok(Partials {
            hidden: g.hidden,
            count: selected.len(),
            values,
        })
    }
}

fn worker_handle_metadata_bytes() -> Result<usize> {
    let scoped =
        retained_metadata::Owner::<std::thread::ScopedJoinHandle<'static, Result<()>>>::bytes_for(
            MAX_WORKERS,
        )?;
    let pooled =
        retained_metadata::Owner::<std::sync::Mutex<RowJob<'static>>>::bytes_for(MAX_WORKERS)?;
    Ok(scoped.max(pooled))
}

// Only retained application-owned headers are handled here. Public payloads
// and std thread internals keep their ordinary destructors; compiler moves,
// registers and opaque OS/runtime storage remain outside the wipe claim.
#[allow(unsafe_code)]
mod retained_metadata {
    use super::*;
    use std::mem::MaybeUninit;

    /// Sealed, fixed-capacity live-item registry. MaybeUninit owns BOTH the
    /// slot representations and outer Vec header after their normal Drop;
    /// erased all-zero bytes never masquerade as a live Vec or JoinHandle.
    pub(super) struct Owner<T> {
        storage: MaybeUninit<Vec<MaybeUninit<T>>>,
        first: Zeroizing<usize>,
        end: Zeroizing<usize>,
        #[cfg(test)]
        witness: Option<std::sync::Arc<Witness>>,
    }

    impl<T> Owner<T> {
        pub(super) fn new(capacity: usize) -> Result<Self> {
            if capacity == 0 || capacity > MAX_WORKERS {
                return Err(Error::LimitExceeded("CPU retained header count"));
            }
            Self::bytes_for(capacity)?;
            let mut storage = Vec::<MaybeUninit<T>>::new();
            storage
                .try_reserve_exact(capacity)
                .map_err(|_| Error::LimitExceeded("CPU retained header allocation"))?;
            if storage.capacity() != capacity {
                return Err(Error::LimitExceeded("CPU retained header owned capacity"));
            }
            let bytes = capacity
                .checked_mul(size_of::<T>())
                .ok_or(Error::ArithmeticOverflow)?;
            // SAFETY: reserved allocation contains exactly capacity aligned T
            // slots. MaybeUninit<T> permits any bits; zero all spare bytes before
            // private headers enter it, then make all MaybeUninit slots live.
            unsafe {
                std::ptr::write_bytes(storage.as_mut_ptr().cast::<u8>(), 0, bytes);
                storage.set_len(capacity);
            }
            Ok(Self {
                storage: MaybeUninit::new(storage),
                first: Zeroizing::new(0),
                end: Zeroizing::new(0),
                #[cfg(test)]
                witness: None,
            })
        }

        pub(super) fn bytes_for(capacity: usize) -> Result<usize> {
            let bytes = capacity
                .checked_mul(size_of::<T>())
                .and_then(|value| value.checked_add(size_of::<Self>()))
                .ok_or(Error::ArithmeticOverflow)?;
            if bytes > MAX_PRIVATE_HOST_BYTES {
                return Err(Error::LimitExceeded("CPU retained header capacity"));
            }
            Ok(bytes)
        }

        fn slots(&self) -> &Vec<MaybeUninit<T>> {
            // SAFETY: constructor initializes outer Vec; only Drop ends it.
            unsafe { self.storage.assume_init_ref() }
        }
        fn slots_mut(&mut self) -> &mut Vec<MaybeUninit<T>> {
            // SAFETY: as above, and exclusive owner access.
            unsafe { self.storage.assume_init_mut() }
        }
        pub(super) fn metadata_bytes(&self) -> Result<usize> {
            Self::bytes_for(self.slots().capacity())
        }
        pub(super) fn push(&mut self, item: T) -> Result<()> {
            if *self.end >= self.slots().len() {
                return Err(Error::LimitExceeded("CPU retained live header count"));
            }
            let index = *self.end;
            self.slots_mut()[index].write(item);
            *self.end += 1;
            Ok(())
        }
        pub(super) fn get(&self, index: usize) -> Result<&T> {
            if index < *self.first || index >= *self.end {
                return Err(Error::InvalidBufferLength);
            }
            // SAFETY: each first..end slot was initialized exactly once and
            // hasn't been consumed. No mutation while borrowed.
            Ok(unsafe { self.slots()[index].assume_init_ref() })
        }
        pub(super) fn get_mut(&mut self, index: usize) -> Result<&mut T> {
            if index < *self.first || index >= *self.end {
                return Err(Error::InvalidBufferLength);
            }
            // SAFETY: same initialized range proof, plus exclusive access.
            Ok(unsafe { self.slots_mut()[index].assume_init_mut() })
        }
        pub(super) fn take_first(&mut self) -> Option<T> {
            if *self.first >= *self.end {
                return None;
            }
            let index = *self.first;
            *self.first += 1;
            // SAFETY: take removes exactly one live item; the old header bits
            // remain only MaybeUninit bytes and are erased with full capacity.
            Some(unsafe { self.slots()[index].assume_init_read() })
        }
        #[cfg(test)]
        pub(super) fn observe(mut self, witness: std::sync::Arc<Witness>) -> Self {
            self.witness = Some(witness);
            self
        }
    }

    unsafe fn wipe_bytes(pointer: *mut u8, bytes: usize) {
        for index in 0..bytes {
            // SAFETY: caller proves a complete live, uniquely owned allocation
            // of at least bytes. No T is live in that span at this stage.
            unsafe { std::ptr::write_volatile(pointer.add(index), 0) };
        }
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
    }

    impl<T> Drop for Owner<T> {
        fn drop(&mut self) {
            // Current sealed consumers have non-panicking standard destructors:
            // [Vec<u8>;3] and ScopedJoinHandle<Result<()>>. Normal payload drops
            // happen before any header erasure; consumed handles are already
            // joined by parallel_rows, preserving original first-error order.
            let first = Zeroizing::new(*self.first);
            let end = Zeroizing::new(*self.end);
            let slots = self.slots_mut();
            for slot in &mut slots[*first..*end] {
                // SAFETY: exactly the remaining initialized live range. Never
                // drop consumed slots, uninitialized spare capacity or twice.
                unsafe { slot.assume_init_drop() };
            }
            let bytes = slots.capacity() * size_of::<T>();
            let pointer = slots.as_mut_ptr().cast::<u8>();
            // SAFETY: all values are now dead; the whole slot allocation is
            // owned, writable, and contains only MaybeUninit representations.
            unsafe { wipe_bytes(pointer, bytes) };
            #[cfg(test)]
            if let Some(witness) = &self.witness {
                // SAFETY: bytes have just been initialized by volatile zeros;
                // inspection finishes while the allocation is still owned.
                let zeros = unsafe { std::slice::from_raw_parts(pointer, bytes) };
                witness
                    .slot_zero
                    .store(zeros.iter().all(|&b| b == 0), Ordering::SeqCst);
                witness.slot_bytes.store(bytes, Ordering::SeqCst);
            }
            // SAFETY: Vec<MaybeUninit<T>> only releases the already-erased
            // allocation, never invokes T Drop. Then its header itself is dead.
            unsafe {
                self.storage.assume_init_drop();
                wipe_bytes(
                    self.storage.as_mut_ptr().cast::<u8>(),
                    size_of::<Vec<MaybeUninit<T>>>(),
                );
            }
            self.first.zeroize();
            self.end.zeroize();
            #[cfg(test)]
            if let Some(witness) = &self.witness {
                // SAFETY: dead outer header is MaybeUninit storage initialized
                // entirely to zeros; neither Vec nor T is read after erasure.
                let header = unsafe {
                    std::slice::from_raw_parts(
                        self.storage.as_ptr().cast::<u8>(),
                        size_of::<Vec<MaybeUninit<T>>>(),
                    )
                };
                witness
                    .header_zero
                    .store(header.iter().all(|&b| b == 0), Ordering::SeqCst);
                witness
                    .counts_zero
                    .store(*self.first == 0 && *self.end == 0, Ordering::SeqCst);
                witness.calls.fetch_add(1, Ordering::SeqCst);
                if let Some(live) = &witness.live_workers {
                    witness
                        .joined
                        .store(live.load(Ordering::SeqCst) == 0, Ordering::SeqCst);
                }
            }
        }
    }

    #[cfg(test)]
    #[derive(Default)]
    pub(super) struct Witness {
        pub(super) slot_zero: AtomicBool,
        pub(super) header_zero: AtomicBool,
        pub(super) counts_zero: AtomicBool,
        pub(super) joined: AtomicBool,
        pub(super) slot_bytes: std::sync::atomic::AtomicUsize,
        pub(super) calls: std::sync::atomic::AtomicUsize,
        pub(super) live_workers: Option<std::sync::Arc<std::sync::atomic::AtomicUsize>>,
    }
}

fn checked_sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |sum, &value| {
        sum.checked_add(value).ok_or(Error::ArithmeticOverflow)
    })
}
fn bytes(values: &Vec<f32>) -> Result<usize> {
    values
        .capacity()
        .checked_mul(size_of::<f32>())
        .ok_or(Error::ArithmeticOverflow)
}
fn zeroed(length: usize) -> Result<Zeroizing<Vec<f32>>> {
    let mut values = Zeroizing::new(Vec::new());
    values
        .try_reserve_exact(length)
        .map_err(|_| Error::LimitExceeded("CPU expert private allocation"))?;
    if bytes(&values)? > MAX_PRIVATE_HOST_BYTES {
        return Err(Error::LimitExceeded("CPU expert private capacity"));
    }
    let capacity = values.capacity();
    values.resize(capacity, 0.0);
    Ok(values)
}
fn row(packed: &[u8], index: usize, width: usize) -> Result<&[u8]> {
    let first = index.checked_mul(width).ok_or(Error::ArithmeticOverflow)?;
    let end = first.checked_add(width).ok_or(Error::ArithmeticOverflow)?;
    packed.get(first..end).ok_or(Error::InvalidBufferLength)
}
fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(Error::InvalidFormat("CPU expert execution cancelled"))
    } else {
        Ok(())
    }
}
fn finite(value: f32, category: &'static str) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(Error::InvalidFormat(category))
    }
}

struct Scratch {
    decoded: Zeroizing<Vec<f32>>,
    block: Zeroizing<[f32; 256]>,
    sums: Zeroizing<[f32; LANES]>,
}
impl Scratch {
    fn new(width: usize) -> Result<Self> {
        let decoded = zeroed(width)?;
        let total = bytes(&decoded)?
            .checked_add(WORKER_FIXED_PRIVATE_BYTES)
            .ok_or(Error::ArithmeticOverflow)?;
        if total > MAX_WORKER_PRIVATE_BYTES {
            return Err(Error::LimitExceeded("CPU expert worker scratch"));
        }
        Ok(Self {
            decoded,
            block: Zeroizing::new([0.; 256]),
            sums: Zeroizing::new([0.; LANES]),
        })
    }
}

// All retained row pointers, bounds and completions live in the original
// full-capacity typed header owner, erased only after synchronous broadcast.
struct RowJob<'a> {
    first: usize,
    rows: &'a mut [f32],
    result: Option<Result<()>>,
}

fn pooled_rows(
    pool: &rayon::ThreadPool,
    output: &mut [f32],
    width: usize,
    cancel: &AtomicBool,
    hook: &(impl Fn(usize) + Sync),
    run: impl Fn(usize, &mut Scratch) -> Result<f32> + Sync,
) -> Result<()> {
    let failed = AtomicBool::new(false);
    let count = pool.current_num_threads().min(output.len());
    if count == 0 || count > MAX_WORKERS {
        return Err(Error::InvalidBufferLength);
    }
    let rows_per_worker = output.len().div_ceil(count);
    let result = (|| {
        let mut jobs = retained_metadata::Owner::new(MAX_WORKERS)?;
        let mut live = 0;
        for (worker, rows) in output.chunks_mut(rows_per_worker).enumerate() {
            jobs.push(std::sync::Mutex::new(RowJob {
                first: worker * rows_per_worker,
                rows,
                result: None,
            }))?;
            live += 1;
        }
        // Each broadcast index owns exactly one disjoint chunk. It is never
        // stolen, and returning unit keeps runtime result storage content-free.
        // catch_unwind is INSIDE the lock: a user hook panic cannot poison it.
        pool.broadcast(|context| {
            if context.index() >= live {
                return;
            }
            let Ok(slot) = jobs.get(context.index()) else {
                failed.store(true, Ordering::Relaxed);
                return;
            };
            let Ok(mut job) = slot.lock() else {
                failed.store(true, Ordering::Relaxed);
                return;
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut scratch = Scratch::new(width)?;
                let first = job.first;
                for (local, target) in job.rows.iter_mut().enumerate() {
                    if failed.load(Ordering::Relaxed) {
                        return Ok(());
                    }
                    check_cancel(cancel)?;
                    hook(first + local);
                    *target = run(first + local, &mut scratch)?;
                }
                Ok(())
            }))
            .unwrap_or(Err(Error::InvalidFormat("CPU expert worker panic")));
            if result.is_err() {
                failed.store(true, Ordering::Relaxed);
            }
            job.result = Some(result);
        });
        // Broadcast has completed every callback before inspection/header drop.
        let mut first_error = None;
        for index in 0..live {
            let result = jobs
                .get(index)?
                .lock()
                .map_err(|_| Error::InvalidFormat("CPU expert pool completion"))?
                .result
                .take()
                .ok_or(Error::InvalidFormat("CPU expert pool incomplete"))?;
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        if failed.load(Ordering::Relaxed) && first_error.is_none() {
            return Err(Error::InvalidFormat("CPU expert pool incomplete"));
        }
        first_error.map_or(Ok(()), Err)
    })();
    if result.is_err() {
        output.zeroize();
    }
    result
}

fn parallel_rows(
    workers: &WorkerPlan,
    output: &mut [f32],
    width: usize,
    cancel: &AtomicBool,
    hook: &(impl Fn(usize) + Sync),
    run: impl Fn(usize, &mut Scratch) -> Result<f32> + Sync,
) -> Result<()> {
    let failed = AtomicBool::new(false);
    let count = workers.count.min(output.len());
    if count == 0 {
        return Err(Error::InvalidBufferLength);
    }
    let rows_per_worker = output.len().div_ceil(count);
    let result = std::thread::scope(|scope| {
        // The exact accepted fixed-capacity header bytes were precharged before
        // private input fill. Scoped lifetimes stay wholly inside this closure.
        let mut handles = retained_metadata::Owner::new(MAX_WORKERS)?;
        let mut first_error = None;
        for (worker, rows) in output.chunks_mut(rows_per_worker).enumerate() {
            let first = worker * rows_per_worker;
            let failed = &failed;
            let run = &run;
            let task = std::thread::Builder::new().spawn_scoped(scope, move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if workers.pinned {
                        affinity::pin(workers.cores[worker])?;
                    }
                    let mut scratch = Scratch::new(width)?;
                    for (local, target) in rows.iter_mut().enumerate() {
                        if failed.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        check_cancel(cancel)?;
                        hook(first + local);
                        *target = run(first + local, &mut scratch)?;
                    }
                    Ok(())
                }))
                .unwrap_or(Err(Error::InvalidFormat("CPU expert worker panic")));
                if result.is_err() {
                    failed.store(true, Ordering::Relaxed);
                }
                result
            });
            match task {
                Ok(handle) => {
                    // count <= MAX_WORKERS and each push follows one successful
                    // spawn; this cannot exceed the exact admitted capacity.
                    handles.push(handle)?;
                }
                Err(_) => {
                    failed.store(true, Ordering::Relaxed);
                    first_error = Some(Error::LimitExceeded("CPU expert worker spawn"));
                    break;
                }
            }
        }
        // Every handle is consumed/joined, including after partial spawn or panic.
        // No worker can outlive the borrowed activation/output buffers.
        while let Some(handle) = handles.take_first() {
            let result = match handle.join() {
                Ok(result) => result,
                Err(_) => {
                    failed.store(true, Ordering::Relaxed);
                    Err(Error::InvalidFormat("CPU expert worker panic"))
                }
            };
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    });
    if result.is_err() {
        output.zeroize();
    }
    result
}

fn dot_row(
    backend: Backend,
    dtype: TensorType,
    packed: &[u8],
    input: &[f32],
    scratch: &mut Scratch,
    cancel: &AtomicBool,
) -> Result<f32> {
    if dtype.row_bytes(input.len() as u64)? != packed.len() as u64
        || input.len() > scratch.decoded.len()
    {
        return Err(Error::InvalidBufferLength);
    }
    match backend {
        Backend::Portable => {
            let decoded = &mut scratch.decoded[..input.len()];
            dtype.decode(packed, decoded)?;
            scratch.sums.fill(0.);
            for (column, (&weight, &value)) in decoded.iter().zip(input).enumerate() {
                if column.is_multiple_of(256) {
                    check_cancel(cancel)?;
                }
                finite(weight, "CPU expert non-finite packed weight")?;
                scratch.sums[column % LANES] += weight * value;
            }
            let result = reduce64(&mut scratch.sums);
            finite(result, "CPU expert non-finite projection")
        }
        Backend::Avx2 => avx::dot(dtype, packed, input, scratch, cancel),
        Backend::Avx2Fma => avx::dot_fma(dtype, packed, input, scratch, cancel),
    }
}

fn reduce64(sums: &mut [f32; LANES]) -> f32 {
    let mut stride = 32;
    while stride != 0 {
        for lane in 0..stride {
            sums[lane] += sums[lane + stride];
        }
        stride /= 2;
    }
    sums[0]
}

pub fn avx2_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}
pub fn fma_available() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        avx2_available()
            && std::is_x86_feature_detected!("f16c")
            && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod avx {
    use super::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    struct Sums([__m256; 8]);
    impl Zeroize for Sums {
        fn zeroize(&mut self) {
            for value in &mut self.0 {
                // SAFETY: __m256 has no invalid all-zero bit pattern. Each
                // destination is an initialized owned 256-bit stack object.
                unsafe {
                    std::ptr::write_volatile(value, std::mem::zeroed());
                }
            }
            std::sync::atomic::compiler_fence(Ordering::SeqCst);
        }
    }

    /// The original64-lane tree, entirely in registers: strides32,16,8,4,2,1.
    /// Every add has the same two operands and f32 round point as reduce64.
    /// In particular, this is neither a horizontal sum nor reassociation/FMA.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn finish(sums: &Sums) -> Result<f32> {
        let a0 = _mm256_add_ps(sums.0[0], sums.0[4]);
        let a1 = _mm256_add_ps(sums.0[1], sums.0[5]);
        let a2 = _mm256_add_ps(sums.0[2], sums.0[6]);
        let a3 = _mm256_add_ps(sums.0[3], sums.0[7]);
        let b0 = _mm256_add_ps(a0, a2);
        let b1 = _mm256_add_ps(a1, a3);
        let c = _mm256_add_ps(b0, b1);
        let d = _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps::<1>(c));
        let e = _mm_add_ps(d, _mm_movehl_ps(d, d));
        let f = _mm_add_ss(e, _mm_shuffle_ps::<0x55>(e, e));
        finite(_mm_cvtss_f32(f), "CPU expert non-finite projection")
    }

    #[cfg(test)]
    #[allow(unsafe_code)]
    pub(super) fn test_reduce(values: &[f32; LANES]) -> Result<f32> {
        if !avx2_available() {
            return Err(Error::InvalidFormat("CPU expert AVX2 unavailable"));
        }
        let mut sums = Zeroizing::new(Sums([unsafe { std::mem::zeroed() }; 8]));
        for (bank, vector) in sums.0.iter_mut().enumerate() {
            // SAFETY: detected AVX2, eight complete disjoint owning lane spans.
            *vector = unsafe { _mm256_loadu_ps(values.as_ptr().add(bank * 8)) };
        }
        // SAFETY: detected AVX2 and the original64 complete lanes above.
        unsafe { finish(&sums) }
    }

    pub(super) fn dot(
        dtype: TensorType,
        packed: &[u8],
        input: &[f32],
        scratch: &mut Scratch,
        cancel: &AtomicBool,
    ) -> Result<f32> {
        if !avx2_available() {
            return Err(Error::InvalidFormat("CPU expert AVX2 unavailable"));
        }
        // SAFETY: runtime ISA detection above; dot_row validated complete packed
        // row/input geometry. Every vector load/store is bounded below.
        unsafe { run(dtype, packed, input, scratch, cancel) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn run(
        dtype: TensorType,
        packed: &[u8],
        input: &[f32],
        scratch: &mut Scratch,
        cancel: &AtomicBool,
    ) -> Result<f32> {
        // These layouts have finite integer quants and bounded binary16 scale
        // products. Decode directly into SIMD registers rather than expanding
        // every row to a temporary f32 block. Keep separate multiply/add and the
        // original eight accumulators (64 lanes), including every round point.
        if matches!(dtype, TensorType::Q4K | TensorType::Q5_0 | TensorType::Q8_0) {
            // SAFETY: this function has AVX2 enabled and dot_row proved the full
            // original block geometry. Each helper load is bounded by its block.
            return unsafe { packed_dot(dtype, packed, input, scratch, cancel) };
        }
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let chunk = (dtype.block_elements() as usize).max(LANES);
        let row_bytes = usize::try_from(dtype.row_bytes(input.len() as u64)?)
            .map_err(|_| Error::ArithmeticOverflow)?;
        if chunk > scratch.block.len() || row_bytes != packed.len() {
            return Err(Error::InvalidBufferLength);
        }
        let mut column = 0usize;
        while column < input.len() {
            check_cancel(cancel)?;
            let elements = chunk.min(input.len() - column);
            // For first=0 the byte offset is exactly0. Nonzero columns are full
            // quantization-block boundaries even when the final F32 tail is odd.
            let byte_first = if column == 0 {
                0
            } else {
                usize::try_from(dtype.row_bytes(column as u64)?)
                    .map_err(|_| Error::ArithmeticOverflow)?
            };
            let byte_count = usize::try_from(dtype.row_bytes(elements as u64)?)
                .map_err(|_| Error::ArithmeticOverflow)?;
            let byte_end = byte_first
                .checked_add(byte_count)
                .ok_or(Error::ArithmeticOverflow)?;
            let source = packed
                .get(byte_first..byte_end)
                .ok_or(Error::InvalidBufferLength)?;
            dtype.decode(source, &mut scratch.block[..elements])?;
            if scratch.block[..elements]
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(Error::InvalidFormat("CPU expert non-finite packed weight"));
            }
            let vectors = elements / 8;
            for vector in 0..vectors {
                let local = vector * 8;
                let bank = ((column + local) % LANES) / 8;
                // SAFETY: local+8<=elements<=256 and column+elements<=input.len;
                // byte decoding filled all weight lanes. Unaligned reads allowed.
                let (weight, activation) = unsafe {
                    (
                        _mm256_loadu_ps(scratch.block.as_ptr().add(local)),
                        _mm256_loadu_ps(input.as_ptr().add(column + local)),
                    )
                };
                // Explicit mul/add preserves every original per-lane round point.
                sums.0[bank] = _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
            }
            if !elements.is_multiple_of(8) {
                // Only unquantized final tails can reach here. Update the same
                // lanes using owning guarded scalar storage, never reordered.
                for (bank, vector) in sums.0.iter().enumerate() {
                    // SAFETY: bank<8; exactly8 floats inside the64-element array.
                    unsafe {
                        _mm256_storeu_ps(scratch.sums.as_mut_ptr().add(bank * 8), *vector);
                    }
                }
                for local in vectors * 8..elements {
                    let lane = (column + local) % LANES;
                    scratch.sums[lane] += scratch.block[local] * input[column + local];
                }
                for (bank, vector) in sums.0.iter_mut().enumerate() {
                    // SAFETY: bank<8; each initialized owning scalar span has8 floats.
                    *vector = unsafe { _mm256_loadu_ps(scratch.sums.as_ptr().add(bank * 8)) };
                }
            }
            column += elements;
        }
        // SAFETY: run has AVX2 enabled; sums owns all original64 lane results.
        unsafe { finish(&sums) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn packed_dot(
        dtype: TensorType,
        packed: &[u8],
        input: &[f32],
        _scratch: &mut Scratch,
        cancel: &AtomicBool,
    ) -> Result<f32> {
        if std::is_x86_feature_detected!("f16c") {
            // SAFETY: run already requires AVX2; F16C is detected here. The
            // original dot_row proved the complete packed row/input geometry.
            return unsafe { packed_f16c_dot(dtype, packed, input, cancel) };
        }
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let mask = _mm256_set1_epi32(15);
        let elements = dtype.block_elements() as usize;
        let block_bytes = dtype.block_bytes() as usize;
        for (block_index, block) in packed.chunks_exact(block_bytes).enumerate() {
            check_cancel(cancel)?;
            let first = block_index * elements;
            let scale = crate::quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            finite(scale, "CPU expert non-finite packed weight")?;
            if dtype == TensorType::Q4K {
                let minimum = crate::quant::f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
                finite(minimum, "CPU expert non-finite packed weight")?;
                let scales = &block[4..16];
                for group in 0..4 {
                    // Two explicit inline calls avoid the out-of-line array
                    // construction/closure observed in the fixed-public CPU
                    // profile. Every original f32 multiply round point stays.
                    let factors = [
                        q4_factors(scales, group * 2, scale, minimum),
                        q4_factors(scales, group * 2 + 1, scale, minimum),
                    ];
                    for vector in 0..4 {
                        let local = vector * 8;
                        // SAFETY: 16+group*32+local+8 <= 144.
                        let bytes = unsafe {
                            _mm_loadl_epi64(block.as_ptr().add(16 + group * 32 + local).cast())
                        };
                        let quants = _mm256_cvtepu8_epi32(bytes);
                        let q = [
                            _mm256_and_si256(quants, mask),
                            _mm256_srli_epi32::<4>(quants),
                        ];
                        for side in 0..2 {
                            let weight = _mm256_sub_ps(
                                _mm256_mul_ps(
                                    _mm256_set1_ps(factors[side].0),
                                    _mm256_cvtepi32_ps(q[side]),
                                ),
                                _mm256_set1_ps(factors[side].1),
                            );
                            // SAFETY: first+group*64+side*32+local+8 is inside
                            // the original 256-element input block.
                            let activation = unsafe {
                                _mm256_loadu_ps(
                                    input.as_ptr().add(first + group * 64 + side * 32 + local),
                                )
                            };
                            let bank = side * 4 + vector;
                            sums.0[bank] =
                                _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
                        }
                    }
                }
            } else {
                let high = if dtype == TensorType::Q5_0 {
                    u32::from_le_bytes(block[2..6].try_into().unwrap())
                } else {
                    0
                };
                for vector in 0..4 {
                    let local = vector * 8;
                    let quants = if dtype == TensorType::Q8_0 {
                        // SAFETY: 2+local+8 <= 34, signed bytes are original Q8.
                        let bytes =
                            unsafe { _mm_loadl_epi64(block.as_ptr().add(2 + local).cast()) };
                        _mm256_cvtepi8_epi32(bytes)
                    } else {
                        // SAFETY: 6+local%16+8 <= 22.
                        let bytes =
                            unsafe { _mm_loadl_epi64(block.as_ptr().add(6 + local % 16).cast()) };
                        let bytes = _mm256_cvtepu8_epi32(bytes);
                        let low = if local < 16 {
                            _mm256_and_si256(bytes, mask)
                        } else {
                            _mm256_srli_epi32::<4>(bytes)
                        };
                        let shifts = _mm256_setr_epi32(
                            local as i32,
                            local as i32 + 1,
                            local as i32 + 2,
                            local as i32 + 3,
                            local as i32 + 4,
                            local as i32 + 5,
                            local as i32 + 6,
                            local as i32 + 7,
                        );
                        let bit = _mm256_and_si256(
                            _mm256_srlv_epi32(_mm256_set1_epi32(high as i32), shifts),
                            _mm256_set1_epi32(1),
                        );
                        _mm256_sub_epi32(
                            _mm256_or_si256(low, _mm256_slli_epi32::<4>(bit)),
                            _mm256_set1_epi32(16),
                        )
                    };
                    let weight = _mm256_mul_ps(_mm256_set1_ps(scale), _mm256_cvtepi32_ps(quants));
                    // SAFETY: first+local+8 <= input.len() by original32 blocks.
                    let activation = unsafe { _mm256_loadu_ps(input.as_ptr().add(first + local)) };
                    let bank = ((first + local) % LANES) / 8;
                    sums.0[bank] = _mm256_add_ps(sums.0[bank], _mm256_mul_ps(weight, activation));
                }
            }
        }
        // SAFETY: packed_dot has AVX2 enabled and preserves original64 lanes.
        unsafe { finish(&sums) }
    }

    #[inline(always)]
    fn q4_factors(scales: &[u8], index: usize, scale: f32, minimum: f32) -> (f32, f32) {
        let (s, m) = if index < 4 {
            (scales[index] & 63, scales[index + 4] & 63)
        } else {
            (
                (scales[index + 4] & 15) | ((scales[index - 4] >> 6) << 4),
                (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
            )
        };
        (scale * f32::from(s), minimum * f32::from(m))
    }

    // Keep dtype decisions outside the hot loop. Original64 lanes and every
    // multiply/add round point remain; no activation quantization or FMA.
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn packed_f16c_dot(
        dtype: TensorType,
        packed: &[u8],
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<f32> {
        // SAFETY: caller detected both features and validated the full row.
        unsafe {
            match dtype {
                TensorType::Q4K => packed_q4_f16c(packed, input, cancel),
                TensorType::Q5_0 => packed_32_f16c::<true>(packed, input, cancel),
                TensorType::Q8_0 => packed_32_f16c::<false>(packed, input, cancel),
                _ => Err(Error::InvalidFormat("CPU packed SIMD layout")),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn half_pair(bits: u32) -> Result<(f32, f32)> {
        let values = _mm_cvtph_ps(_mm_cvtsi32_si128(bits as i32));
        let first = _mm_cvtss_f32(values);
        let second = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(values, values));
        finite(first, "CPU expert non-finite packed weight")?;
        finite(second, "CPU expert non-finite packed weight")?;
        Ok((first, second))
    }

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn packed_q4_f16c(packed: &[u8], input: &[f32], cancel: &AtomicBool) -> Result<f32> {
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let mask = _mm256_set1_epi32(15);
        for (index, block) in packed.chunks_exact(144).enumerate() {
            check_cancel(cancel)?;
            // SAFETY: this target has F16C; each full block has its four-byte
            // half pair. Finite binary16 scales have exact binary32 encodings.
            let (scale, minimum) =
                unsafe { half_pair(u32::from_le_bytes(block[..4].try_into().unwrap()))? };
            let first = index * 256;
            macro_rules! group {
                ($g:literal) => {{
                    let (s0, m0) = q4_factors(&block[4..16], $g * 2, scale, minimum);
                    let (s1, m1) = q4_factors(&block[4..16], $g * 2 + 1, scale, minimum);
                    let s0 = _mm256_set1_ps(s0);
                    let m0 = _mm256_set1_ps(m0);
                    let s1 = _mm256_set1_ps(s1);
                    let m1 = _mm256_set1_ps(m1);
                    macro_rules! vector {
                        ($v:literal) => {{
                            // SAFETY: full144-byte Q4 block, full256-value
                            // input block. Literal g<4/v<4 bounds every load;
                            // original row geometry was checked by dot_row.
                            unsafe {
                                let bytes = _mm_loadl_epi64(
                                    block.as_ptr().add(16 + $g * 32 + $v * 8).cast(),
                                );
                                let quants = _mm256_cvtepu8_epi32(bytes);
                                let low = _mm256_cvtepi32_ps(_mm256_and_si256(quants, mask));
                                let high = _mm256_cvtepi32_ps(_mm256_srli_epi32::<4>(quants));
                                let w0 = _mm256_sub_ps(_mm256_mul_ps(s0, low), m0);
                                let w1 = _mm256_sub_ps(_mm256_mul_ps(s1, high), m1);
                                let x0 =
                                    _mm256_loadu_ps(input.as_ptr().add(first + $g * 64 + $v * 8));
                                let x1 = _mm256_loadu_ps(
                                    input.as_ptr().add(first + $g * 64 + 32 + $v * 8),
                                );
                                sums.0[$v] = _mm256_add_ps(sums.0[$v], _mm256_mul_ps(w0, x0));
                                sums.0[$v + 4] =
                                    _mm256_add_ps(sums.0[$v + 4], _mm256_mul_ps(w1, x1));
                            }
                        }};
                    }
                    vector!(0);
                    vector!(1);
                    vector!(2);
                    vector!(3);
                }};
            }
            group!(0);
            group!(1);
            group!(2);
            group!(3);
        }
        // SAFETY: this target has AVX2 and the original complete64 sums.
        unsafe { finish(&sums) }
    }

    #[target_feature(enable = "avx2,f16c")]
    unsafe fn packed_32_f16c<const Q5: bool>(
        packed: &[u8],
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<f32> {
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let bytes_per_block = if Q5 { 22 } else { 34 };
        macro_rules! block32 {
            ($block:expr, $first:expr, $bank:literal) => {{
                check_cancel(cancel)?;
                let block = $block;
                let first = $first;
                // SAFETY: full original32-value block and detected F16C.
                let (scale, _) =
                    unsafe { half_pair(u32::from(u16::from_le_bytes([block[0], block[1]])))? };
                let scale = _mm256_set1_ps(scale);
                let high = if Q5 {
                    u32::from_le_bytes(block[2..6].try_into().unwrap())
                } else {
                    0
                };
                macro_rules! vector {
                    ($v:literal) => {{
                        // SAFETY: literal v<4 bounds the eight-byte packed load
                        // and eight-value activation load in the original full
                        // block. Q5/8 dtype geometry was checked by dot_row.
                        unsafe {
                            let quants = if Q5 {
                                let bytes =
                                    _mm_loadl_epi64(block.as_ptr().add(6 + ($v * 8) % 16).cast());
                                let bytes = _mm256_cvtepu8_epi32(bytes);
                                let low = if $v < 2 {
                                    _mm256_and_si256(bytes, _mm256_set1_epi32(15))
                                } else {
                                    _mm256_srli_epi32::<4>(bytes)
                                };
                                let shifts = _mm256_setr_epi32(
                                    $v * 8,
                                    $v * 8 + 1,
                                    $v * 8 + 2,
                                    $v * 8 + 3,
                                    $v * 8 + 4,
                                    $v * 8 + 5,
                                    $v * 8 + 6,
                                    $v * 8 + 7,
                                );
                                let bit = _mm256_and_si256(
                                    _mm256_srlv_epi32(_mm256_set1_epi32(high as i32), shifts),
                                    _mm256_set1_epi32(1),
                                );
                                _mm256_sub_epi32(
                                    _mm256_or_si256(low, _mm256_slli_epi32::<4>(bit)),
                                    _mm256_set1_epi32(16),
                                )
                            } else {
                                let bytes = _mm_loadl_epi64(block.as_ptr().add(2 + $v * 8).cast());
                                _mm256_cvtepi8_epi32(bytes)
                            };
                            let weight = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(quants));
                            let activation = _mm256_loadu_ps(input.as_ptr().add(first + $v * 8));
                            sums.0[$bank + $v] = _mm256_add_ps(
                                sums.0[$bank + $v],
                                _mm256_mul_ps(weight, activation),
                            );
                        }
                    }};
                }
                vector!(0);
                vector!(1);
                vector!(2);
                vector!(3);
            }};
        }
        let mut pairs = packed.chunks_exact(bytes_per_block * 2);
        for (index, pair) in pairs.by_ref().enumerate() {
            block32!(&pair[..bytes_per_block], index * 64, 0);
            block32!(&pair[bytes_per_block..], index * 64 + 32, 4);
        }
        if !pairs.remainder().is_empty() {
            // Original quantized rows can end with one full32-value block.
            block32!(pairs.remainder(), input.len() - 32, 0);
        }
        // SAFETY: this target has AVX2 and every original64 lane is retained.
        unsafe { finish(&sums) }
    }

    pub(super) fn dot_fma(
        dtype: TensorType,
        packed: &[u8],
        input: &[f32],
        scratch: &mut Scratch,
        cancel: &AtomicBool,
    ) -> Result<f32> {
        if !fma_available() {
            return Err(Error::InvalidFormat("CPU expert AVX2/F16C/FMA unavailable"));
        }
        // SAFETY: all three features checked; dot_row proved original packing.
        unsafe {
            match dtype {
                TensorType::Q4K => packed_q4_fma(packed, input, cancel),
                TensorType::Q5_0 => packed_32_fma::<true>(packed, input, cancel),
                TensorType::Q8_0 => packed_32_fma::<false>(packed, input, cancel),
                _ => run(dtype, packed, input, scratch, cancel),
            }
        }
    }

    #[target_feature(enable = "avx2,f16c,fma")]
    unsafe fn packed_q4_fma(packed: &[u8], input: &[f32], cancel: &AtomicBool) -> Result<f32> {
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let mask = _mm256_set1_epi32(15);
        for (index, block) in packed.chunks_exact(144).enumerate() {
            check_cancel(cancel)?;
            // SAFETY: this target has F16C; each full block has its four-byte
            // half pair. Finite binary16 scales have exact binary32 encodings.
            let (scale, minimum) =
                unsafe { half_pair(u32::from_le_bytes(block[..4].try_into().unwrap()))? };
            let first = index * 256;
            macro_rules! group {
                ($g:literal) => {{
                    let (s0, m0) = q4_factors(&block[4..16], $g * 2, scale, minimum);
                    let (s1, m1) = q4_factors(&block[4..16], $g * 2 + 1, scale, minimum);
                    let s0 = _mm256_set1_ps(s0);
                    let m0 = _mm256_set1_ps(m0);
                    let s1 = _mm256_set1_ps(s1);
                    let m1 = _mm256_set1_ps(m1);
                    macro_rules! vector {
                        ($v:literal) => {{
                            // SAFETY: full144-byte Q4 block, full256-value
                            // input block. Literal g<4/v<4 bounds every load;
                            // original row geometry was checked by dot_row.
                            unsafe {
                                let bytes = _mm_loadl_epi64(
                                    block.as_ptr().add(16 + $g * 32 + $v * 8).cast(),
                                );
                                let quants = _mm256_cvtepu8_epi32(bytes);
                                let low = _mm256_cvtepi32_ps(_mm256_and_si256(quants, mask));
                                let high = _mm256_cvtepi32_ps(_mm256_srli_epi32::<4>(quants));
                                let w0 = _mm256_sub_ps(_mm256_mul_ps(s0, low), m0);
                                let w1 = _mm256_sub_ps(_mm256_mul_ps(s1, high), m1);
                                let x0 =
                                    _mm256_loadu_ps(input.as_ptr().add(first + $g * 64 + $v * 8));
                                let x1 = _mm256_loadu_ps(
                                    input.as_ptr().add(first + $g * 64 + 32 + $v * 8),
                                );
                                sums.0[$v] = _mm256_fmadd_ps(w0, x0, sums.0[$v]);
                                sums.0[$v] = _mm256_fmadd_ps(w1, x1, sums.0[$v]);
                            }
                        }};
                    }
                    vector!(0);
                    vector!(1);
                    vector!(2);
                    vector!(3);
                }};
            }
            group!(0);
            group!(1);
            group!(2);
            group!(3);
        }
        // SAFETY: this target has AVX2 and the bounded32 accumulated lanes.
        unsafe { finish(&sums) }
    }

    #[target_feature(enable = "avx2,f16c,fma")]
    unsafe fn packed_32_fma<const Q5: bool>(
        packed: &[u8],
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<f32> {
        let mut sums = Zeroizing::new(Sums([_mm256_setzero_ps(); 8]));
        let bytes_per_block = if Q5 { 22 } else { 34 };
        macro_rules! block32 {
            ($block:expr, $first:expr, $bank:literal) => {{
                check_cancel(cancel)?;
                let block = $block;
                let first = $first;
                // SAFETY: full original32-value block and detected F16C.
                let (scale, _) =
                    unsafe { half_pair(u32::from(u16::from_le_bytes([block[0], block[1]])))? };
                let scale = _mm256_set1_ps(scale);
                let high = if Q5 {
                    u32::from_le_bytes(block[2..6].try_into().unwrap())
                } else {
                    0
                };
                macro_rules! vector {
                    ($v:literal) => {{
                        // SAFETY: literal v<4 bounds the eight-byte packed load
                        // and eight-value activation load in the original full
                        // block. Q5/8 dtype geometry was checked by dot_row.
                        unsafe {
                            let quants = if Q5 {
                                let bytes =
                                    _mm_loadl_epi64(block.as_ptr().add(6 + ($v * 8) % 16).cast());
                                let bytes = _mm256_cvtepu8_epi32(bytes);
                                let low = if $v < 2 {
                                    _mm256_and_si256(bytes, _mm256_set1_epi32(15))
                                } else {
                                    _mm256_srli_epi32::<4>(bytes)
                                };
                                let shifts = _mm256_setr_epi32(
                                    $v * 8,
                                    $v * 8 + 1,
                                    $v * 8 + 2,
                                    $v * 8 + 3,
                                    $v * 8 + 4,
                                    $v * 8 + 5,
                                    $v * 8 + 6,
                                    $v * 8 + 7,
                                );
                                let bit = _mm256_and_si256(
                                    _mm256_srlv_epi32(_mm256_set1_epi32(high as i32), shifts),
                                    _mm256_set1_epi32(1),
                                );
                                _mm256_sub_epi32(
                                    _mm256_or_si256(low, _mm256_slli_epi32::<4>(bit)),
                                    _mm256_set1_epi32(16),
                                )
                            } else {
                                let bytes = _mm_loadl_epi64(block.as_ptr().add(2 + $v * 8).cast());
                                _mm256_cvtepi8_epi32(bytes)
                            };
                            let weight = _mm256_mul_ps(scale, _mm256_cvtepi32_ps(quants));
                            let activation = _mm256_loadu_ps(input.as_ptr().add(first + $v * 8));
                            sums.0[$v] = _mm256_fmadd_ps(weight, activation, sums.0[$v]);
                        }
                    }};
                }
                vector!(0);
                vector!(1);
                vector!(2);
                vector!(3);
            }};
        }
        let mut pairs = packed.chunks_exact(bytes_per_block * 2);
        for (index, pair) in pairs.by_ref().enumerate() {
            block32!(&pair[..bytes_per_block], index * 64, 0);
            block32!(&pair[bytes_per_block..], index * 64 + 32, 4);
        }
        if !pairs.remainder().is_empty() {
            // Quantized rows can end with one full32-value block.
            block32!(pairs.remainder(), input.len() - 32, 0);
        }
        // SAFETY: this target has AVX2 and the bounded32 accumulated lanes.
        unsafe { finish(&sums) }
    }

    #[cfg(test)]
    pub(super) fn test_half_pair(bits: u32) -> Result<(f32, f32)> {
        if !avx2_available() || !std::is_x86_feature_detected!("f16c") {
            return Err(Error::InvalidFormat("CPU expert F16C unavailable"));
        }
        // SAFETY: both ISA features were detected above.
        unsafe { half_pair(bits) }
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
mod avx {
    pub(super) fn dot_fma(
        _: TensorType,
        _: &[u8],
        _: &[f32],
        _: &mut Scratch,
        _: &AtomicBool,
    ) -> Result<f32> {
        Err(Error::InvalidFormat("CPU expert AVX2/F16C/FMA unavailable"))
    }
    use super::*;
    pub(super) fn dot(
        _: TensorType,
        _: &[u8],
        _: &[f32],
        _: &mut Scratch,
        _: &AtomicBool,
    ) -> Result<f32> {
        Err(Error::InvalidFormat("CPU expert AVX2 unavailable"))
    }
}

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod affinity {
    use super::*;
    use std::io::Read;
    const MAX_CPUS: usize = 4096;
    const WORDS: usize = MAX_CPUS / usize::BITS as usize;
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, bytes: usize, mask: *mut usize) -> i32;
        fn sched_setaffinity(pid: i32, bytes: usize, mask: *const usize) -> i32;
    }
    fn allowed() -> Result<[usize; WORDS]> {
        let mut mask = [0usize; WORDS];
        // SAFETY: pid0 is this calling thread, mask is aligned initialized
        // storage of the exact passed length; libc writes only that extent.
        let result = unsafe { sched_getaffinity(0, size_of_val(&mask), mask.as_mut_ptr()) };
        if result != 0 {
            return Err(Error::InvalidFormat("CPU expert affinity unavailable"));
        }
        Ok(mask)
    }
    #[cfg(test)]
    pub(super) fn test_allowed() -> Result<[usize; WORDS]> {
        allowed()
    }
    fn topology(cpu: usize, field: &str) -> Result<u32> {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/{field}");
        let mut bytes = Vec::new();
        std::fs::File::open(path)?
            .take(65)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 64 {
            return Err(Error::InvalidFormat("CPU expert topology size"));
        }
        std::str::from_utf8(&bytes)
            .ok()
            .and_then(|text| text.trim().parse().ok())
            .ok_or(Error::InvalidFormat("CPU expert topology value"))
    }
    pub(super) fn physical_cores(count: usize) -> Result<[usize; MAX_WORKERS]> {
        let mask = allowed()?;
        let mut cores = [0usize; MAX_WORKERS];
        let mut identities = [(0u32, 0u32); MAX_WORKERS];
        let mut found = 0;
        for cpu in 0..MAX_CPUS {
            if mask[cpu / usize::BITS as usize] & (1usize << (cpu % usize::BITS as usize)) == 0 {
                continue;
            }
            let id = (
                topology(cpu, "physical_package_id")?,
                topology(cpu, "core_id")?,
            );
            if identities[..found].contains(&id) {
                continue;
            }
            cores[found] = cpu;
            identities[found] = id;
            found += 1;
            if found == count {
                return Ok(cores);
            }
        }
        Err(Error::LimitExceeded("CPU expert allowed physical cores"))
    }
    pub(super) fn pin(cpu: usize) -> Result<()> {
        if cpu >= MAX_CPUS {
            return Err(Error::InvalidFormat("CPU expert affinity CPU"));
        }
        let mut mask = [0usize; WORDS];
        mask[cpu / usize::BITS as usize] = 1usize << (cpu % usize::BITS as usize);
        // SAFETY: applies only to this scoped worker, CPU index/whole mask are
        // checked and the kernel cannot access outside the passed extent.
        let result = unsafe { sched_setaffinity(0, size_of_val(&mask), mask.as_ptr()) };
        if result != 0 {
            return Err(Error::InvalidFormat("CPU expert worker affinity"));
        }
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::{GgufFile, GgufLimits};
    use std::sync::atomic::{AtomicU64, AtomicUsize};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    const KINDS: [TensorType; 7] = [
        TensorType::F32,
        TensorType::F16,
        TensorType::Bf16,
        TensorType::Q5_0,
        TensorType::Q8_0,
        TensorType::Q4K,
        TensorType::Q6K,
    ];

    struct Fixture {
        path: std::path::PathBuf,
        model: GgufFile,
        expected: [Vec<f32>; 3],
        hidden: usize,
        ff: usize,
        experts: usize,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    // Independent encoders and numerical values; this oracle does not call
    // TensorType::decode, packed CPU kernels or the model scalar executor.
    fn block(kind: TensorType, seed: usize) -> (Vec<u8>, Vec<f32>) {
        match kind {
            TensorType::F32 => {
                let value = (seed % 13) as f32 / 32. - 0.1875;
                (value.to_le_bytes().to_vec(), vec![value])
            }
            TensorType::F16 => {
                let choices: [(u16, f32); 7] = [
                    (0, 0.),
                    (0x8000, -0.),
                    (0x3c00, 1.),
                    (0xbc00, -1.),
                    (0x3800, 0.5),
                    (0xb800, -0.5),
                    (1, f32::from_bits(0x33800000)),
                ];
                let (bits, value) = choices[seed % choices.len()];
                (bits.to_le_bytes().to_vec(), vec![value])
            }
            TensorType::Bf16 => {
                let choices: [(u16, f32); 7] = [
                    (0, 0.),
                    (0x8000, -0.),
                    (0x3f80, 1.),
                    (0xbf80, -1.),
                    (0x3f00, 0.5),
                    (0xbf00, -0.5),
                    (1, f32::from_bits(0x00010000)),
                ];
                let (bits, value) = choices[seed % choices.len()];
                (bits.to_le_bytes().to_vec(), vec![value])
            }
            TensorType::Q5_0 => {
                let mut bytes = vec![0u8; 22];
                bytes[..2].copy_from_slice(&0x3400u16.to_le_bytes());
                let mut expected = vec![0.; 32];
                let mut high = 0u32;
                for (column, value) in expected.iter_mut().enumerate() {
                    let quant = (column * 7 + seed * 3) % 32;
                    let offset = 6 + column % 16;
                    bytes[offset] |= ((quant & 15) as u8) << ((column / 16) * 4);
                    high |= ((quant >> 4) as u32) << column;
                    *value = 0.25 * (quant as i32 - 16) as f32;
                }
                bytes[2..6].copy_from_slice(&high.to_le_bytes());
                (bytes, expected)
            }
            TensorType::Q8_0 => {
                let mut bytes = vec![0u8; 34];
                bytes[..2].copy_from_slice(&0x3400u16.to_le_bytes());
                let mut expected = vec![0.; 32];
                for (column, value) in expected.iter_mut().enumerate() {
                    let quant = ((column * 11 + seed * 5) % 63) as i32 - 31;
                    bytes[2 + column] = quant as i8 as u8;
                    *value = 0.25 * quant as f32;
                }
                (bytes, expected)
            }
            TensorType::Q4K => {
                let mut bytes = vec![0u8; 144];
                bytes[..2].copy_from_slice(&0x3400u16.to_le_bytes());
                bytes[2..4].copy_from_slice(&0x3000u16.to_le_bytes());
                let mut scales = [0u8; 8];
                let mut minima = [0u8; 8];
                for group in 0..8 {
                    scales[group] = ((group * 19 + seed * 7) % 64) as u8;
                    minima[group] = ((group * 13 + seed * 11) % 64) as u8;
                    if group < 4 {
                        bytes[4 + group] |= scales[group];
                        bytes[8 + group] |= minima[group];
                    } else {
                        bytes[8 + group] = (scales[group] & 15) | ((minima[group] & 15) << 4);
                        bytes[group] |= (scales[group] >> 4) << 6;
                        bytes[4 + group] |= (minima[group] >> 4) << 6;
                    }
                }
                let mut expected = vec![0.; 256];
                for (column, value) in expected.iter_mut().enumerate() {
                    let group = column / 32;
                    let quant = ((column * 7 + seed * 3) % 16) as u8;
                    bytes[16 + (group / 2) * 32 + column % 32] |= quant << ((group % 2) * 4);
                    *value = (0.25 * f32::from(scales[group])) * f32::from(quant)
                        - 0.125 * f32::from(minima[group]);
                }
                (bytes, expected)
            }
            TensorType::Q6K => {
                let mut bytes = vec![0u8; 210];
                bytes[208..].copy_from_slice(&0x3400u16.to_le_bytes());
                for scale in 0..16 {
                    bytes[192 + scale] = (((scale * 5 + seed) % 11) as i32 - 5) as i8 as u8;
                }
                let mut expected = vec![0.; 256];
                for half in 0..2 {
                    for quarter in 0..4 {
                        for lane in 0..32 {
                            let column = half * 128 + quarter * 32 + lane;
                            let quant = ((column * 7 + seed * 3) % 64) as u8;
                            let low = half * 64 + lane + (quarter % 2) * 32;
                            bytes[low] |= (quant & 15) << ((quarter / 2) * 4);
                            bytes[128 + half * 32 + lane] |= (quant >> 4) << (quarter * 2);
                            let scale = bytes[192 + half * 8 + quarter * 2 + lane / 16] as i8;
                            expected[column] =
                                (0.25 * f32::from(scale)) * (quant as i32 - 32) as f32;
                        }
                    }
                }
                (bytes, expected)
            }
        }
    }

    fn encoded_row(kind: TensorType, columns: usize, seed: usize) -> (Vec<u8>, Vec<f32>) {
        let elements = kind.block_elements() as usize;
        let mut packed = Vec::new();
        let mut decoded = Vec::new();
        for position in 0..columns / elements {
            let (bytes, values) = block(kind, seed + position);
            packed.extend(bytes);
            decoded.extend(values);
        }
        (packed, decoded)
    }

    fn fixture(kinds: [TensorType; 3], hidden: usize, ff: usize, experts: usize) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "kuno-cpu-packed-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut payload = Vec::new();
        let mut offsets = [0u64; 3];
        let mut expected: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::new());
        let columns = [hidden, hidden, ff];
        let rows = [ff, ff, hidden];
        for part in 0..3 {
            payload.resize(payload.len().next_multiple_of(32), 0);
            offsets[part] = payload.len() as u64;
            for expert in 0..experts {
                for row in 0..rows[part] {
                    let (bytes, values) =
                        encoded_row(kinds[part], columns[part], part * 23 + expert * 31 + row);
                    payload.extend(bytes);
                    expected[part].extend(values);
                }
            }
        }
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        for (part, name) in ["gate", "up", "down"].into_iter().enumerate() {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(3u32.to_le_bytes());
            for dimension in [columns[part], rows[part], experts] {
                bytes.extend((dimension as u64).to_le_bytes());
            }
            bytes.extend(kinds[part].gguf_id().to_le_bytes());
            bytes.extend(offsets[part].to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.extend(payload);
        std::fs::write(&path, bytes).unwrap();
        let model = GgufFile::open(&path, GgufLimits::default()).unwrap();
        Fixture {
            path,
            model,
            expected,
            hidden,
            ff,
            experts,
        }
    }

    fn sparse_f32_fixture(hidden: usize, ff: usize, experts: usize) -> Fixture {
        let path = std::env::temp_dir().join(format!(
            "kuno-cpu-packed-sparse-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let tensor_bytes = (hidden * ff * experts * 4) as u64;
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(3u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        for (part, name) in ["gate", "up", "down"].into_iter().enumerate() {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(3u32.to_le_bytes());
            let (columns, rows) = if part == 2 {
                (ff, hidden)
            } else {
                (hidden, ff)
            };
            for dimension in [columns, rows, experts] {
                bytes.extend((dimension as u64).to_le_bytes());
            }
            bytes.extend(0u32.to_le_bytes());
            bytes.extend((part as u64 * tensor_bytes).to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        std::fs::write(&path, &bytes).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(bytes.len() as u64 + 3 * tensor_bytes)
            .unwrap();
        let model = GgufFile::open(&path, GgufLimits::default()).unwrap();
        Fixture {
            path,
            model,
            expected: std::array::from_fn(|_| Vec::new()),
            hidden,
            ff,
            experts,
        }
    }

    // Independently visits each lane's entire column sequence first, then the
    // original tree. This does not call reduce64 or either packed dot path.
    fn reference_dot(weights: &[f32], input: &[f32]) -> f32 {
        let mut lanes = [0f32; 64];
        for (lane, value) in lanes.iter_mut().enumerate() {
            for column in (lane..input.len()).step_by(64) {
                *value += weights[column] * input[column];
            }
        }
        for stride in [32, 16, 8, 4, 2, 1] {
            for lane in 0..stride {
                lanes[lane] += lanes[lane + stride];
            }
        }
        lanes[0]
    }
    fn expected_experts(f: &Fixture, selected: &[usize], input: &[f32]) -> Vec<f32> {
        let mut result = Vec::new();
        for &expert in selected {
            let mut act = vec![0f32; f.ff];
            for (row, target) in act.iter_mut().enumerate() {
                let first = (expert * f.ff + row) * f.hidden;
                let gate = reference_dot(&f.expected[0][first..first + f.hidden], input);
                let up = reference_dot(&f.expected[1][first..first + f.hidden], input);
                let sigmoid = if gate >= 0. {
                    1. / (1. + (-gate).exp())
                } else {
                    let e = gate.exp();
                    e / (1. + e)
                };
                *target = (gate * sigmoid) * up;
            }
            for row in 0..f.hidden {
                let first = (expert * f.hidden + row) * f.ff;
                result.push(reference_dot(&f.expected[2][first..first + f.ff], &act));
            }
        }
        result
    }
    fn set<'r, 'm>(reader: &'r PositionalReader<'m>, f: &'m Fixture) -> PackedExperts<'r, 'm> {
        PackedExperts::new(
            reader,
            f.model.tensor("gate").unwrap(),
            f.model.tensor("up").unwrap(),
            f.model.tensor("down").unwrap(),
        )
        .unwrap()
    }
    fn input(columns: usize) -> Vec<f32> {
        (0..columns)
            .map(|column| ((column * 17 % 37) as f32 - 18.) / 512.)
            .collect()
    }
    fn same_bits(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "fixed-public numeric index{index}"
            );
        }
    }

    fn assert_metadata_wiped(witness: &retained_metadata::Witness, bytes: usize) {
        assert_eq!(witness.calls.load(Ordering::SeqCst), 1);
        assert_eq!(witness.slot_bytes.load(Ordering::SeqCst), bytes);
        assert!(witness.slot_zero.load(Ordering::SeqCst));
        assert!(witness.header_zero.load(Ordering::SeqCst));
        assert!(witness.counts_zero.load(Ordering::SeqCst));
    }

    #[test]
    fn public_job_headers_wipe_full_capacity_on_drop_partial_error_and_unwind() {
        for mode in 0..3 {
            let witness = std::sync::Arc::new(retained_metadata::Witness::default());
            let attempt =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
                    let mut jobs = retained_metadata::Owner::<[Vec<u8>; 3]>::new(MAX_SELECTED)?
                        .observe(witness.clone());
                    assert_eq!(
                        jobs.metadata_bytes()?,
                        retained_metadata::Owner::<[Vec<u8>; 3]>::bytes_for(MAX_SELECTED)?
                    );
                    for rank in 0..2 {
                        jobs.push(std::array::from_fn(|_| Vec::new()))?;
                        for part in 0..3 {
                            // Original public payloads remain nonzero and use their
                            // normal Vec destructors; only retained headers wipe.
                            let payload = &mut jobs.get_mut(rank)?[part];
                            payload.try_reserve_exact(16).unwrap();
                            payload.resize(7 + part, (13 + rank + part) as u8);
                            if mode == 1 && rank == 1 && part == 0 {
                                return Err(Error::InvalidFormat("fixed-public partial fill"));
                            }
                        }
                    }
                    assert_eq!(jobs.get(1)?[2], vec![16; 9]);
                    if mode == 2 {
                        panic!("fixed-public owning metadata unwind");
                    }
                    Ok(())
                }));
            match mode {
                0 => assert!(matches!(attempt, Ok(Ok(())))),
                1 => assert!(matches!(attempt, Ok(Err(_)))),
                _ => assert!(attempt.is_err()),
            }
            // All ten slots, including eight never-live spare slots, and the
            // dead outer Vec header are inspected before their owners release.
            assert_metadata_wiped(&witness, MAX_SELECTED * size_of::<[Vec<u8>; 3]>());
        }
    }

    #[test]
    fn public_job_headers_wipe_after_actual_positional_read_error_and_cancel() {
        for cancel_after_load in [false, true] {
            let f = fixture([TensorType::F32; 3], 8, 8, 2);
            let reader = f.model.positional_reader().unwrap();
            let witness = std::sync::Arc::new(retained_metadata::Witness::default());
            let cancel = AtomicBool::new(false);
            let result = (|| -> Result<()> {
                let mut jobs = retained_metadata::Owner::<[Vec<u8>; 3]>::new(MAX_SELECTED)?
                    .observe(witness.clone());
                jobs.push(std::array::from_fn(|_| Vec::new()))?;
                let tensor = f.model.tensor("gate").unwrap();
                let offset = Zeroizing::new(8 * 8 * 4u64);
                let packed = &mut jobs.get_mut(0)?[0];
                packed.resize(*offset as usize, 0);
                reader.read_range(tensor, *offset, packed)?;
                assert!(packed.iter().any(|&byte| byte != 0));
                if cancel_after_load {
                    cancel.store(true, Ordering::Relaxed);
                    check_cancel(&cancel)?;
                } else {
                    std::fs::OpenOptions::new()
                        .write(true)
                        .open(&f.path)?
                        .set_len(64)?;
                    let second = &mut jobs.get_mut(0)?[1];
                    second.resize(*offset as usize, 91);
                    let result = reader.read_range(f.model.tensor("up").unwrap(), *offset, second);
                    assert!(second.iter().all(|&byte| byte == 0));
                    result?;
                }
                Ok(())
            })();
            assert!(result.is_err());
            assert_metadata_wiped(&witness, MAX_SELECTED * size_of::<[Vec<u8>; 3]>());
        }
    }

    #[test]
    fn consumed_scoped_handle_headers_wipe_only_after_all_consumers_join() {
        let alive = std::sync::Arc::new(AtomicUsize::new(0));
        let completed = AtomicUsize::new(0);
        let jobs_witness = std::sync::Arc::new(retained_metadata::Witness {
            live_workers: Some(alive.clone()),
            ..retained_metadata::Witness::default()
        });
        let handles_witness = std::sync::Arc::new(retained_metadata::Witness {
            live_workers: Some(alive.clone()),
            ..retained_metadata::Witness::default()
        });
        let mut jobs = retained_metadata::Owner::<[Vec<u8>; 3]>::new(MAX_SELECTED)
            .unwrap()
            .observe(jobs_witness.clone());
        jobs.push(std::array::from_fn(|_| Vec::new())).unwrap();
        jobs.get_mut(0).unwrap()[0].extend([17, 29, 41]);
        let mut failures = 0;
        std::thread::scope(|scope| {
            let mut handles = retained_metadata::Owner::new(MAX_WORKERS)
                .unwrap()
                .observe(handles_witness.clone());
            assert!(handles.metadata_bytes().unwrap() <= worker_handle_metadata_bytes().unwrap());
            for index in 0..4 {
                let jobs = &jobs;
                let alive = &alive;
                let completed = &completed;
                handles
                    .push(
                        std::thread::Builder::new()
                            .spawn_scoped(scope, move || -> Result<()> {
                                struct Live<'a>(&'a AtomicUsize);
                                impl Drop for Live<'_> {
                                    fn drop(&mut self) {
                                        self.0.fetch_sub(1, Ordering::SeqCst);
                                    }
                                }
                                alive.fetch_add(1, Ordering::SeqCst);
                                let _live = Live(alive);
                                assert_eq!(jobs.get(0)?[0], [17, 29, 41]);
                                completed.fetch_add(1, Ordering::SeqCst);
                                if index == 1 {
                                    return Err(Error::InvalidFormat("fixed-public joined error"));
                                }
                                if index == 2 {
                                    panic!("fixed-public joined unwind");
                                }
                                Ok(())
                            })
                            .unwrap(),
                    )
                    .unwrap();
            }
            while let Some(handle) = handles.take_first() {
                if !matches!(handle.join(), Ok(Ok(()))) {
                    failures += 1;
                }
            }
            assert!(handles.get(0).is_err());
            assert_eq!(alive.load(Ordering::SeqCst), 0);
        });
        assert_eq!(failures, 2);
        assert_eq!(completed.load(Ordering::SeqCst), 4);
        assert_metadata_wiped(
            &handles_witness,
            MAX_WORKERS * size_of::<std::thread::ScopedJoinHandle<'_, Result<()>>>(),
        );
        assert!(handles_witness.joined.load(Ordering::SeqCst));
        drop(jobs);
        assert_metadata_wiped(&jobs_witness, MAX_SELECTED * size_of::<[Vec<u8>; 3]>());
        assert!(jobs_witness.joined.load(Ordering::SeqCst));
    }

    #[test]
    fn actual_fixed_header_capacities_are_admitted_before_private_input_fill() {
        let jobs = retained_metadata::Owner::<[Vec<u8>; 3]>::new(MAX_SELECTED).unwrap();
        let header_bytes = jobs.metadata_bytes().unwrap() + worker_handle_metadata_bytes().unwrap();
        assert!(header_bytes < 4096);
        assert!(retained_metadata::Owner::<[Vec<u8>; 3]>::new(0).is_err());
        assert!(retained_metadata::Owner::<[Vec<u8>; 3]>::new(MAX_WORKERS + 1).is_err());
        assert!(matches!(
            retained_metadata::Owner::<[Vec<u8>; 3]>::bytes_for(usize::MAX),
            Err(Error::ArithmeticOverflow)
        ));
        assert!(jobs.get(0).is_err());
        assert!(
            checked_sum(&[MAX_PRIVATE_HOST_BYTES, header_bytes]).unwrap() > MAX_PRIVATE_HOST_BYTES
        );
    }

    #[test]
    fn all_seven_original_formats_match_independent64_lane_expectations() {
        let cancel = AtomicBool::new(false);
        for kind in KINDS {
            let f = fixture([kind; 3], 256, 256, 3);
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let x = input(256);
            let expected = expected_experts(&f, &[2, 0], &x);
            for backend in [Backend::Portable, Backend::Avx2] {
                if backend == Backend::Avx2 && !avx2_available() {
                    continue;
                }
                let op = ExpertOperator::new(WorkerPlan::portable(3).unwrap(), backend).unwrap();
                let got = op.execute(&weights, &[2, 0], &x, &cancel).unwrap();
                assert_eq!(got.ranks(), 2);
                assert_eq!(got.hidden(), 256);
                assert_eq!(weights.hidden(), 256);
                assert_eq!(weights.feed_forward(), 256);
                same_bits(got.active(), &expected);
                assert_eq!(got.full().len(), MAX_SELECTED * 256);
                assert!(got.full()[512..].iter().all(|v| v.to_bits() == 0));
                assert!(got.owned_bytes() <= MAX_PRIVATE_HOST_BYTES);
            }
        }
    }

    #[test]
    fn actual_q4_gate_up_and_both_down_formats_preserve_all_ten_rank_order() {
        let cancel = AtomicBool::new(false);
        for down in [TensorType::Q5_0, TensorType::Q8_0] {
            let f = fixture([TensorType::Q4K, TensorType::Q4K, down], 512, 256, 10);
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let x = input(512);
            let selected = [9, 3, 6, 0, 8, 2, 5, 1, 7, 4];
            for count in [1, 3, 5, 8, 10] {
                let expected = expected_experts(&f, &selected[..count], &x);
                for backend in [Backend::Portable, Backend::Avx2] {
                    if backend == Backend::Avx2 && !avx2_available() {
                        continue;
                    }
                    let op =
                        ExpertOperator::new(WorkerPlan::portable(4).unwrap(), backend).unwrap();
                    let got = op
                        .execute(&weights, &selected[..count], &x, &cancel)
                        .unwrap();
                    same_bits(got.active(), &expected);
                    assert!(
                        got.full()[count * 512..]
                            .iter()
                            .all(|value| value.to_bits() == 0)
                    );
                }
            }
        }
    }

    #[test]
    fn row_odd_tails_subnormals_signedzeros_and_no_reassociation_match_bits() {
        let cancel = AtomicBool::new(false);
        for kind in KINDS {
            let widths: &[usize] = match kind {
                TensorType::F32 | TensorType::F16 | TensorType::Bf16 => {
                    &[1, 3, 5, 63, 64, 65, 127, 129, 256]
                }
                TensorType::Q5_0 | TensorType::Q8_0 => &[32, 64, 96, 256],
                _ => &[256, 512],
            };
            for &width in widths {
                let (packed, decoded) = encoded_row(kind, width, 7);
                let x = input(width);
                let expected = reference_dot(&decoded, &x);
                for backend in [Backend::Portable, Backend::Avx2] {
                    if backend == Backend::Avx2 && !avx2_available() {
                        continue;
                    }
                    let got = dot_row(
                        backend,
                        kind,
                        &packed,
                        &x,
                        &mut Scratch::new(width).unwrap(),
                        &cancel,
                    )
                    .unwrap();
                    assert_eq!(got.to_bits(), expected.to_bits());
                }
            }
        }
        // A left-to-right sum is deliberately different from the required tree.
        let mut weights = vec![0.; 65];
        weights[0] = 1e20;
        weights[1] = 1.;
        weights[64] = -1e20;
        let x = vec![1.; 65];
        assert_ne!(reference_dot(&weights, &x), weights.iter().sum::<f32>());
        let packed: Vec<_> = weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        for backend in [Backend::Portable, Backend::Avx2] {
            if backend == Backend::Avx2 && !avx2_available() {
                continue;
            }
            assert_eq!(
                dot_row(
                    backend,
                    TensorType::F32,
                    &packed,
                    &x,
                    &mut Scratch::new(65).unwrap(),
                    &cancel
                )
                .unwrap()
                .to_bits(),
                reference_dot(&weights, &x).to_bits()
            );
        }
    }

    #[test]
    fn invalid_selection_input_and_public_budget_fail_before_execution() {
        let f = fixture([TensorType::F32; 3], 32, 32, 3);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let op = ExpertOperator::new(WorkerPlan::portable(2).unwrap(), Backend::Portable).unwrap();
        let cancel = AtomicBool::new(false);
        let x = input(32);
        for selected in [vec![], vec![0, 0], vec![3], vec![0; 11]] {
            assert!(op.execute(&weights, &selected, &x, &cancel).is_err());
        }
        assert!(op.execute(&weights, &[0], &x[..31], &cancel).is_err());
        let mut bad = x.clone();
        bad[5] = f32::NAN;
        assert!(op.execute(&weights, &[0], &bad, &cancel).is_err());
        cancel.store(true, Ordering::Relaxed);
        assert!(op.execute(&weights, &[0], &x, &cancel).is_err());
        assert!(WorkerPlan::portable(0).is_err());
        assert!(WorkerPlan::portable(17).is_err());
        assert!(row(&[], usize::MAX, 4).is_err());
        assert!(checked_sum(&[usize::MAX, 1]).is_err());
        assert_eq!(f.experts, 3);
    }

    #[test]
    fn actual_sparse_metadata_admission_rejects_large_jobs_before_any_payload_read() {
        let f = sparse_f32_fixture(2560, 640, 10);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let op = ExpertOperator::new(WorkerPlan::portable(16).unwrap(), Backend::Portable).unwrap();
        // A stale descriptor would make any actual payload read return FileChanged.
        // The public original-byte admission must reject even before that point.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&f.path)
            .unwrap()
            .set_len(64)
            .unwrap();
        let ids: Vec<_> = (0..10).collect();
        let error = op.execute(&weights, &ids, &vec![0.; 2560], &AtomicBool::new(false));
        assert!(matches!(
            error,
            Err(Error::LimitExceeded("CPU expert public packed capacity"))
        ));
    }

    #[test]
    fn caller_private_remainder_rejects_before_reads_and_preserves_original_math() {
        let f = fixture([TensorType::F32; 3], 32, 32, 3);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let op = ExpertOperator::new(WorkerPlan::portable(2).unwrap(), Backend::Portable).unwrap();
        let cancel = AtomicBool::new(false);
        let x = input(32);
        let original = op.execute(&weights, &[2, 0], &x, &cancel).unwrap();
        let bounded = op
            .execute_with_private_limit(
                &weights,
                &[2, 0],
                &x,
                &cancel,
                MAX_PRIVATE_HOST_BYTES - 32768,
            )
            .unwrap();
        same_bits(original.full(), bounded.full());
        for limit in [0, MAX_PRIVATE_HOST_BYTES + 1, usize::MAX] {
            assert!(matches!(
                op.execute_with_private_limit(&weights, &[2, 0], &x, &cancel, limit),
                Err(Error::LimitExceeded("CPU expert shared private remainder"))
            ));
        }
        // A positional payload read now fails, but an insufficient admitted
        // remainder must reject before any private input copy or model read.
        std::fs::OpenOptions::new()
            .write(true)
            .open(&f.path)
            .unwrap()
            .set_len(64)
            .unwrap();
        assert!(matches!(
            op.execute_with_private_limit(&weights, &[2, 0], &x, &cancel, 1),
            Err(Error::LimitExceeded("CPU expert private host capacity"))
        ));
    }

    #[test]
    fn callable_packed_projection_matches_independent_public_values_and_capacity() {
        for kind in KINDS {
            let f = fixture([kind; 3], 256, 256, 2);
            let reader = f.model.positional_reader().unwrap();
            let matrix = PackedProjection::new(&reader, f.model.tensor("gate").unwrap()).unwrap();
            assert_eq!(matrix.columns(), 256);
            assert_eq!(matrix.rows(), 256);
            assert_eq!(
                matrix.packed_bytes(),
                kind.row_bytes(256).unwrap() as usize * 256
            );
            let x = input(256);
            let expected: Vec<_> = (0..256)
                .map(|row| reference_dot(&f.expected[0][(256 + row) * 256..(257 + row) * 256], &x))
                .collect();
            for backend in [Backend::Portable, Backend::Avx2] {
                if backend == Backend::Avx2 && !avx2_available() {
                    continue;
                }
                let op = ExpertOperator::new(WorkerPlan::portable(4).unwrap(), backend).unwrap();
                assert_eq!(op.worker_count(), 4);
                assert!(op.backend() == backend);
                let got = op.project(&matrix, 1, &x, &AtomicBool::new(false)).unwrap();
                same_bits(got.values(), &expected);
                assert!(got.owned_bytes() <= MAX_PRIVATE_HOST_BYTES);
                assert!(op.project(&matrix, 2, &x, &AtomicBool::new(false)).is_err());
            }
        }
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn register_reduction_preserves_original_tree_bits_and_nonfinite_rejection() {
        if !avx2_available() {
            return;
        }
        let compare = |lanes: [f32; LANES]| {
            let mut original = lanes;
            let expected = reduce64(&mut original);
            let actual = avx::test_reduce(&lanes);
            if expected.is_finite() {
                assert_eq!(actual.unwrap().to_bits(), expected.to_bits());
            } else {
                assert!(actual.is_err());
            }
        };
        compare([0.; LANES]);
        compare([-0.; LANES]);
        compare([f32::from_bits(1); LANES]);
        compare([f32::MAX; LANES]);
        let mut order = [0.; LANES];
        order[0] = 1e20;
        order[16] = -1e20;
        order[32] = 1.;
        let mut scalar = order;
        assert_eq!(reduce64(&mut scalar), 0.);
        compare(order);
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for lane in 0..LANES {
                let mut values = [1.; LANES];
                values[lane] = bad;
                compare(values);
            }
        }
        let mut bits = 0x3d11_713du32;
        for _ in 0..2048 {
            let values = std::array::from_fn(|_| {
                bits = bits.wrapping_mul(1664525).wrapping_add(1013904223);
                // All finite exponents, including subnormals and magnitudes
                // that overflow intermediate adds. This also varies signs.
                let finite_bits = if bits & 0x7f80_0000 == 0x7f80_0000 {
                    bits ^ 0x0080_0000
                } else {
                    bits
                };
                f32::from_bits(finite_bits)
            });
            compare(values);
        }
    }

    #[test]
    fn direct_simd_packed_decode_preserves_round_points_for_varied_half_scales() {
        if !avx2_available() {
            return;
        }
        let scales = [0u16, 0x8000, 1, 0x83ff, 0x0400, 0x1401, 0xb801, 0x7bff];
        for kind in [TensorType::Q4K, TensorType::Q5_0, TensorType::Q8_0] {
            for seed in 0..16 {
                let (mut packed, _) = encoded_row(kind, 2560, seed);
                for (block_index, block) in packed
                    .chunks_exact_mut(kind.block_bytes() as usize)
                    .enumerate()
                {
                    block[..2].copy_from_slice(
                        &scales[(seed + block_index) % scales.len()].to_le_bytes(),
                    );
                    if kind == TensorType::Q4K {
                        block[2..4].copy_from_slice(
                            &scales[(seed + block_index + 3) % scales.len()].to_le_bytes(),
                        );
                    }
                }
                let x: Vec<_> = (0..2560)
                    .map(|i| ((i * 17 + seed * 31) % 257) as f32 / 129.0 - 1.0)
                    .collect();
                let mut scalar = Scratch::new(2560).unwrap();
                let mut vector = Scratch::new(2560).unwrap();
                let cancel = AtomicBool::new(false);
                let expected =
                    dot_row(Backend::Portable, kind, &packed, &x, &mut scalar, &cancel).unwrap();
                let actual =
                    dot_row(Backend::Avx2, kind, &packed, &x, &mut vector, &cancel).unwrap();
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "kind={kind:?} seed={seed}"
                );
                for bits in [0x7c00u16, 0xfc00, 0x7e01] {
                    packed[..2].copy_from_slice(&bits.to_le_bytes());
                    assert!(
                        dot_row(Backend::Avx2, kind, &packed, &x, &mut vector, &cancel).is_err()
                    );
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn hardware_half_pairs_match_all_finite_software_encodings_and_reject_nonfinite() {
        if !avx2_available() || !std::is_x86_feature_detected!("f16c") {
            return;
        }
        for bits in 0..=u16::MAX {
            let opposite = bits ^ 0x8000;
            let a = crate::quant::f16_to_f32(bits);
            let b = crate::quant::f16_to_f32(opposite);
            let actual = avx::test_half_pair(u32::from(bits) | (u32::from(opposite) << 16));
            if a.is_finite() && b.is_finite() {
                let (first, second) = actual.unwrap();
                assert_eq!(first.to_bits(), a.to_bits());
                assert_eq!(second.to_bits(), b.to_bits());
            } else {
                assert!(actual.is_err());
            }
        }
    }
    #[test]
    fn explicit_fma_original_experts_have_bounded_error_and_preserve_inactive_capacity() {
        if !fma_available() {
            return;
        }
        for down in [TensorType::Q5_0, TensorType::Q8_0] {
            let f = fixture([TensorType::Q4K, TensorType::Q4K, down], 256, 256, 3);
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let exact = ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Avx2)
                .unwrap();
            let fast =
                ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Avx2Fma)
                    .unwrap();
            let cancel = AtomicBool::new(false);
            for pattern in 0..4 {
                let mut x = input(256);
                for (i, v) in x.iter_mut().enumerate() {
                    *v *= ((i + pattern * 17) % 23) as f32 / 11. - 1.;
                }
                let expected = exact.execute(&weights, &[2, 0], &x, &cancel).unwrap();
                let actual = fast.execute(&weights, &[2, 0], &x, &cancel).unwrap();
                // Synthetic complete projections include cancellation near
                // zero. Bound absolute error by the whole output's infinity
                // norm, rather than dividing by each cancelled element.
                let norm = expected.full().iter().map(|v| v.abs()).fold(0f32, f32::max);
                for (&a, &e) in actual.full().iter().zip(expected.full()) {
                    assert!(
                        a.is_finite() && (a - e).abs() <= 5e-4 + 1e-5 * norm,
                        "public FMA error a={a} e={e} norm={norm}"
                    );
                }
                assert!(actual.full()[512..].iter().all(|v| v.to_bits() == 0));
            }
            cancel.store(true, Ordering::Release);
            assert!(
                fast.execute(&weights, &[2, 0], &input(256), &cancel)
                    .is_err()
            );
        }
    }
    #[test]
    fn explicit_fma_rows_match_independent_f64_dot_and_reject_bad_scales() {
        if !fma_available() {
            return;
        }
        for kind in [TensorType::Q4K, TensorType::Q5_0, TensorType::Q8_0] {
            for columns in [256usize, 640, 2560] {
                if !columns.is_multiple_of(kind.block_elements() as usize) {
                    continue;
                }
                for seed in 0..16 {
                    let (mut packed, decoded) = encoded_row(kind, columns, seed);
                    let x: Vec<_> = (0..columns)
                        .map(|i| ((i * 17 + seed * 31) % 257) as f32 / 129. - 1.)
                        .collect();
                    let pairs = decoded
                        .iter()
                        .zip(&x)
                        .map(|(&w, &v)| f64::from(w) * f64::from(v));
                    let reference = pairs.clone().sum::<f64>();
                    let l1 = pairs.map(f64::abs).sum::<f64>();
                    let cancel = AtomicBool::new(false);
                    let mut scratch = Scratch::new(columns).unwrap();
                    let a = dot_row(Backend::Avx2Fma, kind, &packed, &x, &mut scratch, &cancel)
                        .unwrap();
                    assert!((f64::from(a) - reference).abs() <= 1e-5 + 1e-5 * l1);
                    for bits in [0x7c00u16, 0xfc00, 0x7e01] {
                        packed[..2].copy_from_slice(&bits.to_le_bytes());
                        assert!(
                            dot_row(Backend::Avx2Fma, kind, &packed, &x, &mut scratch, &cancel)
                                .is_err()
                        );
                    }
                }
            }
        }
    }

    #[cfg(all(
        target_os = "linux",
        target_pointer_width = "64",
        not(target_arch = "s390x")
    ))]
    #[test]
    #[allow(unsafe_code)]
    fn selected_mapped_prefetch_preserves_results_counts_and_cancel_boundary() {
        let f = fixture(
            [TensorType::Q4K, TensorType::Q4K, TensorType::Q5_0],
            256,
            256,
            3,
        );
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let control =
            ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Portable)
                .unwrap();
        let candidate =
            ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Portable)
                .unwrap()
                .with_mapped_prefetch();
        let x = input(256);
        let cancel = AtomicBool::new(false);
        assert!(!control.mapped_prefetch());
        assert!(candidate.mapped_prefetch());
        assert!(candidate.execute(&weights, &[2, 0], &x, &cancel).is_err());
        assert_eq!(candidate.prefetch_calls(), 0);
        // SAFETY: test exclusively owns immutable temporary inode.
        unsafe {
            f.model.enable_mapped_weights().unwrap();
        }
        let expected = control.execute(&weights, &[2, 0], &x, &cancel).unwrap();
        let actual = candidate.execute(&weights, &[2, 0], &x, &cancel).unwrap();
        same_bits(actual.full(), expected.full());
        assert_eq!(candidate.prefetch_calls(), 6);
        assert_eq!(control.prefetch_calls(), 0);
        cancel.store(true, Ordering::Relaxed);
        assert!(candidate.execute(&weights, &[2, 0], &x, &cancel).is_err());
        assert_eq!(candidate.prefetch_calls(), 6);
    }

    #[test]
    #[allow(unsafe_code)]
    fn residual_complete_experts_bound_original_float_error_and_report_actual_passes() {
        if !avx2_available() {
            return;
        }
        for down in [TensorType::Q5_0, TensorType::Q8_0] {
            let f = fixture([TensorType::Q4K, TensorType::Q4K, down], 256, 256, 3);
            // SAFETY: test exclusively owns immutable fixture inode.
            unsafe {
                f.model.enable_mapped_weights().unwrap();
            }
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let control =
                ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Portable)
                    .unwrap();
            let candidate =
                ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Avx2)
                    .unwrap()
                    .with_q8_residual()
                    .unwrap();
            assert_eq!(
                candidate.q8_residual_passes(),
                if cfg!(feature = "cpu-q8-two-pass") {
                    2
                } else {
                    3
                }
            );
            let x = input(256);
            let cancel = AtomicBool::new(false);
            let expected = control.execute(&weights, &[2, 0], &x, &cancel).unwrap();
            let actual = candidate.execute(&weights, &[2, 0], &x, &cancel).unwrap();
            let norm = expected.full().iter().map(|v| v.abs()).fold(0f32, f32::max);
            for (&a, &b) in actual.full().iter().zip(expected.full()) {
                assert!((a - b).abs() <= 1e-3 + 1e-4 * norm);
            }
            cancel.store(true, Ordering::Relaxed);
            assert!(candidate.execute(&weights, &[2, 0], &x, &cancel).is_err());
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn mapped_expert_jobs_match_read_jobs_and_join_before_borrowed_headers_erase() {
        for down in [TensorType::Q5_0, TensorType::Q8_0] {
            let f = fixture([TensorType::Q4K, TensorType::Q4K, down], 256, 256, 3);
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let op = ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Avx2)
                .unwrap();
            let x = input(256);
            let cancel = AtomicBool::new(false);
            let expected = op.execute(&weights, &[2, 0], &x, &cancel).unwrap();
            // SAFETY: this fixture inode is not modified during any map borrow.
            unsafe {
                f.model.enable_mapped_weights().unwrap();
            }
            let actual = op.execute(&weights, &[2, 0], &x, &cancel).unwrap();
            same_bits(actual.full(), expected.full());
            let witness = std::sync::Arc::new(retained_metadata::Witness::default());
            {
                let mut jobs = retained_metadata::Owner::<[PackedJob<'_>; 3]>::new(MAX_SELECTED)
                    .unwrap()
                    .observe(witness.clone());
                jobs.push(std::array::from_fn(|_| {
                    PackedJob::Mapped(
                        reader
                            .mapped_range(f.model.tensor("gate").unwrap(), 0, 144)
                            .unwrap()
                            .unwrap(),
                    )
                }))
                .unwrap();
                std::thread::scope(|scope| {
                    let jobs = &jobs;
                    scope.spawn(move || assert_eq!(jobs.get(0).unwrap()[0].as_ref().len(), 144));
                });
            }
            assert_metadata_wiped(&witness, MAX_SELECTED * size_of::<[PackedJob<'_>; 3]>());
            cancel.store(true, Ordering::Relaxed);
            assert!(op.execute(&weights, &[2, 0], &x, &cancel).is_err());
            cancel.store(false, Ordering::Relaxed);
            same_bits(
                op.execute(&weights, &[2, 0], &x, &cancel).unwrap().full(),
                expected.full(),
            );
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn quantized_complete_experts_match_independent_approximate_oracle_and_clear_failed_calls() {
        if !avx2_available() {
            return;
        }
        fn oracle(weights: &[f32], values: &[f32]) -> f32 {
            let mut result = 0f64;
            for (w, x) in weights.chunks_exact(32).zip(values.chunks_exact(32)) {
                let scale = x.iter().map(|v| v.abs()).fold(0f32, f32::max) / 127.;
                for (&weight, &value) in w.iter().zip(x) {
                    let q = if scale > 0. {
                        (value / scale).round_ties_even().clamp(-127., 127.)
                    } else {
                        0.
                    };
                    result += f64::from(weight) * f64::from(q) * f64::from(scale);
                }
            }
            result as f32
        }
        for down in [TensorType::Q5_0, TensorType::Q8_0] {
            let f = fixture([TensorType::Q4K, TensorType::Q4K, down], 256, 256, 2);
            // SAFETY: this test exclusively owns the immutable temporary inode.
            unsafe {
                f.model.enable_mapped_weights().unwrap();
            }
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            let values = input(256);
            let op = ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Avx2)
                .unwrap()
                .with_q8_activations()
                .unwrap();
            assert!(op.q8_activations());
            let cancel = AtomicBool::new(false);
            let mut expected = Vec::new();
            for expert in [1, 0] {
                let mut activations = vec![0.; 256];
                for (row, value) in activations.iter_mut().enumerate() {
                    let first = (expert * 256 + row) * 256;
                    let gate = oracle(&f.expected[0][first..first + 256], &values);
                    let up = oracle(&f.expected[1][first..first + 256], &values);
                    *value = crate::ops::silu(gate) * up;
                }
                for row in 0..256 {
                    let first = (expert * 256 + row) * 256;
                    expected.push(oracle(&f.expected[2][first..first + 256], &activations));
                }
            }
            let first = op.execute(&weights, &[1, 0], &values, &cancel).unwrap();
            for (&actual, &reference) in first.active().iter().zip(&expected) {
                assert!((actual - reference).abs() <= 0.01 + reference.abs() * 0.0005);
            }
            assert!(first.full()[512..].iter().all(|v| v.to_bits() == 0));
            assert!(
                op.execute_with(&weights, &[1, 0], &values, &cancel, |_| panic!(
                    "fixed public panic"
                ))
                .is_err()
            );
            assert!(
                op.execute_with(&weights, &[1, 0], &values, &cancel, |_| cancel
                    .store(true, Ordering::Release))
                    .is_err()
            );
            cancel.store(false, Ordering::Release);
            let fresh = op.execute(&weights, &[1, 0], &values, &cancel).unwrap();
            same_bits(first.full(), fresh.full());
        }
        assert!(
            ExpertOperator::new(WorkerPlan::portable(2).unwrap(), Backend::Avx2)
                .unwrap()
                .with_q8_activations()
                .is_err()
        );
        assert!(
            ExpertOperator::new_pooled(WorkerPlan::portable(2).unwrap(), Backend::Portable)
                .unwrap()
                .with_q8_activations()
                .is_err()
        );
        let witness = std::sync::Arc::new(retained_metadata::Witness::default());
        {
            let mut owner = retained_metadata::Owner::new(MAX_SELECTED + 1)
                .unwrap()
                .observe(witness.clone());
            owner.push(quantized::Input::empty()).unwrap();
            owner
                .get_mut(0)
                .unwrap()
                .fill(&[1.; 256], &AtomicBool::new(false))
                .unwrap();
        }
        assert_metadata_wiped(&witness, (MAX_SELECTED + 1) * size_of::<quantized::Input>());
    }

    #[test]
    fn foreign_original_triple_rejects_and_unchanged_fixture_still_works() {
        let a = fixture([TensorType::F32; 3], 8, 8, 1);
        let b = fixture([TensorType::F32; 3], 8, 8, 1);
        let reader = a.model.positional_reader().unwrap();
        assert!(
            PackedExperts::new(
                &reader,
                a.model.tensor("gate").unwrap(),
                b.model.tensor("up").unwrap(),
                a.model.tensor("down").unwrap()
            )
            .is_err()
        );
        let op = ExpertOperator::new(WorkerPlan::portable(1).unwrap(), Backend::Portable).unwrap();
        let got = op
            .execute(&set(&reader, &a), &[0], &input(8), &AtomicBool::new(false))
            .unwrap();
        same_bits(got.active(), &expected_experts(&a, &[0], &input(8)));
    }

    #[test]
    fn controlled_worker_panic_and_midwork_cancel_join_every_worker_and_clear_output() {
        let workers = WorkerPlan::portable(4).unwrap();
        for panic in [false, true] {
            let cancel = AtomicBool::new(false);
            let alive = AtomicUsize::new(0);
            let completed = AtomicUsize::new(0);
            struct Live<'a>(&'a AtomicUsize);
            impl Drop for Live<'_> {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let mut output = [7f32; 47];
            let hook = |row| {
                if row == 3 {
                    if panic {
                        panic!("fixed-public worker fault")
                    } else {
                        cancel.store(true, Ordering::Relaxed);
                    }
                }
            };
            let result = parallel_rows(&workers, &mut output, 65, &cancel, &hook, |row, _| {
                alive.fetch_add(1, Ordering::SeqCst);
                let _live = Live(&alive);
                completed.fetch_add(1, Ordering::SeqCst);
                Ok(row as f32)
            });
            assert!(result.is_err());
            assert_eq!(alive.load(Ordering::SeqCst), 0);
            assert!(output.iter().all(|value| value.to_bits() == 0));
            assert!(completed.load(Ordering::SeqCst) <= 47);
        }
    }

    #[test]
    fn expert_panic_cancel_and_nonfinite_never_return_partial_output() {
        let f = fixture([TensorType::F32; 3], 8, 8, 2);
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let op = ExpertOperator::new(WorkerPlan::portable(3).unwrap(), Backend::Portable).unwrap();
        let cancel = AtomicBool::new(false);
        assert!(
            op.execute_with(&weights, &[1, 0], &input(8), &cancel, |row| {
                if row == 2 {
                    panic!("fixed-public stage fault")
                }
            })
            .is_err()
        );
        assert!(
            op.execute_with(&weights, &[1, 0], &input(8), &cancel, |row| {
                if row == 2 {
                    cancel.store(true, Ordering::Relaxed)
                }
            })
            .is_err()
        );
        let mut packed = vec![0u8; 8 * 4];
        packed[4..8].copy_from_slice(&f32::NAN.to_le_bytes());
        for backend in [Backend::Portable, Backend::Avx2] {
            if backend == Backend::Avx2 && !avx2_available() {
                continue;
            }
            assert!(
                dot_row(
                    backend,
                    TensorType::F32,
                    &packed,
                    &[1.; 8],
                    &mut Scratch::new(8).unwrap(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
            assert!(
                dot_row(
                    backend,
                    TensorType::F32,
                    &packed[..4],
                    &[1.; 8],
                    &mut Scratch::new(8).unwrap(),
                    &AtomicBool::new(false)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn pooled_original_formats_repeat_without_changing_bits_or_rank_order() {
        for kind in KINDS {
            let f = fixture([kind; 3], 256, 256, 3);
            let reader = f.model.positional_reader().unwrap();
            let weights = set(&reader, &f);
            for backend in [Backend::Portable, Backend::Avx2] {
                if backend == Backend::Avx2 && !avx2_available() {
                    continue;
                }
                let operator =
                    ExpertOperator::new_pooled(WorkerPlan::portable(3).unwrap(), backend).unwrap();
                for selected in [&[2, 0][..], &[1][..], &[0, 2, 1][..]] {
                    let got = operator
                        .execute(&weights, selected, &input(256), &AtomicBool::new(false))
                        .unwrap();
                    same_bits(got.active(), &expected_experts(&f, selected, &input(256)));
                    assert!(
                        got.full()[selected.len() * 256..]
                            .iter()
                            .all(|v| v.to_bits() == 0)
                    );
                }
            }
        }
    }

    #[test]
    fn pooled_failure_joins_clears_and_leaves_no_request_for_next_call() {
        let op = ExpertOperator::new_pooled(WorkerPlan::portable(4).unwrap(), Backend::Portable)
            .unwrap();
        let alive = AtomicUsize::new(0);
        struct Live<'a>(&'a AtomicUsize);
        impl Drop for Live<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }
        for fault in 0..3 {
            let cancel = AtomicBool::new(false);
            let mut output = [7.; 47];
            let result = op.rows(
                &mut output,
                65,
                &cancel,
                &|row| {
                    if row == 3 {
                        if fault == 0 {
                            panic!("fixed-public pool panic");
                        }
                        if fault == 1 {
                            cancel.store(true, Ordering::Relaxed);
                        }
                    }
                },
                |row, _| {
                    alive.fetch_add(1, Ordering::SeqCst);
                    let _live = Live(&alive);
                    if row == 3 && fault == 2 {
                        return Err(Error::InvalidFormat("fixed-public pool error"));
                    }
                    Ok(row as f32)
                },
            );
            assert!(result.is_err());
            assert_eq!(alive.load(Ordering::SeqCst), 0);
            assert!(output.iter().all(|v| v.to_bits() == 0));
            cancel.store(false, Ordering::Relaxed);
            op.rows(&mut output, 65, &cancel, &|_| {}, |row, _| Ok(row as f32))
                .unwrap();
            same_bits(&output, &(0..47).map(|row| row as f32).collect::<Vec<_>>());
        }
        let mut empty = [];
        assert!(
            op.rows(&mut empty, 65, &AtomicBool::new(false), &|_| {}, |_, _| Ok(
                0.
            ))
            .is_err()
        );
        let mut tiny = [9.];
        assert!(
            op.rows(&mut tiny, 65, &AtomicBool::new(true), &|_| {}, |_, _| Ok(
                0.
            ))
            .is_err()
        );
        assert_eq!(tiny, [0.]);
    }

    #[test]
    fn pooled_headers_erase_full_capacity_after_join_and_error() {
        for error in [false, true] {
            let witness = std::sync::Arc::new(retained_metadata::Witness::default());
            let mut output = [7.; 47];
            let mut owner = retained_metadata::Owner::new(MAX_WORKERS)
                .unwrap()
                .observe(witness.clone());
            owner
                .push(std::sync::Mutex::new(RowJob {
                    first: 17,
                    rows: &mut output,
                    result: Some(if error {
                        Err(Error::InvalidFormat("public error"))
                    } else {
                        Ok(())
                    }),
                }))
                .unwrap();
            assert!(owner.metadata_bytes().unwrap() <= worker_handle_metadata_bytes().unwrap());
            drop(owner);
            assert_metadata_wiped(
                &witness,
                MAX_WORKERS * size_of::<std::sync::Mutex<RowJob<'_>>>(),
            );
        }
    }

    #[test]
    fn concurrent_cpu_really_runs_with_parent_and_joins_all_failure_paths() {
        let f = fixture(
            [TensorType::Q4K, TensorType::Q4K, TensorType::Q8_0],
            256,
            256,
            2,
        );
        let reader = f.model.positional_reader().unwrap();
        let weights = set(&reader, &f);
        let op = ExpertOperator::new_pooled(WorkerPlan::portable(2).unwrap(), Backend::Portable)
            .unwrap();
        for failure in 0..5 {
            let cancel = AtomicBool::new(false);
            let started = AtomicBool::new(false);
            let release = AtomicBool::new(false);
            let parent_done = AtomicBool::new(false);
            let result = op.execute_alongside_with(
                &weights,
                &[1, 0],
                &input(256),
                &cancel,
                MAX_PRIVATE_HOST_BYTES,
                || {
                    // A bounded public host-test synchronization only. No engine
                    // timeout or private request instrumentation is introduced.
                    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                    while !started.load(Ordering::Acquire) {
                        assert!(std::time::Instant::now() < deadline);
                        std::thread::yield_now();
                    }
                    release.store(true, Ordering::Release);
                    parent_done.store(true, Ordering::Release);
                    match failure {
                        1 => Err(Error::InvalidFormat("fixed-public parent error")),
                        2 => panic!("fixed-public parent panic"),
                        3 => {
                            cancel.store(true, Ordering::Relaxed);
                            Ok(17)
                        }
                        _ => Ok(17),
                    }
                },
                |row| {
                    if row == 0 {
                        started.store(true, Ordering::Release);
                        while !release.load(Ordering::Acquire) {
                            std::thread::yield_now();
                        }
                        if failure == 4 {
                            panic!("fixed-public CPU panic");
                        }
                    }
                },
            );
            assert!(parent_done.load(Ordering::Acquire));
            if failure == 0 {
                let (got, parent) = result.unwrap();
                assert_eq!(parent, 17);
                same_bits(got.active(), &expected_experts(&f, &[1, 0], &input(256)));
            } else {
                assert!(result.is_err());
            }
            // The reusable pool has no abandoned callback/request after error.
            cancel.store(false, Ordering::Relaxed);
            let got = op.execute(&weights, &[0], &input(256), &cancel).unwrap();
            same_bits(got.active(), &expected_experts(&f, &[0], &input(256)));
        }
        let scoped =
            ExpertOperator::new(WorkerPlan::portable(2).unwrap(), Backend::Portable).unwrap();
        for (operator, limit) in [
            (&scoped, MAX_PRIVATE_HOST_BYTES),
            (&op, 32768),
            (&op, MAX_PRIVATE_HOST_BYTES + 1),
        ] {
            assert!(
                operator
                    .execute_with_private_limit_alongside(
                        &weights,
                        &[0],
                        &input(256),
                        &AtomicBool::new(false),
                        limit,
                        || -> Result<()> { panic!("must reject before parent work") }
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn pooled_workers_reuse_fixed_threads_and_preserve_parent_affinity() {
        let Ok(plan) = WorkerPlan::physical(2) else {
            return;
        };
        let before = affinity::test_allowed().unwrap();
        let op = ExpertOperator::new_pooled(plan, Backend::Portable).unwrap();
        let mut original = None;
        for _ in 0..8 {
            let ids = std::sync::Mutex::new(std::collections::HashSet::new());
            op.rows(
                &mut [0.; 32],
                65,
                &AtomicBool::new(false),
                &|_| {
                    ids.lock().unwrap().insert(std::thread::current().id());
                },
                |row, _| Ok(row as f32),
            )
            .unwrap();
            let ids = ids.into_inner().unwrap();
            assert_eq!(ids.len(), 2);
            if let Some(expected) = &original {
                assert_eq!(&ids, expected);
            } else {
                original = Some(ids);
            }
            assert_eq!(affinity::test_allowed().unwrap(), before);
        }
    }

    #[test]
    #[ignore = "fixed-public scheduling-only timing; not expert or model throughput"]
    fn public_worker_scheduling_calibration() {
        for pooled in [false, true, true, false] {
            let plan = WorkerPlan::physical(16).unwrap();
            let op = if pooled {
                ExpertOperator::new_pooled(plan, Backend::Avx2)
            } else {
                ExpertOperator::new(plan, Backend::Avx2)
            }
            .unwrap();
            let cancel = AtomicBool::new(false);
            let mut output = [0.; 256];
            let begin = std::time::Instant::now();
            for _ in 0..64 {
                for _ in 0..2 {
                    op.rows(&mut output, 2560, &cancel, &|_| {}, |row, _| Ok(row as f32))
                        .unwrap();
                }
            }
            println!(
                "public_scheduling_only pooled={pooled} calls=128 workers=16 elapsed_us={}",
                begin.elapsed().as_micros()
            );
            assert!(
                output
                    .iter()
                    .enumerate()
                    .all(|(row, value)| *value == row as f32)
            );
        }
    }

    #[test]
    fn real_physical_worker_plan_is_scoped_and_never_changes_parent_affinity() {
        #[cfg(target_os = "linux")]
        {
            // Missing constrained-container topology is an explicit unsupported
            // plan, never silently replaced by an SMT/unpinned worker plan.
            let Ok(plan) = WorkerPlan::physical(1) else {
                return;
            };
            let before = affinity::test_allowed().unwrap();
            assert_eq!(plan.count(), 1);
            let f = fixture([TensorType::F32; 3], 8, 8, 1);
            let reader = f.model.positional_reader().unwrap();
            let op = ExpertOperator::new(plan, Backend::Portable).unwrap();
            let got = op
                .execute(&set(&reader, &f), &[0], &input(8), &AtomicBool::new(false))
                .unwrap();
            same_bits(got.active(), &expected_experts(&f, &[0], &input(8)));
            // Discovery still sees the parent's original allowed set after the
            // worker exited; pinning happened only in those scoped threads.
            assert_eq!(affinity::test_allowed().unwrap(), before);
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod affinity {
    use super::*;
    pub(super) fn physical_cores(_: usize) -> Result<[usize; MAX_WORKERS]> {
        Err(Error::InvalidFormat(
            "CPU expert physical cores require Linux",
        ))
    }
    pub(super) fn pin(_: usize) -> Result<()> {
        Err(Error::InvalidFormat(
            "CPU expert physical cores require Linux",
        ))
    }
}
