//! Buffer-only laboratory glue. All private allocations are arena-owned.
//! Numerical faults accumulate until the whole-token owner rejects/cleans up.
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use cubecl_runtime::storage::ManagedResource;
use kuno_text::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zeroize::{Zeroize, Zeroizing};
static METADATA_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static COMPUTE_SCRATCH_CLEAR: AtomicBool = AtomicBool::new(false);
static COMPUTE_SCRATCH_CLEAR_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) fn select_compute_scratch_clear(selected: bool) {
    COMPUTE_SCRATCH_CLEAR.store(selected, Ordering::Release);
}
pub(crate) fn compute_scratch_clear_enabled() -> bool {
    COMPUTE_SCRATCH_CLEAR.load(Ordering::Acquire)
}
pub(crate) fn compute_scratch_clear_calls() -> u64 {
    COMPUTE_SCRATCH_CLEAR_CALLS.load(Ordering::Relaxed)
}
pub(crate) fn grouped_scratch_clear_enabled() -> bool {
    native_scratch_clear_enabled() || compute_scratch_clear_enabled()
}
static NATIVE_SCRATCH_CLEAR: AtomicBool = AtomicBool::new(false);
static NATIVE_SCRATCH_CLEAR_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) fn select_native_scratch_clear(selected: bool) {
    NATIVE_SCRATCH_CLEAR.store(selected, Ordering::Release);
}
pub(crate) fn native_scratch_clear_enabled() -> bool {
    NATIVE_SCRATCH_CLEAR.load(Ordering::Acquire)
}
pub(crate) fn native_scratch_clear_calls() -> u64 {
    NATIVE_SCRATCH_CLEAR_CALLS.load(Ordering::Relaxed)
}
static RETAINED_METADATA: AtomicBool = AtomicBool::new(false);
pub(crate) fn select_retained_metadata(selected: bool) {
    RETAINED_METADATA.store(selected, Ordering::Release);
}
pub fn metadata_calls() -> u64 {
    METADATA_CALLS.load(Ordering::Relaxed)
}
/// Fixed-public experiment only. Ordinary callers preserve get_resource's
/// submission semantics. The selected API inspects FULL retained owners only;
/// subsequent extent/alias checks and every actual data fence stay unchanged.
pub fn resources(
    client: &ComputeClient<WgpuRuntime>,
    handles: Vec<Handle>,
) -> Result<Vec<ManagedResource<cubecl::wgpu::WgpuResource>>> {
    if RETAINED_METADATA.load(Ordering::Acquire) {
        retained_resources(client, handles)
    } else {
        handles
            .into_iter()
            .map(|h| client.get_resource(h).map_err(fault))
            .collect()
    }
}
/// Original global-source owners always require non-submitting retained
/// inspection, independently of the legacy lab's optional metadata policy.
pub(crate) fn retained_resources(
    client: &ComputeClient<WgpuRuntime>,
    handles: Vec<Handle>,
) -> Result<Vec<ManagedResource<cubecl::wgpu::WgpuResource>>> {
    let sliced = handles
        .iter()
        .filter(|h| h.offset_start.unwrap_or(0) != 0 || h.offset_end.unwrap_or(0) != 0)
        .count();
    let count = handles.len();
    retained_batches(handles, |batch| {
        let result = client.retained_resource_metadata(batch).map_err(|cause| {
            if crate::strata_port::enabled() {
                eprintln!("strata_metadata_failure count={count} sliced={sliced}");
                // PUBLIC fixtures only: static lifecycle diagnostics.
                eprintln!("strata_static_runtime_diagnostic={cause}");
                if let Err(launch) = client.flush() {
                    eprintln!("strata_static_launch_diagnostic={launch}");
                }
            }
            Error::InvalidFormat("resident retained metadata lookup rejected")
        })?;
        METADATA_CALLS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        Ok(result)
    })
}
/// The backend admits at most128 full bindings per query, not128 owners per
/// model. Retain ALL original owners/results across batches; the caller still
/// proves complete extents and cross-batch aliasing before accepting the model.
fn retained_batches<H: Clone, R>(
    handles: Vec<H>,
    mut query: impl FnMut(Vec<H>) -> Result<Vec<R>>,
) -> Result<Vec<R>> {
    if handles.is_empty() || handles.len() > 4096 {
        return Err(Error::LimitExceeded("resident full metadata inventory"));
    }
    let mut result = Vec::with_capacity(handles.len());
    for batch in handles.chunks(128) {
        let rows = query(batch.to_vec())?;
        if rows.len() != batch.len() {
            return Err(Error::InvalidFormat("resident metadata batch incomplete"));
        }
        result.extend(rows);
    }
    Ok(result)
}

struct ScrubBytes(cubecl::bytes::Bytes);
impl Drop for ScrubBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
use crate::private_arena::{BufferId, PrivateArena};

// This flag has no environment or private-protocol control. Its only caller is
// the dedicated compiled-public, single-request diagnostic entry point.
static COMPILED_PUBLIC_DIAGNOSTICS: AtomicBool = AtomicBool::new(false);
pub(crate) fn enable_compiled_public_diagnostics() {
    COMPILED_PUBLIC_DIAGNOSTICS.store(true, Ordering::Release);
}
/// Public-diagnostic-only step timing: prints a static label and elapsed time.
pub(crate) fn public_diagnostics() -> bool {
    COMPILED_PUBLIC_DIAGNOSTICS.load(Ordering::Acquire)
}
pub(crate) fn public_step(label: &'static str, started: std::time::Instant) {
    if COMPILED_PUBLIC_DIAGNOSTICS.load(Ordering::Acquire) {
        eprintln!("public_load_step {label} {:.3}", started.elapsed().as_secs_f64());
    }
}
#[track_caller]
pub fn fault(cause: impl std::fmt::Display) -> Error {
    if COMPILED_PUBLIC_DIAGNOSTICS.load(Ordering::Acquire) {
        let at = std::panic::Location::caller();
        // One line: the public stderr filter drops continuation lines.
        eprintln!("public_device_error {}:{} {}", at.file(), at.line(), cause.to_string().replace('\n', " | "));
    }
    Error::InvalidFormat("resident graph device operation")
}
pub fn cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        Err(Error::InvalidFormat("resident graph cancelled"))
    } else {
        Ok(())
    }
}
pub fn words(arena: &PrivateArena, id: BufferId) -> Result<usize> {
    Ok(arena.layout(id).map_err(fault)?.logical_words)
}
pub fn handle(arena: &PrivateArena, id: BufferId) -> Result<Handle> {
    crate::resident_gdn::logical_handle(arena, id)
}
pub fn arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    // SAFETY: arena identity, epoch and exact logical bounds checked by handle.
    Ok(unsafe { ArrayArg::from_raw_parts(handle(arena, id)?, words(arena, id)?) })
}
pub fn raw_arg(handle: Handle, words: usize) -> Result<ArrayArg<WgpuRuntime>> {
    if handle.size_in_used() < (words.checked_mul(4).ok_or(Error::ArithmeticOverflow)?) as u64 {
        return Err(Error::InvalidBufferLength);
    }
    // SAFETY: caller owns public handle on same stream and verified element type.
    Ok(unsafe { ArrayArg::from_raw_parts(handle, words) })
}
pub fn grid(words: usize) -> Result<CubeCount> {
    if words == 0 || words.div_ceil(64) > 65535 {
        return Err(Error::LimitExceeded("resident kernel grid"));
    }
    Ok(CubeCount::Static(words.div_ceil(64) as u32, 1, 1))
}
#[allow(clippy::manual_range_contains)]
#[cube]
fn finite(x: f32) -> bool {
    x <= 3.4028235e38_f32 && x >= -3.4028235e38_f32
}
#[cube(launch)]
pub fn verify_finite(values: &Array<f32>, fault: &mut Array<Atomic<u32>>) {
    let i = ABSOLUTE_POS;
    if i < values.len() && !finite(values[i]) {
        fault[0].fetch_or(1);
    }
}
#[cube(launch)]
pub fn verify_status(status: &Array<u32>, fault: &mut Array<Atomic<u32>>) {
    let i = ABSOLUTE_POS;
    if i < status.len() && status[i] != 1 {
        fault[0].fetch_or(2);
    }
}
#[cube(launch)]
pub fn copy_words(input: &Array<u32>, output: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < output.len() {
        output[i] = input[i];
    }
}
#[cube(launch)]
pub fn fence_packet(values: &Array<f32>, fault: &Array<u32>, packet: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < values.len() {
        packet[i] = u32::reinterpret(values[i]);
    }
    if i == values.len() {
        packet[i] = fault[0];
    }
}
pub fn validate(arena: &PrivateArena, output: BufferId, fault_id: BufferId) -> Result<()> {
    if output == fault_id || words(arena, fault_id)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    // Strata's execution strategy checks real published outputs instead of
    // scanning every intermediate tensor. Physical span checks and complete
    // private-owner erasure are separate and remain mandatory.
    if crate::strata_port::enabled() {
        return Ok(());
    }
    verify_finite::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        grid(words(arena, output)?)?,
        CubeDim::new_1d(64),
        arg(arena, output)?,
        arg(arena, fault_id)?,
    );
    Ok(())
}
pub fn status(arena: &PrivateArena, region: BufferId, fault_id: BufferId) -> Result<()> {
    if region == fault_id || words(arena, fault_id)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    if crate::strata_port::enabled() {
        return Ok(());
    }
    verify_status::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        grid(words(arena, region)?)?,
        CubeDim::new_1d(64),
        arg(arena, region)?,
        arg(arena, fault_id)?,
    );
    Ok(())
}

/// Fold only initialized active token rows. Inactive MAX8 status words may be
/// zero or belong to earlier work; neither is evidence about this operation.
/// Same-stream dispatch precedes scratch reuse and joins the final model fault.
#[allow(clippy::collapsible_if)] // Cube DSL eagerly evaluates storage in AND operands.
#[cube(launch)]
fn verify_status_batch(
    status: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
    first: usize,
    stride: usize,
    width: usize,
) {
    let token = CUBE_POS_Y as usize;
    let element = ABSOLUTE_POS_X as usize;
    // Cube DSL lowers boolean operands eagerly. Keep the storage read INSIDE
    // the active-row branch so an inactive grid row never indexes a subregion.
    if token < count && element < width {
        if status[first + token * stride + element] != 1 {
            fault[0].fetch_or(2);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn status_batch(
    arena: &PrivateArena,
    region: BufferId,
    fault_id: BufferId,
    count: usize,
    first: usize,
    stride: usize,
    width: usize,
) -> Result<()> {
    if region == fault_id || words(arena, fault_id)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    if crate::strata_port::enabled() {
        return Ok(());
    }
    crate::resident_batch_projection::validate_view(
        count,
        first,
        stride,
        width,
        words(arena, region)?,
    )?;
    let CubeCount::Static(x, _, _) = grid(width)? else {
        return Err(Error::InvalidBufferLength);
    };
    verify_status_batch::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        CubeCount::Static(x, count as u32, 1),
        CubeDim::new_1d(64),
        arg(arena, region)?,
        arg(arena, fault_id)?,
        count,
        first,
        stride,
        width,
    );
    Ok(())
}
/// Decode windows skip per-layer scratch/status re-zeroing: every active row
/// is overwritten by its producer and fault words are cleared once per window.
/// Verified bit-identical (greedy outputs) against the clearing path.
/// Diagnostic host timers (content-free labels and durations), printed with
/// KUNO_CLAUDE_PIPE_TRACE at request end.
pub fn claude_time(label: &'static str, since: std::time::Instant) {
    static T: std::sync::OnceLock<std::sync::Mutex<std::collections::BTreeMap<&'static str, u128>>> = std::sync::OnceLock::new();
    *T.get_or_init(Default::default).lock().unwrap().entry(label).or_default() += since.elapsed().as_nanos();
    let _ = &CLAUDE_TIMES;
    CLAUDE_TIMES.get_or_init(|| T.get().unwrap());
}
pub static CLAUDE_TIMES: std::sync::OnceLock<&'static std::sync::Mutex<std::collections::BTreeMap<&'static str, u128>>> = std::sync::OnceLock::new();
pub fn decode_skips_layer_clears() -> bool {
    !crate::strata_prompt::active()
}
pub fn zero(arena: &PrivateArena, id: BufferId) -> Result<()> {
    let layout = arena.layout(id).map_err(fault)?;
    if native_scratch_clear_enabled() && layout.owned_bytes <= 64 * 1024 * 1024 {
        zero_batch(arena, &[id])?;
        NATIVE_SCRATCH_CLEAR_CALLS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        return Ok(());
    }
    crate::private_arena::erase_full_span::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        CubeCount::Static(layout.workgroups_x, layout.workgroups_y, 1),
        CubeDim::new_1d(64),
        unsafe {
            ArrayArg::from_raw_parts(arena.handle(id).map_err(fault)?.clone(), layout.owned_words)
        },
    );
    Ok(())
}
/// Batch only consecutive full-owner scratch clears at the existing stream
/// boundary. Public16-owner/64MiB limits remain; larger public spans retain
/// the original shader clear. Never changes final private arena erasure.
pub fn zero_many(arena: &PrivateArena, ids: &[BufferId]) -> Result<()> {
    for chunk in ids.chunks(if compute_scratch_clear_enabled() {
        4
    } else {
        16
    }) {
        let bytes = chunk.iter().try_fold(0usize, |n, &id| {
            n.checked_add(arena.layout(id).map_err(fault)?.owned_bytes)
                .ok_or(Error::ArithmeticOverflow)
        })?;
        if compute_scratch_clear_enabled() && chunk.len() >= 2 && bytes <= 64 * 1024 * 1024 {
            crate::scratch_clear_batch::launch(arena, chunk)?;
            COMPUTE_SCRATCH_CLEAR_CALLS
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| Error::ArithmeticOverflow)?;
        } else if native_scratch_clear_enabled() && bytes <= 64 * 1024 * 1024 {
            zero_batch(arena, chunk)?;
            NATIVE_SCRATCH_CLEAR_CALLS
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
                .map_err(|_| Error::ArithmeticOverflow)?;
        } else {
            for &id in chunk {
                zero(arena, id)?;
            }
        }
    }
    Ok(())
}
/// Ordered native clears of original full owning scratch pages. The parent
/// retains the arena through original completed-fence/full-zero acceptance.
pub fn zero_batch(arena: &PrivateArena, ids: &[BufferId]) -> Result<()> {
    if ids.is_empty() || ids.len() > 16 {
        return Err(Error::InvalidBufferLength);
    }
    let mut handles = Vec::with_capacity(ids.len());
    for &id in ids {
        let layout = arena.layout(id).map_err(fault)?;
        let handle = arena.handle(id).map_err(fault)?;
        if handle.size() != layout.owned_bytes as u64
            || handle.size_in_used() != layout.owned_bytes as u64
        {
            return Err(Error::InvalidBufferLength);
        }
        handles.push(handle.clone());
    }
    arena
        .client()
        .map_err(fault)?
        .clear_private_owners(handles)
        .map_err(fault)
}
/// Source allocations remain tracked until I/O arena reset after a fence.
pub fn upload_f32(
    io: &mut PrivateArena,
    arena: &PrivateArena,
    id: BufferId,
    values: &[f32],
) -> Result<()> {
    if values.len() != words(arena, id)? || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidBufferLength);
    }
    if !std::ptr::eq(
        io.client().map_err(fault)?.info(),
        arena.client().map_err(fault)?.info(),
    ) {
        return Err(Error::InvalidFormat("resident upload context mismatch"));
    }
    let source = io.upload_f32(values).map_err(fault)?;
    copy_words::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        grid(values.len())?,
        CubeDim::new_1d(64),
        arg(io, source)?,
        arg(arena, id)?,
    );
    Ok(())
}
pub fn read_f32(arena: &PrivateArena, id: BufferId) -> Result<Zeroizing<Vec<f32>>> {
    let bytes = ScrubBytes(
        arena
            .client()
            .map_err(fault)?
            .read_one(handle(arena, id)?)
            .map_err(fault)?,
    );
    let values = f32::from_bytes(&bytes.0);
    if values.len() != words(arena, id)? || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("resident output invalid"));
    }
    Ok(Zeroizing::new(values.to_vec()))
}
/// ONE readback drains prior commands and includes the accumulated fault word.
pub fn read_fence(
    arena: &PrivateArena,
    values: BufferId,
    fault_id: BufferId,
    packet: BufferId,
) -> Result<Zeroizing<Vec<f32>>> {
    let n = words(arena, values)?;
    if words(arena, packet)? != n + 1
        || words(arena, fault_id)? != 1
        || values == packet
        || fault_id == packet
    {
        return Err(Error::InvalidBufferLength);
    }
    fence_packet::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        grid(n + 1)?,
        CubeDim::new_1d(64),
        arg(arena, values)?,
        arg(arena, fault_id)?,
        arg(arena, packet)?,
    );
    let bytes = ScrubBytes(
        arena
            .client()
            .map_err(fault)?
            .read_one(handle(arena, packet)?)
            .map_err(fault)?,
    );
    let packed = u32::from_bytes(&bytes.0);
    if packed.len() != n + 1 || packed[n] != 0 {
        return Err(Error::InvalidFormat("resident token numerical failure"));
    }
    let output: Vec<f32> = packed[..n].iter().map(|v| f32::from_bits(*v)).collect();
    let output = Zeroizing::new(output);
    if output.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("resident output invalid"));
    }
    Ok(output)
}

/// Accept one little-endian fault word. Packet width is fixed independently
/// of vocabulary/context and rejects malformed bytes before interpretation.
pub(crate) fn accept_fault_packet(bytes: &[u8]) -> Result<()> {
    let word: [u8; 4] = bytes.try_into().map_err(|_| Error::InvalidBufferLength)?;
    if u32::from_le_bytes(word) != 0 {
        return Err(Error::InvalidFormat("resident token numerical failure"));
    }
    Ok(())
}
fn validate_fault_words(logical_words: usize) -> Result<()> {
    if logical_words != 1 {
        return Err(Error::InvalidBufferLength);
    }
    Ok(())
}
/// This completed same-stream read drains the final combine/status commands.
/// Only the accumulated fault word is downloaded; the local Bytes lease drops
/// before return, preserving the owner's admission/sanitizer readback seam.
pub fn read_fault_fence(arena: &PrivateArena, fault_id: BufferId) -> Result<()> {
    validate_fault_words(words(arena, fault_id)?)?;
    let bytes = ScrubBytes(
        arena
            .client()
            .map_err(fault)?
            .read_one(handle(arena, fault_id)?)
            .map_err(fault)?,
    );
    accept_fault_packet(&bytes.0)
}

pub fn arg_prefix(arena: &PrivateArena, id: BufferId, n: usize) -> Result<ArrayArg<WgpuRuntime>> {
    let capacity = words(arena, id)?;
    if n == 0 || n > capacity {
        return Err(Error::InvalidBufferLength);
    }
    let h = handle(arena, id)?.offset_end(((capacity - n) * 4) as u64);
    Ok(unsafe { ArrayArg::from_raw_parts(h, n) })
}
pub fn status_prefix(
    arena: &PrivateArena,
    id: BufferId,
    n: usize,
    fault_id: BufferId,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if id == fault_id || words(arena, fault_id)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    verify_status::launch::<WgpuRuntime>(
        arena.client().map_err(fault)?,
        grid(n)?,
        CubeDim::new_1d(64),
        arg_prefix(arena, id, n)?,
        arg(arena, fault_id)?,
    );
    Ok(())
}
pub fn read_prefix_f32(
    arena: &PrivateArena,
    id: BufferId,
    n: usize,
) -> Result<Zeroizing<Vec<f32>>> {
    let capacity = words(arena, id)?;
    if n == 0 || n > capacity {
        return Err(Error::InvalidBufferLength);
    }
    let bytes = ScrubBytes(
        arena
            .client()
            .map_err(fault)?
            .read_one(handle(arena, id)?.offset_end(((capacity - n) * 4) as u64))
            .map_err(fault)?,
    );
    let values = f32::from_bytes(&bytes.0);
    if values.len() != n || values.iter().any(|v| !v.is_finite()) {
        return Err(Error::InvalidFormat("resident score invalid"));
    }
    Ok(Zeroizing::new(values.to_vec()))
}
pub fn read_prefix_u32(
    arena: &PrivateArena,
    id: BufferId,
    n: usize,
) -> Result<Zeroizing<Vec<u32>>> {
    read_handle_prefix_u32(arena, handle(arena, id)?, n)
}
/// Read an exact prefix of an already-attested private view. Preserve both
/// existing offsets when narrowing; no whole-owner or unrelated tail readback.
pub(crate) fn read_handle_prefix_u32(
    arena: &PrivateArena,
    view: Handle,
    n: usize,
) -> Result<Zeroizing<Vec<u32>>> {
    let capacity =
        usize::try_from(view.size_in_used() / 4).map_err(|_| Error::ArithmeticOverflow)?;
    if !view.size_in_used().is_multiple_of(4) {
        return Err(Error::InvalidBufferLength);
    }
    if n == 0 || n > capacity {
        return Err(Error::InvalidBufferLength);
    }
    let bytes = ScrubBytes(
        arena
            .client()
            .map_err(fault)?
            .read_one(view.offset_end(((capacity - n) * 4) as u64))
            .map_err(fault)?,
    );
    let values = u32::from_bytes(&bytes.0);
    if values.len() != n {
        return Err(Error::InvalidBufferLength);
    }
    Ok(Zeroizing::new(values.to_vec()))
}
pub struct TransientSelector<'a> {
    pub io: &'a mut PrivateArena,
}
impl crate::resident_qsa::ResidentQsaSelector for TransientSelector<'_> {
    fn select_transient(
        &mut self,
        arena: &PrivateArena,
        layout: crate::qsa_gpu::QsaLayout,
        scores: crate::resident_gdn::StatusRegion,
        status: crate::resident_gdn::StatusRegion,
        cancel: &AtomicBool,
    ) -> Result<crate::resident_qsa::PrivateSelection> {
        self::cancel(cancel)?;
        let values = read_prefix_f32(arena, scores.buffer, scores.words)?;
        let flags = read_prefix_u32(arena, status.buffer, status.words)?;
        self::cancel(cancel)?;
        crate::resident_qsa::PrivateSelection::from_scores(layout, &values, &flags)
    }
    fn upload_selected(
        &mut self,
        arena: &PrivateArena,
        target: BufferId,
        selection: &crate::resident_qsa::PrivateSelection,
        cancel: &AtomicBool,
    ) -> Result<()> {
        self::cancel(cancel)?;
        if !std::ptr::eq(
            self.io.client().map_err(fault)?.info(),
            arena.client().map_err(fault)?.info(),
        ) {
            return Err(Error::InvalidFormat("resident selection context mismatch"));
        }
        let mut padded = Zeroizing::new(vec![0_u32; words(arena, target)?]);
        if selection.ids().is_empty() || selection.ids().len() > padded.len() {
            return Err(Error::InvalidBufferLength);
        }
        padded[..selection.ids().len()].copy_from_slice(selection.ids());
        let source = self.io.upload_u32(&padded).map_err(fault)?;
        copy_words::launch::<WgpuRuntime>(
            arena.client().map_err(fault)?,
            grid(padded.len())?,
            CubeDim::new_1d(64),
            arg(self.io, source)?,
            arg(arena, target)?,
        );
        self::cancel(cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapped_word_erasure_covers_every_odd_unaligned_span_and_preserves_boundaries() {
        for offset in 0..16 {
            for length in 0..=240 {
                let mut data = [0xa5_u8; 257];
                // SAFETY: this initialized exclusive synthetic span stays live
                // for the whole call; all 16 alignments and byte tails are tested.
                unsafe {
                    cubecl_runtime::transient::wipe_mapped_full_span(
                        data.as_mut_ptr().add(offset),
                        length,
                    )
                };
                assert!(data[..offset].iter().all(|&x| x == 0xa5));
                assert!(data[offset..offset + length].iter().all(|&x| x == 0));
                assert!(data[offset + length..].iter().all(|&x| x == 0xa5));
            }
        }
    }
    #[test]
    fn retained_metadata_rejects_foreign_streams_offset_views_empty_and_growth() {
        use cubecl::stream_id::StreamId;
        use cubecl_runtime::transient::retained_metadata_bindings as validate;
        let stream = StreamId { value: 73 };
        let owner = || Handle::new(stream, 32768).binding();
        assert!(validate(&[owner()], stream).is_ok());
        assert!(validate(&(0..128).map(|_| owner()).collect::<Vec<_>>(), stream).is_ok());
        assert!(validate(&[], stream).is_err());
        assert!(validate(&(0..129).map(|_| owner()).collect::<Vec<_>>(), stream).is_err());
        assert!(
            validate(
                &[Handle::new(StreamId { value: 74 }, 32768).binding()],
                stream
            )
            .is_err()
        );
        assert!(validate(&[Handle::new(stream, 0).binding()], stream).is_err());
        for (first, last) in [(1, 0), (0, 1), (32768, 0), (0, u64::MAX)] {
            let h = Handle::new(stream, 32768)
                .offset_start(first)
                .offset_end(last);
            assert!(validate(&[h.binding()], stream).is_err());
        }
    }
    #[test]
    fn native_clear_rejects_slices_foreign_empty_unaligned_and_excess_owners() {
        use cubecl::stream_id::StreamId;
        use cubecl_runtime::transient::clear_owner_bindings as validate;
        let stream = StreamId { value: 73 };
        let owner = |size| Handle::new(stream, size).binding();
        assert!(validate(&[owner(32768)], stream).is_ok());
        assert!(validate(&[], stream).is_err());
        assert!(
            validate(
                &(0..16).map(|_| owner(4 * 1024 * 1024)).collect::<Vec<_>>(),
                stream
            )
            .is_ok()
        );
        for size in [0, 3, 64 * 1024 * 1024 + 4, u64::MAX] {
            assert!(validate(&[owner(size)], stream).is_err());
        }
        assert!(validate(&(0..17).map(|_| owner(4)).collect::<Vec<_>>(), stream).is_err());
        for (start, end) in [(Some(4), None), (None, Some(4))] {
            let mut b = owner(32768);
            b.offset_start = start;
            b.offset_end = end;
            assert!(validate(&[b], stream).is_err());
        }
        let mut b = owner(32768);
        b.stream = StreamId { value: 74 };
        assert!(validate(&[b], stream).is_err());
    }
    #[test]
    fn fault_only_fence_accepts_exact_zero_and_rejects_all_failure_bits() {
        assert!(accept_fault_packet(&[0; 4]).is_ok());
        for bit in 0..32 {
            assert!(accept_fault_packet(&(1_u32 << bit).to_le_bytes()).is_err());
        }
        assert!(accept_fault_packet(&u32::MAX.to_le_bytes()).is_err());
        // Interpret exactly four bytes without relying on host slice alignment.
        assert!(accept_fault_packet(&[0x7f, 0, 0, 0, 0][1..]).is_ok());
        assert!(accept_fault_packet(&[0x7f, 0, 0, 0, 1][1..]).is_err());
    }
    #[test]
    fn malformed_fault_extent_or_packet_is_never_accepted() {
        assert!(validate_fault_words(1).is_ok());
        for n in [0, 2, 256, usize::MAX] {
            assert!(validate_fault_words(n).is_err());
        }
        for n in [0, 1, 2, 3, 5, 8, 256] {
            assert!(accept_fault_packet(&vec![0; n]).is_err());
        }
    }
    #[test]
    fn portable_dispatch_bounds_are_checked() {
        assert!(grid(0).is_err());
        assert!(grid(65535 * 64 + 1).is_err());
        assert!(matches!(
            grid(65535 * 64).unwrap(),
            CubeCount::Static(65535, 1, 1)
        ));
    }
    #[test]
    fn atomic_validation_and_packet_bits_compile_checked_and_unchecked() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for variant in 0..5 {
                let mut b = KernelBuilder::default();
                b.scope
                    .register_type::<u32>(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));
                b.scope
                    .register_type::<usize>(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));
                b.scope
                    .register_type::<f32>(StorageType::Scalar(ElemType::Float(FloatKind::F32)));
                let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
                let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                let a = StorageType::Atomic(ElemType::UInt(UIntKind::U32));
                if variant == 0 {
                    let x = b.input_array(Type::new(f)).into();
                    let fault = b.output_array(Type::new(a)).into();
                    verify_finite::expand(&mut b.scope, x, fault);
                } else if variant == 1 {
                    let x = b.input_array(Type::new(u)).into();
                    let fault = b.output_array(Type::new(a)).into();
                    verify_status::expand(&mut b.scope, x, fault);
                } else if variant == 2 {
                    let x = b.input_array(Type::new(u)).into();
                    let y = b.output_array(Type::new(u)).into();
                    copy_words::expand(&mut b.scope, x, y);
                } else if variant == 3 {
                    let x = b.input_array(Type::new(f)).into();
                    let fault = b.input_array(Type::new(u)).into();
                    let y = b.output_array(Type::new(u)).into();
                    fence_packet::expand(&mut b.scope, x, fault, y);
                } else {
                    let x = b.input_array(Type::new(u)).into();
                    let fault = b.output_array(Type::new(a)).into();
                    let args: Vec<NativeExpand<usize>> =
                        (0..4).map(|_| b.scalar(u).into()).collect();
                    verify_status_batch::expand(
                        &mut b.scope,
                        x,
                        fault,
                        args[0].clone(),
                        args[1].clone(),
                        args[2].clone(),
                        args[3].clone(),
                    );
                }
                let source = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("resident_util"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        u,
                    )
                    .unwrap()
                    .to_string();
                assert!(source.contains("@compute"));
                if variant < 2 {
                    assert!(source.contains("atomicOr("));
                }
                if variant == 3 {
                    assert!(source.contains("bitcast<u32>"));
                }
                if variant == 4 {
                    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
                    wgpu::naga::valid::Validator::new(
                        wgpu::naga::valid::ValidationFlags::all(),
                        wgpu::naga::valid::Capabilities::empty(),
                    )
                    .validate(&module)
                    .unwrap();
                    assert!(source.contains("atomicOr("));
                    for index in 0..4 {
                        assert!(
                            source.contains(&format!("scalars_u32[{index}]")),
                            "{source}"
                        );
                    }
                    assert!(source.contains("global_id.x"), "{source}");
                    assert!(source.contains("workgroup_id.y"), "{source}");
                    let body = source.split("fn resident_util(").nth(1).unwrap();
                    // The storage load must not be hoisted before the active
                    // mask, especially for count1 variable-width QSA planes.
                    assert!(
                        body.find("if ").unwrap() < body.find("buffer_0_global[").unwrap(),
                        "{source}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod retained_batch_tests {
    use super::*;
    #[test]
    fn native_shared_metadata_preserves_full_inventory_across_backend_batch_limit() {
        for count in [1, 128, 129, 137, 379, 2048, 4096] {
            let owners: Vec<_> = (0..count).collect();
            let actual = retained_batches(owners.clone(), |batch| {
                assert!(!batch.is_empty() && batch.len() <= 128);
                Ok(batch)
            })
            .unwrap();
            assert_eq!(actual, owners);
        }
        assert!(retained_batches(Vec::<usize>::new(), Ok).is_err());
        assert!(retained_batches(vec![0; 4097], Ok).is_err());
        let mut calls = 0;
        let failure = retained_batches((0..379).collect::<Vec<_>>(), |mut batch| {
            calls += 1;
            if calls == 2 {
                batch.pop();
            }
            Ok(batch)
        });
        assert!(failure.is_err());
        assert_eq!(calls, 2);
        let mut calls = 0;
        assert!(
            retained_batches((0..379).collect::<Vec<_>>(), |batch| {
                calls += 1;
                if calls == 2 {
                    Err(Error::InvalidFormat("synthetic metadata failure"))
                } else {
                    Ok(batch)
                }
            })
            .is_err()
        );
        assert_eq!(calls, 2);
    }
}
