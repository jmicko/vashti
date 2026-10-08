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
        let result = client.retained_resource_metadata(handles).map_err(fault)?;
        METADATA_CALLS
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::ArithmeticOverflow)?;
        Ok(result)
    } else {
        handles
            .into_iter()
            .map(|h| client.get_resource(h).map_err(fault))
            .collect()
    }
}
struct ScrubBytes(cubecl::bytes::Bytes);
impl Drop for ScrubBytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
use crate::private_arena::{BufferId, PrivateArena};

pub fn fault(_: impl std::fmt::Display) -> Error {
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
        CubeCount::Static(x, crate::quant_matvec::matmul::MAX_BATCH as u32, 1),
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
pub fn zero(arena: &PrivateArena, id: BufferId) -> Result<()> {
    let layout = arena.layout(id).map_err(fault)?;
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
