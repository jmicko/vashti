//! Owning I/O for the future actual MAX8 model coordinator. No state commit,
//! scalar helper change, default selection, source cache or additional MAIN.
//! Every completed map drops its raw lease before the actual admission's ONE
//! mandatory audited cleanup, ordered source release and MAIN fence credit.
#![allow(dead_code)] // Native whole-model coordinator/device acceptance pending.
use super::*;
#[cfg(test)]
use crate::executor::checked_resident_io_audit;
use crate::{
    quant_matvec::matmul::MAX_BATCH, resident_batch_projection::validate_view,
    resident_gr::batch::full_private,
};
use cubecl::server::Handle;
#[path = "resident_count1_io.rs"]
mod count1;
pub(crate) use count1::{Count1RouterInput, read_count1_router_input};
#[cfg(test)]
use cubecl_runtime::transient::TransientIoReceipt;
#[cfg(test)]
use kuno_text::TensorType;

const EMBEDDING: usize = 2560;
const CHANNELS: usize = 4 * EMBEDDING;
const EXPERTS: usize = 512;
const VOCABULARY: usize = 248320;
const SOURCE_BYTES: usize = 524288;
const ROUTER_BYTES: usize = 32768;
const OUTPUT_BYTES: usize = 1048576;
const EMBEDDING_HOST_CAP: usize = 768 * 1024;
const ROUTER_HOST_CAP: usize = 64 * 1024;
const OUTPUT_HOST_CAP: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) struct InputView {
    pub(crate) first: usize,
    pub(crate) stride: usize,
    pub(crate) words: usize,
}
impl Zeroize for InputView {
    fn zeroize(&mut self) {
        self.first.zeroize();
        self.stride.zeroize();
        self.words.zeroize();
    }
}
#[derive(Clone, Copy)]
pub(crate) struct RouterView {
    pub(crate) first: usize,
    pub(crate) stride: usize,
    pub(crate) words: usize,
}
impl Zeroize for RouterView {
    fn zeroize(&mut self) {
        self.first.zeroize();
        self.stride.zeroize();
        self.words.zeroize();
    }
}
fn error() -> Error {
    Error::InvalidFormat("native batch owning IO")
}
fn guarded<T>(admission: &ResidentAdmission, f: impl FnOnce() -> Result<T>) -> Result<T> {
    guarded_poison(&admission.sequence_poison(), f)
}
fn guarded_poison<T>(poison: &AtomicBool, f: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        _ => {
            poison.store(true, Ordering::Release);
            Err(error())
        }
    }
}
fn active(arena: &PrivateArena, admission: &ResidentAdmission, cancel: &AtomicBool) -> Result<()> {
    u::cancel(cancel)?;
    if !admission.transient_io()
        || admission.sequence_poison().load(Ordering::Acquire)
        || admission.retired_context().load(Ordering::Acquire)
    {
        return Err(error());
    }
    admission.check_client(arena.client().map_err(u::fault)?)
}
fn shape(graph: BoundGraph<'_, '_>) -> Result<()> {
    graph.verify_unchanged()?;
    let g = graph.geometry();
    if g.residual.width != EMBEDDING
        || g.residual.streams != 4
        || g.vocabulary != VOCABULARY
        || g.experts != EXPERTS
        || g.experts_used != 10
        || g.layers != 48
    {
        return Err(error());
    }
    Ok(())
}
fn packet(
    count: usize,
    start: usize,
    context: usize,
    tokens: &[u32; MAX_BATCH],
    positions: &[u32; MAX_BATCH],
) -> Result<()> {
    if !(1..=MAX_BATCH).contains(&count)
        || context > 4096
        || start.checked_add(count).is_none_or(|n| n > context)
        || (0..count)
            .any(|t| tokens[t] as usize >= VOCABULARY || positions[t] as usize != start + t)
        || tokens[count..]
            .iter()
            .chain(&positions[count..])
            .any(|&x| x != 0)
    {
        return Err(error());
    }
    Ok(())
}
fn physical<const N: usize>(
    arena: &PrivateArena,
    ids: [BufferId; N],
    words: [usize; N],
    bytes: [usize; N],
) -> Result<[Handle; N]> {
    if ids.iter().enumerate().any(|(i, id)| ids[..i].contains(id)) {
        return Err(error());
    }
    let mut handles: [Option<Handle>; N] = std::array::from_fn(|_| None);
    let mut resources: [Option<_>; N] = std::array::from_fn(|_| None);
    let client = arena.client().map_err(u::fault)?;
    for i in 0..N {
        let h = full_private(arena, ids[i], words[i])?;
        if h.size() != bytes[i] as u64
            || (i > 0
                && handles[0]
                    .as_ref()
                    .is_none_or(|first| first.stream != h.stream))
        {
            return Err(error());
        }
        resources[i] = Some(client.get_resource(h.clone()).map_err(u::fault)?);
        handles[i] = Some(h);
    }
    for i in 0..N {
        let p = resources[i].as_ref().ok_or(error())?.resource();
        if p.offset != 0
            || p.size != bytes[i] as u64
            || p.buffer.size() != p.size
            || resources[..i]
                .iter()
                .flatten()
                .any(|r| r.resource().buffer == p.buffer)
        {
            return Err(error());
        }
    }
    Ok(handles.map(|h| h.expect("validated owning handle")))
}
fn source(bytes: usize) -> Result<Zeroizing<Vec<u8>>> {
    let mut source = Zeroizing::new(Vec::new());
    source.try_reserve_exact(bytes).map_err(|_| error())?;
    if source.capacity() != bytes {
        return Err(error());
    }
    source.resize(bytes, 0);
    Ok(source)
}
fn fill_embedding(
    source: &mut [u8],
    view: InputView,
    count: usize,
    cancel: &AtomicBool,
    mut row: impl FnMut(usize, &mut [f32; EMBEDDING]) -> Result<()>,
) -> Result<()> {
    if !(1..=MAX_BATCH).contains(&count) {
        return Err(error());
    }
    validate_view(MAX_BATCH, view.first, view.stride, CHANNELS, view.words)?;
    if source.len() != SOURCE_BYTES
        || BufferLayout::new(view.words).map_err(u::fault)?.owned_bytes != SOURCE_BYTES
    {
        return Err(error());
    }
    let mut decoded = Zeroizing::new([0f32; EMBEDDING]);
    for t in 0..count {
        u::cancel(cancel)?;
        row(t, &mut decoded)?;
        if decoded.iter().any(|x| !x.is_finite()) {
            return Err(error());
        }
        for stream in 0..4 {
            for (d, &value) in decoded.iter().enumerate() {
                let at = (view.first + t * view.stride + stream * EMBEDDING + d) * 4;
                source[at..at + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            }
        }
    }
    u::cancel(cancel)
}
/// Destination is already retained in the admitted actual Model arena before
/// any source read/copy. This queues only; no fence/cleanup/inference deadline.
#[allow(clippy::too_many_arguments)]
pub(crate) fn enqueue_embedding(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    admission: &ResidentAdmission,
    destination: BufferId,
    view: InputView,
    count: usize,
    start: usize,
    context: usize,
    tokens: &Zeroizing<[u32; MAX_BATCH]>,
    positions: &Zeroizing<[u32; MAX_BATCH]>,
    cancel: &AtomicBool,
) -> Result<()> {
    guarded(admission, || {
        active(arena, admission, cancel)?;
        shape(graph)?;
        packet(count, start, context, tokens, positions)?;
        let view = Zeroizing::new(view);
        validate_view(MAX_BATCH, view.first, view.stride, CHANNELS, view.words)?;
        let [handle] = physical(arena, [destination], [view.words], [SOURCE_BYTES])?;
        let tensor = graph.tensor("token_embd.weight")?;
        if tensor.info().row_len() != EMBEDDING as u64
            || tensor.info().row_count() != VOCABULARY as u64
        {
            return Err(error());
        }
        let row_bytes = usize::try_from(tensor.info().row_bytes()).map_err(|_| error())?;
        if row_bytes == 0
            || row_bytes > EMBEDDING * 4
            || tensor.info().dtype().row_bytes(EMBEDDING as u64)? as usize != row_bytes
        {
            return Err(error());
        }
        let mut packed = Zeroizing::new([0u8; EMBEDDING * 4]);
        let mut upload = source(SOURCE_BYTES)?;
        if upload.capacity() + std::mem::size_of_val(&*packed) + EMBEDDING * 4 + 4096
            > EMBEDDING_HOST_CAP
        {
            return Err(error());
        }
        fill_embedding(&mut upload, *view, count, cancel, |t, decoded| {
            tensor.read_row(u64::from(tokens[t]), &mut packed[..row_bytes])?;
            tensor.info().dtype().decode(&packed[..row_bytes], decoded)
        })?;
        graph.verify_unchanged()?;
        active(arena, admission, cancel)?;
        arena
            .client()
            .map_err(u::fault)?
            .write_private(handle, std::mem::take(&mut *upload))
            .map_err(u::fault)?;
        u::cancel(cancel)
    })
}

#[cube(launch)]
fn router_packet(
    input: &Array<f32>,
    fault: &Array<u32>,
    output: &mut Array<u32>,
    count: usize,
    first: usize,
    stride: usize,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i < output.len() {
        if i < MAX_BATCH * EXPERTS {
            let token = i / EXPERTS;
            let column = i % EXPERTS;
            if token < count {
                output[i] = u32::reinterpret(input[first + token * stride + column]);
            } else {
                output[i] = 0;
            }
        } else if i == MAX_BATCH * EXPERTS {
            output[i] = fault[0];
        } else {
            output[i] = 0;
        }
    }
}
#[cube(launch)]
fn output_packet(input: &Array<f32>, fault: &Array<u32>, output: &mut Array<u32>, width: usize) {
    let i = ABSOLUTE_POS_X as usize;
    if i < output.len() {
        if i < width {
            output[i] = u32::reinterpret(input[i]);
        } else if i == width {
            output[i] = fault[0];
        } else {
            output[i] = 0;
        }
    }
}
struct Bytes(cubecl::bytes::Bytes);
impl AsRef<[u8]> for Bytes {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}
impl Drop for Bytes {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
/// This exact production producer is also driven by fake host leases/tests.
/// Decode failures release lease/guard and never reach audit/checkpoint/credit.
fn completed_map<L: AsRef<[u8]>, T>(
    read: impl FnOnce() -> Result<L>,
    decode: impl FnOnce(&[u8]) -> Result<T>,
) -> Result<T> {
    let result = {
        let lease = read()?;
        decode(lease.as_ref())?
    };
    Ok(result)
}
fn read(arena: &PrivateArena, handle: Handle, retired: &AtomicBool) -> Result<Bytes> {
    arena
        .client()
        .map_err(u::fault)?
        .read_one(handle)
        .map(Bytes)
        .map_err(|_| {
            retired.store(true, Ordering::Release);
            error()
        })
}
fn word(bytes: &[u8], i: usize) -> Result<u32> {
    let start = i.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
    let end = start.checked_add(4).ok_or(Error::ArithmeticOverflow)?;
    Ok(u32::from_le_bytes(
        bytes
            .get(start..end)
            .ok_or(error())?
            .try_into()
            .map_err(|_| error())?,
    ))
}
pub(crate) struct RouterLogits {
    rows: Box<Zeroizing<[[f32; EXPERTS]; MAX_BATCH]>>,
    count: Zeroizing<usize>,
}
impl RouterLogits {
    fn zero(count: usize) -> Result<Self> {
        if !(1..=MAX_BATCH).contains(&count)
            || std::mem::size_of::<Self>() + MAX_BATCH * EXPERTS * 4 + ROUTER_BYTES + 4096
                > ROUTER_HOST_CAP
        {
            return Err(error());
        }
        // Heap guard initialized zero BEFORE read/fill. Filled rows never move
        // inline on return: only this owning Box pointer is transferred.
        Ok(Self {
            rows: Box::new(Zeroizing::new([[0.; EXPERTS]; MAX_BATCH])),
            count: Zeroizing::new(count),
        })
    }
    pub(crate) fn row(&self, t: usize) -> Result<&[f32; EXPERTS]> {
        if t >= *self.count {
            return Err(error());
        }
        Ok(&self.rows[t])
    }
    fn accept(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() != ROUTER_BYTES || word(bytes, MAX_BATCH * EXPERTS)? != 0 {
            return Err(error());
        }
        for i in 0..bytes.len() / 4 {
            let bits = word(bytes, i)?;
            if i < *self.count * EXPERTS {
                let value = f32::from_bits(bits);
                if !value.is_finite() {
                    return Err(error());
                }
                self.rows[i / EXPERTS][i % EXPERTS] = value;
            } else if i != MAX_BATCH * EXPERTS && bits != 0 {
                return Err(error());
            }
        }
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_router(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    admission: &ResidentAdmission,
    router: BufferId,
    packet: BufferId,
    fault: BufferId,
    view: RouterView,
    count: usize,
    cancel: &AtomicBool,
) -> Result<RouterLogits> {
    guarded(admission, || {
        active(arena, admission, cancel)?;
        shape(graph)?;
        if !(1..=MAX_BATCH).contains(&count) {
            return Err(error());
        }
        let view = Zeroizing::new(view);
        validate_view(MAX_BATCH, view.first, view.stride, EXPERTS, view.words)?;
        let [input, packet_handle, fault_handle] = physical(
            arena,
            [router, packet, fault],
            [view.words, MAX_BATCH * EXPERTS + 1, 1],
            [ROUTER_BYTES, ROUTER_BYTES, ROUTER_BYTES],
        )?;
        let client = arena.client().map_err(u::fault)?;
        router_packet::launch::<WgpuRuntime>(
            client,
            u::grid(ROUTER_BYTES / 4)?,
            CubeDim::new_1d(64),
            u::raw_arg(input, view.words)?,
            u::raw_arg(fault_handle, 1)?,
            u::raw_arg(packet_handle.clone(), ROUTER_BYTES / 4)?,
            count,
            view.first,
            view.stride,
        );
        let retired = admission.retired_context();
        // The admission owns mandatory audit even under token-only cadence.
        admission.readback_audited(cancel, || {
            let mut out = RouterLogits::zero(count)?;
            completed_map(
                || read(arena, packet_handle, &retired),
                |bytes| {
                    out.accept(bytes)?;
                    Ok(out)
                },
            )
        })
    })
}
fn accept_output(bytes: &[u8], width: usize) -> Result<Zeroizing<Vec<f32>>> {
    if width != VOCABULARY || bytes.len() != OUTPUT_BYTES || word(bytes, width)? != 0 {
        return Err(error());
    }
    for i in 0..bytes.len() / 4 {
        let bits = word(bytes, i)?;
        if (i < width && !f32::from_bits(bits).is_finite()) || (i > width && bits != 0) {
            return Err(error());
        }
    }
    let mut out = Zeroizing::new(Vec::new());
    out.try_reserve_exact(width).map_err(|_| error())?;
    if out.capacity() != width || out.capacity() * 4 + bytes.len() > OUTPUT_HOST_CAP {
        return Err(error());
    }
    for i in 0..width {
        out.push(f32::from_bits(word(bytes, i)?));
    }
    Ok(out)
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn read_last_logits(
    graph: BoundGraph<'_, '_>,
    arena: &PrivateArena,
    admission: &ResidentAdmission,
    logits: BufferId,
    packet: BufferId,
    fault: BufferId,
    cancel: &AtomicBool,
) -> Result<Zeroizing<Vec<f32>>> {
    guarded(admission, || {
        active(arena, admission, cancel)?;
        shape(graph)?;
        let [input, packet_handle, fault_handle] = physical(
            arena,
            [logits, packet, fault],
            [VOCABULARY, VOCABULARY + 1, 1],
            [OUTPUT_BYTES, OUTPUT_BYTES, ROUTER_BYTES],
        )?;
        output_packet::launch::<WgpuRuntime>(
            arena.client().map_err(u::fault)?,
            u::grid(OUTPUT_BYTES / 4)?,
            CubeDim::new_1d(64),
            u::raw_arg(input, VOCABULARY)?,
            u::raw_arg(fault_handle, 1)?,
            u::raw_arg(packet_handle.clone(), OUTPUT_BYTES / 4)?,
            VOCABULARY,
        );
        let retired = admission.retired_context();
        admission.readback_audited(cancel, || {
            completed_map(
                || read(arena, packet_handle, &retired),
                |bytes| accept_output(bytes, VOCABULARY),
            )
        })
    })
}
fn accept_fault(bytes: &[u8]) -> Result<()> {
    if bytes.len() != ROUTER_BYTES || bytes.chunks_exact(4).any(|b| b != [0; 4]) {
        return Err(error());
    }
    Ok(())
}
/// Used for Advance first proof and second final proof AFTER PLE maps/leases.
pub(crate) fn read_fault(
    arena: &PrivateArena,
    admission: &ResidentAdmission,
    fault: BufferId,
    cancel: &AtomicBool,
) -> Result<()> {
    guarded(admission, || {
        active(arena, admission, cancel)?;
        let [handle] = physical(arena, [fault], [1], [ROUTER_BYTES])?;
        let retired = admission.retired_context();
        admission.readback_audited(cancel, || {
            completed_map(|| read(arena, handle, &retired), accept_fault)
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;
    fn set(bytes: &mut [u8], i: usize, bits: u32) {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&bits.to_le_bytes());
    }
    #[test]
    fn bounded_embedding_producer_keeps_every_hc_bit_inactive_gap_and_owned_tail() {
        for count in [1, 3, 5, 8] {
            let view = InputView {
                first: 3,
                stride: CHANNELS + 11,
                words: 3 + MAX_BATCH * (CHANNELS + 11) + 7,
            };
            let mut out = source(SOURCE_BYTES).unwrap();
            fill_embedding(&mut out, view, count, &AtomicBool::new(false), |t, row| {
                for (d, x) in row.iter_mut().enumerate() {
                    *x = if d.is_multiple_of(19) {
                        -0.
                    } else {
                        (t * 17 + d % 37) as f32 / 41.
                    };
                }
                Ok(())
            })
            .unwrap();
            for i in 0..out.len() / 4 {
                let active = i >= view.first
                    && (i - view.first) / view.stride < count
                    && (i - view.first) % view.stride < CHANNELS;
                if active {
                    let t = (i - view.first) / view.stride;
                    let d = (i - view.first) % view.stride % EMBEDDING;
                    let expected = if d.is_multiple_of(19) {
                        -0f32
                    } else {
                        (t * 17 + d % 37) as f32 / 41.
                    };
                    assert_eq!(word(&out, i).unwrap(), expected.to_bits());
                } else {
                    assert_eq!(word(&out, i).unwrap(), 0);
                }
            }
        }
        let mut out = source(SOURCE_BYTES).unwrap();
        let bad = InputView {
            first: usize::MAX,
            stride: usize::MAX,
            words: 1,
        };
        assert!(
            fill_embedding(&mut out, bad, 1, &AtomicBool::new(false), |_, _| panic!(
                "public unexpected row"
            ))
            .is_err()
        );
        let view = InputView {
            first: 0,
            stride: CHANNELS,
            words: 8 * CHANNELS,
        };
        let cancel = AtomicBool::new(false);
        assert!(
            fill_embedding(&mut out, view, 3, &cancel, |t, row| {
                row.fill(0.25);
                if t == 1 {
                    cancel.store(true, Ordering::Release);
                }
                Ok(())
            })
            .is_err()
        );
        assert!(source(usize::MAX).is_err());
    }
    #[test]
    fn original_all_seven_row_codecs_are_bounded_and_preserve_decoded_bits() {
        for dtype in [
            TensorType::F32,
            TensorType::F16,
            TensorType::Bf16,
            TensorType::Q5_0,
            TensorType::Q8_0,
            TensorType::Q4K,
            TensorType::Q6K,
        ] {
            let n = dtype.row_bytes(EMBEDDING as u64).unwrap() as usize;
            assert!(n <= EMBEDDING * 4);
            let packed = Zeroizing::new(vec![0u8; n]);
            let mut expected = Zeroizing::new([0f32; EMBEDDING]);
            dtype.decode(&packed, &mut expected[..]).unwrap();
            let mut out = source(SOURCE_BYTES).unwrap();
            let view = InputView {
                first: 0,
                stride: CHANNELS,
                words: 8 * CHANNELS,
            };
            fill_embedding(&mut out, view, 1, &AtomicBool::new(false), |_, row| {
                dtype.decode(&packed, row)
            })
            .unwrap();
            for stream in 0..4 {
                for (d, x) in expected.iter().enumerate() {
                    assert_eq!(word(&out, stream * EMBEDDING + d).unwrap(), x.to_bits());
                }
            }
        }
    }
    #[test]
    fn boxed_router_rows_keep_their_heap_address_and_reject_every_bad_extent_region() {
        for count in [1, 3, 5, 8] {
            let mut bytes = Zeroizing::new(vec![0; ROUTER_BYTES]);
            for i in 0..count * EXPERTS {
                set(
                    &mut bytes,
                    i,
                    if i % 7 == 0 {
                        (-0f32).to_bits()
                    } else {
                        ((i % 31) as f32 / 37.).to_bits()
                    },
                );
            }
            let mut rows = RouterLogits::zero(count).unwrap();
            let address = rows.rows.as_ptr();
            rows.accept(&bytes).unwrap();
            let moved = rows;
            assert_eq!(address, moved.rows.as_ptr());
            assert!(moved.row(count).is_err());
            for t in 0..count {
                for (d, x) in moved.row(t).unwrap().iter().enumerate() {
                    assert_eq!(x.to_bits(), word(&bytes, t * EXPERTS + d).unwrap());
                }
            }
            for index in [MAX_BATCH * EXPERTS, ROUTER_BYTES / 4 - 1] {
                set(&mut bytes, index, 1);
                assert!(RouterLogits::zero(count).unwrap().accept(&bytes).is_err());
                set(&mut bytes, index, 0);
            }
            if count < 8 {
                set(&mut bytes, count * EXPERTS, (-0f32).to_bits());
                assert!(RouterLogits::zero(count).unwrap().accept(&bytes).is_err());
                set(&mut bytes, count * EXPERTS, 0);
            }
            set(&mut bytes, 0, f32::NAN.to_bits());
            assert!(RouterLogits::zero(count).unwrap().accept(&bytes).is_err());
            set(&mut bytes, 0, 0);
            assert!(
                RouterLogits::zero(count)
                    .unwrap()
                    .accept(&bytes[..bytes.len() - 4])
                    .is_err()
            );
            bytes.extend([0; 4]);
            assert!(RouterLogits::zero(count).unwrap().accept(&bytes).is_err());
        }
    }
    #[test]
    fn output_and_fault_require_full_owner_padding_and_return_only_guarded_exact_capacity() {
        let mut bytes = Zeroizing::new(vec![0; OUTPUT_BYTES]);
        set(&mut bytes, 0, (-0f32).to_bits());
        set(&mut bytes, VOCABULARY - 1, 0.25f32.to_bits());
        let out = accept_output(&bytes, VOCABULARY).unwrap();
        assert_eq!(out.capacity(), VOCABULARY);
        assert_eq!(out[0].to_bits(), (-0f32).to_bits());
        assert_eq!(out[VOCABULARY - 1], 0.25);
        assert!(out.capacity() * 4 + OUTPUT_BYTES <= OUTPUT_HOST_CAP);
        for (i, bits) in [
            (VOCABULARY, 2),
            (OUTPUT_BYTES / 4 - 1, (-0f32).to_bits()),
            (17, f32::INFINITY.to_bits()),
        ] {
            set(&mut bytes, i, bits);
            assert!(accept_output(&bytes, VOCABULARY).is_err());
            set(&mut bytes, i, 0);
        }
        assert!(accept_output(&bytes[..bytes.len() - 4], VOCABULARY).is_err());
        bytes.extend([0; 4]);
        assert!(accept_output(&bytes, VOCABULARY).is_err());
        let mut fault = Zeroizing::new(vec![0; ROUTER_BYTES]);
        assert!(accept_fault(&fault).is_ok());
        for i in [0, ROUTER_BYTES / 4 - 1] {
            set(&mut fault, i, 1);
            assert!(accept_fault(&fault).is_err());
            set(&mut fault, i, 0);
        }
        assert!(accept_fault(&fault[..fault.len() - 1]).is_err());
    }
    struct Lease {
        bytes: Vec<u8>,
        events: Rc<RefCell<Vec<&'static str>>>,
    }
    impl AsRef<[u8]> for Lease {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            self.bytes.zeroize();
            self.events.borrow_mut().push("drop");
        }
    }
    #[test]
    fn actual_completed_map_producer_drops_lease_before_admission_and_no_error_can_return() {
        for failure in 0..3 {
            let events = Rc::new(RefCell::new(Vec::new()));
            let result = completed_map(
                || {
                    events.borrow_mut().push("read");
                    if failure == 1 {
                        return Err(error());
                    }
                    Ok(Lease {
                        bytes: vec![0; 4],
                        events: events.clone(),
                    })
                },
                |_| {
                    events.borrow_mut().push("decode");
                    if failure == 2 {
                        return Err(error());
                    }
                    Ok(Box::new(Zeroizing::new([17f32; 16])))
                },
            );
            assert_eq!(result.is_ok(), failure == 0);
            let expected: &[&str] = if failure == 1 {
                &["read"]
            } else {
                &["read", "decode", "drop"]
            };
            assert_eq!(&*events.borrow(), expected);
        }
    }
    #[test]
    fn production_error_panic_and_cancel_guard_set_the_same_sticky_poison() {
        let poison = Arc::new(AtomicBool::new(false));
        assert_eq!(guarded_poison(&poison, || Ok(19)).unwrap(), 19);
        assert!(!poison.load(Ordering::Acquire));
        let alias = poison.clone();
        assert!(guarded_poison::<()>(&poison, || Err(error())).is_err());
        assert!(alias.load(Ordering::Acquire));
        assert!(guarded_poison::<()>(&poison, || panic!("fixed public injected panic")).is_err());
        assert!(alias.load(Ordering::Acquire));
        let cancel = AtomicBool::new(true);
        assert!(guarded_poison(&poison, || u::cancel(&cancel)).is_err());
        assert!(alias.load(Ordering::Acquire));
        // Success cannot clear a previously failed whole-sequence owner.
        guarded_poison(&poison, || Ok(())).unwrap();
        assert!(alias.load(Ordering::Acquire));
    }
    #[test]
    fn exact_guarded_sources_and_router_heap_wipe_the_entire_retained_capacity() {
        let mut upload = source(SOURCE_BYTES).unwrap();
        upload.fill(0xa7);
        let capacity = upload.capacity();
        upload.zeroize();
        assert!(upload.is_empty());
        assert_eq!(upload.capacity(), capacity);
        // The zeroize Vec implementation writes all retained allocation bytes;
        // u8 has no invalid representations and the owner stays allocated.
        let retained = unsafe { std::slice::from_raw_parts(upload.as_ptr(), capacity) };
        assert!(retained.iter().all(|&x| x == 0));
        let mut rows = RouterLogits::zero(3).unwrap();
        for row in rows.rows.iter_mut() {
            row.fill(0.75);
        }
        let address = rows.rows.as_ptr();
        rows.rows.zeroize();
        assert_eq!(address, rows.rows.as_ptr());
        assert!(rows.rows.iter().flatten().all(|x| x.to_bits() == 0));
        assert!(word(&[], usize::MAX).is_err());
        assert!(word(&[], usize::MAX / 4).is_err());
    }
    #[test]
    fn production_audit_retirement_rejects_failed_partial_empty_and_overflow_receipts() {
        let good = TransientIoReceipt {
            epoch: 7,
            upload_bytes: 32_768,
            uniform_bytes: 512,
            readback_bytes: 32_768,
            inspected_bytes: 66_048,
            inspected_buffers: 4,
        };
        let retired = AtomicBool::new(false);
        checked_resident_io_audit(&retired, || Ok(good)).unwrap();
        assert!(!retired.load(Ordering::Acquire));
        for bad in [
            TransientIoReceipt {
                inspected_bytes: 66_047,
                ..good
            },
            TransientIoReceipt {
                inspected_buffers: 0,
                ..good
            },
            TransientIoReceipt {
                upload_bytes: 0,
                uniform_bytes: 0,
                readback_bytes: 0,
                inspected_bytes: 0,
                ..good
            },
            TransientIoReceipt {
                upload_bytes: u64::MAX,
                ..good
            },
        ] {
            let retired = AtomicBool::new(false);
            assert!(checked_resident_io_audit(&retired, || Ok(bad)).is_err());
            assert!(retired.load(Ordering::Acquire));
        }
        let retired = AtomicBool::new(false);
        assert!(checked_resident_io_audit(&retired, || Err(error())).is_err());
        assert!(retired.load(Ordering::Acquire));
        let retired = AtomicBool::new(false);
        assert!(
            checked_resident_io_audit(&retired, || panic!("fixed public audit panic")).is_err()
        );
        assert!(retired.load(Ordering::Acquire));
    }
    #[test]
    fn separate_runtime_component_and_host_caps_match_approved_coordinator() {
        assert_eq!(2 * (SOURCE_BYTES + 8 * 2 * 512), 1_064_960);
        assert_eq!(2 * (8 * 2 * 512 + ROUTER_BYTES), 81_920);
        assert_eq!(2 * ROUTER_BYTES, 65_536);
        assert_eq!(OUTPUT_BYTES + VOCABULARY * 4, 2_041_856);
        const { assert!(SOURCE_BYTES + 2 * EMBEDDING * 4 + 4096 < EMBEDDING_HOST_CAP) };
        assert!(
            ROUTER_BYTES + std::mem::size_of::<[[f32; EXPERTS]; MAX_BATCH]>() + 4096
                < ROUTER_HOST_CAP
        );
    }
    #[test]
    fn packet_overflow_bad_ids_inactive_payloads_and_cancel_reject_before_fill() {
        for count in [1, 3, 5, 8] {
            let mut t = [0; 8];
            let mut p = [0; 8];
            for i in 0..count {
                t[i] = (i + 1) as u32;
                p[i] = (4096 - count + i) as u32;
            }
            assert!(packet(count, 4096 - count, 4096, &t, &p).is_ok());
            p[0] += 1;
            assert!(packet(count, 4096 - count, 4096, &t, &p).is_err());
        }
        assert!(packet(9, 0, 4096, &[0; 8], &[0; 8]).is_err());
        assert!(packet(1, usize::MAX, usize::MAX, &[0; 8], &[0; 8]).is_err());
        assert!(
            packet(
                1,
                0,
                4096,
                &[VOCABULARY as u32, 0, 0, 0, 0, 0, 0, 0],
                &[0; 8]
            )
            .is_err()
        );
        assert!(packet(1, 0, 4096, &[0, 1, 0, 0, 0, 0, 0, 0], &[0; 8]).is_err());
    }
    #[test]
    fn packet_shaders_validate_offline_and_keep_counts_bases_strides_runtime() {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, FloatKind, StorageType, Type, UIntKind};
        use cubecl::server::ExecutionMode;
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
            for router in [false, true] {
                let mut b = KernelBuilder::default();
                let f = StorageType::Scalar(ElemType::Float(FloatKind::F32));
                let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
                b.scope.register_type::<f32>(f);
                b.scope.register_type::<u32>(u);
                b.scope.register_type::<usize>(u);
                let input = b.input_array(Type::new(f)).into();
                let fault = b.input_array(Type::new(u)).into();
                let output = b.output_array(Type::new(u)).into();
                let count = b.scalar(u).into();
                if router {
                    let first = b.scalar(u).into();
                    let stride = b.scalar(u).into();
                    router_packet::expand(&mut b.scope, input, fault, output, count, first, stride);
                } else {
                    output_packet::expand(&mut b.scope, input, fault, output, count);
                }
                let shader = WgslCompiler::default()
                    .compile(
                        b.build(
                            KernelSettings::default()
                                .cube_dim(CubeDim::new_1d(64))
                                .kernel_name("native_owning_packet"),
                        ),
                        &WgpuCompilationOptions::default(),
                        mode,
                        u,
                    )
                    .unwrap()
                    .to_string();
                assert!(shader.contains("bitcast<u32>"));
                let module = wgpu::naga::front::wgsl::parse_str(&shader).unwrap();
                wgpu::naga::valid::Validator::new(
                    wgpu::naga::valid::ValidationFlags::all(),
                    wgpu::naga::valid::Capabilities::empty(),
                )
                .validate(&module)
                .unwrap();
                assert!(shader.contains("scalars"));
            }
        }
    }
}
