//! Connected request owner for the pinned Intel sampler and coupled MTP chain.
//! No global content cache: TLS is an exclusive, wiping, stack-scoped request.
use super::{HISTORY_MAX, Params, VOCAB, coupled_base, coupled_counter, penalty_rows};
use crate::{
    private_arena::{BufferId, BufferLayout, PrivateArena},
    resident_util as u,
};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::*,
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments},
};
use kuno_text::{Error, Result, model::GraphBindingId};
use std::{
    cell::RefCell,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zeroize::{Zeroize, Zeroizing};
// Full arena roots, not offset upload destinations. write_private requires the
// entire authenticated owner; min32KiB allocator padding is charged explicitly.
pub(crate) const ROOT_BYTES: [u64; 7] = [32768, 131072, 262144, 32768, 32768, 32768, 32768];
pub(crate) const MAIN_BYTES: u64 = 557056;
pub(crate) const HOST_BYTES: usize = 2 * 1024 * 1024;
// Source, destination and independent inspection allowances for one maximal
// queued draft/target window, plus complete request-owner erase inspection.
pub(crate) const RUNTIME_BYTES: u64 =
    4 * MAIN_BYTES + 4 * (131072 + 32768 + 32768 + 32768 + 7 * 32768);
const P: usize = 0;
const HIST: usize = 1;
const CAND: usize = 2;
const OUT: usize = 3;
const RING: usize = 4;
const CONTROL: usize = 5;
const BASE: usize = 6;
// Original CONTROL is one32KiB owner. Each possible draft borrows a disjoint
// aligned4KiB row, so all round controls can be uploaded before graph capture.
const DRAFT_CONTROL_ROW_BYTES: u64 = 4096;
fn draft_control_range(index: usize) -> Result<std::ops::Range<u64>> {
    if index >= crate::strata_port::verify_capacity() - 1 {
        return Err(reject());
    }
    let start = (index as u64)
        .checked_mul(DRAFT_CONTROL_ROW_BYTES)
        .ok_or_else(reject)?;
    let end = start
        .checked_add(DRAFT_CONTROL_ROW_BYTES)
        .ok_or_else(reject)?;
    if end > ROOT_BYTES[CONTROL] {
        return Err(reject());
    }
    Ok(start..end)
}
fn draft_controls(position: usize) -> Result<Zeroizing<Vec<u8>>> {
    if !(1..=HISTORY_MAX).contains(&position) {
        return Err(reject());
    }
    let mut bytes = Zeroizing::new(vec![0u8; ROOT_BYTES[CONTROL] as usize]);
    for j in 0..crate::strata_port::verify_capacity() - 1 {
        let counter = position.checked_add(j).ok_or_else(reject)?;
        if counter > HISTORY_MAX {
            break;
        }
        let row = draft_control_range(j)?;
        for (dest, value) in bytes[row.start as usize..row.start as usize + 12]
            .chunks_exact_mut(4)
            .zip([(counter - 1) as u32, j as u32, position as u32])
        {
            dest.copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}
fn draft_control(root: &Handle, index: usize) -> Result<Handle> {
    if root.offset_start.unwrap_or(0) != 0
        || root.offset_end.unwrap_or(0) != 0
        || root.size_in_used() != ROOT_BYTES[CONTROL]
    {
        return Err(reject());
    }
    let row = draft_control_range(index)?;
    Ok(root
        .clone()
        .offset_start(row.start)
        .offset_end(ROOT_BYTES[CONTROL] - row.end))
}
pub(crate) fn active() -> bool {
    ACTIVE.with(|s| s.borrow().is_some())
}
fn reject() -> Error {
    Error::InvalidFormat("native sampling request owner/state")
}
thread_local! {static ACTIVE:RefCell<Option<State>>=const{RefCell::new(None)};}
struct State {
    client: ComputeClient<WgpuRuntime>,
    roots: [Handle; 7],
    params: Params,
    target: GraphBindingId,
    draft: Option<u64>,
    device_residuals: bool,
    retired: Arc<AtomicBool>,
    consumed: Zeroizing<Vec<u32>>,
    window: Zeroizing<Vec<u32>>,
    // phase0 idle,1 target staged,2 target launched,3 draft round.
    coordinates: Zeroizing<[usize; 4]>,
}
#[must_use]
pub(crate) struct Scope {
    _not_send: std::marker::PhantomData<std::rc::Rc<()>>,
    closed: bool,
    retired: Arc<AtomicBool>,
}
impl Scope {
    /// Complete arena/model ownership remains with the caller throughout this
    /// lexical scope. All roots are proved here, retained on failure by the arena,
    /// and never acquire independent free authority in TLS or offset consumers.
    /// # Safety
    /// The sole request worker retains these complete arena owners and their
    /// pre-device grants until close/retirement has joined all users. It must
    /// not reset/close either model or change its client while this scope lives.
    pub(crate) unsafe fn new(
        arena: &PrivateArena,
        ids: [BufferId; 7],
        target: GraphBindingId,
        draft: Option<u64>,
        params: Params,
        prefix: &[u32],
        retired: Arc<AtomicBool>,
    ) -> Result<Self> {
        params.validate()?;
        if !crate::strata_intel_core::enabled()
            || !arena.public_owned_zero_audit_enabled()
            || retired.load(Ordering::Acquire)
            || prefix.len() > HISTORY_MAX
            || prefix.iter().any(|&id| id >= VOCAB as u32)
        {
            return Err(reject());
        }
        let roots: Vec<_> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                if ids[..i].contains(&id)
                    || arena.layout(id).map_err(u::fault)?
                        != BufferLayout::new(ROOT_BYTES[i] as usize / 4).map_err(u::fault)?
                {
                    return Err(reject());
                }
                let h = arena.handle(id).cloned().map_err(u::fault)?;
                if h.size() != ROOT_BYTES[i]
                    || h.size_in_used() != ROOT_BYTES[i]
                    || h.offset_start.unwrap_or(0) != 0
                    || h.offset_end.unwrap_or(0) != 0
                {
                    return Err(reject());
                }
                Ok(h)
            })
            .collect::<Result<_>>()?;
        let roots: [Handle; 7] = roots.try_into().map_err(|_| reject())?;
        if roots.iter().any(|h| h.stream != roots[0].stream)
            || roots.iter().enumerate().any(|(i, h)| {
                roots[..i].iter().any(|a| {
                    a.clone().binding().memory.allocation_id()
                        == h.clone().binding().memory.allocation_id()
                })
            })
        {
            return Err(reject());
        }
        let mut consumed = Zeroizing::new(Vec::with_capacity(HISTORY_MAX));
        consumed.extend_from_slice(prefix);
        let state = State {
            client: arena.client().map_err(u::fault)?.clone(),
            roots,
            params,
            target,
            draft,
            device_residuals: false,
            retired: retired.clone(),
            consumed,
            window: Zeroizing::new(Vec::with_capacity(crate::strata_port::verify_capacity())),
            coordinates: Zeroizing::new([0; 4]),
        };
        ACTIVE.with(|s| {
            let mut s = s.borrow_mut();
            if s.is_some() {
                return Err(reject());
            }
            *s = Some(state);
            Ok(())
        })?;
        Ok(Self {
            _not_send: std::marker::PhantomData,
            closed: false,
            retired,
        })
    }
    pub(crate) fn stage_target(&self, position: usize, inputs: &[u32]) -> Result<()> {
        with(|s| {
            stage_window(
                &s.consumed,
                &mut s.window,
                &mut s.coordinates,
                position,
                inputs,
            )
        })
    }
    pub(crate) fn commit_target(&self, n: usize) -> Result<()> {
        with(|s| commit_window(&mut s.consumed, &mut s.window, &mut s.coordinates, n))
    }
    /// Explicit selected-controller policy. Legacy sampler users retain their
    /// existing host carry packet until they select original device consumers.
    pub(crate) fn use_device_residuals(&self) -> Result<()> {
        with(|s| {
            if s.coordinates[0] != 0 {
                return Err(reject());
            }
            s.device_residuals = true;
            Ok(())
        })
    }
    pub(crate) fn begin_round(&self, position: usize, next: u32) -> Result<()> {
        with(|s| {
            if s.coordinates[0] != 0 || position != s.consumed.len() || next >= VOCAB as u32 {
                return Err(reject());
            }
            if s.params.greedy() || s.draft.is_none() {
                return Ok(());
            }
            let controls = draft_controls(position)?;
            let h = s.params.penalty_last_n.min(HISTORY_MAX);
            let history = coupled_base(&s.consumed, next, h)?;
            s.params.counter = position as u64;
            let words = s.params.words(1, h, 0, 2)?;
            let mut bytes = Zeroizing::new(vec![0; ROOT_BYTES[BASE] as usize]);
            for (d, w) in bytes[..128].chunks_exact_mut(8).zip(words.iter()) {
                d.copy_from_slice(&w.to_le_bytes());
            }
            // Full padded base is the original native4096-slot ring base, even if h<cap.
            bytes[256..256 + HISTORY_MAX * 4].fill(0xff);
            if h > 0 {
                for (d, w) in bytes[256 + (HISTORY_MAX - h) * 4..256 + HISTORY_MAX * 4]
                    .chunks_exact_mut(4)
                    .zip(history.iter())
                {
                    d.copy_from_slice(&w.to_le_bytes());
                }
            }
            bytes[256 + (HISTORY_MAX - 1) * 4..256 + HISTORY_MAX * 4]
                .copy_from_slice(&next.to_le_bytes());
            s.write(BASE, bytes)?;
            // CPU controls for every possible draft are staged once. The
            // original GPU bridge still validates the actual positions/token,
            // constructs history and publishes the real sampled proposal.
            s.write(CONTROL, controls)?;
            s.coordinates[0] = 3;
            s.coordinates[2] = position;
            s.coordinates[3] = 0;
            Ok(())
        })
    }
    /// Join prior consumers, erase all seven complete owners, then inspect every
    /// byte including padding. The parent's scope-close then erases audit I/O.
    pub(crate) fn close(mut self) -> Result<()> {
        let state = ACTIVE.with(|s| s.borrow_mut().take()).ok_or_else(reject)?;
        let result = (|| {
            state
                .client
                .clear_private_owners(state.roots.to_vec())
                .map_err(u::fault)?;
            for root in &state.roots {
                let mut bytes = state
                    .client
                    .read_private_cleanup(root.clone())
                    .map_err(u::fault)?;
                let valid = bytes.len() == root.size() as usize && bytes.iter().all(|&v| v == 0);
                bytes.zeroize();
                if !valid {
                    return Err(reject());
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.retired.store(true, Ordering::Release);
        }
        self.closed = true;
        result
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        if !self.closed {
            self.retired.store(true, Ordering::Release);
            ACTIVE.with(|s| {
                s.borrow_mut().take();
            });
        }
    }
}
fn with<T>(f: impl FnOnce(&mut State) -> Result<T>) -> Result<T> {
    ACTIVE.with(|s| {
        let mut s = s.borrow_mut();
        let s = s.as_mut().ok_or_else(reject)?;
        if s.retired.load(Ordering::Acquire) {
            return Err(reject());
        }
        f(s)
    })
}
impl State {
    fn write(&self, root: usize, mut bytes: Zeroizing<Vec<u8>>) -> Result<()> {
        if bytes.len() != ROOT_BYTES[root] as usize {
            return Err(reject());
        }
        self.client
            .write_private(self.roots[root].clone(), std::mem::take(&mut *bytes))
            .map_err(u::fault)
    }
    fn context(&self, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
        if !std::ptr::eq(self.client.info(), client.info()) {
            return Err(reject());
        }
        Ok(())
    }
}
pub(crate) fn device_residuals(
    client: &ComputeClient<WgpuRuntime>,
    binding: GraphBindingId,
) -> Result<bool> {
    if !active() {
        return Ok(false);
    }
    with(|s| {
        s.context(client)?;
        if s.target != binding {
            return Err(reject());
        }
        Ok(s.device_residuals)
    })
}
/// Runs inside the original authenticated target packet I/O closure, after the
/// real target head. No host logits, arbitrary compute owner, or shadow routes.
pub(crate) fn target_pick(
    client: &ComputeClient<WgpuRuntime>,
    binding: GraphBindingId,
    logits: Handle,
    fault: Handle,
    count: usize,
    start: usize,
) -> Result<Option<Handle>> {
    if !active() {
        return Ok(None);
    }
    with(|s| {
        s.context(client)?;
        if s.target != binding
            || s.coordinates[0] != 1
            || s.coordinates[1] != start
            || s.window.len() != count
        {
            return Err(reject());
        }
        let h = s.params.penalty_last_n.min(HISTORY_MAX);
        s.params.counter = start as u64;
        let history = penalty_rows(&s.consumed, &s.window, h)?;
        let mut bytes = Zeroizing::new(vec![0; ROOT_BYTES[HIST] as usize]);
        for (d, w) in bytes.chunks_exact_mut(4).zip(history.iter()) {
            d.copy_from_slice(&w.to_le_bytes());
        }
        s.write(HIST, bytes)?;
        let words = s.params.words(count, h, 0, 2)?;
        let mut bytes = Zeroizing::new(vec![0; ROOT_BYTES[P] as usize]);
        for (d, w) in bytes.chunks_exact_mut(8).zip(words.iter()) {
            d.copy_from_slice(&w.to_le_bytes());
        }
        s.write(P, bytes)?;
        let mut empty = Zeroizing::new(vec![0; ROOT_BYTES[OUT] as usize]);
        for pair in empty[..count * 8].chunks_exact_mut(8) {
            pair[..4].copy_from_slice(&(VOCAB as u32).to_le_bytes());
            pair[4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        }
        s.write(OUT, empty)?;
        // All complete roots were authenticated in Scope::new. Target authenticates
        // its own original logits and sticky fault before entering this closure.
        unsafe {
            super::launch(
                client,
                [
                    logits,
                    s.roots[HIST].clone(),
                    s.roots[CAND].clone(),
                    s.roots[OUT].clone(),
                    fault,
                    s.roots[P].clone(),
                ],
                &s.params,
                count,
                h,
                0,
                2,
            )?;
        }
        s.coordinates[0] = 2;
        Ok(Some(s.roots[OUT].clone()))
    })
}
/// Coupled MTP uses the same request chain/uniform as its verifying target row.
/// The real front's device position table supplies the draw counter. Consecutive
/// drafts feed the actual preceding GPU token and extend only the device ring.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draft_pick(
    client: &ComputeClient<WgpuRuntime>,
    binding: u64,
    cell: usize,
    logits: Handle,
    positions: Handle,
    input: Handle,
    greedy: Handle,
    confidence: Handle,
    fault: Handle,
    cjk: Option<[Handle; 2]>,
) -> Result<bool> {
    if !active() {
        return Ok(false);
    }
    with(|s| {
        s.context(client)?;
        if s.draft != Some(binding) {
            return Err(reject());
        }
        if s.params.greedy() {
            return Ok(false);
        }
        let counter = coupled_counter(cell)?;
        let j = (counter as usize)
            .checked_sub(s.coordinates[2])
            .ok_or_else(reject)?;
        if s.draft != Some(binding)
            || s.coordinates[0] != 3
            || j != s.coordinates[3]
            || j >= crate::strata_port::verify_capacity() - 1
            || cell >= HISTORY_MAX
        {
            return Err(reject());
        }
        let control = draft_control(&s.roots[CONTROL], j)?;
        if let Some(maps) = &cjk
            && (maps[0].size_in_used() < crate::resident_mtp::cjk::MAP_BYTES as u64
                || maps[1].size_in_used() < (VOCAB * 4) as u64)
        {
            return Err(reject());
        }
        let mut prepare = vec![
            s.roots[P].clone(),
            s.roots[HIST].clone(),
            s.roots[RING].clone(),
            s.roots[BASE].clone(),
            positions,
            control.clone(),
            input,
            fault.clone(),
            s.roots[OUT].clone(),
        ];
        if let Some(maps) = &cjk {
            prepare.push(maps[1].clone());
        }
        bridge(
            client,
            if cjk.is_some() {
                Bridge::PrepareCjk
            } else {
                Bridge::Prepare
            },
            prepare,
        )?;
        s.params.counter = counter;
        let h = s.params.penalty_last_n.min(HISTORY_MAX);
        let owners = [
            logits,
            s.roots[HIST].clone(),
            s.roots[CAND].clone(),
            s.roots[OUT].clone(),
            fault.clone(),
            s.roots[P].clone(),
        ];
        // SAFETY: bridge publishes the same parameter/history domain as the
        // selected shader. Public map owners remain retained by the MTP model.
        unsafe {
            if cjk.is_some() {
                super::launch_cjk(client, owners, &s.params, h)?;
            } else {
                super::launch(client, owners, &s.params, 1, h, 0, 2)?;
            }
        }
        let mut publish = vec![
            s.roots[OUT].clone(),
            s.roots[RING].clone(),
            greedy,
            control,
            fault,
            confidence,
        ];
        if let Some(maps) = &cjk {
            publish.push(maps[0].clone());
        }
        bridge(
            client,
            if cjk.is_some() {
                Bridge::PublishCjk
            } else {
                Bridge::Publish
            },
            publish,
        )?;
        s.coordinates[3] += 1;
        Ok(true)
    })
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum Bridge {
    Prepare,
    Publish,
    PrepareCjk,
    PublishCjk,
}
impl Bridge {
    fn source(self) -> String {
        let source = include_str!("strata-sampling-bridge.wgsl")
            .split_once(match self {
                Self::Prepare | Self::PrepareCjk => "// PREPARE_BEGIN\n",
                Self::Publish | Self::PublishCjk => "// PUBLISH_BEGIN\n",
            })
            .unwrap()
            .1
            .split_once("// END")
            .unwrap()
            .0
            .to_owned();
        match self {
            Self::PrepareCjk=>source
                .replace("@compute @workgroup_size(256)","@group(0) @binding(9) var<storage,read> inverse:array<u32>;\n@compute @workgroup_size(256)")
                .replace("if arrayLength(&params)<32u", "if arrayLength(&inverse)<248320u||arrayLength(&params)<32u")
                .replace("lane!=16u&&lane!=17u", "lane!=1u&&lane!=3u&&lane!=16u&&lane!=17u")
                .replace("params[16]=positions[cell]+1u;", "params[1]=106299u;params[3]=26u;params[16]=positions[cell]+1u;")
                .replace("history[i]=ring[4096u+j-h+i];", "let token=ring[4096u+j-h+i];var compact=-1;if token>=0&&token<248320 {let index=inverse[u32(token)];if index<106299u {compact=i32(index);}else if index!=0xffffffffu {atomicOr(&fault[0],0x10000000u);}}history[i]=compact;"),
            Self::PublishCjk=>source
                .replace("@compute @workgroup_size(64)","@group(0) @binding(6) var<storage,read> ids:array<u32>;\n@compute @workgroup_size(64)")
                .replace("if arrayLength(&picks)<1u", "if arrayLength(&ids)<106299u||arrayLength(&picks)<1u")
                .replace("pick.x>=248320u", "pick.x>=106299u")
                .replace("confidence[64]=pick.y;", "let original=ids[pick.x];if original>=248320u {atomicOr(&fault[0],0x10000000u);token[0]=248320u;return;}confidence[64]=pick.y;")
                .replace("ring[4096u+j]=i32(pick.x)","ring[4096u+j]=i32(original)")
                .replace("token[0]=pick.x", "token[0]=original"),
            _=>source,
        }
    }
    fn lanes(self) -> u32 {
        if matches!(self, Self::Prepare | Self::PrepareCjk) {
            256
        } else {
            64
        }
    }
}
impl KernelMetadata for Bridge {
    fn name(&self) -> &'static str {
        match self {
            Self::Prepare => "StrataCoupledDraftDevicePrepare",
            Self::Publish => "StrataCoupledDraftDevicePublish",
            Self::PrepareCjk => "StrataCoupledCjkDraftDevicePrepare",
            Self::PublishCjk => "StrataCoupledCjkDraftDevicePublish",
        }
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(self.lanes()))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Bridge {
    fn compile(
        &self,
        _: &mut AutoCompiler,
        _: &cubecl::WgpuCompilationOptions,
        _: ExecutionMode,
        _: StorageType,
    ) -> std::result::Result<CompiledKernel<AutoCompiler>, CompilationError> {
        Ok(CompiledKernel {
            entrypoint_name: "main".into(),
            debug_name: None,
            source: self.source(),
            repr: None,
            cube_dim: CubeDim::new_1d(self.lanes()),
            debug_info: None,
        })
    }
}
fn bridge(client: &ComputeClient<WgpuRuntime>, k: Bridge, handles: Vec<Handle>) -> Result<()> {
    // Parent supplies same-device authenticated complete owning pages. Reject
    // stream/physical aliasing before the first unsafe bridge dispatch as well.
    for (i, h) in handles.iter().enumerate() {
        if h.stream != handles[0].stream
            || h.size_in_used() < 4
            || handles[..i].iter().any(|a| {
                a.clone().binding().memory.allocation_id()
                    == h.clone().binding().memory.allocation_id()
            })
        {
            return Err(reject());
        }
    }
    let mut args = KernelArguments::new();
    for h in handles {
        args = args.with_buffer(h.binding());
    }
    unsafe {
        client.launch_unchecked(Box::new(k), CubeCount::Static(1, 1, 1), args);
    }
    Ok(())
}
pub(crate) fn sources() -> Vec<CompiledKernel<AutoCompiler>> {
    [
        Bridge::Prepare,
        Bridge::Publish,
        Bridge::PrepareCjk,
        Bridge::PublishCjk,
    ]
    .into_iter()
    .map(|k| CompiledKernel {
        entrypoint_name: "main".into(),
        debug_name: None,
        source: k.source(),
        repr: None,
        cube_dim: CubeDim::new_1d(k.lanes()),
        debug_info: None,
    })
    .collect()
}

// These are the actual controller-to-I/O state transitions, separated from
// device ownership so rejection/accepted-prefix semantics can be checked on CPU.
fn stage_window(
    consumed: &[u32],
    window: &mut Zeroizing<Vec<u32>>,
    coordinates: &mut [usize; 4],
    position: usize,
    inputs: &[u32],
) -> Result<()> {
    if !matches!(coordinates[0], 0 | 3)
        || position != consumed.len()
        || inputs.is_empty()
        || inputs.len() > crate::strata_port::verify_capacity()
        || position
            .checked_add(inputs.len())
            .is_none_or(|n| n > HISTORY_MAX)
        || inputs.iter().any(|&id| id >= VOCAB as u32)
    {
        return Err(reject());
    }
    window.zeroize();
    window.extend_from_slice(inputs);
    coordinates[0] = 1;
    coordinates[1] = position;
    Ok(())
}
fn commit_window(
    consumed: &mut Zeroizing<Vec<u32>>,
    window: &mut Zeroizing<Vec<u32>>,
    coordinates: &mut [usize; 4],
    n: usize,
) -> Result<()> {
    if coordinates[0] != 2
        || n == 0
        || n > window.len()
        || consumed.len() != coordinates[1]
        || consumed
            .len()
            .checked_add(n)
            .is_none_or(|v| v > HISTORY_MAX)
    {
        return Err(reject());
    }
    consumed.extend_from_slice(&window[..n]);
    window.zeroize();
    coordinates[0] = 0;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_coupled_draft_controls_are_staged_once_in_disjoint_original_rows() {
        let count = crate::strata_port::verify_capacity() - 1;
        for position in [1, 63, 2048, HISTORY_MAX - count + 1, HISTORY_MAX] {
            let bytes = draft_controls(position).unwrap();
            assert_eq!(bytes.len(), ROOT_BYTES[CONTROL] as usize);
            let mut expected = vec![0u8; bytes.len()];
            let mut end = 0;
            for j in 0..count {
                let row = draft_control_range(j).unwrap();
                assert!(row.start >= end);
                assert_eq!(row.start % 4096, 0);
                assert_eq!(row.end - row.start, DRAFT_CONTROL_ROW_BYTES);
                assert!(row.end <= ROOT_BYTES[CONTROL]);
                end = row.end;
                let counter = position + j;
                if counter <= HISTORY_MAX {
                    let words: Vec<_> = bytes[row.start as usize..row.start as usize + 12]
                        .chunks_exact(4)
                        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                        .collect();
                    assert_eq!(words, [(counter - 1) as u32, j as u32, position as u32]);
                    assert_eq!(
                        coupled_counter(words[0] as usize).unwrap() as usize,
                        position + j
                    );
                    expected[row.start as usize..row.start as usize + 12]
                        .copy_from_slice(&bytes[row.start as usize..row.start as usize + 12]);
                }
            }
            assert_eq!(&*bytes, &expected);
        }
        for invalid in [0, HISTORY_MAX + 1, usize::MAX] {
            assert!(draft_controls(invalid).is_err());
        }
        for invalid in [count, usize::MAX] {
            assert!(draft_control_range(invalid).is_err());
        }
    }
    #[test]
    fn native_sampling_request_commits_only_verified_input_prefix_and_rejects_duplicate_commit() {
        for count in 1..=crate::strata_port::verify_capacity() {
            for n in 1..=count {
                let mut consumed = Zeroizing::new(vec![1, 2, 3]);
                let mut window = Zeroizing::new(Vec::new());
                let mut c = [0; 4];
                let inputs: Vec<_> = (10..10 + count as u32).collect();
                stage_window(&consumed, &mut window, &mut c, 3, &inputs).unwrap();
                assert!(commit_window(&mut consumed, &mut window, &mut c, n).is_err());
                c[0] = 2;
                commit_window(&mut consumed, &mut window, &mut c, n).unwrap();
                assert_eq!(&*consumed, &[&[1, 2, 3][..], &inputs[..n]].concat());
                assert!(window.is_empty());
                assert_eq!(c[0], 0);
                assert!(commit_window(&mut consumed, &mut window, &mut c, n).is_err());
                let len = consumed.len();
                stage_window(&consumed, &mut window, &mut c, len, &[99]).unwrap();
                assert_eq!(&*window, &[99]);
            }
        }
    }
    #[test]
    fn native_sampling_request_preflight_and_prices_preserve_complete_owners() {
        let consumed = vec![1, 2];
        let mut window = Zeroizing::new(vec![7]);
        let mut c = [0; 4];
        for (position, inputs) in [
            (0, vec![1]),
            (2, vec![]),
            (2, vec![VOCAB as u32]),
            (2, vec![0; crate::strata_port::verify_capacity() + 1]),
        ] {
            assert!(stage_window(&consumed, &mut window, &mut c, position, &inputs).is_err());
            assert_eq!(&*window, &[7]);
            assert_eq!(c, [0; 4]);
        }
        assert_eq!(ROOT_BYTES.into_iter().sum::<u64>(), MAIN_BYTES);
        for bytes in ROOT_BYTES {
            assert_eq!(
                BufferLayout::new(bytes as usize / 4).unwrap().owned_bytes as u64,
                bytes
            );
        }
        if crate::strata_intel_core::enabled() {
            assert_eq!(
                crate::strata_global_moe::Workspace::requests()
                    .into_iter()
                    .sum::<u64>(),
                crate::strata_global_moe::MAIN_BYTES
            );
        }
        assert!(RUNTIME_BYTES >= 4 * MAIN_BYTES + 4 * ROOT_BYTES[HIST]);
        assert!(
            HOST_BYTES
                >= 2 * ROOT_BYTES[CAND] as usize + 4 * ROOT_BYTES[HIST] as usize + HISTORY_MAX * 4
        );
    }
    #[test]
    fn native_sampling_request_bridges_and_selected_packet_lower_offline() {
        use wgpu::naga as n;
        let mut all: Vec<_> = sources().into_iter().map(|s| s.source).collect();
        all.push(crate::strata_packet::selected_shader());
        for source in all {
            let module = n::front::wgsl::parse_str(&source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            let info = n::valid::Validator::new(
                n::valid::ValidationFlags::all(),
                n::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
            n::back::spv::write_vec(
                &module,
                &info,
                &n::back::spv::Options {
                    lang_version: (1, 3),
                    zero_initialize_workgroup_memory:
                        n::back::spv::ZeroInitializeWorkgroupMemoryMode::None,
                    ..Default::default()
                },
                Some(&n::back::spv::PipelineOptions {
                    shader_stage: n::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
    }
}
