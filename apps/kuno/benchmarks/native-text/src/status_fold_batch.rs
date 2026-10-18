//! Opt-in status folds at existing ready-before-reuse boundaries, no acceptance.
use crate::{
    private_arena::{BufferId, PrivateArena},
    resident_util as u,
};
use cubecl::{
    ir::{ElemType, StorageType, UIntKind},
    prelude::{ComputeClient, CubeCount, CubeDim},
    wgpu::{AutoCompiler, WgpuRuntime},
};
use cubecl_runtime::{
    compiler::{CompilationError, CubeTask},
    id::KernelId,
    kernel::{CompiledKernel, KernelMetadata},
    server::{ExecutionMode, Handle, KernelArguments, MetadataBindingInfo},
};
use kuno_text::{Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use zeroize::{Zeroize, Zeroizing};
static SELECTED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
pub(crate) fn select(selected: bool) {
    SELECTED.store(selected, Ordering::Release);
}
pub(crate) fn selected() -> bool {
    SELECTED.load(Ordering::Acquire)
}
pub(crate) fn calls() -> u64 {
    CALLS.load(Ordering::Relaxed)
}
/// Model-local permission selects only the original target status spans. The
/// legacy process-global experiment remains independent and cannot earn these
/// actual dispatch receipts.
#[derive(Default)]
pub(crate) struct PublicPolicy {
    selected: bool,
    dispatched: u64,
}
impl PublicPolicy {
    pub(crate) fn state(&self) -> (bool, u64) {
        (self.selected, self.dispatched)
    }
    pub(crate) fn validate_selection(&self) -> Result<()> {
        if self.dispatched != 0 {
            return Err(Error::InvalidFormat(
                "target grouped status selection after dispatch",
            ));
        }
        Ok(())
    }
    /// Enclosing model preflights every workspace before changing any policy.
    pub(crate) fn enable(&mut self) {
        self.selected = true;
    }
    /// Advance the original counter for a fold in a genuinely replayed native
    /// program. Only the private joined replay receipt supplies its mode; the
    /// stage caller must first check its original pending owner/sequence ticket.
    /// This grants no numerical/fault/cursor acceptance and enqueues no new fold.
    pub(crate) fn complete_native_replay(
        &mut self,
        arena: &PrivateArena,
        fault: BufferId,
        completed: &crate::resident_model::target_replay::Completed,
    ) -> Result<()> {
        completed.check_fault_owner(arena, fault)?;
        let next = next_native_replay_dispatch(
            self.selected,
            completed.grouped_status_folds(),
            self.dispatched,
        )?;
        self.dispatched = next;
        Ok(())
    }
    pub(crate) fn launch(
        &mut self,
        arena: &PrivateArena,
        views: &[View],
        fault: BufferId,
    ) -> Result<()> {
        let next = next_public_dispatch(self.selected, self.dispatched)?;
        launch(arena, views, fault)?;
        self.dispatched = next;
        Ok(())
    }
}
fn next_public_dispatch(selected: bool, current: u64) -> Result<u64> {
    if !selected {
        return Err(Error::InvalidFormat(
            "unselected target grouped status dispatch",
        ));
    }
    current.checked_add(1).ok_or(Error::ArithmeticOverflow)
}

fn next_native_replay_dispatch(selected: bool, recorded: bool, current: u64) -> Result<u64> {
    if selected != recorded || !selected && current != 0 {
        return Err(Error::InvalidFormat(
            "native replay grouped fold policy mismatch",
        ));
    }
    if selected {
        next_public_dispatch(true, current)
    } else {
        Ok(current)
    }
}
fn family_inventory(native: bool) -> [u64; 3] {
    // Native GR writers fold directly into the sticky model fault. GDN and
    // QSA still each execute one real grouped check per original layer.
    [if native { 0 } else { 97 }, 36, 12]
}

/// Public counts only. A proposal joins every actual original target role and
/// is published by the model only after all original family commits and health.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct Completed {
    pub(crate) chunks: u64,
    pub(crate) gr_dispatches: u64,
    pub(crate) gdn_dispatches: u64,
    pub(crate) qsa_dispatches: u64,
}
impl Completed {
    pub(crate) fn dispatches(self) -> [u64; 3] {
        [self.gr_dispatches, self.gdn_dispatches, self.qsa_dispatches]
    }
    pub(crate) fn propose(
        self,
        selected: bool,
        before: [u64; 3],
        after: [u64; 3],
        expected: [u64; 3],
    ) -> Result<Self> {
        self.propose_for_mode(
            crate::strata_intel_core::enabled(),
            selected,
            before,
            after,
            expected,
        )
    }
    fn propose_for_mode(
        self,
        native: bool,
        selected: bool,
        before: [u64; 3],
        after: [u64; 3],
        expected: [u64; 3],
    ) -> Result<Self> {
        let inventory = family_inventory(native);
        let accepted = inventory.map(|n| self.chunks.checked_mul(n));
        if accepted != self.dispatches().map(Some) {
            return Err(Error::InvalidFormat(
                "target grouped status prior accepted join",
            ));
        }
        if before != self.dispatches() {
            return Err(Error::InvalidFormat(
                "target grouped status unaccepted dispatches",
            ));
        }
        if !selected {
            if self != Self::default() || after != before {
                return Err(Error::InvalidFormat(
                    "unselected target grouped status credit",
                ));
            }
            return Ok(self);
        }
        // These are the real DASLab48 roles. Native GR writes its sticky
        // fault directly; legacy GR includes FinalAllRows and both layer halves.
        if expected != inventory {
            return Err(Error::InvalidFormat(
                "target grouped status actual family inventory",
            ));
        }
        for i in 0..3 {
            if after[i].checked_sub(before[i]) != Some(expected[i]) {
                return Err(Error::InvalidFormat(
                    "target grouped status missing or surplus fold",
                ));
            }
        }
        Ok(Self {
            chunks: self
                .chunks
                .checked_add(1)
                .ok_or(Error::ArithmeticOverflow)?,
            gr_dispatches: after[0],
            gdn_dispatches: after[1],
            qsa_dispatches: after[2],
        })
    }
    pub(crate) fn receipt(self, selected: bool, actual: [u64; 3]) -> Result<(bool, Self)> {
        self.receipt_for_mode(crate::strata_intel_core::enabled(), selected, actual)
    }
    fn receipt_for_mode(
        self,
        native: bool,
        selected: bool,
        actual: [u64; 3],
    ) -> Result<(bool, Self)> {
        let inventory = family_inventory(native);
        if actual != self.dispatches() || selected == (self.chunks == 0) {
            return Err(Error::InvalidFormat(
                "target grouped status incomplete receipt",
            ));
        }
        let expected = inventory.map(|n| self.chunks.checked_mul(n));
        if expected != self.dispatches().map(Some) {
            return Err(Error::InvalidFormat(
                "target grouped status accepted family join",
            ));
        }
        Ok((selected, self))
    }
}

#[cfg(test)]
mod public_policy_tests {
    use super::*;
    #[test]
    fn cold_capture_and_replay_preserve_actual_native_and_legacy_fold_inventories() {
        for native in [false, true] {
            let expected = if native { [0, 36, 12] } else { [97, 36, 12] };
            let mut accepted = Completed::default();
            for frame in 0..7u64 {
                let before = accepted.dispatches();
                let after = std::array::from_fn(|i| before[i] + expected[i]);
                accepted = accepted
                    .propose_for_mode(native, true, before, after, expected)
                    .unwrap();
                assert_eq!(accepted.chunks, frame + 1);
                assert_eq!(
                    accepted.receipt_for_mode(native, true, after).unwrap().1,
                    accepted
                );
                for role in 0..3 {
                    let mut missing = after;
                    missing[role] += 1;
                    assert!(
                        accepted
                            .propose_for_mode(native, true, after, missing, expected)
                            .is_err()
                    );
                    let mut wrong = expected;
                    wrong[role] += 1;
                    assert!(
                        accepted
                            .propose_for_mode(
                                native,
                                true,
                                after,
                                std::array::from_fn(|i| after[i] + wrong[i]),
                                wrong
                            )
                            .is_err()
                    );
                }
            }
        }
        assert_eq!(family_inventory(true), [0, 36, 12]);
        assert_eq!(family_inventory(false), [97, 36, 12]);
        assert!(
            Completed::default()
                .propose(
                    true,
                    [0; 3],
                    family_inventory(!crate::strata_intel_core::enabled()),
                    family_inventory(!crate::strata_intel_core::enabled())
                )
                .is_err()
        );
    }
    #[test]
    fn recorded_native_folds_advance_once_only_in_the_original_selected_mode() {
        for selected in [false, true] {
            for recorded in [false, true] {
                for current in [0, 1, 35, 36, 47, 48, 999, u64::MAX] {
                    let result = next_native_replay_dispatch(selected, recorded, current);
                    let expected = if selected != recorded || !selected && current != 0 {
                        None
                    } else if selected {
                        current.checked_add(1)
                    } else {
                        Some(current)
                    };
                    assert_eq!(result.ok(), expected);
                }
            }
        }
        let before = [0, 36, 12];
        let mut after = before;
        for _ in 0..36 {
            after[1] = next_native_replay_dispatch(true, true, after[1]).unwrap();
        }
        for _ in 0..12 {
            after[2] = next_native_replay_dispatch(true, true, after[2]).unwrap();
        }
        let cold = Completed::default()
            .propose_for_mode(true, true, [0; 3], before, [0, 36, 12])
            .unwrap();
        let replay = cold
            .propose_for_mode(true, true, before, after, [0, 36, 12])
            .unwrap();
        assert_eq!(replay.dispatches(), [0, 72, 24]);
        assert_eq!(replay.chunks, 2);
    }
    #[test]
    fn immutable_target_policy_does_not_change_the_old_global_flag() {
        let global = selected();
        let mut policy = PublicPolicy::default();
        assert_eq!(policy.state(), (false, 0));
        policy.validate_selection().unwrap();
        assert!(next_public_dispatch(false, 0).is_err());
        policy.enable();
        assert_eq!(policy.state(), (true, 0));
        assert_eq!(next_public_dispatch(true, 0).unwrap(), 1);
        assert!(next_public_dispatch(true, u64::MAX).is_err());
        policy.dispatched = 1;
        assert!(policy.validate_selection().is_err());
        assert_eq!(selected(), global);
    }
    #[test]
    fn actual_completion_rejects_wrong_previous_credit_and_overflow() {
        let bad = Completed {
            chunks: 1,
            ..Completed::default()
        };
        assert!(
            bad.propose(true, [0; 3], [97, 36, 12], [97, 36, 12])
                .is_err()
        );
        let bad = Completed {
            chunks: u64::MAX,
            ..Completed::default()
        };
        assert!(
            bad.propose(true, [0; 3], [97, 36, 12], [97, 36, 12])
                .is_err()
        );
        assert!(bad.receipt(true, [0; 3]).is_err());
    }
}

pub(crate) struct View {
    region: BufferId,
    span: Zeroizing<[usize; 4]>,
}
impl View {
    pub(crate) fn new(
        region: BufferId,
        count: usize,
        first: usize,
        stride: usize,
        width: usize,
    ) -> Self {
        Self {
            region,
            span: Zeroizing::new([count, first, stride, width]),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Kernel(u8);
impl Kernel {
    fn source(self) -> String {
        let mut s = String::from("struct Params {count:u32,first:u32,stride:u32,width:u32}\n");
        for i in 0..self.0 {
            s.push_str(&format!(
                "@group(0) @binding({i}) var<storage,read> status{i}:array<u32>;\n"
            ));
        }
        s.push_str(&format!("@group(0) @binding({}) var<storage,read_write> fault:array<atomic<u32>>;\n@group(0) @binding({}) var<uniform> params:array<Params,{}>;\n",self.0,self.0+1,self.0));
        s.push_str(
            "@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {\n",
        );
        for i in 0..self.0 {
            s.push_str(&format!("if id.y<params[{i}].count && id.x<params[{i}].width {{if status{i}[params[{i}].first+id.y*params[{i}].stride+id.x]!=1u {{atomicOr(&fault[0],2u);}}}}\n"));
        }
        s.push('}');
        s
    }
}
impl KernelMetadata for Kernel {
    fn name(&self) -> &'static str {
        "GroupedActiveStatusFold"
    }
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(*self)
            .cube_dim(CubeDim::new_1d(64))
    }
    fn address_type(&self) -> StorageType {
        StorageType::Scalar(ElemType::UInt(UIntKind::U32))
    }
}
impl CubeTask<AutoCompiler> for Kernel {
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
            cube_dim: CubeDim::new_1d(64),
            debug_info: None,
        })
    }
}
pub(crate) fn launch(arena: &PrivateArena, views: &[View], fault: BufferId) -> Result<()> {
    if !matches!(views.len(), 3 | 5 | 8) || u::words(arena, fault)? != 1 {
        return Err(Error::InvalidBufferLength);
    }
    let client = arena.client().map_err(u::fault)?;
    let mut handles = Vec::new();
    handles
        .try_reserve_exact(views.len() + 1)
        .map_err(|_| Error::LimitExceeded("grouped status handles"))?;
    let mut args = KernelArguments::new();
    let mut info = MetadataBindingInfo::custom(Vec::new());
    info.data
        .try_reserve_exact(views.len() * 2)
        .map_err(|_| Error::LimitExceeded("grouped status INFO"))?;
    let mut width_max = 0usize;
    let mut count_max = 0usize;
    for (i, v) in views.iter().enumerate() {
        if v.region == fault || views[..i].iter().any(|old| old.region == v.region) {
            return Err(Error::InvalidBufferLength);
        }
        let [count, first, stride, width] = *v.span;
        crate::resident_batch_projection::validate_view(
            count,
            first,
            stride,
            width,
            u::words(arena, v.region)?,
        )?;
        if [count, first, stride, width]
            .iter()
            .any(|&n| n > u32::MAX as usize)
        {
            return Err(Error::InvalidBufferLength);
        }
        handles.push(arena.handle(v.region).map_err(u::fault)?.clone());
        args = args.with_buffer(u::handle(arena, v.region)?.binding());
        info.data.push(count as u64 | ((first as u64) << 32));
        info.data.push(stride as u64 | ((width as u64) << 32));
        width_max = width_max.max(width);
        count_max = count_max.max(count);
    }
    handles.push(arena.handle(fault).map_err(u::fault)?.clone());
    args = args.with_buffer(u::handle(arena, fault)?.binding());
    let handle_bytes = handles
        .capacity()
        .checked_mul(std::mem::size_of::<Handle>())
        .ok_or(Error::ArithmeticOverflow)?;
    if handle_bytes > 4096 {
        return Err(Error::LimitExceeded("grouped status metadata"));
    }
    let stream = handles.last().unwrap().stream;
    for h in &handles {
        if h.stream != stream
            || h.offset_start.unwrap_or(0) != 0
            || h.offset_end.unwrap_or(0) != 0
            || h.size() != h.size_in_used()
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let metadata = u::resources(client, handles)?;
    if metadata.len() != views.len() + 1 {
        return Err(Error::InvalidBufferLength);
    }
    let bytes = metadata
        .capacity()
        .checked_mul(std::mem::size_of_val(&metadata[0]))
        .and_then(|n| n.checked_add(handle_bytes))
        .and_then(|n| {
            n.checked_add(args.buffers.capacity() * std::mem::size_of_val(&args.buffers[0]))
        })
        .and_then(|n| n.checked_add(info.data.capacity() * 8 + std::mem::size_of_val(views)))
        .ok_or(Error::ArithmeticOverflow)?;
    if bytes > 4096 {
        return Err(Error::LimitExceeded("grouped status metadata"));
    }
    for (i, r) in metadata.iter().enumerate() {
        let r = r.resource();
        let region = if i == views.len() {
            fault
        } else {
            views[i].region
        };
        let n = arena.layout(region).map_err(u::fault)?.owned_bytes as u64;
        if r.offset != 0
            || r.size != n
            || r.buffer.size() != n
            || metadata[..i]
                .iter()
                .any(|old| old.resource().buffer == r.buffer)
        {
            return Err(Error::InvalidBufferLength);
        }
    }
    let groups = width_max.div_ceil(64);
    if groups == 0 || groups > 65535 {
        return Err(Error::InvalidBufferLength);
    }
    // SAFETY: same validated logical spans and distinct actual full owners as
    // the original folds, ready before reuse/fence. Only fault bit2 is written.
    // Private runtime coordinates remain in the original wiping INFO ledger.
    unsafe {
        client.launch_unchecked(
            Box::new(Kernel(views.len() as u8)),
            CubeCount::Static(groups as u32, count_max as u32, 1),
            args.with_info(info),
        );
    }
    CALLS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map_err(|_| Error::ArithmeticOverflow)?;
    Ok(())
}
struct Scrub(cubecl::bytes::Bytes);
impl Drop for Scrub {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
// Fixed-public probe upload; ownership transfers immediately to write_private's
// wiping guard. The source words remain separately Zeroizing.
fn word_bytes(words: &[u32]) -> Result<Vec<u8>> {
    let n = words
        .len()
        .checked_mul(4)
        .ok_or(Error::ArithmeticOverflow)?;
    let mut bytes = Zeroizing::new(vec![0u8; n]);
    for (dst, word) in bytes.chunks_exact_mut(4).zip(words) {
        dst.copy_from_slice(&word.to_le_bytes());
    }
    Ok(std::mem::take(&mut *bytes))
}
pub(crate) fn probe(client: &ComputeClient<WgpuRuntime>, retired: Arc<AtomicBool>) -> Result<()> {
    let mut a = PrivateArena::new_transient_on_configured(client.clone(), 512 * 1024, retired)
        .map_err(u::fault)?;
    let widths = [2, 3, 5, 65, 257, 4, 17, 129];
    let ids = widths
        .map(|n| a.allocate_zeroed(8 * n).map_err(u::fault))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let fault = a.allocate_zeroed(1).map_err(u::fault)?;
    let duplicate = [ids[0], ids[0], ids[1]].map(|id| View::new(id, 1, 0, 1, 1));
    if launch(&a, &duplicate, fault).is_ok() {
        return Err(Error::InvalidBufferLength);
    }
    for members in [3, 5, 8] {
        for count in [1, 3, 8] {
            for case in 0..5 {
                let views: Vec<_> = ids[..members]
                    .iter()
                    .zip(&widths[..members])
                    .map(|(&id, &width)| View::new(id, count, 0, width, width))
                    .collect();
                for (i, &id) in ids[..members].iter().enumerate() {
                    let n = a.layout(id).map_err(u::fault)?.owned_words;
                    let mut words = Zeroizing::new(vec![9u32; n]);
                    words[..count * widths[i]].fill(1);
                    if (case == 1 || case == 4) && i == members - 1 {
                        words[count * widths[i] - 1] = 0;
                    }
                    if case == 2 && i == 0 {
                        words[0] = 0;
                    }
                    client
                        .write_private(a.handle(id).map_err(u::fault)?.clone(), word_bytes(&words)?)
                        .map_err(u::fault)?;
                }
                let mut f =
                    Zeroizing::new(vec![0u32; a.layout(fault).map_err(u::fault)?.owned_words]);
                if case == 3 || case == 4 {
                    f[0] = 8;
                }
                client
                    .write_private(a.handle(fault).map_err(u::fault)?.clone(), word_bytes(&f)?)
                    .map_err(u::fault)?;
                launch(&a, &views, fault)?;
                let bytes = Scrub(client.read_one(u::handle(&a, fault)?).map_err(u::fault)?);
                let expected = if case == 4 {
                    10
                } else if case == 1 || case == 2 {
                    2
                } else if case == 3 {
                    8
                } else {
                    0
                };
                if bytes.0.len() != 4
                    || u32::from_le_bytes(
                        bytes.0[..]
                            .try_into()
                            .map_err(|_| Error::InvalidBufferLength)?,
                    ) != expected
                {
                    return Err(Error::InvalidFormat("grouped status active/sticky probe"));
                }
                drop(bytes);
                cubecl::future::block_on(client.sanitize_transient_io_audited())
                    .map_err(u::fault)?;
            }
        }
    }
    a.close().map_err(u::fault)?;
    println!(
        "public_grouped_status_fold groups=3,5,8 counts=1,3,8 active_and_sticky=true inactive_ignored=true"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use wgpu::naga as n;

    type Offline<T> = std::result::Result<T, &'static str>;

    fn compiled_status_source(source: &str, members: usize) -> n::Module {
        let module = n::front::wgsl::parse_str(source).unwrap();
        let info = n::valid::Validator::new(
            n::valid::ValidationFlags::all(),
            n::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points.len(), 1);
        assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        assert!(module.functions.is_empty());
        let mut bindings = Vec::new();
        for (_, global) in module.global_variables.iter() {
            let binding = global.binding.as_ref().unwrap();
            assert_eq!(binding.group, 0);
            bindings.push(binding.binding);
            if binding.binding as usize == members + 1 {
                assert_eq!(global.space, n::AddressSpace::Uniform);
                let n::TypeInner::Array { base, stride, .. } = module.types[global.ty].inner else {
                    panic!("status parameters are not an array");
                };
                assert_eq!(stride, 16);
                let n::TypeInner::Struct { ref members, span } = module.types[base].inner else {
                    panic!("status parameters are not a struct");
                };
                assert_eq!(span, 16);
                assert_eq!(
                    members.iter().map(|m| m.offset).collect::<Vec<_>>(),
                    [0, 4, 8, 12]
                );
            } else {
                let n::AddressSpace::Storage { access } = global.space else {
                    panic!("unexpected status memory space");
                };
                assert_eq!(
                    access.contains(n::StorageAccess::STORE),
                    binding.binding as usize == members
                );
                let n::TypeInner::Array { base, stride, .. } = module.types[global.ty].inner else {
                    panic!("status storage is not an array");
                };
                assert_eq!(stride, 4);
                let scalar = match module.types[base].inner {
                    n::TypeInner::Atomic(scalar) if binding.binding as usize == members => scalar,
                    n::TypeInner::Scalar(scalar) if (binding.binding as usize) < members => scalar,
                    _ => panic!("unexpected status/fault element type"),
                };
                assert_eq!(scalar.kind, n::ScalarKind::Uint);
                assert_eq!(scalar.width, 4);
            }
        }
        bindings.sort_unstable();
        assert_eq!(bindings, (0..members as u32 + 2).collect::<Vec<_>>());
        let words = n::back::spv::write_vec(
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
        let mut atomic_ors = 0;
        let mut at = 5;
        while at < words.len() {
            let length = (words[at] >> 16) as usize;
            assert!(length > 0 && at + length <= words.len());
            atomic_ors += usize::from(words[at] & 65535 == 241); // OpAtomicOr.
            at += length;
        }
        assert_eq!(atomic_ors, members);
        module
    }

    #[derive(Clone, Debug)]
    enum Value {
        Word(u32),
        Bool(bool),
        Id([u32; 3]),
        Params([u32; 4]),
        Global(u32, Vec<u32>),
        Local(usize),
    }
    impl Value {
        fn word(self) -> Offline<u32> {
            match self {
                Self::Word(value) => Ok(value),
                _ => Err("expected word"),
            }
        }
        fn boolean(self) -> Offline<bool> {
            match self {
                Self::Bool(value) => Ok(value),
                _ => Err("expected boolean"),
            }
        }
        fn index(self, index: u32) -> Offline<Self> {
            match self {
                Self::Id(id) => id
                    .get(index as usize)
                    .copied()
                    .map(Self::Word)
                    .ok_or("id index"),
                Self::Params(params) => params
                    .get(index as usize)
                    .copied()
                    .map(Self::Word)
                    .ok_or("parameter index"),
                Self::Global(binding, mut path) => {
                    path.push(index);
                    Ok(Self::Global(binding, path))
                }
                _ => Err("unexpected indexed value"),
            }
        }
    }

    // Interpret the generated Naga control flow, including Emit timing. An
    // eagerly emitted storage load is observed even when its later branch is
    // inactive; a handwritten copy of the shader's if expression would miss it.
    struct Invocation<'a> {
        module: &'a n::Module,
        function: &'a n::Function,
        params: &'a [[u32; 4]],
        status: &'a [Vec<u32>],
        id: [u32; 3],
        fault: &'a mut u32,
        touched: &'a mut BTreeSet<(usize, usize)>,
        values: Vec<Option<Value>>,
        locals: Vec<Option<Value>>,
    }
    impl Invocation<'_> {
        fn expression(&mut self, handle: n::Handle<n::Expression>) -> Offline<Value> {
            if let Some(value) = &self.values[handle.index()] {
                return Ok(value.clone());
            }
            let value = match self.function.expressions[handle].clone() {
                n::Expression::Literal(n::Literal::U32(value)) => Value::Word(value),
                n::Expression::Literal(n::Literal::I32(value)) => {
                    Value::Word(u32::try_from(value).map_err(|_| "negative word")?)
                }
                n::Expression::Literal(n::Literal::Bool(value)) => Value::Bool(value),
                n::Expression::FunctionArgument(0) => Value::Id(self.id),
                n::Expression::GlobalVariable(global) => Value::Global(
                    self.module.global_variables[global]
                        .binding
                        .as_ref()
                        .ok_or("unbound global")?
                        .binding,
                    Vec::new(),
                ),
                n::Expression::LocalVariable(local) => Value::Local(local.index()),
                n::Expression::AccessIndex { base, index } => {
                    self.expression(base)?.index(index)?
                }
                n::Expression::Access { base, index } => {
                    let index = self.expression(index)?.word()?;
                    self.expression(base)?.index(index)?
                }
                n::Expression::As { expr, .. } => self.expression(expr)?,
                n::Expression::Load { pointer } => match self.expression(pointer)? {
                    Value::Local(local) => {
                        self.locals[local].clone().ok_or("uninitialized local")?
                    }
                    Value::Global(binding, path) if binding as usize == self.params.len() + 1 => {
                        match path.as_slice() {
                            [member] => Value::Params(
                                *self.params.get(*member as usize).ok_or("uniform member")?,
                            ),
                            [member, field] => Value::Word(
                                *self
                                    .params
                                    .get(*member as usize)
                                    .and_then(|p| p.get(*field as usize))
                                    .ok_or("uniform field")?,
                            ),
                            _ => return Err("uniform load shape"),
                        }
                    }
                    Value::Global(binding, path) if (binding as usize) < self.params.len() => {
                        let member = binding as usize;
                        let [count, first, stride, width] = self.params[member];
                        if self.id[1] >= count || self.id[0] >= width {
                            return Err("inactive status read");
                        }
                        let [index] = path.as_slice() else {
                            return Err("status load shape");
                        };
                        if *index != first + self.id[1] * stride + self.id[0] {
                            return Err("wrong active status span");
                        }
                        let word = *self.status[member]
                            .get(*index as usize)
                            .ok_or("out of bounds status read")?;
                        if !self.touched.insert((member, *index as usize)) {
                            return Err("duplicate status read");
                        }
                        Value::Word(word)
                    }
                    _ => return Err("unexpected storage load"),
                },
                n::Expression::Binary { op, left, right } => {
                    let left = self.expression(left)?;
                    let right = self.expression(right)?;
                    match op {
                        n::BinaryOperator::Add => {
                            Value::Word(left.word()?.wrapping_add(right.word()?))
                        }
                        n::BinaryOperator::Multiply => {
                            Value::Word(left.word()?.wrapping_mul(right.word()?))
                        }
                        n::BinaryOperator::Less => Value::Bool(left.word()? < right.word()?),
                        n::BinaryOperator::NotEqual => Value::Bool(left.word()? != right.word()?),
                        n::BinaryOperator::LogicalAnd => {
                            Value::Bool(left.boolean()? && right.boolean()?)
                        }
                        _ => return Err("unexpected binary operator"),
                    }
                }
                _ => return Err("unexpected generated expression"),
            };
            self.values[handle.index()] = Some(value.clone());
            Ok(value)
        }
        fn block(&mut self, block: &n::Block) -> Offline<()> {
            for statement in block {
                match statement {
                    n::Statement::Emit(range) => {
                        for handle in range.clone() {
                            self.expression(handle)?;
                        }
                    }
                    n::Statement::If {
                        condition,
                        accept,
                        reject,
                    } => {
                        if self.expression(*condition)?.boolean()? {
                            self.block(accept)?;
                        } else {
                            self.block(reject)?;
                        }
                    }
                    n::Statement::Store { pointer, value } => {
                        let Value::Local(local) = self.expression(*pointer)? else {
                            return Err("nonlocal store");
                        };
                        let value = self.expression(*value)?;
                        self.locals[local] = Some(value);
                    }
                    n::Statement::Atomic {
                        pointer,
                        fun,
                        value,
                        result,
                    } => {
                        if *fun != n::AtomicFunction::InclusiveOr {
                            return Err("non-OR atomic");
                        }
                        let Value::Global(binding, path) = self.expression(*pointer)? else {
                            return Err("non-global atomic");
                        };
                        if binding as usize != self.params.len()
                            || path != [0]
                            || self.expression(*value)?.word()? != 2
                        {
                            return Err("wrong fault mutation");
                        }
                        let old = *self.fault;
                        *self.fault |= 2;
                        if let Some(result) = result {
                            self.values[result.index()] = Some(Value::Word(old));
                        }
                    }
                    n::Statement::Block(block) => self.block(block)?,
                    n::Statement::Return { value: None } => return Ok(()),
                    _ => return Err("unexpected generated statement"),
                }
            }
            Ok(())
        }
    }
    fn invoke(
        module: &n::Module,
        params: &[[u32; 4]],
        status: &[Vec<u32>],
        id: [u32; 3],
        fault: &mut u32,
        touched: &mut BTreeSet<(usize, usize)>,
    ) -> Offline<()> {
        let function = &module.entry_points[0].function;
        let mut invocation = Invocation {
            module,
            function,
            params,
            status,
            id,
            fault,
            touched,
            values: vec![None; function.expressions.len()],
            locals: vec![None; function.local_variables.len()],
        };
        for (local, variable) in function.local_variables.iter() {
            if let Some(init) = variable.init {
                let value = invocation.expression(init)?;
                invocation.locals[local.index()] = Some(value);
            }
        }
        invocation.block(&function.body)
    }
    fn grouped_dispatch(
        module: &n::Module,
        params: &[[u32; 4]],
        status: &[Vec<u32>],
        prior: u32,
    ) -> (u32, BTreeSet<(usize, usize)>) {
        let width = params.iter().map(|p| p[3]).max().unwrap().div_ceil(64) * 64;
        let count = params.iter().map(|p| p[0]).max().unwrap();
        let mut fault = prior;
        let mut touched = BTreeSet::new();
        for y in 0..count {
            for x in 0..width {
                invoke(module, params, status, [x, y, 0], &mut fault, &mut touched).unwrap();
            }
        }
        (fault, touched)
    }
    // Independent original schedule: each scalar fold visits only the active
    // width of its own strided rows, with no shared maximum or rounded grid.
    fn original_scalar_folds(
        params: &[[u32; 4]],
        status: &[Vec<u32>],
        prior: u32,
    ) -> (u32, BTreeSet<(usize, usize)>) {
        let mut fault = prior;
        let mut touched = BTreeSet::new();
        for (member, &[count, first, stride, width]) in params.iter().enumerate() {
            for (token, row) in status[member][first as usize..]
                .chunks(stride as usize)
                .take(count as usize)
                .enumerate()
            {
                for (element, &word) in row.iter().take(width as usize).enumerate() {
                    touched.insert((member, first as usize + token * stride as usize + element));
                    if word != 1 {
                        fault |= 2;
                    }
                }
            }
        }
        (fault, touched)
    }

    #[test]
    fn generated_status3_5_8_memory_and_exact_active_spans_match_original_folds() {
        let widths = [1, 2, 3, 5, 63, 64, 65, 129, 257];
        for members in [3, 5, 8] {
            let module = compiled_status_source(&Kernel(members as u8).source(), members);
            for count in [1, 2, 3, 5, 8] {
                for offset in 0..3 {
                    let params: Vec<[u32; 4]> = (0..members)
                        .map(|i| {
                            let width = widths[(i + offset) % widths.len()];
                            [
                                count - i as u32 % count,
                                (i + offset) as u32 % 4,
                                width + i as u32 % 3 + 1,
                                width,
                            ]
                        })
                        .collect();
                    let mut status: Vec<Vec<u32>> = params
                        .iter()
                        .enumerate()
                        .map(|(i, &[active, first, stride, width])| {
                            let mut owner =
                                vec![[0, 9, u32::MAX][i % 3]; (first + 8 * stride + 7) as usize];
                            for row in owner[first as usize..]
                                .chunks_mut(stride as usize)
                                .take(active as usize)
                            {
                                row[..width as usize].fill(1);
                            }
                            crate::resident_batch_projection::validate_view(
                                active as usize,
                                first as usize,
                                stride as usize,
                                width as usize,
                                owner.len(),
                            )
                            .unwrap();
                            owner
                        })
                        .collect();
                    assert_eq!(
                        grouped_dispatch(&module, &params, &status, 0),
                        original_scalar_folds(&params, &status, 0)
                    );
                    for member in 0..members {
                        let [active, first, stride, width] = params[member];
                        for (x, y) in [(0, 0), (width - 1, active - 1)] {
                            let index = (first + y * stride + x) as usize;
                            for bad in [0, 2, u32::MAX] {
                                status[member][index] = bad;
                                for prior in [0, 8, u32::MAX] {
                                    let mut fault = prior;
                                    invoke(
                                        &module,
                                        &params,
                                        &status,
                                        [x, y, 0],
                                        &mut fault,
                                        &mut BTreeSet::new(),
                                    )
                                    .unwrap();
                                    assert_eq!(
                                        fault,
                                        original_scalar_folds(&params, &status, prior).0
                                    );
                                }
                            }
                            status[member][index] = 1;
                        }
                    }
                    for (owner, &[active, first, stride, width]) in status.iter_mut().zip(&params) {
                        owner[(first + (active - 1) * stride + width - 1) as usize] = 0;
                    }
                    assert_eq!(
                        grouped_dispatch(&module, &params, &status, 8),
                        original_scalar_folds(&params, &status, 8)
                    );
                }
            }
        }
    }

    #[test]
    fn generated_guard_proof_rejects_a_hoisted_inactive_storage_read() {
        let source = Kernel(3).source();
        let read = "status0[params[0].first+id.y*params[0].stride+id.x]";
        let hoisted = source.replacen(read, "eager_status0", 1).replacen(
            "id:vec3<u32>) {\n",
            &format!("id:vec3<u32>) {{\nlet eager_status0={read};\n"),
            1,
        );
        assert_ne!(source, hoisted);
        let module = compiled_status_source(&hoisted, 3);
        let params = [[1, 0, 1, 1], [3, 0, 65, 65], [2, 0, 3, 3]];
        let status = [vec![1; 8], vec![1; 8 * 65], vec![1; 8 * 3]];
        for id in [[1, 0, 0], [0, 1, 0]] {
            assert_eq!(
                invoke(&module, &params, &status, id, &mut 0, &mut BTreeSet::new()),
                Err("inactive status read")
            );
        }
    }

    #[test]
    fn original_active_status_predicates_and_uniform_layout_validate_offline() {
        for n in [3, 5, 8] {
            let s = Kernel(n).source();
            let module = wgpu::naga::front::wgsl::parse_str(&s).unwrap();
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(s.matches("atomicOr(&fault[0],2u)").count(), n as usize);
            assert_eq!(s.matches("id.y<params[").count(), n as usize);
            assert!(!s.contains("status0[0]="));
        }
    }
}
