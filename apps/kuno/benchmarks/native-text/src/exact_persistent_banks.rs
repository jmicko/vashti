//! UNUSED fixed DAS144 allocation leaf. Quotes, actual root publication and
//! construction-origin authority are deliberately distinct. No admission,
//! loaded-weight, source attachment, heap or healthy-recovery credit lives here.
use super::{ExpertBanks, OriginalPackedProfile};
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::{WgpuResource, WgpuRuntime};
use cubecl_common::stream_id::StreamId;
use cubecl_runtime::{
    config::{CubeClRuntimeConfig, RuntimeConfig, memory::PersistentMemory},
    memory_management::MemoryAllocationMode,
};
use kuno_text::{
    Error, GgufFile, Result,
    model::{BoundGraph, RoutedWeights},
};
use std::{
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

const ROOTS: usize = 6;
const SLOTS: usize = 144;
const SEGMENT: usize = 72;
const MAX_RESIDENT_ROOT: u64 = 64 * 1024 * 1024;
const MAX_ALIGNMENT: u64 = 1024 * 1024;

fn rejected() -> Error {
    Error::InvalidFormat("exact persistent DAS144 bank rejected")
}

/// A checked metadata quote; Copy does not confer allocation/admission credit.
#[derive(Clone, Copy)]
pub(crate) struct ExactBankQuote {
    layer: usize,
    alignment: u64,
    logical: [u64; ROOTS],
    backing: [u64; ROOTS],
    total: u64,
}
impl ExactBankQuote {
    fn from_metadata(
        layer: usize,
        alignment: u64,
        metadata: [(u32, [u64; 3], u64, u64); 3],
    ) -> Result<Self> {
        let expected = crate::daslab_profile::routed_types(layer).ok_or_else(rejected)?;
        if !(4..=MAX_ALIGNMENT).contains(&alignment) || !alignment.is_power_of_two() {
            return Err(rejected());
        }
        let mut logical = [0; ROOTS];
        let mut backing = [0; ROOTS];
        let mut total = 0u64;
        for (role, (dtype, dimensions, row_bytes, tensor_bytes)) in metadata.into_iter().enumerate()
        {
            let dims = if role == 2 {
                [640, 2560, 512]
            } else {
                [2560, 640, 512]
            };
            let format = kuno_text::TensorType::from_gguf_id(dtype)?;
            if dtype != expected[role]
                || dimensions != dims
                || row_bytes != format.row_bytes(dims[0])?
            {
                return Err(rejected());
            }
            let expert = row_bytes
                .checked_mul(dims[1])
                .ok_or(Error::ArithmeticOverflow)?;
            if expert.checked_mul(512) != Some(tensor_bytes) {
                return Err(rejected());
            }
            let bytes = expert
                .checked_mul(SEGMENT as u64)
                .ok_or(Error::ArithmeticOverflow)?;
            let actual = aligned(bytes, alignment)?;
            for part in 0..2 {
                let index = role * 2 + part;
                logical[index] = bytes;
                backing[index] = actual;
                total = total.checked_add(actual).ok_or(Error::ArithmeticOverflow)?;
            }
        }
        Ok(Self {
            layer,
            alignment,
            logical,
            backing,
            total,
        })
    }
    /// Pure arithmetic only. Parent must supply its authenticated original
    /// admission upper/debt/limit and retain every other owner and margin.
    pub(crate) fn precharged_upper(&self, upper: u64, limit: u64) -> Result<u64> {
        let next = upper
            .checked_add(self.total)
            .ok_or(Error::ArithmeticOverflow)?;
        if next > limit {
            return Err(Error::LimitExceeded("exact bank quote capacity"));
        }
        Ok(next)
    }
    pub(crate) fn backing_requests(&self) -> &[u64; ROOTS] {
        &self.backing
    }
    pub(crate) fn total_bytes(&self) -> u64 {
        self.total
    }
}
fn aligned(bytes: u64, alignment: u64) -> Result<u64> {
    if bytes == 0
        || !bytes.is_multiple_of(4)
        || !(4..=MAX_ALIGNMENT).contains(&alignment)
        || !alignment.is_power_of_two()
    {
        return Err(rejected());
    }
    let extent = bytes
        .checked_add(alignment - 1)
        .ok_or(Error::ArithmeticOverflow)?
        & !(alignment - 1);
    if extent > MAX_RESIDENT_ROOT {
        return Err(rejected());
    }
    Ok(extent)
}
fn enabled(config: &CubeClRuntimeConfig) -> Result<()> {
    if !matches!(config.memory.persistent_memory, PersistentMemory::Enabled) {
        return Err(rejected());
    }
    Ok(())
}

/// Private construction-origin witness, never derived from metadata alone.
/// No Send/Clone/serialization: scope is the unchanged initialization stream.
pub(crate) struct ConstructionOrigin<'a> {
    client: &'a ComputeClient<WgpuRuntime>,
    config: Arc<CubeClRuntimeConfig>,
    stream: StreamId,
    alignment: u64,
    retired: &'a AtomicBool,
    _current_thread: PhantomData<Rc<()>>,
}
impl<'a> ConstructionOrigin<'a> {
    /// # Safety
    /// Call at fresh pinned Wgpu client initialization using this SAME Enabled
    /// snapshot, before any unsafe client stream override. Parent exclusively
    /// owns the current stream/client allocation and mode scope until return;
    /// no concurrent allocation/stream/property/mode mutation is allowed.
    /// `retired` is the original context's mandatory retirement authority.
    /// Global config/current stream alone do NOT prove these obligations for
    /// an arbitrary already-existing client.
    pub(crate) unsafe fn from_fresh_current_client(
        client: &'a ComputeClient<WgpuRuntime>,
        initialized_config: Arc<CubeClRuntimeConfig>,
        retired: &'a AtomicBool,
    ) -> Result<Self> {
        enabled(&initialized_config)?;
        if !Arc::ptr_eq(&initialized_config, &CubeClRuntimeConfig::get())
            || retired.load(Ordering::Acquire)
        {
            return Err(rejected());
        }
        let alignment = client.properties().memory.alignment;
        aligned(4, alignment)?;
        Ok(Self {
            client,
            config: initialized_config,
            stream: StreamId::current(),
            alignment,
            retired,
            _current_thread: PhantomData,
        })
    }
    fn check(&self) -> Result<()> {
        enabled(&self.config)?;
        if self.retired.load(Ordering::Acquire)
            || StreamId::current() != self.stream
            || self.client.properties().memory.alignment != self.alignment
            || !Arc::ptr_eq(&self.config, &CubeClRuntimeConfig::get())
        {
            return Err(rejected());
        }
        Ok(())
    }
}

pub(crate) struct ExactBankPlan<'model, 'file> {
    graph: BoundGraph<'model, 'file>,
    file: &'file GgufFile,
    weights: RoutedWeights<'file>,
    quote: ExactBankQuote,
}
impl<'model, 'file> ExactBankPlan<'model, 'file> {
    pub(crate) fn for_layer(
        origin: &ConstructionOrigin<'_>,
        graph: BoundGraph<'model, 'file>,
        file: &'file GgufFile,
        layer: usize,
    ) -> Result<Self> {
        origin.check()?;
        if file.shard_count() != 2 || file.tensors().len() != crate::daslab_profile::TENSOR_COUNT {
            return Err(rejected());
        }
        OriginalPackedProfile::DaslabIq3.validate_graph(graph)?;
        let weights = super::routed_weights(graph, layer)?;
        let tensors = [weights.gate, weights.up, weights.down];
        if tensors.iter().any(|t| !t.belongs_to(file)) {
            return Err(rejected());
        }
        let metadata = tensors.map(|t| -> Result<_> {
            let info = t.info();
            let dims: [u64; 3] = info.dimensions().try_into().map_err(|_| rejected())?;
            Ok((
                info.dtype().gguf_id(),
                dims,
                info.row_bytes(),
                info.byte_len(),
            ))
        });
        let [gate, up, down] = metadata;
        let quote = ExactBankQuote::from_metadata(layer, origin.alignment, [gate?, up?, down?])?;
        file.verify_file_unchanged()?;
        Ok(Self {
            graph,
            file,
            weights,
            quote,
        })
    }
    pub(crate) fn quote(&self) -> &ExactBankQuote {
        &self.quote
    }

    /// Parent precharge callback runs before ANY bank root allocation. It must
    /// charge exact requests to its original held admission owner, preserving
    /// all other debts/caps/margins; callback success is not leaf proof of that
    /// accounting. Allocation/source publication remains unused until a future
    /// executing parent joins the unsafe origin and mandatory close lifecycle.
    pub(crate) fn allocate(
        self,
        origin: &mut ConstructionOrigin<'_>,
        precharge: impl FnOnce(&ExactBankQuote) -> Result<()>,
    ) -> Result<ExactAllocatedBank> {
        origin.check()?;
        if self.quote.alignment != origin.alignment {
            return Err(rejected());
        }
        self.graph.verify_unchanged()?;
        self.file.verify_file_unchanged()?;
        precharge(&self.quote)?;
        origin.check()?;
        let client = origin.client;
        let bank = persistent_scope(
            |mode| {
                // SAFETY: private unsafe construction-origin witness requires
                // unchanged current client stream and serialized MAIN scope.
                unsafe {
                    client.allocation_mode(mode);
                }
                Ok(())
            },
            origin.retired,
            || {
                ExpertBanks::new_with_profile(
                    client,
                    self.weights,
                    SLOTS,
                    OriginalPackedProfile::DaslabIq3,
                )
            },
        )?;
        let validated = catch_unwind(AssertUnwindSafe(|| {
            origin.check()?;
            self.graph.verify_unchanged()?;
            self.file.verify_file_unchanged()?;
            bank.check_client(client)?;
            bank.check_names(self.weights)?;
            let mut roots: Vec<WgpuResource> = Vec::with_capacity(ROOTS);
            if roots.capacity() != ROOTS {
                return Err(rejected());
            }
            for (role, owner) in [bank.gate(), bank.up(), bank.down()]
                .into_iter()
                .enumerate()
            {
                if owner.segments().len() != 2 {
                    return Err(rejected());
                }
                for (part, segment) in owner.segments().iter().enumerate() {
                    let i = role * 2 + part;
                    if segment.first_slot() != part * SEGMENT || segment.capacity() != SEGMENT {
                        return Err(rejected());
                    }
                    let h = segment.handle();
                    validate_handle(&self.quote, i, origin.stream, h, segment.words())?;
                    let resource = client.get_resource(h.clone()).map_err(|_| rejected())?;
                    let r = resource.resource();
                    validate_extent(&self.quote, i, r.offset, r.size, r.buffer.size())?;
                    if roots.iter().any(|old| old.buffer == r.buffer) {
                        return Err(rejected());
                    }
                    roots.push(r.clone());
                }
            }
            let roots: [WgpuResource; ROOTS] = roots.try_into().map_err(|_| rejected())?;
            Ok(roots)
        }));
        match validated {
            Ok(Ok(roots)) => Ok(ExactAllocatedBank {
                bank,
                roots,
                quote: self.quote,
            }),
            Ok(Err(error)) => {
                origin.retired.store(true, Ordering::Release);
                // No release credit: parent must join the original close proof.
                Err(error)
            }
            Err(_) => {
                origin.retired.store(true, Ordering::Release);
                Err(rejected())
            }
        }
    }
}

/// Actual retained six-root ownership, not a heap/source/close certificate.
pub(crate) struct ExactAllocatedBank {
    bank: ExpertBanks,
    roots: [WgpuResource; ROOTS],
    quote: ExactBankQuote,
}
impl ExactAllocatedBank {
    pub(crate) fn bank(&self) -> &ExpertBanks {
        &self.bank
    }
    pub(crate) fn actual_buffer_bytes(&self) -> Result<u64> {
        self.roots.iter().enumerate().try_fold(0u64, |n, (i, r)| {
            validate_extent(&self.quote, i, r.offset, r.size, r.buffer.size())?;
            n.checked_add(r.buffer.size())
                .ok_or(Error::ArithmeticOverflow)
        })
    }
}
fn validate_extent(
    quote: &ExactBankQuote,
    i: usize,
    offset: u64,
    used: u64,
    backing: u64,
) -> Result<()> {
    if i >= ROOTS || offset != 0 || used != quote.logical[i] || backing != quote.backing[i] {
        return Err(rejected());
    }
    Ok(())
}
fn validate_handle(
    quote: &ExactBankQuote,
    i: usize,
    stream: StreamId,
    h: &Handle,
    words: usize,
) -> Result<()> {
    // Reject offsets BEFORE size_in_used(), which subtracts unchecked forged
    // offsets. Both None and Some(0) represent the original full owner.
    if i >= ROOTS || h.offset_start.unwrap_or(0) != 0 || h.offset_end.unwrap_or(0) != 0 {
        return Err(rejected());
    }
    if h.stream != stream
        || h.size() != quote.logical[i]
        || h.size_in_used() != quote.logical[i]
        || u64::try_from(words).ok().and_then(|n| n.checked_mul(4)) != Some(quote.logical[i])
        || cubecl_runtime::public_external::is_public_external_binding(&h.clone().binding())
    {
        return Err(rejected());
    }
    Ok(())
}

/// MAIN-only original mode API; catch task panic before restoring Auto.
/// Failed restoration retires too. It is an ordered mode restoration, never
/// a device completion, free, erasure, or healthy-recovery proof.
pub(crate) fn persistent_scope<T>(
    mut mode: impl FnMut(MemoryAllocationMode) -> Result<()>,
    retired: &AtomicBool,
    task: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        mode(MemoryAllocationMode::Persistent)?;
        task()
    }));
    let restored = catch_unwind(AssertUnwindSafe(|| mode(MemoryAllocationMode::Auto)));
    match (result, restored) {
        (Ok(Ok(value)), Ok(Ok(()))) => Ok(value),
        (Ok(Err(error)), Ok(Ok(()))) => {
            retired.store(true, Ordering::Release);
            Err(error)
        }
        _ => {
            retired.store(true, Ordering::Release);
            Err(rejected())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn metadata(layer: usize) -> [(u32, [u64; 3], u64, u64); 3] {
        // Independent inspected original public fixture, not routed_types or
        // the allocator's quote math. No GGUF/model payload is read.
        static RECEIPT: std::sync::OnceLock<serde_json::Value> = std::sync::OnceLock::new();
        let receipt = RECEIPT.get_or_init(|| {
            serde_json::from_str(include_str!(
                "../../text/receipts/daslab-iq3-artifact-20261009.json"
            ))
            .unwrap()
        });
        ["gate", "up", "down"].map(|role| {
            let name = format!("blk.{layer}.ffn_{role}_exps.weight");
            let t = receipt["tensor_inventory"]
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"].as_str() == Some(&name))
                .unwrap();
            let dimensions = std::array::from_fn(|i| t["dimensions"][i].as_u64().unwrap());
            (
                t["type_id"].as_u64().unwrap() as u32,
                dimensions,
                t["row_bytes"].as_u64().unwrap(),
                t["tensor_bytes"].as_u64().unwrap(),
            )
        })
    }
    #[test]
    fn exact_persistent_all48_original_roots_and_six_formats() {
        let mut total = 0;
        let mut formats = std::collections::BTreeSet::new();
        for layer in 0..48 {
            let m = metadata(layer);
            formats.extend(m.iter().map(|m| m.0));
            let q = ExactBankQuote::from_metadata(layer, 256, m).unwrap();
            assert_eq!(q.layer, layer);
            assert_eq!(q.logical, q.backing);
            assert!(q.backing.iter().all(|&n| n <= MAX_RESIDENT_ROOT));
            for (role, metadata) in m.iter().enumerate() {
                assert_eq!(q.logical[2 * role], metadata.3 / 512 * 72);
            }
            total += q.total;
        }
        assert_eq!(
            formats.into_iter().collect::<Vec<_>>(),
            [18, 20, 21, 22, 23, 42]
        );
        assert_eq!(total, 14_144_716_800u64);
    }
    #[test]
    fn exact_persistent_quotes_reject_source_alignment_extent_and_overflow() {
        for alignment in [0, 1, 2, 3, 12, MAX_ALIGNMENT + 4, u64::MAX] {
            assert!(ExactBankQuote::from_metadata(0, alignment, metadata(0)).is_err());
        }
        for (bytes, alignment) in [
            (0, 256),
            (3, 256),
            (MAX_RESIDENT_ROOT + 4, 256),
            (u64::MAX - 3, 256),
        ] {
            assert!(aligned(bytes, alignment).is_err());
        }
        assert_eq!(aligned(4, 32768).unwrap(), 32768);
        assert!(ExactBankQuote::from_metadata(48, 256, metadata(0)).is_err());
        let m = metadata(0);
        for role in 0..3 {
            for field in 0..4 {
                let mut bad = m;
                match field {
                    0 => bad[role].0 = 0,
                    1 => bad[role].1[2] -= 1,
                    2 => bad[role].2 += 4,
                    _ => bad[role].3 -= 4,
                }
                assert!(ExactBankQuote::from_metadata(0, 256, bad).is_err());
            }
        }
        let q = ExactBankQuote::from_metadata(0, 256, m).unwrap();
        assert_eq!(q.precharged_upper(1, q.total + 1).unwrap(), q.total + 1);
        assert!(q.precharged_upper(1, q.total).is_err());
        assert!(q.precharged_upper(u64::MAX, u64::MAX).is_err());
    }
    #[test]
    fn exact_persistent_actual_root_extent_is_not_an_unallocated_quote() {
        let q = ExactBankQuote::from_metadata(0, 4096, metadata(0)).unwrap();
        for i in 0..ROOTS {
            validate_extent(&q, i, 0, q.logical[i], q.backing[i]).unwrap();
            for (offset, used, backing) in [
                (4, q.logical[i], q.backing[i]),
                (0, q.logical[i] - 4, q.backing[i]),
                (0, q.logical[i], q.backing[i] + 4),
            ] {
                assert!(validate_extent(&q, i, offset, used, backing).is_err());
            }
            // Original metadata handles only: no GPU allocation or completion.
            let stream = StreamId { value: 17 };
            let words = usize::try_from(q.logical[i] / 4).unwrap();
            let h = Handle::new(stream, q.logical[i]);
            validate_handle(&q, i, stream, &h, words).unwrap();
            validate_handle(
                &q,
                i,
                stream,
                &h.clone().offset_start(0).offset_end(0),
                words,
            )
            .unwrap();
            for offset in [4, u64::MAX] {
                assert!(
                    validate_handle(&q, i, stream, &h.clone().offset_start(offset), words).is_err()
                );
                assert!(
                    validate_handle(&q, i, stream, &h.clone().offset_end(offset), words).is_err()
                );
            }
            assert!(validate_handle(&q, i, StreamId { value: 18 }, &h, words).is_err());
            assert!(
                validate_handle(&q, i, stream, &Handle::new(stream, q.logical[i] - 4), words)
                    .is_err()
            );
            assert!(validate_handle(&q, i, stream, &h, words - 1).is_err());
            assert!(validate_handle(&q, i, stream, &h, usize::MAX).is_err());
        }
        assert!(validate_extent(&q, ROOTS, 0, 0, 0).is_err());
    }
    #[test]
    fn exact_persistent_scope_restores_auto_before_return_error_or_panic() {
        for outcome in 0..3 {
            let events = RefCell::new(Vec::new());
            let retired = AtomicBool::new(false);
            let result = persistent_scope(
                |mode| {
                    events.borrow_mut().push(match mode {
                        MemoryAllocationMode::Auto => 0,
                        MemoryAllocationMode::Persistent => 1,
                    });
                    Ok(())
                },
                &retired,
                || {
                    events.borrow_mut().push(2);
                    match outcome {
                        0 => Ok(7),
                        1 => Err(rejected()),
                        _ => panic!("fixed public panic fixture"),
                    }
                },
            );
            assert_eq!(*events.borrow(), [1, 2, 0]);
            assert_eq!(result.is_ok(), outcome == 0);
            assert_eq!(retired.load(Ordering::Acquire), outcome != 0);
        }
        let retired = AtomicBool::new(false);
        assert!(
            persistent_scope(
                |mode| if matches!(mode, MemoryAllocationMode::Auto) {
                    Err(rejected())
                } else {
                    Ok(())
                },
                &retired,
                || Ok(())
            )
            .is_err()
        );
        assert!(retired.load(Ordering::Acquire));
        for fail_persistent in [true, false] {
            for panic in [false, true] {
                let events = RefCell::new(Vec::new());
                let retired = AtomicBool::new(false);
                let result = persistent_scope(
                    |mode| {
                        let persistent = matches!(mode, MemoryAllocationMode::Persistent);
                        events.borrow_mut().push(persistent);
                        if persistent == fail_persistent {
                            if panic {
                                panic!("fixed public mode panic fixture");
                            }
                            return Err(rejected());
                        }
                        Ok(())
                    },
                    &retired,
                    || Ok(()),
                );
                assert!(result.is_err());
                assert!(retired.load(Ordering::Acquire));
                assert_eq!(*events.borrow(), [true, false]);
            }
        }
    }
    #[test]
    fn exact_persistent_disabled_enforced_and_unsafe_origin_are_not_inferred() {
        let mut config = CubeClRuntimeConfig::default();
        assert!(enabled(&config).is_ok());
        for policy in [PersistentMemory::Disabled, PersistentMemory::Enforced] {
            config.memory.persistent_memory = policy;
            assert!(enabled(&config).is_err());
        }
        let source = include_str!("exact_persistent_banks.rs")
            .split("#[cfg(test)]")
            .next()
            .unwrap();
        assert!(source.contains("unsafe fn from_fresh_current_client"));
        assert!(source.contains("precharge(&self.quote)?;"));
        assert!(
            source.find("precharge(&self.quote)?;").unwrap()
                < source.find("let bank = persistent_scope(").unwrap()
        );
        assert!(source.contains("fn validate_handle("));
        assert!(source.contains("h.stream != stream"));
        assert!(source.contains("old.buffer == r.buffer"));
        assert!(
            source.find("h.offset_start.unwrap_or(0) != 0").unwrap()
                < source.find("h.size_in_used()").unwrap()
        );
    }
}
