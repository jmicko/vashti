//! Transient packed expert banks for a fixed public laboratory experiment.
//! Stores original weights only; never persists expert access/rank history.
use crate::bank_slots::{BankSlots, routed_capacity, segmented_capacity};
use crate::quant_matvec;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::{BoundGraph, RoutedWeights};
use kuno_text::{Error, Result, TensorType, TensorView};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use zeroize::Zeroize;

#[path = "exact_persistent_banks.rs"]
#[allow(dead_code)] // Source-only fixed144 allocation leaf; no selected consumer.
pub(crate) mod exact_persistent;

/// Explicit original-weight laboratory profile. Selection belongs to the
/// constructor; never infer a profile from a failed admission or source read.
/// Neither variant changes route coverage, slot lifetime or completion gates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum OriginalPackedProfile {
    #[default]
    Q4,
    UnslothQ3,
    DaslabIq3,
}
pub(crate) const UNSLOTH_Q3_BANK_BYTES_144: usize = 19_998_441_472;

impl OriginalPackedProfile {
    pub(crate) fn is_low_bit(self) -> bool {
        matches!(self, Self::UnslothQ3 | Self::DaslabIq3)
    }
    pub(crate) fn validate_capacity(self, capacity: usize) -> Result<()> {
        validate_capacity(capacity)?;
        if (self == Self::UnslothQ3 && capacity != 144)
            || (self == Self::DaslabIq3 && capacity != crate::daslab_profile::BANK_SLOTS)
        {
            return Err(Error::InvalidFormat(
                "Unsloth Q3 requires 144 original slots",
            ));
        }
        Ok(())
    }
    /// IQ support first uses the original generic grouped kernels. The Q4
    /// shared/fused/integer/word/block implementations are separate algorithms.
    pub(crate) fn validate_shortcuts(
        self,
        shared_moe: bool,
        fused: bool,
        integer: bool,
        word: bool,
        block: bool,
    ) -> Result<()> {
        if self.is_low_bit() && (shared_moe || fused || integer || word || block) {
            return Err(Error::InvalidFormat(
                "Unsloth Q3 requires generic original packed MoE",
            ));
        }
        Ok(())
    }
    pub(crate) fn ple_row_bytes(self) -> usize {
        match self {
            Self::Q4 => 110,
            Self::UnslothQ3 | Self::DaslabIq3 => 90,
        }
    }
    pub(crate) fn validate_ple(
        self,
        dtype: TensorType,
        dimensions: &[u64],
        row_bytes: u64,
    ) -> Result<()> {
        let expected = match self {
            Self::Q4 => TensorType::Q5_0,
            Self::UnslothQ3 | Self::DaslabIq3 => TensorType::Iq4Nl,
        };
        if dtype != expected
            || dimensions != [160, 320001536]
            || row_bytes != self.ple_row_bytes() as u64
        {
            return Err(Error::InvalidFormat("original packed profile PLE source"));
        }
        Ok(())
    }
    /// Validates every bound main-graph tensor before model/device allocation.
    /// Q4 keeps its existing admission contract; Q3 is the inspected Unsloth
    /// mixed profile, not a general low-bit or requantized-weight fallback.
    pub(crate) fn validate_graph(self, graph: BoundGraph<'_, '_>) -> Result<()> {
        if self == Self::Q4 {
            return Ok(());
        }
        graph.verify_unchanged()?;
        if self == Self::DaslabIq3 {
            if graph.geometry().layers != crate::daslab_profile::LAYER_COUNT
                || graph.geometry().experts != crate::daslab_profile::EXPERT_COUNT
            {
                return Err(Error::InvalidFormat("DASLab original graph geometry"));
            }
            let inventory = graph
                .tensor_names()
                .map(|name| Ok((name, graph.tensor(name)?.info().dtype().gguf_id())))
                .collect::<Result<Vec<_>>>()?;
            crate::daslab_profile::validate_inventory(inventory)
                .map_err(|_| Error::InvalidFormat("DASLab original tensor inventory"))?;
        }
        if graph.geometry().layers != 48 {
            return Err(Error::InvalidFormat(
                "Unsloth Q3 requires 48 original layers",
            ));
        }
        for name in graph.tensor_names() {
            let tensor = graph.tensor(name)?;
            let info = tensor.info();
            if name.ends_with("_exps.weight") {
                // Complete triples and canonical per-layer names are checked
                // below, before any bank allocation or source staging.
                continue;
            }
            let expected = if self == Self::DaslabIq3 {
                TensorType::from_gguf_id(
                    crate::daslab_profile::expected_dtype(name)
                        .ok_or(Error::InvalidFormat("DASLab original tensor name"))?,
                )?
            } else {
                q3_main_dtype(name, info.dimensions())
            };
            if info.dtype() != expected {
                return Err(Error::InvalidFormat(
                    "Unsloth Q3 original main tensor dtype",
                ));
            }
            if name == "per_layer_token_embd.weight" {
                self.validate_ple(info.dtype(), info.dimensions(), info.row_bytes())?;
            }
        }
        for layer in 0..48 {
            let weights = routed_weights(graph, layer)?;
            validate_profile_weights(self, weights)?;
        }
        graph.verify_unchanged()
    }
}

fn q3_main_dtype(name: &str, dimensions: &[u64]) -> TensorType {
    if name == "output.weight" {
        TensorType::Q6K
    } else if name == "per_layer_token_embd.weight" {
        TensorType::Iq4Nl
    } else if name.ends_with("indexer.q_proj.weight") || name.ends_with("indexer.k_proj.weight") {
        TensorType::Bf16
    } else if dimensions.len() == 1
        || name.ends_with("ffn_gate_inp.weight")
        || name.ends_with("hc_attn_inject.weight")
        || name.ends_with("hc_ffn_inject.weight")
        || name.ends_with("ssm_alpha.weight")
        || name.ends_with("ssm_beta.weight")
        || name.ends_with("ssm_conv1d.weight")
        || name.ends_with("ple_conv1d.weight")
    {
        TensorType::F32
    } else {
        TensorType::Q8_0
    }
}

fn q3_expert_types(layer: usize) -> Result<[TensorType; 3]> {
    if layer >= 48 {
        return Err(Error::OutOfBounds);
    }
    let gate = if layer == 2 {
        TensorType::Iq4Xs
    } else {
        TensorType::Iq3Xxs
    };
    let down = if matches!(layer, 2 | 4 | 30 | 46 | 47) {
        TensorType::Q8_0
    } else {
        TensorType::Iq4Nl
    };
    Ok([gate, gate, down])
}

fn routed_weights<'a>(graph: BoundGraph<'_, 'a>, layer: usize) -> Result<RoutedWeights<'a>> {
    Ok(RoutedWeights {
        gate: graph.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        up: graph.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        down: graph.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    })
}

fn validate_profile_metadata(
    profile: OriginalPackedProfile,
    layer: usize,
    parts: [(TensorType, &[u64]); 3],
) -> Result<()> {
    if profile == OriginalPackedProfile::Q4 {
        return validate_segmented_metadata(parts);
    }
    let expected = if profile == OriginalPackedProfile::DaslabIq3 {
        crate::daslab_profile::routed_types(layer)
            .ok_or(Error::OutOfBounds)?
            .map(TensorType::from_gguf_id)
            .into_iter()
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::InvalidFormat("DASLab original routed triple"))?
    } else {
        q3_expert_types(layer)?
    };
    for (index, (dtype, dimensions)) in parts.into_iter().enumerate() {
        if dtype != expected[index]
            || dimensions
                != if index == 2 {
                    &[640, 2560, 512]
                } else {
                    &[2560, 640, 512]
                }
        {
            return Err(Error::InvalidFormat("Unsloth Q3 original expert metadata"));
        }
    }
    Ok(())
}

fn validate_profile_weights(
    profile: OriginalPackedProfile,
    weights: RoutedWeights<'_>,
) -> Result<()> {
    let name = weights.gate.info().name();
    let layer = name
        .strip_prefix("blk.")
        .and_then(|s| s.strip_suffix(".ffn_gate_exps.weight"))
        .and_then(|s| s.parse::<usize>().ok())
        .ok_or(Error::InvalidFormat("original packed profile layer name"))?;
    for (tensor, suffix) in [weights.gate, weights.up, weights.down]
        .into_iter()
        .zip(["gate", "up", "down"])
    {
        if tensor.info().name() != format!("blk.{layer}.ffn_{suffix}_exps.weight") {
            return Err(Error::InvalidFormat("original packed profile layer name"));
        }
    }
    validate_profile_metadata(
        profile,
        layer,
        [weights.gate, weights.up, weights.down].map(|t| (t.info().dtype(), t.info().dimensions())),
    )
}

/// Immutable allocation extent; expose only readonly handles/geometry. Filling
/// changes PUBLIC weight bytes, never these allocations or their offsets.
pub struct BankSegment {
    handle: Handle,
    first_slot: usize,
    capacity: usize,
    words: usize,
}
impl BankSegment {
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
    pub fn first_slot(&self) -> usize {
        self.first_slot
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn words(&self) -> usize {
        self.words
    }
}
pub struct Bank {
    segments: Vec<BankSegment>,
    expert_bytes: usize,
    words: usize,
    reserved: usize,
}
impl Bank {
    pub fn single(&self) -> Result<&BankSegment> {
        if self.segments.len() != 1 {
            return Err(Error::InvalidFormat(
                "segmented bank requires resident block MoE",
            ));
        }
        Ok(&self.segments[0])
    }
    pub fn segments(&self) -> &[BankSegment] {
        &self.segments
    }
    fn resolve(&self, slot: usize) -> Result<(&BankSegment, usize)> {
        let index = resolve_segment(
            self.segments.iter().map(|s| (s.first_slot, s.capacity)),
            slot,
        )?;
        let segment = &self.segments[index];
        Ok((segment, slot - segment.first_slot))
    }
    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }
    pub fn words(&self) -> usize {
        self.words
    }
}
fn resolve_segment(parts: impl IntoIterator<Item = (usize, usize)>, slot: usize) -> Result<usize> {
    parts
        .into_iter()
        .enumerate()
        .find_map(|(i, (first, count))| {
            first
                .checked_add(count)
                .filter(|&end| slot >= first && slot < end)
                .map(|_| i)
        })
        .ok_or(Error::InvalidBufferLength)
}
static NEXT_BANK_INSTANCE: AtomicU64 = AtomicU64::new(1);
/// Opaque constructor-only allocation generation. No pointer/allocator reuse
/// can recreate a previous instance, even if the same model slots are rebuilt.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct BankInstance(u64);
fn next_instance(counter: &AtomicU64) -> Result<BankInstance> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .map(BankInstance)
        .map_err(|_| Error::ArithmeticOverflow)
}

pub struct ExpertBanks {
    gate: Bank,
    up: Bank,
    down: Bank,
    instance: BankInstance,
    pub hidden: usize,
    pub ff: usize,
    pub experts: usize,
    pub slots: BankSlots,
    names: [String; 3],
    origin: ComputeClient<WgpuRuntime>,
    profile: OriginalPackedProfile,
}

#[cfg(test)]
pub(crate) fn fixture_bank_instance() -> BankInstance {
    next_instance(&NEXT_BANK_INSTANCE).expect("public test bank identity")
}

fn segment_capacities(capacity: usize) -> Result<Vec<usize>> {
    validate_capacity(capacity)?;
    Ok(
        if let Some(parts) = crate::bank_slots::segmented_capacities(capacity) {
            parts.to_vec()
        } else {
            vec![capacity]
        },
    )
}
fn bank_segments(t: TensorView<'_>, capacity: usize) -> Result<(usize, Vec<usize>)> {
    if t.info().dimensions().len() != 3 {
        return Err(Error::InvalidFormat("expert bank rank"));
    }
    packed_segments(
        t.info().row_bytes() as usize,
        t.info().dimensions()[1] as usize,
        capacity,
    )
}
fn packed_segments(row_bytes: usize, rows: usize, capacity: usize) -> Result<(usize, Vec<usize>)> {
    let mut expert = 0;
    let mut bytes = Vec::new();
    for count in segment_capacities(capacity)? {
        let (e, b) = packed_bank_bytes(row_bytes, rows, count)?;
        expert = e;
        bytes.push(b);
    }
    Ok((expert, bytes))
}
/// Public metadata restriction for the focused 64+32 experiment. This is not a
/// permissive generic bank: all 48 original triples must match before allocation.
fn validate_segmented_metadata(parts: [(TensorType, &[u64]); 3]) -> Result<()> {
    let [(gate, gd), (up, ud), (down, dd)] = parts;
    if gate != TensorType::Q4K
        || up != TensorType::Q4K
        || !matches!(down, TensorType::Q5_0 | TensorType::Q8_0)
        || gd != [2560, 640, 512]
        || ud != gd
        || dd != [640, 2560, 512]
    {
        return Err(Error::InvalidFormat("segmented original expert metadata"));
    }
    Ok(())
}
/// Only the three original 16-slot component extents used by the fixed-public
/// CPU/GPU laboratory. A full-bin request proves the complete physical owner;
/// shaders continue to bind the original logical packed bytes only.
fn fixed_lab_page_bytes(logical_bytes: usize) -> Result<usize> {
    if !matches!(logical_bytes, 14_745_600 | 18_022_400 | 27_852_800) {
        return Err(Error::InvalidFormat("fixed-public expert bank extent"));
    }
    usize::try_from(
        crate::main_admission::page_bytes(logical_bytes as u64)
            .map_err(|_| Error::ArithmeticOverflow)?,
    )
    .map_err(|_| Error::ArithmeticOverflow)
}
fn fixed_lab_full_page_handle(
    logical_bytes: usize,
    allocate: impl FnOnce(usize) -> Handle,
) -> Result<Handle> {
    let owned_bytes = fixed_lab_page_bytes(logical_bytes)?;
    let owner = allocate(owned_bytes);
    if owner.size() != owned_bytes as u64
        || owner.size_in_used() != owner.size()
        || owner.offset_start.unwrap_or(0) != 0
        || owner.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidFormat("fixed-public expert bank full owner"));
    }
    Ok(owner.offset_end((owned_bytes - logical_bytes) as u64))
}
pub fn configured_static_cache(raw: Option<&str>, default: usize) -> Result<usize> {
    let bytes = match raw {
        None => default,
        Some(raw) if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) => raw
            .parse()
            .map_err(|_| Error::InvalidFormat("KUNO_LAB_STATIC_CACHE_BYTES integer bytes"))?,
        Some(_) => {
            return Err(Error::InvalidFormat(
                "KUNO_LAB_STATIC_CACHE_BYTES integer bytes",
            ));
        }
    };
    if !(64 * 1024 * 1024..=16 * 1024 * 1024 * 1024).contains(&bytes) {
        return Err(Error::LimitExceeded("laboratory static cache"));
    }
    Ok(bytes)
}
pub(crate) fn check_ordered_segmented(capacity: usize, ordered: bool) -> Result<()> {
    if segmented_capacity(capacity) && ordered {
        return Err(Error::InvalidFormat(
            "segmented slots do not yet support ordered public upload",
        ));
    }
    Ok(())
}
pub fn validate_ordered_segmented(capacity: usize) -> Result<()> {
    check_ordered_segmented(
        capacity,
        std::env::var("KUNO_LAB_ORDERED_PUBLIC_UPLOAD").as_deref() == Ok("1"),
    )
}
pub fn validate_segmented_configuration(
    capacity: usize,
    explicit_cache: Option<usize>,
    budget: usize,
    block: bool,
    batch: bool,
    word_down: bool,
    resident: bool,
) -> Result<()> {
    validate_segmented_configuration_with_profile(
        capacity,
        explicit_cache,
        budget,
        block,
        batch,
        word_down,
        resident,
        OriginalPackedProfile::Q4,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_segmented_configuration_with_profile(
    capacity: usize,
    explicit_cache: Option<usize>,
    budget: usize,
    block: bool,
    batch: bool,
    word_down: bool,
    resident: bool,
    profile: OriginalPackedProfile,
) -> Result<()> {
    profile.validate_capacity(capacity)?;
    let gate_policy = match profile {
        OriginalPackedProfile::Q4 => block,
        OriginalPackedProfile::UnslothQ3 | OriginalPackedProfile::DaslabIq3 => !block,
    };
    if segmented_capacity(capacity)
        && (explicit_cache != Some(budget) || !gate_policy || batch || word_down || !resident)
    {
        return Err(Error::InvalidFormat(match profile {
            OriginalPackedProfile::Q4 => {
                "segmented slots require explicit static cache, resident block MoE, legacy fill and generic down"
            }
            OriginalPackedProfile::UnslothQ3 | OriginalPackedProfile::DaslabIq3 => {
                "Unsloth Q3 requires explicit static cache, resident generic grouped MoE and original fill"
            }
        }));
    }
    Ok(())
}
fn packed_bank_bytes(row_bytes: usize, rows: usize, capacity: usize) -> Result<(usize, usize)> {
    let expert = row_bytes
        .checked_mul(rows)
        .ok_or(Error::ArithmeticOverflow)?;
    let bytes = expert
        .checked_mul(capacity)
        .ok_or(Error::ArithmeticOverflow)?;
    if !expert.is_multiple_of(4) || bytes == 0 || bytes > 128 * 1024 * 1024 {
        return Err(Error::LimitExceeded("expert bank buffer"));
    }
    Ok((expert, bytes))
}
pub fn validate_capacity(capacity: usize) -> Result<()> {
    if !routed_capacity(capacity) {
        return Err(Error::LimitExceeded("expert bank slots"));
    }
    Ok(())
}
const RESIDENCY_MARGIN: usize = 512 * 1024 * 1024;
const MAX_DEVICE_BYTES: usize = 26 * 1024 * 1024 * 1024;
fn validate_residency_budget(
    bank_bytes: usize,
    cache_bytes: usize,
    device_bytes: usize,
) -> Result<()> {
    validate_residency_budget_with_limit(bank_bytes, cache_bytes, device_bytes, MAX_DEVICE_BYTES)
}
fn validate_residency_budget_with_limit(
    bank_bytes: usize,
    cache_bytes: usize,
    device_bytes: usize,
    limit: usize,
) -> Result<()> {
    if device_bytes > limit
        || bank_bytes
            .checked_add(cache_bytes)
            .and_then(|n| n.checked_add(RESIDENCY_MARGIN))
            .is_none_or(|n| n > device_bytes)
    {
        return Err(Error::LimitExceeded(
            "expert banks and cache exceed device budget",
        ));
    }
    Ok(())
}
/// Inspect all original public tensor extents before ANY expert bank is created.
/// Retains the existing margin and live allocator telemetry enforcement.
/// Admission bounds declared bank/cache charges, not all driver/staging use.
pub fn preflight_residency(
    graph: BoundGraph<'_, '_>,
    capacity: usize,
    cache_bytes: usize,
    device_bytes: usize,
) -> Result<usize> {
    preflight_residency_with_profile(
        graph,
        capacity,
        cache_bytes,
        device_bytes,
        OriginalPackedProfile::Q4,
    )
}
pub(crate) fn preflight_residency_with_profile(
    graph: BoundGraph<'_, '_>,
    capacity: usize,
    cache_bytes: usize,
    device_bytes: usize,
    profile: OriginalPackedProfile,
) -> Result<usize> {
    profile.validate_capacity(capacity)?;
    profile.validate_graph(graph)?;
    if segmented_capacity(capacity) && graph.geometry().layers != 48 {
        return Err(Error::InvalidFormat("segmented graph requires 48 layers"));
    }
    let mut total = 0usize;
    for layer in 0..graph.geometry().layers {
        let weights = routed_weights(graph, layer)?;
        total = total
            .checked_add(ExpertBanks::allocation_with_profile(
                weights, capacity, profile,
            )?)
            .ok_or(Error::ArithmeticOverflow)?;
    }
    validate_profile_residency(profile, capacity, total, cache_bytes, device_bytes)?;
    Ok(total)
}
fn validate_profile_residency(
    profile: OriginalPackedProfile,
    capacity: usize,
    total: usize,
    cache_bytes: usize,
    device_bytes: usize,
) -> Result<()> {
    const GIB: usize = 1024 * 1024 * 1024;
    profile.validate_capacity(capacity)?;
    if profile == OriginalPackedProfile::UnslothQ3 && total != UNSLOTH_Q3_BANK_BYTES_144 {
        return Err(Error::InvalidFormat("Unsloth Q3 full graph page charge"));
    }
    if profile == OriginalPackedProfile::DaslabIq3
        && total as u64 != crate::daslab_profile::selected_bank_bytes().physical
    {
        return Err(Error::InvalidFormat("DASLab full graph page charge"));
    }
    if profile == OriginalPackedProfile::Q4
        && segmented_capacity(capacity)
        && total
            != (if capacity == 160 {
                27
            } else if capacity == 144 {
                24
            } else {
                capacity / 32 * 6
            }) * 1024
                * 1024
                * 1024
    {
        return Err(Error::InvalidFormat("segmented full graph page charge"));
    }
    if matches!(capacity, 128 | 144 | 160) || crate::strata_port::enabled() && capacity == 192 {
        // Q3 uses smaller expert banks, but its mixed-precision shared weights
        // exceed the old five-GiB cache once allocator page rounding is charged.
        // Admit only the explicit measured-profile candidate, retaining Q4's
        // original envelope and the checked bank/cache/margin budget below.
        let accepted_cache = if profile.is_low_bit() {
            matches!(cache_bytes, n if n == 5 * GIB || n == 8 * GIB)
        } else {
            cache_bytes == (if capacity == 160 { 3 } else { 5 }) * GIB
        };
        if !accepted_cache || device_bytes != 31 * 1024 * 1024 * 1024 {
            return Err(Error::InvalidFormat("expanded fixed residency envelope"));
        }
        validate_residency_budget_with_limit(
            total,
            cache_bytes,
            device_bytes,
            31 * 1024 * 1024 * 1024,
        )?;
    } else {
        validate_residency_budget(total, cache_bytes, device_bytes)?;
    }
    Ok(())
}
impl ExpertBanks {
    pub fn allocation_requests(weights: RoutedWeights<'_>, capacity: usize) -> Result<Vec<u64>> {
        Self::allocation_requests_with_profile(weights, capacity, OriginalPackedProfile::Q4)
    }
    pub(crate) fn allocation_requests_with_profile(
        weights: RoutedWeights<'_>,
        capacity: usize,
        profile: OriginalPackedProfile,
    ) -> Result<Vec<u64>> {
        profile.validate_capacity(capacity)?;
        if profile.is_low_bit() {
            validate_profile_weights(profile, weights)?;
        }
        if profile == OriginalPackedProfile::Q4 && segmented_capacity(capacity) {
            validate_segmented_metadata(
                [weights.gate, weights.up, weights.down]
                    .map(|t| (t.info().dtype(), t.info().dimensions())),
            )?;
        }
        let mut requests = Vec::new();
        for t in [weights.gate, weights.up, weights.down] {
            for bytes in bank_segments(t, capacity)?.1 {
                requests.push(bytes as u64);
            }
        }
        Ok(requests)
    }
    pub fn allocation(weights: RoutedWeights<'_>, capacity: usize) -> Result<usize> {
        Self::allocation_with_profile(weights, capacity, OriginalPackedProfile::Q4)
    }
    pub(crate) fn allocation_with_profile(
        weights: RoutedWeights<'_>,
        capacity: usize,
        profile: OriginalPackedProfile,
    ) -> Result<usize> {
        Self::allocation_requests_with_profile(weights, capacity, profile)?
            .into_iter()
            .try_fold(0usize, |total, bytes| {
                total
                    .checked_add(
                        if crate::strata_port::enabled()
                            && profile == OriginalPackedProfile::DaslabIq3
                        {
                            bytes as usize
                        } else {
                            (bytes as usize)
                                .checked_next_power_of_two()
                                .ok_or(Error::ArithmeticOverflow)?
                        },
                    )
                    .ok_or(Error::ArithmeticOverflow)
            })
    }
    pub fn new(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
        capacity: usize,
    ) -> Result<Self> {
        Self::new_with_profile(client, weights, capacity, OriginalPackedProfile::Q4)
    }
    pub(crate) fn new_with_profile(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
        capacity: usize,
        profile: OriginalPackedProfile,
    ) -> Result<Self> {
        if !crate::strata_port::enabled() || profile != OriginalPackedProfile::DaslabIq3 {
            return Self::new_inner(client, weights, capacity, false, false, profile);
        }
        use cubecl_runtime::{
            config::{CubeClRuntimeConfig, RuntimeConfig, memory::PersistentMemory},
            memory_management::MemoryAllocationMode,
        };
        if !matches!(
            CubeClRuntimeConfig::get().memory.persistent_memory,
            PersistentMemory::Enabled
        ) {
            return Err(Error::InvalidFormat("port exact bank allocator config"));
        }
        let alignment = client.properties().memory.alignment;
        let requests = Self::allocation_requests_with_profile(weights, capacity, profile)?;
        if alignment == 0
            || alignment > 4096
            || requests.iter().any(|n| !n.is_multiple_of(alignment))
        {
            return Err(Error::InvalidFormat("port exact bank alignment"));
        }
        // The resident admission holds exclusive stream allocation authority. Scope
        // restoration precedes all dependent operations, including failure paths.
        let retired = AtomicBool::new(false);
        let bank = exact_persistent::persistent_scope(
            |mode: MemoryAllocationMode| {
                unsafe {
                    client.allocation_mode(mode);
                }
                Ok(())
            },
            &retired,
            || Self::new_inner(client, weights, capacity, false, false, profile),
        )?;
        let mut roots = Vec::new();
        for owner in [&bank.gate, &bank.up, &bank.down] {
            for segment in &owner.segments {
                let resource = client
                    .get_resource(segment.handle.clone())
                    .map_err(|_| Error::InvalidFormat("port exact bank root"))?;
                let root = resource.resource();
                let logical = segment.words as u64 * 4;
                if root.offset != 0
                    || root.size != logical
                    || root.buffer.size() != logical
                    || roots.contains(&root.buffer)
                {
                    return Err(Error::InvalidFormat("port exact bank physical extent"));
                }
                roots.push(root.buffer.clone());
            }
        }
        if roots.len() != 6 {
            return Err(Error::InvalidFormat("port exact bank root count"));
        }
        Ok(bank)
    }
    /// Narrow operator-lab seam: original Q4 gate/up and Q5/Q8 down, exactly
    /// sixteen slots. The original constructor/model allocation is unchanged.
    pub(crate) fn new_full_pages_for_cpu_gpu_lab(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
    ) -> Result<Self> {
        validate_segmented_metadata(
            [weights.gate, weights.up, weights.down]
                .map(|tensor| (tensor.info().dtype(), tensor.info().dimensions())),
        )
        .map_err(|_| Error::InvalidFormat("fixed-public original expert bank metadata"))?;
        Self::new_inner(client, weights, 16, true, false, OriginalPackedProfile::Q4)
    }
    /// A separate stationary secondary bank. Preserve ordinary flat48 geometry.
    pub(crate) fn new_secondary(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
    ) -> Result<Self> {
        validate_segmented_metadata(
            [weights.gate, weights.up, weights.down]
                .map(|t| (t.info().dtype(), t.info().dimensions())),
        )?;
        Self::new_inner(client, weights, 48, false, true, OriginalPackedProfile::Q4)
    }
    pub(crate) fn secondary_requests(weights: RoutedWeights<'_>) -> Result<Vec<u64>> {
        validate_segmented_metadata(
            [weights.gate, weights.up, weights.down]
                .map(|t| (t.info().dtype(), t.info().dimensions())),
        )?;
        [weights.gate, weights.up, weights.down]
            .into_iter()
            .flat_map(|t| {
                [32, 16].map(move |n| {
                    packed_bank_bytes(
                        t.info().row_bytes() as usize,
                        t.info().dimensions()[1] as usize,
                        n,
                    )
                    .map(|(_, b)| b as u64)
                })
            })
            .collect()
    }
    fn new_inner(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
        capacity: usize,
        full_lab_pages: bool,
        secondary: bool,
        profile: OriginalPackedProfile,
    ) -> Result<Self> {
        let gate = Matrix::new(weights.gate)?;
        let up = Matrix::new(weights.up)?;
        let down = Matrix::new(weights.down)?;
        if gate.inputs() != up.inputs()
            || gate.outputs() != up.outputs()
            || down.inputs() != gate.outputs()
            || down.outputs() != gate.inputs()
            || gate.experts() != up.experts()
            || gate.experts() != down.experts()
            || gate.experts() != 512
            || gate.inputs() > 4096
            || gate.outputs() > 4096
        {
            return Err(Error::InvalidFormat("expert bank geometry"));
        }
        Self::allocation_with_profile(weights, capacity, profile)?;
        let create = |t: TensorView<'_>| -> Result<Bank> {
            let (expert_bytes, sizes) = if secondary {
                let (expert, first) = packed_bank_bytes(
                    t.info().row_bytes() as usize,
                    t.info().dimensions()[1] as usize,
                    32,
                )?;
                let (_, second) = packed_bank_bytes(
                    t.info().row_bytes() as usize,
                    t.info().dimensions()[1] as usize,
                    16,
                )?;
                (expert, vec![first, second])
            } else {
                bank_segments(t, capacity)?
            };
            let mut first_slot = 0;
            let mut words = 0;
            let mut reserved = 0;
            let mut segments = Vec::new();
            for bytes in sizes {
                let count = bytes / expert_bytes;
                segments.push(BankSegment {
                    handle: if full_lab_pages {
                        fixed_lab_full_page_handle(bytes, |owned| client.empty(owned))?
                    } else {
                        client.empty(bytes)
                    },
                    first_slot,
                    capacity: count,
                    words: bytes / 4,
                });
                first_slot += count;
                words += bytes / 4;
                reserved += if crate::strata_port::enabled()
                    && profile == OriginalPackedProfile::DaslabIq3
                {
                    bytes
                } else {
                    bytes.next_power_of_two()
                };
            }
            Ok(Bank {
                segments,
                expert_bytes,
                words,
                reserved,
            })
        };
        let instance = next_instance(&NEXT_BANK_INSTANCE)?;
        Ok(Self {
            profile,
            instance,
            gate: create(weights.gate)?,
            up: create(weights.up)?,
            down: create(weights.down)?,
            hidden: gate.inputs(),
            ff: gate.outputs(),
            experts: gate.experts(),
            slots: BankSlots::new(capacity, gate.experts())
                .map_err(|_| Error::InvalidFormat("expert slots"))?,
            origin: client.clone(),
            names: [weights.gate, weights.up, weights.down].map(|t| t.info().name().into()),
        })
    }
    pub fn instance(&self) -> BankInstance {
        self.instance
    }
    pub(crate) fn profile(&self) -> OriginalPackedProfile {
        self.profile
    }
    pub fn gate(&self) -> &Bank {
        &self.gate
    }
    pub fn up(&self) -> &Bank {
        &self.up
    }
    pub fn down(&self) -> &Bank {
        &self.down
    }
    pub fn reserved(&self) -> usize {
        self.gate.reserved + self.up.reserved + self.down.reserved
    }
    pub fn check_client(&self, client: &ComputeClient<WgpuRuntime>) -> Result<()> {
        if !std::ptr::eq(self.origin.info(), client.info()) {
            return Err(Error::InvalidFormat("expert bank foreign context"));
        }
        Ok(())
    }
    pub fn check_names(&self, weights: RoutedWeights<'_>) -> Result<()> {
        if self.profile.is_low_bit() {
            validate_profile_weights(self.profile, weights)?;
        }
        if [weights.gate, weights.up, weights.down]
            .iter()
            .zip(&self.names)
            .any(|(t, n)| t.info().name() != n)
        {
            return Err(Error::InvalidFormat("expert bank weight names"));
        }
        Ok(())
    }
    pub(crate) fn source_names(&self) -> [&str; 3] {
        self.names.each_ref().map(String::as_str)
    }
    pub fn check(
        &self,
        weights: RoutedWeights<'_>,
        input: &[f32],
        selected: &[(usize, f32)],
    ) -> Result<()> {
        if self.profile.is_low_bit() {
            self.check_names(weights)?;
        }
        if [weights.gate, weights.up, weights.down]
            .iter()
            .zip(&self.names)
            .any(|(t, n)| t.info().name() != n)
            || input.len() != self.hidden
            || input.iter().any(|x| !x.is_finite())
            || selected.len() != 10
            || selected
                .iter()
                .any(|&(e, p)| e >= self.experts || !p.is_finite() || p < 0.)
        {
            return Err(Error::InvalidFormat("expert bank input"));
        }
        Ok(())
    }
}
#[cube(launch)]
fn copy_words(source: &Array<u32>, destination: &mut Array<u32>, offset: usize) {
    let i = ABSOLUTE_POS;
    if i < source.len() {
        destination[offset + i] = source[i];
    }
}

pub fn fill(
    client: &ComputeClient<WgpuRuntime>,
    bank: &Bank,
    tensor: TensorView<'_>,
    expert: usize,
    slot: usize,
    cancel: &AtomicBool,
) -> Result<(u64, u128, u128)> {
    fill_source(client, bank, tensor, expert, slot, cancel, None)
}

/// Same expert bytes/scatter as legacy fill, with an explicitly retained full
/// source page and mapped owned staging. Pending owners survive every failure.
pub(crate) fn fill_ordered(
    client: &ComputeClient<WgpuRuntime>,
    bank: &Bank,
    tensor: TensorView<'_>,
    expert: usize,
    slot: usize,
    cancel: &AtomicBool,
    sources: &mut crate::ordered_public::PendingSources<Handle>,
) -> Result<(u64, u128, u128)> {
    fill_source(client, bank, tensor, expert, slot, cancel, Some(sources))
}
fn fill_source(
    client: &ComputeClient<WgpuRuntime>,
    bank: &Bank,
    tensor: TensorView<'_>,
    expert: usize,
    slot: usize,
    cancel: &AtomicBool,
    sources: Option<&mut crate::ordered_public::PendingSources<Handle>>,
) -> Result<(u64, u128, u128)> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::InvalidFormat("GPU execution cancelled"));
    }
    if tensor.info().dimensions().len() != 3 {
        return Err(Error::InvalidFormat("expert bank rank"));
    }
    let offset = expert
        .checked_mul(bank.expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let (segment, local_slot) = bank.resolve(slot)?;
    let dst = local_slot
        .checked_mul(bank.expert_bytes / 4)
        .ok_or(Error::ArithmeticOverflow)?;
    if expert >= tensor.info().dimensions()[2] as usize
        || dst
            .checked_add(bank.expert_bytes / 4)
            .is_none_or(|n| n > segment.words)
    {
        return Err(Error::InvalidBufferLength);
    }
    let source_bytes = if sources.is_some() {
        crate::main_admission::page_bytes(bank.expert_bytes as u64)
            .map_err(|_| Error::InvalidBufferLength)? as usize
    } else {
        bank.expert_bytes
    };
    let mut packed = if sources.is_some() {
        let mut data = Vec::new();
        data.try_reserve_exact(source_bytes)
            .map_err(|_| Error::LimitExceeded("public expert upload scratch"))?;
        data.resize(source_bytes, 0u8);
        data
    } else {
        vec![0u8; bank.expert_bytes]
    };
    let read = Instant::now();
    tensor.read_range(offset as u64, &mut packed[..bank.expert_bytes])?;
    let read_ns = read.elapsed().as_nanos();
    let upload = Instant::now();
    let src = if let Some(sources) = sources {
        sources.enqueue(
            packed,
            |bytes| Ok(client.empty(bytes)),
            |owner, data| {
                if owner.stream != segment.handle.stream
                    || owner.size() != data.len() as u64
                    || owner.size_in_used() != owner.size()
                {
                    return Err(Error::InvalidFormat(
                        "ordered public source layout or stream",
                    ));
                }
                client
                    .write_private(owner, data)
                    .map_err(|_| Error::InvalidFormat("ordered public source upload failed"))
            },
        )?
    } else {
        // Immutable PUBLIC weights retain the original ordinary upload policy.
        client.create_public_owned(packed)
    };
    copy_words::launch::<WgpuRuntime>(
        client,
        CubeCount::Static((bank.expert_bytes as u32 / 4).div_ceil(256), 1, 1),
        CubeDim::new_1d(256),
        // SAFETY: checked word-aligned expert geometry and destination capacity.
        unsafe { ArrayArg::from_raw_parts(src, bank.expert_bytes / 4) },
        unsafe { ArrayArg::from_raw_parts(segment.handle.clone(), segment.words) },
        dst,
    );
    let upload_ns = upload.elapsed().as_nanos();
    // Only logical original words are copied; full-page padding stays zero.
    // Owned sources remain pending until the next checked completed read fence.
    Ok((bank.expert_bytes as u64, read_ns, upload_ns))
}

/// A byte-exact span inside an untrimmed bank allocation. Allocation padding
/// is retained outside the view; it must never become part of an expert upload.
fn public_slot_span(
    expert_bytes: usize,
    local_slot: usize,
    capacity: usize,
    words: usize,
    handle_bytes: u64,
) -> Result<(u64, u64)> {
    let logical = words.checked_mul(4).ok_or(Error::ArithmeticOverflow)?;
    if expert_bytes == 0
        || !expert_bytes.is_multiple_of(4)
        || local_slot >= capacity
        || expert_bytes.checked_mul(capacity) != Some(logical)
        || handle_bytes < u64::try_from(logical).map_err(|_| Error::ArithmeticOverflow)?
    {
        return Err(Error::InvalidBufferLength);
    }
    let start = local_slot
        .checked_mul(expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let end = start
        .checked_add(expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    if end > logical {
        return Err(Error::InvalidBufferLength);
    }
    Ok((
        u64::try_from(start).map_err(|_| Error::ArithmeticOverflow)?,
        handle_bytes
            .checked_sub(u64::try_from(end).map_err(|_| Error::ArithmeticOverflow)?)
            .ok_or(Error::ArithmeticOverflow)?,
    ))
}

/// Prepare only original PUBLIC weight bytes for a checked existing slot.
/// Caller binds the source names/device and exact pending plan, retains full
/// bank owners and commits only after the original audited owning completion.
pub(crate) fn prepare_public_existing(
    bank: &Bank,
    tensor: TensorView<'_>,
    expert: usize,
    slot: usize,
    cancel: &AtomicBool,
) -> Result<(Handle, Vec<u8>, u128)> {
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::InvalidFormat("GPU execution cancelled"));
    }
    let dimensions = tensor.info().dimensions();
    if dimensions.len() != 3
        || tensor.info().row_bytes().checked_mul(dimensions[1]) != Some(bank.expert_bytes as u64)
        || expert as u64 >= dimensions[2]
    {
        return Err(Error::InvalidFormat("direct original expert geometry"));
    }
    let offset = (expert as u64)
        .checked_mul(bank.expert_bytes as u64)
        .ok_or(Error::ArithmeticOverflow)?;
    if offset
        .checked_add(bank.expert_bytes as u64)
        .is_none_or(|end| end > tensor.info().byte_len())
    {
        return Err(Error::InvalidBufferLength);
    }
    let (segment, local_slot) = bank.resolve(slot)?;
    if segment.handle.offset_start.unwrap_or(0) != 0 || segment.handle.offset_end.unwrap_or(0) != 0
    {
        return Err(Error::InvalidFormat("direct original bank root view"));
    }
    let (start, trim_end) = public_slot_span(
        bank.expert_bytes,
        local_slot,
        segment.capacity,
        segment.words,
        segment.handle.size(),
    )?;
    let view = segment
        .handle
        .clone()
        .offset_start(start)
        .offset_end(trim_end);
    let mut packed = Vec::new();
    packed
        .try_reserve_exact(bank.expert_bytes)
        .map_err(|_| Error::LimitExceeded("public direct expert upload scratch"))?;
    packed.resize(bank.expert_bytes, 0u8);
    let read = Instant::now();
    tensor.read_range(offset, &mut packed)?;
    let read_ns = read.elapsed().as_nanos();
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::InvalidFormat("GPU execution cancelled"));
    }
    Ok((view, packed, read_ns))
}

/// Ten original expert triples share the unchanged source-window allowance.
/// This bound is PUBLIC geometry; no route or slot appears in an allocation key.
pub(crate) const PUBLIC_EXISTING_MAX_COMPONENTS: usize = 30;
const PUBLIC_EXISTING_METADATA: usize = 8192;

fn public_existing_batch_bounds(
    components: usize,
    metadata_bytes: usize,
    capacities: impl IntoIterator<Item = (usize, usize)>,
    source_precharge: u64,
) -> Result<u64> {
    if components == 0
        || components > PUBLIC_EXISTING_MAX_COMPONENTS
        || metadata_bytes > PUBLIC_EXISTING_METADATA
        || source_precharge > 40 * 1024 * 1024
    {
        return Err(Error::LimitExceeded("public existing expert window"));
    }
    let mut count = 0usize;
    let mut logical = 0u64;
    let mut charged = 0u64;
    for (len, capacity) in capacities {
        let page = crate::main_admission::page_bytes(len as u64)
            .map_err(|_| Error::InvalidBufferLength)?;
        if len == 0 || !len.is_multiple_of(4) || capacity < len || capacity as u64 > page {
            return Err(Error::InvalidBufferLength);
        }
        count = count.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        logical = logical
            .checked_add(len as u64)
            .ok_or(Error::ArithmeticOverflow)?;
        charged = charged.checked_add(page).ok_or(Error::ArithmeticOverflow)?;
    }
    if count != components || charged != source_precharge {
        return Err(Error::InvalidBufferLength);
    }
    Ok(logical)
}

/// Schedule immutable original PUBLIC weights into already checked slot views.
/// The caller retains full roots and the canonical pin, and holds the ORIGINAL
/// per-component source precharge through its audited owning read_fault. No
/// absent GPU source, smaller dispatch count or ordinary staging grants credit.
/// Actual public Vec capacities fit those same pages; descriptor metadata fits
/// the original workspace's separate 512-KiB temporary-host allowance.
pub(crate) fn write_public_existing(
    client: &ComputeClient<WgpuRuntime>,
    writes: Vec<(Handle, Vec<u8>)>,
    source_precharge: u64,
    cancel: &AtomicBool,
) -> Result<(u64, u128)> {
    let metadata = writes
        .capacity()
        .checked_mul(std::mem::size_of::<(Handle, Vec<u8>)>())
        .ok_or(Error::ArithmeticOverflow)?;
    let bytes = public_existing_batch_bounds(
        writes.len(),
        metadata,
        writes.iter().map(|(_, data)| (data.len(), data.capacity())),
        source_precharge,
    )?;
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::InvalidFormat("GPU execution cancelled"));
    }
    let upload = Instant::now();
    client
        .write_public_owned_batch(writes)
        .map_err(|_| Error::InvalidFormat("direct original expert upload"))?;
    Ok((bytes, upload.elapsed().as_nanos()))
}

// One-dimensional portable grid; stay one 256-lane workgroup below 64 MiB.
const MAX_BATCH_BYTES: usize = 65_535 * 256 * 4;
const MIN_CONFIG_BATCH_BYTES: usize = 4 * 1024 * 1024;
const MAX_CONFIG_BATCH_BYTES: usize = 64 * 1024 * 1024;
pub const DEFAULT_BATCH_BYTES: usize = 8 * 1024 * 1024;
/// Operator-only comparison window. The effective portable grid bound is a
/// little smaller than 64 MiB; chunks never split a packed expert bundle.
pub fn configured_batch_bytes(raw: Option<&str>) -> Result<usize> {
    let bytes = match raw {
        None => DEFAULT_BATCH_BYTES,
        Some(raw) if !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()) => {
            raw.parse().map_err(|_| {
                Error::InvalidFormat("KUNO_LAB_EXPERT_BATCH_BYTES must be integer 4..64 MiB")
            })?
        }
        Some(_) => {
            return Err(Error::InvalidFormat(
                "KUNO_LAB_EXPERT_BATCH_BYTES must be integer 4..64 MiB",
            ));
        }
    };
    if !(MIN_CONFIG_BATCH_BYTES..=MAX_CONFIG_BATCH_BYTES).contains(&bytes) {
        return Err(Error::LimitExceeded(
            "KUNO_LAB_EXPERT_BATCH_BYTES must be 4..64 MiB",
        ));
    }
    Ok(bytes)
}
/// Same public packed payload and runtime private slots across alternatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScatterMode {
    Dynamic1d,
    Specialized1d,
    Planes3d,
}
impl ScatterMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dynamic1d => "dynamic-1d",
            Self::Specialized1d => "public-specialized-1d",
            Self::Planes3d => "division-free-3d",
        }
    }
}
pub fn configured_scatter(raw: Option<&str>) -> Result<ScatterMode> {
    match raw {
        None | Some("0") => Ok(ScatterMode::Dynamic1d),
        Some("1") => Ok(ScatterMode::Specialized1d),
        Some("2") => Ok(ScatterMode::Planes3d),
        _ => Err(Error::InvalidFormat(
            "KUNO_LAB_EXPERT_SCATTER must be 0, 1 or 2",
        )),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BatchLayout {
    component_words: [usize; 3],
    expert_words: usize,
    experts_per_batch: usize,
}
impl BatchLayout {
    fn new(component_bytes: [usize; 3], capacity: usize, batch_bytes: usize) -> Result<Self> {
        if !(MIN_CONFIG_BATCH_BYTES..=MAX_CONFIG_BATCH_BYTES).contains(&batch_bytes)
            || !(10..=72).contains(&capacity)
            || component_bytes
                .iter()
                .any(|&n| n == 0 || !n.is_multiple_of(4))
        {
            return Err(Error::InvalidFormat("public expert batch geometry"));
        }
        let mut total = 0usize;
        for bytes in component_bytes {
            if bytes
                .checked_mul(capacity)
                .is_none_or(|n| n > 128 * 1024 * 1024)
            {
                return Err(Error::LimitExceeded("public expert batch destination"));
            }
            total = total.checked_add(bytes).ok_or(Error::ArithmeticOverflow)?;
        }
        let experts_per_batch = (batch_bytes.min(MAX_BATCH_BYTES) / total).min(10);
        if experts_per_batch == 0 {
            return Err(Error::LimitExceeded("public expert batch bundle"));
        }
        Ok(Self {
            component_words: component_bytes.map(|n| n / 4),
            expert_words: total / 4,
            experts_per_batch,
        })
    }
    fn validate_misses(
        self,
        misses: &[(usize, usize)],
        experts: usize,
        capacity: usize,
    ) -> Result<()> {
        if misses.len() > 10 || experts == 0 || capacity > experts {
            return Err(Error::InvalidBufferLength);
        }
        for (i, &(expert, slot)) in misses.iter().enumerate() {
            if expert >= experts
                || slot >= capacity
                || misses[..i].iter().any(|&(e, s)| e == expert || s == slot)
            {
                return Err(Error::InvalidFormat("public expert batch selection"));
            }
        }
        Ok(())
    }
    fn scatter_grid(self, mode: ScatterMode, count: usize) -> Result<[u32; 3]> {
        let bytes = self.bytes(count)?;
        let dims = match mode {
            ScatterMode::Dynamic1d | ScatterMode::Specialized1d => {
                [(bytes / 4).div_ceil(256), 1, 1]
            }
            ScatterMode::Planes3d => [
                self.component_words
                    .into_iter()
                    .max()
                    .ok_or(Error::InvalidBufferLength)?
                    .div_ceil(256),
                3,
                count,
            ],
        };
        if dims.into_iter().any(|n| n == 0 || n > 65_535) {
            return Err(Error::LimitExceeded("public expert scatter grid"));
        }
        Ok(dims.map(|n| n as u32))
    }
    fn bytes(self, count: usize) -> Result<usize> {
        if count == 0 || count > self.experts_per_batch {
            return Err(Error::InvalidBufferLength);
        }
        self.expert_words
            .checked_mul(count)
            .and_then(|n| n.checked_mul(4))
            .ok_or(Error::ArithmeticOverflow)
    }
}
#[derive(Default)]
pub struct FillStats {
    pub bytes: u64,
    pub read_ns: u128,
    pub upload_ns: u128,
    pub experts: u64,
}

/// Upload one bounded concatenation of ORIGINAL public packed weights per
/// chunk; scatter into unchanged gate/up/down bank slots on the same stream.
/// Private choices are ten validated RUNTIME uniforms, never compile constants
/// or a private temporary GPU array. No activation/request bytes enter packed.
/// Bank metadata is not committed here. Caller discards a partially changed
/// bank/whole token on ANY error and commits only the ordered successful plan.
pub fn fill_plan(
    client: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
    weights: RoutedWeights<'_>,
    misses: &[(usize, usize)],
    cancel: &AtomicBool,
    batch_bytes: usize,
    scatter: ScatterMode,
) -> Result<FillStats> {
    let cancelled = || -> Result<()> {
        if cancel.load(Ordering::Acquire) {
            Err(Error::InvalidFormat("GPU execution cancelled"))
        } else {
            Ok(())
        }
    };
    cancelled()?;
    banks.gate.single()?;
    banks.up.single()?;
    banks.down.single()?;
    banks.check_client(client)?;
    banks.check_names(weights)?;
    let parts = [
        (&banks.gate, weights.gate, banks.hidden, banks.ff),
        (&banks.up, weights.up, banks.hidden, banks.ff),
        (&banks.down, weights.down, banks.ff, banks.hidden),
    ];
    let layout = BatchLayout::new(
        parts.map(|(bank, _, _, _)| bank.expert_bytes),
        banks.slots.capacity(),
        batch_bytes,
    )?;
    layout.validate_misses(misses, banks.experts, banks.slots.capacity())?;
    for (bank, tensor, columns, rows) in parts {
        let matrix = Matrix::new(tensor)?;
        if matrix.inputs() != columns
            || matrix.outputs() != rows
            || matrix.experts() != banks.experts
            || bank.expert_bytes.checked_mul(banks.slots.capacity()) != Some(bank.words * 4)
        {
            return Err(Error::InvalidFormat("public expert batch tensor geometry"));
        }
        quant_matvec::validate_layout(
            tensor.info().dtype().gguf_id(),
            rows,
            columns,
            bank.expert_bytes,
            columns,
            rows,
        )
        .map_err(|_| Error::InvalidFormat("public expert batch packing"))?;
        if bank.single()?.handle.stream != banks.gate.single()?.handle.stream {
            return Err(Error::InvalidFormat("public expert batch stream"));
        }
    }
    let mut stats = FillStats::default();
    // Public original bytes only. Reuse the same bounded capacity between
    // chunks; create_from_slice owns its copies before the next overwrite.
    let mut packed = Vec::new();
    for chunk in misses.chunks(layout.experts_per_batch) {
        cancelled()?;
        let bytes = layout.bytes(chunk.len())?;
        let [gx, gy, gz] = layout.scatter_grid(scatter, chunk.len())?;
        packed.resize(bytes, 0);
        let mut offset = 0usize;
        let mut slots = zeroize::Zeroizing::new([0_u32; 10]);
        for (rank, &(expert, slot)) in chunk.iter().enumerate() {
            slots[rank] = slot as u32;
            for (bank, tensor, _, _) in parts {
                cancelled()?;
                let end = offset
                    .checked_add(bank.expert_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                let source_offset = expert
                    .checked_mul(bank.expert_bytes)
                    .ok_or(Error::ArithmeticOverflow)?;
                let read = Instant::now();
                tensor.read_range(source_offset as u64, &mut packed[offset..end])?;
                stats.read_ns += read.elapsed().as_nanos();
                offset = end;
            }
        }
        if offset != bytes {
            return Err(Error::InvalidBufferLength);
        }
        cancelled()?;
        let upload = Instant::now();
        let source = client.create_from_slice(&packed);
        if source.stream != banks.gate.single()?.handle.stream {
            return Err(Error::InvalidFormat("public expert batch source stream"));
        }
        cancelled()?;
        // SAFETY: all source bounds/word alignment/grid, fixed tensor layouts,
        // destination capacities, unique experts/slots and runtime selections
        // checked above. Handles remain live on this stream until submission
        // drains. Each source word is copied exactly once to one public bank.
        unsafe {
            macro_rules! launch_scatter {
                ($kernel:ident) => {{
                    $kernel::launch_unchecked::<WgpuRuntime>(
                        client,
                        CubeCount::Static(gx, gy, gz),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(source, bytes / 4),
                        ArrayArg::from_raw_parts(
                            banks.gate.single()?.handle.clone(),
                            banks.gate.words,
                        ),
                        ArrayArg::from_raw_parts(banks.up.single()?.handle.clone(), banks.up.words),
                        ArrayArg::from_raw_parts(
                            banks.down.single()?.handle.clone(),
                            banks.down.words,
                        ),
                        layout.component_words[0],
                        layout.component_words[1],
                        layout.component_words[2],
                        slots[0],
                        slots[1],
                        slots[2],
                        slots[3],
                        slots[4],
                        slots[5],
                        slots[6],
                        slots[7],
                        slots[8],
                        slots[9],
                    );
                }};
            }
            match scatter {
                ScatterMode::Dynamic1d => launch_scatter!(scatter_public_weights),
                ScatterMode::Specialized1d => launch_scatter!(scatter_public_weights_specialized),
                ScatterMode::Planes3d => launch_scatter!(scatter_public_weights_planes),
            }
        }
        stats.upload_ns += upload.elapsed().as_nanos();
        stats.bytes += bytes as u64;
        stats.experts += chunk.len() as u64;
        // Public packed scratch stays reusable; choices in slots zeroize.
        cancelled()?;
    }
    Ok(stats)
}
#[cube]
#[allow(clippy::too_many_arguments)]
fn scatter_slot(
    rank: usize,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
    s8: u32,
    s9: u32,
) -> u32 {
    let mut slot = s0;
    if rank == 1 {
        slot = s1;
    }
    if rank == 2 {
        slot = s2;
    }
    if rank == 3 {
        slot = s3;
    }
    if rank == 4 {
        slot = s4;
    }
    if rank == 5 {
        slot = s5;
    }
    if rank == 6 {
        slot = s6;
    }
    if rank == 7 {
        slot = s7;
    }
    if rank == 8 {
        slot = s8;
    }
    if rank == 9 {
        slot = s9;
    }
    slot
}
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_public_weights(
    source: &Array<u32>,
    gate: &mut Array<u32>,
    up: &mut Array<u32>,
    down: &mut Array<u32>,
    gate_words: usize,
    up_words: usize,
    down_words: usize,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
    s8: u32,
    s9: u32,
) {
    let i = ABSOLUTE_POS;
    if i < source.len() {
        let bundle = gate_words + up_words + down_words;
        let rank = i / bundle;
        let word = i % bundle;
        let slot = scatter_slot(rank, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9);
        if word < gate_words {
            gate[slot as usize * gate_words + word] = source[i];
        } else if word < gate_words + up_words {
            up[slot as usize * up_words + word - gate_words] = source[i];
        } else {
            down[slot as usize * down_words + word - gate_words - up_words] = source[i];
        }
    }
}

// Only admitted PUBLIC tensor widths specialize this kernel.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_public_weights_specialized(
    source: &Array<u32>,
    gate: &mut Array<u32>,
    up: &mut Array<u32>,
    down: &mut Array<u32>,
    #[comptime] gate_words: usize,
    #[comptime] up_words: usize,
    #[comptime] down_words: usize,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
    s8: u32,
    s9: u32,
) {
    let i = ABSOLUTE_POS;
    if i < source.len() {
        let bundle = comptime!(gate_words + up_words + down_words);
        let rank = i / bundle;
        let word = i % bundle;
        let slot = scatter_slot(rank, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9);
        if word < gate_words {
            gate[slot as usize * gate_words + word] = source[i];
        } else if word < gate_words + up_words {
            up[slot as usize * up_words + word - gate_words] = source[i];
        } else {
            down[slot as usize * down_words + word - gate_words - up_words] = source[i];
        }
    }
}

// Plane/rank are uniform per workgroup; no per-word quotient/remainder.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn scatter_public_weights_planes(
    source: &Array<u32>,
    gate: &mut Array<u32>,
    up: &mut Array<u32>,
    down: &mut Array<u32>,
    gate_words: usize,
    up_words: usize,
    down_words: usize,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    s4: u32,
    s5: u32,
    s6: u32,
    s7: u32,
    s8: u32,
    s9: u32,
) {
    let i = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let plane = CUBE_POS_Y;
    let rank = CUBE_POS_Z as usize;
    let bundle = gate_words + up_words + down_words;
    let slot = scatter_slot(rank, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9);
    if plane == 0 {
        if i < gate_words {
            gate[slot as usize * gate_words + i] = source[rank * bundle + i];
        }
    } else if plane == 1 {
        if i < up_words {
            up[slot as usize * up_words + i] = source[rank * bundle + gate_words + i];
        }
    } else {
        if i < down_words {
            down[slot as usize * down_words + i] =
                source[rank * bundle + gate_words + up_words + i];
        }
    }
}

#[cube(launch)]
fn erase(data: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < data.len() {
        data[i] = 0.;
    }
}
#[cube(launch)]
fn erase_pair(a: &mut Array<u32>, b: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < a.len() {
        a[i] = 0;
    }
    if i < b.len() {
        b[i] = 0;
    }
}
#[cube(launch)]
fn erase_six(
    a: &mut Array<u32>,
    b: &mut Array<u32>,
    c: &mut Array<u32>,
    d: &mut Array<u32>,
    e: &mut Array<u32>,
    f: &mut Array<u32>,
) {
    let i = ABSOLUTE_POS;
    if i < a.len() {
        a[i] = 0;
    }
    if i < b.len() {
        b[i] = 0;
    }
    if i < c.len() {
        c[i] = 0;
    }
    if i < d.len() {
        d[i] = 0;
    }
    if i < e.len() {
        e[i] = 0;
    }
    if i < f.len() {
        f[i] = 0;
    }
}
pub fn clear(client: &ComputeClient<WgpuRuntime>, handles: &[(Handle, usize)]) {
    let max = handles.iter().map(|(_, len)| *len).max().unwrap_or(0);
    if max == 0 {
        return;
    }
    let count = CubeCount::Static((max as u32).div_ceil(256), 1, 1);
    let dim = CubeDim::new_1d(256);
    // SAFETY: caller tracks exact 32-bit allocation counts; erasing float/u32
    // storage as words writes the same zero bits. All handles are distinct.
    unsafe {
        let arg = |i: usize| ArrayArg::from_raw_parts(handles[i].0.clone(), handles[i].1);
        match handles.len() {
            2 => erase_pair::launch::<WgpuRuntime>(client, count, dim, arg(0), arg(1)),
            6 => erase_six::launch::<WgpuRuntime>(
                client,
                count,
                dim,
                arg(0),
                arg(1),
                arg(2),
                arg(3),
                arg(4),
                arg(5),
            ),
            _ => {
                for (handle, len) in handles {
                    erase::launch::<WgpuRuntime>(
                        client,
                        CubeCount::Static((*len as u32).div_ceil(256), 1, 1),
                        dim,
                        ArrayArg::from_raw_parts(handle.clone(), *len),
                    );
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn execute(
    client: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
    weights: RoutedWeights<'_>,
    input: &[f32],
    slots: &[usize],
    probabilities: &[f32],
    cancel: &AtomicBool,
    block_moe: bool,
    word_down: bool,
) -> Result<Vec<f32>> {
    banks.gate.single()?;
    banks.up.single()?;
    banks.down.single()?;
    let selected = slots.len();
    let shape = quant_matvec::moe::MoeShape::new(selected, banks.hidden, banks.ff, 65535)
        .map_err(|_| Error::InvalidFormat("expert launch shape"))?;
    if selected != 10
        || probabilities.len() != selected
        || slots
            .iter()
            .any(|&s| s * banks.gate.expert_bytes / 4 >= banks.gate.words)
        || probabilities.iter().any(|p| !p.is_finite() || *p < 0.)
    {
        return Err(Error::InvalidBufferLength);
    }
    for (tensor, columns, rows, bank) in [
        (weights.gate, banks.hidden, banks.ff, &banks.gate),
        (weights.up, banks.hidden, banks.ff, &banks.up),
        (weights.down, banks.ff, banks.hidden, &banks.down),
    ] {
        quant_matvec::validate_layout(
            tensor.info().dtype().gguf_id(),
            rows,
            columns,
            bank.expert_bytes,
            columns,
            rows,
        )
        .map_err(|_| Error::InvalidFormat("expert packed layout"))?;
    }
    if word_down {
        quant_matvec::moe_word_down::validate(weights.down.info().dtype().gguf_id(), banks.ff)
            .map_err(|_| Error::InvalidFormat("word down packed layout"))?;
    }
    let slots: Vec<u32> = slots.iter().map(|&s| s as u32).collect();
    let x = client.create_from_slice(f32::as_bytes(input));
    let sl = client.create_from_slice(u32::as_bytes(&slots));
    let p = client.create_from_slice(f32::as_bytes(probabilities));
    let act = client.empty(selected * banks.ff * 4);
    let parts = client.empty(selected * banks.hidden * 4);
    let out = client.empty(banks.hidden * 4);
    let scratch = vec![
        (x.clone(), input.len()),
        (sl.clone(), selected),
        (p.clone(), selected),
        (act.clone(), selected * banks.ff),
        (parts.clone(), selected * banks.hidden),
        (out.clone(), banks.hidden),
    ];
    let result = (|| {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::InvalidFormat("GPU execution cancelled"));
        }
        // SAFETY: all scalar buffer lengths, bank indices, and packed dimensions validated above.
        unsafe {
            if block_moe {
                quant_matvec::moe_block::moe_gate_up::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(shape.gate_up_cubes, 1, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(banks.gate.single()?.handle.clone(), banks.gate.words),
                    ArrayArg::from_raw_parts(banks.up.single()?.handle.clone(), banks.up.words),
                    ArrayArg::from_raw_parts(x, input.len()),
                    ArrayArg::from_raw_parts(sl.clone(), selected),
                    ArrayArg::from_raw_parts(act.clone(), selected * banks.ff),
                    selected,
                    banks.hidden,
                    banks.ff,
                    weights.gate.info().dtype().gguf_id(),
                    weights.up.info().dtype().gguf_id(),
                );
            } else {
                quant_matvec::moe::moe_gate_up::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(shape.gate_up_cubes, 1, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(banks.gate.single()?.handle.clone(), banks.gate.words),
                    ArrayArg::from_raw_parts(banks.up.single()?.handle.clone(), banks.up.words),
                    ArrayArg::from_raw_parts(x, input.len()),
                    ArrayArg::from_raw_parts(sl.clone(), selected),
                    ArrayArg::from_raw_parts(act.clone(), selected * banks.ff),
                    selected,
                    banks.hidden,
                    banks.ff,
                    weights.gate.info().dtype().gguf_id(),
                    weights.up.info().dtype().gguf_id(),
                );
            }
            if word_down {
                quant_matvec::moe_word_down::moe_down::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(shape.down_cubes, 1, 1),
                    CubeDim::new_1d(quant_matvec::moe_word_down::PHYSICAL_LANES as u32),
                    ArrayArg::from_raw_parts(banks.down.single()?.handle.clone(), banks.down.words),
                    ArrayArg::from_raw_parts(act, selected * banks.ff),
                    ArrayArg::from_raw_parts(sl, selected),
                    ArrayArg::from_raw_parts(parts.clone(), selected * banks.hidden),
                    selected,
                    banks.hidden,
                    banks.ff,
                    weights.down.info().dtype().gguf_id(),
                );
            } else {
                quant_matvec::moe::moe_down::launch_unchecked::<WgpuRuntime>(
                    client,
                    CubeCount::Static(shape.down_cubes, 1, 1),
                    CubeDim::new_1d(64),
                    ArrayArg::from_raw_parts(banks.down.single()?.handle.clone(), banks.down.words),
                    ArrayArg::from_raw_parts(act, selected * banks.ff),
                    ArrayArg::from_raw_parts(sl, selected),
                    ArrayArg::from_raw_parts(parts.clone(), selected * banks.hidden),
                    selected,
                    banks.hidden,
                    banks.ff,
                    weights.down.info().dtype().gguf_id(),
                );
            }
            quant_matvec::moe::moe_reduce::launch_unchecked::<WgpuRuntime>(
                client,
                CubeCount::Static(shape.reduce_cubes, 1, 1),
                CubeDim::new_1d(64),
                ArrayArg::from_raw_parts(parts, selected * banks.hidden),
                ArrayArg::from_raw_parts(p, selected),
                ArrayArg::from_raw_parts(out.clone(), banks.hidden),
                selected,
                banks.hidden,
            );
        }
        let mut bytes = client
            .read_one(out)
            .map_err(|_| Error::InvalidFormat("expert output read"))?;
        let values = f32::from_bytes(&bytes);
        let result = if values.len() != banks.hidden || values.iter().any(|v| !v.is_finite()) {
            Err(Error::InvalidFormat("expert output values"))
        } else if cancel.load(Ordering::Relaxed) {
            Err(Error::InvalidFormat("GPU execution cancelled"))
        } else {
            Ok(values.to_vec())
        };
        bytes.zeroize();
        result
    })();
    clear(client, &scratch);
    result
}

#[cfg(test)]
mod fixed_lab_page_tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn original_sixteen_slot_q5_q8_banks_own_full_pages_with_exact_logical_views() {
        let stream = cubecl::stream_id::StreamId::current();
        for down_expert in [1_126_400usize, 1_740_800] {
            let mut charged = 0usize;
            for expert in [921_600usize, 921_600, down_expert] {
                let logical = 16 * expert;
                let page = logical.next_power_of_two();
                let requested = Cell::new(0);
                let view = fixed_lab_full_page_handle(logical, |bytes| {
                    requested.set(bytes);
                    Handle::new(stream, bytes as u64)
                })
                .unwrap();
                assert_eq!(requested.get(), page);
                assert_eq!(view.size(), page as u64);
                assert_eq!(view.size_in_used(), logical as u64);
                assert_eq!(view.offset_start.unwrap_or(0), 0);
                assert_eq!(view.offset_end, Some((page - logical) as u64));
                assert_eq!(view.stream, stream);
                assert_eq!(view.size_in_used() as usize / expert, 16);
                charged += page;
            }
            assert_eq!(charged, 64 * 1024 * 1024);
            assert_eq!(charged * 2, 134_217_728);
        }
    }
    #[test]
    fn full_page_lab_rejects_nonoriginal_extents_and_sliced_or_wrong_sized_owners() {
        let stream = cubecl::stream_id::StreamId::current();
        for logical in [0, 4, 14_745_604, 16 * 1024 * 1024, usize::MAX] {
            assert!(
                fixed_lab_full_page_handle(logical, |_| {
                    panic!("nonoriginal extent reached allocation")
                })
                .is_err()
            );
        }
        for bad in 0..3 {
            assert!(
                fixed_lab_full_page_handle(14_745_600, |page| match bad {
                    0 => Handle::new(stream, page as u64 + 32768),
                    1 => Handle::new(stream, page as u64).offset_start(4),
                    _ => Handle::new(stream, page as u64).offset_end(4),
                })
                .is_err()
            );
        }
    }
}

#[cfg(test)]
mod instance_tests {
    use super::*;
    #[test]
    fn allocation_instances_cannot_reuse_old_proofs_even_at_exhaustion() {
        let c = AtomicU64::new(41);
        let a = next_instance(&c).unwrap();
        let b = next_instance(&c).unwrap();
        assert!(a != b);
        let end = AtomicU64::new(u64::MAX - 1);
        assert!(next_instance(&end).is_ok());
        assert!(matches!(
            next_instance(&end),
            Err(Error::ArithmeticOverflow)
        ));
        assert!(matches!(
            next_instance(&end),
            Err(Error::ArithmeticOverflow)
        ));
        assert_eq!(end.load(Ordering::Relaxed), u64::MAX);
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    #[test]
    fn batch_window_overrides_are_bounded_and_default_does_not_change_legacy_selection() {
        assert_eq!(configured_batch_bytes(None).unwrap(), 8 * 1024 * 1024);
        for mib in [4, 8, 16, 64] {
            assert_eq!(
                configured_batch_bytes(Some(&(mib * 1024 * 1024).to_string())).unwrap(),
                mib * 1024 * 1024
            );
        }
        for invalid in [
            "",
            "0",
            "4194303",
            "67108865",
            "184467440737095516160",
            "8MiB",
            "-1",
            "+8388608",
            " 8388608",
        ] {
            assert!(configured_batch_bytes(Some(invalid)).is_err());
        }
    }
    #[test]
    fn actual_packed_experts_respect_4_8_16_64_mib_windows_and_portable_grid() {
        for down in [1_126_400, 1_740_800] {
            let bundle = 2 * 921_600 + down;
            for mib in [4, 8, 16, 64] {
                let target = mib * 1024 * 1024;
                let l = BatchLayout::new([921_600, 921_600, down], 72, target).unwrap();
                assert_eq!(
                    l.experts_per_batch,
                    (target.min(MAX_BATCH_BYTES) / bundle).min(10)
                );
                let chunks: Vec<_> = (0..10)
                    .collect::<Vec<_>>()
                    .chunks(l.experts_per_batch)
                    .map(|c| l.bytes(c.len()).unwrap())
                    .collect();
                assert!(chunks.iter().all(|&bytes| bytes <= target
                    && bytes <= MAX_BATCH_BYTES
                    && bytes.is_multiple_of(bundle)));
                assert_eq!(chunks.iter().sum::<usize>(), 10 * bundle);
            }
        }
        // A window cannot admit even one larger original public bundle.
        assert!(BatchLayout::new([6_553_600; 3], 10, DEFAULT_BATCH_BYTES).is_err());
        assert!(BatchLayout::new([4; 3], 10, MIN_CONFIG_BATCH_BYTES - 1).is_err());
        assert!(BatchLayout::new([4; 3], 10, MAX_CONFIG_BATCH_BYTES + 1).is_err());
    }
    #[test]
    fn original_q4_with_q5_or_q8_ten_misses_fit_one_bounded_grid() {
        for down in [1126400, 1740800] {
            let l = BatchLayout::new([921600, 921600, down], 64, MAX_CONFIG_BATCH_BYTES).unwrap();
            assert_eq!(l.experts_per_batch, 10);
            assert_eq!(l.bytes(10).unwrap(), 10 * (1843200 + down));
            assert!(l.bytes(10).unwrap() <= MAX_BATCH_BYTES);
            assert!(l.bytes(10).unwrap().div_ceil(4 * 256) <= 65535);
            l.validate_misses(
                &(0..10).map(|n| (511 - n, n * 6)).collect::<Vec<_>>(),
                512,
                64,
            )
            .unwrap();
        }
    }
    #[test]
    fn unusually_large_legal_public_bundles_split_only_at_expert_boundaries() {
        let l = BatchLayout::new([6553600; 3], 10, MAX_CONFIG_BATCH_BYTES).unwrap();
        assert_eq!(l.experts_per_batch, 3);
        let misses = (0..10).map(|n| (20 + n, 9 - n)).collect::<Vec<_>>();
        l.validate_misses(&misses, 512, 10).unwrap();
        let sizes = misses
            .chunks(l.experts_per_batch)
            .map(|c| l.bytes(c.len()).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(sizes, [58982400, 58982400, 58982400, 19660800]);
        assert!(
            sizes
                .iter()
                .all(|&n| n <= MAX_BATCH_BYTES && n.is_multiple_of(4))
        );
        assert_eq!(sizes.iter().sum::<usize>(), 10 * 3 * 6553600);
    }
    #[test]
    fn rejects_invalid_geometry_modified_choices_and_destination_overflow_before_copy() {
        for b in [
            [0, 4, 4],
            [3, 4, 4],
            [usize::MAX, 4, 4],
            [128 * 1024 * 1024, 4, 4],
        ] {
            assert!(BatchLayout::new(b, 10, DEFAULT_BATCH_BYTES).is_err());
        }
        assert!(BatchLayout::new([4; 3], 9, DEFAULT_BATCH_BYTES).is_err());
        assert!(BatchLayout::new([4; 3], 513, DEFAULT_BATCH_BYTES).is_err());
        let l = BatchLayout::new([4; 3], 16, DEFAULT_BATCH_BYTES).unwrap();
        for misses in [
            &[(512, 0)][..],
            &[(0, 16)][..],
            &[(1, 0), (1, 2)][..],
            &[(1, 0), (2, 0)][..],
        ] {
            assert!(l.validate_misses(misses, 512, 16).is_err());
        }
        assert!(
            l.validate_misses(&(0..11).map(|n| (n, n)).collect::<Vec<_>>(), 512, 16)
                .is_err()
        );
        assert!(l.bytes(0).is_err());
        assert!(l.bytes(11).is_err());
        assert!(l.validate_misses(&[], 512, 16).is_ok());
    }
    #[test]
    fn scatter_exactly_maps_nonconsecutive_slots_and_keeps_unselected_words_unchanged() {
        // Independent per-expert bank fixtures, including slot63 and unequal
        // component extents. Each u32 can stand for any original packed word.
        let sizes = [3usize, 5, 2];
        let capacity = 64;
        let choices = [(511usize, 63usize), (7, 0), (301, 12)];
        let original: Vec<Vec<u32>> = sizes
            .iter()
            .enumerate()
            .map(|(part, &words)| {
                (0..512 * words)
                    .map(|i| (part as u32) * 1000000 + i as u32)
                    .collect()
            })
            .collect();
        let mut packed = Vec::new();
        for &(expert, _) in &choices {
            for (part, &words) in sizes.iter().enumerate() {
                packed.extend_from_slice(&original[part][expert * words..(expert + 1) * words]);
            }
        }
        let mut destination: Vec<Vec<u32>> = sizes
            .iter()
            .map(|&words| vec![0xdeadbeef; words * capacity])
            .collect();
        let bundle = sizes.iter().sum::<usize>();
        for (i, &value) in packed.iter().enumerate() {
            let rank = i / bundle;
            let word = i % bundle;
            let slot = choices[rank].1;
            if word < sizes[0] {
                destination[0][slot * sizes[0] + word] = value;
            } else if word < sizes[0] + sizes[1] {
                destination[1][slot * sizes[1] + word - sizes[0]] = value;
            } else {
                destination[2][slot * sizes[2] + word - sizes[0] - sizes[1]] = value;
            }
        }
        for (part, &words) in sizes.iter().enumerate() {
            for slot in 0..capacity {
                let actual = &destination[part][slot * words..(slot + 1) * words];
                if let Some(&(expert, _)) = choices.iter().find(|&&(_, s)| s == slot) {
                    assert_eq!(
                        actual,
                        &original[part][expert * words..(expert + 1) * words]
                    );
                } else {
                    assert!(actual.iter().all(|&v| v == 0xdeadbeef));
                }
            }
        }
    }
    fn scatter_wgsl(
        scatter: ScatterMode,
        mode: cubecl::server::ExecutionMode,
        public_words: [usize; 3],
    ) -> String {
        use cubecl::compute::KernelBuilder;
        use cubecl::ir::{ElemType, StorageType, Type, UIntKind};
        use cubecl::wgpu::WgslCompiler;
        use cubecl::{Compiler, KernelSettings, WgpuCompilationOptions};
        let u = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let mut b = KernelBuilder::default();
        b.scope.register_type::<u32>(u);
        b.scope.register_type::<usize>(u);
        let source = b.input_array(Type::new(u)).into();
        let gate = b.output_array(Type::new(u)).into();
        let up = b.output_array(Type::new(u)).into();
        let down = b.output_array(Type::new(u)).into();
        // Ten slots are ALWAYS runtime scalars, independent of widths/mode.
        let s0 = b.scalar(u).into();
        let s1 = b.scalar(u).into();
        let s2 = b.scalar(u).into();
        let s3 = b.scalar(u).into();
        let s4 = b.scalar(u).into();
        let s5 = b.scalar(u).into();
        let s6 = b.scalar(u).into();
        let s7 = b.scalar(u).into();
        let s8 = b.scalar(u).into();
        let s9 = b.scalar(u).into();
        match scatter {
            ScatterMode::Dynamic1d => {
                let g = b.scalar(u).into();
                let p = b.scalar(u).into();
                let d = b.scalar(u).into();
                scatter_public_weights::expand(
                    &mut b.scope,
                    source,
                    gate,
                    up,
                    down,
                    g,
                    p,
                    d,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    s8,
                    s9,
                );
            }
            ScatterMode::Specialized1d => scatter_public_weights_specialized::expand(
                &mut b.scope,
                source,
                gate,
                up,
                down,
                public_words[0],
                public_words[1],
                public_words[2],
                s0,
                s1,
                s2,
                s3,
                s4,
                s5,
                s6,
                s7,
                s8,
                s9,
            ),
            ScatterMode::Planes3d => {
                let g = b.scalar(u).into();
                let p = b.scalar(u).into();
                let d = b.scalar(u).into();
                scatter_public_weights_planes::expand(
                    &mut b.scope,
                    source,
                    gate,
                    up,
                    down,
                    g,
                    p,
                    d,
                    s0,
                    s1,
                    s2,
                    s3,
                    s4,
                    s5,
                    s6,
                    s7,
                    s8,
                    s9,
                );
            }
        }
        WgslCompiler::default()
            .compile(
                b.build(
                    KernelSettings::default()
                        .cube_dim(CubeDim::new_1d(256))
                        .kernel_name("public_expert_scatter"),
                ),
                &WgpuCompilationOptions::default(),
                mode,
                u,
            )
            .unwrap()
            .to_string()
    }
    #[test]
    fn every_scatter_has_ten_runtime_slot_loads_but_only_public_geometry_specializes() {
        use cubecl::server::ExecutionMode;
        for scatter in [
            ScatterMode::Dynamic1d,
            ScatterMode::Specialized1d,
            ScatterMode::Planes3d,
        ] {
            for mode in [ExecutionMode::Checked, ExecutionMode::Unchecked] {
                let source = scatter_wgsl(scatter, mode, [230_400, 230_400, 435_200]);
                assert!(source.contains("@compute") && source.contains("@binding(3)"));
                for slot in 0..10 {
                    assert!(
                        source.contains(&format!("scalars_u32[{slot}]")),
                        "missing runtime slot {slot}: {source}"
                    );
                }
                if scatter == ScatterMode::Specialized1d {
                    assert!(!source.contains("scalars_u32[10]"));
                    assert!(
                        source.contains("u32(230400)") && source.contains("u32(435200)"),
                        "{source}"
                    );
                } else {
                    for shape in 10..13 {
                        assert!(source.contains(&format!("scalars_u32[{shape}]")));
                    }
                }
                if scatter == ScatterMode::Planes3d {
                    assert!(
                        !source.contains(" / ") && !source.contains(" % "),
                        "3D introduced division: {source}"
                    );
                }
            }
        }
        // Guard the source-level specialization contract, not just one shader.
        let code = include_str!("expert_banks.rs");
        let signature = code
            .split("pub fn scatter_public_weights_specialized(")
            .nth(1)
            .unwrap()
            .split(") {")
            .next()
            .unwrap();
        let public_fields = signature
            .lines()
            .filter(|line| line.contains("#[comptime]"))
            .map(str::trim)
            .collect::<Vec<_>>();
        assert_eq!(
            public_fields,
            [
                "#[comptime] gate_words: usize,",
                "#[comptime] up_words: usize,",
                "#[comptime] down_words: usize,"
            ]
        );
        for rank in 0..10 {
            assert!(signature.contains(&format!("s{rank}: u32,")));
        }
    }
    #[test]
    fn public_width_specialization_changes_code_and_modes_do_not_capture_choices() {
        use cubecl::server::ExecutionMode::Unchecked;
        assert_ne!(
            scatter_wgsl(ScatterMode::Specialized1d, Unchecked, [3, 5, 2]),
            scatter_wgsl(ScatterMode::Specialized1d, Unchecked, [257, 513, 89])
        );
        assert_eq!(
            scatter_wgsl(ScatterMode::Dynamic1d, Unchecked, [3, 5, 2]),
            scatter_wgsl(ScatterMode::Dynamic1d, Unchecked, [257, 513, 89])
        );
        assert_eq!(
            scatter_wgsl(ScatterMode::Planes3d, Unchecked, [3, 5, 2]),
            scatter_wgsl(ScatterMode::Planes3d, Unchecked, [257, 513, 89])
        );
    }
    #[test]
    fn scatter_modes_and_three_dimensional_portable_grids_reject_invalid_before_upload() {
        assert_eq!(configured_scatter(None).unwrap(), ScatterMode::Dynamic1d);
        for (raw, expected) in [
            ("0", ScatterMode::Dynamic1d),
            ("1", ScatterMode::Specialized1d),
            ("2", ScatterMode::Planes3d),
        ] {
            assert_eq!(configured_scatter(Some(raw)).unwrap(), expected);
        }
        for invalid in ["", "3", "-1", " 1", "true"] {
            assert!(configured_scatter(Some(invalid)).is_err());
        }
        for down in [1_126_400, 1_740_800] {
            let layout = BatchLayout::new([921_600, 921_600, down], 72, 64 * 1024 * 1024).unwrap();
            for count in [1, 3, 10] {
                for mode in [
                    ScatterMode::Dynamic1d,
                    ScatterMode::Specialized1d,
                    ScatterMode::Planes3d,
                ] {
                    let grid = layout.scatter_grid(mode, count).unwrap();
                    assert!(grid.into_iter().all(|n| n > 0 && n <= 65_535));
                    if mode == ScatterMode::Planes3d {
                        assert_eq!(grid, [((down / 4).div_ceil(256)) as u32, 3, count as u32]);
                    } else {
                        assert_eq!(
                            grid,
                            [
                                ((layout.bytes(count).unwrap() / 4).div_ceil(256)) as u32,
                                1,
                                1
                            ]
                        );
                    }
                }
            }
            assert!(layout.scatter_grid(ScatterMode::Planes3d, 0).is_err());
            assert!(layout.scatter_grid(ScatterMode::Planes3d, 11).is_err());
        }
    }
    #[test]
    fn three_dimensional_copy_matches_original_bits_tails_and_all_ten_nontrivial_slots() {
        // Independent oracle gathers the original selected expert for each bank.
        let sizes = [257usize, 513, 89];
        let capacity = 72usize;
        let bundle = sizes.iter().sum::<usize>();
        let all = [
            (511usize, 71usize),
            (0, 0),
            (47, 55),
            (3, 7),
            (300, 34),
            (100, 63),
            (9, 1),
            (17, 64),
            (80, 13),
            (97, 26),
        ];
        let original: Vec<Vec<u32>> = sizes
            .iter()
            .enumerate()
            .map(|(plane, &words)| {
                (0..512 * words)
                    .map(|n| {
                        0x7f80_0000_u32
                            ^ (n as u32).wrapping_mul(2_654_435_761)
                            ^ ((plane as u32) << 28)
                    })
                    .collect()
            })
            .collect();
        let layout = BatchLayout::new(sizes.map(|n| n * 4), capacity, DEFAULT_BATCH_BYTES).unwrap();
        for count in [1, 3, 10] {
            let choices = &all[..count];
            let grid = layout.scatter_grid(ScatterMode::Planes3d, count).unwrap();
            let mut source = Vec::new();
            for &(expert, _) in choices {
                for (plane, &words) in sizes.iter().enumerate() {
                    source
                        .extend_from_slice(&original[plane][expert * words..(expert + 1) * words]);
                }
            }
            let mut copies: Vec<Vec<u32>> = sizes
                .iter()
                .map(|&n| vec![0xdead_beef_u32; n * capacity])
                .collect();
            let mut visits = vec![0_u8; source.len()];
            for rank in 0..grid[2] as usize {
                for plane in 0..grid[1] as usize {
                    for i in 0..grid[0] as usize * 256 {
                        if i < sizes[plane] {
                            let start = sizes[..plane].iter().sum::<usize>();
                            let src = rank * bundle + start + i;
                            copies[plane][choices[rank].1 * sizes[plane] + i] = source[src];
                            visits[src] += 1;
                        }
                    }
                }
            }
            assert!(visits.iter().all(|&n| n == 1));
            for (plane, &words) in sizes.iter().enumerate() {
                for slot in 0..capacity {
                    let actual = &copies[plane][slot * words..(slot + 1) * words];
                    if let Some(&(expert, _)) = choices.iter().find(|&&(_, s)| s == slot) {
                        assert_eq!(
                            actual,
                            &original[plane][expert * words..(expert + 1) * words]
                        );
                    } else {
                        assert!(actual.iter().all(|&n| n == 0xdead_beef));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod capacity_tests {
    use super::*;
    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;
    #[test]
    fn actual_q4_q5_q8_original_extents_keep_72_in_the_same_pages_as_64() {
        // Independently derive original GGUF block sizes and row geometry.
        // gate/up: 2560 inputs x640 outputs Q4_K; down: 640x2560 Q5_0/Q8_0.
        let shapes = [
            (
                quant_matvec::Q4_K,
                2560,
                640,
                2560 / 256 * 144,
                921_600,
                64 * MIB,
            ),
            (
                quant_matvec::Q5_0,
                640,
                2560,
                640 / 32 * 22,
                1_126_400,
                128 * MIB,
            ),
            (
                quant_matvec::Q8_0,
                640,
                2560,
                640 / 32 * 34,
                1_740_800,
                128 * MIB,
            ),
        ];
        for (dtype, columns, rows, row_bytes, expert, page) in shapes {
            quant_matvec::validate_layout(dtype, rows, columns, expert, columns, rows).unwrap();
            let (actual, b64) = packed_bank_bytes(row_bytes, rows, 64).unwrap();
            let (_, b72) =
                packed_bank_bytes(row_bytes, rows, crate::bank_slots::MAX_FLAT_CAPACITY).unwrap();
            assert_eq!(actual, expert);
            assert_eq!(b64, expert * 64);
            assert_eq!(b72, expert * 72);
            assert!(b72 <= 128 * MIB);
            assert_eq!(b64.next_power_of_two(), page);
            assert_eq!(b72.next_power_of_two(), page);
        }
        assert_eq!(
            packed_bank_bytes(2560 / 256 * 144, 640, 72).unwrap().1,
            66_355_200
        );
        assert_eq!(
            packed_bank_bytes(640 / 32 * 22, 2560, 72).unwrap().1,
            81_100_800
        );
        assert_eq!(
            packed_bank_bytes(640 / 32 * 34, 2560, 72).unwrap().1,
            125_337_600
        );
        // 73 would double BOTH gate/up charges and is not admitted.
        assert_eq!((921_600usize * 73).next_power_of_two(), 128 * MIB);
        assert!(validate_capacity(73).is_err());
    }
    #[test]
    fn whole_graph_budget_is_bounded_before_bank_allocations() {
        let b72 = 48 * (64 + 64 + 128) * MIB;
        assert_eq!(b72, 12 * GIB);
        validate_residency_budget(b72, 8 * GIB, 26 * GIB).unwrap();
        assert!(validate_residency_budget(b72, 6 * GIB, 10 * GIB).is_err());
        // Nvidia's original 16-slot geometry remains 3 GiB across 48 layers.
        validate_residency_budget(48 * (16 + 16 + 32) * MIB, 6 * GIB, 10 * GIB).unwrap();
        assert!(validate_residency_budget(b72, 8 * GIB, 26 * GIB + 1).is_err());
        assert!(validate_residency_budget(usize::MAX, 1, 26 * GIB).is_err());
        assert!(validate_residency_budget(0, 0, RESIDENCY_MARGIN - 1).is_err());
        for invalid in [0, 9, 73, usize::MAX] {
            assert!(validate_capacity(invalid).is_err());
        }
        assert!(packed_bank_bytes(640 / 32 * 34, 2560, 78).is_err());
        assert!(packed_bank_bytes(usize::MAX, 2, 72).is_err());
    }
}

#[cfg(test)]
mod segmented_tests {
    use super::*;
    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;
    #[test]
    fn expanded160_adds_only_one_original_sixteen_slot_shard_with_full_page_charge() {
        assert_eq!(segment_capacities(160).unwrap(), [72, 72, 16]);
        for down in [440, 680] {
            let mut total = 0;
            for (row, rows) in [(1440, 640), (1440, 640), (down, 2560)] {
                let (_, parts) = packed_segments(row, rows, 160).unwrap();
                assert_eq!(parts, [row * rows * 72, row * rows * 72, row * rows * 16]);
                total += parts
                    .into_iter()
                    .map(usize::next_power_of_two)
                    .sum::<usize>();
            }
            assert_eq!(48 * total, 27 * GIB);
            validate_residency_budget_with_limit(48 * total, 3 * GIB, 31 * GIB, 31 * GIB).unwrap();
            assert!(
                validate_residency_budget_with_limit(48 * total, 5 * GIB, 31 * GIB, 31 * GIB)
                    .is_err()
            );
        }
        for (slot, index) in [(0, 0), (71, 0), (72, 1), (143, 1), (144, 2), (159, 2)] {
            assert_eq!(
                resolve_segment([(0, 72), (72, 72), (144, 16)], slot).unwrap(),
                index
            );
        }
        assert!(resolve_segment([(0, 72), (72, 72), (144, 16)], 160).is_err());
    }
    #[test]
    fn expanded144_reuses_the_same_complete_pools_and_bounds_both_shards() {
        assert_eq!(segment_capacities(144).unwrap(), [72, 72]);
        for down in [440, 680] {
            let mut total = 0;
            for (row, rows) in [(1440, 640), (1440, 640), (down, 2560)] {
                let (_, parts) = packed_segments(row, rows, 144).unwrap();
                assert_eq!(parts, [row * rows * 72; 2]);
                total += parts
                    .into_iter()
                    .map(usize::next_power_of_two)
                    .sum::<usize>();
            }
            assert_eq!(48 * total, 24 * GIB);
            validate_residency_budget_with_limit(48 * total, 5 * GIB, 31 * GIB, 31 * GIB).unwrap();
        }
        for (slot, index) in [(0, 0), (71, 0), (72, 1), (143, 1)] {
            assert_eq!(resolve_segment([(0, 72), (72, 72)], slot).unwrap(), index);
        }
        assert!(resolve_segment([(0, 72), (72, 72)], 144).is_err());
    }
    #[test]
    fn expanded128_proves_two_original_shards_and_separate_complete_envelope() {
        assert_eq!(segment_capacities(128).unwrap(), [64, 64]);
        for down in [440, 680] {
            let mut total = 0;
            for (row, rows) in [(1440, 640), (1440, 640), (down, 2560)] {
                let (_, parts) = packed_segments(row, rows, 128).unwrap();
                assert_eq!(parts, [row * rows * 64; 2]);
                total += parts
                    .into_iter()
                    .map(usize::next_power_of_two)
                    .sum::<usize>();
            }
            assert_eq!(48 * total, 24 * GIB);
            validate_residency_budget_with_limit(48 * total, 5 * GIB, 31 * GIB, 31 * GIB).unwrap();
            assert!(validate_residency_budget(48 * total, 5 * GIB, 31 * GIB).is_err());
            assert!(
                validate_residency_budget_with_limit(48 * total, 7 * GIB, 31 * GIB, 31 * GIB)
                    .is_err()
            );
        }
        assert!(check_ordered_segmented(128, true).is_err());
        for (explicit, block, batch, word, resident) in [
            (None, true, false, false, true),
            (Some(5 * GIB), false, false, false, true),
            (Some(5 * GIB), true, true, false, true),
            (Some(5 * GIB), true, false, true, true),
            (Some(5 * GIB), true, false, false, false),
        ] {
            assert!(
                validate_segmented_configuration(
                    128,
                    explicit,
                    5 * GIB,
                    block,
                    batch,
                    word,
                    resident
                )
                .is_err()
            );
        }
    }
    #[test]
    fn exact_original_segment_charge_and_full_envelope() {
        let mut layer = 0;
        for (row, rows, expert, expected) in [
            (1440, 640, 921600, [64, 32]),
            (1440, 640, 921600, [64, 32]),
            (440, 2560, 1126400, [128, 64]),
        ] {
            let (e, bytes) = packed_segments(row, rows, 96).unwrap();
            assert_eq!(e, expert);
            assert_eq!(bytes, [expert * 64, expert * 32]);
            let pages: Vec<_> = bytes.iter().map(|n| n.next_power_of_two()).collect();
            assert_eq!(pages, expected.map(|n| n * MIB));
            assert!(bytes.iter().all(|&n| n <= 128 * MIB));
            layer += pages.iter().sum::<usize>();
        }
        let (_, q8) = packed_segments(680, 2560, 96).unwrap();
        assert_eq!(
            q8.iter().map(|n| n.next_power_of_two()).collect::<Vec<_>>(),
            [128 * MIB, 64 * MIB]
        );
        assert_eq!(layer, 384 * MIB);
        assert_eq!(48 * layer, 18 * GIB);
        validate_residency_budget(48 * layer, 6 * GIB, 26 * GIB).unwrap();
        assert!(validate_residency_budget(48 * layer, 8 * GIB, 26 * GIB).is_err());
        assert!(validate_residency_budget(48 * layer, 6 * GIB, 10 * GIB).is_err());
        // Existing full admission must ALSO fit public vectors and live
        // baseline. These are conditional ceiling arithmetic, not telemetry.
        let private_plan = 785186816;
        let base = 18 * GIB + 6 * GIB + private_plan + 64 * MIB + 64 * MIB + 40 * MIB;
        assert_eq!(26 * GIB - base, 1186136064);
        assert_eq!(26 * GIB - base, 1131 * MIB + 196608);
        assert!(packed_segments(usize::MAX, 2, 96).is_err());
        for cap in [9, 73, 95, 97, 127, 129, usize::MAX] {
            assert!(segment_capacities(cap).is_err());
        }
        assert_eq!(segment_capacities(72).unwrap(), [72]);
    }
    #[test]
    fn every_original_triple_must_match_and_unsupported_options_fail() {
        let gate = [2560, 640, 512];
        let down = [640, 2560, 512];
        for dtype in [TensorType::Q5_0, TensorType::Q8_0] {
            validate_segmented_metadata([
                (TensorType::Q4K, &gate),
                (TensorType::Q4K, &gate),
                (dtype, &down),
            ])
            .unwrap();
        }
        for layer in 0..48 {
            let mut invalid_gate = gate;
            invalid_gate[2] = 511;
            for pos in 0..48 {
                let g = if pos == layer { &invalid_gate } else { &gate };
                let result = validate_segmented_metadata([
                    (TensorType::Q4K, g),
                    (TensorType::Q4K, &gate),
                    (TensorType::Q5_0, &down),
                ]);
                assert_eq!(result.is_err(), pos == layer);
            }
        }
        for bad in [TensorType::F32, TensorType::Q6K, TensorType::Q4K] {
            assert!(
                validate_segmented_metadata([
                    (TensorType::Q4K, &gate),
                    (TensorType::Q4K, &gate),
                    (bad, &down)
                ])
                .is_err()
            );
        }
        let wrong = [640, 2560, 512, 1];
        assert!(
            validate_segmented_metadata([
                (TensorType::Q4K, &gate),
                (TensorType::Q4K, &gate),
                (TensorType::Q5_0, &wrong)
            ])
            .is_err()
        );
        let wrong = [2560, 641, 512];
        assert!(
            validate_segmented_metadata([
                (TensorType::Q4K, &wrong),
                (TensorType::Q4K, &gate),
                (TensorType::Q5_0, &down)
            ])
            .is_err()
        );
        validate_segmented_configuration(96, Some(6 * GIB), 6 * GIB, true, false, false, true)
            .unwrap();
        for (cache, block, batch, word, resident) in [
            (None, true, false, false, true),
            (Some(8 * GIB), true, false, false, true),
            (Some(6 * GIB), false, false, false, true),
            (Some(6 * GIB), true, true, false, true),
            (Some(6 * GIB), true, false, true, true),
            (Some(6 * GIB), true, false, false, false),
        ] {
            assert!(
                validate_segmented_configuration(96, cache, 6 * GIB, block, batch, word, resident)
                    .is_err()
            );
        }
        validate_segmented_configuration(64, None, 8 * GIB, false, true, true, false).unwrap();
        check_ordered_segmented(64, true).unwrap();
        check_ordered_segmented(96, false).unwrap();
        assert!(check_ordered_segmented(96, true).is_err());
        assert_eq!(configured_static_cache(None, 8 * GIB).unwrap(), 8 * GIB);
        assert_eq!(
            configured_static_cache(Some("6442450944"), 8 * GIB).unwrap(),
            6 * GIB
        );
        for invalid in [
            "",
            "0",
            "67108863",
            "17179869185",
            "-1",
            "+6442450944",
            " 6442450944",
            "1.5",
            "184467440737095516160",
        ] {
            assert!(configured_static_cache(Some(invalid), 8 * GIB).is_err());
        }
    }
    #[test]
    fn public_existing_spans_cover_only_original_slots_and_preserve_padding() {
        let bytes = 1024;
        for capacity in [16usize, 32, 64] {
            let logical = bytes * capacity;
            let allocated = logical as u64 + 4096;
            for local in 0..capacity {
                let (start, end_trim) =
                    public_slot_span(bytes, local, capacity, logical / 4, allocated).unwrap();
                assert_eq!(start, (local * bytes) as u64);
                assert_eq!(allocated - start - end_trim, bytes as u64);
                assert!(end_trim >= 4096);
            }
            assert!(public_slot_span(bytes, capacity, capacity, logical / 4, allocated).is_err());
            assert!(public_slot_span(bytes, 0, capacity, logical / 4, logical as u64 - 1).is_err());
            assert!(public_slot_span(bytes, 0, capacity, logical / 4 + 1, allocated).is_err());
        }
        for bytes in [0usize, 3, usize::MAX] {
            assert!(public_slot_span(bytes, 0, 1, 1, 4).is_err());
        }
        assert!(public_slot_span(4, 0, usize::MAX, usize::MAX, u64::MAX).is_err());
    }
    #[test]
    fn daslab_existing_views_cover_both72_segments_without_their_page_tails() {
        // Independent pinned original expert byte counts, including Q2_0 and
        // the rare IQ4_XS gate/up. Every projection preserves a full root tail.
        for bytes in [627200usize, 704000, 524800, 870400, 921600, 460800] {
            let logical = bytes * 72;
            let page = (logical.max(32768)).next_power_of_two() as u64;
            for global in [0usize, 71, 72, 143] {
                let shard = resolve_segment([(0, 72), (72, 72)], global).unwrap();
                let local = global - shard * 72;
                let (first, trim) = public_slot_span(bytes, local, 72, logical / 4, page).unwrap();
                assert_eq!(first, (local * bytes) as u64);
                assert_eq!(page - first - trim, bytes as u64);
                assert_eq!(first + bytes as u64 + trim, page);
                assert!(trim >= page - logical as u64);
                assert!(first.is_multiple_of(4));
                assert!(first + bytes as u64 <= logical as u64);
            }
        }
    }
    #[test]
    fn public_existing_window_keeps_original_source_precharge_and_bounded_host_capacities() {
        for count in [1usize, 2, 10] {
            for down in [460800usize, 921600] {
                let capacities: Vec<_> = (0..count)
                    .flat_map(|_| [524800usize, 524800, down])
                    .map(|n| (n, n))
                    .collect();
                let charge: u64 = capacities
                    .iter()
                    .map(|&(n, _)| crate::main_admission::page_bytes(n as u64).unwrap())
                    .sum();
                let logical: u64 = capacities.iter().map(|&(n, _)| n as u64).sum();
                let metadata = count * 3 * std::mem::size_of::<(Handle, Vec<u8>)>();
                assert_eq!(
                    public_existing_batch_bounds(
                        count * 3,
                        metadata,
                        capacities.iter().copied(),
                        charge,
                    )
                    .unwrap(),
                    logical
                );
                assert!(charge <= 30 * 1024 * 1024);
                assert!(
                    public_existing_batch_bounds(
                        count * 3,
                        metadata,
                        capacities.iter().copied(),
                        charge - 1,
                    )
                    .is_err()
                );
            }
        }
        for (count, metadata, entries, charge) in [
            (0, 0, vec![], 0),
            (31, 0, vec![(4, 4); 31], 31 * 32768),
            (1, PUBLIC_EXISTING_METADATA + 1, vec![(4, 4)], 32768),
            (1, 0, vec![(4, 4)], 40 * 1024 * 1024 + 1),
            (1, 0, vec![(0, 0)], 32768),
            (1, 0, vec![(3, 3)], 32768),
            (1, 0, vec![(4, 3)], 32768),
            (1, 0, vec![(4, 32769)], 32768),
            (2, 0, vec![(4, 4)], 32768),
        ] {
            assert!(public_existing_batch_bounds(count, metadata, entries, charge).is_err());
        }
    }
    #[test]
    fn global_slots_resolve_once_across_boundary_and_failure_is_transactional() {
        let parts = [(0, 64), (64, 32)];
        for (slot, expected) in [(0, 0), (63, 0), (64, 1), (95, 1)] {
            assert_eq!(resolve_segment(parts, slot).unwrap(), expected);
        }
        for slot in [96, usize::MAX] {
            assert!(resolve_segment(parts, slot).is_err());
        }
        let mut bank = BankSlots::new(96, 512).unwrap();
        let initial: Vec<_> = (0..96).collect();
        let p = bank.plan(&initial).unwrap();
        bank.commit(p).unwrap();
        // Global LRU remains one plan. All ten hits protect their physical
        // slots regardless of segment. A partial physical fill can corrupt
        // either segment: dropping host plan is NOT permission to reuse bank.
        let p = bank
            .plan(&[95, 63, 64, 200, 0, 65, 201, 94, 62, 32])
            .unwrap();
        assert_eq!(p.uploads, [(200, 1), (201, 2)]);
        assert_eq!(p.slots, [95, 63, 64, 1, 0, 65, 2, 94, 62, 32]);
        drop(p);
        let p = bank
            .plan(&[95, 63, 64, 200, 0, 65, 201, 94, 62, 32])
            .unwrap();
        assert_eq!(p.uploads, [(200, 1), (201, 2)]);
        bank.commit(p).unwrap();
    }
}

#[cfg(test)]
mod q3_profile_tests {
    use super::*;
    const GIB: usize = 1024 * 1024 * 1024;

    /// Explicit local original-model admission only: no device client,
    /// generation, expert payload copy, or model-wide float expansion.
    #[test]
    #[ignore = "set KUNO_TEST_Q3_DIR to the pinned Unsloth UD-Q3_K_XL shard directory"]
    fn installed_q3_sharded_profile_admits_exact_original_graph_and_owned_charges() {
        use kuno_text::{GgufFile, GgufLimits, model::Qwen4Exp};
        let directory = std::env::var_os("KUNO_TEST_Q3_DIR")
            .map(std::path::PathBuf::from)
            .expect("KUNO_TEST_Q3_DIR must explicitly select the original public shard directory");
        let paths = [
            "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf",
            "Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf",
            "Qwen3.8-Flash-Next-UD-Q3_K_XL-00003-of-00003.gguf",
        ]
        .map(|name| directory.join(name));
        let file = GgufFile::open_shards(&paths, GgufLimits::default()).unwrap();
        let model = Qwen4Exp::bind(&file).unwrap();
        let graph = model.bound_graph();
        let profile = OriginalPackedProfile::UnslothQ3;
        profile.validate_graph(graph).unwrap();
        assert_eq!(
            preflight_residency_with_profile(graph, 144, 5 * GIB, 31 * GIB, profile).unwrap(),
            19_998_441_472
        );
        assert_eq!(
            crate::resident_model::ple_batch::PleBatchWorkspace::allocation_requests_with_profile(
                graph, profile
            )
            .unwrap()
            .iter()
            .sum::<u64>(),
            6_258_688
        );
        graph.verify_unchanged().unwrap();
    }

    #[test]
    fn q3_allocation_api_uses_inspected_original_tensor_geometry_before_device_work() {
        use kuno_text::{GgufFile, GgufLimits};
        use std::fs::OpenOptions;
        use std::io::Write;
        static NEXT: AtomicU64 = AtomicU64::new(0);
        struct Fixture(std::path::PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        fn text(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        for layer in [0, 2, 4] {
            let fixture = Fixture(std::env::temp_dir().join(format!(
                "kuno-q3-owner-{}-{}.gguf",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            )));
            let kinds = q3_expert_types(layer).unwrap();
            let mut bytes = b"GGUF".to_vec();
            bytes.extend(3u32.to_le_bytes());
            bytes.extend(3u64.to_le_bytes());
            bytes.extend(0u64.to_le_bytes());
            let mut offset = 0u64;
            for (index, suffix) in ["gate", "up", "down"].into_iter().enumerate() {
                text(&mut bytes, &format!("blk.{layer}.ffn_{suffix}_exps.weight"));
                bytes.extend(3u32.to_le_bytes());
                let dimensions = if index == 2 {
                    [640u64, 2560, 512]
                } else {
                    [2560u64, 640, 512]
                };
                for dimension in dimensions {
                    bytes.extend(dimension.to_le_bytes());
                }
                bytes.extend(kinds[index].gguf_id().to_le_bytes());
                bytes.extend(offset.to_le_bytes());
                offset +=
                    kinds[index].row_bytes(dimensions[0]).unwrap() * dimensions[1] * dimensions[2];
            }
            bytes.resize(bytes.len().next_multiple_of(32), 0);
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&fixture.0)
                .unwrap();
            file.write_all(&bytes).unwrap();
            // Sparse public extent only: no weight payload allocation/write/read,
            // device client, map or GPU command is used by this metadata test.
            file.set_len(bytes.len() as u64 + offset).unwrap();
            drop(file);
            let model = GgufFile::open(&fixture.0, GgufLimits::default()).unwrap();
            let weights = RoutedWeights {
                gate: model
                    .tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))
                    .unwrap(),
                up: model
                    .tensor(&format!("blk.{layer}.ffn_up_exps.weight"))
                    .unwrap(),
                down: model
                    .tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
                    .unwrap(),
            };
            let requests = ExpertBanks::allocation_requests_with_profile(
                weights,
                144,
                OriginalPackedProfile::UnslothQ3,
            )
            .unwrap();
            assert_eq!(requests.len(), 6);
            let expected_gate = if layer == 2 { 62_668_800 } else { 45_158_400 };
            let expected_down = if layer == 0 { 66_355_200 } else { 125_337_600 };
            assert_eq!(
                requests,
                [
                    expected_gate,
                    expected_gate,
                    expected_gate,
                    expected_gate,
                    expected_down,
                    expected_down
                ]
            );
            assert_eq!(
                ExpertBanks::allocation_with_profile(
                    weights,
                    144,
                    OriginalPackedProfile::UnslothQ3
                )
                .unwrap(),
                if layer == 0 {
                    384 * 1024 * 1024
                } else {
                    512 * 1024 * 1024
                }
            );
            assert!(ExpertBanks::allocation_requests(weights, 144).is_err());
            assert!(
                ExpertBanks::allocation_requests_with_profile(
                    weights,
                    128,
                    OriginalPackedProfile::UnslothQ3
                )
                .is_err()
            );
            assert!(
                validate_profile_weights(
                    OriginalPackedProfile::UnslothQ3,
                    RoutedWeights {
                        gate: weights.gate,
                        up: weights.down,
                        down: weights.up,
                    }
                )
                .is_err()
            );
        }
    }

    #[test]
    fn q3_profile_preserves_exact_mixed_types_in_all_48_layers() {
        let gate = [2560, 640, 512];
        let down = [640, 2560, 512];
        let mut iq4_gate_layers = 0;
        let mut q8_down_layers = 0;
        for layer in 0..48 {
            let kinds = q3_expert_types(layer).unwrap();
            iq4_gate_layers += usize::from(kinds[0] == TensorType::Iq4Xs);
            q8_down_layers += usize::from(kinds[2] == TensorType::Q8_0);
            assert_eq!(
                kinds[0],
                if layer == 2 {
                    TensorType::Iq4Xs
                } else {
                    TensorType::Iq3Xxs
                }
            );
            assert_eq!(
                kinds[2],
                if [2, 4, 30, 46, 47].contains(&layer) {
                    TensorType::Q8_0
                } else {
                    TensorType::Iq4Nl
                }
            );
            let parts = [
                (kinds[0], &gate[..]),
                (kinds[1], &gate[..]),
                (kinds[2], &down[..]),
            ];
            validate_profile_metadata(OriginalPackedProfile::UnslothQ3, layer, parts).unwrap();
            assert!(validate_profile_metadata(OriginalPackedProfile::Q4, layer, parts).is_err());
            for position in 0..3 {
                let mut bad = parts;
                bad[position].0 = TensorType::F32;
                assert!(
                    validate_profile_metadata(OriginalPackedProfile::UnslothQ3, layer, bad)
                        .is_err()
                );
            }
        }
        assert_eq!((iq4_gate_layers, q8_down_layers), (1, 5));
        assert!(q3_expert_types(48).is_err());
        let wrong = [2560, 640, 511];
        assert!(
            validate_profile_metadata(
                OriginalPackedProfile::UnslothQ3,
                0,
                [
                    (TensorType::Iq3Xxs, &wrong),
                    (TensorType::Iq3Xxs, &gate),
                    (TensorType::Iq4Nl, &down)
                ]
            )
            .is_err()
        );
        // A rare layer cannot accidentally inherit the common layer profile.
        assert!(
            validate_profile_metadata(
                OriginalPackedProfile::UnslothQ3,
                2,
                [
                    (TensorType::Iq3Xxs, &gate),
                    (TensorType::Iq3Xxs, &gate),
                    (TensorType::Q8_0, &down)
                ]
            )
            .is_err()
        );
        assert!(
            validate_profile_metadata(
                OriginalPackedProfile::UnslothQ3,
                4,
                [
                    (TensorType::Iq3Xxs, &gate),
                    (TensorType::Iq3Xxs, &gate),
                    (TensorType::Iq4Nl, &down)
                ]
            )
            .is_err()
        );
    }

    #[test]
    fn q3_profile_derives_real_packed_rows_and_full_physical_bank_charge() {
        let mut logical = 0usize;
        let mut physical = 0usize;
        for layer in 0..48 {
            let kinds = q3_expert_types(layer).unwrap();
            for (index, kind) in kinds.into_iter().enumerate() {
                let (columns, rows) = if index == 2 { (640, 2560) } else { (2560, 640) };
                let row = kind.row_bytes(columns).unwrap() as usize;
                assert_eq!(
                    row,
                    match kind {
                        TensorType::Iq3Xxs => 980,
                        TensorType::Iq4Xs => 1360,
                        TensorType::Iq4Nl => 360,
                        TensorType::Q8_0 => 680,
                        _ => unreachable!(),
                    }
                );
                let (expert, segments) = packed_segments(row, rows, 144).unwrap();
                assert_eq!(segments, [expert * 72; 2]);
                assert_eq!(
                    expert,
                    match kind {
                        TensorType::Iq3Xxs => 627200,
                        TensorType::Iq4Xs => 870400,
                        TensorType::Iq4Nl => 921600,
                        TensorType::Q8_0 => 1740800,
                        _ => unreachable!(),
                    }
                );
                for bytes in segments {
                    assert!(bytes <= 128 * 1024 * 1024);
                    assert!(bytes.is_multiple_of(4));
                    logical += bytes;
                    physical += bytes.next_power_of_two();
                }
            }
        }
        assert_eq!(logical, 15_700_377_600);
        assert_eq!(physical, UNSLOTH_Q3_BANK_BYTES_144);
        assert_eq!(physical, 18 * GIB + 640 * 1024 * 1024);
        validate_profile_residency(
            OriginalPackedProfile::UnslothQ3,
            144,
            physical,
            5 * GIB,
            31 * GIB,
        )
        .unwrap();
        validate_profile_residency(
            OriginalPackedProfile::UnslothQ3,
            144,
            physical,
            8 * GIB,
            31 * GIB,
        )
        .unwrap();
        for (cap, bytes, cache, device) in [
            (144, logical, 5 * GIB, 31 * GIB),
            (144, 24 * GIB, 5 * GIB, 31 * GIB),
            (128, physical, 5 * GIB, 31 * GIB),
            (144, physical, 6 * GIB, 31 * GIB),
            (144, physical, 10 * GIB, 31 * GIB),
            (144, physical, 5 * GIB, 26 * GIB),
            (144, usize::MAX, 5 * GIB, 31 * GIB),
        ] {
            assert!(
                validate_profile_residency(
                    OriginalPackedProfile::UnslothQ3,
                    cap,
                    bytes,
                    cache,
                    device
                )
                .is_err()
            );
        }
        validate_profile_residency(OriginalPackedProfile::Q4, 144, 24 * GIB, 5 * GIB, 31 * GIB)
            .unwrap();
        assert!(
            validate_profile_residency(OriginalPackedProfile::Q4, 144, physical, 5 * GIB, 31 * GIB)
                .is_err()
        );
        assert!(
            validate_profile_residency(OriginalPackedProfile::Q4, 144, 24 * GIB, 8 * GIB, 31 * GIB)
                .is_err()
        );
    }

    #[test]
    fn q3_profile_is_explicit_and_rejects_every_q4_shortcut() {
        assert_eq!(OriginalPackedProfile::default(), OriginalPackedProfile::Q4);
        let q3 = OriginalPackedProfile::UnslothQ3;
        validate_segmented_configuration_with_profile(
            144,
            Some(5 * GIB),
            5 * GIB,
            false,
            false,
            false,
            true,
            q3,
        )
        .unwrap();
        for (capacity, cache, block, batch, word, resident) in [
            (144, None, false, false, false, true),
            (128, Some(5 * GIB), false, false, false, true),
            (144, Some(5 * GIB), true, false, false, true),
            (144, Some(5 * GIB), false, true, false, true),
            (144, Some(5 * GIB), false, false, true, true),
            (144, Some(5 * GIB), false, false, false, false),
        ] {
            assert!(
                validate_segmented_configuration_with_profile(
                    capacity,
                    cache,
                    5 * GIB,
                    block,
                    batch,
                    word,
                    resident,
                    q3
                )
                .is_err()
            );
        }
        q3.validate_shortcuts(false, false, false, false, false)
            .unwrap();
        for index in 0..5 {
            let mut options = [false; 5];
            options[index] = true;
            assert!(
                q3.validate_shortcuts(options[0], options[1], options[2], options[3], options[4])
                    .is_err()
            );
        }
        OriginalPackedProfile::Q4
            .validate_shortcuts(true, true, true, true, true)
            .unwrap();
    }

    #[test]
    fn q3_profile_matches_original_main_and_paired_ple_formats() {
        for (name, dimensions, dtype) in [
            ("token_embd.weight", &[2560, 248320][..], TensorType::Q8_0),
            ("output.weight", &[2560, 248320][..], TensorType::Q6K),
            (
                "per_layer_token_embd.weight",
                &[160, 320001536][..],
                TensorType::Iq4Nl,
            ),
            (
                "blk.3.indexer.q_proj.weight",
                &[2560, 512][..],
                TensorType::Bf16,
            ),
            (
                "blk.3.indexer.k_proj.weight",
                &[2560, 128][..],
                TensorType::Bf16,
            ),
            (
                "blk.0.ffn_gate_inp.weight",
                &[2560, 512][..],
                TensorType::F32,
            ),
            (
                "blk.0.hc_attn_inject.weight",
                &[10240, 4][..],
                TensorType::F32,
            ),
            ("blk.0.ssm_alpha.weight", &[2560, 48][..], TensorType::F32),
            ("blk.0.ssm_conv1d.weight", &[4, 10240][..], TensorType::F32),
            ("blk.1.ple_conv1d.weight", &[4, 10240][..], TensorType::F32),
            ("blk.0.hc_attn_norm.weight", &[10240][..], TensorType::F32),
            (
                "blk.0.hc_attn_down.weight",
                &[10240, 320][..],
                TensorType::Q8_0,
            ),
        ] {
            assert_eq!(q3_main_dtype(name, dimensions), dtype);
        }
        let dimensions = [160, 320001536];
        let q3 = OriginalPackedProfile::UnslothQ3;
        q3.validate_ple(TensorType::Iq4Nl, &dimensions, 90).unwrap();
        OriginalPackedProfile::Q4
            .validate_ple(TensorType::Q5_0, &dimensions, 110)
            .unwrap();
        for (dtype, dims, bytes) in [
            (TensorType::Q5_0, &dimensions[..], 110),
            (TensorType::Iq4Nl, &dimensions[..], 110),
            (TensorType::Iq4Nl, &[160, 320001535][..], 90),
            (TensorType::Iq4Nl, &[159, 320001536][..], 90),
        ] {
            assert!(q3.validate_ple(dtype, dims, bytes).is_err());
        }
        assert!(
            OriginalPackedProfile::Q4
                .validate_ple(TensorType::Iq4Nl, &dimensions, 90)
                .is_err()
        );
    }
}

#[cfg(test)]
mod daslab_profile_tests {
    use super::*;
    #[test]
    fn original_triples_use_exact_row_geometry_and_page_charge() {
        let profile = OriginalPackedProfile::DaslabIq3;
        let mut physical = 0usize;
        for layer in 0..48 {
            let raw = crate::daslab_profile::routed_types(layer).unwrap();
            let kinds = raw.map(|id| TensorType::from_gguf_id(id).unwrap());
            let parts = [
                (kinds[0], &[2560, 640, 512][..]),
                (kinds[1], &[2560, 640, 512][..]),
                (kinds[2], &[640, 2560, 512][..]),
            ];
            validate_profile_metadata(profile, layer, parts).unwrap();
            for (index, kind) in kinds.into_iter().enumerate() {
                let (columns, rows) = if index == 2 { (640, 2560) } else { (2560, 640) };
                let row = columns / kind.block_elements() as usize * kind.block_bytes() as usize;
                physical += 2 * (row * rows * 72).next_power_of_two();
            }
            let mut wrong = parts;
            wrong[2].0 = TensorType::Q8_0;
            assert!(validate_profile_metadata(profile, layer, wrong).is_err());
        }
        assert_eq!(physical, 18_723_373_056);
        validate_profile_residency(profile, 144, physical, 8 << 30, 31 << 30).unwrap();
        assert!(validate_profile_residency(profile, 144, physical + 1, 8 << 30, 31 << 30).is_err());
        assert!(validate_profile_residency(profile, 128, physical, 8 << 30, 31 << 30).is_err());
        profile
            .validate_ple(TensorType::Iq4Nl, &[160, 320001536], 90)
            .unwrap();
        assert!(
            profile
                .validate_shortcuts(false, true, false, false, false)
                .is_err()
        );
    }
}
