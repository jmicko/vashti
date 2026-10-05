//! Transient packed expert banks for a fixed public laboratory experiment.
//! Stores original weights only; never persists expert access/rank history.
use crate::bank_slots::{BankSlots, MAX_CAPACITY, MIN_ROUTED_CAPACITY};
use crate::quant_matvec;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::{BoundGraph, RoutedWeights};
use kuno_text::{Error, Result, TensorView};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use zeroize::Zeroize;

/// Immutable allocation extent; expose only readonly handles/geometry. Filling
/// changes PUBLIC weight bytes, never these allocations or their offsets.
pub struct Bank {
    handle: Handle,
    expert_bytes: usize,
    words: usize,
    reserved: usize,
}
impl Bank {
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
    pub fn expert_bytes(&self) -> usize {
        self.expert_bytes
    }
    pub fn words(&self) -> usize {
        self.words
    }
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
}

#[cfg(test)]
pub(crate) fn fixture_bank_instance() -> BankInstance {
    next_instance(&NEXT_BANK_INSTANCE).expect("public test bank identity")
}

fn bank_bytes(t: TensorView<'_>, capacity: usize) -> Result<(usize, usize)> {
    if t.info().dimensions().len() != 3 {
        return Err(Error::InvalidFormat("expert bank rank"));
    }
    packed_bank_bytes(
        t.info().row_bytes() as usize,
        t.info().dimensions()[1] as usize,
        capacity,
    )
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
    if !(MIN_ROUTED_CAPACITY..=MAX_CAPACITY).contains(&capacity) {
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
    if device_bytes > MAX_DEVICE_BYTES
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
    validate_capacity(capacity)?;
    let mut total = 0usize;
    for layer in 0..graph.geometry().layers {
        let prefix = format!("blk.{layer}.");
        let weights = RoutedWeights {
            gate: graph.tensor(&format!("{prefix}ffn_gate_exps.weight"))?,
            up: graph.tensor(&format!("{prefix}ffn_up_exps.weight"))?,
            down: graph.tensor(&format!("{prefix}ffn_down_exps.weight"))?,
        };
        total = total
            .checked_add(ExpertBanks::allocation(weights, capacity)?)
            .ok_or(Error::ArithmeticOverflow)?;
    }
    validate_residency_budget(total, cache_bytes, device_bytes)?;
    Ok(total)
}
impl ExpertBanks {
    pub fn allocation(weights: RoutedWeights<'_>, capacity: usize) -> Result<usize> {
        validate_capacity(capacity)?;
        let mut total = 0usize;
        for t in [weights.gate, weights.up, weights.down] {
            total = total
                .checked_add(bank_bytes(t, capacity)?.1.next_power_of_two())
                .ok_or(Error::ArithmeticOverflow)?;
        }
        Ok(total)
    }
    pub fn new(
        client: &ComputeClient<WgpuRuntime>,
        weights: RoutedWeights<'_>,
        capacity: usize,
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
        Self::allocation(weights, capacity)?;
        let create = |t| -> Result<Bank> {
            let (expert_bytes, bytes) = bank_bytes(t, capacity)?;
            Ok(Bank {
                handle: client.empty(bytes),
                expert_bytes,
                words: bytes / 4,
                reserved: bytes.next_power_of_two(),
            })
        };
        let instance = next_instance(&NEXT_BANK_INSTANCE)?;
        Ok(Self {
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
        if [weights.gate, weights.up, weights.down]
            .iter()
            .zip(&self.names)
            .any(|(t, n)| t.info().name() != n)
        {
            return Err(Error::InvalidFormat("expert bank weight names"));
        }
        Ok(())
    }
    pub fn check(
        &self,
        weights: RoutedWeights<'_>,
        input: &[f32],
        selected: &[(usize, f32)],
    ) -> Result<()> {
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
    if cancel.load(Ordering::Relaxed) {
        return Err(Error::InvalidFormat("GPU execution cancelled"));
    }
    if tensor.info().dimensions().len() != 3 {
        return Err(Error::InvalidFormat("expert bank rank"));
    }
    let offset = expert
        .checked_mul(bank.expert_bytes)
        .ok_or(Error::ArithmeticOverflow)?;
    let dst = slot
        .checked_mul(bank.expert_bytes / 4)
        .ok_or(Error::ArithmeticOverflow)?;
    if expert >= tensor.info().dimensions()[2] as usize
        || dst
            .checked_add(bank.expert_bytes / 4)
            .is_none_or(|n| n > bank.words)
    {
        return Err(Error::InvalidBufferLength);
    }
    let mut packed = vec![0u8; bank.expert_bytes];
    let read = Instant::now();
    tensor.read_range(offset as u64, &mut packed)?;
    let read_ns = read.elapsed().as_nanos();
    let upload = Instant::now();
    let src = client.create_from_slice(&packed);
    copy_words::launch::<WgpuRuntime>(
        client,
        CubeCount::Static((bank.expert_bytes as u32 / 4).div_ceil(256), 1, 1),
        CubeDim::new_1d(256),
        // SAFETY: checked word-aligned expert geometry and destination capacity.
        unsafe { ArrayArg::from_raw_parts(src, bank.expert_bytes / 4) },
        unsafe { ArrayArg::from_raw_parts(bank.handle.clone(), bank.words) },
        dst,
    );
    let upload_ns = upload.elapsed().as_nanos();
    // These are immutable PUBLIC original GGUF bytes, as in the dense weight
    // cache. No activation, slot, probability or request content enters packed.
    // create_from_slice owns its public copy before this bounded scratch drops.
    // Volatile wiping public weights twice was measurable CPU overhead; private
    // scratch/readbacks still require erasure and the separate staging audit.
    drop(packed);
    Ok((bank.expert_bytes as u64, read_ns, upload_ns))
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
            || !(MIN_ROUTED_CAPACITY..=MAX_CAPACITY).contains(&capacity)
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
        if bank.handle.stream != banks.gate.handle.stream {
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
        if source.stream != banks.gate.handle.stream {
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
                        ArrayArg::from_raw_parts(banks.gate.handle.clone(), banks.gate.words),
                        ArrayArg::from_raw_parts(banks.up.handle.clone(), banks.up.words),
                        ArrayArg::from_raw_parts(banks.down.handle.clone(), banks.down.words),
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
                    ArrayArg::from_raw_parts(banks.gate.handle.clone(), banks.gate.words),
                    ArrayArg::from_raw_parts(banks.up.handle.clone(), banks.up.words),
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
                    ArrayArg::from_raw_parts(banks.gate.handle.clone(), banks.gate.words),
                    ArrayArg::from_raw_parts(banks.up.handle.clone(), banks.up.words),
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
                    ArrayArg::from_raw_parts(banks.down.handle.clone(), banks.down.words),
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
                    ArrayArg::from_raw_parts(banks.down.handle.clone(), banks.down.words),
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
            let (_, b72) = packed_bank_bytes(row_bytes, rows, MAX_CAPACITY).unwrap();
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
