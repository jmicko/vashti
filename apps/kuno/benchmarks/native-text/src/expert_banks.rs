//! Transient packed expert banks for a fixed public laboratory experiment.
//! Stores original weights only; never persists expert access/rank history.
use crate::bank_slots::BankSlots;
use crate::quant_matvec;
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::wgpu::WgpuRuntime;
use kuno_text::cpu::Matrix;
use kuno_text::model::RoutedWeights;
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
    let expert = (t.info().row_bytes() as usize)
        .checked_mul(t.info().dimensions()[1] as usize)
        .ok_or(Error::ArithmeticOverflow)?;
    let bytes = expert
        .checked_mul(capacity)
        .ok_or(Error::ArithmeticOverflow)?;
    if !expert.is_multiple_of(4) || bytes == 0 || bytes > 128 * 1024 * 1024 {
        return Err(Error::LimitExceeded("expert bank buffer"));
    }
    Ok((expert, bytes))
}
impl ExpertBanks {
    pub fn allocation(weights: RoutedWeights<'_>, capacity: usize) -> Result<usize> {
        if !(10..=64).contains(&capacity) {
            return Err(Error::LimitExceeded("expert bank slots"));
        }
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
