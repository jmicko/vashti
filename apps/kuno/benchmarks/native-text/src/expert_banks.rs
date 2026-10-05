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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use zeroize::Zeroize;

pub struct Bank {
    pub handle: Handle,
    pub expert_bytes: usize,
    pub words: usize,
    pub reserved: usize,
}
pub struct ExpertBanks {
    pub gate: Bank,
    pub up: Bank,
    pub down: Bank,
    pub hidden: usize,
    pub ff: usize,
    pub experts: usize,
    pub slots: BankSlots,
    names: [String; 3],
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
        Ok(Self {
            gate: create(weights.gate)?,
            up: create(weights.up)?,
            down: create(weights.down)?,
            hidden: gate.inputs(),
            ff: gate.outputs(),
            experts: gate.experts(),
            slots: BankSlots::new(capacity, gate.experts())
                .map_err(|_| Error::InvalidFormat("expert slots"))?,
            names: [weights.gate, weights.up, weights.down].map(|t| t.info().name().into()),
        })
    }
    pub fn reserved(&self) -> usize {
        self.gate.reserved + self.up.reserved + self.down.reserved
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
    // Public original weight bytes; erasing bounds host scratch lifetime too.
    packed.zeroize();
    Ok((bank.expert_bytes as u64, read_ns, upload_ns))
}

#[cube(launch)]
fn erase(data: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i < data.len() {
        data[i] = 0.;
    }
}
pub fn clear(client: &ComputeClient<WgpuRuntime>, handles: &[(Handle, usize)]) {
    for (handle, len) in handles {
        erase::launch::<WgpuRuntime>(
            client,
            CubeCount::Static((*len as u32).div_ceil(256), 1, 1),
            CubeDim::new_1d(256),
            // SAFETY: caller constructs lengths from the actual allocation.
            unsafe { ArrayArg::from_raw_parts(handle.clone(), *len) },
        );
    }
}

pub fn execute(
    client: &ComputeClient<WgpuRuntime>,
    banks: &ExpertBanks,
    weights: RoutedWeights<'_>,
    input: &[f32],
    slots: &[usize],
    probabilities: &[f32],
    cancel: &AtomicBool,
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
