//! Protected stationary second-device experts. No model duplication, saved
//! routes or production API. The creating thread owns every command and cleanup.
use crate::{
    bank_slots::BankSlots,
    expert_banks::{self, BankInstance, ExpertBanks},
    main_admission::{self, MainAdmission},
    private_arena::{BufferId, PrivateArena},
    quant_matvec::grouped_moe as k,
    resident_util as u,
};
use cubecl::prelude::*;
use cubecl::wgpu::{RuntimeOptions, WgpuRuntime, WgpuSetup, init_device};
use kuno_text::cpu_expert::Partials;
use kuno_text::model::{BoundGraph, RoutedWeights};
use kuno_text::{Error, GgufFile, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zeroize::{Zeroize, Zeroizing};

const H: usize = 2560;
const F: usize = 640;
const LAYERS: usize = 48;
const IO: u64 = 16 * 1024 * 1024;
const PRIVATE: usize = 2 * 1024 * 1024;
const DEVICE: u64 = 10 * 1024 * 1024 * 1024;
pub(crate) const LIVE_HOST: usize = 384 * 1024;
struct Readback(cubecl::bytes::Bytes);
impl Drop for Readback {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}
fn full_arg(arena: &PrivateArena, id: BufferId) -> Result<ArrayArg<WgpuRuntime>> {
    let l = arena.layout(id).map_err(u::fault)?;
    // SAFETY: the tracked full exclusive page stays owned through the checked fence.
    Ok(unsafe {
        ArrayArg::from_raw_parts(arena.handle(id).map_err(u::fault)?.clone(), l.owned_words)
    })
}
fn err() -> Error {
    Error::InvalidFormat("stationary secondary expert transaction")
}
fn adm(_: main_admission::AdmissionError) -> Error {
    err()
}

pub(crate) struct Frequency(Zeroizing<Vec<u32>>);
impl Frequency {
    pub(crate) fn new() -> Result<Self> {
        let mut v = Vec::new();
        v.try_reserve_exact(LAYERS * 512).map_err(|_| err())?;
        if v.capacity() != LAYERS * 512 {
            return Err(err());
        }
        v.resize(LAYERS * 512, 0);
        Ok(Self(Zeroizing::new(v)))
    }
    pub(crate) fn record(
        &mut self,
        layer: usize,
        ids: impl IntoIterator<Item = usize>,
    ) -> Result<()> {
        if layer >= LAYERS {
            return Err(err());
        }
        for id in ids {
            if id >= 512 {
                return Err(err());
            }
            let n = &mut self.0[layer * 512 + id];
            *n = n.checked_add(1).ok_or(Error::ArithmeticOverflow)?;
        }
        Ok(())
    }
    fn select(&self, layer: usize, primary: &BankSlots) -> Result<Zeroizing<[usize; 48]>> {
        if layer >= LAYERS || primary.capacity() != 144 {
            return Err(err());
        }
        let mut rank = Zeroizing::new([0usize; 512]);
        for (i, n) in rank.iter_mut().enumerate() {
            *n = i;
        }
        rank.sort_unstable_by(|&a, &b| {
            self.0[layer * 512 + b]
                .cmp(&self.0[layer * 512 + a])
                .then(a.cmp(&b))
        });
        let mut out = Zeroizing::new([0usize; 48]);
        let mut n = 0;
        for &id in rank.iter() {
            if resident_slot(primary, id)?.is_none() {
                out[n] = id;
                n += 1;
                if n == 48 {
                    return Ok(out);
                }
            }
        }
        Err(err())
    }
}
fn resident_slot(bank: &BankSlots, id: usize) -> Result<Option<usize>> {
    if id >= 512 {
        return Err(err());
    }
    for slot in 0..bank.capacity() {
        if bank.resident_expert(slot).map_err(|_| err())? == Some(id) {
            return Ok(Some(slot));
        }
    }
    Ok(None)
}
fn weights<'f>(g: BoundGraph<'_, 'f>, layer: usize) -> Result<RoutedWeights<'f>> {
    Ok(RoutedWeights {
        gate: g.tensor(&format!("blk.{layer}.ffn_gate_exps.weight"))?,
        up: g.tensor(&format!("blk.{layer}.ffn_up_exps.weight"))?,
        down: g.tensor(&format!("blk.{layer}.ffn_down_exps.weight"))?,
    })
}
pub(crate) async fn new_client() -> Result<ComputeClient<WgpuRuntime>> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut a: Vec<_> = instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .await
        .into_iter()
        .filter(|a| {
            a.get_info().vendor == 0x10de
                && a.get_info().device_type == wgpu::DeviceType::DiscreteGpu
        })
        .collect();
    if a.len() != 1 {
        return Err(err());
    }
    let adapter = a.remove(0);
    let info = adapter.get_info();
    let limits = adapter.limits();
    if !adapter.features().contains(wgpu::Features::SUBGROUP)
        || info.subgroup_min_size < 16
        || info.subgroup_max_size > 32
        || limits.max_storage_buffers_per_shader_stage < 8
        || limits.max_storage_buffer_binding_size < 128 * 1024 * 1024
        || limits.max_compute_invocations_per_workgroup < 128
    {
        return Err(err());
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Kuno stationary secondary public fixture"),
            required_features: wgpu::Features::SUBGROUP,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
            ..Default::default()
        })
        .await
        .map_err(|_| err())?;
    let selected = init_device(
        WgpuSetup {
            instance,
            adapter,
            device,
            queue,
            backend: wgpu::Backend::Vulkan,
        },
        RuntimeOptions {
            tasks_max: 256,
            memory_config: crate::private_arena::required_memory_configuration(),
        },
    );
    Ok(WgpuRuntime::client(&selected))
}

/// Readiness comes only from joined CPU jobs and a checked secondary fence.
/// The original non-B70 rank order remains unchanged in the host bridge.
pub(crate) enum HostPartials {
    Cpu(Partials),
    Joined(Completed),
}
pub(crate) struct Completed {
    values: Zeroizing<Vec<f32>>,
    count: Zeroizing<usize>,
}
impl HostPartials {
    pub(crate) fn ranks(&self) -> usize {
        match self {
            Self::Cpu(p) => p.ranks(),
            Self::Joined(p) => *p.count,
        }
    }
    pub(crate) fn hidden(&self) -> usize {
        H
    }
    pub(crate) fn full(&self) -> &[f32] {
        match self {
            Self::Cpu(p) => p.full(),
            Self::Joined(p) => &p.values,
        }
    }
    pub(crate) fn active(&self) -> &[f32] {
        &self.full()[..self.ranks() * H]
    }
    pub(crate) fn owned_bytes(&self) -> usize {
        match self {
            Self::Cpu(p) => p.owned_bytes(),
            Self::Joined(p) => p.values.capacity() * 4,
        }
    }
}
pub(crate) struct Selection {
    pub(crate) cpu: Zeroizing<Vec<usize>>,
    gpu: Zeroizing<Vec<usize>>,
    slots: Zeroizing<[usize; 10]>,
    pub(crate) instance: BankInstance,
}
impl Selection {
    pub(crate) fn gpu_count(&self) -> usize {
        self.gpu.len()
    }
}

pub(crate) struct Secondary {
    client: ComputeClient<WgpuRuntime>,
    retired: Arc<AtomicBool>,
    arena: PrivateArena,
    ids: Zeroizing<[BufferId; 9]>,
    banks: Vec<ExpertBanks>,
    ledger: MainAdmission,
    nonce: u64,
    queued: bool,
    closed: bool,
    pub(crate) calls: u64,
    pub(crate) expert_calls: u64,
    pub(crate) cpu_calls: u64,
    pub(crate) uploaded_bytes: u64,
    pub(crate) bank_setup_ms: u64,
}
impl Secondary {
    pub(crate) fn build<'p>(
        client: ComputeClient<WgpuRuntime>,
        file: &GgufFile,
        graph: BoundGraph<'_, '_>,
        primary: impl Fn(usize) -> Result<&'p ExpertBanks>,
        frequency: &Frequency,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        if graph.geometry().layers != LAYERS || !file.weights_mapped() {
            return Err(err());
        }
        file.verify_file_unchanged()?;
        let start = std::time::Instant::now();
        let requests = [
            k::MAP_WORDS,
            8 * H,
            80 * F,
            80 * H,
            80,
            8 * H,
            4800,
            1,
            10 * H + 1,
        ];
        let private = requests.iter().try_fold(0u64, |n, &v| {
            Ok::<_, Error>(n + main_admission::page_bytes((v * 4) as u64).map_err(adm)?)
        })?;
        let bank_charge = (192u64 * 1024 * 1024) * LAYERS as u64;
        main_admission::check_capacity(
            DEVICE,
            &[
                bank_charge,
                PRIVATE as u64,
                IO,
                40 * 1024 * 1024,
                512 * 1024 * 1024,
            ],
        )
        .map_err(adm)?;
        let baseline = client.memory_usage().map_err(u::fault)?.bytes_reserved;
        let mut ledger = MainAdmission::new(DEVICE, baseline).map_err(adm)?;
        ledger.precharge(&[private]).map_err(adm)?;
        let retired = Arc::new(AtomicBool::new(false));
        let mut arena = PrivateArena::new_transient(client.clone(), PRIVATE, IO, retired.clone())
            .map_err(u::fault)?;
        arena.enable_public_owned_zero_audit().map_err(u::fault)?;
        let mut ids = Vec::new();
        for words in requests {
            ids.push(arena.allocate_zeroed(words).map_err(u::fault)?);
        }
        let ids: [BufferId; 9] = ids.try_into().map_err(|_| err())?;
        ledger
            .observe_after(client.memory_usage().map_err(u::fault)?.bytes_reserved)
            .map_err(adm)?;
        let mut out = Self {
            client,
            retired,
            arena,
            ids: Zeroizing::new(ids),
            banks: Vec::with_capacity(LAYERS),
            ledger,
            nonce: 0,
            queued: false,
            closed: false,
            calls: 0,
            expert_calls: 0,
            cpu_calls: 0,
            uploaded_bytes: 0,
            bank_setup_ms: 0,
        };
        for layer in 0..LAYERS {
            u::cancel(cancel)?;
            let original = primary(layer)?;
            let w = weights(graph, layer)?;
            original.check_names(w)?;
            let selected = frequency.select(layer, &original.slots)?;
            let requests = ExpertBanks::secondary_requests(w)?;
            out.ledger.precharge(&requests).map_err(adm)?;
            let mut bank = ExpertBanks::new_secondary(&out.client, w)?;
            out.ledger.observe_after(out.usage()?).map_err(adm)?;
            let plan = bank.slots.plan(&*selected).map_err(|_| err())?;
            if plan.uploads.len() != 48 {
                return Err(err());
            }
            for window in plan.uploads.chunks(8) {
                let mut charge = Vec::new();
                for _ in window {
                    for p in [bank.gate(), bank.up(), bank.down()] {
                        charge.push(p.expert_bytes() as u64);
                    }
                }
                let rounded = charge.iter().try_fold(0u64, |n, &v| {
                    Ok::<_, Error>(n + main_admission::page_bytes(v).map_err(adm)?)
                })?;
                if rounded > 40 * 1024 * 1024 {
                    return Err(err());
                }
                out.ledger.precharge(&charge).map_err(adm)?;
                for &(expert, slot) in window {
                    for (packed, tensor) in [bank.gate(), bank.up(), bank.down()]
                        .into_iter()
                        .zip([w.gate, w.up, w.down])
                    {
                        let (bytes, _, _) =
                            expert_banks::fill(&out.client, packed, tensor, expert, slot, cancel)?;
                        out.uploaded_bytes = out
                            .uploaded_bytes
                            .checked_add(bytes)
                            .ok_or(Error::ArithmeticOverflow)?;
                    }
                }
                out.ledger.observe_after(out.usage()?).map_err(adm)?;
                crate::executor::sanitize_resident_io(&out.client)?;
                out.ledger.completed_fence(out.usage()?).map_err(adm)?;
            }
            bank.slots.commit(plan).map_err(|_| err())?;
            out.banks.push(bank);
        }
        file.verify_file_unchanged()?;
        out.nonce = out.client.begin_transient_io_reuse().map_err(u::fault)?;
        if out.nonce == 0 {
            return Err(err());
        }
        out.numeric_probe(graph, primary, cancel)?;
        out.calls = 0;
        out.expert_calls = 0;
        out.cpu_calls = 0;
        out.bank_setup_ms = start.elapsed().as_millis() as u64;
        Ok(out)
    }
    fn numeric_probe<'p>(
        &mut self,
        graph: BoundGraph<'_, '_>,
        primary: impl Fn(usize) -> Result<&'p ExpertBanks>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        let mut seen = 0u8;
        let mut x = Zeroizing::new(vec![0f32; H]);
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i * 37 % 97) as f32 - 48.) / 97.;
        }
        let mut decoded = Zeroizing::new(vec![0f32; H]);
        let mut activation = Zeroizing::new(vec![0f32; F]);
        for layer in 0..LAYERS {
            let w = weights(graph, layer)?;
            let bit = if w.down.info().dtype().gguf_id() == 6 {
                1
            } else {
                2
            };
            if seen & bit != 0 {
                continue;
            }
            let mut ids = Zeroizing::new([0usize; 4]);
            for (n, slot) in [0, 31, 32, 47].into_iter().enumerate() {
                ids[n] = self.banks[layer]
                    .slots
                    .resident_expert(slot)
                    .map_err(|_| err())?
                    .ok_or_else(err)?;
            }
            let selection = self.select(layer, &*ids, primary(layer)?)?;
            self.enqueue(layer, &selection, &x, cancel)?;
            let actual = self.finish(&selection, &*ids, None, cancel)?;
            for (rank, &expert) in ids.iter().enumerate() {
                for r in 0..F {
                    w.gate.decode_row((expert * F + r) as u64, &mut decoded)?;
                    let gate = decoded
                        .iter()
                        .zip(x.iter())
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>() as f32;
                    w.up.decode_row((expert * F + r) as u64, &mut decoded)?;
                    let up = decoded
                        .iter()
                        .zip(x.iter())
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>() as f32;
                    let sig = if gate >= 0. {
                        1. / (1. + (-gate).exp())
                    } else {
                        let e = gate.exp();
                        e / (1. + e)
                    };
                    activation[r] = gate * sig * up;
                }
                for r in 0..H {
                    w.down
                        .decode_row((expert * H + r) as u64, &mut decoded[..F])?;
                    let reference = decoded[..F]
                        .iter()
                        .zip(activation.iter())
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>() as f32;
                    let value = actual.full()[rank * H + r];
                    if !value.is_finite()
                        || (value - reference).abs() > 5e-4 + 5e-5 * reference.abs()
                    {
                        return Err(Error::InvalidFormat(
                            "secondary independent original-row numerical oracle",
                        ));
                    }
                }
            }
            seen |= bit;
        }
        if seen != 3 {
            return Err(err());
        }
        println!(
            "public_secondary_original_rows families=2 boundary_slots=4 numerical_and_full_owners=true"
        );
        Ok(())
    }
    fn usage(&self) -> Result<u64> {
        Ok(self.client.memory_usage().map_err(u::fault)?.bytes_reserved)
    }
    pub(crate) fn select(
        &self,
        layer: usize,
        ids: &[usize],
        primary: &ExpertBanks,
    ) -> Result<Selection> {
        if self.closed || self.retired.load(Ordering::Acquire) || self.queued || ids.len() > 10 {
            return Err(err());
        }
        let bank = self.banks.get(layer).ok_or_else(err)?;
        let mut s = Selection {
            cpu: Zeroizing::new(Vec::with_capacity(10)),
            gpu: Zeroizing::new(Vec::with_capacity(10)),
            slots: Zeroizing::new([0; 10]),
            instance: bank.instance(),
        };
        for (i, &id) in ids.iter().enumerate() {
            if ids[..i].contains(&id) || resident_slot(&primary.slots, id)?.is_some() {
                return Err(err());
            }
            if let Some(slot) = resident_slot(&bank.slots, id)? {
                s.slots[s.gpu.len()] = slot;
                s.gpu.push(id);
            } else {
                s.cpu.push(id);
            }
        }
        Ok(s)
    }
    pub(crate) fn check_membership(
        &self,
        layer: usize,
        primary: &ExpertBanks,
        ids: &[usize; 10],
        mask: u16,
        instance: Option<BankInstance>,
    ) -> Result<()> {
        if self.closed || self.retired.load(Ordering::Acquire) {
            return Err(err());
        }
        let bank = self.banks.get(layer).ok_or_else(err)?;
        if mask != 0 && instance != Some(bank.instance()) {
            return Err(err());
        }
        for (r, &id) in ids.iter().enumerate() {
            if mask & (1 << r) != 0
                && (resident_slot(&bank.slots, id)?.is_none()
                    || resident_slot(&primary.slots, id)?.is_some())
            {
                return Err(err());
            }
        }
        Ok(())
    }
    pub(crate) fn enqueue(
        &mut self,
        layer: usize,
        s: &Selection,
        mixed: &[f32],
        cancel: &AtomicBool,
    ) -> Result<()> {
        if self.queued
            || self.closed
            || s.gpu.is_empty()
            || mixed.len() != H
            || mixed.iter().any(|v| !v.is_finite())
        {
            return Err(err());
        }
        u::cancel(cancel)?;
        let bank = self.banks.get(layer).ok_or_else(err)?;
        if bank.instance() != s.instance {
            return Err(err());
        }
        bank.check_client(&self.client)?;
        let wwords = self
            .ids
            .each_ref()
            .map(|&id| u::words(&self.arena, id))
            .into_iter()
            .collect::<Result<Vec<_>>>()?;
        for &id in self.ids.iter() {
            let l = self.arena.layout(id).map_err(u::fault)?;
            let h = self.arena.handle(id).map_err(u::fault)?;
            if h.size() != l.owned_bytes as u64
                || h.size_in_used() != h.size()
                || h.offset_start.unwrap_or(0) != 0
                || h.offset_end.unwrap_or(0) != 0
            {
                return Err(err());
            }
        }
        u::zero_batch(&self.arena, &self.ids[..])?;
        let mut map = Zeroizing::new(vec![0u32; wwords[0]]);
        map[..4].copy_from_slice(&[1, 0, 1, s.gpu.len() as u32]);
        for r in 0..s.gpu.len() {
            if resident_slot(&bank.slots, s.gpu[r])? != Some(s.slots[r]) {
                return Err(err());
            }
            map[4 + r] = s.slots[r] as u32;
            map[84 + r] = 1;
            map[164 + r * 8] = r as u32;
        }
        self.write(0, &map)?;
        let input = Zeroizing::new(mixed.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        self.write(1, &input)?;

        let down = if bank.down().expert_bytes() == 640 * 2560 * 34 / 32 {
            8
        } else {
            6
        };
        let g = k::Geometry::stationary_secondary([
            k::Projection::new(12, H, F, 512).map_err(Error::InvalidFormat)?,
            k::Projection::new(12, H, F, 512).map_err(Error::InvalidFormat)?,
            k::Projection::new(down, F, H, 512).map_err(Error::InvalidFormat)?,
        ])
        .map_err(Error::InvalidFormat)?;
        let spans = k::Spans::new([0, 0, H, 0, F, 0, H, 0, 0, H, 0]);
        let owner = |i: usize| self.arena.handle(self.ids[i]).cloned().map_err(u::fault);
        let owners = [
            owner(0)?,
            owner(1)?,
            owner(2)?,
            owner(3)?,
            owner(4)?,
            owner(5)?,
            owner(6)?,
        ];
        let banks = std::array::from_fn(|shard| {
            [bank.gate(), bank.up(), bank.down()].map(|p| p.segments()[shard].handle().clone())
        });
        crate::next_shared_moe::launch_pair(
            &self.client,
            g,
            &spans,
            banks,
            owners,
            [32, 16],
            false,
            false,
        )?;
        unsafe {
            fold::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static(
                    self.ids
                        .iter()
                        .map(|&id| {
                            self.arena
                                .layout(id)
                                .expect("validated full secondary owner")
                                .owned_words
                        })
                        .max()
                        .ok_or_else(err)?
                        .div_ceil(64) as u32,
                    1,
                    1,
                ),
                CubeDim::new_1d(64),
                full_arg(&self.arena, self.ids[0])?,
                full_arg(&self.arena, self.ids[1])?,
                full_arg(&self.arena, self.ids[2])?,
                full_arg(&self.arena, self.ids[3])?,
                full_arg(&self.arena, self.ids[4])?,
                full_arg(&self.arena, self.ids[5])?,
                full_arg(&self.arena, self.ids[6])?,
                full_arg(&self.arena, self.ids[7])?,
                s.gpu.len(),
            );
        }
        self.client.flush().map_err(u::fault)?;
        self.queued = true;
        Ok(())
    }
    fn write(&self, i: usize, words: &[u32]) -> Result<()> {
        let id = self.ids[i];
        let l = self.arena.layout(id).map_err(u::fault)?;
        if words.len() > l.logical_words {
            return Err(err());
        }
        let mut bytes = Zeroizing::new(vec![0u8; l.owned_bytes]);
        for (i, &v) in words.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        self.client
            .write_private(
                self.arena.handle(id).map_err(u::fault)?.clone(),
                std::mem::take(&mut *bytes),
            )
            .map_err(u::fault)
    }
    pub(crate) fn finish(
        &mut self,
        s: &Selection,
        original: &[usize],
        cpu: Option<Partials>,
        cancel: &AtomicBool,
    ) -> Result<HostPartials> {
        if s.gpu.is_empty() {
            return cpu.map(HostPartials::Cpu).ok_or_else(err);
        }
        if !self.queued {
            return Err(err());
        }
        // The packet combines the original-rank partial prefix and sticky fault.
        unsafe {
            packet::launch_unchecked::<WgpuRuntime>(
                &self.client,
                CubeCount::Static((10 * H + 1).div_ceil(64) as u32, 1, 1),
                CubeDim::new_1d(64),
                u::arg(&self.arena, self.ids[3])?,
                u::arg(&self.arena, self.ids[7])?,
                u::arg(&self.arena, self.ids[8])?,
            );
        }
        let bytes = self
            .client
            .read_one(self.arena.handle(self.ids[8]).map_err(u::fault)?.clone())
            .map_err(u::fault)?;
        let bytes = Readback(bytes);
        let values = u32::from_bytes(&bytes.0);
        if values.len()
            != self
                .arena
                .layout(self.ids[8])
                .map_err(u::fault)?
                .owned_words
        {
            return Err(Error::InvalidFormat("secondary return packet extent"));
        }
        if values[10 * H] != 0 {
            println!("public_secondary_fault flags={}", values[10 * H]);
            return Err(Error::InvalidFormat("secondary full-owner device fold"));
        }
        if values[10 * H + 1..].iter().any(|&v| v != 0) {
            return Err(Error::InvalidFormat("secondary return packet padding"));
        }
        let mut out = Completed {
            values: Zeroizing::new(vec![0f32; 10 * H]),
            count: Zeroizing::new(original.len()),
        };
        join_rows(
            original,
            &s.cpu,
            &s.gpu,
            cpu.as_ref(),
            &values[..10 * H],
            &mut out.values,
        )?;
        drop(bytes);
        drop(cpu);
        u::cancel(cancel)?;
        self.client
            .recycle_transient_io(self.nonce)
            .map_err(u::fault)?;
        self.ledger.completed_fence(self.usage()?).map_err(adm)?;
        self.queued = false;
        self.calls += 1;
        self.expert_calls += s.gpu.len() as u64;
        self.cpu_calls += s.cpu.len() as u64;
        Ok(HostPartials::Joined(out))
    }
    pub(crate) fn close(&mut self) -> Result<()> {
        if self.closed {
            return Ok(());
        }
        if self.nonce != 0 {
            self.client
                .end_transient_io_reuse(self.nonce)
                .map_err(u::fault)?;
            self.nonce = 0;
        }
        let expected = self.ids.iter().try_fold(0u64, |n, &id| {
            Ok::<_, Error>(n + self.arena.layout(id).map_err(u::fault)?.owned_bytes as u64)
        })?;
        self.arena.close().map_err(u::fault)?;
        let audit = self.arena.owned_zero_audit_receipt().ok_or_else(err)?;
        if audit.buffers != 9 || audit.bytes != expected {
            return Err(err());
        }
        self.banks.clear();
        self.client.memory_cleanup();
        crate::executor::sanitize_resident_io(&self.client)?;
        self.closed = true;
        println!(
            "public_secondary_closed=true owned_buffers={} owned_bytes={}",
            audit.buffers, audit.bytes
        );
        Ok(())
    }
}
impl Drop for Secondary {
    fn drop(&mut self) {
        if !self.closed && self.close().is_err() {
            self.retired.store(true, Ordering::Release);
        }
    }
}

fn join_rows(
    original: &[usize],
    cpu_ids: &[usize],
    gpu_ids: &[usize],
    cpu: Option<&Partials>,
    gpu: &[u32],
    out: &mut [f32],
) -> Result<()> {
    if original.len() > 10
        || cpu_ids.len() + gpu_ids.len() != original.len()
        || gpu.len() != 10 * H
        || out.len() != 10 * H
        || original
            .iter()
            .enumerate()
            .any(|(r, id)| *id >= 512 || original[..r].contains(id))
        || cpu_ids.iter().any(|id| gpu_ids.contains(id))
    {
        return Err(err());
    }
    if cpu_ids.is_empty() {
        if cpu.is_some() {
            return Err(err());
        }
    } else if cpu.is_none_or(|p| {
        p.ranks() != cpu_ids.len()
            || p.hidden() != H
            || p.full()[p.active().len()..]
                .iter()
                .any(|v| v.to_bits() != 0)
    }) {
        return Err(err());
    }
    for (r, &id) in original.iter().enumerate() {
        if let Some(j) = cpu_ids.iter().position(|&e| e == id) {
            out[r * H..(r + 1) * H]
                .copy_from_slice(&cpu.ok_or_else(err)?.active()[j * H..(j + 1) * H]);
        } else if let Some(j) = gpu_ids.iter().position(|&e| e == id) {
            for f in 0..H {
                out[r * H + f] = f32::from_bits(gpu[j * H + f]);
            }
        } else {
            return Err(err());
        }
    }
    if out.iter().any(|v| !v.is_finite())
        || out[original.len() * H..].iter().any(|v| v.to_bits() != 0)
        || gpu[gpu_ids.len() * H..].iter().any(|&v| v != 0)
    {
        return Err(err());
    }
    Ok(())
}

#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::collapsible_if, clippy::eq_op)]
fn fold(
    map: &Array<u32>,
    input: &Array<u32>,
    act: &Array<u32>,
    part: &Array<u32>,
    prob: &Array<u32>,
    output: &Array<u32>,
    status: &Array<u32>,
    fault: &mut Array<Atomic<u32>>,
    count: usize,
) {
    let i = ABSOLUTE_POS;
    let mut bad = 0u32;
    if count == 0 || count > 10 {
        bad |= 256;
    }
    if i == 0 {
        if map[0] != 1 || map[1] != 0 || map[2] != 1 || map[3] != count as u32 {
            bad |= 256;
        }
    }
    if i < input.len() {
        if i < 2560 {
            let v = f32::reinterpret(input[i]);
            if v != v || f32::abs(v) > f32::new(f32::MAX) {
                bad |= 1;
            }
        } else if input[i] != 0 {
            bad |= 1;
        }
    }
    if i < act.len() {
        if i < count * 640 {
            let v = f32::reinterpret(act[i]);
            if v != v || f32::abs(v) > f32::new(f32::MAX) {
                bad |= 2;
            }
        } else if act[i] != 0 {
            bad |= 2;
        }
    }
    if i < part.len() {
        if i < count * 2560 {
            let v = f32::reinterpret(part[i]);
            if v != v || f32::abs(v) > f32::new(f32::MAX) {
                bad |= 4;
            }
        } else if part[i] != 0 {
            bad |= 4;
        }
    }
    if i < status.len() {
        let expected = if i < count || (i >= 80 && i < 80 + count) {
            1u32
        } else {
            0u32
        };
        if status[i] != expected {
            bad |= 8;
        }
    }
    if i < prob.len() {
        if prob[i] != 0 {
            bad |= 16;
        }
    }
    if i < output.len() {
        if output[i] != 0 {
            bad |= 32;
        }
    }
    if i < map.len() && i >= 804 {
        if map[i] != 0 {
            bad |= 64;
        }
    }
    if i < fault.len() && i > 0 {
        if fault[i].load() != 0 {
            bad |= 128;
        }
    }
    if bad != 0 {
        fault[0].fetch_or(bad);
    }
}
#[cube(launch, launch_unchecked)]
fn packet(part: &Array<u32>, fault: &Array<u32>, out: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    if i < 25600 {
        out[i] = part[i];
    }
    if i == 25600 {
        out[i] = fault[0];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn secondary_partition_rejects_foreign_boundaries_and_preserves_flat48() {
        for down in [6, 8] {
            let p = [
                k::Projection::new(12, H, F, 512).unwrap(),
                k::Projection::new(12, H, F, 512).unwrap(),
                k::Projection::new(down, F, H, 512).unwrap(),
            ];
            let flat = k::Geometry::new(p, 1, 48).unwrap();
            let secondary = k::Geometry::stationary_secondary(p).unwrap();
            for (first, count) in [(0, 32), (32, 16)] {
                let words = p.map(|v| v.segment_words(count).unwrap());
                secondary.validate_shard(first, count, words).unwrap();
                assert!(flat.validate_shard(first, count, words).is_err());
            }
            assert!(
                secondary
                    .validate_shard(0, 48, p.map(|v| v.segment_words(48).unwrap()))
                    .is_err()
            );
            assert!(
                secondary
                    .validate_shard(31, 16, p.map(|v| v.segment_words(16).unwrap()))
                    .is_err()
            );
        }
    }
    #[test]
    fn complementary_selection_preserves_primary_and_prefill_frequency() {
        let mut f = Frequency::new().unwrap();
        f.record(0, [3, 3, 2, 2, 2, 1]).unwrap();
        let mut bank = BankSlots::new(144, 512).unwrap();
        let plan = bank.plan(&[2, 7]).unwrap();
        bank.commit(plan).unwrap();
        let ids = f.select(0, &bank).unwrap();
        assert_eq!(&ids[..3], &[3, 1, 0]);
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&7));
        assert!(f.record(48, [0]).is_err());
        assert!(f.record(0, [512]).is_err());
    }
    #[test]
    fn joined_gpu_rows_keep_noncontiguous_original_order_and_reject_duplicate_missing_nonfinite() {
        let mut gpu = vec![0u32; 10 * H];
        gpu[..H].fill(2f32.to_bits());
        gpu[H..2 * H].fill(3f32.to_bits());
        let mut out = vec![0f32; 10 * H];
        join_rows(&[9, 5], &[], &[5, 9], None, &gpu, &mut out).unwrap();
        assert_eq!(out[0], 3.);
        assert_eq!(out[H], 2.);
        assert!(join_rows(&[9, 9], &[], &[5, 9], None, &gpu, &mut out).is_err());
        assert!(join_rows(&[9, 6], &[], &[5, 9], None, &gpu, &mut out).is_err());
        gpu[0] = f32::NAN.to_bits();
        assert!(join_rows(&[9, 5], &[], &[5, 9], None, &gpu, &mut out).is_err());
    }
}
