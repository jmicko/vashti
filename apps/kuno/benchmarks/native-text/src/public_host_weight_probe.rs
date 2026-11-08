//! Device-parented compiled-public host-placement and ownership acceptance.
//! No new device, model file, request bytes, CubeCL owner, or timing claim.
//! Original Q2_0 four-code blocks are decoded and reduced directly from the
//! immutable non-device-local allocation. The first hardware gate is deliberately
//! small; a whole-model mirror and GPU source-table connection are not implemented.
use crate::public_host_weight_pool as pool;
use std::{borrow::Cow, ffi::OsString, path::PathBuf, time::Duration};
use zeroize::{Zeroize, Zeroizing};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WORDS: usize = 1024;
const PAGE: u64 = (WORDS * 4) as u64;
const INPUT_FIRST: usize = 3;
const OUTPUT_FIRST: usize = 8;
const DOT_FIRST: usize = OUTPUT_FIRST + pool::ROWS * pool::COLUMNS;
const READBACK_BYTES: u64 = PAGE * 3;
const NORMAL_CASES: usize = 2;
const FAULT_CASES: usize = 5;
const FIXTURE_HOST_CAP: u64 = 8 * 1024 * 1024;

pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    allowed_build: bool,
) -> std::result::Result<PathBuf, &'static str> {
    if !allowed_build
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "public-host-weight-probe"
    {
        return Err("public host weight fixed card/build required");
    }
    let (mut gate, mut receipt) = (false, None);
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|p| key.as_encoded_bytes().starts_with(p))
        {
            continue;
        }
        if key == "KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY" && !gate && value == "1" {
            gate = true;
        } else if key == "KUNO_PUBLIC_RECEIPT" && receipt.is_none() && !value.is_empty() {
            receipt = Some(PathBuf::from(value));
        } else {
            return Err("public host weight inherited/duplicate option rejected");
        }
    }
    if !gate {
        return Err("public host weight explicit public gate required");
    }
    receipt.ok_or("public host weight new numeric receipt required")
}
pub(crate) fn validate_limits(limits: &wgpu::Limits) -> std::result::Result<(), &'static str> {
    if limits.max_compute_invocations_per_workgroup < 64
        || limits.max_compute_workgroup_size_x < 64
        || limits.max_compute_workgroup_storage_size < 264
        || limits.max_storage_buffers_per_shader_stage < 4
        || limits.max_bindings_per_bind_group < 4
        || limits.max_bind_groups < 1
        || limits.max_storage_buffer_binding_size < pool::BUFFER_BYTES
        || limits.max_buffer_size < pool::BUFFER_BYTES
        || limits.max_compute_workgroups_per_dimension < 1
    {
        return Err("public host weight actual device limits rejected");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    SourceTail,
    InputNan,
    InputTail,
    OutputDirty,
    Sticky,
}
impl Fault {
    fn bits(self) -> u32 {
        match self {
            Self::None => 0,
            Self::SourceTail => 1,
            Self::InputNan => 2,
            Self::InputTail => 4,
            Self::OutputDirty => 8,
            Self::Sticky => 256 | 32,
        }
    }
}
fn input() -> Zeroizing<Vec<u32>> {
    let mut out = Zeroizing::new(vec![0; WORDS]);
    for column in 0..pool::COLUMNS {
        out[INPUT_FIRST + column] = (((column * 13 % 17) as f32 - 8.) / 8.).to_bits();
    }
    out
}
fn expected(fixture: pool::Fixture, input: &[u32]) -> Result<Zeroizing<Vec<u32>>> {
    if input.len() != WORDS {
        return Err("public host weight CPU input extent".into());
    }
    let mut weights = Zeroizing::new(vec![0.; pool::ROWS * pool::COLUMNS]);
    kuno_text::TensorType::from_gguf_id(42)?.decode(&fixture.packed(), &mut weights)?;
    let mut output = Zeroizing::new(vec![0; WORDS]);
    for row in 0..pool::ROWS {
        let mut lanes = [0.; 64];
        for column in 0..64 {
            let value = weights[row * 64 + column];
            output[OUTPUT_FIRST + row * 64 + column] = value.to_bits();
            lanes[column] = value * f32::from_bits(input[INPUT_FIRST + column]);
        }
        // Same original low+high then32-lane binary reduction, no FMA.
        for lane in 0..32 {
            lanes[lane] += lanes[lane + 32];
        }
        for step in [16, 8, 4, 2, 1] {
            for lane in 0..step {
                lanes[lane] += lanes[lane + step];
            }
        }
        output[DOT_FIRST + row] = lanes[0].to_bits();
        lanes.zeroize();
    }
    Ok(output)
}
fn bytes(words: &[u32]) -> Zeroizing<Vec<u8>> {
    Zeroizing::new(words.iter().flat_map(|word| word.to_le_bytes()).collect())
}
fn initialized(device: &wgpu::Device, words: &[u32], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compiled-public-host-probe-owner"),
        size: (words.len() * 4) as u64,
        usage,
        mapped_at_creation: true,
    });
    buffer
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(&bytes(words));
    buffer.unmap();
    buffer
}
fn wait(device: &wgpu::Device) -> Result<()> {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .map_err(|_| "public host weight probe GPU join failed")?;
    Ok(())
}
fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Zeroizing<Vec<u32>>> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result.is_ok());
        });
    wait(device)?;
    if !rx
        .recv()
        .map_err(|_| "public host weight map completion missing")?
    {
        return Err("public host weight probe readback map failed".into());
    }
    let lease = buffer.slice(..).get_mapped_range();
    let words = Zeroizing::new(
        lease
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    );
    drop(lease);
    buffer.unmap();
    Ok(words)
}
#[derive(Clone, Copy)]
struct Counters {
    buffers: isize,
    memory: isize,
}
fn counters(device: &wgpu::Device) -> Counters {
    let c = device.get_internal_counters().hal;
    Counters {
        buffers: c.buffers.read(),
        memory: c.buffer_memory.read(),
    }
}
fn shader(prefix: usize) -> Result<String> {
    if ![0, 3].contains(&prefix) {
        return Err("public host weight shader fixed prefix".into());
    }
    Ok(SOURCE.replace("@PREFIX@", &prefix.to_string()))
}
fn validate_readback(
    actual: &[u32],
    fixture: pool::Fixture,
    fault: Fault,
    x: &[u32],
    initial: &[u32],
    initial_fault: &[u32],
) -> Result<()> {
    if actual.len() != WORDS * 3
        || x.len() != WORDS
        || initial.len() != WORDS
        || initial_fault.len() != WORDS
        || actual[2 * WORDS..] != *x
    {
        return Err("public host weight complete readback/immutable input extent".into());
    }
    if fault == Fault::None {
        let oracle = expected(fixture, x)?;
        if actual[..WORDS] != oracle[..] || actual[WORDS..2 * WORDS].iter().any(|&word| word != 0) {
            return Err("public host weight original packed numeric/full owner mismatch".into());
        }
    } else if actual[..WORDS] != *initial
        || actual[WORDS] != fault.bits()
        || actual[WORDS + 1..2 * WORDS] != initial_fault[1..]
        || actual[WORDS] & initial_fault[0] != initial_fault[0]
    {
        return Err("public host weight fault publication/full owner mismatch".into());
    }
    Ok(())
}

struct Case {
    placement: pool::Placement,
    allocation: u64,
    counter_live: isize,
    zeroed: u64,
}
fn case(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fixture: pool::Fixture,
    fault_kind: Fault,
) -> Result<Case> {
    let before = counters(device);
    let source = pool::PublicHostWeightPool::new(device, fixture)?;
    let live = counters(device);
    if live.memory != before.memory || ![0, 1].contains(&(live.buffers - before.buffers)) {
        return Err("public host weight imported HAL counter accounting".into());
    }
    let placement = source.placement();
    let allocation = source.allocated_bytes();
    let mut x = input();
    let mut initial = Zeroizing::new(vec![0_u32; WORDS]);
    let mut initial_fault = Zeroizing::new(vec![0_u32; WORDS]);
    match fault_kind {
        Fault::InputNan => x[INPUT_FIRST + 7] = f32::NAN.to_bits(),
        Fault::InputTail => x[WORDS - 1] = 1,
        Fault::OutputDirty => initial[WORDS - 1] = 0xd00d,
        Fault::Sticky => initial_fault[0] = 256,
        Fault::None | Fault::SourceTail => (),
    }
    let gpu_input = initialized(
        device,
        &x,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let gpu_output = initialized(
        device,
        &initial,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let gpu_fault = initialized(
        device,
        &initial_fault,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compiled-public-host-probe-readback"),
        size: READBACK_BYTES,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compiled-public-host-q2-original"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader(fixture.prefix())?)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("compiled-public-host-q2-original"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.source().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: gpu_input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: gpu_output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gpu_fault.as_entire_binding(),
            },
        ],
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&gpu_output, 0, &readback, 0, PAGE);
    encoder.copy_buffer_to_buffer(&gpu_fault, 0, &readback, PAGE, PAGE);
    encoder.copy_buffer_to_buffer(&gpu_input, 0, &readback, 2 * PAGE, PAGE);
    queue.submit([encoder.finish()]);
    wait(device)?;
    if cubecl_common::future::block_on(error_scope.pop()).is_some() {
        return Err("public host weight probe validation failed".into());
    }
    let actual = read(device, &readback)?;
    validate_readback(&actual, fixture, fault_kind, &x, &initial, &initial_fault)?;
    // Explicitly zero/inspect every complete ordinary input+output+fault span.
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.clear_buffer(&gpu_input, 0, None);
    encoder.clear_buffer(&gpu_output, 0, None);
    encoder.clear_buffer(&gpu_fault, 0, None);
    encoder.copy_buffer_to_buffer(&gpu_output, 0, &readback, 0, PAGE);
    encoder.copy_buffer_to_buffer(&gpu_fault, 0, &readback, PAGE, PAGE);
    encoder.copy_buffer_to_buffer(&gpu_input, 0, &readback, 2 * PAGE, PAGE);
    queue.submit([encoder.finish()]);
    if read(device, &readback)?.iter().any(|&word| word != 0) {
        return Err("public host weight complete output zero audit".into());
    }
    drop(bind);
    drop(layout);
    drop(pipeline);
    drop(module);
    gpu_input.destroy();
    gpu_output.destroy();
    gpu_fault.destroy();
    readback.destroy();
    drop((gpu_input, gpu_output, gpu_fault, readback));
    let close = source.close()?;
    wait(device)?;
    let after = counters(device);
    if !close.source_unchanged
        || !close.full_allocation_zero
        || !close.ledger_closed
        || close.allocation_bytes != allocation
        || after.buffers != before.buffers
        || after.memory != before.memory
    {
        return Err("public host weight full allocation/ledger/HAL close mismatch".into());
    }
    Ok(Case {
        placement,
        allocation,
        counter_live: live.buffers - before.buffers,
        zeroed: close.allocation_bytes,
    })
}

pub(crate) fn run(
    adapter: &wgpu::Adapter,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    limits: &wgpu::Limits,
) -> Result<serde_json::Value> {
    validate_limits(limits)?;
    validate_limits(&device.limits())?;
    // SAFETY: read borrowed identities only. Actual Vulkan objects must belong
    // to one setup; no raw API object is destroyed or concurrently submitted.
    let a = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host weight Vulkan adapter required")?;
    let d = unsafe { device.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host weight Vulkan device required")?;
    let q = unsafe { queue.as_hal::<wgpu::hal::api::Vulkan>() }
        .ok_or("public host weight Vulkan queue required")?;
    if a.raw_physical_device() != d.raw_physical_device()
        || a.shared_instance().raw_instance().handle()
            != d.shared_instance().raw_instance().handle()
        || q.raw_device().handle() != d.raw_device().handle()
        || q.as_raw() != d.raw_queue()
    {
        return Err("public host weight mixed setup identity rejected".into());
    }
    drop((a, d, q));
    if adapter.get_info().device_type != wgpu::DeviceType::DiscreteGpu {
        return Err("public host weight discrete GPU required".into());
    }
    // Compiled public sources only; the ordinary descriptor bytes do not include
    // opaque driver caches/metadata. No private payload erasure claim is made.
    crate::main_admission::check_capacity(FIXTURE_HOST_CAP, &[pool::HOST_CAP, 16 * PAGE, 65536])?;
    wait(device)?;
    let mut cases = Vec::with_capacity(NORMAL_CASES + FAULT_CASES);
    cases.push(case(device, queue, pool::Fixture::Aligned, Fault::None)?);
    cases.push(case(device, queue, pool::Fixture::Prefix3, Fault::None)?);
    cases.push(case(
        device,
        queue,
        pool::Fixture::BadPublicTail,
        Fault::SourceTail,
    )?);
    for fault in [
        Fault::InputNan,
        Fault::InputTail,
        Fault::OutputDirty,
        Fault::Sticky,
    ] {
        cases.push(case(device, queue, pool::Fixture::Prefix3, fault)?);
    }
    Ok(serde_json::json!({
        "scope":"compiled_public_nonlocal_host_q2_operator_only",
        "normal_cases":NORMAL_CASES,"fault_cases":FAULT_CASES,
        "original_dtype":42,"rows":pool::ROWS,"columns":pool::COLUMNS,
        "exact_decoded_values":NORMAL_CASES*pool::ROWS*pool::COLUMNS,
        "exact_original64_dots":NORMAL_CASES*pool::ROWS,
        "memory_type_indices":cases.iter().map(|c|c.placement.memory_type).collect::<Vec<_>>(),
        "heap_indices":cases.iter().map(|c|c.placement.heap).collect::<Vec<_>>(),
        "memory_property_flags":cases.iter().map(|c|c.placement.property_flags).collect::<Vec<_>>(),
        "memory_heap_flags":cases.iter().map(|c|c.placement.heap_flags).collect::<Vec<_>>(),
        "physical_allocation_bytes":cases.iter().map(|c|c.allocation).collect::<Vec<_>>(),
        "hal_buffer_count_deltas_live":cases.iter().map(|c|c.counter_live).collect::<Vec<_>>(),
        "host_weight_ledger_cap_bytes":pool::HOST_CAP,
        "host_weight_buffer_bytes":pool::BUFFER_BYTES,
        "host_weight_bytes_freed":cases.iter().map(|c|c.zeroed).sum::<u64>(),
        "all_host_weight_ledgers_closed":true,"all_host_allocations_cpu_zero_verified":true,
        "all_full_source_gpu_padding_checked":true,"all_sources_immutable_after_gpu":true,
        "hal_buffer_memory_excludes_external_host_pool":true,
        "hal_counters_available":cases.iter().all(|c|c.counter_live==1),
        "hal_counters_restored":true,"explicit_weight_upload_bytes":0,
        "whole_model_or_expert_mirror":false,"cubecl_registered":false,
        "portable_permanent_pinning_proved":false,"private_requests_accepted":false,
        "hidden_ordinary_buffer_padding_erasure_claim":false,
        "opaque_driver_host_metadata_bounded":false,"performance_accepted":false,
    }))
}

const SOURCE: &str = r#"
const PREFIX:u32=@PREFIX@u;
@group(0) @binding(0) var<storage,read> weights:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;
@group(0) @binding(3) var<storage,read_write> fault:array<atomic<u32>>;
var<workgroup> bad:atomic<u32>;
var<workgroup> full_fault:u32;
var<workgroup> sums:array<f32,64>;
fn finite(word:u32)->bool {return (word&0x7f800000u)!=0x7f800000u;}
fn expected_byte(index:u32)->u32 {
 if index<PREFIX {return 0u;}
 let at=index-PREFIX;
 if at>=72u {return 0u;}
 let row=at/18u;let column=at%18u;
 if column==0u {return 0u;}
 if column==1u {
  var high=0x38u;
  if row==1u {high=0x34u;}
  if row==2u {high=0x3cu;}
  if row==3u {high=0x30u;}
  return high;
 }
 return ((column-2u)*73u+row*41u+9u)&255u;
}
fn expected_word(word:u32)->u32 {
 let at=word*4u;
 return expected_byte(at)|(expected_byte(at+1u)<<8u)|(expected_byte(at+2u)<<16u)|(expected_byte(at+3u)<<24u);
}
fn byte(at:u32)->u32 {return (weights[at/4u]>>(8u*(at%4u)))&255u;}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane:u32) {
 if lane==0u {atomicStore(&bad,0u);}
 workgroupBarrier();
 for(var i=lane;i<16384u;i+=64u) {
  if weights[i]!=expected_word(i) {atomicOr(&bad,1u);}
 }
 for(var i=lane;i<1024u;i+=64u) {
  if i>=3u && i<67u {
   if !finite(input[i]) {atomicOr(&bad,2u);}
  }else{if input[i]!=0u {atomicOr(&bad,4u);}}
  if output[i]!=0u {atomicOr(&bad,8u);}
  if i>0u && atomicLoad(&fault[i])!=0u {atomicOr(&bad,16u);}
 }
 if lane==0u && atomicLoad(&fault[0])!=0u {atomicOr(&bad,32u);}
 workgroupBarrier();
 if lane==0u {
  atomicOr(&fault[0],atomicLoad(&bad));
  full_fault=atomicLoad(&bad)|atomicLoad(&fault[0]);
 }
 workgroupBarrier();
 if workgroupUniformLoad(&full_fault)!=0u {return;}
 for(var row=0u;row<4u;row+=1u) {
  let block=PREFIX+row*18u;
  let d=unpack2x16float(byte(block)|(byte(block+1u)<<8u)).x;
  let code=(byte(block+2u+lane/4u)>>(2u*(lane%4u)))&3u;
  let value=d*f32(i32(code)-1);
  let product=value*bitcast<f32>(input[3u+lane]);
  output[8u+row*64u+lane]=bitcast<u32>(value);
  sums[lane]=product;
  workgroupBarrier();
  if lane<32u {sums[lane]=sums[lane]+sums[lane+32u];}
  workgroupBarrier();
  for(var step=16u;step>0u;step/=2u) {
   if lane<step {sums[lane]=sums[lane]+sums[lane+step];}
   workgroupBarrier();
  }
  if lane==0u {output[264u+row]=bitcast<u32>(sums[0]);}
  workgroupBarrier();
 }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;
    #[test]
    fn actual_limit_guards_include_uniform_fault_word_and_complete_source() {
        let limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 64,
            max_compute_workgroup_size_x: 64,
            max_compute_workgroup_storage_size: 264,
            max_storage_buffers_per_shader_stage: 4,
            max_bindings_per_bind_group: 4,
            max_bind_groups: 1,
            max_storage_buffer_binding_size: pool::BUFFER_BYTES,
            max_buffer_size: pool::BUFFER_BYTES,
            max_compute_workgroups_per_dimension: 1,
            ..Default::default()
        };
        validate_limits(&limits).unwrap();
        let mut bad = limits.clone();
        bad.max_compute_workgroup_storage_size = 263;
        assert!(validate_limits(&bad).is_err());
        bad = limits.clone();
        bad.max_storage_buffer_binding_size = pool::BUFFER_BYTES - 4;
        assert!(validate_limits(&bad).is_err());
        bad = limits.clone();
        bad.max_buffer_size = pool::BUFFER_BYTES - 1;
        assert!(validate_limits(&bad).is_err());
        bad = limits;
        bad.max_storage_buffers_per_shader_stage = 3;
        assert!(validate_limits(&bad).is_err());
    }
    #[test]
    fn registration_rejects_private_or_cross_experiment_inputs() {
        let args: Vec<OsString> = ["lab", "intel", "public-host-weight-probe"]
            .map(Into::into)
            .to_vec();
        let vars: Vec<(OsString, OsString)> = [
            ("KUNO_LAB_PUBLIC_HOST_WEIGHT_ONLY", "1"),
            ("KUNO_PUBLIC_RECEIPT", "new.json"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .to_vec();
        assert!(registration(&args, vars.clone(), true).is_ok());
        assert!(registration(&args, vars.clone(), false).is_err());
        for key in [
            "KUNO_LAB_PRIVATE_PROMPT",
            "KUNO_LAB_TARGET_REUSE",
            "KUNO_PUBLIC_MODEL",
            "KUNO_TEST_INPUT",
        ] {
            let mut bad = vars.clone();
            bad.push((key.into(), "0".into()));
            assert!(registration(&args, bad, true).is_err());
        }
        let mut duplicate = vars.clone();
        duplicate.push(vars[0].clone());
        assert!(registration(&args, duplicate, true).is_err());
        assert!(registration(&args, vars[..1].iter().cloned(), true).is_err());
        assert_eq!(
            OsStr::new("new.json"),
            registration(&args, vars, true).unwrap().as_os_str()
        );
    }
    #[test]
    fn complete_cpu_reference_preserves_all_gaps_and_original_four_codes() {
        let x = input();
        let a = expected(pool::Fixture::Aligned, &x).unwrap();
        let b = expected(pool::Fixture::Prefix3, &x).unwrap();
        assert_eq!(&a[..], &b[..]);
        assert!(a[..OUTPUT_FIRST].iter().all(|&x| x == 0));
        assert!(a[DOT_FIRST + 4..].iter().all(|&x| x == 0));
        assert!(
            a[OUTPUT_FIRST..DOT_FIRST + 4]
                .iter()
                .all(|&x| f32::from_bits(x).is_finite())
        );
        assert!(
            a[OUTPUT_FIRST..DOT_FIRST]
                .iter()
                .any(|&x| x == 1f32.to_bits())
        );
        assert!(expected(pool::Fixture::Aligned, &x[..WORDS - 1]).is_err());
    }
    #[test]
    fn complete_validation_rejects_tail_mutation_and_fault_publication() {
        let x = input();
        let initial = vec![0; WORDS];
        let mut normal = expected(pool::Fixture::Aligned, &x).unwrap().to_vec();
        normal.extend_from_slice(&initial);
        normal.extend_from_slice(&x);
        validate_readback(
            &normal,
            pool::Fixture::Aligned,
            Fault::None,
            &x,
            &initial,
            &initial,
        )
        .unwrap();
        for at in [
            0,
            OUTPUT_FIRST,
            DOT_FIRST,
            WORDS - 1,
            WORDS,
            2 * WORDS - 1,
            3 * WORDS - 1,
        ] {
            let mut corrupted = normal.clone();
            corrupted[at] ^= 1;
            assert!(
                validate_readback(
                    &corrupted,
                    pool::Fixture::Aligned,
                    Fault::None,
                    &x,
                    &initial,
                    &initial
                )
                .is_err()
            );
        }
        assert!(
            validate_readback(
                &normal[..3 * WORDS - 1],
                pool::Fixture::Aligned,
                Fault::None,
                &x,
                &initial,
                &initial
            )
            .is_err()
        );
        for fault in [
            Fault::SourceTail,
            Fault::InputNan,
            Fault::InputTail,
            Fault::OutputDirty,
            Fault::Sticky,
        ] {
            let mut initial_fault = vec![0; WORDS];
            if fault == Fault::Sticky {
                initial_fault[0] = 256;
            }
            let mut frame = initial.clone();
            frame.extend_from_slice(&initial_fault);
            frame.extend_from_slice(&x);
            frame[WORDS] = fault.bits();
            validate_readback(
                &frame,
                pool::Fixture::Prefix3,
                fault,
                &x,
                &initial,
                &initial_fault,
            )
            .unwrap();
            frame[OUTPUT_FIRST] = 1;
            assert!(
                validate_readback(
                    &frame,
                    pool::Fixture::Prefix3,
                    fault,
                    &x,
                    &initial,
                    &initial_fault
                )
                .is_err()
            );
        }
    }
    #[test]
    fn original64_shader_is_uniform_valid_and_sources_are_read_only() {
        for prefix in [0, 3] {
            let source = shader(prefix).unwrap();
            let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
            let info = wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
            let mut bindings = vec![];
            for (handle, var) in module.global_variables.iter() {
                if let Some(binding) = &var.binding {
                    assert!(!info.get_entry_point(0)[handle].is_empty());
                    bindings.push(binding.binding);
                    if let wgpu::naga::AddressSpace::Storage { access } = var.space {
                        assert_eq!(
                            access.contains(wgpu::naga::StorageAccess::STORE),
                            binding.binding >= 2
                        );
                    }
                }
            }
            bindings.sort();
            assert_eq!(bindings, vec![0, 1, 2, 3]);
            wgpu::naga::back::spv::write_vec(
                &module,
                &info,
                &wgpu::naga::back::spv::Options {
                    lang_version: (1, 3),
                    ..Default::default()
                },
                Some(&wgpu::naga::back::spv::PipelineOptions {
                    shader_stage: wgpu::naga::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
            assert!(
                source
                    .find("if workgroupUniformLoad(&full_fault)!=0u")
                    .unwrap()
                    < source.find("output[8u+").unwrap()
            );
            assert!(!source.contains("fma("));
        }
        assert!(shader(1).is_err());
    }
}
