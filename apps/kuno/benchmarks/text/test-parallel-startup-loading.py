#!/usr/bin/env python3
"""CPU-only real scoped-read/upload-completion policy and source-call gates.

No Cargo build, real model, GPU, service, sudo or inference. Optional --deps-dir
links existing real kuno_text/CubeCL libraries for original-descriptor tiny
GGUF tests and actual completion-future type checking. Those retained libraries
do not replace full compilation/hardware acceptance of the new worker.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps-dir', type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text'
source = native / 'src'
gpu = (source / 'strata_global_gpu.rs').read_text()
host = (source / 'strata_global_host.rs').read_text()
plan = (source / 'strata_global_plan.rs').read_text()
shadow = (source / 'strata_prompt_shadow.rs').read_text()
client = (native / 'vendor/cubecl-runtime/src/client.rs').read_text()
server = (native / 'vendor/cubecl-wgpu/src/compute/server.rs').read_text()
stream = (native / 'vendor/cubecl-wgpu/src/compute/stream.rs').read_text()
reader = (root / 'crates/kuno-text/src/gguf_positional.rs').read_text()

arm = gpu.index('let mut pending = upload_completion::Pending::new')
write = gpu.index('client\n', arm)
sync = gpu.index('let completion = client.sync();', write)
save = gpu.index('pending.submitted(', sync)
fill = gpu.index('Some(make_window(end)?)', save)
join = gpu.index('pending\n', fill)
assert arm < write < sync < save < fill < join
assert 'pending.finish()' in gpu[join:join + 150].replace('\n', '').replace(' ', '')
assert 'plan.load_chunk_read_at(graph, &reader' in gpu
assert 'reader.read_range(tensor, source, target)' in plan
assert 'tensor.read_range(source, target)' in plan  # Old and restore paths stay original.
assert host.count('parallel_reads::fill(') == 1
assert 'bytes > SOURCE_WINDOW as usize' in host
assert 'p.count.checked_mul(bytes)' in ''.join(host.split())
assert 'reader.read_range(' in host and 'reader.verify_file_unchanged()?' in host
assert 'parallel_reads::SETUP_BYTES' in host
for item in ['map.bytes_mut().zeroize();', 'fill(map.bytes_mut())?;',
             'map.cache(placement.coherent(), true)', 'PublicExternalHostSource::from_initialized_public_host(']:
    assert item in shadow
assert shadow.index('map.bytes_mut().zeroize();') < shadow.index('fill(map.bytes_mut())?;')
assert shadow.index('fill(map.bytes_mut())?;') < shadow.index('map.cache(placement.coherent(), true)')
assert 'pub fn sync(&self) -> DynFut<Result<(), ServerError>>' in client
server_sync = server[server.index('fn sync(&mut self, stream_id:'):]
assert server_sync.index('execute_streams') < server_sync.index('let result = stream.sync();')
stream_sync = stream[stream.index('pub fn sync('):]
assert stream_sync.index('.flush(') < stream_sync.index('Box::pin(async move')
# Actual source write is synchronously consumed by Scheduler before sync returns
# its future; write_buffer borrows it only while copying into backend staging.
assert 'ScheduleTask::Write {' in server and 'self.scheduler.register(stream_id, task, &[])' in server
assert 'self.write_to_buffer(&buffer, &data);' in stream
assert '.write_buffer(&resource.buffer, resource.offset, data)' in stream
assert 'self.files[tensor.info.shard_index].read_at(target, offset)' in reader
assert 'if result.is_err() {\n            output.fill(0);' in reader

integration = r'''
use kuno_text::{GgufFile, GgufLimits};
use std::{fs, path::PathBuf, sync::atomic::{AtomicUsize, Ordering}};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture { path: PathBuf, moved: PathBuf, model: GgufFile }
impl Drop for Fixture { fn drop(&mut self) {
    let _ = fs::remove_file(&self.path); let _ = fs::remove_file(&self.moved);
} }
fn fixture() -> Fixture {
    let path = std::env::temp_dir().join(format!("kuno-public-loading-{}-{}.gguf",
        std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let moved = path.with_extension("original");
    let mut bytes = b"GGUF".to_vec(); bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes()); bytes.extend(b"w"); bytes.extend(3u32.to_le_bytes());
    for n in [4u64, 2, 64] { bytes.extend(n.to_le_bytes()); }
    bytes.extend(0u32.to_le_bytes()); bytes.extend(0u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    for n in 0..512 { bytes.extend((n as f32).to_le_bytes()); }
    fs::write(&path, bytes).unwrap();
    let model = GgufFile::open(&path, GgufLimits::default()).unwrap();
    Fixture { path, moved, model }
}
#[test]
fn actual_original_positional_reader_fills_final_disjoint_spans_and_preserves_padding() {
    let f = fixture(); let tensor = f.model.tensor("w").unwrap();
    let reader = tensor.positional_reader().unwrap(); let mut bytes = vec![0; 2048+128];
    reads::fill(&mut bytes[..2048], 32, |ordinal, target| {
        reader.read_range(tensor, ordinal as u64*32, target)
    }).unwrap();
    for (n, word) in bytes[..2048].chunks_exact(4).enumerate() {
        assert_eq!(f32::from_le_bytes(word.try_into().unwrap()), n as f32);
    }
    assert!(bytes[2048..].iter().all(|&value| value == 0));
    let mut ordinary = [0;16]; tensor.read_row(2, &mut ordinary).unwrap();
    assert_eq!(f32::from_le_bytes(ordinary[..4].try_into().unwrap()), 8.0);
}
#[test]
fn actual_original_descriptor_survives_path_replacement_but_rejects_mutation_and_foreign_owner() {
    let f = fixture(); let foreign = fixture();
    let tensor = f.model.tensor("w").unwrap(); let reader = tensor.positional_reader().unwrap();
    fs::rename(&f.path, &f.moved).unwrap(); fs::write(&f.path, [255;128]).unwrap();
    let mut bytes = [0;32]; reader.read_range(tensor, 0, &mut bytes).unwrap();
    assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1.0);
    assert!(reader.read_range(foreign.model.tensor("w").unwrap(), 0, &mut bytes).is_err());
    assert!(reader.read_range(tensor, u64::MAX, &mut bytes).is_err());
    fs::OpenOptions::new().write(true).open(&f.moved).unwrap().set_len(64).unwrap();
    bytes.fill(255); assert!(reader.read_range(tensor, 0, &mut bytes).is_err());
    assert!(bytes.iter().all(|&value| value == 0));
}
#[allow(dead_code)]
fn actual_native_completion_future_types(
    client: &cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime>,
    roots: &mut Vec<cubecl::server::Handle>,
) {
    let mut pending = completion::Pending::new(roots, || cubecl::future::block_on(client.sync()));
    let actual = client.sync();
    pending.submitted(move || cubecl::future::block_on(actual));
    let _ = pending.finish();
}
'''
with tempfile.TemporaryDirectory(prefix='kuno-parallel-startup-cpu-') as directory:
    directory = Path(directory)
    harness = directory / 'actual-startup.rs'
    binary = directory / 'actual-startup-tests'
    modules = '\n'.join(f'#[path = "{source / file}"] mod {name};' for name, file in [
        ('reads', 'strata_parallel_reads.rs'), ('completion', 'strata_upload_completion.rs')])
    command = ['rustc', '--edition=2024', '--test', '-D', 'warnings']
    if args.deps_dir:
        command.extend(['-L', f'dependency={args.deps_dir}'])
        for name in ['kuno_text', 'cubecl']:
            choices = list(args.deps_dir.glob(f'lib{name}-*.rlib'))
            if not choices: parser.error(f'missing retained {name} rlib')
            library = max(choices, key=lambda path: path.stat().st_mtime_ns)
            command.extend(['--extern', f'{name}={library}'])
    harness.write_text(modules + (integration if args.deps_dir else ''))
    subprocess.run(command + [str(harness), '-o', str(binary)], check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('actual scoped-read and original upload completion/quarantine policies passed; GPU acceptance pending')
