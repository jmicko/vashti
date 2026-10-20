#!/usr/bin/env python3
"""Compile actual native-Graph device-limit admission with rustc, CPU only.

The tiny SPIR-V fixtures encode entry/size metadata, not executable shaders.
This establishes parser and arithmetic guards, not Vulkan module validity,
device-property accuracy, GPU numerical behavior or runtime recovery.
"""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
graph = (root / 'apps/kuno/benchmarks/native-text/vendor/cubecl-wgpu/src/compute/strata_graph.rs').read_text()
policy = graph.split('// NATIVE_PROGRAM_DEVICE_LIMITS_BEGIN\n', 1)[1].split(
    '// NATIVE_PROGRAM_DEVICE_LIMITS_END', 1)[0]

# Actual physical properties, retained by this Graph; the selected original
# program must pass before native Vulkan shader/pipeline allocation.
new = graph.index('pub(crate) fn new(device: &wgpu::Device)')
for field in ['max_compute_workgroup_subgroups', 'min_subgroup_size',
              'max_subgroup_size', 'max_compute_work_group_invocations',
              'max_compute_work_group_size', 'max_compute_shared_memory_size']:
    assert field in graph[new:graph.index('pub(crate) fn initialize_resources', new)]
assert graph.index('get_physical_device_properties2(', new) < graph.index('let program_limits =', new)
assert 'program_limits: ProgramDeviceLimits,' in graph
assert 'u64::try_from(kernel.shared_size)' in graph
push = graph.index('pub(crate) fn push(')
gate = graph.index('self.program_limits', push)
shader = graph.index('self.device.create_shader_module(', gate)
pipeline = graph.index('self.device.create_compute_pipelines(', shader)
assert push < gate < shader < pipeline
assert '.validate(' in graph[gate:shader]
assert 'p.entry.as_bytes()' in graph[gate:shader]
assert 'p.required_subgroup' in graph[gate:shader] and 'p.shared_bytes' in graph[gate:shader]
# No new backend profiling, fabricated completion, selector limit widening,
# environment override or request-dependent property source in this policy.
for forbidden in ['std::env', 'create_', 'unsafe', 'Atomic', 'finish_joined',
                  'max_subgroups: 64', 'max_subgroups: 32', 'fault(']:
    assert forbidden not in policy

tests = r'''
fn limits() -> ProgramDeviceLimits {
    ProgramDeviceLimits { min_subgroup: 8, max_subgroup: 32, max_subgroups: 64,
        max_invocations: 1024, max_size: [1024, 1024, 64], max_shared_bytes: 49152 }
}
fn metadata(size: [u32; 3]) -> Vec<u32> {
    let mut words = vec![0x07230203, 0x00010300, 0, 128, 0,
        (5 << 16) | 15, 5, 4, 0x6e69616d, 0,
        (6 << 16) | 16, 4, 17];
    words.extend_from_slice(&size); words
}
fn admit(l: ProgramDeviceLimits, size: [u32; 3], width: Option<u32>, shared: Option<u64>)
    -> Result<(), &'static str> {
    l.validate(&metadata(size), b"main", width, shared)
}
#[test]
fn actual_subgroup_cap_gates_new_32_subgroup_shape() {
    assert_eq!(admit(limits(), [512, 1, 1], Some(16), Some(49152)), Ok(()));
    let mut l = limits(); l.max_subgroups = 32;
    assert_eq!(admit(l, [512, 1, 1], Some(16), Some(49152)), Ok(()));
    l.max_subgroups = 31;
    assert_eq!(admit(l, [512, 1, 1], Some(16), Some(49152)),
        Err("native program compute subgroup count limit"));
}
#[test]
fn existing_fixed_shapes_and_partial_subgroup_are_preserved() {
    for size in [1, 16, 32, 64, 128, 256] {
        for width in [16, 32] {
            assert_eq!(admit(limits(), [size, 1, 1], Some(width), Some(32768)), Ok(()));
        }
    }
    let mut l = limits(); l.max_subgroups = 1;
    assert_eq!(admit(l, [32, 1, 1], Some(32), None), Ok(()));
    assert_eq!(admit(l, [33, 1, 1], Some(32), None),
        Err("native program compute subgroup count limit"));
    l.max_subgroups = 0;
    assert_eq!(admit(l, [1, 1, 1], Some(32), None),
        Err("native program compute subgroup count limit"));
}
#[test]
fn automatic_subgroup_choice_is_not_replaced_by_minimum_width_guess() {
    let mut l = limits(); l.max_subgroups = 32;
    assert_eq!(admit(l, [1024, 1, 1], None, Some(0)), Ok(()));
    assert_eq!(admit(l, [1024, 1, 1], Some(32), Some(0)), Ok(()));
    assert_eq!(admit(l, [1024, 1, 1], Some(16), Some(0)),
        Err("native program compute subgroup count limit"));
}
#[test]
fn all_actual_dimension_and_invocation_caps_apply() {
    for size in [[0, 1, 1], [1025, 1, 1], [1, 1025, 1], [1, 1, 65]] {
        assert_eq!(admit(limits(), size, Some(16), None),
            Err("native program workgroup dimension limit"));
    }
    assert_eq!(admit(limits(), [512, 2, 1], Some(16), None), Ok(()));
    assert_eq!(admit(limits(), [513, 2, 1], Some(16), None),
        Err("native program workgroup invocation limit"));
    let mut l = limits(); l.max_size = [u32::MAX; 3]; l.max_invocations = u32::MAX;
    assert_eq!(admit(l, [65536, 65536, 1], Some(16), None),
        Err("native program workgroup invocation overflow"));
}
#[test]
fn actual_required_width_and_shared_footprint_are_bounded() {
    for width in [0, 1, 3, 7, 15, 24, 64] {
        assert_eq!(admit(limits(), [128, 1, 1], Some(width), None),
            Err("native program required subgroup size limit"));
    }
    assert_eq!(admit(limits(), [128, 1, 1], Some(16), Some(49153)),
        Err("native program shared memory limit"));
    assert_eq!(admit(limits(), [128, 1, 1], Some(16), Some(u64::MAX)),
        Err("native program shared memory limit"));
}
#[test]
fn selected_entry_name_and_compute_model_are_exact() {
    let mut words = metadata([128, 1, 1]);
    assert!(limits().validate(&words, b"mai", Some(16), None).is_err());
    assert!(limits().validate(&words, b"mainlonger", Some(16), None).is_err());
    words[6] = 4;
    assert!(limits().validate(&words, b"main", Some(16), None).is_err());
    words[6] = 5; words[7] = 0;
    assert!(limits().validate(&words, b"main", Some(16), None).is_err());
    words[7] = 128;
    assert!(limits().validate(&words, b"main", Some(16), None).is_err());
}
#[test]
fn multiple_entries_use_only_selected_literal_size() {
    let mut words = metadata([128, 1, 1]);
    words.extend_from_slice(&[(5 << 16) | 15, 5, 5, 0x72656874, 0,
        (6 << 16) | 16, 5, 17, 2048, 1, 1]);
    assert_eq!(limits().validate(&words, b"main", Some(16), None), Ok(()));
    assert!(limits().validate(&words, b"ther", Some(16), None).is_err());
    words.extend_from_slice(&[(5 << 16) | 15, 5, 6, 0x6e69616d, 0]);
    assert_eq!(limits().validate(&words, b"main", Some(16), None),
        Err("native program workgroup compute entry"));
}
#[test]
fn malformed_modules_and_instruction_extents_are_rejected_without_panics() {
    for len in 0..5 {
        assert!(limits().validate(&metadata([1, 1, 1])[..len], b"main", Some(16), None).is_err());
    }
    for extra in [vec![0], vec![(20 << 16) | 1, 0]] {
        let mut words = metadata([1, 1, 1]); words.extend(extra);
        assert!(limits().validate(&words, b"main", Some(16), None).is_err());
    }
}
#[test]
fn missing_duplicate_and_non_literal_sizes_are_not_admitted() {
    let mut words = metadata([128, 1, 1]); words.truncate(10);
    assert_eq!(limits().validate(&words, b"main", Some(16), None),
        Err("native program literal workgroup missing"));
    let mut words = metadata([128, 1, 1]);
    words.extend_from_slice(&[(6 << 16) | 16, 4, 17, 128, 1, 1]);
    assert_eq!(limits().validate(&words, b"main", Some(16), None),
        Err("native program literal workgroup mode"));
    for op in [16, 331] {
        let mut words = metadata([128, 1, 1]);
        words.extend_from_slice(&[(6 << 16) | op, 4, 38, 5, 6, 7]);
        assert_eq!(limits().validate(&words, b"main", Some(16), None),
            Err("native program specialized workgroup unsupported"));
    }
    let mut words = metadata([128, 1, 1]);
    words.extend_from_slice(&[(4 << 16) | 71, 10, 11, 25]);
    assert_eq!(limits().validate(&words, b"main", Some(16), None),
        Err("native program overriding workgroup builtin unsupported"));
}
'''
with tempfile.TemporaryDirectory(prefix='kuno-native-program-device-limits-') as directory:
    directory = Path(directory)
    source = directory / 'limits.rs'
    binary = directory / 'limits-tests'
    source.write_text(policy + tests)
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('actual device-limit parser and arithmetic plus pre-allocation call sites passed')
