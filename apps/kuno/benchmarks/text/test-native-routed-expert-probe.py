#!/usr/bin/env python3
"""CPU-only checks of the actual routed-probe fixtures and original group ABI.

Links already-built dependencies; never constructs a GPU/client, opens a model,
builds Cargo, touches services or establishes actual kernel numerical parity.
Full worker type checking and protected compiled-public GPU acceptance remain
required. Pure source functions are copied verbatim into the temporary harness.
"""
import argparse
from pathlib import Path
import re
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps-dir', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
source = root / 'apps/kuno/benchmarks/native-text/src'
shared = (source / 'strata_routed_expert_probe.rs').read_text()
original = (source / 'strata_cubek_expert_probe.rs').read_text()
native = (source / 'strata_native_expert_probe.rs').read_text()
groups = (source / 'strata_prompt_groups.rs').read_text()
allocation = (source / 'strata_prompt_allocation.rs').read_text()
assert 'pub(crate) const DEFAULT_ROWS: usize = kuno_text::ops::ple_transition::MAX_PROMPT_BATCH;' in allocation

def cases(text):
    return [(int(n), control) for n, control in re.findall(r'\((\d+), (\w+)\)', text)]

assert cases(original) == [(32, 'Finite'), (33, 'Finite'), (64, 'Finite'),
    (257, 'Finite'), (33, 'NonfiniteFirst'), (33, 'NonfiniteOther'), (33, 'InvalidRoute')]
assert cases(native) == [(32, 'Finite'), (33, 'Finite'), (64, 'Finite'),
    (256, 'Finite'), (257, 'Finite'), (287, 'Finite'), (288, 'Finite'),
    (383, 'Finite'), (384, 'Finite'), (257, 'NonfiniteFirst'),
    (257, 'NonfiniteOther'), (257, 'InvalidRoutePrefix'), (257, 'InvalidRoute')]
assert 'super::try_expert(' in original and 'if !actual' in original
assert 'super::native_expert_wide::try_expert(' in native
assert 'super::native::run_expert(' in native
assert 'if count >= 256' in native and 'if !actual' in native and 'if actual' in native
for guard in ['!arena.public_owned_zero_audit_enabled()', 'arena.buffer_count() != 0',
              '!std::ptr::eq(client.info(), arena.client()?.info())',
              'error > 1e-4 + 3e-4 * reference.abs()',
              'let expected_fault = if control == Control::Finite { 0 } else { 128 };',
              'arena.reset()?;', 'scan(arena, fault,', 'scan(arena, status,']:
    assert guard in shared, guard
assert shared.index('arena.reset()?;') < shared.index('reports.push(checked?);')

# Original pure group constants/functions, not a replacement grouping oracle.
group_constants = groups[groups.index('pub(crate) const HOST_BYTES:'):groups.index('/// One compiled prompt ABI')]
group_functions = groups[groups.index('const MEMBERS:'):groups.index('/// Real prompt-only views.')]
group_module = '''mod strata_prompt {
    pub fn invalid_buffer_length() -> kuno_text::Error { kuno_text::Error::InvalidBufferLength }
    pub mod allocation { pub const DEFAULT_ROWS: usize = kuno_text::ops::ple_transition::MAX_PROMPT_BATCH; }
    pub mod groups {
        use super::allocation;
        use kuno_text::{Error, Result};
        use zeroize::Zeroizing;
''' + group_constants + group_functions + '} }\n'

fixture = shared[:shared.index('// Guard the genuine returned mapped lease')]
fixture = fixture.replace('use crate::private_arena::{BufferId, PrivateArena};', '')
fixture = fixture.replace('use crate::resident_util as u;', '')
window = shared[shared.index('fn read_window('):shared.index('fn scan(')]
tests = shared[shared.index('#[cfg(test)]\nmod tests {'):]
with tempfile.TemporaryDirectory(prefix='kuno-routed-probe-cpu-') as directory:
    directory = Path(directory)
    harness = directory / 'actual-routed-fixtures.rs'
    binary = directory / 'actual-routed-fixtures'
    # Use CubeCL's actual StreamId re-export so the retained library resolves
    # its own matching cubecl_common version, rather than picking another build.
    harness.write_text('#![allow(dead_code, unused_imports)]\n' + group_module
        + 'mod actual {\n' + fixture + '\nuse cubecl as cubecl_common;\n' + window + tests + '\n}\n')
    command = ['rustc', '--edition=2024', '--test', '-D', 'warnings', '-L', f'dependency={args.deps_dir}']
    for name in ['kuno_text', 'cubecl', 'half', 'zeroize']:
        choices = list(args.deps_dir.glob(f'lib{name}-*.rlib'))
        if not choices:
            parser.error(f'missing retained {name} library')
        library = max(choices, key=lambda path: path.stat().st_mtime_ns)
        command.extend(['--extern', f'{name}={library}'])
    subprocess.run(command + [str(harness), '-o', str(binary)], check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('Original14/native26 case schedules and actual authentic fixtures passed; GPU kernel parity remains pending')
