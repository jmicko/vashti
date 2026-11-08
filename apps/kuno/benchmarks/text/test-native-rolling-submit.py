#!/usr/bin/env python3
"""CPU-only actual native rolling lifecycle/selection/planner/Vulkan-view tests.

No Cargo build, GPU, model, services, sudo or private content. Actual schedule,
submit and rolling modules compile against prebuilt real ash/wgpu/CubeCL rlibs;
metadata-only sibling constants/Wait match the existing source types. The
Graph selection/lifecycle call sites are also checked, not hardware acceptance.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps-dir', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
compute = root / 'apps/kuno/benchmarks/native-text/vendor/cubecl-wgpu/src/compute'
graph = (compute / 'strata_graph.rs').read_text()
marker = '// NATIVE_ROLLING_SELECTION_'
selection = graph.split(marker + 'BEGIN\n', 1)[1].split(marker + 'END', 1)[0]
# Complete owner/planner/cap preflight and sole flight precede first submission.
preflight = graph.index('if metadata > 64 * 1024 * 1024')
push = graph.index('self.recordings.push(recording)', preflight)
assert preflight < push < graph.index('.begin_rolling(')
assert graph.index('self.prepare_last_recording(compute_count, false, false, false)?') < graph.index('.begin_rolling(')
loop = graph.index('for (index, n) in self.nodes.iter().enumerate()')
close = graph.index('end_native_command(&self.device, self.command, boundaries.after(segment))?', loop)
rolling = graph.index('rolling.closed(next, index, false)', close)
begin = graph.index('begin_native_command(&self.device, self.command, boundaries.before(segment))?', rolling)
assert close < rolling < begin
assert graph.index('rolling.finish().map_err(fault)?') < graph.index('self.mark_last_submitted(id)?', loop)
assert graph.index('self.mark_last_submitted(id)?', loop) < graph.index('rolling.accept().map_err(fault)?')
assert 'std::mem::forget(std::mem::take(&mut self.recordings))' in graph
# The opt-in terminal public fault consumes an explicit one-shot only AFTER
# real production closed-prefix submission and leaves original recovery active.
rolling_source = (compute / 'strata_rolling.rs').read_text()
closed = rolling_source.index('let before = self.0.next;')
assert rolling_source.index('self.0.closed(end, nodes, force)?;', closed) < rolling_source.index('PUBLIC_FAILURE.after_window(self.0.next != before)', closed)
assert 'super::super::strata_graph::enabled()' in rolling_source
assert 'KUNO_LAB_LANGUAGE_PUBLIC_ONLY' in rolling_source
assert 'PUBLIC_FAILURE.recovered_queue(waited)' in rolling_source
assert 'PUBLIC_FAILURE.recovered_queue(false)' in rolling_source
assert 'fn native_strata_public_rolling_failure_stats() -> [u64; 3]' in graph
libraries = {}
for name in ['ash', 'wgpu', 'cubecl_runtime']:
    choices = list(args.deps_dir.glob(f'lib{name}-*.rlib'))
    if not choices:
        parser.error(f'missing existing {name} rlib')
    libraries[name] = max(choices, key=lambda path: path.stat().st_mtime_ns)
with tempfile.TemporaryDirectory(prefix='kuno-native-rolling-cpu-') as directory:
    directory = Path(directory)
    harness = (root / 'apps/kuno/benchmarks/text/strata-submit-cpu-tests.rs').read_text()
    harness = harness.replace('../native-text/vendor/cubecl-wgpu/src/compute/strata_schedule.rs', str(compute / 'strata_schedule.rs'))
    source = directory / 'actual-native-tests.rs'
    source.write_text(harness + '\n' + selection)
    binary = directory / 'actual-native-tests'
    command = ['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source), '-L', f'dependency={args.deps_dir}']
    for name, path in libraries.items():
        command.extend(['--extern', f'{name}={path}'])
    command.extend(['-o', str(binary)])
    subprocess.run(command, check=True)
    subprocess.run([str(binary), '--test-threads=1'], check=True)
print('actual native rolling lifecycle, queue-plan/views, and graph call-site controls passed')
