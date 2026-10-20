#!/usr/bin/env python3
"""CPU-only actual prompt GDN source/Naga tests using existing dependencies.

No Cargo/shared-target build, model, GPU, root or installed service operation.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps-dir', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text/src'
text = (native / 'strata_prompt_gdn.rs').read_text()
code = text.split('#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]', 1)[1]
code = '#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]' + code.split('impl KernelMetadata for Op', 1)[0]
code = code.replace('include_str!("', f'include_str!("{native}/')
header = '#![allow(dead_code)]\nmod prompt_gdn {\n'
header += f'#[path="{native / "strata_prompt_gdn_specialize.rs"}"] mod specialize;\n'
libraries = list(args.deps_dir.glob('libwgpu-*.rlib'))
if not libraries:
    parser.error('no existing wgpu rlib under --deps-dir')
wgpu = max(libraries, key=lambda path: path.stat().st_mtime_ns)
with tempfile.TemporaryDirectory(prefix='kuno-prompt-gdn-specialize-') as directory:
    directory = Path(directory)
    source = directory / 'actual-source-tests.rs'
    output = directory / 'actual-source-tests'
    source.write_text(header + code + '}\n')
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source),
                    '-L', f'dependency={args.deps_dir}', '--extern', f'wgpu={wgpu}',
                    '-o', str(output)], check=True)
    subprocess.run([str(output), '--test-threads=1'], check=True)
print('actual prompt GDN source, ordered state/FMA/closing invariants, and Naga/SPIR-V checks passed')
