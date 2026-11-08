#!/usr/bin/env python3
"""CPU-only tests of the actual prompt QSA source generator and Naga lowering.

Use existing Cargo dependencies: --deps-dir PATH/release/deps. This runs no
Cargo build, GPU operation, model load, root command or installed worker.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--deps-dir', required=True, type=Path)
parser.add_argument('--paired', action='store_true', help='also select the opt-in actual paired prompt hook')
parser.add_argument('--pv-gather', action='store_true', help='also select the opt-in actual PV probability gather hook')
args = parser.parse_args()
root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text/src'
text = (native / 'strata_qsa.rs').read_text()
code = text.split('#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]', 1)[1]
code = '#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]' + code.split('impl KernelMetadata for Kernel', 1)[0]
code = code.replace('include_str!("', f'include_str!("{native}/')
header = '''#![allow(dead_code)]
mod qsa_gpu {
    pub const MAX_CONTEXT: usize = 32768;
    pub const MAX_POSITION: u32 = 32767;
    pub const TOKEN: usize = 1152;
}
mod mtp_resident_kernels { pub const CONTEXT: usize = 32768; }
mod strata_qsa {
const WINDOW_SCORE_GRID: [usize; 3] = [256, 1, 1];
'''
header += f'#[path="{native / "strata_prompt_qsa_specialize.rs"}"] mod prompt_specialize;\n'
header += f'#[path="{native / "strata_prompt_qsa_paired.rs"}"] mod prompt_paired;\n'
header += f'#[path="{native / "strata_prompt_qsa_pv_gather.rs"}"] mod prompt_pv_gather;\n'
libraries = {}
for name in ['wgpu', 'half', 'cubecl']:
    matches = list(args.deps_dir.glob(f'lib{name}-*.rlib'))
    if not matches:
        parser.error(f'no existing {name} rlib under --deps-dir')
    libraries[name] = max(matches, key=lambda path: path.stat().st_mtime_ns)
with tempfile.TemporaryDirectory(prefix='kuno-prompt-qsa-specialize-') as directory:
    directory = Path(directory)
    source = directory / 'actual-source-tests.rs'
    output = directory / 'actual-source-tests'
    source.write_text(header + code + '}\n')
    command = ['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source),
               '-L', f'dependency={args.deps_dir}']
    if args.paired:
        command.extend(['--cfg', 'feature="strata-qsa-paired"'])
    if args.pv_gather:
        command.extend(['--cfg', 'feature="strata-qsa-pv-gather"'])
    for name, library in libraries.items():
        command.extend(['--extern', f'{name}={library}'])
    command.extend(['-o', str(output)])
    subprocess.run(command, check=True)
    subprocess.run([str(output), '--test-threads=1'], check=True)
print('actual source generation, signed/grouped Q8 decode, original ordering, and Naga/SPIR-V checks passed')
