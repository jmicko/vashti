#!/usr/bin/env python3
"""Compile the exact public QSA oracle/checker. No model, GPU or Cargo build."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / 'apps/kuno/benchmarks/native-text/src'
text = (NATIVE / 'strata_qsa_select_numeric_probe.rs').read_text()
oracle = text.split('// QSA_ACTIVE_PREFIX_ORACLE_BEGIN\n', 1)[1].split(
    '// QSA_ACTIVE_PREFIX_ORACLE_END', 1
)[0]
assert 'vec![f32::NAN; ROWS * BLOCKS]' in text
assert 'vec![u32::MAX; ROWS * BLOCKS]' in text
assert 'Op::Validate, count, owners.clone(), p' in text
assert 'Op::Select, count, owners, p' in text
assert 'layout.owned_words' in text
assert text.index('let attempt = check(') < text.index('arena.reset()?;') < text.index('reports.push(attempt?);')
worker = (NATIVE / 'strata_worker.rs').read_text()
assert 'let attention_checks = crate::strata_qsa::selection::public_numeric' in worker
assert '"attention_checks":attention_checks' in worker
with tempfile.TemporaryDirectory(prefix='kuno-qsa-active-prefix-oracle-') as directory:
    directory = Path(directory)
    source = directory / 'oracle.rs'
    output = directory / 'oracle-tests'
    source.write_text(oracle)
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source), '-o', str(output)], check=True)
    subprocess.run([str(output), '--test-threads=1'], check=True)
print('actual primitive call-sites, poison owners, full-output scan and reset-before-report checks passed')
