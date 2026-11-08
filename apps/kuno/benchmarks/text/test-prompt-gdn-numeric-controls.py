#!/usr/bin/env python3
"""Compile the exact fixed-public GDN fixture/checker, without GPU or Cargo."""
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[4]
native = root / 'apps/kuno/benchmarks/native-text/src'
text = (native / 'strata_prompt_gdn_numeric_probe.rs').read_text()
fixture = text.split('// PROMPT_GDN_PARITY_FIXTURE_BEGIN\n', 1)[1].split(
    '// PROMPT_GDN_PARITY_FIXTURE_END', 1
)[0]
assert 'Op::RecurrenceFault.scalar_source()' in text
assert 'Box::new(Op::RecurrenceFault)' in text
assert 'Box::new(ScalarRecurrence)' in text
assert 'KernelId::new::<Self>()' in text
assert '!arena.public_owned_zero_audit_enabled()' in text
assert '|| crate::strata_prompt::active()' in text
assert 'layout.logical_words != words' in text
assert 'owner.size_in_used() != layout.owned_bytes as u64' in text
assert 'crate::strata_prompt::storage::overlaps(arena.handle(previous)?, owner)?' in text
assert 'gr_batch_probe::read_native_public(arena, id)?' in text
assert 'words.len() != arena.layout(id)?.owned_words' in text
assert text.index('let attempt = check(') < text.index('arena.reset()?;') < text.index('reports.push(attempt?);')
with tempfile.TemporaryDirectory(prefix='kuno-prompt-gdn-fixture-') as directory:
    directory = Path(directory)
    source = directory / 'actual-fixture.rs'
    output = directory / 'actual-fixture-tests'
    source.write_text(fixture)
    subprocess.run(['rustc', '--edition=2024', '--test', '-D', 'warnings', str(source), '-o', str(output)], check=True)
    subprocess.run([str(output), '--test-threads=1'], check=True)
print('actual fixture/checker and primitive identity/admission/full-owner/reset call-site checks passed')
