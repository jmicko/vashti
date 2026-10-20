#!/usr/bin/env python3
"""Check actual pure QSA fixture geometry/checkers; no GPU or Cargo build."""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "apps/kuno/benchmarks/native-text/src"
text = (NATIVE / "strata_qsa_attention_numeric_probe.rs").read_text()
oracle = text.split("// QSA_ATTENTION_PARITY_ORACLE_BEGIN\n", 1)[1].split(
    "// QSA_ATTENTION_PARITY_ORACLE_END", 1
)[0]
assert "Kernel::PromptAttentionFault.scalar_source()" in text
assert "KernelId::new::<Self>()" in text
assert "Kernel::PromptAttentionFault," in text
assert "gr_batch_probe::read_native_public(arena, id)" in text
assert "arena.layout(id)?.owned_words" in text
assert "arena.upload_u32(&queries)" in text
assert "queries[0] = f32::NAN.to_bits();" in text
assert "reference_partial.as_slice() != actual_partial.as_slice()" in text
assert "reference_stats.as_slice() != actual_stats.as_slice()" in text
assert "reference_fault.as_slice() != actual_fault.as_slice()" in text
assert text.index("let attempt = check(") < text.index("arena.reset()?;") < text.index("reports.push(attempt?);")
with tempfile.TemporaryDirectory(prefix="kuno-qsa-attention-parity-oracle-") as directory:
    directory = Path(directory)
    source = directory / "oracle.rs"
    output = directory / "oracle-tests"
    source.write_text(oracle)
    subprocess.run(
        ["rustc", "--edition=2024", "--test", "-D", "warnings", str(source), "-o", str(output)],
        check=True,
    )
    subprocess.run([str(output), "--test-threads=1"], check=True)
print("distinct original scalar shader, real candidate launch, full-owner parity and reset-before-report checks passed")
