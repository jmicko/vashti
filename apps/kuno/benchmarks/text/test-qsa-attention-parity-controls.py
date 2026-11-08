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
assert "reference_partial.as_slice() == actual_partial.as_slice()" in text
assert "reference_stats.as_slice() == actual_stats.as_slice()" in text
assert "reference_fault.as_slice() != actual_fault.as_slice()" in text
assert "if !all_accepted" in text
assert "public QSA attention original bit or independent F64 acceptance failed" in text
assert 'for (side, partial_words, stats_words)' in text
assert '("original_scalar", &reference_partial, &reference_stats)' in text
assert '("optimized", &actual_partial, &actual_stats)' in text
assert '"original_bitwise_required":!nondyadic' in text
assert "query_offset as u32" in text
assert "[0x2113u32, 0x2487, 0x299d, 0x2c53]" in text
assert text.index("let attempt = check(") < text.index("arena.reset()?;") < text.index("let (report, accepted) = attempt?;") < text.index("reports.push(report);")
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
    f64_source = directory / "f64-oracle.rs"
    f64_output = directory / "f64-oracle-tests"
    f64_source.write_text('#[path = ' + repr(str(NATIVE / "strata_qsa_attention_f64_oracle.rs")).replace("'", '"') + '] mod oracle;\n')
    subprocess.run(
        ["rustc", "--edition=2024", "--test", "-O", "-D", "warnings", str(f64_source), "-o", str(f64_output)],
        check=True,
    )
    subprocess.run([str(f64_output), "--test-threads=1"], check=True)
print("distinct original scalar shader, real candidate launch, full-owner parity and reset-before-report checks passed")
