#!/usr/bin/env python3
"""CPU-only tests of the exact native command-boundary policy source.

No Cargo build, GPU, model, service, sudo, or private content is involved.
The policy and its unit tests are extracted unchanged between explicit markers
from the actual Vulkan encoder, then compiled with the local Rust compiler.
This is policy validation, not complete backend or hardware acceptance.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "apps/kuno/benchmarks/native-text/vendor/cubecl-wgpu/src/compute/strata_graph.rs"
text = SOURCE.read_text()
marker = "// NATIVE_COMMAND_BOUNDARIES_"
policy = text.split(marker + "BEGIN\n", 1)[1].split(marker + "END", 1)[0]
# Inspect real call sites, not a disconnected policy implementation. The
# per-queue hazard maps must stay outside and survive the complete node loop.
start = text.index("let mut previous: [HashMap<vk::Buffer, vk::AccessFlags>; 3]")
loop = text.index("for (index, n) in self.nodes.iter().enumerate()", start)
end = text.index("self.recordings.last_mut().unwrap().ready = true", loop)
assert text.count("let mut previous: [HashMap<vk::Buffer, vk::AccessFlags>; 3]") == 1
assert text.index("let boundaries = CommandBoundaries::from_queues(") < start < loop < end
assert "begin_native_command(&self.device, self.command, boundaries.before(0))?" in text
assert "end_native_command(&self.device, self.command, boundaries.after(segment))?" in text[loop:end]
assert "begin_native_command(&self.device, self.command, boundaries.before(segment))?" in text[loop:end]
assert "let previous = &mut previous[work.plan.segments[segment].queue()]" in text[loop:end]
assert "let accesses = self.node_accesses(n)?" in text[loop:end]
assert ".src_access_mask(before)" in text[loop:end] and ".dst_access_mask(access)" in text[loop:end]
assert text[loop:end].count("native_host_read_boundary(&self.device, self.command, target)?") == 2
with tempfile.TemporaryDirectory(prefix="kuno-native-command-boundary-cpu-") as directory:
    directory = Path(directory)
    source = directory / "policy.rs"
    binary = directory / "policy-tests"
    source.write_text(policy)
    subprocess.run([
        "rustc", "--edition=2024", "--test", "-D", "warnings", str(source), "-o", str(binary)
    ], check=True)
    subprocess.run([str(binary), "--test-threads=1"], check=True)
print("actual native encoder call-site and persistent hazard-map checks passed")
