#!/usr/bin/env python3
"""CPU-only fixed-public sampler policy and real call-site checks.

Compiles the unchanged marked policy and existing target-mode parser directly
with rustc. Does not build Cargo targets, access models/GPUs or run a service.
This does not establish native query-pool completion or hardware timings.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
NATIVE = ROOT / "apps/kuno/benchmarks/native-text"
profile = (NATIVE / "src/strata_prefill_profile.rs").read_text()
worker = (NATIVE / "src/strata_worker.rs").read_text()
prompt = (NATIVE / "src/resident_model_prompt.rs").read_text()
graph = (NATIVE / "vendor/cubecl-wgpu/src/compute/strata_graph.rs").read_text()
port = (NATIVE / "src/strata_port.rs").read_text()
policy = profile.split("// PUBLIC_PREFILL_NATIVE_SAMPLE_BEGIN\n", 1)[1].split(
    "// PUBLIC_PREFILL_NATIVE_SAMPLE_END", 1
)[0]
target = port[port.rfind("#[derive(", 0, port.index("enum NativeProfileSample")):
              port.index("fn native_profile_sample()")]
target_test = """
#[test]
fn prefill_sampling_never_selects_target_verification_policy() {
    assert_eq!(NativeProfileSample::parse("prefill-layer"), None);
    assert_eq!(NativeProfileSample::parse("round-8"), Some(NativeProfileSample::First));
    assert_eq!(NativeProfileSample::parse("warm-round-64"), Some(NativeProfileSample::Warm));
    assert!(NativeProfileSample::First.selects(1, 8));
    assert!(NativeProfileSample::Warm.selects(2, 64));
}
"""
# Authorization is created only after inherited experiment configuration was
# scrubbed, and before any runtime threads. Private entry never enables it.
scrub = worker.index("std::env::remove_var(name)")
public_gate = worker.index("if public_check && prefill_check == Some(true)")
public_enable = worker.index("strata_prefill_profile::enable_compiled_public()")
runtime = worker.index("cubecl::future::block_on(serve(")
assert scrub < public_gate < public_enable < runtime
assert worker.count("strata_prefill_profile::enable_compiled_public()") == 1
assert "run_mode(path, false, None, false, false, false)" in worker
assert 'set_var("KUNO_STRATA_NATIVE_PROFILE", "prefill-layer")' in worker[public_gate:runtime]
assert 'set_var("KUNO_LAB_LANGUAGE_PUBLIC_ONLY", "1")' in worker[public_gate:runtime]
# The RAII scope must include the original completion fence. Enqueue alone is
# insufficient because graph profiling is decided later on actual submission.
sample = prompt.index("let native_sample =")
enqueue = prompt.index("?.enqueue_prompt(", sample)
completed = prompt.index('timing.mark("experts")?;', enqueue)
finish = prompt.index("native_sample.finish();", completed)
assert sample < enqueue < completed < finish
assert 'cubecl::future::block_on(client.sync())' in profile
assert '"public_prefill_native_gpu_profile {}"' in profile
# Anonymous rolling metadata is published only through the existing enabled
# public Timer after its original real sync, never from the backend itself.
assert 'native_strata_graph_rolling_stats()' in profile
mark = profile.index("pub(crate) fn mark(&mut self, name: &'static str)")
sync = profile.index('cubecl::future::block_on(client.sync())', mark)
for field in ['expert_native_rolling_recordings', 'expert_native_rolling_windows', 'expert_native_rolling_segments']:
    assert profile.index('"' + field + '"', mark) > sync
issued = graph.index('let [windows, segments] = rolling.submitted_counts();')
accepted = graph.index('rolling.accept().map_err(fault)?;', issued)
assert graph.index('total.fetch_add(amount,', accepted) > accepted
assert 'pub fn native_strata_graph_rolling_stats() -> [u64; 3]' in graph
assert '"expert_native_rolling_' not in graph
assert 'if let Some(before) = self.0.take()' in profile
assert 'impl Drop for NativeExpertLayer' in profile
# Preserve existing query capacity, timing authorization, serialized diagnostic
# and no-replay semantics; do not turn this into ordinary throughput profiling.
assert 'Ok("round-8" | "warm-round-64" | "prefill-layer")' in graph
assert 'profile_active() && compute_count >= 2048' in graph
assert 'self.nodes.len() > 16384' in graph
assert 'get_query_pool_results' in graph and 'QueryResultFlags::WAIT' in graph
completion = graph[graph.index('fn complete_lane_submission'):graph.index('pub(crate) fn queued_transfer_only')]
assert completion.index('.wait_for_fences(') < completion.index('.collect()?;')
assert completion.index('.collect()?;') < completion.index('.finish_joined(')
assert '!profile_active()' in graph and 'r.profile.is_none()' in graph
# The separate microtiming entry remains opt-in and publishes only after the
# actual owned arena close/audit. It does not replace the53 numerical controls.
timing = worker.index("if matrix_timing {", worker.index("if matrix_check {"))
probe = worker.index("strata_matrix::public_timing", timing)
close = worker.index("arena.close()", probe)
audit = worker.index("arena.owned_zero_audit_receipt()", close)
report = worker.index('"public_prefill_matrix_timing {}"', audit)
numeric = worker.index("strata_matrix::public_numeric", report)
assert timing < probe < close < audit < report < numeric
with tempfile.TemporaryDirectory(prefix="kuno-public-prefill-sample-cpu-") as directory:
    directory = Path(directory)
    source = directory / "policy.rs"
    binary = directory / "policy-tests"
    source.write_text(policy + "\n" + target + "\n" + target_test)
    subprocess.run([
        "rustc", "--edition=2024", "--test", "-D", "warnings", str(source), "-o", str(binary)
    ], check=True)
    subprocess.run([str(binary), "--test-threads=1"], check=True)
print("actual public-only startup, fence-spanning scope, unchanged query caps and matrix close/audit checks passed")
