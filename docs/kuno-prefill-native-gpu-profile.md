# Fixed-public prefill operator timing

The existing `--public-prefill-profile` worker diagnostic now requests one
bounded native timestamp sample: request3, first4096-token chunk, expert layer24.
The first request is a73-token ordinary-prompt warmup; request2 loads/executes the long
fixture; request3 is its warm repeat. Input is compiled public text. There is no
private-wire option, API, configuration field or Vashti change.

Startup first scrubs inherited `KUNO_LAB_`, `KUNO_STRATA_`, `KUNO_TEST_` and
`KUNO_PUBLIC_` variables. Only the explicit compiled-public profiling entry then
sets `KUNO_STRATA_NATIVE_PROFILE=prefill-layer` and its public authorization before
runtime threads exist. The existing target-window parser ignores this new mode,
so it does not select another target-controller experiment. Ordinary private
entry never enables the coarse collector or native sampling guard.

`NativeExpertLayer` enters immediately before the real expert enqueue and remains
active through `Timer::mark("experts")`, whose original `client.sync()` joins
completion. Error/unwind disables sampling and does not report incomplete results.
Success then emits a separate `public_prefill_native_gpu_profile` line containing
only fixed class labels and numeric counts/nanoseconds. Generated small/macro
expert kernels map to `matrix`; GU and DOWN are intentionally combined. Other
classes are `dequant`, `gather`, `activate`, `ordered_copy`, `ordered_fill`, `other`.
No expert IDs, route values, tensor names, tokens or request content are emitted.

The same public coarse `experts` interval also reports two existing backend
host counters: `expert_native_capture_cpu` and `expert_native_wait_cpu`.
Each retains the collector's numeric triple format `[calls,total_ns,total_ns]`;
the third field deliberately repeats this CPU counter rather than representing
an enqueue slice. Deltas cover the interval through its existing real sync,
with no new fence, timestamp pool or private profiling switch. These are
process-global native counters, so the protected public worker must remain the
sole workload. Capture includes native planning/descriptor/command recording and host queue
submission calls; with rolling submission, its later portion overlaps already
submitted GPU work. Wait includes actual native fence wait calls. Neither
is a GPU kernel duration. The old `experts` enqueue field ends before native
recording in sync and therefore does not include capture CPU work.

The same public `experts` interval reports anonymous actual rolling issuance
counters as `[count,0,0]`: `expert_native_rolling_recordings`,
`expert_native_rolling_windows` and `expert_native_rolling_segments`.
A recording is credited only after all original segments and the final joined
fence have actually been queued and its matching original flight accepted.
Windows count nonempty successful prefix submissions, not selector intentions.
The existing Timer publishes deltas only after its actual completion sync;
these process-global counters contain no request identity, route or content and
are never logged/published by ordinary private inference. Timestamp-sampled
recordings deliberately use the original nonrolling path and contribute zero.

The existing backend owns every query pool through the actual matching fence,
requires at least2048 compute nodes, caps recordings at16384 nodes, validates
timestamp geometry, serializes the sampled plan and disables replay. Those
semantics and resource caps are unchanged. A selected interval with no eligible
recording honestly reports `timestamp_calls:0`; it is not a measured operator
breakdown. Timestamps use `BOTTOM_OF_PIPE`; node timing excludes preceding
per-node barrier gaps. This deliberately perturbs overlap and cannot establish
uninstrumented throughput or replace whole-request/fault/erasure acceptance.

The worker also exposes a separate `public_matrix_timing_check(path)` hook for
root's opt-in `--public-prefill-matrix-timing-check` CLI. It invokes the existing
fixed-public original expert microtiming helper, then requires the actual arena
close and owned-zero audit before publishing `public_prefill_matrix_timing`.
Original full numerical controls remain a separate unchanged branch.

CPU preparation: `python3 apps/kuno/benchmarks/text/test-public-prefill-native-sample.py`
compiles the actual marked selector/classifier and actual target-mode parser with
`rustc --test -D warnings`. Three tests cover the unique authorized geometry,
static real generated-kernel classification and separation from target profiling.
Structural checks cover real startup scrubbing, guard lifetime through sync,
unchanged native query caps/no-replay/fence collection, and microtiming close/audit.
Rustfmt and diff checks pass. No full Cargo build, GPU, sudo, services, installation
or hardware result was performed while preparing this change. Root must build and
validate actual query sampling plus ordinary numerical/privacy/recovery gates.


## Expert-batch host breakdown (2026-10-21)

The same original fixed-public request3/layer24 guard also samples six consecutive
backend expert-batch intervals, returned as `host_batch` alongside `host_capture`
in the existing diagnostic JSON (never the inference API). Columns remain calls,
wall_ns, actual CLOCK_THREAD_CPUTIME_ID ns, cpu_clock_unavailable. Labels cover
scheduler drain/admission, exact binding inventory, pipeline/grammar, uniform+
source scratch setup, actual node enqueue, and finishing the original I/O scopes.
Private/disabled calls retain no clock, counter or content. Sampling adds no GPU
command/fence, changes no batch/source/owner proof, and uses the existing public
host-clock feature. Normal rolling issuance is unchanged. Nested node costs are
included in batch enqueue; do not sum these into the parent expert interval.

Purpose: distinguish backend CPU work from queue waits before selecting the next
large performance change. This source-only checkpoint has no new hardware result,
speed gain, API/config/frontend/release change or installation claim. Preserve
original154 CPU/11 runtime/189 GPU controls and complete both-mode acceptance;
these diagnostic fields cannot authorize narrower sources or relaxed limits.


The completed first normal run (source7a7cd5b0/SHA8ed3be10) passes compute and
actual physical recovery. Sampled390-group layer batch costs total about4.6ms
actual thread CPU (bindings0.816ms, grammar0.870ms, uniforms0.351ms, nodes2.110ms,
drain0.318ms, finish0.130ms); whole warm expert interval6.0357s includes1.5703s
before final sync. These are not proof that backend table lookup dominates.
Next public-only `expert_boundary_*` labels split original owner reset, route
enqueue, route readback+original checkpoint, host grouping, map upload and expert
collector/backend return. They retain [calls,wall_ns,thread_cpu_ns], with explicit
`expert_boundary_cpu_clock_unavailable` count when needed. Ordinary disabled
collection creates no clock or new fence. Existing CpuTimer triples are unchanged.
The readback phase includes GPU waits/recycling; thread CPU disambiguates it.
No generation/API/private logging behavior changes, and no gain is claimed.
