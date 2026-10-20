# Fixed-public prefill operator timing

The existing `--public-prefill-profile` worker diagnostic now requests one
bounded native timestamp sample: request3, first4096-token chunk, expert layer24.
The first request remains the short warmup; request2 loads/executes the long
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
