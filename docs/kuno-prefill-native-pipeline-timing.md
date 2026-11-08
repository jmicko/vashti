# Public native pipeline creation timing

This source-only diagnostic extends the existing compiled public prefill profile
and host profile entries. It adds no option, private wire field, fake prompt,
warm-up, pipeline preparation policy, GPU query, dispatch or fence. The original
clean fixture and correctness controls remain unchanged. No build or hardware
result is claimed at this checkpoint.

The existing `strata-public-host-clock` feature supplies a scoped process guard.
Only the public prefill entry can activate it, after the original inherited
environment scrub and public authorization. The backend also requires native
execution, `KUNO_LAB_LANGUAGE_PUBLIC_ONLY=1` and exactly `prefill-layer` or
`prefill-host-layer`. The guard is established before device and model loading,
and disables future samples on return, error or unwind. Private entry never
activates it or emits a snapshot. Inactive timers retain no clock and do not
read the environment, allocate, or change execution. The feature-off source
contains no timing hooks.

Two cumulative rows use fixed labels:

- `server_cache_miss_prepare`: starts after the real in-process pipeline-cache
  miss. Includes cache lookup, source/module compilation and genuine pipeline
  admission, and optional compilation-cache insertion. An in-process hit does
  not contribute. A SPIR-V disk-cache hit still prepares a module/admission and
  contributes. Under native-prepared selection this excludes actual native
  executable creation.
- `native_driver_pipeline_create`: brackets exactly the existing Vulkan
  `create_compute_pipelines` call, after the original native source, subgroup,
  layout and device checks. Shader/layout setup, descriptor-template creation,
  graph-node binding and command encoding are outside this row. It measures host
  API wall time, including time spent waiting inside the driver, with no claim
  of thread CPU time or GPU execution time.

Each row has `[success_count, success_wall_ns, failure_count, failure_wall_ns]`.
Totals saturate rather than wrap. A server preparation returning an error or
unwinding contributes to failure; a successful Vulkan API call contributes to
success even if a later graph operation fails. Neither count certifies enqueue,
completion, correctness, erasure, or recovery.

`public_native_pipeline_creation` records contain those labels, column labels,
cumulative totals, a fixed phase, and the ordinal of the compiled public request.
They contain no kernel identities, variants, INFO, model owners, tensor values,
routes, prompts, or output. Snapshots occur at:

1. `load_complete`, ordinal 0, after the original load timing has been captured;
2. `request_begin`, before the original request timer starts;
3. `last_prompt_input`, immediately after the original prompt endpoint timing;
4. `request_end`, after `session.run` returns, including its decode and cleanup.

Subtract adjacent snapshots to isolate load, prompt processing, and later
request work. The final snapshot is also emitted after an error return; only the
unchanged original completion and recovery receipts may establish success. The
existing final prepared/admission counters retain their exact meaning and are
not substituted by these timing rows.

Serialization adds a small amount of host work to these already diagnostic
profile modes. Their prompt times remain diagnostic, never a clean selection.
There is no established speed improvement: eager preparation could move measured
cost into load and is not implemented. Repeated execution with zero new
pipelines receives no pipeline-creation saving from such a shift.

Three additive CPU controls include the actual production timing helper from
the worker test module: exact activation guards, inactive timers retaining no
clock or counters, and separate success/failure accumulation with saturation.
They require no device. All original AppCPU, runtime and metadata populations
remain required independently; the new controls do not establish hardware or
performance results.
