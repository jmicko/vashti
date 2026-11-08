# Fixed-public target submission backlog experiment

This is an internal, default-off experiment based on `ef2c15f`, following the
[fixed-public target request I/O contract](kuno-native-mtp-all-rows-plan.md#fixed-public-target-request-io-reuse-implemented-and-public-fixture-accepted).
Source preparation does not establish numerical, admission, speed or physical
closing acceptance. The initial controller scope is the original public DASLab
speculative target with full Q8 MTP on the independently identified Intel GPU.
Nvidia and the draft stream retain their existing construction and512 policy.
The native controllers below implement selection; protected labels and hardware
acceptance remain separately owned.

## Exact runtime API and construction boundary

`cubecl_runtime::server::PublicTargetSubmissionBacklog` has only two variants:

| Policy | `task_window()` | `minimum_submission_cost()` |
| --- | ---: | ---: |
| `Default512` (default) | 512 | 0 |
| `Queued2048` | 2048 | 1 |

The explicit target-only constructor is
`ComputeClient::configure_public_target_request_io_with_backlog(policy)`.
`ComputeClient::public_target_submission_backlog()` reports the actual immutable
runtime configuration, not completed inference or a process-global wait count.
The original `configure_public_target_request_io()` still selects `Default512`.
Other runtimes reject the new methods by default. No environment lookup or
environment mutation selects this policy.

Call the constructor before target model construction on the inline target
stream. `Queued2048` requires a native stream with the original `tasks_max=256`.
The actual runtime rejects a non-inline device, a retired stream or any already
configured private I/O stream before flushing or installing the policy. Both
generic private-I/O and fixed-target-I/O construction count as already
configured. The server's original serialized construction guard also remains.
There is no late mutable setter, request-scope selection or warm-reset bypass.

Construction retains the original setup flush, fixed target I/O allocation and
real `drain_submitted()` proof. Only after this real drain does selected2048
start its empty regulation state. Default512 does not reset the original stamp
or counter, preserving its old transition sequence. A failed constructor does
not select a new policy. The configuration getter requires an actual active,
non-retired fixed-target I/O configuration; it cannot credit a generic/draft
stream or a stream that has completed terminal retirement.

## Submission and wait semantics

`SchedulerMultiStream::max_tasks`, `WgpuStream::tasks_max`, the queue submit
points, parameter-copy ordering and numerical kernels remain unchanged.
`WgpuStream::flush_inner()` submits the optional original INFO copy buffer
followed by the original tasks command buffer, then calls the same regulation
point under `CleanupBoundary::ordinary_regulation`.

The original first-submission rule is retained: save its actual submission stamp
with zero accumulated tasks. For every subsequent actual submission, add its
task cost. When that count reaches the policy window, swap the saved stamp with
the current stamp, wait on the **previous** stamp using the original
`device.poll(Wait { submission_index: Some(previous), timeout: None })`, record
the original content-free attribution and reset the task counter only after
that poll. Poll warnings and error handling remain unchanged. Neither policy
replaces this with a wait on the current submission.

Default512 adds exactly `tasks_count`, including the historical zero cost of an
INFO-only submission. Selected2048 adds `max(tasks_count,1)` with saturating
counter addition: a real INFO-only submission costs one slot, and an invalid
huge count cannot wrap away a required wait. A flush with neither tasks nor
pending INFO returns before `queue.submit()` and before regulator accounting;
it earns no submission credit. The selected minimum cost is required to bound
submission count even if an arbitrary caller issues repeated INFO-only flushes.
It changes no canonical kernel/submission arithmetic.

2048 is a **regulation window**, not a guarantee of at most2048 unfinished tasks.
The previous-stamp wait intentionally retains a lagging interval. If every
actual submitted batch costs at most `S`, then the first wait is reached with at
most `W+2S-1` charged tasks outstanding; steady-state outstanding work is bounded
by `2(W+S-1)`, where `W` is the chosen window. Each selected submission costs at
least one, so its unfinished submission count is bounded by the same envelope.
There is no new callback/future list or unbounded application queue.
This envelope assumes the original previous-stamp poll completes successfully;
the retained original warning path on poll failure is not a completion proof.
The unchanged error futures and complete closing gates still reject failed
device execution. The policy cannot establish driver completion by arithmetic.

The canonical scalar/kernel batch threshold remains256. For the approved
ordinary inference path, an original native owner-clear batch can overshoot it:
`resident_util::zero_many()` chunks at most16 owners and
`transient::clear_owner_bindings()` preserves that bound, so255 prior tasks plus
16 clears permits a271-task submission. Actual private read batches add two
tasks per nonempty descriptor and explicitly flush. The only grouped target
`read_async` call is the original or grouped PLE three-owner read, permitting at
most255+6=261 tasks; other target ordinary reads use a single descriptor. These
original bounded arrays, full owner/cap checks and scheduling are unchanged.

For this ordinary regulated inference envelope, `S=271` gives selected2048 at
most4636 unfinished charged tasks/submissions; the comparable positive-task512
envelope is1564. The window adds at most3072 charged task slots to that
conservative source bound. These are not observed driver occupancy. The typed
API checks the256 scheduler threshold; it does not impose an arbitrary private
read-vector271 bound. Any future larger descriptor batch must recompute `S`.

This4636 envelope does **not** cover every raw runtime or closing command.
`CleanupBoundary::ordinary_regulation()` deliberately skips the regulator during
original cleanup. `TransientIo::sanitize_after_drain()` retains its own raw
queue/audit commands and checked drains; full private-owner closing erasure and
single-owner cleanup read maps keep their original real drain/deadline gates.
Original public write-buffer64-write drains also remain outside this ordinary
regulated task envelope. None of these commands, proofs or drains is changed or
credited by the policy.

## Owner, metadata and error-future inventory

No tensor/private buffer, public source buffer, GPU scratch owner, host private
plane, INFO reservation or scheduler queue is added. The regulator retains the
same one submission stamp and one `usize` counter. Separate enum variants carry
the immutable policy without another payload field or allocation. Source tests
compare actual regulator size to a separate copy of the original state layout.

Application-owned allocation and retention boundaries remain in their original
functions, at the same enqueue/flush/fence locations:

- `SchedulerMultiStream::register/execute_streams` still drains its256-task
  application task vector into the original target stream. Moving the later
  driver wait does not retain another scheduler task vector or its handle list.
- `flush_inner()` still calls `prepare_batched_info()`, then performs the same
  submission, `memory_cleanup(false)` and `release_uniforms()` each flush.
  The policy does not increase application uniform-pool retention.
- `TransientIo` still charges full upload/INFO/readback owners and simultaneous
  audit headroom against the128MiB ledger before allocation. Its write-once INFO
  pages, pinned cache records and binding-cache resources stay subject to the
  original full-capacity4MiB metadata checks. The selected wait window neither
  releases these owners early nor grants another capacity allowance.
- `IoScopes::finish()` still appends its original three popped error futures
  per scope, and the same fence/audit paths drain them. Regulation has no error
  scopes, callbacks or future allocations. The same enqueue/flush schedule
  therefore creates **zero additional application error futures** and zero
  additional retained application handle/cache records under this policy.
- Explicit read maps, complete fault and status proofs, request I/O closing
  audits, cleanup deadlines, full original model-owner erasure and terminal
  retirement remain untouched. This does not suppress or defer a proof fence.

wgpu/driver command-buffer retention, native execution metadata and allocation
bytes are opaque. This is also true for the original512 policy. The longer
window may retain more native command buffers and driver references even though
the explicit application owner/cap accounting is unchanged. This document does
**not** claim an exact driver-byte guarantee or add a generic budget. The
original536,870,912-byte shared driver margin, all original caps and fresh full
physical admission are mandatory. Physical full closing/recovery audits prove
only the actual tested case; a failed admission must remain a failed candidate.

## Focused source proof and physical gate

CPU source tests cover independent original512 stamp-sequence equivalence,
selected2048 boundary and previous-stamp waits, reset-after-poll state, first
submission, irregular271/256/small/INFO-only batches, the finite envelope,
counter overflow, original state size and rejection of reconfiguration,
retirement, non-inline selection and a non256 selected scheduler threshold.
All existing numerical/source/owner tests and tolerances remain authoritative.

The controller must report the actual enum, window2048 and minimum actual
submission cost1 separately from process-global attribution spanning target and
draft runtimes. It must prove initial scope and actual target construction,
retain unchanged cold/warm CPU13 full vectors, carries, greedy/repair checks and
full original closure, then compare normal-language control/candidate with CPU
compilation idle and complete fresh admission/recovery. Nested attention/proof
and GPU-wait/map timers overlap and must not be added or treated as isolated
kernel cost. No speed or30-token/s claim follows from fewer regulator waits.

## Native controller contract and CPU receipt

Both public controllers accept only the explicit selector
`KUNO_LAB_TARGET_SUBMISSION_BACKLOG=queued-2048-public`; absence preserves512.
Selection requires original speculative full-Q8 MTP on the independently
identified Intel device, fused MTP sweeps, frozen original HOST/VRAM expert
sources, GPU routing, partition exit, direct fill, inline execution, Verify3
logical/finite fusion, chunk checkpoints and full request reuse. CPU13 also
requires its original warm replay and diagnostic bank placement. Target-only,
Nvidia placement, Q2 draft, English pruning, head broadcast, prepared/live MoE,
old codebooks/scales/factors, two-row GU, staging and three-member arithmetic
are rejected. Existing valid GU integer/stride, DOWN integer or original
four-row/stride, grouped PLE/status, chronological QSA, three-streaming GDN and
packed IQ4 policies compose under their independent scope guards.

The selected typed constructor runs before target model allocation. Cold and
warm CPU13 passes therefore both use the selected runtime backlog; model
arithmetic policies retain their original warm-only activation. Draft uses its
unchanged generic64MiB I/O constructor and must reject the target policy query.
The receipt checks the target's actual immutable policy after construction and
again after original full model/draft closure. It reports window and minimum
actual-submission cost separately, explicitly gives no completed device
execution credit, and does not reinterpret global wait attribution as a
per-target metric. It adds no owner or metadata charge and preserves admission,
all full closing/source proofs, numerical references and tolerances.

Seven real regulator unit tests passed using an exact-source state harness,
with the actual pinned wgpu29.0.4 and vendored CubeCL runtime/common types. The
harness copied the complete non-wasm/wasm regulator source tail and all seven
tests verbatim, plus the original transient I/O fault helper. The stream-tail
SHA256 was
`b2d6793eebfd7b14dbba0e98ffbe2dbfecef6a1083ccf162283c1079da1f3bde`;
the fault-helper SHA256 was
`aeb88ffe32cd2d62b326e4cc85dfdde365aa751faf19ab03d64a1f53bfb50965`.
The standalone temporary harness used the existing
`runtime-review-target-20261008` cache and performed no device creation or GPU
work. Its attribution callback was a no-op, so this receipt proves state,
finite-count and guard behavior, not driver poll completion or hardware timing.
The original vendor crate's full strict compile is required by the native
controller check; no repository manifest was changed for the harness.

The native controller source check passed eight focused backlog tests, then all
44 language-controller,36 CPU13-controller and31 target-helper tests
(111 distinct tests including the eight new ones). Commands used the pinned
standalone native-text manifest, `--locked --no-default-features --features direct-device --bin kuno-text-lab`, each named test filter and `-j2`, with only
the existing `oracle-native-target-20261009` cache. Strict native
`cargo clippy --all-targets -j2 -- -D warnings` passed with the same manifest,
lockfile and pure direct-device feature selection; this compiled the real
vendored runtime source. Owned-source formatting and diff checks passed. CPU
jobs were idle at2026-10-11T19:17:36Z. No GPU/service/system changes were made;
physical admission, numerical replay and throughput remain pending integration.


Physical v166 proof completed2026-10-11: original CPU13 and actual selected target2048/draft512, complete closure and recovery pass. One ordered code200 pair10.451238original512 vs10.482525queued2048 shows no material gain. Original512 stays selected for next candidate; explicit2048 experiment remains available. Full numeric/source/policy evidence: v166_intel_target_backlog_physical and latest coordination handoff. Prior pending statements describe source checkpoint.
