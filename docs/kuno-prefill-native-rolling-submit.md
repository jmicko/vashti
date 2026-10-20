# Native prefill rolling recording/submission candidate

This backend-only candidate addresses an execution difference from pinned Strata
v0.1.41 (`fb58e0db`): the SYCL expert path submits queue operations as later
experts are prepared, while the Rust native graph previously completed planning,
descriptor updates and command recording for a whole expert layer before its
first queue submission. The existing coarse `experts` enqueue field stops before
native capture performed inside sync. Use the added public
`expert_native_capture_cpu` and `expert_native_wait_cpu` counters to separate
that CPU capture time from actual native fence wait time. Public count-only `expert_native_rolling_recordings`,
`expert_native_rolling_windows` and `expert_native_rolling_segments` additionally
prove the intended path actually submitted original segments; they are credited
only after final joined submission/flight acceptance, then published after the
existing real completion sync. These are global
numeric public-diagnostic counters, never private content instrumentation.

The selected path is a fresh, unnamed MAIN recording with a real
`StrataPromptExpertF16Image` operation, at least1024 compute nodes and no more
than32768 total nodes. Timestamp-sampled recordings, named branches, direct
replay, draft/MTP and smaller recordings retain the existing submission path.
This changes scheduling, not arithmetic, the public API or request authority.

## One original recording and complete admission

Before any prefix can run, the original graph validates every physical
resource/descriptor interval, complete per-queue hazard plan, owner inventory,
recording metadata cap, command/pool inventory, queue identities and upload
boundary. The complete bounded `Batches` inventory validates every semaphore,
signal and incoming wait, including the initial INFO upload dependency. All
internal cross-queue wait values must point strictly backward in global segment
order and name the original signal's actual queue. Epoch values are reserved
before submission and never reused on failure.

`prepare_last_recording` first binds this exact recording to the original
lane/nonce/family and one original FlightBook entry, still `submitted=false`,
then resets that lane's actual fence. The existing Recording owns all original
resources, command pools, descriptor pool, pipeline references and semaphores.
No additional flight, fabricated completion or shortened owner lifetime exists.
Later descriptor allocation/update/command recording can still fail; the owned
submission guard remains armed across every such fallible operation and panic.
Distinct descriptor sets are updated while already submitted earlier sets stay
unchanged; command pools are neither reset nor freed while any prefix is live.

## Bounded rolling issuance and final completion

After each segment's real `end_command_buffer`, the guard records its closed
prefix. It issues a prefix after128 newly recorded nodes or64 newly closed
segments, whichever comes first. The final remainder is always issued. Open
command buffers are never submitted. Each original segment is issued in global
plan order, preserving its original queue, complete timeline waits/signals and
all original actual-range hazards and physical entry/exit barriers.

Global order is essential for unwind: every wait on an internal timeline then
names a signal whose host submission has already succeeded. It avoids leaving
an issued queue blocked behind a later unsubmitted prefix after an error.
The copy queue uses the existing adopted `KunoTransferQueue` authority and shared
host lock; original serialized compute/shared queues retain their existing
ownership. No new hardware queue or scope is constructed.

After the whole recording is closed, the final primary-queue submit waits for
both actual last-copy and last-shared timeline values and signals the ONE
original lane fence. Only then can the graph mark its matching original flight
submitted, clear duplicate enqueue metadata and disarm the rolling guard.
Ordinary completion/replay still requires that real final joined fence and
original ownership checks. Completed recordings remain replayable through the
unchanged original batched path; this candidate only overlaps initial capture.

## Error and panic recovery

Three separate recovery fences are allocated before the first prefix. The
guard marks a queue touched BEFORE each attempted Vulkan submission, including
an attempt that returns failure. Any error/panic before successful graph
acceptance queues a real empty submit with its dedicated fence on EVERY touched
queue, then waits every successfully queued recovery fence. It never waits on
an ordinary final fence that was not submitted. Recovery attempts every queue
even if another queue's submit/wait fails. Its waits use the existing120-second
native recovery horizon.

A failed prefix never receives FlightBook completion credit. The original
partial submitted=false flight therefore makes existing `Graph::Drop` refuse
ordinary retirement and conservatively retain the complete Recording/resources/
pools until protected context/process recovery. A failed recovery also retains
possibly-live recovery fences. This terminal quarantine is deliberate: no
unknown Vulkan work permits releasing its storage or reusing its lane.

The existing16MiB temporary submission metadata cap includes the added original
queue index in each bounded Record. A separate4096-byte host reservation covers
the rolling guard, cloned ash dispatch table, queue capability and recovery
fence metadata; a compile-time bound checks that actual size. Existing owner,
retained64MiB metadata, queue and query caps are preserved.

## CPU validation and remaining hardware gates

Run:

```sh
python3 apps/kuno/benchmarks/text/test-native-rolling-submit.py --deps-dir PATH/release/deps
python3 apps/kuno/benchmarks/text/test-native-command-boundaries.py
```

The first compiles the actual complete schedule, submit and rolling Vulkan
modules against existing real ash/wgpu/CubeCL runtime rlibs, with only sibling
metadata Wait/constants supplied by the existing standalone harness. It runs33
CPU tests: original planner and epoch/owner controls, full Vulkan borrowed-view
checks, randomized4096-node complete DAGs, closed-prefix windows, every injected
submission failure, panic after prefix, final-join failure, acceptance failure,
and invalid geometry. Graph source call-site checks verify full admission and
one-flight reservation before issuance, actual command end before prefix,
final queued join before mark/accept and original terminal owner retention.
The second checks real persistent per-physical-queue hazard maps and original
external boundaries. This is not a full worker build or hardware acceptance.

The parent task must compile the integrated worker and validate actual positive
requests, cancellation/error teardown, close/erasure/owned-zero and measured GPU
memory recovery. Then compare public coarse capture/wait, clean4245-token
prompt throughput and short-prompt/decode controls against the same artifact.
The original public one-layer timestamp diagnostic deliberately uses the old
serialized path, so its kernel classes cannot alone measure rolling overlap.

Tradeoff: this uses more host queue-submit calls than whole-queue batching,
although recording work now overlaps already submitted GPU work. The fixed
128-node/64-command window is a conservative first bounded choice. No speedup,
full worker compilation, live hardware success or installation is claimed by
source/CPU validation alone. No services or installed files were changed here.
