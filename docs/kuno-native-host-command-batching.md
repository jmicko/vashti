# Native Vulkan host command batching

Source-only candidate, 2026-10-20. This is a Kuno internal execution change;
there is no API, model, configuration, numerical kernel or Vashti change.
The current installed native32 worker is unchanged by this preparation.

## Exact boundary being preserved

The selected pinned MIT Strata SYCL source records the raw-slot `used` event
after both FP16 dequantization operations and before GU/activation/DOWN
(`sycl/src/prefill/prefill.cpp:3964-3983`). Slot refills wait on that original
event (`3812-3844`). Kuno's original DOWN-dequant release marker and queue
planner encode the same early release. Removing the marker would postpone raw
reuse until after the matrix work and is not this optimization.

Restoring the marker increased actual rolling segments from 24,773 to 48,610
in the complete fixed-public profile. Those are segments inside original
Recordings, not twice as many Recording owners. Capture CPU increased from
about 1.868s to 2.853s while actual wait decreased from 3.719s to 3.545s. These
historical whole-profile values motivate the source change; they do not measure
its individual host-call costs or predict a speedup.

## Complete host-only change

- Allocate original command buffers in at most one driver call per original
  MAIN/COPY/SHARED pool, then retain their exact global segment mapping. The
  original header remains Graph-owned and is never freed individually.
- Preflight actual final/scratch vector capacities against the remaining,
  unchanged 64MiB Graph metadata allowance before command-buffer allocation.
  A construction guard owns newly returned MAIN handles immediately, including
  partial allocation and callback unwind. It frees them only on construction
  failure; successful original Recordings still destroy their complete pool.
  COPY/SHARED allocations remain owned by their original pool throughout.
- Build stable Vulkan timeline/submit views once before the first rolling
  prefix, within the unchanged separately precharged 16MiB submission envelope.
  Their raw pointers refer only to final immutable heap-backed original records.
  No vector grows and no host allocation occurs while submitting rolling runs.
- Within each already-closed rolling window, pack only maximal **contiguous
  same-queue** runs of original `VkSubmitInfo` records into one synchronous
  driver call. Keep each command buffer, semaphore, signal value, wait value,
  stage mask and original order. No run crosses a window or reorders a queue.
- Mark the whole run's queue attempted before the call. A partial failure or
  panic grants no submission/completion credit and invokes original recovery
  fence/drain behavior; original Graph/Recording unknown-completion quarantine
  and FlightBook/request scope checks remain unchanged.

The complete plan still proves every peer dependency strictly backward and
forbids own-queue semaphore waits. Consequently a packed same-queue run cannot
wait on an unsubmitted record later in that run. The original final joined
fence and all actual GPU command counts are unchanged. Existing full-recording
replay retains its original per-queue order.

## CPU evidence and remaining acceptance

Actual production `strata_schedule.rs`, `strata_submit.rs` and
`strata_rolling.rs` typecheck in a hardware-free standalone harness using the
existing real ash/Wgpu/CubeCL-runtime libraries. All **37 CPU controls pass**,
including all original planner/dependency/rolling controls and new allocation
budget/error/unwind, complete actual C-view pointer/signal/wait, closed-window
packing and failed-whole-run recovery controls.

Artifacts: `/media/jmicko/faststore/kuno-development/native-host-command-batching-cpu-20261020/`.
The harness substitutes only sibling metadata constants, the public diagnostic
`enabled` hook and the INFO wait type; it does not instantiate a GPU device or
claim physical private-I/O/Graph lifetime acceptance. Production Graph's narrow
remaining-budget accounting block still needs whole-worker compilation.

Before selecting this candidate, the parent must run the combined original
numeric controls, whole-model prompt correctness, actual injected rolling
failure/fresh recovery, staged tools/MAX8/semantic/privacy acceptance and
physical memory recovery. No performance, install or release is claimed here.

## Read-only Mesa submission review (2026-10-20)

Do not assume packing contiguous VkSubmitInfo records delays intermediate
signals through runtime coalescing. Published upstream Mesa26.2.5
`src/vulkan/runtime/vk_queue.c:423-434` explicitly rejects merging when the
first submit signals and the next contains commands or waits. `vk_queue_merge_submit`
then submits the prior original record before retaining the next. Every Kuno
segment retains its own timeline signal, so this proposed runtime-merge story
is contradicted by that source. No batching rollback or new GPU test was made
from that speculation. This is a source finding, not a direct hardware timeline
measurement or verification of every Kisak distro patch.

Primary source: https://archive.mesa3d.org/mesa-26.2.5.tar.xz
SHA256794b3d1729075dae7ebd47c7283b7ca0cdb5dec668f7f11cc0a5d82f6138a0a5.
Only the two relevant MIT source files and provenance JSON remain under the
artifact base `mesa-driver-source-20261020`; the temporary66MiB archive was
removed. No driver/system setting or installed engine changed for this review.
