# Native transfer and submission review — 2026-10-19

This isolated Kuno-only review starts at52a7f261. It performs no GPU work, sudo,
service change, model load or shared Cargo build. The root task owns hardware
validation and installation. No API, private logging or Vashti change.

## What the actual transfer path does

`strata_global_host` / `public_host_expert_mirror` allocate Vulkan memory with
HOST_VISIBLE and explicitly non-DEVICE_LOCAL memory type AND heap. Selection
prefers HOST_CACHED, then HOST_COHERENT. Model setup writes the mapped allocation,
flushes noncoherent memory, and imports the same initialized VkBuffer into the
immutable CubeCL registry. This is not a new CPU payload copy for each expert.
Do not claim portable pinning from flags alone.

`Sources::stage_prompt_group` resolves retained original sources and uses actual
`PublicHostCopy` descriptors. Native `stage_public_host_copies` drains scheduled
kernel descriptions without submitting, then appends real vkCmdCopyBuffer nodes.
IoScopes accumulates deferred error futures, without polling a fence per copy.
Automatic native submission remains deferred to genuine bounded request/program
boundaries. Raw ring descriptors are exact per-slot physical views in a separate
root. The adopted copy queue really is a distinct transfer-capable family.

Pinned Strata SYCL source remains tagv0.1.41 / fb58e0dbc8399662c0e47c76578c6e878b14f6cf.
Its selected native FP16 prefill also stages eight raw expert blobs and releases
each after GU/DOWN decoding, then uses oneMKL products. Page-locked source arenas
avoid its extra pageable-to-pinned CPU staging. Kuno already avoids that extra
per-expert CPU payload copy; no missing transfer mechanism was established.

The coupled public profile reports27,315,942,400 HOST-copy bytes in39,834
segments on its warm repeat, with21.459s inside the expert interval. At a
hypothetical13GB/s transfer rate those bytes take2.10s; that is an arithmetic
lower bound, NOT a measured bandwidth claim. A direct-path public DMA benchmark
could reuse the actual immutable source registration, PublicHostCopy path and
private raw owner without constructing a full Session. A realistic688KiB segment
fixture needs a bounded larger compiled-public pool with COPY_SRC, not the
old64KiB STORAGE-only test pool silently relabeled as representative.

## Descriptor hazards

The native graph derives read/write classes from actual Naga usage or SPIR-V
binding visibility, but uses each complete WgpuResource descriptor interval.
The expert stage[1] descriptor is80MiB although the two GU/DOWN decoded slots
together use~19.66MB. Its real math accesses do not narrow that inventory.
Encoding further merges by VkBuffer and uses WHOLE_SIZE barriers. This creates
conservative compute-side hazards for disjoint GU/DOWN/slot spans.

However, all these operations use the same compute queue, the raw DMA ring is
physically separate, and the shared atomic fault word already introduces a
write dependency between most kernels. Narrowing only the weight descriptor
does not demonstrate additional copy/compute overlap. Any future refinement must
validate actual kernel accesses and fault publication, not assert narrower
invented ownership. This candidate leaves those guards unchanged.

## Bounded timeline submission candidate

`strata_schedule::Commands::submit` previously made one vkQueueSubmit host call
per segment. An expert graph alternates copy/dequant-release/math segments, so
many thousands of host driver calls can occur inside the coarse expert fence.
The reported enqueue-only timer does not include later graph recording,
planning, submission and waiting. CPU submission cost must therefore be measured
separately before calling this a device-only bottleneck.

The candidate groups the SAME segment command buffers into at most one
vkQueueSubmit call per active physical queue, then keeps the original final
all-queue fence join. Original per-queue order, every timeline signal and wait,
ALL_COMMANDS wait masks, upload-header dependency, raw/DQ owner inventory,
copy-queue locking authority and retained failure ownership are unchanged.

Only the existing explicitly TIMELINE semaphores participate; this helper is
not a binary-semaphore batching API. Vulkan timeline wait-before-signal submission
on a different queue is required. A submit return grants no completion. If any
call fails, the existing graph retains the original recording and resources;
the final join is not issued as if all submissions succeeded.

Validation rejects incomplete/null command inventories, malformed header/signal
order, self/forward/wrong-queue wait edges, inconsistent final-copy/shared
signals, null/aliased owned semaphores and exhausted timeline ranges. Queue
authority also rejects null active queues. A named16MiB temporary metadata
envelope is added to NATIVE_QUEUE_HOST_BYTES. Records and borrowed timeline /
VkSubmitInfo arrays remain address-stable through each host call; source arrays
remain owned independently until the original joined fence.

## CPU acceptance and pending hardware gates

`rustc --test -D warnings` against the actual modified strata_schedule.rs and
strata_submit.rs passes27 tests:23 original planner/scope/dependency tests plus
four new tests. The harness uses the existing real ash, wgpu and CubeCL runtime
rlibs; metadata-only sibling Wait/constants stand-ins match their source types
and sizes. It neither rebuilds dependencies nor constructs a GPU device.
Harness and output: `/tmp/kuno-prefill-submit-review/{tests.rs,results.log}`.
The reproducible harness is also saved at
`apps/kuno/benchmarks/text/strata-submit-cpu-tests.rs`.
Standalone Clippy with all warnings denied stops on the pre-existing
`strata_schedule.rs::update_regions` needless_range_loop. Repeating with only
that inherited lint allowed passes; this is not claimed as strict whole-module
Clippy acceptance. No unrelated loop refactor is included.

New tests inspect the actual borrowed Vulkan command, semaphore, timeline-value,
wait-mask and pNext inventories during their live callback; preserve all three
queue orders; simulate progress of4096-node randomized real-planner DAGs for
both compute lanes and serialized/unserialized modes; reject malformed graphs;
and cover the full32768-node /32769-segment bound and empty queues.

This is NOT a production full-build, GPU correctness, performance or installation
acceptance. The root task must build the actual integrated worker, run protected
complete requests/erasure/close/recovery, and compare recording/submission/wait
times and matched complete prompt performance. Consider this candidate useful
across prefill/decode/startup only if those measurements establish a gain.
