# Native command-buffer boundary review — 2026-10-20

This Kuno-only candidate starts at032e2cdd in the isolated
`codex/kuno-prefill-expert-runtime-audit` worktree. It performs no GPU inference,
sudo, service change, model load or shared Cargo build. The root task owns final
integration, numerical/device validation, matched performance and installation.
No API, private logging, model arithmetic or Vashti behavior changes.

## Selected-path finding

The actual native Vulkan encoder calls `native_memory_boundary` at both the
beginning and end of EVERY command-buffer segment. The barrier has
ALL_COMMANDS source/destination stages and MEMORY_WRITE -> MEMORY_READ|MEMORY_WRITE
accesses. Expert GU/DOWN dequantization releases and intervening DMA naturally
produce many segments. The earlier submission-batching change reduced host
`vkQueueSubmit` calls but retained these GPU barriers; its negligible gain does
not measure their cost. Their contribution is not yet timed.

The candidate retains the global boundary before the FIRST and after the LAST
segment of EACH original physical queue. Only interior segment boundaries omit
this extra global barrier. All command buffers, queue order, per-node buffer
hazards, original timeline values/waits/signals, host-read visibility, resource
inventories and final joined fences remain unchanged.

`CommandBoundaries` derives first/last indices directly from the completed
original plan's segment queue IDs. It has three fixed first/last stack entries,
accepts at most32769 segments and the original compute header, and allocates no
retained resource or metadata. No cap is increased or resource range narrowed.

## Source proof and limits

The proof concerns the actual selected encoder, not command-buffer order alone:

1. `Graph::node_bindings` includes every used WGSL binding with access flags
   derived from Naga entry-point usage. Actual SPIR-V bindings retain the compiled
   READ/READ_WRITE classification. Uniform/INFO and atomic fault bindings are
   included; no special exception allows fault writers to overlap.
2. The SAME `node_bindings` includes the actual physical BDA
   `read_dependencies`, real COPY source/destination, real FILL destination and
   INDIRECT argument. `Graph::push` checks the original bound/initialized
   resources and rejects read dependencies on a non-physical program. The
   candidate does not infer aliases from shader names or offsets.
3. `node_ranges` feeds those exact binding intervals into the unchanged
   `strata_schedule::Planner`, which records real inter-queue RAW/WAR/WAW edges
   and resource unions. `node_accesses` feeds the same bindings into the
   unchanged intra-queue encoder, conservatively merging by actual VkBuffer.
   WHOLE_SIZE buffer barriers are deliberately retained.
4. The three `previous` access maps are initialized once OUTSIDE the complete
   node/segment loop, then indexed by the actual segment's queue. Switching
   command buffers does not reset hazard history. A buffer access crossing a
   same-queue segment boundary receives the same original memory barrier.
5. Cross-queue dependencies retain the original timeline signal/wait inventories
   and ALL_COMMANDS wait masks. These semaphore dependencies supply device
   availability and visibility across queues. Vulkan's semaphore signal access
   scope includes all device accesses, as does the wait's second access scope;
   the candidate does not rely merely on a CPU fence or host submission order.
   [Khronos synchronization specification](https://docs.vulkan.org/spec/latest/chapters/synchronization.html#synchronization-semaphores-waiting).
6. The global boundary at the start/end of every USED queue is retained. This
   preserves native/Wgpu entrance and exit assumptions for cold initialization,
   uploads, retained request rebinding, shared compute and transfers. Real
   `native_host_read_boundary` calls after COPY/FILL still establish
   TRANSFER_WRITE -> HOST_READ visibility for mapped readback owners.
7. `Commands::submit` keeps the original final compute-queue fence submission
   waiting on the LAST transfer and shared-queue timeline values, even when no
   later compute consumer exists. No submit return grants completion. Partial
   failures keep the original recording/resources pinned.
8. Named and opportunistic warm replays use the original retained command buffers
   and exact resource/signature/nonce/family validation. Their encoded boundaries
   already derive from that same immutable plan. No replay-only barrier removal
   or separate unvalidated warm policy exists. Request erasure, IO reuse,
   actual completion and close are unchanged.

These checks establish the intended source-level dependency preservation. They
are not proof of a particular driver's behavior or a hardware speed improvement.
All original source identity, BDA admission, queue ownership, shader finite/math
and fail-closed recovery requirements still apply.

## CPU validation

`python3 apps/kuno/benchmarks/text/test-native-command-boundaries.py` extracts the
EXACT marked policy and its tests from the original modified encoder, compiles
with `rustc --test -D warnings`, and passes four tests:

- Single/main-only queue and external-boundary retention.
- Alternating copy/main/shared segments, releases and deterministic replay policy.
- Full32769-segment inventory, exactly one first/last boundary on every used queue.
- Empty, invalid-queue/header, invalid-index and excess-capacity rejection.

The same runner checks the actual encoder call sites and verifies that the
persistent per-queue hazard map, per-node accesses/memory barriers and both
host-read boundaries still surround the real complete node loop. It does not
construct an ash device or compile the complete backend. Formatting and Git
whitespace checks pass. Full worker build/Clippy and GPU gates are pending.

## Required root acceptance

Before speed testing, the integrated exact worker must pass original numerical
matrix/router/dequant controls and complete request/erasure/close/recovery tests.
Warm reuse, shared draft, actual COPY/FILL readback, cancellation/failure recovery
and private staged thinking/tool-result behavior need their existing appropriate
acceptance gates. Compare matched complete prompt durations only after those
pass. Preserve the installed worker until selection and installation are
explicitly accepted. This change is potentially relevant to any native graph
with multiple segments, including decode, but no decode gain is established.
