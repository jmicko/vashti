# Scoped CPU reads of the original public HOST mirror

## Scope and status

This is a source-only, default-unused bridge for one original DASLab expert
layer. It borrows the existing frozen public HOST mirror used by the GPU. It
does not reread or map the GGUF, copy weights, allocate another mirror, select
CPU execution in normal inference, merge routes, change expert arithmetic, or
claim a speed improvement. The [borrowed CPU batch leaf](kuno-native-borrowed-cpu-batch.md)
remains synchronous and returns unweighted original-rank partials.

Raw borrowed packed views authenticate geometry, not payload origin. This
parent joins those views to the actual original `Plan`, attached `Sources`,
registered external owners, model/graph identity and frozen bank generations.
Full artifact hashes and original loaded-byte validation remain prerequisites
of the existing unsafe source attachment; descriptor/stat checks are not a
replacement for them.

## Typed lifetime and source contract

`Attachment::borrow_cpu_layer(&mut self, prepared, reader, executor, workspace,
layer, cancel)` returns an opaque `CpuLayerBorrow`. The parent must first prove
the existing completed native projection boundary; a narrow model wrapper may
delegate using its actual MoE workspace. The guard borrows the Attachment
exclusively, the same PreparedPlan and the original positional reader. It also
retains shared borrows of the actual executor and workspace, preventing their
reset/fill or replacement while views remain live. It has
no Clone, Debug, serialization, mutable byte or raw pointer interface.

Creation requires all of the following before mapping:

- Attachment is Attached.
- PreparedPlan and Mirror refer to the same actual Plan identity and graph;
  the reader owns that original model and all three original layer tensors.
- The actual executor is frozen on that attached source identity and client.
  The actual workspace has frozen sources enabled and its existing source
  getter verifies the original graph, source identity and all actual bank
  instances/generations. The original Plan independently checks all 48 actual
  banks against its frozen snapshot. No fill, eviction or recency commit is
  allowed while the source is attached.
- All nine layer pages are still Registered. Every owner has the exact original
  role/page coordinate, public external binding, offset0/full logical length,
  original source allocation and matching retained binding shape/stream. The
  private producer-owned non-reused handle and resolved original Wgpu buffer
  prove actual registration identity; the runtime's opaque descriptor ID has
  no public equality interface and is not exposed. Actual Vulkan
  allocations are distinct; a memory object cannot be mapped twice.

The guard owns at most nine private read-only maps, three roles by three HOST
partitions. Each map is of the SAME retained VkDeviceMemory allocation used by
the registered GPU source. Its private pointer cannot escape. Logical spans
are derived by `Plan::address` and `Plan::range`, checked against the original
expert ID, tensor, dtype, row bytes, page, local slot and byte extent. Allocation
padding is never exposed as expert payload.

Guard operations are deliberately narrow:

- `weights()` returns the original normal `PackedExperts` metadata view. It
  performs no payload I/O and owns no weight bytes.
- `expert(id)` accepts only actual HOST experts with virtual slots144..511 and
  returns the leaf's immutable three-span `ExpertView` borrowing this guard.
- `expert_for_host_ordinal(0..368)` and `resident_expert(0..144)` derive IDs from
  the actual Plan for fixed public fixtures. IDs are transient selection data,
  never receipt fields. The former is checked against all three role ranges;
  the latter must classify back to the exact requested VRAM slot.
- `geometry()` exposes only the checked original grouped geometry. Actual
  HOST_CACHED/HOST_COHERENT placement is reported as public per-page metadata;
  neither property is assumed merely because the allocation is host visible.

The guard and its expert views are valid only while the exclusive Attachment
borrow is alive. An Arc of Sources or a CubeCL handle alone does not own or
authorize RAM access.

## Opaque original GPU partial control

`launch_original_gpu_partials(arena, spans, ids, groups, count, cancel)` accepts
the seven actual private arena IDs in native order
`[MAP, INPUT, ACT, PART, PROB, OUTPUT, STATUS]`. Count is1..3 and group count is
bounded by count*10. Complete MAX8/MAX80 owner spans, actual client/stream,
offsets, typed lengths and no-alias checks precede payload dispatch.

The method privately obtains the original registered source handles; it never
returns a handle or passes one to a caller callback. It calls only the existing
original-F32 `grouped_gate_up_frozen` and `grouped_down_frozen` wrappers, through
HOST partitions144/288/432 with lengths144/144/80. All three GU calls precede
the three DOWN calls on the same attached stream. Each kernel retains original
source addresses, full route group membership, F32 dot/reduction order, SiLU,
original rank destinations and status behavior. No probabilities are
multiplied and no resident expert work is performed in this partial-only arm.

The original immutable source pages may be read by both CPU workers and GPU
kernels. Source writes are forbidden throughout the attached lifetime. Existing
CPU initialization flushed noncoherent memory before publication; no GPU writes
occur and this guard performs no invalidate/flush/cache operation. Mapping or
unmapping changes neither allocation ownership nor the GPU binding.

Every GPU call attempts a join on the actual attached client before the method
returns; success requires that join. A final guard Drop also attempts to join
any armed GPU work, including errors or unwind. Failed join grants no completion
or release authority; existing Registered-page fail-safe retention and the
terminal close's own mandatory join prevent unproved detach/zero/free. A
read-only CPU unmap never frees the RAM or changes the GPU source binding. There
is no new global recovery latch or retry policy. CPU batches
must be joined before their input views and guard can drop; the existing
synchronous CPU leaf already provides this guarantee.

The private fixture owner remains responsible for original MAP/input packet
proof, finite/status/inactive/padding/full-owner checks and erasure. This bridge
does not authenticate arbitrary runtime route data or replace those checks.

## Capacity and close

Readonly-map and scoped header ownership is fixed, at most nine maps and one
guard. Exact typed sizes and temporary resource-check capacity are admitted
inside the existing two-MiB planner/mirror and 256-KiB Attachment metadata
reservations before mapping. Those caps, public RAM envelope, registry budget
and private CPU/GPU owner budgets are unchanged. The bridge allocates no model
payload or new source buffer. Callers independently reserve the CPU batch's
original complete worker/input/activation/output envelope and every GPU private
owner before dispatch.

Normal readonly guards unmap only after joined consumers. The exclusive borrow
ends before the existing authenticated whole-model close. Terminal close still
requires original complete model proof, private-zero close, all consumers
dropped, exact detach receipts, joined device, full allocation zero/audit,
flush and free. A successful CPU read is not a zero/free receipt.

## Acceptance gates and limitations

Source tests must cover exact logical bounds excluding padding, HOST-only
classification, malformed role/page/range and repeated map rejection,
registered binding/phase requirements, exact unchanged metadata caps, same
Plan/source identity, no public source handle interface and fail-safe joined
drop ordering. They establish no mapped-device acceptance.

The first physical operator gate uses one fixed original DASLab layer and must
compare CPU borrowed original-F32 partials with original GPU HOST partials
reading the SAME mirror pages. The CPU leaf's source fixture suite covers all
six formats; this first mapped-device gate does not. Use counts1/2/3 and HOST
rank subsets1/3/5/10, with
repeated experts across members but unique ranks within each member. Retain
ten original route ranks/member, filling the other ranks from the actual
resident map; both partial-only arms skip resident work identically. Read back
full owning inputs, outputs, fault/status and inactive/padding regions, and
include complete GPU-input handoff, CPU joined work, full sparse partial upload,
finite check and actual GPU join in the CPU timing. Preserve original numerical
tolerances and closure/recovery requirements. Report actual memory placement.

This gate does not prove count3 hybrid merge, GPU/CPU overlap or whole-model
speed. Model integration would later require original-rank probability merge,
unchanged full original CPU13 absolute0.006 + relative0.0003, whole vocabulary
and hidden vectors, actual experts, real speculative accept/repair, full owning
fault/padding/source epochs and physical memory recovery. No runtime selector,
service/API/frontend, installation or release change is part of this bridge.

## Source receipt (2026-10-11)

The bridge is implemented in the isolated `codex/kuno-cpu-host-borrow` worktree
on root `2e51bd5`, with the controller-owned executor authentication helper
`5b16fd4` applied as `94e41d4`. Existing pre-attachment accessors are unchanged.
There is no model/operator selector or standalone fixture registration in this
leaf; the parent task owns its narrow completed-boundary model wrapper and
physical operator controller.

All **25 attachment/mirror/plan source tests pass**, including six additive
bridge checks. The new tests cover exact role/page indexing, unchanged metadata
caps, Registered-only CPU maps, all nine memory identities before any map,
logical payload/padding/overflow boundaries and exact original resource
identity/extents. Existing Plan tests retain original tensor/format/range,
complete complement, opaque origin/generation, exact mapping and budget checks.
These are metadata/source tests; they do not instantiate Vulkan maps or prove
runtime physical registration.

The locked native laboratory's all-target pure `direct-device` Clippy with
warnings denied passes with `-j2`; formatting and whitespace checks pass. The
readonly bridge and opaque original GPU launcher compile. Source checks used
the released NVMe runtime-review cache; no GPU, sudo, service or installed
binary changes were made.

```text
cargo test --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml -j2 --bin kuno-text-lab --features direct-device public_host_expert_ -- --test-threads=2
cargo clippy --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml -j2 --all-targets --features direct-device -- -D warnings
```

Actual same-allocation CPU/GPU output and timing, mapped-memory placement,
complete owning runtime checks, terminal detach/zero/free and physical memory
recovery remain untested in this leaf. Root owns those gates. No CPU/GPU or
whole-model speed acceptance is claimed.
