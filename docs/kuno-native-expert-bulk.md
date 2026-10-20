# Scoped native expert bulk candidate

This is an opt-in backend candidate (`strata-native-expert-bulk`), not a measured
speed improvement or a new application/API contract. It uses the original model,
quantization, six expert operators, output/fault arithmetic and complete prompt.

The accepted public CPU sample in
`strata-native-fast-cpu-readyfix-20261020/receipt.json` contains 26,338 expert
groups, 158,028 compute launches and 39,225 HOST copy segments. Its warm prompt
has 3.226s of sampled user CPU spread across planning, allocations, resource
resolution, copies, dispatch and other work. Descriptor batching alone was a
small improvement; this change targets the complete repeated launch path.
The sample does not prove that all memset time is unnecessary initialization,
or that all sampled CPU time can be removed from wall time.

## Connected execution

`ComputeClient::with_native_strata_expert_batch` lends an explicit scoped client
to the existing expert function. Generated CubeCL matrix argument/INFO packing
is unchanged. The original Gather, both dequantizations, GU product, activation
and DOWN product all use that client. `PromptLayer::stage_group_on` sends the
checked HOST copies to the same sink; using `Sources`' original client instead
would incorrectly let copies escape the deferred ring order.

The sink has no TLS/global interceptor. It keeps original `Binding` owners and
INFO until the callback ends. It seals escaped clones on success, errors and
panic. Callback failure drops the unsubmitted commands. Only the closed expert
launch/copy family may use this unsafe internal callback; reads, synchronization
and unrelated mutation must not interleave. Ordinary clients are unchanged.

The WGPU backend revalidates the actual command inventory and capacities. It
resolves each distinct original allocation/view/stream once through the original
memory manager. It reuses only exact original pipeline identities, including
execution mode, release boundary and PUBLIC external-read mask. Every original
physical read dependency is retained. It admits the same six-operator sequence
and interleaved copy order into the existing native graph.

Original current reuse nonce, private mutation guard, error scopes, root
initialization, descriptor extents, graph admission/draft/range checks, planner,
hazards, queues, rolling recording and actual completion fences remain. The
same `Arc<Vec<Binding>>` accompanies both compute and transfer nodes and every
recording that references it. Root's owner-retention change retains and charges
that complete original allocator inventory through success, splits, recording
failure and unknown-completion quarantine. A `WgpuResource`/VkBuffer alone is
never substituted for allocator ownership.

INFO is a closed non-symbolic expert path. Original byte records are packed into
full zeroed pages using the existing `InfoPolicy` and `batched_uniform` ledger
allocation/upload. Each node still receives its exact original logical INFO
length. Original full page owners, mapped-source sealing, initialization, nonce,
epoch, copy-before-use and terminal erasure/audit remain. No prompt/model cache
or byte content survives the existing request lifetime.

## Bounds and diagnostics

The tape admits at most 4,608 original compute/copy nodes and 8MiB of actual
command/argument metadata. Its whole command capacity is allocated before the
first insertion, so populated private command storage is not reallocated.
Deferred HOST-copy and INFO allocations have full-capacity wiping owners.
Backend tables admit at most 4,096 distinct original binding views and 32 exact
pipelines. Conservative simultaneous table/INFO scratch is charged to the
original I/O metadata ceiling before allocation; actual capacities are checked.
Original graph retained metadata remains subject to its unchanged 64MiB ceiling.

The existing compiled-public profile receives backend-returned enqueue counts:
`expert_native_bulk_batches`, `expert_native_bulk_launches` and
`expert_native_bulk_copy_segments`. These are actual entry/command counts, never
completion receipts. Ordinary private workers have no additional persistent
counters/logging. Whole-model acceptance must establish one batch per expert
phase, six launches per original group and identical original HOST copy counts.
Per-group `expert_cpu_*` measurements now cover collection; backend work remains
inside the enclosing expert enqueue timer. They must not independently be
presented as a reduction in complete expert work.

## Validation boundary

The connected worker initially typechecked with Intel, native32, paired QSA and
bulk enabled. Focused CPU controls exercise the real sink seal/panic/escaped
clone, original stream, capacity and remeasured malformed inventory, plus actual
INFO page bytes/logical lengths/padding against the original planner. Original
I/O ownership, queue and recording-owner controls are retained.

Completed isolated CPU validation (2026-10-20): the complete worker typecheck
passes; all original 121 I/O owner controls plus the actual new page control pass
(122), original queue controls pass (33), original recording binding-owner
controls pass (2), and actual collector source controls pass (3). The collector
driver includes the production module by path with its real runtime/common types;
no fake device or GPU was constructed. Cargo cannot directly run dependency-only
vendor crate tests from the application workspace, so this small driver lives
outside the repository under `native-expert-bulk-cpu-controls-20261020`.

All evidence is under
`/media/jmicko/faststore/kuno-development/prefill-cubek-cpu-20261020/`:
`native-expert-bulk-check.log`, `native-expert-bulk-owner-controls.log`,
`native-expert-bulk-queue-controls.log`,
`native-expert-bulk-retained-owner-controls.log` and
`native-expert-bulk-collector-controls.log`. Full-worker tests used
`--features strata-intel-core,strata-gpu-adaptive-usage,strata-native-wide32,strata-qsa-paired,strata-native-expert-bulk`
with the isolated target and two compilation jobs.

Actual GPU arithmetic, full model execution, public selection counts, injected
terminal recovery, clean timing, staged API/tool use, installation and physical
recovery remain the integration task's acceptance gates. No GPU or service was
used from this isolated development worktree. Unsupported backends, symbolic
branches or malformed batches reject; there is no silent ordinary fallback
inside an admitted bulk batch. Disabling the feature retains the original path.
