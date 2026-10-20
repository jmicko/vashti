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

Original private-I/O scope, actual reuse nonce when present, private mutation guard, error scopes, root
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

Large prompt prefixes legitimately use fresh private I/O, with no reuse nonce.
The batch's private token binds the actual I/O owner `Arc`, current ledger
epoch, checked per-owner active batch generation and exact `Option` of the
original reuse nonce. INFO packing, every compute/copy enqueue and closing
check that identity; foreign owners, post-erasure epochs, closed/older batches,
retired owners and fresh/reuse transitions reject. A fresh token grants no reuse,
readiness, completion or erasure credit. When an actual reuse lease exists, its
original nonce check and metadata ceiling also remain mandatory.

## Bounds and diagnostics

The complete wide-expert wave admits at most 5,632 original compute/copy nodes
(512 groups times eight physical compute pieces plus three HOST copies), and
8MiB of actual
command/argument metadata. Its whole command capacity is allocated before the
first insertion, so populated private command storage is not reallocated.
Deferred HOST-copy and INFO allocations have full-capacity wiping owners.
Backend tables admit at most 4,096 distinct original binding views and 32 exact
pipelines. The public `NATIVE_EXPERT_BATCH_HOST_BYTES` constant reserves 24MiB
for one active batch, including the original tape/arguments and simultaneous
backend owner/INFO tables. The feature adds that quote to the original prompt
`HOST_BYTES`, through `PROMPT_SOURCE_HOST` and the parent's original host
reservation, before model/payload allocation. Backend scratch must fit that
same bound before allocation; actual capacities are checked. Actual reuse also
charges scratch to its unchanged reusable-owner metadata ceiling. Fresh I/O
uses its own precharged closed-batch envelope and original fresh page ledger,
without borrowing a reuse lease or enlarging a reusable-cache ceiling.
Original graph retained metadata remains subject to its unchanged 64MiB ceiling.

The existing compiled-public profile receives backend-returned enqueue counts:
`expert_native_bulk_batches`, `expert_native_bulk_launches` and
`expert_native_bulk_copy_segments`. These are actual entry/command counts, never
completion receipts. Ordinary private workers have no additional persistent
counters/logging. Whole-model acceptance must establish one batch per expert
phase, six logical phases per original group and identical original HOST copy
counts. The wide wave reports its additional real matrix pieces separately.
Per-group `expert_cpu_*` measurements now cover collection; backend work remains
inside the enclosing expert enqueue timer. They must not independently be
presented as a reduction in complete expert work.

## Complete wide-prefix/tail grammar

The original path still has six logical phases and six real compute launches.
The wide wave keeps Gather, GU-dequant, DOWN-dequant (original release point),
GU product, activation and DOWN product. The connected miss-only ring adds
separately named, literal-role resident GU/DOWN image producers with identical
math and no raw-slot release. Their pair must match; normal streamed DOWN
still requires the original release. See [the complete conditional producer
and genuine-view contract](kuno-native-miss-ring.md). Each product is one original full-range
dispatch or one exact full 128-member prefix followed by its required authentic
tail. A role-bearing wide/tail entry cannot enter another phase, and no copy or
other kernel can split an incomplete pair. All phases must close before INFO
upload or native node admission. Pipeline preparation grants no acceptance.

The validator reads the original 72-byte legacy or 80-byte new generated INFO,
including real scalar padding and all six array buffer/logical lengths. The
seven new U32 values are columns, rows, routed-member count, weight first,
group, member base and piece members. The original Gather carries the *full
prompt* count; its real grid has exactly five workgroups per routed member, so
the validator derives routed count from that grid and preserves the prompt
header bound. Wide prefix is exactly `count /128 *128` starting at zero; a tail
is exactly the remaining 1..127 members. Its input retains the same allocator,
stream and root, with start increased by `prefix *columns *2` and excluded end
increased by `original_used -count *columns *2`. The other five views remain
identical. Full map/status/fault/output extents, real direct grids and original
legacy small/narrow/macro fallbacks remain checked.

Original vector ArrayArgs pass a vector count, and KernelLauncher subsequently
divides logical length by vec4 vectorization. The validator preserves the real
serialized buffer length (`used /8`) versus logical length (`used /32`), rather
than silently changing this existing argument behavior. The existing pure
`KernelLauncher::into_bindings` serializer is now public so CPU controls can
register genuine ArrayArg/scalar types without constructing a client/device;
the method body, INFO lease wiping and launch authority are unchanged.

The returned receipt reports actual physical `launches`, plus `logical_phases`,
`matrix_pieces`, `wide_prefixes`, `wide_blocks` and `tail_pieces`. Public totals
use the corresponding `expert_native_bulk_*` names. For G groups and T actual
tail launches, require logical phases=6G, matrix pieces=2G+T, and real launches
=6G+T. This never describes seven/eight launches as six. Original copy totals
must still match. Counters become available only after the complete original
batch has actually enqueued and its closing scope checks pass.

The authentic host-inventory control uses the original group builder/validator
on 4,096 prompt rows: 158 groups with 257 members and 354 with one member, exactly
40,960 routes. It includes all 85 possible GPU/HOST/shadow source views, actual
typed boxed kernel commands and original live Vec capacities. The fixed tape
has 5,632 slots, but this route-conserving maximum needs 3,900 entries and 4,924
physical nodes. The actual command size is 208 bytes; actual command/argument
inventory is 7,695,200 bytes, 693,408 below the unchanged 8MiB ceiling. Its
conservative backend charge is 10,256,352 bytes within the unchanged 24MiB parent
quote. Node overflow and oversized retained-capacity negatives still reject.
The 4,096-view, 32-pipeline, 64MiB Graph and actual reuse-scope limits are unchanged.

Source-only acceptance for this wave: the actual collector/grammar driver passes
six controls, including original seal/panic/capacity tests and exact complete,
missing-tail, wrong-role, wrong-schema/grid/member interval and all-six-owner
negative controls. Two App controls pass: actual typed ArrayArg/KernelLauncher
registration through scalar-small, narrow, macro and split-wide grammar, and the
original group-validator maximum-inventory proof above. The complete worker
typechecks with the original Intel/native32/paired/bulk feature set. Logs are
`native-expert-piece-grammar-controls.log`,
`native-expert-piece-budget-controls.log` and `native-expert-piece-check.log` in
the isolated evidence directory below. New generated kernel arithmetic, full
real-request logical/physical counts and recovery remain root integration gates;
no GPU, installed service or shared Cargo target was used from this worktree.

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

The first connected 9-feature GPU build passed the existing 103 arithmetic
controls and recovery, then correctly rejected the first full-model prefix:
the initial implementation incorrectly required a reuse nonce for fresh I/O.
That failed attempt establishes no completed-request performance result.
The explicit owner/epoch/generation/fresh-or-reuse repair typechecks. Its actual
scope controls reject foreign/stale/closed/retired owners, epoch changes,
fresh/reuse transitions and generation overflow. The unchanged owner controls
plus page and scope controls pass (124), original queue controls pass (33),
recording binding-owner controls pass (2), and original model-budget/admission
controls pass (32), including the connected parent reservation. The corresponding
`native-expert-fresh-scope-{check,owner-controls,queue-controls,retained-owner-controls,model-budget-controls}.log`
files are in the same isolated evidence directory. This repair still needs the
complete real fresh/reuse model, physical cleanup and public-counter gates.

The fresh-scope connected retry then reached the unchanged strict six-operator
gate and rejected the final dequant release marker. The actual producer order
was correct. `KernelMetadata for Box<dyn CubeTask>` forwarded name/ID/address
metadata but omitted `native_release_after`, so calls on the box returned the
default `false` rather than the original DOWN-dequant `true`. Both the Box and
KernelTask adapters now forward that original flag. This also restores the
intended release boundary/cache identity in the ordinary pipeline path; it does
not relax the batch sequence or change expert arithmetic. The same exact
six-slot predicate is shared with producer CPU coverage: boxed real Ops across
all supported quant pairs and actual generated small/macro/narrow expert type
identities, wrong release/adjacent-order/foreign-kernel negatives, and real
KernelTask-to-Box propagation. No GPU is constructed by those controls.
Both producer controls and the original release-cache identity control pass;
the complete worker typechecks. Evidence files are
`native-expert-box-metadata-controls.log`,
`native-expert-box-original-release-controls.log` and
`native-expert-box-metadata-check.log` in the same isolated directory. The
corrected full pipeline still needs actual model/physical acceptance.

The next connected attempt reached INFO packing and rejected authentic records
larger than the device's descriptor-offset alignment. That alignment constrains
record starts, not lengths. Page packing now uses the unchanged `InfoPolicy::plan`
for each original record, allowing records to span multiple alignment units.
It stops before a full page would wrap to offset zero, retains exact original
logical descriptor lengths and bytes, and still rejects empty/symbolic records,
binding-limit violations and records exceeding one complete fixed page. Complete
page initialization, original ledger owners and terminal erasure are unchanged.
CPU controls now mix actual CubeCL `InfoBuilder` six-array metadata plus the
original five runtime-U32-address scalar arguments (72 bytes, including their
padding) with original 64-byte custom Params across
4/8/16/32/64/256/512-byte alignments and both page
sizes, checking original planner offsets, byte identity, nonoverlap and all zero
padding. No timing is accepted from the preceding failed whole-model attempt.
Both real page controls and all 125 original I/O/page/scope controls pass;
the complete worker typechecks. The new evidence files are
`native-expert-real-info-page-controls.log`,
`native-expert-real-info-owner-controls.log` and
`native-expert-real-info-check.log` in the same isolated directory.

Actual GPU arithmetic, full model execution, public selection counts, injected
terminal recovery, clean timing, staged API/tool use, installation and physical
recovery remain the integration task's acceptance gates. No GPU or service was
used from this isolated development worktree. Unsupported backends, symbolic
branches or malformed batches reject; there is no silent ordinary fallback
inside an admitted bulk batch. Disabling the feature retains the original path.
