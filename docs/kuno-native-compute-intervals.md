# Complete-view compute synchronization

Kuno-only internal candidate after45795611. Preserve the complete connected
pipeline. No numerical, source binding/view, parent admission, network, model,
consumer or installation contract changes.

The existing compute encoder records one previous access mask per whole buffer.
It therefore inserts a WHOLE_SIZE barrier even for disjoint original views.
The coherent successor uses exact COMPLETE declared intervals on each original
queue, while retaining the existing global queue Plan, every cross-queue wait,
release boundary, command segment, actual submission and physical owner.

Before any prefix runs, build bounded per-node compute barrier lists from the
same original resources/BDA dependencies/indirect views and actual access masks.
Check current aliases against previous nodes BEFORE updating any interval;
union read/write aliases within the current node. Distinct gaps remain gaps.
A barrier is needed only for a genuine overlapping access with a writer.
For a genuine hazard, retain WHOLE_SIZE and the original complete supported
COMPUTE/DRAW_INDIRECT/TRANSFER stages. Its source mask includes ALL preceding
access classes on that root; destination covers ALL six supported native
access classes (shader read/write, uniform read, indirect read, transfer
read/write). This conservative full-root dependency permits retiring the old
same-queue interval history. Future accesses cannot lose an older dependency
outside the triggering overlap. No bare queue ordering or CPU fence is treated
as a memory dependency. Dedicated COPY nodes retain the previous independently
validated transfer plan and barriers, untouched. Host-read boundaries remain.

All actual Plan arrays, simultaneous temporary region/hash capacities and
scratch, reallocation peaks, source signature/owners/keys and existing live
Recordings are preflighted inside the original64MiB Graph allowance. Keep the
original32768-node/65536-root/262144-region bounds and a262144-dependency cap;
reject overflow before submission. No narrower source or changed GPU budget.

Anonymous fixed-public counters compare the exact original whole-buffer
algorithm's hypothetical barrier count to the actual encoded interval count,
with real compute node counts. These are not timings or completion grants.
Only the existing compiled-public collector publishes them after real sync.
No private prompt/route trace or persistent request metric is introduced.

Initial actual-source CPU controls cover8192 randomized nodes against an
independent byte-addressed three-queue oracle, aliases/gaps/mixed access classes,
and invalid/overflow inventories. These do not establish GPU correctness.
Require integrated original CPU/129 GPU/offline/full request/count/recovery
gates, then a meaningful clean gain before full staged/terminal acceptance or
installation. Never weaken numerical or ownership checks to accept a speedup.

The connected actual App compiles.54 CPU controls pass: initial three interval
controls,37 original queue/planning/rolling/recovery, four transfer-visibility,
three Recording owners, one scratch capacity, two serializer/budget, two real
KernelTask/Box release and two profiler controls. The final interval suite adds
an allocation-capacity failure check that cannot realloc or alter the old span;
all four interval controls pass. The8192-node oracle additionally checks the
anonymous legacy count against an independent copy of the original whole-root
algorithm. Interval insertion itself rejects a missing precharged slot before
Vec can reallocate. Exact logs: `native-compute-view-final-cpu-20261020.log`
and `native-compute-view-bounds-cpu-20261020.log` under the artifact base.
Together these establish55 distinct current CPU controls; original numerical
GPU and whole-model gates remain pending. No speed or installation claim.

## Frozen interval result and coupled atomic-only successor

Source `f1e09df78ddeedaa556c8fe8d2dda3d5cb1a9d28`, worker SHA256
`e705298b6e8e30fbc3c534c2805a2b9c95eaaf375e59506eb58fb21e2540484a`,
build `strata-parity-compute-views-20261020`, passes release compilation,
408/208 offline layouts, all129 original numerical GPU controls and complete
requests/producer counts/physical recovery. Repeated4245 prefix12.33719s
versus11.54345s, capture2.13977s, wait4.16856s, expert wall8.07530s,
enqueue1.58562s. It encodes404,498 compute barrier entries versus431,700 for
the original whole-root algorithm on164,182 real compute nodes: only6.3% fewer
entries. This is slower, unselected and uninstalled. No clean/staged/terminal
acceptance is claimed. Recovery snapshot14:54:47UTC confirms Intel50,688,000B,
NVIDIA compute0, all faults0, three services200 and normal admission.
Receipts `strata-parity-compute-views-20261020-{operators,profile}` and owned
`strata-parity-compute-views-public-summary-20261020.json` retain exact evidence.

Keep the whole connected implementation. A shared fault buffer is declared
read/write for every atomic OR, keeping many apparent dependencies even when
all actual operations only accumulate error bits. The successor couples the
complete-view alias tracker with a conservative proof over the ACTUAL FINAL
SPIR-V used by each retained Program, including native matrix and physical
expert lowering. It never changes shader instructions, arithmetic or faults.

The proof accepts only unused-result unsigned32 AtomicOr at actual Device scope
on an actual original read/write STORAGE_BUFFER descriptor. Direct access
chains and pointer copies are followed; all pointer uses must be accounted for.
Ordinary/atomic loads, stores/clears, other atomics, used return values, pointer
escapes/calls/unsupported aliases, ambiguous bindings, non-Device scope, large
or malformed/unsupported instruction inventory decline proof and keep ordinary
barriers. No shader name, source comment, static caller bit or presumed error-free
input grants this property. The original validated-module precondition remains;
this bounded scanner does not replace SPIR-V validation. Its zero-allocation
pointer walk is bounded to262144 words/IDs and256 pointer aliases. A constant
Device scope may use signed or unsigned32 integers (Naga emits signed scope
constants); the atomic value/result itself must remain unsigned32.

Original declared views/access flags and the full queue Plan/waits/releases
remain unchanged. ONLY same-queue overlapping accesses that are BOTH genuinely
proven OR-only may omit their mutual memory dependency. Aliases union their
actual access classes: a normal read/write through any descriptor or complete
physical-read dependency prevents the exemption on its overlapping bytes.
Initial clears, ordinary reads, used atomic results and final fault readback
retain their actual dependencies. A real barrier keeps full supported source/
destination classes and original fences/unknown-flight quarantine. Anonymous
compiled-public aggregate counts also record actual proven views, never private
prompt or route data. The guard bitmask adds only fixed Program metadata; actual
access/region/container sizes remain charged to the original64MiB Graph budget.

Actual App compilation and60 CPU controls pass: three actual lowered-SPIR-V
proof controls, six interval controls (including2048 mixed-class nodes against
an independent three-queue byte oracle,8192 original nodes, clear/read/alias
negatives and precharged-capacity failure), and all previous51 queue/transfer/
owner/scratch/serializer/release/profiler controls. Scope/type, consumed-value,
other-atomic, truncation/duplicate and signed-result negatives remain strict.
Initial compile caught test-only enum dependency/new field integration mistakes;
a positive proof control caught Naga's signed scope constant and was repaired
against its actual writer implementation before acceptance. The direct rspirv
entry is test-only and reuses the already existing runtime package/version.
Exact log `native-atomic-or-complete-cpu-20261020.log`. No GPU or speed claim
for the atomic successor yet; require original129 GPU/full-request/counts/
physical recovery, then meaningful clean improvement and full staged/terminal
acceptance before installation.

## Complete atomic candidate result (2b32bbea)

Frozen source `2b32bbeabc8ad5ac35645fbc01ff8a24652edf84`, worker SHA256
`9452beed3c50bfb236fb6aed399865311ae62b19d8b260e9dea927d1266233dd`,
build `strata-parity-atomic-or-20261020`, passes release compilation,
408 ordinary/208 matrix offline layouts, all129 original GPU numerical controls,
three complete requests/actual bulk/wide/tail/miss-ring counts and physical
recovery. The additional `atomic-or-validation.json` passes with actual
proven OR views162,742 on164,182 compute nodes for each long request. Actual
barrier entries241,948 vs431,700 original whole-root entries (44% fewer).
This is a count of BUFFER BARRIER ENTRIES, NOT hardware flushes or barrier calls.

Load78.47156s; first/repeated4245 prefixes13.08708/12.27180s versus preceding
13.20344/12.33719s: essentially unchanged. Repeated capture2.13418s,
enqueue1.58682s, wait4.17544s, expert wall8.07705s. The whole connected candidate
is correct but slower than the earlier complete guard bundle's profiled11.54345s;
no clean A/B, staged/terminal acceptance, sustained gain or installation is
claimed. The installed accepted3fffc266/SHAeac78ce0 remains selected.

Exact receipts `strata-parity-atomic-or-20261020-{operators,profile}` include
original `miss-ring-validation.json`, added `atomic-or-validation.json` and
owned `strata-parity-atomic-or-public-summary-20261020.json`. Final snapshot
15:25:53UTC: Intel50,688,000B, NVIDIA compute0, no new GPU faults, all three
HTTP services200, no active/paused/waiting/recovery admission. No reset/reboot.

## Next coherent source proposal: sealed access-class manifest

Retain the full pipeline and all proofs above. The conservative interval
algorithm retires old full-root history only because each real WHOLE_SIZE
barrier has destination flags for ALL six supported access classes. That can
force unrelated cache/indirect/uniform/transfer synchronization. Removing entry
structs alone does not establish fewer actual pipeline stalls; most math nodes
still have real dependencies. The mask cost is a SOURCE HYPOTHESIS, not yet a
measured hardware cause.

A coherent successor should preflight a SEALED actual access-class union for
each genuine VkBuffer on each original physical queue, over the complete same
immutable original Node/Plan inventory BEFORE any prefix. Derive it from every
original descriptor access, complete physical read dependency and indirect
resource, never BufferUsages guesses, shader names or caller hints. Consume the
manifest immutably; validate every real node class/queue against it, plus exact
original node counts before accepting the plan. Real barriers may then use that
complete same-queue future-class union instead of all six classes. The union
must cover EVERY later access in this recording, preserving the premise that
the old full-root history can retire. Full original ALL_COMMANDS external
memory boundaries and ALL actual cross-queue waits/releases remain necessary
for accesses outside this sealed recording; do not optimize them away.

Simply changing destination to this one node's mask and clearing old history
is WRONG: an older write in untouched bytes may later be read through a different
access class. Either retain a full typed visibility history, or use the complete
sealed manifest above. Preserve the current correct broad fallback if proof is
absent. All manifest/temporary/hash capacities and simultaneous original Plan,
owners, keys, barriers and Recordings must fit the unchanged64MiB allowance.
Avoid adding an unmeasured expensive per-node traversal when existing actual
inventory passes can build the manifest; no narrowed resource view is allowed.

Add actual barrier-CALL counts and exact legacy call counts, separately from
entry counts, to the compiled-public aggregate proof. Validate complete-manifest
membership/unknown-future-class/queue negatives and typed byte oracle controls,
then original129 GPU/full requests/recovery. Only meaningful matched clean
improvement justifies full staged/terminal acceptance and installation. If this
still does not reduce wall cost, the other concrete CPU lead is immutable shared
complete BDA dependency bundles: current per-node Vec clones/packing still
repeat the same full layer inventory. Keep genuine handles/epochs/loan checks;
never substitute fake/narrow dependency declarations.

## Connected sealed manifest, global scopes and execution-only WAR

The next complete implementation retains every preceding pipeline component.
During the ORIGINAL planning pass, one checked binding visit fills both the
original ordered write/read intervals and exact typed compute accesses. After
pushing that node, the actual Planner's read-only `last_segment()` supplies its
real queue; no guessed queue mapping or extra full binding traversal is added.
A bounded ClassBuilder unions real access classes per actual VkBuffer/queue.
The consumed sealed manifest includes the original node count. Every actual
compute node is checked against its sealed root/class membership before a plan
can finish and BEFORE any prefix. Unknown future classes/roots/queues fail closed.

The masks are moved once into the genuine per-root hazard table, with old/new
hash allocation peaks charged. They never mutate there. There is no repeated
separate manifest lookup per dependency. Builder capacity is capped at1MiB
within the unchanged64MiB Graph allowance. Unavailable optional builder evidence
or initial cached-manifest allocation retains the correct broad policy; no
narrow mask is guessed. Existing hard preflight/admission limits still apply.
After complete barrier precomputation the class/root temporaries drop; actual
encoded dependency masks remain in the immutable plan and command buffers.
All original signature/owners/INFO/replay/unknown-flight/erasure checks remain.

A genuine data hazard uses the sealed COMPLETE future-class union, not this
one node's mask. The same original supported stage masks, full external
ALL_COMMANDS memory boundaries and exact cross-queue waits/releases remain.
Retiring full-root history therefore preserves future typed reads in untouched
bytes. Broad fallback still covers all six supported classes. No declared
resource view is narrowed or indirect/physical source dependency omitted.

Two coupled ordinary Vulkan optimizations are included:

- A root with ONLY preceding reads needs execution ordering before a writer,
  with zero memory access masks. An outstanding writer ANYWHERE on the complete
  root prevents this exemption, even when the triggering overlap is read-only.
  Atomic OR read/write states remain writers. Original stage ordering and the
  actual barrier call stay; there is no false GPU completion or early slot reuse.
- Required compute dependencies at the same original node boundary become ONE
  global memory barrier using the union of required source/destination masks.
  This broadens memory resource scope and preserves all original CPU-validated
  views and real lifetime owners; it removes per-buffer driver bookkeeping.
  A pure execution-only boundary emits an actual zero-access barrier. An empty
  dependency list emits none. Dedicated COPY barriers, per-target host-read
  boundaries, actual queue/family policies and command boundaries stay unchanged.

Primary semantics/reference:
[Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html),
compute-to-compute read-then-write and global-vs-per-resource examples. The guide
confirms execution ordering suffices for WAR and recommends global memory
barriers for ordinary compute buffer dependencies. No shader arithmetic or
instruction words change here.

Compiled-public counters now distinguish original/actual internal compute
barrier CALLS, buffer dependency ENTRIES, entries narrowed by the sealed manifest,
and execution-only entries. Counts are not hardware flush or completion proof.
No private request trace or new network/API/configuration/consumer field exists.

The actual combined App compiles and65 CPU controls pass: three final-SPIR-V
proof,11 interval/manifest/mask controls,37 original queue/planning/rolling/
recovery, four transfer, three owner, one scratch, two actual serializer/budget,
two real KernelTask/Box release and two profiler controls. New cases cover
2048 mixed nodes against the independent three-queue byte oracle with sealed
classes, future indirect reads in untouched bytes, wrong future class/root/queue/
count/capacity, pure WAR versus disjoint pending writers, and the ACTUAL global
mask producer including execution-only calls. The original8192-node and mixed
alias oracles remain. Log `native-sealed-global-complete-cpu-20261020.log`.
An integration compile caught an incomplete Option conversion for fallback;
this was fixed before final CPU acceptance. No numerical GPU tolerance/reference
was changed. Frozen release/offline/original129 GPU/full request/count/recovery
and performance results are pending; no installation or speed claim yet.


## Complete sealed/global candidate result (065815af)

Source065815af9f7e1bfe3ed29b5663f669a09fec7bec / worker SHA256
21446b898fcf8c7778ff610b544c0810b474273c437a50c724d95be0658b24f7,
`strata-parity-sealed-global-20261020`, passes65 actual App CPU controls,
408 ordinary/208 matrix offline layouts, all129 unchanged original numerical
GPU controls, three complete requests, actual bulk/wide/tail/miss-ring/OR and
new synchronization-path proofs, graceful exit and actual physical recovery.

Load82.80644s, first/repeated4245 prefix12.69377/11.90766s. Repeated capture
2.18639s, enqueue1.59215s, wait3.77171s, experts7.73099s, attention2.93840s.
The preceding development repeat12.27180s improves modestly; installed clean
11.14739s and Strata5.3515s remain faster. This is NOT a clean comparison,
full staged/terminal acceptance, sustained decode result or installation claim.

Actual long counts:164,182 compute nodes,431,700 original buffer entries,
241,948 actual entries,189,272 sealed-mask data entries and52,676 execution-only
entries. Original internal calls163,798 versus actual162,838: fewer entries
has barely reduced call boundaries. Counters prove actual encoded work, not
hardware flushes. Original root/data/queue/owner protections remain intact.
Receipts `strata-parity-sealed-global-20261020-{operators,profile}`, profile
`sealed-global-validation.json`, owned `strata-parity-sealed-global-public-summary-20261020.json`.
Physical recovery16:03:46UTC: Intel50,688,000B, NVIDIA compute0, faults0,
all three endpoints200 and normal inactive/unpaused admission/waiting0.
Installed source3fffc266 stays selected. No API or Vashti changes.

Next connected candidate: native command-buffer-owned push descriptors on
actually supported/enabled Vulkan devices. Strata uses USM kernel parameters;
Kuno currently allocates/updates/binds one fresh descriptor set per original
compute node. Earlier leaf sampling includes driver memset/allocation costs,
though this is not proof of a dominant cause. Preserve all existing complete
pipeline components. Enable KHR_push_descriptor only for the explicit compiled
native candidate; query the actual maxPushDescriptors. Each real native pipeline
uses a push layout only if its complete original descriptor count fits; all
others keep the existing fresh-set path. Fully write every descriptor before
EVERY dispatch, including after command-buffer/queue boundaries, with identical
raw buffer, offset, size, type, actual INFO epoch and Binding owners. Command
buffers own their copied descriptor state through the original real fences,
rolling recovery/quarantine, erasure and replay signatures. No descriptor is
mutated after use. Keep full BDA declarations and all synchronization. Reserve
fixed stack packing within the original64MiB metadata cap before any prefix.
Count actual pushed/fresh dispatches in the compiled-public bridge. Verify the
actual write packer, unsupported/count fallback, original129 controls, complete
requests/recovery and only then timings; no gain is assumed.
Primary contract: [Khronos push descriptors](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDescriptorSet.html)
and [actual device limit](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDevicePushDescriptorProperties.html).


The complete push-descriptor candidate is now implemented behind explicit
`strata-native-push-descriptors` / builder `--push-descriptors`. Private worker
startup first scrubs inherited selectors, then enables only its compiled policy.
Only the selected native device requests the actually supported KHR extension.
Graph queries maxPushDescriptors, selects each genuine pipeline layout, writes
all original bindings before every dispatch, and uses the unchanged fresh-set
path otherwise. All-push recordings keep a tiny valid pool for original ownership;
mixed recordings size pools for actual fresh nodes. Full resources, BDA reads,
signatures, INFO/allocator owners, original queues and recovery stay intact.
Fixed32-entry descriptor arrays are charged before any prefix inside64MiB.
Actual pushed/fresh dispatch counters join the compiled-public post-sync bridge.

All73 distinct actual App CPU controls pass (`native-push-descriptor-combined-cpu-20261020.log`):
original65 plus three push-selector/real-write-packer/malformed-view controls and
five original fresh-window/writes/budget/order controls, now included directly.
An initial compile caught a usize/u64 counter mismatch and was corrected. The
newly included original fresh-window source-order witness had a stale delimiter
from the older whole-buffer encoder; it now ends at the real post-window Copy
selection and keeps all original ordering assertions. No numerical reference,
threshold or recovery condition was relaxed. Release/offline/original129 GPU/
full request/count/recovery and measured results remain pending. Not installed.


## Complete push-descriptor result (9dc5786a)

Source9dc5786abed869cc1201ece297b6c3fe9cf03b9f, worker SHA256
85b9461bd87283bbc8f580cb2671cc22b047afd9b06fa084b9a43eac29766ed6,
`strata-parity-push-descriptors-20261020`, passes73 actual App CPU controls,
408/208 offline layouts, all129 original numerical GPU controls, complete
original-weight requests, all original bulk/wide/tail/miss-ring/OR/manifest
proofs and actual new push counters, graceful exit and physical recovery.
Actual164,182 long expert compute dispatches ALL use command-owned push
bindings; fresh-set dispatches0. This is a real selected path, not an environment
variable or preflight-only result. Fallback selector/fresh-window paths have CPU
coverage; no unsupported physical GPU's fallback has been newly exercised.

Load118.87836s (regressed/no startup gain), first/repeat4245 prefix13.17020 /
11.83052s. Repeated expert CPU enqueue1.53346s, capture2.06352s, wait3.87759s;
experts7.65247s, attention2.93633s. Versus connected sealed/global11.90766s,
this is unchanged within ordinary variation; no speed win or installed clean
comparison is claimed. All preceding connected pieces remain retained, not
partially reverted. Installed clean11.14739s and Strata5.3515s still lead.
No new clean/staged/terminal acceptance or sustained decode claim; NOT installed.

Receipts `strata-parity-push-descriptors-20261020-{operators,profile}`, profile
`push-descriptor-validation.json` and owned
`strata-parity-push-descriptors-public-summary-20261020.json`.
Physical16:19:39UTC: Intel50,688,000B, NVIDIA compute0, all faults0, endpoints200,
admission inactive/unpaused/no recovery/waiting0. Installed worker/gateway
SHAeac78ce0/d3ed07f4 reverified unchanged. No API or Vashti changes.
Next is a protected fixed-public leaf-CPU diagnostic on THIS complete bundle,
using original DSO-aware helper and raw-data deletion; no additional performance
claim from instrumented sampling. Use actual symbols to distinguish the remaining
allocation, immutable binding/dependency packing, planner and driver costs.


## Complete push-bundle CPU diagnostic and next coherent host work

The protected DSO-aware fixed-public diagnostic on source9dc5786a/SHA85b9461b
passes complete requests, raw-record deletion and actual physical recovery.
Receipt `strata-native-push-descriptors-dso-cpu-20261020/receipt.json`; owned
`strata-native-push-descriptors-dso-cpu-public-summary-20261020.json`.
All726,548 raw recording bytes were deleted; no private requests, callchains,
registers, stack copies or memory samples were collected. Only fixed-public
aggregate symbols/time windows persist. Instrumented repeated4245 prefix11.87723s
is a diagnostic, NOT a new clean comparison or sustained inference result.

Repeated prompt samples3.48241s user CPU (693 samples): libc memset0.46231s,
actual Planner::push_accesses0.25628s, fill_compute_ranges closure0.14070s,
malloc0.10050s, libc statx0.09548s, actual compute Recorder::push0.09045s,
memmove0.08543s, real guarded native-buffer lookup0.07538s, hash lookup0.07035s.
These are leaf samples, not inclusive/wall times. In particular memset/statx
callers are not known from this sample; do NOT invent a filesystem-per-kernel,
privacy failure or driver-only explanation. Allocation/planning/binding work is
still a distributed category; there is no single proven dominant function.

Physical recovery16:23:02UTC: Intel50,688,000B, NVIDIA compute0, faults0,
all endpoints200 and admission inactive/unpaused/no recovery/waiting0. Both
completed development candidates remain connected, unselected and uninstalled.
Installed worker3fffc266/SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 remain.

Next source investigation should combine complete immutable binding metadata
reuse across ORIGINAL capture passes with immutable full BDA source bundles,
not another isolated change of shader arithmetic or queue guesses. Current
signature, original Planner, class manifest, compute interval pass and descriptor
encoder repeatedly materialize/look up the same complete original views. A
capture-owned sealed inventory could reuse actual raw identity/view/type/access
facts while retaining every original field/signature, full dependency, original
queue-selected Plan and all ownership/INFO/flight/erasure checks. It must be
built from actual immutable nodes, check counts/membership before any prefix,
revalidate actual raw roots under EACH real reacquired Wgpu destruction guard,
and never retain an unguarded handle without revalidation. No stale source or
same-bytes identity substitute is allowed. Actual allocation/peak capacities,
old/new tables, complete source bundles and wipes must remain within64MiB;
if optional proof is unavailable keep the current correct path. Sharing a full
BDA bundle must preserve every original binding, view, stream and publication
lifetime, not narrow source declarations or omit admission checks. This remains
an investigation/proposal, NOT an implemented or measured gain. Use coarse
compiled-public subphase timing if needed to choose the connected changes;
never publish private metadata. Keep all current connected pipeline pieces.

After implementation: original CPU/oracle/budget/lifecycle controls, full129
unchanged numerical GPU controls, complete real requests and actual producer
counts plus physical recovery. Only a meaningful matched CLEAN gain justifies
terminal recovery/full staged thinking/tools/MAX8/semantic/privacy acceptance
and the official installation transaction. Installed clean11.14739s versus
Strata5.3515s remains authoritative. The parity goal is active and unmet.


## Connected original-signature inventory and full source bundles

The complete next host candidate is implemented. It indexes the SAME original
replay signature allocation: that already contains every original descriptor,
full physical read and indirect buffer handle/offset/size. It does not make a
second full access image or change any signature field. A capture-only index
validates exact original node count, complete schema, full root membership and
bounds and all used capacity before any prefix. It is bound to the actual
signature allocation identity/length, not an equal-byte clone. Optional evidence
or allocation unavailable means the unchanged original view-lookup path.

The snapshot additionally borrows the SAME actual Keys owner. Every later
capture guard revalidates that exact owner, every actual live VkBuffer and its
physical size under the real Wgpu destruction lock. Only a reader borrowing
that guard can access the cached original view facts. Original signature/node
inventory is immutable through these passes and moves intact into Recording.
Planner, class collection, compute intervals, COPY visibility and native
push/fresh descriptors/copies/indirect encoding reuse the same complete views.
Access flags and OR proof still come from the original actual Program; original
queue selection, all full source declarations, bounds, owners and waits stay.
Each encoder node is decoded once, after closing/issuing a prior prefix and
preparing the fresh window. Readers drop before ready mutation, original guard
release, node drops, Wgpu polling, cleanup or erasure. The capture-only index
and root metadata are wiped before Keys drop; no handle cache crosses a request.

Index/root/sort peak is capped1MiB, charged inside the original64MiB alongside
the unchanged signature, original owners, retained Recordings, Plan, compute/
transfer state, descriptors and all scratch. Actual allocation capacities are
checked. Full-capacity volatile word/byte erasure covers unused metadata too.
Current hard admission/preflight limits still apply; this is not a claim that
all metadata-pressure configurations have been live exercised.

Coupled full-read sharing remains confined to ONE real original expert batch.
The existing binding table first validates/resolves every original binding and
retains its genuine allocator owner. Identical ORDERED exact table-index lists
can share an immutable Arc<Vec<WgpuResource>>; order, duplicates, every original
view and per-node admission checks remain. Distinct binding indices, even with
equal bytes, never match. No source cache crosses a batch/publication lifetime.
At most16 cached lists and4096 total cached views; overflow uses independently
materialized full original views, without narrowing or skipping checks. New
indices/scratch are wiped on all exits. Original non-bulk nodes remain Vec-owned.
Each actual capture separately deduplicates/charges these source bundles while
retaining original allocator Binding bundles and unique full resource owners
through every real flight, split, quarantine and recovery path.

The actual host-reservation formula is shared by producer and actual-INFO budget
control. It charges original argument plus worst resolved-view images, all
cache/scratch/Arc/table headers and real pipeline/INFO containers. The fixture
charges18,915,776B of the unchanged25,165,824B host reservation; the worst full
8MiB tape/5632-node case plus miss-ring metadata also fits. No private/device/
parent/capture cap is enlarged. Original release points/closed logical grammar,
real INFO producer and all shader arithmetic remain unchanged.

Compiled-public counters distinguish actually cached nodes/views, actual full
root revalidations and actual shared-source nodes/unique bundles, after the
original real sync. No network, KNT2/KNR2, model, configuration or Vashti field
changes. No private metadata is published or persisted.

All80 distinct actual App CPU controls pass (original73 plus four capture-index
and three source-bundle controls), log `native-capture-inventory-combined-cpu-20261020.log`.
They retain original alias/queue/rolling/terminal/owner/budget/math producer
checks, and cover complete full views, BDA/indirect, equal-byte foreign signature,
changed/missing roots, count/overflow/truncation/capacity negatives, actual unused
capacity erasure, exact ordered/duplicated/distinct source keys, fallback after
cache bounds, error rollback and independent batch lifetimes. Initial integration
compile caught Option/Result budget propagation and a live immutable signature
borrow spanning ready mutation; both were repaired before final CPU acceptance.
No numerical reference/tolerance or recovery criterion was relaxed. Release,
offline layouts, original129 numerical GPU/full requests/actual counters/
physical recovery and performance are next. Not installed or selected.


## Complete capture/source result (bfbfc0c3)

Source`bfbfc0c3d51d39409e364530fef44ad65d93495d`, worker SHA256
`ac01855f2dc9002208f81b88533a364657594d4bd4d6e1e9e7c94b1113baa90d`,
`strata-parity-capture-inventory-20261020`, passes80 actual App CPU controls,
408/208 offline layouts, all129 original numerical GPU controls, three complete
original-weight requests, all original bulk/wide/tail/miss-ring/OR/mask/push
proofs and actual new capture/source counters, graceful exit and physical
recovery. All original mathematical references and tolerances remain unchanged.

Load82.47800s, first/repeated4245 prefix13.14888/11.78149s. Repeated experts
7.58404s, enqueue1.51518s, capture2.01684s, wait3.88843s; attention2.94794s.
Versus preceding11.83052s, this is unchanged within ordinary variation, NOT a
meaningful win or clean comparison. Installed clean11.14739s and matched
Strata5.3515s still lead. No new clean/staged/terminal/installed or sustained
decode acceptance. The complete connected bundle is retained, unselected and
uninstalled; do not roll back individual components from these partial timings.

Actual repeated prompt:204,463 original nodes ALL use the cache;8,968,280
actual cached view accesses,21,782 actual full guarded-root revalidations.
52,676 actual source-reading nodes retain96 distinct full source bundles summed
over the real batch captures. This is within-batch sharing; never interpret it
as authorization for a persistent cross-request/publication cache. It proves
that despite source-handle sharing, complete source lists are still expanded
into the flat signature and walked repeatedly by planning and compute passes.

Receipts `strata-parity-capture-inventory-20261020-{operators,profile}`, profile
`capture-inventory-validation.json`, owned
`strata-parity-capture-inventory-public-summary-20261020.json`.
Recovery16:53:49UTC: Intel50,688,000B, NVIDIA compute0, faults0, endpoints200,
admission inactive/unpaused/no recovery/waiting0. Installed remains source3fffc266
with gateway36626d30. No network/API/config/model/Vashti change or release.

## Next coherent source proposal: factor complete source declarations end to end

The actual8.97 million view accesses versus96 full source-bundle instances
changes the next target. Do NOT spend another GPU block only on faster lookups
or byte copies. Complete source declarations are still repeatedly expanded and
walked across signature/planner/compute representations. Factor their immutable
representation and consume it throughout the connected pipeline instead.

Required connected implementation:

- Intern a full actual read bundle ONCE per capture from the SAME immutable
  genuine Arc/source owner and original binding table. Retain every ordered
  original buffer identity, offset, size, duplicate and full declaration. No
  same-byte identity substitution, narrower ranges, guessed shader-name proof,
  missing physical dependency or cross-batch/publication/request source cache.
- A structured signature may reference that complete canonical list from each
  original node, rather than expand it thousands of times. Preserve ALL original
  effective signature fields, node order, dispatch, INFO, pipeline and ownership
  distinctions. Equality/replay must be exactly equivalent to original expanded
  comparison; a hash may reject candidates but NEVER grant equality alone.
  Keep an independent original-expansion oracle in CPU controls. Original
  allocator/recording owners and IO epoch checks remain essential.
- Carry the complete shared declarations through actual original planning and
  compute hazard processing, not just through argument cloning. Any omission of
  repeated tracking work needs a SEALED complete original write inventory and
  strict equivalence proof. Only sources with NO original write anywhere on
  that full root across ALL physical queues in this recording can qualify.
  A writer in unrelated bytes, inline transfer, COPY, private overlay/loan,
  unknown future access or unsupported proof must keep the full original path.
  Do not assume that a weight-containing root is immutable: public arenas can
  alias private scratch. Existing CPU source publication/pinning guards stay.
- Keep the complete original source footprint in external join/conflict and
  lifetime ownership declarations even if proven redundant INTERNAL read
  bookkeeping is factored. Preserve original node/queue order, exact segments,
  every required peer wait/release and first/last external memory boundary.
  Compare plans against the unchanged old planner and independent byte oracle,
  including future writers, disjoint writes, queue changes, aliased roots,
  unknown scope/capacity/epoch and terminal recovery negatives. This is a
  different representation of the same full declarations, not permission to
  simply remove read_dependencies or clear history from assumed immutability.
- Build capture-only indexes only AFTER an actual replay miss. The present
  prototype builds its index before the replay decision; it is unused on a hit.
  Avoid adding that work to named decode replay. Prevalidated per-node layout
  records can also avoid re-parsing immutable signature headers in each pass.
  Combine these with the full declaration change, rather than running tiny
  isolated timing experiments. Current prototype has no new sustained decode
  or full staged acceptance, so do not install it as if that were verified.
- Retain real Wgpu guard reacquisition/root revalidation, full original views,
  actual counts/membership, original public/native grammar and INFO producers,
  BoundGraph/IO/owner/fence/quarantine/erasure checks and unchanged24/64MiB caps.
  Charge all canonical lists, per-node references, sort/hash/temporary peaks and
  full wipes before any prefix. Missing optimization proof keeps original
  behavior, with no increase to memory or admission limits.

This is a connected source proposal, not yet implemented or measured. Then run
original80 CPU plus producer/oracle/equivalence/budget controls, all129 numerical
GPU gates, actual full requests/new producer counts and physical recovery.
Only a meaningful matched clean gain earns terminal/full staged thinking/tools/
MAX8/semantic/privacy acceptance and official installation. No further Ollama
baseline runs. The Strata parity goal remains active and unmet.


## Connected canonical full-source representation

The complete successor is implemented. At actual capture the same genuine
immutable source Arc/allocator binding table supplies one canonical ordered full
view list. Lifetime root retention visits each actual shared source owner once
instead of expanding it per node. Every descriptor, full BDA view/duplicate,
indirect operand, original node order, pipeline, release/shared flag, dispatch,
INFO/resource field and ownership constraint remains. Ordinary non-shared
captures keep the original flat representation. Missing compact capacity keeps
the original flat path; all actual allocation capacities are charged.

Canonical nodes reference complete source lists, not presumed smaller accesses.
An independent original expanded-word oracle checks every field. Replay equality
is exactly equivalent to original expansion, including mixed flat/structured and
separately owned equal lists: comparisons use complete actual view tuples and
all node fields, never a hash-only match or source-pointer shortcut. Source
interning itself still requires the SAME original immutable Arc/binding owner;
no equal-byte identity substitution occurs. Original allocator/recording owners
and IO epochs remain the authority for real lifetimes.

After a real replay MISS, a capture-only evidence object checks every stored
full view/indirect against actual guarded root membership/extent, then binds all
signature allocation identities. Every later actual Wgpu guard revalidates the
same Keys owner, all live raw roots and sizes. Canonical nodes use their already
validated layout, rather than re-parsing an expanded signature. No new index is
built on named decode replay hits. Original flat capture lookup remains available.

A separate optional SEALED whole-recording write inventory uses the actual
original program access flags and every descriptor/COPY/FILL in ALL queues.
Unknown classes decline proof. ONLY a root with an original FULL physical read
view and no write anywhere on the entire root can be factored internally. A
writer in disjoint bytes, COPY, inline/shared work, private overlay/loan or any
future node prevents it. Never infer immutability from weight labels or fault
values. All original host publication/pinning and BoundGraph/IO checks stay.
A truncated/incorrect source/reference/count cannot publish a canonical signature.

The original complete source declaration remains referenced by each node.
Certified immutable members need not be expanded into each INTERNAL hazard
update. All other original members retain their ordered full views. The original
Planner separately retains the certified ORIGINAL whole-root footprint in its
canonical accesses; external joins/conflicts, ownership, every node/queue/segment,
peer wait/release and first/last ALL_COMMANDS boundaries remain. The planner
rejects an unfactored access to a certified root rather than silently hide a
writer. Actual descriptor/kernel dispatches still bind every original resource.
Compute tracking excludes only proven never-written roots, which cannot create
an internal data hazard; mutable/unknown roots keep the existing complete logic.
No memory dependency or completion/fence credit is fabricated.

All canonical views/nodes/source references, root/sort/write-proof/active-index
lists, full volatile wipes, pending planner footprints and finished metadata
are charged within unchanged64MiB. Proof unavailable retains full original
tracking. The original24MiB host and8MiB tape caps remain. Actual compiled-public
counters additionally prove canonical lists, certified full roots and logically
factored source declarations, after the original real sync. There is no new
HTTP/KNT2/KNR2/config/model/Vashti field or private-content persistence.

All86 actual App CPU controls pass (`native-canonical-source-combined-cpu-20261020.log`):
original80 plus six signature/ownership/proof/equivalence controls. New cases
compare EVERY expanded word/replay mutation, separately owned equal source
lists, unknown references/counts/budgets, actual signature/root ownership,
missing/full/partial physical roots, unknown classes, future/disjoint/peer writes,
and original compute dependencies. Eight512-node mixed recordings on both lanes
with serialization on/off match the original planner's EXACT segments, waits,
signals, releases, node mapping and full external footprints/conflicts. Original
independent byte/alias oracles and all mathematical references remain unchanged.
Release/offline/original129 numerical GPU/full request/producer/recovery and
performance gates remain next. No installation, speed or sustained decode claim.


## Enqueue admission without an unused range image

Canonical-source candidate5b173ad8/SHA
cc0aa6aa6b2694c3474138090c2437692213658a482e9c77eabb9b418f00b47a
passes86 App CPU,408/208 offline layouts, all129 original numerical GPU controls,
three complete public requests/all producer proofs and actual physical recovery.
Repeated4245 profile11.1239s improves prior11.7815s; clean repeat10.7871s
versus installed11.1474s is a modest3.2% difference, not parity or installation
acceptance. Actual capture CPU2.0168 ->1.3818s; enqueue remains1.5056s.
Actual96 canonical full lists,4,860 certified-root instances and2,675,866
factored declarations retain all full original external footprints. Actual
cached-view work8,968,280 ->3,616,548; nodes204,463/guarded-root checks21,782
unchanged. All previous components remain; no sustained/staged/terminal
acceptance or install claim. Clean recovery marker is present; verify the
independent clean host-after receipt before selecting anything.

The next connected internal change removes only an UNUSED enqueue range image:
outside the ENTIRE prepared/active draft-source lifetime, original admission
unconditionally joins all native submissions and does not consume incoming
ranges. Continue that SAME join at EACH node. Full original owning extents,
including indirect dispatch, are checked at enqueue. Actual original raw roots
are resolved under the SAME real capture guard before signature/replay/Plan,
encoding or submission, including destroyed/foreign-device rejection. No
GPU command can bypass that gate. All per-node source/descriptor declarations,
owning Arcs, budget/IO/epoch checks and external footprints remain unchanged.
An invalid raw resource may be rejected at capture rather than enqueue, always
before GPU publication. No raw handle is cached across phases.

From source PREPARATION through real joined retirement or terminal drain, retain
EVERY original enqueue range, write-protection test, full actual-flight conflict
check and selective join. Active-request absence is insufficient: prepared but
inactive owners are protected too. Unknown/partial scopes never take the fast
path. No-scope nodes retain the original unconditional join, including INFO
visibility debt and readiness completion; this does not skip synchronization.
Actual public-only post-sync counters distinguish both admission paths. No
network/API/configuration/model/frontend change.

Required validation: original86 CPU plus protected-owner lifecycle controls,
actual guarded-buffer destruction/foreign-device/budget GPU test, all129 original
numerical gates, full real requests and actual producer/admission counts, then
physical recovery. A meaningful clean gain must still earn original terminal/
full thinking/tools/MAX8/semantic/privacy acceptance and official installation.

Implementation3b35ea10056012107d720aaff510bbb3638da175 is frozen as
`strata-parity-unscoped-admission-20261020`, worker SHA
237477796fd59406e45abbb95ac770ef6654d073662d6a611559cebf1188d5a4.
All90 App CPU controls pass (original86 plus four actual protected-scope
controls). The protected actual Wgpu guard lifecycle test passes, with physical
recovery; test executable SHA25a57092b65550b7aded6a5f56e8b126d464b2ac046b0348986b4075f7dbfdd2.
Offline408 ordinary/208 matrix layouts pass; shader math is unchanged.
The original129 numerical gates and three full profiled requests are now running
in `kuno-native-unscoped-admission-connected-20261020`, swap/core0.
Artifact/receipt prefix `strata-parity-unscoped-admission-20261020`; no result
or installation claim until terminal proof and actual recovery.

Independent canonical-source clean recovery17:41:56UTC confirms Intel50,688,000B,
NVIDIA compute0, no new or existing driver faults, all services200 and normal
unpaused admission/waiting0. Previous installed source3fffc266/SHAeac78ce0
and unchanged gateway36626d30/SHAd3ed07f4 remain selected.


## Subgroup-independent GDN register-pressure candidate

Complete enqueue candidate3b35ea10/SHA23747779 passes original129 GPU controls,
three full requests/all original and new producer proofs, exit and physical
recovery17:49:54UTC. Repeated profile11.0527s versus11.1239s is unchanged. Actual
204,463 unscoped admissions, zero protected admissions in that prefill interval;
enqueue1.4037s, capture1.3777s, wait3.9248s. Keep the complete source, unselected;
no staged/terminal/clean/sustained or installation acceptance is claimed.

Compiler receipts also expose spills in several SIMD32 programs. The next
connected candidate addresses register pressure for prompt GDN recurrence. Its
128-invocation workgroup uses ONLY whole-workgroup coordinates/barriers and
explicitly ordered per-invocation32-row state/FMA loops, with no subgroup
arithmetic. Keep every expression, exact state mapping, causal prefetch/SLM
barrier, finite/fault check, INFO/owner/budget and host launch validation.
Compile ONLY the opt-in recurrence entry as required SIMD16 in BOTH ordinary
Wgpu and retained native pipelines; all other kernels and the unchanged scalar
numerical reference keep original metadata. Smaller hardware subgroups can
reduce per-thread register pressure without changing logical invocation IDs.
No performance result or shader-specific spill attribution is assumed yet.

The opt-in `strata-gdn-subgroup16` changes only the literal compiler entry name.
Both actual consumers inspect COMPLETE emitted validated SPIR-V: exactly one
named GLCompute128x1x1 entry, only Shader capability, only original workgroup/local
builtins, original Workgroup barriers, Device/Workgroup atomic scopes, no
extensions, subgroup capabilities/instructions, alternate execution modes or
specialized geometry. Unknown proof retains the original policy. The literal
name scopes this optimization; the actual module proof establishes absence of
subgroup dependencies. Existing device-owner marking, physical subgroup/stage/
workgroup limits and all original numerical gates remain mandatory.

The ordinary Wgpu separate-entry path obtains the SAME proof from the complete
validated Naga module lowered to words; overrides never qualify. No globally
disabled robustness, varying subgroup policy, relaxed arithmetic, modified
FP32 reduction, narrower dependencies or new memory cap is introduced. Existing
public-only numeric entry adds static compiler-scope delimiters to distinguish
reference/selected recurrence ISA; private inference never executes that entry.
Actual ten complete-output/next-state bitwise GDN cases, nine original QSA
bitwise and independent F64 cases, all remaining129 numerical controls and full
request/recovery proof remain the next gates. No network or Vashti change.

All94 actual App CPU controls pass. First frozen215a11da/SHAf9028910
failed OFFLINE standalone validation because the new validator branch omitted
the existing HAL Workgroup-layout canonicalization (raw Naga adds ArrayStride
on Workgroup arrays). No GPU test was started for that artifact. Correct the
offline producer to apply the SAME checked normalization already used by both
actual shader factories before spirv-val; no validator/numerical rule is relaxed.
The corrected successor retains the exact same kernel and pipeline behavior.

Corrected frozen source022d799bbf76d58391c8dedfe316cb8b392d0afc / worker
SHAe3ba33c026b50a9a7c3442f1e8e8ca80c1c676687d35a40182f705aae7691fa8,
build`strata-parity-gdn-subgroup16-v2-20261020`. Offline408 ordinary/208 matrix
layouts pass, including both actual named GDN entries318/323 through the
unchanged spirv-val gate after actual HAL normalization. The protected unit
`kuno-native-gdn-subgroup16-connected-20261020` now runs original129 GPU controls
then full requests/all inherited producer proofs and actual compiler-scope
selection/spill evidence. No actual GPU correctness/performance acceptance yet.
Both tests and workers retain swap/core0 and real recovery checks.


## Completed GDN experiment and current performance checkpoint

Source022d799b / SHAe3ba33c0 passes all original129 numerical GPU controls,
three full real public requests, every inherited full declaration/queue/source/
miss-ring/push/canonical/admission producer proof, graceful exit and physical
recovery18:07:51UTC. The new BOTH-pipeline ISA assertion then FAILS: ordinary
Wgpu still compiles SIMD32 (26 spills/52 fills), while the actual retained
native GDN pipeline is SIMD16 (1706 instructions, ZERO spills/fills). Original
reference ordinary/native remain SIMD32 (26:52 and53:122). Do NOT suppress this
assertion or describe both paths as selected16. Unknown ordinary-module proof
correctly retains its original policy; the exact cause is not established.
No unsafe subgroup bypass or weakened numerical tolerance was introduced.

This is a numerically valid native-path experiment, not complete new producer
acceptance or installation. The diagnostic is explicitly marked
`new_both_pipeline_producer_proof_passed:false` in owned
`strata-parity-gdn-subgroup16-v2-public-diagnostic-20261020.json`; the original
failed wrapper log is retained unchanged. Root receipts remain under
`strata-parity-gdn-subgroup16-v2-20261020-{operators,profile}`.
Repeated profile11.051815s versus preceding11.052699s is unchanged. GDN
recurrence/norm coarse interval0.436133 ->0.401232s, but QSA attention/merge
1.467331s and the whole prompt do not materially improve. Do not spend the
next pass tuning an unused ordinary fallback just to make this candidate
look complete; native GDN spill elimination is NOT the main remaining cost.
No clean/staged/terminal/sustained acceptance or install was attempted.

Current run's real gain is canonical full-source capture2.016843 ->1.381753s,
with clean repeated4245-token10.787071s versus installed11.147391s (3.2%).
That still trails same-host matched Strata5.3515s; the parity goal remains active
and unmet. All complete coupled changes remain retained as opt-in development,
not reverted from small isolated speed outcomes. Installed worker3fffc266/
SHAeac78ce0 and image-enabled gateway36626d30/SHAd3ed07f4 are unchanged.
Independent18:12:57UTC check: Intel50,688,000B, NVIDIA compute0, faults0,
all services200, admission inactive/unpaused/recoveryfalse/waiting0. No GPU
job remains active. Owned bounded credential refresh64142 remains until18:59UTC.

### Next focused work: actual attention compiler output

The remaining QSA attention/merge costs roughly1.47s versus the reference's
roughly0.68s instrumented interval. Compare actual selected kernel instructions
and memory traffic, not another generic graph-lookup rewrite or Ollama baseline.
Do not assume the reference has256 GRFs or uses its opt-in XMX path.

A CPU-only, offline, device-free extraction now works against the SAME pinned
AOT reference. Its61,501,536-byte `sycl-spir64_gen` bundle concatenates1429 ELF
device images; treating it as one ELF or disassembling only the first image is
wrong. The first images include opt-in XMX kernels and do not prove selection.
`strata-public-aot-qsa-gdn-metadata-20261020.json` records29 matching kernel
metadata rows. Actual `qsa_decode_attn_batch` default INT8 (`Li1`) images and
`gdn_rec_cols_pipe_kernel` are SIMD32/128GRFs. XMX launch images are SIMD16,
but pinned source says `STRATA_PROMPT_ATTN_XMX` defaults0; no such override was
added to the accepted reference. Static metadata is NOT a new live selection or
performance result. Next extract/disassemble the corresponding actual default
INT8 image and compare to original Kuno public compiler code, including FMA,
SLM/shared loads, sends and materialized products, before choosing a complete
attention change. Retain original nine bitwise + independent F64 controls.

Reproducible CPU-only extractor:
`/media/jmicko/faststore/kuno-development/inspect-strata-public-aot-metadata-20261020.py`.
Use retained image7b7f0a2a9c0d783fc3c1cb7250ac8ded9accea48f5bc54983cee8bbdbedfefc2,
rootless Podman, no network/device/model mounts, read-only build mount and
512MiB /tmp tmpfs,2 CPU/2GiB/no swap. Within it:
`/opt/intel/oneapi/compiler/latest/bin/compiler/clang-offload-bundler --type=o --unbundle --targets=sycl-spir64_gen --input=/engine/strata --output=/tmp/strata-device`.
The script walks complete ELF section tables and emits ONLY public static
code metadata. No GPU was opened, no live request was run, and extracted
device bytes stayed in deleted container tmpfs. Host logs
`strata-public-aot-{tools-inventory,sections,metadata-extract,device-sections}-20261020.log`
record tool/path preparation; first inventory's exit127 is missing `rg` INSIDE
the image, not Podman or GPU failure. `ocloc disasm` is available in that image.

The reference's larger stream-all ring is conditional: pinned
`prefill.cpp::ring_slots` permits the default large ring only when nonresident
pairs are below10%; otherwise it returns STAGE8. Do not infer that Kuno simply
missed a384-slot default without inspecting actual placement and byte budgets.
No API/KNT2/KNR2/configuration/model identity or Vashti consumer change.


## Connected explicit QK FMA candidate

CPU-only extraction/disassembly of the actual pinned default INT8 image
(`attn_chunk_kernel_2ad0a4`, not opt-in lane-cell/XMX) establishes chronological
FMA chains in QK: one float multiply followed by seven float MADs per head,
then the original XOR reduction. Strata's99 static float MAD instructions
include84 QK and12 PV plus setup. Its actual shader is SIMD32/128GRFs.
The earlier source comment claiming universal Strata contract(off) was incorrect
for this AOT binary; no build flag or source assumption substitutes for this
observed code. The preserved legacy product-rounding postpass remains unchanged.

New opt-in `strata-qsa-fma` changes ONLY the seven chronological per-lane QK
accumulation statements to explicit FMA, retaining the separately rounded FIRST
product, exact operands/first-to-last term order and every XOR step. All public
INT8 dequantization products and other remaining FMuls retain original
NoContraction. Query/SLM layout, paired-cell reuse/full guarded tail, probability
gather/padding, max/exp/sums, chronological PV FMAs, stores, all finite/fault and
causality checks, INFO, dispatch, original owners/epochs and queue/lifetimes
remain unchanged. No unsafe fast-math flags or blanket precision relaxation.
Keep all complete preceding coupled components including native GDN candidate.

The original scalar GPU reference is untouched. All original nine bitwise QSA
cases and three independent operation-bounded F64 cases MUST pass unchanged,
as must the other original129 controls; no numerical acceptance rule is widened
for FMA. New CPU controls restore the complete original source byte-for-byte by
undoing only these statements across all eight status/fault/paired/gather
variants; real Naga/SPIR-V proves exactly7 additional Fmas and7 fewer FMuls per
actual dot, every remaining product precise, identical group/barrier/shuffle
geometry and fault predicates. Test the whole connected pipeline after all
original numerical checks, before claiming speed or installation.

Public static ISA lives under `strata-public-attention-isa-20261020`. Extraction
selected-0.json is GDN (directory qsa), selected-1.json is default INT8 QSA
(directory gdn); directory names reflect initial extraction order and are NOT
selection evidence. `.text.*.asm` and `.ze_info` files are hidden dot files;
`rg --files`/plain ls omit them. No need to rerun ocloc assuming they are absent.
Duplicate `dump/` is the same QSA disassembly. Public original Kuno numeric
compiler code is extracted in `kuno-public-qsa-last-four-isa-20261020.txt`.
Native-code SUMMARY PRECEDES its assembly in Mesa output; splitting only at
summary ends would omit the last program's assembly. No GPU, model mount,
private trace, driver, service or system configuration was used for extraction.
No API, protocol/configuration/model identity or Vashti consumer change.


## Complete FMA result and coupled uniform-QK successor

Frozene854a89d45ba8c38528366f74c13e2506c641707 / SHA
420dc7d09da202b712dedcfca33f0d9d5e7b031de20962063f339f546e2eba0e
(`strata-parity-qsa-fma-20261020`) passes111 actual CPU controls,408/208
offline layouts, all129 original numerical GPU cases, complete real requests/
all inherited producer proofs, actual QK native FMA ISA proof and physical
recovery18:33:15UTC. All nine original bitwise QSA cases pass; the three
non-dyadic cases ALSO now match original scalar partial/stats bits while passing
their unchanged independent F64 bounds. Actual native optimized ISA4127
instructions/348 MADs/zero spills versus previous4461 instructions. Exact
compiler scopes disambiguate reference2002/1599 and optimized4109/4127; do
not infer ordinary/native identities from interleaved old unscoped summaries.

Repeated profile11.174104s versus preceding11.051815s is unchanged/slower;
QSA attention/merge1.459168s versus1.467331s is unchanged. Expert groups also
change26338 ->26673 (same original weights/seed/prompt; FMA changes actual
FP32 routes). Do not claim an overall speed gain from the reduced static code.
The complete FMA source is retained; no partial rollback. Owned summary
`strata-parity-qsa-fma-public-summary-20261020.json`, root receipts
`strata-parity-qsa-fma-20261020-{operators,profile}`. No clean/full staged/
terminal/sustained or installation acceptance. GDN ordinary16 proof is still
missing; actual native16/ordinary32 limitation remains visible.

The next COMPLETE opt-in `strata-qsa-uniform` couples three compiler changes,
while keeping FMA, paired/padded/local-fault and all earlier components:

- Broadcast immutable selected token IDs from subgroup lane zero. Every lane
  originally reads the SAME `row[c0/c1]` word: the paired cell indices are
  subgroup-uniform and the original whole-workgroup initialization barrier
  precedes this immutable read phase. Broadcast preserves the exact original
  selected ID/causal failure, and explicitly exposes uniformity to the compiler.
  No replacement expert/token or guessed smaller footprint is admitted.
- Materialize each of the five `lane XOR distance` indices once, then use
  explicit subgroupShuffle at those exact indices for every original QK,
  maximum and sum step. Physical subgroup32 and all original addition/XOR
  pairings stay. No operation reordering, new barrier or subgroup-size change.
- Broadcast the exact original lane-ZERO score (`dot/16`) before checking it
  for finiteness outside the lane-zero store branch. This samples the SAME
  original score, even for arbitrary NaN payloads/other-lane values. Every
  invocation contributes the same old lane-zero failure to the existing sticky
  OR; final fault/status/output is unchanged. A raw all-lane finite check without
  the score broadcast is NOT equivalent and is explicitly rejected by the CPU
  counterexample. Original lane-zero SLM/status stores and all other predicates
  stay. The existing local-fault rewrite follows, so valid lane-zero stores no
  longer carry divergent per-invocation sticky-fault SSA state.

No arithmetic, dequantization rounding, operands, max/exp, chronological PV,
private owner/extent/header/epoch, dependency footprint, queue, actual Wgpu
guard, lifetime/fence/quarantine, budget or erasure gate changes. The independent
scalar reference is untouched. New complete-source undo controls compare every
remaining byte; actual SPIR-V proves50 BroadcastFirst sites, zero ShuffleXor,
all original250 indexed reduction steps plus12 existing PV gathers, unchanged
256-invocation geometry and original workgroup barriers. Initial CPU collector
used incorrect numeric opcodes for Shuffle/ShuffleXor; corrected against the
local SPIR-V enum (345/346). This is a test-collector fix, not weakened shader
or numerical acceptance. All original129 GPU controls remain mandatory.
The opt-in requires the full FMA/paired/padded/local-fault feature bundle.

A fresh bounded credential refresh now owns outer128882/loop128884, parent3556,
started18:39:26UTC, ends20:39:26UTC; log/PID/owner files
`sudo-refresh-qsa-uniform-20261020.*`. Verified prior64142 loop was retired;
the shared shell is untouched. Never kill its process group. No API/protocol/
configuration/model identity or Vashti change.

Frozen uniform-QK source84316ccb4d44a00b0530b2f5f2e6105cf4872baf / worker
SHA7e6b09917b984857c48285d72a8cc773150e249662b1e2733f4055db880915d0,
build`strata-parity-qsa-uniform-20261020`, passes114 App CPU and408/208
offline layouts. All129 original numerical GPU controls have now passed with
physical recovery; full model requests/producer/compiler/group-count/recovery
and performance are running in `kuno-native-qsa-uniform-connected-20261020`.
No performance/install acceptance yet. Every preceding exit and guarded QK
decision is subgroup-uniform: the original full fixed32 groups keep lane zero
active at every immutable-ID and original-score broadcast. No partial subgroup
or arbitrarily chosen first-lane value is substituted.


## Completed uniform attention and disjoint expert scratch candidate

Uniform-QK84316ccb / SHA7e6b0991 passes114 App CPU,408/208 offline layouts,
all129 original numerical GPU controls, full real requests, actual compiler
and all inherited producer proofs. Actual routed group counts remain exactly
[7663,26673,26673], identical to the FMA reference. Repeated4245 prefix11.128987s
and QSA attention/merge1.495390s are unchanged versus FMA11.174104/1.459168s.
No clean/staged/terminal/sustained/install acceptance or overall gain claim.
Public summary`strata-parity-qsa-uniform-public-summary-20261020.json`; unit
`kuno-native-qsa-uniform-connected-20261020` exits successfully. Physical
18:53:52UTC recovery: Intel50,688,000B, NVIDIA compute0, all driver faults0,
HTTP200 and normal inactive/unpaused admission/waiting0. Installed identities
remain unchanged. All complete components stay retained opt-in.

Source inspection rules out a per-QSA-dispatch INFO upload: stream flush
already prepares those pages together. Do not implement another batching API
based on that discarded hypothesis. The concrete expert difference is scratch
reuse: Gather, GU, Activate and Down all reuse the same packed-input area;
all groups reuse the same GU area. Both dequantizers also declare the entire
shared image area despite accessing disjoint actual roles/slots. Strata uses
separate grouped Xs/GU/Hh spans and two FP16 image slots.

The next complete opt-in`strata-expert-scratch-ring` / builder
`--expert-scratch-ring` couples TWO genuine packed-input/GU slots and separate
GU/DOWN FP16 image descriptors. Slots alternate by group modulo2, matching
original DQ2. Input and hidden remain intentionally shared WITHIN one slot,
as in the old six-phase program; different slots do not overlap. Slot0 GU is
the original admitted GU view. Two20MiB input views and slot1's20MiB GU fit
inside the original80MiB working0 area, using60MiB without increasing any
priced root, placement, original host/graph/tape cap or allocation. DQ role
views carve the original working1 two-slot image area. Shader-relative image
offsets become zero ONLY with these real rebased descriptors. Both dequantizer
and matrix consumer use those exact views; no scheduler-only footprint narrowing.

The original shaders, arithmetic, raw source declarations/miss ring,6 logical
phases and authentic full/wide/tail products remain unchanged. The SAME closed
batch grammar sees the actual new views and array metadata; no special grammar
exception is introduced. The graph resolves every new original owner under its
actual guard, computes hazards from the complete descriptors, and keeps original
copy waits, INFO epochs, lane/fence, budget, quarantine and erasure behavior.
All static views are prepared before enqueueing the batch; malformed extents,
alignment, capacity, streams and overlapping roles are rejected. This removes
false scratch conflicts but is not a claim of complete cross-expert parallelism:
remaining global pipeline dependencies still order real consumers.

Original129 GPU controls plus full requests/identical routed group counts,
actual new scratch counts, unchanged producer/owner proofs, physical recovery
and measured timings are required. Any numerical/guard failure remains a
failure; do not relax existing references. No API/KNT2/KNR2/configuration/model
identity or Vashti change. No installation or speed claim before acceptance.


The complete source now passes117 selected actual App CPU controls (the prior
114 plus real disjoint ownership/extent controls and both actual slot layouts
through original scalar/macro/narrow/wide+tail/resident/streamed grammar).
Log`native-expert-scratch-complete-cpu-20261020.log`. An initial broad `scratch`
test filter also selected four unrelated old feature-profile fixtures; those
failed their existing scratch/profile expectations. They remain unchanged and
the full failed log`native-expert-scratch-new-cpu-20261020.log` is retained.
The117-control pass is an exact named set, NOT a whole-suite pass or a claim
that those four unrelated configurations were fixed. No GPU acceptance yet.


Frozen76836aa8 / SHA36efb2eb (`strata-parity-expert-scratch-20261020`)
passes117 final-source CPU controls,408/208 offline and all129 original GPU
controls. The first whole-model request FAILS the unchanged raw route-header
guard in`strata_prompt_groups.rs:37`; no prompt timing or full producer
acceptance exists. Do not install or remove that guard. The original wrapper
records compute_passed=false and physical_recovery_passed=true. Actual
19:08:55UTC recovery confirms Intel50,688,000B, NVIDIA compute0, driver faults0,
all services200 and normal inactive/unpaused admission/waiting0. No GPU wedge.
Swap0/core0 were checked from the actual wrapper cgroup/rlimit before any worker
and recorded in`strata-parity-expert-scratch-20261020-protection.json`.

Next diagnosis retains the ENTIRE scratch candidate and original numerical/
header guards. A fixed-public-only static phase marker reports the failing
model layer without copying tensor, route/expert IDs, request/response bytes or
private state. No private worker selects this entry/diagnostic. This is a
correctness investigation, not a speed result or partial feature rollback.


The failure is now located in source: the ORIGINAL dequantizer's exact
`arrayLength(packed)==PROMPT_WORKING_WORDS/4` check correctly rejects smaller
role descriptors. The earlier complete scratch candidate failed to connect its
new views to a corresponding exact shader layout. Do not call this a GPU hang
or loosen the original guard. The phase-only diagnostic4507e153/SHA23c898f0
was frozen but NOT run; source inspection found the cause without that extra
whole-model run. The owned failure summary is
`strata-parity-expert-scratch-failure-public-summary-20261020.json`.

The corrected COMPLETE candidate adds a DISTINCT`SlotDequant` kernel identity.
All original Dequant/ResidentDequant sources and full80MiB checks remain
byte-for-byte unchanged. The slot variant changes ONLY that output-layout
predicate: exact role-sized output array AND BOTH relative offsets zero.
It retains every source, raw, table, map, quantization, alignment, finite/fault
and output-bound predicate, and the same streamed/resident release class.
Actual producer dispatch selects this variant only with the real role views;
legacy producers and original129 controls select the original kernels.
Native batch operator names/grammar, shape/owner/array metadata checks and all
scheduling/erasure controls remain original. No generic acceptance exception.

All118 selected actual CPU controls pass, including whole-source undo for all
12 role/residency variants, distinct kernel IDs, original release classes and
the unchanged original grammar with both real scratch slots. The new public
GPU gate adds42 separate exact-role dequant cases, using the SAME original
CPU-decoded FP16 oracle and inspecting the COMPLETE original80MiB output owner,
including unchanged bytes surrounding a real rebased descriptor. It covers
six original quant formats, HOST/resident valid inputs, original malformed
source extent/alignment/nonfinite cases, plus wrong role-view length and a
nonzero-but-aligned relative image offset. Invalid layouts must fault BEFORE
source access/write. New cases do not replace or relax the original129 controls;
no private entry point, tensor/request trace or extra private observation.
The new public report key is laboratory-only, not a Kuno client API change.
Next: freeze/offline420 layouts, original129 plus42 new GPU cases, complete
requests/producer/group-count/recovery and timings. No acceptance assumed.


## Next complete attention code-size candidate

Corrected scratch sourceb1fb992b / SHA37cf3a1e, build
`strata-parity-expert-scratch-v2-20261020`, passes118 App CPU and420/208
offline layouts. Protected unit`kuno-native-expert-scratch-v2-connected-20261020`
is running original129 and new42 GPU controls, then full requests/producer/
routed-count/recovery/performance gates. No accepted result assumed.

Independent source/actual-ISA review finds a second concrete difference:
paired QK emits BOTH a full-pair specialization and its complete guarded tail.
The previous optimized native shader contains4320 instructions/348 MADs versus
the smaller default Strata image. Static count is not dynamic performance proof.
The new opt-in`strata-qsa-compact` / builder`--qsa-compact` retains the COMPLETE
paired/uniform/FMA/padded-PV/local-fault bundle but emits the original guarded
body once for every pair. It does not revert pairing, contraction or fault policy.

The source producer FIRST verifies byte-for-byte that erasing ONLY each of the
12 `valid0` and12 `valid1` per-head guards makes the guarded body identical to
its previous full-pair specialization. When both are true the arithmetic and
stores are identical; when either is false the same original guarded path was
already selected. Both are subgroup-uniform, with all32 original lanes active
inside a selected body. Every token/causal/dequant failure before the branch,
shared query/score/probability barrier, original XOR tree and score sample,
chronological FMA/softmax/PV, output/status and closing fault remains. Whole
source outside this replaced branch stays byte-identical, including all real
owners/footprints/headers/INFO/budgets/queues/fences/erasure. No scalar reference
or numerical threshold changes. This requires the full existing uniform bundle.

New controls compare the ENTIRE source after restoring only the duplicate
branch, reject drift between its two arithmetic bodies, and inspect ACTUAL
compiled status/fault SPIR-V plus the selected production hook. Expected static
lane-zero samples50->26, indexed shuffles262->142 and FMAs348->180 remove ONLY
those duplicated QK bodies; the original geometry, shared globals and barriers
stay. These expectations are pending validation, not claimed results.
Original129 plus42 scratch GPU controls, complete requests/identical routed
counts and physical recovery precede speed claims. CPU-heavy compilation waits
until the running scratch timing test finishes. No API/KNT2/KNR2/configuration/
model identity or Vashti change; installed services remain unchanged.


Corrected scratchb1fb992b/SHA37cf3a1e now passes all118 selected CPU,
420/208 offline layouts, the original129 PLUS42 new actual GPU controls,
complete real requests, ALL inherited producer/compiler/slot/group proofs and
physical recovery. Actual scratch slots96/192/192 across48/96/96 batches;
groups[7663,26673,26673] remain exactly the original FMA/uniform counts.
Repeated profile11.128987 ->10.624366s (4.53% shorter); first long
12.449061 ->12.021062s. Experts6.920840 ->6.421638s. Actual compute barrier
calls164,876 ->85,049, while37,359 real rolling segments stay. Expert capture
1.403028s is essentially unchanged; the real wait interval3.927201 ->3.462373s
is shorter but is NOT a pure kernel-time attribution. QSA attention/merge
1.494770s remains. No clean/full staged/terminal/sustained/install acceptance.
Reference Strata5.3515s still leads. All components are retained.

Root receipts`strata-parity-expert-scratch-v2-20261020-{operators,profile}`,
`expert-scratch-validation.json` and the owned
`strata-parity-expert-scratch-v2-public-summary-20261020.json` preserve exact
source/artifact, all old/new proofs and actual swap/core0 attestation. Actual
19:24:36UTC recovery: Intel50,688,000B, NVIDIA compute0, all faults0, services200,
normal inactive/unpaused/no recovery/waiting0. Installed binaries unchanged.

The complete compact-QK successor now passes120 actual App CPU controls
(`native-qsa-compact-complete-cpu-20261020.log`). Those include all preceding118,
whole-source reconstruction and actual selected status/fault pipeline/SPIR-V.
Actual static broadcasts50->26, shuffles262->142 and FMAs348->180 are confirmed,
with identical256-invocation geometry, shared globals and original barriers.
No numerical reference/guard changes, shader drift exception or partial rollback.
Freeze/offline, original129+42 GPU/full request/compiler/group/recovery/timing
are next; no speed or installation claim for compact-QK yet.


Frozen compact-QK5ac1e4e34a9e7923816d099915f1b50771721d29 / SHA
1a789884c13fcb0dd60533f98db4b6c424b65b6dbb1795a490170da9b6d55e50
(`strata-parity-qsa-compact-20261020`) passes120 CPU,420/208 offline,
original129+42 GPU controls, full requests/all inherited and new producer
proofs, exact original routed counts, actual compiler and physical recovery.
Native SIMD32 ISA4320->2654 instructions,348->180 MADs, zero spills/fills.
The smaller program does NOT improve measured QSA attention/merge:1.505209s
versus1.494770s. Repeat profile10.677227s versus scratch10.624366s is unchanged;
first long11.971158s. Keep ALL complete components. No partial rollback or speed
claim from the static code size. Experts6.465395s, capture1.399600s, actualwait
3.497518s and enqueue1.413363s show the remaining distributed host/compute cost.

Owned summary`strata-parity-qsa-compact-public-summary-20261020.json`; all root
receipts/proofs use`strata-parity-qsa-compact-20261020-{operators,profile}`.
Actual19:36:50UTC independent recovery is healthy/clear (see the shared shell;
profile receipt recovery19:35:08UTC): Intel50,688,000B, NVIDIA compute0, all
faults0, HTTP200, normal inactive/unpaused/no recovery/waiting0. Installed worker
and gateway remain unchanged. No clean/staged/terminal/sustained acceptance yet.

Protected`kuno-native-qsa-compact-acceptance-20261020` now runs original terminal
negative recovery, a FRESH clean45/4245/4245-token check, then full staged
thinking/tools/MAX8/code/privacy acceptance ONLY if the clean FIRST and REPEATED
long prefixes EACH beat the installed native32 reference by the existing5% gate.
No threshold reduction, warmup relocation, prompt cache, altered workload or
comparison with a different model. New wrapper
`check-native-qsa-compact-acceptance-20261020.py` verifies frozen source/hash,
all171 original/new controls, actual client swap/core0, and every step's full
physical recovery (including services, NVIDIA, driver faults and admission),
not a legacy passed flag written before finally. If speed gate fails, keep the
complete candidate unselected and investigate the remaining host path; do not
perform installation or unrelated API/UI work. Goal remains active/unmet.


## Clean selection outcome and fixed-public CPU-clock diagnosis

Compact5ac1e4e / SHA1a789884 passes the original terminal-negative partial-submit
recovery test and a FRESH complete clean45/4245/4245 run, with actual recovery.
Clean FIRST long13.711003s versus installed12.962279s is5.78% SLOWER; repeated
10.288472s versus11.147391s is7.71% shorter. The original5%-at-EACH-position
selection gate FAILS. `clean-selection.json` records both positions and false;
do not cherry-pick the repeat, lower the gate or install this candidate.
The full staged API/semantic check was deliberately NOT started after that
failed gate. No new staged/terminal-success-reuse/sustained/install acceptance.
Terminal failure retains its original explicit process-exit/quarantine policy;
its passed recovery is not an ordinary reusable-session/owned-zero claim.

The coupled source stays intact. Next diagnosis adds explicit opt-in
`strata-public-host-clock` / `--public-host-clock`, never deployed policy.
Its new vendor feature depends only on the ALREADY LOCKED libc crate; the lock
change is one existing dependency edge, no version/download/system change.
The existing fixed-public selected-layer guard (`PROFILE_ACTIVE`, authorized
native profile selector and `KUNO_LAB_LANGUAGE_PUBLIC_ONLY=1`) enables eight
coarse capture marks. Every mark records BOTH monotonic wall and actual current
thread CPU ns. A missing/invalid/nonmonotonic thread clock is counted explicitly,
not silently reported as zero CPU work. Ordinary private code constructs no active
clock profiler, records no extra counters or diagnostic data, and default
builds do not contain this optional clock path.

Phases: inventory/signature; snapshot/preparation; original queue plan/classes;
original COPY dependencies; original compute dependencies; retained/rolling
preparation; command encoding/rolling; finalization. Complete original signatures,
actual owning guards, full descriptor/BDA views, INFO epochs, node ordering,
real queues/headers/fences, source visibility, original budgets/quarantine and
private erasure remain unchanged. No added GPU submission, wait, event/query,
barrier or memory access. The anonymous static256-byte counters and tiny timer
carry only labels/counts/durations, no pointer, model value, route/expert ID,
prompt, response, stack/register/memory sample or file content. The existing
compiled-public sampled-layer JSON gains a clearly separate`host_capture`
map with columns[calls,wall_ns,thread_cpu_ns,cpu_clock_unavailable]. No Kuno
client API, KNT2/KNR2/configuration/model identity or Vashti change.

CPU controls exercise actual enabled/disabled clock behavior and inspect the
actual public-only graph gate/eight static marks. Numerical shaders remain
unchanged; all prior components and original references/guards stay. Compile
only AFTER the active GPU acceptance unit has ended. Then original CPU/GPU/
complete request/recovery gates and the fixed-public sampled-layer diagnostic
locate the next connected optimization. Do not call elapsed capture time pure
CPU cost or infer that wall-minus-CPU is exclusively GPU waiting. These clocks
are diagnostic, not accepted performance. Clean/source/private product policy
and existing selection threshold remain unchanged.


Actual19:46:19UTC recovery after the failed clean selection: Intel50,688,000B,
NVIDIA compute0, driver faults0, allHTTP200, normal inactive/unpaused admission,
recovery=false/waiting0. No staging or installation was started, no GPU work
remains from acceptance. Installed identities remain unchanged.
One source hypothesis is already ruled out: the original std KernelLauncher
uses a thread-local reused IR Scope and reused INFO lease. Do not attribute
host time to reconstructing an IR Scope on every launch. A prepared-launch
proposal must demonstrate which actual remaining work it removes. Keep full
private wiping, owner checks and original global visibility; memset symbols
alone never justify deleting erasure or lifetime validation.


All123 selected actual App CPU controls now pass, including disabled timing,
real Linux thread/wall clocks with explicit availability and the ACTUAL public
only graph gate/eight marks. Source has no new GPU operation or numerical
change. Logs`native-host-clock-{cpu-build,complete-cpu}-20261020.log`.
An initial editing helper stopped before writing Graph because its final-command
marker also matched an earlier segment boundary; corrected to the exact final
encoding boundary. Only our incidental main/mod formatting was restored before
applying the intended edits. No failed or partial edit was frozen/run on GPU.
No shader/numerical/header/reference/selection gate was relaxed.
Freeze/offline/original129+42/full request/host-clock/recovery remain next.


### Normal-schedule host-clock follow-up

Frozen44f8d68f/SHA41d0e593 completes123 selected CPU,420/208 offline,
original129+42 GPU, complete73/4245/4245 public requests, original compiler/
producer/count/erasure/recovery gates and all eight distinct wall/thread-CPU
rows with zero unavailable clocks. Repeat10.6932s is a diagnostic, not a clean
selection result. Sampled layer24 reports encoding8.2430ms, retained preparation
7.7313ms, queue planning1.9695ms, compute dependencies1.5124ms, and smaller
remaining stages. These are actual CPU intervals for the INSTRUMENTED layer:
the existing GPU timestamp sampler serializes queue planning and disables its
rolling path. They cannot establish the ranking on the normal execution path.
Actual19:58:17UTC recovery is clear/healthy; installed identities unchanged.
Owned summary`strata-parity-host-clock-public-summary-20261020.json`.

The complete follow-up retains that diagnostic and adds the explicit compiled
lab-only`--public-prefill-host-profile SETTINGS` entry under the same optional
`strata-public-host-clock` feature. It selects`prefill-host-layer` only after
scrubbing all inherited experiment environment variables before threads.
Private/KNT2/API entries always pass false. The original GPU timestamp selector
is unchanged and does not recognize this value: no GPU timestamp commands,
serialized queue override, or rolling-disable override is added. The same fixed
public third request/layer24 activates just eight wall/thread-CPU marks. Its
public diagnostic identifies`host_only_normal_schedule=true`; actual timestamp
calls and GPU class samples must be zero. Numerical math, full memory ownership,
barriers/timelines, schedule, budgets and erasure are unchanged. All125 selected
CPU controls pass including the original123 plus all host admission combinations
and a witness of the unchanged original GPU/rolling selector. Exact-source
420/208 offline,171 GPU, full public requests and physical recovery follow.
Normal-path results, not the serialized sampler alone, will choose the next
connected optimization. No API/frontend/install/release change.


### Complete push-template encoding candidate

The next candidate retains the complete bundle and all original descriptor view,
owner/range/type/count checks. On the existing explicitly selected push path,
Vulkan1.1+ devices with all actual required entry points may create one
PUSH_DESCRIPTORS descriptor-update template per immutable pipeline layout. A
fixed32-entry stack array describes the SAME ordered storage/uniform bindings,
one descriptor each, reading complete VkDescriptorBufferInfo values at their
actual C-stride offsets. Each dispatch still validates and writes EVERY original
view and calls the command with that pipeline's own template/layout/set0.
Only rebuilding WriteDescriptorSet structures per dispatch is removed. No
shader/math/fault/queue/barrier/source/recording/epoch/budget/erasure change.
Template creation failure or unsupported device keeps the original complete
push implementation; non-push devices keep the original fresh-set path.

Templates own no model content, buffer handles or data pointers; they are
layout-only driver objects destroyed with their pipeline after original joined
completion. Unknown completion preserves the entire original Graph as before.
The fixed creation array is explicitly added to the existing conservative stack
reservation. Anonymous aggregate template dispatches extend only the existing
compiled-public laboratory profile, never inference API/private tracing.
Official contract: Vulkan vkCmdPushDescriptorSetWithTemplateKHR and
VkDescriptorUpdateTemplateCreateInfo reference pages (Vulkan1.1 plus enabled
VK_KHR_push_descriptor); actual instance/device versions and function pointers
are checked. Original per-call type/view guards remain in both implementations.
CPU controls compare every template offset/type/binding and decoded live input
view against the original complete writes, including malformed extent/type
rejection. Original GPU references and full requests remain authoritative.
This is an unselected backend candidate; no installation or speed claim.


All128 selected actual App CPU controls pass (original125 plus three template
controls). Every original packer test remains intact; the original writes and
template route share the same complete-view/type validator. Source changes add
one optional layout-only handle to each pipeline, checked effective instance
and physical Vulkan versions plus all three actual entry points, and explicit
post-completion destruction. Creation/unsupported fallback remains the original
full writes; no partial descriptor update/cache is permitted. Anonymous template
dispatch totals are appended after all old laboratory counter indices, preserving
the original schema fields and old index meanings. Capture stack reservation
includes the actual32 template entries in addition to every old term. Exact-source
GPU/full-request/normal-path host clock and clean selection gates remain next.
Primary references:
[push command](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPushDescriptorSetWithTemplate.html),
[template creation](https://docs.vulkan.org/refpages/latest/refpages/source/VkDescriptorUpdateTemplateCreateInfo.html).


Normal-clock source`d88c904f6d38f2c208060b0f7a6fe836b03d9714`, frozen SHA
`84373929977b9ceb6b0d8954f4be6991ed3593c1e6f5d518db888c72fde98fd9`, passes
125 CPU,420/208 offline,129+42 original/role-image GPU controls, the complete
original public proof chain, then the new normal-schedule fixed-public run.
Same73/4245/4245 inputs, seed0/temp0, correct323, exact7663/26673/26673 groups.
Host-only mode actually emits ZERO GPU timestamps and zero class GPU samples,
keeps84 real rolling recordings/1491 windows/37959 segments. It samples the SAME
third-request/layer24/count4096 window: actual CPU encoding9.183503ms, retained
preparation8.361003ms, queue planning2.012011ms, compute dependencies1.532520ms,
inventory0.671660ms, snapshot0.325420ms, copies0.118290ms, finalization0.026190ms.
All eight real thread clocks available. Encoding+retained are79% of sampled
thread CPU; this is sampled CPU attribution, NOT whole-request GPU time.
Normal diagnostic firstlong12.2337s/repeat10.7068s; not clean selection evidence.
Actual independent20:15:45UTC: Intel50,688,000B,NVIDIA compute0,all faults0,
allHTTP200, admission inactive/unpaused/recoveryfalse/waiting0. Source/installed
identities unchanged. Owned summary`strata-parity-host-clock-normal-20261020-public-summary.json`.

Complete template source`03815d0b` is frozen in
`strata-parity-push-template-20261020`;128 selected CPU and420/208 offline pass.
The protected exact-source171-GPU/full-proof and both original-GPU-sampler /
normal-host-only full request runs are active. All original guards, references,
source IDs and thresholds are retained. No clean/staged/install result yet.


Template source03815d0b8f47c547bf4729927dd2f98f9dd8a974 / frozenSHA
f0c926f285eea8f8f7c225ff580564da10a83699a5e353077cc9db5b1663eb82 passes
128 CPU,420/208 offline, original129+42 GPU, full original proof chain and both
three-request GPU-sampler / normal-host-only runs. Actual template dispatches
46746/166220/166220 equal EVERY original push dispatch in both modes; fresh
sets0. All original full source/math/threshold/count/ownership checks remain.
Normal diagnostic firstlong11.9980s/repeat10.6374s vs prior12.2337/10.7068s.
These are diagnostics, not clean speed acceptance or a Strata parity claim.
Sampled encoding CPU9.184->8.501ms, retained8.361->7.540ms; the latter preparation
code is unchanged, so do not attribute all timing variation to templates.
Actual independent20:23:40UTC recovery is clear/healthy: Intel50,688,000B,
NVIDIA compute0, all faults0, HTTP200, normal admission. Installed identities
unchanged. Owned`strata-parity-push-template-20261020-public-summary.json`.
Protected terminal-negative/fresh clean45/4245/4245 acceptance is next, retaining
original5%-faster at EACH first/repeat threshold. Staged API/semantic acceptance
runs only after that unchanged clean gate passes. No installation yet.


Next candidate direction after the template clean gate: completed recording
resource reuse. The original command allocation is ALREADY bulk per actual pool;
never propose another bulk-allocation fix. Retained-preparation CPU includes
original Commands::new (copy/shared pools, three timeline semaphores and bulk
command-buffer allocation), actual owner/recording transfer, boundaries and
rolling preflight. Attribute any proposed improvement to work actually removed.
A reusable resource must be extracted only at the SAME original idle eviction
point that already permits destruction, after the original flight book proves
not live; named live/private ownership and unknown-completion quarantine stay.
No reset on a merely submitted/ready flag. Actual queue families/header/pools,
all command handles, monotonic timeline next_base, actual allocated capacities
(including old+new simultaneous scratch), source/header/buffer identity,64MiB
caps, complete re-encoding of every descriptor, all original barriers and final
join fences must remain authoritative. Current source DOES NOT implement reuse.
If full pool recycling proves too intrusive, peer-pool/peer-command reuse with
identical actual per-queue capacity is a smaller coherent step; MAIN stays owned
by the new original recording pool/header. Any compatibility miss keeps original
construction. No unchecked cache of private content or skipped erasure.


### Template clean selection and safe checkpoint

Exact source03815d0b8f47c547bf4729927dd2f98f9dd8a974 / frozen worker
SHAf0c926f285eea8f8f7c225ff580564da10a83699a5e353077cc9db5b1663eb82
passes the actual submitted-prefix terminal-negative check: expected worker exit1,
fired1, TWO actual recovery queues waited, failed queues0. Intentional process
exit/quarantine remains part of that test; no ordinary reusable-session/owned-zero
claim replaces it. A subsequent FRESH clean45/4245/4245 process completes all
three correct323 requests and physical recovery, but FAILS the unchanged clean
selection gate:

| Position | Installed native32 | Complete template candidate | Original5% gate |
| --- | ---: | ---: | --- |
| First long |12.962279132s|13.780554488s|FAIL |
| Repeated long |11.147390805s|10.383566824s|PASS |

Repeated long is6.85% shorter than installed, FIRST is6.31% slower. Do not
cherry-pick repeat, change warm-up input, move JIT work out of the clock, change
thresholds or claim Strata parity (matched repeat5.3515s). The prior compact
candidate was13.7110/10.2885s; no meaningful whole-request template gain over that
candidate is established. Every prior component is retained; none rolled back.
Clean cold model load117.6644s, first45-token request10.4057s and3.5317s output;
these remain separate metrics. Staged API/semantic acceptance was NOT STARTED,
installation was NOT STARTED, and no new sustained/decode/frontend/release
acceptance is claimed. Owned summary:
`strata-parity-push-template-clean-selection-public-summary-20261020.json`.
All helpers/artifacts are under`/media/jmicko/faststore/kuno-development`;
acceptance prefix`strata-parity-push-template-acceptance-20261020`.

Independent actual20:30:46UTC state: Intel50,688,000B,NVIDIA compute0, every
kernel fault counter0, Kuno/Ollama/Vashti200, admission inactive/unpaused,
recoveryfalse/waiting0. Both protected units ended. Installed worker3fffc266/
SHAeac78ce0 and gateway36626d30/SHAd3ed07f4 remain untouched. No GPU/device reset
or reboot was used. Goal remains active/unmet. No GPU test is left running.

Additional fixed-public LOAD attribution is owned at
`strata-parity-push-template-public-load-phases-20261020.json`: labels are START
arrivals, not per-stage completed timers. Target0.6196s, global_sources2.3877s,
shared_target_weights76.5421s, shared_draft77.3815s, request_rebind_begin80.4441s;
worker load80.4315s in that separate diagnostic. Thus the interval containing
original global expert source construction is~74.15s. Do not blame all load
cost on shader compilation or infer disk byte volume/bandwidth from phase time.
Actual source pointers: Sources::load in strata_global_sources.rs sequentially
loads Cache, Host, builds tables, optional exchange inventory and verifies files;
strata_global_gpu.rs / strata_global_host.rs retain the complete parallel-reader
and original ownership/publication fences. No loading change is implemented here.

Next work retains the COMPLETE candidate, focuses on recording resource
preparation and first-use cost, and requires actual removal/reuse of a measured
cost. Commands already allocates in bulk per pool. Any reuse must preserve the
original completed-flight authority and cap OLD+NEW simultaneous allocated
metadata, pool families/header/command ownership, monotonic timeline epochs,
full fresh descriptors/real barriers/final fences, original private reset and
erasure boundaries. A Vulkan command-pool lifecycle reset, if implemented only
after that authority, is not a device/driver reset; device/driver resets remain
prohibited. No reuse code has been added yet. Keep root-required work bounded:
owned sudo refresh128882/128884 expires20:39:26UTC (sharedshell3556 is NOT ours
to kill); if sudo later blocks meaningful protected progress, leave this safe
state and stop that run per goal policy rather than repeated workarounds.


### Complete idle-recording re-encoding candidate

Reuse only an original idle, ready, unnamed, unprofiled MAIN recording in the
SAME Graph/family, after the original flight book proves it not live. Only the
existing large expert rolling class and full32-binding push support qualify.
New/old actual queue counts must match exactly, all original commands/header
must be present, and the original64MiB retained/command metadata preflight must
still pass. Mismatches retain original construction. No new cache/heap buffer.
Reuse keeps that recording's original three command pools, complete command
inventory, descriptor pool and timeline semaphores/monotonic next_base. It resets
only those completed command pools, then re-encodes every original node, live
binding, real range barrier and final queue fence. Different per-queue ordering
is handled by an allocation-free permutation: old plan storage is scratch only
AFTER the recording is made not-ready, and is replaced by the complete new plan.
Header stays in its original MAIN pool. All actual command handles stay with
their original physical queue family. No arithmetic/source/precision change.

The physical source guard is released before retiring any old resource owner,
then reacquired at the original encoding boundary with full current validation.
No guard crosses owner retirement. Original request erasure clears the same
recording inventory; unknown completion/quarantine never authorizes reuse.
Actual old+new metadata is conservatively charged before any reset. The newly
created but unsubmitted scratch pool/descriptor pool are discarded only once
original pool ownership has transferred into the replacement recording.
Anonymous actual reuse counts extend only the existing public diagnostic.
No API/protocol/model/config/frontend/install/release change. All prior coupled
components, numerical references, descriptor/owner/hazard/epoch guards remain.


Implemented re-encoding retains the existing Recording object and its original
MAIN/COPY/SHARED pools, descriptor pool, commands and timeline next_base. It
requires the original flight-book not-live test, exact family/header identity,
ready/unnamed/unprofiled state and original large expert rolling eligibility;
newly created scratch pools must be unsubmitted, with full32-binding push support.
The same old+new64MiB preflight remains conservative, including actual old command
Vec capacity. Matching queue COUNTS allows a different queue ORDER. Three linear
partition passes and permutation cycles reuse only the old plan's existing node
index scratch; no new heap allocation or command handle is needed. Every original
pool/command remains owned by the same recording. It is marked not-ready before
any pool reset and every original node/binding/barrier/fence is re-encoded.
Normal incompatible requests retain the original construction path. No old
recording survives original private erasure via a new side cache.

One CPU compile caught the retained metadata variable needed by later fresh-set
window preflight; corrected by carrying the SAME conservative checked metadata
through both construction branches. No failed build was run on GPU, no capacity
check was removed. All132 selected actual App CPU controls now pass: prior128
plus exhaustive small queue permutations, large32769-command mixed layouts,
pre-mutation malformed inventory rejection, and checked actual-capacity budget
boundaries/overflow. Exact-source original171 GPU/full proof/normal-host profile,
physical recovery and subsequent clean/terminal/staged/install gates remain next.


Exact-count source cdbb041b / SHA82e95810 completes all132 CPU,420/208 offline,
original129+42 GPU, original producer/compiler/count proofs, BOTH whole-model
profile modes, correct323 and physical recovery, but FAILS the new actual-path
exercise gate: ZERO recordings reused in every request. These completed timings
are FALLBACK behavior, not proof of re-encoding correctness or a gain. Preserve
`strata-parity-command-reencode-20261020-reencode-public-summary.json` with
passed=false. No clean/staged/install acceptance was run. Normal first11.9949s,
repeat10.6238s remain diagnostic only. Actual20:55:58UTC clear/healthy, allfaults0.
Live read-only vulkaninfo identifies B70 vendor8086/devicee223 with actual
maxPushDescriptors32; both hardware GPUs report32. A supposed device16 limit is
NOT the observed reason. This read-only capability query saved no configuration and requested no reset.

The complete extension retains exact no-allocation permutation when applicable,
and admits a completed recording with SUFFICIENT original per-queue capacity.
Unused original command handles stay in explicitly charged per-queue spare Vecs;
no GPU buffer/command/pool/handle is added, borrowed or fabricated. New CPU handle
vectors are prepared before marking not-ready/resetting anything, with original
old+new simultaneous capacity, headers and temporary preparation storage charged
against the original budget. Allocation/capacity mismatch keeps original fresh
construction. Real model data/resources never enter this metadata cache.
Private reset/erasure still clears the original recordings, with no side cache.
Each queue retains its actual original handles across shrinking/growing plans,
all original descriptor writes/barriers are re-encoded, next_base is unchanged.
The original exact path remains for matching inventories with no inactive handles.

Eligibility now records actual push-only status per recording and verifies the
ACTUAL new node layouts, rather than assuming all potential32-binding layouts
are used. One conservative word covers the new flag/padding. Anonymous workload,
fresh-pool and idle-candidate counters distinguish an unused selector from
capacity misses. Actual capacity-reuse and retained-spare counts must be nonzero
in long requests before this extension can claim runtime validation. Two new
CPU controls exercise repeated shrink/grow/permutation with full handle/physical
queue conservation and non-mutating rejection of insufficient capacity, unknown
handles and budget/overflow. No numerical reference, threshold, private ownership,
header/family/epoch/budget/erasure or timing gate is weakened.


All134 selected CPU controls pass before the final conservative push-only flag
word was also charged for the incoming recording (old recordings already charge
it). Final-source recompilation/control run follows; this adds capacity accounting,
not a guard exemption. The prepared inactive-handle route includes both actual
Vec capacities and preparation-stack metadata; no allocated backend handle can
vaporize when fewer commands are active. Repeated CPU shrink/grow reaches the
full original4096-handle inventory again, checking every handle exactly once and
its original pool family at every stage. The two original exact permutation
routes and all previous numerical checks remain intact.


### Coupled completed-resource retirement and adoption

Capacity source2137a0f1/SHA7ee191f9 passes134 CPU,420/208 offline,171 original
GPU controls and both full original request/proof chains, but again FAILS actual
path exercise. Counters establish eligible/fresh captures18/84/83, idle candidates0,
reuse0. The issue is absence of an idle eligible recording at selection, not just
queue capacity. No speed acceptance or installation. Earlier exact-count-cause
inference was insufficient; retain these failed exercise receipts honestly.

The coupled successor preserves all components and connects resource reuse to
ORIGINAL idle retirement points (unnamed LRU eviction and completed I/O cleanup),
plus performs the original new-plan access joins before attempting reuse. It
retains at most ONE typed IdlePools object: device/Graph/family identity, original
MAIN/COPY/SHARED pools, original timeline semaphores/next_base, header and per-queue
command HANDLE inventories only. It contains NO Plan, signature, resource owner,
model/input/route/expert value or prompt state. Before transfer it uses original
completed-flight/all-queue evidence and resets those original nonpending command
pools. The same old recording resources/signature/plan are then destroyed, before
original I/O sanitization/zeroization proceeds. No erasure or owner check is skipped.
Normal Graph destruction drops this content-free device allocation; unknown
completion retains it conservatively until protected-process teardown.

Cache adoption requires exact original Graph/device/family/header ownership and
sufficient per-queue capacity, performs the same checked old+new metadata/temporary
reservation, and builds a NEW Commands with the COMPLETE original fresh Plan and
all fresh current owners/bindings/barriers. All original epochs remain monotonic.
Only already-reset driver objects are reused. No new backend buffer/command/handle
is allocated by adoption, no fake executable plan or completion token is created.
The original prepare_last_recording still enforces its actual plan access/fence/
INFO epoch checks after adoption. Earlier joins use that SAME new-plan access
inventory, after source leases are released; no weaker proxy or host flag grants
completion. No API/config/model identity/frontend/release change.


The coupled implementation also preserves the original protected-draft-source
alias rejection BEFORE earlier access joins; prepare_last_recording repeats its
original admission/fence/INFO checks. Original private cleanup still clears the
entire recording/signature/plan/resource inventory. The only survivor is a typed
IdlePools value with raw device/Graph/family IDs, reset pools, semaphores and
command-handle inventories. All136 selected actual CPU controls pass, including
exhaustive field destructuring/type checks that reject adding private buffers,
values or plans to that cache, and source-bound witnesses of real flight/exports/
handoff joins and source-lease release ahead of retirement. Actual old+new cache
and preparation capacities remain charged inside the original64MiB checks.
No device reset/reboot, numerical-reference, threshold or privacy-guard change.
A source-edit helper initially missed a formatted marker and a closing helper
brace; corrected before the successful build. No partial/failed build ran on GPU.
Freeze,420/208 offline,171 actual GPU/full request and actual retirement/adoption
proofs follow. Numerical fallback success never substitutes for cache execution.


The ebd753ed exact coupled source was frozen and built before extending adoption
across small unkeyed MAIN captures; no GPU run of that intermediate freeze is
claimed. A large reset pool must service small captures too to remove their
repeated pool/semaphore setup; its inactive original handles then keep that pool
eligible for retirement again. Named, sampled or different-family work retains
original paths. All new plans, source/owner/descriptor/hazard checks remain full.
Public reports add clearly named PROCESS TOTAL retirement/adoption counters,
since I/O cleanup can retire pools between expert timer intervals. Private calls
never enable that report. Per-expert counters retain their original delta scope.
The same136 controls and full exact-source GPU/retirement/adoption proofs are
required on the final combined source before clean performance acceptance.


### Completed-pool reuse runtime and clean selection outcome

Exact final source865a9c166acc83203fd82c4052a2f6ee2e2240fd,
workerSHA361978398e3c798a6c59eb09dc091719d0d6862787b5ea72a5ab95175c9c3a28,
build`strata-parity-idle-pools-coupled-20261020`:136 selected actual CPU controls,
420 total offline shaders/208 matrix layouts, original129 GPU+42 exact-role image
controls, all original source/owner/producer/compiler/count/budget guards and BOTH
whole-model public profiles PASS. Actual cache reuse is now exercised: normal
expert intervals27/89/96 re-encoded records,3912/36183/40107 original commands,
1062/33777/41109 spare command observations. Process-total retired/adopted pools
61/60,179/178,297/296 across requests; every long request increases actual
adoptions. Counts are accumulated observations, not distinct backend allocations.
Small captures now qualify, so recording/command totals must not be incorrectly
bounded by large-only rolling counters. The original large rolling counters and
all producer/numerical assertions remain unchanged. New exercise proof is true.

Normal sampled layer24 total threadCPU(ns): retained_prepare117090,
command_encoding6072022, queue_plan1950090, compute_dependencies1513390,
inventory_signature666700, snapshot_prepare318230, copy_dependencies119791,
finalize25310. Earlier retained/encoding8.36/9.18ms become0.117/6.072ms; this
proves lower actual targeted CPU overhead, not clean whole-request acceptance.
Normal long prompts11.608595581/10.128802681s; expert phase7.224606472/5.904627620s.
Third normal attention2.949376696s; QSA attention/merge1.505852085s and expert
postcapture wait3.538181488s remain substantial. GPU-sampled class sums serialize
one layer; do not interpret their shares as normal overlapped critical-path time.

Original terminal-negative test PASS:2 real recovery queues waited,0 failed,
protected worker exit required; no reusable-session/zero-owned claim. The fresh
clean45/4245/4245 requests PASS with correct fixed-public arithmetic, full output
frames and actual physical recovery. However unchanged speed SELECTION FAILS:
firstlong13.247788385s versus installed12.962279132s; repeat9.680801649s versus
11.147390805s. First is2.20% slower; repeat13.16% shorter. Never hide first-use
compilation, change the short warmup, select just repeat or relax5%-at-EACH gate.
No staged API/tool/semantic/sustained acceptance or installation was started.
Prepared installation and post-install helpers are explicitly UNEXECUTED.
Installed worker3fffc266/gateway36626d30 remain; Strata repeat5.3515s still leads.

Owned fixed-public summaries under`/media/jmicko/faststore/kuno-development`:
`strata-parity-idle-pools-coupled-20261020-reencode-public-summary.json`,
`idle-pools-coupled-full-public-counters-20261020.json`,
`strata-parity-idle-pools-coupled-acceptance-20261020-clean-selection-public-summary.json`.
Actual21:47:13UTC: Intel50,688,000B,NVIDIA compute0,all driver faults0,allHTTP200,
normal admission. No GPU job remains; no GPU/device reset/reboot. Goal ACTIVE/UNMET.

Next coherent candidate should address matrix QSA math/staging together, using
pinned MIT upstream`qsa_prompt_attn_xmx.dp.cpp` v2 for design comparison. Its hi/lo
Q/P, exact INT8 codes plus per64 FP16 scales, packing, softmax and PV interact.
The existing short/scalar references and original bitwise/F64 bounds must remain;
any large-only selection needs ADDITIONAL actual large-path independent F64,
full-owner/tail, nonfinite, causal/offset and fault validation. Old <=8-row tests
using fallback cannot prove a new >=16-row kernel. Validate real shared storage,
actual CMMA subgroup support and full original source/owner/epoch/erasure guards.
No XMX code is implemented at this checkpoint. No API/protocol/configuration,
model identity/frontend/version/release change.


### Complete matrix-attention successor under implementation

The next opt-in `strata-qsa-xmx` retains the entire865a9c16 bundle. One Rust
CubeCL kernel implements QK hi/lo half staging, exact INT8 codes/per64 half scales,
per64-cell softmax and scaled probability hi/lo PV as a connected layout. It
writes the SAME partial/stat/fault owners consumed by the original merge; no new
private buffer, model source or persistence. Original short(<16-row), scalar,
decode and MTP paths remain. Selection requires actual8x16x16 half->F32 CMMA,
128 threads and58,660B real SLM; unsupported hardware keeps the original path.
Full private owner/offset/causal/fault admission remains. No speed claim yet.

Keep the original twelve small numerical controls and exact F64 bounds. Add a
separate large control (16/31/32 rows, offsets0/32/64, long32768-capacity boundary)
that MUST launch actual CMMA and checks both original scalar and candidate against
the SAME independent mathematical oracle. The original8-row oracle entry keeps
its domain; a new32-row entry shares identical loops and derived error bounds.
Negative future-selection/nonfinite controls must preserve sticky fault and full
inactive/tail zeros; do not claim failed-input payload math is accepted. Original
shader references and old bitwise requirements remain unchanged. Public reports
record anonymous actual XMX launches only through existing compiled-public gate.
No release/install/API/config/model identity/frontend change is authorized by this
experiment; full exact-source runtime/clean/staged acceptance is still required.


Final-source CPU compile and144 selected controls PASS (all previous136 plus
original QSA geometry/oracle controls and new tile/split tests). Initial CubeCL
compilation exposed constant-expression expansion and an unsupported early-return
form; corrected before any device execution. Half-scale decoding uses exact
32-bit exponent/significand construction, so no unrequested shaderInt16 feature
is introduced. The original mathematical oracle loops/operation-derived bounds
are unchanged; its old eight-row entry still rejects larger input. New large
compiler delimiters are distinct, preserving original scalar/FMA ISA scopes.
Next: freeze full bundle, validate420 WGSL variants plus216 matrix layouts,
original171 GPU controls plus7 new actual-XMX controls, both request profiles,
actual launch-count proof and physical recovery. No GPU or speed result yet.


First XMX sourcee2935521/SHAa38e2801 passes144 CPU and420 WGSL/216 matrix offline
variants, but the protected operator run FAILS BEFORE actual XMX launch: selector
correctly rejects58,660B SLM. Read-only live vulkaninfo proves B70 devicee223
maxComputeSharedMemorySize=49152B (NVIDIA likewise49152). Do not infer that SYCL's
larger local memory is exposed by Vulkan; no device-limit/feature/ownership guard
was bypassed. Failed run exits and physically recovers; no whole-model or speed
run occurs. Preserve`strata-parity-qsa-xmx-first-failure-public-20261020.json`.

The connected correction stages8 query heads at a time, processes two K scale
groups concurrently across all8 original subgroups, and processes128 PV dimensions
at a time using paired probability images in the same Q hi/lo allocation. K/V
still share one64-cell code image only AFTER full original workgroup fences.
FP16 scale words are read from the original cache rather than kept in a new SLM
array. The physical requirement is now48,420B, inside the ACTUAL49,152B limit.
All12 heads,64 cells,256 dimensions, per64 scales and external original slots
remain; no precision/source/fault/causal/oracle/acceptance guard is relaxed.
CPU tile coverage is updated to the real coupled layout; original144 controls,
all offline variants and original171+7 actual GPU controls must pass again on
this exact correction before any whole-model speed/selection claim.


48KiB source3ac1f893/SHAdcc438de again passes144 CPU and all420/216 offline
variants. Actual hardware admission now succeeds, but the new public fixture's
FULL padded fault allocation violates production's strict exact4-byte fault
view, so the operator run stops with InvalidBufferLength BEFORE XMX execution.
No math/performance acceptance. Physical recovery passes. The correction derives
the genuine logical fault handle from its original arena BufferId, as production
already does. The entire authentic parent is retained and read afterward for
padding-zero checks; the production exact4-byte check remains UNCHANGED. No fake
length/view, source/owner/epoch waiver or test threshold change is introduced.


### First actual matrix-attention execution and precision checkpoint

Final logical-view source585dc471c56879466a9bcbe40f6a8229ff06e6a9 /
SHAfd43c5cb4cece90f4987f50a71d482072a3d2ca21fae6ea80eaff9b762ca03fb
passes144 CPU and420/216 offline. Hardware selector and authentic logical fault
view now pass, and actual ordinary+retained native compiler scopes each show
SIMD16,48 DPAS,5777 instructions,0 spills/fills. First large causal(start0,count16,
capacity4096,offset0) executes, but the original independent F64 bound FAILS.
Original scalar mismatches0; candidate partial mismatches78071,max_abs
0.0038284343515728736,max_error_to_bound25.629098220813116; candidate stats
mismatches696,max_abs0.0018138062745212125,max_error_to_bound46.61242040278987.
Full original owner/tail/fault checks precede this F64 comparison. No whole-model
profile or speed test follows a failed numerical acceptance. No install.

Postmortem public summary
`strata-parity-qsa-xmx-logical-precision-postmortem-public-20261020.json` preserves
passed=false and physical_recovery_passed=true, exact source/hash and real ISA
rows. Root wrapper stops before all downstream proof/acceptance stages. Earlier
58KiB and padded-fault failures were unexercised; this one IS actual math failure.

Additional actual-compiled SPIRV CPU control finds exactly2 complete
F32->F16->F32 conversions feeding F32 residual subtraction (Q and P). The entire
selected145 CPU set passes. This rules out the simple CPU-optimizer-dropping-casts
hypothesis at SPIRV level; it does NOT prove the driver's dynamic rounding or
which matrix/scaling step causes the error. Next fixed-public GPU control should
measure the actual shared hi/lo stage against host half rounding, then verify
DPAS operands, scale groups and accumulation. Keep all original F64 bounds,
old12 references and new7 full-batch cases. No weakening/fake fixture/partial
component rollback or performance-led abandonment is acceptable.

Independent22:25:14UTC: Intel50,688,000B,NVIDIA compute0,all driver faults0,
allHTTP200 and normal admission; no GPU job remains. Installed native32/gateway
unchanged. New bounded sudo refresh297061 ends2026-10-21 00:17:49UTC, owned parent
sharedshell3556 must never be killed. Goal ACTIVE/UNMET. No reset/reboot, new API,
configuration/model identity/frontend/release change.
