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
