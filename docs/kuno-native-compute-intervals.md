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
