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
