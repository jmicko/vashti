# Bounded fresh-descriptor batches for native capture

This is a backend-only source candidate based on `3fffc266`. It changes no
Vashti/API/configuration/model/context contract and retains no prompt, route,
image or output content. Source ownership is isolated in
`kuno-prefill-transfer-review`; root owns GPU acceptance and installation.

## Measured target and source difference

The fixed-public native-wide profile at
`strata-parity-native-wide-20261020-profile/receipt.json` reports26,338 active
expert groups in its final4245-token request: expert enqueue1.577s,
native capture1.861s and native wait3.696s, with7.313s total expert phase time.
The subsequent root native32 profile reports capture1.906s. These public
measurements motivate reducing CPU capture overhead; they do not attribute all
capture time to descriptors or guarantee a one-second saving.

Every expert currently queues six compute kernels: gather, two dequantizations,
GU, activation and DOWN. Thus26,338 groups produce158,028 compute nodes before
other phase work. `Graph::execute` originally called
`vkAllocateDescriptorSets` and `vkUpdateDescriptorSets` once per compute node,
constructing separate info/write vectors each time. The selected pinned Strata
path directly enqueues dequantization, oneMKL GU, SwiGLU and oneMKL DOWN on its
stream; no source-visible equivalent of Kuno's per-node native descriptor
allocation/update loop exists. Its opaque driver/backend costs must not be
inferred to be zero.

## Complete candidate

A window contains at most128 **original nodes**, including Copy/Fill nodes.
Every compute node still receives its own newly allocated descriptor set with
its original exact pipeline layout. Copy/Fill keep their original node indices
and receive no descriptor. The descriptor pool, command buffers, full resources
and signature already belong to the original Recording before any new bulk
operation; Vulkan set lifetime and release remain pool-owned.

For each window, the production helper validates the original ordered binding
counts, actual layouts, complete owner intervals and buffer identities, then
finishes immutable info/type arrays. It allocates all fresh sets in one Vulkan
call and builds the actual write array through `Plan::writes`. This returns
`VkWriteDescriptorSet` values borrowing the completed immutable info array;
Rust forbids growing/replacing that array while the writes are live. One update
call initializes only these new, never-bound sets. Every subsequent bind and
dispatch remains in its original node order.

There is no descriptor cache, set recycling, update-after-use, authority reuse,
new completion credit, command compaction or hazard narrowing. Existing actual
range preflight, per-node barriers, queue/segment boundaries, INFO timeline
waits, rolling-prefix guard, final fence and failure quarantine are unchanged.
A failed later window reaches the same armed rolling recovery; fresh unused
sets remain in the already-owned pool until real submitted users drain or are
quarantined. Dropping a completed host window drops only handle/index vectors,
never Vulkan sets or original resources.

The new helper bounds128 sets,32 bindings per compute node and4096 binding
infos/types/writes. It computes costs using actual Rust Vulkan structure sizes,
with a conservative4KiB reserve for headers/stack state. Maximum new temporary
metadata is387,072B on this build, below the explicit512KiB limit, and is also
charged against the existing64MiB aggregate metadata limit before allocation.
The previous host window is dropped before preparing another, so two transient
windows are never charged as one. Vulkan descriptor counts/set counts in the
original recording pool remain unchanged.

## CPU evidence and remaining gates

Four tests run the **actual production plan and Vulkan write builder**, linked
to the same Ash library: unique original node/set/binding order; bounded window
and bulk-call inventory up to32,768 nodes; real metadata caps and incomplete
inventory rejection; and actual buffer/offset/range/type/pointer preservation.
Synthetic handle values are CPU fixtures only, with no GPU completion claim.
A separate production-helper footprint invocation reports128 nodes,128 sets,
4096 bindings and387,072B peak metadata. Full worker `cargo check` with the
native32 feature passes in the isolated target (28.18s). The original queue,
rolling, I/O-owner, binding, parameter and transient-erasure fixture filters
also pass154 CPU tests (0.25s); no fixture was disabled or substituted. The
artifacts are `descriptor-whole-check.log` and
`descriptor-original-owner-controls.log` under the isolated
`prefill-cubek-cpu-20261020` directory. No GPU was used by this task.

Root's protected acceptance must retain original numerical/owner/erasure and
actual failure-recovery gates, then profile the complete connected candidate.
The source review by the independent runtime agent found no clear pointer,
pool-lifetime, rolling-failure or command-order blocker. It is source review,
not proof of live Vulkan performance or physical recovery.

## Separate expert-wide assessment

Both existing expert shapes satisfy N128/K32 arithmetic geometry:
GU1280x2560 and DOWN2560x640. Gather/activation already produce member-contiguous
FP16 inputs. The dense wide kernel is not a drop-in expert implementation:
GU must retain route checks and fault128, and DOWN must scatter to route*2560,
maintain the original full-capacity status plane and its unique feature-zero
rule. A genuine mapped epilogue is required.

The latest public histogram has2321 groups with>=256 members (8.8% of groups),
covering1,357,660 of2,037,120 routed members (66.6%). It does not report>=512 or
multiples of128. Therefore the current full-group `count>=512 && count%128==0`
wide selection could hit rarely; no speed estimate follows from that histogram.
A future wide full128-row prefix plus unchanged M32 tail could cover more large
groups, with authentic member bases and original mapped/fault/status stores.
That is a separate candidate after dense-wide ISA/numerical/profile acceptance;
no expert kernel was changed in this descriptor work.
