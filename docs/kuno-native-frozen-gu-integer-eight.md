# Frozen GU-only packed integer experiment

This source leaf is an explicit alternate mathematical candidate. It retains
the original DASLab packed gate/up weights, real private rank MAP and current
DOWN/GDN/router behavior. It does not claim original64 F32 equivalence,
whole-model numerical acceptance or measured speed. Original CPU13 tolerances
and target-greedy verification remain unchanged.

Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
`sycl/src/kernels/cuda/iq_kernels.dp.cpp` uses Q8 activation blocks, packed
integer dot products and eight lanes per grouped projection row. The native
frozen control instead reconstructs each packed weight as F32 and retains the
original64 lane tree. This leaf reuses Kuno's independently C-fixtured original
integer decoder and its already declared two-residual-Q8 quantizer; it does
not copy Strata's single-Q8 half scales or its integer-division finish formula.
Packing/quantizer provenance and MIT notices remain in
`iq_moe_integer-NOTICE.txt`.

`quant_matvec::frozen_gu_integer` provides only preparation, GU projection,
full-owner audit and pure preflight functions. Parent integration must select
it explicitly and immutably on frozen sources with partition exit, count1..3
and a full wave starting at token0. No default source selection changes here.
It rejects competing private/workgroup tables, GU factors, two-row GU,
three-member geometry and live-member reduction. Parent must additionally
reject packed HOST staging and shared-DOWN scales in the first composed trial.
Four-row DOWN remains compatible and unchanged.

The original five public partitions remain `(first_slot,slot_count)`
`(0,72)`, `(72,72)`, `(144,144)`, `(288,144)`, `(432,80)`. GU launches
`(80,groups,1)` with64 physical lanes: eight paired output rows, eight lanes
per row. `groups` is a bounded launch envelope1..80; actual MAP groups,
membership, slots, ranks and count remain runtime data. Inactive partitions
return uniformly before source payload reads, initialization and barriers.
Weights are decoded as four signed bytes, reused across both residual planes
and up to three RHS members. Each lane processes32-value blocks in ascending
stride8 order, accumulates bounded i32 dots, applies the existing F32 integer
factors and plane scales, then uses a fixed F32 tree4/2/1. This is the new
declared arithmetic. The stable native SwiGLU expression, ACT addresses and
feature0 ready status are retained; existing downstream finite/fold/acceptance
proofs remain mandatory.

Allocate `SCRATCH_WORDS=16384` as the exact logical request to obtain one full
64KiB private owner. `LOGICAL_WORDS=4384` describes meaningful frames only and
must not be passed to the allocator: it rounds to32KiB. Header words0..2 are
readiness flags, header3..63 are zero. Three1440-word frames begin at64:
640 first-plane packed words,640 residual-plane packed words,80 first F32
scale words,80 residual F32 scale words. Inactive frames and4384..16383 remain
zero. Prepare runs `(1,3,1)`,128 lanes, once per real layer. Each token
workgroup exclusively clears and publishes its own readiness flag and frame;
workgroup0 clears only header3..63 and the owning tail. There is no cross-group
header race. Every shared cell is initialized explicitly; the runtime does
not zero workgroup memory.

The project uses the prepared owner directly from storage. It does not copy
all three frames into per-workgroup memory. Original public codebooks receive
literal lane/index/value writes, once per group, and share one uniform
initialization barrier. Reduction storage is1536B, plus the largest8192B
IQ2_S grid; prepare uses16004B. Parent must check device limits and physically
authenticate all complete, disjoint, same-client/stream roots, source binding
identity, immutable pending ranks, epochs, all owning tails and64KiB budget
before enqueue. The supplied Handle checks are supplementary to those proofs.

`prepare(client,geometry,spans,owners7,scratch,count)` clears the whole scratch
owner itself. `launch(client,geometry,spans,banks2,owners7,scratch,first_slot,
slot_count,groups,count)` replaces one source's GU call only. `audit(client,
scratch,fault,count)` scans every word and ORs sticky fault8 on unfinished,
nonfinite-scale, inactive, header or tail corruption. Parent should prepare
once, run the original five GU/DOWN source pairs with GU replaced, audit, then
retain all original finite/fold/reduction/transaction commit gates. Scratch
must participate in startup inventory, actual runtime physical proofs, reset,
full owning zero-close and memory recovery. INFO/runtime/host increments are
separately exposed by the leaf; no owner reuse or uncharged allocation is
assumed.

Focused CPU tests cover all48 actual mixed layers/five partitions, competing
geometry rejection, full64KiB allocation, disjoint producer reset ownership,
actual WGSL/Naga/SPIR-V with workgroup zero initialization disabled, signed
packed-dot instructions, full grid word equality, no Function arrays,
declared initialization/barriers, guard-negative mutations and eight-row/
three-real-route independent quantized projection vectors. Physical operator,
CPU13 and matched full-language trials remain parent-owned and pending.
