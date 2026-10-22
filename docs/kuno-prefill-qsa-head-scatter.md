# Ordered twelve-head prompt reduction

Optional complete source continuation,2026-10-22. No build, GPU, speed or
installation claim at implementation. This remains a Rust engine generating
WGSL/SPIR-V; no SYCL/C++ engine is embedded. No API/KNT2/KNR2/model/context/
configuration/frontend change; required Vashti:none.

## Evidence and connected implementation

Pinned MIT Strata fb58e0dbc8399662c0e47c76578c6e878b14f6cf contains
`reduce12` in `sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp`, under
STRATA_EXPERIMENTAL_SM60. It is NOT the selected same-host Intel default.
Do not attribute the5.3515s reference result to this inactive optimization.
It demonstrates a topology reducing12 separate five-step head butterflies
to16 exchanges. Our current selected paired attention has130 static XOR/Quad
exchanges including softmax; this candidate has42, retaining48 PV gathers.
Static counts alone are not a latency or speed result.

`strata-qsa-head-scatter` / `--qsa-head-scatter` explicitly retains the COMPLETE
query-plane, GDN fault-mask, medium-expert, vertical-Quad, native-prepared,
prefetch, raw-bank and cohort bundle. Only the two prompt attention variants
change, before original local-fault closing. Decode/MTP/scalar references stay.

All24 original eight-product dot accumulations and168 chronological FMAs
remain, including valid-cell guards and every original query/cache read. Hold
each head partial in a scalar. Padded heads12..15 are zero. Exchange/reduce
16->8->4->2->1 head values across XOR16/8/4/2, then XOR1. Each lane pair
gets head lane/2. The even lane publishes the identical score/status address.
Both lanes check that head's finite predicate, so the original all-head sticky
fault and original closing/status reduction remain. Invalid cells still use
the original sentinel, cannot supply cache addresses, and publish no score.

Unlike Strata's commutative spelling, every pair explicitly orders the lower
lane subtree FIRST after exchange. This preserves the EXACT original lane-zero
expression tree for each head, including signed zero and nonfinite operands;
no reliance on associativity or finite-only commutativity. Division by16,
softmax/PV chronology, barriers, owners, 12KiB Q layout, workgroup256/subgroup32,
source extents and every original memory-budget/fault guard remain. Native
vertical swaps lower the same XOR2 participants: four swaps plus38 XOR, versus
26 plus104 previously. The generic native postpass itself is unchanged.

## Required acceptance and limits

All245 prior App CPU controls plus4 genuine controls: complete per-head
ordered symbolic expression trees; IEEE bit/fault equivalence; exact full-source
inversion/source-drift refusal; actual SPIR-V ABI/FMA/multiply/barrier/loop/
gather and130->42 exchange inventories plus selected constructor composition.
Original constructor tests compose the new hook while keeping their separate
old-stage invariants. The existing native Quad inverse still requires every
nonreplacement word/header/ID unchanged, with the exact new selected site count.
Intended249 App controls; unchanged15 real runtime,429 WGSL/296 layouts,350
GPU controls, BOTH original18-script chains, actual compiler ISA, unchanged
zero-spill and independent bitwise/F64 bounds and physical recovery.

Register pressure can increase because head partials now overlap; native
assembly must decide that. Do not relax the spill gate or claim a gain from
source counts. Complete optional source remains even if a single diagnostic
is slower; fix errors in the connected implementation. No CPU compilation
while protected whole-model GPU timings are active. Clean selection and
staged private-runtime/tool/thinking/API/recovery gates precede installation.

## Earlier bundle still being tested

Source8a49206b1909b920dca10f98fa4364791c4ea8c4, worker SHA256
dc3084dcfaa18ba6a894a5e06cf7e6b6ab04c67d958bd79377d04c0b58511aae
passes245 App CPU/15 runtime/429 WGSL/296 layouts. Its protected operator
worker passed and recovered; the full normal/profile chains are still running
at this source checkpoint. Public artifact prefix: strata-parity-query-gdn-20261022.
The initial CPU harness used a wrong new test namespace and correctly refused
zero selected tests; failed log/selection are preserved with initial-harness-
namespace suffix. Correct full245 selection then passed. No source gate removed.
