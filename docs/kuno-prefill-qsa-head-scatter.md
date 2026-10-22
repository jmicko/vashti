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

## First CPU build checkpoint

Sourcee695691d compiles. All four NEW controls pass independently, including
real SPIR-V and exact IEEE/symbolic ordered trees. The full original selection
correctly stops at the separate original vertical-mask inventory, which still
expects26 occurrences of each XOR mask. Complete new distribution is exactly
XOR1=4, Quad(vertical)=4, XOR4=6, XOR8=10, XOR16=18, plus48 untouched gathers.
Update that feature-specific exact inventory; retain the original26 counts
when scatter is disabled. Original failure log is preserved, and full249 CPU
checks must rerun before worker/offline/GPU acceptance. No GPU run for e695691d.

The preceding query/GDN bundle completed BOTH full chains and all350GPU
controls with recovery. Normal first/repeat10.301409836/9.011673934s; profile
10.331076692/9.090600187s, versus earlier normal10.411413116/9.141123116s.
Profiling points to GDN recurrence/norm392.155678->276.009677ms, while QSA
attention/merge1336.786381->1338.678668ms is essentially unchanged. Do not
attribute the gain to query planes from this combined diagnostic. Complete
public receipt SHA2562d6fb304bf677e61c22e1fd66d03100bb134538405c1a3c029d57bc38a54066b.
Actual QSA ISA2945instructions/130sends/SIMD32/128GRF/zero spills/fills;
reference ISA unchanged1814b26ac88692e20d3714abd88ea2e303455f1a7569f32fc4b2927bc8710ba7.
Actual recovery22:16:21UTC: Intel50,688,000B,NVIDIA compute0, faults0,
services200, admission normal. No installation or Strata parity claim.

## Connected depth-first register frontier

Source90ce0c9e/worker6084555c passes249 App CPU/15 runtime/429 WGSL/296 layouts.
The protected original operator worker and full profile computation pass and
physically recover. Actual QSA is2415instructions/108sends/SIMD32/128GRF, but
28spills/90fills. The unchanged native zero-spill guard rejects the chain BEFORE
normal scheduling/outer full aggregate. Preserve failed receipts; no complete
350-control/both-chain/clean/staged/install acceptance for this source. Profile
first/repeat9.765336101/8.598320364s is diagnostic only. Actual recovery
22:29:26UTC: Intel50,688,000B,NVIDIA compute0, faults0,services200,admission normal.
The reduction is retained; fix its live-value scheduling instead of abandoning
the algorithm or relaxing the spill guard.

New `strata-qsa-head-frontier` / `--qsa-head-frontier` requires the COMPLETE
ordered-head bundle. It schedules the SAME15 internal ordered nodes per cell
in depth-first order. Query head order0,8,4,2,10,6,1,9,5,3,11,7 replaces0..11.
All24 immutable Q-vector addresses occur once, and the same next-head
lookahead remains. The first head0 was already prefetched by the untouched
outer raw-pair pipeline. Every per-head chronological seven-FMA dot and its
original valid-cell predicate remains. Combine child sums immediately, then
retire them. Symbolic frontier has at most4 nonzero subtree values per cell,
versus12 simultaneous whole-head sums; actual registers/spills remain a GPU
compiler acceptance question.

Only register-only exchange nodes execute additionally for invalid cells, on
the original initialized zero partials. They cannot read a cache address,
write an owner or alter a fault. Every final score/status/fault remains behind
the original valid-cell predicate; original sentinel and closing code remain.
Same ordered operands,42 exchanges/48 gathers/168 dot FMAs, workgroup256,
barriers, full source/owner budgets, backend postpass and precision policy.
No new state allocation or array. Exact whole-region reconstruction rejects
any unexpected preceding source drift before selection.

Source only; four additional controls cover full-source inverse/drift refusal,
all query addresses/unique heads/symbolic live frontier, all15 nodes/original
dot members/valid predicates, and actual SPIR-V ABI/geometry/load/store/FMA/
barrier/loop/shuffle inventories plus selected composition. All249 prior
controls remain; intended253 App CPU, unchanged15 runtime/429 WGSL/296 layouts/
350 GPU, BOTH original18-script chains and unchanged zero-spill/F64/bitwise/
physical-recovery gates precede claims. No API/frontend or installation change.
