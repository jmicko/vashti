# Bounded scalar QSA memory lookahead

Optional backend-only successor, selected by `strata-qsa-prefetch` and explicit
builder option `--qsa-prefetch`. It requires the complete existing paired,
PV-gather, FMA, uniform, compact and local-fault bundle. The builder requires
those explicit options; Cargo features also enforce their dependency chain.
Packed-cache, reference-schedule and XMX combinations are rejected rather than
silently bypassed. No API, model, context, private logging or new GPU owner.

## Exact evidence and connected source

Frozen42efd634/worker6ba68800 actual compiled-public synchronous scopes identify
selected `PromptAttentionFault` hash06a30aec. Its complete saved native code is
`kuno-current-public-qsa-optimized-complete-isa-20261022.txt`, SHA256
`0612701bbd8f35f390d789858d95fda327c069a8474c124bcbe7eb1e81feba08`, under
`/media/jmicko/faststore/kuno-development`. The adjacent complete proof binds
the original operator receipt, source and worker. Summary: WG256/SG32,128 GRFs,
2654 instructions,112 sends,two loops,zero spills/fills. Compiler cycle estimates
are not measured timing. The1599-instruction program is the original scalar
oracle; earlier unrelated312/2222-instruction programs are not QSA evidence.

Actual code loads/waits/decodes both current K fragments before first Q SLM
loads5082/5084. Next head Q loads5245/5248 follow the preceding complete dots.
PV7596/7599 loads code/scale, consumes them7601/7603, decodes/checks7608–7611,
then issues probability SLM7619 and waits7627 before the12 MADs. Pinned default
Strata INT8 code overlaps its V loads1550/1553 with probability SLM sends1567–1581
before consuming V1583–1589. Both actual programs are SG32/128 GRFs.

The successor couples these bounded phases in one generated shader:

- Carry only raw code/scale words and exact token IDs for the next two-cell
  K pair. Every original cell is read once under its original causal bound;
  current decoded operands survive before carry words are overwritten.
- Load head0's two Q vectors before K decoding, then carry one next head's two
  Q vectors before computing the current head. At most current+next heads are
  live; this is not an all-query register cache. Every original dot, explicit
  FMA, XOR reduction, lane-zero score, status and finite predicate is retained.
- Fetch four separately guarded V code/scale pairs and their exact probability
  gathers before decoding/consuming those cells. Execute cell-major order and
  the original12 FMAs per live cell. Masked/tail cells skip arithmetic exactly;
  zero multiplication does not replace a skip.

Query/SLM types, binding ABI, dispatch geometry, all barriers, complete cache
descriptors, source/physical ownership, hazards, faults, precision/denormal flags,
final erasure, decode/MTP and independent scalar oracle remain unchanged.

## Validation and selection

Source controls cover whole-source inversion outside the bounded phases, exact
head arithmetic restoration, all64-cell tails and invalid selections, independent
cell-major contribution inventories, exact query vector coordinates, prerequisite
and source-drift refusal, actual Naga/SPIR-V binding/shared/geometry/barrier
identity and all original precise FMul annotations. Static FMA count changes
180→216 solely because four PV bodies replace one; QK stays168, dynamic PV
remains12 per live cell. Indexed subgroup sites142→178 and first-lane sites26→28
likewise reflect the bounded emitted bodies, not extra dynamic cell contributions.
Original compact controls keep their arithmetic assertions; only the optional
production-selector expectation composes the new hook.

No build, CPU control execution, GPU run, performance or installation is claimed
by the implementation commit. Root must run original full controls and unchanged
nine bitwise/three independent F64 attention cases, all faults/tails/full-owner
audits and both complete proof chains. Current source scheduling may be sunk by
the driver: inspect the actual selected native code to prove future K/V and Q/
probability sends precede consumption, and require no spills. Preserve the whole
coupled candidate while resolving compiler/quality errors; do not infer speed
from source order or relax numerical/ownership gates. Entire old QSA attention+
merge is about1.5s; differently scoped Strata~0.68s only suggests an approximately
0.8s opportunity, not a promised saving or a multi-second gain.
