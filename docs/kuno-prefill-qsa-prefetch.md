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


## Root validation checkpoint — 2026-10-22

Merged complete implementation as c5a17af5ff6e0561e4a691501e3f47e36e200d3a
(cherry-pick of ae24a145). Frozen artifact directory under the development
artifact base is `strata-parity-qsa-prefetch-20261022`; worker SHA256
`78e7a60e22e65d815e1c1e8b62ce01be83e9abb79a09c5649d3ae3f6f4dc1e28`.
All original213 selected App controls plus7 new controls passed, as did all13
actual runtime controls (original11 retained),429 WGSL shaders,280 matrix
layouts, all326 GPU controls, and BOTH complete original18-script
profile/normal proof and recovery chains. This is the full coupled bank/release
and attention candidate, not an isolated partial optimization.

Exact helper and receipt identities under the artifact base:

- `check-qsa-prefetch-connected-20261022.py` SHA256
  `7c30d5f2795127cc7460e1a89e023498f5550d2e91140afebbe8242bcc55e964`.
- `qsa-prefetch-original-harness-20261022/manifest.json` SHA256
  `af853ef8e21b8819862d59da7bd13884e6b1a119b736acb64c23cf420fba5715`.
  All immediate predecessor scripts are preserved byte-exact in `original/`.
  Besides path changes, ONLY the selected prefetch static MAD population becomes
  216. Original numerical tolerances, precise products, fault, physical owner,
  source, capacity, erasure and recovery predicates remain unchanged.
- `strata-parity-qsa-prefetch-20261022-expert-cohorts-public-summary.json`
  SHA256 `554375e5db922cb6c2f85ca0ef10f2508337ba5f3e3d42ec26618dba60bb11e1`.
- `qsa-prefetch-comparison-20261022.json` SHA256
  `2af971b991b90fde0a594a823f465a5c8d262c3defbc8e361fd79f7051c9d0ca`.

### Actual compiler result

Complete synchronous reference/optimized compiler scopes are exported by
`extract-prefetch-public-qsa-scoped-isa-20261022.py`.
`kuno-prefetch-public-qsa-optimized-complete-isa-20261022.txt` SHA256
`5ae58070035c4debf49a590d3a31c6ce7a1701e47786f205afd8a7cc16dcd844`
contains the actual selected code AFTER its Mesa summary. It is SIMD32,128
GRFs,2982 instructions,130 sends,two loops,216 MADs,zero spills/fills.
The original scalar oracle's full ISA SHA remains
`1814b26ac88692e20d3714abd88ea2e303455f1a7569f32fc4b2927bc8710ba7`.

The intended lookahead survives compilation. Complete-export line anchors:
Q head0 loads6110/6112 precede K decoding6141; next K loads6305/6307 and
6357/6359 precede head0 arithmetic6395. Next-head Q loads6388/6391 precede
current-head arithmetic6395. PV probabilities8840/8899/8973/9043 and four
V code/scale pairs8852–8853/8913–8914/8984–8985/9056–9057 all precede
first decode9088 and first accumulation9104. Machine-checked line evidence is
`qsa-prefetch-native-schedule-proof-20261022.json`, SHA256
`84d9cad87aba643e7abbccf873f00f705d7648e358f8ee9cb7cef6b10a1bace6`.
Instruction ordering is evidence of actual early sends, not measured overlap
duration or a performance guarantee.

### Matched diagnostic result and decision

All rows use the original compiled-public73/4245/4245-token sequence, no prefix
cache reuse, exact original model/placement and complete selected bundle.
These are host-instrumented diagnostics, not clean selection receipts.

| Schedule | Prompt | Previous bank/release seconds | With prefetch seconds |
| --- | --- | ---: | ---: |
| Normal | Short73 | 3.513170431 | 3.590674452 |
| Normal | First4245 | 10.424577118 | 10.501310285 |
| Normal | Repeat4245 | 9.283137399 | 9.380901735 |
| Profile | Short73 | 3.450164342 | 3.553828183 |
| Profile | First4245 | 10.427255190 | 10.594575759 |
| Profile | Repeat4245 | 9.365058297 | 9.449852848 |

Normal long evaluations are0.74%/1.05% slower in these samples; no useful gain.
Normal repeat whole attention bucket2.950384074→2.961908847s is also essentially
unchanged; this bucket is broader than the individual QSA kernel. Do not infer
an isolated kernel latency from it. Extra instructions/sends are a possible
offset to hidden latency, not a proven causal attribution. No uninstrumented
selection, staged quality, or installation was warranted. Keep the entire
implementation available behind its explicit option; do not undo one half
based on partial timing. Recheck structural differences in Strata's complete
attention path before assuming more source reordering will close the gap.
Same-host Strata5.3515s remains ahead; goal remains UNMET.

Protected unit `kuno-qsa-prefetch-20261022.service` completed exit0,
inactive/dead. Verified swap/core0 throughout. Final snapshot17:30:44UTC:
same boot, Intel50,688,000B, NVIDIA compute0, all driver fault counters0,
Kuno/Ollama/Vashti200, admission active=false/paused=false/waiting=0/
recovery_required=false. Snapshot `qsa-prefetch-final-state-20261022.json`
SHA256 `551f34ea6fcd60cb7f3c08e37c90beefdf62056a82e99c35acfed64316d621db`.
No benchmark worker or compiler remains. Installed worker301f1fd1/SHA0e226827
and gateway36626d30/SHAd3ed07f4 hashes reverified unchanged. No API, KNT2/KNR2,
model, context, configuration, frontend, release, reset or reboot change.
Required Vashti changes: none.
