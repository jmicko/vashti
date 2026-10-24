# Complete granular small-expert pipeline

Source candidate composed after accepted source94613ad7/full1867. Not yet
compiled or measured. Keep all previous complete variants. No API, model,
Vashti, install, release, driver or boot change.

The full four-C carried small kernels spill at128GRF. The short-lived four-C
variants do not spill but their macro sample is slower than the older two-C
path. Another source difference is physical workgroup size: current macro64
uses M64/N128/WG256; macro32 uses M32/N128/WG128; cohort16 uses M16/N128/WG64.
All give each SG16 a16-member by32-output/four-C cell.

For one64-member GU product, the current macro64 grid offers10 workgroups;
DOWN offers20. [Intel specifies32 Xe-cores for this B70](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics/specifications.html).
These block counts suggest possible underfill, not measured occupancy or proof
of how Intel places a workgroup across cores. Do not describe oneMKL's opaque
internal tile as known from its GEMM API. Actual normal macro counts are
4218 macro32 and6532 macro64 launches, with383820 physical workgroups in the
completed repeated fixture. This is substantial production work.

## Compose all changes together

For BOTH regular small and cohort16 consumers, use an M16/N64 tile with
WG32/two SG16. Preserve the same16x32/four-C subgroup arithmetic, input reuse,
original initialized padding and logical count categories. The64-member GU
example becomes80 workgroups and DOWN160, with the SAME active subgroup/cell
work and arithmetic. Partial counts require exact ceil/termination accounting;
do not assume a constant workgroup multiplier for every count.

Publication reuses the existing independent eight-member helper. Its actual
scratch span is `PLANE_POS*128..+128` F32 values: two real subgroups need256
F32 values/1024 B, reused sequentially for the four accumulators with the
original subgroup fences. Prove the entire actual span and all layout modes;
never lower the advertised reservation without lowering/proving the actual
allocation. Preserve raw GU, the checker, ordered DOWN/pending2 and invalid
sentinels. Original tile8 and the accepted large eight-C kernels stay intact.

Replace the eight-A/two-B temporal bank with FOUR A and TWO B fragments for
the four C cells. In each K32 iteration:

1. Bank0 starts with rows0/8 at K and B(K), genuinely initialized/carried.
2. Prefetch rows0/8 and B at K+16 before current products.
3. Consume C0/C1 at K; load rows16/24 at K into the released bank0 A storage.
4. Consume C0/C1 at K+16; load rows16/24 at K+16 into released bank1 A storage.
5. Consume C2/C3 at K. If another iteration exists, refill bank0 rows0/8 and
   B at K+32 before consuming C2/C3 at K+16.

Every C still accumulates ascending K16 products. Interleaving independent C
cells changes no reduction order within a cell. Only the first two A rows and
B need cross-loop carry; late rows have genuine current-iteration definitions.
No speculative final read: admitted K2560/640 are divisible by32 and the
K+32 refill is guarded. Prove actual reaching definitions, not source comments.

After the unchanged original GU/DOWN role guards, use their genuine constant
K/N geometry throughout the new product and publication address calculations.
Keep original runtime INFO and guards: constants grant no input/owner authority.
Cohort role geometry is already constant. Do not change precision, finite
predicates, source extents, status, image layout or any original model weights.

## Authentic host and runtime composition

Use distinct exact generated roles for granular macro32/64 GU/DOWN and
cohort16. Logical macro32/64 retain their original32..63/64..255 count ranges;
their physical M tile is16 in this variant. Real host grids are
`rows/64 x ceil(count/16)` and cohort `rows/64 x2 xslots`; actual WG32.
Reject old grids/WG, forged nested names and new-role/tile8 aliases.

Extend the closed native classifiers/grammars only for these exact genuine
roles, after authentic role classification. Retain the original Gather,
activation, inverse, six owners, scalars, maps, packet fields, phase order,
source releases, command limits and full erasure. No payload or owner is
cached as validation. Original producers/slots are sufficient: dequantization
and FP16 image layouts do not change. No new allocation or extra phase.

Count actual granular products and grids separately. Preserve semantic full
small counts and honest physical workgroups; displaced N128/local/full-carry
producer counters must be zero. Add an explicit new counter-helper branch
without crediting a kernel that did not execute. Profile genuine generated
types as macro/cohort; do not relabel based on names or expected counts alone.

## Acceptance

Finish BOTH consumers, launch/grids, exact runtime roles, counters and compiler
proofs before live speed testing. Retain all422 AppCPU,18/86 runtime,8+9 metadata,
10 native,429 WGSL,320 original matrix and all1867 GPU controls. Plan additional
CPU lattice/scratch/padding, actual emitted-bank/constant-stride/negative-module
and genuine profile-class controls, plus exact runtime role/grid/phase/owner
positive and forgery negatives. Freeze exact names/counts before acceptance.

Add116 old/new/F64 complete-owner regular/cohort cases and4 mixed-slot pairs,
covering partial counts, both roles, bad routes, independent halves, nonfinite
values, nonzero slots/prefixes and original tile8. Compare with the preceding
actual carried program and keep its controls too: planned1987 fresh GPU cases.
No numerical tolerance, original guard, capacity or physical recovery relaxation.

Then both original full model/compiler/owner/counter/erasure/recovery chains,
actual native spill/instruction inventories and the original CLEAN first/repeat
gate. Keep the entire candidate when diagnosing results. Do not abandon one
of the coupled liveness/granularity pieces based on an isolated timing.


## Composed source checkpoint (2026-10-24)

Feature `strata-small-granular` depends on the whole preceding carry bundle and
forwards `cubecl-wgpu/kuno-small-granular` to the exact runtime classifier. The
builder requires explicit `--small-granular` plus the existing complete flags.
New genuine namespaces are `native::granular_small` and
`cohorts::granular_small`. Logical macro32/64 guards remain; both physical
shapes are M16/N64/WG32, publication allocation256 F32. Original six-role
and cohort48-u32 owner/phase contracts remain. All old entry functions remain
available to the original numerical controls.

Both consumers call one four-A/two-B interleaved K32 core. All per-cell K16
reductions retain ascending order. New public controls compare actual preceding
true-carried and granular programs, including four mixed eight-slot pairs.
These fixtures contain fixed public data only and are not private request logs.
Additional plan: three AppCPU cases (coverage/scratch, genuine profile roles,
48 emitted layout proofs), three regular native grammar cases and two cohort
cases. Planned total425 AppCPU,422 feature-off,18/91 runtime,1987 fresh GPU.
CPU/compiler and live outcomes remain pending; no speed or acceptance claim.
