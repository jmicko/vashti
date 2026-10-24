# Coupled weights-first transposed global matrices

Proposed next complete variant after accepted7a54656c, not implemented/tested.
Backend-only benchmark feature; no model/API/configuration/Vashti/install/driver
change. Preserve complete N64/image/four-C/quant variants and their evidence.

## Verified source reason and limitation

Pinned MIT Stratafb58e0dbc8399662c0e47c76578c6e878b14f6cf,
sycl/src/prefill/gemm.dp.cpp:760–769 (BF16) and799–809 (FP16), passes W as first
BLAS operand with transposition and X second without transposition, dimensions
N,T,K and float output leading dimension ldy. Its comment describes
Y^T[N,T] = W[N,K] . X^T[K,T]. Kuno currently uses X as cooperative matrixA and
W as matrixB. N64 has now eliminated all actual spills without reducing full
request time; the global memory/operand mapping remains a concrete difference.
The oneMKL implementation is opaque and can transform its API orientation;
this is not proof of its precise internal tile or a promised speedup.

## Complete arithmetic and output change

Use current M128/N64/WG512/SG16 geometry and two-C/two-bank footprint. Subgroup
coordinates remain member=(sg/4)*16 and column=(sg%4)*16. MatrixA now loads two
original W8xK16 row-major fragments at column and column+8; matrixB loads the
original X16xK16 as K16x16 column-major. Each C is logical8 output-columns by
16 members. Keep four chronological K16 products per K32, both genuine banks
and guarded final loads. No repack/copy/new owner or operand SLM.

Both C matrices MUST store column-major: first at out_first+member*out_stride+
column, second at that address+8 columns. Their leading dimension stays the
original output stride. Changing only operands or only publication is wrong.
All genuine GR320 rows remain full N64 tiles. Original logical bounds, dtype,
weight prefixes, holes/nonalias, capabilities and finite consumer stay exact.

Dense exact native entry must recognize9 loads/4 products/2 column-major stores,
now input binding0 contributes3 cooperative loads and weight binding1 contributes6.
Retain original4-C12/8/4 and ordinary N64 input6/weight3 rules. New entry identity,
actual native positive/negative proof, static layout and CFG tests are required.

## Expert publication and rare failure semantics

Same original roles/headers/prefix128/N64 physical grid, sources and16KiB original
subgroup scratch. A new8x16 C spans16 members and8 output columns. Fast path may
store column-major directly only when all16 routes are valid. DOWN column-zero
publication writes original pending2 for all16 genuine members; GU still needs
original validating activation. Original tail kernels and all source/retained-
capacity/erasure guarantees stay unchanged.

On a bad route in either half, preserve original8-member raw/checked granularity.
Compute two subgroup-uniform validity flags, each using all lanes but only its
own lower/upper8 route predicates. Publish C column-major into the original
128-F32 subgroup scratch (stride8), then snapshot/read it with member=i/8 and
column=i%8. For each valid route in a wholly valid8-member half, retain raw value
and DOWN pending2, even if a different half is invalid. In an invalid8 half,
retain original scalar finite-zero/fault128 and column-zero GU/status rules;
invalid-route storage stays untouched. Do not sanitize the valid half merely
because another half is bad, or reinterpret pending2 as accepted status.

Add compound controls: invalid member0 + nonfinite column0, and invalid member8
+ nonfinite column0, with257 members, both GU/DOWN and old/new. These force
mixed raw/checked halves and test full output/status/fault/source owners. Existing
independent F64 tolerance and original bitwise comparisons remain; no shortcut
reference or dropped poisoned case. This adds8 cases beyond ordinary24 expert
pairs. If actual hardware arithmetic differs bitwise, diagnose it and document
scope before any acceptance; do not silently weaken the numerical oracle.

## Closed selection and full acceptance

Explicit feature strata-global-transpose composes the complete image/N64 bundle.
Use distinct genuine dense/expert identities, exact role-bound grammar at N64
X grid, anonymous path and physical-workgroup counters, full preflight and
original logical M128 credit. Existing original scopes remain unchanged.

Keep all372 AppCPU,18/53 runtime,8+9 metadata,10 native controls, original compiler
sets and837 GPU cases. Add3 dense/2 expert/1 generated-identity CPU controls and
2 runtime grid/role/owner controls: planned378 CPU,18/55 runtime. New24+16 real
compiler cases require inverted operand bindings, both banks/chronology, actual
column-major stores and original fallback fences with no product-loop SLM.
Dense21 + expert32 adds53 GPU cases: planned890, explicit exact inventories.

Only after complete source/compiler/numeric/metadata checks run both original
whole-model chains, original precision/status/finite/owner/capacity/reuse/rolling/
erasure proofs, actual new path counters, compiled ISA and physical recovery.
Then interpret full normal/diagnostic timing; a promising candidate gets original
CLEAN45/4245/4245 comparison, target5.3515s unchanged. No component speed rollback.
