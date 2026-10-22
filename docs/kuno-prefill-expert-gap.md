# Expert execution gap versus the pinned Strata Intel path

Research and complete candidate, 2026-10-22. Reference is MIT Strata
fb58e0dbc8399662c0e47c76578c6e878b14f6cf under
`/media/jmicko/faststore/kuno-development/strata-parity-reference-build-20261020/source`.
No API, model, configuration, frontend or release change.

## Measured evidence and scope

Latest full candidate c5a17af5 repeat4245 prompt9.380901735s: expert interval
5.145774869s, attention2.961908847s. Expert interval includes routing/readback,
host enqueue and backend capture plus GPU execution/completion. Nested timings
must not be summed into it again. Capture0.511086744s and native wait3.275145859s
are host wall timers, not sampled CPU seconds or individual GPU kernel times.
Fixed-public sampled layer24/start0/count4096 alone cannot be multiplied by96
to claim a full-prompt breakdown: the second chunk has149 prompt inputs and
different group sizes.

The Strata *instrumented* repeat is6.2893s; the clean reference is5.3515s.
Its event phases include endpoint/queue idle overhead. Expert-related phases
sum4.679s (grouping617ms,gather97,copy wait87,dequant875,GU545,DOWN2358,combine100).
They are not identical scopes to Kuno's expert interval. Its QSA attention/merge
is680ms versus Kuno's fenced family1.500092289s, not a proven isolated kernel
ratio. Both expert execution and attention remain substantial leads.

## Actual selected source differences

- Strata `sycl/src/prefill/gemm.dp.cpp:773` calls dpct BLAS, which resolves
  half×half→float products through oneMKL in `sycl/include/dpct/blas_utils.hpp`.
  This is Intel's tuned implementation. Kuno's Rust/CubeCL kernels use explicit
  fixed8/16/32-row families and128-row prefixes. Merely lowering the prefix
  threshold128 was already tested and was slower; do not repeat that as a new idea.
- The Intel default uses ordinary FP16 dequantization plus GEMM, not CUDA fused
  quantized MoE. `sycl/CMakeLists.txt:211` selects the fused stub. Do not
  attribute the same-host reference to an inactive CUDA/MMQ optimization.
- Strata's default `ring_slots` uses routed-only8 slots when nonresident share
  exceeds10%; the reference holds11608 of24576 pairs, so cross-layer stream-all
  is NOT selected. Kuno now has two real32MiB banks. Increasing slots blindly
  is not a missing Strata default.
- Strata uses direct pointer/queue operations. Kuno retains full source/owner
  validation, planning and descriptors around authentic captured operations.
  Earlier capture/cohort improvements already reduced this cost. Do not call
  all remaining GPU-completion wait CPU overhead.
- Current small native/cohort matrices load only one K16 operand set and
  immediately consume it. The wide staged kernels already pipeline their
  operands; this candidate targets the unpipelined direct-global small families.

## Complete optional candidate

`strata-expert-register-prefetch` / `--expert-register-prefetch` requires
the complete original global-narrow/cohort bundle. The SAME helper is used by
both original global8/global16 and cohort8/cohort16 paths. Read two independent
K16 operand sets before consuming them, then update each original accumulator
in chronological K16 order. Guard a final single K16 block before any next
read. No new allocation, shared memory, binding, output, route/status/finite
predicate, source extent, publication fence, dequantization or mathematical
precision change. Old one-block path stays available. Source order alone does
not prove native load overlap or speed.

The fixed-public sampler previously misclassified generated wide/tail kernels
as `other`. Correct that using their genuine generated program types, and emit
eight static matrix-family count/ns buckets. Preserve original seven aggregate
classes, their complete partition, sample authorization/coordinates, timing
mode and private-disabled behavior. No raw program name or content is emitted.
Old `matrix`/ `other` subsets are therefore NOT directly comparable to corrected
subsets. Whole prompt metrics remain comparable within their diagnostic limits.

Run all original220 selected CPU controls plus genuine program-classification
and guarded chronological-walk controls, unchanged runtime13,429 WGSL/280
layouts, all326 GPU controls and both complete source/owner/math/recovery chains.
Only exact feature-specific static matrix load/product counts double; dynamic
work and original arithmetic/owner/fault limits remain. No speed or installation
claim at implementation time. Inspect corrected real matrix-family costs and
retain the complete candidate while resolving errors. Clean selection/staged
quality/install gates remain separate and unchanged.


## Coupled attention lowering lead

Actual Strata QSA assembly has32 indirect register-move lines; current compact
Kuno327 and prefetched Kuno332. Raw instruction counts are not latency and the
paired Kuno loop executes fewer iterations. More importantly, exact reduction
bodies show Strata's constant XOR8/4/2/1 become direct register permutations,
while Kuno's `prompt_uniform` converted original XOR butterflies into indexed
shuffles. That metadata optimization and indexed-lane spelling need not be
coupled. Retain ALL uniform immutable IDs/lane-zero score/fault policy and the
entire paired/compact/prefetch bundle; restore130 exact constant XOR operations
in place of indexed lane lookups. Original130 reduction participant selections
and chronological additions are identical. The48 remaining indexed PV gathers
stay intact; no precision or fault-policy relaxation.

Explicit option `--qsa-native-xor` / `strata-qsa-native-xor` requires the
complete QSA prefetch bundle. Whole-source inverse, actual SPIR-V geometry/
binding/SLM/barrier/operation inventories and fail-closed source-drift controls
are separate from existing full bitwise/F64 GPU controls. Two existing selected
source expectations compose the new hook without changing their independent
original arithmetic tests. Native ISA must prove fewer indirect operations and
zero spills; no speed claim at implementation time. Compose with expert-register
candidate; no rollback of the preceding complete changes.

The old external assembly directories were accidentally named in reversed order:
`.../strata-public-attention-isa-20261020/gdn/` contains the actual QSA symbol,
and `qsa/` contains GDN. Select by actual symbol + matching .ze_info, never the
directory label. Existing QSA V-load evidence1550/1553 is indeed in the real QSA
file under the old `gdn/` directory. No original artifact is changed.
`strata-attention-isa-identities-20261022.json` independently binds each symbol,
metadata, full assembly SHA and actual SIMD/GRF/SLM values. AOT extraction order
is not a semantic name. This correction does not change any benchmark result.


## First expert-register run and complete next bundle

Frozen8b7490a543408cfb1a343680c267c913d70718ef, workerSHA
0d821b0b49e371f104a9276219d34d85b45335035eea9c66ce727139b7665ac7,
artifact `strata-parity-expert-register-20261022`, passes222 selected CPU,
runtime13,429 WGSL/280 matrix layouts, all326 GPU controls and the complete
inner18-script chains. Both normal/profile receipts attest computation and
physical recovery. Normal first10.591999060/repeat9.341873441s; profiled
10.523338456/9.442025625s. Compared to preceding normal10.501310285/
9.380901735s this is neutral, not a meaningful gain or clean selection.

The OUTER aggregate fails a new profiling-only assertion: it expected ordinary
wide_gu calls, but this complete serving bundle selects the distinct generated
`expert_wide_gu_direct::ExpertWideGuDirect` program. It remained in `other`.
Original numerical, source, owner, capacity, erasure and recovery checks passed;
do not claim the outer wrapper or systemd unit succeeded. `--collect` removes
failed units too: inactive/not-found with default ExecMainStatus0 does not prove
exit0. Original failed wrapper/log/receipts are preserved. No repeat hardware
run to hide that error.

Actual partial corrected fixed layer has macro236calls/13.065ms,
cohort278/10.814ms, tailGU48/3.452ms,tailDOWN48/1.389ms,wideDOWN49/2.895ms.
Narrow0 and ordinarywideGU0 are legitimate absent paths;58 calls/5.449ms remain
in other, which cannot be relabelled as measured directGU by inference.
These are diagnostic one-layer timestamps, not full-prompt rates.

The successor retains ALL expert and QSA prefetch code, adds native-XOR
commita022e933, and explicitly classifies the direct-GU program with its
genuine generated type. Its CPU fixture now verifies all five ordinary/direct
wide/tail types (the four original names remain unchanged for old controls).
Next frozen full225-CPU/13-runtime/429/280/326-GPU chain must also require each
selected actual matrix family and exact per-family sums equals aggregate matrix.
No uninstrumented/staged/install or Strata-parity claim. No service/settings
change; accepted native-prepared worker and gateway remain installed.


## Next coupled Q8 finite-predicate candidate

The actual native-XOR175b1c2f public numerical/operator run passes; its scoped
native code still has332 indirect register moves (3097 instructions/130 sends/
SIMD32/128GRFs/zero spills), so the SPIR-V XOR spelling alone does NOT make
Mesa emit Strata's cheap permutations. Keep it as a semantic operation for
further lowering; never claim the expected assembly improvement happened.
Whole timing/aggregate results are pending separately.

Strata omits many checks that Kuno must retain. One exact simplification is
available without weakening them: every signed8-bit code times an FP16 scale
is finite iff the scale is finite. Max finite magnitude is128*65504, far below
F32 overflow. NaN/positive and negative infinity, including zero*infinity,
remain failures. Half subnormals become normal representable F32 values.
Replace16 repeated K-product finite predicates with the two equivalent scale
predicates, keeping ALL16 original multiplications and subsequent FMAs,
queries, scores, access/causal/status checks and sticky fault semantics.

Explicit `--qsa-scale-finite` / `strata-qsa-scale-finite` requires the whole
native-XOR/prefetch bundle. Refuse any drift in the exact finite function,
signed-byte decoder, FP16 unpack/selection or all16 original product expressions.
CPU validation exhausts all65536 half encodings ×256 signed bytes, verifies
whole-source inverse, unchanged actual ABI/SLM/geometry/barriers/products/
subgroup instructions and full selected source composition.

Reuse the FOUR existing independent scalar-GPU poison controls (positive/negative
infinity, NaN, zero*infinity) on this ordinary-layout path as ADDITIONAL controls.
Original twelve attention controls and every original326 GPU case remain;
the complete inventory becomes330. Existing `packed_cache_checks` is only the
legacy fixed-public receipt field name; this option does not select packed cache.
No original tolerance, owner/fault/readback/erasure or numerical reference is
changed. No new device capability, allocation, API or persistence. Source-only
until complete228CPU/runtime13/offline429/280/full330 and recovery pass.
