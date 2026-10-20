# Actual prompt-attention scalar parity fixture — 2026-10-20

This compiled-public diagnostic accompanies the prompt-only QSA inner-loop
specialization. It does not alter normal inference, grant a prompt scope, or
claim a speed result. The parent owns full compilation, GPU tests, requesterasure
and physical recovery.

`strata_qsa_attention_numeric_probe.rs` launches the actual selected
`Kernel::PromptAttentionFault` through its original launch validation. A separate
diagnostic `CubeTask`, with a distinct cache identity, compiles that same kernel's
unchanged `scalar_source()` as its GPU oracle. It includes all original prompt
row/cache/fault rewrites; it does not call the specialized source twice under
different names. No weights, descriptors, pointer authority or model nonce are
invented. Both dispatches use actual complete private-arena owners and the same
client, query, packed INT8+FP16 K/V cache, and causal selected cells.

Seven valid cases cover query counts1/3/4/7/8; positions0,31,60,62,2047,4094 and
32760; the64-cell,2048/2051-cell and32768-context boundaries; and all33 attention
chunks. Two actual negative controls use one future selected cell and one NaN
query. Deliberate NaN bits use the original wiping raw-word uploader, preserving
the normal finite-float upload guard.

The original nine cases are retained, followed by three finite queries with
nontrivial F32 significands and non-power-of-two FP16 K/V scales. Query offsets
are0,32 and64. Their query/selected owners contain the real complete global rows;
the partial/statistics owners retain the original local eight-query layout.
Unused query rows are poisoned with NaNs, so a wrong offset becomes a real
fault. Exact owner role lengths and nonoverlap are checked before both launches.
The existing host launch wrapper permits these offsets for counts at most8 even
without a PromptScope; its original row/context validation is exercised too.
No wrapper guard is bypassed or weakened for these additional cases.

The original nine cases still require **bit-identical complete partial and
statistics owners**, including every inactive query/chunk and physical tail.
All twelve cases independently demand exact zero inactive regions and exact
complete fault parity:0 for valid inputs,1 for each negative control, and zero
physical padding. Fixed public inputs stay in wiping RAM and actual readback
leases; the original whole-owner reader includes its runtime I/O audit. The
parent worker still owns final close/zero audit.

The extra three non-dyadic cases exposed finite bit differences after legal
F32 contraction/rounding while the original nine remained bit-identical. Their
new independent acceptance does not choose either GPU result as ground truth.
`strata_qsa_attention_f64_oracle.rs` decodes the original host INT8 codes/FP16
scales exactly, reads the actual F32 query bits and causal selected IDs, then
computes each64-cell chunk's scores, maximum, exponential denominator and weighted
V sum in F64. It checks **both** unchanged scalar and optimized complete outputs
against that mathematical result. Its head/query-offset/chunk/feature loops do
not mirror GPU lanes, parse generated shaders or consume GPU intermediates.
The actual reference arrays and error bounds are held in caller-owned wiping RAM.

Each live word gets an operation-derived error bound; the observed difference
between GPU implementations never enters it. With unit roundoff `u=2^-24` and
`gamma(n)=n*u/(1-n*u)`:

- Exact Q8 decode is verified for these original finite public operands. A
  changed fixture requiring a rounded decode is rejected instead of silently
  omitting that error term.
- QK permits `gamma(13)*sum(abs(q*k))/16`: one product, seven sequential lane
  adds and five subgroup adds per reduction path. Division contributes at most
  2.5ULP, bounded by5u times the possible score magnitude. F64 reference summation
  receives its separate negligible rounding allowance.
- The maximum inherits the greatest score bound. Subtraction error and
  exponential sensitivity propagate it to each weight. The explicit exp
  allowance follows [WGSL floating-point accuracy](https://www.w3.org/TR/WGSL/#floating-point-accuracy):
  `(3+2*abs(x))ULP`, with one normal F32 ULP bounded by2u times magnitude.
  A fixture outside the finite, normal exponential domain is rejected.
- The positive denominator uses a six-add pair/tree bound. The64-term weighted
  V sum uses a conservative `gamma(128)` weighted L1 bound covering even
  unfused multiply-plus-add implementations, plus propagated weight error.

No fitted absolute floor, FTZ replacement, ignored NaN or comparison only against
the legacy GPU result is used. Each component must pass its own bound, rather
than a tolerance scaled by the largest unrelated owner value. Original matrix
IEEE references/tolerances, shader arithmetic and production compilation policy
are untouched. Only the new three cases use F64 numerical acceptance; the nine
original bitwise gates remain mandatory.

Aggregate diagnosis retains scalar-versus-optimized bit mismatch counts,
absolute/relative/ULP distances and nonfinite counts. For each GPU path versus
F64 it additionally reports complete checked-word/mismatch/nonfinite counts,
maximum absolute error, infinity-norm normalized error, maximum individual bound
and maximum error-to-bound ratio. No buffer arrays are printed. A healthy math
failure is collected after each genuine reset and still rejects the full run;
runtime, owner, tail or fault failures abort immediately.

Results belong in a separate `qsa_math_checks` array, preserving existing matrix,
router/dequant and active-prefix selection case counts. Only fixed case names,
geometry and boolean/numeric checks are reported; no arrays or content are
persisted. Each case drains/resets the original arena before its result is
published, including on failed comparisons.

`apps/kuno/benchmarks/text/test-qsa-attention-parity-controls.py` compiles the
actual pure geometry/checker region and independent oracle module. Three original
CPU tests retain causal/poison/owner/fault/offset coverage; three F64 tests check
actual non-dyadic geometry0/32/64, exact unpacking and finite bounds, and reject
wrong query offsets, K codes, head/chunk order, inactive/physical tails and NaNs.
The script also checks the distinct scalar task, actual candidate launch, both
F64 consumers and reset-before-result call sites. These are CPU controls, not
GPU parity, device ownership or whole-model quality proof. The parent must run
all twelve actual kernels and existing controls before accepting a new candidate.
