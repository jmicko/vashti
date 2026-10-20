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

The oracle and candidate must have **bit-identical complete partial and
statistics owners**, including every inactive query/chunk and physical tail.
An independent coordinate checker also demands zeros in every inactive region.
Their complete fault owners must agree exactly:0 for valid inputs,1 for each
negative control, and zero physical padding. Fixed public buffer data stays in
wiping RAM and actual readback leases; the original whole-owner reader includes
the full runtime I/O audit. The parent worker still owns final close/zero audit.

Aggregate numeric diagnosis prints mismatch counts, maximum finite absolute and
relative error, maximum ULP distance, and nonfinite mismatch counts before any
final bitwise rejection. These measurements set no acceptance tolerance. A pure
math mismatch is collected across all twelve cases after each genuine reset,
then still fails the complete check; runtime, owner, tail or fault failures
abort immediately. This permits inspecting all three rounding-sensitive cases
without converting failed parity into a passing receipt.

Results belong in a separate `qsa_math_checks` array, preserving existing matrix,
router/dequant and active-prefix selection case counts. Only fixed case names,
geometry and boolean/numeric checks are reported; no arrays or content are
persisted. Each case drains/resets the original arena before its result is
published, including on failed comparisons.

`apps/kuno/benchmarks/text/test-qsa-attention-parity-controls.py` compiles the
actual pure geometry/checker region and verifies the original scalar, actual
candidate, whole-owner readback, exact comparison, NaN upload and cleanup call
sites. Its three CPU tests check causal inputs, poison tails, all active/inactive
chunk coordinates, physical padding, fault rejection, exact global query offsets,
and numeric diagnosis without relaxing bitwise acceptance. They are not GPU parity
or device ownership proof.
