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

The oracle and candidate must have **bit-identical complete partial and
statistics owners**, including every inactive query/chunk and physical tail.
An independent coordinate checker also demands zeros in every inactive region.
Their complete fault owners must agree exactly:0 for valid inputs,1 for each
negative control, and zero physical padding. Fixed public buffer data stays in
wiping RAM and actual readback leases; the original whole-owner reader includes
the full runtime I/O audit. The parent worker still owns final close/zero audit.

Results belong in a separate `qsa_math_checks` array, preserving existing matrix,
router/dequant and active-prefix selection case counts. Only fixed case names,
geometry and boolean/numeric checks are reported; no arrays or content are
persisted. Each case drains/resets the original arena before its result is
published, including on failed comparisons.

`apps/kuno/benchmarks/text/test-qsa-attention-parity-controls.py` compiles the
actual pure geometry/checker region and verifies the original scalar, actual
candidate, whole-owner readback, exact comparison, NaN upload and cleanup call
sites. Its two CPU tests check causal inputs, poison tails, all active/inactive
chunk coordinates, physical padding and fault rejection. They are not GPU parity
or device ownership proof.
