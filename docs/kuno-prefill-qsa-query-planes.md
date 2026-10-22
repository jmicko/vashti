# Contiguous query-vector planes in prompt attention

Complete optional source candidate,2026-10-22. No test, timing or installation
claim at source implementation. Continue the complete expert-macro64/vertical
Quad bundle from `kuno-prefill-expert-macro64.md`; do not discard correct
components based on a partial speed report. No API/KNT2/KNR2/config/model/
context/frontend change; Vashti:none. Installed accepted worker is unchanged.

## Evidence and hypothesis

The selected NC2 QK loop reads two vec4 values per lane for each head. Original
physical indices are head*64+lane*2 and+1:32-byte stride between neighboring
lanes for either vector instruction. Both unchanged768-vector stores and
24 actual prefetched query read sites are known from exact selected source.
Strata's logical float4 source is similar, but compiles through IGC. Do not
claim this permutation is copied from a Strata implementation or already faster.

Intel's general oneAPI SLM guidance discusses avoiding bank conflicts:
https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2024-2/shared-local-memory.html
This is motivation, not a verified B70 bank-count or conflict-counter result.
The narrower stride may help Mesa's split vec4 accesses. Actual native code and
whole-prompt measurements, rather than an assumed bank model, decide the gain.

## Complete connected change

`strata-qsa-query-planes` / `--qsa-query-planes` explicitly requires the complete
expert-macro64 and vertical-Quad bundle. It transforms ONLY the two prompt QSA
attention variants after scale-finite and before original fault closing. Decode,
MTP, original scalar oracle and every other shader stay unchanged.

Keep the exact array<vec4<f32>,768>,12KiB owner and original256-thread geometry.
For logical vector v, write physical index
(v/64)*64+(v%2)*32+(v%64)/2. Each head's first vectors occupy32 contiguous
vec4 entries, then its second vectors occupy another32. Reads are
head*64+lane and head*64+32+lane. The actual global query address is unchanged.
No value conversion, arithmetic reassociation, reduction/FMA/precise-product
change, extra load/store/allocation, address outside the same owner, skipped
fault/causal/source/extent guard, barrier removal or memory-budget reduction.
All K/PV lookahead, expert staging/publication and other coupled features remain.

The physical layout is a full768-vector bijection. Every3072 scalar bit,
including negative zero and nonfinite payloads, reaches exactly the same dot
component. Original store/reader barriers retain their locations. Three CPU
controls independently prove whole-source inversion and changed-source refusal;
all actual writes and every logical component's bit identity; and real SPIR-V
ABI/geometry/load/store/FMul/barrier/loop/shuffle inventories plus production
constructor composition. Prior constructor controls explicitly compose this
additional layout hook; their original independent stage/math proofs remain.

## Required acceptance

All238 prior AppCPU controls plus3 new (intended241), unchanged15 actual
runtime controls, all429 WGSL/296 layouts, all350 original GPU controls,
BOTH full original18-script chains with compiler/source/real production counts/
owner/status/fault/erasure and physical recovery. No CPU compilation during GPU
timing. Exact26 vertical swaps/104 remaining XOR/48 gathers and all original
native arithmetic/no-spill predicates remain. New layout needs no separate
synthetic GPU substitute: the existing actual scalar and independentF64 QSA
cases, all four scale poison cases and whole-model requests exercise it.
Clean speed selection and full staged private-runtime/tool/thinking/MAX8/API/
recovery acceptance still precede installation. Goal remains UNMET until a
matched clean same-workload result establishes Strata parity.
