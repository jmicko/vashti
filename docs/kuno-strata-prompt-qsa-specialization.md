# Prompt QSA scalar specialization candidate (2026-10-20)

This is a Kuno-only arithmetic-preserving candidate. It is not an installed
worker, a GPU acceptance result, or a speed claim. No API, model, precision,
prompt retention, tool, Vashti, or context-setting change is needed.

## Typed query SLM follow-up

The actual selected upstream kernel reads each head's eight query components
through two `sycl::float4` shared-memory loads
(`sycl/src/kernels/cuda/qsa_decode_attn.dp.cpp:229–232`). The expanded Rust
shader still used eight individual scalar SLM loads. The follow-up makes this
physical access explicit: the same12KiB shared image is768 `vec4<f32>` values.
Each of256 threads initializes three disjoint vectors from the unchanged scalar
global query owner, before the same original barrier. Each head/lane then loads
two genuine vectors and consumes components in the original left-to-right dot
order. There is no96-register per-thread query cache or new owner/descriptor.

CPU coverage checks exact initialization coverage, every head/lane/component
coordinate, actual Naga WorkGroup array type/stride/extent, original four
barriers, NoContraction products and explicit PV FMAs. All six controls pass;
independent source review finds no bounds/causality/closing change. Original
scalar source and decode/MTP remain unchanged. GPU math, complete requests and
measured performance remain required before selection; vector source alone is
not a bandwidth or speed result.

## Selected upstream path and concrete difference

Pinned MIT Strata `fb58e0d` normally selects its scalar INT8-KV sparse attention
path. `sycl/src/prefill/prefill.cpp:3152` falls back to batches of32 queries;
`qsa_prompt_attn.dp.cpp:2009` rejects the optional XMX route when its mode is
off, and `qsa_prompt_attn_xmx.dp.cpp:366` defaults that mode to0. Both selected
implementations use256 threads, SIMD32,64-cell partial chunks,12 query heads
per KV head, head width256, and a separate partial merge. An inactive XMX path
is not evidence that the selected Rust port is missing that algorithm.

The selected upstream `qsa_decode_attn.dp.cpp:66` loads eight signed Q8 values
from two packed words and one FP16 scale. Its QK and PV loops at225–235 and
269–305 explicitly unroll the12 query heads, eight dot components, subgroup
reductions, and scalar PV accumulators. Kuno's original generated prompt shader
keeps a dynamically indexed `k[8]`, `accum[12]`, and the fixed-extent loops in
the actual Naga-produced SPIR-V. Each original scalar `kv()` call also emits
its own code and scale-word load. Native compiler CSE/register allocation may
reduce some cost later; source differences alone cannot establish a speedup.

## Connected change

`strata_prompt_qsa_specialize.rs` expands only the already-rebased
`PromptAttention` and `PromptAttentionFault` generated source. It changes all
of the connected fixed extents together:

- Twelve explicit per-thread stores fill the same3072-word shared Q row.
- Each lane's eight-aligned K group loads two code words and one packed scale
  word, decodes eight signed values, and performs the original finite checks.
  An eight-value group cannot cross the original64-value scale boundary.
- Twelve explicit QK blocks retain each original left-to-right eight-product
  dot and XOR reduction order16,8,4,2,1.
- Maximum and probability-sum reductions retain that same original XOR order.
- Twelve scalar PV accumulators keep the original chronological cell loop,
  explicit FMA operations, partial stores, and finite checks.

The logical scalar K-load requests fall from sixteen words per lane/cell to
three. This is a compiler/source request count, not a physical-traffic estimate:
loads share cache lines, packed scales are shared, and a native compiler may
already eliminate repeated operations. The arithmetic operation count remains
the same.

Dynamic cell/chunk/time-step bounds, row poisoning/sentinels, causal guards,
full owning roots/strides, query-row rebasing, launch geometry, all four
workgroup barriers, status/fault closures, capture identity, math/denormal
flags, and final ownership/erasure checks are unchanged. No fast math,
reassociation, new approximation, XMX, DSP, binding layout or authority is
introduced. The base WGSL file is unchanged. Decode/MTP/merge shaders retain
their original generated source. `Kernel::scalar_source` exposes the exact
original generated shader internally for the separate compiled-public golden
control; it does not create a runtime setting or request route.

## CPU acceptance and remaining gate

The exact real source generator is tested using existing dependency artifacts,
without a Cargo/shared-target build, GPU operation, root command, service change,
or model load:

```sh
python3 apps/kuno/benchmarks/text/test-prompt-qsa-specialize.py \
  --deps-dir /media/jmicko/faststore/kuno-development/root-native-text-target-20261005/release/deps
```

All four tests pass: source/closing invariants, unchanged non-prompt source,
all signed byte patterns and finite nonnegative FP16 scale representations,
and actual Naga validation/SPIR-V lowering of both fault/status variants.
Standalone `spirv-val --target-env vulkan1.3` accepts both emitted modules.

In the fault variant, actual `OpLoopMerge` count falls13→3, and both function
arrays disappear; the status variant retains one additional original bounded
closing loop. Both variants preserve four `OpControlBarrier` instructions.
Fault SPIR-V grows4965→11199 words; the expansion could increase native
instruction/register pressure, so it still needs measured GPU evaluation.
The native compiler and actual hardware remain authoritative.

Next gate is the separate real-arena fixed-public attention control against
the unchanged scalar shader, existing full shader/numerical checks, complete
request/state erasure/recovery, then the matched public prompt profile. Do not
weaken original IEEE/denormal, owner, fault, or numerical acceptance limits to
select this candidate. No private content should be profiled or saved.

## Rounding-sensitive golden gate and candidate repair

The initial nine actual GPU controls with dyadic inputs passed, but the three
additional non-dyadic controls rejected the first expansion: partial maximum
absolute differences were about1.67e-6,3.81e-6 and5.72e-6; stats differences
reached7.63e-6, with no nonfinite mismatch. The bitwise gate remains strict;
these controls are not accepted and no threshold or reference is changed.

Inspection found that the existing native graph's
`preserve_native_product_rounding` postpass runs only for generated source
containing `STRATA_ROUND_PRODUCTS`. Neither original attention nor the first
expanded shader had that marker. Fixed expansion lets a native compiler see
neighboring products/adds that the dynamic array/loops previously materialized
across stores or control flow, so its implicit contraction can differ.

The candidate generator now adds the existing marker only to the specialized
prompt attention source. The original scalar golden source, base WGSL and
non-prompt paths remain unchanged. Five exact-source CPU tests pass, including
the real native postpass: every emitted FMul result receives NoContraction,
all12 explicit chronological PV FMA instructions remain, and the postpass is
idempotent. Both actual postpass modules pass standalone Vulkan1.3 SPIR-V
validation. Fault SPIR-V is11514 words after annotation; loop/array/barrier
evidence is unchanged.

This is a source/compiler repair candidate, not proof of restored GPU parity:
the original dynamic kernel's native contraction choices still require the
same twelve actual GPU golden controls before acceptance. Do not modify the
original reference to force agreement.
