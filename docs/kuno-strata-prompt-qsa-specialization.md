# Prompt QSA scalar specialization candidate (2026-10-20)

This is a Kuno-only arithmetic-preserving candidate. It is not an installed
worker, a GPU acceptance result, or a speed claim. No API, model, precision,
prompt retention, tool, Vashti, or context-setting change is needed.

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
