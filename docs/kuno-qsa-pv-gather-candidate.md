# Prompt attention probability-gather candidate — 2026-10-20

This is an explicit `strata-qsa-pv-gather` source candidate, with CPU validation
only. It can follow either the original prompt specialization or the separately
enabled paired-QK specialization. Root owns central Cargo/build selectors,
protected GPU acceptance and installation. No API, private request switch,
numerical bound, scalar golden reference or installed worker changes here.

## The selected Intel reference

The actual public reference is pinned MIT sourcefb58e0db and AOT bmg-g31 binary
SHA256 `16a6ed700650b61675ac9ae8f8fc3e1c803b566b120f70495b599b07ff7095b3`
under `/media/jmicko/faststore/kuno-development/strata-parity-reference-build-20261020`.
The reference client permits neither `STRATA_PROMPT_ATTN_XMX` nor
`STRATA_ATTN_LANECELL`; the timing client adds only `STRATA_PREFILL_TIMING=1`.

`sycl/src/prefill/prefill.cpp:3152` first calls `qsa_prompt_attn_batch`.
`qsa_prompt_attn.dp.cpp:2010` tries XMX, whose mode defaults to0 and returns false
at `qsa_prompt_attn_xmx.dp.cpp:366-368`. The caller then dispatches the older
`qsa_decode_attn_batch` in32-query tiles (`prefill.cpp:653,3154-3158`). Its
LANECELL flag is false by default (`qsa_decode_attn.dp.cpp:600-645`), so int8 KV
selects `attn_chunk_kernel<1,false>` atlines684-693. The experimental pre75 path
is not this selection. Grid is33 cell chunks by2 KV heads byquery count, with
WG256 and explicitly required SG32. QK uses eight consecutive dimensions per
lane, left-to-right local sums and XOR16/8/4/2/1; softmax pairs cells lane and
lane+32. PV is one dimension per thread and twelve independent accumulators,
folded in ascending cell order. The separate merge folds chunks in order.

CPU parsing the original binary's embedded ELF `.ze_info` confirms the selected
batch kernel name `attn_chunk_kernel_2ad0a4...ILi1E` atoffset26377008:128 GRFs,
SIMD32, eight EU threads,15,872 bytes declared SLM. The batch merge kernel
`attn_merge_kernel_338175` atoffset26229680 also has128 GRFs/SIMD32/eight threads.
There is no evidenced256-GRF or SG16 advantage in this reference. The SYCL build
uses `-fp-model=precise` and correctly rounded divide/sqrt (`sycl/CMakeLists.txt:39-66`),
while softmax and merge explicitly use `sycl::native::exp`. Native WGSL exp
already lowers to hardware `math.exp` with the exponential-base conversion;
there is no observed software-library exp bottleneck to remove blindly.

## Measured gap and concrete lowering evidence

`strata-native-fast-cpu-readyfix-20261020/receipt.json` is a completed public
diagnostic atsource8c91e6ab. Its warm4245 prefixes are11.8742/11.7921s and last
attention/merge interval1.6466s. The original upstream timing reference's
attention interval is680ms. These are different instrumentation boundaries,
not a controlled isolated-kernel throughput ratio. The source routes and
hardware are verified; do not infer identical output/expert trajectories.

The actual Mesa dump
`strata-parity-native-paired-fast-bulk-20261020-operators/compiler-public-stderr.txt`
shows no attention spills. Its optimized fault source hash
`a68e3d2f8d99c03c` has twelve distinct uniform `load.slm.d32x1t.a32` probability
reads in the PV loop, each followed by the original per-head FMA. The source
stores probabilities head-major: `sp[h*64+c]`. Reusing query reads in QK did not
remove these PV memory sends. Compiler instruction/branch estimates are not
measured hardware duration.

## Complete bounded change

The new module changes the internal probability array from768 to780 F32 words
and all of its head strides from64 to65. It is exactly+48 declared shared bytes:
15,620 becomes15,668. There is no alias, additional root, global allocation,
scratch lease or guessed capability. Twelve padding words, at`h*65+64`, are
never read or written. All live QK/softmax cells retain their original values;
only internal addresses change. The padded stride removes the original
power-of-two spacing of the twelve simultaneously gathered head addresses.

At each existing PV cell, lane0..11 loads its head's probability once; all other
lanes set a local temporary to0. After that conditional reconverges, twelve
constant subgroup broadcasts reconstruct the exact original twelve probability
bits in every output lane. The original twelve F32 FMAs then consume those bits
and the original V value in the same chronological order. Cache/causal checks,
NaN/fault checks, all global owners/strides and all four barriers are unchanged.
This aims to replace twelve uniform memory messages by one distributed gather
and register broadcasts. It is a Mesa-targeted candidate, not an assertion that
the selected upstream source explicitly performs this transformation.

Naga27 deliberately emits WGSL `subgroupBroadcast` as
`OpGroupNonUniformShuffle` for Vulkan1.1 compatibility
(`back/spv/subgroup.rs:184-188`). The actual generated SPIR-V is checked for
twelve such instructions, subgroup scope3 and exact source lanes0..11. There
is no SPIR-V mutation to invent another primitive or change subgroup authority.

## CPU checks and required next gates

The direct-Rust source harness uses prebuilt dependencies, separate temporary
binaries and no Cargo/shared target/GPU/model/service work. Thirteen checks pass
with PV enabled on base specialization and with both PV and paired QK enabled.
The default/base and paired-only combinations also retain the same harness.
Controls include:

- Complete-source reversal: only the declared padding/index and probability
  read/broadcast transformations differ; QK, softmax, global stores, finite
  checks and closing are otherwise identical.
- All0..64 cell tails, all768 live head/cell coordinates, NaN-poisoned unread
  padding, preserved negative-zero/NaN bit patterns, and a wrong-head negative
  control using distinct public values.
- Actual Naga/SPIR-V, original WG256, exact15,668 shared bytes, twelve constant
  subgroup broadcasts, all FMul NoContraction decorations, twelve explicit PV
  FMAs and four barriers.

The existing nine strict bitwise plus three independent F64 attention GPU
references remain unchanged, as do full owner/tail/fault controls. They must
pass again for this exact source, followed by complete requests, erasure and
physical recovery. Inspect actual ISA for the intended one-gather/register
broadcast lowering and any spill/occupancy regression; source intent alone is
not a result. Then compare clean matched whole-request times. No speed gain,
upstream parity or installation is claimed from this checkpoint.
