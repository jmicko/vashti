# Qwen21 graph-local positional-matrix sharing

Candidate based on Vashti `fb2d1d0`, stable-diffusion.cpp
`88411ef1e0688ff2df1010aeeb5d92b2d8cea2be`, and GGML
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d`. Updated patch 001 SHA-256:
`ed75739ccb193099f23d4600cc4a8846f9d17fe6c2cec7591a7c9cb30c2d0134`.
The installed residency manifest continues to describe the accepted older build;
this source candidate has no installation or whole-workflow speed receipt.

## Change and lifetime

Qwen21 originally creates `CONT(PERMUTE(pe,3,0,1,2))` for each Q and K in
each of its 32 layers. GGML's graph deduplication uses tensor pointers, so these
are 64 separate materializations. The model now creates one prepared tensor and
passes it to every layer. Its existing prelude cut group includes this tensor,
so segmented graphs reuse it through `INPUT_PREVIOUS_CUT` rather than repeat
preparation in every segment.

The legacy `Rope::apply_rope` path keeps preparation at its original position.
The prepared path removes only the repeated PE preparation: activation
permutations, views, repeats, two multiplies and in-place sum remain identical.
Generic attention and other models continue to call the legacy path. Positional
values, circular adjustment, prefix STORE/REUSE slicing, input uploads and
request cleanup are unchanged.

Prepared PE belongs to one compute graph. Existing graph-cut accounting,
synchronization and `GraphEndGuard` clear its transient storage; it is neither
a prefix-cache entry nor a retained cross-step/request tensor. Segmented
execution keeps one additional prepared buffer until the last consuming layer,
with one existing cut-cache capture copy.

For the public generation shape, the target-only PE is `8214*1024 = 8,411,136`
bytes; editing is `4056*1024 = 4,153,344` bytes. Avoiding 63 materializations per
diffusion call removes approximately 24.68 GiB and 12.19 GiB of destination
writes over 50 calls respectively, plus matching reads and first-step prefix
traffic. These are operation counts, not measured speedups.

## CPU regression and source proof

`rope-sharing-test.cpp` uses synthetic values only and explicitly initializes
the CPU backend. Coverage includes an independent separately rounded scalar
matrix oracle, bitwise legacy/prepared equality, interleaved and noninterleaved
pairs, `D=4,L=2` ambiguous shapes, `D=128`, odd lengths, multiple heads/batches,
generation and ordered two-reference layouts, different condition lengths, all
circular flags, full-prefix and target-only embeddings, and unchanged raw PE.

The real miniature Qwen21 graphs cover NONE, STORE and REUSE modes and require
one shared prepared node. A 64-consumer graph checks 64 original CONT nodes
versus one shared CONT and 128 views of the same prepared pointer. A two-layer
segmented execution uses the actual planner, cut cache and bindings; it must
match monolithic output exactly, release cached storage, then succeed with a
fresh graph and changed coefficients.

The test compiles one C++ translation unit against candidate source headers and
the accepted standard engine/GGML archives, whose non-inline implementation is
unchanged. Example with the accepted local build:

```sh
rope_src=/tmp/kuno-image-shared-rope-source
rope_build=/tmp/kuno-image-resident-native-v1/build-standard
rope_cuda=/home/jmicko/dev/vashti/target/kuno-toolchain/cuda-13.0/lib64
c++ -std=c++17 -O1 -DGGML_MAX_NAME=160 \
  -I "$rope_src/src" -I "$rope_src/ggml/include" -I "$rope_src/include" \
  crates/kuno-image/native/rope-sharing-test.cpp \
  -Wl,--start-group "$rope_build/libstable-diffusion.a" \
  "$rope_build/ggml/src/libggml.a" "$rope_build/ggml/src/libggml-base.a" \
  "$rope_build/ggml/src/libggml-cpu.a" \
  "$rope_build/ggml/src/ggml-vulkan/libggml-vulkan.a" \
  "$rope_build/ggml/src/ggml-cuda/libggml-cuda.a" \
  "$rope_cuda/libcublas_static.a" "$rope_cuda/libcublasLt_static.a" \
  "$rope_cuda/libcudart_static.a" "$rope_cuda/libculibos.a" \
  -Wl,--end-group -lgomp -lvulkan -lcuda -ldl -lrt -lpthread \
  -o /tmp/kuno-image-shared-rope-test
/tmp/kuno-image-shared-rope-test
```

Fresh checkout `/tmp/kuno-image-shared-rope-proof` at the pinned engine revision
passed patch-001 `git apply --check`, apply, and byte-for-byte equality between
its full non-GGML `git diff HEAD` and patch 001. Candidate source also matches
patch 002 and patch 003 exactly at the pinned GGML revision. `git diff --check`
passes. The documented one-translation-unit compile and complete CPU regression
passed on 2026-10-01, printing `RoPE scalar oracle, exact sharing, Qwen prefix
layouts and segmented lifetime passed`. Independent source/test review found
no blocking issue. No engine rebuild,
GPU test, request content read, installed-service operation or deployment was
performed for this candidate.
