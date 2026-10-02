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
the standard engine/GGML archives. The initial CPU receipt used the accepted
archives; the device-buffer follow-up uses the fresh standard build below:

```sh
rope_src=/tmp/kuno-image-shared-rope-standard-v1/source
rope_build=/tmp/kuno-image-shared-rope-standard-v1/build
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
no blocking issue. The later standard rebuild receipt is below. No GPU test,
request content read, installed-service operation or deployment was performed.

## Optional device oracle

The same executable accepts `--device CUDA|Vulkan|SYCL <PCI-address>` when linked
with that backend. Device selection requires a GPU, exact PCI match, a supported
backend family and exactly one match before initialization. It has no fallback.
Default execution initializes only CPU and also checks the upload/readback path
through CPU backend buffers; it does not enumerate or initialize GPUs.

The optional mode uploads synthetic Q/K and PE to real device buffers, checks
every compute node's backend support, and executes legacy and prepared variants
in separate fresh contexts. Their readbacks must match bit-for-bit in RAM.
Raw Q/K/PE must remain unchanged. Prepared PE also matches an independent
physical-permutation oracle. All readback outputs are marked as outputs so the
allocator cannot recycle them before comparison.

Forty cases cover ambiguous `D=4,L=2`, both pair layouts, differing condition
lengths, two reference prefixes, full and suffix PE, all circular modes, an odd
65-row/multiple-batch case, and public 8214/4056-row targets with 32 heads.
Only one variant's device workspace is live. The largest Q+K+PE input is
277,567,488 bytes. Size-only reservation measures workspace before allocation;
the test enforces a 2 GiB workspace cap and 64 MiB of live free-memory headroom.
It explicitly calls the pinned allocator's real reserve after size measurement.
Each case reports scalar shape/index/pass fields; core dumps and backend logs
are disabled. Device-path numerical and topology failures return fixed phase
codes while unwinding resources; best-effort teardown contains backend
exceptions. Synchronization orders operations; deferred-error detection and
device-fault recovery remain separate validation. This is an arithmetic oracle,
not a speed test.

After the standard build finished, the updated single-TU compile and complete
CPU test passed against its fresh archives. GPU mode was not executed. No
product source, engine archive or gateway binary changed for this test follow-up.

## Standard CUDA/Vulkan build receipt

Compiled repository commit `734bd110cb894b720ce7a98e44af7a6802c8a933` with
the patch hashes above. Fresh directories:

- Source: `/tmp/kuno-image-shared-rope-standard-v1/source`
- Build: `/tmp/kuno-image-shared-rope-standard-v1/build`
- Cargo target: `/tmp/kuno-image-shared-rope-standard-v1/cargo-target`
- Detailed receipt: `/tmp/kuno-image-shared-rope-standard-v1/build-provenance.json`
  (SHA-256 `400412832440122e1ebdc7b5b2f77d9fd768336f75032e190bb427325efd8d9b`).

Copied the accepted standard source, reset only its copy, applied new patch001,
and proved exact engine/GGML diffs. Replayed all 57 accepted C++ argv arrays
sequentially with only source/build prefix substitutions. Compiler:
`c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`. Existing native version macros
remain `88411ef`/`unknown`. The copied CMake cache is byte-identical configuration;
it is not a fresh CMake/GGML build. Accepted CUDA producer paths remain unchanged.

The engine archive contains exactly 72 unique members: 57 freshly compiled C++
objects and 15 byte-identical reused C objects. All five reused GGML archives
match their accepted and original producer hashes. Engine archive SHA-256:
`f0bae1e12938bcd9bf926f1973d20bd703c946e11a2a07de569fcd661eb72c9f`.

Fresh Cargo build passed with
`cargo build --release --locked --offline -p kuno --features native-image -j 1 -vv`.
Rust: `rustc 1.95.0 (59807616e 2026-04-14)`. Cargo.lock SHA-256:
`b680a386b1352c17f5848b2784e4c0ef15ededa6fa0a24ff6ca47c712483a54a`.
The native build-script receipt confirms CUDA/Vulkan static links and no SYCL
development flag. Gateway size is 725,280,480 bytes; SHA-256:
`6c6409189ee4edd740095776d5df8d9dfa9645dfa95736f1c97b65b15f676938`.
Post-build verification confirms the accepted source/archive/GGML inputs are
unchanged and both resident cleanup API symbols are present. Total build time:
411.463 seconds with one compiler job. The gateway was not executed or packaged.
