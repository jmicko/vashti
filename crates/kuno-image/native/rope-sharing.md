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
codes while unwinding resources. Both legacy and prepared Q/K outputs must be
finite in every case (fixed phases 19/20). Backend loading and initialization
are inside the fixed exception boundary. A thrown backend operation marks
ownership uncertain: teardown skips further waits and frees, and phase 21
prevents another case from using that backend. Remaining ownership is left to
process teardown. Synchronization orders operations; deferred-error detection and
device-fault recovery remain separate validation. This is an arithmetic oracle,
not a speed test.

`rope-sharing-failure-test.cpp` uses CPU inputs to exercise late Q/K NaN and
positive/negative infinity failures followed by same-backend reuse. A fake
backend throws from the pinned graph-compute synchronization path to verify
that teardown does not retry its wait or free its owned handles. The fixture
also checks the fixed entry exception before any backend loading; it does not
enumerate or execute a GPU.

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

## Persistent restore after host restart

The `/tmp` artifacts above are historical and were removed by the host restart.
On 2026-10-01 local time (2026-10-02 UTC), the standard candidate and safe oracle
were restored under the persistent worktree target directory on kernel
`7.0.0-38-generic`. Repository source was
`ac5ea5ef620d78b70f0e339f4e9e48b7d87a9700`; product source remains identical
to `734bd110cb894b720ce7a98e44af7a6802c8a933`. Only test/documentation changes
follow that product commit.

Artifact root:
`/home/jmicko/.codex/worktrees/kuno-comfy-speed-review/vashti/target/shared-rope-standard-v2`.
It contains fresh `source`, `build`, `cargo-target`, compiler temporary files,
logs, `rope-sharing-test`, and `build-provenance.json`. Final detailed receipt
SHA-256 is `43e5ef99e5b9988873558563ca2c6f87c2f287255a0e57782b04fad0366d433b`.
The executed `rebuild.py` SHA-256 is
`f41d32ed391ee2a750d2b19da2f6a9c3a10fe5089789c9ee4b4d8543581f6ba5`.
Receipt annotations were added after the build to explain argv destination
rewrites and record the CPU output hash; the script and binaries were unchanged.

The persistent original producers were
`/home/jmicko/.codex/worktrees/kuno-image-performance/vashti/target/kuno-sd-source-88411ef-46ce05099d24f16f`
and `kuno-sd-build-88411ef-46ce05099d24f16f-cuda` in the same target directory.
Independent source and GGML clones checked out the pinned revisions above,
applied only patches 001 and 002, and passed exact full-diff equality checks.
Patch 003 is absent. The original engine archive SHA-256 is
`0485532f359ccd1cf7bef71f8aeba07117d8bf4a10ede6756543e0fa84180400`.

All 57 original Ninja C++ commands were replayed sequentially. Source/build
prefixes were replaced, and relative `-o` and `-MF` destinations became absolute
paths to the same files under the fresh build working directory. Relative `-MT`
targets and all other argv entries were preserved. The receipt records each
original and executed argv. Compiler identity, native `88411ef`/`unknown`
version macros, optimization flags and CUDA producer paths are unchanged.
The CMake cache is byte-identical to this persistent original producer
(SHA-256 `91a35cddd0118bbf0fa83c537470622babfcca8b969ad9daf0a6a27371cd4f79`);
it is not the lost residency cache or a fresh CMake/GGML build. It confirms
CUDA/Vulkan enabled, SYCL disabled and static GGML.

The restored engine archive has 72 unique members: 57 freshly compiled C++
objects and 15 byte-identical original C objects. Five GGML archives were copied
intact with byte-identical producer hashes. Post-build verification repeated
the source pins, exact patch diffs, flags and unchanged original-producer checks.
New artifacts:

| Artifact under the persistent root | Bytes | SHA-256 |
| --- | ---: | --- |
| `build/libstable-diffusion.a` | 50,557,902 | `3fc0342fd44eb42bd69451debd1bfeb5ef8515623f42ae4aec100d8459c11cfd` |
| `cargo-target/release/kuno` | 725,288,672 | `9865c0d4f075b3f7eab05c4907f9da0c4c68d49ab06f539c4f05b0814061b557` |
| `rope-sharing-test` | 725,868,280 | `7bb1bf4a640a9a97a702a099753af4b4da78c9debee42f379d387813fbdc2236` |

Fresh offline locked release Cargo used the command above with default features
plus `native-image`, one compiler job, the same Rust/Cargo toolchain and unchanged
Cargo.lock hash. Its build-script output confirms static CUDA/Vulkan links with
the original CUDA producer paths and no SYCL development flag.

The safe oracle from `ac5ea5e` compiled against the fresh standard archives and
passed the complete default CPU suite. CPU output SHA-256 is
`9e364b1a782e1b839490deb5708365842efd2f2a70481f5097df4972e5ee5cd0`.
Invalid `--device CPU invalid` returned fixed phase 1 and exit 2 before backend
initialization. The receipt includes all compile/link commands and oracle source
hash. Total restore time was 384.478 seconds. No GPU test, gateway execution,
service change, packaging or installation was performed; GPU arithmetic and
whole-workflow performance remain unverified for this restored standard build.
