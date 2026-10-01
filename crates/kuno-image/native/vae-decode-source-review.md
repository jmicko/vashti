# Qwen 2.1 SYCL decode source diagnosis

This review covers the private prototype at repository `cb34430`, engine
`88411ef1e0688ff2df1010aeeb5d92b2d8cea2be` with current 001, and GGML
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d` plus 002 and the pre-IM2COL-fix
003 snapshot `c0c42594baf1e9a03a7e1c86303eabbffd64a3ab5c4107e122e3b9b36edcba11`.
No GPU or service operation was performed for this review.

## Confirmed CPU cause of a legacy launch rejection

The full private edit run completed sampling then exited before image delivery.
A separate content-free filtered run identified `IM2COL` at the SYCL
compute-forward catch. The small 512-square image had completed successfully.
These live observations came from the coordinating backend task; the tests below
independently establish a host rejection before GPU submission.

The actual pinned `WAN::WanVAE` 2D Qwen 2.1 decoder graph has 996 nodes and 46
IM2COL nodes at both tested sizes. The decoder's default convolution route calls
`ggml_conv_2d`, which builds an F16 IM2COL tensor and GEMM. Its final upsample
uses 288 channels at full resolution before the last stage reduces to 144.

| Output dimensions | Largest F16 IM2COL bytes | IM2COL nodes above 4 GiB | Legacy global work-items |
| --- | ---: | ---: | ---: |
| 512 x 512 | 1,358,954,496 | 0 | 738,197,504 |
| 1248 x 832 | 5,382,733,824 | 2 | 2,923,954,176 |

For the larger 288-channel 3x3 convolution, the old dispatch global range is
`(832, 1248, 2816)` and local range `(1, 1, 256)`. Each global component fits an
integer, but their product exceeds `INT32_MAX`.

The bundled compiler 2026.1.1 defaults to
`__SYCL_ID_QUERIES_FIT_IN_INT__=1`. Its
`sycl/detail/id_queries_fit_in_int.hpp::checkValueRange(nd_range)` checks the
**linear global product** with `__builtin_mul_overflow` into a signed integer.
The queue's host submission path calls this check. An independent CPU test of
the exact header rejects the larger range with `sycl::errc::nd_range`, accepts
the small range and `INT32_MAX`, and rejects `INT32_MAX+1`. No queue is created.
Compiling the same CPU test with `-fsycl-id-queries-range=size_t` changes both
FIT_IN_INT and FIT_IN_UINT to zero and accepts all four ranges. Preprocess-only
`-E -dM` did not reliably reflect this flag; actual compiled host code did.

The final private executable already contains the embedded backend option
`-ze-intel-greater-than-4GB-buffer-required`. That option is separate from the
host's range check. Changing final linker flags alone cannot remove a check
already compiled into the IM2COL host object. A bounded dispatch or a correct
compile-time range option still needs independent GPU indexing/numeric tests.
These CPU results do not establish that the full image succeeds after that fix,
or exclude a subsequent allocation or kernel failure.

## Recovery and memory remain separate concerns

`image.cpp::decode_image_outputs` calls
`StableDiffusionGGML::decode_first_stage`, then `VAE::decode`,
`WanVAERunner::_compute`, and `GGMLRunner::compute/execute_graph`. Spatial tiling
starts disabled. The existing retry only activates for
`GGML_STATUS_ALLOC_FAILED`; Qwen 2.1 then uses 32-latent tiles. Execution/device
failures are deliberately not retried as allocation failures.

Pinned `ggml_sycl_compute_forward` catches a SYCL exception and calls
`std::exit(1)`, bypassing the shim's structured failure and VAE retry. Other
SYCL CHECK_TRY_ERROR paths abort. The confirmed host range rejection explains
this process behavior without requiring a kernel reset or an OOM claim.

`GGMLRunner::runner_end` releases graph workspace, prefix cache and managed
weight residency. It does not destroy the backend context's cached legacy
pool or retained oneDNN scratchpad map. `retained_runtime_buffer_bytes` counts
workspace/cache/staging/cuts, not these backend-owned temporary caches. The
legacy pool retains up to 256 buffers and uses a 1.05 allocation growth factor.
Consequently tracked residency can undercount actual retained backend memory.
This is a source-identified pressure risk, not the proven reason for this exit.
IM2COL itself allocates no scratch; it writes its planned output tensor.

A later isolated VAE GPU test should decode deterministic synthetic 64-channel
latents at 32x32 and 78x52 with identical weights, then compare standalone VAE,
after-diffusion VAE and 32-latent tiling. Emit only fixed stage/op/status enums
and numeric sizes, free-memory/peak counts and timings. Discard decoded data in
RAM and retain the same Landlock/no-swap restrictions. Do not enable raw GGML
or vendor debug output or print model/tensor names. This diagnostic was proposed,
not executed by this review.

## Reproduce the CPU metadata proof

`vae-graph-shapes-test.cpp` instantiates the actual decoder blocks and builds
its graph with both GGML contexts set to `no_alloc=true`. All graph nodes and
source tensors must have null data and backend buffers. The test uses default
VAE attention, no runner/backend, and aborting stubs for accidental backend
capability calls or tensor writes. Metadata contexts consume a bounded 32 MiB;
no model files or tensor data are loaded. It checks actual graph counts, types,
output extents and the largest IM2COL allocation against independent constants.

Use the exact finalized 160-name parent objects and base archive described in
the private build manifest; this does not link the SYCL backend:

```bash
"$ONEAPI/compiler/2026.1/bin/icpx" -std=c++17 -O2 -Wall -Wextra -Werror \
  -shared-intel -fiopenmp -DGGML_MAX_NAME=160 \
  -isystem "$SOURCE/src" -isystem "$SOURCE/include" \
  -isystem "$SOURCE/ggml/include" -isystem "$SOURCE/ggml/src/ggml-sycl" \
  crates/kuno-image/native/vae-graph-shapes-test.cpp \
  "$BUILD/CMakeFiles/stable-diffusion.dir/src/core/ggml_extend.cpp.o" \
  "$BUILD/CMakeFiles/stable-diffusion.dir/src/core/ggml_tensor_utils.cpp.o" \
  "$BUILD/ggml/src/libggml-base.a" \
  -o "$CPU_TARGET/vae-graph-shapes-test"
```

The metadata proof passed with the exact compiler and warnings denied. GNU
C++ also compiled and ran the test without warnings-as-errors; with that flag it
reports an existing conservative `out_dim` initialization warning in the vendor
constructor. Do not change the vendor VAE code solely to silence this diagnostic.

The tested executable's ELF dependencies contain compiler/OpenMP helpers and
system libraries, with no SYCL, MKL or oneDNN dependency. Supply only the reviewed
compiler library directory to its standalone CPU launch. It emitted the two
numeric table rows above and `device_execution=false`.

## Separate wide-byte copy defect (original generation size)

After bounding IM2COL submission, the coordinating task's original 1184 x 1776
generation completed 50 sampling steps, then reported device loss during VAE
decode with a corresponding GPU reset. That observation does not establish OOM
or identify which kernel failed. The independently extended metadata test
confirms a different concrete arithmetic defect in the unchanged legacy SYCL
copy implementation.

At 1184 x 1776 the actual decoder has **seven** non-contiguous F32 CONT nodes
whose source or destination byte offset exceeds `INT32_MAX`. Six contain
605,601,792 scalar elements and have final offset 2,422,407,164 bytes. A sliced
view contains only 302,800,896 elements but still reaches source offset
2,420,304,380 bytes. All logical element indices fit signed32. At 1248 x 832
there are zero such nodes; the largest CONT byte offset is 1,196,163,068.
The metadata test allocates no tensor data, backend, device or model.

`GGML_OP_CONT` dispatches to `ggml_sycl_dup`, then `ggml_sycl_cpy`. The old
non-contiguous F32/F16/BF16 and legacy quantized paths narrowed GGML's byte
strides to `int` in their launch wrappers and computed signed-int byte offsets.
The copy fix carries `size_t` byte strides through every wrapper and kernel,
uses a shared wide offset helper, and preserves `int64_t` scalar counts and
dimensions. Conversion bodies, packed-block arithmetic, launch geometry,
queue ordering and the same-type contiguous memcpy fast path are unchanged.
This is an internal development-SYCL fix; ordinary CUDA/Vulkan and API contracts
are unaffected. It is a proven invalid-address source bug, but its role in the
observed reset remains unverified until independent GPU tests and full decode.

`sycl-copy-offset-test.cpp` independently enumerates coordinates to check the
helper against padded strides, all 24 four-axis permutations, >2/>4/>8 GiB
offsets, multiple packed blocks of width32/256, and the exact large activation's
signed32 boundary and last index. **5,785 CPU checks passed** with GNU C++ and
warnings denied. The modified actual `cpy.cpp` compiled offline with the exact
2026.1 SDK, MAX_NAME160/F16/DNNL configuration; it retains two existing warnings
in `dpct/helper.hpp` and has no new warning.

`sycl-copy-test.cpp` is a standalone numeric oracle. Its default CPU mode
passed **630** exact checks: F32/F16 non-contiguous CONT, padded Q8_0 block
copy/dequantization, and all eight source/destination-strided F32/F16 copy and
conversion directions with write canaries. Explicit GPU mode2 uses a stride
`2^31+64`; mode3 uses `2^32+64`. Each large case has only six logical elements,
initializes and checks all accessed bytes, touches no giant host array, and
allocates one large backend-owned buffer to isolate pointer addressing from
large launch counts. Run only in the coordinating task's cleared-environment,
Landlock, no-swap cgroup with a unique physical BDF. No GPU test was executed by
this source review. BF16 and other quantized formats share the widened helper
but do not yet have numerical device acceptance from this oracle.

The exact oneDNN3.11.4 source does account for large GEMM buffers:
`jit/gen_kernel.cpp` computes A/B/C byte spans and requests A64 whenever any
span exceeds `UINT32_MAX`; reference MatMul requests stateless addressing for
large buffers. The narrower Xe systolic implementation explicitly refuses
buffers above `INT32_MAX`. This does not prove which implementation executed
the failing decode. `gemm.hpp::row_gemm` still computes `k*m` as signed int
before conversion to a batch stride; for batch1 that stride may be unused,
but it is a separate source issue to investigate. No GEMM behavior changed here.

## Reproduce the CPU range proof

`sycl-range-limit-test.cpp` calls the actual host header helper directly, without
queue, device, context, kernel submission or user input. Both variants passed
with warnings denied. Supply the reviewed compiler library directory while
compiling/linking and at CPU startup:

```bash
"$ONEAPI/compiler/2026.1/bin/icpx" -fsycl -std=c++17 -O2 \
  -Wall -Wextra -Werror crates/kuno-image/native/sycl-range-limit-test.cpp \
  -o "$CPU_TARGET/sycl-range-limit-default"
"$ONEAPI/compiler/2026.1/bin/icpx" -fsycl -std=c++17 -O2 \
  -fsycl-id-queries-range=size_t -DKUNO_EXPECT_SIZE_T_RANGES=1 \
  -Wall -Wextra -Werror crates/kuno-image/native/sycl-range-limit-test.cpp \
  -o "$CPU_TARGET/sycl-range-limit-size-t"
```

The default emitted four cases, FIT_IN_INT=1, FIT_IN_UINT=0 and two rejections;
the size_t variant emitted both flags zero and zero rejections. Both emitted
`device_execution=false`. No GPU admission, image generation, source patch,
service configuration or installed binary changed in this review.
