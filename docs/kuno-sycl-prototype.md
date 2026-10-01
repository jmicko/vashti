# Optional Intel SYCL image prototype

This is a development experiment, not an installed backend or an API change.
Apply `003-sycl-convrot.patch` to pinned GGML
`4bf5f6000653b7881d00963cd6ddb665ccd62a8d`, independently after patch 002.
Patch 003 changes only `src/ggml-sycl`; normal CUDA/Vulkan source is untouched.

The native ConvRot path accepts contiguous signed INT8 two-dimensional weights,
per-output FP32 scales, optional FP32 bias, and either contiguous FP32 input
(group 0 or 256) or the existing packed group-256 input. Logical input batches
are flattened into rows. Packed activations have rows padded to four, followed
by the logical rows' FP32 scales. Padding is zeroed. The normalized H4 tensor
product and nearest-even quantization match the existing finite-input contract.
The signed INT32 dot product is gated at K <= 132104 (including -128 weights
and +/-127 activations). Unsupported geometry/types and split weights are
rejected by `supports_op`; raw I8 never reaches the generic float converter.

oneMKL performs signed I8 x I8 -> I32 GEMM with standard compute mode. A dependent
kernel applies activation scale, weight scale, then bias. Scratch returns to the
existing LIFO pool on the same in-order queue. The rare legacy-pool overflow
branch drains the queue before actual USM/Level Zero free. The new operations
are excluded from SYCL graph capture, which oneMKL does not support. Zero-scale
rows quantize to zero; nonfinite scaled values deterministically quantize to
zero rather than invoking an undefined integer conversion. Nonfinite model
activations are not an accuracy-supported input envelope.

The backend exposes the actual Level Zero PCI-address extension as `device_id`.
Only an exact canonical 12-character domain:bus:slot.function is accepted;
unavailable/invalid identity is null, never inferred from names or ordinals.

The same patch includes an independently reviewed oneDNN unmasked SDPA
experiment, enabled only by exact `GGML_SYCL_FA_ONEDNN_NO_MASK=1`. It targets
head-128 MHA, batch one, at least 32 query rows. It omits the graph's mask Add
and mask input rather than allocating a zero mask, separates cache keys, and
validates partition ports. Masked attention retains its graph. Fused-kernel
selection, numerical error and workspace/peak memory remain GPU acceptance
requirements. Its helper test is in the separate SDPA review commit.

## Local validation

- The CPU-only geometry/PCI test passes 40 cases, including K overflow,
  malformed packing/scales/strides/batches and strict BDF rejection.
- The independent arithmetic test passes 30 CPU cases: plain and rotated
  direct input, packed rotation, odd rows/N, flattened batches, bias/no bias,
  zero rows, -128/+127 weights, exact quantized bytes and deliberate rounding
  ties. It uses a dense H4 tensor-product oracle, not backend transform code.
  A direct batched GPU case exposed a one-byte quantization error at a half tie:
  with row maximum `.8125`, FP32 scale then FP32 reciprocal then FP32 multiply
  maps `.40625` to exactly `63.5`, which must round to 64. Ordinary device
  division can shift that product below the tie. The quantizer now uses explicit
  Intel IMF nearest-even FP32 division and multiplication at all three
  boundaries, then the existing `rint`; the oracle tolerance is unchanged.
  Added fractional tie cases cover positive/negative ties, direct and packed
  H4 input, and flattened batches. The CPU oracle explicitly selects
  `FE_TONEAREST`. GPU acceptance of this fix remains pending.
  See the [Intel IMF device-library contract](https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2024-2/imf-device-library.html).
- `icpx -fsycl` compiled the new kernels; the full private SYCL backend builds.
- GPU tests and full image inference have not been accepted by this commit.

The pinned private compiler/runtime bundle is Intel compiler 2026.1.1-325,
MKL 2026.1.0-236 and oneDNN 2026.0.2-40. The repository's ignored target bundle
manifests retain official Intel URLs, archive SHA256 values and licenses.
Its reduced package set has no official MKL CMake config; this prototype uses
a private imported target with the same dynamic BLAS library closure as the
validated vendor matrix probe. The extra SYCL-devel package is only 1,335,800
bytes and does not supply that missing config. No system package is installed.

`target/configure-sycl.py` and `target/finish-sycl-tests.py` in the isolated
prototype worktree preserve the exact initial shared-build recipe. These are
private experiments, not installer inputs. The compiler-generated shared
backend embeds SYCL device images. PIC object archives can be reused for a
static native engine, but its final link must use `icpx -fsycl`. Compile all
GGML C and C++ objects and oracle headers with `GGML_MAX_NAME=160`, matching
the image engine; standalone GGML's default 64 is an incompatible ABI. Generated
OpenMP dependencies use `iomp5`; do not add a second GNU OpenMP runtime.

Root schedules all GPU executions. The arithmetic test's explicit GPU command
is `sycl-convrot-arithmetic-test --device SYCL <physical PCI address>`. The
existing `int8-linear-benchmark` should run tiny acceptance before fixed large
shape IDs 1, 2 and 3; its fourth phase is a different-precision dense comparison,
not byte-equivalent INT8 arithmetic. Run `attention-layout-test --device SYCL
<physical PCI address>` with the no-mask experiment disabled, then enabled.
Do not infer whole-image speed from GEMM timing alone.

Before any production adoption, validate cancellation/drain, privacy-safe
worker output, source pins, managed-runtime packaging, physical selection,
model-loading ownership, full masked/unmasked numerical attention, image
quality, complete workflow timing and host/GPU high-water memory. The current
service and public backend preference remain unchanged.


## Bounded IM2COL launch experiment

The pinned compiler defaults to `__SYCL_ID_QUERIES_FIT_IN_INT__=1`.
Its host-side `sycl::detail::checkValueRange(nd_range)` checks the product of
all global dimensions against signed INT32. The full-resolution 288-channel,
3x3, 1248x832 decoder patch launch is `(832,1248,2816)`, or 2,923,954,176
work-items. A CPU-only call to this actual SDK helper rejects that range with
`sycl::errc::nd_range`; it does not initialize a queue or enumerate a device.
This explains a pre-submit failure independently of GPU memory availability.
A final-link large-buffer option does not change the host range check.

Patch 003 now slices 2D and 3D IM2COL dispatch rows into disjoint ranges capped
at both 65,535 rows and INT32_MAX total work-items. The pure helper checks
`width * padded_channels` before multiplying, rejects impossible single-row
ranges, and computes the remaining row bound. For the failing geometry, the
ranges cover rows `[0,611)` and `[611,832)`. Both pass the pinned SDK check.
Kernel indices add the row offset before the existing batch, depth, height,
stride, dilation and padding calculations. Tensor and pointer arithmetic remain
64-bit; convolution math, packing, numerical precision and output allocation
are unchanged. The worker still needs the full 5,382,733,824-byte output tensor.
Row slicing is not a memory-saving decode implementation. A single row whose
width times padded channels exceeds INT32_MAX remains outside this prototype's
supported launch envelope and is rejected before submission.

`crates/kuno-image/native/sycl-im2col-test.cpp` is an independent numeric-only
oracle. Compile it with the pinned `icpx -fsycl`, GGML_MAX_NAME=160, and the
patched GGML include/library paths. With no arguments it uses only the CPU
backend. It passes 141,632 exact values across odd 2D shapes, F16/F32 output,
batch two, padding/stride/dilation, a 65,536-row batch boundary, and 3D F16/F32
shapes. Its direct SDK regression rejects the original global range and accepts
seven bounded ranges with exact contiguous row coverage. The helper also checks
zero/negative dimensions and signed-product boundaries.

The root task controls explicit GPU execution under Landlock and zero swap:
`sycl-im2col-test --device SYCL <physical PCI address> <mode>`.
Mode 1 runs the tiny and row/batch/3D cases; mode 2 checks a 78x52,
1152-channel decoder stage; modes 3 and 4 check full 1248x832 stages with
144 and 288 channels. Large tests stage input in 1 MiB host chunks and read back
bounded independent output samples, including offsets around 2 GiB and 4 GiB.
No model, prompt, image or output file is involved. Large CPU/GPU buffers remain
subject to the enclosing memory policy. GPU acceptance, full decoder execution,
and whole-workflow timing for this candidate are pending.
