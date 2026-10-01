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
- The independent arithmetic test passes 27 CPU cases: plain and rotated
  direct input, packed rotation, odd rows/N, flattened batches, bias/no bias,
  zero rows, -128/+127 weights, exact quantized bytes and deliberate rounding
  ties. It uses a dense H4 tensor-product oracle, not backend transform code.
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
