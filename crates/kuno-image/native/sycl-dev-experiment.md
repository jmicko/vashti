# Private SYCL image experiment

This is development guidance, not a supported installed runtime or a speed
claim. Ordinary CUDA/Vulkan builds and public API behavior remain unchanged.
The private bundle under `target/toolchains/kuno-oneapi-2026.1` contains compiler
2026.1.1, oneMKL 2026.1.0 and oneDNN package 2026.0.2 (API version 3.11.4).
Verify its existing signed-package manifest rather than using system packages.

## No-mask SDPA addition

The pinned GGML oneDNN route originally required a mask. Qwen's cached warm
diffusion attention supplies no mask and uses ordinary multi-head attention;
the MKL attention route additionally requires grouped-query attention. Enabling
the existing oneDNN switch alone therefore cannot accelerate that warm graph.

The incremental change to `fattn-onednn.cpp` and its new
`fattn-onednn-shapes.hpp` adds a separate graph without the mask/Add input.
It creates no zero mask and requests no external score/probability allocation.
Masked graphs retain their existing Add operation; the compiled-partition key
includes mask presence. Graph arithmetic uses strict math with FP32 score/scale
intermediates and the existing FP16 inputs/output plus conversion to GGML FP32.
This does not promise bit-identical results to native FP32 attention.

The new unmasked route is disabled unless
`GGML_SYCL_FA_ONEDNN_NO_MASK=1`. It is restricted to the existing Xe2 hardware
gate, batch one, MHA, head dimension 128, at least 32 queries, validated scalar
strides/types, finite positive scale with a representable FP16 reciprocal, no
ALiBi/softcap/sinks, and known default/FP32 accumulator precision. It preserves
the optional KV ceiling. Unsupported static shapes or library-rejected graph
partitions retain native TILE attention. The helper also rejects invalid masked
layouts rather than passing their raw pointer to a contiguous oneDNN graph.

The pinned [oneDNN 3.11.4 SDP pattern](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/backend/dnnl/patterns/sdp.cpp)
and [primitive configuration](https://github.com/uxlfoundation/oneDNN/blob/v3.11.4/src/graph/backend/dnnl/kernels/sdp_primitive_config.cpp)
support optional scale and mask nodes. The public partition API exposes engine
kind and graph inputs/outputs, but does not expose internal kernel kind or
scratch size. One supported partition is insufficient evidence that the library
used the fused SDP primitive instead of materializing a score matrix internally.

Before running a full private recipe, a synthetic GPU test must therefore check
numeric accuracy, actual fused-kernel selection, and peak allocation. A
4056-query, 8192-key, 32-head score tensor would occupy about 4 GiB in FP32.
Capture vendor verbose output only for synthetic data, in RAM, and emit static
kernel categories/counts and numeric timings/memory rather than raw text.
The existing attention oracle should cover masked/unmasked, partial rows/keys,
FP16 and FP32 accumulator choices, head padding, scaling and V views. Testing
the CPU eligibility helper does not establish any GPU correctness or speed.

## Offline validation completed

`onednn-sdpa-shapes-test.cpp` tests the actual hardware-independent helper using
GGML metadata only. It passed 199 accepted layouts/scales and 39 rejected cases,
including token/head-transposed V, padded strides, mask dimensions, masked GQA,
nonfinite/underflow/overflow scales, and unsupported precision/types. It does
not enumerate devices or load any model.

Compile it against the pinned base archive, for example:

```bash
c++ -std=c++17 -O2 -DGGML_MAX_NAME=160 \
  -I "$SOURCE/ggml/include" -I "$SOURCE/ggml/src/ggml-sycl" \
  crates/kuno-image/native/onednn-sdpa-shapes-test.cpp \
  "$BUILD/ggml/src/libggml-base.a" -lgomp -lpthread \
  -o "$BUILD/onednn-sdpa-shapes-test"
"$BUILD/onednn-sdpa-shapes-test"
```

The modified translation unit passed `icpx -fsycl -std=c++17 -fsyntax-only`
with the bundled oneDNN/MKL include paths, `GGML_SYCL_DNNL=1`,
`GGML_SYCL_WARP_SIZE=16` and `GGML_MAX_NAME=160`. The compiler emitted one existing
`dpct/helper.hpp` return-path warning. No GPU execution or full build occurred.

## Build and routing blockers to resolve

- Validate the engine revision and exact 001 diff independently of the GGML
  revision and exact combined 002 plus development SYCL diff. Include new files
  with intent-to-add; never bypass the existing exact-source check. Content
  address a fresh build directory with the complete patch bundle.
- The current Rust build script links Vulkan and optional CUDA only. A private
  opt-in must additionally link the static GGML SYCL archive and dynamic vendor
  closure, while preserving the normal build path. Its final native link needs
  the `icpx -fsycl` driver to process SYCL device images; a normal GNU linker
  command alone is insufficient. Use a separate Cargo target directory.
- A development CMake configuration uses the exact private `icpx` and `icx`,
  `SD_SYCL=ON`, static engine/GGML, no examples/WebP/WebM, the existing patched
  GGML, and explicit oneDNN/TBB/MKL package roots. Initially disable SYCL graph
  replay, host-memory fallback and system-USM fallback; do not change driver or
  VMM behavior without measurement. Keeping Vulkan enabled satisfies the
  existing baseline adapter, but its exact SYCL linking still needs an opt-in.
- The extracted MKL directory currently contains headers/libraries but no
  `MKLConfig.cmake`; the pinned CMake requires `MKL::MKL_SYCL::BLAS`. Obtain the
  matching private config package or provide a reviewed private imported target
  with the verified dynamic closure. Do not let CMake silently find another MKL.
  The oneDNN package config exists and uses the private TBB config.
- The current shim ranks only physical-PCI-matched CUDA/Vulkan backends; the
  pinned SYCL properties do not set `device_id`. Publish a stable normalized BDF
  from `sycl::ext::intel::info::device::pci_address`, retain it in device context,
  and require a unique exact match. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` limits
  the backend but is not physical identity. Never select device zero by assumption.
- SYCL prefix caches, capability probes and working-memory budgets require
  deliberate validation. Do not label a SYCL route as Vulkan in diagnostics,
  silently fall back to a different card/CPU, or expose CUDA Sage attention as a
  supported SYCL choice.

## Privacy and dynamic runtime

Private linking requires the bundle's compiler, MKL, oneDNN, TBB, UMF and TCM
library directories. Keep them in a reviewed explicit `LD_LIBRARY_PATH` only
for the standalone development worker. The installed worker's `env_clear()`
currently removes that path and all experimental flags. Do not widen inherited
environment variables or alter the public service to make the experiment work.
Use `readelf`/`ldd` and a synthetic launch under the actual restrictions to
check both direct dependencies and vendor libraries loaded later with `dlopen`.

Every content-bearing worker must retain the existing no-swap check before
reading input, `MemorySwapMax=0`, disabled core dumps and nondumpable process,
and inherited Landlock ABI 3 write restrictions. Dynamic libraries are readable;
shader/JIT caches, `/tmp` files and `/dev/shm` stores remain unwritable. No broad
filesystem exception should be added for oneAPI. Suppress all raw library logs
and exception text through the worker's existing pipe/filter policy.

For a clean environment set `HOME=/nonexistent`,
`SYCL_CACHE_PERSISTENT=0`, `SYCL_DUMP_IMAGES=0`, `SYCL_UR_TRACE=0`,
`SYCL_CACHE_TRACE=0`, vendor verbose/JIT dump/profiling off, and preserve the
existing CUDA/Mesa cache-disable settings. Keep SYCL's in-memory compiled-code
cache available; it is transient within the worker. The deprecated
`SYCL_CACHE_DISABLE_PERSISTENT` has no effect according to
[Intel's runtime environment documentation](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md).
Do not enable attention fingerprint diagnostics, which print tensor values.
If initialization tries a forbidden write, diagnose with synthetic input and
bounded content-free data rather than weakening privacy for a real request.

These are integration requirements. No installed service or public worker was
modified, and no GPU quality, performance, cancellation, device loss, no-swap
or dynamic-runtime acceptance has been established by this change.
