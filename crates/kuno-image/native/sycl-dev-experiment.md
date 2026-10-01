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
GGML metadata only. It passed 199 accepted layouts/scales and 47 rejected cases,
including token/head-transposed V, padded strides, mask dimensions, masked GQA,
nonfinite/underflow/overflow scales, unsupported precision/types, and independently
contiguous malformed output extents on all four axes in masked/unmasked graphs. It does
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
`dpct/helper.hpp` return-path warning. That translation-unit check involved no GPU execution or full build. A later
genuine static parent-engine configuration compiled all 57 C++ translation units
and 15 third-party C units with the current exact 001 and the same 160-byte name
ABI, then archived their 72 objects. This does not establish a complete linked
SYCL worker or GPU acceptance.

## Development bridge and remaining build requirements

The Rust bridge now requires `KUNO_DEV_SYCL_BUILD=1` and an explicit absolute
`KUNO_ONEAPI_ROOT`. It pins 001 independently, then pins 002 outside the SYCL
directory and `003-sycl-convrot.patch` inside that directory. New files must be
intent-to-add for the exact diff. Ordinary builds still require Vulkan and the
unmodified 002 source contract. The private gate additionally requires actual
`SD_SYCL=ON`, `GGML_STATIC=ON`, static engine configuration, and an explicit
`-DGGML_MAX_NAME=160` in both CMake C/C++ flags. The initial standalone GGML
build used its default name size 64 and must not be mixed with engine headers.

Only the development shim is compiled with `KUNO_NATIVE_SYCL_DEV`. A standalone
Qwen 2.1 worker then requires exact `KUNO_EXPERIMENTAL_QWEN_SYCL=1` to select
SYCL. This request requires one matching physical PCI identity; absent or
duplicate SYCL candidates fail, and it does not benchmark a fallback backend.
Without the runtime flag, ordinary CUDA/Vulkan preference is preserved. The
existing public child environment does not forward this flag. SYCL reports its
own backend label and CUDA Fast/Sage attention remains unsupported on SYCL.

The shim requests resident FP16 prefix storage with the same conservative
9 GiB reserve as Vulkan. The current 001 model concat change now recognizes
SYCL alongside CUDA/Vulkan and preserves the cache's FP16 layout during reuse.
The cache oracle covers this backend; GPU numerical and performance acceptance
remain separate from this static source/build proof.

CPU backend-choice tests passed with and without the compile gate. A native
feature build-script test passed the exact build-flag parser; both shim variants
passed C++ syntax checks. The full development engine link and GPU execution
remain unverified. The following requirements still apply:

- Validate the engine revision and exact 001 diff independently of the GGML
  revision and exact combined 002 plus development SYCL diff. Include new files
  with intent-to-add; never bypass the existing exact-source check. Content
  address a fresh build directory with the complete patch bundle.
- The private build script links the static GGML SYCL archive and reviewed
  dynamic vendor closure, preserving the normal CUDA/Vulkan path. Its final link needs
  the `icpx -fsycl` driver to process SYCL device images; a normal GNU linker
  command alone is insufficient. The private wrapper adds `-fPIC` and an
  explicit trailing `-lsycl` because Rust supplies `-nodefaultlibs`. Use a separate Cargo target directory.
- A development CMake configuration uses the exact private `icpx` and `icx`,
  `SD_SYCL=ON`, static engine/GGML, no examples/WebP/WebM, the existing patched
  GGML, and explicit oneDNN/TBB/MKL package roots. Set `CMAKE_POSITION_INDEPENDENT_CODE=ON` for the engine objects used by
  Rust's PIE executable. Pass OpenMP C++ flags/library names as typed CMake
  `STRING` values, preserving the exact configuration validation. Initially
  disable SYCL graph replay, host-memory fallback and system-USM fallback; do not change driver or
  VMM behavior without measurement. Keeping Vulkan enabled satisfies the
  existing baseline adapter; the explicit development build gate also permits
  a SYCL-only private engine with Vulkan disabled.
- The extracted MKL directory currently contains headers/libraries but no
  `MKLConfig.cmake`; the pinned CMake requires `MKL::MKL_SYCL::BLAS`. Obtain the
  matching private config package or provide a reviewed private imported target
  with the verified dynamic closure. Do not let CMake silently find another MKL.
  The oneDNN package config exists and uses the private TBB config.
- The pinned SYCL properties originally did not set `device_id`. The 003 patch
  must publish a stable normalized BDF
  from `sycl::ext::intel::info::device::pci_address`, retain it in device context,
  and require a unique exact match. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` limits
  the backend but is not physical identity. Never select device zero by assumption.
- SYCL prefix caches, capability probes and working-memory budgets require
  deliberate validation. Do not label a SYCL route as Vulkan in diagnostics,
  silently fall back to a different card/CPU, or expose CUDA Sage attention as a
  supported SYCL choice.

Use the private compiler's final-link driver and a separate Cargo target:

```bash
KUNO_DEV_SYCL_BUILD=1 KUNO_ONEAPI_ROOT="$ONEAPI" \
KUNO_SD_SOURCE_DIR="$SOURCE" KUNO_SD_BUILD_DIR="$BUILD" \
CARGO_TARGET_DIR="$PRIVATE_CARGO_TARGET" \
RUSTFLAGS="-Clinker=$ONEAPI/compiler/2026.1/bin/icpx -Clink-arg=-fsycl" \
cargo build --locked -p kuno --release --features native-image
```

The genuine private CMake configuration uses these settings in addition to the
explicit compiler/package roots described above:

```bash
cmake -G Ninja -S "$SOURCE" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$ONEAPI/compiler/2026.1/bin/icx" \
  -DCMAKE_CXX_COMPILER="$ONEAPI/compiler/2026.1/bin/icpx" \
  -DCMAKE_C_FLAGS=-DGGML_MAX_NAME=160 \
  -DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=160 \
  -DSD_SYCL=ON -DSD_CUDA=OFF -DSD_VULKAN=OFF -DSD_METAL=OFF \
  -DSD_BUILD_SHARED_LIBS=OFF -DSD_BUILD_SHARED_GGML_LIB=OFF \
  -DSD_USE_UPSTREAM_GGML=OFF -DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF \
  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DGGML_BACKEND_DL=OFF \
  -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON \
  -DGGML_SYCL_GRAPH=OFF -DGGML_SYCL_HOST_MEM_FALLBACK=OFF -DGGML_NATIVE=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DOpenMP_CXX_FLAGS=-fiopenmp '-DOpenMP_CXX_LIB_NAMES=iomp5;pthread' \
  -DOpenMP_iomp5_LIBRARY="$ONEAPI/compiler/2026.1/lib/libiomp5.so" \
  -DOpenMP_pthread_LIBRARY=/lib/x86_64-linux-gnu/libpthread.a \
  -DMKL_DIR="$PRIVATE_MKL_CONFIG" \
  "-DCMAKE_PREFIX_PATH=$ONEAPI/compiler/2026.1;$ONEAPI/dnnl/2026.0;$ONEAPI/tbb/2023.1"
```

Supply the reviewed bundle's explicit library directories to the configure and
build processes. `GGML_STATIC` adds `-static` to GGML's CMake link probes; the
parent's global `-fsycl` makes automatic C++ OpenMP detection fail with that
static probe. The explicit `-fiopenmp`/`iomp5` paths above configure the actual
compiler/link target, and a separate dynamic CPU-only link probe verifies them.
Do not edit cache text to bypass this failure.

Final Rust linking must remain dynamic for the vendor runtime even though the
engine/GGML archives are static. A private linker wrapper can invoke the exact
`icpx -fsycl -Xs -ze-intel-greater-than-4GB-buffer-required` driver and supply its
explicit compiler/MKL/TBB/oneDNN/UMF/TCM library paths. Do not propagate CMake's
global `-static` into that link. A CPU-only wrapper probe linked successfully and
its ELF dependencies included `libsycl.so.9` and `libiomp5.so`; this does not
validate SYCL device images or a runtime launch.

Reusing completed PIC GGML objects is a development optimization only. Require
exact 002/003 source diffs, `GGML_MAX_NAME=160`, matching ISA/FP16/oneDNN/OpenMP
flags, disabled graph/host fallback, and finalized archive hashes before reuse.
Linux shared-export visibility and PIC differences must be recorded explicitly;
64-name objects are incompatible. The native parent objects must still be built
from its genuine configuration and exact current 001. Never copy mutable peer
build output or claim that copied archives were compiled by the static config.

The CMake OpenMP configuration must use the reviewed `iomp5` runtime for private
SYCL; normal builds retain GNU OpenMP. The standalone launcher needs explicit
library paths for direct and `dlopen` dependencies. Compilation and final link
do not prove that a runtime will launch with the privacy restrictions applied.

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
`SYCL_CACHE_PERSISTENT=0`, `SYCL_UR_TRACE=0`, and leave
`SYCL_DUMP_IMAGES` and `SYCL_CACHE_TRACE` unset in the cleared environment.
Setting `SYCL_DUMP_IMAGES=0` still activates dumping in the tested runtime;
Landlock rejected the attempted SPIR-V file write during synthetic initialization.
Keep vendor verbose/JIT dump/profiling off, and preserve the
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
