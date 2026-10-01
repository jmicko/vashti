# Private oneMKL matrix probe

This development-only synthetic executable is not linked into Kuno or its normal
native build. It establishes arithmetic and kernel throughput before accepting
an optional SYCL dependency. It loads no models, requests, prompts, or images.

Use a separately verified Intel oneAPI compiler/MKL runtime. The tested bundle
used compiler 2026.1.1-325 and MKL 2026.1.0-236, unpacked privately without
changing system packages or drivers. With `ONEAPI_ROOT` pointing at its
`opt/intel/oneapi`, compile:

```bash
"$ONEAPI_ROOT/compiler/2026.1/bin/icpx" -fsycl -std=c++17 -O2 -DMKL_ILP64 \
  -I "$ONEAPI_ROOT/mkl/2026.1/include" \
  crates/kuno-image/native/oneapi-matrix-benchmark.cpp \
  -L "$ONEAPI_ROOT/mkl/2026.1/lib" \
  -L "$ONEAPI_ROOT/compiler/2026.1/lib" \
  -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
  -lpthread -ldl -lm -o target/oneapi-matrix-benchmark
```

Supply the verified compiler, MKL, TBB, UMF and TCM runtime directories through
`LD_LIBRARY_PATH`; do not install a build-path rpath in a release binary.
First run `--cpu-oracle`, which never enumerates GPUs. GPU runs require explicit
ownership of an idle card, then `--pci 0000:2d:00.0 --shape 0 --iterations 1` for
tiny cases and `--shape 2 --iterations 5` for the large fused MLP shape. Set
`ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_VERBOSE=0` and `SYCL_UR_TRACE=0`. The PCI address must match exactly; no
other GPU or CPU fallback is accepted.

Type 1 is signed I8×I8→I32; type 2 is the same integer-valued F16×F16→F32.
Four tiny cases independently validate every output, including odd/padded rows,
zero rows and signed extrema. Large cases add 64 independent CPU dot products
and compare every integer/half output. Inputs deliberately bound all sums to
exact F32 integers; this does not compare real ConvRot activation quantization
against reconstructed dense models. Cleanup drains submitted work before freeing
host/device buffers, and checks asynchronous errors after each wait.

First-call and warmed wall time include the completed GEMM, not input uploads or
result readback. Earlier tiny kernels may warm shared library state; first-call
is not a process-cold measurement. The largest case explicitly owns
1,149,272,064 device bytes plus the same host amount, excluding vendor workspace.

See [measured results](../../../docs/kuno-image-performance.md#private-intel-vendor-library-probe).
oneMKL SYCL's dynamic linking requirements must be resolved explicitly before
shipping an optional backend. No new dependency is implied for the ordinary
static CUDA/Vulkan engine.
