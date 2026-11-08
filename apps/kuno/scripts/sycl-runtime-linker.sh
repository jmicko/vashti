#!/bin/sh
# Developer-only final-link recipe. Never use for the standard gateway package.
# With KUNO_SYCL_AOT_DEVICE (build time, e.g. bmg-g31) and KUNO_OCLOC_DIR (the
# directory holding ocloc), the GGML SYCL device code is compiled ahead of time
# for that GPU, so workers start without JIT-compiling every kernel.
set -eu
: "${KUNO_ONEAPI_ROOT:?Set KUNO_ONEAPI_ROOT to the reviewed extracted oneAPI bundle}"
kuno_sdk_root="$KUNO_ONEAPI_ROOT"
ONEAPI_ROOT="$kuno_sdk_root"
LD_LIBRARY_PATH="$kuno_sdk_root/compiler/2026.1/lib:$kuno_sdk_root/mkl/2026.1/lib:$kuno_sdk_root/tbb/2023.1/lib:$kuno_sdk_root/dnnl/2026.0/lib:$kuno_sdk_root/umf/1.1/lib:$kuno_sdk_root/tcm/1.5/lib"
large_buffers="-ze-intel-greater-than-4GB-buffer-required"
aot=""
case "$*" in *ggml-sycl*) aot="${KUNO_SYCL_AOT_DEVICE:-}" ;; esac
if [ -n "$aot" ]; then
  : "${KUNO_OCLOC_DIR:?Set KUNO_OCLOC_DIR to the directory containing ocloc for AOT}"
  PATH="$KUNO_OCLOC_DIR:$PATH"
  LD_LIBRARY_PATH="${KUNO_OCLOC_LIB_DIR:-$KUNO_OCLOC_DIR}:$LD_LIBRARY_PATH"
  set -- -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \
    "-device $aot -internal_options $large_buffers" "$@"
else
  set -- -Xs "$large_buffers" "$@"
fi
export ONEAPI_ROOT LD_LIBRARY_PATH PATH
exec "$kuno_sdk_root/compiler/2026.1/bin/icpx" \
  -fsycl -fPIC -shared-intel "$@" \
  -L"$kuno_sdk_root/compiler/2026.1/lib" -lsycl -lsvml -lirng -limf -lintlc -lirc \
  -L"$kuno_sdk_root/mkl/2026.1/lib" \
  -Wl,--no-as-needed -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
  -Wl,--as-needed
