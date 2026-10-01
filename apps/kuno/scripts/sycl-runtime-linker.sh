#!/bin/sh
# Developer-only final-link recipe. Never use for the standard gateway package.
set -eu
: "${KUNO_ONEAPI_ROOT:?Set KUNO_ONEAPI_ROOT to the reviewed extracted oneAPI bundle}"
kuno_sdk_root="$KUNO_ONEAPI_ROOT"
ONEAPI_ROOT="$kuno_sdk_root"
LD_LIBRARY_PATH="$kuno_sdk_root/compiler/2026.1/lib:$kuno_sdk_root/mkl/2026.1/lib:$kuno_sdk_root/tbb/2023.1/lib:$kuno_sdk_root/dnnl/2026.0/lib:$kuno_sdk_root/umf/1.1/lib:$kuno_sdk_root/tcm/1.5/lib"
export ONEAPI_ROOT LD_LIBRARY_PATH
exec "$kuno_sdk_root/compiler/2026.1/bin/icpx" \
  -fsycl -fPIC -shared-intel -Xs -ze-intel-greater-than-4GB-buffer-required "$@" \
  -L"$kuno_sdk_root/compiler/2026.1/lib" -lsycl -lsvml -lirng -limf -lintlc -lirc \
  -L"$kuno_sdk_root/mkl/2026.1/lib" \
  -Wl,--no-as-needed -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
  -Wl,--as-needed
