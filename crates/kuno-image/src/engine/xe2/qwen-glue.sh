#!/bin/sh
# Rebuilds qwen.bin from qwen.cl: OpenCL C -> SPIR-V (oneAPI clang + llvm-spirv) -> native B70
# (bmg-g31) module (ocloc). Usage: qwen-glue.sh <oneAPI root> <ocloc binary>
set -eu
dir=$(cd "$(dirname "$0")" && pwd)
sdk=$1; ocloc=$2
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
LD_LIBRARY_PATH="$sdk/compiler/2026.1/lib" "$sdk/compiler/2026.1/bin/compiler/clang" -cc1 \
  -triple spir64-unknown-unknown -cl-std=CL3.0 -finclude-default-header \
  -cl-ext=+cl_khr_fp64,+cl_khr_fp16,+cl_intel_subgroups,+cl_khr_subgroups,+__opencl_c_subgroups,+__opencl_c_fp64 \
  -emit-llvm-bc -O2 "$dir/qwen.cl" -o "$tmp/qwen.bc"
LD_LIBRARY_PATH="$sdk/compiler/2026.1/lib" "$sdk/compiler/2026.1/bin/compiler/llvm-spirv" "$tmp/qwen.bc" -o "$tmp/qwen.spv"
(cd "$tmp" && "$ocloc" compile -spirv_input -file "$tmp/qwen.spv" -device bmg-g31 -output qwen)
cp "$tmp/qwen_bmg.bin" "$dir/qwen.bin"
