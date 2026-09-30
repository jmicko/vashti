#!/usr/bin/env bash
# Optional development build. The resulting executable embeds the image engine;
# installed users need Vulkan drivers, not Python, CMake, or a C++ compiler.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
source_dir="$root/target/kuno-sd-source-88411ef"
build_dir="$root/target/kuno-sd-build-88411ef-int8"
cuda_options=(-DSD_CUDA=OFF)
cuda_libraries=()
case ${KUNO_IMAGE_CUDA:-0} in
    0) ;;
    1)
        : "${KUNO_CUDA_ROOT:?Set KUNO_CUDA_ROOT to a CUDA toolkit with static cuBLAS libraries}"
        if [[ ! -x "$KUNO_CUDA_ROOT/bin/nvcc" ]]; then
            echo 'KUNO_CUDA_ROOT must contain bin/nvcc.' >&2
            exit 1
        fi
        build_dir="$root/target/kuno-sd-build-88411ef-cuda"
        cuda_options=(-DSD_CUDA=ON -DGGML_STATIC=ON
            "-DCUDAToolkit_ROOT=$KUNO_CUDA_ROOT"
            "-DCMAKE_CUDA_COMPILER=$KUNO_CUDA_ROOT/bin/nvcc"
            "-DCMAKE_CUDA_ARCHITECTURES=${KUNO_CUDA_ARCHITECTURES:-native}")
        cuda_lib="$KUNO_CUDA_ROOT/lib64"
        [[ -d "$cuda_lib" ]] || cuda_lib="$KUNO_CUDA_ROOT/lib"
        cuda_libraries=("$build_dir/ggml/src/ggml-cuda/libggml-cuda.a"
            "$cuda_lib/libcublas_static.a" "$cuda_lib/libcublasLt_static.a"
            "$cuda_lib/libcudart_static.a" "$cuda_lib/libculibos.a" -lcuda -ldl -lrt -lpthread)
        ;;
    *) echo 'KUNO_IMAGE_CUDA must be 0 or 1.' >&2; exit 1 ;;
esac
revision=88411ef1e0688ff2df1010aeeb5d92b2d8cea2be
for command in git cmake ninja cargo c++; do
    if ! command -v "$command" >/dev/null; then
        echo "Missing build tool: $command" >&2
        echo 'On Linux Mint/Ubuntu: sudo apt install build-essential cmake ninja-build glslc libvulkan-dev spirv-headers glslang-tools' >&2
        echo 'Install the Rust toolchain separately if cargo is missing.' >&2
        exit 1
    fi
done
if [[ ! -e "$source_dir" ]]; then
    mkdir -p -- "$(dirname -- "$source_dir")"
    git init "$source_dir"
    git -C "$source_dir" remote add origin https://github.com/leejet/stable-diffusion.cpp.git
    git -C "$source_dir" fetch --depth 1 origin "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
    git -C "$source_dir" submodule update --init --recursive --depth 1
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$revision" ]]; then
    echo 'Native source is not the pinned revision; preserving the existing directory.' >&2
    exit 1
fi
patch_file="$root/apps/kuno/patches/001-qwen-quantized-config.patch"
if [[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]]; then
    git -C "$source_dir" apply --check "$patch_file"
    git -C "$source_dir" apply "$patch_file"
fi
# Require exactly our reviewed patch; never overwrite local dependency edits.
if ! git -C "$source_dir" diff --no-color --no-ext-diff --binary HEAD | cmp -s "$patch_file" -; then
    echo 'Native source has changes beyond the pinned Kuno patch; preserving them.' >&2
    exit 1
fi
vulkan_options=()
if [[ -n ${VULKAN_SDK:-} ]]; then
    glslc_path="$VULKAN_SDK/bin/glslc"
    vulkan_options+=("-DVulkan_INCLUDE_DIR=$VULKAN_SDK/include")
elif glslc_path=$(command -v glslc); then
    :
else
    echo 'Missing glslc. Set VULKAN_SDK to a recent LunarG SDK x86_64 directory.' >&2
    exit 1
fi
mkdir -p -- "$build_dir"
if ! "$glslc_path" --target-env=vulkan1.3 \
    "$source_dir/ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/integer_dot.comp" \
    -o "$build_dir/kuno-integer-dot-probe.spv"; then
    echo 'Qwen INT8 requires glslc with GL_EXT_integer_dot_product support.' >&2
    echo 'Use a recent LunarG Vulkan SDK and set VULKAN_SDK to its x86_64 directory.' >&2
    exit 1
fi
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    "-DVulkan_GLSLC_EXECUTABLE=$glslc_path" "${vulkan_options[@]}" \
    "${cuda_options[@]}" \
    -DSD_VULKAN=ON -DSD_BUILD_EXAMPLES=OFF -DSD_WEBP=OFF -DSD_WEBM=OFF \
    -DSD_BUILD_SHARED_LIBS=OFF -DSD_USE_UPSTREAM_GGML=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" --parallel "${KUNO_BUILD_JOBS:-6}"
# Exercise quantized embedding math against a dense, independent synthetic oracle.
c++ -std=c++17 -O2 -DGGML_MAX_NAME=160 \
    -I "$source_dir/src" -I "$source_dir/ggml/include" -I "$source_dir/include" \
    "$root/crates/kuno-image/native/embedding-test.cpp" \
    -Wl,--start-group "$build_dir/libstable-diffusion.a" \
    "$build_dir"/ggml/src/libggml*.a "$build_dir/ggml/src/ggml-vulkan/libggml-vulkan.a" \
    "${cuda_libraries[@]}" \
    -Wl,--end-group -lgomp -lvulkan -o "$build_dir/kuno-embedding-test"
"$build_dir/kuno-embedding-test"
# Synthetic CPU oracle/fallback regression; optional device acceptance is run
# separately with kuno-attention-layout-test --device CUDA <PCI address>.
c++ -std=c++17 -O2 -DGGML_MAX_NAME=160 \
    -I "$source_dir/src" -I "$source_dir/ggml/include" -I "$source_dir/include" \
    "$root/crates/kuno-image/native/attention-layout-test.cpp" \
    -Wl,--start-group "$build_dir/libstable-diffusion.a" \
    "$build_dir"/ggml/src/libggml*.a "$build_dir/ggml/src/ggml-vulkan/libggml-vulkan.a" \
    "${cuda_libraries[@]}" \
    -Wl,--end-group -lgomp -lvulkan -o "$build_dir/kuno-attention-layout-test"
"$build_dir/kuno-attention-layout-test"
cd -- "$root"
KUNO_SD_SOURCE_DIR="$source_dir" KUNO_SD_BUILD_DIR="$build_dir" \
    cargo build --locked -p kuno --release --features native-image
