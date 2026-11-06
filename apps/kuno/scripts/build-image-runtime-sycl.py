#!/usr/bin/env python3
"""Build the selected managed SYCL worker; never execute, install or package it.

--plan validates complete pinned source and prints commands without configuring
or compiling. A real build requires a reviewed oneAPI/MKL configuration and
fresh native/Cargo output directories. The standard CUDA/Vulkan builder is separate.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
CLEARED_BUILD_ENV = ("CFLAGS", "CXXFLAGS", "LDFLAGS", "CARGO_ENCODED_RUSTFLAGS")
# Ahead-of-time device code for the supported Intel GPU (Arc Pro B70); the
# worker then starts without JIT-compiling the GGML SYCL kernels.
AOT_DEVICE = "bmg-g31"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments])


def reject_untracked_source(source):
    # CMake GLOB can compile files absent from git diff. Do not use
    # --exclude-standard: ignored files must be rejected, too.
    require(not git(source, "ls-files", "--others", "--", ".", ":(exclude)ggml"),
            "engine source contains untracked or ignored files")
    require(not git(source / "ggml", "ls-files", "--others"),
            "GGML source contains untracked or ignored files")


def selected_pins():
    tree = ast.parse((ROOT / "apps/kuno/scripts/package-image-runtime.py").read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(isinstance(name, ast.Name)
                and name.id == "PINS" for name in node.targets))


def plan(args):
    pins = selected_pins()
    source = args.source.resolve()
    reject_untracked_source(source)
    sdk = args.oneapi.resolve()
    patches = ROOT / "apps/kuno/patches"
    engine_patch = patches / "001-qwen-quantized-config.patch"
    ggml_patch = patches / "002-vulkan-flash-shapes.patch"
    sycl_patch = patches / "003-sycl-convrot.patch"
    for key, patch in [("engine_patch_sha256", engine_patch),
                       ("ggml_patch_sha256", ggml_patch),
                       ("sycl_patch_sha256", sycl_patch)]:
        require(digest(patch) == pins[key], "selected package/source pins differ")
    require(git(source, "rev-parse", "HEAD").decode().strip() == pins["engine_revision"],
            "engine source pin differs")
    require(git(source / "ggml", "rev-parse", "HEAD").decode().strip() == pins["ggml_revision"],
            "GGML source pin differs")
    base = ["diff", "--no-color", "--no-ext-diff", "--binary", "HEAD", "--"]
    require(git(source, *base, ".", ":(exclude)ggml") == engine_patch.read_bytes(),
            "engine complete source diff differs")
    require(git(source / "ggml", *base, ".", ":(exclude)src/ggml-sycl") == ggml_patch.read_bytes(),
            "GGML complete non-SYCL diff differs")
    require(git(source / "ggml", *base, "src/ggml-sycl") == sycl_patch.read_bytes(),
            "GGML complete SYCL diff differs")
    bundle = hashlib.sha256(engine_patch.read_bytes() + ggml_patch.read_bytes() + sycl_patch.read_bytes()).hexdigest()[:16]
    output = (args.output or ROOT / "target" / ("kuno-managed-sycl-" + bundle)).resolve()
    native, cargo = output / "native", output / "cargo"
    require(not native.exists() and not cargo.exists(), "preserve existing native/Cargo artifacts; choose fresh output")
    compiler = sdk / "compiler/2026.1"
    mkl = str(args.mkl_config.resolve()) if args.mkl_config else "<KUNO_MKL_CONFIG_DIR>"
    configure = ["cmake", "-G", "Ninja", "-S", str(source), "-B", str(native),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_C_COMPILER=" + str(compiler / "bin/icx"),
        "-DCMAKE_CXX_COMPILER=" + str(compiler / "bin/icpx"),
        "-DCMAKE_C_FLAGS=-DGGML_MAX_NAME=160", "-DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME=160",
        "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG", "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG",
        "-DSD_SYCL=ON", "-DSD_CUDA=OFF", "-DSD_VULKAN=OFF", "-DSD_METAL=OFF",
        "-DSD_BUILD_SHARED_LIBS=OFF", "-DSD_BUILD_SHARED_GGML_LIB=OFF", "-DSD_USE_UPSTREAM_GGML=OFF",
        "-DSD_BUILD_EXAMPLES=OFF", "-DSD_WEBP=OFF", "-DSD_WEBM=OFF", "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_STATIC=ON", "-DGGML_BACKEND_DL=OFF", "-DGGML_SYCL=ON", "-DGGML_SYCL_TARGET=INTEL",
        "-DGGML_SYCL_F16=ON", "-DGGML_SYCL_DNN=ON", "-DGGML_SYCL_GRAPH=OFF",
        "-DGGML_SYCL_HOST_MEM_FALLBACK=OFF", "-DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON",
        "-DGGML_NATIVE=OFF", "-DGGML_BUILD_TESTS=OFF", "-DGGML_BUILD_EXAMPLES=OFF",
        "-DKUNO_SYCL_WHOLE_ROPE_BYPASS=ON", "-DGGML_SYCL_DEVICE_ARCH=" + AOT_DEVICE,
        "-DOpenMP_CXX_FLAGS=-fiopenmp", "-DOpenMP_CXX_LIB_NAMES=iomp5;pthread",
        "-DOpenMP_iomp5_LIBRARY=" + str(compiler / "lib/libiomp5.so"),
        "-DOpenMP_pthread_LIBRARY=/lib/x86_64-linux-gnu/libpthread.a", "-DMKL_DIR=" + mkl,
        "-DCMAKE_PREFIX_PATH=" + ";".join(str(sdk / name) for name in ["compiler/2026.1", "dnnl/2026.0", "tbb/2023.1"])]
    libs = ":".join(str(sdk / name / "lib") for name in ["compiler/2026.1", "mkl/2026.1", "tbb/2023.1", "dnnl/2026.0", "umf/1.1", "tcm/1.5"])
    environment = dict(ONEAPI_ROOT=str(sdk), KUNO_ONEAPI_ROOT=str(sdk), LD_LIBRARY_PATH=libs,
        KUNO_DEV_SYCL_BUILD="1", KUNO_SD_SOURCE_DIR=str(source), KUNO_SD_BUILD_DIR=str(native),
        CARGO_TARGET_DIR=str(cargo), CARGO_BUILD_JOBS="1",
        RUSTFLAGS="-Clinker=" + str(ROOT / "apps/kuno/scripts/sycl-runtime-linker.sh"),
        KUNO_SYCL_AOT_DEVICE=AOT_DEVICE,
        KUNO_OCLOC_DIR=str(args.ocloc.resolve()) if args.ocloc else "<KUNO_OCLOC_DIR>")
    return dict(source_pins=pins, source=str(source), output=str(output), configure=configure,
                build=["cmake", "--build", str(native), "--parallel", "1"],
                cargo=["cargo", "build", "--locked", "--release", "-p", "kuno", "--features", "native-image"],
                environment=environment, worker=str(cargo / "release/kuno"),
                O_level="-O3", whole_default="source remains unset/OFF; managed builder explicitly ON",
                missing_inputs=([] if args.mkl_config and (args.mkl_config / "MKLConfig.cmake").is_file() else ["reviewed MKLConfig.cmake"]) +
                               ([] if args.ocloc and (args.ocloc / "ocloc").is_file() else ["ocloc for " + AOT_DEVICE + " AOT"]),
                cleared_environment=list(CLEARED_BUILD_ENV),
                runtime_execution=False, installation=False, packaging=False)


def build_environment(overrides):
    environment = os.environ.copy()
    for key in CLEARED_BUILD_ENV:
        environment.pop(key, None)
    environment.update(overrides)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.environ.get("KUNO_SD_SOURCE_DIR"), required="KUNO_SD_SOURCE_DIR" not in os.environ)
    parser.add_argument("--oneapi", type=Path, default=os.environ.get("KUNO_ONEAPI_ROOT"), required="KUNO_ONEAPI_ROOT" not in os.environ)
    parser.add_argument("--mkl-config", type=Path, default=os.environ.get("KUNO_MKL_CONFIG_DIR"))
    parser.add_argument("--ocloc", type=Path, default=os.environ.get("KUNO_OCLOC_DIR"),
                        help="directory containing ocloc (Intel GPU offline compiler) for AOT device code")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    try:
        receipt = plan(args)
        if args.plan:
            print(json.dumps(receipt, indent=2))
            return
        require(not receipt["missing_inputs"], "set KUNO_MKL_CONFIG_DIR to the reviewed MKL CMake configuration and KUNO_OCLOC_DIR to ocloc")
        for name in ["compiler/2026.1/bin/icx", "compiler/2026.1/bin/icpx", "compiler/2026.1/lib/libiomp5.so"]:
            require((args.oneapi / name).is_file(), "reviewed oneAPI build input missing")
        environment = build_environment(receipt["environment"])
        output = Path(receipt["output"])
        output.mkdir(parents=True, exist_ok=True)
        for name in ["configure", "build", "cargo"]:
            subprocess.run(receipt[name], cwd=ROOT, env=environment, check=True)
        receipt["worker_sha256"] = digest(Path(receipt["worker"]))
        (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(receipt["worker"])
    except (OSError, ValueError, StopIteration, subprocess.CalledProcessError):
        parser.exit(1, "Managed SYCL build verification failed; no worker approved.\n")


if __name__ == "__main__":
    main()
