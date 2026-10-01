fn main() {
    println!("cargo:rustc-check-cfg=cfg(kuno_native_cuda)");
    println!("cargo:rustc-check-cfg=cfg(kuno_native_sycl_dev)");
    println!("cargo:rerun-if-env-changed=KUNO_SD_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=KUNO_SD_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=KUNO_DEV_SYCL_BUILD");
    println!("cargo:rerun-if-env-changed=KUNO_ONEAPI_ROOT");
    println!("cargo:rerun-if-changed=native/shim.cpp");
    println!("cargo:rerun-if-changed=native/cache-budget.h");
    println!("cargo:rerun-if-changed=native/backend-choice.h");
    println!("cargo:rerun-if-changed=native/experimental-precision.h");
    #[cfg(feature = "native-sd")]
    {
        use std::{path::PathBuf, process::Command};
        let dev_sycl = dev_sycl_flag(std::env::var("KUNO_DEV_SYCL_BUILD").ok().as_deref())
            .expect("KUNO_DEV_SYCL_BUILD must be absent, 0, or exactly 1");
        let source = PathBuf::from(
            std::env::var("KUNO_SD_SOURCE_DIR")
                .expect("set KUNO_SD_SOURCE_DIR to the pinned source checkout"),
        );
        let build = PathBuf::from(
            std::env::var("KUNO_SD_BUILD_DIR")
                .expect("set KUNO_SD_BUILD_DIR to its static Vulkan build"),
        );
        let revision = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&source)
            .output()
            .expect("read native revision");
        assert!(revision.status.success());
        assert_eq!(
            String::from_utf8(revision.stdout).unwrap().trim(),
            "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be",
            "unsupported stable-diffusion.cpp revision"
        );
        let patch = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../../apps/kuno/patches/001-qwen-quantized-config.patch");
        let diff = Command::new("git")
            .args([
                "diff",
                "--no-color",
                "--no-ext-diff",
                "--binary",
                "HEAD",
                "--",
                ".",
                ":(exclude)ggml",
            ])
            .current_dir(&source)
            .output()
            .expect("read native patch");
        println!("cargo:rerun-if-changed={}", patch.display());
        println!(
            "cargo:rerun-if-changed={}",
            source.join("src/model/common/ggml_block.hpp").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            source.join("src/model/te/llm.hpp").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            source.join("src/core/ggml_extend.cpp").display()
        );
        println!(
            "cargo:rerun-if-changed={}",
            source.join("src/pipeline/diffusion_engine.cpp").display()
        );
        for file in [
            "src/core/runner_cache.h",
            "src/core/runner_cache.cpp",
            "src/core/ggml_runner.cpp",
            "src/core/params_buffer_type.h",
            "src/model_manager.cpp",
            "src/model_loader.cpp",
            "src/model_loader.h",
            "src/model_loader_files.cpp",
            "src/model_io/int8_dense.h",
        ] {
            println!("cargo:rerun-if-changed={}", source.join(file).display());
        }
        println!(
            "cargo:rerun-if-changed={}",
            source
                .join("src/model/diffusion/qwen_image_2_1.hpp")
                .display()
        );
        let expected = std::fs::read(patch).expect("native compatibility patch");
        assert!(
            diff.status.success() && diff.stdout == expected,
            "native source must contain exactly the pinned Kuno patch"
        );
        let ggml = source.join("ggml");
        let ggml_revision = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(&ggml)
            .output()
            .expect("read GGML revision");
        assert!(ggml_revision.status.success());
        assert_eq!(
            String::from_utf8(ggml_revision.stdout).unwrap().trim(),
            "4bf5f6000653b7881d00963cd6ddb665ccd62a8d",
            "unsupported GGML revision"
        );
        let ggml_patch = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../../apps/kuno/patches/002-vulkan-flash-shapes.patch");
        println!("cargo:rerun-if-changed={}", ggml_patch.display());
        let mut ggml_command = Command::new("git");
        ggml_command.args(["diff", "--no-color", "--no-ext-diff", "--binary", "HEAD"]);
        if dev_sycl {
            ggml_command.args(["--", ".", ":(exclude)src/ggml-sycl"]);
        }
        let ggml_diff = ggml_command
            .current_dir(&ggml)
            .output()
            .expect("read GGML patch");
        assert!(
            ggml_diff.status.success()
                && ggml_diff.stdout == std::fs::read(ggml_patch).expect("GGML compatibility patch"),
            "GGML source must contain exactly the pinned Kuno patch"
        );
        if dev_sycl {
            let sycl_patch = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
                .join("../../apps/kuno/patches/003-sycl-convrot.patch");
            println!("cargo:rerun-if-changed={}", sycl_patch.display());
            let sycl_diff = Command::new("git")
                .args([
                    "diff",
                    "--no-color",
                    "--no-ext-diff",
                    "--binary",
                    "HEAD",
                    "--",
                    "src/ggml-sycl",
                ])
                .current_dir(&ggml)
                .output()
                .expect("read development SYCL patch");
            assert!(
                sycl_diff.status.success()
                    && sycl_diff.stdout
                        == std::fs::read(sycl_patch).expect("pinned development SYCL patch"),
                "development SYCL source must contain exactly the separate pinned patch"
            );
            for file in [
                "CMakeLists.txt",
                "ggml-sycl.cpp",
                "convrot.cpp",
                "convrot.hpp",
                "convrot-policy.hpp",
                "pci-address.hpp",
                "im2col.cpp",
                "im2col-launch-policy.hpp",
                "fattn-onednn.cpp",
                "fattn-onednn-shapes.hpp",
            ] {
                println!(
                    "cargo:rerun-if-changed={}",
                    ggml.join("src/ggml-sycl").join(file).display()
                );
            }
        }
        for file in [
            "src/ggml-vulkan/ggml-vulkan.cpp",
            "src/ggml-vulkan/vulkan-int8-m8-tuning.h",
            "src/ggml-vulkan/CMakeLists.txt",
            "src/ggml-vulkan/vulkan-shaders/CMakeLists.txt",
            "src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp",
            "src/ggml-vulkan/vulkan-shaders/flash_attn_cm1_m8.comp",
            "src/ggml-vulkan/vulkan-shaders/mul_mat_i8_tensorwise_m8_cm1.comp",
        ] {
            println!("cargo:rerun-if-changed={}", ggml.join(file).display());
        }
        let cache =
            std::fs::read_to_string(build.join("CMakeCache.txt")).expect("native build cache");
        println!(
            "cargo:rerun-if-changed={}",
            build.join("CMakeCache.txt").display()
        );
        for required in [
            "SD_USE_UPSTREAM_GGML:BOOL=OFF",
            "SD_BUILD_SHARED_LIBS:BOOL=OFF",
        ] {
            assert!(
                cache.lines().any(|l| l == required),
                "native build must retain patched FP8 support and static Vulkan linkage"
            );
        }
        let vulkan = cache.lines().any(|line| line == "SD_VULKAN:BOOL=ON");
        if dev_sycl {
            for required in ["SD_SYCL:BOOL=ON", "GGML_STATIC:BOOL=ON"] {
                assert!(
                    cache.lines().any(|line| line == required),
                    "development SYCL build configuration missing {required}"
                );
            }
            // Standalone GGML defaults to a different tensor-name size. Every
            // private archive must share the engine/shim's explicit ABI.
            for prefix in ["CMAKE_C_FLAGS:STRING=", "CMAKE_CXX_FLAGS:STRING="] {
                assert!(
                    cache
                        .lines()
                        .filter_map(|line| line.strip_prefix(prefix))
                        .any(|flags| flags
                            .split_whitespace()
                            .any(|flag| flag == "-DGGML_MAX_NAME=160")),
                    "private SYCL archives require GGML_MAX_NAME=160 in both C/C++ flags"
                );
            }
            println!("cargo:rustc-cfg=kuno_native_sycl_dev");
        } else {
            assert!(vulkan, "ordinary native image builds require Vulkan");
        }
        let mut shim = cc::Build::new();
        if dev_sycl {
            shim.define("KUNO_NATIVE_SYCL_DEV", "1");
        }
        shim.cpp(true)
            .define("GGML_MAX_NAME", "160")
            .flag_if_supported("-std=c++17")
            .include(source.join("include"))
            .include(source.join("ggml/include"))
            .file("native/shim.cpp")
            .compile("kuno_sd_shim");
        for archive in [
            "libstable-diffusion.a",
            "ggml/src/libggml.a",
            "ggml/src/libggml-base.a",
            "ggml/src/libggml-cpu.a",
        ] {
            println!("cargo:rerun-if-changed={}", build.join(archive).display());
        }
        for directory in [&build, &build.join("ggml/src")] {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
        for name in ["stable-diffusion", "ggml", "ggml-cpu"] {
            println!("cargo:rustc-link-lib=static={name}");
        }
        if vulkan {
            let directory = build.join("ggml/src/ggml-vulkan");
            println!("cargo:rustc-link-search=native={}", directory.display());
            println!(
                "cargo:rerun-if-changed={}",
                directory.join("libggml-vulkan.a").display()
            );
            println!("cargo:rustc-link-lib=static=ggml-vulkan");
        }
        if dev_sycl {
            let sycl = build.join("ggml/src/ggml-sycl");
            assert!(
                sycl.join("libggml-sycl.a").is_file(),
                "development SYCL static archive missing"
            );
            println!("cargo:rustc-link-search=native={}", sycl.display());
            println!(
                "cargo:rerun-if-changed={}",
                sycl.join("libggml-sycl.a").display()
            );
            println!("cargo:rustc-link-lib=static=ggml-sycl");
        }
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=stdc++");
        if dev_sycl {
            let root = PathBuf::from(
                std::env::var("KUNO_ONEAPI_ROOT")
                    .expect("set KUNO_ONEAPI_ROOT to the reviewed private bundle"),
            );
            assert!(
                root.is_absolute() && root.is_dir(),
                "KUNO_ONEAPI_ROOT must be an existing absolute directory"
            );
            for (directory, library) in [
                ("compiler/2026.1/lib", "sycl"),
                ("mkl/2026.1/lib", "mkl_sycl_blas"),
                ("mkl/2026.1/lib", "mkl_intel_ilp64"),
                ("mkl/2026.1/lib", "mkl_sequential"),
                ("mkl/2026.1/lib", "mkl_core"),
                ("dnnl/2026.0/lib", "dnnl"),
                ("tbb/2023.1/lib", "tbb"),
            ] {
                let path = root.join(directory).join(format!("lib{library}.so"));
                assert!(
                    path.is_file(),
                    "private SYCL dynamic library missing: {}",
                    path.display()
                );
                println!(
                    "cargo:rustc-link-search=native={}",
                    path.parent().unwrap().display()
                );
                println!("cargo:rerun-if-changed={}", path.display());
                println!("cargo:rustc-link-lib=dylib={library}");
            }
            // Runtime dlopen dependencies also need these private directories;
            // root's standalone launcher supplies an explicit clean environment.
            for directory in ["umf/1.1/lib", "tcm/1.5/lib"] {
                assert!(
                    root.join(directory).is_dir(),
                    "private SYCL runtime directory missing"
                );
                println!(
                    "cargo:rustc-link-search=native={}",
                    root.join(directory).display()
                );
            }
            if cache
                .lines()
                .any(|line| line == "GGML_OPENMP_ENABLED:INTERNAL=ON")
            {
                assert!(
                    cache
                        .lines()
                        .any(|line| line == "OpenMP_CXX_LIB_NAMES:STRING=iomp5;pthread"
                            || line == "OpenMP_CXX_LIB_NAMES:STRING=iomp5"),
                    "private SYCL build must use the reviewed Intel OpenMP runtime"
                );
                let openmp = root.join("compiler/2026.1/lib/libiomp5.so");
                assert!(openmp.is_file(), "private Intel OpenMP runtime missing");
                println!("cargo:rerun-if-changed={}", openmp.display());
                println!("cargo:rustc-link-lib=dylib=iomp5");
            }
            for name in ["dl", "rt", "pthread", "m"] {
                println!("cargo:rustc-link-lib={name}");
            }
        } else {
            println!("cargo:rustc-link-lib=gomp");
        }
        if vulkan {
            println!("cargo:rustc-link-lib=vulkan");
        }
        if cache.lines().any(|line| line == "SD_CUDA:BOOL=ON") {
            println!("cargo:rustc-cfg=kuno_native_cuda");
            assert!(
                cache.lines().any(|line| line == "GGML_STATIC:BOOL=ON"),
                "CUDA image builds must statically link the CUDA runtime and BLAS libraries"
            );
            let cuda = build.join("ggml/src/ggml-cuda");
            println!("cargo:rustc-link-search=native={}", cuda.display());
            println!(
                "cargo:rerun-if-changed={}",
                cuda.join("libggml-cuda.a").display()
            );
            println!("cargo:rustc-link-lib=static=ggml-cuda");
            for (key, name) in [
                ("CUDA_cublas_static_LIBRARY", "cublas_static"),
                ("CUDA_cublasLt_static_LIBRARY", "cublasLt_static"),
                ("CUDA_cudart_static_LIBRARY", "cudart_static"),
                ("CUDA_culibos_LIBRARY", "culibos"),
            ] {
                let prefix = format!("{key}:FILEPATH=");
                let path = PathBuf::from(
                    cache
                        .lines()
                        .find_map(|line| line.strip_prefix(&prefix))
                        .expect("CUDA static library missing from native build cache"),
                );
                assert!(
                    path.is_file(),
                    "CUDA static library missing: {}",
                    path.display()
                );
                println!("cargo:rerun-if-changed={}", path.display());
                println!(
                    "cargo:rustc-link-search=native={}",
                    path.parent().unwrap().display()
                );
                println!("cargo:rustc-link-lib=static={name}");
            }
            if !cache.lines().any(|line| line == "GGML_CUDA_NO_VMM:BOOL=ON") {
                println!("cargo:rustc-link-lib=cuda");
            }
            for name in ["dl", "rt", "pthread"] {
                println!("cargo:rustc-link-lib={name}");
            }
        }
    }
}

#[cfg(any(feature = "native-sd", test))]
fn dev_sycl_flag(value: Option<&str>) -> Result<bool, &'static str> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("invalid_development_build_flag"),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sycl_build_requires_exact_explicit_opt_in() {
        assert_eq!(super::dev_sycl_flag(None), Ok(false));
        assert_eq!(super::dev_sycl_flag(Some("0")), Ok(false));
        assert_eq!(super::dev_sycl_flag(Some("1")), Ok(true));
        for value in ["", "true", "01", "1 ", "2"] {
            assert!(super::dev_sycl_flag(Some(value)).is_err());
        }
    }
}
