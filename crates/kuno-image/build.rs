fn main() {
    println!("cargo:rustc-check-cfg=cfg(kuno_native_cuda)");
    println!("cargo:rerun-if-env-changed=KUNO_SD_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=KUNO_SD_BUILD_DIR");
    println!("cargo:rerun-if-changed=native/shim.cpp");
    println!("cargo:rerun-if-changed=native/cache-budget.h");
    println!("cargo:rerun-if-changed=native/backend-choice.h");
    #[cfg(feature = "native-sd")]
    {
        use std::{path::PathBuf, process::Command};
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
        let ggml_diff = Command::new("git")
            .args(["diff", "--no-color", "--no-ext-diff", "--binary", "HEAD"])
            .current_dir(&ggml)
            .output()
            .expect("read GGML patch");
        assert!(
            ggml_diff.status.success()
                && ggml_diff.stdout == std::fs::read(ggml_patch).expect("GGML compatibility patch"),
            "GGML source must contain exactly the pinned Kuno patch"
        );
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
            "SD_VULKAN:BOOL=ON",
            "SD_USE_UPSTREAM_GGML:BOOL=OFF",
            "SD_BUILD_SHARED_LIBS:BOOL=OFF",
        ] {
            assert!(
                cache.lines().any(|l| l == required),
                "native build must retain patched FP8 support and static Vulkan linkage"
            );
        }
        cc::Build::new()
            .cpp(true)
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
            "ggml/src/ggml-vulkan/libggml-vulkan.a",
        ] {
            println!("cargo:rerun-if-changed={}", build.join(archive).display());
        }
        for directory in [
            &build,
            &build.join("ggml/src"),
            &build.join("ggml/src/ggml-vulkan"),
        ] {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
        for name in [
            "stable-diffusion",
            "ggml",
            "ggml-cpu",
            "ggml-vulkan",
            "ggml-base",
        ] {
            println!("cargo:rustc-link-lib=static={name}");
        }
        for name in ["stdc++", "gomp", "vulkan"] {
            println!("cargo:rustc-link-lib={name}");
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
