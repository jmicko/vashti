//! Compiled kernels. The SPIR-V is checked in so building needs no shader compiler; the test
//! below recompiles every source with `glslangValidator` when it is installed and fails if a
//! checked-in binary is stale (`KUNO_UPDATE_SHADERS=1` rewrites them).
pub const GEMM: &[u8] = include_bytes!("shaders/gemm.spv");

#[cfg(test)]
mod tests {
    use std::{path::Path, process::Command};

    #[test]
    fn checked_in_spirv_matches_sources() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/engine/gpu/shaders");
        let out = std::env::temp_dir().join(format!("kuno-shader-{}.spv", std::process::id()));
        if Command::new("glslangValidator")
            .arg("--version")
            .output()
            .is_err()
        {
            eprintln!("glslangValidator not installed; skipping");
            return;
        }
        let mut sources = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let source = entry.unwrap().path();
            if source.extension().is_none_or(|e| e != "comp") {
                continue;
            }
            sources += 1;
            let status = Command::new("glslangValidator")
                .args(["-V", "--target-env", "vulkan1.3", "--quiet", "-o"])
                .arg(&out)
                .arg(&source)
                .status()
                .unwrap();
            assert!(status.success(), "{} does not compile", source.display());
            let fresh = std::fs::read(&out).unwrap();
            let binary = source.with_extension("spv");
            if std::env::var_os("KUNO_UPDATE_SHADERS").is_some() {
                std::fs::write(&binary, &fresh).unwrap();
            }
            assert!(
                std::fs::read(&binary).ok().as_deref() == Some(&fresh[..]),
                "{} is stale; rerun with KUNO_UPDATE_SHADERS=1",
                binary.display()
            );
        }
        let _ = std::fs::remove_file(&out);
        assert!(sources > 0);
    }
}
