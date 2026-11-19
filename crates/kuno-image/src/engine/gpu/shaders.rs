//! Compiled kernels. The SPIR-V is checked in so building needs no shader compiler; the test
//! below recompiles every source with `glslangValidator` when it is installed and fails if a
//! checked-in binary is stale (`KUNO_UPDATE_SHADERS=1` rewrites them).
pub const GEMM: &[u8] = include_bytes!("shaders/gemm.spv");
pub const GEMM16: &[u8] = include_bytes!("shaders/gemm16.spv");
pub const GEMM8: &[u8] = include_bytes!("shaders/gemm8.spv");
pub const MIX: &[u8] = include_bytes!("shaders/mix.spv");
pub const NORM: &[u8] = include_bytes!("shaders/norm.spv");
pub const HEADS: &[u8] = include_bytes!("shaders/heads.spv");
pub const MERGE: &[u8] = include_bytes!("shaders/merge.spv");
pub const NORMQ8: &[u8] = include_bytes!("shaders/normq8.spv");
pub const QK8: &[u8] = include_bytes!("shaders/qk8.spv");
pub const VALUE8: &[u8] = include_bytes!("shaders/value8.spv");
pub const SCOREMAX: &[u8] = include_bytes!("shaders/scoremax.spv");
pub const ROWMAX: &[u8] = include_bytes!("shaders/rowmax.spv");
pub const MIXTOP: &[u8] = include_bytes!("shaders/mixtop.spv");
pub const MERGE8: &[u8] = include_bytes!("shaders/merge8.spv");
pub const QK8S: &[u8] = include_bytes!("shaders/qk8s.spv");
pub const VALUE8S: &[u8] = include_bytes!("shaders/value8s.spv");
pub const MERGEQ8: &[u8] = include_bytes!("shaders/mergeq8.spv");
pub const RESID8: &[u8] = include_bytes!("shaders/resid8.spv");
pub const EMBED8: &[u8] = include_bytes!("shaders/embed8.spv");
pub const PARTS8: &[u8] = include_bytes!("shaders/parts8.spv");
pub const NORM16: &[u8] = include_bytes!("shaders/norm16.spv");
pub const MERGE16S: &[u8] = include_bytes!("shaders/merge16s.spv");
pub const GATE16: &[u8] = include_bytes!("shaders/gate16.spv");
pub const ADD: &[u8] = include_bytes!("shaders/add.spv");
pub const FINAL8: &[u8] = include_bytes!("shaders/final8.spv");
pub const GATEQ8: &[u8] = include_bytes!("shaders/gateq8.spv");
pub const ROTATE8: &[u8] = include_bytes!("shaders/rotate8.spv");
pub const TNORM: &[u8] = include_bytes!("shaders/tnorm.spv");
pub const TQKV: &[u8] = include_bytes!("shaders/tqkv.spv");
pub const TATTEND: &[u8] = include_bytes!("shaders/tattend.spv");
pub const TRESID: &[u8] = include_bytes!("shaders/tresid.spv");
pub const TGATE: &[u8] = include_bytes!("shaders/tgate.spv");
pub const TADD: &[u8] = include_bytes!("shaders/tadd.spv");
pub const RESIDUAL: &[u8] = include_bytes!("shaders/residual.spv");
pub const GATE: &[u8] = include_bytes!("shaders/gate.spv");

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
