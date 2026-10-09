//! Registration for the compiled model-free attention selector diagnostic.
use std::{ffi::OsString, path::PathBuf};
pub(crate) fn registration(
    args: &[OsString],
    env: impl IntoIterator<Item = (OsString, OsString)>,
    allowed_build: bool,
) -> std::result::Result<PathBuf, &'static str> {
    if !allowed_build
        || args.len() != 3
        || !matches!(args[1].to_str(), Some("intel" | "nvidia"))
        || args[2] != "qsa-chronological-probe"
    {
        return Err("public QSA selection fixed card/build required");
    }
    let (mut gate, mut receipt) = (false, None);
    for (key, value) in env {
        if ![b"KUNO_LAB_".as_slice(), b"KUNO_TEST_", b"KUNO_PUBLIC_"]
            .iter()
            .any(|p| key.as_encoded_bytes().starts_with(p))
        {
            continue;
        }
        if key == "KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY" && !gate && value == "1" {
            gate = true;
        } else if key == "KUNO_PUBLIC_RECEIPT" && receipt.is_none() && !value.is_empty() {
            receipt = Some(PathBuf::from(value));
        } else {
            return Err("public QSA selection inherited/duplicate option rejected");
        }
    }
    if !gate {
        return Err("public QSA selection explicit public gate required");
    }
    receipt.ok_or("public QSA selection new numeric receipt required")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qsa_probe_rejects_duplicate_inherited_and_missing_public_gates() {
        for card in ["intel", "nvidia"] {
            let args = ["lab", card, "qsa-chronological-probe"].map(OsString::from);
            let env = vec![
                ("KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY".into(), "1".into()),
                ("KUNO_PUBLIC_RECEIPT".into(), "/public/numeric".into()),
            ];
            registration(&args, env.clone(), true).unwrap();
            assert!(registration(&args, env.clone(), false).is_err());
            for key in [
                "KUNO_LAB_PROMPT",
                "KUNO_TEST_MODEL",
                "KUNO_LAB_TARGET_QSA",
                "KUNO_LAB_QSA_CHRONOLOGICAL_PUBLIC_ONLY",
                "KUNO_PUBLIC_RECEIPT",
            ] {
                let mut bad = env.clone();
                bad.push((key.into(), "1".into()));
                assert!(registration(&args, bad, true).is_err());
            }
            for omit in 0..2 {
                let mut bad = env.clone();
                bad.remove(omit);
                assert!(registration(&args, bad, true).is_err());
            }
        }
    }
}
