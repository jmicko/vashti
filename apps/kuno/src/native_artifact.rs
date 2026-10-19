//! Root-enrolled model identity cache. Only model metadata is persisted here.
//! Never used for prompts, responses, token/KV state or content fingerprints.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, os::unix::fs::MetadataExt, path::PathBuf};
type Result<T> = std::result::Result<T, &'static str>;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    dev: u64,
    ino: u64,
    bytes: u64,
    mtime: i64,
    mtime_ns: i64,
    ctime: i64,
    ctime_ns: i64,
}
impl Identity {
    fn of(file: &File) -> Result<Self> {
        let m = file.metadata().map_err(|_| "native_artifact_unavailable")?;
        if !m.is_file() {
            return Err("native_artifact_invalid");
        };
        Ok(Self {
            dev: m.dev(),
            ino: m.ino(),
            bytes: m.len(),
            mtime: m.mtime(),
            mtime_ns: m.mtime_nsec(),
            ctime: m.ctime(),
            ctime_ns: m.ctime_nsec(),
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identity: Option<Identity>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub shards: Vec<Artifact>,
    pub mtp: Artifact,
    pub device_id: u32,
}
/// Returns whether a full hash was necessary. Verify unchanged metadata after
/// hashing as well: no raced read can enroll a fresh identity for mixed bytes.
pub fn verify(artifact: &Artifact, trust_identity: bool) -> Result<(Identity, bool)> {
    if !artifact.path.is_absolute()
        || artifact.sha256.len() != 64
        || !artifact.sha256.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("native_artifact_invalid");
    }
    let mut file = File::open(&artifact.path).map_err(|_| "native_artifact_unavailable")?;
    let before = Identity::of(&file)?;
    if trust_identity && artifact.identity.as_ref() == Some(&before) {
        return Ok((before, false));
    }
    let mut hash = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|_| "native_artifact_unavailable")?;
        if n == 0 {
            break;
        }
        hash.update(&buf[..n]);
    }
    if format!("{:x}", hash.finalize()) != artifact.sha256 || Identity::of(&file)? != before {
        return Err("native_artifact_changed");
    }
    // Reopening detects a path replaced during verification, including same-size files.
    let reopened = File::open(&artifact.path).map_err(|_| "native_artifact_unavailable")?;
    if Identity::of(&reopened)? != before {
        return Err("native_artifact_changed");
    }
    Ok((before, true))
}
/// Cached identities are trusted only from a root-owned non-writable settings
/// descriptor. Development/untrusted settings always use complete hash checks.
pub fn trusted_settings(file: &File) -> bool {
    file.metadata()
        .is_ok_and(|m| m.is_file() && m.uid() == 0 && m.mode() & 0o022 == 0)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    #[test]
    fn unchanged_identity_skips_hash_but_mutations_and_untrusted_settings_do_not() {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(b"original").unwrap();
        let mut a = Artifact {
            path: f.path().to_owned(),
            sha256: format!("{:x}", Sha256::digest(b"original")),
            identity: None,
        };
        let (id, hashed) = verify(&a, true).unwrap();
        assert!(hashed);
        a.identity = Some(id);
        assert!(!verify(&a, true).unwrap().1);
        assert!(verify(&a, false).unwrap().1);
        std::fs::write(f.path(), b"modified").unwrap();
        assert!(verify(&a, true).is_err());
    }
    #[test]
    fn replacing_path_is_not_a_cache_hit() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("model");
        std::fs::write(&p, b"same").unwrap();
        let mut a = Artifact {
            path: p.clone(),
            sha256: format!("{:x}", Sha256::digest(b"same")),
            identity: None,
        };
        a.identity = Some(verify(&a, true).unwrap().0);
        std::fs::rename(&p, d.path().join("old")).unwrap();
        std::fs::write(&p, b"same").unwrap();
        assert!(verify(&a, true).unwrap().1);
    }
}
