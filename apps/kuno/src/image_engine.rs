//! Optional, root-owned image runtimes. This module never loads a GPU library.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom},
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

pub const SYCL_RUNTIME: &str = "qwen21-sycl-2026.1-v1";
pub const RUNTIME_ROOT: &str = "/usr/local/lib/kuno/runtimes";
/// ABI 3 is the resident ABI 2 plus MiniMax video jobs and progress frames.
/// ABI 3 added video jobs and progress frames; ABI 4 adds text-only, last-frame and
/// reference-image video. The gateway sends video only to ABI 4 workers.
pub const VIDEO_WORKER_ABI: u32 = 4;
const ENGINE_REVISION: &str = "88411ef1e0688ff2df1010aeeb5d92b2d8cea2be";
const GGML_REVISION: &str = "4bf5f6000653b7881d00963cd6ddb665ccd62a8d";
const MAX_MANIFEST: u64 = 64 * 1024;
const MAX_MEMBER: u64 = 1024 * 1024 * 1024;
const MAX_TOTAL: u64 = 3 * MAX_MEMBER;
type Result<T> = std::result::Result<T, &'static str>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageEngine {
    #[default]
    Native,
    ManagedSycl,
}
impl ImageEngine {
    pub fn is_native(&self) -> bool {
        *self == Self::Native
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMember {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeManifest {
    pub version: u32,
    pub id: String,
    pub worker_abi: u32,
    pub ggml_max_name: u32,
    pub engine_revision: String,
    pub ggml_revision: String,
    pub engine_patch_sha256: String,
    pub ggml_patch_sha256: String,
    pub sycl_patch_sha256: String,
    pub files: Vec<RuntimeMember>,
}

fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn member_name_valid(name: &str) -> bool {
    name == "worker"
        || (((name.starts_with("lib") && name.contains(".so"))
            || ((name.starts_with("LICENSE-") || name.starts_with("NOTICE-"))
                && name.ends_with(".txt")))
            && name.len() <= 128
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
            && !name.contains(".."))
}
impl RuntimeManifest {
    fn validate(&self) -> Result<()> {
        if self.version != 1
            || self.id != SYCL_RUNTIME
            || !matches!(self.worker_abi, 1..=VIDEO_WORKER_ABI)
            || self.ggml_max_name != 160
            || self.engine_revision != ENGINE_REVISION
            || self.ggml_revision != GGML_REVISION
            || !hash_valid(&self.engine_patch_sha256)
            || !hash_valid(&self.ggml_patch_sha256)
            || !hash_valid(&self.sycl_patch_sha256)
            || !(2..=96).contains(&self.files.len())
        {
            return Err("image_managed_runtime_incompatible");
        }
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        for file in &self.files {
            if !member_name_valid(&file.name)
                || !names.insert(file.name.as_str())
                || !(1..=MAX_MEMBER).contains(&file.bytes)
                || !hash_valid(&file.sha256)
            {
                return Err("image_managed_runtime_invalid");
            }
            total = total
                .checked_add(file.bytes)
                .ok_or("image_managed_runtime_invalid")?;
        }
        if !names.contains("worker") || total > MAX_TOTAL {
            return Err("image_managed_runtime_invalid");
        }
        Ok(())
    }
}

fn open_regular(path: &Path, owner: Option<u32>, executable: bool) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| "image_managed_runtime_unavailable")?;
    let metadata = file
        .metadata()
        .map_err(|_| "image_managed_runtime_invalid")?;
    if !metadata.is_file()
        || owner.is_some_and(|uid| metadata.uid() != uid || metadata.mode() & 0o022 != 0)
        || (executable && metadata.mode() & 0o111 == 0)
        || metadata.mode() & 0o6000 != 0
    {
        return Err("image_managed_runtime_untrusted");
    }
    Ok(file)
}
fn trusted_directory(path: &Path, owner: u32) -> Result<()> {
    let metadata = fs::symlink_metadata(path).map_err(|_| "image_managed_runtime_unavailable")?;
    if !metadata.is_dir() || metadata.uid() != owner || metadata.mode() & 0o022 != 0 {
        return Err("image_managed_runtime_untrusted");
    }
    Ok(())
}
fn manifest_at(directory: &Path, owner: Option<u32>) -> Result<(RuntimeManifest, Vec<u8>)> {
    let mut bytes = Vec::new();
    open_regular(&directory.join("manifest.json"), owner, false)?
        .take(MAX_MANIFEST + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "image_managed_runtime_invalid")?;
    if bytes.len() as u64 > MAX_MANIFEST {
        return Err("image_managed_runtime_invalid");
    }
    let manifest: RuntimeManifest =
        serde_json::from_slice(&bytes).map_err(|_| "image_managed_runtime_invalid")?;
    manifest.validate()?;
    Ok((manifest, bytes))
}
fn verified_member(directory: &Path, member: &RuntimeMember, owner: Option<u32>) -> Result<File> {
    let mut file = open_regular(
        &directory.join(&member.name),
        owner,
        member.name == "worker",
    )?;
    if file
        .metadata()
        .map_err(|_| "image_managed_runtime_invalid")?
        .len()
        != member.bytes
    {
        return Err("image_managed_runtime_corrupt");
    }
    if member.name == "worker" {
        let mut header = [0u8; 64];
        file.read_exact(&mut header)
            .map_err(|_| "image_managed_runtime_incompatible")?;
        if &header[..7] != b"\x7fELF\x02\x01\x01"
            || !matches!(u16::from_le_bytes([header[16], header[17]]), 2 | 3)
            || u16::from_le_bytes([header[18], header[19]]) != 62
            || u32::from_le_bytes(header[20..24].try_into().unwrap()) != 1
        {
            return Err("image_managed_runtime_incompatible");
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|_| "image_managed_runtime_corrupt")?;
    }
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut count = 0u64;
    loop {
        let n = file
            .read(&mut buffer)
            .map_err(|_| "image_managed_runtime_corrupt")?;
        if n == 0 {
            break;
        }
        count += n as u64;
        if count > member.bytes {
            return Err("image_managed_runtime_corrupt");
        }
        digest.update(&buffer[..n]);
    }
    if count != member.bytes || format!("{:x}", digest.finalize()) != member.sha256 {
        return Err("image_managed_runtime_corrupt");
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|_| "image_managed_runtime_corrupt")?;
    Ok(file)
}

#[derive(Clone)]
pub struct VerifiedRuntime {
    pub directory: PathBuf,
    pub manifest: RuntimeManifest,
}
type Stamp = (u64, u64, u64, i64, i64, i64, i64, u32, u32);
type CachedRuntime = Option<(VerifiedRuntime, Vec<Stamp>)>;
static VERIFIED: OnceLock<Mutex<CachedRuntime>> = OnceLock::new();
fn stamp(metadata: fs::Metadata) -> Stamp {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.len(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
        metadata.mode(),
        metadata.uid(),
    )
}
impl VerifiedRuntime {
    pub fn load(id: &str) -> Result<Self> {
        if id != SYCL_RUNTIME {
            return Err("image_managed_runtime_incompatible");
        }
        let directory = Path::new(RUNTIME_ROOT).join(id);
        for parent in directory.ancestors() {
            trusted_directory(parent, 0)?;
        }
        let mut cache = VERIFIED
            .get_or_init(|| Mutex::new(None))
            .lock()
            .map_err(|_| "image_managed_runtime_unavailable")?;
        if let Some((runtime, previous)) = cache.as_ref()
            && runtime.stamps(0).is_ok_and(|actual| actual == *previous)
        {
            return Ok(runtime.clone());
        }
        *cache = None;
        let runtime = Self::load_at(&directory, 0)?;
        *cache = Some((runtime.clone(), runtime.stamps(0)?));
        Ok(runtime)
    }
    fn load_at(directory: &Path, owner: u32) -> Result<Self> {
        trusted_directory(directory, owner)?;
        let (manifest, _) = manifest_at(directory, Some(owner))?;
        let names: BTreeSet<_> = manifest
            .files
            .iter()
            .map(|m| m.name.as_str())
            .chain(["manifest.json"])
            .collect();
        for entry in fs::read_dir(directory).map_err(|_| "image_managed_runtime_unavailable")? {
            let entry = entry.map_err(|_| "image_managed_runtime_invalid")?;
            if !entry
                .file_name()
                .to_str()
                .is_some_and(|name| names.contains(name))
            {
                return Err("image_managed_runtime_invalid");
            }
        }
        for member in &manifest.files {
            verified_member(directory, member, Some(owner))?;
        }
        Ok(Self {
            directory: directory.into(),
            manifest,
        })
    }
    pub fn worker(&self) -> PathBuf {
        self.directory.join("worker")
    }
    fn stamps(&self, owner: u32) -> Result<Vec<Stamp>> {
        let mut stamps = vec![stamp(
            fs::symlink_metadata(&self.directory)
                .map_err(|_| "image_managed_runtime_unavailable")?,
        )];
        for name in std::iter::once("manifest.json")
            .chain(self.manifest.files.iter().map(|m| m.name.as_str()))
        {
            stamps.push(stamp(
                open_regular(&self.directory.join(name), Some(owner), name == "worker")?
                    .metadata()
                    .map_err(|_| "image_managed_runtime_invalid")?,
            ));
        }
        Ok(stamps)
    }
}

/// Root-side bundle preparation; hashes apply to the exact opened bytes. Source
/// files may be user-owned, but never symlinks. Installed files are root-owned.
pub fn bundle_artifacts(
    directory: &Path,
    expected_manifest: &str,
) -> Result<Vec<crate::deployment::Artifact>> {
    if !directory.is_absolute() || !hash_valid(expected_manifest) {
        return Err("image_runtime_bundle_invalid");
    }
    let (manifest, bytes) = manifest_at(directory, None)?;
    if format!("{:x}", Sha256::digest(&bytes)) != expected_manifest {
        return Err("image_runtime_manifest_checksum_mismatch");
    }
    let base = Path::new(RUNTIME_ROOT)
        .strip_prefix("/")
        .unwrap()
        .join(&manifest.id);
    let mut artifacts = vec![crate::deployment::Artifact {
        path: base.join("manifest.json"),
        bytes,
        mode: 0o644,
    }];
    for member in &manifest.files {
        let file = verified_member(directory, member, None)?;
        let mut bytes = Vec::new();
        file.take(member.bytes + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| "image_managed_runtime_corrupt")?;
        if bytes.len() as u64 != member.bytes
            || format!("{:x}", Sha256::digest(&bytes)) != member.sha256
        {
            return Err("image_managed_runtime_corrupt");
        }
        artifacts.push(crate::deployment::Artifact {
            path: base.join(&member.name),
            bytes,
            mode: if member.name == "worker" {
                0o755
            } else {
                0o644
            },
        });
    }
    Ok(artifacts)
}

/// Keep an installed optional runtime in every replacement transaction even
/// when the selected image engine is native. Never adopt an untrusted directory.
pub fn installed_artifacts() -> Result<Vec<crate::deployment::Artifact>> {
    let directory = Path::new(RUNTIME_ROOT).join(SYCL_RUNTIME);
    match fs::symlink_metadata(&directory) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err("image_managed_runtime_unavailable"),
        Ok(_) => {}
    }
    VerifiedRuntime::load(SYCL_RUNTIME)?;
    let (_, bytes) = manifest_at(&directory, Some(0))?;
    bundle_artifacts(&directory, &format!("{:x}", Sha256::digest(&bytes)))
}

/// Fixed, cleared vendor environment. No inherited cache/debug settings, SDK
/// paths or writable exceptions. Device identity is still verified by the worker.
pub fn sycl_command(
    runtime: &VerifiedRuntime,
    models: &str,
    pci: &str,
    previews: bool,
    resident: bool,
) -> std::process::Command {
    let mut command = std::process::Command::new(runtime.worker());
    command
        .args([
            if resident {
                "image-worker-resident"
            } else {
                "image-worker"
            },
            models,
            pci,
        ])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("HOME", "/nonexistent")
        .env("LD_LIBRARY_PATH", &runtime.directory)
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .env("__GL_SHADER_DISK_CACHE", "0")
        .env("CUDA_CACHE_DISABLE", "1")
        .env("SYCL_CACHE_PERSISTENT", "0")
        .env("NEO_CACHE_PERSISTENT", "0")
        .env("SYCL_UR_TRACE", "0")
        .env("GGML_SYCL_ENABLE_GRAPH", "0")
        .env("GGML_SYCL_USM_SYSTEM", "0")
        .env("ONEDNN_VERBOSE", "0")
        .env("DNNL_VERBOSE", "0")
        .env("MKL_VERBOSE", "0")
        .env("XPTI_TRACE_ENABLE", "0")
        .env("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu")
        .env("KUNO_EXPERIMENTAL_QWEN_SYCL", "1")
        .env("GGML_SYCL_FA_ONEDNN_NO_MASK", "1");
    if previews {
        command.env("KUNO_IMAGE_PREVIEWS", "1");
    }
    command
}

/// Called by the standard executable's single-threaded private entrypoint.
/// Restrictions survive exec, covering vendor constructors and their threads.
pub fn launch(id: &str, models: &str, pci: &str, previews: bool, resident: bool) -> Result<()> {
    use std::os::unix::process::CommandExt;
    crate::image_privacy::protect_process()?;
    crate::image_privacy::require_no_swap()?;
    crate::image_privacy::restrict_filesystem()?;
    let runtime = VerifiedRuntime::load(id)?;
    if resident && runtime.manifest.worker_abi < 2 {
        return Err("image_managed_runtime_incompatible");
    }
    let _error = sycl_command(&runtime, models, pci, previews, resident).exec();
    Err("image_managed_runtime_exec_failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    fn worker_bytes() -> Vec<u8> {
        let mut bytes = vec![0; 64];
        bytes[..7].copy_from_slice(b"\x7fELF\x02\x01\x01");
        bytes[16] = 3;
        bytes[18] = 62;
        bytes[20] = 1;
        bytes
    }
    fn manifest() -> RuntimeManifest {
        RuntimeManifest {
            version: 1,
            id: SYCL_RUNTIME.into(),
            worker_abi: 1,
            ggml_max_name: 160,
            engine_revision: ENGINE_REVISION.into(),
            ggml_revision: GGML_REVISION.into(),
            engine_patch_sha256: "1".repeat(64),
            ggml_patch_sha256: "2".repeat(64),
            sycl_patch_sha256: "3".repeat(64),
            files: vec![
                RuntimeMember {
                    name: "worker".into(),
                    bytes: 64,
                    sha256: format!("{:x}", Sha256::digest(worker_bytes())),
                },
                RuntimeMember {
                    name: "libsycl.so.9".into(),
                    bytes: 3,
                    sha256: format!("{:x}", Sha256::digest(b"def")),
                },
            ],
        }
    }
    fn bundle() -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        fs::set_permissions(d.path(), fs::Permissions::from_mode(0o700)).unwrap();
        fs::write(d.path().join("worker"), worker_bytes()).unwrap();
        fs::set_permissions(d.path().join("worker"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(d.path().join("libsycl.so.9"), b"def").unwrap();
        fs::write(
            d.path().join("manifest.json"),
            serde_json::to_vec(&manifest()).unwrap(),
        )
        .unwrap();
        for name in ["manifest.json", "libsycl.so.9"] {
            fs::set_permissions(d.path().join(name), fs::Permissions::from_mode(0o644)).unwrap();
        }
        d
    }
    #[test]
    fn manifest_rejects_paths_duplicates_abi_and_bounds() {
        manifest().validate().unwrap();
        for name in [
            "../worker",
            "/libsycl.so",
            "libx/evil.so",
            "lib..so",
            "libx.so\n",
            "manifest.json",
        ] {
            let mut m = manifest();
            m.files[1].name = name.into();
            assert!(m.validate().is_err());
        }
        let mut m = manifest();
        m.files.push(m.files[0].clone());
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.worker_abi = 2;
        assert!(m.validate().is_ok());
        m.worker_abi = 3;
        assert!(m.validate().is_ok());
        m.worker_abi = 4;
        assert!(m.validate().is_ok());
        m.worker_abi = 5;
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.ggml_max_name = 64;
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.id = "other".into();
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.files[0].bytes = MAX_MEMBER + 1;
        assert!(m.validate().is_err());
        let mut m = manifest();
        m.files[0].sha256 = "A".repeat(64);
        assert!(m.validate().is_err());
    }
    #[test]
    fn verification_rejects_tampering_symlinks_writable_and_nonexecutables() {
        let d = bundle();
        let uid = unsafe { libc::geteuid() };
        VerifiedRuntime::load_at(d.path(), uid).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid.wrapping_add(1)).is_err());
        fs::write(d.path().join("libsycl.so.9"), b"xyz").unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::write(d.path().join("libsycl.so.9"), b"def").unwrap();
        fs::set_permissions(d.path().join("worker"), fs::Permissions::from_mode(0o777)).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::set_permissions(d.path().join("worker"), fs::Permissions::from_mode(0o644)).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::set_permissions(d.path().join("worker"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::remove_file(d.path().join("libsycl.so.9")).unwrap();
        symlink(d.path().join("worker"), d.path().join("libsycl.so.9")).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
    }
    #[test]
    fn directory_manifest_and_member_inventory_are_protected() {
        let d = bundle();
        let uid = unsafe { libc::geteuid() };
        let runtime = VerifiedRuntime::load_at(d.path(), uid).unwrap();
        let previous = runtime.stamps(uid).unwrap();
        fs::write(d.path().join("libsycl.so.9"), b"different length").unwrap();
        assert_ne!(runtime.stamps(uid).unwrap(), previous);
        fs::write(d.path().join("libsycl.so.9"), b"def").unwrap();
        fs::write(d.path().join("libextra.so"), b"xyz").unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::remove_file(d.path().join("libextra.so")).unwrap();
        fs::set_permissions(d.path(), fs::Permissions::from_mode(0o777)).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::set_permissions(d.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let manifest_path = d.path().join("manifest.json");
        fs::remove_file(&manifest_path).unwrap();
        symlink(d.path().join("worker"), &manifest_path).unwrap();
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
    }
    #[test]
    fn runtime_artifacts_restore_together_when_activation_fails() {
        use crate::deployment::{Host, Transaction};
        struct Simulated {
            fail_once: bool,
        }
        impl Host for Simulated {
            fn preflight(&mut self) -> Result<()> {
                Ok(())
            }
            fn original_active(&mut self) -> Result<bool> {
                Ok(true)
            }
            fn stop_original(&mut self) -> Result<()> {
                Ok(())
            }
            fn reload(&mut self) -> Result<()> {
                Ok(())
            }
            fn start_private_upstream(&mut self) -> Result<()> {
                Ok(())
            }
            fn verify_private_upstream(&mut self) -> Result<()> {
                Ok(())
            }
            fn start_gateway(&mut self) -> Result<()> {
                Ok(())
            }
            fn verify_gateway(&mut self) -> Result<()> {
                if self.fail_once {
                    self.fail_once = false;
                    Err("synthetic_activation_failure")
                } else {
                    Ok(())
                }
            }
            fn stop_gateway(&mut self) -> Result<()> {
                Ok(())
            }
            fn restore_original(&mut self, _: bool) -> Result<()> {
                Ok(())
            }
            fn verify_original(&mut self, _: bool) -> Result<()> {
                Ok(())
            }
        }
        let bundle = bundle();
        // A real runtime includes versioned and SONAME library aliases. Their
        // inventory exceeds the gateway's former 32-file journal bound.
        let mut manifest = manifest();
        for index in 0..20 {
            for suffix in ["so.1", "so.1.0"] {
                let name = format!("libclosure{index}.{suffix}");
                fs::write(bundle.path().join(&name), b"public library fixture").unwrap();
                fs::set_permissions(bundle.path().join(&name), fs::Permissions::from_mode(0o644))
                    .unwrap();
                manifest.files.push(RuntimeMember {
                    name,
                    bytes: 22,
                    sha256: format!("{:x}", Sha256::digest(b"public library fixture")),
                });
            }
        }
        fs::write(
            bundle.path().join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let destination = tempfile::tempdir().unwrap();
        let digest = format!(
            "{:x}",
            Sha256::digest(fs::read(bundle.path().join("manifest.json")).unwrap())
        );
        let artifacts = bundle_artifacts(bundle.path(), &digest).unwrap();
        assert!(artifacts.len() > 32);
        let transaction =
            Transaction::open(destination.path(), &destination.path().join("setup")).unwrap();
        let mut host = Simulated { fail_once: false };
        transaction.apply(&artifacts, &mut host).unwrap();
        let mut changed = artifacts.clone();
        changed[1].bytes = b"new worker".to_vec();
        host.fail_once = true;
        assert_eq!(
            transaction.replace(&changed, &mut host),
            Err("upgrade_failed_previous_version_restored")
        );
        transaction.verify_installed().unwrap();
        for artifact in &artifacts {
            assert_eq!(
                fs::read(destination.path().join(&artifact.path)).unwrap(),
                artifact.bytes
            );
        }
        let installed_directory = destination
            .path()
            .join(Path::new(RUNTIME_ROOT).strip_prefix("/").unwrap())
            .join(SYCL_RUNTIME);
        VerifiedRuntime::load_at(&installed_directory, unsafe { libc::geteuid() }).unwrap();
        assert_ne!(
            fs::metadata(installed_directory.join("libclosure0.so.1"))
                .unwrap()
                .ino(),
            fs::metadata(installed_directory.join("libclosure0.so.1.0"))
                .unwrap()
                .ino(),
            "installed aliases must be separate regular files"
        );
        transaction.restore(&mut host).unwrap();
        for artifact in artifacts {
            assert!(!destination.path().join(artifact.path).exists());
        }
    }
    #[test]
    fn special_members_and_wrong_worker_architecture_fail_without_blocking() {
        let d = bundle();
        let uid = unsafe { libc::geteuid() };
        fs::remove_file(d.path().join("libsycl.so.9")).unwrap();
        let path =
            std::ffi::CString::new(d.path().join("libsycl.so.9").as_os_str().as_encoded_bytes())
                .unwrap();
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(VerifiedRuntime::load_at(d.path(), uid).is_err());
        fs::remove_file(d.path().join("libsycl.so.9")).unwrap();
        fs::write(d.path().join("libsycl.so.9"), b"def").unwrap();
        fs::set_permissions(
            d.path().join("libsycl.so.9"),
            fs::Permissions::from_mode(0o644),
        )
        .unwrap();
        let mut bytes = worker_bytes();
        bytes[18] = 183;
        fs::write(d.path().join("worker"), bytes).unwrap();
        assert_eq!(
            VerifiedRuntime::load_at(d.path(), uid).err().unwrap(),
            "image_managed_runtime_incompatible"
        );
    }
    #[test]
    fn prepared_artifacts_have_fixed_paths_modes_and_exact_checksums() {
        let d = bundle();
        let bytes = fs::read(d.path().join("manifest.json")).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let a = bundle_artifacts(d.path(), &digest).unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[1].mode, 0o755);
        assert_eq!(a[2].mode, 0o644);
        assert!(
            a.iter()
                .all(|f| f.path.starts_with("usr/local/lib/kuno/runtimes"))
        );
        assert!(bundle_artifacts(d.path(), &"0".repeat(64)).is_err());
    }
    #[test]
    fn sycl_environment_is_fixed_and_has_no_dump_or_trace_paths() {
        let runtime = VerifiedRuntime {
            directory: "/synthetic/runtime".into(),
            manifest: manifest(),
        };
        let c = sycl_command(&runtime, "/models", "0000:2d:00.0", true, false);
        let env: std::collections::BTreeMap<_, _> = c.get_envs().collect();
        let expected: std::collections::BTreeMap<_, _> = [
            ("PATH", "/usr/bin:/bin"),
            ("HOME", "/nonexistent"),
            ("LD_LIBRARY_PATH", "/synthetic/runtime"),
            ("MESA_SHADER_CACHE_DISABLE", "true"),
            ("__GL_SHADER_DISK_CACHE", "0"),
            ("CUDA_CACHE_DISABLE", "1"),
            ("SYCL_CACHE_PERSISTENT", "0"),
            ("NEO_CACHE_PERSISTENT", "0"),
            ("SYCL_UR_TRACE", "0"),
            ("GGML_SYCL_ENABLE_GRAPH", "0"),
            ("GGML_SYCL_USM_SYSTEM", "0"),
            ("ONEDNN_VERBOSE", "0"),
            ("DNNL_VERBOSE", "0"),
            ("MKL_VERBOSE", "0"),
            ("XPTI_TRACE_ENABLE", "0"),
            ("ONEAPI_DEVICE_SELECTOR", "level_zero:gpu"),
            ("KUNO_EXPERIMENTAL_QWEN_SYCL", "1"),
            ("GGML_SYCL_FA_ONEDNN_NO_MASK", "1"),
            ("KUNO_IMAGE_PREVIEWS", "1"),
        ]
        .into_iter()
        .map(|(key, value)| (std::ffi::OsStr::new(key), Some(std::ffi::OsStr::new(value))))
        .collect();
        assert_eq!(env, expected);
        for forbidden in [
            "SYCL_DUMP_IMAGES",
            "SYCL_CACHE_TRACE",
            "LD_PRELOAD",
            "KUNO_CONTROL_TOKEN",
        ] {
            assert!(!env.contains_key(std::ffi::OsStr::new(forbidden)));
        }
        assert_eq!(c.get_program(), "/synthetic/runtime/worker");
    }
}
