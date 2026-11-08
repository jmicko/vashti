//! Recoverable installation transaction, independent of service-manager I/O.
//! The systemd adapter verifies ownership and readiness. Tests use
//! a simulated service manager to inject crashes and startup/activation failures.

use crate::{journal::Journal, storage};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
};

type Result<T> = std::result::Result<T, &'static str>;

/// Eight gateway artifacts plus a manifest and up to 96 optional-runtime
/// members fit within this bound. Writers and readers must share the limit so
/// a successful transaction can always be reopened and rolled back.
pub const MAX_DEPLOYMENT_FILES: usize = 128;

#[derive(Clone)]
pub struct Artifact {
    /// Relative to the installation root; parents must be real directories.
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub mode: u32,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedFile {
    path: PathBuf,
    digest: String,
    mode: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Prepared,
    Installing,
    Starting,
    Complete,
    Restoring,
    Restored,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    root: PathBuf,
    phase: Phase,
    original_active: bool,
    files: Vec<OwnedFile>,
}

/// Errors are classified codes, never raw service logs or request information.
/// Methods must be idempotent. `preflight` verifies listener and service ownership
/// and rejects an unrelated gateway service or listener before any mutation.
pub trait Host {
    fn preflight(&mut self) -> Result<()>;
    fn original_active(&mut self) -> Result<bool>;
    fn stop_original(&mut self) -> Result<()>;
    fn reload(&mut self) -> Result<()>;
    fn start_private_upstream(&mut self) -> Result<()>;
    fn verify_private_upstream(&mut self) -> Result<()>;
    fn start_gateway(&mut self) -> Result<()>;
    fn verify_gateway(&mut self) -> Result<()>;
    fn stop_gateway(&mut self) -> Result<()>;
    fn restore_original(&mut self, was_active: bool) -> Result<()>;
    fn verify_original(&mut self, was_active: bool) -> Result<()>;
    fn activate_gateway(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct Transaction {
    root: PathBuf,
    journal_path: PathBuf,
    _lock: Journal,
}

impl Transaction {
    pub fn open(root: &Path, state: &Path) -> Result<Self> {
        let root = root
            .canonicalize()
            .map_err(|_| "installation_root_unavailable")?;
        let lock = Journal::open(state).map_err(|_| "setup_state_unavailable_or_in_use")?;
        Ok(Self {
            root,
            journal_path: state.join("takeover.json"),
            _lock: lock,
        })
    }

    pub fn phase(&self) -> Result<Option<Phase>> {
        Ok(self.load()?.map(|r| r.phase))
    }

    pub fn verify_installed(&self) -> Result<()> {
        let record = self.load()?.ok_or("installation_not_found")?;
        if record.phase != Phase::Complete {
            return Err("installation_not_complete");
        }
        self.check_files(&record.files, false)
    }

    pub fn verify_recovery_files(&self) -> Result<()> {
        let record = self.load()?.ok_or("installation_not_found")?;
        if !matches!(record.phase, Phase::Complete | Phase::Restoring) {
            return Err("installation_not_recoverable");
        }
        self.check_files(&record.files, false)
    }

    pub fn apply(&self, artifacts: &[Artifact], host: &mut impl Host) -> Result<()> {
        if artifacts.is_empty() {
            return Err("empty_deployment");
        }
        let files = self.describe(artifacts)?;
        if let Some(record) = self.load()? {
            if record.phase == Phase::Complete {
                if !same_files(&record.files, &files) {
                    return Err("deployment_already_installed_with_different_files");
                }
                self.check_files(&record.files, false)?;
                host.verify_private_upstream()?;
                host.verify_gateway()?;
                return host.activate_gateway();
            }
            if record.phase != Phase::Restored {
                return Err("interrupted_setup_restore_required");
            }
        }
        host.preflight()?;
        for file in &files {
            let path = self.checked_path(&file.path)?;
            if fs::symlink_metadata(&path).is_ok() || fs::symlink_metadata(staged(&path)).is_ok() {
                return Err("installation_path_already_exists");
            }
        }
        let mut record = Record {
            version: 1,
            root: self.root.clone(),
            phase: Phase::Prepared,
            original_active: host.original_active()?,
            files,
        };
        self.save(&record)?;
        let result = (|| {
            record.phase = Phase::Installing;
            self.save(&record)?;
            host.stop_original()?;
            for artifact in artifacts {
                self.install(artifact)?;
            }
            host.reload()?;
            record.phase = Phase::Starting;
            self.save(&record)?;
            host.start_private_upstream()?;
            host.verify_private_upstream()?;
            host.start_gateway()?;
            host.verify_gateway()?;
            record.phase = Phase::Complete;
            self.save(&record)?;
            host.activate_gateway()
        })();
        if let Err(code) = result {
            if self.restore(host).is_err() {
                return Err("takeover_failed_restore_required");
            }
            return Err(code);
        }
        Ok(())
    }

    /// Replace a complete installation under the existing exclusive setup lock.
    /// Ordinary activation failure restores the exact previous artifacts. A
    /// process crash retains the normal recoverable restore journal.
    pub fn replace(&self, artifacts: &[Artifact], host: &mut impl Host) -> Result<()> {
        let record = self.load()?.ok_or("installation_not_found")?;
        if record.phase != Phase::Complete {
            return Err("interrupted_setup_restore_required");
        }
        self.check_files(&record.files, false)?;
        if artifacts.is_empty() {
            return Err("empty_deployment");
        }
        let files = self.describe(artifacts)?;
        if same_files(&record.files, &files) {
            return self.apply(artifacts, host);
        }
        // New artifacts must be checked before stopping a working installation.
        // In particular, adding the CLI launcher cannot displace an unrelated
        // existing executable and then rely on service rollback to recover.
        for file in &files {
            if !record.files.iter().any(|old| old.path == file.path) {
                let path = self.checked_path(&file.path)?;
                if fs::symlink_metadata(&path).is_ok()
                    || fs::symlink_metadata(staged(&path)).is_ok()
                {
                    return Err("installation_path_already_exists");
                }
            }
        }
        let previous = record
            .files
            .iter()
            .map(|file| {
                Ok(Artifact {
                    path: file.path.clone(),
                    mode: file.mode,
                    bytes: fs::read(self.checked_path(&file.path)?)
                        .map_err(|_| "previous_installation_read_failed")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // Recheck the snapshot before removing any installed files.
        if !same_files(&record.files, &self.describe(&previous)?) {
            return Err("owned_file_changed_restore_refused");
        }
        self.restore(host)?;
        if self.apply(artifacts, host).is_err() {
            return if self.apply(&previous, host).is_ok() {
                Err("upgrade_failed_previous_version_restored")
            } else {
                Err("upgrade_failed_restore_required")
            };
        }
        Ok(())
    }

    /// Used both for failed installation recovery and explicit uninstall.
    /// Compare every existing owned file before stopping anything. Never remove
    /// an administrator's subsequent edit or follow a replaced parent symlink.
    pub fn restore(&self, host: &mut impl Host) -> Result<()> {
        let Some(mut record) = self.load()? else {
            return Ok(());
        };
        if record.phase == Phase::Restored {
            return Ok(());
        }
        self.check_files(&record.files, true)?;
        for file in &record.files {
            if let Some(actual) = existing(&staged(&self.checked_path(&file.path)?))?
                && actual != file.digest
            {
                return Err("staged_file_changed_restore_refused");
            }
        }
        record.phase = Phase::Restoring;
        self.save(&record)?;
        host.stop_gateway()?;
        host.stop_original()?;
        for file in record.files.iter().rev() {
            let path = self.checked_path(&file.path)?;
            // Recheck immediately before unlinking; do not overwrite later edits.
            if let Some(actual) = existing(&path)? {
                if actual != file.digest {
                    return Err("owned_file_changed_restore_refused");
                }
                fs::remove_file(&path).map_err(|_| "owned_file_remove_failed")?;
                sync_parent(&path)?;
            }
            let temporary = staged(&path);
            if let Some(actual) = existing(&temporary)? {
                if actual != file.digest {
                    return Err("staged_file_changed_restore_refused");
                }
                fs::remove_file(&temporary).map_err(|_| "staged_file_remove_failed")?;
                sync_parent(&temporary)?;
            }
        }
        host.reload()?;
        host.restore_original(record.original_active)?;
        host.verify_original(record.original_active)?;
        record.phase = Phase::Restored;
        self.save(&record)
    }

    fn describe(&self, artifacts: &[Artifact]) -> Result<Vec<OwnedFile>> {
        if artifacts.len() > MAX_DEPLOYMENT_FILES {
            return Err("deployment_artifact_limit_exceeded");
        }
        let mut files = Vec::new();
        for artifact in artifacts {
            self.checked_path(&artifact.path)?;
            if files.iter().any(|f: &OwnedFile| f.path == artifact.path)
                || !matches!(artifact.mode, 0o600 | 0o644 | 0o755)
            {
                return Err("invalid_deployment_artifact");
            }
            files.push(OwnedFile {
                path: artifact.path.clone(),
                digest: digest(&artifact.bytes),
                mode: artifact.mode,
            });
        }
        Ok(files)
    }

    fn checked_path(&self, relative: &Path) -> Result<PathBuf> {
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("unsafe_installation_path");
        }
        let mut path = self.root.clone();
        let count = relative.components().count();
        for (index, component) in relative.components().enumerate() {
            path.push(component.as_os_str());
            match fs::symlink_metadata(&path) {
                Ok(metadata)
                    if metadata.file_type().is_symlink()
                        || (index + 1 < count && !metadata.is_dir()) =>
                {
                    return Err("unsafe_installation_path");
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => return Err("installation_path_unreadable"),
            }
        }
        Ok(path)
    }

    fn install(&self, artifact: &Artifact) -> Result<()> {
        let path = self.checked_path(&artifact.path)?;
        let parent = path.parent().ok_or("unsafe_installation_path")?;
        // New runtime-visible directories need explicit permissions even under
        // a restrictive sudo umask. Preserve every pre-existing directory mode.
        let relative_parent = parent
            .strip_prefix(&self.root)
            .map_err(|_| "unsafe_installation_path")?;
        let mut directory = self.root.clone();
        for component in relative_parent.components() {
            directory.push(component);
            match fs::create_dir(&directory) {
                Ok(()) => {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        fs::set_permissions(&directory, fs::Permissions::from_mode(0o755))
                            .map_err(|_| "installation_directory_permissions_failed")?;
                    }
                    sync_parent(&directory)?;
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(_) => return Err("installation_directory_failed"),
            }
        }
        self.checked_path(&artifact.path)?;
        // Materialize the complete inode before exposing the destination. link
        // is atomic and fails on an occupied path, unlike overwrite-by-rename.
        let temporary = staged(&path);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(artifact.mode).custom_flags(libc::O_NOFOLLOW);
        }
        let mut file = options
            .open(&temporary)
            .map_err(|_| "artifact_stage_failed")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(artifact.mode))
                .map_err(|_| "artifact_permissions_failed")?;
        }
        let result = (|| {
            file.write_all(&artifact.bytes)
                .map_err(|_| "artifact_write_failed")?;
            file.sync_all().map_err(|_| "artifact_sync_failed")?;
            fs::hard_link(&temporary, &path).map_err(|_| "artifact_publish_failed")?;
            sync_parent(&path)
        })();
        let _ = fs::remove_file(&temporary);
        result
    }

    fn check_files(&self, files: &[OwnedFile], allow_missing: bool) -> Result<()> {
        for file in files {
            match existing(&self.checked_path(&file.path)?)? {
                Some(actual)
                    if actual == file.digest
                        && mode_matches(&self.checked_path(&file.path)?, file.mode)? => {}
                None if allow_missing => {}
                _ => return Err("owned_file_changed_restore_refused"),
            }
        }
        Ok(())
    }

    fn load(&self) -> Result<Option<Record>> {
        match fs::symlink_metadata(&self.journal_path) {
            Ok(metadata) if !metadata.is_file() || metadata.len() > 1024 * 1024 => {
                return Err("setup_journal_invalid");
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err("setup_journal_unreadable"),
        }
        let bytes = match fs::read(&self.journal_path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(_) => return Err("setup_journal_unreadable"),
        };
        let record: Record = serde_json::from_slice(&bytes).map_err(|_| "setup_journal_invalid")?;
        if record.version != 1
            || record.root != self.root
            || record.files.is_empty()
            || record.files.len() > MAX_DEPLOYMENT_FILES
            || record
                .files
                .iter()
                .any(|f| f.digest.len() != 64 || !f.digest.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err("setup_journal_invalid");
        }
        for file in &record.files {
            self.checked_path(&file.path)?;
        }
        Ok(Some(record))
    }

    fn save(&self, record: &Record) -> Result<()> {
        storage::write_json(&self.journal_path, record).map_err(|_| "setup_journal_write_failed")
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn mode_matches(path: &Path, expected: u32) -> Result<bool> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        Ok(fs::metadata(path)
            .map_err(|_| "owned_file_unreadable")?
            .permissions()
            .mode()
            & 0o777
            == expected)
    }
    #[cfg(not(unix))]
    {
        let _ = (path, expected);
        Ok(true)
    }
}
fn staged(path: &Path) -> PathBuf {
    let mut name = path
        .file_name()
        .expect("validated artifact path")
        .to_os_string();
    name.push(".kuno-pending");
    path.with_file_name(name)
}
fn existing(path: &Path) -> Result<Option<String>> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.is_file() => Err("owned_path_not_regular"),
        Ok(_) => Ok(Some(digest(
            &fs::read(path).map_err(|_| "owned_file_unreadable")?,
        ))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err("owned_file_unreadable"),
    }
}
fn same_files(left: &[OwnedFile], right: &[OwnedFile]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(a, b)| a.path == b.path && a.digest == b.digest && a.mode == b.mode)
}
fn sync_parent(path: &Path) -> Result<()> {
    File::open(path.parent().ok_or("unsafe_installation_path")?)
        .and_then(|f| f.sync_all())
        .map_err(|_| "installation_directory_sync_failed")
}
