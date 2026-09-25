//! An in-flight marker, host boot identity, and exclusive lock; no request information.
use std::{
    fs::{self, File, OpenOptions},
    io,
    path::{Path, PathBuf},
};

pub struct Journal {
    directory: PathBuf,
    _lock: File,
}

fn options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    options
}

impl Journal {
    pub fn open(directory: &Path) -> io::Result<Self> {
        let boot = fs::read_to_string("/proc/sys/kernel/random/boot_id")?;
        Self::open_for_boot(directory, boot.trim())
    }

    fn open_for_boot(directory: &Path, boot: &str) -> io::Result<Self> {
        if !valid_boot_id(boot) {
            return Err(io::Error::other("invalid_host_boot_identity"));
        }
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(directory)?;
        if !fs::symlink_metadata(directory)?.file_type().is_dir() {
            return Err(io::Error::other("state_directory_must_not_be_a_symlink"));
        }
        let lock = options()
            .create(true)
            .truncate(false)
            .open(directory.join("instance.lock"))?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: lock owns a valid descriptor; flock only changes its lock.
            if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                return Err(io::Error::other("state_directory_in_use"));
            }
        }
        #[cfg(not(unix))]
        return Err(io::Error::other("persistent_admission_requires_unix"));
        let journal = Self {
            directory: directory.into(),
            _lock: lock,
        };
        journal.reconcile_boot(boot)?;
        Ok(journal)
    }

    fn reconcile_boot(&self, current: &str) -> io::Result<()> {
        let path = self.directory.join("host-boot.json");
        let previous: Option<String> = match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.is_file() && metadata.len() <= 80 => {
                let value: String =
                    serde_json::from_slice(&fs::read(&path)?).map_err(io::Error::other)?;
                if !valid_boot_id(&value) {
                    return Err(io::Error::other("invalid_saved_boot_identity"));
                }
                Some(value)
            }
            Ok(_) => return Err(io::Error::other("invalid_saved_boot_identity")),
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        // A local upstream cannot retain GPU work across a host reboot. A
        // gateway-only restart gives no such assurance. Legacy empty markers
        // without a recorded boot identity still require explicit recovery.
        if previous.as_deref().is_some_and(|old| old != current) && self.uncertain()? {
            let marker = fs::symlink_metadata(self.directory.join("upstream-in-flight"))?;
            if !marker.is_file() || marker.len() != 0 {
                return Err(io::Error::other("recovery_marker_changed"));
            }
            self.finish()?;
        }
        if previous.as_deref() != Some(current) {
            crate::storage::write_json(&path, &current)?;
        }
        Ok(())
    }

    pub fn uncertain(&self) -> io::Result<bool> {
        match fs::symlink_metadata(self.directory.join("upstream-in-flight")) {
            Ok(_) => Ok(true),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(error) => Err(error),
        }
    }

    pub fn begin(&self) -> io::Result<()> {
        options()
            .create_new(true)
            .open(self.directory.join("upstream-in-flight"))?
            .sync_all()?;
        self.sync_directory()
    }

    pub fn finish(&self) -> io::Result<()> {
        fs::remove_file(self.directory.join("upstream-in-flight"))?;
        self.sync_directory()
    }

    fn sync_directory(&self) -> io::Result<()> {
        File::open(&self.directory)?.sync_all()
    }
}

fn valid_boot_id(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if matches!(i, 8 | 13 | 18 | 23) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    const FIRST: &str = "11111111-1111-4111-8111-111111111111";
    const SECOND: &str = "22222222-2222-4222-8222-222222222222";

    #[test]
    fn only_a_changed_host_boot_clears_uncertainty() {
        let directory = tempfile::tempdir().unwrap();
        let journal = Journal::open_for_boot(directory.path(), FIRST).unwrap();
        journal.begin().unwrap();
        assert!(Journal::open_for_boot(directory.path(), SECOND).is_err());
        assert!(journal.uncertain().unwrap());
        drop(journal);
        let restarted = Journal::open_for_boot(directory.path(), FIRST).unwrap();
        assert!(restarted.uncertain().unwrap());
        drop(restarted);
        let rebooted = Journal::open_for_boot(directory.path(), SECOND).unwrap();
        assert!(!rebooted.uncertain().unwrap());
        rebooted.begin().unwrap();
        rebooted.finish().unwrap();
    }

    #[test]
    fn legacy_and_corrupt_state_cannot_prove_a_reboot() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("upstream-in-flight"), []).unwrap();
        let legacy = Journal::open_for_boot(directory.path(), FIRST).unwrap();
        assert!(legacy.uncertain().unwrap());
        drop(legacy);
        fs::write(directory.path().join("host-boot.json"), b"bad data").unwrap();
        assert!(Journal::open_for_boot(directory.path(), SECOND).is_err());
        assert!(directory.path().join("upstream-in-flight").exists());
    }

    #[test]
    fn changed_marker_is_preserved_on_reboot() {
        let directory = tempfile::tempdir().unwrap();
        drop(Journal::open_for_boot(directory.path(), FIRST).unwrap());
        fs::write(
            directory.path().join("upstream-in-flight"),
            b"administrator edit",
        )
        .unwrap();
        assert!(Journal::open_for_boot(directory.path(), SECOND).is_err());
        assert_eq!(
            fs::read(directory.path().join("upstream-in-flight")).unwrap(),
            b"administrator edit"
        );
    }
}
