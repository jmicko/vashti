//! Only a fixed marker and an exclusive process lock; no request information.
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
        Ok(Self {
            directory: directory.into(),
            _lock: lock,
        })
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
