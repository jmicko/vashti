//! Atomic, private, content-free operational state writes.
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::Path,
};

pub fn write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    let temporary = path.with_extension("pending");
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    // A prior interrupted write is never used as the committed state.
    if temporary.try_exists()? {
        if !fs::symlink_metadata(&temporary)?.file_type().is_file() {
            return Err(io::Error::other("state_temp_not_regular"));
        }
        fs::remove_file(&temporary)?;
    }
    let mut file = options.open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    File::open(
        path.parent()
            .ok_or_else(|| io::Error::other("state_parent_missing"))?,
    )?
    .sync_all()
}
