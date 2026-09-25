//! Curated, immutable model downloads. Only model artifacts reach disk; metadata
//! imported from an image never supplies a URL or a destination path.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    ffi::CString,
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::{
        fs::{OpenOptionsExt, PermissionsExt},
        io::AsRawFd,
    },
    path::Path,
};

type Result<T> = std::result::Result<T, &'static str>;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Collection {
    pub id: String,
    pub repository: String,
    pub revision: String,
    pub license: String,
    pub files: Vec<Artifact>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}
pub fn qwen() -> Collection {
    serde_json::from_str(include_str!("../collections/qwen-image-2512-fp8.json"))
        .expect("compiled collection")
}
pub fn collection(id: &str) -> Result<Collection> {
    match id {
        "qwen-image-2512-fp8" => Ok(qwen()),
        "qwen-image-2.1-int8" => Ok(serde_json::from_str(include_str!(
            "../collections/qwen-image-2.1-int8.json"
        ))
        .expect("compiled collection")),
        _ => Err("unknown_model_collection"),
    }
}
fn open(path: &Path, create: bool) -> Result<File> {
    let f = OpenOptions::new()
        .read(true)
        .write(create)
        .create(create)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|_| "model_file_unavailable")?;
    if !f
        .metadata()
        .map_err(|_| "model_file_unavailable")?
        .is_file()
    {
        return Err("model_file_not_regular");
    }
    Ok(f)
}
fn hash(file: &mut File) -> Result<(u64, String)> {
    file.seek(SeekFrom::Start(0))
        .map_err(|_| "model_read_failed")?;
    let mut hash = Sha256::new();
    let mut size = 0;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).map_err(|_| "model_read_failed")?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
        size += n as u64;
    }
    Ok((size, format!("{:x}", hash.finalize())))
}
fn matches(file: &mut File, artifact: &Artifact) -> Result<bool> {
    let (size, digest) = hash(file)?;
    Ok(size == artifact.size && digest == artifact.sha256)
}
/// Installation verifies the exact curated files before exposing them through
/// its read-only mount. The public weights contain no inference content.
pub fn verify(directory: &Path, id: &str) -> Result<()> {
    for artifact in collection(id)?.files {
        let name = artifact
            .path
            .rsplit('/')
            .next()
            .ok_or("collection_invalid")?;
        eprintln!("kuno: Verifying {name}");
        let mut file = open(&directory.join(name), false)?;
        if !matches(&mut file, &artifact)? {
            return Err("existing_model_checksum_mismatch");
        }
        if file
            .metadata()
            .map_err(|_| "model_file_unavailable")?
            .permissions()
            .mode()
            & 4
            == 0
        {
            return Err("image_model_requires_service_read_permission");
        }
    }
    Ok(())
}

/// Fast startup check after installation's full hash verification. Revalidating
/// artifact contents remains the downloader/installer's responsibility.
pub fn ready(directory: &Path, id: &str) -> Result<()> {
    for artifact in collection(id)?.files {
        let name = artifact
            .path
            .rsplit('/')
            .next()
            .ok_or("collection_invalid")?;
        let file = open(&directory.join(name), false)?;
        if file.metadata().map_err(|_| "model_file_unavailable")?.len() != artifact.size {
            return Err("image_model_size_mismatch");
        }
    }
    Ok(())
}
fn available(directory: &Path) -> Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let name =
        CString::new(directory.as_os_str().as_bytes()).map_err(|_| "model_directory_invalid")?;
    let mut s = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(name.as_ptr(), s.as_mut_ptr()) } != 0 {
        return Err("model_space_unavailable");
    }
    let s = unsafe { s.assume_init() };
    Ok(s.f_bavail.saturating_mul(s.f_frsize))
}
fn resume_range(header: &str, offset: u64, size: u64) -> bool {
    let Some(value) = header.strip_prefix("bytes ") else {
        return false;
    };
    let Some((range, total)) = value.split_once('/') else {
        return false;
    };
    let Some((start, end)) = range.split_once('-') else {
        return false;
    };
    start.parse::<u64>().ok() == Some(offset)
        && total.parse::<u64>().ok() == Some(size)
        && end.parse::<u64>().ok() == size.checked_sub(1)
}
fn client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(30))
        .timeout(std::time::Duration::from_secs(7200))
        .redirect(reqwest::redirect::Policy::custom(|attempt| {
            let url = attempt.url();
            let host = url.host_str().unwrap_or("");
            if attempt.previous().len() >= 10
                || url.scheme() != "https"
                || !(host == "huggingface.co"
                    || host.ends_with(".huggingface.co")
                    || host.ends_with(".hf.co"))
            {
                attempt.error("unapproved_model_redirect")
            } else {
                attempt.follow()
            }
        }))
        .build()
        .map_err(|_| "model_download_client_failed")
}
/// Called on a blocking CLI thread. Exclusive store lock, space checks, bounded
/// streams, immutable revisions, digest verification, and atomic activation.
pub fn fetch(id: &str, directory: &Path) -> Result<()> {
    let collection = collection(id)?;
    if !directory.is_absolute() {
        return Err("model_directory_must_be_absolute");
    }
    fs::create_dir_all(directory).map_err(|_| "model_directory_unavailable")?;
    if !fs::symlink_metadata(directory)
        .map_err(|_| "model_directory_unavailable")?
        .is_dir()
    {
        return Err("model_directory_not_regular");
    }
    let lock = open(&directory.join(".download-lock"), true)?;
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err("model_download_already_running");
    }
    let client = client()?;
    for artifact in &collection.files {
        let name = artifact
            .path
            .rsplit('/')
            .next()
            .ok_or("collection_invalid")?;
        let destination = directory.join(name);
        if destination
            .try_exists()
            .map_err(|_| "model_file_unavailable")?
        {
            eprintln!("kuno: Verifying {name}");
            if !matches(&mut open(&destination, false)?, artifact)? {
                return Err("existing_model_checksum_mismatch");
            }
            continue;
        }
        let partial = directory.join(format!("{}.partial", artifact.sha256));
        let mut file = open(&partial, true)?;
        let mut offset = file.metadata().map_err(|_| "model_file_unavailable")?.len();
        if offset >= artifact.size {
            if offset == artifact.size && matches(&mut file, artifact)? {
                file.set_permissions(fs::Permissions::from_mode(0o644))
                    .map_err(|_| "model_permissions_failed")?;
                fs::hard_link(&partial, &destination).map_err(|_| "model_activation_failed")?;
                fs::remove_file(&partial).map_err(|_| "model_activation_failed")?;
                File::open(directory)
                    .and_then(|f| f.sync_all())
                    .map_err(|_| "model_sync_failed")?;
                continue;
            }
            file.set_len(0).map_err(|_| "model_write_failed")?;
            offset = 0;
        }
        if available(directory)? < artifact.size - offset + 512 * 1024 * 1024 {
            return Err("insufficient_model_disk_space");
        }
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            collection.repository, collection.revision, artifact.path
        );
        let mut request = client.get(url).header("Accept-Encoding", "identity");
        if offset > 0 {
            request = request.header("Range", format!("bytes={offset}-"));
        }
        eprintln!(
            "kuno: Downloading {name} ({offset}/{} bytes)",
            artifact.size
        );
        let mut response = request.send().map_err(|_| "model_download_failed")?;
        if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            if !response
                .headers()
                .get("Content-Range")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|h| resume_range(h, offset, artifact.size))
            {
                return Err("model_download_range_invalid");
            }
        } else if response.status() == reqwest::StatusCode::OK {
            file.set_len(0).map_err(|_| "model_write_failed")?;
            offset = 0;
        } else {
            return Err("model_download_rejected");
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|_| "model_write_failed")?;
        let mut bytes = offset;
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut progress = std::time::Instant::now();
        loop {
            let n = response
                .read(&mut buffer)
                .map_err(|_| "model_download_interrupted_resume_available")?;
            if n == 0 {
                break;
            }
            bytes = bytes
                .checked_add(n as u64)
                .ok_or("model_download_too_large")?;
            if bytes > artifact.size {
                return Err("model_download_too_large");
            }
            file.write_all(&buffer[..n])
                .map_err(|_| "model_write_failed")?;
            if progress.elapsed().as_secs() >= 15 {
                eprintln!("kuno: {name}: {}%", bytes * 100 / artifact.size);
                progress = std::time::Instant::now();
            }
        }
        if bytes != artifact.size {
            return Err("model_download_incomplete_resume_available");
        }
        file.sync_all().map_err(|_| "model_sync_failed")?;
        eprintln!("kuno: Verifying {name}");
        if !matches(&mut file, artifact)? {
            return Err("downloaded_model_checksum_mismatch");
        }
        // No replacement of a file created by another application during download.
        file.set_permissions(fs::Permissions::from_mode(0o644))
            .map_err(|_| "model_permissions_failed")?;
        fs::hard_link(&partial, &destination).map_err(|_| "model_activation_failed")?;
        fs::remove_file(&partial).map_err(|_| "model_activation_failed")?;
        File::open(directory)
            .and_then(|f| f.sync_all())
            .map_err(|_| "model_sync_failed")?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checksum_works_on_a_small_worker_stack() {
        // Tokio blocking workers commonly have a 2 MiB stack. Nested download
        // and verification buffers must not put megabytes on that stack.
        std::thread::Builder::new()
            .stack_size(256 * 1024)
            .spawn(|| {
                let mut file = tempfile::tempfile().unwrap();
                file.write_all(b"abc").unwrap();
                let (size, digest) = hash(&mut file).unwrap();
                assert_eq!(size, 3);
                assert_eq!(
                    digest,
                    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
    #[test]
    fn immutable_manifest_has_complete_reference_dependencies() {
        let c = qwen();
        assert_eq!(c.files.len(), 3);
        assert_eq!(c.revision.len(), 40);
        for a in c.files {
            assert_eq!(a.sha256.len(), 64);
            assert!(a.size > 0);
            assert!(!a.path.contains(".."));
        }
    }
    #[test]
    fn resume_rejects_wrong_offsets_totals_and_unspecified_ranges() {
        assert!(resume_range("bytes 5-9/10", 5, 10));
        for s in [
            "bytes 0-9/10",
            "bytes 5-8/10",
            "bytes 5-9/*",
            "bytes 5-9/11",
        ] {
            assert!(!resume_range(s, 5, 10));
        }
    }
    #[test]
    fn symlinks_and_nonregular_files_are_not_download_targets() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("link");
        std::os::unix::fs::symlink("missing", &p).unwrap();
        assert!(open(&p, true).is_err());
        assert!(open(d.path(), false).is_err());
    }
}
