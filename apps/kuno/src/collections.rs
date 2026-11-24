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
    /// Files Kuno builds locally from downloaded ones, with pinned checksums.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub derived: Vec<Derived>,
}
/// A file Kuno builds from verified downloads. Deterministic, so pinned: a LoRA merged into
/// an INT8 ConvRot model (`base`, `lora`, `multiplier`; see `lora_merge`), or one downloaded
/// file converted for the engines (`convert`; see `kuno_image::engine::convert`).
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Derived {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiplier: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub convert: Option<Conversion>,
}
/// One downloaded file (`input`, its repository path) converted as a part of a model.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Conversion {
    /// `transformer`, `text_encoder` or `vae`.
    pub part: String,
    pub input: String,
}
impl Derived {
    fn artifact(&self) -> Artifact {
        Artifact {
            path: self.path.clone(),
            size: self.size,
            sha256: self.sha256.clone(),
            keep: true,
        }
    }
}
/// Where a downloaded file lives: its file name, or for a conversion's input (removed once
/// converted) a name made from its whole repository path, so that parts never collide.
fn local_name(artifact: &Artifact) -> Result<String> {
    if artifact.keep {
        return Ok(artifact
            .path
            .rsplit('/')
            .next()
            .ok_or("collection_invalid")?
            .to_owned());
    }
    if artifact.path.contains("..") || artifact.path.starts_with('/') {
        return Err("collection_invalid");
    }
    Ok(format!(".source-{}", artifact.path.replace('/', "--")))
}
/// The files a ready collection holds (kept downloads and derived files), by local name.
fn local_artifacts(collection: Collection) -> Result<Vec<(String, Artifact)>> {
    let mut out = Vec::new();
    for artifact in collection.files {
        if !artifact.keep {
            continue;
        }
        out.push((local_name(&artifact)?, artifact));
    }
    for derived in &collection.derived {
        out.push((derived.path.clone(), derived.artifact()));
    }
    Ok(out)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: String,
    pub size: u64,
    pub sha256: String,
    /// False for a conversion's input: removed once its output is verified.
    #[serde(default = "yes", skip_serializing_if = "is_true")]
    pub keep: bool,
}
fn yes() -> bool {
    true
}
fn is_true(value: &bool) -> bool {
    *value
}
pub fn qwen() -> Collection {
    serde_json::from_str(include_str!("../collections/qwen-image-2512-fp8.json"))
        .expect("compiled collection")
}
/// Number of collections `collection` knows; keep in step with its match arms.
pub const KNOWN_COLLECTIONS: usize = 5;
pub fn collection(id: &str) -> Result<Collection> {
    match id {
        "qwen-image-2512-fp8" => Ok(qwen()),
        "qwen-image-2512-int8" => Ok(serde_json::from_str(include_str!(
            "../collections/qwen-image-2512-int8.json"
        ))
        .expect("compiled collection")),
        "qwen-image-2.1-int8" => Ok(serde_json::from_str(include_str!(
            "../collections/qwen-image-2.1-int8.json"
        ))
        .expect("compiled collection")),
        kuno_protocol::MINIMAX_H3_COLLECTION => Ok(serde_json::from_str(include_str!(
            "../collections/minimax-h3-int8.json"
        ))
        .expect("compiled collection")),
        kuno_protocol::MINIMAX_H3_REFERENCE_COLLECTION => Ok(serde_json::from_str(include_str!(
            "../collections/minimax-h3-ref2va-int8.json"
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
    for (name, artifact) in local_artifacts(collection(id)?)? {
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
    for (name, artifact) in local_artifacts(collection(id)?)? {
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
    for artifact in collection.files.iter().filter(|a| a.keep) {
        download(&client, &collection, artifact, directory)?;
    }
    convert_all(&client, &collection, directory)?;
    for derived in collection.derived.iter().filter(|d| d.convert.is_none()) {
        derive(directory, derived)?;
    }
    Ok(())
}

/// Downloads one artifact to its local name: resumable, bounded, verified, then activated
/// atomically. A present file must already match its pin.
fn download(
    client: &reqwest::blocking::Client,
    collection: &Collection,
    artifact: &Artifact,
    directory: &Path,
) -> Result<()> {
    let name = &local_name(artifact)?;
    let destination = directory.join(name);
    if destination
        .try_exists()
        .map_err(|_| "model_file_unavailable")?
    {
        eprintln!("kuno: Verifying {name}");
        if !matches(&mut open(&destination, false)?, artifact)? {
            return Err("existing_model_checksum_mismatch");
        }
        return Ok(());
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
            return Ok(());
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
    Ok(())
}

/// Space kept free beyond what a download and the pending conversions need.
const MARGIN: u64 = 512 * 1024 * 1024;

/// Downloads every conversion's input and converts it, overlapping the two: a converter
/// thread works through inputs as they arrive while the next one downloads. A download
/// starts only when the disk holds it plus every pending output; otherwise it waits for a
/// conversion to finish (which removes its input). With ample space downloads run ahead;
/// with little, it alternates: download, convert, remove, next.
fn convert_all(
    client: &reqwest::blocking::Client,
    collection: &Collection,
    directory: &Path,
) -> Result<()> {
    let mut todo = Vec::new();
    for derived in &collection.derived {
        let Some(conversion) = &derived.convert else {
            continue;
        };
        let input = collection
            .files
            .iter()
            .find(|a| a.path == conversion.input)
            .ok_or("collection_invalid")?;
        let destination = directory.join(&derived.path);
        if destination
            .try_exists()
            .map_err(|_| "model_file_unavailable")?
        {
            eprintln!("kuno: Verifying {}", derived.path);
            if !matches(&mut open(&destination, false)?, &derived.artifact())? {
                return Err("existing_model_checksum_mismatch");
            }
            if !input.keep {
                remove_if_present(&directory.join(local_name(input)?))?;
            }
            continue;
        }
        todo.push((derived, conversion, input));
    }
    if todo.is_empty() {
        return Ok(());
    }
    let mut sizes = Vec::with_capacity(todo.len());
    for (derived, _, input) in &todo {
        let source = directory.join(local_name(input)?);
        let have = fs::metadata(&source).map(|m| m.len()).unwrap_or(0).min(input.size);
        let partial = fs::metadata(directory.join(format!("{}.partial", input.sha256)))
            .map(|m| m.len())
            .unwrap_or(0)
            .min(input.size);
        sizes.push((input.size - have.max(partial), derived.size));
    }
    schedule(
        &sizes,
        || available(directory),
        |index| download(client, collection, todo[index].2, directory),
        |index| {
            let (derived, conversion, input) = todo[index];
            convert_one(collection, directory, derived, conversion, input)
        },
    )
}

/// Runs `fetch` then `convert` for each job (`(bytes still to download, output bytes)`), in
/// order, converting on a second thread while later jobs download. A download starts only
/// when `space` covers it, its output and every pending output, plus `MARGIN`; otherwise it
/// waits for a conversion (each frees its input). Nothing pending and still too little
/// space is an error.
fn schedule(
    jobs: &[(u64, u64)],
    space: impl Fn() -> Result<u64>,
    fetch: impl Fn(usize) -> Result<()>,
    convert: impl Fn(usize) -> Result<()> + Sync,
) -> Result<()> {
    let (queue_in, queue) = std::sync::mpsc::channel::<usize>();
    let (finished, results) = std::sync::mpsc::channel::<Result<usize>>();
    std::thread::scope(|scope| {
        let convert = &convert;
        scope.spawn(move || {
            for index in queue {
                let result = convert(index);
                let failed = result.is_err();
                let _ = finished.send(result.map(|_| index));
                if failed {
                    break;
                }
            }
        });
        let mut pending = 0u64;
        let mut running = 0usize;
        let wait = |pending: &mut u64, running: &mut usize| -> Result<()> {
            let index = results.recv().map_err(|_| "model_conversion_failed")??;
            *pending -= jobs[index].1;
            *running -= 1;
            Ok(())
        };
        for (index, &(remaining, output)) in jobs.iter().enumerate() {
            while space()? < remaining + output + pending + MARGIN {
                if running == 0 {
                    return Err("insufficient_model_disk_space");
                }
                wait(&mut pending, &mut running)?;
            }
            fetch(index)?;
            queue_in.send(index).map_err(|_| "model_conversion_failed")?;
            pending += output;
            running += 1;
        }
        drop(queue_in);
        while running > 0 {
            wait(&mut pending, &mut running)?;
        }
        Ok(())
    })
}
fn remove_if_present(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("model_write_failed"),
    }
}

/// Converts one downloaded file, checks the output against its pin, activates it and
/// removes the input if it is not kept.
fn convert_one(
    collection: &Collection,
    directory: &Path,
    derived: &Derived,
    conversion: &Conversion,
    input: &Artifact,
) -> Result<()> {
    let source = directory.join(local_name(input)?);
    let destination = directory.join(&derived.path);
    if derived.path.contains("..") || derived.path.starts_with('/') {
        return Err("collection_invalid");
    }
    let parent = destination.parent().ok_or("collection_invalid")?;
    fs::create_dir_all(parent).map_err(|_| "model_directory_unavailable")?;
    let partial = directory.join(format!("{}.partial", derived.sha256));
    remove_if_present(&partial)?;
    eprintln!("kuno: Converting {}", conversion.input);
    let started = std::time::Instant::now();
    convert_file(
        &conversion.part,
        &source,
        &partial,
        &format!("{}@{}", collection.repository, collection.revision),
    )?;
    let mut file = open(&partial, false)?;
    if !matches(&mut file, &derived.artifact())? {
        let _ = fs::remove_file(&partial);
        return Err("derived_model_checksum_mismatch");
    }
    file.set_permissions(fs::Permissions::from_mode(0o644))
        .map_err(|_| "model_permissions_failed")?;
    fs::hard_link(&partial, &destination).map_err(|_| "model_activation_failed")?;
    fs::remove_file(&partial).map_err(|_| "model_activation_failed")?;
    if !input.keep {
        remove_if_present(&source)?;
    }
    for d in [parent, directory] {
        File::open(d)
            .and_then(|f| f.sync_all())
            .map_err(|_| "model_sync_failed")?;
    }
    eprintln!(
        "kuno: {} ready in {} s",
        derived.path,
        started.elapsed().as_secs()
    );
    Ok(())
}
#[cfg(feature = "rust-image")]
fn convert_file(part: &str, input: &Path, output: &Path, source: &str) -> Result<()> {
    use kuno_image::engine::convert::{Part, convert};
    let part = match part {
        "transformer" => Part::Transformer,
        "text_encoder" => Part::TextEncoder,
        "vae" => Part::Vae,
        _ => return Err("collection_invalid"),
    };
    convert(part, input, output, source)
        .map(|_| ())
        .map_err(|_| "model_conversion_failed")
}
#[cfg(not(feature = "rust-image"))]
fn convert_file(_: &str, _: &Path, _: &Path, _: &str) -> Result<()> {
    Err("model_conversion_not_in_this_build")
}

/// Builds a derived file from verified downloads; an existing one must match its pin.
fn derive(directory: &Path, derived: &Derived) -> Result<()> {
    let destination = directory.join(&derived.path);
    let artifact = derived.artifact();
    if destination
        .try_exists()
        .map_err(|_| "model_file_unavailable")?
    {
        eprintln!("kuno: Verifying {}", derived.path);
        if !matches(&mut open(&destination, false)?, &artifact)? {
            return Err("existing_model_checksum_mismatch");
        }
        return Ok(());
    }
    let partial = directory.join(format!("{}.partial", derived.sha256));
    match fs::remove_file(&partial) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(_) => return Err("model_write_failed"),
    }
    if available(directory)? < derived.size + 512 * 1024 * 1024 {
        return Err("insufficient_model_disk_space");
    }
    eprintln!("kuno: Merging the turbo LoRA into {}", derived.path);
    let started = std::time::Instant::now();
    let (Some(base), Some(lora), Some(multiplier)) =
        (&derived.base, &derived.lora, derived.multiplier)
    else {
        return Err("collection_invalid");
    };
    let digest = crate::lora_merge::merge(
        &directory.join(base),
        &directory.join(lora),
        &partial,
        multiplier,
        |done, total| {
            if done % 26 == 0 || done == total {
                eprintln!("kuno: {}: {}%", derived.path, done * 100 / total);
            }
        },
    )?;
    let file = open(&partial, false)?;
    let size = file.metadata().map_err(|_| "model_file_unavailable")?.len();
    if digest != derived.sha256 || size != derived.size {
        let _ = fs::remove_file(&partial);
        return Err("derived_model_checksum_mismatch");
    }
    file.set_permissions(fs::Permissions::from_mode(0o644))
        .map_err(|_| "model_permissions_failed")?;
    fs::hard_link(&partial, &destination).map_err(|_| "model_activation_failed")?;
    fs::remove_file(&partial).map_err(|_| "model_activation_failed")?;
    File::open(directory)
        .and_then(|f| f.sync_all())
        .map_err(|_| "model_sync_failed")?;
    eprintln!(
        "kuno: {} ready in {} s",
        derived.path,
        started.elapsed().as_secs()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    /// The download/convert schedule on a simulated disk: inputs of 10 GB that convert to 5
    /// GB. With room for one input and its output it alternates; with room for everything,
    /// downloads run ahead of conversions; it never needs more than the disk holds.
    #[test]
    fn schedule_fits_the_disk() {
        use std::sync::Mutex;
        const GB: u64 = 1 << 30;
        let run = |capacity: u64| -> (Result<()>, Vec<String>, u64) {
            let disk = Mutex::new((0u64, 0u64)); // used, peak
            let log = Mutex::new(Vec::new());
            let jobs = vec![(10 * GB, 5 * GB); 4];
            let space = || {
                let used = disk.lock().unwrap().0;
                Ok(capacity.saturating_sub(used))
            };
            let grow = |by: i64| {
                let mut d = disk.lock().unwrap();
                d.0 = (d.0 as i64 + by) as u64;
                d.1 = d.1.max(d.0);
            };
            let result = schedule(
                &jobs,
                space,
                |i| {
                    grow(10 * GB as i64);
                    log.lock().unwrap().push(format!("d{i}"));
                    Ok(())
                },
                |i| {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    grow(5 * GB as i64);
                    grow(-(10 * GB as i64));
                    log.lock().unwrap().push(format!("c{i}"));
                    Ok(())
                },
            );
            let peak = disk.lock().unwrap().1;
            (result, log.into_inner().unwrap(), peak)
        };
        // The least that can finish: the last input, every output and the margin. Downloads
        // still overlap where the disk allows, never past it.
        let least = 10 * GB + 4 * 5 * GB + MARGIN;
        let (result, log, peak) = run(least);
        assert!(result.is_ok());
        assert_eq!(log.len(), 8);
        assert!(peak <= least - MARGIN);
        // One byte less: it stops before running out, with what it finished kept.
        let (result, log, peak) = run(least - 1);
        assert_eq!(result, Err("insufficient_model_disk_space"));
        assert!(peak < least - MARGIN && log.iter().filter(|e| e.starts_with('c')).count() == 3);
        // Ample: every download starts before the first conversion ends.
        let (result, ample, _) = run(1000 * GB);
        assert!(result.is_ok());
        assert_eq!(&ample[..4], ["d0", "d1", "d2", "d3"]);
        assert!(schedule(&[(10 * GB, 5 * GB)], || Ok(GB), |_| Ok(()), |_| Ok(())).is_err());
    }
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
    fn video_collections_pin_downloads_and_their_derived_models() {
        let keyframe = collection(kuno_protocol::MINIMAX_H3_COLLECTION).unwrap();
        let reference = collection(kuno_protocol::MINIMAX_H3_REFERENCE_COLLECTION).unwrap();
        // The models share the text encoder and VAEs; only the DiT and LoRA differ.
        let shared = |c: &Collection| -> Vec<String> {
            c.files.iter().skip(2).map(|f| f.sha256.clone()).collect()
        };
        assert_eq!(shared(&keyframe), shared(&reference));
        assert_ne!(keyframe.files[0].sha256, reference.files[0].sha256);
        for c in [keyframe, reference] {
            video_collection_is_complete(c);
        }
    }
    fn video_collection_is_complete(c: Collection) {
        assert_eq!(c.files.len(), 5);
        assert_eq!(c.revision.len(), 40);
        let names: Vec<_> = local_artifacts(collection(&c.id).unwrap())
            .unwrap()
            .into_iter()
            .map(|(name, a)| {
                assert_eq!(a.sha256.len(), 64);
                name
            })
            .collect();
        assert_eq!(c.derived.len(), 1);
        let d = &c.derived[0];
        let (base, lora) = (d.base.as_ref().unwrap(), d.lora.as_ref().unwrap());
        assert!(names.contains(base) && names.contains(lora));
        assert_eq!(names.last(), Some(&d.path));
        // The merge rewrites tensors in place, so the size equals the base's.
        let base = c.files.iter().find(|f| f.path.ends_with(base.as_str())).unwrap();
        assert_eq!(base.size, d.size);
    }
    #[test]
    fn int8_2512_converts_every_downloaded_file_once() {
        let c = collection("qwen-image-2512-int8").unwrap();
        assert_eq!(c.repository, "Qwen/Qwen-Image-2512");
        assert_eq!(c.files.len(), c.derived.len());
        assert!(c.files.iter().all(|f| !f.keep && f.sha256.len() == 64));
        for f in &c.files {
            let d: Vec<_> = c
                .derived
                .iter()
                .filter(|d| d.convert.as_ref().is_some_and(|v| v.input == f.path))
                .collect();
            assert_eq!(d.len(), 1, "{}", f.path);
            assert!(d[0].path.starts_with("qwen-image-2512-int8/"));
        }
        // Only the converted files stay.
        let local = local_artifacts(c).unwrap();
        assert!(local.iter().all(|(name, _)| name.starts_with("qwen-image-2512-int8/")));
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
