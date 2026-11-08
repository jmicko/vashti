//! Load-time WGSL module prewarm. GPU code only: the fixed-public cache writer
//! records each compiled kernel's WGSL source and entry point (shape constants,
//! never request content) under the code cache; serving workers rebuild those
//! modules on background threads while weights stream from disk, and the
//! server takes a prebuilt module only when entry, mode and source match exactly.
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

static RECORD: OnceLock<PathBuf> = OnceLock::new();
type Ready = HashMap<u64, (String, String, bool, bool, Vec<wgpu::ShaderModule>)>;
// Uses of each source in this recording process (one module per kernel id).
static USES: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();
static READY: OnceLock<Mutex<Ready>> = OnceLock::new();

pub(crate) fn source_key(entry: &str, source: &str, checked: bool) -> u64 {
    key(entry, source, checked)
}
fn key(entry: &str, source: &str, checked: bool) -> u64 {
    // FNV-1a; the full source is compared on use, so this only buckets.
    let mut h = 0xcbf29ce484222325u64;
    for b in entry.bytes().chain([0, checked as u8]).chain(source.bytes()) {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// Only the fixed-public cache writer calls this.
pub fn set_wgsl_record_dir(dir: PathBuf) {
    if std::fs::create_dir_all(&dir).is_ok() {
        let _ = RECORD.set(dir);
    }
}

pub(crate) fn record(entry: &str, source: &str, checked: bool) {
    let Some(dir) = RECORD.get() else { return };
    if entry.contains('\n') {
        return;
    }
    let k = key(entry, source, checked);
    let use_index = {
        let Ok(mut uses) = USES.get_or_init(|| Mutex::new(HashMap::new())).lock() else { return };
        let n = uses.entry(k).or_insert(0);
        *n += 1;
        *n - 1
    };
    // One file per concurrent use: `<key>.wgsl`, then `<key>-1.wgsl`, ...
    let name = if use_index == 0 { format!("{k:016x}.wgsl") } else { format!("{k:016x}-{use_index}.wgsl") };
    let path = dir.join(name);
    if path.exists() {
        return;
    }
    let tmp = dir.join(format!(".{k:016x}-{use_index}.{}", std::process::id()));
    let body = format!("{entry}\n{}\n{source}", if checked { "checked" } else { "unchecked" });
    if std::fs::write(&tmp, body).is_ok() {
        let _ = std::fs::rename(&tmp, &path);
    }
}

pub(crate) fn take(entry: &str, source: &str, checked: bool) -> Option<wgpu::ShaderModule> {
    let ready = READY.get()?;
    let mut map = ready.lock().ok()?;
    let k = key(entry, source, checked);
    let hit = map.get(&k).is_some_and(|(e, s, c, native, _)| {
        e == entry && s == source && *c == checked && *native == super::strata_graph::enabled()
    });
    // Each kernel id needs its own module; several may share one source.
    if hit { map.get_mut(&k).and_then(|v| v.4.pop()) } else { None }
}

/// Rebuild every recorded module on `threads` detached background threads.
/// Failures are ignored: the server then compiles that kernel lazily as before.
pub fn prewarm_wgsl(dir: &Path, device: wgpu::Device, threads: usize) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let files: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "wgsl"))
        .collect();
    if files.is_empty() {
        return;
    }
    let ready = READY.get_or_init(|| Mutex::new(HashMap::new()));
    let files = std::sync::Arc::new(Mutex::new(files));
    for _ in 0..threads.max(1) {
        let files = files.clone();
        let device = device.clone();
        let _ = std::thread::Builder::new()
            .name("kuno-prewarm".into())
            .stack_size(8 * 1024 * 1024)
            .spawn(move || loop {
                let Some(path) = files.lock().ok().and_then(|mut f| f.pop()) else { return };
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                let mut parts = text.splitn(3, '\n');
                let (Some(entry), Some(mode), Some(source)) = (parts.next(), parts.next(), parts.next())
                else {
                    continue;
                };
                let checked = mode == "checked";
                let native = super::strata_graph::enabled();
                let Ok(module) = crate::backend::compile_wgsl(&device, entry, source, checked) else {
                    super::strata_pipeline::CLAUDE_PIPELINE_TIME[9]
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                };
                if let Ok(mut map) = ready.lock() {
                    map.entry(key(entry, source, checked))
                        .or_insert_with(|| {
                            (entry.to_owned(), source.to_owned(), checked, native, Vec::new())
                        })
                        .4
                        .push(module);
                }
            });
    }
}
