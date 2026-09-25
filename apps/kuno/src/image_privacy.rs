//! Linux image workers have no writable content store. Pipes are supplied by
//! the parent. Writes are limited to GPU devices and the process's own ephemeral
//! task controls (CUDA names internal threads through /proc/self/task/*/comm).
use std::{
    fs,
    os::fd::{AsRawFd, FromRawFd, OwnedFd},
    os::unix::fs::OpenOptionsExt,
};

/// Require an effective no-swap cgroup (or a host with swap disabled). This is
/// checked before reading the recipe. An ordinary shell is not sufficient.
pub fn require_no_swap() -> Result<(), &'static str> {
    let swaps = fs::read_to_string("/proc/swaps").map_err(|_| "image_swap_status_unavailable")?;
    if swaps.lines().skip(1).all(|line| line.trim().is_empty()) {
        return Ok(());
    }
    let groups = fs::read_to_string("/proc/self/cgroup").map_err(|_| "image_cgroup_unavailable")?;
    let group = groups
        .lines()
        .find_map(|line| line.strip_prefix("0::/"))
        .ok_or("image_requires_cgroup_v2_no_swap")?;
    if group.split('/').any(|part| part == ".." || part == ".") {
        return Err("image_cgroup_invalid");
    }
    let root = std::path::Path::new("/sys/fs/cgroup");
    let mut path = root.join(group);
    loop {
        if fs::read_to_string(path.join("memory.swap.max"))
            .ok()
            .is_some_and(|v| v.trim() == "0")
        {
            return Ok(());
        }
        if path == root || !path.pop() {
            break;
        }
    }
    Err("image_requires_memory_swap_max_zero")
}

/// Landlock ABI 3 also covers truncate. The native library and its new threads
/// inherit this restriction. Do not silently downgrade on older kernels.
pub fn require_landlock() -> Result<(), &'static str> {
    let abi = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            std::ptr::null::<u8>(),
            0,
            1,
        )
    };
    if abi < 3 {
        return Err("image_requires_landlock_abi_3");
    }
    Ok(())
}

pub fn restrict_filesystem() -> Result<(), &'static str> {
    require_landlock()?;
    const WRITE: u64 = 1 << 1;
    const WRITE_OPERATIONS: u64 = WRITE | (((1u64 << 15) - 1) & !((1u64 << 4) - 1));
    #[repr(C)]
    struct Ruleset {
        handled_access_fs: u64,
    }
    #[repr(C, packed)]
    struct PathRule {
        allowed_access: u64,
        parent_fd: i32,
    }
    let attr = Ruleset {
        handled_access_fs: WRITE_OPERATIONS,
    };
    let fd = unsafe {
        libc::syscall(
            libc::SYS_landlock_create_ruleset,
            &attr,
            std::mem::size_of::<Ruleset>(),
            0,
        )
    };
    if fd < 0 {
        return Err("image_filesystem_rules_failed");
    }
    let rules = unsafe { OwnedFd::from_raw_fd(fd as i32) };
    let mut devices = vec![std::path::PathBuf::from("/dev/null")];
    for directory in ["/dev/dri", "/dev/nvidia-caps"] {
        if let Ok(entries) = fs::read_dir(directory) {
            devices.extend(entries.flatten().map(|entry| entry.path()));
        }
    }
    if let Ok(entries) = fs::read_dir("/dev") {
        devices.extend(
            entries
                .flatten()
                .filter(|entry| entry.file_name().as_encoded_bytes().starts_with(b"nvidia"))
                .map(|entry| entry.path()),
        );
    }
    use std::os::unix::fs::FileTypeExt;
    for path in devices {
        // No directory exception: even shader caches and /dev/shm files must
        // remain unwritable. Device nodes are opened with O_NOFOLLOW.
        if !fs::symlink_metadata(&path).is_ok_and(|m| m.file_type().is_char_device()) {
            continue;
        }
        let device = fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(path)
            .map_err(|_| "image_device_rules_failed")?;
        let rule = PathRule {
            allowed_access: WRITE,
            parent_fd: device.as_raw_fd(),
        };
        if unsafe { libc::syscall(libc::SYS_landlock_add_rule, rules.as_raw_fd(), 1, &rule, 0) }
            != 0
        {
            return Err("image_device_rules_failed");
        }
    }
    // CUDA's pthread_setname_np path opens a newly created thread's comm with
    // O_TRUNC. Task IDs do not exist when this policy is installed, so allow
    // writes/truncation beneath this process's task directory, without creation
    // or removal rights. Procfs controls are not a persistent content store.
    // Proc fd symlinks still resolve to their underlying file's Landlock policy.
    let tasks = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_PATH | libc::O_CLOEXEC)
        .open("/proc/self/task")
        .map_err(|_| "image_filesystem_rules_failed")?;
    let task_rule = PathRule {
        allowed_access: WRITE | (1 << 14),
        parent_fd: tasks.as_raw_fd(),
    };
    if unsafe {
        libc::syscall(
            libc::SYS_landlock_add_rule,
            rules.as_raw_fd(),
            1,
            &task_rule,
            0,
        )
    } != 0
    {
        return Err("image_filesystem_rules_failed");
    }
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0
        || unsafe { libc::syscall(libc::SYS_landlock_restrict_self, rules.as_raw_fd(), 0) } != 0
    {
        return Err("image_filesystem_restriction_failed");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn child_cannot_persist_content() {
        if let Some(path) = std::env::var_os("KUNO_TEST_LANDLOCK_CHILD") {
            // Restricted CI containers may not expose Landlock at all. A
            // production worker must fail closed there, never skip isolation.
            match restrict_filesystem() {
                Ok(()) => (),
                Err("image_requires_landlock_abi_3") => return,
                Err(code) => panic!("{code}"),
            }
            let path = std::path::PathBuf::from(path);
            assert_eq!(fs::read(path.join("existing")).unwrap(), b"fixture");
            assert!(fs::write(path.join("existing"), b"changed").is_err());
            assert!(fs::write(path.join("new"), b"content").is_err());
            assert!(fs::create_dir(path.join("new-dir")).is_err());
            assert!(fs::remove_file(path.join("existing")).is_err());
            assert!(fs::rename(path.join("existing"), path.join("moved")).is_err());
            let existing = fs::File::open(path.join("existing")).unwrap();
            let tid = unsafe { libc::syscall(libc::SYS_gettid) };
            let alias = format!("/proc/self/task/{tid}/fd/{}", existing.as_raw_fd());
            assert!(fs::write(alias, b"changed via proc").is_err());
            std::thread::spawn(|| {
                let tid = unsafe { libc::syscall(libc::SYS_gettid) };
                fs::write(format!("/proc/self/task/{tid}/comm"), b"kuno-test").unwrap();
            })
            .join()
            .unwrap();
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("existing"), b"fixture").unwrap();
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "image_privacy::tests::child_cannot_persist_content",
            ])
            .env("KUNO_TEST_LANDLOCK_CHILD", directory.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert_eq!(
            fs::read(directory.path().join("existing")).unwrap(),
            b"fixture"
        );
    }
}
