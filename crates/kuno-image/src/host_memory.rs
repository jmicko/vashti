use std::{io, path::Path};

#[cfg(feature = "native-sd")]
pub(crate) fn available() -> u64 {
    available_with(|path| std::fs::read_to_string(path)).unwrap_or(0)
}

// Respect ancestor limits as well as host availability. Unknown accounting
// disables the optional RAM cache; it must not prevent ordinary GPU inference.
fn available_with(mut read: impl FnMut(&Path) -> io::Result<String>) -> Option<u64> {
    let info = read(Path::new("/proc/meminfo")).ok()?;
    let mut available = info.lines().find_map(|line| {
        let mut fields = line.strip_prefix("MemAvailable:")?.split_whitespace();
        let kib = fields.next()?.parse::<u64>().ok()?;
        (fields.next()? == "kB" && fields.next().is_none()).then(|| kib.checked_mul(1024))?
    })?;
    let groups = read(Path::new("/proc/self/cgroup")).ok()?;
    let group = groups.lines().find_map(|line| line.strip_prefix("0::/"))?;
    if group.starts_with('/') || group.split('/').any(|part| part == ".." || part == ".") {
        return None;
    }
    let root = Path::new("/sys/fs/cgroup");
    let mut path = root.join(group);
    loop {
        let mut current = None;
        for name in ["memory.max", "memory.high"] {
            let value = match read(&path.join(name)) {
                Ok(value) => value,
                Err(error) if path == root && error.kind() == io::ErrorKind::NotFound => continue,
                Err(_) => return None,
            };
            if value.trim() != "max" {
                let limit = value.trim().parse::<u64>().ok()?;
                let used = match current {
                    Some(value) => value,
                    None => {
                        let value = read(&path.join("memory.current"))
                            .ok()?
                            .trim()
                            .parse::<u64>()
                            .ok()?;
                        current = Some(value);
                        value
                    }
                };
                available = available.min(limit.saturating_sub(used));
            }
        }
        if path == root {
            break;
        }
        if !path.pop() {
            return None;
        }
    }
    Some(available)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    const GIB: u64 = 1024 * 1024 * 1024;
    fn files() -> BTreeMap<String, String> {
        [
            (
                "/proc/meminfo",
                format!("MemAvailable: {} kB\n", 48 * GIB / 1024),
            ),
            ("/proc/self/cgroup", "0::/parent/worker\n".into()),
            (
                "/sys/fs/cgroup/parent/worker/memory.max",
                (32 * GIB).to_string(),
            ),
            ("/sys/fs/cgroup/parent/worker/memory.high", "max".into()),
            (
                "/sys/fs/cgroup/parent/worker/memory.current",
                (4 * GIB).to_string(),
            ),
            ("/sys/fs/cgroup/parent/memory.max", (24 * GIB).to_string()),
            ("/sys/fs/cgroup/parent/memory.high", "max".into()),
            (
                "/sys/fs/cgroup/parent/memory.current",
                (10 * GIB).to_string(),
            ),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect()
    }
    fn measure(files: &BTreeMap<String, String>) -> Option<u64> {
        available_with(|path| {
            files
                .get(path.to_str().unwrap())
                .cloned()
                .ok_or_else(|| io::Error::from(io::ErrorKind::NotFound))
        })
    }
    #[test]
    fn respects_ancestor_limit_and_pressure_threshold() {
        let mut input = files();
        assert_eq!(measure(&input), Some(14 * GIB));
        input.insert(
            "/sys/fs/cgroup/parent/worker/memory.high".into(),
            (6 * GIB).to_string(),
        );
        assert_eq!(measure(&input), Some(2 * GIB));
        input.insert(
            "/sys/fs/cgroup/parent/worker/memory.current".into(),
            (7 * GIB).to_string(),
        );
        assert_eq!(measure(&input), Some(0));
    }
    #[test]
    fn unknown_or_invalid_accounting_disables_cache() {
        let mut input = files();
        input.remove("/sys/fs/cgroup/parent/memory.max");
        assert_eq!(measure(&input), None);
        for group in [
            "0::/parent/../worker",
            "0::/parent/./worker",
            "0:://worker",
            "1:memory:/worker",
        ] {
            let mut input = files();
            input.insert("/proc/self/cgroup".into(), group.into());
            assert_eq!(measure(&input), None);
        }
        let mut input = files();
        input.insert(
            "/proc/meminfo".into(),
            format!("MemAvailable: {} kB", u64::MAX),
        );
        assert_eq!(measure(&input), None);
    }
    #[test]
    fn honors_a_limit_at_the_cgroup_namespace_root() {
        let mut input = files();
        input.insert("/proc/self/cgroup".into(), "0::/".into());
        input.insert("/sys/fs/cgroup/memory.max".into(), (12 * GIB).to_string());
        input.insert(
            "/sys/fs/cgroup/memory.current".into(),
            (5 * GIB).to_string(),
        );
        assert_eq!(measure(&input), Some(7 * GIB));
    }
}
