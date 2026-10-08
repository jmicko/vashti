//! Numeric-only diagnostics for compiled public laboratory fixtures.
//! No argv, paths, prompts, model output or residency selections are recorded.
use std::collections::BTreeMap;
pub(crate) fn snapshot() -> Option<BTreeMap<&'static str, u64>> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let mut result = parse_stat(&stat)?;
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    for (key, name) in [
        ("read_bytes:", "read_bytes"),
        ("write_bytes:", "write_bytes"),
    ] {
        let value = io
            .lines()
            .find_map(|line| line.strip_prefix(key))?
            .trim()
            .parse()
            .ok()?;
        result.insert(name, value);
    }
    Some(result)
}
fn parse_stat(stat: &str) -> Option<BTreeMap<&'static str, u64>> {
    let tail = stat
        .rsplit_once(") ")?
        .1
        .split_whitespace()
        .collect::<Vec<_>>();
    let mut result = BTreeMap::new();
    // Tail starts at Linux stat field3 (state). Times are clock ticks; no
    // assumed HZ or wall-time interpretation is attached to these counters.
    for (index, name) in [
        (7, "minor_faults"),
        (9, "major_faults"),
        (11, "user_ticks"),
        (12, "system_ticks"),
    ] {
        result.insert(name, tail.get(index)?.parse().ok()?);
    }
    Some(result)
}
pub(crate) fn delta(
    before: Option<BTreeMap<&'static str, u64>>,
    after: Option<BTreeMap<&'static str, u64>>,
) -> Option<BTreeMap<&'static str, u64>> {
    let before = before?;
    let after = after?;
    before
        .into_iter()
        .map(|(key, n)| Some((key, after.get(key)?.checked_sub(n)?)))
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_only_numeric_counters_and_rejects_missing_or_decreasing() {
        let raw = "123 (public fixture ) name) R 1 2 3 4 5 6 7 8 9 10 11 12 13";
        let values = parse_stat(raw).unwrap();
        assert_eq!(values.get("minor_faults"), Some(&7));
        assert_eq!(values.get("major_faults"), Some(&9));
        assert_eq!(values.get("user_ticks"), Some(&11));
        assert_eq!(values.get("system_ticks"), Some(&12));
        assert_eq!(
            delta(Some(values.clone()), Some(values.clone()))
                .unwrap()
                .values()
                .sum::<u64>(),
            0
        );
        assert!(delta(Some(values.clone()), Some(BTreeMap::new())).is_none());
        assert!(delta(None, Some(values)).is_none());
        assert!(parse_stat("123 (fixture) R").is_none());
    }
}
