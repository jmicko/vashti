//! Content-free budgets for the installed automatic-recovery supervisor.
//! Only bounded counters, host boot identity, and allowlisted failure
//! categories are stored: no prompt, image, response, summary, client
//! identifier, request path, or backend text is persisted or logged.
use crate::storage;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

pub const STATE_FILE: &str = "/var/lib/kuno-setup/auto-recovery.json";
const MAX_STATE_BYTES: u64 = 4096;
const MAX_ATTEMPTS: u32 = 5;
const MAX_BOOT_ATTEMPTS: u32 = 5;
const BACKOFF_SECONDS: [u64; 6] = [30, 60, 120, 240, 480, 1800];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub version: u32,
    pub mode: String,
    pub attempts: u32,
    pub attempts_this_boot: u32,
    pub consecutive_failures: u32,
    pub last_failure: Option<String>,
    pub next_eligible_unix: u64,
    pub boot_id: String,
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn boot_id() -> Result<String, &'static str> {
    let value = fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map_err(|_| "recovery_boot_identity_unavailable")?;
    let value = value.trim().to_owned();
    if valid_boot_id(&value) {
        Ok(value)
    } else {
        Err("recovery_boot_identity_invalid")
    }
}

pub fn load(path: &Path) -> Result<Option<State>, &'static str> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("recovery_state_unreadable"),
    };
    if !metadata.file_type().is_file() || metadata.len() > MAX_STATE_BYTES {
        return Err("recovery_state_invalid");
    }
    let bytes = fs::read(path).map_err(|_| "recovery_state_unreadable")?;
    let state: State = serde_json::from_slice(&bytes).map_err(|_| "recovery_state_invalid")?;
    state.validate()?;
    Ok(Some(state))
}

pub fn load_or_init(path: &Path) -> Result<State, &'static str> {
    let boot = boot_id()?;
    if let Some(mut state) = load(path)? {
        state.note_boot(&boot);
        return Ok(state);
    }
    let state = fresh(&boot);
    save(path, &state)?;
    Ok(state)
}

/// Explicit operator recovery must stay available even when the budget record
/// cannot be interpreted. It holds only counters and fixed failure codes, so a
/// manual command replaces an unusable record instead of leaving the host
/// without a recovery path. The automatic path still refuses changed state.
pub fn reset_for_manual(path: &Path) -> Result<State, &'static str> {
    let boot = boot_id()?;
    let mut state = match load(path) {
        Ok(Some(state)) => state,
        Ok(None) => fresh(&boot),
        Err(_) => {
            clear(path)?;
            fresh(&boot)
        }
    };
    state.note_boot(&boot);
    state.reset_for_manual();
    save(path, &state)?;
    Ok(state)
}

pub fn clear(path: &Path) -> Result<(), &'static str> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err("recovery_state_remove_failed"),
    }
}

fn fresh(boot: &str) -> State {
    State {
        version: 1,
        mode: "auto".into(),
        attempts: 0,
        attempts_this_boot: 0,
        consecutive_failures: 0,
        last_failure: None,
        next_eligible_unix: 0,
        boot_id: boot.into(),
    }
}

pub fn save(path: &Path, state: &State) -> Result<(), &'static str> {
    state.validate()?;
    if let Ok(metadata) = fs::symlink_metadata(path)
        && (!metadata.file_type().is_file() || metadata.len() > MAX_STATE_BYTES)
    {
        return Err("recovery_state_invalid");
    }
    storage::write_json(path, state).map_err(|_| "recovery_state_write_failed")
}

impl State {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.version != 1
            || !matches!(self.mode.as_str(), "auto" | "backoff" | "manual_required")
            || self.attempts > 1000
            || self.attempts_this_boot > 1000
            || self.consecutive_failures > 1000
            || !valid_boot_id(&self.boot_id)
            || self
                .last_failure
                .as_deref()
                .is_some_and(|value| !valid_failure(value))
        {
            return Err("recovery_state_invalid");
        }
        Ok(())
    }

    /// A changed boot identity proves that no process from the previous boot
    /// survived, which is the same proof the in-flight journal requires before
    /// it clears an uncertainty marker across a reboot.
    pub fn note_boot(&mut self, boot: &str) {
        if self.boot_id != boot {
            *self = fresh(boot);
        }
    }

    pub fn should_attempt(&self, now: u64) -> bool {
        self.mode != "manual_required"
            && self.next_eligible_unix <= now
            && self.attempts < MAX_ATTEMPTS
            && self.attempts_this_boot < MAX_BOOT_ATTEMPTS
    }

    pub fn record_attempt(&mut self, now: u64) -> Result<(), &'static str> {
        if !self.should_attempt(now) {
            return Err("recovery_budget_exhausted");
        }
        self.attempts = self.attempts.saturating_add(1);
        self.attempts_this_boot = self.attempts_this_boot.saturating_add(1);
        self.mode = "backoff".into();
        self.next_eligible_unix = now.saturating_add(BACKOFF_SECONDS[0]);
        Ok(())
    }

    pub fn record_success(&mut self) {
        self.mode = "auto".into();
        self.attempts = 0;
        self.consecutive_failures = 0;
        self.last_failure = None;
        self.next_eligible_unix = 0;
    }

    pub fn record_failure(&mut self, code: &'static str, now: u64) {
        self.consecutive_failures = self.consecutive_failures.saturating_add(1);
        self.last_failure = Some(classify(code).into());
        self.next_eligible_unix = now.saturating_add(backoff_seconds(self.consecutive_failures));
        if self.attempts >= MAX_ATTEMPTS || self.attempts_this_boot >= MAX_BOOT_ATTEMPTS {
            self.mode = "manual_required".into();
        } else {
            self.mode = "backoff".into();
        }
    }

    pub fn reset_for_manual(&mut self) {
        self.attempts_this_boot = 0;
        self.record_success();
    }
}

pub fn backoff_seconds(failures: u32) -> u64 {
    BACKOFF_SECONDS
        .get(failures.saturating_sub(1) as usize)
        .copied()
        .unwrap_or(*BACKOFF_SECONDS.last().unwrap_or(&1800))
}

/// Maps a fixed internal code to a small allowlist, so neither stored state nor
/// logs can carry an unexpected value.
pub fn classify(code: &'static str) -> &'static str {
    match code {
        "gateway_service_changed_recovery_refused" => "gateway_changed",
        "service_cgroup_invalid" | "service_cgroup_not_empty" => "cgroup_unproven",
        "recovery_marker_changed" | "recovery_marker_unreadable" => "marker_changed",
        "managed_state_directory_changed" | "managed_worker_state_directory_changed" => {
            "state_directory_changed"
        }
        "recovery_files_changed" | "setup_journal_invalid" => "installed_state_changed",
        "activation_marker_write_failed" | "gateway_not_ready" => "activation_failed",
        "upstream_not_ready_at_startup" | "gateway_upstream_mismatch" => "upstream_not_ready",
        _ => "recovery_step_failed",
    }
}

fn valid_failure(value: &str) -> bool {
    matches!(
        value,
        "gateway_changed"
            | "cgroup_unproven"
            | "marker_changed"
            | "state_directory_changed"
            | "installed_state_changed"
            | "activation_failed"
            | "upstream_not_ready"
            | "recovery_step_failed"
    )
}

fn valid_boot_id(value: &str) -> bool {
    value.len() == 36
        && value.bytes().enumerate().all(|(index, byte)| {
            if matches!(index, 8 | 13 | 18 | 23) {
                byte == b'-'
            } else {
                byte.is_ascii_hexdigit()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> State {
        State {
            version: 1,
            mode: "auto".into(),
            attempts: 0,
            attempts_this_boot: 0,
            consecutive_failures: 0,
            last_failure: None,
            next_eligible_unix: 0,
            boot_id: "11111111-1111-4111-8111-111111111111".into(),
        }
    }

    #[test]
    fn recovery_budget_and_backoff_are_bounded() {
        let mut state = state();
        for attempt in 0..MAX_ATTEMPTS {
            let now = state.next_eligible_unix;
            state.record_attempt(now).unwrap();
            state.record_failure("recovery_step_failed", now);
            if attempt + 1 < MAX_ATTEMPTS {
                assert!(!state.should_attempt(now));
            }
        }
        assert_eq!(state.mode, "manual_required");
        assert!(!state.should_attempt(u64::MAX));
        assert_eq!(backoff_seconds(1), 30);
        assert_eq!(backoff_seconds(5), 480);
        assert_eq!(backoff_seconds(6), 1800);
    }

    #[test]
    fn successful_recovery_resets_the_episode_but_not_the_boot_budget() {
        let mut state = state();
        for _ in 0..MAX_BOOT_ATTEMPTS {
            state.record_attempt(0).unwrap();
            state.record_success();
        }
        assert_eq!(state.attempts, 0);
        assert_eq!(state.attempts_this_boot, MAX_BOOT_ATTEMPTS);
        assert!(!state.should_attempt(u64::MAX));
        state.reset_for_manual();
        assert!(state.should_attempt(0));
    }

    #[test]
    fn records_reject_unknown_or_unbounded_values() {
        let mut value = state();
        value.mode = "anything".into();
        assert_eq!(value.validate(), Err("recovery_state_invalid"));
        value = state();
        value.last_failure = Some("prompt text".into());
        assert_eq!(value.validate(), Err("recovery_state_invalid"));
        value = state();
        value.attempts = 1001;
        assert_eq!(value.validate(), Err("recovery_state_invalid"));
    }

    #[test]
    fn state_round_trip_is_content_free_and_symlink_safe() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let mut value = state();
        value.record_attempt(0).unwrap();
        save(&path, &value).unwrap();
        let loaded = load(&path).unwrap().unwrap();
        assert_eq!(loaded.attempts, 1);
        let target = directory.path().join("target");
        fs::write(&target, b"").unwrap();
        fs::remove_file(&path).unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(load(&path).is_err());
        // Explicit recovery replaces the unusable record instead of refusing.
        let reset = reset_for_manual(&path).unwrap();
        assert!(reset.should_attempt(0));
        assert!(fs::symlink_metadata(&path).unwrap().file_type().is_file());
        assert_eq!(fs::read(&target).unwrap(), b"");
    }

    #[test]
    fn a_changed_boot_restores_the_budget_and_manual_recovery_replaces_changed_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("state.json");
        let mut value = state();
        for _ in 0..MAX_BOOT_ATTEMPTS {
            let now = value.next_eligible_unix;
            value.record_attempt(now).unwrap();
            value.record_failure("recovery_step_failed", now);
        }
        assert_eq!(value.mode, "manual_required");
        save(&path, &value).unwrap();
        // A reboot proves no previous process survived, so the episode budget
        // starts again rather than leaving the host without supervision.
        let mut rebooted = load(&path).unwrap().unwrap();
        rebooted.note_boot("22222222-2222-4222-8222-222222222222");
        assert_eq!(rebooted, fresh("22222222-2222-4222-8222-222222222222"));
        assert!(rebooted.should_attempt(0));
        // Explicit recovery is available even when the record is unusable.
        let mut tampered = state();
        tampered.last_failure = Some("prompt text".into());
        fs::write(&path, serde_json::to_vec(&tampered).unwrap()).unwrap();
        assert_eq!(load(&path), Err("recovery_state_invalid"));
        let reset = reset_for_manual(&path).unwrap();
        assert!(reset.should_attempt(0));
        assert_eq!(load(&path).unwrap().unwrap(), reset);
    }

    #[test]
    fn only_allowlisted_failure_categories_are_stored_or_printed() {
        assert_eq!(classify("worker_reap_failed"), "recovery_step_failed");
        assert_eq!(
            classify("gateway_service_changed_recovery_refused"),
            "gateway_changed"
        );
        for code in [
            "recovery_step_failed",
            "gateway_changed",
            "cgroup_unproven",
            "marker_changed",
            "state_directory_changed",
            "installed_state_changed",
            "activation_failed",
            "upstream_not_ready",
        ] {
            assert!(valid_failure(classify(code)));
        }
    }
}
