use kuno::deployment::{Artifact, Host, Phase, Transaction};
use std::path::Path;

#[derive(Default)]
struct FakeHost {
    active: bool,
    gateway: bool,
    private: bool,
    events: Vec<&'static str>,
    fail_at: Option<&'static str>,
    crash_at: Option<&'static str>,
}

impl FakeHost {
    fn event(&mut self, name: &'static str) -> Result<(), &'static str> {
        self.events.push(name);
        if self.crash_at == Some(name) {
            self.crash_at = None;
            panic!("simulated setup process crash");
        }
        if self.fail_at == Some(name) {
            self.fail_at = None;
            return Err("simulated_service_failure");
        }
        Ok(())
    }
}

impl Host for FakeHost {
    fn preflight(&mut self) -> Result<(), &'static str> {
        self.event("preflight")
    }
    fn original_active(&mut self) -> Result<bool, &'static str> {
        self.event("original_active")?;
        Ok(self.active)
    }
    fn stop_original(&mut self) -> Result<(), &'static str> {
        self.event("stop_original")?;
        self.active = false;
        self.private = false;
        Ok(())
    }
    fn reload(&mut self) -> Result<(), &'static str> {
        self.event("reload")
    }
    fn start_private_upstream(&mut self) -> Result<(), &'static str> {
        self.event("start_private")?;
        self.private = true;
        self.active = true;
        Ok(())
    }
    fn verify_private_upstream(&mut self) -> Result<(), &'static str> {
        self.event("verify_private")?;
        if self.private {
            Ok(())
        } else {
            Err("private_upstream_not_ready")
        }
    }
    fn start_gateway(&mut self) -> Result<(), &'static str> {
        self.event("start_gateway")?;
        self.gateway = true;
        Ok(())
    }
    fn verify_gateway(&mut self) -> Result<(), &'static str> {
        self.event("verify_gateway")?;
        if self.gateway {
            Ok(())
        } else {
            Err("gateway_not_ready")
        }
    }
    fn stop_gateway(&mut self) -> Result<(), &'static str> {
        self.event("stop_gateway")?;
        self.gateway = false;
        Ok(())
    }
    fn restore_original(&mut self, active: bool) -> Result<(), &'static str> {
        self.event("restore_original")?;
        self.active = active;
        self.private = false;
        Ok(())
    }
    fn verify_original(&mut self, active: bool) -> Result<(), &'static str> {
        self.event("verify_original")?;
        assert_eq!(self.active, active);
        Ok(())
    }
    fn activate_gateway(&mut self) -> Result<(), &'static str> {
        self.event("activate_gateway")
    }
}

fn artifacts() -> Vec<Artifact> {
    vec![
        Artifact {
            path: "etc/systemd/system/ollama.service.d/zzzz-kuno.conf".into(),
            bytes: b"[Service]\nEnvironment=\"OLLAMA_HOST=127.0.0.1:11435\"\n".to_vec(),
            mode: 0o644,
        },
        Artifact {
            path: "etc/kuno/control-token".into(),
            bytes: b"test-credential".to_vec(),
            mode: 0o600,
        },
        Artifact {
            path: "usr/local/lib/kuno/kuno".into(),
            bytes: b"test executable".to_vec(),
            mode: 0o755,
        },
    ]
}

fn unchanged_user_override(root: &Path) -> std::path::PathBuf {
    let path = root.join("etc/systemd/system/ollama.service.d/zz-single-model.conf");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, "[Service]\nEnvironment=OLLAMA_MAX_LOADED_MODELS=1\n").unwrap();
    path
}

#[test]
fn successful_setup_repeat_and_restore_preserve_preexisting_settings() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let original = unchanged_user_override(root.path());
    let before = std::fs::read(&original).unwrap();
    let transaction = Transaction::open(root.path(), state.path()).unwrap();
    let mut host = FakeHost {
        active: true,
        ..Default::default()
    };
    transaction.apply(&artifacts(), &mut host).unwrap();
    assert_eq!(transaction.phase().unwrap(), Some(Phase::Complete));
    let calls = host.events.len();
    transaction.apply(&artifacts(), &mut host).unwrap();
    assert_eq!(
        &host.events[calls..],
        &["verify_private", "verify_gateway", "activate_gateway"]
    );
    let journal = std::fs::read_to_string(state.path().join("takeover.json")).unwrap();
    assert!(!journal.contains("test-credential"));
    transaction.restore(&mut host).unwrap();
    assert!(host.active && !host.private && !host.gateway);
    assert_eq!(std::fs::read(original).unwrap(), before);
    assert!(
        artifacts()
            .iter()
            .all(|a| !root.path().join(&a.path).exists())
    );
    let calls = host.events.len();
    transaction.restore(&mut host).unwrap();
    assert_eq!(host.events.len(), calls);
}

#[test]
fn each_service_failure_rolls_back_and_preserves_the_original_active_state() {
    for active in [false, true] {
        for failure in [
            "stop_original",
            "reload",
            "start_private",
            "verify_private",
            "start_gateway",
            "verify_gateway",
            "activate_gateway",
        ] {
            let root = tempfile::tempdir().unwrap();
            let state = tempfile::tempdir().unwrap();
            let original = unchanged_user_override(root.path());
            let before = std::fs::read(&original).unwrap();
            let transaction = Transaction::open(root.path(), state.path()).unwrap();
            let mut host = FakeHost {
                active,
                fail_at: Some(failure),
                ..Default::default()
            };
            assert_eq!(
                transaction.apply(&artifacts(), &mut host),
                Err("simulated_service_failure"),
                "{failure}"
            );
            assert_eq!(
                transaction.phase().unwrap(),
                Some(Phase::Restored),
                "{failure}"
            );
            assert_eq!(host.active, active);
            assert!(!host.gateway && !host.private);
            assert_eq!(std::fs::read(original).unwrap(), before);
            assert!(
                artifacts()
                    .iter()
                    .all(|a| !root.path().join(&a.path).exists())
            );
        }
    }
}

#[test]
fn interrupted_activation_can_resume_without_reinstalling_files() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let transaction = Transaction::open(root.path(), state.path()).unwrap();
    let mut host = FakeHost {
        active: true,
        crash_at: Some("activate_gateway"),
        ..Default::default()
    };
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || transaction.apply(&artifacts(), &mut host)
        ))
        .is_err()
    );
    assert_eq!(transaction.phase().unwrap(), Some(Phase::Complete));
    let calls = host.events.len();
    transaction.apply(&artifacts(), &mut host).unwrap();
    assert_eq!(
        &host.events[calls..],
        &["verify_private", "verify_gateway", "activate_gateway"]
    );
}

#[test]
fn interrupted_setup_can_be_restored_after_reopening_the_journal() {
    for crash in [
        "stop_original",
        "reload",
        "start_private",
        "verify_private",
        "start_gateway",
        "verify_gateway",
    ] {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut host = FakeHost {
            active: true,
            crash_at: Some(crash),
            ..Default::default()
        };
        {
            let transaction = Transaction::open(root.path(), state.path()).unwrap();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || transaction.apply(&artifacts(), &mut host)
                ))
                .is_err()
            );
        }
        let transaction = Transaction::open(root.path(), state.path()).unwrap();
        assert_eq!(
            transaction.apply(&artifacts(), &mut host),
            Err("interrupted_setup_restore_required")
        );
        transaction.restore(&mut host).unwrap();
        assert!(host.active && !host.gateway && !host.private);
    }
}

#[test]
fn edited_owned_files_block_restore_before_service_changes() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let transaction = Transaction::open(root.path(), state.path()).unwrap();
    let mut host = FakeHost {
        active: true,
        ..Default::default()
    };
    transaction.apply(&artifacts(), &mut host).unwrap();
    let changed = root.path().join(&artifacts()[0].path);
    std::fs::write(&changed, "later administrator edit").unwrap();
    let calls = host.events.len();
    assert_eq!(
        transaction.restore(&mut host),
        Err("owned_file_changed_restore_refused")
    );
    assert_eq!(host.events.len(), calls);
    assert_eq!(
        std::fs::read_to_string(changed).unwrap(),
        "later administrator edit"
    );
}

#[test]
fn occupied_paths_and_failed_preflight_cannot_change_a_service() {
    for occupied in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let state = tempfile::tempdir().unwrap();
        let mut host = FakeHost {
            active: true,
            fail_at: (!occupied).then_some("preflight"),
            ..Default::default()
        };
        if occupied {
            let path = root.path().join(&artifacts()[0].path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "belongs to someone else").unwrap();
        }
        let transaction = Transaction::open(root.path(), state.path()).unwrap();
        assert!(transaction.apply(&artifacts(), &mut host).is_err());
        assert_eq!(host.events, vec!["preflight"]);
        assert_eq!(transaction.phase().unwrap(), None);
    }
}

#[test]
fn failed_restore_remains_recoverable_and_preserves_complete_staging_files() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let transaction = Transaction::open(root.path(), state.path()).unwrap();
    let mut host = FakeHost {
        active: true,
        ..Default::default()
    };
    transaction.apply(&artifacts(), &mut host).unwrap();
    let path = root.path().join(&artifacts()[0].path);
    let stage = path.with_file_name("zzzz-kuno.conf.kuno-pending");
    std::fs::hard_link(&path, &stage).unwrap();
    host.fail_at = Some("restore_original");
    assert!(transaction.restore(&mut host).is_err());
    assert_eq!(transaction.phase().unwrap(), Some(Phase::Restoring));
    assert!(!stage.exists());
    transaction.restore(&mut host).unwrap();
    assert!(host.active && !host.gateway && !host.private);
}

#[test]
fn a_recovery_journal_cannot_be_reused_with_a_different_installation_root() {
    let root = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let mut host = FakeHost {
        active: true,
        ..Default::default()
    };
    {
        let transaction = Transaction::open(root.path(), state.path()).unwrap();
        transaction.apply(&artifacts(), &mut host).unwrap();
    }
    let transaction = Transaction::open(other.path(), state.path()).unwrap();
    let calls = host.events.len();
    assert_eq!(transaction.restore(&mut host), Err("setup_journal_invalid"));
    assert_eq!(host.events.len(), calls);
}

#[cfg(unix)]
#[test]
fn symlink_parents_and_traversal_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let state = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    std::os::unix::fs::symlink(outside.path(), root.path().join("etc")).unwrap();
    let transaction = Transaction::open(root.path(), state.path()).unwrap();
    let mut host = FakeHost::default();
    assert_eq!(
        transaction.apply(&artifacts(), &mut host),
        Err("unsafe_installation_path")
    );
    let invalid = [Artifact {
        path: "../outside".into(),
        bytes: vec![],
        mode: 0o600,
    }];
    assert_eq!(
        transaction.apply(&invalid, &mut host),
        Err("unsafe_installation_path")
    );
    assert!(host.events.is_empty());
}
