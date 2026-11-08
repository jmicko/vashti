//! Installer-owned Bash environment, separate from the Ollama service override.
//!
//! No shell is executed and no user startup contents are backed up on disk.
//! The protected journal contains only account identity and our own blocks.
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    net::SocketAddr,
    os::unix::{
        fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        io::{AsRawFd, FromRawFd},
    },
    path::{Path, PathBuf},
};

type Result<T> = std::result::Result<T, &'static str>;
const STATE: &str = "/var/lib/kuno-setup/cli-environment.json";
const LIMIT: u64 = 1024 * 1024;
const START: &str = "# >>> Kuno Ollama CLI >>>\n";
const END: &str = "# <<< Kuno Ollama CLI <<<\n";
const SSH_START: &str = "# >>> Kuno Ollama CLI noninteractive >>>\n";
const SSH_END: &str = "# <<< Kuno Ollama CLI noninteractive <<<\n";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Account {
    name: String,
    uid: u32,
    gid: u32,
    home: PathBuf,
    shell: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Owned {
    prefix: Option<String>,
    suffix: String,
    separator_newline: bool,
    original_exists: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Managed {
    name: String,
    owned: Owned,
    // Only present in a durable, incomplete update. These are owned blocks,
    // never a copy of a user's startup contents.
    prior: Option<Owned>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    version: u32,
    account: Account,
    host: String,
    pending: bool,
    #[serde(default)]
    removing: bool,
    files: Vec<Managed>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Snapshot {
    text: String,
    dev: u64,
    ino: u64,
    mode: u32,
    uid: u32,
    gid: u32,
}

struct Change {
    name: String,
    before: Option<Snapshot>,
    after: Option<String>,
}

/// Prepared without changing either shell files or services. The installer must
/// verify the upstream listener before `apply`, and hold its setup lock.
pub struct ShellExportPlan {
    state: PathBuf,
    previous: Option<Snapshot>,
    account: Account,
    home_identity: (u64, u64),
    changes: Vec<Change>,
    record: Option<Record>,
    pending_record: Option<Record>,
    host: String,
}

pub type ShellRestorePlan = ShellExportPlan;

#[derive(Debug)]
pub struct Activation {
    pub user: String,
    pub host: String,
    pub changed: bool,
}

impl Activation {
    /// Applies to the current Bash process; new terminals read managed startup.
    /// No sourcing of user-controlled files is requested from the root installer.
    pub fn activation_command(&self) -> String {
        if self.host.is_empty() {
            "unset OLLAMA_HOST".to_owned()
        } else {
            format!("export OLLAMA_HOST='{}'", self.host)
        }
    }
}

pub fn prepare(upstream: SocketAddr) -> Result<ShellExportPlan> {
    prepare_for_user(upstream, None)
}

pub fn prepare_for_user(upstream: SocketAddr, user: Option<&str>) -> Result<ShellExportPlan> {
    require_root()?;
    if !upstream.ip().is_loopback() || upstream.port() == 0 {
        return Err("cli_environment_private_endpoint_required");
    }
    let state = Path::new(STATE);
    let previous = read_state(state)?;
    let old = decode(previous.as_ref())?;
    let requested = user
        .map(str::to_owned)
        .or_else(|| std::env::var("SUDO_USER").ok());
    let account = if let Some(old) = &old {
        // The enrolled account always wins over a later administrator's SUDO_USER.
        if user.is_some_and(|name| name != old.account.name) {
            return Err("cli_environment_different_user_already_enrolled");
        }
        let account = resolve_account(&old.account.name)?;
        if account != old.account {
            return Err("cli_environment_account_changed");
        }
        account
    } else {
        resolve_account(
            requested
                .as_deref()
                .ok_or("cli_environment_user_required_use_sudo_or_explicit_user")?,
        )?
    };
    prepare_at(
        state,
        previous,
        old,
        account,
        Some(format!("http://{upstream}")),
    )
}

pub fn prepare_restore() -> Result<Option<ShellRestorePlan>> {
    require_root()?;
    let state = Path::new(STATE);
    let previous = read_state(state)?;
    let Some(old) = decode(previous.as_ref())? else {
        return Ok(None);
    };
    let account = resolve_account(&old.account.name)?;
    if account != old.account {
        return Err("cli_environment_account_changed");
    }
    prepare_at(state, previous, Some(old), account, None).map(Some)
}

fn require_root() -> Result<()> {
    if unsafe { libc::geteuid() } != 0 {
        return Err("root_required_run_this_command_with_sudo");
    }
    Ok(())
}

fn resolve_account(name: &str) -> Result<Account> {
    let name_c = std::ffi::CString::new(name).map_err(|_| "cli_environment_user_invalid")?;
    let mut buf = vec![0u8; 65536];
    let mut entry: libc::passwd = unsafe { std::mem::zeroed() };
    let mut found = std::ptr::null_mut();
    let status = unsafe {
        libc::getpwnam_r(
            name_c.as_ptr(),
            &mut entry,
            buf.as_mut_ptr().cast(),
            buf.len(),
            &mut found,
        )
    };
    if status != 0 || found.is_null() {
        return Err("cli_environment_user_not_found");
    }
    if entry.pw_uid == 0 {
        return Err("cli_environment_nonroot_user_required");
    }
    let read = |p: *const libc::c_char| -> Result<String> {
        if p.is_null() {
            return Err("cli_environment_account_invalid");
        }
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_str()
            .map(str::to_owned)
            .map_err(|_| "cli_environment_account_invalid")
    };
    let shell = PathBuf::from(read(entry.pw_shell)?);
    if shell.file_name().and_then(|s| s.to_str()) != Some("bash") {
        return Err("cli_environment_shell_unsupported_bash_required");
    }
    let home = PathBuf::from(read(entry.pw_dir)?);
    if !home.is_absolute() {
        return Err("cli_environment_home_invalid");
    }
    Ok(Account {
        name: read(entry.pw_name)?,
        uid: entry.pw_uid,
        gid: entry.pw_gid,
        home,
        shell,
    })
}

// Linux filesystem credentials are thread-local. Privileged setup never reads
// or creates startup files with root's filesystem access. Account-owned regular
// files only, O_NOFOLLOW, and an open canonical home directory constrain paths.
struct FsIdentity {
    uid: libc::uid_t,
    gid: libc::gid_t,
}
impl FsIdentity {
    fn enter(account: &Account) -> Result<Self> {
        let old_gid = unsafe { libc::setfsgid(account.gid) } as libc::gid_t;
        let old_uid = unsafe { libc::setfsuid(account.uid) } as libc::uid_t;
        let current = unsafe { libc::setfsuid(account.uid) } as libc::uid_t;
        let current_gid = unsafe { libc::setfsgid(account.gid) } as libc::gid_t;
        if current != account.uid || current_gid != account.gid {
            unsafe {
                libc::setfsuid(old_uid);
                libc::setfsgid(old_gid);
            }
            return Err("cli_environment_user_access_failed");
        }
        Ok(Self {
            uid: old_uid,
            gid: old_gid,
        })
    }
}
impl Drop for FsIdentity {
    fn drop(&mut self) {
        unsafe {
            libc::setfsuid(self.uid);
            libc::setfsgid(self.gid);
        }
    }
}

fn open_home(account: &Account) -> Result<File> {
    let _identity = FsIdentity::enter(account)?;
    let canonical =
        fs::canonicalize(&account.home).map_err(|_| "cli_environment_home_unavailable")?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(canonical)
        .map_err(|_| "cli_environment_home_unavailable")?;
    let meta = file
        .metadata()
        .map_err(|_| "cli_environment_home_unavailable")?;
    if meta.uid() != account.uid || meta.mode() & 0o022 != 0 {
        return Err("cli_environment_home_permissions_invalid");
    }
    Ok(file)
}

fn valid_name(name: &str) -> bool {
    matches!(
        name,
        ".bashrc" | ".bash_profile" | ".bash_login" | ".profile"
    )
}

fn read_startup(home: &File, name: &str, account: &Account) -> Result<Option<Snapshot>> {
    if !valid_name(name) {
        return Err("cli_environment_record_invalid");
    }
    let _identity = FsIdentity::enter(account)?;
    let name = std::ffi::CString::new(name).unwrap();
    let fd = unsafe {
        libc::openat(
            home.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NONBLOCK | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        if std::io::Error::last_os_error().kind() == std::io::ErrorKind::NotFound {
            return Ok(None);
        }
        return Err("cli_environment_startup_unavailable_or_symlink");
    }
    let file = unsafe { File::from_raw_fd(fd) };
    snapshot(file, account.uid, false).map(Some)
}

fn snapshot(file: File, uid: u32, protected: bool) -> Result<Snapshot> {
    let metadata = file
        .metadata()
        .map_err(|_| "cli_environment_file_unavailable")?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != uid
        || metadata.mode() & 0o7000 != 0
        || (protected && metadata.mode() & 0o077 != 0)
    {
        return Err("cli_environment_file_permissions_invalid");
    }
    if metadata.len() > LIMIT {
        return Err("cli_environment_file_too_large");
    }
    let mut text = String::new();
    file.take(LIMIT + 1)
        .read_to_string(&mut text)
        .map_err(|_| "cli_environment_file_encoding_invalid")?;
    if text.len() as u64 > LIMIT {
        return Err("cli_environment_file_too_large");
    }
    Ok(Snapshot {
        text,
        dev: metadata.dev(),
        ino: metadata.ino(),
        mode: metadata.mode() & 0o777,
        uid: metadata.uid(),
        gid: metadata.gid(),
    })
}

fn read_state(path: &Path) -> Result<Option<Snapshot>> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err("cli_environment_record_unavailable"),
    };
    snapshot(file, unsafe { libc::geteuid() }, true).map(Some)
}

fn decode(previous: Option<&Snapshot>) -> Result<Option<Record>> {
    let Some(previous) = previous else {
        return Ok(None);
    };
    let record: Record =
        serde_json::from_str(&previous.text).map_err(|_| "cli_environment_record_invalid")?;
    if record.version != 1
        || (record.removing && !record.pending)
        || record.files.len() != 2
        || record.files[0].name != ".bashrc"
        || !matches!(
            record.files[1].name.as_str(),
            ".bash_profile" | ".bash_login" | ".profile"
        )
        || (!record.pending && record.files.iter().any(|f| f.prior.is_some()))
    {
        return Err("cli_environment_record_invalid");
    }
    // Record text never becomes shell source without validation.
    let address = record
        .host
        .strip_prefix("http://")
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .ok_or("cli_environment_record_invalid")?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err("cli_environment_record_invalid");
    }
    for (i, file) in record.files.iter().enumerate() {
        validate_owned(&file.owned, i == 0)?;
        if !record.removing
            && (file.owned.prefix
                != if i == 0 {
                    Some(prefix(&record.host))
                } else {
                    None
                }
                || file.owned.suffix != suffix(&record.host))
        {
            return Err("cli_environment_record_invalid");
        }
        if let Some(prior) = &file.prior {
            validate_owned(prior, i == 0)?;
        }
    }
    Ok(Some(record))
}

fn validate_owned(owned: &Owned, bashrc: bool) -> Result<()> {
    let host = owned
        .suffix
        .strip_prefix(START)
        .and_then(|s| s.strip_prefix("export OLLAMA_HOST='"))
        .and_then(|s| s.strip_suffix(&format!("'\n{END}")))
        .ok_or("cli_environment_record_invalid")?;
    let addr = host
        .strip_prefix("http://")
        .and_then(|s| s.parse::<SocketAddr>().ok())
        .ok_or("cli_environment_record_invalid")?;
    if !addr.ip().is_loopback()
        || addr.port() == 0
        || owned.prefix != if bashrc { Some(prefix(host)) } else { None }
    {
        return Err("cli_environment_record_invalid");
    }
    Ok(())
}

fn prefix(host: &str) -> String {
    format!(
        "{SSH_START}case $- in\n  *i*) ;;\n  *) export OLLAMA_HOST='{host}' ;;\nesac\n{SSH_END}"
    )
}
fn suffix(host: &str) -> String {
    format!("{START}export OLLAMA_HOST='{host}'\n{END}")
}
fn markers(text: &str) -> bool {
    [START, END, SSH_START, SSH_END]
        .iter()
        .any(|s| text.contains(s.trim_end()))
}

fn remove_owned(text: &str, owned: &Owned) -> Result<String> {
    let mut base = text.to_owned();
    if let Some(prefix) = &owned.prefix {
        if base.matches(prefix).count() != 1 {
            return Err("cli_environment_owned_block_conflict");
        }
        base = base.replacen(prefix, "", 1);
    }
    if base.matches(&owned.suffix).count() != 1 {
        return Err("cli_environment_owned_block_conflict");
    }
    let start = base.find(&owned.suffix).unwrap();
    let end = start + owned.suffix.len();
    // Undo the newline inserted for an unterminated original file only while
    // this is still the tail; later user text must not become joined together.
    let remove_start = if owned.separator_newline
        && end == base.len()
        && start > 0
        && base.as_bytes()[start - 1] == b'\n'
    {
        start - 1
    } else {
        start
    };
    base.replace_range(remove_start..end, "");
    if markers(&base) {
        return Err("cli_environment_owned_block_conflict");
    }
    Ok(base)
}

fn recover_base(
    current: Option<&Snapshot>,
    managed: &Managed,
    pending: bool,
    removing: bool,
) -> Result<(String, bool, Option<Owned>)> {
    if let Some(current) = current {
        if let Ok(base) = remove_owned(&current.text, &managed.owned) {
            return Ok((
                base,
                managed.owned.original_exists,
                Some(managed.owned.clone()),
            ));
        }
        if pending {
            if let Some(prior) = &managed.prior {
                return remove_owned(&current.text, prior)
                    .map(|base| (base, prior.original_exists, Some(prior.clone())));
            }
            if !markers(&current.text) {
                return Ok((
                    current.text.clone(),
                    if removing {
                        managed.owned.original_exists
                    } else {
                        true
                    },
                    None,
                ));
            }
        }
    } else if pending && managed.prior.is_none() {
        return Ok((
            String::new(),
            if removing {
                managed.owned.original_exists
            } else {
                false
            },
            None,
        ));
    }
    Err("cli_environment_owned_block_conflict")
}

fn prepare_at(
    state: &Path,
    previous: Option<Snapshot>,
    old: Option<Record>,
    account: Account,
    host: Option<String>,
) -> Result<ShellExportPlan> {
    validate_state_parent(state)?;
    let home = open_home(&account)?;
    let meta = home
        .metadata()
        .map_err(|_| "cli_environment_home_unavailable")?;
    let mut rc = read_startup(&home, ".bashrc", &account)?;
    let login = if let Some(old) = &old {
        // A newly added higher-priority login file would bypass our enrollment.
        let selected = old.files[1].name.clone();
        for name in [".bash_profile", ".bash_login", ".profile"] {
            if name == selected {
                break;
            }
            if read_startup(&home, name, &account)?.is_some() {
                return Err("cli_environment_login_startup_changed");
            }
        }
        selected
    } else {
        let mut selected = ".profile";
        for name in [".bash_profile", ".bash_login", ".profile"] {
            if read_startup(&home, name, &account)?.is_some() {
                selected = name;
                break;
            }
        }
        selected.to_owned()
    };
    let login_snapshot = read_startup(&home, &login, &account)?;
    let mut changes = Vec::new();
    let mut files = Vec::new();
    for (i, (name, before)) in [(".bashrc".to_owned(), rc.take()), (login, login_snapshot)]
        .into_iter()
        .enumerate()
    {
        let (base, original_exists, prior) = if let Some(old) = &old {
            recover_base(before.as_ref(), &old.files[i], old.pending, old.removing)?
        } else {
            let text = before.as_ref().map_or("", |f| f.text.as_str());
            if markers(text) {
                return Err("cli_environment_unenrolled_block_conflict");
            }
            (text.to_owned(), before.is_some(), None)
        };
        let after = if let Some(host) = &host {
            let owned = Owned {
                prefix: if i == 0 { Some(prefix(host)) } else { None },
                suffix: suffix(host),
                separator_newline: !base.is_empty() && !base.ends_with('\n'),
                original_exists,
            };
            let text = format!(
                "{}{}{}{}",
                owned.prefix.as_deref().unwrap_or(""),
                base,
                if owned.separator_newline { "\n" } else { "" },
                owned.suffix
            );
            files.push(Managed {
                name: name.clone(),
                owned,
                prior,
            });
            Some(text)
        } else {
            let owned = prior.unwrap_or_else(|| old.as_ref().unwrap().files[i].owned.clone());
            files.push(Managed {
                name: name.clone(),
                owned,
                prior: None,
            });
            if !original_exists && base.is_empty() {
                None
            } else {
                Some(base)
            }
        };
        changes.push(Change {
            name,
            before,
            after,
        });
    }
    let mut pending_record = host.as_ref().map(|host| Record {
        version: 1,
        account: account.clone(),
        host: host.clone(),
        pending: true,
        removing: false,
        files: files.clone(),
    });
    if host.is_none() {
        pending_record = old.clone().map(|mut r| {
            r.pending = true;
            r.removing = true;
            r.files = files;
            r
        });
    }
    let record = host.as_ref().and(pending_record.clone()).map(|mut r| {
        r.pending = false;
        for f in &mut r.files {
            f.prior = None
        }
        r
    });
    Ok(ShellExportPlan {
        state: state.to_owned(),
        previous,
        account,
        home_identity: (meta.dev(), meta.ino()),
        changes,
        record,
        pending_record,
        host: host.unwrap_or_default(),
    })
}

impl ShellExportPlan {
    /// Caller verifies the real upstream endpoint and holds the setup lock.
    pub fn apply(self) -> Result<Activation> {
        if read_state(&self.state)? != self.previous {
            return Err("cli_environment_record_changed");
        }
        let home = open_home(&self.account)?;
        let meta = home
            .metadata()
            .map_err(|_| "cli_environment_home_unavailable")?;
        if (meta.dev(), meta.ino()) != self.home_identity {
            return Err("cli_environment_home_changed");
        }
        for change in &self.changes {
            if read_startup(&home, &change.name, &self.account)? != change.before {
                return Err("cli_environment_startup_changed");
            }
        }
        let mut changed = false;
        if let Some(pending) = &self.pending_record {
            write_state(&self.state, Some(pending))?;
        }
        let mut written = Vec::new();
        for (i, change) in self.changes.iter().enumerate() {
            if change.before.as_ref().map(|f| &f.text) == change.after.as_ref() {
                continue;
            }
            if read_startup(&home, &change.name, &self.account)? != change.before {
                self.rollback(&home, &written)?;
                return Err("cli_environment_startup_changed");
            }
            written.push(i);
            if let Err(error) = write_startup(&home, change, &self.account, change.after.as_deref())
            {
                self.rollback(&home, &written)?;
                return Err(error);
            }
            changed = true;
        }
        if let Err(error) = write_state(&self.state, self.record.as_ref()) {
            self.rollback(&home, &written)?;
            return Err(error);
        }
        Ok(Activation {
            user: self.account.name,
            host: self.host,
            changed,
        })
    }

    fn rollback(&self, home: &File, written: &[usize]) -> Result<()> {
        for &i in written.iter().rev() {
            let change = &self.changes[i];
            let current = read_startup(home, &change.name, &self.account)?;
            if current.as_ref().map(|f| &f.text) == change.before.as_ref().map(|f| &f.text) {
                continue;
            }
            if current.as_ref().map(|f| &f.text) != change.after.as_ref() {
                return Err("cli_environment_rollback_conflict");
            }
            write_startup_expected(
                home,
                change,
                &self.account,
                change.before.as_ref().map(|f| f.text.as_str()),
                current.as_ref(),
            )?;
        }
        write_state_text(&self.state, self.previous.as_ref().map(|f| f.text.as_str()))
    }
}

fn write_startup(
    home: &File,
    change: &Change,
    account: &Account,
    text: Option<&str>,
) -> Result<()> {
    write_startup_expected(home, change, account, text, change.before.as_ref())
}

#[cfg(test)]
thread_local! {
    static BEFORE_PUBLISH: std::cell::RefCell<Option<Box<dyn FnOnce()>>> = const { std::cell::RefCell::new(None) };
}

fn recheck_publication(
    home: &File,
    change: &Change,
    account: &Account,
    expected: Option<&Snapshot>,
) -> Result<()> {
    #[cfg(test)]
    BEFORE_PUBLISH.with(|hook| {
        let callback = hook.borrow_mut().take();
        if let Some(callback) = callback {
            callback();
        }
    });
    if read_startup(home, &change.name, account)?.as_ref() != expected {
        return Err("cli_environment_startup_changed");
    }
    Ok(())
}

fn write_startup_expected(
    home: &File,
    change: &Change,
    account: &Account,
    text: Option<&str>,
    expected: Option<&Snapshot>,
) -> Result<()> {
    let mut writer = account.clone();
    if let Some(before) = &change.before {
        writer.gid = before.gid;
    }
    let _identity = FsIdentity::enter(&writer)?;
    let name = std::ffi::CString::new(change.name.as_str()).unwrap();
    if let Some(text) = text {
        let temp_name = format!(
            ".kuno-cli-{}-{}",
            std::process::id(),
            change.name.trim_start_matches('.')
        );
        let temp = std::ffi::CString::new(temp_name).unwrap();
        let fd = unsafe {
            libc::openat(
                home.as_raw_fd(),
                temp.as_ptr(),
                libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                0o600,
            )
        };
        if fd < 0 {
            return Err("cli_environment_atomic_write_failed");
        }
        let mut file = unsafe { File::from_raw_fd(fd) };
        let result = (|| {
            file.write_all(text.as_bytes())
                .map_err(|_| "cli_environment_atomic_write_failed")?;
            let (mode, gid) = change
                .before
                .as_ref()
                .map_or((0o644, account.gid), |f| (f.mode, f.gid));
            if unsafe { libc::fchown(file.as_raw_fd(), account.uid, gid) } != 0 {
                return Err("cli_environment_owner_preservation_failed");
            }
            file.set_permissions(fs::Permissions::from_mode(mode))
                .map_err(|_| "cli_environment_mode_preservation_failed")?;
            file.sync_all()
                .map_err(|_| "cli_environment_atomic_write_failed")?;
            recheck_publication(home, change, account, expected)?;
            if unsafe {
                libc::renameat(
                    home.as_raw_fd(),
                    temp.as_ptr(),
                    home.as_raw_fd(),
                    name.as_ptr(),
                )
            } != 0
            {
                return Err("cli_environment_atomic_write_failed");
            }
            home.sync_all()
                .map_err(|_| "cli_environment_atomic_write_failed")
        })();
        if result.is_err() {
            unsafe {
                libc::unlinkat(home.as_raw_fd(), temp.as_ptr(), 0);
            }
        }
        result
    } else {
        recheck_publication(home, change, account, expected)?;
        if unsafe { libc::unlinkat(home.as_raw_fd(), name.as_ptr(), 0) } != 0 {
            return Err("cli_environment_atomic_write_failed");
        }
        home.sync_all()
            .map_err(|_| "cli_environment_atomic_write_failed")
    }
}

fn write_state(path: &Path, record: Option<&Record>) -> Result<()> {
    let text = record
        .map(serde_json::to_string_pretty)
        .transpose()
        .map_err(|_| "cli_environment_record_invalid")?;
    write_state_text(path, text.as_deref())
}

fn validate_state_parent(path: &Path) -> Result<()> {
    let parent = path.parent().ok_or("cli_environment_record_invalid")?;
    let metadata =
        fs::symlink_metadata(parent).map_err(|_| "cli_environment_record_directory_unavailable")?;
    if !metadata.is_dir()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o077 != 0
    {
        return Err("cli_environment_record_directory_permissions_invalid");
    }
    Ok(())
}

fn write_state_text(path: &Path, text: Option<&str>) -> Result<()> {
    validate_state_parent(path)?;
    let parent = path.parent().unwrap();
    if let Some(text) = text {
        let temp = parent.join(format!(".cli-environment-{}.tmp", std::process::id()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temp)
            .map_err(|_| "cli_environment_record_write_failed")?;
        let result = (|| {
            file.write_all(text.as_bytes())
                .map_err(|_| "cli_environment_record_write_failed")?;
            file.sync_all()
                .map_err(|_| "cli_environment_record_write_failed")?;
            fs::rename(&temp, path).map_err(|_| "cli_environment_record_write_failed")?;
            File::open(parent)
                .and_then(|f| f.sync_all())
                .map_err(|_| "cli_environment_record_write_failed")
        })();
        if result.is_err() {
            let _ = fs::remove_file(temp);
        }
        result
    } else {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(_) => return Err("cli_environment_record_write_failed"),
        }
        File::open(parent)
            .and_then(|f| f.sync_all())
            .map_err(|_| "cli_environment_record_write_failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    struct Fixture {
        _dir: tempfile::TempDir,
        state: PathBuf,
        account: Account,
    }
    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let home = dir.path().join("home");
            let setup = dir.path().join("setup");
            fs::create_dir(&home).unwrap();
            fs::create_dir(&setup).unwrap();
            fs::set_permissions(&home, fs::Permissions::from_mode(0o700)).unwrap();
            fs::set_permissions(&setup, fs::Permissions::from_mode(0o700)).unwrap();
            Self {
                state: setup.join("cli-environment.json"),
                account: Account {
                    name: "fixture".into(),
                    uid: unsafe { libc::geteuid() },
                    gid: unsafe { libc::getegid() },
                    home,
                    shell: "/bin/bash".into(),
                },
                _dir: dir,
            }
        }
        fn put(&self, name: &str, text: &str) {
            fs::write(self.account.home.join(name), text).unwrap();
        }
        fn text(&self, name: &str) -> String {
            fs::read_to_string(self.account.home.join(name)).unwrap()
        }
        fn plan(&self, port: Option<u16>) -> Result<ShellExportPlan> {
            let before = read_state(&self.state)?;
            let old = decode(before.as_ref())?;
            prepare_at(
                &self.state,
                before,
                old,
                self.account.clone(),
                port.map(|p| format!("http://127.0.0.1:{p}")),
            )
        }
        fn record(&self) -> Record {
            decode(read_state(&self.state).unwrap().as_ref())
                .unwrap()
                .unwrap()
        }
    }

    #[test]
    fn install_update_restore_preserves_previous_settings_and_unterminated_tail() {
        let f = Fixture::new();
        let rc = "# user settings\nexport OLLAMA_HOST=http://example.invalid:9000";
        let login = "# login\nexport PATH=/user/bin:$PATH\n";
        f.put(".bashrc", rc);
        f.put(".profile", login);
        fs::set_permissions(
            f.account.home.join(".bashrc"),
            fs::Permissions::from_mode(0o640),
        )
        .unwrap();
        let status = f.plan(Some(11433)).unwrap().apply().unwrap();
        assert_eq!(
            status.activation_command(),
            "export OLLAMA_HOST='http://127.0.0.1:11433'"
        );
        assert!(f.text(".bashrc").contains(rc));
        assert_eq!(
            fs::metadata(f.account.home.join(".bashrc")).unwrap().mode() & 0o777,
            0o640
        );
        assert_eq!(
            fs::metadata(f.account.home.join(".bashrc")).unwrap().uid(),
            f.account.uid
        );
        assert_eq!(fs::metadata(&f.state).unwrap().mode() & 0o777, 0o600);
        assert!(
            !fs::read_to_string(&f.state)
                .unwrap()
                .contains("example.invalid")
        );
        f.plan(Some(11435)).unwrap().apply().unwrap();
        assert!(!f.text(".bashrc").contains("127.0.0.1:11433"));
        assert!(f.text(".bashrc").contains("127.0.0.1:11435"));
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), rc);
        assert_eq!(f.text(".profile"), login);
        assert!(!f.state.exists());
    }

    #[test]
    fn missing_files_are_created_and_removed_without_adding_a_bash_profile() {
        let f = Fixture::new();
        f.plan(Some(11433)).unwrap().apply().unwrap();
        assert!(f.account.home.join(".profile").is_file());
        assert!(!f.account.home.join(".bash_profile").exists());
        f.plan(None).unwrap().apply().unwrap();
        assert!(!f.account.home.join(".bashrc").exists());
        assert!(!f.account.home.join(".profile").exists());
    }

    #[test]
    fn later_unrelated_edits_survive_update_and_restore() {
        let f = Fixture::new();
        f.put(".bashrc", "# original\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let rc = format!("{}# new setting\n", f.text(".bashrc"));
        f.put(".bashrc", &rc);
        f.plan(Some(11435)).unwrap().apply().unwrap();
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), "# original\n# new setting\n");
    }

    #[test]
    fn bash_interactive_and_noninteractive_guard_receive_selected_host() {
        let f = Fixture::new();
        f.put(".bashrc", "case $- in\n  *i*) ;;\n  *) return;;\nesac\nexport OLLAMA_HOST=http://old.invalid:11434\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        // Fixed fixture shell source only, never production user content.
        let noninteractive = std::process::Command::new("/bin/bash")
            .args([
                "--noprofile",
                "--norc",
                "-c",
                "source \"$HOME/.bashrc\"; printf %s \"$OLLAMA_HOST\"",
            ])
            .env_clear()
            .env("HOME", &f.account.home)
            .output()
            .unwrap();
        assert!(noninteractive.status.success());
        assert_eq!(noninteractive.stdout, b"http://127.0.0.1:11433");
        let interactive = std::process::Command::new("/bin/bash")
            .arg("--noprofile")
            .arg("--rcfile")
            .arg(f.account.home.join(".bashrc"))
            .args(["-ic", "printf %s \"$OLLAMA_HOST\""])
            .env_clear()
            .env("HOME", &f.account.home)
            .output()
            .unwrap();
        assert!(interactive.status.success());
        assert!(interactive.stdout.ends_with(b"http://127.0.0.1:11433"));
    }

    #[test]
    fn selects_existing_login_priority_and_detects_new_higher_priority_file() {
        let f = Fixture::new();
        f.put(".bash_login", "# existing login\n");
        f.put(".profile", "# untouched\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        assert_eq!(f.record().files[1].name, ".bash_login");
        assert_eq!(f.text(".profile"), "# untouched\n");
        f.put(".bash_profile", "# changed startup priority\n");
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_login_startup_changed")
        ));
    }

    #[test]
    fn refuses_symlinks_hardlinks_and_large_files_without_mutating() {
        let f = Fixture::new();
        let other = f.account.home.join("unrelated");
        fs::write(&other, "safe").unwrap();
        symlink(&other, f.account.home.join(".bashrc")).unwrap();
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_startup_unavailable_or_symlink")
        ));
        assert_eq!(fs::read_to_string(&other).unwrap(), "safe");
        fs::remove_file(f.account.home.join(".bashrc")).unwrap();
        fs::hard_link(&other, f.account.home.join(".bashrc")).unwrap();
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_file_permissions_invalid")
        ));
        fs::remove_file(f.account.home.join(".bashrc")).unwrap();
        f.put(".bashrc", &"x".repeat(LIMIT as usize + 1));
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_file_too_large")
        ));
        assert!(!f.state.exists());
    }

    #[test]
    fn canonical_home_symlink_is_supported_but_startup_links_are_not() {
        let mut f = Fixture::new();
        let link = f._dir.path().join("home-link");
        symlink(&f.account.home, &link).unwrap();
        f.account.home = link;
        f.plan(Some(11433)).unwrap().apply().unwrap();
        assert!(f.text(".bashrc").contains("11433"));
        f.plan(None).unwrap().apply().unwrap();
    }

    #[test]
    fn owned_block_edits_and_unenrolled_markers_are_conflicts() {
        let f = Fixture::new();
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let edited = f.text(".bashrc").replace("11433", "11499");
        f.put(".bashrc", &edited);
        assert!(matches!(
            f.plan(Some(11435)),
            Err("cli_environment_owned_block_conflict")
        ));
        assert!(matches!(
            f.plan(None),
            Err("cli_environment_owned_block_conflict")
        ));
        assert_eq!(f.text(".bashrc"), edited);
        let other = Fixture::new();
        other.put(".bashrc", START);
        assert!(matches!(
            other.plan(Some(11433)),
            Err("cli_environment_unenrolled_block_conflict")
        ));
    }

    #[test]
    fn rejects_changes_between_preflight_and_apply() {
        let f = Fixture::new();
        f.put(".bashrc", "# before\n");
        let plan = f.plan(Some(11433)).unwrap();
        f.put(".bashrc", "# after\n");
        assert_eq!(plan.apply().unwrap_err(), "cli_environment_startup_changed");
        assert_eq!(f.text(".bashrc"), "# after\n");
        assert!(!f.state.exists());
    }

    #[test]
    fn unfinished_enrollment_and_update_resume_from_known_owned_states() {
        let f = Fixture::new();
        f.put(".bashrc", "# before\n");
        let plan = f.plan(Some(11433)).unwrap();
        write_state(&f.state, plan.pending_record.as_ref()).unwrap();
        let home = open_home(&f.account).unwrap();
        write_startup(
            &home,
            &plan.changes[0],
            &f.account,
            plan.changes[0].after.as_deref(),
        )
        .unwrap();
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let plan = f.plan(Some(11435)).unwrap();
        write_state(&f.state, plan.pending_record.as_ref()).unwrap();
        write_startup(
            &home,
            &plan.changes[0],
            &f.account,
            plan.changes[0].after.as_deref(),
        )
        .unwrap();
        // A second endpoint update can normalize both old/new known blocks.
        f.plan(Some(11437)).unwrap().apply().unwrap();
        assert!(!f.record().pending);
        assert!(f.text(".profile").contains("11437"));
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), "# before\n");
    }

    #[test]
    fn unfinished_restore_is_idempotently_recoverable() {
        let f = Fixture::new();
        f.put(".bashrc", "# before\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let plan = f.plan(None).unwrap();
        write_state(&f.state, plan.pending_record.as_ref()).unwrap();
        let home = open_home(&f.account).unwrap();
        write_startup(
            &home,
            &plan.changes[0],
            &f.account,
            plan.changes[0].after.as_deref(),
        )
        .unwrap();
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), "# before\n");
        assert!(!f.state.exists());
        assert!(!f.account.home.join(".profile").exists());
    }

    #[test]
    fn ordinary_apply_failure_rolls_back_prior_files_and_metadata() {
        let f = Fixture::new();
        f.put(".bashrc", "# before\n");
        let plan = f.plan(Some(11433)).unwrap();
        // An existing atomic temporary name forces the second write to fail.
        let collision = f
            .account
            .home
            .join(format!(".kuno-cli-{}-profile", std::process::id()));
        fs::write(&collision, "untouched").unwrap();
        assert_eq!(
            plan.apply().unwrap_err(),
            "cli_environment_atomic_write_failed"
        );
        assert_eq!(f.text(".bashrc"), "# before\n");
        assert!(!f.account.home.join(".profile").exists());
        assert!(!f.state.exists());
        assert_eq!(fs::read_to_string(collision).unwrap(), "untouched");
    }

    #[test]
    fn metadata_cannot_inject_shell_source_or_escape_startup_directory() {
        let f = Fixture::new();
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let mut record = f.record();
        record.files[0].owned.suffix = "echo injected\n".into();
        write_state(&f.state, Some(&record)).unwrap();
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_record_invalid")
        ));
    }
    #[test]
    fn interrupted_update_can_be_restored_even_when_files_have_different_hosts() {
        let f = Fixture::new();
        f.put(".bashrc", "# original\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let plan = f.plan(Some(11435)).unwrap();
        write_state(&f.state, plan.pending_record.as_ref()).unwrap();
        let home = open_home(&f.account).unwrap();
        write_startup(
            &home,
            &plan.changes[0],
            &f.account,
            plan.changes[0].after.as_deref(),
        )
        .unwrap();
        let restore = f.plan(None).unwrap();
        write_state(&f.state, restore.pending_record.as_ref()).unwrap();
        write_startup(
            &home,
            &restore.changes[0],
            &f.account,
            restore.changes[0].after.as_deref(),
        )
        .unwrap();
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), "# original\n");
        assert!(!f.account.home.join(".profile").exists());
        assert!(!f.state.exists());
    }

    #[test]
    fn protects_record_directory_and_rejects_forged_file_paths() {
        let f = Fixture::new();
        fs::set_permissions(f.state.parent().unwrap(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(
            f.plan(Some(11433)),
            Err("cli_environment_record_directory_permissions_invalid")
        ));
        fs::set_permissions(f.state.parent().unwrap(), fs::Permissions::from_mode(0o700)).unwrap();
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let mut record = f.record();
        record.files[0].name = "../unrelated".into();
        write_state(&f.state, Some(&record)).unwrap();
        assert!(matches!(
            f.plan(None),
            Err("cli_environment_record_invalid")
        ));
    }

    #[test]
    fn update_failure_restores_original_enrollment_and_current_endpoint() {
        let f = Fixture::new();
        f.put(".bashrc", "# user\n");
        f.plan(Some(11433)).unwrap().apply().unwrap();
        let rc = f.text(".bashrc");
        let profile = f.text(".profile");
        let old_record = fs::read_to_string(&f.state).unwrap();
        let plan = f.plan(Some(11435)).unwrap();
        let collision = f
            .account
            .home
            .join(format!(".kuno-cli-{}-profile", std::process::id()));
        fs::write(collision, "untouched").unwrap();
        assert_eq!(
            plan.apply().unwrap_err(),
            "cli_environment_atomic_write_failed"
        );
        assert_eq!(f.text(".bashrc"), rc);
        assert_eq!(f.text(".profile"), profile);
        assert_eq!(fs::read_to_string(&f.state).unwrap(), old_record);
        assert!(!f.record().pending);
    }
    #[test]
    fn rejects_user_edit_during_temporary_write_without_overwriting_it() {
        let f = Fixture::new();
        f.put(".bashrc", "# before\n");
        let plan = f.plan(Some(11433)).unwrap();
        let path = f.account.home.join(".bashrc");
        BEFORE_PUBLISH.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                fs::write(path, "# user edit during publish\n").unwrap()
            }))
        });
        assert_eq!(
            plan.apply().unwrap_err(),
            "cli_environment_rollback_conflict"
        );
        assert_eq!(f.text(".bashrc"), "# user edit during publish\n");
        assert!(!f.account.home.join(".profile").exists());
        // Pending journal owns no user content and can preserve the new text.
        f.plan(Some(11433)).unwrap().apply().unwrap();
        f.plan(None).unwrap().apply().unwrap();
        assert_eq!(f.text(".bashrc"), "# user edit during publish\n");
    }
}
