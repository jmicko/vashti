use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};

const MAX_MODELS: usize = 4096;
const WEEK: u64 = 7 * 86400;

#[derive(Clone, Debug, Deserialize)]
pub struct CatalogModel {
    pub name: String,
    pub digest: String,
}
#[derive(Default, Deserialize)]
pub struct Catalog {
    pub models: Vec<CatalogModel>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelUsage {
    pub name: String,
    pub digest: String,
    pub real_calls: u64,
    pub last_real_use_unix: u64,
}

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UsageSnapshot {
    pub models: Vec<ModelUsage>,
    pub unattributed_calls: u64,
    pub persistence_failed: bool,
}

pub struct Usage {
    path: Option<PathBuf>,
    state: Mutex<UsageSnapshot>,
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

impl Catalog {
    pub fn resolve(&self, name: &str) -> Option<&CatalogModel> {
        if self.models.len() > MAX_MODELS {
            return None;
        }
        self.models.iter().find(|m| {
            (m.name == name || m.name == format!("{name}:latest"))
                && !m.name.is_empty()
                && m.name.len() <= 256
                && m.digest.len() == 64
                && m.digest.bytes().all(|b| b.is_ascii_hexdigit())
        })
    }
}

impl Usage {
    pub fn open(directory: Option<&Path>) -> Result<Self, &'static str> {
        let path = directory.map(|d| d.join("usage.json"));
        let state = match &path {
            Some(path) if path.try_exists().map_err(|_| "usage_read_failed")? => {
                let metadata = fs::symlink_metadata(path).map_err(|_| "usage_read_failed")?;
                if !metadata.is_file() {
                    return Err("usage_path_not_regular");
                }
                if metadata.len() > 2 * 1024 * 1024 {
                    return Err("usage_file_too_large");
                }
                let data = fs::read(path).map_err(|_| "usage_read_failed")?;
                let state: UsageSnapshot =
                    serde_json::from_slice(&data).map_err(|_| "usage_file_invalid")?;
                if state.models.len() > MAX_MODELS
                    || state.models.iter().any(|m| {
                        m.name.len() > 256
                            || m.digest.len() != 64
                            || !m.digest.bytes().all(|b| b.is_ascii_hexdigit())
                    })
                {
                    return Err("usage_file_invalid");
                }
                state
            }
            _ => UsageSnapshot::default(),
        };
        Ok(Self {
            path,
            state: Mutex::new(state),
        })
    }

    pub fn snapshot(&self) -> UsageSnapshot {
        self.state.lock().unwrap().clone()
    }

    pub fn record(&self, model: Option<&CatalogModel>, timestamp: u64) {
        let mut state = self.state.lock().unwrap();
        if let Some(model) = model {
            let index = state.models.iter().position(|m| m.digest == model.digest);
            if let Some(index) = index {
                let row = &mut state.models[index];
                row.name.clone_from(&model.name);
                row.real_calls = row.real_calls.saturating_add(1);
                row.last_real_use_unix = row.last_real_use_unix.max(timestamp);
            } else {
                if state.models.len() >= MAX_MODELS {
                    let oldest = state
                        .models
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, m)| m.last_real_use_unix)
                        .unwrap()
                        .0;
                    state.models.swap_remove(oldest);
                }
                state.models.push(ModelUsage {
                    name: model.name.clone(),
                    digest: model.digest.clone(),
                    real_calls: 1,
                    last_real_use_unix: timestamp,
                });
            }
        } else {
            state.unattributed_calls = state.unattributed_calls.saturating_add(1);
        }
        if let Some(path) = &self.path {
            state.persistence_failed = false;
            if crate::storage::write_json(path, &*state).is_err() {
                state.persistence_failed = true;
            }
        }
    }

    /// Tests/probes never call `record`. Eligibility also requires that the
    /// same artifact still exists in the current catalog; expiry deletes no files.
    pub fn eligible(&self, catalog: &Catalog, timestamp: u64) -> Vec<ModelUsage> {
        let current: BTreeMap<_, _> = catalog
            .models
            .iter()
            .map(|m| (&m.digest, &m.name))
            .collect();
        self.snapshot()
            .models
            .into_iter()
            .filter(|m| {
                m.last_real_use_unix <= timestamp
                    && timestamp - m.last_real_use_unix <= WEEK
                    && current.contains_key(&m.digest)
            })
            .collect()
    }
}
