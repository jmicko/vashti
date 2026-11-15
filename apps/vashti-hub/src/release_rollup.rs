use std::collections::HashMap;

use serde::Serialize;
use sqlx::{Row, SqliteConnection};

use super::{AppError, PrereleaseRollup, merge_release_notes, release_version};

#[derive(Debug, Serialize)]
pub struct RetainedRelease {
    pub version: String,
    pub targets: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct RollupPlan {
    pub version: String,
    pub incorporated_versions: Vec<String>,
    pub retained_releases: Vec<RetainedRelease>,
    pub reclaimable_bytes: i64,
    #[serde(skip)]
    notes: Option<String>,
    #[serde(skip)]
    incorporated: Vec<PrereleaseRollup>,
    #[serde(skip)]
    latest_stable: Option<semver::Version>,
}

struct Release {
    version: semver::Version,
    label: String,
    status: String,
    notes: Option<String>,
    artifacts: Vec<Artifact>,
}

struct Artifact {
    target: String,
    bytes: i64,
    signed: bool,
}

pub async fn preview(db: &mut SqliteConnection, target: &str) -> Result<RollupPlan, AppError> {
    let rows = sqlx::query(
        "SELECT r.version, r.release_status, r.notes, a.target, a.size_bytes, a.update_signature
         FROM releases r LEFT JOIN release_artifacts a ON a.release_version = r.version",
    )
    .fetch_all(db)
    .await?;
    let mut releases = HashMap::<String, Release>::new();
    for row in rows {
        let label: String = row.try_get("version")?;
        if !releases.contains_key(&label) {
            releases.insert(
                label.clone(),
                Release {
                    version: release_version(&label)?,
                    label: label.clone(),
                    status: row.try_get("release_status")?,
                    notes: row.try_get("notes")?,
                    artifacts: Vec::new(),
                },
            );
        }
        if let Some(target) = row.try_get::<Option<String>, _>("target")? {
            releases
                .get_mut(&label)
                .expect("inserted release")
                .artifacts
                .push(Artifact {
                    target,
                    bytes: row.try_get("size_bytes")?,
                    signed: row
                        .try_get::<Option<String>, _>("update_signature")?
                        .is_some(),
                });
        }
    }
    let selected = releases
        .get(target)
        .ok_or_else(|| AppError::not_found("release_not_found", "Release not found"))?;
    if selected.status != "prerelease" {
        return Err(AppError::conflict(
            "release_already_promoted",
            "Only a prerelease can incorporate earlier prereleases",
        ));
    }
    let stable = releases
        .values()
        .filter(|r| r.status == "released" && r.version.pre.is_empty())
        .map(|r| &r.version)
        .max();

    // Keep the newest build of every target. A Vashti-only release must not remove
    // the only current Kuno/runtime (or a platform not included in that release).
    let mut newest_targets = HashMap::<(&str, bool), &semver::Version>::new();
    for release in releases.values() {
        for artifact in &release.artifacts {
            for signed_only in [false, true] {
                if signed_only && !artifact.signed {
                    continue;
                }
                let newest = newest_targets
                    .entry((&artifact.target, signed_only))
                    .or_insert(&release.version);
                if release.version > **newest {
                    *newest = &release.version;
                }
            }
        }
    }
    let mut candidates = releases
        .values()
        .filter(|release| {
            release.status == "prerelease"
                && release.version < selected.version
                && stable.is_none_or(|stable| release.version > *stable)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| b.version.cmp(&a.version));
    let mut plan = RollupPlan {
        version: target.to_owned(),
        incorporated_versions: Vec::new(),
        retained_releases: Vec::new(),
        reclaimable_bytes: 0,
        notes: selected.notes.clone(),
        incorporated: Vec::new(),
        latest_stable: stable.cloned(),
    };
    for release in candidates {
        let mut needed = release
            .artifacts
            .iter()
            .filter(|artifact| {
                [false, true].into_iter().any(|signed_only| {
                    newest_targets
                        .get(&(artifact.target.as_str(), signed_only))
                        .is_some_and(|newest| **newest == release.version)
                })
            })
            .map(|artifact| artifact.target.clone())
            .collect::<Vec<_>>();
        needed.sort();
        if !needed.is_empty() {
            plan.retained_releases.push(RetainedRelease {
                version: release.label.clone(),
                targets: needed,
            });
            continue;
        }
        plan.reclaimable_bytes += release
            .artifacts
            .iter()
            .map(|artifact| artifact.bytes)
            .sum::<i64>();
        plan.incorporated_versions.push(release.label.clone());
        plan.incorporated.push(PrereleaseRollup {
            version: release.label.clone(),
            notes: release.notes.clone(),
        });
    }
    Ok(plan)
}

pub fn require_confirmed(plan: &RollupPlan, versions: &[String]) -> Result<(), AppError> {
    let mut confirmed = versions.to_vec();
    let mut expected = plan.incorporated_versions.clone();
    confirmed.sort();
    expected.sort();
    if confirmed != expected {
        return Err(AppError::conflict(
            "rollup_changed",
            "Available releases changed. Reopen the confirmation before continuing.",
        ));
    }
    Ok(())
}

pub async fn apply(
    db: &mut SqliteConnection,
    plan: &RollupPlan,
    promote: bool,
) -> Result<(), AppError> {
    let notes = merge_release_notes(&plan.version, plan.notes.as_deref(), &plan.incorporated);
    sqlx::query("UPDATE releases SET notes = ?, release_status = ? WHERE version = ? AND release_status = 'prerelease'")
        .bind(notes).bind(if promote { "released" } else { "prerelease" }).bind(&plan.version)
        .execute(&mut *db).await?;
    for version in &plan.incorporated_versions {
        sqlx::query("DELETE FROM download_events WHERE release_version = ?")
            .bind(version)
            .execute(&mut *db)
            .await?;
        sqlx::query("DELETE FROM release_artifacts WHERE release_version = ?")
            .bind(version)
            .execute(&mut *db)
            .await?;
        sqlx::query("DELETE FROM releases WHERE version = ? AND release_status = 'prerelease'")
            .bind(version)
            .execute(&mut *db)
            .await?;
    }
    Ok(())
}

impl RollupPlan {
    pub fn require_promotion(&self) -> Result<(), AppError> {
        super::require_stable_version(&self.version)?;
        let target = release_version(&self.version)?;
        if self
            .latest_stable
            .as_ref()
            .is_some_and(|latest| target <= *latest)
        {
            return Err(AppError::conflict(
                "release_not_newer",
                "A promoted release must be newer than the current stable release",
            ));
        }
        Ok(())
    }

    pub fn preserve_previous(&mut self) {
        self.incorporated.clear();
        self.incorporated_versions.clear();
        self.reclaimable_bytes = 0;
    }
}
