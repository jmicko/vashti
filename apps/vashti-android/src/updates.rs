use serde::{Deserialize, Serialize};
use vashti_update_manifest::{SCHEMA_VERSION, SignedArtifact, verify_release_signature};

const HUB: &str = "https://vashti.chat";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    Stable,
    Prerelease,
}

impl Channel {
    fn name(self) -> &'static str {
        match self {
            Self::Stable => "stable",
            Self::Prerelease => "prerelease",
        }
    }
}

#[derive(Deserialize)]
struct Manifest {
    schema_version: u32,
    channel: String,
    release_status: String,
    artifact: SignedArtifact,
    download_url: String,
    signature: String,
}

#[derive(Debug, Serialize)]
pub struct AppUpdate {
    version: String,
}

fn download_path(version: &str) -> Result<String, String> {
    let parsed = semver::Version::parse(version.strip_prefix('v').unwrap_or(version))
        .map_err(|_| "Invalid app version".to_string())?;
    // Hub accepts SemVer prereleases but not build metadata.
    if !parsed.build.is_empty() {
        return Err("Invalid app version".into());
    }
    Ok(format!("/releases/v{parsed}/vashti-android.apk"))
}

fn validate_manifest(
    manifest: Manifest,
    channel: Channel,
    current: &str,
) -> Result<Option<AppUpdate>, String> {
    if manifest.schema_version != SCHEMA_VERSION
        || manifest.channel != channel.name()
        || !matches!(manifest.release_status.as_str(), "released" | "prerelease")
        || (channel == Channel::Stable && manifest.release_status != "released")
        || manifest.artifact.target != "android-universal"
        || manifest.artifact.filename != "vashti-android.apk"
        || manifest.download_url != download_path(&manifest.artifact.version)?
    {
        return Err("Invalid Android update manifest".into());
    }
    verify_release_signature(&manifest.artifact, &manifest.signature)
        .map_err(|_| "Android update signature verification failed".to_string())?;
    let remote = semver::Version::parse(manifest.artifact.version.trim_start_matches('v'))
        .map_err(|error| error.to_string())?;
    let installed = semver::Version::parse(current).map_err(|error| error.to_string())?;
    Ok((remote > installed).then_some(AppUpdate {
        version: manifest.artifact.version,
    }))
}

#[cfg(target_os = "android")]
#[tauri::command]
pub async fn native_check_app_update(channel: Channel) -> Result<Option<AppUpdate>, String> {
    use std::time::Duration;

    // Deliberately separate from the authenticated server transport: no session cookies,
    // credentials, or server-selected URLs are sent to the public release service.
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| error.to_string())?;
    let mut response = client
        .get(format!(
            "{HUB}/api/updates/{}/android-universal",
            channel.name()
        ))
        .send()
        .await
        .map_err(|_| "Could not reach the app update service".to_string())?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !response.status().is_success() {
        return Err("The app update service is unavailable".into());
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        if body.len() + chunk.len() > 256 * 1024 {
            return Err("App update manifest is too large".into());
        }
        body.extend_from_slice(&chunk);
    }
    let manifest =
        serde_json::from_slice(&body).map_err(|_| "Invalid app update response".to_string())?;
    validate_manifest(manifest, channel, env!("CARGO_PKG_VERSION"))
}

#[cfg(target_os = "android")]
#[tauri::command]
pub fn native_open_app_update(app: tauri::AppHandle, version: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    let path = download_path(&version)?;
    app.opener()
        .open_url(format!("{HUB}{path}"), None::<&str>)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Manifest {
        Manifest {
            schema_version: SCHEMA_VERSION,
            channel: "prerelease".into(),
            release_status: "prerelease".into(),
            artifact: SignedArtifact {
                version: "v0.12.0".into(),
                target: "android-universal".into(),
                filename: "vashti-android.apk".into(),
                sha256: "ccc370c9ad603fbc70cab3146418dd437b7665fd7ab81724e1afaa7b1306eda7".into(),
                size_bytes: 50_400_986,
            },
            download_url: "/releases/v0.12.0/vashti-android.apk".into(),
            signature: "I2JFTUw8rpqv6x8ZFIcJpTGprxgr7xwYrPz5OThv4QKWm2LBBPDn4utS4u6wJQFBoHgSVmyqUDX2XpCk+7FSCg==".into(),
        }
    }

    #[test]
    fn apk_links_are_confined_to_the_official_release_path() {
        assert_eq!(
            download_path("v0.12.0").unwrap(),
            "/releases/v0.12.0/vashti-android.apk"
        );
        for value in [
            "../evil",
            "https://evil.test",
            "1.2.3/evil",
            "1.2.3+evil",
            "1.2.3-beta/evil",
        ] {
            assert!(download_path(value).is_err());
        }
        assert_eq!(HUB, "https://vashti.chat");
    }

    #[test]
    fn rejects_wrong_target_channel_path_and_invalid_signature() {
        assert!(validate_manifest(fixture(), Channel::Stable, "0.12.0").is_err());
        let mut manifest = fixture();
        manifest.artifact.target = "linux-x86_64".into();
        assert!(validate_manifest(manifest, Channel::Prerelease, "0.11.0").is_err());
        let mut manifest = fixture();
        manifest.download_url = "https://evil.test/app.apk".into();
        assert!(validate_manifest(manifest, Channel::Prerelease, "0.11.0").is_err());
        let mut manifest = fixture();
        manifest.signature = "invalid".into();
        assert!(
            validate_manifest(manifest, Channel::Prerelease, "0.11.0")
                .unwrap_err()
                .contains("signature")
        );
        let mut manifest = fixture();
        manifest.schema_version += 1;
        assert!(validate_manifest(manifest, Channel::Prerelease, "0.11.0").is_err());
    }

    #[test]
    fn signed_release_only_notifies_for_a_newer_available_version() {
        let update = validate_manifest(fixture(), Channel::Prerelease, "0.9.0")
            .unwrap()
            .unwrap();
        assert_eq!(update.version, "v0.12.0");
        assert!(
            validate_manifest(fixture(), Channel::Prerelease, "0.12.0")
                .unwrap()
                .is_none()
        );
        assert!(
            validate_manifest(fixture(), Channel::Prerelease, "0.13.0")
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn beta_download_paths_are_supported_without_allowing_path_injection() {
        assert_eq!(
            download_path("v0.13.0-beta.1").unwrap(),
            "/releases/v0.13.0-beta.1/vashti-android.apk"
        );
        assert!(download_path("v0.13.0+build.1").is_err());
        assert!(download_path("../0.13.0-beta.1").is_err());
    }
}
