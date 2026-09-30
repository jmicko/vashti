//! Cached, bounded prerequisite checks; never probe GPU work or user content.
use crate::image_runtime::ImageConfig;
use kuno_protocol::{
    ImageCapabilities, ImageWorkflowCapability, QWEN_RECIPE, QWEN21_EDIT_RECIPE, QWEN21_RECIPE,
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const REFRESH: Duration = Duration::from_secs(5);
struct Cache {
    value: ImageCapabilities,
    checked: Option<Instant>,
    refreshing: bool,
}
pub(crate) struct ImageReadiness {
    cache: Mutex<Cache>,
    fast_attention: bool,
}

impl ImageReadiness {
    pub fn new(config: Option<&ImageConfig>, fast_attention: bool) -> Arc<Self> {
        Arc::new(Self {
            fast_attention,
            cache: Mutex::new(Cache {
                value: describe(
                    config,
                    cfg!(feature = "native-image"),
                    fast_attention,
                    Some("image_runtime_unavailable"),
                    |_| true,
                ),
                checked: None,
                refreshing: false,
            }),
        })
    }

    pub async fn snapshot(self: &Arc<Self>, config: Option<ImageConfig>) -> ImageCapabilities {
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.refreshing || cache.checked.is_some_and(|t| t.elapsed() < REFRESH) {
                return cache.value.clone();
            }
            cache.refreshing = true;
        }
        let (send, receive) = tokio::sync::oneshot::channel();
        let cache = self.clone();
        // The refresh survives a disconnected discovery request, so even slow
        // filesystems permit only one outstanding probe per gateway.
        tokio::spawn(async move {
            let fast_attention = cache.fast_attention;
            let value =
                tokio::task::spawn_blocking(move || probe(config.as_ref(), fast_attention)).await;
            let mut guard = cache.cache.lock().unwrap();
            if let Ok(value) = value {
                guard.value = value;
            }
            guard.checked = Some(Instant::now());
            guard.refreshing = false;
            let _ = send.send(guard.value.clone());
        });
        match tokio::time::timeout(Duration::from_secs(1), receive).await {
            Ok(Ok(value)) => value,
            _ => self.cache.lock().unwrap().value.clone(),
        }
    }
}

fn probe(config: Option<&ImageConfig>, fast_attention: bool) -> ImageCapabilities {
    let privacy = if cfg!(feature = "native-image") && config.is_some() {
        (crate::image_privacy::require_no_swap().is_err()
            || crate::image_privacy::require_landlock().is_err())
        .then_some("image_privacy_unavailable")
    } else {
        None
    };
    describe(
        config,
        cfg!(feature = "native-image"),
        fast_attention,
        privacy,
        |id| config.is_some_and(|c| crate::collections::ready(&c.models, id).is_ok()),
    )
}

/// Degraded gateway startup must not let an unsafe parent accept image content.
pub(crate) async fn require_privacy() -> Result<(), &'static str> {
    tokio::task::spawn_blocking(|| {
        crate::image_privacy::require_no_swap()
            .and_then(|_| crate::image_privacy::require_landlock())
            .map_err(|_| "image_privacy_unavailable")
    })
    .await
    .map_err(|_| "image_runtime_unavailable")?
}

fn describe(
    config: Option<&ImageConfig>,
    native: bool,
    fast_attention: bool,
    privacy: Option<&str>,
    mut models_ready: impl FnMut(&str) -> bool,
) -> ImageCapabilities {
    let mut images = ImageCapabilities {
        api_version: 1,
        access: "inference".into(),
        state: "ready".into(),
        reason: None,
        generate_path: "/kuno/v1/images/generate".into(),
        response_formats: vec![kuno_protocol::IMAGE_STREAM_MEDIA_TYPE.into()],
        workflows: Vec::new(),
    };
    let reason = if config.is_none() {
        Some("image_generation_not_configured")
    } else if !native {
        Some("image_engine_not_in_this_build")
    } else {
        None
    };
    if let Some(reason) = reason {
        images.state = if config.is_none() {
            "disabled"
        } else {
            "unsupported"
        }
        .into();
        images.reason = Some(reason.into());
        return images;
    }
    for collection in &config.unwrap().collections {
        let reason = if !models_ready(collection) {
            Some("image_models_unavailable")
        } else {
            privacy
        };
        let ids: &[&str] = if collection == "qwen-image-2512-fp8" {
            &[QWEN_RECIPE]
        } else {
            &[QWEN21_RECIPE, QWEN21_EDIT_RECIPE]
        };
        for id in ids {
            let edit = *id == QWEN21_EDIT_RECIPE;
            if images.workflows.iter().any(|w| w.id == *id) {
                continue;
            }
            images.workflows.push(ImageWorkflowCapability {
                id: (*id).into(),
                state: if reason.is_some() {
                    "unavailable"
                } else {
                    "ready"
                }
                .into(),
                reason: reason.map(str::to_owned),
                reference_images_min: u32::from(edit),
                reference_images_max: if edit { 10 } else { 0 },
                attention: if *id == QWEN_RECIPE || !fast_attention {
                    vec!["standard".into()]
                } else {
                    vec!["standard".into(), "fast".into()]
                },
            });
        }
    }
    let available = images
        .workflows
        .iter()
        .filter(|w| w.state == "ready")
        .count();
    if available == 0 {
        images.state = "unavailable".into();
        images.reason = images.workflows.first().and_then(|w| w.reason.clone());
    } else if available < images.workflows.len() {
        images.state = "partial".into();
    }
    images
}

/// Rechecked per request; discovery's cache is advisory, never authorization.
pub(crate) async fn require_workflow(
    config: ImageConfig,
    workflow: &str,
    fast_attention: bool,
) -> Result<(), &'static str> {
    let images = tokio::task::spawn_blocking(move || probe(Some(&config), fast_attention))
        .await
        .map_err(|_| "image_runtime_unavailable")?;
    match images.workflows.iter().find(|w| w.id == workflow) {
        Some(w) if w.state == "ready" => Ok(()),
        Some(w) if w.reason.as_deref() == Some("image_models_unavailable") => {
            Err("image_models_unavailable")
        }
        Some(_) => Err("image_privacy_unavailable"),
        None => Err("image_engine_not_in_this_build"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ImageConfig {
        ImageConfig {
            models: "/synthetic/models".into(),
            pci_address: "0000:01:00.0".into(),
            timeout_seconds: 60,
            collections: vec!["qwen-image-2512-fp8".into(), "qwen-image-2.1-int8".into()],
        }
    }
    #[test]
    fn missing_collection_does_not_hide_other_workflows_and_recovers() {
        let c = config();
        let partial = describe(Some(&c), true, true, None, |id| id == "qwen-image-2.1-int8");
        assert_eq!(partial.state, "partial");
        assert_eq!(partial.workflows.len(), 3);
        assert_eq!(
            partial.workflows[0].reason.as_deref(),
            Some("image_models_unavailable")
        );
        assert_eq!(partial.workflows[2].reference_images_max, 10);
        assert_eq!(
            describe(Some(&c), true, true, None, |_| true).state,
            "ready"
        );
        assert_eq!(
            describe(
                Some(&c),
                true,
                true,
                Some("image_privacy_unavailable"),
                |_| true
            )
            .state,
            "unavailable"
        );
    }
    #[test]
    fn disabled_and_unsupported_do_not_touch_artifacts() {
        assert_eq!(
            describe(None, true, true, None, |_| panic!("artifact probe")).state,
            "disabled"
        );
        assert_eq!(
            describe(Some(&config()), false, false, None, |_| panic!(
                "artifact probe"
            ))
            .state,
            "unsupported"
        );
    }

    #[test]
    fn fast_attention_is_advertised_only_for_a_supported_cuda_image_device() {
        let c = config();
        let vulkan = describe(Some(&c), true, false, None, |_| true);
        assert!(vulkan.workflows.iter().all(|w| w.attention == ["standard"]));
        let cuda = describe(Some(&c), true, true, None, |_| true);
        assert_eq!(cuda.workflows[0].attention, ["standard"]);
        assert_eq!(cuda.workflows[1].attention, ["standard", "fast"]);
    }
    #[tokio::test]
    async fn readiness_is_cached_and_has_no_private_paths() {
        let c = config();
        let cache = ImageReadiness::new(Some(&c), false);
        let a = cache.snapshot(Some(c.clone())).await;
        let b = cache.snapshot(Some(c)).await;
        assert_eq!(
            serde_json::to_value(&a).unwrap(),
            serde_json::to_value(&b).unwrap()
        );
        assert!(!serde_json::to_string(&a).unwrap().contains("/synthetic"));
        assert!(!cache.cache.lock().unwrap().refreshing);
    }
}
