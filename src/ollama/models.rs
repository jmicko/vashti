use serde::Deserialize;

#[derive(Debug)]
pub struct OllamaModel {
    pub name: String,
    pub supports_images: bool,
}

#[derive(Debug, Deserialize)]
pub struct TagsResponse {
    #[serde(default)]
    models: Vec<TagModel>,
}

#[derive(Debug, Deserialize)]
struct TagModel {
    name: Option<String>,
    model: Option<String>,
    details: Option<ModelDetails>,
}

#[derive(Debug, Deserialize)]
struct ModelDetails {
    family: Option<String>,
    #[serde(default)]
    families: Vec<String>,
}

impl TagsResponse {
    pub fn into_models(self) -> Vec<OllamaModel> {
        self.models
            .into_iter()
            .filter_map(|model| {
                let name = model.name.or(model.model)?;
                let supports_images = supports_images(&name, model.details.as_ref());

                Some(OllamaModel {
                    name,
                    supports_images,
                })
            })
            .collect()
    }
}

fn supports_images(name: &str, details: Option<&ModelDetails>) -> bool {
    let name = name.to_ascii_lowercase();
    if ["llava", "bakllava", "moondream", "minicpm-v"]
        .iter()
        .any(|needle| name.contains(needle))
    {
        return true;
    }

    let Some(details) = details else {
        return false;
    };

    details
        .family
        .iter()
        .chain(details.families.iter())
        .map(|family| family.to_ascii_lowercase())
        .any(|family| family.contains("vision") || family.contains("clip"))
}
