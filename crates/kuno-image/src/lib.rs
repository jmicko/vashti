//! Bounded, in-memory ComfyUI execution-graph import. Never execute imported
//! nodes, follow URLs, open model paths, or infer settings from canvas notes.
#[cfg(any(feature = "native-sd", test))]
mod host_memory;
mod json;
mod png;
pub mod preview;
mod qwen21;
pub mod references;
use kuno_protocol::{ImageRecipe, ImportedImageRecipe, QWEN_RECIPE};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
pub const MAX_PNG_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_METADATA_BYTES: usize = 1024 * 1024;
type Result<T> = std::result::Result<T, &'static str>;

/// Verify a worker result before returning it to the content owner. Dimensions
/// are checked before allocating the decode buffer, so backend output is bounded.
pub fn validate_output(bytes: &[u8], width: u32, height: u32) -> Result<()> {
    // The native encoder emits only these three chunk types. Inspect the full
    // file before decoding: metadata can also occur after the pixel stream.
    if bytes.len() > 20 * 1024 * 1024 || !bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Err("image_worker_output_invalid");
    }
    let mut at = 8usize;
    let mut ended = false;
    while bytes.len().saturating_sub(at) >= 12 {
        let size = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let end = at
            .checked_add(size)
            .and_then(|n| n.checked_add(12))
            .filter(|end| *end <= bytes.len())
            .ok_or("image_worker_output_invalid")?;
        let kind = &bytes[at + 4..at + 8];
        if !matches!(kind, b"IHDR" | b"IDAT" | b"IEND") {
            return Err("image_worker_output_invalid");
        }
        if kind == b"IEND" {
            ended = size == 0 && end == bytes.len();
            break;
        }
        at = end;
    }
    if !ended {
        return Err("image_worker_output_invalid");
    }
    let mut reader = ::png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .map_err(|_| "image_worker_output_invalid")?;
    let info = reader.info();
    if info.width != width
        || info.height != height
        || !matches!(
            info.color_type,
            ::png::ColorType::Rgb | ::png::ColorType::Rgba
        )
        || info.bit_depth != ::png::BitDepth::Eight
        || !info.uncompressed_latin1_text.is_empty()
        || !info.compressed_latin1_text.is_empty()
        || !info.utf8_text.is_empty()
    {
        return Err("image_worker_output_invalid");
    }
    let size = reader
        .output_buffer_size()
        .filter(|s| *s <= 2048 * 2048 * 4)
        .ok_or("image_worker_output_invalid")?;
    let mut decoded = vec![0; size];
    reader
        .next_frame(&mut decoded)
        .map_err(|_| "image_worker_output_invalid")?;
    reader.finish().map_err(|_| "image_worker_output_invalid")?;
    Ok(())
}

#[cfg(test)]
mod output_tests {
    use super::*;
    #[test]
    fn output_rejects_metadata_after_pixels_and_trailing_bytes() {
        let mut bytes = Vec::new();
        {
            let mut encoder = ::png::Encoder::new(&mut bytes, 64, 64);
            encoder.set_color(::png::ColorType::Rgb);
            encoder.set_depth(::png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&vec![0; 64 * 64 * 3])
                .unwrap();
        }
        validate_output(&bytes, 64, 64).unwrap();
        let mut metadata = bytes[..bytes.len() - 12].to_vec();
        png::chunk(&mut metadata, b"tEXt", b"prompt\0synthetic");
        png::chunk(&mut metadata, b"IEND", &[]);
        assert!(validate_output(&metadata, 64, 64).is_err());
        bytes.push(0);
        assert!(validate_output(&bytes, 64, 64).is_err());
    }
}

pub fn import_png(bytes: &[u8], output: Option<&str>) -> Result<ImportedImageRecipe> {
    let prompt = png::prompt(bytes)?;
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    import_with_dimensions(&prompt, output, Some((width, height)))
}
pub fn import_prompt(bytes: &[u8], selected: Option<&str>) -> Result<ImportedImageRecipe> {
    import_with_dimensions(bytes, selected, None)
}
fn import_with_dimensions(
    bytes: &[u8],
    selected: Option<&str>,
    dimensions: Option<(u32, u32)>,
) -> Result<ImportedImageRecipe> {
    if bytes.len() > MAX_METADATA_BYTES {
        return Err("image_metadata_too_large");
    }
    let value = json::unique(bytes)?;
    let nodes = value.as_object().ok_or("comfy_graph_invalid")?;
    if nodes.is_empty() || nodes.len() > 1024 || nodes.keys().any(|s| s.len() > 128) {
        return Err("comfy_graph_invalid");
    }
    let outputs: Vec<_> = nodes
        .iter()
        .filter(|(_, n)| {
            matches!(
                n["class_type"].as_str(),
                Some("SaveImage" | "PreviewImage" | "SaveImageAdvanced")
            )
        })
        .map(|(id, _)| id.as_str())
        .collect();
    let output = match selected {
        Some(id) if outputs.contains(&id) => id,
        Some(_) => return Err("comfy_output_not_found"),
        None if outputs.len() == 1 => outputs[0],
        None if outputs.len() > 1 => return Err("comfy_outputs_ambiguous"),
        None => return Err("comfy_output_not_found"),
    };
    // Dispatch from this output's active VAE, never from disconnected notes.
    if let Some(decode_id) = nodes[output]["inputs"]["images"][0].as_str()
        && let Some(vae_id) = nodes
            .get(decode_id)
            .and_then(|n| n["inputs"]["vae"][0].as_str())
        && nodes
            .get(vae_id)
            .is_some_and(|n| n["inputs"]["vae_name"] == "qwen_image_2.1_vae_bf16.safetensors")
    {
        return qwen21::import(nodes, output, dimensions);
    }
    let mut graph = Graph {
        nodes,
        visited: BTreeSet::new(),
    };
    let save = graph.node(
        output,
        &["SaveImage", "PreviewImage"],
        &["images", "filename_prefix"],
    )?;
    let decode = graph.link(save, "images", &["VAEDecode"], &["samples", "vae"])?;
    let vae = graph.link(decode, "vae", &["VAELoader"], &["vae_name"])?;
    equals(vae, "vae_name", "qwen_image_vae.safetensors")?;
    let sampler = graph.link(
        decode,
        "samples",
        &["KSampler"],
        &[
            "seed",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
            "model",
            "positive",
            "negative",
            "latent_image",
        ],
    )?;
    equals(sampler, "sampler_name", "euler")?;
    equals(sampler, "scheduler", "simple")?;
    if number(sampler, "denoise")? != 1.0 {
        return Err("comfy_denoise_unsupported");
    }
    let sampling = graph.link(
        sampler,
        "model",
        &["ModelSamplingAuraFlow"],
        &["shift", "sampling", "model"],
    )?;
    if sampling.contains_key("sampling") {
        equals(sampling, "sampling", "flow")?;
    }
    let model = graph.link(
        sampling,
        "model",
        &["UNETLoader"],
        &["unet_name", "weight_dtype"],
    )?;
    equals(model, "unet_name", "qwen_image_2512_fp8_e4m3fn.safetensors")?;
    equals(model, "weight_dtype", "default")?;
    let positive = graph.link(sampler, "positive", &["CLIPTextEncode"], &["text", "clip"])?;
    let negative = graph.link(sampler, "negative", &["CLIPTextEncode"], &["text", "clip"])?;
    if positive.get("clip") != negative.get("clip") {
        return Err("comfy_text_encoders_differ");
    }
    let clip = graph.link(
        positive,
        "clip",
        &["CLIPLoader"],
        &["clip_name", "type", "device"],
    )?;
    equals(clip, "clip_name", "qwen_2.5_vl_7b_fp8_scaled.safetensors")?;
    equals(clip, "type", "qwen_image")?;
    equals(clip, "device", "default")?;
    let prompt = graph.text(positive.get("text").ok_or("comfy_prompt_missing")?)?;
    let negative_prompt = graph.text(negative.get("text").ok_or("comfy_prompt_missing")?)?;
    let latent = graph.link(
        sampler,
        "latent_image",
        &["EmptySD3LatentImage"],
        &["width", "height", "batch_size"],
    )?;
    if integer(latent, "batch_size")? != 1 {
        return Err("comfy_batch_unsupported");
    }
    let recipe = ImageRecipe {
        attention: Default::default(),
        workflow: QWEN_RECIPE.into(),
        prompt,
        negative_prompt,
        seed: integer(sampler, "seed")?.to_string(),
        width: u32::try_from(integer(latent, "width")?)
            .map_err(|_| "image_dimensions_unsupported")?,
        height: u32::try_from(integer(latent, "height")?)
            .map_err(|_| "image_dimensions_unsupported")?,
        steps: u32::try_from(integer(sampler, "steps")?)
            .map_err(|_| "image_sampling_settings_invalid")?,
        cfg: number(sampler, "cfg")?,
        flow_shift: Some(number(sampling, "shift")?),
        reference_images: Vec::new(),
    };
    recipe.validate()?;
    Ok(ImportedImageRecipe {
        recipe,
        selected_output: output.into(),
        required_reference_images: 0,
        warnings: vec![
            "source_model_hashes_unknown",
            "cross_engine_pixel_identity_unverified",
        ],
    })
}
struct Graph<'a> {
    nodes: &'a Map<String, Value>,
    visited: BTreeSet<String>,
}
impl<'a> Graph<'a> {
    fn node(&mut self, id: &str, types: &[&str], keys: &[&str]) -> Result<&'a Map<String, Value>> {
        // This recognizer has a fixed-depth chain, and bounds every indirect string
        // traversal. Sharing the same encoder is valid; arbitrary graph recursion is not.
        if self.visited.len() > 64 {
            return Err("comfy_graph_too_deep");
        }
        self.visited.insert(id.into());
        let n = self
            .nodes
            .get(id)
            .and_then(Value::as_object)
            .ok_or("comfy_node_missing")?;
        if !n
            .get("class_type")
            .and_then(Value::as_str)
            .is_some_and(|s| types.contains(&s))
        {
            return Err("comfy_active_node_unsupported");
        }
        let i = n
            .get("inputs")
            .and_then(Value::as_object)
            .ok_or("comfy_inputs_invalid")?;
        if i.keys().any(|k| !keys.contains(&k.as_str())) {
            return Err("comfy_active_input_unsupported");
        }
        Ok(i)
    }
    fn reference(value: &Value) -> Result<&str> {
        let a = value.as_array().ok_or("comfy_link_invalid")?;
        if a.len() != 2 || a[1].as_u64() != Some(0) {
            return Err("comfy_link_invalid");
        }
        a[0].as_str().ok_or("comfy_link_invalid")
    }
    fn link(
        &mut self,
        inputs: &Map<String, Value>,
        key: &str,
        types: &[&str],
        keys: &[&str],
    ) -> Result<&'a Map<String, Value>> {
        let id = Self::reference(inputs.get(key).ok_or("comfy_link_missing")?)?;
        self.node(id, types, keys)
    }
    fn text(&mut self, value: &Value) -> Result<String> {
        if let Some(text) = value.as_str() {
            return Ok(text.into());
        }
        let id = Self::reference(value)?;
        let input = self.node(
            id,
            &["PrimitiveStringMultiline", "PrimitiveString"],
            &["value"],
        )?;
        input
            .get("value")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or("comfy_prompt_not_literal")
    }
}
fn equals(i: &Map<String, Value>, k: &str, w: &str) -> Result<()> {
    if i.get(k).and_then(Value::as_str) == Some(w) {
        Ok(())
    } else {
        Err("comfy_recipe_setting_unsupported")
    }
}
fn number(i: &Map<String, Value>, k: &str) -> Result<f64> {
    i.get(k)
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or("comfy_number_invalid")
}
fn integer(i: &Map<String, Value>, k: &str) -> Result<u64> {
    i.get(k)
        .and_then(Value::as_u64)
        .ok_or("comfy_integer_invalid")
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<u8> {
        let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
        png::chunk(&mut out, b"IHDR", &[0; 13]);
        let mut metadata = b"prompt\0".to_vec();
        metadata.extend_from_slice(include_bytes!("../tests/fixtures/qwen.json"));
        png::chunk(&mut out, b"tEXt", &metadata);
        png::chunk(&mut out, b"IEND", &[]);
        out
    }
    fn graph() -> Value {
        serde_json::from_slice(&png::prompt(&fixture()).unwrap()).unwrap()
    }
    fn import(v: &Value) -> Result<ImportedImageRecipe> {
        import_prompt(&serde_json::to_vec(v).unwrap(), None)
    }
    #[test]
    fn reference_graph_maps_active_recipe_and_indirect_prompts() {
        let r = import_png(&fixture(), None).unwrap().recipe;
        assert_eq!((r.width, r.height, r.steps), (832, 1248, 100));
        assert_eq!(r.cfg, 5.0);
        assert!((r.flow_shift.unwrap() - 3.1).abs() < 1e-12);
        assert_eq!(r.seed, "684489560344628");
        assert_eq!(r.prompt, "A red wooden boat on a quiet lake.");
        assert_eq!(r.negative_prompt, "Blurry, low quality.");
        assert_eq!(r.workflow, QWEN_RECIPE);
    }
    #[test]
    #[ignore = "requires KUNO_REFERENCE_PNG; the user's image is not a build dependency"]
    fn local_reference_png_has_expected_settings() {
        let bytes = std::fs::read(std::env::var_os("KUNO_REFERENCE_PNG").unwrap()).unwrap();
        let r = import_png(&bytes, None).unwrap().recipe;
        assert_eq!((r.width, r.height, r.steps), (832, 1248, 100));
        assert_eq!(r.seed, "684489560344628");
        assert_eq!(r.cfg, 5.0);
        assert!((r.flow_shift.unwrap() - 3.1).abs() < 1e-12);
    }
    #[test]
    fn active_changes_fail_but_disconnected_notes_and_loras_do_not() {
        let mut v = graph();
        v["disconnected"] = serde_json::json!({"class_type":"LoraLoader","inputs":{"url":"https://example.invalid/model"}});
        assert!(import(&v).is_ok());
        v["110"]["inputs"]["model"] = serde_json::json!(["disconnected", 0]);
        assert!(matches!(import(&v), Err("comfy_active_node_unsupported")));
        for (node, key, value) in [
            ("106", "scheduler", Value::from("karras")),
            ("106", "denoise", Value::from(0.5)),
            ("107", "batch_size", Value::from(2)),
            ("105", "unet_name", Value::from("different.safetensors")),
            ("106", "unrecognized_option", Value::from(true)),
        ] {
            let mut v = graph();
            v[node]["inputs"][key] = value;
            assert!(import(&v).is_err());
        }
    }
    #[test]
    fn ambiguous_outputs_bad_links_and_duplicate_json_are_rejected() {
        let mut v = graph();
        v["second"] = v["60"].clone();
        assert!(matches!(import(&v), Err("comfy_outputs_ambiguous")));
        assert!(import_prompt(&serde_json::to_vec(&v).unwrap(), Some("60")).is_ok());
        let mut v = graph();
        v["108"]["inputs"]["text"] = serde_json::json!(["91", 1]);
        assert!(import(&v).is_err());
        v["108"]["inputs"]["text"] = serde_json::json!(["108", 0]);
        assert!(import(&v).is_err());
        assert!(matches!(
            import_prompt(b"{\"x\":{},\"x\":{}}", None),
            Err("comfy_json_invalid_or_ambiguous")
        ));
        assert!(json::unique(b"{\"inputs\":{\"seed\":1,\"seed\":2}}").is_err());
    }
    #[test]
    fn seeds_survive_javascript_and_invalid_resource_bounds_fail() {
        let mut v = graph();
        v["106"]["inputs"]["seed"] = Value::from(u64::MAX);
        let r = import(&v).unwrap().recipe;
        assert_eq!(r.seed, u64::MAX.to_string());
        assert!(serde_json::to_value(&r).unwrap()["seed"].is_string());
        for (node, key, value) in [
            ("106", "steps", Value::from(0)),
            ("107", "width", Value::from(833)),
            ("106", "seed", Value::from(-1)),
            ("110", "shift", Value::from(100)),
        ] {
            let mut v = graph();
            v[node]["inputs"][key] = value;
            assert!(import(&v).is_err());
        }
    }
}

#[cfg(feature = "native-sd")]
pub mod native;
