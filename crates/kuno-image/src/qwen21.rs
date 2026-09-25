use super::*;
use kuno_protocol::{QWEN21_EDIT_RECIPE, QWEN21_RECIPE};

pub(super) fn import(
    nodes: &Map<String, Value>,
    output: &str,
    dimensions: Option<(u32, u32)>,
) -> Result<ImportedImageRecipe> {
    let mut graph = Graph {
        nodes,
        visited: BTreeSet::new(),
    };
    let save = graph.node(
        output,
        &["SaveImage", "PreviewImage", "SaveImageAdvanced"],
        &[
            "images",
            "filename_prefix",
            "format",
            "format.bit_depth",
            "format.input_color_space",
        ],
    )?;
    if save.contains_key("format") {
        equals(save, "format", "png")?;
    }
    if save.contains_key("format.input_color_space") {
        equals(save, "format.input_color_space", "sRGB")?;
    }
    let sixteen_bit = match save.get("format.bit_depth").and_then(Value::as_str) {
        Some("16-bit") => true,
        Some("8-bit") | None => false,
        _ => return Err("comfy_output_format_unsupported"),
    };
    let decode = graph.link(save, "images", &["VAEDecode"], &["samples", "vae"])?;
    let vae = graph.link(decode, "vae", &["VAELoader"], &["vae_name"])?;
    equals(vae, "vae_name", "qwen_image_2.1_vae_bf16.safetensors")?;
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
    let model_id = link_slot(sampler.get("model"), 0)?;
    let model = if nodes
        .get(model_id)
        .is_some_and(|n| n["class_type"] == "QwenImage21Cache")
    {
        let cache = graph.node(
            model_id,
            &["QwenImage21Cache"],
            &["model", "device", "dtype"],
        )?;
        equals(cache, "device", "auto")?;
        equals(cache, "dtype", "default")?;
        graph.link(
            cache,
            "model",
            &["UNETLoader"],
            &["unet_name", "weight_dtype"],
        )?
    } else {
        graph.node(model_id, &["UNETLoader"], &["unet_name", "weight_dtype"])?
    };
    equals(
        model,
        "unet_name",
        "qwen_image_2.1_int8_convrot.safetensors",
    )?;
    equals(model, "weight_dtype", "default")?;
    let encoder_id = link_slot(sampler.get("positive"), 0)?;
    if link_slot(sampler.get("negative"), 1)? != encoder_id {
        return Err("comfy_text_encoders_differ");
    }
    let mut keys = vec!["clip", "prompt", "negative_prompt", "resolution", "vae"];
    let image_keys: Vec<String> = (1..=10).map(|i| format!("images.image_{i}")).collect();
    keys.extend(image_keys.iter().map(String::as_str));
    let encoding = graph.node(encoder_id, &["TextEncodeQwenImage21"], &keys)?;
    if integer(encoding, "resolution")? != 1024 {
        return Err("comfy_reference_resolution_unsupported");
    }
    let clip = graph.link(
        encoding,
        "clip",
        &["CLIPLoader"],
        &["clip_name", "type", "device"],
    )?;
    equals(clip, "clip_name", "qwen3vl_8b_int8_convrot.safetensors")?;
    equals(clip, "type", "qwen_image")?;
    equals(clip, "device", "default")?;
    let mut required = 0;
    let mut gap = false;
    for key in &image_keys {
        if encoding.contains_key(key) {
            if gap {
                return Err("comfy_reference_slots_not_contiguous");
            }
            let input = graph.link(encoding, key, &["LoadImage"], &["image", "upload"])?;
            if !input.get("image").is_some_and(Value::is_string) {
                return Err("comfy_reference_invalid");
            }
            required += 1;
        } else {
            gap = true;
        }
    }
    if required > 0 && encoding.get("vae") != decode.get("vae") {
        return Err("comfy_reference_vae_differs");
    }
    let mut latent = sampler.get("latent_image").ok_or("comfy_link_missing")?;
    let (latent_id, _) = any_link(latent)?;
    if nodes
        .get(latent_id)
        .is_some_and(|n| n["class_type"] == "ComfySwitchNode")
    {
        let switch = graph.node(
            latent_id,
            &["ComfySwitchNode"],
            &["switch", "on_true", "on_false"],
        )?;
        let branch = switch
            .get("switch")
            .and_then(Value::as_bool)
            .ok_or("comfy_switch_not_literal")?;
        latent = switch
            .get(if branch { "on_true" } else { "on_false" })
            .ok_or("comfy_link_missing")?;
    }
    let (latent_id, slot) = any_link(latent)?;
    let (width, height) = if latent_id == encoder_id && slot == 2 && required > 0 {
        dimensions.ok_or("comfy_edit_output_dimensions_required")?
    } else {
        if slot != 0 {
            return Err("comfy_link_invalid");
        }
        let empty = graph.node(
            latent_id,
            &["EmptyLatentImage"],
            &["width", "height", "batch_size"],
        )?;
        if integer(empty, "batch_size")? != 1 {
            return Err("comfy_batch_unsupported");
        }
        let size = if empty.get("width").is_some_and(Value::is_array) {
            let id = link_slot(empty.get("width"), 0)?;
            if link_slot(empty.get("height"), 1)? != id {
                return Err("comfy_dimensions_invalid");
            }
            let resolution = graph.node(
                id,
                &["ResolutionSelector"],
                &["aspect_ratio", "megapixels", "multiple"],
            )?;
            let ratio = resolution
                .get("aspect_ratio")
                .and_then(Value::as_str)
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.split_once(':'))
                .ok_or("comfy_dimensions_invalid")?;
            let a: f64 = ratio.0.parse().map_err(|_| "comfy_dimensions_invalid")?;
            let b: f64 = ratio.1.parse().map_err(|_| "comfy_dimensions_invalid")?;
            let mp = number(resolution, "megapixels")?;
            let multiple = integer(resolution, "multiple")?;
            if !(0.1..=4.0).contains(&mp)
                || !(1.0..=32.0).contains(&a)
                || !(1.0..=32.0).contains(&b)
                || !matches!(multiple, 8 | 16 | 32 | 64)
            {
                return Err("comfy_dimensions_invalid");
            }
            let unit = multiple as f64;
            (
                ((mp * 1024. * 1024. * a / b).sqrt() / unit).round() as u32 * multiple as u32,
                ((mp * 1024. * 1024. * b / a).sqrt() / unit).round() as u32 * multiple as u32,
            )
        } else {
            (
                u32::try_from(integer(empty, "width")?).map_err(|_| "comfy_dimensions_invalid")?,
                u32::try_from(integer(empty, "height")?).map_err(|_| "comfy_dimensions_invalid")?,
            )
        };
        if dimensions.is_some_and(|d| d != size) {
            return Err("comfy_output_dimensions_disagree");
        }
        size
    };
    let recipe = ImageRecipe {
        attention: Default::default(),
        workflow: if required == 0 {
            QWEN21_RECIPE
        } else {
            QWEN21_EDIT_RECIPE
        }
        .into(),
        prompt: graph.text(encoding.get("prompt").ok_or("comfy_prompt_missing")?)?,
        negative_prompt: graph.text(
            encoding
                .get("negative_prompt")
                .ok_or("comfy_prompt_missing")?,
        )?,
        seed: integer(sampler, "seed")?.to_string(),
        width,
        height,
        steps: u32::try_from(integer(sampler, "steps")?)
            .map_err(|_| "image_sampling_settings_invalid")?,
        cfg: number(sampler, "cfg")?,
        flow_shift: None,
        reference_images: Vec::new(),
    };
    recipe.validate()?;
    let mut warnings = vec![
        "source_model_hashes_unknown",
        "cross_engine_pixel_identity_unverified",
    ];
    if required > 0 {
        warnings.push("original_reference_images_required");
    }
    if sixteen_bit {
        warnings.push("native_output_is_8_bit");
    }
    Ok(ImportedImageRecipe {
        recipe,
        selected_output: output.into(),
        required_reference_images: required,
        warnings,
    })
}
fn any_link(value: &Value) -> Result<(&str, u64)> {
    let a = value
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or("comfy_link_invalid")?;
    Ok((
        a[0].as_str().ok_or("comfy_link_invalid")?,
        a[1].as_u64().ok_or("comfy_link_invalid")?,
    ))
}
fn link_slot(value: Option<&Value>, slot: u64) -> Result<&str> {
    let (id, actual) = any_link(value.ok_or("comfy_link_missing")?)?;
    if actual != slot {
        return Err("comfy_link_invalid");
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph(edit: bool) -> Value {
        serde_json::from_str(if edit {
            include_str!("../tests/fixtures/qwen21-edit.json")
        } else {
            include_str!("../tests/fixtures/qwen21.json")
        })
        .unwrap()
    }
    fn import_graph(value: &Value, size: (u32, u32)) -> Result<ImportedImageRecipe> {
        import_with_dimensions(&serde_json::to_vec(value).unwrap(), None, Some(size))
    }
    #[test]
    fn generation_and_edit_follow_only_active_links() {
        let generated = import_graph(&graph(false), (1184, 1776)).unwrap();
        assert_eq!(generated.recipe.workflow, QWEN21_RECIPE);
        assert_eq!(generated.recipe.steps, 50);
        assert_eq!(generated.recipe.cfg, 1.0);
        assert!(generated.recipe.flow_shift.is_none());
        assert!(generated.warnings.contains(&"native_output_is_8_bit"));
        assert_eq!(
            (generated.recipe.width, generated.recipe.height),
            (1184, 1776)
        );
        assert!(generated.recipe.validate_execution().is_ok());
        let mut edit = graph(true);
        // Disconnected square resolution and image compare do not select output size.
        edit["13"]["inputs"]["megapixels"] = Value::from(9000);
        let edited = import_graph(&edit, (1248, 832)).unwrap();
        assert_eq!(edited.recipe.workflow, QWEN21_EDIT_RECIPE);
        assert_eq!(
            (
                edited.recipe.width,
                edited.recipe.height,
                edited.recipe.steps
            ),
            (1248, 832, 25)
        );
        assert_eq!(edited.recipe.cfg, 4.0);
        assert_eq!(edited.required_reference_images, 1);
        assert!(edited.recipe.reference_images.is_empty());
        assert_eq!(
            edited.recipe.validate_execution(),
            Err("image_edit_requires_references")
        );
        for i in 2..=10 {
            edit["485"]["inputs"][format!("images.image_{i}")] = serde_json::json!(["470", 0]);
        }
        assert_eq!(
            import_graph(&edit, (1248, 832))
                .unwrap()
                .required_reference_images,
            10
        );
        edit["485"]["inputs"]["images.image_11"] = serde_json::json!(["470", 0]);
        assert!(import_graph(&edit, (1248, 832)).is_err());
    }
    #[test]
    fn bad_links_settings_and_unknown_active_nodes_fail_closed() {
        for (node, key, value) in [
            ("482", "negative", serde_json::json!(["485", 0])),
            ("482", "positive", serde_json::json!(["missing", 0])),
            ("484", "model", serde_json::json!(["missing", 0])),
            ("484", "dtype", serde_json::json!("float16")),
            ("485", "resolution", serde_json::json!(512)),
            ("485", "images.image_1", serde_json::json!(["470", 1])),
            ("483", "switch", serde_json::json!(["470", 0])),
            (
                "479",
                "vae_name",
                serde_json::json!("qwen_image_vae.safetensors"),
            ),
        ] {
            let mut v = graph(true);
            v[node]["inputs"][key] = value;
            assert!(import_graph(&v, (1248, 832)).is_err(), "{node}.{key}");
        }
    }
    #[test]
    #[ignore = "requires local user reference PNGs; no user content in fixtures"]
    fn local_qwen21_reference_pngs_import() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (file, size, steps, references) in [
            ("Qwen_image_2.1_00078.png", (1184, 1776), 50, 0),
            ("Qwen_image_edit_2.1_00076.png", (1248, 832), 25, 1),
        ] {
            let result = import_png(&std::fs::read(root.join(file)).unwrap(), None).unwrap();
            assert_eq!((result.recipe.width, result.recipe.height), size);
            assert_eq!(result.recipe.steps, steps);
            assert_eq!(result.required_reference_images, references);
        }
    }
}
