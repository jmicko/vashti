//! Model conversion: a model file as published (BF16, FP16, F32 or FP8 weights) into the
//! form the engines load without conversion. Large matrices become INT8 in the rotated basis
//! (groups of 256, one f32 scale per row, `<name>.weight_scale` as `[rows, 1]`); other tensors
//! keep their stored type. One input file gives one output file, so a model split over
//! several files converts file by file as each arrives.
//!
//! Tensor names follow the original publisher's (the Hugging Face layout); the Wan 2.1 VAE's
//! are renamed to the layout `qwen_vae` reads.
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use rayon::prelude::*;

use super::{dit::quantize, safetensors::Safetensors};

/// Which part of a model a file belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Qwen Image's diffusion transformer.
    Transformer,
    /// A Qwen2.5-VL language model used as text encoder.
    TextEncoder,
    /// The Wan 2.1 VAE (Qwen Image's).
    Vae,
}

/// The format written into every output file's metadata.
pub const FORMAT: &str = "kuno-int8-convrot-g256-v1";

/// What happens to one input tensor.
enum Plan {
    Copy(String),
    Int8(String),
    Drop,
}

fn plan(part: Part, name: &str) -> Plan {
    match part {
        Part::Transformer => {
            let name = name.strip_prefix("model.diffusion_model.").unwrap_or(name);
            let block_matrix = name.starts_with("transformer_blocks.")
                && name.ends_with(".weight")
                && [
                    ".attn.to_q.",
                    ".attn.to_k.",
                    ".attn.to_v.",
                    ".attn.add_q_proj.",
                    ".attn.add_k_proj.",
                    ".attn.add_v_proj.",
                    ".attn.to_out.0.",
                    ".attn.to_add_out.",
                    "_mlp.net.0.proj.",
                    "_mlp.net.2.",
                    "_mod.1.",
                ]
                .iter()
                .any(|p| name.contains(p));
            if block_matrix || name == "norm_out.linear.weight" {
                Plan::Int8(name.to_owned())
            } else {
                Plan::Copy(name.to_owned())
            }
        }
        Part::TextEncoder => {
            if name == "lm_head.weight" {
                return Plan::Drop;
            }
            let matrix = name.starts_with("model.layers.")
                && name.ends_with(".weight")
                && [
                    ".q_proj.",
                    ".k_proj.",
                    ".v_proj.",
                    ".o_proj.",
                    ".gate_proj.",
                    ".up_proj.",
                    ".down_proj.",
                ]
                .iter()
                .any(|p| name.contains(p));
            if matrix {
                Plan::Int8(name.to_owned())
            } else {
                Plan::Copy(name.to_owned())
            }
        }
        Part::Vae => Plan::Copy(wan21_name(name).unwrap_or_else(|| name.to_owned())),
    }
}

/// The Wan 2.1 decoder's name for a Hugging Face (diffusers) VAE tensor; None for names
/// that are not the decoder's (kept as they are).
pub fn wan21_name(name: &str) -> Option<String> {
    let resnet = |rest: &str| -> Option<String> {
        let (part, suffix) = rest.split_once('.')?;
        let to = match part {
            "norm1" => "residual.0",
            "conv1" => "residual.2",
            "norm2" => "residual.3",
            "conv2" => "residual.6",
            "conv_shortcut" => "shortcut",
            _ => return None,
        };
        Some(format!("{to}.{suffix}"))
    };
    if let Some(rest) = name.strip_prefix("post_quant_conv.") {
        return Some(format!("conv2.{rest}"));
    }
    let rest = name.strip_prefix("decoder.")?;
    if let Some(r) = rest.strip_prefix("conv_in.") {
        return Some(format!("decoder.conv1.{r}"));
    }
    if let Some(r) = rest.strip_prefix("conv_out.") {
        return Some(format!("decoder.head.2.{r}"));
    }
    if let Some(r) = rest.strip_prefix("norm_out.") {
        return Some(format!("decoder.head.0.{r}"));
    }
    if let Some(r) = rest.strip_prefix("mid_block.attentions.0.") {
        return Some(format!("decoder.middle.1.{r}"));
    }
    if let Some(r) = rest.strip_prefix("mid_block.resnets.") {
        let (index, r) = r.split_once('.')?;
        let at = match index {
            "0" => 0,
            "1" => 2,
            _ => return None,
        };
        return Some(format!("decoder.middle.{at}.{}", resnet(r)?));
    }
    let r = rest.strip_prefix("up_blocks.")?;
    let (block, r) = r.split_once('.')?;
    let block: usize = block.parse().ok()?;
    if let Some(r) = r.strip_prefix("resnets.") {
        let (index, r) = r.split_once('.')?;
        let index: usize = index.parse().ok()?;
        return Some(format!(
            "decoder.upsamples.{}.{}",
            4 * block + index,
            resnet(r)?
        ));
    }
    let r = r.strip_prefix("upsamplers.0.")?;
    Some(format!("decoder.upsamples.{}.{r}", 4 * block + 3))
}

/// An output tensor: name, type, shape and how to make its bytes.
struct Out {
    name: String,
    kind: &'static str,
    shape: Vec<usize>,
    bytes: usize,
    source: String,
    /// 0 copy, 1 INT8 matrix, 2 its scales.
    what: u8,
}

/// Sizes of what was converted.
#[derive(Debug, Default)]
pub struct Converted {
    pub tensors: usize,
    pub int8: usize,
    pub bytes: u64,
}

/// Converts one file of `part` from `input` to `output` (written to `output` with a `.part`
/// suffix, then renamed).
pub fn convert(
    part: Part,
    input: &Path,
    output: &Path,
    source: &str,
) -> Result<Converted, &'static str> {
    let file = Safetensors::open(input)?;
    let mut names: Vec<&str> = file.names().collect();
    names.sort_unstable();
    let mut outs = Vec::new();
    for name in names {
        let shape = file.shape(name).ok_or("model_tensor_missing")?.to_vec();
        let kind = file.kind(name).ok_or("model_tensor_invalid")?;
        match plan(part, name) {
            Plan::Drop => {}
            Plan::Copy(to) => {
                let size = file.bytes(name).ok_or("model_tensor_invalid")?;
                outs.push(Out {
                    name: to,
                    kind,
                    shape,
                    bytes: size,
                    source: name.to_owned(),
                    what: 0,
                });
            }
            Plan::Int8(to) => {
                let [rows, columns] = shape[..] else {
                    return Err("model_tensor_invalid");
                };
                if !columns.is_multiple_of(256) {
                    return Err("model_tensor_invalid");
                }
                let base = to.strip_suffix(".weight").ok_or("model_tensor_invalid")?;
                outs.push(Out {
                    name: format!("{base}.weight_scale"),
                    kind: "F32",
                    shape: vec![rows, 1],
                    bytes: rows * 4,
                    source: name.to_owned(),
                    what: 2,
                });
                outs.push(Out {
                    name: to,
                    kind: "I8",
                    shape,
                    bytes: rows * columns,
                    source: name.to_owned(),
                    what: 1,
                });
            }
        }
    }
    outs.sort_by(|a, b| a.name.cmp(&b.name));
    // Header: tensors in name order, data offsets in the same order.
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".into(),
        serde_json::json!({ "format": "pt", "kuno_format": FORMAT, "kuno_source": source }),
    );
    let mut at = 0usize;
    for out in &outs {
        header.insert(
            out.name.clone(),
            serde_json::json!({ "dtype": out.kind, "shape": out.shape, "data_offsets": [at, at + out.bytes] }),
        );
        at += out.bytes;
    }
    let mut text =
        serde_json::to_vec(&serde_json::Value::Object(header)).map_err(|_| "convert_failed")?;
    while !text.len().is_multiple_of(8) {
        text.push(b' ');
    }
    let partial = output.with_extension("safetensors.part");
    let mut to = BufWriter::with_capacity(
        1 << 24,
        File::create(&partial).map_err(|_| "convert_write_failed")?,
    );
    let write = |to: &mut BufWriter<File>, bytes: &[u8]| {
        to.write_all(bytes).map_err(|_| "convert_write_failed")
    };
    write(&mut to, &(text.len() as u64).to_le_bytes())?;
    write(&mut to, &text)?;
    let mut done = Converted::default();
    // An INT8 matrix and its scales come from one quantization, written in name order.
    let mut pending: Option<(String, Vec<i8>, Vec<f32>)> = None;
    for out in &outs {
        if out.what == 0 {
            write(&mut to, &file.raw(&out.source)?.2)?;
        } else {
            if pending
                .as_ref()
                .is_none_or(|(source, _, _)| *source != out.source)
            {
                let [rows, columns] = file.shape(&out.source).ok_or("model_tensor_missing")?[..]
                else {
                    return Err("model_tensor_invalid");
                };
                let weight = file.floats(&out.source, &[rows, columns])?;
                let scale = fp8_scale(&file, &out.source)?;
                let weight: Vec<f32> = if scale == 1.0 {
                    weight
                } else {
                    weight.into_par_iter().map(|v| v * scale).collect()
                };
                let (bytes, scales) = quantize(&weight, columns);
                pending = Some((out.source.clone(), bytes, scales));
            }
            let (_, bytes, scales) = pending.as_ref().ok_or("convert_failed")?;
            if out.what == 1 {
                // SAFETY: i8 and u8 have the same size and alignment.
                let raw =
                    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), bytes.len()) };
                write(&mut to, raw)?;
                done.int8 += 1;
            } else {
                let raw: Vec<u8> = scales.iter().flat_map(|s| s.to_le_bytes()).collect();
                write(&mut to, &raw)?;
            }
        }
        done.tensors += 1;
        done.bytes += out.bytes as u64;
    }
    to.flush().map_err(|_| "convert_write_failed")?;
    to.into_inner()
        .map_err(|_| "convert_write_failed")?
        .sync_all()
        .map_err(|_| "convert_write_failed")?;
    std::fs::rename(&partial, output).map_err(|_| "convert_write_failed")?;
    Ok(done)
}

/// An FP8 matrix's scale in "scaled FP8" files (`<base>.scale_weight`), else 1.
fn fp8_scale(file: &Safetensors, weight: &str) -> Result<f32, &'static str> {
    let base = weight.strip_suffix(".weight").unwrap_or(weight);
    let name = format!("{base}.scale_weight");
    if file.shape(&name).is_some() {
        Ok(file.f32(&name, &[])?[0])
    } else {
        Ok(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vae_names_map_to_the_wan_layout() {
        let cases = [
            ("post_quant_conv.weight", "conv2.weight"),
            ("decoder.conv_in.bias", "decoder.conv1.bias"),
            (
                "decoder.mid_block.resnets.1.norm2.gamma",
                "decoder.middle.2.residual.3.gamma",
            ),
            (
                "decoder.mid_block.attentions.0.to_qkv.weight",
                "decoder.middle.1.to_qkv.weight",
            ),
            (
                "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
                "decoder.upsamples.4.shortcut.weight",
            ),
            (
                "decoder.up_blocks.2.upsamplers.0.resample.1.weight",
                "decoder.upsamples.11.resample.1.weight",
            ),
            (
                "decoder.up_blocks.3.resnets.2.conv2.bias",
                "decoder.upsamples.14.residual.6.bias",
            ),
            ("decoder.norm_out.gamma", "decoder.head.0.gamma"),
            ("decoder.conv_out.weight", "decoder.head.2.weight"),
        ];
        for (from, to) in cases {
            assert_eq!(wan21_name(from).as_deref(), Some(to), "{from}");
        }
        assert_eq!(wan21_name("encoder.conv_in.weight"), None);
    }

    /// `KUNO_CONVERT_PART=transformer|text_encoder|vae KUNO_CONVERT_IN=<file or dir>
    /// KUNO_CONVERT_OUT=<dir>`: converts every input file into the output directory.
    #[test]
    #[ignore = "converts model files"]
    fn convert_files() {
        let part = match std::env::var("KUNO_CONVERT_PART").unwrap().as_str() {
            "transformer" => Part::Transformer,
            "text_encoder" => Part::TextEncoder,
            _ => Part::Vae,
        };
        let input = std::path::PathBuf::from(std::env::var("KUNO_CONVERT_IN").unwrap());
        let out = std::path::PathBuf::from(std::env::var("KUNO_CONVERT_OUT").unwrap());
        std::fs::create_dir_all(&out).unwrap();
        let mut inputs: Vec<_> = if input.is_dir() {
            std::fs::read_dir(&input)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect()
        } else {
            vec![input]
        };
        inputs.sort();
        for path in inputs {
            let started = std::time::Instant::now();
            let to = out.join(path.file_name().unwrap());
            let done = convert(
                part,
                &path,
                &to,
                &std::env::var("KUNO_CONVERT_SOURCE").unwrap_or_default(),
            )
            .unwrap();
            println!(
                "{}: {} tensors ({} int8), {:.2} GB, {:.1} s",
                path.file_name().unwrap().to_string_lossy(),
                done.tensors,
                done.int8,
                done.bytes as f64 / 1e9,
                started.elapsed().as_secs_f64()
            );
        }
    }
}
