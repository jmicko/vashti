//! Approved public-only diagnostic: capture CPU routing at position23 after
//! checking positions0..23 against the existing fixed32 full-logit reference.
//! No prompt arguments, hidden-state file, inference API or runtime changes.

#[path = "../../../apps/kuno/benchmarks/native-text/src/q3_public_reference.rs"]
mod q3_public_reference;
#[path = "../../../apps/kuno/benchmarks/native-text/src/q3_router_reference.rs"]
pub mod q3_router_reference;

use kuno_text::model::{MatrixExecutor, Qwen4Exp, RoutedWeights, greedy};
use kuno_text::threaded::ThreadedExecutor;
use kuno_text::{Error, GgufFile, GgufLimits, Qwen35Tokenizer, TensorView, TokenizerLimits};
use q3_router_reference::{EXPERTS, LAYERS, Layer, MODEL_BYTES, POSITION, POSITIONS, Reference};
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use zeroize::Zeroizing;

struct Inputs {
    directory: PathBuf,
    logits: PathBuf,
    routers: PathBuf,
}
fn registration(
    args: &[OsString],
    environment: impl IntoIterator<Item = (OsString, OsString)>,
    default_build: bool,
) -> Result<Inputs, &'static str> {
    if args.len() != 1 || !default_build {
        return Err("public Q3 router writer requires default build and no prompt arguments");
    }
    let mut settings = BTreeMap::new();
    for (key, value) in environment {
        if !(key.as_encoded_bytes().starts_with(b"KUNO_LAB_")
            || key.as_encoded_bytes().starts_with(b"KUNO_TEST_")
            || key.as_encoded_bytes().starts_with(b"KUNO_PUBLIC_"))
        {
            continue;
        }
        if !matches!(
            key.to_str(),
            Some(
                "KUNO_LAB_Q3_ROUTER_PUBLIC_ONLY"
                    | "KUNO_TEST_Q3_DIR"
                    | "KUNO_PUBLIC_Q3_REFERENCE"
                    | "KUNO_PUBLIC_Q3_ROUTERS"
            )
        ) || value.is_empty()
            || settings.insert(key, value).is_some()
        {
            return Err("public Q3 router writer rejects other inputs/settings");
        }
    }
    if settings.remove(OsStr::new("KUNO_LAB_Q3_ROUTER_PUBLIC_ONLY")) != Some("1".into()) {
        return Err("public Q3 router writer requires its explicit public-only gate");
    }
    Ok(Inputs {
        directory: settings
            .remove(OsStr::new("KUNO_TEST_Q3_DIR"))
            .ok_or("verified original Q3 directory required")?
            .into(),
        logits: settings
            .remove(OsStr::new("KUNO_PUBLIC_Q3_REFERENCE"))
            .ok_or("verified fixed32 full-logit reference required")?
            .into(),
        routers: settings
            .remove(OsStr::new("KUNO_PUBLIC_Q3_ROUTERS"))
            .ok_or("new fixed public router output required")?
            .into(),
    })
}

fn rejected() -> Error {
    Error::InvalidFormat("public Q3 router capture order or shape")
}
fn layer_from_name(name: &str, suffix: &str) -> kuno_text::Result<usize> {
    let body = name
        .strip_prefix("blk.")
        .and_then(|n| n.strip_suffix(suffix))
        .ok_or_else(rejected)?;
    let layer: usize = body.parse().map_err(|_| rejected())?;
    if layer >= LAYERS || body != layer.to_string() {
        return Err(rejected());
    }
    Ok(layer)
}

/// Only run() is delegated. Returning None from routed_moe leaves the graph's
/// original per-expert projections flowing through this wrapper unchanged.
struct Capture<E> {
    inner: E,
    position: usize,
    next_layer: usize,
    pending_layer: Option<usize>,
    pending_scores: Option<Zeroizing<[f32; EXPERTS]>>,
    captured: Vec<Layer>,
}
impl<E> Capture<E> {
    fn new(inner: E) -> Self {
        Self {
            inner,
            position: 0,
            next_layer: 0,
            pending_layer: None,
            pending_scores: None,
            captured: Vec::with_capacity(LAYERS),
        }
    }
    fn begin_position(&mut self, position: usize) -> kuno_text::Result<()> {
        if position > POSITION || self.pending_layer.is_some() || self.pending_scores.is_some() {
            return Err(rejected());
        }
        self.position = position;
        self.next_layer = 0;
        Ok(())
    }
    fn completed_position(&self) -> kuno_text::Result<()> {
        if self.next_layer != LAYERS
            || self.pending_layer.is_some()
            || self.pending_scores.is_some()
        {
            return Err(rejected());
        }
        if self.position == POSITION && self.captured.len() != LAYERS {
            return Err(rejected());
        }
        Ok(())
    }
    fn capture_scores(&mut self, layer: usize, scores: &[f32]) -> kuno_text::Result<()> {
        if layer != self.next_layer
            || self.pending_layer.is_some()
            || scores.len() != EXPERTS
            || scores.iter().any(|v| !v.is_finite())
        {
            return Err(rejected());
        }
        self.pending_layer = Some(layer);
        self.next_layer += 1;
        if self.position == POSITION {
            let mut owner = Zeroizing::new([0.; EXPERTS]);
            owner.copy_from_slice(scores);
            self.pending_scores = Some(owner);
        }
        Ok(())
    }
    fn capture_selection(
        &mut self,
        layer: usize,
        selected: &[(usize, f32)],
    ) -> kuno_text::Result<()> {
        if self.pending_layer != Some(layer) || selected.len() != 10 {
            return Err(rejected());
        }
        if self.position == POSITION {
            if self.captured.len() != layer {
                return Err(rejected());
            }
            let scores = self.pending_scores.take().ok_or_else(rejected)?;
            self.captured.push(Layer::new(&*scores, selected)?);
        } else if self.pending_scores.is_some() {
            return Err(rejected());
        }
        self.pending_layer = None;
        Ok(())
    }
}
impl<E: MatrixExecutor> MatrixExecutor for Capture<E> {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> kuno_text::Result<Vec<f32>> {
        let result = self.inner.run(tensor, expert, input, cancel)?;
        if tensor.info().name().ends_with(".ffn_gate_inp.weight") {
            if expert != 0 || input.len() != 2560 || tensor.info().dimensions() != [2560, 512] {
                return Err(rejected());
            }
            let layer = layer_from_name(tensor.info().name(), ".ffn_gate_inp.weight")?;
            self.capture_scores(layer, &result)?;
        }
        Ok(result)
    }
    fn routed_moe(
        &mut self,
        weights: RoutedWeights<'_>,
        input: &[f32],
        selected: &[(usize, f32)],
        _cancel: &AtomicBool,
    ) -> kuno_text::Result<Option<Vec<f32>>> {
        let layer = layer_from_name(weights.gate.info().name(), ".ffn_gate_exps.weight")?;
        if input.len() != 2560
            || layer_from_name(weights.up.info().name(), ".ffn_up_exps.weight")? != layer
            || layer_from_name(weights.down.info().name(), ".ffn_down_exps.weight")? != layer
        {
            return Err(rejected());
        }
        self.capture_selection(layer, selected)?;
        Ok(None)
    }
}

fn require_original(file: &GgufFile) -> kuno_text::Result<()> {
    let mut formats = BTreeMap::new();
    for tensor in file.tensors().values() {
        *formats.entry(tensor.dtype().gguf_id()).or_insert(0usize) += 1;
    }
    if file.file_len() != MODEL_BYTES
        || file.tensors().len() != 1224
        || formats
            != [
                (0, 557),
                (8, 502),
                (14, 1),
                (18, 94),
                (20, 44),
                (23, 2),
                (30, 24),
            ]
            .into()
    {
        return Err(Error::InvalidFormat(
            "public Q3 original artifact geometry/formats",
        ));
    }
    file.verify_file_unchanged()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::set_hook(Box::new(|_| {
        eprintln!("public_q3_router_writer_failure panic=true")
    }));
    let inputs = registration(
        &std::env::args_os().collect::<Vec<_>>(),
        std::env::vars_os(),
        !cfg!(any(
            feature = "cpu-work-stealing-lab",
            feature = "cpu-q8-two-pass"
        )),
    )?;
    let reference = q3_public_reference::Reference::load(inputs.logits)?;
    reference.require_positions(POSITIONS)?;
    let paths: Vec<_> = (1..=3)
        .map(|i| {
            inputs.directory.join(format!(
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-{i:05}-of-00003.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default())?;
    require_original(&file)?;
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default())?;
    if tokenizer.encode("!", true)? != [0] {
        return Err("public Q3 router token binding rejected".into());
    }
    let graph = Qwen4Exp::bind(&file)?;
    let mut sequence = graph.sequence(POSITIONS)?;
    let mut executor = Capture::new(ThreadedExecutor::new(16)?);
    let cancel = AtomicBool::new(false);
    let mut input = 0;
    for position in 0..=POSITION {
        executor.begin_position(position)?;
        let mut completed_layers = 0;
        let logits =
            Zeroizing::new(
                graph.step(&mut executor, &mut sequence, input, &cancel, |_| {
                    completed_layers += 1;
                })?,
            );
        executor.completed_position()?;
        if completed_layers != LAYERS || sequence.position() != position + 1 {
            return Err("public Q3 router complete position rejected".into());
        }
        reference.verify(position, &logits)?;
        input = greedy(&logits)?;
        println!(
            "public_q3_router_position={position} full_reference_verified=true layers={LAYERS}"
        );
    }
    require_original(&file)?;
    sequence.reset();
    Reference::new(executor.captured)?.write_new(inputs.routers)?;
    println!(
        "public_q3_router_complete verified_positions=24 capture_position=23 layers=48 scores_per_layer=512 selected_per_layer=10"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn environment() -> Vec<(OsString, OsString)> {
        [
            ("KUNO_LAB_Q3_ROUTER_PUBLIC_ONLY", "1"),
            ("KUNO_TEST_Q3_DIR", "/public/q3"),
            ("KUNO_PUBLIC_Q3_REFERENCE", "/public/full32.bin"),
            ("KUNO_PUBLIC_Q3_ROUTERS", "/public/routers.bin"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .into()
    }
    #[test]
    fn only_explicit_public_default_no_prompt_entry_is_accepted() {
        let args = [OsString::from("public-writer")];
        assert!(registration(&args, environment(), true).is_ok());
        assert!(registration(&args, environment(), false).is_err());
        assert!(registration(&[], environment(), true).is_err());
        assert!(registration(&[args[0].clone(), "prompt".into()], environment(), true).is_err());
        for key in [
            "KUNO_LAB_TRANSIENT_IO",
            "KUNO_TEST_GGUF",
            "KUNO_PUBLIC_RECEIPT",
            "KUNO_LAB_Q3_PUBLIC_POSITIONS",
        ] {
            let mut vars = environment();
            vars.push((key.into(), "0".into()));
            assert!(registration(&args, vars, true).is_err());
        }
        for index in 0..4 {
            let mut vars = environment();
            vars.remove(index);
            assert!(registration(&args, vars, true).is_err());
        }
    }
    #[test]
    fn actual_router_capture_requires_score_then_route_in_order_and_complete48() {
        let scores = [0.; EXPERTS];
        let selected = kuno_text::ops::router_topk(&scores, 10, true).unwrap();
        let mut capture = Capture::new(());
        capture.begin_position(POSITION).unwrap();
        assert!(capture.capture_selection(0, &selected).is_err());
        assert!(capture.capture_scores(1, &scores).is_err());
        assert!(capture.completed_position().is_err());
        for layer in 0..LAYERS {
            capture.capture_scores(layer, &scores).unwrap();
            assert!(capture.capture_scores(layer, &scores).is_err());
            assert!(
                capture
                    .capture_selection((layer + 1) % LAYERS, &selected)
                    .is_err()
            );
            capture.capture_selection(layer, &selected).unwrap();
        }
        capture.completed_position().unwrap();
        assert_eq!(capture.captured.len(), LAYERS);
        assert!(capture.begin_position(POSITION + 1).is_err());
        assert_eq!(
            layer_from_name("blk.47.ffn_gate_inp.weight", ".ffn_gate_inp.weight").unwrap(),
            47
        );
        for name in [
            "blk.048.ffn_gate_inp.weight",
            "blk.48.ffn_gate_inp.weight",
            "blk.0.other.weight",
        ] {
            assert!(layer_from_name(name, ".ffn_gate_inp.weight").is_err());
        }
    }
    #[test]
    fn noncaptured_positions_keep_no_router_vectors_or_records() {
        let scores = [0.; EXPERTS];
        let selected = kuno_text::ops::router_topk(&scores, 10, true).unwrap();
        let mut capture = Capture::new(());
        for position in 0..POSITION {
            capture.begin_position(position).unwrap();
            for layer in 0..LAYERS {
                capture.capture_scores(layer, &scores).unwrap();
                assert!(capture.pending_scores.is_none());
                capture.capture_selection(layer, &selected).unwrap();
            }
            capture.completed_position().unwrap();
            assert!(capture.captured.is_empty());
        }
    }
}
