//! Compiled public sparse admission/layout fixtures. No real weights or prompts.
//! The small bound target admits only the two borrowed matrices; it does not
//! establish complete 48-layer target execution or actual Q2 prediction quality.
use super::*;
use crate::GgufLimits;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

type Tensor = (String, Vec<u64>, TensorType);

struct Fixture {
    path: PathBuf,
    offsets: BTreeMap<String, u64>,
}
impl Fixture {
    fn create(tensors: Vec<Tensor>, metadata: Vec<(String, String)>) -> Self {
        fn string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend((value.len() as u64).to_le_bytes());
            bytes.extend(value.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend((tensors.len() as u64).to_le_bytes());
        bytes.extend((metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            string(&mut bytes, &key);
            bytes.extend(8_u32.to_le_bytes());
            string(&mut bytes, &value);
        }
        let mut relative = BTreeMap::new();
        let mut extent = 0_u64;
        for (name, dimensions, dtype) in tensors {
            string(&mut bytes, &name);
            bytes.extend((dimensions.len() as u32).to_le_bytes());
            for &dimension in &dimensions {
                bytes.extend(dimension.to_le_bytes());
            }
            bytes.extend(dtype.gguf_id().to_le_bytes());
            bytes.extend(extent.to_le_bytes());
            relative.insert(name, extent);
            let rows = dimensions[1..].iter().product::<u64>();
            extent += dtype.row_bytes(dimensions[0]).unwrap() * rows;
            extent = extent.next_multiple_of(32);
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        let data = bytes.len() as u64;
        let path = std::env::temp_dir().join(format!(
            "kuno-strata-mtp-public-{}-{}.gguf",
            std::process::id(),
            NEXT_BINDING.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(data + extent).unwrap();
        Self {
            path,
            offsets: relative.into_iter().map(|(n, o)| (n, data + o)).collect(),
        }
    }
    fn metadata() -> Vec<(String, String)> {
        [
            ("general.architecture", "qwen4exp-mtp"),
            (
                "strata.mtp.source",
                "Qwen/Qwen3.8-Flash-Next BF16 checkpoint, mtp.* tensors",
            ),
            ("strata.mtp.source_sha256", SOURCE_MANIFEST_SHA256),
            ("strata.mtp.expert_format", "q2_0"),
            ("strata.mtp.expert_quantizer", "per-block MSE scale search"),
        ]
        .map(|(k, v)| (k.into(), v.into()))
        .into()
    }
    fn tensors() -> Vec<Tensor> {
        SPECS
            .iter()
            .map(|s| (s.name.into(), s.dimensions.into(), s.dtype))
            .collect()
    }
    fn strata() -> Self {
        Self::create(Self::tensors(), Self::metadata())
    }
    fn target(embedding: TensorType, head: TensorType) -> Self {
        Self::create(
            vec![
                (
                    "token_embd.weight".into(),
                    vec![D as u64, VOCAB as u64],
                    embedding,
                ),
                ("output.weight".into(), vec![D as u64, VOCAB as u64], head),
            ],
            vec![],
        )
    }
    fn open(&self) -> GgufFile {
        GgufFile::open(&self.path, GgufLimits::default()).unwrap()
    }
    fn write(&self, name: &str, relative: u64, bytes: &[u8]) {
        let mut file = OpenOptions::new().write(true).open(&self.path).unwrap();
        file.seek(SeekFrom::Start(self.offsets[name] + relative))
            .unwrap();
        file.write_all(bytes).unwrap();
    }
    fn append(&self) {
        OpenOptions::new()
            .append(true)
            .open(&self.path)
            .unwrap()
            .write_all(&[0])
            .unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn bound_target(file: &GgufFile) -> crate::model::Qwen4Exp<'_> {
    crate::model::mtp_test_target(file).unwrap()
}

#[test]
fn strict_profile_borrows_original_target_and_folds_only_static_norms_once() {
    let fixture = Fixture::strata();
    fixture.write("mtp.pre_fc_norm_hidden.weight", 0, &0.5_f32.to_le_bytes());
    let file = fixture.open();
    let target_fixture = Fixture::target(TensorType::Iq4Xs, TensorType::Q6K);
    let target_file = target_fixture.open();
    let target = bound_target(&target_file);
    let graph = MtpGraph::bind_strata_q2(&file, target.bound_graph()).unwrap();
    assert_eq!(graph.profile(), MtpProfile::StrataQ2);
    assert!(graph.target_binding_id() == Some(target.bound_graph().binding_id()));
    assert_eq!(graph.tensor_names().count(), 33);
    assert!(
        graph
            .tensor("token_embd.weight")
            .unwrap()
            .belongs_to(&target_file)
    );
    assert!(
        graph
            .tensor("output.weight")
            .unwrap()
            .belongs_to(&target_file)
    );
    assert!(
        graph
            .tensor("blk.48.attn_q.weight")
            .unwrap()
            .belongs_to(&file)
    );
    assert_eq!(
        graph.tensor("blk.48.attn_q.weight").unwrap().info().name(),
        "mtp.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(
        graph.static_vector("blk.48.nextn.hnorm.weight").unwrap()[0],
        1.5
    );
    let mut raw = vec![0.; WIDE];
    graph
        .tensor("blk.48.nextn.hnorm.weight")
        .unwrap()
        .decode_row(0, &mut raw)
        .unwrap();
    assert_eq!(raw[0], 0.5);
    assert_eq!(
        graph.static_vector("blk.48.nextn.hnorm.weight").unwrap()[0],
        1.5
    );
    for missing in [
        "blk.48.nextn.eh_proj.weight",
        "blk.48.ffn_gate_exps.weight",
        "blk.48.ffn_up_exps.weight",
        "unknown.weight",
    ] {
        assert!(graph.tensor(missing).is_err());
    }
    graph.verify_unchanged().unwrap();
    assert!(MtpGraph::bind(&file).is_err());
}

#[test]
fn admission_rejects_wrong_inventory_geometry_types_norms_and_provenance() {
    let target_fixture = Fixture::target(TensorType::Iq4Xs, TensorType::Q6K);
    let target_file = target_fixture.open();
    let target = bound_target(&target_file);
    for bad in 0..8 {
        let mut tensors = Fixture::tensors();
        let mut metadata = Fixture::metadata();
        match bad {
            0 => {
                tensors.pop();
            }
            1 => tensors.push(("extra.weight".into(), vec![1], F32)),
            2 => tensors[10].1 = vec![2560, 640, 512],
            3 => tensors[10].2 = TensorType::Q8_0,
            4 => metadata[3].1 = "q2_k".into(),
            5 => metadata[2].1 = "0".repeat(64),
            6 => metadata.push(("qwen4exp.attention.scale".into(), "1".into())),
            _ => {}
        }
        let fixture = Fixture::create(tensors, metadata);
        if bad == 7 {
            fixture.write("mtp.pre_fc_norm_hidden.weight", 0, &f32::NAN.to_le_bytes());
        }
        assert!(MtpGraph::bind_strata_q2(&fixture.open(), target.bound_graph()).is_err());
    }
    let fixture = Fixture::strata();
    let file = fixture.open();
    for (embedding, head) in [
        (TensorType::Q8_0, TensorType::Q6K),
        (TensorType::Iq4Xs, TensorType::Q8_0),
    ] {
        let bad_fixture = Fixture::target(embedding, head);
        let bad_file = bad_fixture.open();
        let bad_target = bound_target(&bad_file);
        assert!(MtpGraph::bind_strata_q2(&file, bad_target.bound_graph()).is_err());
    }
    let mut geometry = target.bound_graph().geometry();
    geometry.residual.streams = 1;
    assert!(validate_target_geometry(geometry).is_err());
    geometry = target.bound_graph().geometry();
    geometry.qsa.rope_theta = 1_000_000.;
    assert!(validate_target_geometry(geometry).is_err());
}

#[test]
fn paired_source_and_unique_binding_guards_detect_either_source_mutation() {
    for mutate_target in [false, true] {
        let fixture = Fixture::strata();
        let file = fixture.open();
        let target_fixture = Fixture::target(TensorType::Iq4Xs, TensorType::Q6K);
        let target_file = target_fixture.open();
        let target = bound_target(&target_file);
        let graph = MtpGraph::bind_strata_q2(&file, target.bound_graph()).unwrap();
        let other = MtpGraph::bind_strata_q2(&file, target.bound_graph()).unwrap();
        let mut sequence = graph.sequence(2).unwrap();
        assert!(other.validate_input(&sequence, 0, &[0.; WIDE]).is_err());
        assert_eq!(sequence.position(), 0);
        if mutate_target {
            target_fixture.append();
        } else {
            fixture.append();
        }
        assert!(matches!(graph.verify_unchanged(), Err(Error::FileChanged)));
        assert!(
            graph
                .append_kv(
                    &mut Executor::default(),
                    &mut sequence,
                    0,
                    &[0.; WIDE],
                    &AtomicBool::new(false)
                )
                .is_err()
        );
        assert!(sequence.poisoned);
        assert!(sequence.keys.is_empty() && sequence.values.is_empty());
    }
}

#[test]
fn cached_projection_source_change_before_commit_erases_completed_prefix() {
    struct Mutating<'a> {
        inner: Executor,
        source: &'a Fixture,
    }
    impl MatrixExecutor for Mutating<'_> {
        fn run(
            &mut self,
            tensor: TensorView<'_>,
            expert: usize,
            input: &[f32],
            cancel: &AtomicBool,
        ) -> Result<Vec<f32>> {
            // Simulate a cache-only projection, which never reads source bytes
            // itself. Mutate after the graph's initial paired-source guard.
            if tensor.info().name() == "mtp.layers.0.self_attn.v_proj.weight" {
                self.source.append();
            }
            self.inner.run(tensor, expert, input, cancel)
        }
    }
    for mutate_target in [false, true] {
        let fixture = Fixture::strata();
        let file = fixture.open();
        let target_fixture = Fixture::target(TensorType::Iq4Xs, TensorType::Q6K);
        let target_file = target_fixture.open();
        let target = bound_target(&target_file);
        let graph = MtpGraph::bind_strata_q2(&file, target.bound_graph()).unwrap();
        let cancel = AtomicBool::new(false);
        let mut sequence = graph.sequence(3).unwrap();
        graph
            .append_kv(
                &mut Executor::default(),
                &mut sequence,
                1,
                &[0.; WIDE],
                &cancel,
            )
            .unwrap();
        assert_eq!(sequence.position(), 1);
        let mut executor = Mutating {
            inner: Executor::default(),
            source: if mutate_target {
                &target_fixture
            } else {
                &fixture
            },
        };
        assert!(matches!(
            graph.append_kv(&mut executor, &mut sequence, 2, &[0.; WIDE], &cancel),
            Err(Error::FileChanged)
        ));
        assert!(sequence.poisoned && sequence.keys.is_empty() && sequence.values.is_empty());
        assert_eq!(sequence.position(), 0);
    }
}

#[test]
fn separate_fc_conditioning_has_five_calls_and_retains_global_stream_amplitudes() {
    let hidden: Vec<_> = [1_f32, 2., 4., 8.]
        .into_iter()
        .flat_map(|x| vec![x; D])
        .collect();
    let normalized = condition_hidden(&hidden, &[1.; WIDE]).unwrap();
    let embedding = vec![3.; D];
    let mut calls = vec![];
    let out = project_conditioned(&embedding, &normalized, |is_hidden, input| {
        calls.push(is_hidden);
        Ok(private(
            input
                .iter()
                .map(|x| x * if is_hidden { 2. } else { 0.5 })
                .collect(),
        ))
    })
    .unwrap();
    assert_eq!(calls, [false, true, true, true, true]);
    for (index, actual) in out.iter().enumerate() {
        let normalized = (f64::from(hidden[index]) / (21.25 + f64::from(EPS)).sqrt()) as f32;
        assert_eq!(actual.to_bits(), (normalized * 2. + 1.5).to_bits());
    }
    assert!(
        project_conditioned(&embedding, &normalized, |_, _| Ok(private(vec![0.; D - 1]))).is_err()
    );
    assert!(
        project_conditioned(&embedding, &normalized, |_, _| Ok(private(vec![
            f32::MAX;
            D
        ])))
        .is_err()
    );
}

#[test]
fn combined_expert_projection_keeps_gate_half_before_up_half() {
    let values: Vec<_> = (0..1280)
        .map(|i| {
            if i < 640 {
                -0.5 + i as f32 / 640.
            } else {
                1. + (i - 640) as f32 / 640.
            }
        })
        .collect();
    let out = activate_gate_up(&values).unwrap();
    for (i, actual) in out.iter().enumerate() {
        let gate = values[i];
        let expected =
            (f64::from(gate) / (1. + (-f64::from(gate)).exp())) * f64::from(values[640 + i]);
        assert!((f64::from(*actual) - expected).abs() < 1e-7);
    }
    assert!(activate_gate_up(&values[..1279]).is_err());
    let mut invalid = values;
    invalid[640] = f32::NAN;
    assert!(activate_gate_up(&invalid).is_err());
    assert!(activate_gate_up(&[f32::MAX; 1280]).is_err());
}

#[test]
fn packed_expert_rows_preserve_halves_at_boundary_expert_ids() {
    let fixture = Fixture::strata();
    for expert in [0_u64, 127, 128, 511] {
        for (row, scale, code) in [(0, 0x3c00_u16, 0x55_u8), (640, 0x4000, 0xff)] {
            let mut packed = vec![];
            for _ in 0..40 {
                packed.extend(scale.to_le_bytes());
                packed.extend([code; 16]);
            }
            fixture.write(
                "mtp.layers.0.mlp.experts.gate_up_proj",
                (expert * 1280 + row) * 720,
                &packed,
            );
        }
        let mut down = vec![];
        for _ in 0..10 {
            down.extend(0x3800_u16.to_le_bytes());
            down.extend([0xaa; 16]);
        }
        fixture.write(
            "mtp.layers.0.mlp.experts.down_proj",
            (expert * 2560 + 2559) * 180,
            &down,
        );
    }
    let file = fixture.open();
    let gate_up = file
        .tensor("mtp.layers.0.mlp.experts.gate_up_proj")
        .unwrap();
    let down = file.tensor("mtp.layers.0.mlp.experts.down_proj").unwrap();
    assert_eq!(
        (gate_up.info().row_bytes(), down.info().row_bytes()),
        (720, 180)
    );
    for expert in [0_u64, 127, 128, 511] {
        let mut row = vec![99.; D];
        gate_up.decode_row(expert * 1280, &mut row).unwrap();
        assert!(row.iter().all(|x| x.to_bits() == 0)); // code1 → signed0
        gate_up.decode_row(expert * 1280 + 640, &mut row).unwrap();
        assert!(row.iter().all(|x| *x == 4.)); // code3 → signed2 ×scale2
        let mut last = vec![99.; 640];
        down.decode_row(expert * 2560 + 2559, &mut last).unwrap();
        assert!(last.iter().all(|x| *x == 0.5)); // code2 → signed1 ×scale.5
    }
    let mut invalid = [9_f32; 64];
    assert!(TensorType::Q2_0.decode(&[0; 17], &mut invalid).is_err());
    assert_eq!(invalid, [9.; 64]);
}

#[derive(Default)]
struct Executor {
    calls: Vec<(String, usize)>,
    fail: bool,
}
impl MatrixExecutor for Executor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        _: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let name = tensor.info().name();
        self.calls.push((name.into(), expert));
        if self.fail && name == "mtp.layers.0.self_attn.v_proj.weight" {
            return Err(Error::InvalidFormat("compiled public MTP failure"));
        }
        let rows = tensor.info().dimensions()[1] as usize;
        Ok((0..rows)
            .map(|i| {
                if name == "mtp.layers.0.mlp.gate.weight" {
                    -(i as f32) * 0.125
                } else {
                    input[i % input.len()] * 0.03125
                        + (i % 17) as f32 * 0.001953125
                        + expert as f32 * 0.0009765625
                }
            })
            .collect())
    }
}

fn same_cache(a: &MtpSequence, b: &MtpSequence) {
    assert_eq!(a.position(), b.position());
    for (a, b) in [(&a.keys, &b.keys), (&a.values, &b.values)] {
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn full_and_append_paths_share_aligned_kv_and_repair_a_provisional_suffix() {
    let fixture = Fixture::strata();
    let file = fixture.open();
    let target_fixture = Fixture::target(TensorType::Iq4Xs, TensorType::Q6K);
    let target_file = target_fixture.open();
    let target = bound_target(&target_file);
    let graph = MtpGraph::bind_strata_q2(&file, target.bound_graph()).unwrap();
    let cancel = AtomicBool::new(false);
    let mut full = graph.sequence(4).unwrap();
    let mut appended = graph.sequence(4).unwrap();
    let mut full_executor = Executor::default();
    let mut append_executor = Executor::default();
    for c in 0..3 {
        // Actual schedule R(c), T(c+1) at position c; no zero/dummy cell.
        let carry: Vec<_> = (0..WIDE).map(|i| ((i + c) % 31) as f32 * 0.03125).collect();
        let output = graph
            .step(
                &mut full_executor,
                &mut full,
                (c + 1) as u32,
                &carry,
                &cancel,
            )
            .unwrap();
        assert_eq!((output.logits.len(), output.hidden.len()), (VOCAB, WIDE));
        graph
            .append_kv(
                &mut append_executor,
                &mut appended,
                (c + 1) as u32,
                &carry,
                &cancel,
            )
            .unwrap();
        same_cache(&full, &appended);
    }
    assert_eq!(
        full_executor
            .calls
            .iter()
            .filter(|(n, _)| n == "mtp.fc_embedding.weight")
            .count(),
        3
    );
    assert_eq!(
        full_executor
            .calls
            .iter()
            .filter(|(n, _)| n == "mtp.fc_hidden.weight")
            .count(),
        12
    );
    assert_eq!(
        full_executor
            .calls
            .iter()
            .filter(|(n, _)| n == "mtp.layers.0.mlp.experts.gate_up_proj")
            .count(),
        30
    );
    assert!(
        full_executor
            .calls
            .iter()
            .all(|(n, _)| !n.contains("indexer"))
    );
    assert!(
        append_executor
            .calls
            .iter()
            .all(|(n, _)| !n.contains("mlp") && n != "output.weight")
    );
    full.rewind(1).unwrap();
    let mut fresh = graph.sequence(4).unwrap();
    for c in 0..3 {
        let carry: Vec<_> = (0..WIDE).map(|i| ((i + c) % 31) as f32 * 0.03125).collect();
        graph
            .append_kv(
                &mut Executor::default(),
                &mut fresh,
                (c + 1) as u32,
                &carry,
                &cancel,
            )
            .unwrap();
        if c > 0 {
            graph
                .append_kv(
                    &mut Executor::default(),
                    &mut full,
                    (c + 1) as u32,
                    &carry,
                    &cancel,
                )
                .unwrap();
        }
    }
    same_cache(&full, &fresh);
    let mut failing = Executor {
        fail: true,
        ..Executor::default()
    };
    assert!(
        graph
            .append_kv(&mut failing, &mut full, 4, &[0.; WIDE], &cancel)
            .is_err()
    );
    assert_eq!(full.position(), 0);
    assert!(full.poisoned && full.keys.is_empty() && full.values.is_empty());
}
