//! Compiled public synthetic graph tests. The sparse fixture is deliberately
//! not a strict sidecar admission or an independent full-head numeric oracle.
use super::*;
use std::fs::{self, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        fn string(bytes: &mut Vec<u8>, text: &str) {
            bytes.extend((text.len() as u64).to_le_bytes());
            bytes.extend(text.as_bytes());
        }
        // Real front dimensions, only four compiled embedding rows, ten routed
        // experts and a small output. Projection payloads are sparse holes:
        // the deterministic executor never reads them. Only the public fixed
        // embedding rows are read by the real conditioner.
        let tensors: &[(&str, &[u64])] = &[
            ("token_embd.weight", &[2560, 4]),
            ("blk.48.nextn.eh_proj.weight", &[5120, 2560]),
            ("blk.48.hc_attn_down.weight", &[10240, 320]),
            ("blk.48.hc_attn_up.weight", &[320, 10240]),
            ("blk.48.hc_attn_inject.weight", &[10240, 4]),
            ("blk.48.attn_q.weight", &[2560, 12288]),
            ("blk.48.attn_k.weight", &[2560, 512]),
            ("blk.48.attn_v.weight", &[2560, 512]),
            ("blk.48.attn_output.weight", &[6144, 2560]),
            ("blk.48.hc_ffn_down.weight", &[10240, 320]),
            ("blk.48.hc_ffn_up.weight", &[320, 10240]),
            ("blk.48.hc_ffn_inject.weight", &[10240, 4]),
            ("blk.48.ffn_gate_inp.weight", &[2560, 512]),
            ("blk.48.ffn_gate_exps.weight", &[2560, 640, 10]),
            ("blk.48.ffn_up_exps.weight", &[2560, 640, 10]),
            ("blk.48.ffn_down_exps.weight", &[640, 2560, 10]),
            ("blk.48.ffn_gate_shexp.weight", &[2560, 640]),
            ("blk.48.ffn_up_shexp.weight", &[2560, 640]),
            ("blk.48.ffn_down_shexp.weight", &[640, 2560]),
            ("blk.48.nextn.hc_head_down.weight", &[10240, 320]),
            ("blk.48.nextn.hc_head_up.weight", &[320, 10240]),
            ("output.weight", &[2560, 8]),
        ];
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3_u32.to_le_bytes());
        bytes.extend((tensors.len() as u64).to_le_bytes());
        bytes.extend(0_u64.to_le_bytes());
        let mut offset = 0_u64;
        for &(name, dimensions) in tensors {
            string(&mut bytes, name);
            bytes.extend((dimensions.len() as u32).to_le_bytes());
            for &dimension in dimensions {
                bytes.extend(dimension.to_le_bytes());
            }
            bytes.extend(0_u32.to_le_bytes()); // F32, never a quant-format test
            bytes.extend(offset.to_le_bytes());
            offset += 4 * dimensions.iter().product::<u64>();
            offset = offset.next_multiple_of(32);
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        let data_offset = bytes.len() as u64;
        let path = std::env::temp_dir().join(format!(
            "kuno-mtp-kv-public-{}-{}.gguf",
            std::process::id(),
            NEXT_BINDING.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(&bytes).unwrap();
        file.set_len(data_offset + offset).unwrap();
        file.seek(SeekFrom::Start(data_offset)).unwrap();
        for token in 0..4 {
            for column in 0..D {
                let value = ((column * (token + 3) % 23) as f32 - 11.) * 0.125;
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
        Self(path)
    }
    fn open(&self) -> GgufFile {
        GgufFile::open(&self.0, crate::GgufLimits::default()).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn graph(file: &GgufFile, binding: u64) -> MtpGraph<'_> {
    let mut vectors = BTreeMap::new();
    for (name, width) in [
        ("blk.48.nextn.enorm.weight", D),
        ("blk.48.nextn.hnorm.weight", WIDE),
        ("blk.48.hc_attn_norm.weight", WIDE),
        ("blk.48.attn_q_norm.weight", 256),
        ("blk.48.attn_k_norm.weight", 256),
        ("blk.48.hc_ffn_norm.weight", WIDE),
        ("blk.48.nextn.hc_head_norm.weight", WIDE),
    ] {
        vectors.insert(name.into(), vec![1.; width]);
    }
    vectors.insert("blk.48.ffn_gate_inp_shexp.weight".into(), vec![0.01; D]);
    MtpGraph {
        file,
        vectors,
        names: file.tensors().keys().cloned().collect(),
        binding,
    }
}

#[derive(Default)]
struct Executor {
    calls: Vec<String>,
    fail: Option<&'static str>,
    cancel_at: Option<&'static str>,
    invalid: Option<&'static str>,
}
impl MatrixExecutor for Executor {
    fn run(
        &mut self,
        tensor: TensorView<'_>,
        expert: usize,
        input: &[f32],
        cancel: &AtomicBool,
    ) -> Result<Vec<f32>> {
        let name = tensor.info().name();
        self.calls.push(name.into());
        if self.fail == Some(name) {
            return Err(Error::InvalidFormat("compiled public projection failure"));
        }
        if self.cancel_at == Some(name) {
            cancel.store(true, Ordering::Relaxed);
        }
        let mut output: Vec<_> = (0..tensor.info().dimensions()[1] as usize)
            .map(|i| {
                if name == "blk.48.ffn_gate_inp.weight" {
                    -(i as f32) * 0.125 // original top-k selects these ten experts
                } else {
                    input[i % input.len()] * 0.03125
                        + input[(i * 13 + 7) % input.len()] * 0.015625
                        + ((i % 17) as f32 - 8.) * 0.001953125
                        + expert as f32 * 0.0009765625
                }
            })
            .collect();
        if self.invalid == Some(name) {
            output[0] = f32::NAN;
        }
        Ok(output)
    }
}

fn hidden(position: usize) -> Vec<f32> {
    (0..WIDE)
        .map(|i| ((i * (position + 7) % 31) as f32 - 15.) * 0.0625)
        .collect()
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}
fn same_cache(a: &MtpSequence, b: &MtpSequence) {
    assert_eq!(a.position(), b.position());
    assert_eq!(bits(&a.keys), bits(&b.keys));
    assert_eq!(bits(&a.values), bits(&b.values));
    assert_eq!(
        (a.keys.len(), a.values.len()),
        (a.position() * 512, a.position() * 512)
    );
    assert!(!a.poisoned && !b.poisoned);
}

#[test]
fn append_matches_full_step_cache_bits_and_skips_all_output_work() {
    let fixture = Fixture::new();
    let file = fixture.open();
    let graph = graph(&file, 1);
    let mut full = graph.sequence(4).unwrap();
    let mut cache = graph.sequence(4).unwrap();
    let mut full_executor = Executor::default();
    let mut cache_executor = Executor::default();
    let cancel = AtomicBool::new(false);
    for position in 0..3 {
        graph
            .step(
                &mut full_executor,
                &mut full,
                position as u32,
                &hidden(position),
                &cancel,
            )
            .unwrap();
        graph
            .append_kv(
                &mut cache_executor,
                &mut cache,
                position as u32,
                &hidden(position),
                &cancel,
            )
            .unwrap();
        same_cache(&full, &cache);
    }
    assert_eq!(full_executor.calls.len(), 3 * 51);
    assert_eq!(cache_executor.calls.len(), 3 * 8);
    assert!(cache_executor.calls.iter().all(|n| matches!(
        n.as_str(),
        "blk.48.nextn.eh_proj.weight"
            | "blk.48.hc_attn_down.weight"
            | "blk.48.hc_attn_up.weight"
            | "blk.48.attn_k.weight"
            | "blk.48.attn_v.weight"
    )));
    assert!(cache.keys.iter().any(|x| x.abs() > 0.01));
    assert_ne!(bits(&cache.keys[..512]), bits(&cache.keys[512..1024]));
    // A full step following cache-only preparation sees exactly the same
    // causal context and produces identical outputs, not just matching lengths.
    let expected = graph
        .step(&mut full_executor, &mut full, 3, &hidden(3), &cancel)
        .unwrap();
    let actual = graph
        .step(&mut cache_executor, &mut cache, 3, &hidden(3), &cancel)
        .unwrap();
    same_cache(&full, &cache);
    assert_eq!(bits(&expected.hidden), bits(&actual.hidden));
    assert_eq!(bits(&expected.logits), bits(&actual.logits));
}

#[test]
fn rewind_and_teacher_forced_repair_replace_provisional_conditioning() {
    let fixture = Fixture::new();
    let file = fixture.open();
    let graph = graph(&file, 1);
    let cancel = AtomicBool::new(false);
    let mut sequence = graph.sequence(4).unwrap();
    let mut executor = Executor::default();
    for position in 0..3 {
        graph
            .step(
                &mut executor,
                &mut sequence,
                position as u32,
                &hidden(position),
                &cancel,
            )
            .unwrap();
    }
    let committed_keys = bits(&sequence.keys[..512]);
    let provisional_keys = bits(&sequence.keys[512..]);
    sequence.rewind(1).unwrap();
    assert_eq!(bits(&sequence.keys), committed_keys);
    assert_eq!((sequence.keys.len(), sequence.values.len()), (512, 512));
    let mut fresh = graph.sequence(4).unwrap();
    graph
        .step(&mut Executor::default(), &mut fresh, 0, &hidden(0), &cancel)
        .unwrap();
    for position in 1..3 {
        // Same token with a different real target carry must replace draft KV.
        let repaired = hidden(position + 4);
        graph
            .append_kv(
                &mut executor,
                &mut sequence,
                position as u32,
                &repaired,
                &cancel,
            )
            .unwrap();
        graph
            .step(
                &mut Executor::default(),
                &mut fresh,
                position as u32,
                &repaired,
                &cancel,
            )
            .unwrap();
        same_cache(&sequence, &fresh);
    }
    assert_ne!(bits(&sequence.keys[512..]), provisional_keys);
    sequence.reset();
    assert_eq!(
        (
            sequence.position(),
            sequence.keys.len(),
            sequence.values.len()
        ),
        (0, 0, 0)
    );
    graph
        .append_kv(&mut executor, &mut sequence, 0, &hidden(0), &cancel)
        .unwrap();
    assert_eq!(bits(&sequence.keys), committed_keys);
}

#[test]
fn append_preflight_errors_preserve_completed_prefix_without_projection() {
    let fixture = Fixture::new();
    let file = fixture.open();
    let foreign = graph(&file, 2);
    let graph = graph(&file, 1);
    let mut sequence = graph.sequence(2).unwrap();
    let mut executor = Executor::default();
    let cancel = AtomicBool::new(false);
    let h = hidden(0);
    graph
        .append_kv(&mut executor, &mut sequence, 0, &h, &cancel)
        .unwrap();
    let snapshot = (
        bits(&sequence.keys),
        bits(&sequence.values),
        sequence.position(),
    );
    let calls = executor.calls.len();
    let mut invalid = h.clone();
    invalid[17] = f32::INFINITY;
    for (owner, token, input) in [
        (&foreign, 0, h.as_slice()),
        (&graph, VOCAB as u32, h.as_slice()),
        (&graph, 0, &h[..WIDE - 1]),
        (&graph, 0, invalid.as_slice()),
    ] {
        assert!(
            owner
                .append_kv(&mut executor, &mut sequence, token, input, &cancel)
                .is_err()
        );
        assert_eq!(
            (
                bits(&sequence.keys),
                bits(&sequence.values),
                sequence.position()
            ),
            snapshot
        );
        assert!(!sequence.poisoned);
    }
    assert_eq!(executor.calls.len(), calls);
    graph
        .append_kv(&mut executor, &mut sequence, 1, &hidden(1), &cancel)
        .unwrap();
    let snapshot = (bits(&sequence.keys), bits(&sequence.values));
    let calls = executor.calls.len();
    assert!(
        graph
            .append_kv(&mut executor, &mut sequence, 2, &hidden(2), &cancel)
            .is_err()
    );
    assert_eq!((bits(&sequence.keys), bits(&sequence.values)), snapshot);
    assert_eq!(sequence.position(), 2);
    assert_eq!(executor.calls.len(), calls);
}

#[test]
fn append_execution_failure_and_cancellation_clear_and_poison_entire_cache() {
    let fixture = Fixture::new();
    let file = fixture.open();
    let graph = graph(&file, 1);
    let h = hidden(0);
    for mode in 0..4 {
        let cancel = AtomicBool::new(false);
        let mut sequence = graph.sequence(3).unwrap();
        graph
            .append_kv(&mut Executor::default(), &mut sequence, 0, &h, &cancel)
            .unwrap();
        let mut executor = Executor::default();
        match mode {
            0 => executor.fail = Some("blk.48.attn_v.weight"),
            1 => executor.invalid = Some("blk.48.attn_k.weight"),
            2 => executor.cancel_at = Some("blk.48.attn_v.weight"),
            _ => cancel.store(true, Ordering::Relaxed),
        }
        assert!(
            graph
                .append_kv(&mut executor, &mut sequence, 1, &hidden(1), &cancel)
                .is_err()
        );
        assert_eq!(
            (
                sequence.position(),
                sequence.keys.len(),
                sequence.values.len()
            ),
            (0, 0, 0)
        );
        assert!(sequence.poisoned);
        assert!(sequence.rewind(0).is_err());
        cancel.store(false, Ordering::Relaxed);
        let mut recovered = Executor::default();
        assert!(
            graph
                .append_kv(&mut recovered, &mut sequence, 0, &h, &cancel)
                .is_err()
        );
        assert!(recovered.calls.is_empty());
        sequence.reset();
        graph
            .append_kv(&mut recovered, &mut sequence, 0, &h, &cancel)
            .unwrap();
        let mut fresh = graph.sequence(3).unwrap();
        graph
            .step(&mut Executor::default(), &mut fresh, 0, &h, &cancel)
            .unwrap();
        same_cache(&sequence, &fresh);
    }
}

#[test]
fn full_step_failure_after_shared_kv_publication_keeps_original_poison_rule() {
    let fixture = Fixture::new();
    let file = fixture.open();
    let graph = graph(&file, 1);
    let mut sequence = graph.sequence(2).unwrap();
    let mut executor = Executor {
        fail: Some("blk.48.attn_output.weight"),
        ..Executor::default()
    };
    let result = graph.step(
        &mut executor,
        &mut sequence,
        0,
        &hidden(0),
        &AtomicBool::new(false),
    );
    assert!(result.is_err());
    assert!(executor.calls.iter().any(|n| n == "blk.48.attn_v.weight"));
    assert_eq!(
        (
            sequence.position(),
            sequence.keys.len(),
            sequence.values.len()
        ),
        (0, 0, 0)
    );
    assert!(sequence.poisoned);
}

#[test]
fn shared_completion_rejects_error_after_cache_append_without_half_commit() {
    let mut sequence = MtpSequence::new(2, 1).unwrap();
    sequence.poisoned = true;
    let mut key = vec![1.; 512];
    append_key_value(&mut sequence, &mut key, &[2.; 512]).unwrap();
    assert_eq!(
        (
            sequence.keys.len(),
            sequence.values.len(),
            sequence.position()
        ),
        (512, 512, 0)
    );
    assert!(
        sequence
            .complete::<()>(Err(Error::InvalidFormat("MTP cancelled")))
            .is_err()
    );
    assert_eq!(
        (
            sequence.keys.len(),
            sequence.values.len(),
            sequence.position()
        ),
        (0, 0, 0)
    );
    assert!(sequence.poisoned);
}
