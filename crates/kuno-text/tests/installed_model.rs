use std::collections::BTreeSet;

use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TensorType, TokenizerLimits};

/// Deliberately ignored in normal CI: the artifact is not downloaded/copied by
/// this test. Only public model geometry and individually bounded rows are read;
/// raw metadata strings, token arrays and weight values are never printed.
#[test]
#[ignore = "requires an existing Qwen3.8-Flash-Next mixed-format GGUF via KUNO_TEST_GGUF"]
fn installed_qwen_geometry_and_individual_rows() {
    let path = std::env::var_os("KUNO_TEST_GGUF").expect("KUNO_TEST_GGUF is required");
    let limits = GgufLimits {
        max_tensor_read_bytes: 1024 * 1024,
        ..GgufLimits::default()
    };
    let model = GgufFile::open(path, limits).unwrap();
    assert_eq!(
        model.metadata()["general.architecture"].as_str(),
        Some("qwen4exp")
    );
    let ple = model.tensor("per_layer_token_embd.weight").unwrap();
    assert_eq!(ple.info().dimensions(), [160, 320001536]);
    assert_eq!(ple.info().dtype(), TensorType::Q5_0);
    assert_eq!(ple.info().row_bytes(), 110);
    let mut decoded = [0.0; 160];
    ple.decode_row(0, &mut decoded).unwrap();
    assert!(decoded.iter().all(|value| value.is_finite()));

    let mut formats = BTreeSet::new();
    for info in model.tensors().values() {
        if formats.contains(&info.dtype().gguf_id()) {
            continue;
        }
        assert!(info.row_bytes() <= 1024 * 1024);
        assert!(info.row_len() <= 262_144);
        let mut decoded = vec![0.0; info.row_len() as usize];
        model
            .tensor(info.name())
            .unwrap()
            .decode_row(0, &mut decoded)
            .unwrap();
        assert!(decoded.iter().all(|value| value.is_finite()));
        formats.insert(info.dtype().gguf_id());
    }
    assert_eq!(formats, BTreeSet::from([0, 1, 6, 8, 12, 14, 30]));
    model.verify_file_unchanged().unwrap();
}

#[test]
#[ignore = "requires existing target GGUF and public raw fixture via KUNO_TEST_GGUF/KUNO_TEST_BASELINE"]
fn installed_tokenizer_matches_ollama_raw_baseline_token_counts() {
    let path = std::env::var_os("KUNO_TEST_GGUF").expect("KUNO_TEST_GGUF is required");
    let fixture = std::env::var_os("KUNO_TEST_BASELINE").expect("KUNO_TEST_BASELINE is required");
    let model = GgufFile::open(path, GgufLimits::default()).unwrap();
    let tokenizer = Qwen35Tokenizer::from_gguf(&model, TokenizerLimits::default()).unwrap();
    assert_eq!(tokenizer.vocabulary_len(), 248320);
    assert_eq!(tokenizer.eos_token_id(), 248046);
    assert_eq!(tokenizer.bos_token_id(), 248044);
    let fixture: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(fixture).unwrap()).unwrap();
    assert_eq!(fixture["fixture_kind"], "public_synthetic_benchmark");
    for (key, count) in [("prompt", 75), ("alternate_prompt", 80)] {
        let input = fixture[key].as_str().unwrap();
        let encoded = tokenizer.encode(input, true).unwrap();
        assert_eq!(encoded.len(), count);
        assert_eq!(tokenizer.decode(&encoded).unwrap(), input);
    }
}
