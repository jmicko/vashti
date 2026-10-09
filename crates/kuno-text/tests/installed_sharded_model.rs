//! Explicit opt-in checks of the pinned complete public Unsloth Q3 artifact.
//! No arbitrary prompt, generation, external engine or GPU is involved.
use kuno_text::model::Qwen4Exp;
use kuno_text::{GgufFile, GgufLimits, Qwen35Tokenizer, TensorType, TokenizerLimits};
use std::collections::BTreeSet;

#[test]
#[ignore = "requires all three verified public UD-Q3_K_XL shards via KUNO_TEST_Q3_DIR"]
fn actual_complete_q3_binds_tokenizer_graph_and_bounded_rows() {
    let dir = std::path::PathBuf::from(
        std::env::var_os("KUNO_TEST_Q3_DIR").expect("KUNO_TEST_Q3_DIR is required"),
    );
    let paths: Vec<_> = (1..=3)
        .map(|i| {
            dir.join(format!(
                "Qwen3.8-Flash-Next-UD-Q3_K_XL-{i:05}-of-00003.gguf"
            ))
        })
        .collect();
    let file = GgufFile::open_shards(&paths, GgufLimits::default()).unwrap();
    assert_eq!(file.file_len(), 89_986_353_824);
    assert_eq!(file.tensors().len(), 1224);
    assert!(
        file.tensors()
            .values()
            .all(|t| t.shard_index() == 1 || t.shard_index() == 2)
    );
    let graph = Qwen4Exp::bind(&file).unwrap();
    assert_eq!(graph.bound_graph().geometry().layers, 48);
    let tokenizer = Qwen35Tokenizer::from_gguf(&file, TokenizerLimits::default()).unwrap();
    assert_eq!(tokenizer.vocabulary_len(), 248320);
    assert_eq!(tokenizer.eos_token_id(), 248046);
    let mut types = BTreeSet::new();
    for info in file.tensors().values() {
        if !types.insert(info.dtype().gguf_id()) {
            continue;
        }
        let mut row = vec![0.; info.row_len() as usize];
        file.tensor(info.name())
            .unwrap()
            .decode_row(0, &mut row)
            .unwrap();
        assert!(row.iter().all(|v| v.is_finite()));
    }
    assert_eq!(types, BTreeSet::from([0, 8, 14, 18, 20, 23, 30]));
    let ple = file.tensor("per_layer_token_embd.weight").unwrap();
    assert_eq!(ple.info().dtype(), TensorType::Iq4Nl);
    assert_eq!(ple.info().row_bytes(), 90);
    file.verify_file_unchanged().unwrap();
}
