use std::collections::BTreeMap;

use kuno_text::{Error, MetadataValue, Qwen35Tokenizer, TokenizerLimits, Utf8StreamDecoder};
use serde_json::Value;

fn value(value: &Value) -> MetadataValue {
    match value {
        Value::Bool(value) => MetadataValue::Bool(*value),
        Value::String(value) => MetadataValue::String(value.clone()),
        Value::Number(value) => MetadataValue::Unsigned(value.as_u64().unwrap()),
        Value::Array(values) => MetadataValue::Array(values.iter().map(self::value).collect()),
        _ => panic!("invalid synthetic fixture value"),
    }
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/qwen35-python.json")).unwrap()
}

fn metadata() -> BTreeMap<String, MetadataValue> {
    fixture()["metadata"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(key, item)| (key.clone(), value(item)))
        .collect()
}

fn tokenizer(limits: TokenizerLimits) -> Qwen35Tokenizer {
    Qwen35Tokenizer::from_metadata(&metadata(), limits).unwrap()
}

#[test]
fn exact_ids_match_independent_pinned_python_reference_fixtures() {
    let fixture = fixture();
    let tokenizer = tokenizer(TokenizerLimits::default());
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let expected: Vec<u32> = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect();
        let actual = tokenizer
            .encode(
                case["input"].as_str().unwrap(),
                case["parse_special"].as_bool().unwrap(),
            )
            .unwrap();
        assert_eq!(actual, expected, "synthetic reference case {index}");
        assert_eq!(
            tokenizer.decode(&actual).unwrap(),
            case["input"].as_str().unwrap(),
            "synthetic roundtrip case {index}"
        );
    }
}

#[test]
fn omitted_optional_bpe_boundary_flags_preserve_exact_raw_encoding() {
    let mut metadata = metadata();
    metadata.remove("tokenizer.ggml.add_bos_token");
    metadata.remove("tokenizer.ggml.add_eos_token");
    let tokenizer = Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).unwrap();
    for case in fixture()["cases"].as_array().unwrap() {
        let expected = case["expected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        assert_eq!(
            tokenizer
                .encode(
                    case["input"].as_str().unwrap(),
                    case["parse_special"].as_bool().unwrap()
                )
                .unwrap(),
            expected
        );
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        metadata.insert(key.into(), MetadataValue::Bool(true));
        assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
        metadata.remove(key);
    }
}

#[test]
fn control_is_opt_in_user_defined_is_always_matched_and_longest_wins() {
    let tokenizer = tokenizer(TokenizerLimits::default());
    assert!(tokenizer.encode("", true).unwrap().is_empty());
    assert_eq!(
        tokenizer.encode("<|test|>", true).unwrap(),
        [tokenizer.bos_token_id()]
    );
    assert!(tokenizer.encode("<|test|>", false).unwrap().len() > 1);
    let short = tokenizer.encode("<tool>", false).unwrap();
    let long = tokenizer.encode("<tool>extra", false).unwrap();
    assert_eq!(short.len(), 1);
    assert_eq!(long.len(), 1);
    assert_ne!(short, long);
    assert_eq!(tokenizer.token_bytes(long[0]).unwrap(), b"<tool>extra");
    assert!(tokenizer.encode("<reserved>", true).unwrap().len() > 1);
}

#[test]
fn combining_marks_digit_boundaries_and_rank_priority_do_not_silently_change_ids() {
    let tokenizer = tokenizer(TokenizerLimits::default());
    // These complete tokens exist in the vocabulary. The regex still has to
    // prevent merges across single-digit pretoken boundaries.
    assert_eq!(tokenizer.encode("12", false).unwrap(), [49, 50]);
    assert_eq!(tokenizer.encode("_j\u{301}", false).unwrap().len(), 1);
    assert_eq!(tokenizer.encode("abc", false).unwrap().len(), 1);
    assert_eq!(tokenizer.encode("abab", false).unwrap().len(), 1);
    assert_eq!(tokenizer.encode("ababab", false).unwrap().len(), 2);
}

#[test]
fn all_byte_tokens_decode_without_assuming_individual_utf8_validity() {
    let tokenizer = tokenizer(TokenizerLimits::default());
    for byte in 0_u32..256 {
        assert_eq!(tokenizer.token_bytes(byte).unwrap(), &[byte as u8]);
    }
    assert_eq!(tokenizer.decode(&[0xe2, 0x82, 0xac]).unwrap(), "€");
    assert_eq!(tokenizer.decode(&[0xf0, 0x9f, 0x8c, 0x88]).unwrap(), "🌈");
    assert!(matches!(
        tokenizer.token_bytes(u32::MAX),
        Err(Error::OutOfBounds)
    ));
}

#[test]
fn signed_integer_token_types_from_real_gguf_are_accepted_but_negatives_are_rejected() {
    let mut metadata = metadata();
    let MetadataValue::Array(types) = metadata.get_mut("tokenizer.ggml.token_type").unwrap() else {
        unreachable!()
    };
    for kind in types.iter_mut() {
        *kind = MetadataValue::Signed(kind.as_u64().unwrap() as i64);
    }
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_ok());
    let MetadataValue::Array(types) = metadata.get_mut("tokenizer.ggml.token_type").unwrap() else {
        unreachable!()
    };
    types[0] = MetadataValue::Signed(-1);
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
}

#[test]
fn stream_decode_retains_only_incomplete_tails_and_replaces_invalid_sequences() {
    let mut decoder = Utf8StreamDecoder::new(1024);
    assert_eq!(decoder.push_bytes(&[0xe2]).unwrap(), "");
    assert_eq!(decoder.push_bytes(&[0x82]).unwrap(), "");
    assert_eq!(decoder.push_bytes(&[0xac]).unwrap(), "€");
    assert_eq!(
        decoder.push_bytes(&[0xff, b'a', 0xf0, 0x90]).unwrap(),
        "\u{fffd}a"
    );
    assert_eq!(decoder.push_bytes(b"b").unwrap(), "\u{fffd}b");
    assert_eq!(decoder.push_bytes(&[0xf0, 0x9f]).unwrap(), "");
    assert_eq!(decoder.finish(), "\u{fffd}");
    assert_eq!(decoder.finish(), "");
    assert_eq!(decoder.push_bytes(&[0xe2]).unwrap(), "");
    decoder.clear();
    assert_eq!(decoder.push_bytes(b"x").unwrap(), "x");
}

#[test]
fn streamed_replacement_matches_utf8_lossy_across_every_synthetic_chunk_boundary() {
    let bytes = [
        b'a', 0xe2, 0x82, 0xac, 0xff, 0xf0, 0x9f, 0x8c, 0x88, 0xc0, 0x80, b'z', 0xf0, 0x90,
    ];
    let expected = String::from_utf8_lossy(&bytes);
    for first in 0..=bytes.len() {
        for second in first..=bytes.len() {
            let mut decoder = Utf8StreamDecoder::new(1024);
            let mut actual = decoder.push_bytes(&bytes[..first]).unwrap();
            actual.push_str(&decoder.push_bytes(&bytes[first..second]).unwrap());
            actual.push_str(&decoder.push_bytes(&bytes[second..]).unwrap());
            actual.push_str(&decoder.finish());
            assert_eq!(actual, expected, "chunk boundaries {first},{second}");
        }
    }
}

#[test]
fn input_piece_token_merge_and_decode_limits_fail_closed() {
    let limited = tokenizer(TokenizerLimits {
        max_input_bytes: 2,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.encode("abc", false),
        Err(Error::LimitExceeded("tokenizer input bytes"))
    ));
    let limited = tokenizer(TokenizerLimits {
        max_piece_bytes: 2,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.encode("abc", false),
        Err(Error::LimitExceeded("tokenizer pretoken bytes"))
    ));
    let limited = tokenizer(TokenizerLimits {
        max_output_tokens: 1,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.encode("12", false),
        Err(Error::LimitExceeded("tokenizer output tokens"))
    ));
    let limited = tokenizer(TokenizerLimits {
        max_merge_steps: 0,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.encode("abc", false),
        Err(Error::LimitExceeded("tokenizer merge steps"))
    ));
    let limited = tokenizer(TokenizerLimits {
        max_decoded_bytes: 2,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.decode(&[0xe2, 0x82, 0xac]),
        Err(Error::LimitExceeded("tokenizer decoded bytes"))
    ));
    let mut decoder = Utf8StreamDecoder::new(1);
    assert!(matches!(
        decoder.push_bytes(b"ab"),
        Err(Error::LimitExceeded("tokenizer decode chunk bytes"))
    ));
    let limited = tokenizer(TokenizerLimits {
        regex_backtrack_limit: 0,
        ..TokenizerLimits::default()
    });
    assert!(matches!(
        limited.encode("   a", false),
        Err(Error::LimitExceeded("tokenizer regex execution"))
    ));
}

#[test]
fn missing_metadata_unknown_variants_implicit_bos_and_invalid_ids_are_rejected() {
    for key in [
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.eos_token_id",
    ] {
        let mut metadata = metadata();
        metadata.remove(key);
        assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
    }
    let mut metadata = metadata();
    metadata.insert(
        "tokenizer.ggml.pre".into(),
        MetadataValue::String("qwen2".into()),
    );
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
    metadata.insert(
        "tokenizer.ggml.pre".into(),
        MetadataValue::String("qwen35".into()),
    );
    metadata.insert(
        "tokenizer.ggml.add_bos_token".into(),
        MetadataValue::Bool(true),
    );
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
    metadata.insert(
        "tokenizer.ggml.add_bos_token".into(),
        MetadataValue::Bool(false),
    );
    metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Unsigned(u64::MAX),
    );
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
}

#[test]
fn construction_budgets_and_malformed_vocabularies_and_merges_are_rejected() {
    for limits in [
        TokenizerLimits {
            max_vocabulary: 255,
            ..TokenizerLimits::default()
        },
        TokenizerLimits {
            max_merges: 0,
            ..TokenizerLimits::default()
        },
        TokenizerLimits {
            max_allocation_bytes: 1,
            ..TokenizerLimits::default()
        },
        TokenizerLimits {
            max_special_tokens: 0,
            ..TokenizerLimits::default()
        },
        TokenizerLimits {
            max_token_bytes: 1,
            ..TokenizerLimits::default()
        },
    ] {
        assert!(matches!(
            Qwen35Tokenizer::from_metadata(&metadata(), limits),
            Err(Error::LimitExceeded(_))
        ));
    }
    let mut metadata = metadata();
    let MetadataValue::Array(tokens) = metadata.get_mut("tokenizer.ggml.tokens").unwrap() else {
        unreachable!()
    };
    tokens[0] = tokens[1].clone();
    assert!(Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()).is_err());
    let mut metadata = self::metadata();
    let MetadataValue::Array(merges) = metadata.get_mut("tokenizer.ggml.merges").unwrap() else {
        unreachable!()
    };
    merges.push(merges[0].clone());
    assert!(matches!(
        Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()),
        Err(Error::InvalidFormat("duplicate ranked merge pair"))
    ));
    let mut metadata = self::metadata();
    let MetadataValue::Array(types) = metadata.get_mut("tokenizer.ggml.token_type").unwrap() else {
        unreachable!()
    };
    types[0] = MetadataValue::Unsigned(6);
    assert!(matches!(
        Qwen35Tokenizer::from_metadata(&metadata, TokenizerLimits::default()),
        Err(Error::InvalidFormat("unsupported tokenizer token type"))
    ));
}
