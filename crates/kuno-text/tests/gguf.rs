use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use kuno_text::{Error, GgufFile, GgufLimits, MetadataValue, TensorType};

static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);

impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "kuno-text-fixture-{}-{}.gguf",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        file.write_all(bytes).unwrap();
        Self(path)
    }

    fn open(&self) -> kuno_text::Result<GgufFile> {
        GgufFile::open(&self.0, GgufLimits::default())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct Tensor<'a> {
    name: &'a str,
    dimensions: Vec<u64>,
    kind: u32,
    offset: u64,
    bytes: Vec<u8>,
}

fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}

fn header(version: u32, tensors: u64, entries: u64) -> Vec<u8> {
    let mut result = b"GGUF".to_vec();
    result.extend(version.to_le_bytes());
    result.extend(tensors.to_le_bytes());
    result.extend(entries.to_le_bytes());
    result
}

fn model(tensors: &[Tensor<'_>]) -> Vec<u8> {
    let mut result = header(3, tensors.len() as u64, 3);
    string(&mut result, "general.architecture");
    result.extend(8_u32.to_le_bytes());
    string(&mut result, "qwen4exp");
    string(&mut result, "general.alignment");
    result.extend(4_u32.to_le_bytes());
    result.extend(32_u32.to_le_bytes());
    string(&mut result, "fixture.tokens");
    result.extend(9_u32.to_le_bytes());
    result.extend(8_u32.to_le_bytes());
    result.extend(3_u64.to_le_bytes());
    for token in ["first", "second", "☀"] {
        string(&mut result, token);
    }
    for tensor in tensors {
        string(&mut result, tensor.name);
        result.extend((tensor.dimensions.len() as u32).to_le_bytes());
        for dimension in &tensor.dimensions {
            result.extend(dimension.to_le_bytes());
        }
        result.extend(tensor.kind.to_le_bytes());
        result.extend(tensor.offset.to_le_bytes());
    }
    let data_offset = result.len().next_multiple_of(32);
    result.resize(data_offset, 0);
    for tensor in tensors {
        let start = data_offset + tensor.offset as usize;
        result.resize(result.len().max(start + tensor.bytes.len()), 0);
        result[start..start + tensor.bytes.len()].copy_from_slice(&tensor.bytes);
    }
    result
}

fn ple() -> Tensor<'static> {
    let mut bytes = Vec::new();
    for row in 0..3 {
        for _ in 0..5 {
            bytes.extend(0x3c00_u16.to_le_bytes());
            bytes.extend(u32::MAX.to_le_bytes());
            bytes.extend([row | row << 4; 16]);
        }
    }
    Tensor {
        name: "per_layer_token_embd.weight",
        dimensions: vec![160, 3],
        kind: 6,
        offset: 0,
        bytes,
    }
}

#[test]
fn reads_metadata_and_only_requested_q5_0_ple_rows() {
    let fixture = Fixture::new(&model(&[ple()]));
    let model = fixture.open().unwrap();
    assert_eq!(model.version(), 3);
    assert_eq!(
        model.metadata()["general.architecture"].as_str(),
        Some("qwen4exp")
    );
    let tokens = model.metadata()["fixture.tokens"].as_array().unwrap();
    assert_eq!(tokens[2].as_str(), Some("☀"));
    let tensor = model.tensor("per_layer_token_embd.weight").unwrap();
    let info = tensor.info();
    assert_eq!(info.dimensions(), [160, 3]);
    assert_eq!(info.dtype(), TensorType::Q5_0);
    assert_eq!(info.row_bytes(), 110);
    assert_eq!(info.row_count(), 3);
    assert_eq!(info.byte_len(), 330);
    assert_eq!(info.element_count(), 480);
    let mut decoded = [0.0; 160];
    tensor.decode_row(2, &mut decoded).unwrap();
    assert_eq!(decoded, [2.0; 160]);
    let mut packed = [0; 110];
    tensor.read_row(1, &mut packed).unwrap();
    assert_eq!(&packed[..6], &[0, 0x3c, 0xff, 0xff, 0xff, 0xff]);
    assert_eq!(&packed[6..22], &[0x11; 16]);
    model.verify_file_unchanged().unwrap();
}

#[test]
fn mixed_tensor_types_have_independent_aligned_ranges() {
    let floats = [1.0_f32, 2.0, 3.0, 4.0];
    let first = Tensor {
        name: "first",
        dimensions: vec![2, 2],
        kind: 0,
        offset: 0,
        bytes: floats.into_iter().flat_map(f32::to_le_bytes).collect(),
    };
    let mut second_bytes = vec![0x00, 0x3c];
    second_bytes.extend([3; 32]);
    let second = Tensor {
        name: "second",
        dimensions: vec![32, 1],
        kind: 8,
        offset: 32,
        bytes: second_bytes,
    };
    let fixture = Fixture::new(&model(&[first, second]));
    let model = fixture.open().unwrap();
    assert_eq!(model.tensors().len(), 2);
    let first = model.tensor("first").unwrap();
    let mut decoded = [0.0; 2];
    first.decode_row(1, &mut decoded).unwrap();
    assert_eq!(decoded, [3.0, 4.0]);
    let mut last_bytes = [0; 4];
    first.read_range(12, &mut last_bytes).unwrap();
    assert_eq!(last_bytes, 4.0_f32.to_le_bytes());
    let mut decoded = [0.0; 32];
    model
        .tensor("second")
        .unwrap()
        .decode_row(0, &mut decoded)
        .unwrap();
    assert_eq!(decoded, [3.0; 32]);
}

#[test]
fn rows_and_slices_are_checked_before_reads() {
    let fixture = Fixture::new(&model(&[ple()]));
    let model = fixture.open().unwrap();
    let tensor = model.tensor("per_layer_token_embd.weight").unwrap();
    assert!(model.tensor("missing").is_none());
    assert!(matches!(
        tensor.read_row(3, &mut [0; 110]),
        Err(Error::OutOfBounds)
    ));
    assert!(matches!(
        tensor.read_row(0, &mut [0; 109]),
        Err(Error::InvalidBufferLength)
    ));
    assert!(matches!(
        tensor.decode_row(0, &mut [0.0; 159]),
        Err(Error::InvalidBufferLength)
    ));
    assert!(matches!(
        tensor.decode_row(u64::MAX, &mut [0.0; 160]),
        Err(Error::OutOfBounds)
    ));
    assert!(matches!(
        tensor.read_range(329, &mut [0; 2]),
        Err(Error::OutOfBounds)
    ));
    assert!(matches!(
        tensor.read_range(u64::MAX, &mut [0; 2]),
        Err(Error::ArithmeticOverflow)
    ));
    tensor.read_range(330, &mut []).unwrap();
}

#[test]
fn read_limit_applies_to_both_raw_reads_and_decoded_row_scratch() {
    let fixture = Fixture::new(&model(&[ple()]));
    let limits = GgufLimits {
        max_tensor_read_bytes: 109,
        ..GgufLimits::default()
    };
    let model = GgufFile::open(&fixture.0, limits).unwrap();
    let tensor = model.tensor("per_layer_token_embd.weight").unwrap();
    assert!(matches!(
        tensor.read_row(0, &mut [0; 110]),
        Err(Error::LimitExceeded("tensor read bytes"))
    ));
    assert!(matches!(
        tensor.decode_row(0, &mut [0.0; 160]),
        Err(Error::LimitExceeded("tensor read bytes"))
    ));
    tensor.read_range(0, &mut [0; 16]).unwrap();
}

#[test]
fn accepts_v2_and_rejects_other_versions_and_endianness() {
    let mut bytes = model(&[]);
    bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
    assert_eq!(Fixture::new(&bytes).open().unwrap().version(), 2);
    bytes[4..8].copy_from_slice(&4_u32.to_le_bytes());
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::UnsupportedVersion(4))
    ));
    bytes[..4].copy_from_slice(b"FUGG");
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat(_))
    ));
}

#[test]
fn rejects_payload_truncation_dimension_overflow_and_partial_blocks() {
    let bytes = model(&[ple()]);
    assert!(matches!(
        Fixture::new(&bytes[..bytes.len() - 1]).open(),
        Err(Error::InvalidFormat("tensor exceeds file"))
    ));
    let bad = Tensor {
        name: "overflow",
        dimensions: vec![u64::MAX, 2],
        kind: 0,
        offset: 0,
        bytes: vec![0; 4],
    };
    assert!(matches!(
        Fixture::new(&model(&[bad])).open(),
        Err(Error::ArithmeticOverflow)
    ));
    let bad = Tensor {
        name: "partial",
        dimensions: vec![161, 1],
        kind: 6,
        offset: 0,
        bytes: vec![0; 110],
    };
    assert!(matches!(
        Fixture::new(&model(&[bad])).open(),
        Err(Error::InvalidFormat(_))
    ));
    let bad = Tensor {
        name: "zero",
        dimensions: vec![0, 1],
        kind: 0,
        offset: 0,
        bytes: vec![0; 4],
    };
    assert!(matches!(
        Fixture::new(&model(&[bad])).open(),
        Err(Error::InvalidFormat("zero tensor dimension"))
    ));
}

#[test]
fn rejects_unknown_types_unaligned_offsets_and_overlapping_tensors() {
    let bad = Tensor {
        name: "unsupported",
        dimensions: vec![32],
        kind: 999,
        offset: 0,
        bytes: vec![0; 32],
    };
    assert!(matches!(
        Fixture::new(&model(&[bad])).open(),
        Err(Error::UnsupportedTensorType(999))
    ));
    let bad = Tensor {
        name: "unaligned",
        dimensions: vec![1],
        kind: 0,
        offset: 1,
        bytes: vec![0; 4],
    };
    assert!(matches!(
        Fixture::new(&model(&[bad])).open(),
        Err(Error::InvalidFormat("unaligned tensor offset"))
    ));
    let first = Tensor {
        name: "first",
        dimensions: vec![16],
        kind: 0,
        offset: 0,
        bytes: vec![0; 64],
    };
    let second = Tensor {
        name: "second",
        dimensions: vec![1],
        kind: 0,
        offset: 32,
        bytes: vec![0; 4],
    };
    assert!(matches!(
        Fixture::new(&model(&[first, second])).open(),
        Err(Error::InvalidFormat("overlapping tensors"))
    ));
}

#[test]
fn rejects_duplicate_names_and_duplicate_metadata_keys() {
    let first = Tensor {
        name: "same",
        dimensions: vec![1],
        kind: 0,
        offset: 0,
        bytes: vec![0; 4],
    };
    let second = Tensor {
        name: "same",
        dimensions: vec![1],
        kind: 0,
        offset: 32,
        bytes: vec![0; 4],
    };
    assert!(matches!(
        Fixture::new(&model(&[first, second])).open(),
        Err(Error::InvalidFormat("duplicate tensor name"))
    ));
    let mut bytes = header(3, 0, 2);
    for _ in 0..2 {
        string(&mut bytes, "duplicate");
        bytes.extend(4_u32.to_le_bytes());
        bytes.extend(1_u32.to_le_bytes());
    }
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat("duplicate metadata key"))
    ));
}

#[test]
fn metadata_counts_array_counts_strings_and_total_allocation_are_bounded() {
    let fixture = Fixture::new(&header(3, u64::MAX, 0));
    assert!(matches!(
        fixture.open(),
        Err(Error::LimitExceeded("tensor count"))
    ));
    let fixture = Fixture::new(&header(3, 0, u64::MAX));
    assert!(matches!(
        fixture.open(),
        Err(Error::LimitExceeded("metadata count"))
    ));

    let fixture = Fixture::new(&model(&[ple()]));
    for limits in [
        GgufLimits {
            max_header_bytes: 23,
            ..GgufLimits::default()
        },
        GgufLimits {
            max_metadata_allocation_bytes: 127,
            ..GgufLimits::default()
        },
        GgufLimits {
            max_string_bytes: 1,
            ..GgufLimits::default()
        },
        GgufLimits {
            max_array_elements: 2,
            ..GgufLimits::default()
        },
    ] {
        assert!(matches!(
            GgufFile::open(&fixture.0, limits),
            Err(Error::LimitExceeded(_))
        ));
    }
    let mut bytes = header(3, 0, 1);
    string(&mut bytes, "huge");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(8_u32.to_le_bytes());
    bytes.extend(u64::MAX.to_le_bytes());
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::LimitExceeded("array elements"))
    ));
}

#[test]
fn invalid_utf8_boolean_alignment_and_nested_arrays_fail_closed() {
    let mut bytes = header(3, 0, 1);
    bytes.extend(1_u64.to_le_bytes());
    bytes.push(0xff);
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat("string is not UTF-8"))
    ));

    let mut bytes = header(3, 0, 1);
    string(&mut bytes, "bool");
    bytes.extend(7_u32.to_le_bytes());
    bytes.push(2);
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat("invalid boolean"))
    ));

    let mut bytes = header(3, 0, 1);
    string(&mut bytes, "general.alignment");
    bytes.extend(4_u32.to_le_bytes());
    bytes.extend(3_u32.to_le_bytes());
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat("unsupported tensor alignment"))
    ));

    let mut bytes = header(3, 0, 1);
    string(&mut bytes, "nested");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    assert!(matches!(
        Fixture::new(&bytes).open(),
        Err(Error::InvalidFormat("unsupported or nested array type"))
    ));
}

#[test]
fn header_truncations_return_errors_without_panics() {
    let bytes = model(&[ple()]);
    // Every prefix of the bounded fixture is malformed until the entire tensor
    // payload exists. This exercises incomplete scalars, strings and headers.
    for length in 0..bytes.len() {
        assert!(
            Fixture::new(&bytes[..length]).open().is_err(),
            "length={length}"
        );
    }
}

#[test]
fn read_only_descriptor_detects_in_place_changes() {
    let fixture = Fixture::new(&model(&[ple()]));
    let model = fixture.open().unwrap();
    let file = OpenOptions::new().write(true).open(&fixture.0).unwrap();
    file.set_len(model.file_len() - 1).unwrap();
    assert!(matches!(
        model.verify_file_unchanged(),
        Err(Error::FileChanged)
    ));
    assert!(matches!(
        model
            .tensor("per_layer_token_embd.weight")
            .unwrap()
            .read_row(0, &mut [0; 110]),
        Err(Error::FileChanged)
    ));
}

#[test]
fn metadata_supports_signed_unsigned_floats_and_boolean_arrays() {
    let mut bytes = header(3, 0, 4);
    string(&mut bytes, "unsigned");
    bytes.extend(10_u32.to_le_bytes());
    bytes.extend(u64::MAX.to_le_bytes());
    string(&mut bytes, "signed");
    bytes.extend(11_u32.to_le_bytes());
    bytes.extend(i64::MIN.to_le_bytes());
    string(&mut bytes, "float");
    bytes.extend(12_u32.to_le_bytes());
    bytes.extend(1.25_f64.to_le_bytes());
    string(&mut bytes, "flags");
    bytes.extend(9_u32.to_le_bytes());
    bytes.extend(7_u32.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    bytes.extend([0, 1]);
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    let fixture = Fixture::new(&bytes);
    let model = fixture.open().unwrap();
    assert_eq!(model.metadata()["unsigned"].as_u64(), Some(u64::MAX));
    assert_eq!(model.metadata()["signed"].as_i64(), Some(i64::MIN));
    assert_eq!(model.metadata()["float"].as_f64(), Some(1.25));
    assert_eq!(
        model.metadata()["flags"].as_array().unwrap(),
        [MetadataValue::Bool(false), MetadataValue::Bool(true)]
    );
}

#[test]
fn a_large_sparse_model_file_does_not_cause_a_full_payload_allocation() {
    let tensor = Tensor {
        name: "sparse",
        dimensions: vec![1, 1_000_000_000],
        kind: 0,
        offset: 0,
        bytes: vec![0; 4],
    };
    let fixture = Fixture::new(&model(&[tensor]));
    let file = OpenOptions::new().write(true).open(&fixture.0).unwrap();
    let payload_offset = file.metadata().unwrap().len() - 4;
    file.set_len(payload_offset + 4_000_000_000).unwrap();
    let mut writer = File::options().write(true).open(&fixture.0).unwrap();
    writer
        .seek(SeekFrom::Start(payload_offset + 4_000_000_000 - 4))
        .unwrap();
    writer.write_all(&17.0_f32.to_le_bytes()).unwrap();
    let model = fixture.open().unwrap();
    let tensor = model.tensor("sparse").unwrap();
    assert_eq!(tensor.info().byte_len(), 4_000_000_000);
    let mut row = [0.0; 1];
    tensor.decode_row(999_999_999, &mut row).unwrap();
    assert_eq!(row, [17.0]);
}
