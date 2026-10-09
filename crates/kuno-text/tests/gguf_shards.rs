use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use kuno_text::{Error, GgufFile, GgufLimits};

static NEXT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
enum Value {
    U16(u16),
    I32(i32),
    Text(&'static str),
    Floats(Vec<f64>),
}

fn string(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend((text.len() as u64).to_le_bytes());
    bytes.extend(text.as_bytes());
}

#[derive(Clone)]
struct Shard {
    index: u16,
    count: u16,
    total: i32,
    version: u32,
    extra: Vec<(&'static str, Value)>,
    tensor: Option<(&'static str, u64, Vec<f32>)>,
}

impl Shard {
    fn bytes(&self) -> (Vec<u8>, usize) {
        let mut metadata = vec![
            ("split.no", Value::U16(self.index)),
            ("split.count", Value::U16(self.count)),
            ("split.tensors.count", Value::I32(self.total)),
        ];
        metadata.extend(self.extra.clone());
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(self.version.to_le_bytes());
        bytes.extend(u64::from(self.tensor.is_some()).to_le_bytes());
        bytes.extend((metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            string(&mut bytes, key);
            match value {
                Value::U16(value) => {
                    bytes.extend(2u32.to_le_bytes());
                    bytes.extend(value.to_le_bytes());
                }
                Value::I32(value) => {
                    bytes.extend(5u32.to_le_bytes());
                    bytes.extend(value.to_le_bytes());
                }
                Value::Text(value) => {
                    bytes.extend(8u32.to_le_bytes());
                    string(&mut bytes, value);
                }
                Value::Floats(values) => {
                    bytes.extend(9u32.to_le_bytes());
                    bytes.extend(12u32.to_le_bytes());
                    bytes.extend((values.len() as u64).to_le_bytes());
                    for value in values {
                        bytes.extend(value.to_le_bytes());
                    }
                }
            }
        }
        if let Some((name, offset, values)) = &self.tensor {
            string(&mut bytes, name);
            bytes.extend(1u32.to_le_bytes());
            bytes.extend((values.len() as u64).to_le_bytes());
            bytes.extend(0u32.to_le_bytes());
            bytes.extend(offset.to_le_bytes());
        }
        let data_offset = bytes.len().next_multiple_of(32);
        bytes.resize(data_offset, 0);
        if let Some((_, offset, values)) = &self.tensor {
            bytes.resize(data_offset + *offset as usize, 0);
            for value in values {
                bytes.extend(value.to_le_bytes());
            }
        }
        (bytes, data_offset)
    }
}

fn shards() -> Vec<Shard> {
    (0..3)
        .map(|index| Shard {
            index,
            count: 3,
            total: 2,
            version: 3,
            extra: if index == 0 {
                vec![("general.architecture", Value::Text("qwen4exp"))]
            } else {
                vec![]
            },
            tensor: match index {
                1 => Some(("first", 0, vec![1., 2., 3., 4.])),
                2 => Some(("second", 32, vec![10., 11., 12., 13.])),
                _ => None,
            },
        })
        .collect()
}

struct Fixture {
    directory: PathBuf,
    paths: Vec<PathBuf>,
}

impl Fixture {
    fn new(shards: &[Shard]) -> Self {
        Self::raw(shards.iter().map(|shard| shard.bytes().0).collect())
    }

    fn raw(files: Vec<Vec<u8>>) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "kuno-shards-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let paths = files
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                let path = directory.join(format!("source-{index}.gguf"));
                fs::write(&path, bytes).unwrap();
                path
            })
            .collect();
        Self { directory, paths }
    }

    fn open(&self) -> kuno_text::Result<GgufFile> {
        GgufFile::open_shards(&self.paths, GgufLimits::default())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[test]
fn metadata_only_primary_and_unordered_sources_read_exact_original_tensors() {
    let fixture = Fixture::new(&shards());
    let order = [&fixture.paths[2], &fixture.paths[0], &fixture.paths[1]];
    let model = GgufFile::open_shards(&order, GgufLimits::default()).unwrap();
    assert_eq!(model.shard_count(), 3);
    assert_eq!(model.tensors().len(), 2);
    assert_eq!(model.metadata()["split.no"].as_u64(), Some(0));
    assert_eq!(model.metadata()["split.tensors.count"].as_i64(), Some(2));
    assert_eq!(
        model.metadata()["general.architecture"].as_str(),
        Some("qwen4exp")
    );
    assert_eq!(
        model.file_len(),
        fixture
            .paths
            .iter()
            .map(|path| fs::metadata(path).unwrap().len())
            .sum::<u64>()
    );
    for (name, source, expected) in [
        ("first", 1, [1., 2., 3., 4.]),
        ("second", 2, [10., 11., 12., 13.]),
    ] {
        let tensor = model.tensor(name).unwrap();
        assert_eq!(tensor.info().shard_index(), source);
        let mut values = [0.; 4];
        tensor.decode_row(0, &mut values).unwrap();
        assert_eq!(values, expected);
        assert!(matches!(
            tensor.read_range(15, &mut [0; 2]),
            Err(Error::OutOfBounds)
        ));
    }
    model.verify_file_unchanged().unwrap();
}

#[test]
fn incomplete_sets_and_duplicate_sources_are_rejected() {
    let fixture = Fixture::new(&shards());
    for path in &fixture.paths {
        assert!(matches!(
            GgufFile::open(path, GgufLimits::default()),
            Err(Error::InvalidFormat("complete split GGUF sources required"))
        ));
    }
    assert!(fixture.open().is_ok());
    assert!(GgufFile::open_shards(&fixture.paths[..2], GgufLimits::default()).is_err());
    let duplicate = [&fixture.paths[0], &fixture.paths[1], &fixture.paths[1]];
    assert!(matches!(
        GgufFile::open_shards(&duplicate, GgufLimits::default()),
        Err(Error::InvalidFormat(
            "split GGUF missing or duplicate index"
        ))
    ));
    let absent = [
        &fixture.paths[0],
        &fixture.paths[1],
        &fixture.directory.join("missing"),
    ];
    assert!(matches!(
        GgufFile::open_shards(&absent, GgufLimits::default()),
        Err(Error::Io(_))
    ));
    assert!(GgufFile::open_shards::<PathBuf>(&[], GgufLimits::default()).is_err());
}

#[test]
fn split_contract_rejects_bad_counts_versions_names_and_duplicate_metadata() {
    let mut cases = Vec::new();
    let mut bad = shards();
    bad[2].count = 4;
    cases.push(bad);
    let mut bad = shards();
    bad[2].index = 3;
    cases.push(bad);
    let mut bad = shards();
    bad[2].total = 3;
    cases.push(bad);
    let mut bad = shards();
    for shard in &mut bad {
        shard.total = 3;
    }
    cases.push(bad);
    let mut bad = shards();
    bad[2].total = -1;
    cases.push(bad);
    let mut bad = shards();
    bad[2].version = 2;
    cases.push(bad);
    let mut bad = shards();
    bad[2].tensor.as_mut().unwrap().0 = "first";
    cases.push(bad);
    let mut bad = shards();
    bad[1]
        .extra
        .push(("general.architecture", Value::Text("other")));
    cases.push(bad);
    let mut bad = shards();
    bad[0]
        .extra
        .push(("fixture.floats", Value::Floats(vec![0.])));
    bad[2]
        .extra
        .push(("fixture.floats", Value::Floats(vec![-0.])));
    cases.push(bad);
    for (index, bad) in cases.iter().enumerate() {
        assert!(
            matches!(Fixture::new(bad).open(), Err(Error::InvalidFormat(_))),
            "case {index}"
        );
    }
    let mut good = shards();
    good[1]
        .extra
        .push(("general.architecture", Value::Text("qwen4exp")));
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    good[0]
        .extra
        .push(("fixture.floats", Value::Floats(vec![nan, -0.])));
    good[2]
        .extra
        .push(("fixture.floats", Value::Floats(vec![nan, -0.])));
    assert!(Fixture::new(&good).open().is_ok());
}

#[test]
fn each_tensor_extent_is_checked_against_its_own_file() {
    let sources = shards();
    let mut files: Vec<_> = sources.iter().map(|shard| shard.bytes().0).collect();
    files[1].pop();
    assert!(matches!(
        Fixture::raw(files).open(),
        Err(Error::InvalidFormat("tensor exceeds file"))
    ));
    let mut sources = shards();
    sources[1].tensor.as_mut().unwrap().1 = 1;
    assert!(matches!(
        Fixture::new(&sources).open(),
        Err(Error::InvalidFormat("unaligned tensor offset"))
    ));
}

#[test]
fn header_allocation_entry_and_tensor_limits_apply_to_the_whole_set() {
    let sources = shards();
    let fixture = Fixture::new(&sources);
    let total_header = sources
        .iter()
        .map(|source| source.bytes().1 as u64)
        .sum::<u64>();
    let limits = GgufLimits {
        max_header_bytes: total_header - 1,
        ..GgufLimits::default()
    };
    assert!(matches!(
        GgufFile::open_shards(&fixture.paths, limits),
        Err(Error::LimitExceeded("header bytes"))
    ));
    let limits = GgufLimits {
        max_metadata_entries: 9,
        ..GgufLimits::default()
    };
    assert!(matches!(
        GgufFile::open_shards(&fixture.paths, limits),
        Err(Error::LimitExceeded("metadata count"))
    ));
    let limits = GgufLimits {
        max_tensors: 1,
        ..GgufLimits::default()
    };
    assert!(GgufFile::open_shards(&fixture.paths, limits).is_err());
    // Each file's metadata fits 1100 bytes; the combined retained set does not.
    let limits = GgufLimits {
        max_metadata_allocation_bytes: 1100,
        ..GgufLimits::default()
    };
    assert!(matches!(
        GgufFile::open_shards(&fixture.paths, limits),
        Err(Error::LimitExceeded("metadata allocation bytes"))
    ));
    let limits = GgufLimits {
        max_header_bytes: total_header,
        max_metadata_entries: 10,
        max_tensors: 2,
        ..GgufLimits::default()
    };
    assert!(GgufFile::open_shards(&fixture.paths, limits).is_ok());
}

#[test]
fn changes_in_any_source_invalidate_even_reads_of_another_shard() {
    for changed in 0..3 {
        let fixture = Fixture::new(&shards());
        let model = fixture.open().unwrap();
        let reader = model.positional_reader().unwrap();
        let tensor = model.tensor("first").unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(&fixture.paths[changed])
            .unwrap();
        file.write_all(&[0]).unwrap();
        assert!(matches!(
            model.verify_file_unchanged(),
            Err(Error::FileChanged)
        ));
        assert!(matches!(
            reader.verify_file_unchanged(),
            Err(Error::FileChanged)
        ));
        assert!(matches!(
            reader.mapped_range(tensor, 0, 4),
            Err(Error::FileChanged)
        ));
        let mut output = [0xa5; 4];
        assert!(matches!(
            reader.read_range(tensor, 0, &mut output),
            Err(Error::FileChanged)
        ));
        assert_eq!(output, [0; 4]);
        assert!(matches!(
            tensor.read_range(0, &mut output),
            Err(Error::FileChanged)
        ));
    }
}

#[cfg(unix)]
#[test]
#[allow(unsafe_code)]
fn positional_and_mapped_reads_keep_original_sources_after_path_replacement() {
    let fixture = Fixture::new(&shards());
    let model = fixture.open().unwrap();
    let other = fixture.open().unwrap();
    let reader = model.positional_reader().unwrap();
    assert!(reader.belongs_to(&model));
    assert!(!reader.belongs_to(&other));
    assert!(
        reader
            .read_range(other.tensor("first").unwrap(), 0, &mut [0; 4])
            .is_err()
    );
    assert!(
        reader
            .mapped_range(model.tensor("second").unwrap(), 0, 16)
            .unwrap()
            .is_none()
    );
    // SAFETY: only directory entries are replaced; all inspected inodes stay immutable.
    unsafe {
        model.enable_mapped_weights().unwrap();
    }
    assert!(model.weights_mapped());
    for (index, path) in fixture.paths.iter().enumerate() {
        fs::rename(path, fixture.directory.join(format!("pinned-{index}"))).unwrap();
        fs::write(path, b"unrelated replacement inode").unwrap();
    }
    for (name, expected) in [("first", 1.0f32), ("second", 10.0f32)] {
        let tensor = model.tensor(name).unwrap();
        let mut output = [0; 4];
        reader.read_range(tensor, 0, &mut output).unwrap();
        assert_eq!(output, expected.to_le_bytes());
        assert_eq!(reader.mapped_range(tensor, 0, 4).unwrap().unwrap(), output);
        tensor.read_range(0, &mut output).unwrap();
        assert_eq!(output, expected.to_le_bytes());
    }
    model.verify_file_unchanged().unwrap();
    // Cache hits still check the complete set of inspected descriptors.
    unsafe {
        model.enable_mapped_weights().unwrap();
    }
}

#[cfg(all(
    target_os = "linux",
    target_pointer_width = "64",
    not(target_arch = "s390x")
))]
#[test]
#[allow(unsafe_code)]
fn cache_hints_use_checked_nonprimary_source_geometry() {
    let mut sources = shards();
    sources[2].tensor = Some(("second", 4096, vec![13.; 8192]));
    let fixture = Fixture::new(&sources);
    let model = fixture.open().unwrap();
    let reader = model.positional_reader().unwrap();
    let tensor = model.tensor("second").unwrap();
    // SAFETY: inspected files stay immutable for all mapping consumers.
    unsafe {
        model.enable_mapped_weights().unwrap();
    }
    reader.prefetch_range(tensor, 0, 32768).unwrap();
    assert!(
        model
            .advise_gpu_source_not_needed(tensor, 0, 32768)
            .unwrap()
            >= 4096
    );
    assert!(model.advise_gpu_source_pageout(tensor, 0, 32768).unwrap() >= 4096);
    assert!(reader.prefetch_range(tensor, 32767, 2).is_err());
    assert!(
        model
            .advise_gpu_source_not_needed(tensor, 32767, 2)
            .is_err()
    );
    assert!(model.advise_gpu_source_pageout(tensor, 32767, 2).is_err());
    let mut output = [0; 4];
    reader.read_range(tensor, 32764, &mut output).unwrap();
    assert_eq!(output, 13.0f32.to_le_bytes());
    assert_eq!(
        reader.mapped_range(tensor, 32764, 4).unwrap().unwrap(),
        output
    );
}
