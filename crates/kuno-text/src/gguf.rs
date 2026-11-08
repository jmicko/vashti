//! Bounded GGUF v2/v3 metadata and read-only tensor access.
//!
//! Tensor offsets and sizes are validated before returning a model. Unsupported
//! formats, overlapping tensors, malformed dimensions and unbounded allocations
//! fail closed. Open descriptors pin the inspected source inodes; replacement
//! paths cannot redirect subsequent reads to a different model.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use crate::{Error, Result, TensorType};

#[path = "gguf_positional.rs"]
mod positional;
pub use positional::{DIRECT_BOUNCE_BYTES, DirectSource, PositionalReader};

#[derive(Clone, Debug)]
pub struct GgufLimits {
    pub max_header_bytes: u64,
    /// Conservative accounting of retained strings, vector items and map nodes.
    pub max_metadata_allocation_bytes: u64,
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_metadata_entries: u64,
    pub max_tensors: u64,
    /// Maximum one-shot tensor read, including internal packed row buffers.
    pub max_tensor_read_bytes: u64,
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self {
            max_header_bytes: 128 * 1024 * 1024,
            max_metadata_allocation_bytes: 256 * 1024 * 1024,
            max_string_bytes: 8 * 1024 * 1024,
            max_array_elements: 4 * 1024 * 1024,
            max_metadata_entries: 16_384,
            max_tensors: 65_536,
            max_tensor_read_bytes: 64 * 1024 * 1024,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MetadataValue {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Unsigned(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Signed(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(value) => Some(*value),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[MetadataValue]> {
        match self {
            Self::Array(value) => Some(value),
            _ => None,
        }
    }
}

/// Immutable tensor geometry; dimension zero is the contiguous GGML row width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    dtype: TensorType,
    offset: u64,
    byte_len: u64,
    row_bytes: u64,
    row_count: u64,
    element_count: u64,
    shard_index: usize,
}

impl TensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn dimensions(&self) -> &[u64] {
        &self.dimensions
    }
    pub fn dtype(&self) -> TensorType {
        self.dtype
    }
    /// Shard-local tensor-data-relative offset, not an absolute file position.
    pub fn offset(&self) -> u64 {
        self.offset
    }
    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }
    pub fn row_len(&self) -> u64 {
        self.dimensions[0]
    }
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }
    /// Higher dimensions are flattened with dimension one varying fastest.
    pub fn row_count(&self) -> u64 {
        self.row_count
    }
    pub fn element_count(&self) -> u64 {
        self.element_count
    }
    /// Index of the inspected source, in split.no order (zero for a single file).
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }
}

struct GgufSource {
    file: Mutex<File>,
    /// Monotonic ns (since STAT_EPOCH) of the last successful identity check.
    checked: std::sync::atomic::AtomicU64,
    mapping: OnceLock<memmap2::Mmap>,
    file_len: u64,
    modified: Option<SystemTime>,
    data_offset: u64,
}

/// Minimum interval between identity re-checks of one source (0 = always).
/// Serving workers verify identity thousands of times per prompt; they may
/// set an interval, so a change is still detected within it.
static STAT_INTERVAL_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub fn set_identity_check_interval(interval: std::time::Duration) {
    STAT_INTERVAL_NS.store(interval.as_nanos().min(u64::MAX as u128) as u64, std::sync::atomic::Ordering::Relaxed);
}
fn stat_clock() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    // +1 so a fresh source (checked == 0) is always verified first.
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64 + 1
}
impl GgufSource {
    fn recently_checked(&self) -> bool {
        use std::sync::atomic::Ordering;
        let last = self.checked.load(Ordering::Relaxed);
        last != 0
            && stat_clock().saturating_sub(last) < STAT_INTERVAL_NS.load(Ordering::Relaxed)
    }
    fn check_stat(&self, file: &File) -> Result<()> {
        use std::sync::atomic::Ordering;
        let now = stat_clock();
        let last = self.checked.load(Ordering::Relaxed);
        if last != 0 && now.saturating_sub(last) < STAT_INTERVAL_NS.load(Ordering::Relaxed) {
            return Ok(());
        }
        let stat = file.metadata()?;
        if stat.len() != self.file_len || stat.modified().ok() != self.modified {
            self.checked.store(0, Ordering::Relaxed);
            return Err(Error::FileChanged);
        }
        self.checked.store(now, Ordering::Relaxed);
        Ok(())
    }
}

/// Read-only model descriptor and bounded retained metadata. Debug formatting is
/// intentionally omitted: metadata may include model-defined templates.
pub struct GgufFile {
    sources: Vec<GgufSource>,
    version: u32,
    limits: GgufLimits,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, TensorInfo>,
    header_bytes: u64,
    allocation_bytes: u64,
    metadata_entries: u64,
    total_file_len: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>, limits: GgufLimits) -> Result<Self> {
        let model = Self::inspect(path.as_ref(), limits)?;
        if model.metadata.contains_key("split.count") {
            if split_integer(&model.metadata, "split.count")? != 1
                || split_integer(&model.metadata, "split.no")? != 0
                || split_integer(&model.metadata, "split.tensors.count")?
                    != model.tensors.len() as u64
            {
                return Err(Error::InvalidFormat("complete split GGUF sources required"));
            }
        } else if model.metadata.keys().any(|key| key.starts_with("split.")) {
            return Err(Error::InvalidFormat("incomplete split GGUF metadata"));
        }
        Ok(model)
    }

    /// Inspect an explicit complete split without merging, copying or reopening
    /// payload files. Paths may be unordered; each tensor keeps its original fd
    /// and shard-local offset. All limits apply to the complete inspected set.
    /// Shard zero supplies the primary metadata and may contain no tensors.
    /// Missing metadata on later shards is allowed; duplicated metadata must
    /// match. Split count, indices, GGUF version and declared tensor total must
    /// agree, with exactly one source per contiguous index. At most 1024 sources
    /// are admitted. Ordinary `open` rejects an incomplete multi-source split.
    pub fn open_shards<P: AsRef<Path>>(paths: &[P], limits: GgufLimits) -> Result<Self> {
        // Explicitly bound descriptor/map owners independently of tensor count:
        // a malformed set could otherwise contain arbitrarily many empty shards.
        if paths.len() < 2 || paths.len() > 1024 {
            return Err(Error::LimitExceeded("split source count"));
        }
        let owner_bytes = (paths.len() as u64)
            // Cover the temporary sorted owners and both original and final
            // source-vector storage while descriptors are moved into one owner.
            .checked_mul(
                (std::mem::size_of::<(u64, Self)>() + 2 * std::mem::size_of::<GgufSource>()) as u64,
            )
            .ok_or(Error::ArithmeticOverflow)?;
        let mut remaining = limits.clone();
        remaining.max_metadata_allocation_bytes = remaining
            .max_metadata_allocation_bytes
            .checked_sub(owner_bytes)
            .ok_or(Error::LimitExceeded("metadata allocation bytes"))?;
        let mut inspected = Vec::new();
        inspected
            .try_reserve_exact(paths.len())
            .map_err(|_| Error::LimitExceeded("split source allocation"))?;
        let mut total_file_len = 0u64;
        let mut expected_tensors = None;
        for path in paths {
            let shard = Self::inspect(path.as_ref(), remaining.clone())?;
            let count = split_integer(&shard.metadata, "split.count")?;
            let index = split_integer(&shard.metadata, "split.no")?;
            let total = split_integer(&shard.metadata, "split.tensors.count")?;
            if total > limits.max_tensors {
                return Err(Error::LimitExceeded("tensor count"));
            }
            if count != paths.len() as u64 || index >= count {
                return Err(Error::InvalidFormat("split GGUF count or index"));
            }
            if expected_tensors.is_some_and(|expected| expected != total) {
                return Err(Error::InvalidFormat(
                    "split GGUF declared tensor count mismatch",
                ));
            }
            expected_tensors = Some(total);
            remaining.max_header_bytes -= shard.header_bytes;
            remaining.max_metadata_allocation_bytes -= shard.allocation_bytes;
            remaining.max_metadata_entries -= shard.metadata_entries;
            remaining.max_tensors -= shard.tensors.len() as u64;
            total_file_len = total_file_len
                .checked_add(shard.total_file_len)
                .ok_or(Error::ArithmeticOverflow)?;
            inspected.push((index, shard));
        }
        inspected.sort_unstable_by_key(|(index, _)| *index);
        for (expected, (index, _)) in inspected.iter().enumerate() {
            if *index != expected as u64 {
                return Err(Error::InvalidFormat(
                    "split GGUF missing or duplicate index",
                ));
            }
        }
        let mut ordered = inspected.into_iter();
        let (_, mut model) = ordered
            .next()
            .ok_or(Error::InvalidFormat("split GGUF empty"))?;
        model
            .sources
            .try_reserve_exact(paths.len() - 1)
            .map_err(|_| Error::LimitExceeded("split source allocation"))?;
        for (index, shard) in ordered {
            if shard.version != model.version {
                return Err(Error::InvalidFormat("split GGUF version mismatch"));
            }
            for (key, value) in shard.metadata {
                if key == "split.no" {
                    continue;
                }
                if let Some(original) = model.metadata.get(&key) {
                    if !metadata_equal(original, &value) {
                        return Err(Error::InvalidFormat("split GGUF metadata mismatch"));
                    }
                } else {
                    model.metadata.insert(key, value);
                }
            }
            for (name, mut tensor) in shard.tensors {
                tensor.shard_index = index as usize;
                if model.tensors.insert(name, tensor).is_some() {
                    return Err(Error::InvalidFormat("split GGUF duplicate tensor name"));
                }
            }
            model.sources.extend(shard.sources);
        }
        if Some(model.tensors.len() as u64) != expected_tensors {
            return Err(Error::InvalidFormat(
                "split GGUF total tensor count mismatch",
            ));
        }
        model.header_bytes = limits.max_header_bytes - remaining.max_header_bytes;
        model.allocation_bytes =
            limits.max_metadata_allocation_bytes - remaining.max_metadata_allocation_bytes;
        model.metadata_entries = limits.max_metadata_entries - remaining.max_metadata_entries;
        model.total_file_len = total_file_len;
        model.limits = limits;
        model.verify_file_unchanged()?;
        Ok(model)
    }

    fn inspect(path: &Path, limits: GgufLimits) -> Result<Self> {
        let file = File::open(path)?;
        let stat = file.metadata()?;
        if !stat.is_file() {
            return Err(Error::InvalidFormat("model is not a regular file"));
        }
        let file_len = stat.len();
        let modified = stat.modified().ok();
        let mut reader = HeaderReader {
            inner: BufReader::with_capacity(16 * 1024, file),
            position: 0,
            allocated: 0,
            file_len,
            limits: &limits,
        };
        if reader.bytes::<4>()? != *b"GGUF" {
            return Err(Error::InvalidFormat("little-endian GGUF magic required"));
        }
        let version = reader.u32()?;
        if version != 2 && version != 3 {
            return Err(Error::UnsupportedVersion(version));
        }
        let tensor_count = reader.u64()?;
        let metadata_count = reader.u64()?;
        if tensor_count > limits.max_tensors {
            return Err(Error::LimitExceeded("tensor count"));
        }
        if metadata_count > limits.max_metadata_entries {
            return Err(Error::LimitExceeded("metadata count"));
        }
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            reader.charge_allocation(128)?;
            let key = reader.string()?;
            if key.is_empty() {
                return Err(Error::InvalidFormat("empty metadata key"));
            }
            let tag = reader.u32()?;
            let value = reader.value(tag, true)?;
            if metadata.insert(key, value).is_some() {
                return Err(Error::InvalidFormat("duplicate metadata key"));
            }
        }
        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(MetadataValue::Unsigned(value)) => *value,
            _ => return Err(Error::InvalidFormat("alignment is not unsigned")),
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 4096 {
            return Err(Error::InvalidFormat("unsupported tensor alignment"));
        }
        let mut tensors = BTreeMap::new();
        for _ in 0..tensor_count {
            reader.charge_allocation(256)?;
            let name = reader.string()?;
            if name.is_empty() {
                return Err(Error::InvalidFormat("empty tensor name"));
            }
            let rank = reader.u32()?;
            if rank == 0 || rank > 4 {
                return Err(Error::InvalidFormat(
                    "tensor rank must be between one and four",
                ));
            }
            let mut dimensions = Vec::with_capacity(rank as usize);
            let mut element_count = 1_u64;
            for _ in 0..rank {
                let dimension = reader.u64()?;
                if dimension == 0 {
                    return Err(Error::InvalidFormat("zero tensor dimension"));
                }
                element_count = element_count
                    .checked_mul(dimension)
                    .ok_or(Error::ArithmeticOverflow)?;
                dimensions.push(dimension);
            }
            let dtype = TensorType::from_gguf_id(reader.u32()?)?;
            let offset = reader.u64()?;
            if !offset.is_multiple_of(alignment) {
                return Err(Error::InvalidFormat("unaligned tensor offset"));
            }
            let row_bytes = dtype.row_bytes(dimensions[0])?;
            let row_count = element_count / dimensions[0];
            let byte_len = row_bytes
                .checked_mul(row_count)
                .ok_or(Error::ArithmeticOverflow)?;
            // Charge the second retained name before cloning it.
            reader.charge_allocation(name.len() as u64)?;
            let info = TensorInfo {
                name: name.clone(),
                dimensions,
                dtype,
                offset,
                byte_len,
                row_bytes,
                row_count,
                element_count,
                shard_index: 0,
            };
            if tensors.insert(name, info).is_some() {
                return Err(Error::InvalidFormat("duplicate tensor name"));
            }
        }
        let data_offset = align_up(reader.position, alignment)?;
        if data_offset > limits.max_header_bytes {
            return Err(Error::LimitExceeded("header bytes"));
        }
        if data_offset > file_len {
            return Err(Error::InvalidFormat("missing aligned tensor data"));
        }
        let mut intervals: Vec<_> = tensors
            .values()
            .map(|tensor| (tensor.offset, tensor.byte_len))
            .collect();
        intervals.sort_unstable();
        let mut previous_end = 0_u64;
        for (offset, length) in intervals {
            let end = offset
                .checked_add(length)
                .ok_or(Error::ArithmeticOverflow)?;
            if offset < previous_end {
                return Err(Error::InvalidFormat("overlapping tensors"));
            }
            if end > file_len - data_offset {
                return Err(Error::InvalidFormat("tensor exceeds file"));
            }
            previous_end = end;
        }
        let allocation_bytes = reader.allocated;
        let file = reader.inner.into_inner();
        let after = file.metadata()?;
        if after.len() != file_len || after.modified().ok() != modified {
            return Err(Error::FileChanged);
        }
        Ok(Self {
            sources: vec![GgufSource {
                checked: std::sync::atomic::AtomicU64::new(0),
                mapping: OnceLock::new(),
                file: Mutex::new(file),
                file_len,
                modified,
                data_offset,
            }],
            version,
            limits,
            metadata,
            tensors,
            header_bytes: data_offset,
            allocation_bytes,
            metadata_entries: metadata_count,
            total_file_len: file_len,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    /// Aggregate size of all inspected source files, including their headers.
    pub fn file_len(&self) -> u64 {
        self.total_file_len
    }
    /// Primary source's data offset. Split tensors use their own source offset;
    /// this value must never be combined with a nonzero-shard tensor offset.
    pub fn data_offset(&self) -> u64 {
        self.sources[0].data_offset
    }
    pub fn shard_count(&self) -> usize {
        self.sources.len()
    }
    /// Callers must not log raw values: model-management artifacts can define
    /// templates or example messages even though live content is never present.
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        self.tensors
            .get(name)
            .map(|info| TensorView { model: self, info })
    }

    /// Detect in-place size/mtime changes in every inspected source, including
    /// metadata-only shards. This is a consistency guard, not a digest check
    /// against a malicious same-size/same-mtime rewrite.
    pub fn verify_file_unchanged(&self) -> Result<()> {
        for source in &self.sources {
            if source.recently_checked() {
                continue;
            }
            let file = source
                .file
                .lock()
                .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
            source.check_stat(&file)?;
        }
        Ok(())
    }
}

/// A tensor view is tied to its inspected model, preventing accidental cross-file
/// use of offsets. Reads copy only the requested bytes into caller-owned memory.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    model: &'a GgufFile,
    info: &'a TensorInfo,
}

impl<'a> TensorView<'a> {
    fn source(self) -> &'a GgufSource {
        &self.model.sources[self.info.shard_index]
    }
    /// Cache ownership is tied to the same pinned descriptor, not tensor names
    /// or paths that can be replaced. This is transient identity, not a digest.
    pub fn belongs_to(self, model: &GgufFile) -> bool {
        std::ptr::eq(self.model, model)
    }

    pub fn info(self) -> &'a TensorInfo {
        self.info
    }

    /// Clone the already-validated source descriptors for bounded parallel
    /// positional reads. The reader borrows this tensor's original model;
    /// paths are never reopened and no content cache is created.
    pub fn positional_reader(self) -> Result<PositionalReader<'a>> {
        self.model.positional_reader()
    }

    pub fn read_range(self, byte_offset: u64, output: &mut [u8]) -> Result<()> {
        let len = u64::try_from(output.len()).map_err(|_| Error::ArithmeticOverflow)?;
        if len > self.model.limits.max_tensor_read_bytes {
            return Err(Error::LimitExceeded("tensor read bytes"));
        }
        if byte_offset
            .checked_add(len)
            .ok_or(Error::ArithmeticOverflow)?
            > self.info.byte_len
        {
            return Err(Error::OutOfBounds);
        }
        let source = self.source();
        let absolute = source
            .data_offset
            .checked_add(self.info.offset)
            .and_then(|start| start.checked_add(byte_offset))
            .ok_or(Error::ArithmeticOverflow)?;
        self.model.verify_file_unchanged()?;
        let mut file = source
            .file
            .lock()
            .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
        source.check_stat(&file)?;
        file.seek(SeekFrom::Start(absolute))?;
        file.read_exact(output)?;
        source.check_stat(&file)?;
        drop(file);
        self.model.verify_file_unchanged()
    }

    pub fn read_row(self, row: u64, output: &mut [u8]) -> Result<()> {
        if row >= self.info.row_count {
            return Err(Error::OutOfBounds);
        }
        if output.len() as u64 != self.info.row_bytes {
            return Err(Error::InvalidBufferLength);
        }
        let offset = row
            .checked_mul(self.info.row_bytes)
            .ok_or(Error::ArithmeticOverflow)?;
        self.read_range(offset, output)
    }

    /// Read one row, then decode it. Only the bounded packed row is allocated;
    /// decoded storage belongs to the caller. Worker privacy controls apply to
    /// request-derived uses of these buffers and are outside this model reader.
    pub fn decode_row(self, row: u64, output: &mut [f32]) -> Result<()> {
        if output.len() as u64 != self.info.row_len() {
            return Err(Error::InvalidBufferLength);
        }
        if row >= self.info.row_count {
            return Err(Error::OutOfBounds);
        }
        if self.info.row_bytes > self.model.limits.max_tensor_read_bytes {
            return Err(Error::LimitExceeded("tensor read bytes"));
        }
        let count = usize::try_from(self.info.row_bytes).map_err(|_| Error::ArithmeticOverflow)?;
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(count)
            .map_err(|_| Error::LimitExceeded("tensor row allocation"))?;
        packed.resize(count, 0);
        self.read_row(row, &mut packed)?;
        self.info.dtype.decode(&packed, output)
    }
}

fn split_integer(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<u64> {
    metadata
        .get(key)
        .and_then(|value| match value {
            MetadataValue::Unsigned(value) => Some(*value),
            // GGUF writers use signed I32 for split.tensors.count.
            MetadataValue::Signed(value) => u64::try_from(*value).ok(),
            _ => None,
        })
        .ok_or(Error::InvalidFormat("split GGUF integer metadata"))
}

fn metadata_equal(left: &MetadataValue, right: &MetadataValue) -> bool {
    match (left, right) {
        (MetadataValue::Float(left), MetadataValue::Float(right)) => {
            left.to_bits() == right.to_bits()
        }
        (MetadataValue::Array(left), MetadataValue::Array(right)) => {
            left.len() == right.len() && left.iter().zip(right).all(|(l, r)| metadata_equal(l, r))
        }
        _ => left == right,
    }
}

fn align_up(offset: u64, alignment: u64) -> Result<u64> {
    offset
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(Error::ArithmeticOverflow)
}

struct HeaderReader<'a, R> {
    inner: R,
    position: u64,
    allocated: u64,
    file_len: u64,
    limits: &'a GgufLimits,
}

impl<R: Read> HeaderReader<'_, R> {
    fn read(&mut self, output: &mut [u8]) -> Result<()> {
        let len = u64::try_from(output.len()).map_err(|_| Error::ArithmeticOverflow)?;
        self.ensure_available(len)?;
        self.inner.read_exact(output)?;
        self.position += len;
        Ok(())
    }

    fn ensure_available(&self, len: u64) -> Result<()> {
        let end = self
            .position
            .checked_add(len)
            .ok_or(Error::ArithmeticOverflow)?;
        if end > self.limits.max_header_bytes {
            return Err(Error::LimitExceeded("header bytes"));
        }
        if end > self.file_len {
            return Err(Error::InvalidFormat("truncated header"));
        }
        Ok(())
    }

    fn charge_allocation(&mut self, len: u64) -> Result<()> {
        self.allocated = self
            .allocated
            .checked_add(len)
            .ok_or(Error::ArithmeticOverflow)?;
        if self.allocated > self.limits.max_metadata_allocation_bytes {
            return Err(Error::LimitExceeded("metadata allocation bytes"));
        }
        Ok(())
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        self.read(&mut bytes)?;
        Ok(bytes)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self) -> Result<String> {
        let length = self.u64()?;
        if length > self.limits.max_string_bytes {
            return Err(Error::LimitExceeded("string bytes"));
        }
        self.ensure_available(length)?;
        self.charge_allocation(length)?;
        let length = usize::try_from(length).map_err(|_| Error::ArithmeticOverflow)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| Error::LimitExceeded("string allocation"))?;
        bytes.resize(length, 0);
        self.read(&mut bytes)?;
        String::from_utf8(bytes).map_err(|_| Error::InvalidFormat("string is not UTF-8"))
    }

    fn value(&mut self, tag: u32, allow_array: bool) -> Result<MetadataValue> {
        Ok(match tag {
            0 => MetadataValue::Unsigned(u64::from(self.bytes::<1>()?[0])),
            1 => MetadataValue::Signed(i64::from(self.bytes::<1>()?[0] as i8)),
            2 => MetadataValue::Unsigned(u64::from(u16::from_le_bytes(self.bytes()?))),
            3 => MetadataValue::Signed(i64::from(i16::from_le_bytes(self.bytes()?))),
            4 => MetadataValue::Unsigned(u64::from(self.u32()?)),
            5 => MetadataValue::Signed(i64::from(i32::from_le_bytes(self.bytes()?))),
            6 => MetadataValue::Float(f64::from(f32::from_le_bytes(self.bytes()?))),
            7 => MetadataValue::Bool(match self.bytes::<1>()?[0] {
                0 => false,
                1 => true,
                _ => return Err(Error::InvalidFormat("invalid boolean")),
            }),
            8 => MetadataValue::String(self.string()?),
            9 if allow_array => {
                let element_tag = self.u32()?;
                let count = self.u64()?;
                if count > self.limits.max_array_elements {
                    return Err(Error::LimitExceeded("array elements"));
                }
                let minimum_size = match element_tag {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10..=12 => 8,
                    _ => return Err(Error::InvalidFormat("unsupported or nested array type")),
                };
                self.ensure_available(
                    count
                        .checked_mul(minimum_size)
                        .ok_or(Error::ArithmeticOverflow)?,
                )?;
                self.charge_allocation(
                    count
                        .checked_mul(std::mem::size_of::<MetadataValue>() as u64)
                        .ok_or(Error::ArithmeticOverflow)?,
                )?;
                let count = usize::try_from(count).map_err(|_| Error::ArithmeticOverflow)?;
                let mut items = Vec::new();
                items
                    .try_reserve_exact(count)
                    .map_err(|_| Error::LimitExceeded("array allocation"))?;
                for _ in 0..count {
                    items.push(self.value(element_tag, false)?);
                }
                MetadataValue::Array(items)
            }
            10 => MetadataValue::Unsigned(self.u64()?),
            11 => MetadataValue::Signed(i64::from_le_bytes(self.bytes()?)),
            12 => MetadataValue::Float(f64::from_le_bytes(self.bytes()?)),
            _ => return Err(Error::InvalidFormat("unsupported metadata type")),
        })
    }
}
