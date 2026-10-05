//! Bounded GGUF v2/v3 metadata and read-only tensor access.
//!
//! Tensor offsets and sizes are validated before returning a model. Unsupported
//! formats, overlapping tensors, malformed dimensions and unbounded allocations
//! fail closed. An open descriptor pins the inspected inode; replacement paths
//! cannot redirect subsequent reads to a different model.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

use crate::{Error, Result, TensorType};

#[path = "gguf_positional.rs"]
mod positional;
pub use positional::PositionalReader;

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
    /// Tensor-data-relative offset, not an absolute file position.
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
}

/// Read-only model descriptor and bounded retained metadata. Debug formatting is
/// intentionally omitted: metadata may include model-defined templates.
pub struct GgufFile {
    file: Mutex<File>,
    file_len: u64,
    modified: Option<SystemTime>,
    version: u32,
    data_offset: u64,
    limits: GgufLimits,
    metadata: BTreeMap<String, MetadataValue>,
    tensors: BTreeMap<String, TensorInfo>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>, limits: GgufLimits) -> Result<Self> {
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
            };
            if tensors.insert(name, info).is_some() {
                return Err(Error::InvalidFormat("duplicate tensor name"));
            }
        }
        let data_offset = align_up(reader.position, alignment)?;
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
        let file = reader.inner.into_inner();
        let after = file.metadata()?;
        if after.len() != file_len || after.modified().ok() != modified {
            return Err(Error::FileChanged);
        }
        Ok(Self {
            file: Mutex::new(file),
            file_len,
            modified,
            version,
            data_offset,
            limits,
            metadata,
            tensors,
        })
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn file_len(&self) -> u64 {
        self.file_len
    }
    pub fn data_offset(&self) -> u64 {
        self.data_offset
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

    /// Detect in-place size/mtime changes. This is an inexpensive consistency
    /// guard, not a digest check against a malicious same-size/same-mtime rewrite.
    pub fn verify_file_unchanged(&self) -> Result<()> {
        let file = self
            .file
            .lock()
            .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
        self.check_stat(&file)
    }

    fn check_stat(&self, file: &File) -> Result<()> {
        let stat = file.metadata()?;
        if stat.len() != self.file_len || stat.modified().ok() != self.modified {
            return Err(Error::FileChanged);
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
    /// Cache ownership is tied to the same pinned descriptor, not tensor names
    /// or paths that can be replaced. This is transient identity, not a digest.
    pub fn belongs_to(self, model: &GgufFile) -> bool {
        std::ptr::eq(self.model, model)
    }

    pub fn info(self) -> &'a TensorInfo {
        self.info
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
        let absolute = self
            .model
            .data_offset
            .checked_add(self.info.offset)
            .and_then(|start| start.checked_add(byte_offset))
            .ok_or(Error::ArithmeticOverflow)?;
        let mut file = self
            .model
            .file
            .lock()
            .map_err(|_| Error::InvalidFormat("model reader lock poisoned"))?;
        self.model.check_stat(&file)?;
        file.seek(SeekFrom::Start(absolute))?;
        file.read_exact(output)?;
        Ok(())
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
