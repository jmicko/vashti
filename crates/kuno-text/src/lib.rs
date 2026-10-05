//! Native Rust text-engine foundations.
//!
//! The GGUF reader borrows model files read-only and loads only bounded metadata
//! or explicitly requested tensor slices. It never allocates a complete model,
//! downloads artifacts, invokes another engine, or stores inference content.
//! Scalar decoding is a correctness reference, not an optimized inference path.

#![deny(unsafe_code)]

pub mod cpu;
pub mod gguf;
pub mod model;
pub mod ops;
pub mod quant;
pub mod threaded;
pub mod tokenizer;

pub use gguf::{GgufFile, GgufLimits, MetadataValue, TensorInfo, TensorView};
pub use quant::TensorType;
pub use tokenizer::{Qwen35Tokenizer, TokenizerLimits, Utf8StreamDecoder};

/// Errors deliberately omit file paths, metadata values and payload bytes.
#[derive(Debug)]
pub enum Error {
    Io(std::io::ErrorKind),
    InvalidFormat(&'static str),
    UnsupportedVersion(u32),
    UnsupportedTensorType(u32),
    LimitExceeded(&'static str),
    ArithmeticOverflow,
    OutOfBounds,
    InvalidBufferLength,
    FileChanged,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(kind) => write!(f, "model I/O failed: {kind:?}"),
            Self::InvalidFormat(reason) => write!(f, "invalid GGUF: {reason}"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported GGUF version {version}"),
            Self::UnsupportedTensorType(kind) => write!(f, "unsupported GGUF tensor type {kind}"),
            Self::LimitExceeded(limit) => write!(f, "model resource limit exceeded: {limit}"),
            Self::ArithmeticOverflow => f.write_str("model size arithmetic overflow"),
            Self::OutOfBounds => f.write_str("tensor range is out of bounds"),
            Self::InvalidBufferLength => f.write_str("tensor buffer has an invalid length"),
            Self::FileChanged => f.write_str("model file changed after inspection"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.kind())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
