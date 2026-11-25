//! The image engine. Kuno's own executable is the image and video worker (`kuno
//! image-worker-resident`, started by the gateway). Earlier versions could instead launch a
//! separately installed runtime bundle; configurations naming one (`"engine":
//! "managed_sycl"`) are still read and are served by this executable.

/// Whether this executable carries the image engine (the default build does).
pub const EMBEDDED_ENGINE: bool = cfg!(feature = "rust-image");
