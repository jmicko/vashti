# Kuno native text foundations

This crate is a dependency-free, safe Rust foundation for Kuno's future native
text engine. It does not yet implement a model graph, tokenization, generation,
GPU execution or Ollama API integration. No other inference engine is linked.

`GgufFile::open(path, GgufLimits)` parses little-endian GGUF v2/v3 and validates
the complete tensor directory against the file. It keeps one read-only file
descriptor and bounded metadata. It does not map or allocate the whole model,
copy Ollama's artifacts, or download anything. File replacement cannot redirect
the descriptor; size/mtime changes fail subsequent reads. This guard is not a
cryptographic integrity check against same-size/same-mtime modification.

`metadata()` and `tensors()` expose inspected values/geometry. Treat raw metadata
as model-management content: callers must not log templates or token arrays.
`tensor(name)` returns a view bound to the inspected file. Its `read_range` and
`read_row` copy only requested bytes into caller-owned buffers. `decode_row`
allocates one bounded packed row and decodes into caller-owned `f32` storage.
Concurrent reads are correct but serialized by the descriptor lock; async reads,
direct I/O, row caching and zero-copy GPU staging are later work.

Scalar decoding supports F32, F16, BF16, Q4_K, Q5_0, Q6_K and Q8_0. Mixed formats
are preserved per tensor, including the existing Qwen model's 160-element Q5_0
PLE rows of exactly 110 packed bytes. Quantization blocks may not cross row
boundaries. The scalar implementations are correctness references; they do not
promise optimized matrix multiplication throughput. Packing provenance and its
MIT notice are recorded in `NOTICE`.

Default limits are 128 MiB consumed header bytes, 256 MiB conservatively accounted
metadata allocations, 8 MiB per string, 4,194,304 array elements, 16,384 metadata
entries, 65,536 tensors and 64 MiB per tensor read. Callers may lower these limits
or explicitly budget a different known model. Nested arrays, invalid UTF-8,
duplicate keys/names, unsupported tensor types, invalid alignments/dimensions,
integer overflow, truncated payloads and overlapping tensors fail closed.

This crate only handles artifact content. Request lifetime, cancellation,
no-swap/core policy and full-capacity clearing of request-derived buffers remain
the native worker's responsibility; model-reader tests do not establish those
runtime privacy guarantees.

Validation:

```sh
cargo test -p kuno-text --all-targets
cargo clippy -p kuno-text --all-targets -- -D warnings
cargo fmt -p kuno-text -- --check
```

The `inspect` example reports only file/tensor counts, quantization-type bytes and
the known PLE tensor's geometry. It prints no templates, tokenizer data or weight
values and does not request tensor payload slices. Its bounded 16 KiB header
buffer may prefetch bytes immediately adjacent to the header:

```sh
cargo run -p kuno-text --example inspect -- /path/to/model.gguf
```

An ignored installed-artifact check additionally reads one row per required
format and one 110-byte PLE row. It emits no weight values:

```sh
KUNO_TEST_GGUF=/path/to/model.gguf cargo test -p kuno-text --test installed_model -- --ignored
```
