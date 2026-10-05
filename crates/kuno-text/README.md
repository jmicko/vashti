# Kuno native text foundations

This crate is a Rust foundation for Kuno's future native
text engine. It does not yet implement a model graph, generation,
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

The model reader only handles artifact content; the tokenizer processes transient
request text. Request lifetime, cancellation,
no-swap/core policy and full-capacity clearing of request-derived buffers remain
the native worker's responsibility; model-reader tests do not establish those
runtime privacy guarantees.

The tokenizer is a bounded Rust implementation of the artifact's `gpt2` /
`qwen35` byte-level BPE. `Qwen35Tokenizer::from_gguf` validates the vocabulary,
types, ranked merges, complete byte alphabet and special IDs. `encode(text,
parse_special)` uses the exact Unicode/combining-mark regex, lowest merge rank
and leftmost ties. USER_DEFINED literals always match; CONTROL literals require
`parse_special=true`; the longest literal at the first match wins. It never adds
BOS/EOS or applies a chat template. Unknown variants fail clearly. The Rust
`aho-corasick`, `fancy-regex` and `zeroize` crates provide special matching,
bounded lookahead/Unicode regex and buffer clearing; no C/C++ tokenizer is used.

Tokenizer defaults allow 1,000,000 vocabulary entries, 2,000,000 merges, 16 KiB
per token, 192 MiB conservative construction accounting and 256 specials.
Request bounds are 1 MiB input, 256 KiB per pretoken, 262,144 output tokens,
4,194,304 heap-pop steps and 1,000,000 regex backtracking steps. ID-pair heap BPE
avoids quadratic scans and string concatenation. Output/scratch storage is
reserved before encoding and zeroized on internal cleanup; returned content is
the caller's responsibility. Bounded allocations and worker protection do not
establish complete process privacy on their own.

`token_bytes` returns raw bytes, including incomplete UTF8 token fragments.
`stream_decoder().push_bytes` combines fragments before replacement, emitting
only complete text and retaining at most three bytes. `finish` flushes an
incomplete tail as a replacement character; `clear` and Drop zero the entire
four-byte tail. Temporary decode buffers use zeroization. Chunk reads are capped
at 1 MiB; complete `decode(ids)` output is capped at 4 MiB.

Tests compare exact token IDs against 32 independent synthetic cases produced
by the pinned Python reference, covering Unicode, combining marks, whitespace,
digits, contractions, rank/tie ordering and special types. An additional ignored
installed-model check matches the public Ollama raw benchmark's 75/80 prompt
token counts without inference or printing token IDs/content:

```sh
KUNO_TEST_GGUF=/path/to/model.gguf KUNO_TEST_BASELINE=/path/to/qwen38-baseline.json \
  cargo test -p kuno-text --test installed_model -- --ignored
```

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
