# Kuno native text foundations

This crate contains a bounded Rust GGUF reader, Qwen35 byte BPE tokenizer,
checked graph operators, and an experimental sequential Qwen4exp graph. The
reference can generate tokens directly from the existing mixed-format model;
no other inference engine is linked. It is not connected to Kuno HTTP routing,
model discovery or the installed service. Production privacy/lifecycle and
matched performance acceptance remain separate milestones.

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
Ordinary concurrent reads are correct but serialized by the descriptor lock.
The separate `positional_reader()` clones the already-inspected open descriptor
and provides bounded Unix positional reads without changing that cursor. It
pins the original model/tensor identity, checks unchanged size/mtime before and
after reads, and clears the full caller slice on I/O or late-consistency failure.
It never reopens by path, maps/copies the complete model or converts weights.
Async/direct I/O, row caching and zero-copy GPU staging remain later work.

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

The graph's exact formulas and artifact scope are recorded in
[the graph contract](../../docs/kuno-native-text-graph.md); live numeric receipts
are in [the performance record](../../docs/kuno-text-performance.md). The
`reference_step` and `reference_generate` examples are operator-only public
synthetic probes, run with swap and core dumps disabled. They are not interfaces
for real user content or request logging.

`Qwen4Exp::bind(&GgufFile)` validates the supported geometry and loads bounded
static norm/conv vectors. `sequence(context_limit)` creates separate transient
state capped at 4096 tokens. `step(executor,sequence,token,cancel,progress)`
executes all 48 layers, including PLE, all ten routed experts plus the shared
expert, GDN and causal block-sparse attention, then returns vocabulary logits.
Greedy sampling has a fixed ascending-ID tie rule; other sampling and vision
are not implemented. Sequence state cannot cross model bindings. Execution
errors discard partial state and require explicit reset. The model binding
never contains request-dependent router, recurrent, KV or PLE caches.

The crate denies unsafe code except focused volatile erase helpers and the
explicit CPU expert module's audited AVX2, Linux affinity and retained-metadata
owners. Model I/O, scalar quant decoding and tokenization have no unsafe code.
Returned token/logit/intermediate buffers remain the worker's cleanup
responsibility. Protected subprocess containment, cancellation during long
operators, crash cleanup, GPU memory erasure and concurrent-request acceptance
are still required before production exposure.

`cpu_expert` adds original packed CPU projections and selected-expert execution,
with explicit portable/AVX2 backends and1..16 unpinned or physical-core workers.
All seven weight layouts retain the GPU's64-lane accumulation/reduction order;
explicit multiply/add avoids FMA and reassociation. Original sign-stable SiLU
remains unchanged. `PackedProjection` and `PackedExperts` borrow one positional
reader and its original tensor views. Calls read only selected original packed
components and return guarded projections or unweighted original-rank partials.
All selected ranks execute, including zero-probability ranks. Routing, one final
rank-ordered probability sum, shared experts and whole-model acceptance remain
the caller's responsibility.

Public packed jobs have a40-MiB cap; controlled private buffers, full retained
job/worker metadata and <=16 workers' scratch have an independent1-MiB component
cap. Admission precedes private fill; actual capacities are checked. Scoped
workers finish before borrowed input/output owners release. Owned private
capacities and metadata erase on completion/error/unwind; original public weight
payloads deallocate normally. No register, compiler-copy, opaque std-thread or
OS-stack erasure claim is made. New scoped threads are created for each phase;
their measured overhead remains an optimization question.

Source acceptance passes105 core tests in debug and release, with two installed
checks ignored, strict Clippy and focused source review. Real-model CPU/GPU
numeric and joint timing comparisons remain pending. These helpers do not
change scalar/default execution or installed HTTP behavior. See the
[decode plan](../../docs/kuno-native-text-decode.md) and
[current source acceptance](../../docs/kuno-text-performance.md#original-packed-rust-cpu-expert-source-acceptance-2026-10-05).
