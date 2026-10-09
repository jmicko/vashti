# Original Q6_K integer dense candidate

This is an opt-in laboratory operator, with no engine or mode registration in
this commit. Native Rust emits WGSL; no C library is linked or used for inference.
The unchanged default uses the original F32 arithmetic.

`q6_integer_matvec::prepare` quantizes one finite frozen F32 input into two
32-value Q8 residual planes. The caller owns complete disjoint input, scratch,
output and fault pages, runtime INFO offsets, source identity, precharge,
completion, fault rejection, and erasure. At 2560 columns scratch is 5760 logical
bytes before owning-page rounding. `project` consumes the original 210-byte
Q6_K blocks directly, including signed subscales per 16 elements. It does not
repack, expand, or retain additional model weights. A 248320-by-2560 head still
reads 521472000 original bytes: fifteen 16384-row tiles plus 2560 rows.

The quantizer uses F32 maxabs/127 scales, explicit ties to even, and a second
residual plane. It rejects a non-finite input, reconstructed first product, or
residual before integer conversion. Zero and scale-underflow blocks produce zero
codes. Fault flags are never cleared by the kernels. This arithmetic and the
subgroup reduction differ from original F32 decoding; correctness of the
alternate operator is not numerical parity of the main graph.

`q6_integer_probe::validate_configuration` must precede device creation. A root
registration must require default features, fixed public gate
`KUNO_LAB_Q6_INTEGER_PUBLIC_ONLY=1`, a new `KUNO_PUBLIC_RECEIPT`, actual enabled
SUBGROUP16..32, the checked device limits, and
`private_arena::required_memory_configuration()`. The leaf initializes no
device. It accepts no model paths, prompt text, or other lab inputs. Its four
fixed geometries include the first/full and final head-tile extents, nonzero
byte offsets, partial row groups, two fresh private owners, complete source
padding, owning guards, finite/fault checks and runtime I/O audits. The main
pool is capped at 256 MiB, private owners at 1 MiB, and transient I/O at 128 MiB.
The fixed payload/read/upload/snapshot capacity plan is bounded at 320 MiB host.
Its main-pool release receipt excludes opaque driver resources and physical
recovery; root must verify those. No private production or speed acceptance is
claimed. `dot4I8Packed` can be lowered to scalar emulation by the backend when
native packed-dot capability is absent; protected physical measurement is
needed to establish an actual benefit.

## Independent fixture

`generate.py --source-dir <verified-source-directory>` extracts and compiles
only the pinned original C Q6_K block/decoder with contraction disabled, for
development fixture generation. It verifies both source SHA256 values first.
The independent IEEE half conversion preserves signed zero and subnormals.
The public fixture contains 28 original blocks and 7168 decoded F32 values;
25 finite blocks (6400 values) are checked bit-for-bit, and 3 non-finite half
blocks are rejection fixtures. The first 16 blocks cover every signed-i8
subscale, every Q6 code, and all four low/high quarters. `receipt.json` records
exact source/file digests. No downloaded model bytes are in this fixture.

Primary source: [GGML b11232](https://github.com/ggml-org/llama.cpp/tree/b11232),
`ggml/src/ggml-common.h` (`block_q6_K`) and `ggml/src/ggml-quants.c`
(`dequantize_row_q6_K`). The unpacking layout and per16 signed-scale packed-dot
approach were also compared with [Strata fb58e0](https://github.com/Niko1221/Strata/tree/fb58e0dbc8399662c0e47c76578c6e878b14f6cf),
`src/kernels/cuda/native_mmvq.cu`. Strata's Q8_1 arithmetic and optional model
repacking are not used. The native operator is independently written Rust/WGSL;
the upstream MIT notices are retained in `NOTICE`.
