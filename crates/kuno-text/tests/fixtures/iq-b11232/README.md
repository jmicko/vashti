# Public independent IQ scalar oracle

These are synthetic packed blocks plus the exact F32 bits produced by the
independently compiled llama.cpp **b11232** scalar C dequantizers. The fixture
contains no model weight, inference prompt, output, or user content.

`generate.py` downloads only the pinned public `ggml-common.h` and
`ggml-quants.c`, requires their exact SHA256 values, extracts the three upstream
structs/dequantizers and their codebooks verbatim, and compiles them in a temporary
directory with contraction disabled. Its half conversion uses separate IEEE
mathematical equations. No Kuno Rust code is used to generate expected values.
Python and a C compiler are development tools only. Ordinary Rust tests consume
the committed fixture and do not require either tool or the network.

A record is one original packed block followed by its little-endian F32 output
bit patterns. `receipt.json` records complete source hashes, block counts,
geometry and fixture hashes. Run `python3 generate.py` to regenerate; optionally
pass `--source-dir` for an offline directory holding the exact pinned sources.

Coverage:

- IQ3_XXS: first four blocks exercise all 256 nonlinear grid entries, all 128
  parity sign codes, both grid quartets in every octet and all 16 scale codes.
- IQ4_NL: all 16 nonlinear values in both nibble halves, including sign changes.
- IQ4_XS: first eight blocks exercise every signed six-bit subscale (-32..31),
  all high/low field locations and all nonlinear values in both nibble halves.
- Each format also includes positive/negative zero, minimum/maximum subnormal,
  minimum normal, positive/negative scales, a non-power-of-two scale and maximum
  finite binary16 scale. Every output bit, including signed zero, is compared.

This validates **scalar decoding**. It does not establish optimized CPU SIMD,
GPU shaders, complete model execution, quality or performance for these formats.
