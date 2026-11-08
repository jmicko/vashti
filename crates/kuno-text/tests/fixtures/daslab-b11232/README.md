# Public independent DASLab packing oracle

The committed synthetic blocks use original GGML IQ2_S (ID22), IQ3_S (ID21),
Q5_K (ID13), and Q2_0 (ID42) layouts. Expected F32 bit patterns come from the
independently compiled scalar C dequantizers in llama.cpp **b11232**, never
from Kuno's scalar or SIMD implementation. There are no model weights, prompts,
inference outputs, or private user data in these files.

`generate.py` checks exact SHA256 digests before extracting the four upstream
structs/dequantizers, the original scale/min helper and public IQ codebooks.
The C oracle uses independent IEEE half equations and is compiled with
`-ffp-contract=off`. Python, a C compiler and upstream downloads are needed only
for regeneration. Kuno runtime and ordinary Rust tests consume the committed
bytes and never invoke that reference engine, a C library, or the network.

A record is one packed block followed by its exact little-endian F32 output
bits. `receipt.json` records each GGUF ID, block count, geometry and SHA256.
Run `python3 generate.py`, or pass `--source-dir` containing the exact pinned
`ggml-common.h` and `ggml-quants.c` to regenerate without network access.

Coverage:

- IQ2_S: first 32 blocks cover all 1,024 grid entries, all 256 original stored sign
  bytes and all 16 subscale nibbles. A bijective index permutation mixes the
  high two-bit fields across the four octets rather than keeping them uniform.
- IQ3_S: first 8 blocks cover all 512 grid entries, all 256 stored sign bytes and
  all 16 odd-scale codes. A second bijective permutation mixes each high index
  bit and both quartet positions within every eight-value octet.
- Q5_K: first 8 blocks cover all 64 scale/min codes, split high/low scale fields,
  all 32 quant values and their independent high-bit planes in every group.
- Q2_0: all four codes occur in all consecutive two-bit positions, including
  the fourth code's `+2` value. The block is 64 elements/18 bytes; a 640-value
  row occupies180 bytes. This is distinct from Q2_K and TQ2_0.
- Each format also uses signed zero, minimum/maximum subnormal, minimum normal,
  positive/negative and non-power-of-two scales, plus the maximum finite half.
  Q5_K additionally varies its separate minimum half through those boundaries.

Rust acceptance compares all 20,032 scalar and SIMD fixture values bit for bit,
then tests SIMD dot products against independently stored C weights with the
unchanged F32 original 64-lane accumulation tree. Native-shaped CPU matrices and
complete experts are checked too. This does not prove GPU parity, whole-model
quality, or performance; those require separate protected acceptance runs.
