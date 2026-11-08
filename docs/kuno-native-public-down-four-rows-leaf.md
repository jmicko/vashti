# Source-only frozen four-row down leaf

This opt-in leaf retains the original target's packed IQ4_NL and Q2_0 weights,
F32 accumulation, route/member order, owning spans, readiness writes and full
verification. It is not a physical correctness or performance acceptance.

`Geometry::with_four_row_down()` requires the authenticated frozen512,
inactive-exit, tile8, hidden2560/FF640 profile. `four_row_down()` reports the
immutable selection. It composes independently with public codebook placement,
gate/up factors and down scales. Defaults and existing down kernels are unchanged.

The parent must extend its actual `DownKernel` enum and source factories to
choose `grouped_down_frozen_four_rows` or
`grouped_down_frozen_four_rows_shared_scales`. Both have the original down
entrypoint signature. With this geometry selected, `g.grid(groups, true)` is
`(640, groups, 1)` and CubeDim remains64. Gate/up grids remain unchanged. Do not
attach a flagged geometry to an original one-row down entrypoint.

Each physical thread owns one of16 lanes in one of four rows. Its four separate
virtual accumulators retain columns `c = iteration*64 + lane + virtual*16`.
Local additions `(partial0 + partial2) + (partial1 + partial3)` reproduce the
original strides32 and16, followed by the original shared strides8,4,2,1.
Inactive groups exit uniformly before shared writes. All512 reduction cells are
initialized; IQ4 scale sharing uniformly initializes all80 scales before use.

`down_shared_memory_bytes()` reports2048B plain and2368B with IQ4 scales; Q2
remains2048B even with the independent scale flag. This is per-kernel workgroup
scratch, with no new MAIN owner, storage binding or INFO charge. Gate/up SLM is
independent. The complete five-source/30-group down grid falls from384000 to
96000 cubes per layer; each active expert's row count falls2560 to640.

Partial register storage rises from8 to32 F32 values per thread. Original packed
weight traffic and dot work do not shrink. The grid ratio is not a throughput
prediction; register pressure or host PCIe reads can limit the gain.

Focused CPU acceptance is in `src/down_four_row_tests.rs`: all48 profiles,
prerequisites/composition/charges, independent original C fixture weights and
exact64-lane versus four-row dots for both formats, actual checked/unchecked
WGSL and SPIR-V with explicit None workgroup zero initialization, complete
shared-cell initialization, barriers and inactive exit. Parent integration,
actual completed-policy receipts, full original CPU13 GPU gate and matched
normal language timing remain required.
