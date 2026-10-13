# Count4 compact shuffle DOWN activation staging

This unused source-only candidate extends the explicit exact-tree shuffle leaf
at `efb7930`. No parent, environment, API, feature or installed selector changes.
`launch_compact_shuffle_staged` shares every original compact launch check and
uses the separate appended `ProjectCompactShuffleStaged` kernel identity.
Original compact/shuffle entries and count1..3 tails are unchanged.

All64 local invocations, after the same uniform width, record, geometry,
original MAP member and source-readiness guards, initialize one1440-word
workgroup array. Each member's360 words are copied from its original arbitrary
route frame in the full DOWN owner; unused members become zero. The64-lane
stride covers each word exactly once. One converged barrier precedes consumers.
The four frame offsets then address this cache. Original two-plane packed dots,
weight decoding, F32 scales/additions, eight-lane shuffle tree, actual original
route outputs/status and complete owner audits remain unchanged. There is no
host readback, routing substitute, new MAIN/INFO owner or allocation-cap change.

The cache costs5760B of workgroup storage. Explicit parent admission must call
`validate_staged_limits` in addition to enabling the shuffle16/32 subgroups;
ordinary DOWN limits remain unchanged. This bound is below the existing16004B
GU requirement of the intended composed path. No physical device acceptance is
implied by that arithmetic.

Scheduling inspiration is MIT Strata revision
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, SYCL
`sycl/src/kernels/cuda/iq_kernels.dp.cpp`, `native_down_multi_kernel` around
2050–2113: it stages original quantized activations before grouped DOWN dots.
Existing `iq_moe_integer-NOTICE.txt` retains attribution. Kuno keeps its original
two-plane representation and arithmetic rather than Strata's single-Q8 format.

Each eight-row workgroup previously issued360 activation-word reads per live
member per row. Staging issues360 per live member once: source-level8x fewer
storage reads for those planes, with equivalent workgroup reads afterward.
For40 real routes and2560 output rows, that is147456000 to18432000 source-level
bytes per layer. These are load-expression counts, not measured DRAM traffic;
cache hits, the added barrier and SLM occupancy can make this neutral or slower.

Added tests lower both original DOWN formats, check one barrier,5760B SLM,
64 lanes, unchanged arithmetic/guards and distinct identities. Independent
layout tests cover lengths1..4, noncontiguous route permutations, exact u32 bit
copies, single reset writers, inactive zeros and untouched full source owners.
Projection checks compare original and staged bits for both formats. These are
source proofs; full original CPU13 vectors/carries/greedies, real repair,
owner/source closure and fresh normal timings remain required before acceptance.

**Tests, compilation, Clippy and GPU validation have not been run for this
leaf.** Root owns the shared cache and is retaining compiler-idle timings.
Rust formatting and whitespace checks pass at this source-only handoff.
