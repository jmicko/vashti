# Count4 compact shuffle DOWN activation staging

This currently unselected candidate extends the explicit exact-tree shuffle leaf
at `efb7930`. Root integration, source validation and the GPU numerical gate
are complete; normal timing was neutral, so the candidate remains disabled. There is no environment, API, feature or installed selector change.
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

Root integrated `5a64362` as `41b9d3d`, selected the fixed-public trial in
`d8b58b6a6bccd5fea9a68f15e04ec4b2be362a6e`, and completed the full direct-device
suite: **1346 passed, one ignored**. Strict release direct-device Clippy,
formatting and diff checks pass. The frozen v197 build completed with SHA256
`358f23af90229acbef58c00a90000169c86395120ce783985d186db47f67f802`.
The usage interruption initially left the GPU trial pending; the user then
refreshed sudo and the root completed both gates using that exact frozen build.
Full original cold/warm CPU13 vectors, carries, greedies, real repair and all
owner closures pass. Receipt flags confirm shuffletrue, stagedtrue and
primary-planefalse. Normal code200 measured **10.8717765711 continuations/s**:
199 continuations in18.304276095s,52 rounds,147 accepted drafts,5 recomputations.
This is neutral/slightly slower than shuffle10.955189006 and compaction10.906246956.
There is no normal speedup claim and both experimental selectors remainfalse.

The short CPU13 warm duration improved from2.234336549s to1.778226038s,
which did **not** predict normal code200 throughput. Preserve that distinction:
the CPU13 warm fixture and real-code run differ in routes/placement/context.
Do not prioritize further kernels solely from that diagnostic. Both trials
restored admission, actual GPU baseline, HTTP services and unchanged classified
fault counts. Exact evidence is in the v197 aggregate receipts and the
`native-next-v197-staged-down-{warm,code}-intel-20261013` directories under the
development base. No installed/API/Vashti change.

## Fixed-public root experiment

Root integration can select this leaf only under the count4
compact shuffle path, through compile-time `STAGED_PUBLIC_PROBE`. Host adapter
admission calls `validate_staged_limits` for the applicable protected inputs;
shader subgroup/full-coverage and original source guards remain in force. Both
probe receipts expose `compiled_count4_compact_down_staged`. No runtime/API
selector or allocation-cap change. Original primary-plane approximation is
false. Full source tests, strict Clippy and unchanged CPU13 must pass before
normal timing; source selection alone is not numerical or speed acceptance.
