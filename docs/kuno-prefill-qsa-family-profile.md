# Fixed-public QSA family phase profile

The existing compiled-public prefill profiler now splits the complete
`strata_prompt_qsa_full::run` family into four consecutive fenced intervals.
This is diagnostic decomposition, not a new private profile option, arithmetic
change or production inference barrier. No API, request wire, state authority,
owner, primitive parameter or original operator order changes.

| Public label | Exact existing execution covered |
| --- | --- |
| `qsa_family_producers` | Positions, Q split, Q/K/IQ norm+RoPE and original flag joins, Q8 pack, initial dead key, append, shared pooled means/tail, pooled norm+RoPE and final pooled flag join. |
| `qsa_family_score_select` | Every original256-query batch's active-grid score kernel, score validation and weighted radix selection. |
| `qsa_family_attention_merge` | Every original32-query tile's actual prompt attention and merge kernels; scratch is reused in the original tile order. |
| `qsa_family_gate` | Original packed FP16 gate, including its original current activation owner validation. |

`Timer::for_public_arena` returns disabled BEFORE any arena/client lookup when
the existing compiled-public `ENABLED` collector is off. Private inference
therefore adds no client sync, enqueue, flush or request-dependent timer. Only
the same explicit `--public-prefill-profile` startup which already creates the
coarse collector can enable these subintervals. The terminal public negative
entry does not enable it. Each mark retains the original `Timer::mark` real
completion sync and numeric `[calls,total_ns,enqueue_ns]` format. No new backend
query authorization, arbitrary private flag or separate data store is created.

The outer `qsa_attention` interval still surrounds the whole family; the four
new values are its breakdown and MUST NOT be added to that outer total again.
Extra public fences can change diagnostic scheduling and reuse. Their totals
include host enqueue/native capture/fence waits, not just isolated GPU kernel
execution. Compare their own whole request with the prior public run and a
separate clean controlled request; do not call a fenced public profile clean
throughput.

## Correct scope of the current Strata comparison

The complete current c680 public third-request receipt is
`strata-parity-vector-gdn-20261020-profile/receipt.json` in the external benchmark
root. It reports24 QSA layer/chunk calls:330.689ms input projections,
1842.440ms whole QSA family and108.855ms output projection, total2281.984ms.
Those existing public timers could not identify how much of1842.440ms was
producers, scoring/selection, attention/merge or gating.

The matched original pinned Strata warm diagnostic reports `qsa proj`99ms,
`qsa indexer`0ms (rounded), `qsa select`70ms and `qsa attn`680ms, total849ms.
Its `qsa proj` includes Q/K/V/IQ/IK projection, ordinary pre-attention
normalization/RoPE/append/split and the later gate/output projection:
`sycl/src/prefill/prefill.cpp:2890-2968,3161-3165`. Its `qsa attn` covers only
the attention+merge loop (`:3127-3161`). Thus Rust1842ms versus Strata680ms is
NOT a measured attention-primitive ratio. The complete QSA aggregates are the
better comparison, still bounded by diagnostic timing conventions and different
host costs. Original `PfTimer` event intervals include endpoint and queue idle
time; Rust intervals use actual completion fences.

Actual selected source is the default FP32 Q8 split-chunk attention, not an
inactive CUDA tensor path or optional XMX shortcut. `qsa_prompt_attn.dp.cpp`
returns false for Intel unless the XMX function opts in; its default mode is0.
The prefill fallback calls `qsa_decode_attn_batch` in32-query tiles,256 threads,
SG32,64 selected cells per chunk,12 heads per KV head and2 KV heads. The
selected default `LANE_CELL` flag is false. CMake uses precise FP math,
correctly rounded divide/sqrt, SG32 and per-kernel AOT splitting; it has no
large-GRF override (`sycl/CMakeLists.txt:39-67`). CPU reading the retained
reference's selected Q8 AOT `.ze_info` confirms SIMD32,128 GRFs,15872-byte SLM
and no private/spill memory allocation declared for that kernel. Current native
ISA/register/spill statistics have not been collected, so no compiler occupancy
or actual SLM-vector-message equality is claimed from Naga/SPIR-V alone.

## Source/CPU acceptance and next gate

`python3 apps/kuno/benchmarks/text/test-public-qsa-family-split.py` passes two
CPU tests compiling the ACTUAL admission method with metadata-only stand-ins:
disabled private collection never looks up an arena client or starts a timer,
and enabled collection uses the original arena and preserves its error. Source
checks cover all four real stage boundaries, original tile/kernel order and no
direct private synchronization. This is admission/source validation, not an
actual GPU fence, worker compile or numerical/performance acceptance.
The existing sampler's three CPU tests also pass after root's separate
terminal-entry literal update. Rustfmt/diff checks pass. Root owns integrated
build, actual protected positive/negative hardware gates and physical recovery.

Do not optimize the selector blindly. Its register-key and fixed-loop expansion
still differs from the actual unrolled selected Strata kernel; the new timings
must first establish its contribution. Similarly a WGSL `exp`/SYCL `native::exp`
source difference is not proof of a costly math-library path without actual
backend lowering/ISA evidence. Preserve current IEEE/finite/fault/tail controls.
