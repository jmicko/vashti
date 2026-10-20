# Native startup source comparison, 2026-10-20

This is a source investigation, not a new loading benchmark or permission to
change loading while prompt-processing acceptance is in progress. Root owns the
GPUs, protected clients and eventual runtime selection. No API/Vashti change is
proposed and no private content is collected.

Existing public receipts under `/media/jmicko/faststore/kuno-development` report:

| Receipt | Startup measurement |
| --- | --- |
| `strata-parity-native-rolling-20261020-clean-long/receipt.json` | `load_ns` 102,394,336,683 |
| `strata-parity-native-wide32-20261020-profile/receipt.json` | `load_ns` 102,554,215,022 |
| `strata-matched-reference-20261020/receipt.json` | `load_observed_seconds` 34.412434660014696 |

The reference value includes parent-observed container/engine startup until
READY. Native `load_ns` includes worker setup through its READY plus the actual
`Session::load`; it excludes waiting for a private first request. These are
useful coarse comparisons, not isolated SSD, allocation, DMA or compilation
measurements. Cold file-page history and runtime/compiler setup can differ.

The native32 actual capacity receipt owns 24,445,798,400 B of resident expert
bytes and 25,846,528,000 B of HOST expert bytes: **50,292,326,400 B total**.
There are 11,998 resident pairs and 12,578 HOST pairs. All 83,617,662,656 B of
the original shards are not copied into these tiers. In particular the large
PLE tensor uses the direct source path. Dense weights, shared embedding/head,
MTP and initialization have additional costs; do not divide the full shard size
by the reported load duration and call that a measured disk bandwidth.

## Concrete selected-path differences

* `strata_global_sources.rs:159–162` loads the GPU cache then HOST complement
  sequentially. `strata_global_gpu.rs:105–140` builds one original 16 MiB
  resident window, uploads it, and joins before building the next. File reads,
  packing and DMA cannot overlap between those windows. Each window is bulk
  zeroed twice: `Vec::resize` followed by `Plan::load_chunk_range_with`'s
  `destination.fill(0)` (`strata_global_plan.rs:573`).
* `strata_global_host.rs:259–304` processes each HOST role serially through a
  single 1 MiB scratch window. It reads the original role, copies it into the
  final mapped page, then zeroizes the scratch. Before filling, the entire
  final mapped allocation is zeroized (`strata_prompt_shadow.rs:83–85`). The
  25.85 GB HOST payload therefore has an extra bounce plus two explicit full
  zero passes; these operations are currently interleaved with single-reader
  I/O. This is a source traffic count, not measured elapsed time.
* `TensorView::read_range` (`crates/kuno-text/src/gguf.rs:533–561`) uses the
  original source mutex, seek/read_exact and pre/post identity checks. Enabling
  mapped weights does not silently change this function into mapped/direct
  reads. The model already provides bounded original-descriptor
  `PositionalReader`/mapped-range helpers; their identity checks remain required.
* Pinned MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`,
  `sycl/src/program/generate.cpp:4715`, calls `gguf_src.mirror(..., 8, ...)`.
  `sycl/src/core/gguf_expert_source.cpp:139–155,185–210` has eight positional
  readers copy each role directly into disjoint final pinned mirror spans.
  This is the selected GGUF mirror path, not the separate experts.bin mmap
  backend. Its cache fill has separate pipelined/serial branches; do not assume
  an inactive branch explains the native gap.

Native static dense preparation additionally joins after every original weight
upload (`strata_static_weights.rs:250–266`). Existing executor `weight_read_ns`
and `weight_upload_ns` cover that weight cache, not complete expert GPU/HOST
construction. Upload timing ends before the subsequent explicit join, so those
counters alone cannot attribute the 102-second total.

## Existing public-only phase attribution

The worker already emits exact complete lines `public_prefill_phase LABEL`
from its explicit compiled-public callback (`strata_worker.rs:417–420`). Private
entry does not emit them. Root's external public helper can timestamp only this
finite label sequence with parent `time.monotonic_ns()` and retain numeric
arrival differences; raw stderr and arbitrary diagnostic labels need not be
saved. The prepared helper also timestamps the first request boundary.

| Exact label | Interval until the next label |
| --- | --- |
| `target` | Target constructor, public vectors, original private/workspace initialization |
| `global_sources` | Static dense preparation, GPU experts, HOST experts/embedding, address tables and final attachment |
| `shared_target_weights` | Shared original head preparation |
| `shared_draft` | Shared Q2 MTP construction |
| first `request_rebind_begin` | End boundary after completed Session loading, before request work |

These correspond to `strata_engine.rs:479,508,515,519,558`. Optional wire READY
arrival (`strata_worker.rs:369`) bounds earlier adapter/model/tokenizer setup
versus subsequent Session preparation; READY-to-target also includes the small
compiled-public fixture preparation. First capacity receipt is another existing
post-load boundary. Arrival times include pipe/thread scheduling and are not
GPU event times. Existing `public_prefill_phase_time` values are request-only
and cannot reconstruct startup intervals from old receipts.

## Bounded proposal after attribution

If global-source loading dominates, first prepare a coherent bounded pipeline:
use a small fixed worker pool of original positional readers to populate
disjoint **final** HOST spans, while retaining complete initialized padding,
pre/post original descriptor identity and source bounds. Avoid an expert-sized
bounce only where all existing checks permit direct final-span reads. Join every
reader before sealing/registering or exposing any page. Any failed worker must
join all readers before original owner cleanup; no partial readiness is granted.

Then consider two bounded 16 MiB resident windows so read/packing of the next
can overlap the previous real DMA. Precharge both source windows and backend
staging, retain them through their actual fences, preserve original full-byte
initialization, and join all submitted copies on success and failure before any
source is reused or released. Do not remove privacy/erasure or public model
initialization protections to obtain a load result. No duplicate model, larger
unbounded source envelope or changed expert placement is needed.

If constructor or MTP instead dominates, investigate that measured interval
first. No loading implementation or startup speed claim follows from this
source-only note. Prompt-processing parity remains the current priority.

## Opt-in bounded implementation, 2026-10-20

The later fixed-public CPU profile
`strata-native-fast-cpu-readyfix-20261020/receipt.json` measured the
`global_sources` arrival at 2.035910497 s and `shared_target_weights` at
97.967252358 s: 95.931341861 s for the interval above. Its USER-only samples
sum to 14.327 s in this interval (4.884 s `memmove`, 3.477 s HOST load,
2.709 s the fill closure and 2.422 s `memset`). This confirms the source
construction interval dominates, but does not attribute the remaining wall time
between kernel work, disk waits and GPU fences. It does not justify eliminating
initialization or identity checks.

`strata-parallel-loading` now selects one connected startup candidate:

* HOST expert payload is read directly into its original already-initialized
  mapped allocation by at most eight scoped readers. They share the original
  `PositionalReader` and disjoint safe mutable slices. Each source read retains
  its descriptor/model identity, tensor bounds, absolute source bounds and
  pre/post stat checks. A replaced pathname is never reopened. Every worker is
  joined, including on spawn/read error or panic, before the original mapping
  can flush, unmap, seal or register. Original allocation padding remains zero;
  embedding and loan-shadow preparation keep their previous paths.
* Resident cache fill uses the same original checked layout and at most 16 MiB
  per source Vec. It retains the actual previous `client.sync()` completion
  future while reading/packing the next window. CubeCL synchronously consumes
  the moved write source and submits its staging copy before this future is
  returned, so one next CPU window overlaps one actual backend window. It joins
  the previous completion before another write or any owner publication. The
  original full-byte window initialization, root identity checks and final graph
  identity checks remain in force.
* The completion guard is armed before scheduling. A next-read error/panic
  joins the actual previous completion; a synchronous scheduling/sync-creation
  failure drains through the original client. A failed or panicked drain retains
  all original destination roots for terminal process teardown. It cannot return
  a ready cache, pretend completion or claim reusable/erased owners.
* Reservation retains the existing two 16 MiB upload allowances and 1 MiB
  source allowance. The option adds exactly 2,621,440 B for eight requested
  256 KiB thread stacks plus 64 KiB per-thread setup/TLS allowance. Job metadata
  stays within the existing separate global cap. These are explicit requested
  bounds, not a claim about unrelated OS allocator/kernel accounting. There is
  no model-sized temporary or new persistent model copy.

The normal/restore cache paths are unchanged when the option is disabled.
No request processing, private profiling, graph scheduling, inference API,
Vashti storage or privacy contract changes. The existing four public phase
markers can measure the whole improvement; this candidate adds no diagnostic
fields or production content logging.

`python3 apps/kuno/benchmarks/text/test-parallel-startup-loading.py` runs eight
direct tests of the actual scoped-reader and upload-completion policies.
`--deps-dir` can link the retained real `kuno_text` and CubeCL libraries without
rebuilding them, adding two tiny fixed-public GGUF tests and type checking the
actual native completion future: **10 tests passed**. Checks cover exact
disjoint fill/tails, initialized padding, invalid geometry, worker error/panic
joins, source pathname replacement versus original-FD mutation/foreign-owner
rejection, actual join-on-error and unknown-completion quarantine. Source gates
also verify the original mapped initialization/flush ordering and actual CubeCL
submit/completion boundary. These checks do not compile the complete changed
worker, perform real DMA/fault injection, validate GPU recovery or measure load
speed. Full opt-in worker compilation and protected positive/recovery/loading
acceptance remain root's gates before runtime selection.

Exact CPU checks run in the isolated startup worktree (all passed):

```sh
python3 apps/kuno/benchmarks/text/test-parallel-startup-loading.py
python3 apps/kuno/benchmarks/text/test-parallel-startup-loading.py --deps-dir /media/jmicko/faststore/kuno-development/root-native-text-target-20261005/release/deps
rustfmt --edition 2024 --check apps/kuno/benchmarks/native-text/src/strata_global_gpu.rs apps/kuno/benchmarks/native-text/src/strata_global_host.rs apps/kuno/benchmarks/native-text/src/strata_global_plan.rs apps/kuno/benchmarks/native-text/src/strata_parallel_reads.rs apps/kuno/benchmarks/native-text/src/strata_upload_completion.rs
git diff --check
```

The additional reservation is **2.5 MiB** (2.62144 decimal MB), with original
two-window capacity unchanged. No actual GPU, system service or installed
binary was accessed or changed by these checks.
