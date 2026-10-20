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
