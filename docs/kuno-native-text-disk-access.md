# Qwen3.8-Flash-Next disk access, 2026-10-08

Research and completed fixed-public laboratory measurements. This does not
change model placement, inference behavior, installed services, or the API.

## Expected behavior

Qwen's deliberate off-accelerator component is the additional51B n-gram
embedding table. Its paper describes host-memory lookup and asynchronous
prefetch at Layer2. It does not require continuously paging MoE experts from
disk. See the [architecture paper, section2.3](https://arxiv.org/html/2608.30320v1).

Disk-backed lazy lookup is nevertheless a supported implementation tradeoff.
The installed daemon reports Ollama0.35.1; that release
[pins llama.cpp b11232](https://github.com/ollama/ollama/blob/v0.35.1/LLAMA_CPP_VERSION).
Its [CLI documentation](https://github.com/ggml-org/llama.cpp/blob/b11232/tools/cli/README.md)
defines mmap loading and lazy embedding-row reads, enabled automatically for
eligible tensors above4GiB. This is separate from CPU expert placement. The
exact running Qwen worker's effective lazy setting was not measured here; source
defaults are not a new Ollama performance receipt.

The [exact Ollama artifact](https://ollama.com/library/qwen3.8-flash-next:125b-a6b-q4_K_M/blobs/89059c880c2f)
matches local blob89059c880c2f. Live file size is119150722112bytes. Its n-gram
tensor `per_layer_token_embd.weight` is Q5_0, dimensions160×320001536:
35200168960bytes, about32.78GiB. A large file does not imply every page must
be resident to generate a token.

## What the native laboratory actually does

| Component | Current source access | When physical disk I/O can occur |
| --- | --- | --- |
| N-gram lookup | `resident_ple_batch::embedding` requests16 selected110-byte rows per token through `TensorView::read_row`/`read_range`, using seek/read on the original descriptor; the standalone `ple_chunk` path has the same bounded lookup | A requested file page is absent from Linux's page cache; reads can include readahead |
| CPU expert | `cpu_expert` uses original mapped packed ranges and selected-range WILLNEED advice through `gguf_positional` | First touch or access after eviction of a mapped source page; advice may initiate reads |
| Adaptive GPU promotion | `expert_banks::prepare_public_existing` reads original selected weights into a bounded public owner before uploading into existing GPU slots | Source pages are absent from the page cache |

`read_range` calls can be served entirely by cached RAM. Conversely, mapped
CPU access can cause disk I/O without an explicit read call. “CPU miss” means
an expert is absent from VRAM, not that it is absent from RAM. PCIe weight
uploads and storage reads are separate counters.

Original token-embedding row reads and a cold dense-cache fill can also access
the descriptor. These totals are process-wide, not an exclusive three-category
accounting system. No per-component physical-I/O attribution exists yet.

N-gram logical requests total1760bytes/token, about1.72MiB across1024 steps.
Physical reads need not equal that amount because pages and readahead are
larger. Hashing uses the current token and two predecessors; after the repeated
public128-ID sequence crosses its first boundary, its n-gram lookups repeat.
Its expert routes can still change with attention and recurrent state.

## Completed longer diagnostic

Both v75 runs use one frozen binary, after the unchanged original128-ID
fixture closes. They then create a fresh1227-token context and teacher-force
the same public sequence for1024 additional steps. All1024 greedy results
match exactly between modes. This is a cache-settling diagnostic, not ordinary
free-running generation or a general throughput acceptance test.

| Measurement | Fixed144-slot placement | Fully adaptive144-slot placement |
| --- | ---: | ---: |
| Total physical reads across1024 steps | 20.922GiB | 20.430GiB |
| First128-step window reads | 8.161GiB | 7.693GiB |
| Last128-step window reads | 0.550GiB | 0.889GiB |
| First window tokens/s | 5.990 | 6.159 |
| Last window tokens/s | 7.475 | 9.405 |
| Last window VRAM expert-hit fraction | 29.97% | 89.18% |

All windows record zero physical writes. The adaptive run completes4086
promotions and uploads12.387GiB of original weights. Its disk total is therefore
not a counter of promotion bytes. The fixed run has no promotion uploads and
still reads20.922GiB. The adaptive run followed fixed, so OS cache warming is
an order confound. These samples do not isolate the whole speed gain or prove
that a particular physical read came from the n-gram table or an expert.

Our counters use `/proc/self/io` `read_bytes`, which measures actual storage
traffic on this block-backed filesystem, rather than requested read lengths.
See [kernel I/O accounting](https://docs.kernel.org/filesystems/proc.html).
The decline in reads and major faults is consistent with warming. It does not
establish either zero eviction or repeated hot-page thrashing. The observed
adaptive cgroup had zero memory-limit/OOM events and a17.2GiB reported peak;
there is no evidence here that the48GiB limit caused reclaim. That observation
must not be extrapolated to the completed fixed run's unrecorded memory history.

The next attribution diagnostic should separate logical n-gram/expert/promotion
access and sample file-page refaults, reclaim, memory-pressure time, and storage
I/O over the same windows. [`workingset_refault_file`](https://docs.kernel.org/admin-guide/cgroup-v2.html)
specifically counts accesses to previously evicted file pages. Use aggregate
numeric diagnostics only; private rows, routes and token histories remain
transient. Do not change the six-field accepted steady schema in place.

## Interpretation and resume point

Cold reads are acceptable. The optimization target is unnecessary repeated
reads and exposed stalls, not a blanket zero-disk-read requirement. Neither
the current evidence nor total model size justifies installing extra RAM yet.
Earlier claims that warm transfers always originate in resident RAM apply only
to the short zero-read samples, not all requests or the longer diagnostic.

Source and artifact hashes, all windows, original acceptance and cleanup are
recorded in [the numeric receipt](../apps/kuno/benchmarks/text/receipts/next-adaptive-steady-20261008.json).
The user also approved an exact short expert-trace/replay ceiling. Its isolated
source candidates are `990c52d` (trace/preload seams) and
`0740b6bd741e8e379b2168caadd0b6ad9c6e1139` (runner controls), not yet integrated
or device accepted. Root still needs the orchestration module and main hook.
Do not substitute wrong experts or report that pending ceiling as measured.

## Physical memory and GPU eviction follow-up

Live read-only inventory confirms two32GB DIMMs configured at2667MT/s. Linux
reports67335241728bytes of system RAM; the B70 reports34242297856bytes of device
memory and NVIDIA reports12288MiB. Use one consistent binary unit below:

| Quantity | GiB |
| --- | ---: |
| Main model file | 110.968 |
| Linux-visible RAM | 62.711 |
| B70 physical VRAM | 31.891 |
| RTX3060 VRAM | 12.000 |
| Combined visible RAM and VRAM | 106.602 |
| Model exceeds combined physical capacity by | 4.366 |

These are total capacities, not free bytes or a unified allocatable pool.
Inference also needs OS/application memory, private request state, runtime
buffers and allocation padding. Duplicate host/GPU copies consume capacity
twice. Current selected native runs use B70 and CPU only; a tested stationary
NVIDIA bank was slower and remains excluded. Adding16GiB RAM would increase
raw combined capacity to roughly122.6GiB, but does not establish that the whole
model plus current duplicates/buffers fits or runs faster at a lower RAM speed.

The original routed expert payload is75GiB. Populating144 slots in each layer
retains21.094GiB of logical expert bytes on B70. v75 reports25.205GiB for the
whole MAIN allocator reservation, not the expert banks alone. The n-gram table
is32.783GiB but only selected rows are required per token. Keeping all experts
and useful table rows is a more relevant goal than preloading the whole file.

Adaptive eviction overwrites a selected slot inside its retained GPU roots;
it does not download the victim. The residency ledger changes only after
original owning completion. Weights are immutable original quantized bytes,
so dropping the GPU copy loses no computed model state. Original bytes remain
on disk and may remain in the OS file cache. The temporary upload vector is
released; there is no dedicated retained host expert payload cache in this
configuration. Adaptive expert source pages skip explicit discard advice,
but this neither pins them nor guarantees their continued RAM residency.

GPU-to-RAM victim migration is feasible. It can preserve an expert whose host
pages are no longer resident. It also adds readback/PCIe/completion work and
requires an independently bounded host cache before the GPU slot is overwritten.
If source bytes already exist in RAM, downloading a second copy is wasted work.
A future policy could retain original host bundles when useful and demote
GPU-only victims to RAM when retention benefit justifies the transfer. Count
unique coverage and duplicated bytes separately; reserve OS/runtime headroom.
No cache or eviction behavior was changed for this inventory.

## Actual model-drive read speed

Read-only tests used the existing model file on the Samsung990EVOPlus2TB NVMe.
Its live link reports8GT/s×4. Aligned O_DIRECT reads bypassed Linux's page cache;
there were no writes, cache flushes, GPU operations or service changes.

| Access pattern, queue depth1 | Measured speed/latency |
| --- | --- |
| Two2GiB sequential samples,8MiB requests | 3.332 and3.493GB/s (3.103 and3.254GiB/s) |
| Two256MiB scattered samples,1MiB requests | 2.209 and2.213GB/s (2.057 and2.061GiB/s) |
| 1024 scattered4KiB requests | 25.275MB/s;132.156µs median,148.212µs95th percentile |

Random offsets use seed424242 and4096-byte alignment. Sequential samples start
at4GiB and64GiB; the two large scattered samples use successive distinct random
offset sets. The small scattered sample resets that seed. These short samples
measure this drive/file/link, not long thermal behavior or the engine's effective
expert throughput. Engine reads use the page cache and overlap with CPU/GPU
work; direct-I/O speed does not predict token rate by itself. Small sparse reads
are latency-sensitive even when the drive has high sequential bandwidth.
