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
The user also approved an exact short expert-trace/replay ceiling. Trace/preload
and runner candidates `990c52d`/`0740b6b` are integrated as `d45dc4b`/`366d847`;
orchestration is `aef04f2`. The protected v76 replay is accepted: seven correct
steps in596125512ns (11.74249tokens/s), all3360 expert calls on GPU, no expert
uploads/CPU work or observed physical disk I/O. Original gates and both extra
full closes pass. This is a short current-engine ideal-residency measurement,
not a hardware maximum. See the
[exact numeric receipt](../apps/kuno/benchmarks/text/receipts/next-public-oracle-20261008.json).

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

## Bidirectional cache transfer and published fast reports

GPU-to-RAM demotion adds real traffic. PCIe can carry opposite-direction
traffic concurrently, and independent copies/computation can overlap when
queues, copy engines, dependencies and host buffers permit it. Intel describes
these conditions in its
[GPU overlap guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/asynchronous-and-overlapping-data-transfers.html).
That is a capability, not proof of simultaneous sustained transfer on this B70.
The current promotion path waits for owning completion before the following
token; it exposes one WGPU queue. Merely adding a victim download would not
remove that dependency. Downloads/uploads can also compete for GPU and host
memory bandwidth. No paired bidirectional benchmark or bandwidth-saturation
attribution has been performed here.

A two-way cache should retain an already available original host bundle rather
than download it again. Demotion of a GPU-only bundle must complete into a
bounded RAM owner before slot overwrite. Compare complete promotion/demotion
wall time and effective token rate against a one-way control before selecting
it. Hiding DMA requires explicit scheduling and independent ownership, not an
assumption that full duplex makes downloads free.

The following are original maintainer measurements checked2026-10-08. They have
not been reproduced on this machine. MTP verifies several proposed tokens in
one pass; its gain depends on how many proposals the model accepts.

| Source | Reported setup and decode | Important difference from our run |
| --- | --- | --- |
| [Strata0.1.26 matrix](https://github.com/Niko1221/Strata/blob/main/bench/results/2026-09-29-speed-0126/README.md) | RTX5070 12GB, Ryzen7600,64GB DDR5-5200; IQ3_XXS61.6 / IQ3_S53.3tokens/s at4K,49.0 /45.5 at128K | MTP spec4, greedy256 output tokens, one sample/cell; loading excluded. Smaller quantizations; Coder rows are excluded because they prune experts. |
| [Strata Intel port](https://github.com/Niko1221/Strata/blob/main/docs/INTEL.md#arc-pro-b70-with-a-model-that-does-not-fit-flash-next-iq3_s-2026-10-07-branch-f-141-intel) | B70, SYCL/xe/kernel7; IQ3_S30.5 prose /40.8 code tokens/s,200 output tokens,32K capacity | MTP spec4; medians of3–5 interleaved pairs. Pinned host mirror, PLE in RAM, measured H2D13.3GB/s; experimental port on another host. |
| [FreeToken-Kai](https://github.com/yuuki-net/FreeToken-Kai#measured-results), [host details](https://github.com/yuuki-net/FreeToken-Kai/blob/kai/docs/kai.md#test-hardware) | Two RTX3060 12GB, i5-12600KF,128GB DDR5-4000, WSL2; NVFP4 experts/FP8 dense:18–20 ordinary decode,30–36 MTP code/tool calls,11–14 MTP prose | Different packed weights, host residency and decode machinery. MTP can also be slower; figures are workload-specific. |
| [Closest Strata four-bit comparison](https://github.com/Niko1221/Strata/blob/main/docs/UNSLOTH_Q4.md#what-to-expect-rtx-5070-12-gb-64-gb-ram-windows-one-short-greedy-prompt) | RTX5070 12GB,64GB RAM, UD-Q4_K_XL111.3GB;40GiB host budget:6.7–8.6tokens/s, mean7.9 across six short greedy runs | MTP enabled; physical expert reads dominate at550–750ms per roughly3.5-token verify round. OS file-cache state changes the result. Still a different quantization/host. |

The reports establish plausible optimization opportunities, including on B70,
but do not establish30–50tokens/s for our119.15GB Q4_K_M artifact. Our current
75GiB expert payload and ordinary single-token decode are materially different.
The four-bit disk-limited report is the closest memory-footprint comparison.
Measure ideal correct GPU residency first. Then compare exposed CPU/I/O time,
remaining GPU/dispatch time and multi-token execution; no cache-only speed
promise follows from these reports.

## Smaller-quant feasibility follow-up

The user suggests a smaller quant. The full ISTA IQ3_S, retaining all512 experts,
is a meaningful candidate: two published shards total83617662656bytes
(77.875GiB), versus110.968GiB for our existing model. Shards are54817524224 and
28800138432bytes at revision `ed59f92082b1e93c0e96d60a8b11aab089b52f09`.
See the [pinned artifact](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09/IQ3_S).

Its [published mixed-format allocation](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/blob/main/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.rco-allocation.txt)
lists seven currently unsupported types: IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS,
Q2_0 and Q5_K. This allocation document was fetched read-only on2026-10-08,
SHA256 `df10add9e3389cf5930e1dd527df4c0b24b38597a7e4652aa3a1ab2d3e12fd9b`;
validate actual pinned GGUF headers before implementation. No model payload
was downloaded or inference performed with it.

Current `kuno-text` supports seven other scalar formats, owns one GGUF file,
and rejects unsupported tensor types while parsing. Native banks/optimized CPU
experts restrict original Q4_K gate/up and Q5_0/Q8_0 down; native PLE restricts
Q5_0 rows. IQ3_S therefore requires split-file ownership, the missing scalar
and native GPU formats, IQ4_NL PLE, format-specific CPU fallback/optimization
and a validated bank/admission profile. Preserve routing/cache/completion
machinery and the original Q4 acceptance. Establish a separate quant-specific
reference/output fixture rather than requiring Q4's output IDs or weakening
numeric/owning gates. The smaller footprint may reduce capacity/traffic, but
cannot by itself remove the current11.74token/s replay's GPU/synchronization
cost. No smaller drop-in artifact has been verified.
