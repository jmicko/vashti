# Original three-row CPU HOST / GPU resident overlap operator

## Proposed scope, 2026-10-11

This is a review proposal, not an implemented or accepted execution path. The
isolated branch starts at root `2dbc897`. Implementation requires root's review
of this contract. No compilation, device operation, service change, source
activation, public API, installation or performance claim accompanies it.

The first candidate is one fixed original DASLab layer (`layer0`, H2560,
FF640, 512 original experts), with **exactly three real synthetic input rows**.
Use the already authenticated frozen144 VRAM slots and368 HOST complement on
the same Intel client. Retain the existing actual nine read-only HOST maps;
do not read/map the GGUF again, duplicate weights, replace an expert, change
source placement, fill/evict a bank, or change its immutable generation.

The [CPU borrowed batch](kuno-native-borrowed-cpu-batch.md),
[same-allocation bridge](kuno-native-public-cpu-host-borrow.md), and
[original joint calibration](kuno-native-text-cpu-gpu-calibration.md) are the
prerequisites. The existing `public-host-cpu-batch-probe` remains unchanged.
Its HOST-only partial timings establish neither a complete hybrid layer nor
overlap. The old count1 hybrid owner/map is not permission for three rows.

Four rows are deferred. The current CPU batch and original frozen GPU wave
guard admit only1..3. A later four-row candidate needs a separate explicit
CPU bound, real four-input computation, original GPU control and complete
four-row owner proofs. Do not widen these existing guards or use a fabricated
fourth row to connect it. This experiment does not depend on that extension.

## Source and lifetime authority

Start with the existing fixed-public construction sequence: original full
CPU13 cold check, completed144-slot preparation, accepted warm reset, original
source attachment, original warm token0, and completed native projection
boundary. The one-layer guard retains the actual model/workspace/executor,
Attachment, PreparedPlan, positional reader, original Plan and same client.
No native request, actual model route, MTP execution or four-row policy is
activated during the synthetic operator.

All48 bank identities/generations and the attached source identity remain
checked through the existing `check_public_host_cpu_borrow` and Plan checks.
The nine HOST pages remain Registered, with original role/page coordinates,
bindings and physical VkDeviceMemory identity. HOST_CACHED/HOST_COHERENT are
reported actual placement properties; no cache invalidate/flush or source
write is introduced. CPU and GPU may read the same sealed bytes concurrently.

The narrow new GPU seam is an opaque method on `CpuLayerBorrow` accepting
the existing seven arena IDs in native order
`[MAP, INPUT, ACT, PART, PROB, OUTPUT, STATUS]`, original Spans, count3,
group count and cancellation. A fixed internal source-scope enum admits either
all five original partitions or the two resident partitions. No public enum,
bare handle, mutable source byte or caller callback is returned. The existing
HOST-only method retains its original signature and behavior.

The resident branch must obtain the actual original six bank-root bindings
through a narrow executor-owned helper, inside the borrow. It rechecks the
same Plan/graph/client/frozen bank identities, original tensor names/types and
both72-slot segment extents. Full handle offsets may be absent or explicit0;
nonzero trims reject before used-size subtraction. Actual resource offset,
used/logical span, backing size, stream and physical nonaliasing are proved
before enqueue. Conservative bank admission and its actual buffer proof are
unchanged; this helper cannot credit exact-persistent allocation or extra slots.

The source guard stays on the coordinating thread. Only already checked
immutable ExpertViews, Member/Rank headers, inputs and the existing pooled
operator are borrowed by a scoped CPU task. Do not make the map guard Send/Sync
using unsafe annotations. All CPU row jobs and the outer scoped task join
before views or maps release. The GPU launcher retains its existing armed
join/drop behavior and actual same-stream join on success/error/unwind.
The terminal Attachment close still owns detach, full source zero/audit and
free. CPU unmap is not a source-release receipt.

## Equal work and three arms

Every packet contains three distinct actual H-wide input vectors and ten
original ranks per member. IDs come from the authenticated resident map and
HOST complement; their transient values are never serialized. Within each
member IDs are unique. Shared-expert cases deliberately reuse experts across
members at different original ranks; disjoint cases use30 distinct experts.
GroupMap retains original member/rank destinations and exact probabilities.

The same artifact runs all three arms against the same source state:

| Arm | Complete computation |
| --- | --- |
| AllGpu | Five original-F32 GU partition calls, five original-F32 DOWN calls, then the unchanged grouped_reduce over every original rank. This includes the actual HOST partitions; it is not the old HOST-only control. |
| SerialCpuThenGpu | Actual GPU-input handoff; joined original CPU batch for HOST ranks; real sparse partial encoding/upload; two original VRAM GU calls and two DOWN calls; completed original-rank merge. CPU and GPU phases are deliberately serialized. |
| Concurrent | The same checked handoff and CPU work starts in a scoped task while the coordinating thread enqueues/joins the same two VRAM partitions. Join both consumers, upload real CPU partials and complete the same merge. |

Source partitions remain `(0,72)`, `(72,72)`, `(144,144)`, `(288,144)`,
`(432,80)`. All GU calls for a GPU arm precede its DOWN calls. Original weight
addresses, F32 multiply/add, 64-lane reduction tree and SiLU remain unchanged.
Original count3 GPU arithmetic and CPU arithmetic are compared under their
existing tolerance; source assertions alone do not prove identical device bits.

When all ranks are resident, explicitly skip the CPU call. When all ranks are
HOST, explicitly skip resident GPU calls. Neither case is credited as actual
CPU/GPU overlap. No zero-sized grid or ready flag stands in for missing work.
Record actual successful calls and joined consumers, separately from mode.

The proposed first compiled cases use the existing explicit15 physical-core
AVX2 pool: HOST ranks5 shared,5 disjoint,0 shared,10 shared, and sparse ranks
`{0,4,9}` with signed/zero-rich inputs and probabilities. Later worker sweeps
are separate; do not delay this first comparison with a new scheduler.
For every configuration first complete all three numeric arms, then one
priming triplet, then three timed triplets with rotated order
`AllGpu/Serial/Concurrent`, `Concurrent/AllGpu/Serial`,
`Serial/Concurrent/AllGpu`. These are warm fixed-layer measurements. Repeated
experts, priming and references condition CPU/GPU caches; record that limitation.

## Real sparse upload and rank merge

Use the existing original GPU copy to the full32768B input-handoff owner and
read its actual bytes. Validate MAP/probability owning packets, finite inputs,
original member strides and inactive/padding zero before CPU execution.
Do not substitute regenerated CPU fixture inputs for the GPU-produced handoff.

The unchanged CPU batch returns the full30-rank unweighted307200B logical
output, with missing ranks positive zero. Actual selected masks/headers,
source views and joined completion authorize CPU readiness. After leaf scratch
and caller input leases release, encode into the existing full524288B CPU
upload owner using the approved bounded Vec<u8> copy. Its physical tail stays
zero. Include encoding, upload, validation and actual GPU completion in timing.

Add a separate bounded three-row merge kernel and a private rank-owner page.
It chooses exactly one accepted CPU or GPU partial for every original rank,
and performs `sum += probability[rank] * partial` in rank0..9 order. Do not
reduce CPU/GPU halves separately, compact/reorder probability multiplication,
introduce FMA or discard a rank because its probability is zero. Every active
partial is finite/ready, including zero-probability ranks. GPU status checks
require original GU/DOWN readiness only for GPU-owned ranks; HOST readiness
comes from the joined typed CPU output. Do not fabricate original GPU GU flags
or HOST activations. CPU masks, GroupMap and probabilities must agree exactly.

The all-GPU form of the new merge is an independent bitwise control against
the unchanged grouped_reduce before split timings. Runtime source identities,
counts, masks and original ranks never specialize a shader/pipeline. Only the
fixed public H/FF/three-row shape and original dtypes may do so. Any malformed
owner/ready/count/span or nonfinite value sets the shared sticky fault; it
does not select another kernel, source or arithmetic path.

## Owning budget before allocation

Reuse the existing4MiB standalone operator MAIN cap. The proposed complete
inventory is15 private owners with the following original exclusive-page
charges. Reference snapshots remain GPU-owned rather than surviving beside
CPU workers as additional HOST payloads.

| Owner | Logical words | Full backing bytes |
| --- | ---: | ---: |
| MAP | 804 | 32768 |
| INPUT | 20480 | 131072 |
| ACT | 51200 | 262144 |
| PART | 204800 | 1048576 |
| PROB | 80 | 32768 |
| OUTPUT | 20480 | 131072 |
| STATUS | 4800 | 32768 |
| CPU partial upload | 76800 | 524288 |
| Row inspection | 25600 | 131072 |
| Fault | 1 | 32768 |
| Input handoff | 7680 | 32768 |
| Reference ACT | 51200 | 262144 |
| Reference PART | 204800 | 1048576 |
| Reference OUTPUT | 20480 | 131072 |
| Private rank owners | bounded fixed header/masks, at most256 | 32768 |

Total backing is **3964928B**, leaving229376B below4194304B. This is a
source quote, not an allocation receipt. Authenticate exact original layout,
actual owning spans, physical roots/aliases and aggregate charged bytes before
private fill. No global MAIN cap, source/cache reservation, bank geometry,
driver512MiB margin or general allocator policy changes are permitted.

Keep the original1MiB phased private HOST cap. Unchanged batch payload capacity
is414720B (three inputs,30 activations and30 outputs);15 existing32KiB worker
reservations add491520B, before exact worker/job/header overhead. The original
32768B handoff and16384B caller-header allowances are also admitted before
fill. Use the leaf's checked byte API and actual capacities, not these payload
totals alone. Reject if the complete phase cannot fit; no worker fallback.
The encoding phase retains only the307200B CPU output,524288B encoding owner
and admitted headers. Release packet/input readback leases before that phase.
Perform post-run reference reads one full131072B row owner at a time, keeping
all simultaneous capacities within the same cap. Scoped controller/pool
headers and tasks are counted; OS/pool history erasure is not claimed.

The existing configured transient runtime remains capped at128MiB. Before
implementation acceptance, freeze a finite schedule manifest accounting for
each actual INFO record, upload owner, readback/inspection owner and sanitizer
phase. Each2048B INFO owner is included and no phase exceeds the unchanged512
dispatch ledger. Validate actual capacities before fill. Drop all readback
leases and audit/sanitize transient I/O between arms and after post-run owning
checks; never accumulate the old operator's180 pairs without checkpoints.
Source bindings/maps and the existing model are retained across these I/O
checkpoints; no context, epoch, poison or public source is reset.

## Numeric and timing acceptance

Before timing, compare all raw original-rank partials and merged outputs with
the same-artifact AllGpu control. Use the existing partial bound
`1e-4 + 3e-4 * max(abs(cpu), abs(gpu))`, and existing core/merged bound
`1e-3 + 3e-3 * max(abs(reference), abs(candidate))`; do not relax either.
CPU batch/count1 bit comparisons and the independent pinned-C scalar decoder
remain prior source gates. All-GPU new-merge versus grouped_reduce is bitwise.

For every arm inspect the complete owning input/MAP/probability, ACT/PART,
CPU upload, OUTPUT, status/fault and their inactive/gap/physical padding regions.
Resident GPU rows must reproduce reference GU/partial values. HOST CPU
partials must reproduce reference partials; HOST GPU ACT/status stay zero in
split arms and are not reported as GPU-computed. All unselected member/rank
cells stay positive zero; ready/owner fields have exact expected values.
Check malformed count/rank/source/span, missing readiness, p=0 with nonfinite
partial, sticky prior fault and inactive/padding mutation in focused tests.

The contiguous **completed candidate wall** begins before the arm's actual
input handoff/scheduling and ends after both consumers, real upload/merge,
full finite/fault completion, actual GPU join and its transient I/O cleanup.
No enqueue-only interval, max of isolated CPU/GPU durations or theoretical
overlap replaces that wall. Separate component intervals and host scheduling
intersection are diagnostic only, not GPU hardware timestamps.

Independent raw/reference comparisons run outside that candidate wall for
every arm, with identical requirements. Report their wall and the subsequent
checked owning inspection/I/O-cleanup wall separately, plus the complete
run wall including setup, references and terminal closure. The sum of separated
walls is not advertised as one contiguous request latency. No completed
timing receipt is published until all comparisons and cleanup succeed.

Final success requires full operator private-zero receipt, unchanged complete
target/source close, authenticated detach/zero/free receipts, final file/graph
consistency and no retirement/cancellation. Root independently verifies actual
Intel memory recovery and classified fault state. Receipt fields contain only
fixed case metadata, actual aggregate work/byte/audit counts and durations.
No inputs, partials, probabilities, expert IDs, source addresses or content are
written to disk/logs. Content-free failure stages do not publish partial timing.

## Implementation boundary and later gates

After review, proposed owned additions are a new fixed-public operator/merge
module, the opaque resident/all-partitions launcher in the existing borrow,
and the narrow executor bank helper. Root owns main registration and protected
runner policy. Proposed fixed mode is `public-host-cpu-overlap-probe` with
`KUNO_LAB_HOST_CPU_OVERLAP_PUBLIC_ONLY=1`, the existing fixed8GiB cache,
original source directory, CPU13 reference and public receipt. Reject inherited
conflicting/unknown Kuno laboratory flags before any device/model access.
No new runtime environment flag activates this in inference.

Source checks cover private owner arithmetic/aliases, actual source-scope
classification, same full original kernels, scoped joined scheduling on both
errors/panics, sparse/shared/disjoint original-rank math and exact capacity
phase ordering. Existing CPU batch source/math tests remain unchanged.
Build/Clippy and protected physical acceptance are separate. Do not compile or
use a GPU during root's benchmark holds.

Only a later separately approved model consumer could use these results. It
would require actual dynamic route export, whole-model source/owner budgets,
unchanged full CPU13 absolute0.006 + relative0.0003 logit/hidden checks, exact
greedy/expert identity, real speculation/repair and normal200-token timings.
This synthetic layer proposal grants none of that authority and predicts no
tokens/s gain. Four-row and adaptive placement remain separate candidates.
