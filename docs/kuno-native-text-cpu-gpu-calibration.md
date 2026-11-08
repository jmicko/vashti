# Original CPU/GPU expert calibration laboratory

Approved source-only implementation checkpoint, 2026-10-05. The implementation
worktree starts from root `d1a0446`, including both corrected packed CPU commits.
Root owns physical model/GPU tests and CLI registration; source checks cannot
establish device parity, overlap benefit or whole-model speed. Numeric acceptance
precedes explicit timing. The prepared-host API remains deferred.

The original plan below is preserved as a dated snapshot; its requests to await
approval/base refresh are satisfied by this checkpoint. Actual signature/lifetime
accounting must be frozen before source acceptance. Capability clarification:
the original 64-lane math must also admit the existing legacy copy kernel's
256-lane workgroup, and seven storage bindings including INFO. No default,
production worker, public API or installed behavior changes here.

Original proposal SHA256: `3d1ff3bc2796d1bf85841c079ec86c9cff92fa6f5c690dba27109bc31d4d1d06`.
See [the decode plan](kuno-native-text-decode.md) for the later hybrid-model gate.

## Implemented source checkpoint, paused 2026-10-05

Root `5f2ed09`, `b2b69cc`, `1ef151f` and `e386c79` integrate the read-only
owner inventory, bounded joint operator, borrowed caller graph and corrected
calibration ordering. Their author equivalents are `6cd9a246`, `c16359f5`,
`dc915f6b` and `db3a18f7`. Read [the paused checkpoint](kuno-native-text-checkpoint.md)
before resuming. No CLI or physical acceptance is established for this operator.

The actual typed entry borrows one caller-owned `GgufFile` and `BoundGraph`,
the configured client, device limits, the actual shared retirement `Arc`, explicit
configuration/purpose and cancellation. Its sealed numeric permit is bound to
the file/client/graph/configuration, actual retirement owner and parent thread;
at most 16 subsequent explicit calibration samples may share that permit.
One-card numeric success does not authorize timing before root both-card
acceptance or enable an automatic split policy.

The dated snapshot's **reference ladder before timed execution** is superseded
for Calibration: mandatory hit/filler priming remains first, then actual CPU/GPU
consumers, joined work, owning maps, bridge/merge and checked I/O cleanup complete
before independent references execute. Numeric retains references before
candidate work. Every numerical comparison, immutable input/probability/map
check, file consistency check and full capacity check still precedes publication
of the candidate's local slot plan. Numerical gates and original rank math are
unchanged. Reference time is excluded from operator counters; bridge/merge and
mandatory cleanup are included in the measured work.

Required priming, the numeric permit run and previous samples' post-run
references still condition caches. Rotating groups beyond the CPU LLC does not
establish storage-cold timing or whole-model cost. Prepared-hot CPU execution
remains deferred; the current operator reads the original file on each call.

Actual source accounting freezes 18 private pages/884,736 bytes, two public
16-slot banks/134,217,728 bytes, original legacy source precharge up to
41,943,040 bytes and an independent 64-MiB runtime hard cap within a 256-MiB
physical envelope. Controlled private application storage is 4 MiB (1 MiB CPU,
2 MiB references/snapshots, 1 MiB bridge/metadata). The bridge inventory now
includes a separate 4,096-byte scoped-control/header reservation, for 870,400
bytes. No duplicate model binding, converted weights or prepared payload cache
is introduced.

## Numeric-only failure classification (2026-10-06)

The fixed-public producer now replaces a failed `Purpose::Numeric` result with
one constant stage label. Stages distinguish configuration, device limits,
control/worker proof, geometry/original banks, startup I/O/capacity/host budget,
allocation/physical owners, fixture upload/priming/references, candidate planning
and joined work, CPU partial checks, merge/numerical acceptance, final audit,
file consistency, metadata/slot commit, sentinels and erasure. For example,
`public CPU/GPU numeric startup-capacity rejected` identifies that boundary
without recording a memory value or the original backend error.

The stage labels add no log output, raw error formatting, tensor values, rank
trace or partial receipt. Calibration retains its existing errors and performs
no diagnostic logging I/O. Successful reports, numerical tolerances, allocation
and I/O caps, original math and ownership cleanup remain unchanged. Root still
owns compilation and protected physical reruns; the labels locate a failed gate
and do not establish numerical or performance acceptance.

The physical-owner stage is further split into client/metadata, private and
reference owner shape, public-bank owner shape, resource lookup, private and
reference physical extents, public-bank offset/size/backing, and alias checks.
These remain constant labels without offsets, addresses or payload dumps.

## Fixed-public full-page bank allocation follow-up (2026-10-06)

Source review found that the original `ExpertBanks::new` requests logical bank
bytes from an exclusive pool whose backing allocation can be smaller than the
conservative power-of-two page charge. Such an allocation does not satisfy this
operator's existing exact physical-page invariant. The focused operator now
uses `new_full_pages_for_cpu_gpu_lab`: original Q4 gate/up and Q5/Q8 down metadata,
exactly 16 slots, and only the original 14,745,600/18,022,400/27,852,800-byte logical
component lengths. It requests the complete 16/32-MiB page and retains that owner
through an offset-zero handle whose end is clipped to the original logical span.
Original bank capacities, packed launch lengths, read/fill paths and math are
unchanged. The ordinary `ExpertBanks::new` constructor keeps its original
allocation behavior.

Existing full-page precharges already account for two 64-MiB bank triples. The
physical extent/backing/nonaliasing checks and 256-MiB envelope are unchanged.
Host test sources exercise the actual allocation/view helper for both down
formats and reject foreign extents or malformed full owners before use. The
stage tests continue to reject a logical-sized backing. Formatting/whitespace
and source review are separate from root-owned compilation, protected device
acceptance and any timing result.

## Fixed-public calibration helper source, 2026-10-06

`cpu_gpu_expert_probe::run_fixed_calibration(client, file, graph, limits,
retired_arc, cancel)` is a separate explicit laboratory entry. It borrows the
already inspected original file and caller graph, the existing configured
client/limits and actual shared retirement owner. Main registration, protected
physical execution and both-card numerical acceptance remain root-owned. This
source handoff must be integrated with root's complete physical-bank allocation
fix (`1946020`); it does not replace that invariant or enable a production path.

The compiled subset is Q5_0 and Q8_0 down, original group 0/vector 0, AVX2 with
16 proven physical-core workers, ten misses and CPU miss counts 0/5/10. Each of
the six points first completes separate Concurrent and CPU-first numeric runs
for those exact configurations. It then takes four samples in
Concurrent/CPU-first/CPU-first/Concurrent order: **12 numeric gates and 24 timing
samples total**, two timing tickets per original permit. Each point retains its
two original sealed NumericPermits until all matching samples finish; returned
calibration permits are dropped. The existing exact file/client/graph/config,
retirement Arc/thread identity and shared 16-ticket ceiling are unchanged. No
caller-supplied group, worker, route, split, mode or sample count is accepted.

The helper returns fixed-size typed point/sample arrays containing the existing
aggregate Report, explicit mode and three separate wall intervals. Existing
numeric-only Report fields and the old four numeric CLI cases stay unchanged:

- `Report.timing` contains completed CPU, GPU and joined scheduling intervals;
  `Report.bridge_and_merge` measures the completed CPU upload/original-rank
  merge, finite/fault owning maps and checked I/O.
- `candidate_completed_wall` spans the start of the joined schedule through
  CPU partial validation and completed merge. It excludes independent oracles.
- `acceptance_cleanup_wall` begins after reference computation and numerical
  comparisons, before immutable owner maps, then covers final fences/file
  checks, local slot publication, public sentinels, full MAIN erasure and final
  checked I/O. It also excludes independent oracles.
- `full_run_including_references_wall` spans the entire typed run, including
  setup, priming, original reads, references/oracles, numerical acceptance,
  cleanup, final file verification and local owner destruction.

Candidate and cleanup are separate intervals with references between them;
their sum is not a contiguous full-run wall. The helper preserves every launch,
owning map, fence, comparison, reference ordering, source/main/runtime/private
host charge and final cleanup gate. Timing observations are stack/public
metadata; bounded aggregate reports are under 32 KiB and contain no private
vectors or IDs. No report is returned until all six points complete cleanup and
the original graph/file/retirement owner pass final consistency checks. The
helper performs no logging, file publication or automatic split selection.

Original numeric gates, required priming and earlier post-run references warm
OS/host/device caches. Fixed group 0 is a narrow warmed original-read operator
experiment, not the 32-group working-set study, prepared-host performance,
storage-cold timing or full-model decode speed. Parent must retain compiler-idle
protected execution and numerical/lifecycle evidence before interpreting
results. Host test sources exercise the same fixed-point driver for gate
failures, exact permit reuse, ABBA/sample bounds and publication intervals. No
compilation, test execution or device timing accompanies this source handoff.

## Approved original snapshot

### Protected fixed calibration registration

The owner laboratory registers `resident-cpu-gpu-expert-calibration` separately
from the unchanged four-case numeric mode. Only the card and compiled mode may
be passed; a receipt destination and transient I/O plus full owning audits are
required before device/model access. The protected launcher additionally
verifies actual8-GiB, swap0/core0 containment. It calls the fixed helper on the
same configured client, borrowed original blob and shared retirement owner.
The prior both-card Q5/Q8 complete-expert numeric acceptance is a prerequisite;
each new exact configuration also completes its own numeric gates before timing.

The receipt uses scope `fixed-public-original-cpu-gpu-expert-calibration`, six
points, 12 numeric gates and 24 samples. It records the fixed group0/vector0,
AVX2/16 workers, ten original ranks, original-read warmed-cache scope, existing
numeric/audit fields, checked integer nanosecond timings, and explicit
candidate/cleanup/full-with-references walls. Original permits, selections,
input vectors and model payloads are never serialized. Unexpected timing/mode
values reject publication. Completion requires final original-file, cancel and
retirement checks after all helper cleanup; no full-model or automatic placement
claim follows. A destination write failure can leave an incomplete public lab
file, but fails without the completion marker and cannot pass the run validator.

The registration's source tests and physical execution are separate acceptance
steps; see the current owner checkpoint for exact completed results.

Bounded original CPU/GPU expert acceptance and joint calibration
Source-only proposal, 2026-10-05; implementation awaits approval.

Base and scope

Use the clean managed worktree /home/jmicko/.codex/worktrees/kuno-text-ops/vashti,
currently371b9d19d7faae393262c13c1ec1fe066b604d33. The CPU prerequisite is
55dfe3434652c5e4d730660dc67623fc8d0a0833 plus metadata repair
d8c08873bc05f561f775da9baa6dd9ca96ed07fc from the text-loader worktree.
CPU source SHA256 a4c58eacd22a9ca08a52b361fb6016428b773dd1b7396f11fcff731db2ebd9bb.
Refresh the implementation branch to the parent's agreed CPU integration commit
after approval. Preserve the existing coordinator and all experimental defaults.

This adds a separate fixed-public one-layer operator laboratory, not a native
hybrid decode connector. It neither reads real routes/prompts nor executes a
whole model. Parent selects the actual client, configures owned I/O exactly once,
supplies the already-open original GGUF, and performs protected physical runs.
No source, main registration, GPU, service or model payload action accompanies
this plan. A later caller registration belongs to an explicitly approved seam.

1. Concrete operator and references

Proposed new lab modules: cpu_gpu_expert_probe.rs (owning acceptance runner) and
cpu_gpu_expert_schedule.rs (bounded split, start/join/fence driver and receipts).
One narrow Rust/CubeCL rank_merge kernel can be a child of quant_matvec. All
payloads retain the existing original bytes and F32 activations. No activation
quantization, FMA introduction, probability factoring, expert pruning or altered
64-lane reduction schedule is permitted.

The runner receives the parent's existing client, GgufFile/BoundGraph, actual
limits, and an explicit public ProbeConfig. Config selects Acceptance or later
Calibration, Portable or Avx2, exactly1/4/8/16 physical-core workers, missesM in
0..10, CPUmissesK in0..M, and SerializedCpuFirst/SerializedGpuFirst/Concurrent.
There is no automatic ISA/device fallback. WorkerPlan::physical must prove the
requested count of allowed distinct physical cores; unsupported plans reject
before private allocation. WorkerPlan changes scoped workers only, not parent
affinity. No timing mode is selected before both-card numeric acceptance.

Select actual original layer triples from metadata, one per down format. The
stored metadata first contains Q8 at layer0 and Q5 at layer6; execution derives
and verifies these from the opened model rather than trusting those numbers.
Required gate/up shapes [2560,640,512], both Q4_K, and down [640,2560,512], Q5_0
or Q8_0. Per expert bytes are gate/up921600 each, down1126400/1740800, complete
2969600/3584000. A complete expert performs4915200 weight/input products plus
640 SiLU operations. Use PositionalReader pinned to the original opened model;
never reopen a model path, hash/stream the full file, convert a complete expert
to persisted F32, or load the projector.

Reference ladder, before timed execution:
- Independent bounded row decoder: TensorType::decode one original row, scalar
  partial sums lanes0..63 visiting lane+64*tile, tree32/16/8/4/2/1, original
  sign-stable SiLU and full down. Reuse one guarded decoded row; no whole expert
  F32 expansion. Return all ten original-rank activations/partials.
- Existing Matrix::apply is a separate left-to-right mathematical oracle. It is
  compared within the existing core tolerance, not treated as bitwise64-tree.
- Stage1 PackedProjection gate/up/down and ExpertOperator complete experts:
  Portable versus Avx2 must preserve bits for the independent64-tree CPU values
  on actual original slices. Project supplies raw-gate/up acceptance without a
  mutable intermediate accessor into the production CPU operator.
- Existing all-ten ResidentMoe with its original legacy public fills is the
  unchanged GPU operator reference. Inspect its original activation and
  unweighted partial owners, output and every completion status using complete
  owning maps. Reference state/bank publication is local to this fixed operator
  fixture; no ResidentModel token/state authority is invoked or implied.

Keep existing bank-probe tolerances: GPU/tree stages atol1e-4+rtol3e-4; original
core output atol1e-3+rtol3e-3. Compare CPU bit schedules independently. Do not
relax tolerances to accommodate CPU/GPU exp, contraction or denormal differences;
record the failed stage for this fixed-public fixture and investigate it.

2. Actual split, complete rank math and real upload path

Each sample supplies exactly ten distinct public fixture IDs and the same ten
fixed normalized probability bits, including a second case with +0/-0 ranks.
Input is the existing deterministic2560-wide bank fixture, plus a fixed signed
and zero-rich variant. No original selected rank can disappear because p==0.

For declaredM misses, prime only the ten-minus-M selected hits outside timing,
through actual public fills, completed consumers and local accepted bank plans.
Verify actual committed BankSlots produce exactly those hits/M misses before
the sample. Assign K of the declared miss ranks to CPU in original rank order.
Protect all GPU hits and chosen GPUmisses before selecting any victim. Only
F=M-K misses enter the actual legacy fill plan; CPU ranks stay absent from the
candidate bank. Fresh/reprimed identical state is required between paired runs.

CPU executes the K IDs using PackedExperts/ExpertOperator::execute; its compact
results are unweighted and in requested order, with full ten-rank logical zero
tail. GPU executes G=10-K selected GPU ranks in original relative order using
the unchanged moe_block gate/up and moe_gpu down kernels, runtime selectedG,
original slot bytes,64-lane tree and sign-stable SiLU. Grid envelopes stay at
the full public ten-rank geometry, with the existing runtime selected guard.
Candidate work has distinct
ACT/PART owners. For G==0 skip GPU expert dispatch explicitly; for K==0 skip the
CPU call explicitly. No zero-grid launch or fake zero-weight CPU-owned slot.

The timing path calls actual legacy expert_banks::fill for each original
gate/up/down miss. Aggregate precharge is the same whole-miss source-page SUM,
followed by the existing postfill allocator guard before consumer enqueue.
Do not replace it with ordered upload, scatter batching, artificial memcpy,
new per-component telemetry fences, or an enqueue-only timing substitute.

A separate full CPU partial GPU owner and runtime owner/index table preserve
rank ownership. CPU joins and validates its guard before a moved owned-private
upload; it never writes a buffer concurrently used by GPU kernels. The final
rank_merge reads exactly one accepted unweighted owner for every rank and does
sum += probability[rank]*partial in rank0..9 order, unchanged from moe_reduce.
No sum of separately reduced CPU and GPU outputs. All counts, slot/index/rank
ownership and offsets are runtime private data; only public dtype/geometry may
specialize a shader. Finite/ready checks cover every active rank, even p==0.

3. Owning budgets and proofs

Standalone fixture retains two16-slot actual original bank triples, candidate
and unchanged GPU reference, processed one down format at a time. Both formats
round to64MiB per bank (16MiB gate+16MiB up+32MiB down), total134217728B.
Do not allocate48 banks or the native model private785186816B for this fixture.

Planned PRIVATE full pages, minimum32KiB/power-of-two:
- Shared input/probabilities/merged-output/fault/owner-index:5x32768=163840B.
- CPU compact-rank partial upload:131072B.
- Candidate GPU slots/ACT/PART/status:32768+32768+131072+32768=229376B.
- Existing all-ten reference ResidentMoe five pages:262144B.
- Reference output:32768B.
- Raw original gate/up comparator outputs:2x32768=65536B.
Total18 full pages884736B, within a1MiB private arena cap. Raw-gate/up comparators
reuse full base-zero bindings with runtime output_first for bounded prefix/gap
tests; never pass unaligned token/expert storage subviews. All full owners must
be distinct where either writes, on the same actual client/stream/arena epoch.
Require64-lane workgroups, grid dimension>=25600, >=6 storage bindings plus
INFO, INFO<=512B,128MiB public storage/array support and alignment<=512B.
Public matrix views retain original bounded launch views while full physical
lookups prove their complete public bank owners. Require physical offset0,
full resource size, validated packing/capacity/slot bytes and no writable alias.

Each outstanding missing expert has rounded original source pages1+1+2MiB;
F<=10 gives41943040B MAIN source precharge. No credit before a completed consumer
fence, dropped map leases, checked sanitizer and actual telemetry. Initial
physical aggregate uses a256MiB ceiling and separately reserves the configured
64MiB runtime cap: two banks134217728 +private884736 +sources41943040
runtime67108864 +actual existing baseline must fit268435456B. Thus baseline
must be <=24281088B before allocations; otherwise reject. MAIN ledger tracks
actual MAIN only, independent of runtime ownership accounting. No cap increase.

Incremental operator runtime plan reserves8MiB within the unchanged configured
64MiB hardcap. Initial conservative inventory per clean sample is private full
upload sources262144B, complete owning reference/candidate maps <=884736B,
two fault reads65536B, at most128 INFO records x2048B audited262144B, complete
MAIN zero-inspection spans884736B and two8KiB slab tails16384B =2375680B.
Double the complete inventory for simultaneous source/target inspection
headroom:4751360B <8MiB. Freeze an actual signature/dispatch inventory before
source acceptance; priming/reference/compilation work must have completed
cleanup before this sample envelope starts. Do not silently retain old sample
IO or rely on slab slack. Each map is full physical length, leases drop inside
the completed callback before sanitizer; maps are inspected one at a time.

Application private-host cap4MiB: Stage1-controlled CPU operator1MiB includes
all16x32KiB worker reservations; guarded oracle/snapshots2MiB; moved bridge,
owner/rank/job metadata and counters1MiB. Inspect actual capacities before fill,
including original committed bank states and staged plans; use the Stage1
sealed retained header owner and its exact-capacity tests. Public CPU payload
retains its own40MiB cap. Combined fixture PUBLIC application-host cap48MiB
includes that plus live legacy component/oracle scratch; no simultaneous whole
second CPU job copy. Runtime-owned host remains independently capped64MiB.
These are API-visible owned allocations, not claims about OS allocator bins,
driver caches, machine stacks or compiler/register erasure.

4. Completed measurement and the current hot-payload API gap

The first supported calibration is actual read-path work. CPUexecute includes
positional GGUF reads; GPU uses original cursor-guarded legacy read/upload/copy.
Warmed OS file pages can be conditioned outside timing and labeled honestly as
OS-cache-conditioned reads. Never call them preloaded host execution or infer
SSD-cold timing without an independently controlled run. No drop_caches or
shared-machine memory manipulation is requested.

Use one guarded cancel-aware start gate for the CPU orchestrator and GPU parent.
Partial spawn/panic/cancel aborts the gate and joins every worker; ordinary
Barrier cannot be left waiting after failed spawn. Only the parent thread uses
the GPU client. Concurrent: release both, CPU executes K jobs while parent fills
F actual misses and enqueues G actual consumers, completes owning GPU fault /
status fence and cleanup, joins CPU, validates CPU guard, uploads CPU partials,
queues exact rank merge, and completes second fault/output cleanup. Serialized
CPU-first and GPU-first perform identical work and ownership checks. Both joined
end time and completed GPU observation are recorded, followed by full numerical
maps/validation outside the operator timing and explicit final erasure before
any receipt. Include bridge/merge/mandatory cleanup in end-to-end time; report
it separately from CPU/read and GPU/enqueue nested counters.

Current Stage1execute always reads GGUF and does not accept borrowed prepared
payloads. Therefore a same-bytes-hot host CPU control needs a separately approved
narrow immutable PreparedExperts job owner plus execute_prepared reusing the
same compute functions and guarded headers. A corresponding public fixture
fill facade must reuse the original create_from_slice+copy_words path while
excluding only the model read. Prepared data must be original descriptor-bound
bytes, selected original order, <=40MiB total and scoped to this one calibration
case. It cannot become a decoded/request cache. This is a follow-up dependency,
not an existing API and not necessary for the first real-read acceptance fixture.

Timing protocol after numeric acceptance: rotate32 public ten-expert groups
(320 distinct original experts per layer) without retaining them together;
working set~906MiB/Q5 or~1.07GiB/Q8 greatly exceeds a typical CPU LLC. Record
actual public hardware/cache geometry instead of assuming a cache bandwidth.
Prewarm shader compilation, fixture setup and public worker configuration outside
samples; actual allocations, worker creation/pinning, reads, guards/fences that
the operator would perform remain inside the appropriate measured path.
Search discreteM0..10/K0..M (66 split points), worker1/4/8/16, both actual dtypes;
one explicit bounded subset per parent invocation, not an unbounded search loop.
At most16 recorded matched samples per point. Parent runs compiler-idle ABBA
Concurrent versus Serialized with identical work, plus isolated CPU and GPU
controls; all timings retain end completions. A joint improvement is evidence
for this operator/resource overlap only, not whole-model decode speed or an
automatic split rule. Existing small_cpu projector and nested enqueue timers
are not calibration baselines.

5. Focused acceptance and host/offline test seams

Parent first runs both cards for Q4/Q5 and Q4/Q8 complete actual experts, portable
one-worker independent CPU reference, Avx2 workers1/4/8/16, and masks allGPU,
oneCPUmiss, alternating-fiveCPUmiss and allCPU. CPU/GPU ownership must cover all
ten ranks exactly once. Include every rank position via rotating masks and
zero-probability CPU and GPU ranks. Check raw gate/up, activated rows, every
unweighted partial, final rank-ordered output, status/fault, exact full lengths,
unused/page tails, immutable input, public weight sentinels and owned zero audit.

Host tests exercise the production split/start/join/completed-proof driver with
injected callbacks, not a disconnected policy model: M/K endpoints, missing /
duplicate ranks, wrong count/artifact/dtype/slot, exact capacities/overflow,
spawn-before-gate error, CPU panic/cancel during actual joined work, GPU enqueue /
fault/map/audit error while CPU runs, join-before-private-release, nonfinite CPU
result, and failure never publishes a BankSlots plan or returns partial output.
Reuse Stage1 reader short/error/change and full metadata wipe/join witnesses.
Do not expose its private test hook as a production fault-injection API.

Checked/unchecked offline WGSL tests cover the new rank merge, runtime ownership
branches and all rank reads. Storage loads must be nested within validated
active/index branches because Cube DSL can eagerly lower AND operands. Use
explicit mutable success flags/stores, preserving the prior compiler workaround.
No new performance arithmetic, word-down, subgroup tree or hidden fast math.

GPU late failure, cancellation after fill and before final merge, abandoned plan,
stale epoch/bank identity, partial-copy victim corruption and poisoned retry must
reject the whole operator owner. Explicit abort joins CPU outside admission
locks, then completes drain/erase/audit; failed cleanup retires/quarantines and
cannot produce a clean receipt. Drop sets sticky poison only; actual bank
metadata and private buffers are cleared by their owning cleanup. No rollback,
detached CPU job, partial result publication or CPU fallback on GPU failure.

Receipts contain only fixed-public scope, artifact/binary/device/driver/cpuset
identity, dtype/geometry, explicit worker/M/K/mode/sample settings, completed
wall/phase timings, logical and physical charges, numeric maximum errors and
counts-only erasure/retirement proofs. Never retain real selected-ID histograms,
probability/router traces, user content or private tensors. Root owns main's
approved caller/receipt registration, physical acceptance and any later timing
selection; no public API/catalog/service/default changes are part of this step.

Source basis (read-only)
- docs/kuno-native-text-decode.md at managed base371b9d1.
- text-loader cpu_expert.rs at55dfe+d8c: PackedExperts/PackedProjection,
  ExpertOperator::execute/project, guarded Partials, physical WorkerPlan and
  bounded retained job/scoped-handle ownership.
- text-loader gguf_positional.rs: original descriptor ownership, bounded pread,
  unchanged-file and short/error handling; no shared seek-cursor mutation.
- native lab moe_gpu.rs/moe_block.rs: exact lane visit/tree, sign-stable SiLU,
  unweighted rank partials and original rank0..9 reduction.
- native lab expert_banks.rs::fill_source: current ordinary public read/upload /
  copy path and original expert bytes/full-page charges.
- native lab resident_moe.rs and bank96_probe.rs: actual original all-ten
  reference, status/owning map/capacity/cleanup precedents and numeric tolerances.
- Existing metadata-only show snapshot: layer0 Q8/layer6 Q5 actual routed down;
  implementation revalidates through the opened original artifact.

Approval requested: implement the separate real-read operator acceptance /
joint-calibration modules and optional rank-merge kernel in the managed worktree,
with no main/private arena/core/default edits except a separately agreed caller
seam. Defer prepared hot-payload API changes until independent approval. Parent
keeps all GPU/model-payload/service execution; source checks alone establish no
physical numerical acceptance, performance advantage or full-model hybrid seam.
