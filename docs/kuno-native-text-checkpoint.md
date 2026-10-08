# Native text checkpoint, 2026-10-08

The user-requested private startup-upload extension and pinned llama.cpp review
are complete. Selected source is **3b701d4d9bbeb794351502022b21c30734d501eb**;
the complete fixed-public fixture measures **9.5943 decode tokens/s**, with all
original output, ownership and lifecycle gates passing. This falls within the
preceding timing range and does not establish an additional speed improvement.
The approved10-token/s reference remains unmet; no new Ollama baseline was run.

Zero-tail uploads now cover private constructor/prefill transfers outside
request reuse. Measured prefill private-copy bytes fall about75%; model-bind
copy bytes are unchanged. Public packed weight uploads are unchanged. No
startup time gain or hardware PCIe-traffic reduction is claimed. See **Private
startup uploads accepted** at the end for exact measurements and restoration.
Earlier pending statements and selected versions are historical.

Normal inference is restored and all lab jobs exited. The goal remains paused;
no more physical work is scheduled. Installed binaries/settings, public API,
Vashti and release artifacts are unchanged. The highest-priority new decode
candidate is measured, bounded binding-object reuse; it is not yet implemented.

Backend owner:
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/kuno-native-text`; preserve the canonical Vashti checkout and its work.

## Active overnight offload run, 2026-10-08

The current accepted best is7.1654 tokens/s (144 slots, original full fixture);
10 tokens/s remains unmet. The preceding block established3.5006 tokens/s. Source
`4cf40f3` adds specialized F16C original CPU packed loops with120 core tests,
3 ignored and strict Clippy. Its normal frozen direct-device artifact
`e12a2b2788eabcb2f5d4c6b0588ba7cb18bd531bba4ae8c61ca676fb650a0179`
passes the complete B70/intel96 fixed75/128 model and all original lifecycle/
owning gates: all128 IDs,127 steps in37118ms (3.4215 tokens/s),61491ms prefill.
It establishes correctness but no whole-model speed improvement. Results are
under `kuno-development/next-offload-device-20261008/f16c96-long-v1.*`.

The next candidate expands explicit protected CPU caps to canonical0..10 so
misses beyond the cap can enter the GPU cache during decode. Cap10, which sends
all misses to CPU, leaves its prefilling bank stationary. No default, model
geometry, rank arithmetic, private/source budget or publication/erasure gate
changes. See the
[cache-admission contract](kuno-native-text-count1-hybrid-plan.md#bounded-dynamic-cache-admission-experiment-2026-10-08).
482 direct native tests plus two offline shader suites and23 Python controls
pass. Mixed-cap physical correctness and timing now pass (source5ed5c8c,
protected-launch correction3f72acb, artifact8733aaec…): cap5/intel96 completes
all128 IDs with37595ms decode (3.3781 tokens/s),50682ms prefill,38728 warm
GPU hits and1703 warm GPU expert uploads (5632921600 bytes), and20529 CPU
misses. Warm weight reads add3923ms. It is not a speed win over cap10.

Source d9d5193 ports the GPU count1 expert projections to packed-vector/
subgroup math on original banks and owning spans. Source6253828 batches the
seven original MoE scratch clears via ordered native buffer clears; the same
completed fences, transient-I/O errors, and whole-owning zero audits remain.
Both full75/128 B70/intel96 runs pass all128 IDs plus fresh/poison/cancellation/
full-owning gates. Vector-only decode36821ms (3.4491 tokens/s), prefill48214ms;
plus clears decode37318ms (3.4032 tokens/s), prefill52038ms. These single ordered
samples establish correctness, not a speed improvement. Frozen artifacts are
d99bcffb… and4ffe5a7b… under next-offload-build-20261008-v3/v4; receipts live in
the same next-offload-device directory.484 native tests plus two shader suites
and strict Clippy pass. The updated runtime patch SHA is
0f4137174f5de0e53caf1e30fc517c6e9822ca1861ca628080a78045e707b0d9.
Physical Intel memory returns50688000 bytes after each completed fixture.

The next CPU candidate is a separate compiled `cpu-fma` feature, explicitly
selected in the protected runner. It keeps original Q4_K/Q5_0/Q8_0 weights,
finite/cancel/ownership guards, and call-local wiping scratch, while using
FMA and32 accumulated lanes. Its reduction round points differ; ordinary
Avx2/Portable and installed inference remain unchanged. Independent f64 row
checks use an L1 forward-error bound; complete synthetic expert checks use an
infinity-norm error bound because cancelled elements can approach zero.
The original128 model IDs remain a mandatory, unchanged acceptance gate.
122 core tests pass with3 ignored. Source56340c2, frozen artifact4fa9fbfd…
passes all128 IDs and original owning/lifecycle gates, but decode37323ms
(3.4027 tokens/s), prefill54966ms does not improve the selected control.

Source98df150 adds numeric-only warm decode process counters. Artifact
a2724725… passes the same complete fixture with36790ms decode (3.4520 tokens/s)
and53251ms prefill. During warm decode it incurs61267 major faults and reads
11601731584 bytes from disk, despite zero warm GPU expert uploads. CPU cycle
profiles alone did not expose this stall. Counters report process CPU ticks,
not inferred GPU times; their scope starts after prefill and before decode.

The next explicitly pinned cap10/mapped experiment advises complete original
file pages for weights already retained on GPU at a proved completed boundary.
It uses Linux64 POSIX_FADV_DONTNEED on the original read-only descriptor,
leaves all mappings and borrowed tensor bytes valid, and never drops arbitrary
host caches. Successfully advised bytes are not a claim of actual reclamation.
Advice cost is separately reported outside decode time. All128 original IDs,
full private budgets, fences, erasure and physical recovery remain required.
123 core tests (3 ignored),485 native tests plus two shader suites,25 Python
controls and strict all-target Clippy pass. Source6e74f91 artifact861c20cc… passes the full75/128 fixture with37142ms
decode (3.4193 tokens/s),50760ms prefill. Advice covers18080460800 source bytes
in549ms, but warm decode still reads11718238208 bytes and incurs60783 major
faults: no effective reduction or speed win. Recovery returns50688000B.

The next candidate is explicit `intel128`: two original64-slot shards, exact
24GiB bank charge,5GiB static cache and31GiB reserved ceiling. Existing profiles
retain their26/10GiB ceiling. The original source metadata, all-ten-rank merge,
private inventories and owning proof gates remain. Source83002c9 first rejects at the original512KiB host ledger guard before
allocation. Correction6925024 accounts640KiB separately within the unchanged
12MiB parent envelope. Artifact4f777e06… passes all128 IDs and original owning/
lifecycle gates:35075ms decode (3.6208 tokens/s),49375ms prefill,19238 CPU
misses and41722 GPU hits. Warm decode still reads11150995456 bytes and incurs
63966 major faults. The modest single-sample gain does not meet10 tokens/s or
establish significance. B70 recovers50688000B.486 native tests plus two shader
suites,25 Python controls and strict all-target Clippy pass.

The next explicit candidate uses original full-owner native clears across
GDN/QSA/GR and coordinator scratch, batching adjacent clears within16 owners/
64MiB. Larger public spans keep the original shader path. Final arena erasure,
parent fences and every original full-owning audit remain unchanged. Host gates
pass486 native tests plus two shader suites,26 Python controls and strictClippy;
Physical source d8c1993/artifact74a696a7… passes all original output and
owning/lifecycle gates,35849ms decode (3.5427 tokens/s),50405ms prefill.40767
actual native clear calls are reported. Nested host timings shift among enqueue
and wait phases but total time does not improve. No selected speed win.

The next candidate combines register integer reductions and vector Q5 unpacking
with explicit three-pass residual Q8 CPU activations on original weights. This
is approximate and separately pinned; the original128-ID gate is unchanged.
Actual private metadata is charged inside the original1MiB CPU envelope.124
core tests (3 ignored),486 native tests plus two shader suites,27 Python controls
and strict all-target Clippy pass. Source662a4bf/artifactf48737d4 passes all128
IDs and original owning/fresh/lifecycle gates:36182ms decode (3.5100 tokens/s),
51308ms prefill. Warm reads11150274560 bytes and64084 major faults remain.

Selected CPU source read-ahead (a045454, artifact68052dcf) passes the same
full gates with34944ms decode (3.6344 tokens/s),57714 advice calls for19238
CPU misses. Warm reads10837012480 bytes and58098 major faults remain. This
single sample is only a small change, far below10 tokens/s.126 core tests
(3 ignored),486 native tests plus two shader suites,28 Python controls and
strict all-target Clippy pass. Physical memory returns50688000 bytes.

The next explicit `--pageout-gpu-sources` candidate also asks Linux to reclaim
mapped immutable original model pages after their GPU copies are ready. Only
complete validated GPU source pages are addressed at the original completed
cap10 boundary. Linux COLD/PAGEOUT preserve read-only file contents and borrowed
slices; anonymous/private inference memory is never addressed. This remains
best-effort advice, with no actual reclaimed-byte credit. Mandatory original
owning/output gates remain. Physical source09dcdd9/artifact7a139c08 passes all original gates with35169ms
decode (3.6111 tokens/s),51648ms prefill; readbytes10623344640 remain. The same
frozen binary under the protected root cgroup passes all gates with34984ms
(3.6302 tokens/s),51385ms prefill,10556477440 readbytes. Original blob belongs
to UID997; Linux can silently filter nonowner file PAGEOUT, but temporary root
execution also produces no useful improvement. No permission/model/service
change. Physical B70 recovers50688000 bytes.

Source2bd4f70/artifactcf2a9eee expands to144 original experts in two72-slot
shards within the same24GiB bank/5GiB cache/31GiB device ceiling. All128 IDs,
full original owning and fresh/lifecycle gates pass with34343ms decode
(3.6980 tokens/s),50668ms prefill,16411 CPU misses. Readbytes10352029696 and
58553 major faults remain.487 native tests plus two shader suites,29 Python
controls and strict all-target Clippy pass. This remains below10 tokens/s.

Source de6d086 adds explicit4/8/16 physical worker calibration and separate
compiled two-pass residual Q8.127 core tests (3 ignored),487 native tests plus
both shader suites and30 Python controls pass. The first8-worker launch
(artifact00215ff3) rejects at the original fixed16-worker constructor guard,
before model inference; no accepted speed receipt. Recovery returns50688000B.
The correction retains a typed4/8/16 maximum, requires pooled4/8 and independently
observes actual constructed geometry. The corrected source5295faf/artifactcbe79139 passes all128 IDs and original
owning/fresh/lifecycle gates with8 actual workers:35199ms decode
(3.6081 tokens/s),51175ms prefill. User/system CPU ticks drop to3237/751,
but readbytes10334400512 remain and wall speed does not improve. Sixteen workers
remain the control. Two-pass physical acceptance is pending.

Source ff8d92b adds the original160-slot/72+72+16 profile within27GiB bank,
3GiB static cache and the unchanged31GiB ceiling.490 native tests and both
shader suites,30 Python controls and strict Clippy pass. Frozen artifact
006771c6… rejects its first prefill at a physical owning check before any
accepted timing; no receipt or speed claim. GPU memory recovers50688000B.
Inspection found the actual resident MoE launch proof still assumed two shards.
The correction derives and independently validates all three actual projection
inventories from the checked public table, without weakening full-page/alias/
geometry checks. A host regression rejects missing/extra shards in each
projection.491 native tests plus both shader suites pass. The fixed-public
terminal cleanup now preserves the original content-free typed error instead
of replacing every failure with an opaque chunk rejection; panic payloads
remain suppressed. Corrected sourcec3ff595/artifact8dfbc305 passes all128 IDs and original fresh,
full-owning and cancellation gates, but decode234567ms (0.5414 tokens/s),
prefill61257ms. Three-GiB dense cache thrashes: warm weight reuploads386165831680B,
read59030ms/upload95104ms. It is excluded from speed selection; the accepted
144-slot/5GiB control remains selected. Physical B70 returns50688000B.

A further internal CPU overlap candidate schedules its coordination task in
Rayon's existing pinned pool with `in_place_scope`, retaining the GPU parent
on its original caller thread instead of creating an OS thread every layer.
Original32KiB orchestration /1MiB CPU reservations and full-capacity wiping
metadata/Partials ownership remain. The existing genuine overlap test proves
parent-thread identity and joined completion through all original failures.
127 core tests (3 ignored),491 native tests plus both shader suites and strict
all-target Clippy pass with compiled two-pass arithmetic. Source17101d1/artifact0c908311 passes the full original75/128 model and owning/
fresh/lifecycle gates with two actual residual passes and16 workers:34888ms
decode (3.6402 tokens/s),59177ms prefill. Readbytes10927468544 and62447 major
faults remain; no useful speed win. B70 returns50688000B.
Original144-slot GPU profiling from frozen c3ff595/FMA artifact rejects during
prefill because profiling exhausts the original private-I/O budget; no timing
or acceptance claim, and no budget bypass. Physical memory returns50688000B.
A further buffered original CPU job candidate keeps mapped GPU sources, but
uses bounded call-local original packed vectors for actual CPU positional reads.
127 core tests (3 ignored),491 native tests plus both shader suites,31 Python
controls and strict all-target Clippy pass; physical acceptance now passes: source14a1c92/artifact1864cd13 completes all128 IDs,
original owning/fresh/lifecycle gates with51413ms decode (2.4702 tokens/s),
50962ms prefill and49233 actual original CPU reads. Major faults drop to361,
but readbytes12011974656 and copy/allocation costs worsen wall time. Mapped
reads remain selected. B70 returns50688000B.

The next diagnostic pauses profiling through initialization/prefill, and arms
only the first following native decode with a drop guard. No I/O/admission cap
or original owner proof is changed. Original full-lifetime profiling is rejected
and excluded from timing selection. Sourcee8270af passes491 native tests, both shader suites,32 Python controls,
50 runtime host tests plus two isolated logger configurations and strict Clippy.
The first-following-step diagnostic still rejects with deferred router I/O
under the unchanged cap; same frozen49d016ca artifact with original per-router
cleanup passes all short fixture/owning/fresh/cancel gates. It reports one
profiled step, 64.183445ms total completed GPU kernels (generic custom kernels
33.254779ms, clears6.821887ms), not a clean speed result. Profiling changes
scheduling and is excluded from speed selection. GPU recovers50688000B.

The next explicit source-pageout experiment replaces ineffective COLD/PAGEOUT
with clean original file PTE release before descriptor cache advice. No private
anonymous allocation, source/file contents, inode permission or original proof
budget is changed. A live borrowed slice across full interior pages must refault
identical bytes at the same address; the receipt distinguishes this method from
historical pageout.127 core tests (3 ignored),491 native tests plus both shader
suites,32 Python controls and strict all-target Clippy pass. Physical
source02e1a9b/artifactd53c7a5c passes all128 IDs and original owning/fresh/
lifecycle gates with29994ms decode (4.2342 tokens/s),19879ms prefill. Warm
readbytes fall to5560762368, majorfaults31597, user/systemticks4493/981.
Advice covers25519091712 bytes in1195ms, not an actual freed-byte receipt.
A single ordered sample improves the observed rate; matched significance is
not claimed. Physical B70 returns50688000B.

Next, aggregate successful-call CPU prepare/gate-up/down/total host intervals
are explicitly enabled only in the fixed-public native lab. This isolates
CPU math/read wait from the broader host enqueue interval; no private trace
or installed/API change.128 core tests (3 ignored),491 native tests and both
shader suites,33 Python controls and strict all-target Clippy pass. Physical
source9c4ddef/artifact412f3940 passes all original full-model/owning/lifecycle
gates,24659ms decode (5.1502 tokens/s). CPU prepare347049742ns, gate/up
1988165674ns, down1076907721ns, total3484611488ns over127 steps. Warm
readbytes17580032, majorfaults572; the CPU set is now largely warm. Native
CPU experts consume27.44ms/token and host total194.17ms/token, so remaining
work focuses on execution/fences rather than more CPU math tuning.
The existing fixed-public scheduling diagnostic measures128 broadcasts in
1654/1813us versus scoped60173/57425us; this is not model speed.

The next explicit shared vector projection fuses the original finite fault
coverage into unchanged arithmetic and last-tile owner reads, reducing one
separate kernel per projection. Source checks pass492 native tests, both offline
shader suites,33 Python controls and strict all-target Clippy; physical
acceptance source5f6527f/artifactbcd0e49b passes all128 IDs and original
owning/fresh/lifecycle gates,23457ms decode (5.4142 tokens/s),18688ms
prefill. Warm uploaded bytes0, diskread67944448 bytes. CPU total3701666222ns
over127 steps. Physical memory returns50688000B and classified kernel
fault counters remain0. This is one ordered warm sample, not a paired
comparison. The warm original native-clear control on the exactv20
artifact passes all128 IDs and owning/lifecycle gates, but4.9075 tokens/s
with16384 warm read bytes does not improve the5.1502 sample. Native clears
remain unselected.

The user granted exclusive GPUs. Installed Kuno had an idle managed worker
holding15978409984 B70 bytes with no active request; a graceful Kuno restart
released it and physical VRAM returned50688000. There was no GPU reset/unbind
or reboot. Kuno admission is currently paused for this run, active=false,
waiting0, recovery_required=false. Normal admission must be restored at the
next checkpoint. Source isolation is the same owning worktree; do not touch
the canonical Vashti task. No installed binary/configuration/API change.

## Complete Next shared execution acceptance, 2026-10-08

The user authorized transferring the resident Ornith improvements into complete
Flash-Next inference. This block reaches **3.5006 decode tokens/s** on the B70,
versus the earlier1.691 artifact. The approximately10-token/s target remains
unmet. This is an observed laboratory improvement, not a matched Ollama or
paired historical-artifact comparison. No Ollama baseline was repeated.

Source commits `775d985`, `fc1e89e`, `d351bd6`, `62a52ba` and `1a80d5b` connect
shared packed dense/router kernels, vector recurrent attention, batched private
parameter uploads, the exact original CPU reduction in SIMD registers, and
explicit direct caller device dispatch. Original mapped Ollama weights, routed
GPU/CPU experts, rank merge, full private budgets and owning erasure remain.
The uploaded INFO records have distinct slots and ordered copy/task submission;
the direct device handle retains the original serialized WGPU server and fences.
These remain opt-in fixed-public lab modes, outside installed dependencies.

| Complete75/128 fixture | Prefill ms | Following127 decode ms | Decode tokens/s |
| --- | ---: | ---: | ---: |
| Shared dense/router | 60243 | 83606 | 1.5190 |
| Plus vector GDN | 53080 | 81255 | 1.5630 |
| Plus batched private parameters | 49319 | 44935 | 2.8263 |
| Exact CPU finish,96-slot banks | 54263 | 41974 | 3.0257 |
| Direct device dispatch, standard banks | 48254 | 37941 | 3.3473 |
| Direct device dispatch,96-slot banks | 65978 | 36280 | 3.5006 |

Each row is one ordered compiler-idle sample with its actual source/artifact;
do not infer statistical significance or independent component percentages.
The final96-slot choice completes29205 CPU misses and31755 GPU hits, versus
36916/24044 with standard banks. It improves decode modestly but worsens
prefill. Model bind3272ms follows file warming, not a matched cold-load test.
Seed424242, temperature0 and all128 original native output IDs are unchanged
for every accepted long run. The separate CPU-Q8 trial fails the original-ID
gate and is excluded; its full receipt and failed acceptance are retained.

Selected fixed-public artifact: source
`1a80d5bfc4b01e5984c10daf5c7002869aa26895`, Cargo feature `direct-device`, SHA256
`fe49069954829b2f8514b889b66967564937a67fdaabffe60a58418d437b8ec2`, frozen in
`/media/jmicko/faststore/kuno-development/next-shared-build-20261008-v5-inline`.
Use explicit inline dispatch/intel96, cap10, pooled CPU workers, overlap,
mapped weights, large weight cache, compact INFO, task batching, GPU router,
deferred router IO, retained metadata, fast GPU, shared kernels/GDN and batched
parameter upload. Exact requested/observed fields, hashes, source identities,
all13 samples, tests and postflight are in the
[numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-shared-speed-20261008.json).
The [count1 contract](kuno-native-text-count1-hybrid-plan.md#complete-model-shared-execution-acceptance-2026-10-08)
records reproduction and remaining gates.

119 core tests pass with3 ignored; normal/direct native suites pass479/481
tests plus two separate offline shader suites each.50 imported runtime tests,
22 hybrid Python controls, strict all-target Clippy, formatting and provenance
checks pass. All accepted physical samples pass original row/router/GDN probes,
fresh continuation, the existing cancellation/poison recovery checks and full
1378-owner/818413568-byte MAIN zero audits, source0, without context retirement.
Actual48-GiB cgroups enforce swap0/core0. No real prompt, output or route trace
is stored. These tests do not establish arbitrary-request quality, full-logit
equivalence, long-context/soak or production worker acceptance.

The next speed work should address CPU expert execution and remaining
fine-grained GPU handoffs. A12-second public CPU profile samples52.30% of cycles
in CPU AVX experts and13.30% in the old threaded device server. These are CPU
cycle shares, not wall-time shares; its profiled model run is excluded from
clean speed selection. Final cgroup snapshots show major faults/page-cache
refaults during decode with no swap or OOM, but IO counters are unavailable and
I/O causality is unproven. GPU phase counters are nested host intervals, not
isolated kernel timestamps. Mixed-card text execution and speculative expert
prefetch/MTP remain future experiments.

At postflight all lab jobs exited and Kuno admission is resumed: active=false,
waiting0, paused=false, recovery_required=false. Kuno11434/direct Ollama11435
return HTTP200 with no resident model; Vashti7771 returns HTTP200. Intel
physical VRAM recovers to50688000 bytes; NVIDIA541MiB includes the desktop.
The same boot has zero classified Xe wedge/timeout/GT-reset/memory-fault
messages. No installed binary/configuration, API/catalog/default inference,
Vashti or release change occurred. Required Vashti changes: none. The broader
goal remains paused and incomplete; this directly authorized trial is finished.

The final checkpoint also formats two earlier imported runtime functions and
regenerates provenance; those whitespace changes do not alter the benchmarked
artifact or behavior. The receipt distinguishes its original runtime patch
hash from the checkpoint patch hash. Imported runtime tests/Clippy are rerun
after formatting.

The resident/operator checkpoint below is historical. Its statements that
shared Next arithmetic is not connected and whole-model speed remains1.691
are superseded by this acceptance.

## Resident speed acceptance, 2026-10-08

Ornith now meets the approximately80-token/s resident target:85.35/85.38 for
two fresh75/128 sequences and82.6–83.0 on two EOS-finished public prompts.
All128 IDs match the native scalar reference. The shared-kernel refactor
recheck measures86.00/85.27 with the same IDs. See
[the acceptance contract](kuno-native-ornith.md#resident-speed-acceptance-2026-10-08)
and [aggregate receipt](../apps/kuno/benchmarks/text/receipts/ornith-resident-speed-20261008.json)
for exact source, artifacts, modes, counters and numerical limits.

The main measured fix is cached parallel expert selection (instrumented router
42.064→0.419ms across40 layers), combined with vector packed Q4/Q6 calculations.
Explicit unchecked runtime bounds add roughly9% to the fixed-geometry lab;
bounds-checked vector/router still measures77.5–78.2. Defaults are unchanged.
Teacher-forced128-token comparison to Ollama gives121 top1 and7 rank2 matches,
with identical scalar/fast choices. Free-running Ollama parity is not claimed.

At that earlier checkpoint, backend source through `ffe1cf0` shares these arithmetic kernels and
validates original Flash-Next Q4/Q5/Q8/F32/Q6 slabs plus512/top10 routing. This
is an operator port, not yet complete-model integration. Earlier complete Next
speed remains1.691 tokens/s. The next milestone is approximately10 whole-model
Next tokens/s before speculative FreeToken/Strata work. Ordinary dispatch,
buffer reuse, arithmetic and CPU/GPU offload remain in scope; avoid repeating
Ollama baselines or expanding API/lifecycle work.

Accepted normal artifacts: Ornithv7`55cdb1fa…` at`26829f6`; shared Ornithv8
`b6b81d72…` at`4a06b83`; Next operatorv1`e9a3c833…` at`ffe1cf0`. Two binary
offline shader suites and strict all-bin Clippy pass. All GPU jobs use48-GiB
user cgroups with swap0/core0 and normal explicit private-buffer clearing.
B70 physical memory recovers to50,688,000 bytes with no classified Xe fault.
No installed binary/configuration, gateway/API/catalog, frontend or release
change. Required Vashti changes: none.

At this completed block's postflight all benchmark units exited and inference
admission was resumed: active=false, waiting=0, paused=false and
recovery_required=false. Kuno11434/direct Ollama11435 return HTTP200 with no
resident model; Vashti7771 returns HTTP200. Intel physical VRAM is50,688,000
bytes; NVIDIA596MiB includes the desktop. Kernel classification reports zero
wedged, timeout, GT reset or memory-fault messages on the current boot.

For the next whole-model port, retain the previously measured complete Next
phase boundaries: the old long run spends about186ms/token in attention/GR
enqueue,172ms in router fences,108ms in final proofs and111ms in routed-expert
enqueue. These are host intervals including dependent queued work, not isolated
kernel durations. Port the accepted packed arithmetic and cached512/top10
selection into the original whole-model owners and reduce repeated dispatch/
metadata work. A new operator test alone does not retire this whole-model gate.

The historical checkpoint below describes the original13.9-token/s artifact,
not the currently accepted fast laboratory. Its first result and source-only
block candidate statements are superseded by this section.

## Fully resident Ornith checkpoint, 2026-10-07

The user changed the immediate diagnostic model to the already downloaded
`ornith:35b`, to isolate core performance without offloading. Its Qwen3.5 MoE
adapter is implemented in Rust/WGPU Vulkan, using original packed GGUF weights
and the existing Rust tokenizer. There are no intermediate CPU routing reads:
all40 layers, all256 experts per layer, GDN/full attention and the final head
are GPU resident. This is a separate fixed-public laboratory, not a deployed
engine registration. See [the architecture/measurement contract](kuno-native-ornith.md).

**Source and exact artifact:** `3047d24` introduces the graph; `c174aa9`
handles omitted optional BPE boundary flags and numeric public reference
capture. The accepted execution artifact is
`6917d349c17bbac3eba4e97cfb67a031e96e2c10396a9f473d166f49caa55093`,
frozen in `/media/jmicko/faststore/kuno-development/ornith-resident-build-20261007-v2`.
It retains21,155,768,832 packed weight bytes and235,737,136 private GPU bytes;
915 dispatches execute a complete token. The normal48-GiB cgroup has swap0
and core0. No weight copy, driver change, reset, reboot or install occurred.

**Measured outcome:** two75-prompt/128-output fresh native sequences reach
13.9408 and13.9184 decode tokens/s (127 following steps), with4.987/4.968-second
serial prefill. Native first eight outputs match Ollama; long outputs match the
first14 IDs, then diverge. Both native fresh sequences reproduce identical128
IDs. Thus functioning inference is observed but full-model numerical parity is
**not accepted**. Different activation/reduction arithmetic is a possible
explanation, not an established diagnosis.

Ollama's same public raw prompt/seed424242 runs fully on GPU at75.3222 cold and
83.3857 warm reported decode tokens/s. One numeric-reference run measures
83.5502 and reproduces128 BPE IDs from its in-memory public response. The
identical warm request may reuse Ollama's internal prefix; native resets all
sequence state. Native load3.954seconds follows file warming and is not a
matched cold-load win. All receipts are fixed-public numeric data; no real
prompt, output, image or routing trace enters persistence. See
[the aggregate receipt](../apps/kuno/benchmarks/text/receipts/ornith-resident-foundations-20261007.json).

**Next candidate:** the subsequent explicit `KUNO_LAB_ORNITH_BLOCK_DOT=1`
reuses Q4/Q6 headers and packed byte pairs during subgroup dots. It is off by
default and source-tested only. Do not claim a performance or device-parity
win for it or silently select it. On resume, first compare its individual
packed projections numerically, then teacher-force identical public tokens to
locate long-run divergence and use GPU timestamps to identify the remaining
core bottleneck. Keep the work on speed and arithmetic, not API/lifecycle
expansion. Ornith's14-token/s result does not satisfy Flash-Next's10-token/s
goal or establish Ollama parity.

**Tests:** 118 core tests,477 existing native tests, the new offline shader
inventory,118 Python control tests and strict release Clippy pass. The first
Ornith trial rejected missing optional BOS metadata before GPU creation;
the corrected artifact completes both short and long sequences. No extended
cancellation, multimodal, arbitrary-request, MTP, quality/soak or production
acceptance follows from these tests.

**Earlier heavy-pass outcome:** `9d625d7`'s integrated Flash-Next cap10 path,
with mapped weights, exact AVX2 CPU experts, pooled workers, compact dispatch
metadata, token-level IO, GPU routing and shared-expert overlap, completes the
original75/128 fixture at1.691tokens/s. It remains far below10. Its larger-bank
profiles and aligned upload erasure are source-tested through `9718587`;
expanded-bank device timing is outstanding. The approximate Q8 CPU-activation
long trial diverged and is not selected. Do not combine individual operator
improvements into an unmeasured whole-model gain.

**Restored system:** all benchmark units exited; B70 physical usage recovered
to50,688,000 bytes, NVIDIA398MiB includes the desktop, and no classified Xe
wedged/timeout/reset/memory fault appeared. Kuno admission is unpaused with no
active/waiting request or recovery latch. Kuno11434 and direct Ollama11435
answerHTTP200 with no loaded model; Vashti's existing process on7771 serves
HTTP200. The failed metadata-only test unit was cleared. Installed binaries,
configuration, API/catalog/routing, Vashti and release artifacts are unchanged.
Required Vashti changes: none.

## Joined overlap short ABBA checkpoint, 2026-10-07

The user-authorized pooled-worker and overlap block is finished. Goal remains
paused; no GPU/compiler job is running and installed admission is resumed.
Source`44fd9bc506eedfd701f68c71cb07f47c423cb533` adds explicit pooled count1 CPU/GPU
submission overlap, bounded borrowed orchestrator/parent header capacity, joined
failure paths and unchanged original rank merge/all-family publication.111 core
unit/integration tests,465 native tests,108 Python controls, strict release
all-target Clippy for both crates and formatting pass. Four actual normal-binary
invalid worker/overlap configurations reject before adapter/model access.

Normal artifact SHA`ae9c293443445458ff02faaa79bd88aa29d3222b5dfc3144ab5fa0caa21a61f9`;
source/build`native-count1-overlap-build-20261007-v1`; physical samples/postflight
`native-count1-overlap-device-20261007-v1`, both under faststore's
`kuno-development`. All eight compiler-idle same-artifact cap10 pooled
serial/overlap/overlap/serial short fixtures pass original output IDs, actual CPU
miss counts/fresh continuation, layer0/3 cancellation, poison/recovery and all
four1378-owner/818413568-byte owning erasure audits/source0. Same-stream flush
submits GPU expert work before CPU join; it is not a completion proof or an
instrumented physical-kernel simultaneity claim. The protected serial control
uses the same post-join CPU upload ordering as the overlap candidate.

| Card | Serial short decode ms | Overlap short decode ms | Mean duration difference |
| --- | --- | --- | --- |
| B70 | 4777,4861 | 4653,4796 | 1.96% shorter |
| 3060 | 6460,5979 | 6075,5941 | 3.40% shorter |

These small differences overlap the observed sample variation, especially the
active NVIDIA desktop. Do not select overlap as a reliable performance win or
multiply percentages across distinct artifacts. Pooled worker reuse's prior
short paired benefit remains7.22%/8.31%; overlap is an opt-in scheduling
experiment. Current overlap short rates are approximately1.48 B70 /1.17 NVIDIA
accepted following steps/s, far below the10-token/s working target. No new
Ollama baseline ran. Full-vocabulary logit, hybrid late-finality and original
75/128 gates remain unaccepted, followed by matched repeated long comparisons
and profiling of the remaining GPU/fence work. Short eight IDs do not prove
full numerical equivalence or long stability. Defaults/production are unchanged.

Exact aggregate/source/hash/sample receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-overlap-20261007.json`.
Postflight Intel physical memory returns50688000/34242297856B and Xe classified
wedged/timeout/reset/memory-fault counts remain zero on the same boot. NVIDIA
retains desktop allocations. Both inference listeners respond with no loaded
models; admission active=false/waiting0/paused=false/recovery_required=false.
Sudo cache was refreshed throughout. No installed binary/configuration change,
service restart, hardware reset/reboot, public API/catalog, Vashti or release
change. Required Vashti changes:none. Source is isolated on
`codex/kuno-native-text`; preserve the canonical frontend checkout.

## Pooled worker short ABBA acceptance, 2026-10-07

User-authorized bounded optimization while the goal remains paused. Source
`408abb9e26ed873e740f264db264297025e28539` adds explicit scoped/pooled CPU workers
without changing arithmetic, default execution, frontend or gateway.110 core
unit/integration tests,464 native tests,107 Python controls, strict release
all-target Clippy for both crates and formatting pass. Fixed-public scheduling
control establishes worker setup savings only. Normal artifact SHA
`926abe803dab2d8cb92789ddfda10690a7a1aeb1438a03499cf3d0f4c62132cf`, build
`native-count1-pool-build-20261007-v1`; eight protected compiler-idle same-artifact
short cap10 scoped/pooled/pooled/scoped samples under
`native-count1-pool-device-20261007-v1` (both directories in faststore's
`kuno-development`). All outputs, actual CPU-miss counts, fresh continuation,
layer0/3 cancel, poison/recovery and four complete1378-owner erasure audits pass.

| Card | Scoped short decode ms | Pooled short decode ms | Mean duration reduction |
| --- | --- | --- | --- |
| B70 | 5318,5374 | 4978,4942 | 7.22% |
| 3060 | 6903,6493 | 6173,6110 | 8.31% |

These are seven-step short repeated comparisons, not long throughput or the
10-token/s target. Full-vocabulary logits, hybrid late-finality and75/128 gates
remain unaccepted. Exact receipts:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-pool-20261007.json`.
Intel physical memory recovered to50688000B; content-free Xe classifications
remain zero. NVIDIA retains the active desktop. Admission resumed with no
active/waiting work or recovery latch. No installation/config/service restart,
hardware reset/reboot, API or Vashti change. The next separately authorized
slice is concurrent CPU work and parent GPU expert submission, with bounded
joined ownership and unchanged final merge. No overlap is implemented by408abb9.

## Both-card short hybrid acceptance and constructor correction, 2026-10-07

The user restored sudo and authorized this bounded GPU block; the goal remains
paused. The initial `a8cb5e0` Intel cap0 trial stalled during model construction:
the outer complete-model allocation held MAIN admission while hybrid reservation
called `healthy()` -> `admission.active()` on that same non-reentrant mutex.
The user lab unit was stopped after181 seconds, with no accepted receipt or
owning-erasure claim. Intel physical VRAM returned to50688000 bytes and the
content-free Xe fault classification stayed zero. Preserve that failed attempt
at `native-count1-hybrid-device-20261007-v1`; do not count its misleading systemd
"success"/TERM summary as model acceptance.

Correction `ff3c7d186b5828b6e7b43e9a68b99f048167ca70` supplies a borrowed sealed
construction-allocation proof from inside the existing guarded callback. It
binds the original ledger/stream/sequence and checks poison/retirement flags
without relocking. Arena IDs, worker plan and page checks remain in place;
normal runtime locks and all acceptance/cleanup checks are unchanged.
463 Rust host/offline tests, strict release all-target Clippy, formatting and a
normal Cargo release build pass. The Python validator source is unchanged from
its106-control acceptance, and all four actual receipts pass it again.

Normal binary SHA:
`4382edf25862e600c82eaf59a141fdba37a8fc056a47105da601a9dc3654de4c`.
Build/provenance: `native-count1-hybrid-build-20261007-v2`; physical receipts,
logs, run records and postflight: `native-count1-hybrid-device-20261007-v2`, both
under `/media/jmicko/faststore/kuno-development`. The shared normal target now
contains this new binary; use its own Cargo transcript, never the earlier one.
Aggregate receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-hybrid-device-20261007.json`.

The corrected same-binary cap0 and cap10 short fixtures pass on **both cards**:
original15-token8/7 prefill and all eight chat output IDs, seed424242/temperature0,
seven accepted native steps/scalar0, original SingleQ4All/tile1 policy, actual
completed CPU counts, pre-cancel/poisoned retry rejection, fresh-first and next
output matches, layer0/3 cancellation, all four1378-owner/818413568-byte MAIN
zero audits/source0, normal exit, no retirement and swap0/core0 execution.

| Card | CPU cap | Seven-step decode ms | Completed CPU misses | Fresh CPU misses | Warm expert-weight uploads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intel B70 | 0 | 5440 | 0 | 0 | 2619187200 B |
| Intel B70 | 10 | 5448 | 1256 | 102 | 0 B |
| NVIDIA3060 | 0 | 9056 | 0 | 0 | 4829081600 B |
| NVIDIA3060 | 10 | 6849 | 2259 | 273 | 0 B |

These are **single-order, compiler-idle short observations**, not repeated ABBA,
long throughput, cold-load or production selection. Intel is effectively
unchanged in this pair; NVIDIA duration is24.37% shorter. CPU reads and private
activation/result bridges still occur, so zero expert-weight uploads is not
zero I/O. CPU-assigned misses remain outside the GPU bank; later actual miss
counts differ from cap0, as designed. No routing trace or private vectors are
saved. Scope is the fixed public fixture only. No new Ollama baseline was run;
the10-tokens/s target is unmet and the earlier long1.4258 result remains the
applicable accepted long observation.

**Still outstanding before selection:** full-vocabulary relative logit gates,
hybrid late-layer/second-proof/error/panic/partial-family-commit failure gates,
and the original75/128 whole-model fixture, followed by repeated long cap0/10
ABBA. The short cap10 outputs are not full-vocabulary numerical acceptance.
CPU jobs still join before GPU enqueue, and `cpu_expert::parallel_rows` still
starts/joins two16-worker scopes for each nonempty CPU operator call. A reusable
scoped worker pool or better overlap remains a measured next optimization;
this inspection does not isolate their costs or establish an improvement.
Preserve bounded job/scratch/header ownership and failure joins when changing
workers; do not silently change activation precision or original reduction order.

All four lab units exited; no lab/compiler job from this block remains running.
Installed Kuno admission is restored and verified active=false/waiting0/
paused=false/recovery_required=false. Kuno11434 and the currently configured
direct Ollama11435 both return HTTP200 `/api/ps` with no loaded models (11433
is not the current upstream; no port configuration was changed here). Root xe
query reports50688000/34242297856 bytes; NVIDIA has388/391MiB in two postflight
snapshots including the active desktop. Xe kernel7.0.0-38 classifications for
wedged/timeout/GT-reset/memory-fault messages are all zero in the same boot.
No reset/unbind/reboot, service restart, installed binary/configuration, public
API/catalog/routing, application or release change occurred. Required Vashti
changes: none. This section supersedes the prior source-only status only for
corrected **short** whole-model output/lifecycle/erasure acceptance.

## No-root CPU-miss model registration checkpoint, 2026-10-07

The goal remains paused. The user authorized a bounded source/host pass while
at work, with no sudo or sudo workarounds. Source `a8cb5e00b00f6bb8938df59bc819cef68ad0a7ca`
on `codex/kuno-native-text` registers the separately protected fixed-public
`resident-count1-hybrid-chat` entry. It reserves the existing original count1
CPU-miss workspace, keeps8/7 native prefill, then selects SingleQ4All/dense1 and
cap exactly0 or10 at the completed original boundary. Default inference and
ordinary1376-owner constructors are unchanged.

An actual all48 joined/source-validated MoE permit supplies the checked
zeroizing CPU-miss delta. Overflow is preflighted before any family commit;
the count advances only with final accepted model position after all finishes,
PLE, original bank metadata and health/cancel checks. Cleanup erases it and
failed owners cannot expose it. The compiled public fixture alone records
aggregate counts, never expert/rank traces. Cap10 must prove actual CPU work
both in original decode and fresh continuation; cap0 must prove zero. Both
layer0/3 cancellation probes run the selected cap. The separate validator pins
four1378-owner/818413568-byte audits and all original and hybrid lifecycle
proofs, preserving ordinary1376-owner acceptance.

**Completed without root:**461 Rust host/offline tests,106 Python rejection/
runner controls, strict release all-target Clippy, formatting/diff checks and
normal Cargo release build. Four actual normal-binary negative entry cases
reject missing protections, invalid cap, extra argv and disabled-policy override
before adapter/model access, without a receipt. Python synthetic documents are
validator controls, not hybrid device evidence. The normal binary SHA is
`1c44a37a3c2cd44d41a03946bdd2286bfdab1c727c0843fbe925619cf0ff2197`;
artifact directory is
`/media/jmicko/faststore/kuno-development/native-count1-hybrid-build-20261007-v1`.
It contains the pinned binary, successful Cargo transcript, source identity and
`predevice-rejections.json`. Aggregate source-only receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-hybrid-source-20261007.json`.
The shared normal target executable now belongs to this new build; do not use
it with older9d4f4277 build transcripts. Earlier frozen/durable evidence remains
unchanged.

**Next requires sudo/shared GPU coordination:** pause installed inference
admission, unload/verify actual memory and exclude other GPU work. Run cap0 and
cap10 sequentially on each card using the exact new artifact, for example:

```sh
python3 apps/kuno/benchmarks/text/run-native-count1-hybrid-lab.py \
  --cargo-artifacts /media/jmicko/faststore/kuno-development/native-count1-hybrid-build-20261007-v1/cargo-artifacts.jsonl \
  --model /path/to/the-existing-original-Ollama-blob \
  --card intel --cpu-miss-cap 10 --run-id hybrid-intel-10 \
  --receipt /media/jmicko/faststore/kuno-development/hybrid-intel-10.json
```

The wrapper itself never acquires sudo, pauses services or claims exclusive GPU
access. It verifies swap0/core0/8-GiB execution and exact public eight-token
outputs, then records acceptance only after full lifecycle/audit validation.
Continue hybrid-specific late-final-layer, second-proof and partial-family-
commit fault gates before production selection; matched long timing follows
numerical/lifecycle acceptance. These remain physically unaccepted. CPU jobs
still join before GPU enqueue: no overlap, persistent CPU pool, mapped public
reader, automatic split, hot-request reuse or speed win is established.
The10-token/s target remains unmet; preserve the earlier1.4258 Q4 observation
and do not repeat Ollama baselines.

No GPU/model job, sudo operation, service restart/pause, installation, driver/
hardware reset, API/catalog/routing or Vashti change occurred in this pass.
No compiler or lab job started by this pass remains running. Installed health
and physical GPU memory were not re-audited this turn. Required Vashti changes:
none. This section supersedes the earlier unexposed CPU-miss registration and
counter proposals; it does not supersede their remaining physical acceptance
or introduce a production capability.

## Latest owned-public upload and long comparison, 2026-10-07

`feb76b3` records both-card finite-grid acceptance. `79ba326` adds the narrowly
scoped `create_public_owned(Vec<u8>)`: move immutable original public model bytes
into the ordinary server write queue, avoiding redundant host Vec copies.
Static original GGUF uploads and individual expert sources use it; private-data,
ordered-source, reused bulk-buffer, admission, error and erasure paths remain
unchanged. This is not zero-copy GPU staging. The runtime README records the
exact boundary; the18-file patch SHA is
`64e243409cde4685bc0207876156b595e0081b8d830c1faf66117bb1deb8278f`.
It reconstructs against pinned archives with all modified hashes matching.
459 Rust native tests,49 runtime tests, strict all-target Clippy and formatting
pass; the Python controls are unchanged from their earlier99-test acceptance.

Normal source `79ba326b4de1ff0ee5f7aa4244619bc47378cfb6` release SHA is
`9d4f4277efdcbfcc2349657b9c1bc5d6bd7aef8f3a531c756d38099a4a187144`.
Both adapters pass52 routed/14 original-format/12 expected-failure cases and
the complete eight-token model, fresh continuation, layer0/3 cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits, source0, normal exit/no retirement. Short observations are
Intel5,096ms (read884/upload7ms), NVIDIA7,808ms (read613/upload23ms).
The upload timer is asynchronous enqueue time, not device completion; the
previous Intel run was profiled. These are not matched copy-before/after timing.
Exact evidence: `rust-native-owned-public-upload-device-20261007.json` and
`native-public-owned-build-20261007-v1` under the existing receipt/artifact roots.

The **complete compiler-idle same-binary Intel Q4 ABBA** passes all four original
75/128-output, seed424242/temperature0 model and lifecycle gates. Every sample
uses native tile1, prefill MAX8, actual52-GiB host cap/swap0/core0, the same
borrowed unmodified model, a fresh model owner, and all four original full audits.

| Sample | Profile | Prefill ms | 127-step decode ms | Decode tokens/s |
| --- | --- | ---: | ---: | ---: |
| 1 | single-all | 30704 | 90578 | 1.4021 |
| 2 | single-q4-all | 30019 | 89408 | 1.4205 |
| 3 | single-q4-all | 30308 | 88740 | 1.4311 |
| 4 | single-all | 30281 | 90183 | 1.4082 |

Generic mean1.4052, Q4 mean1.4258 tokens/s: a small observed1.47% rate change,
not the required10 tokens/s. Two repetitions on one card establish only this
bounded comparison, not broad quality or production selection. OS caches were
not flushed, prefill includes lazy loading, and this does not measure hot
cross-request session reuse or the upload copy change in isolation. Process
model caches do not survive samples. No new Ollama baseline was run.
All128 output IDs match, no OOM or swap is recorded, and actual cumulative
unit memory peak is about8.3GiB despite the52-GiB allowance. Preserve numerical
proofs and telemetry at `native-owned-q4-performance-intel-20261007-v1`;
aggregate identities/results: `rust-native-owned-q4-abba-intel-20261007.json`.

Each completed public sample uploads50,656,768,000 expert-weight bytes during
decode:15,413 misses,45,547 hits. Warm original reads take roughly15 seconds;
phase waits also include queued GPU work and cannot isolate kernel times.
The next priority is the existing default-off count1 CPU-miss seam's explicit
whole-model numerical/lifecycle trial, with actual completed-miss evidence,
separate1378-owner/818,413,568-byte validation and cap0 control. The earlier
both-card packed CPU/GPU numeric/calibration acceptance below is already done;
do not rerun that study simply to repeat it. CPU jobs currently join before GPU
enqueue, so no overlap or automatic placement gain is established. A prepared
host/pinned positional or mapped public-weight reader and safe persistent CPU
workers are further measured candidates, not implemented by this pass.

Read-only PCIe inspection identifies Intel's host-facing2b:00.0 bridge and root
port at8GT/s x16; its internal2c:01.0/2d:00.0 fields report2.5GT/s x1 and should
not be used alone as the physical host-link bandwidth. NVIDIA's actual endpoint
reports5GT/s x4 through the chipset. These link fields are not transfer benchmark
measurements; no hardware setting/reset/unbind/reboot occurred.

The goal remains paused. Installed admission is resumed and verified
paused=false/recovery_required=false; both Kuno and direct Ollama /api/ps return
HTTP200 with no models resident, and Kuno /api/tags returns HTTP200. No lab unit
or compiler remains running. A privileged read-only xe query reports50,688,000
VRAM bytes used out of34,242,297,856; NVIDIA reports399MiB including the desktop.
No installed binary/configuration, API/routing,
Vashti, driver or release change was made; required Vashti changes: none.

## Clean paused checkpoint, 2026-10-07

This is the earlier checkpoint, before the additional authorized pass above.
The latest section supersedes its source/artifact and finite-grid status.

Owner branch remains `codex/kuno-native-text` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Source through `d13e95d`
passes459 native Rust host/offline tests, strict all-target Clippy, formatting
and diff checks. All99 Python native-lab tests pass; runtime49 fixture sources
are unchanged. The ordinary main checkout and installed binaries/configuration
are unchanged; installed admission reports paused=false, recovery_required=false.
All source-only agents stopped with clean isolated branches.

**Physically accepted dense candidate:** `b0c83c9`, `ee9434c`, `0741420` remove
seven inactive tile1 dense dispatch slices at the completed `SingleAll`/
`SingleQ4All` boundary, observe actual capacity and reject mismatches before
private clears. Prefill remains MAX8; active packed math, complete owners,
inactive stored values and fault/audit scopes are unchanged. The exact normal
`0741420` binary SHA256 is
`3098e9cb84ed7563f5160030717e9868a759932ee60769492d764a536337ad36`.

Both cards pass all91 matmul cases (84 prior cases plus7 real same-owner MAX8
prefill/count1 replays). The replay's paired GPU maximum absolute error is zero
on both cards; maximum CPU error is1.1920928955078125e-7. The independent strict
validator `7df78cd` passes both actual receipts against shapes independently
read from the original GGUF header. Operator receipts have no vendor field;
actual vendor selection is joined through the pinned unique-discrete-adapter
selection and separate same-binary model receipt. Per-element numerical gates
remain producer checks; aggregate maxima cannot reconstruct their relative
threshold. No timing authority follows from this validator.

Both complete eight-token chat runs pass the same IDs, seven actual native
continuations, fresh prefill/second-token replay, layer0/3 queued cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits with source0, normal exit and no retirement under actual8-GiB/
swap0/core0 protection. Intel generation is5,563ms (read962/upload392ms);
NVIDIA8,675ms (read540/upload770ms). No additional speed gain is established;
Intel has more read time than the prior short run. Exact proofs and identities:
`apps/kuno/benchmarks/text/receipts/rust-native-single-dense-device-20261007.json`,
backed by `/media/jmicko/faststore/kuno-development/native-single-dense-build-20261007-v1`.
The earlier long Intel75/128 fixture at `4d4ecbe` remains1.311tokens/s; it is
neither this new binary's long result nor matched ABBA performance acceptance.

**Source-only follow-ups, not physically accepted:**

- `ec0cae5` narrows count1 grouped finite/reducer public grids toY10/Y1, keeping
  defaultY80/Y8, the same kernel bodies, complete MAX80 owners/status/fault
  scans and active storage writes. Before timing, run the existing52 routed/
  14-format/12-failure numeric inventory and original complete model/lifecycle
  gates on both cards from a fresh normal binary.
- `b6269b4` plus stable assertion fix `3d65a44` derives the unchanged additional
  1MiB bridge runtime envelope (1,015,808bytes). `7546186` normalizes only
  validated inactive logical MIXED rows1..7 in the experimental count1 input
  scan; active arithmetic, owning gaps/padding and router/fault checks remain.
  `d13e95d` keeps the grid fixture and necessary CubeCL NaN/guard spelling
  lint-clean. This hybrid path is still unexposed and device-unaccepted; the
  default GPU path uses its original owners. Do not infer hybrid acceptance
  from ordinary dense tests.

No production native worker/default routing, API, installation, release,
Vashti change or hardware reset/reboot is accepted. No new long timing or CPU/
GPU overlap win is claimed. The normal shared target binary is still the
`3098e9cb…` artifact; Clippy/test binaries are not a new normal Cargo build.

**Next source proposals preserved, implementation not started:**

- After finite device gates, an explicit fixed-public `resident-count1-hybrid-chat`
  CLI can pin cap exactly0|10, SingleQ4All/dense1, the original chat/seed424242/
  temperature0/eight outputs, full zero audit and cancellation checks. It needs
  a separate strict1378-owner/818,413,568-byte receipt rather than weakening the
  ordinary1376-owner validator. Require actual completed CPU-miss count>0 for
  cap10 and zero for cap0; no routing IDs/masks/per-layer content may persist.
- A private zeroizing completed CPU-miss counter can derive a checked delta
  only after all48 actual joined/source/bank proofs pass, check overflow before
  any family commit, and publish only alongside final accepted position after
  all commits/finish/PLE/health/cancel checks. Failed or partial attempts never
  advance it; a healthy completed-boundary getter is required. This is a
  proposal, not a current counter or CLI capability.
- Existing one-shot lifecycle sites can test late_combine47, second-proof
  error/panic after actual PLE preparation, and cancellation after the first
  genuine GDN commit. Each needs exact-site evidence, terminal erasure/retry
  rejection and fresh-owner replay. Existing layer0/3 tests alone do not prove
  joined hybrid work or late publication safety.
- A standalone exact-tree subgroup proposal can first publish64 original lane
  partials to shared memory, keep one unconditional barrier, reconstruct the
  exact original tree independently within each complete16/32/64 subgroup,
  use uniform explicit shuffles (never subgroupAdd), and keep a second
  unconditional barrier. Original local lane0 either publishes its subgroup
  root or performs the exact serial shared-tree fallback. Unsupported/incomplete
  shuffles target each invocation's own active lane; final target0 is selected
  only for complete groups. Feature absence selects the untouched original
  kernel. Never assume local/subgroup lane mapping or width. Require offline
  WGSL/Naga uniformity, paired bitwise finite roots, signed-zero/nonfinite
  classification, forced fallback and actual width/mask/device/full-owner
  gates before dense or grouped integration; GDN sequential sums stay unchanged.
  No source files were created on the isolated `codex/kuno-exact-tree-shuffle`
  branch at `b6269b4`. Pinned source supports PLANE_DIM/UNIT_POS_PLANE/
  plane_ballot/plane_shuffle, but actual card widths are unverified. See the
  [WGSL subgroup specification](https://www.w3.org/TR/WGSL/#subgroups) and
  [shuffle semantics](https://www.w3.org/TR/WGSL/#subgroupshuffle-builtin).

## Additional finite-grid device acceptance, 2026-10-07

The user authorized another bounded optimization turn while leaving the goal
paused. The normal `91efea6a87b05b475a16aa4dfb8e8fa9cba03fef` artifact is
`d6b834687ac0865bd81f2c1eb732c69d4434a849bb3228487a3cfd9dd2ad6018`.
The count1 grouped finite/reducer grids now pass all52 routed cases,
14 original-format cases and12 expected failures on both actual Vulkan adapters.
The independent strict Q4 validator accepts original Q5 layer6/Q8 layer0.
Operator erasure covers1078 owners/153,944,064 bytes, with normal exit and
actual8-GiB/swap0/core0 protection. The operator run proofs now preserve the
actual public adapter marker and independently check the expected vendor.

Both complete eight-token chat/lifecycle runs pass the original IDs, seven
native steps, fresh prefill/second-token replay, layer0/3 queued cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits with source0. NVIDIA short generation observes8,525ms, read545ms,
upload767ms. Intel observes5,739ms, read962ms, upload363ms, but its run overlaps
CPU profiling and is **not** a comparable unprofiled timing. The CPU sample
covers30 seconds spanning startup/prefill/decode and possibly recovery/cleanup:
30.64% of leaf samples are memmove,10.97% memset. This is a lead, not isolated
decode attribution or permission to remove private clearing. No copy-path or
whole-model speed gain follows from these results.

Exact identities, receipts, owning audits and CPU sample scope are recorded in
`apps/kuno/benchmarks/text/receipts/rust-native-single-finite-device-20261007.json`
and the durable `native-single-finite-build-20261007-v1` artifact directory.
The experimental hybrid seam remains unexposed and device-unaccepted. There
is no API, installed service, Vashti, release, driver or hardware reset change.

## Resumed native continuation evidence, 2026-10-06

Root `codex/kuno-native-text` integrates `3735a2c` aggregate native phase timers,
`22c1ac6` protected native count1 continuation, `484fe41` same-binary 52-GiB
decode comparison, and `9fb6a2b` bounded CPU/GPU numeric CLI registration. The
current source passes 400 host/offline tests and strict all-target Clippy. There
is no production native worker, routing change, installation or Vashti change.

The release binary built from `22c1ac6` is frozen as SHA256
`6dcd3e2e59b2d184ff83b264b2f6110883daac116a1e07c9cf1c01c04cca44fd`.
Both cards pass the protected eight-token chat reference with native prefill
8+7 and seven native continuation steps, zero scalar steps. Cancellation,
poisoned retry rejection, fresh first-token recovery, all four original MAIN
audits (1376 pages/818,249,728 bytes each), source audits and normal cleanup pass.
The short runs include warm-up and do not establish a throughput improvement.
See `apps/kuno/benchmarks/text/receipts/rust-native-decode-device-20261006.json`.

The Intel scalar/native/native/scalar comparison ended after its second child,
with an aggregate supervisor failure. Preserve the complete failed directory
`/media/jmicko/faststore/kuno-development/native-decode-performance-intel-20261006-v1`.
Do not claim completed ABBA acceptance or replay it merely to obtain the same
result. Scalar sample1 is accepted at 71.032 seconds for 127 decode steps
(1.788 tokens/s). The native sample2 model receipt records 224.924 seconds
(0.565 tokens/s), the same 128 IDs and all four completed owning audits; its
supervisor sample record was not published. The exact failure location is not
proven. A `/proc/<pid>/io` exit race is a source-identified possibility; fix
bounded process telemetry without weakening model/lifecycle validation.

Native router-fence wall time is 151.757 seconds versus scalar 21.031. These
timers include dependent queued work, not isolated GPU execution. The native
eight-token static projection tile is a candidate bottleneck; a protected
public tile1 experiment is in source development, not yet built/device-accepted.
Its default must remain eight, prefill stays MAX8, and private count/positions
must remain runtime data. Independent numeric checks precede timing.

The new CPU/GPU numeric CLI builds but Intel currently rejects its fixed
operator fixture before aggregate publication. Both failed launches retain
actual swap0/core0/8-GiB protection and content-free run records under
`/media/jmicko/faststore/kuno-development/native-cpu-gpu-build-20261006-v1`.
Do not grant a timing or whole-model placement permit from source tests alone.
Constant stage diagnostics are being prepared without changing numerical gates.

## Current priority amendment, 2026-10-06

### Batched audit and staged CPU-miss acceptance

Root subsequently integrates `4d4ecbe` one mandatory full audited admission
readback, preserving completed lease drop, ordered-source release, context/
cancellation/sequence checks, telemetry and MAIN fence credit. Ordinary
readbacks remain unchanged; there is no general empty-epoch skip. All439 native
host tests and strict all-target Clippy pass. Runtime49 and Python81 fixture
sources are unchanged by this seam. Normal release SHA256 is
`3b192df34bf0c9cfa45d462afc14e341cd270c7cdf03f64ffd7197d3a035d71a`.
Both-card eight-token model/fresh replay/cancellation/recovery/full erasure
checks pass; short generation observes4,820ms Intel and8,701ms NVIDIA. See
`rust-native-audited-fence-device-20261006.json` in the receipt directory below
and `/media/jmicko/faststore/kuno-development/native-audited-fence-build-20261006-v1`.
The longer75-token/128-output Intel fixture also passes the same exact output,
native continuation, fresh replay, cancellation/recovery and four full owning
audits under actual8-GiB/swap0/core0 containment. Prefill is33,486ms;127 decode
steps take96,865ms (**1.311 tokens/s**). Weight reads/uploads are13,858/7,441ms.
See `rust-native-audited-fence-long-intel-20261006.json`. This is a protected
lifecycle observation, not matched ABBA performance acceptance; the10-token/s
target remains unmet. Explicit public count1 dense dispatch remains a separate
source candidate.

Root source through `97d50bc` adds the default-off, explicitly reserved count1
CPU-miss seam (`fc5fea3`, `b9b3f19`, compiler wiring `f391168`, corrected invalid
capacity fixture `97d50bc`). All450 native host/offline tests pass; strict
all-target Clippy passed at `f391168`. This source is not CLI-exposed or
device-accepted. Its selected path joins bounded CPU misses before GPU enqueue;
no CPU/GPU overlap or whole-model speed gain is claimed. Default owners remain
1376/818,249,728 bytes; explicit reservation adds two owners,1378/818,413,568.
No production, API, installation or Vashti behavior changes.

The subsequent upload-erasure batching commit `200b38a` also passes all of the
same both-card10/52/14/12/eight-case operator inventories and complete eight-token
model/cancellation/recovery/erasure gates. Native host tests pass436 and runtime
fault fixtures pass49; strict native/runtime-host Clippy and provenance pass.
Its normal release binary is SHA256
`3ce1dbd109d0251a6e6c6a72ea67bdc779a41451accfce2fea41b00f45db3b68`.
Exact build/run evidence is under
`/media/jmicko/faststore/kuno-development/native-erase-batch-build-20261006-v1`,
with aggregate `rust-native-erase-batch-device-20261006.json` in the receipt
directory below. Intel short generation observes4,916ms with router fence1.602s;
NVIDIA observes9,418ms with fence2.224s. The latter has greater route/read time
and establishes no additional gain. These short observations do not grant
matched performance acceptance. One mandatory audited admission fence and
whole-model count1 CPU-miss connection are the next separate source candidates.

Root integrates `f9e9006` bounded full audit copy/map batching, `ebd0046`
staged CPU-miss/GPU-hit numeric selection, `91d4dd9` protected eight-case CLI
and `793c351` its independent strict receipt validator. Native host/offline
tests pass436, runtime ownership fixtures pass45, Python controls pass81;
strict native/runtime-host Clippy and provenance reconstruction pass.

The normal `91d4dd9` release binary is SHA256
`8f08b872111d04fd97e7bc77c1675242257761386ddfd9a7490fa0d82764cded`.
Both cards pass10 GDN,52 routed/14 packed-format/12 expected failure cases,
and the eight original Q5/Q8 staged cases at CPU caps0/1/5/10. The latter
classifies actual noncontiguous misses, preserves resident GPU hits and merges
original ranks; it grants no cache publication, model or timing authority.
Independent validators pin vendor and original layers6/0.

Both complete eight-token `single-q4-all` model/lifecycle runs also pass unchanged
IDs, seven native continuation steps, fresh continuation replay, queued layer0/3
cancellation, poisoned retry rejection, fresh recovery and all four full
1,376-owner /818,249,728-byte MAIN audits. Processes exit normally without
retirement under actual8-GiB/swap0/core0 containment. Durable evidence is under
`/media/jmicko/faststore/kuno-development/native-audit-staged-build-20261006-v1`;
aggregate identities/proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-audit-staged-device-20261006.json`.

Short generation observations fell12,017→5,307ms Intel and13,478→8,838ms NVIDIA
for seven steps. Router-fence wall time fell8.391→1.861s and6.715→2.225s,
respectively; actual warm file reads remain separately recorded. These are
short lifecycle observations, not matched long-run throughput acceptance or
achievement of the10-tokens/s target. Upload erasure batching and the explicit
whole-model count1 CPU-miss seam remain separate candidates. No production,
API, Vashti, installation, release, driver or hardware change follows.

### Q4 block candidate and both-card operator acceptance

Root integrates `3e20935` original Q4 header reuse, `3f38840` strict protected
profile/comparison controls, `cfe8fa0` completed model-boundary selection and
`22a2413` a narrowly normalized shader-declaration test. 426 Rust host/offline
tests, strict all-target Clippy, formatting and diff checks pass. The existing
62 Python control tests passed at `3f38840`; no later Python change is part of
this build. The count1 CPU-miss plan at `dc440d8` is still a proposal.

The normal `22a2413` release binary has SHA256
`b8008cfe72009a2f8856b9bc39ba6d5c543cee371130365507a0c410e4b1f7da`.
Both cards pass 52 routed-MoE cases, 14 independent packed-format cases and
12 expected failure cases. Six Q4Block cases per card retain paired generic
tile1 references for both down formats and capacities16/64/96, including the
cross-segment boundary. Observed generic maximum absolute error is zero in all
twelve comparisons; the original numerical tolerances are unchanged.
Full owning offsets/tails/status, independent tree/core/scalar references,
unaltered bank ledgers and erasure remain enforced. Each probe audits1,078 MAIN
owners /153,944,064 bytes; controlled host scratch is17,543,168 bytes. Each process
exits normally under actual8-GiB, swap0/core0 containment without retirement.

Durable build/run evidence is under
`/media/jmicko/faststore/kuno-development/native-q4-build-20261006-v1`; aggregate
identities and proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-q4-block-operator-device-20261006.json`.
Both-card complete-model eight-token `single-q4-all` now also passes original
IDs, seven actual native continuation steps, full fresh prefill plus second-token
replay, queued cancellation at layers0/3, poisoned retry rejection and four full
1,376-owner /818,249,728-byte MAIN audits. Original source audits remain empty;
explicit normal cleanup passes without retirement. The exact additional profile
proofs and actual8-GiB/swap0/core0 protection are checked by the public runner.
See `rust-native-single-q4-model-device-20261006.json` in the same receipt directory.

Generation observations are12,017ms Intel and13,478ms NVIDIA for seven steps.
Intel warm weight-read time fell from1,018 to237ms versus the earlier short
single-all run, so its lower wall time does not establish a kernel improvement;
NVIDIA is essentially unchanged. No long Q4 ABBA is launched from these short
checks. Full75/128 timing, batched audit cleanup and a real staged CPU/GPU model
seam remain unaccepted. Strict Q4 operator validation is integrated at `abf9b9e`;
both actual reports pass and all72 Python native-lab tests pass. No production,
default/API/Vashti, install, release or hardware change follows. Admission is
resumed and verified quiescent while the next source candidate is prepared.

### Later owner acceptance

Root integrates `64027cf` public static projection tiles, `dd4aac7` protected
tile controls/exact validators/bounded telemetry exit-race handling, `3122e44`
constant CPU/GPU numeric failure stages, `e7bc6d4` actual tile1 recovery proofs,
and `1946020` narrow full-page public bank allocation for the CPU/GPU lab. The
latest source passes **409 host/offline tests**, strict all-target Clippy,
formatting/diff checks, and **51 Python native laboratory tests**.

Both cards pass **84 projection cases** (all seven original formats, counts
1/3/5/8, public tiles1/4/8), retaining the independent oracle, gaps/tails,
original-weight preservation and full owning erasure. Maximum absolute error
is 1.193e-7 on both cards. Frozen operator binary from `3122e44` is
`51581ade3fa1d2e0fe6ccd9a1ff89c31249721f85f76befd7bee9bfe78690ba5`.

Both cards also pass tile1 whole-model eight-token chat, fresh **tile1** second
token replay and cancellation after queued **tile1** work at layers0/3, with
all four original audits and zero scalar source owners. Frozen `e7bc6d4` binary
is `fe74ea739a03cfa4e1e3a555af912db65d1617c4d17bacc7cbaa243cc2e88ef3`.
Short generation durations are 13.012 seconds Intel and 14.187 seconds NVIDIA
for seven continuations; they include warm-up/cache effects and show no useful
improvement over the earlier short native tile8 runs. No long tile1-only ABBA
was started. Do not claim a matched throughput gain from these correctness runs.

The joint expert fixture failure is localized to strict public-bank physical
extent checks. Logical `client.empty(n)` requests did not guarantee a complete
power-of-two backing allocation under the exclusive allocator's rolling mean.
`1946020` requests the already charged full pages in this fixed lab and retains
the original bounded logical shader views. Ordinary model-bank allocation,
numerical tolerances, budgets and strict physical/nonalias checks are unchanged.
Frozen binary
`bf25aa3ae587b2364871496e8057baf168d9ed487b440a50b7a5aa4b893b9d2b`
passes all four numeric configurations on both cards: Q5 GPU-only/mixed,
Q8 mixed/CPU-only, original ranks0..9 including zero probabilities, concurrent
AVX2/16 physical workers. Every case completes18 MAIN pages/884,736 bytes plus
I/O audits. This permits only the stated subset's operator numeric claim;
other groups/splits/workers, timing, automatic placement and whole-model CPU
dispatch are still unaccepted.

All live jobs use actual8-GiB memory/swap0/core0 guards. Detailed identities,
hashes, counters and limitations are in
`apps/kuno/benchmarks/text/receipts/rust-native-tile-and-cpu-gpu-device-20261006.json`.
Single-member grouped MoE and streaming single-token GDN are the next isolated
candidates. Their later operator acceptance follows below; whole-model and
performance acceptance remain separate. No new Ollama runs, frontend/API/routing,
installation, release or hardware reset occurs.

### Single-token kernel operator acceptance

Root integrates streaming GDN `a0b9d5f`, single-member MoE `d07154f`, and
model-boundary policy/count gates with two probe compile corrections `1b486e7`.
416 host tests pass. Independent source reviews cover both candidates and the
model seam. The profile controls and their strict validators are still being
integrated; no production selection changes.

The exact normal release build at `1b486e7` has SHA256
`61aee450ea2b0b8bd2fbd57feb310a7268607929e8f681f6fbc0d85efa353a35`.
Both GPUs pass the ten-case GDN probe, including odd/actual single-streaming
shapes against unchanged chunk, scalar GPU and independent core references.
Both pass 46 routed-MoE cases, 12 expected failure cases, and 14 independent
packed-format cases (seven original formats, tiles8/1). Count/map guards,
immutable old state, finite checks, inactive offsets/tails/status, full owning
erasure and runtime I/O audits remain enforced. GDN audits160 MAIN owners /
113,868,800 bytes; grouped MoE audits994 /141,950,976 bytes across the probe.
Every protected process exits normally with actual8-GiB, swap0/core0 guards and
no context retirement. These are numerical receipts, not benchmark samples.

Durable build/transcript/run records are under
`/media/jmicko/faststore/kuno-development/native-single-operator-build-20261006-v1`;
aggregate hashes and numerical proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-single-kernel-operator-20261006.json`.
The following later model acceptance supersedes that gate for `single-all`;
compiler-idle matched timing remains separate.

### Combined single-token model acceptance

Controls `1d174d3` and lint fixes `776d58e` connect the public `single-all`
profile. Source tests pass417 Rust /60 Python and strict all-target Clippy.
The exact normal model build at `776d58e`, SHA256
`f05ab6e17de098d4c9f714a39839a5274695f68ab044bb9536ce400b1c2e6142`,
passes both-card original eight-token chat, seven native continuation steps,
actual observed single-all policies, full fresh prefill plus second-token
continuation, queued native cancellations at layers0/3, poisoned retry rejection,
all four complete1376-owner /818,249,728-byte audits, and explicit normal cleanup
without retirement. Each source audit is empty. Actual8-GiB, swap0/core0
containment and the three additional profile completion proofs pass.

Seven-step generation observations are Intel13,053ms and NVIDIA13,566ms,
essentially unchanged from prior short chunk runs. These are correctness/lifecycle
samples with file-cache/JIT variability, not a matched throughput result.
No long kernel-only ABBA is run or speed win selected. Intel's warm weight
read/upload observations total1,018/415ms while the router fence accumulator is
8,508,872,971ns; dependent kernel work and readback also require investigation.
Timers are host interval observations, not isolated GPU kernel execution.

Durable build and records are under
`/media/jmicko/faststore/kuno-development/native-single-model-build-20261006-v1`;
the exact aggregate is
`apps/kuno/benchmarks/text/receipts/rust-native-single-all-model-device-20261006.json`.
Independent whole-model acceptance of the two single-family profiles is not
claimed. The full75/128 fixture and performance selection remain unaccepted.

Root subsequently integrates `cc82731` fixed public CPU/GPU calibration helper
and `4195238` protected registration, preserving `1946020` complete public banks.
420 Rust tests and strict all-target Clippy pass; earlier60 Python tests still
cover unchanged controls. The following bounded timing acceptance is complete.
No production worker/default/API/frontend/install changes.

### Fixed original-read CPU/GPU calibration

The exact normal `4195238` build has SHA256
`b0a2cacda8b3dff0b4b14e2891870f1f9ff6424e9c294aa736e5b78929f82393`.
Both cards complete six fixed Q5/Q8 points, 12 exact-mode numeric gates and24
Concurrent/CPU-first/CPU-first/Concurrent samples, retaining each original
permit through its two tickets. All cases preserve ten original ranks, complete
18 MAIN owners /884,736 bytes plus I/O audits, unchanged tolerances and file
identity. Processes exit normally with actual8-GiB/swap0/core0 guards and no
retirement. No discarded or partial successful sample is accepted.

Two Concurrent samples per point give these **completed candidate** means in
milliseconds (including reads/fills, joined work, private candidate validation
and completed bridge/merge; excluding independent references and final erasure):

| Card / down | GPU10 / CPU0 | GPU5 / CPU5 | GPU0 / CPU10 |
| --- | ---: | ---: | ---: |
| Intel Q5 | 37.180 | 30.782 | 26.970 |
| Intel Q8 | 48.290 | 34.844 | 32.918 |
| NVIDIA Q5 | 40.265 | 33.174 | 25.447 |
| NVIDIA Q8 | 55.296 | 41.922 | 29.205 |

Fixed group0/vector0 and prior numeric/reference work warm the original weight
working set. This narrow operator trial supports further CPU-miss integration;
it is not prepared-host, cold storage, the32-group study, automatic placement or
full-model token throughput. Cleanup and full-with-reference walls are separately
recorded. No private-worker/native-default integration follows from these data.

Durable records are under
`/media/jmicko/faststore/kuno-development/native-cpu-gpu-calibration-build-20261006-v1`;
raw bounded samples, numeric gates, exact identities and hashes are in
`apps/kuno/benchmarks/text/receipts/rust-native-cpu-gpu-calibration-device-20261006.json`.
Next source work is an explicit single-token Q4 header-reuse candidate and a
reviewed CPU-miss/GPU-hit owner/ledger proposal. They remain unaccepted.

The user selected **10 decode tokens/s** as the working native performance target
and explicitly deferred additional/repeated Ollama baseline tests to conserve
development usage. Prioritize measured Rust-engine optimizations after the
applicable numerical/lifecycle checks; expanding the Ollama matrix is not a
resume prerequisite. The completed Intel host-limit experiment must not be
replayed merely to establish the same cache-reclaim result. A corresponding
NVIDIA containment study can wait unless it answers a concrete optimization
question. Preserve historical receipts and label results against the agreed
target rather than claiming a new matched Ollama comparison.

The goal was still paused when this amendment was recorded. No source/runtime
change, GPU load, benchmark or service operation accompanies it. A read-only
source review is preparing the existing native count1 continuation path as a
candidate; acceptance and performance remain unmeasured. Read the
[measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06)
before interpreting the historical practical steps and gates below.

## Latest integrated source and device evidence

Root worktree/branch remain
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, `codex/kuno-native-text`.
The original path resolves through the new `~/dev` symlink; managed worktree Git
pointers, source/read/write access and shared Demons shell are verified.

- `195d8a5`: integrates reviewed numeric candidate `2439d73`.
- `962c37d`: named protected `resident-numeric-boundaries-probe`, rejecting missing
  owned I/O/full audits before creating a device; no production route.
- `80aef9d`: exact aggregate-only numeric validator (author `36d2121`).
- `5ffffa4`: separate same-binary host-limit diagnostic (author `13844b5`).

396 host/offline tests passed before extracting the explicit CLI guard; its
additional focused test passed afterward. Strict all-target Clippy, formatting
and diff checks pass. The numeric validator and memory diagnostic each pass 11
focused Python tests. No existing correctness tolerance or 8-GiB guard changed.

The Cargo-selected binary is frozen durably outside the build target at
`/media/jmicko/faststore/kuno-development/public-native-artifacts/73ce13b2b9fdac27cb85b655bc5543960b9decafafb326900d9656ee931a53a2/kuno-text-lab`.
Its SHA256 is the parent directory name; build snapshot SHA256 is
`79e0f766678c2fd80b0d9826d78017d1992c783026b2b8295010d94225b04a29`.
The missing `/tmp` copies were not treated as live artifacts. A durable older
`110b9926...` release was subsequently found, but the current tests use the new
binary and its actual Cargo identity.

Both cards complete all eight fresh scalar/native owners, full owning maps,
audit/erasure and ordinary family acceptance at position0/count1, protected by
8 GiB, swap0 and core0. All numeric validators and worker units exit0. See
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-device-20261006.json`.
The first Intel launch failed before GPU work because systemd starts in home and
the wrapper path was relative; terminal exit2 is retained. The corrected launch
uses absolute wrapper and frozen executable paths.

| Boundary | Intel bit mismatches / maximum absolute | NVIDIA bit mismatches / maximum absolute |
| --- | --- | --- |
| layer0 combined PLE input | 750 / 7.45e-9 | 0 / 0 |
| projected KEY | 0 / 0 | 0 / 0 |
| projected VALUE | 0 / 0 | 0 / 0 |
| injected residual | 6,579 / 1.49e-8 | 4,185 / 5.96e-8 |

KEY/VALUE depend on separate per-layer token embeddings, not the differing
residual. PLE has known scalar CPU-f64 versus native GPU-f32 topology differences.
Those cannot explain Intel's already differing input. Same nominal f32 schedules
in distinct compiled upstream shaders permit contraction/codegen differences;
source inspection alone does not identify their first offender. These results
locate differences only in this position0/count1 fixture. Whole-model bitwise
acceptance and faster-than-Ollama claims remain false; no arithmetic changed.

## Current host-limit diagnostic

The dedicated public lane fixes one actual executable and fixture, changes only
its declared user cgroup's MemoryMax 8/52/52/8 GiB, keeps swap/core0 and all owning
cleanup/cancellation gates, and records memory/refault/pressure/process-I/O
counters. It does not flush global caches or change production limits. Its
strict 8-GiB correctness checker remains separate. No guard output is fabricated.

The completed Intel experiment is
`kuno-public-performance-intel-20261006-v1.service`, with durable records under
`/media/jmicko/faststore/kuno-development/native-performance-intel-20261006-v1`.
The unit finishes in 1,046.203 seconds, swap peak0, inactive/dead/exit0. All four
model/public-output/full-cleanup/cancellation checks pass, with no cgroup OOM.
Exact identities, model receipts, bounded cgroup records and telemetry hashes are
in `apps/kuno/benchmarks/text/receipts/rust-native-host-limit-intel-20261006.json`.

| Sample / cap | Prefill seconds | 127-step decode seconds | Decode tokens/s | Sampled child disk reads GiB |
| --- | --- | --- | --- | --- |
| 1 / 8 GiB | 91.199 | 128.939 | 0.985 | 123.35 |
| 2 / 52 GiB | 72.232 | 80.438 | 1.579 | 35.46 |
| 3 / 52 GiB | 43.520 | 63.189 | 2.010 | 5.79 |
| 4 / 8 GiB | 91.992 | 130.799 | 0.971 | 126.54 |

The 8-GiB runs repeatedly hit the cgroup limit (58,052/62,003 `max` events),
while both 52-GiB runs have zero such events. Warm weight-read times fall from
65.414/66.696 seconds to 17.996/5.063; upload times stay 7.560–8.432 seconds.
The limit materially biases this laboratory through model file-cache reclaim;
it is not an isolated GPU compute cost. The 52→8 transition also reclaims about
35 GiB before the last child. Naturally changing caches explain some variation
within a cap, so this is not a pure cache-controlled cap-only estimate.

No global caches were flushed. Each child starts a fresh owner; sampled disk
reads and cgroup deltas include later cancellation/fresh replay/audits, outside
the internal decode interval. Per-child current peaks are sampled; unit
`memory.peak` is cumulative. NVIDIA's host-limit experiment is not started.
Matched Ollama production throughput and a true second hot full request remain
unaccepted. Read `apps/kuno/benchmarks/text/performance-lane.md`.

At checkpoint, no public Kuno laboratory unit or native engine child is running,
Ollama `/api/ps` is empty, and NVIDIA has returned to desktop-only baseline.
Main `codex/image-flow-refinements` stays clean and unchanged. No installation,
production limit/configuration, public contract, frontend or release changed.

## Practical next steps after the user resumes

1. Establish basic calculation speed against actual Ollama/ggml before advanced
   expert scheduling. The user explicitly wants this baseline: same resident
   original weights, actual matrix shapes, fixed public inputs, precision and
   batch counts on the same device, with matched CPU workers. Existing operator
   timing wins compare Rust variants only; CPU physical acceptance makes no
   speed claim. Separate completed operator timings, full-layer coordination,
   and whole-model matched GPU/CPU placement; do not subtract nested enqueue
   timers to infer a compute floor. See the later CLI/offload requirement in the
   coordination guide; it changes no installed inference behavior yet.
2. Review the completed Intel cgroup evidence, then obtain NVIDIA evidence under
   the same protocol. All four declared Intel cases have finished; do not replay
   them merely because an earlier observation timed out.
3. The existing complete native `prefill_chunk(&[token], LastLogits)` can continue
   decode after native prefill on the same owner, retaining GPU-f32 PLE. A strict
   default-off public-lab flag is a small next experiment. Require native MAX8,
   owned transient I/O and complete audits; record observed native/scalar decode
   counts. Once native decode starts, never insert a scalar step then switch back:
   scalar advances the model but not the native PLE workspace cursor.
4. Keep the independent scalar/higher-precision oracle. Both-card complete
   fixtures, all-family cancellation/recovery/full cleanup and score comparisons
   precede an old/new timing comparison. Matching the same native implementation
   against itself is not independent numeric acceptance. MAX8 scratch/bridges
   at count1 can be slower; no benefit is established by source inspection.
5. Joint CPU/GPU expert source remains integrated but physically unaccepted;
   preserve the earlier gates below. Long contexts/options/quality/private
   worker, scheduler/catalog/native-default and managed `ollama.` routing remain
   outstanding. No Vashti change, install, public API, release/reset/reboot is
   part of this checkpoint.

## Historical paused checkpoint, 2026-10-05


The user explicitly paused the goal and requested completion of current source
work only. No new GPU benchmark, installation, service/configuration change,
driver reset or reboot is part of this checkpoint. The goal is **paused**, not
complete. No faster-than-Ollama claim or production native-default dispatch is
accepted. Required Vashti changes: **none**.

## Integrated source

Use managed worktree `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
branch `codex/kuno-native-text`. The source checkpoint is `e386c79`:

| Root commit | Author commit | Change |
| --- | --- | --- |
| `5f2ed09` | `6cd9a246` | Read-only fixed-public MoE owner inventory |
| `b2b69cc` | `c16359f5` | Real-read CPU/GPU expert operator laboratory |
| `1ef151f` | `dc915f6b` | Borrow the existing caller graph; avoid duplicate binding |
| `e386c79` | `db3a18f7` | Complete measured work before independent calibration references |

Root completed 391 host/offline tests (zero failures or ignored tests), strict
all-target Clippy, native-lab formatting and `git diff --check`. These are source
checks, not physical acceptance. Exact source hashes, budgets and check evidence
are in `apps/kuno/benchmarks/text/receipts/rust-cpu-gpu-calibration-source-20261005.json`.
The author separately completed 384 release host/offline tests, strict Clippy,
formatting and a normal release build in its isolated branch. That unregistered
author binary is not a root device-tested or installed artifact. The root's
older frozen device artifacts retain their original source/build identities.
The focused independent read-only review of the final calibration-order change
found no material blocker: references follow completed measured consumers and
merge/audit; unchanged numerical gates precede local plan publication; retained
values remain within the charged reference and bridge envelopes.

Read the implemented section of [the calibration contract](kuno-native-text-cpu-gpu-calibration.md).
This operator is source-integrated but has no registered root CLI and has not
executed actual GPU/model payload work. No production model calls the new CPU
operator. Numeric acceptance precedes timing; the sealed local permit alone is
not both-card acceptance or an automatic routing policy.

## Existing physical evidence

- Both cards passed the complete fixed-public 75-prompt/128-output fixture under
  frozen binary `b1ba040f…`; all original output IDs and complete cleanup match.
  See the full Intel/NVIDIA receipts in the performance guide.
- All 44 actual context4/8 lifecycle cases completed under frozen `110b9926…`,
  including late error, actual PLE-ticket panic, cancellation after the first
  GDN commit, poisoned retry rejection and fresh recovery. Both worker units
  exit 0, with full audits and retirement=false.
- All 24 final greedy comparisons matched, but full scores differ: maximum
  absolute delta about 1.04e-5 on Intel and 8.70e-6 on NVIDIA. Both outer
  validators correctly exit 1; whole-model bitwise acceptance remains false.
  No tolerance was relaxed. Matching aggregate statistics do not prove that
  two native partitions produce bit-identical vectors.
- The packed Rust CPU operator passed 16 fixed-public actual original-weight
  AVX2 cases and portable comparisons. These checks do not establish all-ten
  CPU/GPU parity, joint overlap benefit or a connected hybrid model.

See [the performance guide](kuno-text-performance.md), especially
`rust-native-model-lifecycle-device-20261005.json`,
`rust-native-model-full-intel-20261005.json`,
`rust-native-model-full-nvidia-20261005.json` and
`rust-packed-cpu-experts-actual-20261005.json`. No device job is left running by
this checkpoint; do not resume already-terminal recorded sessions.

## Numerical investigation candidate

An isolated source candidate is frozen at
`2439d732966c066d39ef9ee745d208206d98e5ee` in
`/home/jmicko/.codex/worktrees/kuno-text-loader/vashti`, branch
`codex/kuno-native-numeric-boundaries`, based on `8621308`. It is not integrated
or registered on the root branch at this checkpoint. Five focused tests pass
(376 unrelated tests filtered), as do strict all-target Clippy, formatting and
diff checks. No full suite, release build or device test was run for this
candidate. Both its branch and the earlier failure-fixture branch are clean
and preserved. Root's focused source review found no material blocker; an
additional peer review was requested but has not returned. Exact candidate
provenance is recorded in
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-candidate-20261005.json`.

The intended default-unarmed collector compares actual existing layer0 combined
residual, PLE projected KEY, projected VALUE and injected residual, one fixed
public position0/count1 boundary per scalar/fresh-native pair. All eight complete
normal owners must close and audit before aggregate-only results are published.
Actual KEY width is 10,240; VALUE width is 2,560. No state import, math change,
new MAIN pages, user data or whole-model numerical tolerance is introduced.
Controlled diagnostic host storage is bounded at 610,304 bytes within its
independent 1-MiB envelope. Additional audited runtime is 1,114,112 bytes;
actual original worst bound plus this component is 65,204,224 bytes, below the
unchanged 67,108,864-byte hard cap. These source bounds do not establish physical
map/audit acceptance.

The existing scalar PLE tail uses CPU f64 accumulation while the native tail
uses the already-approved GPU f32 path. This is a candidate explanation for
small full-score differences, not proof of first divergence. Scalar/native
projection kernels use the same 64-lane schedule. Observe their actual input
and output boundaries before attributing differences to the PLE tail or shader
lowering.

## Resume gates

1. Review and validate the frozen numerical-boundary candidate, then register
   only its explicit protected public diagnostic. Run actual paired boundaries
   on both cards before accepting a cause or changing any arithmetic.
2. Register the corrected joint expert lab in an explicit protected Numeric-only
   entry using one caller graph/client/retirement owner. Start with a small
   declared subset; do not infer the whole matrix from it. Accept actual original
   Q5/Q8 expert stages, all-ten ownership/rank math, full maps/tails/faults and
   completed erasure on both cards before timing.
3. Correctness containment remains 8 GiB, swap disabled and core dumps disabled.
   Before speed comparisons, design a separate verified performance lane with
   matched actual engine memory containment and captured numeric cgroup counters.
   The original Ollama baseline used 52 GiB while current native correctness
   units use 8 GiB; a prior native 52-GiB receipt exceeded 8 GiB. Buffered file
   cache is cgroup-accounted. This is a plausible benchmark bias, not a proven
   cause: current runs ended before pressure/refault counters were captured.
   Low RSS and the unreliable 256-KiB systemd summary do not settle it. First
   quantify the same-binary 8/52-GiB effect, then compare engines sequentially
   under matched conditions. Do not drop global caches or weaken private limits.
4. Preserve the recorded cold definition: model unloaded, OS cache unflushed.
   Hot full-prefill comparisons require zero Ollama cached prompt tokens. The
   previous repeat reused 71 tokens and is not a fresh-native-prefill comparator.
   Compile outside measured runs and preserve actual binary/source/settings.
   Follow the [current measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06):
   use 10 tokens/s as the practical target, without further Ollama baseline runs.
   The original 7.6–8.0 tokens/s receipt is not Ollama's ceiling. Preserve native
   timing/identity/cache-state evidence and include observed stalls rather than
   claiming a new matched-engine speedup from the working target alone.
5. Only then connect measured CPU/GPU splits into the complete Rust model and
   establish quality/options/tools/long contexts/private worker acceptance,
   scheduler integration and native-default/managed `ollama.` catalog routing.

System memory cache/accounting references:
[kernel cgroup memory documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)
and [kernel process memory documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html).
Hardware availability and sudo should be checked again after the user resumes;
no credentials belong in this document or inference diagnostics.

### Pending packed runtime audit experiment

The batched-INFO laboratory will pack full post-erasure source copies into
fewer bounded staging owners. Every original full source byte, new full stage
and padding remains charged, mapped/inspected and finally erased under the
original caps. No inference math, original MAIN proof or installed path changes.
Sourceb5fafef/artifact2d8f4a9c passes all128 IDs and original owning/fresh/
lifecycle gates:22697ms decode (5.5955 tokens/s),16885ms prefill. Warm
readbytes10211328, uploaded bytes0; CPU total3528544550ns. Final proof
interval falls from4172079892ns to3845052045ns in these single samples.
Physical B70 returns50688000B.51 runtime host tests plus two isolated logger
controls,492 native tests/both shaders,33 Python controls and strict Clippy
pass. This does not meet10 tokens/s or establish a paired speed comparison.

The next candidate shares the same three typed GPU error scopes across each
already-owned single-stream scheduler batch. It adds no deferred task queue or
private owner and preserves all before-acceptance errors/fences. Source/physical
acceptance source4495d8f/artifactb64b9cc5 passes all original128 IDs and
owning/fresh/lifecycle gates:22665ms decode (5.6034 tokens/s),16890ms
prefill. This is essentially the same observed speed as5.5955; no meaningful
speed benefit is established.53 runtime tests/two logger controls,492 native
tests/both shaders,33 Python controls and strict Clippy pass. Physical memory
returns50688000B.

The next diagnostic replaces ambiguous custom `Kernel` profiler names with
static public operator/packing families. No arithmetic, identity, owner or
speed-selection change. First-following profiling remains excluded from speed
selection.

### Actual metadata batches and updated host profile

Source a3248d6 / frozen artifact d461f52b passes the original128-ID full
fixture, fresh continuation, poison and owning/lifecycle gates:22260ms decode
(5.7053 tokens/s),19126ms prefill. Metadata boundary calls fall from635131
to159643; original actual extents, aliases and shard/dtype checks remain.
492 native tests and two offline shader suites,33 Python controls, strict
all-target Clippy pass. Physical B70 memory returns50688000 bytes.

Static family labels source599b523 / artifact2c4f5195 also pass the short
fixture and original owning/lifecycle gates. First-following diagnostic4309
kernels total59.76967ms; dense Q4K13.299167ms, clears6.44854ms, grouped
RMS5.944636ms, GDN4.477242ms, dense Q6K3.967663ms. Profiling forces
extra flushes and differs from the long fixture; it is excluded from speed
selection and does not establish an unprofiled GPU execution floor.

A separate twelve-second root CPU sample of the public v25b fixture keeps raw
perf stacks in private tmpfs, removes them, and saves only static symbols/sample
aggregates. It points to CPU packed expert math, allocator cleanup, and buffer
initialization/inspection. Cycle shares are not elapsed-time shares; the entire
profiled sample is excluded from speed selection even though its output matches.
Next candidate compacts exclusive allocation pages in place, preserving the
original free aging, release conditions and every retained descriptor identity.

Source fd1b864 / artifact22309cf4 accepts the unchanged full128-ID and
owning/fresh/lifecycle fixture with21648ms decode (5.8666 tokens/s),15544ms
prefill. Physical memory returns50688000 bytes; no new classified Xe faults
as of2026-10-08T12:11:53.071467+00:00. Three actual imported allocator host
controls,53 original runtime tests/two logger controls,492 native tests/two
shader suites and strict Clippy pass. One sample;10 tokens/s remains unmet.
The next candidate uses word-sized full-span zero inspection. A standalone
public16-MiB zero-array scan,100 passes, measures754881065ns for byte checks
and46726019ns for safe8-byte words; this is a CPU microbenchmark only, not
whole-model speed acceptance. The original spans/tails/checks remain.

Source998fdcd / frozen artifact8444d9db completes original128 IDs and all
owning/fresh/lifecycle gates:19777ms decode (6.4216 tokens/s),15504ms prefill.
The completed-final-proof interval is2756719001ns/127 steps. Physical B70
returns50688000 bytes.55 original runtime/zero-scan tests,three actual allocator
controls/two logger controls,492 native tests/two shaders and strict Clippy pass.
Next candidate groups consecutive full scratch shader clears (explicit
`--compute-scratch-clear`) without ending the existing compute pass. Final
erasure, every full byte and all original completion/ownership checks remain.

Sourcea6cac90 / frozen artifact6ad5e5ed completes the original full128 IDs
and owning/fresh/lifecycle gates:18654ms decode (6.8082 tokens/s),15494ms
prefill. Independent groups2/3/4 full-owner/padding/unselected controls pass.
494 native tests/two original shader suites,34 Python controls and strict
Clippy pass. Next candidate narrows only active-row dispatch grids; all shader
math and full MAX8 owning allocations remain.10 tokens/s is still unmet.

Source b6049a3/artifactbb1c45f637c2d64926267bd943a5e96a31234b8890d7a96b5fd8b17c99fe27b1
limits GR/GDN/QSA and status grids to their validated active rows. The full
original75/128 fixture and fresh/poison/full-owning/lifecycle gates pass with
17948ms decode (7.0760 tokens/s),17044ms prefill. Physical B70 returns50688000B.
494 native tests, both original offline shader suites and strict Clippy pass.
The next candidate groups statuses at their existing ready-before-reuse
boundaries, preserving active words, sticky faults and complete original gates;
physical acceptance is pending. No installed service/API/frontend changes.

Grouped status source9fbda89/artifact3c7346fb119a96ccd631e2a96eec2c4a199d9fc5f46ffd6250311471ba310044
passes the full original75/128 output and owning/fresh/lifecycle gates:
17724ms decode (7.1654 tokens/s),15279ms prefill,18415 grouped status calls.
495 native tests, both offline shader suites,35 Python controls and strict
Clippy pass. B70 returns50688000B; classified Xe faults remain zero on kernel38.
The next candidate ports GR RMS to max-scaled vector/subgroup reductions;
source gates pass496 native tests plus both shaders,35 Python controls and
strict Clippy; complete physical acceptance is pending. API/installed inference
remain unchanged. The10-token/s target remains unmet.

Shared GR source901f35b/artifact809aace6a5f6d86a5d6d2effa118c685f46d1f9a074f610ae8309fbe6f31739e
passes independent actual-shape/count/invalid/overflow/padding probes and all
original75/128 output, fresh, full-owning and lifecycle gates. Decode17805ms
(7.1328 tokens/s), prefill15240ms,12319 warm shared-GR calls; no speed gain over
v30 in this single sample. B70 returns50688000B. The next candidate batches
three completed PLE diagnostic readbacks, retaining every original validation
and enforcing the simultaneous peak against the unchanged4MiB host cap.
496 native tests, both shaders and strict Clippy pass; physical acceptance
pending. Installed inference and public contracts remain unchanged.

PLE batch source7a2dab6/artifact23c4ccf7ab5775995f449f1f4ac40f167b93211324110091bc3f807f5aafad94
passes all original75/128 output, fresh, full-owning and lifecycle gates:
17745ms decode (7.1569 tokens/s),15431ms prefill. No measurable gain in this
single sample; B70 returns50688000B. The same source recompiled for original
compiled two-pass CPU Q8 (artifact3885e2a9aa2ef3e2e6f001ea85753876d06abe7aff03e3d3bac4b9d02fb2c6c0)
also passes all original gates, but18358ms decode (6.9179 tokens/s),16022ms
prefill is slower. Its CPU operator total rises3006→3686ms against v30 and warm
disk reads remain zero; retain original float FMA as the control. Both feature
sets pass496 native tests, both shader suites and strict Clippy. The next
candidate factors original Q4_K vector dots for dense/expert kernels; independent
rows and full original device acceptance are pending. No API/installed changes.

Factored Q4_K source1f0e078/artifacte71731c34dffd23e94001a21f820f00e7bd5bc9bfbea6b3cfe5f8ba706269a39
passes all original75/128 output, independent rows and owning/fresh/lifecycle
gates, but18110ms decode (7.0127 tokens/s) does not improve v30. Eight workers
on the same frozen artifact also pass all original gates with17943ms decode
(7.0780 tokens/s). CPU user/system ticks fall4026/612→2726/417 against v30,
but CPU operator wall total rises3006→3256ms; retain sixteen workers. Both
physical postflights return50688000B. The next source restores original Q4_K
arithmetic and removes count1 hybrid inactive merge workgroups/repeated
immutable owner-header reads. Source gates pass496 native tests, both shader
suites and strict Clippy; complete device acceptance and a new bounded CPU
profile are pending.10 tokens/s remains unmet; no installation/API/UI changes.

Hybrid fold source5ee51a5/artifact21dba701f30a3a90731f8862b141a6db580dc754bfe507505458c0490939ed14
passes all original75/128 IDs and owning/fresh/lifecycle gates:18126ms decode
(7.0065 tokens/s). No gain; B70 returns50688000B. Best remains v30 at7.1654.
The next candidate replaces byte-at-a-time heap erasure with full-capacity
volatile word stores and directly copies into preallocated guarded owners.
496 native tests/two shaders,57 runtime controls/three actual allocator controls/
two logger controls and strict all-target Clippy pass. Exact unchanged pins and
regenerated23-file provenance patch3f6c6be96aa16958293769027304cf27d4453c6ac3a77edd4415793b1257ad93.
Complete physical acceptance and a refreshed bounded host profile are pending.

Heap erasure source13f591c/artifact54526913e4c600e32a828530de408fe6c99a1e466e5bc80e7a3f4b26c3ecfe76
passes all original gates with17730ms decode (7.1630 tokens/s),15579ms prefill.
B70 returns50688000B. Two bounded host profiles, v36b(all threads) and v36c(main
thread), pass original complete model/lifecycle acceptance and are excluded from
speed selection. Raw stacks are root-owned tmpfs and removed; only static
symbol/sample aggregates remain. Main-thread clearing/copying/allocation remains
visible; cycle shares do not establish elapsed-time attribution.

First-following GPU profile v36d rejects with retired private I/O during its
profiling path; no acceptance or timing selected. Physical B70 returns50688000B
and same-boot classified Xe faults remain zero at2026-10-08T14:00:58.573310+00:00.
Do not bypass its original cap or relax ownership gates to obtain a profile.
The next candidate explicitly reuses only exact-size inspected-zero Readback
owners inside the original I/O cap. Source and complete device acceptance pending.

Readback reuse candidate source gates pass496 native tests/two original shaders,
61 runtime host controls/three actual allocator controls/two logger controls,
36 Python count1 controls and strict all-target Clippy. Provenance23-file patch
fe6c6cfa60fa34f2b784dfb0292a15103f4cb61c608949201e022fec639eccc0;
original four upstream pins/licenses and both locks remain unchanged. Device
acceptance is pending. No source edits or compiler overlap during timings.

Readback reuse source3f16450/artifact0162e03533dc58025dbf4ca146b9552a32590490bf3493473bda565705937155
passes original128 IDs and all fresh/poison/full-owning/lifecycle gates with
17491ms decode (7.2609 tokens/s),48883ms prefill and1900 warm actual reuses.
B70 returns50688000B. The prefill regression is recorded, not attributed or
hidden;10 tokens/s remains unmet. Next candidate retains up to64 small owners
in the same8MiB cap, with explicit actual pointer storage plus owner header
limited to the original4KiB envelope before content. Device acceptance pending.

Small Readback pool source4d17149/artifactef05a2f88c95184639f18c1d45e7eec384f5c3b9055fd6b575918422d3083801
passes all original128-ID/owning/fresh/lifecycle gates with17643ms decode
(7.1983 tokens/s),15251ms prefill,7948 warm reuses. Source gates496 native/two
shaders,62 runtime/three allocator/two logger controls,36 Python controls and
strict Clippy pass. B70 returns50688000B. More reuse does not establish a speed
gain; the existing single best remains v37 at7.2609.

Next candidate uses native primary mapping only for Upload owners: full GPU
clearing twice and direct CPU READ mapping/full zero inspection between passes.
It removes repeated WRITE remaps and Upload verification stages while retaining
all original source bytes, both erase passes, error fences and failure retirement.
Exact capability and original complete-model acceptance are pending; never relax
the original owning gates or budgets. Installed execution remains unchanged.

Primary Upload source gates pass496 native/two shader suites,65 runtime/three
actual allocator/two logger controls,37 Python controls and strict all-target
Clippy. Regenerated23-file provenance patch
f46b89e75ba029cd435f3b0cd0a71715300c306fb4377b69dfa003414d8efd6a
retains original upstream archives/licenses and both locks. Device pending.

Primary Upload source5b4eceb/artifact83e4292a2994430bfe312f1f3c349a9a89fa2e08426672e18d72beca6acdf9cd
starts the correct capability/probes but its all-size clearing run is materially
slower and is deliberately stopped before full output/owning/lifecycle acceptance.
No receipt or speed selection exists. It is a fixed-public fixture; no private
user content was involved. Ordinary process stop returns B70 to50688000B without
GPU reset. Do not enable that candidate by default or represent it as accepted.

The next candidate uses original volatile Upload erasure and inline Readback
map completion on the compiled inline calling thread, preserving original
guarded leases, callbacks, failure retirement and all owning/copy/clear bounds.
Source and complete device acceptance are pending. Original async worker/API
behavior, installed inference and the10-token/s target are unchanged.

Inline Readback source gates pass496 native/two shaders,66 runtime/three actual
allocator/two logger controls,38 Python controls and strict all-target Clippy.
Regenerated23-file provenance a93143eaaa8f6f88155ad8323145214f26a7e51b3107e060ef2f8135a40978e9
retains original upstream pins/licenses and both locks. Device pending.

Inline Readback sourceb345f9c/artifact3e707443f0452d00f9b4bc24608efbad406a2fde1f24bc58acc0eb6d658ef709
passes unchanged original128 IDs, independent arithmetic/fresh/poison and full
1378-owner/818413568-byte normal cleanup/lifecycle gates. Following127 steps
take17047ms (7.4499912 tokens/s), prefill48856ms,9525 warm inline map batches.
Physical B70 returns50688000B and same-boot Xe classifications remain zero.
This is the best single sample so far;10 tokens/s remains unmet.

The next candidate partitions GDN key reductions across four lanes. Arithmetic
summation order changes and complete original acceptance is pending. Installed
execution and all full-owner/cleanup gates remain unchanged.

Partitioned GDN sourceaa3681c/artifactb2efc9e3be1d353687059dbe1a77fb19eb9ff231f8e212063929a2f7791f5f26
passes original128 IDs, independent scalar numeric/invalid-column probes and
all owning/fresh/poison/lifecycle gates:17079ms decode (7.4360 tokens/s),15463ms
prefill. B70 returns50688000B. No measured overall gain; current best remains
v40 at7.4499912.497 native tests/two shaders and strict Clippy pass.

The next candidate combines two original expert-bank shards into two total
projection dispatches rather than four, retaining original packed arithmetic,
full physical proofs and all cleanup gates. Source/device acceptance pending.

Paired original banks source07308b8/artifact6351c8bdf632c9db5ee3b536995ecee41b0312fffe5ad9ddabc8a9550062c401
passes unchanged original128 IDs and all owning/fresh/poison/lifecycle gates.
Following127 steps16933ms (7.5001476 tokens/s), prefill15222ms. B70 returns50688000B.
Single small improvement;10 tokens/s remains unmet. Expanded offline paired
shaders,497 native tests/two other shaders and strict Clippy pass.

Next candidate is separately compiled joined work-stealing CPU row scheduling.
Original arithmetic and all failure/cleanup gates remain; source/device pending.

Work-stealing source gates pass497 native tests/two shaders,128 core controls
(three ignored) with both default broadcast and the separate work-stealing
feature,39 Python runner rejection controls and strict all-target Clippy.
Default and selected core behavior preserve exact original-format bits, bounded
private capacities, joined panic/error/cancel cleanup, erased row headers,
fixed physical worker topology and parent affinity. Complete device pending.

Work-stealing sourceceb88d9/artifactfd3d36bcc38f7cf978a74666e2e132751cf9a7698685f754065b4759b1e2148a
passes all original128 IDs and owning/fresh/poison/lifecycle gates:16410ms decode
(7.7391834 tokens/s),15111ms prefill. CPU phase total2614583418ns, down895490110ns,
gate/up1342271873ns. B70 returns50688000B. Single improved sample;10 remains
unmet. Next candidate shares original Q4 activations in bounded workgroup
memory, retaining exact arithmetic. Complete source/device acceptance pending.

Q4 activation-sharing source gates pass497 native tests/two shaders and strict
all-target Clippy on the separately selected work-stealing build. Shared-input
load spans are runtime-checked and unchanged original row/guard probes remain
required on device. Whole-model device acceptance is pending.

Q4 sharing source7c40189/artifact9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
passes original128 IDs and all owning/fresh/poison/lifecycle gates:16247ms decode,
15464ms prefill. B70 returns50688000B. Next candidate packs four fixed16-lane
Q4 rows into64 invocations, eliminating unused trailing workgroups independently
of the native subgroup width. Reduction order changes; original full acceptance
remains mandatory. Source/device pending;10 tokens/s remains unmet.

Fixed16 Q4 source gates pass498 native controls/two shaders,39 Python controls
and strict all-target Clippy. Tail enumeration covers both64/128 threads and
16/32 native subgroups; every inactive owner index is read exactly once and
active writers are excluded. Fixed16 shuffle peers remain in their own logical
row; shaders validate at256/4096/8192 columns. Device acceptance pending.

Fixed16 Q4 source6a32c25/artifact9a035f53ccf3f35082a1dc6d9ae07f04895f730019c87ab16d57e4d7adb45b1a
passes original128 IDs and owning/fresh/poison/lifecycle gates but is slower:
16700ms decode,16952ms prefill. B70 returns50688000B. Withdraw that geometry
from selection and restore accepted128-lane Q4 source with activation sharing.
Do not assume native subgroup choice or attribute a measured bottleneck from
subgroup bounds alone. Next candidate dispatches only original GPU subset ranks
in shared MoE, retaining private transient counts and all original bounds.
Source/device acceptance pending. Best remains v44 at7.8168277 tokens/s.


### Overnight run paused: 2026-10-08 final checkpoint

User explicitly paused the goal for a debrief. No further optimization runs are
scheduled. The last candidate, active GPU-only MoE dispatch source1a0179b /
frozen02da13e947e1003d365532715a21b71fbcf4a732676b0358b0bc0e006e15ec6b,
passes original128 generated IDs, independent guards, fresh continuation,
poison recovery, full owning erasure and protected lifecycle completion.
127 following steps16616ms (7.6432354 tokens/s), prefill15772ms. It does not
improve the best sample; withdraw it alongside fixed16 Q4. Selected engine source
is restored exactly to7c40189 across native-text and core text code.

Best remains v44:127 following steps16247ms (7.8168277 tokens/s), prefill15464ms,
source7c401893b481d006067d502cc19274b40bc7536f, frozen artifact
9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2.
The first accepted overnight long sample was3.3781088 tokens/s. These are single
fixed-public samples with different selected placement/runtime options, not a
statistical comparison or arbitrary-prompt/long-context/soak acceptance. Target
10 remains unmet. The aggregate now records51 accepted fixtures, including
explicitly excluded diagnostic profiles; stopped/rejected experiments remain
excluded from speed selection.

Selected source gates:497 native controls/two shader suites;128 core controls
(three ignored) in default and separately compiled work-stealing modes;66
runtime host controls, three allocator controls and two logger controls;
39 Python runner controls and strict Clippy. Last candidate also passes complete
protected device acceptance. Restored source matches the already accepted v44
engine exactly; no extra physical run is needed to test identical source.

After all benchmark processes exited, physical B70 usage returned50688000B.
Root Xe classification at2026-10-08T15:45:04.155894+00:00 on boot
665344a9-0e39-4e45-90de-465fa11df0ca / kernel7.0.0-38-generic reports zero
wedged, timeout, GT-reset and memory-fault messages. Public admission resumed:
active=false, waiting=0, paused=false, recovery_required=false. Kuno11434,
direct Ollama11435 and Vashti7771 return HTTP200. No GPU reset or reboot used.
Installed binaries/configuration/model blob/API/catalog/default routing and
Vashti application remain unchanged. No release performed.

Remaining work after a future explicit resume: reduce measured routing fences,
GPU execution/dispatch costs and CPU expert row overhead. The accepted warm
v44 run has zero expert weight uploads/reads, so expert transfer is not the
current dominant measured cost on this fixture. Host-phase timings include
synchronization and overlap and must not be read as pure GPU compute times.
Keep original privacy/correctness/lifecycle bounds; no API or frontend work is
needed for this internal optimization checkpoint.


### Astra request-buffer reuse candidate, 2026-10-08

User authorized another concentrated implementation/benchmark pass toward10
 tokens/s. Owner remains the isolated Kuno checkout. db9f13c adds exact-math
inactive-workgroup Q4 activation-load skipping;324bc92 adds an explicit
request-scoped runtime I/O pool. Original maps/fences/finite checks remain;
private buffers stay owned/charged through the request and receive the original
full two-pass CPU-inspected audit at request close/error. The count1 contract
above is explicitly revised for this request lifetime. No API/frontend changes.

The attempted v44 first-token GPU profile failed the existing private-I/O
bounded runtime gate and is NOT an accepted speed sample. It left B7050688000B
and same-boot Xe classifications zero at2026-10-08T16:02:16.609881+00:00.
No cap was raised. Source work proceeded from accepted v44.

Integrated source tests pass498 native controls/two shader suites,70 runtime
host controls/three allocator/two logger controls and41 Python runner controls.
Runtime strict all-target Clippy passes; integrated Clippy/device acceptance
are pending. Benchmark generation duration will include the final request audit,
with its separate duration and actual reuse counters. Public admission is
paused only for this exclusive benchmark block; restore it at the stopping point.

The first physical same-build control/reuse runs from63a68bb both pass original
128 IDs and full owning/lifecycle acceptance:16108/16357ms respectively. PLE's
independent sanitizer erased the pool each token;126 reuses were insufficient
to exercise the intended optimization. bb678d9 joins that completed boundary
to the request owner. Its corrected run passes every gate but regresses to
23143ms with84673 reuses,381 checkpoints and28ms closing audit. Final proofs
fall2.816→1.179s but routing fences rise3.854→10.396s; this is not selected.
All completed runs leave B7050688000B. Numeric source/artifact/phase receipts
are in`next-request-io-reuse-20261008.json`.

The next candidate a2f1562 batches reused Uniform shader-read→COPY_DST
transitions at a completed checkpoint, removes repeated linear owner searches,
and records scoped readback setup/flush/map-wait aggregates. Original ownership,
full audit and numeric acceptance stay mandatory. Source gates pass498 native
controls/two shader suites,74 runtime host/three allocator/two logger controls;
runtime strict Clippy passes. Physical performance remains pending.


The indexed transition candidate passes original128 IDs but takes22214ms.
INFO interning plus Readback COPY_DST transitions takes21103ms; same-build
reuse-OFF control remains16130ms. Actual full-span Readback clears at the
completed recycle (e9d8032) remove the regression and establish9.3348034 tokens/s:
127 steps13605ms including request-close audit; prefill15314ms. Original128 IDs,
independent numeric/finite guards, fresh continuation, actual cancellation,
poison recovery, owning erasure and protected lifecycle all pass. Map/callback
wait falls9.331→2.454s; this is measured, while the proposed GPU memory-coherence
explanation is an inference. Every completed candidate leaves B7050688000B.
No GPU reset or driver/clock change is involved.

The next candidate preserves original completed checks but uses accept-only
logits/PLE boundaries and one full recycle after the final fault per token.
It also increases exact immutable INFO entries512→1024 under unchanged64KiB
keys and256KiB combined metadata. New runtime API is internal to the isolated
benchmark, not a gateway or frontend contract. Runtime82+3 allocator+2 logger
controls pass with strict Clippy; integrated498 controls/two shader suites and
41 Python controls pass. Physical speed acceptance is pending.


The combined accept-only/INFO1024 candidate3e367ef passes all original device
gates but regresses to16958ms (final receipt authoritative), with5.617s mapped
read waits. It is not selected. Next isolate the same accept-only boundary with
the previously accepted512-entry cache. No new controls or admission limits are
changed. All-family publication and request-close audits remain mandatory.


The512-entry isolated one-recycle candidate8151cf6 also passes every original
gate but regresses (see numeric aggregate). Restore the previously accepted
three completed recycles and512-entry INFO cache. The unused internal
accept-only API does not alter selected execution. Next investigate bounded
nonblocking polling before the original inline map wait, retaining the same
30-second deadline, callbacks and all error acceptance.


Candidate406cf80+a5a0d34 combines bounded bucket lookup with up-to1ms active
native polling only in request-scoped inline readback. Host integration passes
88 unit+3 allocator+2 logger controls. Core math, device placement, full owning
bounds, original map callbacks, completed errors and terminal audits are
unchanged. Normal/default paths retain blocking waits. Integrated native gates
and the protected full device fixture are the remaining acceptance steps.


The combined candidate5df607c/frozenba9844cafee773457a9ac1efa5f5db000fcda2bb22b12b1a72131f9a38dde589
passes twice:127 steps13277/13234ms (about9.57/9.60 tokens/s), with original128
IDs, numeric/finite/owning/fresh/poison/cancellation/lifecycle acceptance intact.
Physical B70 returns50688000B after both runs. One final bounded candidate
removes the SIMD worker's unused decoded-row allocation and constructor GPU
clears that are completely overwritten by full-span Readback copies. Tests and
physical acceptance of those changes remain pending; target10 is still unmet.


Source708f5fe removes only fully overwritten constructor clears;9b51ace makes
CPU decoded scratch lazy for Portable. Runtime88+3 allocator+2 logger controls
and strictClippy pass. Core131 controls (three intentional ignored) pass in
both default and cpu-work-stealing-lab modes, with strict all-targetClippy in
both. The original conservative private budget remains. Integrated498/native
shader validation, frozen normal build and protected physical run are next.


The copy/lazy candidate91cd57e passes all original device gates at13280ms,
with no measurable additional gain over the repeated poll/index build. Its
changes remain byte/arithmetic-equivalent; do not claim a throughput win for
them. The next candidate reduces actual request-scoped zero-tail uploads while
preserving full input/destination ownership and all audits. Warm command-byte
counters are added to distinguish genuine transfer savings from timing noise.
No public inference API or frontend work is involved.


### Astra pass complete: selected source and restored host, 2026-10-08

This final checkpoint supersedes the candidate/pending statements above.
Selected source is `79739d174af41990258a9571c1a8e720d77edc8a`, frozen normal
locked Cargo release artifact
`0df72bd223c7afda807f9fed0793572153bbe40db8ac3fd77ab621f01590d9e6`, with
`direct-device,cpu-fma,cpu-work-stealing`. Its build directory is
`/media/jmicko/faststore/kuno-development/next-offload-build-20261008-v47-zero-tail`.
Original Ollama weights remain read-only and unchanged.

| Selected fixed-public run | Following steps | Decode including closing audit | Decode tokens/s | Prefill |
| --- | ---: | ---: | ---: | ---: |
| `cpu-144-zero-tail-long-v47` | 127 | 13391 ms | 9.4839818 | 15550 ms |
| `cpu-144-zero-tail-repeat-long-v47` | 127 | 13298 ms | 9.5503083 | 15305 ms |

This is about 21–22% faster than the preceding accepted 7.8168277-token/s
sample, but still below the user-approved 10-token/s reference. These two short
fixed-public samples do not establish arbitrary prompt, long-context, quality,
soak or matched-Ollama performance. All original 128 generated IDs, independent
arithmetic/finiteness/guard checks, fresh continuation, actual mid-token
cancellation, poison recovery, full owned-memory erasure and protected lifecycle
checks pass. Timings include the terminal request audit. Both runs enforce
swap zero and core dumps disabled, without persisting real user content.

The main improvement is request-owned I/O reuse with actual completed Readback
clears (`e9d8032`), followed by bounded INFO buckets (`406cf80`) and short bounded
active polling (`a5a0d34`). Selected execution retains all three completed
recycle checkpoints and the 512-entry INFO limit; the correct but slower
one-recycle experiments are excluded. `708f5fe` removes only fully overwritten
constructor clears; `9b51ace` avoids unused SIMD decoded-row scratch. Neither
of the last two has a separately established throughput gain.

`fbae6f7` plus measurement/validation `79739d1` retains request-scoped zero-tail
uploads at the user's explicit preference. Both selected runs encode original
1747845120 bytes as 229218304 copied bytes plus 1518626816 GPU-cleared bytes:
86.8857% fewer upload-copy bytes. The same artifact repeated full correctness.
There is no established additional throughput gain over the immediately prior
13280-ms result. Counters measure encoded command extents, not hardware PCIe
traffic, mapped-store traffic or transfer/compute overlap. This optimization
starts after completed prefill; model loading and unscoped uploads still use
the original path. No startup improvement is claimed.

Final source gates: 498 native controls plus two one-test shader suites;
131 core controls with three intentionally ignored, in both default and
`cpu-work-stealing-lab` builds; 90 runtime host controls, three allocator and
two logger fixtures; 42 Python benchmark/receipt controls. Strict release
all-target Clippy passes for the native lab, runtime and both core feature
modes. Zero-tail host checks include 2048 byte-model cases, original spans,
offsets/padding, signed zero, NaN payloads, alignment and overflow rejection.
Independent source reviews found no blocker. Runtime provenance accounts for
23 modified vendor files and four unchanged published dependency pins/licenses;
composed patch SHA256 is
`e3bb92933bdd4d4cd62e027d0a9a954a3cc3a6f0bc717507158de9788afad386`.

The [numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-request-io-reuse-20261008.json)
records all 14 accepted physical fixtures, including candidates excluded from
speed selection, with exact source/artifact/raw-receipt/run-proof hashes and
numeric phases. Raw public receipts and lifecycle proofs are under
`/media/jmicko/faststore/kuno-development/next-offload-device-20261008`.

All benchmark jobs exited and physical B70 memory returned 50688000 bytes.
Root Xe classification at `2026-10-08T18:16:29.880861+00:00` on boot
`665344a9-0e39-4e45-90de-465fa11df0ca`, kernel `7.0.0-38-generic`, reports
zero wedged, timeout, GT-reset and memory-fault messages. Public admission is
restored: active=false, waiting=0, paused=false, recovery_required=false.
Kuno 11434, direct Ollama 11435 and Vashti 7771 return HTTP 200. No GPU reset,
unbind, reboot, driver or clock change was used. Installed binaries/settings,
model blob, public API/catalog/default routing and Vashti remain unchanged.
The goal stays paused; this explicitly requested pass is complete, with the
speed target still unmet. Required Vashti changes: none.

Next work should start from this selected engine and remaining measured routing/
readback and CPU expert costs, preserving complete output/privacy acceptance.
Warm expert-weight upload/read counters are zero for this fixture; this does
not establish that all transfer costs are irrelevant. Extending zero-tail
handling to loading would require separate correctness and startup measurements.


### Private startup uploads accepted, 2026-10-08

User requested extending the retained zero-tail optimization to loading and
comparing current Kuno with installed Ollama/llama.cpp again. Runtime change
`909ab3b` (reviewed agent source `80f058e`) and instrumentation/provenance
`3b701d4` extend byte-exact prefix/ordered-clear planning to every configured
private upload. Independent constructor/prefill completion and erasure remain;
request reuse is not broadened. Public original weight uploads remain on their
original path. Bounds, full destination ownership, source guards and stage
initialization/audits remain unchanged. No math or model artifact changes.

Frozen normal locked release source
`3b701d4d9bbeb794351502022b21c30734d501eb`, features
`direct-device,cpu-fma,cpu-work-stealing`, artifact SHA256
`26871ebd5f3518f2d8e8e0734dd171527edd6865d68f1e4208e2b55c49a11178`, completes
`cpu-144-startup-zero-tail-long-v47` with all original128 IDs, independent
numerical/finite/guard checks, fresh continuation, actual mid-token cancellation,
poison recovery, all five owning audits and protected lifecycle acceptance.
Swap remains zero and core dumps disabled. Bind1365ms, prefill15324ms,
127 following decode steps13237ms (9.59431895 tokens/s), including9ms closing
audit. This single sample does not establish a further throughput/startup win
against the preceding selected13391/13298ms samples;10 remains unmet.

| Encoded private upload phase | Original bytes | Copied bytes | GPU-cleared tail bytes |
| --- | ---: | ---: | ---: |
| Model bind/probes/constructor | 20643840 | 20643840 | 0 |
| Prefill | 58982400 | 14672064 | 44310336 |
| Warm decode | 1747845120 | 229218304 | 1518626816 |

Prefill private copy payload falls75.12%; bind has no saving in this fixture.
These are encoded command extents, not hardware PCIe measurements. Dense public
weight uploads are excluded; do not describe this as compressing the model.
Bind includes the lab's startup probes, and prefill includes lazy weight use;
neither is a cold-storage end-to-end startup benchmark.

Source gates:498 native controls plus two one-test shader suites;92 runtime
controls plus three allocator/two logger fixtures;43 Python runner controls;
strict release all-target Clippy for native and runtime. The common byte planner
passes2048 byte-model cases plus dense/empty/original-maximum/overflow/counter
separation cases. Parent source review found no blocker. Runtime composed patch
SHA256 `8d32c58ea7f78118e01e57773e4d9f93408d6d723851b18db6735fd7699ab457`;
23 modified vendor files and original dependency/lock/license pins remain.
Exact raw/run/Cargo/artifact hashes and numeric results are in
[the receipt](../apps/kuno/benchmarks/text/receipts/next-startup-zero-tail-20261008.json).

The [updated source comparison](kuno-native-text-count1-hybrid-plan.md#updated-ollamallamacpp-comparison-2026-10-08-research-only)
pins installed Ollama0.35.1 to llama.cpp b11232. First recommended experiment:
cache identical GPU binding objects within the existing request scope, measuring
hits and construction cost. Both backends already pool descriptor storage;
Kuno still rebuilds/validates each binding object, while llama retains descriptor
sets across graph evaluations but updates them per dispatch. Complete binding
memoization is a proposed Kuno optimization, not an established upstream feature
or measured speedup. Further candidates are register-resident recurrent state,
retesting refined CPU integer dots after the disk-fault fix, then overlapping
device-resident expert launch with CPU routing readback. No comparison candidate
was implemented; no new Ollama benchmark or arbitrary-prompt collection ran.

All benchmark processes exited and B70 returned50688000B. Root same-boot Xe
classification at2026-10-08T18:43:11.889831+00:00 reports zero wedged, timeout,
GT-reset and memory-fault messages on kernel7.0.0-38-generic, boot
665344a9-0e39-4e45-90de-465fa11df0ca. Admission resumed with active=false,
waiting=0, paused=false, recovery_required=false. Kuno11434, direct Ollama11435
and Vashti7771 return HTTP200. No reset/reboot or installed service change.
Goal remains paused. Required Vashti changes: none.
