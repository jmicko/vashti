# Native Kuno text engine: next-model handoff

## Current checkpoint: stopped after the combined Strata port test, October13

**Read [the current port checkpoint](kuno-strata-rust-port.md) first.** Work is
paused at the user's request after v221; no GPU jobs or sudo refresher remain.
Automated goal is paused; do not resume automatically. Both GPUs and normal
Kuno/Ollama/Vashti services are available. Fresh16:29:23UTC recovery is recorded
in `native-strata-port-final-pause-20261013.json` under the development base.

Worktree `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`, runtime source `1caf271922abd1dd7aaf3f3d48789e4467362571`.
Original accepted engine remains on `codex/kuno-native-text` at6792453. Canonical
Vashti and deployed services are unchanged; required Vashti changes:none.

Connected broad port includes one-plane IQ arithmetic, parallel GPU routing,
compact real argmax/carry, Q2-to-Q8 runtime dense conversion,106299-token Strata
proposal vocabulary, suffix/MTP policy and request-local two-way adaptive cache.
Newest source expands144 to192 experts/layer with exact persistent roots and
exchanges original VRAM/mapped-RAM cells directly on the GPU in one joined
transaction. Histories/maps remain transient; no output/model conversion is saved.

Latest complete accepted broad-port run is v215/code-Q2:8.458796 real timed
continuations/s through362 emitted tokens at EOS (512 cap). Cache adaptation
alone took6.529365052s for2492 swaps/5.099GB moved. v207/Q8/code200 had10.800044,
which is a different trajectory/scope.30tokens/s is unachieved.

v221192/direct-exchange candidate reaches cold prefill and host materialization,
then rejects a four-row DOWN launch due to two stale shard tables. Internal and
actual process cleanup both pass. d682e04 fixes those tables;1caf271 passes strict
port Clippy/fmt and original direct-device compilation. **No GPU rerun after the
fix**, and **direct GPU exchange has not executed successfully on hardware**.
Latest GPU artifact3e09fb6/SHA b6fc683d... is failed and must not be reused as the
selected implementation. Build a fresh artifact from current runtime source only
on explicit resume; the exact commands, receipts, prior defects and acceptance
sequence are in the port checkpoint.

Full original-engine1346/1ignored regression result is from f0ed07a, not current
source. Final regressions and representative code/prose acceptance remain.
Larger verifier, efficient prefix rollback, XMX prefill, compact KV, auto-sized
residency and general private-input/API integration are unfinished. User explicitly
wants the coupled Rust port, rather than more isolated exact-math leaf tuning.

Everything below is historical background for the preserved original engine;
its old source selection/resume queue is superseded by the checkpoint above.

## Previous accepted engine checkpoint — superseded as the current work queue

Updated **2026-10-13 06:54 UTC / 01:54 America/Chicago**. User requested finishing
this experiment and stopping until tomorrow. **Stopped; automated goal paused;
no benchmark/compiler jobs or queued GPU work.** This opening checkpoint
supersedes every historical queue or interrupted-debug direction below.

Kuno, Ollama and Vashti respond HTTP200. Admission is idle/unpaused with no
recovery latch, B70 actual used50,688,000B, Nvidia compute0, classified faults0,
unchanged boot/kernel. Final fresh protected receipt:
`native-next-final-stop-20261013.json` under the development base. No installed
binary/default, API, frontend, canonical checkout or release change. Required
Vashti changes: **none**. Native30tokens/s remains unachieved; normal speed
remains about **11 real continuation tokens/s**.

## Start here on explicit user resume

Root checkout `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/kuno-native-text`. Canonical `/media/jmicko/faststore/dev/vashti` remains
clean on `codex/image-flow-refinements`, `ae3e70d`; do not move this lab work there.
Read git status/log before editing. Preserve inherited untracked
`apps/kuno/benchmarks/text/__pycache__/`. Never broadly restore/stash another
model's changes to remove diagnostics.

**Current source selection is accepted original compaction.** All three
compile-time experiment selectors are false:
`frozen_gu_integer::four::FIRST_PLANE_PUBLIC_PROBE`,
`frozen_gu_integer::four::shuffle::PUBLIC_PROBE`, and
`frozen_down_integer::four::STAGED_PUBLIC_PROBE`. The alternate leaves and
checked parent wiring remain available. Their immutable frozen artifacts retain
the recorded true/false selections; do not confuse those with current source.

This pass repaired inherited compaction and tested three separate ideas. It
found **no meaningful normal-generation speed gain**. Do not rerun the same
neutral experiments or promote the short CPU13 timing as production speed.
The strongest next investigation is a representative **normal code200** target
profile and actual count4-vs-tail kernel execution credit. Current compact route
counters aggregate counts1..4, and compile-time flags alone do not prove which
leaf dominates normal work. v185 profiled a different CPU13 warm fixture; v197
improved that fixture by20% while normal code200 was unchanged. Establish the
real normal-run bottleneck before another large kernel rewrite. This profiling
extension has not been implemented or authorized to run automatically.

## Changed commits and acceptance

- Inherited `c42379b`, `f6a233a`, `0b72be5`: source compaction, including tails.
- `62594d0`: fixed actual allocator `Some(0)` full-owner trims, checked indirect
  views, count4 metadata, original count1..3 reset coverage, storage ordering,
  complete record auditing and independent DOWN policy credit.
- `fb7b420`: fixed an erroneous count4-only main guard in generic count1..3
  compact DOWN. **Accepted compaction base**, physically working on both gates.
- `fd9c319` (leaf `3bb2c20`), `fdd2732`: primary-plane count4 GU approximation
  and explicit immutable test selection. **Rejected** by unchanged warm full
  CPU13; `6cc52f5` disables it and records evidence. Never loosen tolerances.
- `7d01db5` (leaf `efb7930`), `56d8646`, `70fd470`, `b9ee1be`: exact4/2/1-tree
  subgroup GU/DOWN candidate, full subgroup-count coverage, parent selection,
  independent bit-tree tests. v196 numerical acceptance; normal speed neutral.
- `41b9d3d` (leaf `5a64362`), `d8b58b6`: exact DOWN activation staging,5760B SLM,
  bounded adapter admission and fixed-public selection. v197 source/full GPU
  gates pass; normal speed neutral. `9dc764d` leaves both selectors off.
- Final docs/receipts commit follows these source commits; inspect git log.

The inherited DIAG patch was inspected and saved as
`compaction-v192-inherited-diagnostics-20261012.patch` under the development
base, then removed only from its three inspected files. Do not resurrect it.
The old claim that protected count4+integer DOWN is impossible was false: the
warm four-lookahead label genuinely reaches it. The original stopped failure
and all follow-up corrections are resolved; do not repeat that debug queue.

## Exact results and immutable artifacts

Development base: `/media/jmicko/faststore/kuno-development`.
Aggregate evidence: `apps/kuno/benchmarks/text/receipts/native-strata-optimization-20261009.json`.
All completed GPU trials used public fixtures, swap/core-disabled cgroups,
B70 target and full Q8 MTP on B70; Nvidia was not a compute participant.

| Version | Frozen source | SHA256 | Result |
| --- | --- | --- | --- |
| v193 compaction | `62594d0` | `25a8751d5ae97d337c1e5fb828ce3d5fac443971fc640372d07e4d7213a68984` | Cold pass, warm owning-IO failure; generic DOWN guard fixed in `fb7b420`. |
| v194 compaction | `fb7b420` | `beeb3787b01d682c069e0740698ebb4bfa6aa8352452ff90b64ca8480bcd4c9f` | Full cold/warm CPU13 and normal code200 pass; **10.9062469565** continuations/s. |
| v195 primary-plane GU | `fdd2732` | `28e21d4107b56b200f8dc842fa24c634644a3c9c5e457f5c20245ade599e1ee7` | Cold/prefill pass, full warm carry fixture fails; no normal timing. |
| v196 exact-tree shuffle | `b9ee1be` | `81cd9d54491da08623ac03454476d6cc2985b5161fa21a97d160f7b9f954cabc` | Full cold/warm CPU13 and normal pass; **10.9551890060** continuations/s, neutral. |
| v197 staged DOWN | `d8b58b6` | `358f23af90229acbef58c00a90000169c86395120ce783985d186db47f67f802` | Full cold/warm CPU13 and normal pass; **10.8717765711** continuations/s, neutral. |

All three normal runs emit200 outputs/199timed continuations,52 rounds,
147accepted drafts and5recomputations. Elapsed: v19418.246423430s,
v19618.164907962s, v19718.304276095s. These are single-run comparisons,
not statistical claims. v182 original10.882673 was also effectively the same.
Normal trials were unprofiled with compilers idle. No independent full CPU200
reference exists; full CPU13 and real-repair gates are distinct from that.

Directory patterns under the base:
`native-next-v194-compaction-{build,warm-intel,code-intel}-20261012`,
`native-next-v195-first-plane-{build,warm-intel}-20261012`,
`native-next-v196-shuffle-{build,warm-intel,code-intel}-20261013`,
`native-next-v197-staged-down-{build,warm-intel,code-intel}-20261013`.
Each build has source-commit.txt, SHA256 and artifact manifest; each completed
trial has source verification, intel.json/log and host-before/after.json.
No private content or credentials are in the aggregate receipts.

Source validation at selected v197 `d8b58b6`: **1346 passed, one ignored**;
strict release direct-device Clippy `-D warnings`, fmt and whitespace pass.
Afterward only experiment selectors were restored to false, with formatting
and whitespace rechecked. The frozen v197 binary is the tested selected build;
no new installed binary was produced. All v197 device/owner cleanup and final
protected host recovery pass. Sudo initially expired during the usage-limit
interruption; user refreshed it, so the GPU trials are now completed, not pending.

## Why not infer more from the diagnostic

v185 on `aa90a95` is the first actual **CPU13 warm target** Device profile,
not the10.88 normal-speed run (that was v182 on `dab13e2`). Its4051 spans total
250.572596ms: GU105.174727ms, DOWN69.876932ms, combined69.86%; BF16UP2.612ms.
Profiling changes scheduling. Do not treat CPU enqueue waits as CPU-only work.
v197's short CPU13 warm elapsed fell from v1962.234336549s to1.778226038s, but
normal target verify time remained about13.1–13.25s and throughput about11.
Different routes/placement/context matter. Investigate that mismatch rather
than assuming another warm-fixture kernel gain will transfer.

Strata comparison is also not perfectly matched. Retained
`strata-comparison-evidence-20261009/Strata-strata-iq3_s.json.txt` explicitly
uses `--spec 4`, `--spec-min-p 0.5` and `draft_vocab=cjk`, with alternate Q2 MTP.
Pinned MIT Strata `fb58e0d`, `sycl/src/program/generate.cpp`2215–2220, keeps
MTP cap4 but can expand suffix-lookup verification to6 by default. The retained
27.3code receipt has200outputs/7336.5ms,142/172accepted drafts, but no actual
lookup/window-size counts. **Do not claim it actually used6-token windows or
that deeper MTP alone explains the gap.** Native6/8 extension and alternate
Q2/CJK draft feasibility were not completed this pass. Do not blindly widen
capacity constants; owners, routes, attention/carry and numerical gates matter.

Shelved: BF16 two-row `a9cff9e` source-only (small measured profile share),
paired32 Q6 head CPU13-accepted/default-off/normal-pending, integer16 head
normal-neutral. GU activation staging is a possible future analogue only:
4*1440words require23040B plus up to8192B codebook, with occupancy/adapter-limit
tradeoffs. It is not implemented and should follow representative profiling.

## Resume and stop procedure

Root exclusively owns `root-native-text-target-20261005`, GPUs and service
mutations. Use separate worktrees for any authorized concurrent source work.
No GPU resets, PCI unbinds, reboots or desktop disruption. Keep Kuno transient.
Refresh cached sudo through the idle Demons shell; never type into an active
foreground runner. If sudo expires, ask once or stop at a documented checkpoint;
no privilege workarounds. The automated goal stays paused until user resumes it.

Build committed root source with `helpers-20261008/build-direct-target.py`
and a fresh basename (`CARGO_BUILD_JOBS=2`). Create a fresh output directory.
In the shared shell, from the base:
`sudo -n python3 run-native-strata-public-protected-20261009.py BUILD OUT TRIAL`.
Source hashing can take two minutes before a GPU log appears.

CPU13 trial:
`warm-intel-mtp-four-lookahead-batch-repair-down-integer-grouped-ple-gu-stride-shared-head-gpu-host`.
Normal code200 trial:
`language-code-intel-mtp-four-lookahead-batch-repair-down-integer-grouped-ple-gu-stride-shared-head-gpu-host`.
Original full CPU13 tolerance is `0.006 + 0.0003 * abs(reference)`; preserve
vectors, carries, greedies, real repair and full-owner closure. Normal fixture:
seed424242, temperature0, public Python sorted-list merge prompt, thinkingfalse,
repetition1 and penalties0,200emitted/199timed continuations. No repeated Ollama
baseline is needed. Preserve rejected and neutral results.

The runner restores admission in finally. Confirm actual memory recovery,
HTTP listeners, no active benchmark processes and unchanged kernel-fault
counts. Final helper: `helpers-20261008/final-secondary-state.py`; there is no
top-level host-state.py. Update this opening checkpoint and coordination guide,
commit source/results/docs and leave no queued work when stopping.

## Historical evidence — not the current work queue

v181 on8f2a2e4 accepts the pending fused-four MTP repair with original cold and
selected warm full CPU13 vectors/carries/greedies, actual rejected-prefix
repair, full closure and physical recovery. Selected work joins three real
count4 repairs/six appended cells plus one original tail/two cells, with four
aggregate final fault gates. Original prefill remains false/zero. This is a
correctness run, not a new whole-model speed claim. Normal services recover
01:12:45UTC on the same boot, B70used50688000B, Nvidia compute0, no classified
faults. See `v181_four_lookahead_batched_repair_cpu13_acceptance`.

Completed v182/v183 manual-pass record: batched integer output head is source/device accepted, but
its first16-value variant is neutral in normal generation. v182 on dab13e2,
frozen SHA4b5f5ece164aefaf5465bf20992e3d49a3dfcf6c07ec9cdec205ba197ff6cf44,
measures10.882673 original versus10.880799 integer real continuations/s in one
same-artifact compiler-idle code200 pair. Both199 timed continuations,52 real
rounds,147of155 actual proposals accepted and5 real recomputations. Full cold/
warm CPU13 vectors/carries/greedy and full erasure/recovery pass. Do not call
this a speedup. The paired32 load variant on f95caee / v183 also passes full
cold/warm CPU13; its normal speed is still pending. Neither head variant is
selected for installed/default inference.

Pinned MIT Strata fb58e0d SYCL verify.cpp1537-1541 prepares allT head inputs
then native_mmvq; its wide Q6 kernel uses packed integer dots and32values per
lane. Kuno's old Verify3/4 head already shares weight reads across RHS. The
new head retains original210-byte blocks,16cached tiles and existing GR
scratch; it adds no model owner. Strata B70 also uses direct pinned HOST
expert reads, and replays captured verification graphs. Next priority is a
measured bottleneck: default-off fixed-public first-warm-target-round kernel
profiling on source c742998 / v184. Read
[kernel diagnostic contract](kuno-native-public-target-profile.md). Root strict
pure-direct Clippy,52 controller tests and77 protected-scope Python tests pass;
physical timestamp profile passes in v185 (see current checkpoint above). Profiling itself changes scheduling and
is never a normal-throughput result. Existing host enqueue/phase timers
include GPU backpressure and are not device-exclusive times. No CPU overlap,
new hybrid placement, API or frontend work is selected by this experiment.

User steering after the completed three-way comparison: continue optimization
with target and full Q8 MTP on Intel B70 only; defer Nvidia experiments.
Ordinary installed defaults remain unchanged until their own integration gate.

Earlier normal Intel comparison v177 (superseded by v182 above): **10.442972 original three-row**
and **10.740576 genuine four-row real continuations/s** on source26295f6.
This is one compiler-idle same-artifact ordered code200 pair (+2.85percent),
with199 actual continuations/200 emitted in each arm, not repeated speed
acceptance or native30tokens/s. The earlier v173 repair comparison measured
10.533053 original scalar versus10.603973 batched (+0.67percent). The selected
pre-v173 baseline was v167 at10.482354 real continuations/s
with original bank filling, original512 queue/IQ4 decoder, grouped PLE,
integer GU plus stride-four, integer DOWN and shared full-Q8 Intel MTP.
The public profile filler measured10.478727 and remains unselected. See the
completed v167 receipt below; these are one ordered code200 pair, not30tokens/s.

The borrowed CPU leaf/actual same-RAM bridge and fixed one-layer operator
passed the v172 physical gate on source47b972bf73f79c04acee3c87e922526203c54099,
binary SHA256107958180ec3334cc6e1c84b65976600b790c94539f12ca04cbd6c4f6e06175a.
All45 cases/180 pairs pass actual full partial/inactive/padding/source/fault
checks; maximum absolute error1.071021e-8 under unchanged1e-4/3e-4 bounds.
Cold originalCPU13 and warm token0 full vectors pass. All11 operator owners,
1380 target MAIN owners and432 original HOST pages close, and same-boot normal
services/B70baseline/fault counts recover. Read the original operator contract.

The complete15-worker CPU handoff/compute/upload path is2.324ms versus1.763ms
original-F32 GPU at count3/fiveHOSTranks; at tenHOSTranks3.394ms versus2.852ms.
Keep the GPU path for current batched verification. Some count1 CPU points win,
but this L3-hot synthetic layer does not establish whole-model hybrid benefit,
and the selected integer GPU path was not compared. Exact cases/owners/recovery
are in `v172_attached_cpu_operator_physical`. The four preceding setup failures
(v168..171) are separately recorded; no timing/complete close credit belongs
to a failure. v171 exposed cumulative full-owner I/O audits exceeding128MiB;
v172 uses the existing complete audited sanitizer after every pair, retaining
the cap and recording432.422ms total cleanup outside both operator arms.

Original-math batched MTP repair passes v173 original CPU13 cold/warm full
vectors/carries/greedy and actual rejection repair on frozen source
970f8f4141adc903212fe102ada504b4a42edfe8, pure-direct binary SHA256
69bc565a1396d85d9f9fdaa702620d94b44795c15b4541329f7f4d502c8ec0ec.
One compiler-idle same-artifact code200 pair measures10.533053 original versus
10.603973 batched real continuations/s (+0.67percent); both199 continuations,
130of138 actual drafts/69 rounds/five real target recomputations. New repair
actually completes69 joined calls/130 appended cells/69 final fault gates,
reducing MTP reads337to207. The1.802to1.590s repair interval includes necessary
real target recomputation/full checks, not only readback cost. Keep the explicit
implemented leaf; the next separate four-row candidate retains original scalar
repair and needs its own numerical/whole-model gates. Native30tokens/s remains
unachieved. No repeatedABBA/prose/independentCPU200 or installed/default claim.

Actual accepted bank backing is14,557,240,448B for14,144,716,800B logical payload:
412,523,648B actual padding versus4,166,132,608B unused conservative admission
allowance. All48 layers/288 distinct original VRAM buffers are measured without
GPU queries or additional owners; attached and successful-close snapshots match.
The unused conservative bound is not additional physical VRAM. Existing cold/
warm/normal owner closures pass:1382 target MAIN/828997632B,42 draft private
owners/11698176B and432 HOST allocations/36147609600B freed, backing0.
Last recovery21:43:44UTC: same boot, B70baseline50688000B, Nvidia compute0,
allHTTP200, unpaused/idle/no recovery latch and all classified GPU faults0.
Exact source, admission, counters and complete receipts are recorded in
v173_intel_mtp_batched_repair_and_bank_backing_physical.

Current integrated candidate is genuine four-row verification with three real
MTP proposals, original tails/scalar repair and new count4 arithmetic/metadata
paths. v177 on26295f6 passes both original and selected cold/warm full CPU13
vectors/carries/greedies, real rejection repair, complete close and recovery.
Actual selected warm work completes3 count4 chunks with2256 successful equal
Verify4 dense/fused calls: accepted prefixes1once/4twice, plus one original
count3 tail. Two of3 actual third proposals are accepted; conditional on the
first two being accepted,2of2 thirds are accepted. No independent third-draft
CPU oracle is claimed. Normal code200 adds actual prefix2/3 coverage. Cold/prefill/default
four counters stay false/zero. All1382 original MAIN owners and432 HOST pages
close, and target/draft/source zero/free checks complete. The earlier v176
guard failure is repaired using actual attached frozen-four authority while
original direct demand loading remains capped at3. v174 operators are reused
because these setup fixes change no arithmetic/shader/reference/tolerance/cap.
Normal same-artifact code200 measures10.442972 original versus10.740576
four (+2.85percent). Selected work completes51 count4 chunks and38352 equal
Verify4 calls, accepting147of155 real drafts over52 rounds;46of48 third
proposals whose first two are accepted also accept. Five real recomputations
complete, with repaired hidden/logit maximum absolute error0 and full closures.
Original completes69 rounds/130of138 drafts/five recomputations. All natural
prefix lengths1..4 occur across the two fixtures. Both arms restore same-boot
normal services/B70baseline and zero classified faults. One ordered pair does
not establish repeated/prose/independentCPU200/native30 acceptance. See
v177_actual_frozen_four_cpu13_acceptance and
v177_original_vs_four_normal_intel_pair in the numeric receipt.

GDN retained metadata correction (`cd638c7`, original leaf`b3480fe`) passes
both original/four cold/warm full CPU13 and complete closures/recovery. Frozen
v178 code200 measures10.704157 versus v177four10.740576, about0.34percent
slower in one comparison: no measured gain. Both do identical real work,
147of155 drafts/52 rounds/five genuine repairs and zero repaired-vector error.
Its35 source checks/strict Clippy pass; all arithmetic/shaders/caps are unchanged.
It removes the omitted metadata queue submission but this receipt grants no
isolated GDN speed claim. See v178_gdn_retained_metadata_cpu13_acceptance and
v178_gdn_retained_metadata_four_normal.
The separate original three-row CPU-HOST/GPU-resident overlap operator is
registered on d587d9b, with protected admission19checks and frozen entry8
negatives passing. First v179 physical run reaches four complete case progress
sequences then fails sparse negative fault checking; no completed numeric or
timing receipt is credited to that failed run. Finite guards/diagnostics are
hardened in `47f4c8e`. The v180 retry passes all five cases/75 full arm calls,
all11 negative controls and complete source/owner/recovery checks. Concurrent
CPU/GPU is3.893ms versus5.764ms original-F32 GPU for disjoint HOST experts,
but3.905ms versus2.989ms for shared experts and2.964ms versus2.553ms sparse.
This grants no whole-model or selected-integer GPU speed claim; no consumer is
activated. See `v180_attached_cpu_overlap_physical` and the operator contract.
The unused fused-four MTP cache-repair extension is integrated
as `bd46b2b` (original `7ea302c`), preserving original1..3/default behavior,
math, 42 owners and caps. Its56 focused tests and strict direct Clippy pass.
The approved separate controller composition is integrated as `83967bc`
(original `fc0948c`). Its225 distinct focused tests and strict direct Clippy
pass. Only actual count4 rounds use the fused-four leaf; tails retain old
repair. Actual completed counters join the separate four/tail records to the
full observed Traffic delta, preserving the old max-two receipt. No protected
selector or actual model GPU consumer is selected yet. No normal
API/installed/default selection is changed.

Root combined pure-direct all-target Clippy with warnings denied, full native
crate formatting and diff checks pass on83967bc after integration. CPU/compiler
jobs are idle at this checkpoint. See the complete source-only receipt
`paused_intel_only_mtp_four_repair_source_checkpoint_20261012`.

Historical resume direction, superseded by the compaction queue above: the [four-row repair controller contract](kuno-native-public-four-batched-mtp-repair-controller.md).
Root must register the exact combined protected scope/fixed labels, freeze a
fresh artifact, then run original cold/selected warm full CPU13, all genuine
prefix/repair/closure checks and a same-artifact scalar-four versus joined-four
normal code200 comparison. Keep the original model, CPU13 reference, public
prompt/seed and tolerances. The current artifact v180 tested only the separate
CPU-overlap operator on47f4c8e, before MTP composition; it cannot credit that
new controller. Do not skip physical acceptance or claim30tokens/s.

### Next speed candidate after the paused checkpoint

Read-only inspection of `resident_gr_batch::enqueue_final_head_with_target_capacity`
finds that `FinalAllRows`/`FinalFourRows` intentionally use the ordinary F32
`enqueue_batch` output projection, bypassing the accepted count1 integer Q6
head. A separate real batched integer head for counts3/4 is the next narrow
candidate after fused-four repair. Its proposed four prepared input planes use
23040B in existing NORM scratch; authenticate the full owner, lifetimes and
arithmetic before implementation. No new allocation or device acceptance is
claimed. Independent packed Q6 fixtures/count1..4 and unchanged full CPU13,
real rejection repair and paired normal timings remain required.

Target verification accounts for13.2s of the18.5s selected normal generation;
the head's device share is unmeasured, so this is not a throughput promise.
Global overlapping HOST I/O timers do not isolate removable device latency.
More resident experts need a separate capacity/source design: the measured
actual bank padding is412523648B, and unused conservative admission is not
physical free VRAM. CPU overlap would add48 actual per-layer routing fences;
its original-F32 isolated comparison cannot establish a gain for the selected
integer whole-model path. Do not begin these candidates while the goal is paused.

Accepted GU scheduling physical source isac78f00/v159, frozen binarySHA256
`c04d49a6ca11115c0bba0b1acc38019ae2f6f4b1f323cc0f9b6608eaa639bc7d`.
Encoded-register preparation and explicit GU stride-four pass the unchanged
original CPU13 cold/warm vectors/carries/targetgreedy, real repair, full1381
MAIN owners/828932096B and432 HOST pages/36147609600B freed, draft/source closure
and physical recovery. Compiler-idle, same-artifact ordered code200 pair:
8.636025 control versus8.973138 stride candidate, a preliminary3.90% gain.
Both time199 real continuations; both accept130/138 drafts over69 rounds.
Candidate22.177303819s total,5.051258292s draft,15.299634771s verification,
1.809250708s repair; actual stride17760 GU calls. No repeatedABBA/prose or
independentCPU200 acceptance. Defaults remain unchanged; see checked-in
`v159_encoded_register_gu_stride_physical` receipt. Earlier v158 integer
normal8.642210 is retained as historical evidence.

The requested MTP placement comparison uses source86882f9/v161, frozen binary
SHA256 `95ab177a40131f60342b5dff72cb75486c1d89958709def127b0e079ba777632`.
Intel MTP passes the unchanged original CPU13 cold/warm target vectors/carries,
targetgreedy, real repair, all1381 target MAIN owners, all42 draft private owners,
full432 HOST-page detach/free and physical recovery. Actual receipt verifies
same B70 PCI0000:2d:00.0 with distinct live Vulkan logical devices and runtimes.
Full144 target banks,8GiB static cache and full Q8 draft coexist without widened
caps or fallback. Combined34,166,833,152B owning admission includes512MiB driver
margin, leaving24,776,704B beyond that margin on the fresh baseline query.

Same-artifact compiler-idle ordered code200 so far: Intel MTP9.908034 versus
Nvidia MTP8.876789 continuation tokens/s (199 continuations each), preliminary
11.62% Intel gain. Both accept130/138 drafts over69 rounds and make5 real repairs.
Intel draft3.053354s/verify15.174402s/repair1.836269s, total20.084711873s;
Nvidia draft5.281152s/verify15.321660s/repair1.796661s, total22.418017018s.
This does not isolate PCIe transfer from draft GPU compute/queue costs and is
not repeatedABBA/prose/independentCPU200 acceptance. Warm12-token6.464541 remains
diagnostic only. True target-only comparison completes199 real count1 continuations at8.336136t/s
(23.871972055s), with no draft model/device/public weights/proposals/repairs, full
1381MAIN/432HOST closure and physical recovery. Intel is preliminarily18.86%
faster than target-only; Nvidia6.48%. Retain MTP and investigate Intel placement
as the next internal candidate; ordinary defaults stay unchanged. Required
Vashti changes:none.

The first v160 attempt rejected before model allocation because pinned wgpu29
CoreDevice equality compares registry-local IDs across independent Instances.
Fix86882f9 uses borrowed nonnull/distinct live Vulkan device handles, preserving
adapter/runtime/admission checks; six focused tests and strict all-target
Clippy pass. Legacy no-parent Nvidia scopes explicitly report unverified/null
logical identity; full frozen placement scopes require actual verified identity.
The failure and successful retry are both recorded separately in checked-in
`v160_intel_mtp_registration_failure` and `v161_mtp_placement_comparison`.
No installed/API/default changes. DOWN stride and grouped PLE stay unselected
for the placement comparison; do not combine their effects with placement.

Previous full physical source is21d97e2/v157, frozen binarySHA256
`a483e3f6aefb4ea74cda20457d17bf7bb240c18ff50cd67a3e7ae646ab91df44`.
The pinned40525-row English/code Q8 draft head passes unchanged full CPU13
vectors/carries/targetgreedy, real repair, all42 draft private owners,1380 target
MAIN owners/828866560B,432 HOST pages/36147609600B freed and full physical recovery.
Normal compiler-idle ordered same-artifact code200 is8.293190 shortlist versus
8.525761 full-head control. Both time199 real continuations; shortlist129/140
accepted/proposed drafts over70 rounds, control130/138 over69. Draft time falls
5.239454→4.930069s but extra verification/repair removes that saving. No gain;
shortlist remains default-off. Warm12-token4.152881 is diagnostic only. Exact
source/model/helper hashes, all results and recovery are in receipt section
`v157_public_english_vocab_physical`. No ABBA/prose/independentCPU200/private or
production acceptance. Original map262144B owning charge,40525 rows and public
SHA256369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25 are required.

Accepted source scope for frozen GU integer preparation/eight-lane
projection: ownerprechargee655f32, leaf13c276c, limitfixfb7c528, parent9f80c33
(original agent commitsd68f80d/40bddbb/1657abe/e10e9fb). One full64KiB private owner
and independent HOST/INFO/runtime precharges retain original MAP/weight pages,
DOWN and all-family completion. Actual1 preparation:5 GUlaunches:1 owneraudit
must join deferred actual original routing acceptance. This is explicitly
alternate two-residual-Q8 activation/eight-lane arithmetic; packed oracle tests
are NOT originalF32 equivalence; the unchanged original CPU13/greedy/repair
physically passes v158 as above. Read the frozen GU leaf/parent/controller
contracts. Integrated followups cache encoded weight chunks once in registers(b4b20f9)
and add immutable four-group GU scheduling(50d423d/febf8df/e5182f0/853e0e0).
Combined source checks and the v159 physical gates above pass. Separate
original four-row DOWN stride scheduling is integrated(ec367fb) but remains
unselected/physically untested; the protected guard still needs its explicit
selector handler before a GPU gate. Keep this effect separate from placement.
Grouped PLE leaf/controller are integrated9f8462e/513fe02 onto the accepted
86882f9 HAL device-identity fix. Original source commits33b4de7/cce4f66 remain
clean at `/home/jmicko/.codex/worktrees/kuno-grouped-ple-controller/vashti`.
Read [the exact grouped PLE contract](kuno-native-target-grouped-ple.md).
All13 grouped tests and78 owner controller/helper checks plus strict Clippy pass;
root integrated CPU/source review is underway. Policies remain default-off.
Protected guard/helper integration is being prepared; no new physical or speed
acceptance yet. Fresh freeze must pass unchanged full CPU13/repair/owner/recovery
before Intel-only normal comparisons. Reserve before model; select only after
accepted prefill/source attachment before requestIO. No global GR math or changed
owner budgets. Test PLE separately from DOWN stride before combining.
Root owns the guard/helper and all shared GPU work.

Earlier e16c549/v156 ThreeGdn full gates pass at8.489327 normalcode200; control
8.542111 may overlap agentClippy and is not a compiler-idle causal pair.
Q2 v155 reserves2539257856B public draft weights versus Q85011668992B but
normal7.949402 establishes no speed gain. Three-member8.094793, staging7.870304
also pass full physical gates and remain default-off. v150 ordered4DOWN/shared
head8.119520 control /8.484726 candidate is preliminary, not repeatedABBA.

Last recovered host2026-10-11T17:14:23.384257UTC: same boot, B70baseline50688000B,
NVIDIA compute0, unpaused idle/no recovery, HTTP200 for allthree services, GPU
faults0. This is a historical snapshot, not future availability.
No30tokens/s, independentCPU200 or private/production acceptance. No runtime
installation, API/UI work, release or other task messaging.

## Historical stop checkpoint from 2026-10-10

Updated2026-10-10 after the user requested a few final tests, a stop and a usable
system. GPU/CPU work is STOPPED and the goal remains paused. This file does not
authorize further GPU occupancy, service changes or installation.

## Historical checkpoint

We have real Rust inference with original DASLab IQ3_S Qwen3.8-Flash-Next and Q8
MTP on the3060, but have NOT reached20 or30 tokens/s. Prior repeated v145 GPU
router + inactive-partition-exit code200 result is8.33810 continuation tokens/s.
GLM's ledger fix8c7d838 and preserve-placement source4fad434 were reviewed with no
blocking findings. Preserve-placement now passes hardware gates, but only gives
5.72541 tokens/s in a short all-selected-in-VRAM replay. Normal generation
already preserves post-prefill residents; do not mistake that for a new speedup.

Current integrated kernel source is7364dc8, following private-table leafe054d70.
Frozen v147 passes full original CPU13 vectors/carries/greedies/actual repair,
source/owner closes and physical recovery. Its same-artifact ordered code200 pair
is7.78036 control /7.62161 module-private public tables, both199 timed
continuations. This candidate shows no gain and remains default-off. No ABBA,
prose/soak or independent CPU200 proof was run; no installed/public API change.

Two clean source-only leaves based7364dc8 remain UNINTEGRATED:

- `3594b76de116fc72f536cc296572a281d54231f2` at
  `/home/jmicko/.codex/worktrees/kuno-constant-array-probe/vashti`:
  deduplicated public workgroup grids with literal initialization after uniform
  inactive exit, one barrier,5/6/12KiB total SLM. Three focused shader/SPIR-V/
  immutable-store/geometry tests and strict direct Clippy pass.
- `b56af6af0ef70f30401ae0b881fc4178c9807951` at
  `/home/jmicko/.codex/worktrees/kuno-iq4-down-scales/vashti`, branch
  `codex/kuno-iq4-down-scales`:20 exact IQ4_NL scales in80B SLM, original
  decoding/order/dots, unchanged Q2/default. Three focused shader/SPIR-V/
  C-fixture bit/dot/owner tests, strict direct Clippy and peer review pass.

Their runtime selection, root parser/launcher/receipts and GPU proof are pending;
DO NOT send a guessed workgroup/shared selector to v147. After user-authorized
resumption, integrate leaves (resolve shared Geometry/Sources policy hunks), add
explicit default-off selection and actual-completed-policy receipts, retain
private/workgroup mutual exclusion and scale-policy composition, rerun CPU
proofs, freeze a NEW clean artifact and run full original CPU13 gates before
normal same-artifact language timing. Do not substitute resident experts or
loosen reference checks. Keep focus on speed; no API/UI/auth/cancellation/release
work is needed for this internal pass. Don't repeat old Ollama baselines.

Read the [latest detailed checkpoint](kuno-native-text-checkpoint.md#resumed-shader-experiments-and-usable-stop-checkpoint-2026-10-10)
for exact artifact SHA256s, paths, measurements, tests, flags and ordered gates,
then the active coordination handoff and all-row kernel contract. It supersedes
all historical instructions below. Root receipt is
`apps/kuno/benchmarks/text/receipts/native-strata-optimization-20261009.json`.
No prompts/generated text/raw output tokens/routing IDs/images/credentials were
saved. Existing original model/reference hashes and tolerances are unchanged.

Final live check2026-10-11 01:27:07UTC: both cards recovered (B70 50,688,000B,
no NVIDIA compute owners), Kuno admission idle/unpaused/no recovery latch,
Kuno/Ollama/Vashti HTTP200, same boot and classified GPU faults0. No owned jobs
remain. Installed binaries and canonical frontend were untouched. Required
Vashti changes:none. Confirm fresh availability/sudo before any new GPU run.

### Correct checkout and ownership

| Purpose | Checkout | Branch / checkpoint before this handoff |
| --- | --- | --- |
| Native engine source; work here | `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti` | `codex/kuno-native-text`, kernel integration `7364dc8`; later documentation commit |
| Separate Vashti application task; leave alone | `/media/jmicko/faststore/dev/vashti` (also `/home/jmicko/dev/vashti`) | `codex/image-flow-refinements`, HEAD `ae3e70d` |
| Local frozen binaries, models' numeric reference fixtures, helper scripts, receipts | `/media/jmicko/faststore/kuno-development` | Not the source checkout |

Native tracked source and isolated leaves are committed; pre-existing Python
bytecode remains untracked. Canonical frontend was clean and untouched. Recheck branch,
status, and recent commits before editing; other tasks may subsequently change
them. Preserve unrelated files, including coordination documents. Concurrent
source work requires separate worktrees from an agreed commit; worktrees do not
isolate GPUs, system RAM, services, or the shared compilation directory.

Read `AGENTS.md`, the latest active section of
[the coordination guide](kuno-vashti-coordination.md), and the
[immutable RAM-source/partition contract](kuno-native-mtp-all-rows-plan.md#authenticated-immutable-ram-expert-sources-2026-10-09).
The [current checkpoint](kuno-native-text-checkpoint.md#ram-source-full-model-acceptance-and-stop-checkpoint-2026-10-09)
contains the exact preceding results. Earlier “active” sections in these long
documents are historical, not instructions to repeat every prior experiment.

## Historical detailed handoff, 2026-10-09

The remaining sections are historical reference. The current checkpoint above
supersedes their old acceptance tables and immediate-next-step instructions.

## What is accepted, and what is not

| Component | Actual acceptance | Limitation |
| --- | --- | --- |
| v143 immutable RAM expert sources | B70 full original CPU13 logits/carries, exact greedies, real MTP prefix repair, target/draft closes and host detachment | Uses **CPU routing**; short deterministic physical placement fixture |
| Frozen512 device router | 80 standalone cases on **each** GPU; original selected IDs/order, guarded faults, operator erasure | Not full-model device routing or host-weight arithmetic acceptance |
| Inactive-partition exit candidate | Source tests, shader lowering, parser/launcher checks, Clippy and formatting | **No hardware correctness or timing test yet** |
| Ordinary native public code200 | Older separate artifacts produced about 2–4 continuation tokens/s under different disk/cache conditions | Not yet tested with the v143 RAM-source control or new candidate |
| Installed native engine selection | None from this pass | Laboratory code is not a production catalogue/API backend |

The physically accepted control is source
`f0e088b28038646ad34fcc358a63344c0bbbb5d0`, frozen binary SHA256
`ddbd8c7dba93944146203c72f6fb235eaddca49ae5d4b53478ee5d69ef9794fe`.
Its build directory is:

```text
/media/jmicko/faststore/kuno-development/native-strata-v143-host-alignment-20261009
```

The current candidate source is
`389fb6b388b450f05e0bf05e74859ecc1956af6f`; `cfd4f14` adds checkpoint documents.
The next build must record its actual new HEAD and binary hash. Do not call it
v143 or transfer v143's physical acceptance to the new binary.

### v143 result details worth preserving

The full CPU13 replay uses physical VRAM placement of original experts 0..143
to exercise both storage tiers. It does **not** replace the router's selections,
weights, or tokens. The B70 runs the target and the RTX 3060 runs the original
Q8 MTP draft. The warm replay produces twelve real verified continuations in
2,845,782,894 ns: **4.216765806450167 tokens/s**.

Numeric evidence:

- `expert_uploads=0`, `uploaded_bytes=0`; process physical reads **1,081,344 B**.
- Actual rank-member routes: VRAM **2,590**, RAM **6,530**; source dispatches
  **4,320**. Routes are not unique expert reads or PCIe byte counts.
- Preparing the VRAM bank fills **6,132 experts / 12,535,603,200 B** in
  18.802569481 s; RAM attachment takes **41.434147362 s**. These setup costs are
  excluded from the twelve-continuation timing.
- **432** host pages are registered/detached; actual allocated/CPU-zeroed/freed
  backing is **36,147,609,600 B**, ending with `live_backing_bytes=0`.
- The separately budgeted host reservation is **36,166,705,152 B**. Do not
  report reserved and physically allocated bytes interchangeably.
- Target final close audits **1,378 owners / 826,638,336 private bytes**; the
  draft close and actual repair checks also pass.
- The initial replay is 0.97823 tokens/s with 2,311 uploads / 4,776,960,000 B
  and 4,917,862,400 B of process disk reads. It is not a controlled cold-OS-cache
  measurement: the helper hashes the entire target before model setup.

Zero explicit expert uploads does **not** mean zero PCIe traffic. The GPU reads
the immutable RAM pages across PCIe. The full 83.6 GB model is not claimed to
reside entirely in RAM. Routed missing-expert backing is about 36.15 GB;
file-backed PLE/ngram and other model regions remain separate. See
[disk-access notes](kuno-native-text-disk-access.md) before assuming all disk
reads are errors or that a whole-model mmap means physical residency.

The standalone router's 80 cases comprise 41 numeric cases and 39 fault cases;
12 ties are a subset of the numeric cases. Each card zero-audits 480 owners /
188,743,680 B. Frozen512 means addressable original expert IDs, **not 512 VRAM
residents**. F32 route probabilities remain experimental at 2e-6 absolute /
2e-5 relative tolerance; original selected IDs/order must still match exactly
through the deferred actual-score audit before model commit.

## Historical candidate: previously prioritized experiment

Five immutable expert source partitions share the same 30-group gate/down
dispatch geometry. Much of each partition's scheduled work belongs to groups
owned by another partition. The candidate makes those workgroups exit uniformly
before shared scratch/reductions. Active groups keep the original 64-lane F32
operation order. It preserves the full scratch clear, finite/status checks,
rank fold, exact expert weights, full-family commits, and deferred routing audit.
There is no grid/count trimming, quantization change, new owner, or larger budget.
The scheduling ratio is **not** a predicted throughput multiplier.

The exact opt-in is:

```text
KUNO_LAB_TARGET_SOURCES=frozen-host-public
KUNO_LAB_FROZEN_PARTITION_EXIT=1
```

Absent means false. Only canonical `1` is accepted; do not use `true`, `0`, or
enable it without the frozen-host source policy. It composes with CPU routing
or the separate `KUNO_LAB_TARGET_ROUTER=device-frozen-public` experiment.
Source geometry captures the flag immutably; a completed actual native boundary
must agree with the requested policy before a numeric receipt is accepted.

Implementation source chain:

- `612b7f6`: inactive-partition kernel leaf (peer source `e80dd80`).
- `6413b16`: guarded parser/protected-launcher opt-in (peer source `9eeca92`).
- `389fb6b`: actual immutable source attachment and completed-boundary checks.
- `9538601`: earlier full-model Frozen512 device routing integration.
- `f2d383b`: earlier immutable RAM-source full-model integration.

Relevant source is the standalone Rust laboratory at
`apps/kuno/benchmarks/native-text/`, especially:

| File under `src/` | Purpose |
| --- | --- |
| `daslab_speculative_probe.rs` | Full original CPU13 target/MTP controller and cold/warm correctness gates |
| `daslab_language_probe.rs` | Normal public code/prose generation and timing |
| `resident_model_batch.rs` | Policy parsing and full-model native bridges |
| `resident_frozen_sources.rs` | Five frozen expert sources and immutable exit selection |
| `grouped_moe.rs` | Original and candidate grouped expert kernels |
| `resident_batch_router_gpu.rs`, `.wgsl` | Device routing/source-group maps |
| `resident_batch_router_gpu_probe.rs` | Standalone router numeric/fault cases |
| `public_host_expert_plan.rs`, `public_host_expert_mirror.rs`, `public_host_expert_attachment.rs`, `public_host_bank_completion.rs` | Budget, actual allocation, registration, safe detach/erasure |

Typed external host-buffer ownership/runtime support is in `vendor/cubecl-*`
under the laboratory root, outside `src/`.

This crate has its **own Cargo workspace**. Root-workspace tests/builds do not
build this laboratory. Existing CPU reference math is also in `crates/kuno-text`.
Do not search in an invented `apps/kuno/src/text_lab` directory.

### Crucial local helper trap

`/media/jmicko/faststore/kuno-development/run-native-strata-public-protected-20261009.py`
currently does **not** select the new partition-exit flag. It assembles an
explicit environment dictionary for the transient systemd service. Exporting a
flag in the caller is not a reliable way to select or prove the candidate.
The helper is local evidence infrastructure, not committed engine source; its
hash is recorded in each receipt.

Before testing the candidate, make a small, explicit helper extension and test
its environment selection. A suggested set of **new, not yet implemented** trial
labels is `warm-fast-host`, `warm-fast-gpu-host`, `language-code-fast-host`, and
`language-code-fast-gpu-host`. Their existing `warm-` / `language-code` prefixes
and `-host` / `-gpu-host` suffixes preserve the current fixture/hash/router
selection logic. Add them to the allowlist and set
`env['KUNO_LAB_FROZEN_PARTITION_EXIT']='1'` only for those exact new labels.
Preserve the existing control labels. Do not put this flag into `operator`:
that mode has an exclusive two-variable environment and rejects it before
device creation; it does not exercise the expert arithmetic kernel.

Record the helper change/hash and requested candidate policy. Verify the actual
receipt's `frozen_host_inactive_partition_exit` value. v143 predates this field;
its missing old field is not evidence about a fresh candidate.

## Exact resume sequence

1. Obtain an explicit resume from the user. Read current coordination/spec,
   inspect both worktrees, and inspect the idle shared Demons `shell` panel.
   Confirm the user is not running image/text inference or another GPU agent.
   Inspect live admission and GPU state before changing it. Do not benchmark
   while builds or other GPU clients are running.
2. Make/test the helper selection change above. Keep the **Q8 draft**, current
   source/QSA policies, public references, and model hashes fixed. Do not combine
   this kernel test with Q2 draft, activation quantization, cache changes, or a
   new model. Save a clean committed source checkpoint before freezing a build.
3. Freeze one fresh `direct-device` artifact. Use that same artifact for these
   four gates, with compilers idle and a fresh output directory for every run:

   | Order | Full CPU13 replay | Actual GPU router | Actual partition exit |
   | --- | --- | --- | --- |
   | 1 | CPU-routing control (`warm-host`) | false | false |
   | 2 | CPU-routing exit candidate (new explicit label) | false | true |
   | 3 | GPU512-routing control (`warm-gpu-host`) | true | false |
   | 4 | Composed GPU512 + exit candidate (new explicit label) | true | true |

   All four must pass complete CPU13 vectors/carries, exact greedies, real
   prefix repair, actual policy receipt checks, private/source/draft closes,
   and physical recovery. GPU512 whole-model acceptance is pending independently
   of the exit kernel; gate 2 isolates the kernel first. Do not assume standalone
   router acceptance proves gate 3. CPU controls must have `gpu_router_calls=0`;
   GPU candidates must have positive actual calls. Require both storage tiers
   to be exercised in the CPU13 fixture.
4. Only after these gates, measure **normal original public code200** with the
   fastest accepted combination and an appropriate same-artifact control. The
   existing `language-code-host` and `language-code-gpu-host` labels are controls;
   candidate labels require the explicit helper extension. Record actual accepted
   counts, setup, prefill/decode, reads/routes, policies, closes, and recovery.
   Ordered preliminary runs are not a robust speed claim. Use a matched ABBA
   comparison before claiming a small improvement. Stop repeating experiments
   that do not resolve a specific uncertainty.
5. Update the checkpoint and numeric receipt with source/binary/helper hashes,
   exact results, limitations, and the next concrete step. No frontend/API handoff
   is required for backend-only internal work. End with normal services/admission
   and physically recovered memory. If sudo expires, do not launch another
   dependent privileged command; ask for a refresh and report the checkpoint.
   Let an already launched protected trial reach normal close: its existing
   root helper/service and cleanup do not lose privileges when the timestamp
   expires. Do not spend usage on privilege workarounds.

### Build and existing control invocation

These are example names; choose new names/date if they already exist. The build
helper **creates** its build directory. The trial helper requires its output
directory to **already exist and be empty**. Both helpers hardcode the native
worktree; changing the caller's current directory does not change their source.

```bash
cd /home/jmicko/.codex/worktrees/kuno-image-speed/vashti
git status --short
git branch --show-current
git log -5 --oneline

python3 /media/jmicko/faststore/kuno-development/helpers-20261008/build-direct-target.py \
  native-next-v144-build-20261010

mkdir /media/jmicko/faststore/kuno-development/native-next-v144-cpu-control-20261010
sudo -n python3 /media/jmicko/faststore/kuno-development/run-native-strata-public-protected-20261009.py \
  native-next-v144-build-20261010 native-next-v144-cpu-control-20261010 warm-host
```

Run the protected command in the shared Demons shell, whose sudo credential the
user refreshes. Ordinary tool shells may have a different sudo timestamp scope.
Read the panel first; never type refresh commands into a foreground benchmark's
stdin. On an idle shell, `sudo -n -v` refreshes a still-valid credential without
prompting. Never save/print passwords or hidden control credentials. If it fails,
ask the user once and stop dependent GPU work until access is restored.

The helper takes **three positional arguments**: build basename, output basename,
trial. It has no fourth flag argument and currently no `*-fast-*` labels. Existing
useful labels are `operator`, `warm-host`, `warm-gpu-host`,
`language-code-host`, `language-code-gpu-host`, and corresponding prose controls.

The build uses release `--locked --features direct-device`, writes source commit,
features, Cargo artifact/build logs, and an immutable executable named by SHA256.
Shared `CARGO_TARGET_DIR` is
`/media/jmicko/faststore/kuno-development/root-native-text-target-20261005`.
The helper records HEAD but does **not** capture a dirty source diff: ensure clean
source, do not edit during compilation, and do not infer a source hash from a
directory name. No CPU-inference feature combination belongs in this comparison.

The trial helper authenticates both full original shards for `-host` modes and
checks file identity before/after hashing. This may take about 100–120 seconds
and **warms the OS file cache before setup**. Silence during hashing does not
mean a GPU hang. Keep the verification; label the cache state honestly.

It drains/pauses Kuno, then launches an owned transient systemd service as
`jmicko` with `MemorySwapMax=0`, `LimitCORE=0`, and `RuntimeMaxSec=1200`.
CPU13 memory limit is 48 GiB; language adds 196 MiB; standalone operators use
8 GiB. The protected launcher verifies these actual limits before execution.
The twenty-minute limit is a **laboratory watchdog**, not a production inference
deadline. Do not change production Ollama-compatible long inference to match it.
Do not bypass the launcher, globally disable swap, or raise limits silently.

### Failure and recovery procedure

The helper's outer error is deliberately only `error_type`. Keep the failed
directory, `intel.log`/`nvidia.log`, source verification, and before/after state.
Read the last public stage and original authored error from the relevant log;
do not erase failures or reuse the directory to make a retry appear clean.

The current helper's `finally` resumes admission **before** checking memory
recovery; killing the parent may skip `finally` completely. A successful helper
return is not the entire final-state gate. If a trial is interrupted:

1. Identify only its exact owned unit, `<output-basename>-intel.service` (and the
   NVIDIA sibling for two-card operator modes). Inspect its process/cgroup state.
   Stop that unit if necessary; do not broadly kill Ollama, desktop processes,
   other containers, or unrelated agents. Forced termination is a failed trial,
   not proof of the application's normal private close.
2. Verify the owned process/cgroup is gone, physical memory returned, and no new
   same-boot Xe/NVIDIA fault was recorded. If recovery is incomplete, keep/pause
   admission with `host-state.py pause` and stop further tests. Do not immediately
   resume and submit another model load.
3. Once physically safe, use the existing recovery/status helpers in the shared
   shell:

   ```bash
   sudo -n python3 /media/jmicko/faststore/kuno-development/helpers-20261008/host-state.py resume
   sudo -n python3 /media/jmicko/faststore/kuno-development/helpers-20261008/final-secondary-state.py
   ```

4. Require admission `active=false`, `waiting=0`, `paused=false`,
   `recovery_required=false`; all three HTTP checks 200; no NVIDIA compute owners;
   Intel physical usage back near baseline; and unchanged classified fault
   counts on the same boot. The wrapper currently checks fewer admission fields,
   so verify the complete state manually. NVIDIA desktop VRAM need not be zero.

The last accepted historical snapshot was **2026-10-09 21:23:38 UTC**: Intel
50,688,000 B, no NVIDIA compute owner, all services healthy, no classified GPU
faults. The wrapper's Intel idle bound is 160 MiB. Historical NVIDIA desktop
usage was about 483 MiB. These numbers are not a fresh guarantee; inspect again
on resume. The local helper checks Kuno on 11434, real Ollama on **11435**, and
Vashti on 7771. Earlier chat proposals used other ports; inspect configuration
before changing anything. Internal control is local 11432; do not copy or print
its protected token.

**Do not reset/unbind either GPU, unload drivers, reboot, force clocks, or change
global swap.** A previous Intel reset attempt black-screened the machine despite
the monitor being plugged into the 3060. GPU monitors/desktop clients can also
hold handles. If normal owned-process cleanup fails, report the evidence and
stop for the human rather than trying progressively riskier resets.

## Models, references, and public benchmark definition

Reuse the existing import; do not download another 83.6 GB copy or merge shards
just to use this Rust lab. The DASLab source is pinned to revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`.

Target directory:

```text
/media/jmicko/faststore/ollama/.kuno-import/daslab-next-ed59f92082b1/IQ3_S
```

| File | SHA256 |
| --- | --- |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113` |

Total target file bytes: **83,617,662,656**, about 77.875 GiB. This is the current
IQ3_S model, not the original approximately 120 GB Q4_K_M model. Geometry is
48 layers (36 GDN / 12 QSA), 512 experts with 10 selected, D=2560, FF=640,
HC=10240, vocabulary 248320. The host has approximately 62.71 GiB usable physical
RAM, a 32 GiB B70, and a 12 GiB RTX 3060; available memory is time-dependent.
The user's spare 16 GB may help longer runs, but the short RAM-source fixture
does not establish that adding it is necessary or sufficient.

Default Q8 draft, **retain for the next kernel comparison**:

```text
/media/jmicko/faststore/ollama/kuno-sidecars/unsloth-qwen3.8-flash-next/766911a6b7369840a91dbcd95f9f997acaab6cd6/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
bytes: 4,137,429,120
sha256: 58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1
```

Full thirteen-position original CPU target/carry reference:

```text
/media/jmicko/faststore/kuno-development/daslab-target-carry-reference13-20261009/public-thirteen-carry.bin
sha256: 80650af818ae00344af5b4b3979e9272cc833c641ea4f80d2934cdb6ffa67e02
```

Keep that reference unchanged. A failing candidate does not authorize regenerating
an oracle to agree with it. The controller checks complete vectors/carries, exact
greedy selections, and actual repair branches, not merely plausible text.

The public language fixture in `daslab_language_probe.rs` uses:

- Code: “Write a Python function that merges two sorted lists of integers into
  one sorted list. Explain its time and space complexity.” (35 prompt tokens.)
- Prose: “Explain how a hash table works, including collisions, resizing, and
  average lookup complexity.” (30 prompt tokens.)
- Seed **424242**, temperature **0**, thinking false, repetition penalty 1,
  presence/frequency penalties 0, maximum **200 emitted tokens including EOS**.
- Exact pinned ChatML rendering below; no extra system message or prompt rewrite.

```text
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The first output token is selected during prefill. A complete 200-emission run
therefore has **199 timed decode continuations**, not 200. Use the recorded actual
count if EOS stops it earlier. Report loading/preparation, prefill, continuation
decode, and close separately. Target verification/real repair remain mandatory;
the language harness is not an independent CPU200 quality proof.

Kuno must not persist private prompts, generated images/text, prompt summaries,
raw token streams, logits, routing traces, or reusable private scratch. Fixed
public fixture text may be documented; benchmark files should contain numeric
receipts/public labels, not generated answers. Vashti owns user persistence.
Protected swap/core limits and actual owning closes are part of this existing
laboratory; retain them without expanding this speed pass into an API redesign.

### Strata comparison and retained alternate draft

Strata is open source under MIT. The pinned comparison source is
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` (v0.1.41), retained at:

```text
/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009
```

The temporary Strata installation, build/venv/runtime arena, specifically owned
rootless development image, and listener on 8085 were removed. Original model
shards and comparison evidence were retained; unrelated containers were untouched.
Do not assume Strata is still running or reinstall it just to repeat a baseline.

Its measured Intel public results were about **27.3 code decode tokens/s** and
**36.0 prose tokens/s**. These are comparison results from a different engine/
draft/optimization stack, not matched native controls. Numeric evidence remains:

```text
/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009
```

The preserved alternate draft is:

```text
/media/jmicko/faststore/kuno-development/kuno-strata-mtp-q2-20261009/mtp-q2_0.gguf
bytes: 889,017,216
sha256: a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d
```

Internal `KUNO_LAB_MTP_PROFILE=strata-q2` selects its **raw dtype 42**: 64 values
per 18-byte block, codes {-1,0,1,2} multiplied by an f16 scale. It is **not Q2_K**.
Associated `rt/dense.bin`, `experts.bin`, and `draft_vocab.bin` remain as format
research. Our accepted alternate draft handles the original GGUF profile; do
not claim the whole optimized Strata runtime format or pruned vocabulary is
already implemented. Earlier Q2 trials did not establish a speed win over Q8.
Use it in a later isolated experiment after the current candidate is accepted.

After the partition experiment, a promising next kernel investigation is packed
IQ3 block decoding and more efficient row grouping in pinned
`sycl/src/kernels/cuda/iq_kernels.dp.cpp`. Strata's uniform inactive-group exit
was the immediate inspiration. Study the actual pinned implementation and keep
MIT notices for copied/adapted code. Changes to lane reductions, Q8 activations,
DP4A arithmetic, or vocabulary pruning need their own numerical proof; do not
combine them into the first exit-kernel measurement. Keep inference in Rust;
GPU/API bindings do not justify replacing it with the C++ engine.

## Already-fixed traps: do not undo these fixes

**Actual allocation alignment versus planning envelope.** v139–v142 host setup
failed after `prepare_complete`, `warm_reset_complete`, `attach_start` with
`public host expert source plan rejected`. Registration incorrectly required
actual allocation bytes to be a multiple of conservative 65,536-byte budget
rounding. Layer 0's gate page is **90,316,800 B** and valid at the driver's smaller
alignment; the conservative envelope is 90,374,144 B. `f0e088b` carries actual
Vulkan alignment in the sealed native owner/resource and validates that alignment
plus full bounds/identity/ownership. Do not add pointless padding, falsify actual
charges, weaken the guards, or confuse envelope size with driver requirements.

**Preserve the original failure while still closing.** `be3e0e8` retains the
authored public error after mandatory close attempts; cleanup flags are numeric.
A failed close/audit never produces an accepted receipt. Read the original stage
and error before making speculative architecture changes.

**Host detach order matters.** The GPU must finish, the model must close with a
complete zero audit and be dropped, and queued use must be joined before typed
host source detachment/free. Never free registered host backing while the GPU
can still read it. Retaining backing until safe process teardown on failure is
preferable to a use-after-free. Do not loosen the nonreuse registry/generation
checks to make a test pass.

**Router sentinel fixture was wrong, not the missing-source guard.** The old
fixture expected SHAPE=1 where the correct missing-source fault was MISSING=4;
ID 0 really was selected by that pattern. `fad1b21` independently derives selected
IDs 0/13/14 for its fault fixtures without changing kernel guards. Do not revive
the erroneous explanation that ID 0 was unselected.

## Existing tests and evidence; how to checkpoint new work

Aligned v143 source passed 945 native tests, 2 auxiliary shader tests, one
existing ignored opt-in test, 9 focused plan regressions, strict pure-direct
all-target Clippy, formatting and whitespace checks.

The final source-only candidate passed 45 public parser/reference tests,
30 frozen-source/routing/shader tests, 102 Python protected-launcher tests,
strict pure-direct all-target Clippy and formatting/whitespace. The leaf also
passed 47 grouped-kernel/guard and 4 source tests with all actual DAS mixed
profiles' checked/unchecked WGSL/SPIR-V lowering and preserved active arithmetic
structure. These checks do not replace physical whole-model acceptance.

Useful focused commands after relevant source changes (not needed merely to read
or edit these notes):

```bash
cd /home/jmicko/.codex/worktrees/kuno-image-speed/vashti
export CARGO_TARGET_DIR=/media/jmicko/faststore/kuno-development/root-native-text-target-20261005
cargo test --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml \
  --bin kuno-text-lab --features direct-device -j2 daslab_
cargo test --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml \
  --bin kuno-text-lab --features direct-device -j2 frozen
cargo clippy --locked --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml \
  --all-targets --features direct-device -j2 -- -D warnings
cargo fmt --manifest-path apps/kuno/benchmarks/native-text/Cargo.toml --check
python3 apps/kuno/benchmarks/text/test_native_count1_hybrid_lab.py
git diff --check
```

For expensive builds/tests, use the established shared target directory rather
than accumulating another build tree; do not run these concurrently with timed
GPU trials. Python is public-only benchmark/launcher infrastructure, not a
production Python inference dependency. Clean only your own generated artifacts.

Physical evidence under `/media/jmicko/faststore/kuno-development`:

- `native-strata-v143-host-reference-20261009`: accepted original CPU13 RAM-source
  controller, detailed numeric host/target/draft close evidence, recovery state.
- `native-strata-v143-router-operator-20261009`: accepted both-card 80-case router
  operators. Some conservative whole-model/host-weight acceptance fields remain
  false intentionally; do not rewrite them because individual cases pass.
- `native-strata-v139-frozen-host-reference-20261009`,
  `native-strata-v140-frozen-host-reference-20261009`,
  `native-strata-v141-frozen-host-reference-20261009`,
  `native-strata-v142-public-host-stages-reference-20261009`: preserved failures
  establishing the registration/alignment diagnosis.

Compact checked-in results live in
[`native-strata-optimization-20261009.json`](../apps/kuno/benchmarks/text/receipts/native-strata-optimization-20261009.json)
and the [checkpoint](kuno-native-text-checkpoint.md). New entries must distinguish
source checks, physical operator checks, full-model correctness, ordinary timing,
installation, and release. Record exact source/binary/helper/reference hashes,
trial policy and seed, acceptance failures as well as successes, measured token
counts, actual RAM/VRAM recovery, and which claims remain untested. A source commit
or clean device operator alone never establishes production readiness or speed.


### Intel-only grouped-read/source gate,2026-10-11

Grouped PLE source9f8462e/controller513fe02 and protected registrationde59c75
are now integrated above accepted HALfix86882f9. The integrated read-only review
finds no blocker;117 relevant Rust tests and strict locked pure-direct all-target
Clippy with two CPU jobs pass. Both original owning/fault proofs and unchanged
CPU13 reference/tolerances remain mandatory.27 protected-launcher tests and nine
new external fixed Intel-only PLE/DOWN/combined labels pass exact-selector and
original host-cap checks. Fake Xe metadata was used only in those source checks;
physical trials require a fresh privileged memory snapshot after draining.

Next artifact will independently gate grouped PLE on original cold/warm CPU13,
then compare normal compiler-idle Intel Q8 MTP code200 control/candidate. DOWN
stride stays off during that pair and needs its own original numerical/speed
gate before any combination. New policies remain default-off; source success
does not establish physical numerical, throughput or30tokens/s acceptance.
Root owns GPUs/services; canonical Vashti and installed/public behavior remain
unchanged. Required Vashti changes:none.


### Intel-only grouped PLE physical receipt,2026-10-11 17:41UTC

Frozen66bdd0f/v162 (SHA498736a0bacd6a6d31c9610931658fff0c04cdee790fa996f20a9331ae19a9d2)
passes unchanged originalCPU13 cold/warm target vectors/carries/greedy, real
repair, full1381MAIN/42draft/432HOST-page close and physical recovery. Warm
selection reports8actual completed groupedPLE chunks; cold false/zero. Source
gate117Rust/239Python/9fixed-label checks and strict directClippy pass.

Compiler-idle same-artifact ordered normalcode200 is9.886106control versus
10.022470groupedPLE continuations/s (199timed,130/138drafts,69rounds,5repairs).
Candidate reports74actualgroupedchunks and302readbackbatches versus450control:
148 fewer batches. Both still report618GPUwaits; scoped map intervals remain
about2.63s. This small1.38% preliminary timing does not establish a repeated
gain or removed actualGPUwaits. No math/owner/cap/reference change.

Latest recovery17:41:33UTC: B70baseline50688000B,Nvidia compute0, allHTTP200,
idle/unpaused/no recovery and unchanged classifiedfaults on sameboot. Exact
numeric/source/helper/model/admission/close evidence is under
`v162_intel_grouped_reads_and_down_schedule` in the root native receipt.
DOWNstride and combined original numerical/throughput gates remain pending;
keep them off in the PLEpair. Next separate source work prepares immutable
grouped GR/GDN/QSA status folds (defaultoff), preserving all original checks.
CanonicalVashti/API/installedservices unchanged; requiredVashti changes:none.
No30tokens/s, normalprose, independentCPU200, production or release claim.


### Completed Intel-only v162 schedule comparison,2026-10-11 18:00UTC

The remaining DOWNstride and combined groupedPLE+DOWNstride gates pass original
CPU13 cold/warm vectors/carries/greedy, real repair and full source/owner closure
on frozen66bdd0f/v162. Warm DOWN reports432layers/2160DOWN/4320source calls;
combined also reports8actual groupedPLE chunks. Original reference/tolerances
and all admission/owner limits remain unchanged.

Compiler-idle same-artifact normalcode200:9.886106control,10.022470groupedPLE,
9.567741DOWNstride and9.558671combined continuations/s, each199timed and
130/138drafts/69rounds/5repairs. DOWNstride reports3552layers/17760DOWN/35520
source calls; combined also reports74groupedPLE chunks. These are single ordered
measurements, not repeated gain estimates. Keep DOWNstride off for the next
internal speed comparison; retain the preliminarily fastest groupedPLE selection.

Final recovery18:00:12UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on the sameboot. Exact
numeric/source/admission/closure records join v162_intel_grouped_reads_and_down_schedule.
Next separate candidates are grouped status folds and packed-integer DOWN;
neither is physically accepted yet. Root owns shared GPUs and source isolation.
Required Vashti changes:none; installedAPI/services/canonical checkout unchanged.
No30tokens/s, normalprose, independentCPU200, production or release acceptance.


### Grouped-status integrated source gate,2026-10-11

Grouped status leaf/parent0cdb291 (originalb9ebef5) and public controllers453d2b7
(originalff2578e) are integrated after the completedv162 Intel-only schedule
comparison. Read the normative [status folds](kuno-native-public-grouped-status-folds.md)
and [controller contract](kuno-native-public-grouped-status-controller.md).
Policies remain independent/default-off; original equations, both whole fault
proofs, all-family commit and source/owner closure remain mandatory. Completed
counts publish only after accepted commits and join97GR/36GDN/12QSA per chunk.

Source gates:25focused status checks plus280resident regressions in isolated
parent,90controller/helperRust and33protected guard checks in isolated controller,
strict direct all-target Clippy/fmt/diff pass. Root integrated245textPython checks
and six fixed Intel-only status/PLE+status selector/cap labels pass; synthetic Xe
metadata used solely for source checks. Physical launch still requires fresh
privileged Xe accounting after paused/drained admission.

Frozenv163 source453d2b76bb74a03e5c6aacacc586d48f2fc806a5,
binarySHA256bddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a,
pure direct-device, is undergoing originalCPU13 acceptance. Same-artifact
compiler-idle Intel normal comparison remains required; source acceptance is
not physical numerical or speed acceptance. DOWNstride remains off. Separate
integer DOWN source/parent/controller work is isolated and unselected. Required
Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Completed Intel-only grouped-status comparison,2026-10-11 18:23UTC

Frozen453d2b7/v163 (SHAbddc741e8b7af8a45f00023e709c9008d5bdc6e25d3ad2289403c8cae289f67a)
passes unchanged originalCPU13 cold/warm vectors/carries/greedy/realrepair and
complete1381MAIN/42draft/432HOST-page closure. Warm initialprefill stays
false/allzero;8accepted statuschunks join776GR/288GDN/96QSA.

Compiler-idle same-artifact orderedcode200:10.096522groupedPLEcontrol versus
9.951850groupedPLE+status continuations/s, both199timed,130/138drafts,69rounds,
5real repairs. Candidate reports74accepted statuschunks7178GR/2664GDN/888QSA,
removing31228fixed statusdispatches; GPUwaits584versus618control,302readbatches
each. No throughputgain is established; keepstatus off in the next speedcandidate.
Retain source and independent composition for later work, with defaultoff.
Scoped phases/waits/maps overlap and are not additive.

Last recovery18:23:18UTC: B70baseline50688000B,Nvidia compute0,allHTTP200,
idle/unpaused/no recovery and no classifiedGPUfaults on sameboot. Full numeric
records, hashes/admission/closures: v163_intel_grouped_status_physical. Source
tests and limitations recorded there; no repeatedABBA/prose/CPU200/30t/s claim.

Separate integerDOWN leaf58b7f50(original730e135) and parent28d8ed6(original87a6fbf)
are source-integrated only. Leaf7focused/17existing oracle tests and strict
Clippy pass; independent read-only parent review finds no blocker. Parent
regression/Clippy and controller integration remain pending. Read both integer
DOWN normative contracts. Mandatory original physical accuracy/repair/closure
and matched timing remain required; no accepted new arithmetic or speed yet.
Required Vashti changes:none; installedAPI/services/canonicalcheckout unchanged.


### Intel-only integer DOWN integrated source gate,2026-10-11

Separate default-off integerDOWN is source-integrated: leaf58b7f50(original730e135),
parent28d8ed6(original87a6fbf), tested annotation/deltaee4f200(originaldce5b5b),
controller6d1bb47(original0ed0d59). Read the linked integer DOWN leaf/parent and
[controller](kuno-native-public-frozen-down-integer-controller.md) contracts.
The source keeps original experts/routes/packed weights, full-Q8 speculative
MTP scope and original reference/tolerances, with separate alternate math.
Intel combined admission adds exact65536MAIN+462848runtime while reserving
65536HOST separately;196608INFO is already included in runtime. Original144banks,
8GiBcache, fullQ8 owning inventory and512MiB driver margin remain unchanged.

Leaf7new+17existing quantizer/oracle checks, parent13focused/281resident/20source/
13MoE/4constructor/17existing oracle checks, controller99Rust/38guard checks
and strict direct-device all-targetClippy/fmt/diff pass. Parent counts overlap.
Independent read-only parent review finds no blocker. Root integrated250Python
checks and three fixed Intel-only helper labels/caps pass; synthetic Xe metadata
used only in source checks. Actual physical launches require fresh privileged
Xe accounting after paused/drained admission. No scope or profile fallback.

Frozenv164 source6d1bb47cd54937a2dd08cb05d11062313168c862,
SHAe5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b,
pure direct-device, is undergoing originalCPU13 accuracy/repair/owningclose
and physical recovery. Paired normal speed remains pending. Current trial
retains PLE/GUstride/sharedQ8Intel; original DOWNrows/stride and groupedstatus
stay off to isolate new DOWN arithmetic. Source success does not establish
physical accuracy or speed. Required Vashti changes:none; installedAPI/services/
canonicalcheckout unchanged. No30tokens/s or release acceptance.


### Completed Intel integer DOWN comparison, 2026-10-11 18:50 UTC

Frozen source `6d1bb47cd54937a2dd08cb05d11062313168c862`, v164, binary SHA256
`e5baf349d6d697e388ceff2d27a24bc4a09c99f3ef196505ee44c0a596b5b52b`, passes
unchanged original CPU13 cold/warm full vectors, carries, greedy IDs, real
repairs and complete owning closure. Warm first token reports actual integer
DOWN 48/240/48 and 480 source dispatches; warm full pass reports
432/2160/432 with 4320 sources. Original reference and tolerances are unchanged.

One compiler-idle same-artifact ordered code200 pair measured **10.023723 →
10.486196 continuations/s (+4.61%)**, each timing 199 real continuations,
130/138 accepted drafts, 69 rounds and five repairs. Integer DOWN reports
3552 preparations, 17760 calls, 3552 audits and 35520 original source calls.
Keep integer DOWN as a preliminary faster candidate with integer GU/stride,
grouped PLE and original full-Q8 shared head, all on Intel. Grouped status and
original DOWNstride remain off. Nvidia is deferred at the user's direction.
Single ordered timing is not repeated ABBA or prose acceptance.

All 1382 target owners/828997632 bytes, 42 draft owners and 432 host pages close.
Last physical recovery18:50:02UTC: Intel50688000B, Nvidia compute0, allHTTP200,
idle/unpaused/no recovery latch and no classified GPU faults on the same boot.
Full source tests, hashes, actual admission/selection and closures are recorded
in `v164_intel_integer_down_physical`. No independent CPU200/private/production,
installed defaults/API, 30tokens/s or release claim. Required Vashti changes:none.

Next isolated source candidates: transfer the existing exact packed IQ4 lookup
into actual Verify3 dense IQ4_NL/IQ4_XS (distinct immutable pipeline policy,
unchanged original addresses/scales/dots/reductions), and a finite target GPU
submission-backlog experiment retaining original submissions and proofs.
Neither has physical acceptance yet. Root coordinates GPU and CPU timing;
separate worktrees preserve source isolation. See the new normative contracts
once integrated. No frontend or API work is part of these candidates.


### Intel Verify3 IQ4 decoder completed; queueing candidate next, 2026-10-11

Frozen v165 source6266201b44faacac187db8e1f2353d35f7438b5a passes all original
operator controls plus9408 exact packed-register IQ4 values,12normal+4fault
Verify3 cases and12normal+42fault fused-finite cases on Intel. Default binary
SHA996786943fb7dfc6f41e1ae975cb0659833d8cd372b673d474cd7f0f83195ad0;
pure direct binary SHA5f86159edbad0652e77b60a19046d6ef5933e936e8677c65ec8dc3626e988719.
Original CPU13 cold/warm full vectors, carries, greedy IDs and real repairs
pass without changing reference80650af8 or ABS0.006/REL0.0003. Cold decoder
staysOriginal/zero; warm actual89first/801full selected dense calls. Full1382
MAIN/828997632B,42draft owners and432HOST pages close with physical recovery.

Compiler-idle same-artifact code200 ordered pair measures10.440895control vs
10.408871register decoder (-0.31percent), both199real continuations,
130/138accepted drafts/69rounds/five repairs. This shows no speed gain in this
single pair. Leave IQ4 registers off for the next speed candidate; keep its
explicit implementation and numerical acceptance. No ABBA/prose/CPU200 or
production/default selection follows. Latest recovery19:26UTC: sameboot,
Intel50688000B/Nvidia compute0/allHTTP200/no recovery latch/no new GPU faults.
See full source/actual counts/closures in v165_intel_iq4_registers_physical.

Next runtime candidate is [explicit bounded target backlog2048](kuno-native-public-target-backlog-2048.md),
leafc8bccc4(original525ce3b), clarificationad4af97(original45d944d),
controller77e421c(original3d94692), protected guard633ac25. It retains original
kernels, queue submissions and closing proofs; target is selected before model
construction for bothcold/warm while independent Intel full-Q8 draft remains512.
Its opaque driver retention is not an exact byte guarantee; fresh actual Intel
admission and unchanged512MiB margin/caps remain required.111controller/helper
checks (including8new),7exact-source state-harness checks, strict direct all-target
Clippy and259integratedPython checks plus6fixed helper labels/caps pass. The
harness limitation is recorded in the normative contract. Physical CPU13 and
same-artifact normal speed comparison remain pending. No GPU reset, installed
service/default/API/frontend change or30tokens/s/release claim. Vashti changes:none.


### Intel target queue comparison complete; placement next, 2026-10-11

Frozen v166 source0d35d5c7d1f7daf040622be61009f4770ce2616b,
pure-direct binary450e32a18ff3b9f3d985a27cb9550564c6b7c29c83bdfb09ba0c8aa3199da045,
passes unchanged original cold/warm CPU13 full vectors/carries/greedy/real repair,
actual target2048/minimum1 with independent draft512/minimum0, full1382MAIN/
828997632B,42draft and432HOST close and physical recovery19:41:31UTC.
Actual selected runtime was constructed before model allocation for both passes,
and remained immutable through completed close. No kernel or route changed.

Same-artifact compiler-idle ordered code200 pair10.451238original512 vs
10.482525queued2048(+0.30percent), both199 real continuations/130of138 drafts/
69rounds/five repairs, shows no material gain in one pair. Keep original512
for the next placement candidate. Global submission waits618→281 and
8.294789→1.668612seconds, while mapped-read wait2.472111→8.967485seconds;
work largely moved to a different wait boundary. These overlapping timers
are not isolated GPU kernel cost and cannot be added as a removable speed gain.
Full numerical hashes, actual policy/counters/admission and closing evidence
are in v166_intel_target_backlog_physical. Integrated strict release/direct
all-targetClippy passed. No ABBA/prose/CPU200/private/production or30tokens/s claim.

Latest physical snapshot remains sameboot Intel50688000B/Nvidia compute0,
allHTTP200, idle/unpaused/no recovery latch and no new classifiedGPU faults.
Nvidia remains deferred. Installedservices/defaults/API/frontend unchanged.
Required Vashti changes:none.

Next isolated source preparation is setup-only public profile-ranked empty-bank
fill: preserve all existing prompt/cold-pass residents, use the pinned public
Strata ranking only for empty slots, then freeze the original144/layer bank,
all512 actual IDs and sorted368 HOST complement. No private prompt or future
route profile is recorded and no capacity/arithmetic/route change is selected.
Read docs/kuno-native-public-profile-ranked-bank-filler.md after leaf integration;
its exact artifact/derivation, full owner/cap and CPU13coverage limitations are
authoritative. Protected scope263Python checks +three fixed helper labels/caps
pass; leaf/controller numerical/physical/speed acceptance remains pending.
Source CPU13 selected preservation may be all-hit: original diagnostic HOST
coverage remains a separate required control, never fabricated. A separate
borrowed CPU-batch leaf is being prepared to compute existing RAM-resident
expert weights with original Rust AVX2 math; it is not whole-model inference
or a speed claim. Root owns cache scheduling, shared GPU tests and physical
recovery; separate source branches retain ownership boundaries.


### Intel public ranking tested; CPU/GPU splitting next, 2026-10-11

Fixed profile source is integrated through e5f1689, after normative7c9da20
(original192d662) and leaf1d14a14(original53e889a). Read
[kuno-native-public-profile-ranked-bank-filler.md](kuno-native-public-profile-ranked-bank-filler.md)
and its controller contract. Frozen v167 pure-direct source is
`e5f168916cfeccca051f53b4c5ead8958fbe90f2`, binary SHA256
`03379e116acddc20c0bbf73f253913f8156145f592e8fbd5b4fd635edf1c0430`.
Both separate canonical and selected cold/warm CPU13 gates pass unchanged full
vectors/carries/greedy/real repair, original tolerances, actual original512
queue/full-Q8 Intel MTP, full1382 MAIN/828997632B,42 draft and432 HOST closures.
Selected warm publishes48 layers,4121 filler/2791 preserved residents and
actually serves9120 VRAM/zero HOST members; the separate canonical diagnostic
serves2590 VRAM/6530 HOST. Preserving the preceding cold replay's residents can
make that replay all-hit; it cannot replace HOST coverage or credit a profile
benefit on unseen normal generation. The warm12-token reference-check intervals
are6.647453 versus11.553927 accepted tokens/s, not normal200-token throughput.

The compiler-idle same-artifact ordered normal code200 pair is10.482354 original
versus10.478727 profile filler (-0.035percent), both199 real continuations,
130of138 drafts,69 rounds and five actual repairs. Actual VRAM/HOST members
are56344/46376 versus56079/46641; VRAM share54.852percent versus54.594percent.
Selected normal setup preserves5143 residents and fills1769 empty slots;
actual48-layer publication and unchanged original full512 map are accepted.
There is no measured gain. Keep Original bank filling for subsequent trials,
original512 queue, original IQ4 decoder, grouped PLE and integer GU/DOWN/stride
with shared full-Q8 head; Nvidia remains deferred by the user. Counters are
routed members, not unique weight reads or PCIe bytes.

Exact source, original numerical and setup receipts, aggregate tiers, complete
owner/closure and recovery are recorded in
`v167_intel_profile_filler_physical` of the native Strata numeric receipt.
82 distinct leaf/parent checks,120 controller/helper checks,263 protected Python
checks plus three fixed labels/caps, and integrated strict release/direct
all-target Clippy pass. Latest actual recovery is20:09:26UTC, same boot,
B70 used50688000B, Nvidia compute0, allHTTP200, admission idle/unpaused without
recovery latch and no new classified GPU faults. This is one ordered code pair;
no ABBA/prose/independent CPU200/private/production/30tokens/s acceptance.
Installed services, selections, public API, frontend and release are unchanged.
Required Vashti changes:none.

The separate borrowed CPU batch source is integrated as2e51bd5(original3212bc2),
with53 CPU expert checks passing and two unchanged timing probes ignored,
strict CPU/native direct all-target Clippy, formatting and independent review.
Read [kuno-native-borrowed-cpu-batch.md](kuno-native-borrowed-cpu-batch.md):
counts1..3 use original packed F32/AVX2 arithmetic, joined workers and sparse
unweighted original-rank outputs. This leaf is default-unused; arbitrary spans
do not authenticate payload origin. No connected CPU/GPU inference or speed
claim follows. Separate source owners are preparing an Attachment-anchored
read-only borrow of the SAME original Vulkan RAM and an actual one-layer CPU
versus HOST-GPU partial comparison with readback/encoding/upload/join included.
No duplicate model payload, disk read, new arithmetic kernel, cap relaxation or
count3 merge is authorized by an operator result. Initial fixed-layer synthetic
operator measurements may be unusually L3-hot; actual whole-model CPU13 and
normal timing remain later gates. Root controls physical trials and compiler
scheduling; agents use separate worktrees/caches and cannot independently use
GPUs, change services or release.
### Intel four-row physical checkpoint, 2026-10-11

The user selected Intel-only target and full-Q8 MTP; Nvidia remains deferred.
Frozen v174 source42df9070ef5af165667ba472c93053f66d92a759, pure-direct
binary SHA256f976442266b2d16b463a3211c35968bcae7cc3cee88ca89b5a7862b825abfada,
passes independent original-controlled Verify4 dense and fused operators:
20 numerical cases each,20 plain faults and70 fused faults, all10 formats,
both subgroup widths, unchanged independent CPU-F64 and scalar-GPU controls,
source/inactive/padding/sticky-fault checks and full operator zero audits.
These are operator receipts, not whole-model or performance acceptance.
The unchanged original constructor passes cold/warm CPU13 full vectors,
greedy/carries and actual rejection/repair, full target/draft/HOST closure.
Source compilation may overlap these correctness runs; their diagnostic
intervals carry no compiler-idle normal-performance credit.

The selected four-row constructor fails before its first original count1
prefill is accepted, with InvalidBufferLength. It has not accepted a third
MTP proposal or whole count4 chunk. Root and two independent source owners
trace the cause to GR's exact final-head proof: its reserved logits owner is
993280 logical words, but FinalAllRows and FinalFourRows still prove744960.
The fix must carry the actual constructor-fixed Owners capacity into that
proof while preserving original count1..3 arithmetic/spans and original
CPU13 reference/tolerances. Do not narrow the allocation, choose owner size
from active count, or weaken the proof. Retest a new frozen artifact before
normal same-artifact original/four performance comparison.

All four trials recover the same boot/kernel to B70 used50688000B, Nvidia
compute0, all three HTTP probes200, idle/unpaused admission without recovery
latch and no new classified GPU fault. Latest recovery22:32:25UTC. Exact
source/guard/helper hashes and content-free receipts are in
v174_four_rows_operators_and_initial_owner_failure of the native Strata
receipt. Native30tokens/s remains unachieved. Installed services/defaults,
public protocols and canonical Vashti are unchanged; Vashti changes:none.
