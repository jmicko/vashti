# Native text checkpoint, 2026-10-06

The user intentionally paused the goal again on2026-10-07 to conserve weekly
usage. The already-running benchmark/source-check block is finished; no GPU
job or compiler is left running and installed inference admission is resumed.
The objective remains incomplete. Use **10 decode tokens/s** as the agreed
working target when resumed and do not repeat Ollama baseline tests. Root owns
GPU/service operations; source-only agents use isolated worktrees.

## Clean paused checkpoint, 2026-10-07

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
