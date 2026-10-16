# Strata Rust whole-model port experiment

**Current CPU-only resume (2026-10-16):** source `37dced95f977428b2be4f616a7e04124bca3953c` connects
default native prompt GR row scales, direct BF16 norm/SiLU images and their
original down/injection/up consumers. Mix recomputes F32 values from residual,
scale and gamma.453 focused CPU cases /56 nonempty suites, strict lint and
formatting pass. Read the [current receipt](kuno-strata-rewrite-completion.md#native-prompt-gr-scales-and-packed-image-consumers-2026-10-16)
and [checkpoint](kuno-native-text-checkpoint.md) for the exact current frozen
artifact/shader evidence, substitutions and next gates. Normal artifact
`strata-packed-gr-ordinary-20261016/` has SHA256
`fc0d1c655302c774427905e36932797c5c45eb3c02b9b2baf79403be8e198d7b`;330 offline shader variants pass. Fixed prompt roots stay
unchanged; no physical memory/speed or whole-model correctness claim. Whole
rewrite incomplete; no live speed trials. Historical notes below are superseded
as current direction.

## Active direction — finish the complete Rust port before speed testing (2026-10-14)

The user replaced the old performance goal with an explicit **complete connected
Rust rewrite first** goal. This section supersedes every historical next-step,
artifact selection and benchmark instruction below. Do not run throughput trials
or operator timing until the completeness audit in
[kuno-strata-rewrite-completion.md](kuno-strata-rewrite-completion.md) passes.
Interim checks may compile, validate shaders, check numerical/state behavior and
prove whole-model inference completes and releases its resources. Incidental
execution durations must not select, disable or abandon a partial implementation.
Keep coupled upstream changes together even if an intermediate build is slower.

Working branch is `codex/strata-rust-port` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Starting runtime source is
`7455b6f851ea4868d4d4af8f869fc5c281700cbb`. Canonical Vashti and installed binaries
are unchanged; this work has no API/UI contract change. Pinned MIT upstream remains
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`. No linked foreign inference engine is an
acceptable substitute for the Rust rewrite.

The previous pass added cardinality variants (`903dc9d`, `bd86885`), retained
them as an optional experiment (`cb857a8`), and added one-round Vulkan operator
timestamps (`7455b6f`). Frozen v287 uses `bd86885`, SHA256
`2066c090ea7a6a6145d97bcf98695e51cb468e4633345f844f3054030999200c`.
Frozen v288 uses `7455b6f`, SHA256
`731acece3c8ecfd71405a26f63d314d6229df6eb8d9d890fbf9388ad6959885c`.
Their historical public code/prose executions completed and restored GPU/admission
state; they are **not a completion audit or a reason to reject upstream behavior**.
The timestamped round serializes its nodes and is diagnostic only. Full original
regression after the v288 vendor changes remains outstanding.

Correction to older notes: upstream `row_entries` uses **4/2/1 chunks** for formats
with `Multi<TY>::has`; it is not scalar-only by default. `native_gu_kernel` separates
gate/up output roles before activation. Our prior width-per-launch experiment and
combined GU activation do not yet establish that scheduling parity. SYCL QFUSE and
HC_Q8 are explicitly unsupported in this pinned Intel source; they are not required
Intel port components merely because CUDA has them.

The first connected implementation block ports native separate gate/up/DOWN
4/2/1 chunks and activation into the selected Intel core. Gate/up borrow existing
private PART rows; activation consumes and clears them before DOWN. HOST staging
runs once, and eight additional INFO owners/layer are charged. Runtime source is
in `strata_expert_roles.rs`/`strata_expert_fused.rs`. All 158 WGSL/SPIR-V sources
validate (105 physical-pointer checks), strict core Clippy/build pass, and original
direct-device regression passes 1,346/1 ignored. **No GPU inference or speed trial
was run; full rewrite and numerical execution acceptance remain unfinished.**
The protected runner requires `--execution-check` and rejects operator profiling.

A second connected source block now selects native GR pending-write norm,
sliced BF16 down/injection, fixed reduction and up/mix/commit together. PLE and
compact final/no-head prompt boundaries flush pending writes; the enlarged UP
scratch owner adds exactly 524,288 private bytes to both family and model quotes.
All 163 shader sources validate (105 physical checks), release/core Clippy pass,
and 12 core GR owner/layout tests pass. The full core unit run is **not green**:
1,148 pass, 200 fail, one ignored, including old 144-slot and pre-port-budget
assumptions that still need triage. No GPU inference/speed trial was run. Read the
completion guide's **Connected GR implementation block** for exact substitutions,
logs and unfinished numerical acceptance; do not interpret these checks as a
complete rewrite or model correctness proof.

Global byte-prefix planning and its owning loader now compile; current inference
has **not yet switched** from uniform banks. The actual public profile selects
11,570 pairs within the same expert-byte budget, with 160..342 residents/layer.
The plan defines complete HOST complements and bounded native blob chunks. Root
allocation, imports, BDA pointer tables, routing and adaptive exchange must now
be connected together, replacing the old banks rather than allocating both.
Read the completion guide's **Global byte-prefix and owning-loader block** for
exact code, tests, binding/ownership limits and required continuation. Two
host-only allocation tests and strict core Clippy pass; no GPU load or speed
trial was performed. The earlier core-suite failures remain unresolved.

A bounded 240-minute sudo refresher is now at
`/media/jmicko/faststore/kuno-development/strata-global-port-sudo-20261014.pid`.
Inspect the actual process and cache before protected work. If sudo is unavailable,
leave a documented checkpoint and stop; do not declare the rewrite complete.

## Current checkpoint — packed shared/MTP and projection reuse (2026-10-14)

The goal remains **active and unfinished**: implement the connected native Rust
Intel core and reach at least30 accepted continuations/s. Keep the original scope;
no API, UI, release or deployed worker change belongs to this pass. Worktree:
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/strata-rust-port`. Canonical source and installed services are unchanged.
The detailed older checkpoints below are historical, including old source hashes,
geometry and statements about stopped jobs. Use this section and the aggregate
receipt's `native_intel_packed_extension` for the current state.

Latest frozen v285 source `e5bd05d429c0f37dceb6a54735ad0b9ea39c424e`, SHA256
`1319a190a246ab7adcff7055629021db4672a128ddd5e944de3350189f5a69d2`:
**14.512497 code /12.781750 prose timed continuations/s**, Intel B70 only.
Code emits460 including EOS; prose512 at cap. This is not a consistent speed gain
across both prompts, and30 remains unmet. Keep v282 (14.625229code) and v260
(14.300613prose) artifacts as fallbacks. These are single runs with differing
sequences/lengths, not exact-output speed ratios or private-worker acceptance.

The connected changes retain native Vulkan graph replay, physical source pointers,
selected original HOST expert staging, real GPU routing, authentic private owners
and target-greedy verification:

- `0dad3af`: shared gate/up reuse one Q8 input; shared down, scalar MTP dense and
  its authenticated indexed English head use the six original packed formats.
  MTP gets a real charged32768-byte scratch owner. Final target vocabulary stays
  complete; only draft proposals use the original public English subset.
- `50a112e`: dense Q8 preparation uses subgroup32 contiguous loads/reduction and
  packing, preserving ties-to-even codes/F16 scale/zero second plane; fixed packed
  dot loops are expanded. No new public model conversion or target allocation.
- `e5bd05d`: GDN/QSA consecutive projection bundles reuse one immutable input
  epoch, validating common input spans/widths and distinct output/fault/scratch
  owners. Unsupported BF16/F32 projections keep their ordinary arithmetic.
  Receipt adds `packed_dense_input_prepares`; the older attention-only counter
  was renamed `packed_dense_dispatches` when shared/MTP coverage was added.

| Frozen build | Code t/s | Prose t/s | Meaning |
| --- | ---: | ---: | --- |
| v283 (`0dad3af`) |13.629725|13.002690|Packed shared/MTP integrated, below v282 code|
| v284 (`50a112e`) |13.602412|Not run|Parallel quantizer alone yields no established gain|
| v285 (`e5bd05d`) |14.512497|12.781750|Reuse helps code in this trial; not prose|

v285 code verifies in18.732054s and drafts in7.217603s. Prose verifies in27.251253s
and drafts in7.246658s; its verification windows increase relative to v283.
Generated vectors/tokens have not been independently proved identical across
these builds. Confirm packed input epochs and numerical leaf vectors while
profiling; do not mistake changed output/acceptance for a pure throughput gain.

Checks on v285 source: strict `strata-intel-core` Clippy, fmt/whitespace,
65 WGSL/SPIRV shader variants including13 physical `spirv-val` modules;
original `direct-device` regression1346passed /1ignored. Full Strata-feature
unit suite, independent whole-model state oracle, arbitrary/private text worker,
answer-quality acceptance and long soak are **not established**. Successful
real greedy verification/finite public text does not replace those gates.

Each completed run proves target full-owner closure1565 buffers /1,064,534,016
bytes, MTP44 /11,763,712 bytes, original432 host-page detachment/live backing0,
B70 idle50,688,000 bytes, NVIDIA compute0 and classified GPU faults0. Normal
Kuno/Ollama/Vashti HTTP200 and admission idle/unpaused/no recovery latch return.
No device reset/rebind/reboot or installed binary/service/configuration change.
No content is saved; fixtures use these recorded **public** prompts, seed424242,
temperature0, no penalties, thinkingfalse and512max outputs:

- Code: “Write a Python function that merges two sorted lists of integers into
  one sorted list. Explain its time and space complexity.”
- Prose: “Explain how a hash table works, including collisions, resizing, and
  average lookup complexity.”

Reproduce using the protected runner/builder instructions in the native-core
receipt below, with build `native-next-v285-strata-projection-bundle-build-20261014`
and fresh output directories. Do not compile during GPU timing. Root/swap0/core0
protection is required. Never reset/rebind the Intel card to obtain a result.
The source clone's known `build`, `sycl/build`, `build-sycl` and `build-intel`
paths remain absent; the original comparison evidence and pinned MIT source
are preserved. Alternate Q2 sidecar `kuno-strata-mtp-q2-20261009/mtp-q2_0.gguf`
remains889,017,216 bytes.

Historical next-step proposal (superseded above): pinned Intel Strata uses separate
GU roles and 4/2/1 `row_entries` accumulation (`iq_kernels.dp.cpp:1784`,
`1815`, `4182`; `g_split_multi` is opt-in). Kuno's current GU fuses both roles
and four member accumulators. Lower register pressure/occupancy is a plausible
advantage, not yet a measured bottleneck. Add public-only operator GPU timing
and port that scheduling coherently with actual owner/budget/graph dependencies;
compare fused hyper-connection read stages too. SYCL's `fused_gr.dp.cpp` explicitly
says QFUSE and HC_Q8 are not built, so do not present those as existing Intel
upstream behavior. Unequal global expert allocation, native prefill parity,
eight-row core staging and full numerical acceptance also remain unfinished.

No benchmark/compiler remains after these trials. A bounded90-minute sudo
refresher may still be live; inspect its PID file and actual process before reuse:
`/media/jmicko/faststore/kuno-development/strata-intel-core-continuation-sudo-20261014.pid`.
Goal continuation retains the full30t/s objective; if cached sudo is lost, leave
a tested documented checkpoint without privilege workarounds or a completion claim.

## Connected implementation

The separate `kuno-strata-rust` target/`strata-rust-port` feature reuses the
already-written Rust GGUF/tokenizer,48-layer Qwen graph, PLE, GDN, QSA and gated
residual operators. Rust and generated WGSL/Vulkan execute inference; no C++,
Strata, llama.cpp or SYCL inference engine is linked.

- One Q8_1 activation plane with F16 effective scale; original IQ2/IQ3/IQ4
  packed target weights; grouped GU/DOWN, cooperative activation staging,
  256-thread32-row workgroups, integer dots, subgroup reductions and FMA.
  Counts1..3 and4 use the same policy.
- Parallel GPU top11 routing, real GPU argmax and compact163888-byte carry
  packet. The complete target vocabulary is still evaluated; logits are not
  copied in full to the CPU. Final real outputs must be finite.
- Q2 sidecar dense BF16/Q6 weights converted to Q8 in Rust/RAM at load. Q2
  experts and IQ4 embedding remain original. No converted model file is saved.
- Strata's106299-token draft vocabulary, sorted without adding/removing IDs;
  only proposal logits are pruned. Full target verification remains authoritative.
- Request-local suffix matching with real first-MTP agreement and measured
  draft acceptance/cost policy. Lookup histories and cache maps are zeroized.
- Adaptive expert tier using frequency/recency and decay0.7, every4 controller
  rounds, at most2 swaps/layer. All resident slots are eligible. Evicted original
  weights stay in the host complement; no disk reread is required for promotion.
- The newest source uses exact persistent bank allocations and192 resident
  experts/layer:96+96 VRAM,144+144+32 host. Both host maps and generated source
  plans derive from shared constants; the original feature-off geometry stays
  144/72. Actual six distinct bank roots are checked against the exact quote.
  Public RAM complement drops from36147609600 to31432704000 logical bytes.
- GPU-driven two-way exchange of original resident/host cells in one joined
  queue transaction. It has no activation/prompt binding, no CPU weight staging
  and no temporary model copy. The new logical bijection publishes only after
  every role/swap completes. Ordinary external-source bindings stay read-only;
  the isolated port's owning transaction alone uses this explicit raw device
  write path. **Executed successfully on B70 in v222–v224 code and prose runs.**
- Accepted-prefix repair retains original GDN projected inputs in five owning
  pages per layer (34,209,792B total). Partial acceptance replays convolution
  and recurrence only, retains accepted QSA cells, reconstructs PLE history,
  clears rejected suffixes and commits all-family cursors after a fault fence.
  v223/v224 perform zero full target repair forwards. Inputs/stamps remain
  transient and are erased on close. v224 uses ThreeStreaming for short replay.
- Request-local adaptive MTP lengths estimate conditional prefix acceptance
  against observed costs; it chose almost exclusively four rows in these runs.
  Do not claim adaptive length provided the measured speed improvement.
- Intermediate numerical scans use Strata's leaner policy; final checks and
  complete private owner erasure remain. Public runs permit up to512 outputs
  and respect EOS; the original engine still caps this fixture at200.

Pinned upstream: MIT Strata `fb58e0dbc8399662c0e47c76578c6e878b14f6cf`.
`strata-port-NOTICE.txt` and existing ggml/llama.cpp attribution are retained.

## Actual combined results and failures

All measured rates below are **real timed continuations/s**, excluding the first
confirmed token. Fixed public code fixture,temperature0/seed424242, Intel B70.
Different MTP arithmetic/window lengths can produce different greedy trajectories;
these are not an exact same-output Ollama comparison.

| Candidate | Scope | Rate | Evidence |
| --- | --- | --- | --- |
| v205 Q2 | code200 |8.220500|135/189 accepted drafts,23 repairs |
| v205 Q8 | code200 |8.672648|137/183 accepted,21 repairs |
| v207 Q8 | code200 |10.800044|147/155 accepted,5 repairs; no suffix window actually selected |
| v213 Q2 combined | code200 |9.175268|145/161 accepted,7 repairs;144-slot adaptive cache, compact carry, runtime conversion/vocabulary connected |
| v214 Q2 combined | code512 cap,362 emitted at EOS |7.976630|108 rounds,36 repairs;8.823267451s cache adaptation |
| v215 Q2 batched transfers/checks | same cap,362 at EOS |8.458796|42.677466683s timed,6.529365052s adaptation |

v213 cache performed1246 swaps,2549555200B moved,4.467331086s adaptation.
Its resident route fraction was70888/108960 (~65.1%). v214/v215 performed2492
swaps,5099110400B moved. v215 draft5.224239850s,verify24.334071779s,
repair6.570197023s. These completed runs pass internal owner closure and actual
process/GPU/service recovery. **30tokens/s and complete upstream feature parity
are not achieved.** The last successful broader-port artifact is v215 from
`c1b334c3a5e471e6342e5b9cf5900e520704f864`, SHA256
`2d804b91b493466a1987f68b11ad5780e41037ed4937b021731c85c768336f79`.
It uses144 slots and CPU-staged exchanges, not the newest192/direct-exchange code.

Important failures fixed this pass:

- v204 Q2 exact-Q8 admission rejection (`27f14fb`); v204 Q8 receipt permissions
  (`ec21023`). Missing v204 timings must not be inferred.
- v208/v209 shader launch failure hidden behind retained metadata. Router used
  WGSL-reserved `active`; renamed `live` in `9815e85`. Full new shader coverage
  now includes routers, packet and exchange, plus20 projection variants.
- v212 held an outgoing readback lease after wiping it, blocking the sanitizer.
  `32baf8a` explicitly drops it before mutation; v213 then succeeds.
- v216192-slot prototype retained a144-slot whole-model constructor. Connected
  `0f4c623`; no model execution acceptance for v216.
- v217/v219 exposed stale router table extent and metadata budget. `93b4178`
  sizes the table to its bank; `3e09fb6` raises only the isolated port's per-layer
  metadata budget to64KiB/4MiB aggregate. `c627327`/`21ee0c0` preserve original
  failure diagnostics instead of masking them with a poison notification.
- **v221 reached successful cold prefill,432-page host materialization, frozen
  GPU routing and decode startup, then rejected the first four-row DOWN launch.**
  Its four-row launch wrapper still contained two literal72/144 shard tables.
  `d682e04` connects both to the selected source partitions; `1caf271` passes
  source checks. **No GPU rerun after this fix**, honoring the user's pause.
  The direct GPU exchange therefore has no hardware/timing acceptance yet.

The last tested but failed v221 artifact is source `3e09fb6789f1bd92d5c543671b72b2136c32375f`, SHA256
`b6fc683d5065626c8842ea894fb613a4bc3161254b07402507e224f454ffe9a1`, under
`native-strata-port-v221-build-20261013`. Do not select it as a working artifact
or silently repeat it; it lacks the fixed four-row guards.

## Checks and actual stop receipt

- Latest source: Rust port Clippy `-D warnings`, fmt/whitespace, and original
  `kuno-text-lab` direct-device compilation pass. Saved logs:
  `strata-port-final-clippy2.log`, `strata-port-original-check.log`.
- v221 full new-controller WGSL validation (including exchange) plus20 projection
  WGSL/SPIR-V variants passes. A fresh release artifact from latest source still
  needs its own full shader check. Compiler acceptance is not hardware execution.
-1346 original-engine tests/1ignored passed at `f0ed07a`, **not current source**.
  Full regression and representative code/prose runs remain outstanding.
- v221 failure: internal private close reports complete=1; actual cleanup passes.
  Fresh read-only16:29:23UTC receipt `native-strata-port-final-pause-20261013.json`:
  admission idle/unpaused/no latch, all three listeners200, B70used50688000B,
  Nvidia compute0 (679MiB desktop), classified GPU faults all0, unchanged
  boot`b32654b7-a49b-493d-8305-46da15b6eac6`/kernel7.0.0-38-generic.
- Recorded sudo refresher3421970 terminated and exited. No benchmark/compiler
  job or queued GPU work is left. Do not refresh credentials automatically.

All development artifacts/receipts above live under
`/media/jmicko/faststore/kuno-development`. Benchmark logs omit generated text;
fixed public output is displayed live only. No private input path or content
persistence was added.

## Resume only after explicit user instruction

1. Read AGENTS/coordination, this checkpoint and current git status/log. Preserve
   canonical Vashti and inherited `text/__pycache__/`; do not restore the lab to
   main or overwrite newer work. Original accepted engine remains6792453.
2. Build a fresh immutable `kuno-strata-rust` artifact from latest committed
   runtime source, e.g. `python3 apps/kuno/benchmarks/text/build-strata-port.py
   native-strata-port-v222-build-20261013`. Verify SHA/source; run its
   `--validate-port-shaders` and registration-only preset before GPU execution.
3. With the user's GPUs free and cached sudo available, read the Demons shell
   before typing. Run a fresh protected code/Q2 fixture through `run-strata-port.py`.
   It uses MemorySwapMax=0/LimitCORE=0, checks actual idle memory, pauses admission
   and restores/physically verifies services/devices in finally. Never benchmark
   in an ordinary swappable shell, reuse an output directory or reset a GPU.
4. First required acceptance is the complete192-slot/direct-exchange run after
   d682e04. Check that actual exchanges occur, real output is readable, target
   verification and all owner closes pass, and actual memory/faults recover.
   If it fails, preserve the real cause and fix the combined path; don't switch
   back to isolated original-precision kernel trials or claim performance.
5. Only after that works, evaluate initial ranked residency, faster accepted
   prefix rollback and a6/8-row verifier. Those, XMX prefill, compact long-context
   KV and the general private-input worker remain unfinished. The current bank
   count is fixed192, **not auto-sized**. Do not call the whole upstream rewrite
   complete. Request-local adaptive residency is implemented; inference history
   must never be saved in Kuno.
6. Finish appropriate original-engine regressions and representative code/prose
   acceptance before selecting an artifact. Keep API/UI/release work elsewhere.
   On losing sudo, document a clean checkpoint; avoid privilege workarounds.

## Superseding accepted combined evidence after the explicit resume

All rates use real timed continuations, first confirmed token excluded; public
fixtures, seed424242,temperature0, B70, original DASLab IQ3_S target and original
Strata Q2 MTP (dense runtime conversion). Different approximation policies may
produce different greedy trajectories; no exact same-output Ollama claim.

| Artifact | Source | Code | Prose |
| --- | --- | --- | --- |
| v222 |146e60d|10.718328 (362EOS)|9.979851 (512cap)|
| v223 |fa27e39|12.261074 (362EOS)|12.277046 (512cap)|
| v224 |0f4bfcc|12.302385 (362EOS)|12.241920 (512cap)|

v222 proves all192 slots and real GPU/RAM exchanges:2487 code swaps;775872094ns
cache work. Physical resident banks18,859,622,400B,288distinct root buffers,
zero admission padding/headroom. Cache `public_bytes_moved` records **one leg's
payload**; actual exchange traffic is twice that count (upload plus download).
Do not call it the sum of bidirectional traffic.

v223 code:36 state-only repairs,0 full target recomputations;2.081341754s repair,
29.442771630s timed. Prose:98 state-only repairs,0 full recomputations;
41.622389380s timed. v224:29.343904509s code,41.741818165s prose. The adaptive
controller used108 four-row code windows; prose175 four-row,one three-row and
one two-row window. Streaming replay/length policy are performance-neutral here.

Both fixtures for v222–v224 completed erasure/source closure and restored
Kuno/Ollama/Vashti, unpaused admission, B70used50,688,000B and Nvidia compute0;
classified kernel faults unchanged. v224 byte totals:1562 MAINowners,
863,207,424B, including retained replay inputs. Actual private-state vector
parity against an independent full model is not claimed. PLE accepted-prefix
history tests pass2/2; later combined regressions are still required.

Frozen artifacts under `/media/jmicko/faststore/kuno-development`:
- v222 SHA12d6bbcc452e74b583545a6dace04de55f9efc5ab3358e170d94cbc68b4f3c04
- v223 SHAc050c9c82bc8e7c347dcc0f5ccddafb52564354cee368d10a7fc471163acd491
- v224 SHA5375689c1515b2590c39f2fbf73893b7eba5036a140994367dae46101a650f1d
- v225 SHA82c76739ba715e9b1440ac2d5cdf94af64c5afe0532ab2f278ffe685ec4ddb58

The old paused/v221 stop receipt and commands above are historical. Current
work is authorized; use fresh artifact/output paths and read the shared shell
before commands. Bounded refresher3498864 runs at most240minutes from16:45UTC
Oct13, pidfile`strata-port-v222-sudo-keepalive-20261013.pid`. Stop it on finishing.
Do not infer privilege availability from old receipts. No GPU reset is authorized.

## Queued MTP and optional kernel checkpoint (18:17UTC October13)

Latest accepted two-fixture runtime is01e74a7 (v231), artifact
37e38a94efce573412d609b31ddbbed07e91c890c529d586c9408f5d329cd744.
Code:12.755259continuations/s,362EOS,28.302051784s. Prose:
12.598536continuations/s,512cap,40.560267190s. Real output/target verification,
private/source closure and actual process/GPU/service recovery pass. Cache
traffic remains two-way with one-leg counters.30tokens/s remains unachieved.

`ResidentMtp::draft_window` queues1..3 real provisional cells in one exclusive
call and returns IDs only after its final fault/readback fence. It preserves
all original forward/head/full private checks, uses actual prior GPU hidden
and token for following cells, and retains the existing target verification
and accepted-cache repair. A potential suffix match uses the old scalar route;
no suffix behavior was silently removed. It does not promise the driver never
submits/waits mid-window. There is one explicit final proposal read; original
per-cell waits/readbacks are removed inside that window.

The extra packet is one fully admitted/physically authenticated32KiB private
owner,4logical words, included in failure wiping and complete arena erasure.
MTP close now proves43owners/11,730,944B in the port only. Original engine
remains42/11,698,176. Per-window INFO debt is precharged and retired only after
actual completion. Code89queued windows; explicit MTP draft work3.813815385s
versus v2244.850723538s. Prose draft6.285716461s versus v2247.824939644s.

v228 combined integer head/fused source:11.792263code; prose roughly12.25.
Source ddc6bd3, artifactca75de66d0d5ee8d052dff924b7d0e1f98cfd7b03de9a3a0df024eb4b6ec6194.
Both normal closures/recovery pass; source-specific counters correctly prove
the expert dispatch reduction from10to2per layer. v230/a8731e7 isolates the
changes using the protected runner's `--integer-head` and `--fused-experts`:
head-only12.236724code, fusion-only12.189794code. They are **optional and off**
by default. No repeatable speed gain is claimed. Vocab head still evaluates
all248320target rows; its one-Q8-plane math has a distinct private scratch
clear/proof. Fused projection retains all15original public role pages and
requires14storage buffers/15bindings. No source repack/allocation occurs.

v232/d41b2c3 subgroup-register router is undergoing code GPU validation.
v233 adds explicit candidate eligibility (including extreme-negative ties)
and aligned vector reads of complete original packed expert role roots.
Those are not yet accepted as faster or complete. Compiler validation includes
49projection/head/fused WGSL/SPIR-V variants plus controller shaders.

Original engine regression after the ddc6bd3 port changes:1346pass/1ignored.
An outdated source-string guard test expected a literal144, while the unchanged
feature-off geometry is selected byBANK_SLOTS; updated it in df8250f. This
passing regression predates queued-MTP/v233; final latest regression/Clippy
remain outstanding. Latest hardware receipts remain authoritative, not these
old original numerical gates. No installed default, API/UI or canonical edits.

## Current measured limits and exact next work (October13, active)

v231:12.755259code/12.598536prose. v232 subgroup candidate cache:12.796036code,
neutral in whole generation. v233 aligned vector reads:12.386280code,slower;
therefore wide reads are optional. v234 optional16-lane expert dots:
12.562354code,neutral. Both launch-record extents, row grids, block iteration,
subgroup reductions and pipeline keys change together; no scope or source-
packing shortcut was taken. v234 default prose:13.000929,512outputs,
39.304884028s,95state-only repairs,0full repairs. Code profile is instrumentation
only, **not a normal throughput result**. All completed trials restore GPU and
services with unchanged faults and full43-owner MTP/1562-owner target closure.

Currentv236 optional `--cache-eight` keeps192slots and all math, raises per-layer
promotion bound2→8 and raw joined transaction bound288→1152role cells. Cache
plan envelope now includes16KiB transient planning space, and policy flags are
read once rather than per kernel. Code13.004114,362EOS,27.760445649s;
8337swaps,16,965,324,800B one-leg payload (~33.9GBbidirectional). Host routes
38029/(38029+169331)=18.34%, down from28.35% withtwo swaps. Verify18.632647518s
versus v23221.443143286s; adaptation2.766326764s versus about0.8s, so the net
speed gain is modest. Exact-byte two-way exchange means no quantization change.
Cache-eight remains optional until representative acceptance supports a default.

v234 round8 target GPU profile:135.559ms sum of kernel timestamp intervals.
GU42.297ms andDOWN26.933ms; their fixed five-shard dispatch order infers
**mapped HOST52.153590ms / VRAM17.076773ms** acrossboth. No request-derived IDs
were used or saved to classify them. Router6.446ms; dense finite16.374ms;
Q6dense8.926ms; GDN7.948ms; BF16down7.010ms. This profile is one selected window,
not whole-run average and not CPU wall time. Compared withv206(144slots/Q8draft),
router timestamp dropped16.377→6.446ms; this is not an isolated A/B and did not
translate into a clear normal code rate gain. Saved `public-kernel-costs.json`
is beside the v234 profile receipt. Avoid another speculative tiny kernel trial
without accounting for HOST/kernel/controller costs.

Most useful next work:
1. Cache-eight prose is complete and its result/cleanup is recorded below.
2. Generalize to6/8-row verified suffix windows coherently. Private model
   workspaces already reserveMAX8, but kernels/MAP records/packet/policy still
   assume4. Do **not** simply raise a count: vec4 reductions, GUframe/record
   overlap, DOWNscratch extent, router top11/audit geometry, token carries,
   accepted-prefix commit and all-family completion must agree. Keep real
   target verification and adaptive suffix/MTP choice. This is missing from the
   requested broad Strata port and may amortize much more than the neutral
   scheduling candidates. Longer suffix windows need no7-cell MTP prediction
   when the longer suffix itself is the proposal.
3. Consider selected HOST-cell staging into GPU scratch or better promotion
   scheduling. Current mapped-HOST expert work dominates its VRAM counterpart;
   copies also cost real PCIe traffic. Existing `frozen_host_staging.rs` is
   **30group/three-row,32MiB-per-role**, incompatible with current integer/four-
   row path. Generalize its whole ownership/quote/group-index remapping before
   using it; never turn on an old flag and bypass its rejection. Bulk staging
   is a hypothesis, not a promised speed gain. Current transfers fence the
   whole boundary; no safe concurrent prefetch implementation exists yet.
4. Remaining dense MMVQ/XMX prefill, flexible residency and private-input worker
   stay unfinished. Original regression/Clippy are evidence of isolation, not
   full model/quality/API/release acceptance. Do not broaden into API/UI work.

Sourcev236:strict release port Clippy and original direct-device regression
1346pass/1ignored. v234 compiler checks include49variants ineach default8,
16-lane andwide configuration plus controller shaders; v236 default compiler
and synthetic parser check pass. No GPU reset/rebind/reboot is authorized.

Continue using fresh immutable build/output paths through `run-strata-port.py`.
Options:`--integer-head`, `--fused-experts`, `--wide-experts`, `--expert-lanes16`,
`--cache-eight`, `--profile`; all except normal queued MTP/router/cache behavior
are optional and off. Profile changes timing. These are fixed-public laboratory
selectors, not a Kuno/Vashti configuration contract. Model/artifact identities
and numbers are in aggregate receipts; stdout text is not saved.

Shared shell sudo refresher3498864 is still bounded until20:45UTC(15:45Chicago)
Oct13; pidfile`strata-port-v222-sudo-keepalive-20261013.pid`. Do not extend it or
infer live privilege from these notes. Stop it at actual work completion/pause.
If authorization/cache disappears, finish a clean documented checkpoint, restore
services/GPUs and stop rather than attempting privilege workarounds. Goal active;
30 remains unmet. No installed binary, consumer change or release acceptance.

### Completed v236 cache acceptance and current clean boundary

Optional cache-eight prose passes13.189027continuations/s,512outputs,
38.744328668s,0full recomputations. Host37034/338400routes=10.94%;9913swaps,
20,054,476,800B one-leg transfer (~40.1GBbidirectional); adaptation3.197280281s.
Code13.004114,prose13.189027. This is a modest net improvement, not30.
Default remains2promotions/layer; optional8has measured representative acceptance
but is not a deployed Kuno setting. Full private/source closure and actual
memory/fault/service recovery pass for both. No GPU benchmark/compiler job is
left at this checkpoint. Bounded sudo refresher remains for the active goal.

v236/3e01731 final strict Clippy and full original1346pass/1ignored pass. Exact
fresh cleanup receipt will be`native-strata-port-active-checkpoint-20261013.json`
at the development base. Keep the goal active; no user pause was requested.
The next high-impact work queue is6/8-row suffix verification or an evidence-
based expert-transfer/cache design, not more tiny unprofiled dot variants.

Concrete8-row touch points: `daslab_language_probe.rs`/`strata_port.rs` controller
and cost/lookup windows; model/MoE policy guards and frozen sources; both
`resident_batch_router_gpu_four` Rust/WGSL frame/top11 layouts; GUquad member
expansion/record capacities and `record_dispatch_view`; DOWNframe extent/quote;
`next_verify4_dense`/GR shared projections; `strata-packet.wgsl` and owning host
packet/carry bounds; accepted-prefix controller/replay (ThreeStreaming is valid
only for at most3accepted rows; longer prefixes need a suitable recurrence).
Most private graph workspaces already haveMAX8capacity, but this is not proof
that the four-row kernels or authority checks admit8. Preserve old engine and
exact source bytes. Build one coherent candidate then validate/debug whole-model
execution, as requested. Do not silently fall back to forced experts or tokens.

Fresh final-secondary-state receipt confirms Kuno/Ollama/Vashti200, admission
idle/unpaused/no recovery latch, B70used50,688,000B, Nvidia compute0, and all
classified GPU faults0. No benchmark/compiler job remains. Canonical checkout
is still clean. Source changes stay in the Kuno laboratory worktree.

## Wider verifier completed; latest paused handoff (2026-10-13)

This supersedes preceding active-goal and queued six/eight-row notes. Resume only
when the user returns the GPUs and resumes work. Both protected v252 trials
completed before this pause; no additional GPU test was started afterward.

### Implementation and identities

Changes after `605c39b` through `2780e7a` add the isolated
`strata-eight-window` feature. The original four-row feature remains selectable.
`strata_port::verify_capacity()` governs controller, router, all-family permits,
private packet extents, accepted-prefix snapshots and dense/MoE kernels. New GU
and DOWN kernels share weight decode across up to eight real rows. Router frames
carry 80 routes; DOWN separates 80 route flags from activations with a 128-word
header. Its 128 KiB owner is fully audited. Wider dense projection handles eight
accumulators; longer GDN prefix repair uses the existing chunk recurrence rather
than invalid short streaming replay. PLE accepted-prefix history now admits
2..8 rows with sequential-history tests. Queued MTP permits seven proposals.
No replacement experts, forced tokens or weakened fault rejection are used.

Key correctness fixes: `a2686f5` DOWN flag/header separation; `b988d3b` complete
80-route source proof; `a67521f` full scratch audit/pending acceptance;
`2780e7a` PLE prefix authority through eight rows. Earlier v237–v251 whole-model
failures are not working artifacts. Static failure diagnostics were improved
without recording request content. `383248a` removes duplicate unselected shader
helper modules for strict Clippy; measured binary remains from `2780e7a`.

### Measurements

Fixed public code fixture, seed 424242, temperature 0, original DASLab IQ3_S target
and Q2 MTP on B70, optional eight cache promotions per layer, output cap 512:

| v252 mode | Timed continuations/s | Timed duration | Actual windows | Prefix repairs |
| --- | --- | --- | --- | --- |
| Wider adaptive MTP | 12.372330 | 29.178013 s | 78 five-row + 11 eight-row | 40 |
| Normal four-row MTP | 12.894535 | 27.996356 s | 108 four-row | 36 |

Both emit 362 including EOS / 361 timed continuations, zero full target
recomputations. Wider accepts 303 continuations in five-row windows and 58 in
eight-row windows; one suffix window proposes four tokens. Normal run uses no
suffix window. These are actual target-verified paths, not forced-expert upper
bounds or independent CPU reference acceptance. No same-output Ollama comparison
or arbitrary/private input acceptance is claimed.

Wider draft/verify/repair: 4.776867/19.549433/2.421568 seconds. Cache adaptation
2.398553 s, 7,578 swaps, 15,451,033,600 bytes **one-leg payload** (approximately
30.9 GB combined two-way traffic). Wider resident routes 184,660 / total 229,440;
normal resident 169,331 / total 207,360. Staged packed bytes remain zero. Wider
windows do not improve throughput here. One hypothesis is speculative rejected
rows influencing frequency/recency cache history; accepted-only accounting is
not implemented or measured.

Wider physical reservation: 31,769,661,440 bytes. Every run erases all 1,562 target
private buffers (871,661,568 bytes) and 43 MTP private buffers (11,730,944 bytes).
Public MTP owner is 2,434,793,472 bytes; exact expert banks 18,859,622,400 bytes.
Known receipt limitation: `target_packet.packet_bytes` still prints the legacy
163,888; the eight-row allocation/layout actually requires 327,744 bytes. Correct
this descriptive field before using it to quote a future packet. Physical and
private owner checks use the larger actual extents. No private data is persisted.

Frozen build directory:
`/media/jmicko/faststore/kuno-development/native-next-v252-strata-eight-build-20261013`.
Completed trial directories:
`native-next-v252-strata-eight-forced-code-intel-20261013` and
`native-next-v252-strata-eight-normal-code-intel-20261013` under the same development
base. Despite the first directory's historical name, its MTP cost policy is
adaptive, not constant eight rows. Saved receipts contain aggregate public
fixture metrics; generated text was displayed transiently, not saved.

### Checks, recovery and next direction

Strict release Clippy `-D warnings` passes on `383248a`; fmt/whitespace pass.
Frozen v252 shader validation passes 59 GU/DOWN/projection variants plus 14
controller/dense shader sources. The expanded PLE accepted-prefix suite passes
both tests, including every partial prefix for proposed counts 2..8. Original
`direct-device` regression on `383248a` passes 1,346 tests / 1 ignored; that
feature's tests do not establish full unit coverage for eight-row mode.
The whole-model wider/normal GPU runs establish the measured laboratory path.
No prose eight-row trial, independent complete state-vector oracle, long soak,
private worker or deployment acceptance has been performed.

Fresh physical/service receipt:
`/media/jmicko/faststore/kuno-development/native-strata-port-paused-checkpoint-20261013.json`
(20:46:29 UTC): admission idle/unpaused/no recovery, services HTTP 200, B70
50,688,000 bytes, Nvidia compute zero, all classified faults zero. Bounded sudo
helper 3498864 expired normally at 20:45 UTC and is absent. Goal paused.

Next highest-potential direction remains mapped-HOST expert cost: stage selected
packed HOST experts into GPU scratch once and reuse across GU/DOWN rows, or
improve admission/promotion scheduling based on accepted use. Existing
`frozen_host_staging.rs` is the old 30-group/three-row/32 MiB-per-role operator;
do not enable it unchanged for integer four/eight-row kernels. Coordinate its
actual source-index remapping, private owners, capacity and allocation quotes.
No such new staging implementation is present and no speed gain is promised.
Avoid more tiny unprofiled variants or API/UI work at this checkpoint.

Future build: `build-strata-port.py FRESH_BUILD --eight`; runner:
`run-strata-port.py BUILD FRESH_RESULT code q2 --cache-eight --mtp-eight` for the
wider MTP candidate, omit `--mtp-eight` for normal four-row MTP. These are public
laboratory selectors only; `--eight` requires a fresh immutable build directory.
Eight mode rejects unported head/fused/wide/16-lane combinations. Resume from
this current source, not an old failed binary, but keep v236 selected unless a
new complete normal measurement improves it. No further work is queued.

### Resumed Strata Intel source comparison (October 13)

User returned GPU access and refreshed sudo. Current priority supersedes the
staging proposal above: match concrete differences in Strata's pinned SYCL
expert kernels/device plan before inventing a new transfer design. Its
`verify.cpp` device-only `ar_on()` path calls grouped execution with full group
capacity; our compact GU dispatch caps its indirect Y count at four and loops
remaining groups serially. DOWN already uses the complete group count. First
optional comparison (`--independent-groups`) launches one compact GU workgroup
row per actual distinct expert, preserving original weights, per-group dot and
reduction, record audit/erasure and allocations. A/B control remains four lanes.
No performance or full Strata parity claim before whole-model measurement.

### Intel source-led comparison follow-up (historical implementation notes)

Verified against upstream MIT pin `fb58e0d`, not a CUDA-only inference summary:
- `sycl/src/core/verify.cpp` device-only grouped execution has full expert-group
  parallelism. `206e09a` optional independent GU groups measures 12.830527 code
  versus 13.240844 same-binary four-lane control. Both 362 EOS / 36 prefix repairs;
  independent groups is unselected after this negative result.
- `sycl/src/kernels/cuda/iq_kernels.dp.cpp` defaults `STRATA_GRID_SLM=0` and has
  explicitly unrolled packed byte dots. `d0093c1` adds optional constant-index
  unrolling; `6b815de` optional constant IQ table removes per-workgroup staging.
  Combined trial 11.116707 code, unrolling alone 12.914608; both 362 EOS, no
  full target recomputations and clean recovery. Both options stay off. A WGSL
  constant array is an experimental analogue, not proof of identical SYCL
  machine code or a public GPU constant-buffer implementation.
- Measured v254 actual VRAM at decode: 27,945,762,816 bytes of 34,242,297,856.
  Its plan reserves approximately 31.76 GB. Native dense original tensor payload
  (excluding routed expert triples and PLE table) totals 4,514,173,440 bytes.
  Our ordinary cache charges power-of-two allocation bins and has an 8 GiB cap;
  Strata native buffers/residency instead use available memory more directly.

`23944be`/`ef7187d` implement a separately compiled `strata-resident-224` trial:
exact persistent original-public dense allocations, aligned conservative physical
charges and actual backing-root bounds; 5 GiB static cap / 224 experts per layer.
Expert segments are 112+112; host partitions 128+128+32, preserving all 512
original expert identities. The preflight still charges private state, draft,
source/runtime I/O and driver margin together before loading. This is not a
runtime auto-sizing feature. These preparation notes predate the completed v257/v260 measurements below;
the later receipt establishes the measured geometry and physical recovery.
Build helper accepts `--resident-224`; the runner derives the canonical 5 GiB
cache selector from that artifact's feature manifest. No manual arbitrary cap.
`60addeb` adds a root-only aggregate physical-memory observation at decode;
request content/IDs are not recorded. Do not confuse it with reserved caps.

### Exact initial profile residency comparison (preparation notes; v261 below)

Strata `sycl/src/program/generate.cpp` defines `--expert-profile` to choose
residency instead of first-use admission. Our older `profile-ranked-public`
filler preserves prompt-touched residents, so its 224-slot 13.869639 code result
is not full initialization parity. New optional `--profile-first` chooses the
compiled pinned PUBLIC profile's entire per-layer selected set at the completed
post-prefill setup boundary, retaining wanted existing hits in their old slots
and replacing only unwanted experts via original pending-plan/upload/fault/commit
proofs. No private state/routing trace is saved or consulted. Source profiles
remain per-layer fixed capacities, not Strata's global variable allocation.
This changes only initialization; original non-port preservation stays unchanged.
The completed v261 trial below supersedes this preparation status.

## Completed combined residency pass, October 13, 23:11 UTC

All trials use the fixed public code/prose fixtures, seed 424242, temperature 0,
DASLab IQ3_S target and Q2 MTP on B70 only. First confirmed token is excluded from
rates. Generated answers were displayed transiently, not saved. Complete target
verification remains authoritative; no forced expert substitutions are used.

| Frozen candidate | Connected configuration | Code continuations/s | Prose continuations/s |
| --- | --- | --- | --- |
| v257 / c1ccb46 | exact static cache, 224 resident/layer | 13.737579 | 13.636417 |
| v258 / 2e5eb68 | 224 plus public-ranked empty-slot fill | 13.869639 | not run |
| v260 / 86a98a4 | 240 plus exact cache and ranked fill | 14.063997 | 14.300613 |
| v261 / bb336a1 | 240 plus full per-layer public-profile initialization | 13.884123 | not run |

Selected v260 emits 362 including EOS / 361 timed continuations for code,
512 / 511 at cap for prose. Timed durations are 25.668379 / 35.732735 seconds;
verification 17.278097 / 24.430213 seconds. State-only prefix repairs: 36 / 95;
full target recomputations: zero. HOST routes: 28,600 / 207,360 code, 22,702 /
338,400 prose. Cache adaptation: 2.386399 / 2.122040 seconds, 7,353 / 6,662 swaps.
One-leg exchange payload: 14,885,068,800 / 13,402,086,400 bytes; combined two-way
traffic is twice that. Both runs complete private erasure and physical recovery.

240 experts per layer means 11,520 resident experts, two 120-slot GPU segments
and HOST partitions 128+128+16. Exact expert banks use 23,574,528,000 bytes in
288 roots. Static public tensors account for 4,166,067,200 bytes; static cap is
5,368,709,120. This leaves more physical memory for banks than the old binned
8 GiB cap. Combined preflight is 33,254,625,280 bytes, including 512 MiB shared
driver margin. Code's actual decode VRAM is 32,260,292,608 of 34,242,297,856 bytes.
This is fixed laboratory geometry, not autosizing or Strata's variable allocation.
Target erasure covers 1,562 buffers / 863,207,424 bytes; MTP 43 / 11,730,944.

v261 initializes all selected per-layer slots from the pinned public profile,
protects wanted existing hits and replaces unwanted prompt-touched residents.
It uploaded 7,819 experts, retained 3,701 wanted hits, and completes real code
verification/erasure/recovery. Its single 13.884123 result does not establish a
statistically significant loss or reject Strata's complete allocation design.
Known descriptive limitation: profile digest/table-byte receipt fields retain
legacy top144 values; do not quote them as a top240 digest. Full existing public
profile is read for the new setup; no request routing history is persisted.

Negative local comparisons remain available but optional: independent GU groups
12.830527 versus same-binary four-lane control 13.240844; unrolled packed dots
12.914608; unrolling plus WGSL constant IQ array 11.116707. These are incomplete
translations, not evidence against the connected upstream strategy. In particular
WGSL const-array lowering is not Strata's shared GPU read-only table. Do not keep
running these standalone comparisons as the next development gate.

Failed v256 rejected stale host-page geometry before generation; c1ccb46 fixes
all expert addresses. v259 rejected the old 640 KiB host-metadata allowance;
86a98a4 raises only the 240 fixture to 704 KiB. Both physically cleaned up.
A v257 ranked-fill parser rejection preceded GPU work and was fixed in 2e5eb68.
Do not treat those attempts as throughput results.

Latest a0bd715 checks pass: release port Clippy with `strata-resident-240` and
`-D warnings`, fmt/whitespace, 1,346 original direct-device tests / 1 ignored,
and two selected port tests (all original addresses/role bounds, profile wanted-hit
protection/redirected-write rejection). Frozen v260/v261 shader checks pass 49
projection variants plus controller/dense sources. Original regression proves
feature-off isolation; it is not the full Strata-feature unit suite. No independent
full-model state-vector oracle, long soak, private worker, deployment or complete
upstream parity is established.

Immutable fallback build:
`/media/jmicko/faststore/kuno-development/native-next-v260-strata-resident240-build-20261013`.
Completed results: `native-next-v260-strata-ranked240-code-intel-20261013` and
`native-next-v260-strata-ranked240-prose-intel-20261013`, under the same base.
Reproduction: build fresh with `build-strata-port.py FRESH_BUILD --resident-240`;
protected runner `run-strata-port.py BUILD FRESH_OUT code q2 --cache-eight
--ranked-fill` (or `prose`). Full profile candidate additionally uses
`--profile-first`. Artifact identity and all output directories must be fresh.

Final host evidence:
`/media/jmicko/faststore/kuno-development/native-strata-source-match-final-state-20261013.json`.
Bounded sudo refresher 3648310 stopped at 23:12 UTC; no benchmark/compiler/helper
remains. Canonical checkout and deployed binaries are unchanged.

### Next scope: connected upstream Intel core, not another microbenchmark

The user explicitly reiterated that interdependent changes should be implemented
as a system before throughput judgment. Use MIT source pin
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf` and the existing attributed Rust port.
Inventory every connected loading/placement, router/group-planning, quantized
execution, MTP verification and state-commit dependency. Implement that connected
path, compile/check shaders, then run whole-model code/prose and debug failures.
Keep v260 intact as fallback. Do not require each ingredient to outperform v260
on its own; do not call an experimental analogue a faithful finished port.

Known gaps to resolve in that inventory are global variable expert allocation
versus fixed per-layer capacity; a real shared read-only GPU IQ table versus
workgroup staging/const-array lowering; unified pointer group execution versus
our segmented descriptor/dispatch plans; and connected graph/fusion/dense
quantized execution. These are source differences, not individually proven
causes of the speed gap. Document unavoidable Vulkan/WGSL substitutions and
remaining missing operators precisely. Retain transient memory, real routing,
complete target verification and actual cleanup. No API/UI scope or release
acceptance is implied by this native-engine checkpoint.

## Native Intel core receipt (2026-10-14, in progress)

Source through `9be8a65` adds the coherent `strata-intel-core` feature to the
isolated laboratory workspace. Build with `build-strata-port.py FRESH_BUILD
--intel-core`; run the frozen executable with `run-strata-port.py BUILD
FRESH_OUT code q2 --intel-core` (or `prose`) in the protected shared terminal.
The builder rejects dirty tracked sources and changing HEAD during compilation.
The runner still disables swap/core dumps, selects B70 only, pauses admission,
and restores/checks it in a finally block. There is no installed backend/API change.

The native executor borrows the existing Vulkan device/queue through vendored
wgpu-hal 29.0.4 (licenses retained). Rust ash owns its compute pipelines,
descriptors, fences and bounded command recordings; Rust NAGA/rspirv lower the
shaders. No C++/SYCL inference implementation is linked. Supported buffer device
addresses and shaderInt64 are required. The logical Wgpu zero-work initialization
pass is necessary: otherwise Wgpu may clear a raw-native result before readback.
Producer execution precedes readback; private uploads and scratch fills retain
their actual positions within the graph. Parameter target pages remain stable
and are updated only across completed serialized submissions. Cleanup discards
recordings before original full-owner erasure/audit and source detachment.

Public code runs (seed424242, temperature0, 512-output cap, original DASLab IQ3_S
shards and Q2_0 MTP) complete with362 emitted including EOS /361 timed
continuations, readable output and physical recovery:

| Frozen build | Source | Timed continuations/s | Native submissions/dispatches/replays, process totals |
| --- | --- | ---: | --- |
| v269 | `0e6f2b7` | 10.539086 | 4912 /441263 /54 |
| v270 | `049d493` | 12.112745 | 4912 /441263 /54 |
| v271 | `600a887` | 8.302945 | 4912 /441263 /54 |
| v274 | `1c9c9d8` | 12.416320 | 4960 /441263 /53 |
| v275 | `617537b` | 12.852923 | 4312 /441263 /84 |
| v277 | `f2d4048` | 12.594086 | 4312 /441263 /134 |
| v278 | `55adf7d` | 12.408893 | 4312 /491807 /134 |

All directories are under `/media/jmicko/faststore/kuno-development`, named
`native-next-vNNN-strata-intel-core-build-20261014` and
`native-next-vNNN-strata-intel-core-code-20261014`; `source.json` retains exact
source commit and executable SHA256. v271's duplicated source-page projection
was replaced by physical pointers; retain it as evidence, not a selected path.
v277/v278 have107 graphs containing at least2048 compute dispatches and104
replays in that bucket. Process-wide counts include load/prefill/draft/repair;
they are not per-target GPU timings. v278 actually executes16848 packed dense
attention dispatches. It did not improve the whole-model rate yet. Keep the
coupled implementation; do not veto its components using standalone timings.

Completed runs restore B70 to50,688,000 bytes, NVIDIA compute processes to0,
normal admission/HTTP and unchanged classified GPU fault counters. Each original
private/source closure passes. v278 erases1562 target buffers /863,207,424 bytes
and43 MTP buffers /11,730,944 bytes. There is no private-input, deployment,
independent full-state oracle, long-soak, complete upstream parity or30t/s claim.

### Coupled fetch/rebase continuation

`9dc6eeb` and `9be8a65` implement the missing default Strata kernel-fetch path:
GPU MAP selects actual HOST groups; one role-copy shader reads their original
RAM-mirrored packed bytes contiguously and writes three distinct64MiB private
stage roots. GU/DOWN use physical pointers rebased to the selected group's stage
slot; original VRAM groups keep their resident addresses. Original source
ownership/shape proofs remain. Extra bound stage roots provide genuine buffer
lifetimes and dependency edges for the physical reads. All stages are allocated
upfront in PrivateArena, charged201,326,592 bytes in both the constructor plan
and exact target-extent gate, physical target/MTP admission and the complete
owner closing check, and erased/audited with the target. Nothing is
allocated, read from disk, or persisted per stage. No forced/substituted experts.
Core limits are18 storage buffers /19 bindings; pointer tables include the three
actual stage addresses and never log them.

The same change removes redundant per-matrix scratch clears: packed preparation
writes every active plane, and the authentic subsequent GU preparation clears
its complete header/frame/tail before using that root. Borrowing is restricted to
the attention/PLE stage, before GU preparation. Shared-expert and other callers
retain their original path. Dense types areQ8_0, Q4_K, Q5_K, Q6_K, IQ4_NL and
IQ4_XS; original model weights are not requantized. Activation arithmetic is
single Q8_1, so this does not restore the CPU13/F32 precision contract.

v279 passed56 WGSL/SPIRV sources, including13 physical modules through
`spirv-val`, then rejected the old exact target construction-extent gate before
inference. Cleanup/physical recovery pass. `9be8a65` adds the genuinely owned
stage extent to that gate; v280 runs generation but rejects the old closing owner count. Physical cleanup
passes. `44bb1d1` adds three buffers /201,326,592 bytes to that closing gate and
the same bytes to combined physical admission; `a0c6c3c` bounds every shader copy
to its complete64MiB stage owner. v281 builds but is not GPU-run. v282 is the
fresh combined artifact. Its code run passes at14.625229 timed continuations/s,
456 emitted including EOS /455 timed,31.110624 seconds,143 actual four-row
windows and139 large-graph replays. Target close proves1565 buffers /
1,064,534,016 bytes; staged source copies20592 dispatches, packed dense22308.
Readable public output, original source detachment, private erasure, actual50MB
B70 recovery, no NVIDIA compute process and unchanged faults pass. This answer
has a different token sequence/length from v260, so the rates are not a matched
same-output speed ratio. Prose completes512 /511 timed at13.546277 continuations/s in37.722541s,
169 actual four-row windows /165 large-graph replays. The same private/source
and physical cleanup pass. Current original direct-device regression passes
1346 tests /1 ignored; the full Strata-feature unit suite is not run.
Strict core Clippy passed after the closing/admission fix and the final shader
bound;56 source validations including13 physical SPIRV modules pass. Do not
claim general private inference or30t/s from this receipt.

Next work must stay in the coherent core: finish fetch/rebase numerical/physical
validation, follow the actual GPU/CPU execution costs, and then resolve variable
global residency and remaining quantized shared/MTP/fused operators. Keeping the
fast v260 fallback does not mean removing coupled components because a partial
translation is slower. No Vashti connector changes are required for this lab.

Current frozen coherent artifact: source
`a0c6c3c9c21745faed6ee7362983bcd00f5a72e6`, SHA256
`9b41ea8fb312ad1fd1ddb3f5986eb479832c5ab9d173c0ffdf51188c102f38c6`.
Use the v282 build/code/prose directories above; v260 remains the speed fallback.
Final read-only host inspection at02:26:33 UTC proves Kuno/Ollama/Vashti HTTP200,
admission idle/unpaused/not recovering, B70 used50,688,000 bytes, NVIDIA compute
processes0 and all classified GPU faults0. No GPU/compiler job remains. The
previous bounded sudo helper was stopped; a fresh90-minute helper supports the
active goal under `strata-intel-core-continuation-sudo-20261014.pid`. Stop it when
actual work ends. No reset/reboot/rebind, installed binary or canonical source edit.

Resume from this coherent path. Large graph replay is now real and its capture
cost is small in aggregate; do not spend the next pass rebuilding another graph
analogue. Complete the remaining packed shared/dense/MTP and fusion paths and
measure actual device costs; global unequal residency remains a substantial
unported loading/placement dependency. Any variable-capacity/global pool needs
new truthful owning slot/source metadata, not changed bytes behind the old
frozen provenance. Staging currently supports up to40 union groups /four target
rows; core eight-row verification is not connected. Upstream SYCL/XMX prefill,
CPU expert doorbells, pipeline split and other platform options are not claimed.
New activation arithmetic changes the generated sequence, so independent
whole-state/answer-quality acceptance still matters before private deployment.
The goal remains unfinished; do not mark30t/s or the complete rewrite achieved.
