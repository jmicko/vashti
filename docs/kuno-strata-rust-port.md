# Strata Rust whole-model port experiment

## Current checkpoint: active optimization, October13

The goal is active. User has authorized both GPUs and refreshed sudo; normal
services are restored after every protected trial. Work stays on
`codex/strata-rust-port` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Canonical Vashti, deployed
binaries/defaults, API/UI and release remain unchanged. Required consumer changes:
**none**. Original accepted engine remains `codex/kuno-native-text`/`6792453`.

Latest runtime candidate is3e01731379291b93e85dd312e5b52777b6cc5a2c (v236),
frozen SHAd21039dabbd8b2c8e25d16304f011c0e32284c05de9f82e9511774420cda664f.
Default path atv234 passes prose at13.000929continuations/s and instrumented
code/source/privacy/recovery; v232 normal code is12.796036. v236's optional
cache-eight code is13.004114 and prose13.189027; both complete and recover. **30 not achieved.**
Do not call the current code a complete Strata rewrite or production worker.

Queued MTP, retained-input prefix replay, exact192-slot banks and direct two-way
GPU/RAM exchanges work. Head, fused source, wide reads and16-lane schedule are
implemented/tested candidates but stay optional after neutral/slower timings.
Read the latest measurements/profile/next actions at the end of this document;
old paused/v221 stop sections are historical, not the active handoff.

The user's coherent-port-first strategy remains authoritative; original CPU13
precision gates are superseded for the isolated port only. Real target routing,
verification, physical/source bounds, transient privacy, swap/core-disabled
execution and actual recovery remain required. No private input path is added.

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
