# Strata parity goal — 2026-10-20

The user explicitly requested a new goal to match Strata. Work remains Kuno-only
on `codex/strata-rust-port`, using the agreed integration032e2cdd and original installed
worker c8db19de / SHAfb31e124 as the original comparison baseline. Selected
successor3fffc266 / SHAeac78ce0 is now installed, replacing0771a726 / SHAccdaf886. Preserve the canonical
Vashti checkout and uncommitted coordination files. No API/frontend change,
release, reset/reboot, model/quant substitution or relaxed numerical/ownership
checks is part of this goal. Current GPU permission and sudo apply to this run;
stop protected execution at a safe checkpoint if the credential expires.

## Reference and acceptance scope

Pinned MIT upstream Strata v0.1.41, source
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, is available for source comparison at
`/home/jmicko/.codex/worktrees/kuno-native-split-gguf/strata-research-20261009`.
The old comparison installation/binary/container was removed at the user's
request. Its original public receipts/configuration are retained under
`/media/jmicko/faststore/kuno-development/strata-comparison-evidence-20261009`.

Historical local Strata uses the same original DASLab IQ3_S shards, Q2 MTP,
Intel B70, context32768, int8 KV and direct PLE; spec4/min-p0.5 and CJK draft
vocabulary. Seed424242/temperature0,35-token merge-code prefix0.7058s and30-token
prose prefix0.6373s;200-output code/prose decode27.3/36.0 tokens/s. Current native
Kuno code uses seed0, permits1024 outputs and produces501 rather than the previous
362 tokens; warm prefix1.4305s and decode32.92 tokens/s. These are useful reference
bounds, **not a new matched sustained comparison**.

The upstream B70 report's4095-token prefix980–1002 tokens/s (~4.1–4.2s) uses PLE
RAM, different input/configuration and an earlier host. Kuno's original baseline4245-token
compiled-public check takes24.9658s, toolbox4207/4293 prefixes29.506/29.920s.
Do not claim parity by comparing unlike workloads, truncating tools/history,
changing the quant or reporting isolated first-token speed. Before declaring
parity, require an agreed matched local upstream workload/settings/timing scope
or a sufficiently documented equivalent reference; distinguish prompt/decode,
load/first-use compilation, final erasure and whole-response time.

## Accepted native32 installation

Native worker source3fffc266383917cada88fd176b273db8a6b737f4 / SHA256
`eac78ce0029205c9d503c78a1b2c868082801cf966b4e8a3d3bc7585ece80f4c`
is now installed through the unchanged image-enabled gateway36626d30 /
SHAd3ed07f4. Official backup/rollback:
`/var/lib/kuno-strata-parity-native-wide32-20261020` (previousccdaf886).
The complete configuration and original root-enrolled identities are preserved.
No API, protocol, setting, model identity, Vashti consumer or release change.

Matched clean first/repeated4245-token prefixes14.1743/12.4373 ->12.9623/11.1474s;
first-to-first8.55% and repeat-to-repeat10.37% shorter. Fresh upstream repeat
5.3515s remains ahead: the goal is active, not complete. The actual matrix tile
keeps original work/precision with32 subgroups and exactly49,152B shared;
real compiled kernels eliminate all prior91:77 scratch spills/fills. Expert
matrix paths remain original. See the exact ISA evidence and complete prior
candidate receipts below.

All103 actual GPU controls,402/144 offline layouts, real terminal-prefix error
recovery, fresh successful clean requests, protected full thinking/ten-tool
call/result,64/65/66 boundaries,16 MAX8 tails, two identical complete501-token
code answers with155 semantic cases each, privacy and physical recovery pass.
`native-strata-parity-wide32-install-20261020.log` records the official successful
transaction. Actual installed service checks
`native-strata-parity-wide32-installed-{public,final}-check-20261020` pass too:
cold load101.0112s, first35-token prefix10.0622s including JIT; warm prefix1.0908s,
501 outputs in15.0813s (~33.22 output tokens/s), total19.3138s. No new decode
speed gain is claimed. Nonroot DynamicUser/original model mount, swap/core0,
write_bytes0, denied inspection, zero OOM/driver faults, HTTP200 and normal
admission pass. Subsequent protected numerical tests deliberately unload it
and confirm physical recovery again.

The complete uninstalled combined candidatea3b36795 / SHAd7965806 (native32,
paired QK reuse and descriptor batches) passes all103 original GPU controls and
full public requests/recovery, but profiled prefixes11.8289/11.7768s are
essentially unchanged from native32 alone11.7837/11.8168s. Capture1.9064->1.8394s
is a small diagnostic change, not a demonstrated1s gain. This pipeline is kept;
sourceefaaea7e submits closed prefixes before future descriptor preparation and
8c91e6ab hoists the common full-pair validity branch. Their complete frozen
successor `strata-parity-native-paired-fast-bulk-20261020` / SHA52717687 passes
402/144 offline layouts; actual GPU/correctness and CPU profiling are next.

The new fixed-public parent-only startup markers locate98.1591 of103.4862s
inside `global_sources` before `shared_target_weights`. Target model1.6663s,
shared head0.6375s and MTP2.5484s are smaller. These are pipe-arrival intervals,
not exact internal CPU/GPU times. The next public-only CPU sampling will locate
specific weight-store and command-recording work. No private instrumentation,
request/response storage, cache reuse, quant substitution or hazard weakening.

## Prior combined operator checkpoint

Selected default-native source0771a726 / SHA256
`ccdaf8861819ec3dfb6e92208904dba5f989d79f49d176289f69e265be2e8057`
was installed and backend-validated before the successor above. Gateway source36626d30 / SHA256
`d3ed07f473b28bbb46596a619349d933edd6c166b7af10dd42f9821e4ba3048c`
is unchanged. Official backup/rollback:
`/var/lib/kuno-strata-parity-native-20261020`, previous workerfb31e124.
Complete configuration and original root-enrolled identities are preserved.
All receipt paths below are relative to `/media/jmicko/faststore/kuno-development`;
only public numeric metadata is retained, never request/response text or routes.

The clean45/4245/4245 check completes with correct arithmetic, original close,
physical recovery and zero new driver faults. Load102.3943s; first short prefix
10.4072s includes first-use compilation. Long prefixes14.1743/12.4373s, the repeat
50.2% shorter than the installed baseline24.9658s. Fresh upstream5.3515s remains
well ahead; **the parity goal is not complete**.

`strata-parity-native-rolling-20261020-terminal-negative` genuinely submits a
prefix, fires the public injected error once, successfully waits two real
recovery queues and records zero failed queues. Exit1 is intentional: no normal
TEXT/DONE/CLOSED, owned-zero or reusable-session claim. Protected process exit
physically releases GPU memory; the subsequent fresh clean worker succeeds.
`strata-parity-native-rolling-20261020-staged-api` passes the ten-tool thinking/
call/result round trip, exact64/65/66 thresholds,16 genuine MAX8 tail requests,
complete identical501-token semantic repeats (155 cases each), zero writes,
inspection denial, graceful unload, Ollama after native and actual recovery.
Toolbox4207/4293 prefixes15.8214/14.7448s versus baseline29.506/29.920s.

The official transaction and subsequent actual installed service check pass.
`native-strata-parity-installed-public-check-20261020` confirms two complete,
identical semantic answers through the real listener; cold load100.3023s,
first35-token prefix9.9475s including JIT, repeated prefix1.1636s,501 outputs in
15.2889s (~32.77 output tokens/s), total19.7015s. This is no new decode gain.
`native-strata-parity-installed-final-check-20261020` confirms actual nonroot
DynamicUser worker, original model mount, swap/core0, write_bytes0, denied
unprivileged inspection, zero OOM/driver faults, HTTP200 and normal admission.

The installation includes QSA/GDN specialization, genuine MAX8 headless known
prefix, native narrow/macro matrices, rolling scheduling (ecfb12b9/881d5af3),
explicit public terminal control9003819d/08f052e6, and checked borrowed-layer
bindingsd54a60ab. The original operators, source ownership, error quarantine,
private erasure and full thinking/tools/context contracts are preserved.
No API/private-wire/configuration/model-name/Vashti change or release.

### Earlier operator candidates and numerical comparison

Sourcec6808efc / SHAd1100418 passes all97 actual GPU controls:30 original-weight
matrix,39 router/dequant,6 causal-selector,12 QSA attention and10 GDN recurrence.
Its `strata-parity-vector-gdn-20261020-operators` receipt confirms actual close,
physical recovery and zero accepted numerical mismatches. The original nine QSA
fixtures retain their strict complete-owner bitwise comparison. The three added
non-dyadic fixtures independently check **both** unchanged-scalar and optimized
outputs against F64 mathematical references with operation-derived F32 rounding
bounds; they retain bit-difference diagnostics, full-owner/tail/fault checks and
negative controls. This explicitly corrects the newly introduced bit-identity
criterion for those three fixtures; it does not loosen the original nine or use
one GPU path as the mathematical oracle. See
[the complete numerical contract](kuno-qsa-attention-parity-fixture.md).

The c680 profiled warm4245-token prefixes are14.1363/14.1994s. The second request's
expert interval is8.4810s:1.8790s host enqueue,1.9503s native capture and4.4448s
native wait. QSA attention is1.8424s and GDN recurrence/norm0.4307s. These are
instrumented complete-request intervals, not isolated GPU throughput.
`strata-parity-vector-gdn-20261020-profile` completes cold73 plus both long
requests with correct arithmetic, original request erasure and close/recovery.

`native-parity-vector-gdn-stage-quality-20261020` passes the protected staged
API: ten-tool thinking/call/result,64/65/66 thresholds, MAX8 terminal counts1..8
ascending and descending on the same worker, two identical complete501-token
code answers with155 semantic cases each, write_bytes0, inspection denial,
graceful unload, subsequent ordinary Ollama and physical recovery. Warm code
prefixes are1.0432/0.9266s. Its separate seed424242/temperature0/cap200/context32768
code/prose repeats measure38.3116/31.5556 output tokens/s. These cap-limited
requests do not claim EOS completion or equal output sequences across engines.

The fresh pinned MIT Strata reference remains the unchanged sourcefb58e0db /
SHA16a6ed70. `strata-matched-reference-20261020` measures5.3515s for the same clean
4245-token prefix, with all prompt/conversation caches disabled; its suffix
policy promotes the requested verifier4 to effective6. The separate sustained
`strata-matched-decode-reference-readyfix-20261020` explicitly fixes verifier4,
MTP4 and suffix3 to match Kuno's native policy, verifies the same35/30-token
rendering/tokenizer hashes, seed424242, temperature0, cap200, context32768,
int8 KV and direct PLE. Repeat code/prose rates are27.2472/36.1017 output tokens/s,
without prompt reuse or an EOS claim. Decode is workload-dependent: Kuno leads
this code fixture and trails prose; long-prefix parity is still not met.

The full opt-in CubeK Tilewise plus rolling candidate source881d5af3 /
SHA528bc066 passes all111 actual controls, including six actual CubeK dense and
14 actual expert launches (`strata-parity-tilewise-rolling-20261020-operators`).
Its complete profiled warm prefixes regress to21.0933/20.4503s versus c680's
14.1363/14.1994s (`strata-parity-tilewise-rolling-20261020-profile`). Preserve the
coherent candidate and evidence; it is not selected as the default on this basis.

The latest default-native plus rolling-submission and borrowed-layer-binding
candidate source0771a726 / SHAccdaf886 passes all97 actual controls and complete
public profiling (`strata-parity-native-rolling-bundle-20261020-operators` and
`strata-parity-native-rolling-bundle-20261020-profile`). Warm4245 prefixes are
13.2547/13.2952s. Its second expert interval is7.5203s:1.6801s enqueue,2.0152s
capture and3.6075s wait, with81 actual rolling recordings,1346 windows and24773
segments. QSA1.8426s and GDN recurrence/norm0.4312s are essentially unchanged.
A two-thread CPU shader-validation job overlapped this profile, so it is diagnostic
evidence, not a clean A/B benchmark. The separate clean, actual terminal-error,
staged API and installed acceptance above supersede the earlier pending gates.

Earlier evidence is retained: sourcee9b6bd4c / SHAcf867264 completed a14.4597s
profiled long prefix, then7ada5edd / SHAbba3d0f3 failed the newly added three
non-dyadic bit-identity checks despite30 passing matrix controls. Source6256ebea /
SHA4b4e5b0b connected the native NoContraction product-rounding pass, but legal
finite compiler-rounding differences remained. Those original failure receipts
are not rewritten as passes; the independent F64 contract above supplies the
explicit stronger mathematical acceptance for the new cases. Physical recovery
was confirmed after the failed trials as well.

## Connected high-potential work

- The complete Tilewise balanced M64/Double candidatec5932d91 / SHA4f666ecf
  passes111 GPU controls and full requests but regresses to33.9067s. Receipt
  `strata-parity-balanced-m64-qsa-phases-20261020-profile`: GDN input5.7643s,
  output2.2056s, QSA input1.6619s, experts17.2625s. Preserve it as an unselected
  experimental pipeline; do not install it on source structure alone.
- Public phase attribution now separates the full QSA family: producers76.3ms,
  score/validate/select127.0ms, attention/merge1659.9ms, gate20.7ms in that run.
  Reference QSA849ms includes input/output projections and producer work;
  comparing Rust's whole1.885s family only to upstream680ms attention is invalid.
  See the exact scopes in `kuno-prefill-qsa-family-profile.md`.
- Complete library M128/N128/K32, eight real subgroups, actual36,864B shared,
  source43281fd6 / SHAe53e00e2 passes117 actual GPU controls and complete
  requests. Its repeated profiled prefix28.7764s is still a regression and it
  remains unselected (`strata-parity-wide-capacity-20261020-{operators,profile}`).
- Complete original-native M128/N128/K32,16 real SIMD16 subgroups, actual40,960B
  shared, source9cf2236b / SHAa7ac11f1 passes103 actual GPU controls, including
  two actual full-tile launches and four strict original fallback/tail cases.
  All402 ordinary/144 matrix layouts validate offline. Complete profiled
  prefixes12.8885/12.6279s improve slightly over the selected13.2952s diagnostic,
  but this is not a clean A/B or an accepted installation. Original native
  expert matrices remain unchanged. Receipts:
  `strata-parity-native-wide-20261020-{operators,profile}`.
- A process-local compiler dump during ONLY those fixed-public numerical
  controls identifies actual native-wide FP16/BF16 kernels: requiredSIMD16,
 256 invocations,40,960B shared and128 GRFs. Both spill91/fill77 static values;
  the hotK32 loop contains27 scratch loads and40 scratch stores. These are
  compiler instruction counts, not runtime cycles or measured scratch bandwidth.
  The installed Mesa ANV pre-Xe3 path has no supported256-GRF environment switch.
  A complete32-subgroup candidate keeps the same128x128x32 reuse and halves
  each thread's accumulators, using512 invocations and exactly49,152B shared.
  Source3fffc266 / SHAeac78ce0 now passes actual-device admission, all103
  original GPU controls,402 ordinary/144 matrix offline layouts and complete
  public profiling/recovery. Its repeated4245-token profile is11.8168s
  (first11.7837s), versus16-subgroup12.6279s. GDN input/output projections
  fall0.7500/0.2974 ->0.3657/0.1525s and QSA projections0.2168/0.1005
  ->0.1105/0.0527s. Expert execution7.3597s/capture1.9064s and QSA attention
  1.6581s remain essentially unchanged. Both actual wide kernels have zero
  scratch spills/fills and the same total256 DPAS instructions per workgroup
  K32 iteration. Static instruction counts are not measured cycles. Receipts:
  `strata-parity-native-wide32-20261020-{operators,profile}`; see the
  [exact ISA evidence](kuno-prefill-native-isa-admission.md).
  Full negative/fresh clean/private staged acceptance now passes too:
  `strata-parity-native-wide32-20261020-{terminal-negative,clean-long,staged-api}`.
  Clean firstlong12.9623/repeat11.1474s versus selected14.1743/12.4373s;
  corresponding gains8.55%/10.37%. First-long includes genuine first-use large
  shape compilation and is compared with the original first-long position.
  Full ten-tool thinking/call/result,64/65/66,16 actual MAX8 tails, two identical
  complete501-token code answers with155 semantic cases each, write_bytes0,
  inspection denial, ordinary Ollama and actual physical recovery pass. A root
  build overlapped part of staged execution; do not claim a decode speed gain
  from those timings. An exact-hash official installer is prepared and separately
  reviewed; the official transaction and actual installed checks above now pass.
- Opt-in paired-cell QK attentiona814962d reuses the same query vectors across
  two original selected cells. Ten actual source/Naga/SPIR-V CPU controls pass
  both enabled and disabled; original scalar/math oracles remain unchanged.
  GPU numerical and whole-request evidence is still pending. A complete
  bounded descriptor-batch candidatea3b36795 reduces per-node Vulkan allocate/
  update calls to128-original-node batches while preserving every fresh set,
  queue hazard, flight and recovery owner. Four actual-helper CPU cases, full
  production typecheck and154 original queue/ownership cases pass; actual worst
  temporary metadata387,072B is charged to the original64MiB cap. The complete
  combined artifact sourcea3b36795 / SHAd7965806 includes native32, paired QK
  and descriptor batches;402/144 offline shader layouts pass. Original103 GPU
  controls and whole-request profiling are running before any speed decision.
  No combined speed or installation is claimed yet.
- Actual loaded capacity is11,998 resident expert pairs/24,445,798,400B versus
  the fresh reference11,608 pairs, so lower resident capacity is not the cause
  of the gap. The immutable host complement is registered/pinned and copied
  directly through the original ring; no per-request disk read or extra
  pageable-to-pinned staging is introduced. Actual32K sessions use no borrowed
  expert roots in either engine. The one-layer native public copy timestamps
  imply about13.56GB/s, but are serialized diagnostic copy intervals, not a
  clean end-to-end transfer benchmark. Reference87ms wait-copy is residual
  compute wait, not total DMA time; those two numbers are not comparable.
- Root owns shared GPU execution and installation; isolated agents own native
  matrix preparation and read-only compiler/source comparisons. Original
  metadata caps, real device topology, ownership and numerical gates remain.

Only compiled-public profiling may retain aggregate expert-member histograms
and arithmetic-work estimates; no identities, routes, text, hashes of content
or private request profiling. Production worker logging/persistence stay unchanged.
Use the real GPU numerical controls before accepting complete generation, then
matched speed, thinking/tools/semantic repeat/privacy, actual unload and physical
recovery. Source isolation does not isolate GPUs: only the root runs shared tests.


## First protected measurements in this goal

Artifacts live below `/media/jmicko/faststore/kuno-development`; all inputs in
these checks are compiled public fixtures. No request text, response text,
expert identities or routes are retained. The installed baseline remains
c8db19de / SHAfb31e124; the following candidates are not installed.

- `strata-parity-installed-reference-20261020`: seed424242, temperature0,
  context32768 and200-output cap, matching retained historical Strata settings.
  Kuno35-token code prefix1.53335s,200 outputs35.4123 tokens/s;30-token prose
  prefix1.33445s,200 outputs28.7183 tokens/s. Historical upstream code27.3 and
  prose36.0 tokens/s are workload-dependent bounds, not a current controlled A/B.
- Matrix source6413fa09 / SHA232bea52 and combined matrix+command-boundary
  sourcebfd920cc / SHA f8d5cae3 each pass all original14 full IEEE matrix cases
  and39 router/dequant controls, actual arena close and physical recovery.
  Combined warm4245-token profiled prefix23.86855s versus previous selected
 24.96584s. This is a small gain, not parity. Expert interval15.82034s and
  attention6.32888s still dominate this instrumented workload.
- Public histogram:12,221 of26,673 expert groups have1–7 rows;3,883 have8–15;
  2,850 have16–31. These cover6.8% of useful expert rows but expose expensive
  old small/tail matrix execution. Arithmetic-work estimates are not measured
  hardware FLOP rates or a private-request profile.
- Sourcea2ab2abd / SHA4294bc41 exposes a separate original-weight matrix timing
  check. All22 gate-up/down shapes pass full finite/padding/fault ownership,
  sampled IEEE points and real close/recovery; these sampled points do not
  replace the full numerical controls. `strata-parity-matrix-timing-base-20261020`
  records GU count3/7/31 at371.5/415.4/410.7us versus aligned count8 at59.9us.
  DOWN count3/7/31 is104.3/115.2/100.8us. It includes actual host enqueue and one
  real completion fence for64 repeats after3 warmups. A complete staged M8/M16
  narrow family is being prepared with additional full IEEE tail cases.
- MAX8 headless source7c96b6e4 / SHA14b4c9ec builds, but its first cold45-token
  public request fails the unchanged QSA guarded check. No output is accepted.
  `strata-parity-headless-max8-short-20261020` records failure and confirmed
  physical recovery, normal admission and zero new driver faults. Trace/fix the
  original cause; do not weaken that guard or install this artifact.

The new fixed-public native GPU sampler is restricted to request3/chunk0/layer24
of `--public-prefill-profile`; it is not available through private inference.
Its existing query threshold/caps, actual-fence ownership, no-replay policy and
serial diagnostic execution remain unchanged. Await a real sample before
claiming an operator breakdown. See `kuno-prefill-native-gpu-profile.md`.

## Narrow pipeline and attention continuation

Source6b9811ee / SHA46c27c54 (`strata-parity-narrow-guard-checked-20261020`)
adds the complete M8/M16 staged family without changing accepted M32 arithmetic.
All original14 full IEEE matrix cases remain;12 additional narrow-boundary cases
bring the matrix total to26, plus the unchanged39 router/dequant controls. All65
pass on the B70 with unchanged IEEE thresholds, full output/fault/tail ownership,
actual close and physical recovery (`strata-parity-narrow-operators-20261020`).
The full offline set passes402 ordinary/128 matrix layouts.

All22 original expert-shaped timing cases pass and recover too. The diagnostic
`strata-parity-narrow-matrix-timing-20261020` observes GU count3/7/31 at
61.1/65.4/85.2us versus371.5/415.4/410.7us in the earlier microcheck; DOWN at
25.3/25.0/30.4us versus104.3/115.2/100.8us. This is not whole-request speed or a
controlled Strata comparison; a CPU-only upstream build may overlap. Count16 GU
is82.3us versus69.0us, so there is no claim every shape improves. First-shape
frequency/compilation transients are not a steady count1 result.

The cold45-token MAX8 trial still fails safely. The source-only public diagnostic
now identifies `native Verify3 projection count`, from executor.rs. KnownPrefix
has reserved MAX8 owners, but its projection authority still admits only3 before
ordinary generation selects its independent four-row policies. Do not globally
set that generation flag or weaken the guard: a genuine, bounded known-prefix
projection scope must select the existing Verify4/MAX8 kernel within its own
original request authority. Failure recovery is confirmed and normal admission
restored (`strata-parity-narrow-max8-short-20261020`).

Attention candidatef161190a bounds scoring to the actual causal prefix while
keeping full capacity owners/strides and unread future scores out of selection.
See `kuno-strata-attention-active-prefix.md`. Source0a6c58d2 / SHA7398ba91 builds
this plus the narrow pipeline and native GPU sampler. Its dedicated long-profile
warmup is73 public tokens (two reference lines); the following two measured
4245-token inputs are unchanged. This isolates ordinary large-prefix diagnosis
from the separate failing cold45 short-path test. The short correctness check
keeps all three original45-token inputs and must pass before installation.
No production fallback, request truncation or admission relaxation is introduced.

## Fresh same-host upstream reference and complete candidate

The standalone reference is now actually rebuilt and run; it is not a published
claim. Artifact directory `strata-parity-reference-build-20261020` contains the
unchanged pinned MIT sourcefb58e0db and ggml3cf03257, image provenance and public
transient client. AOTbmg-g31 binary SHA256
`16a6ed700650b61675ac9ae8f8fc3e1c803b566b120f70495b599b07ff7095b3`.
Only the1,538,035,200-byte target dense pack is regenerated; original83.62GB
shards and existing Q2 MTP runtime are reused read-only. No duplicated expert
pack, new host runtime installation or upstream source change.

`strata-matched-reference-20261020` passes3 requests, zero reused/read-skipped
prompt tokens, noncancelled clean exit, actual delegated cgroup containment,
swap/core0 and physical GPU recovery. Same rendered45/4245/4245 public prompts,
seed0/temperature0/cap32/context32768/int8KV/PLEdirect/prefill4096. Warm long
prefixes **5.5006/5.3515s**; first4096-token chunk4.127/3.987s. First45 prefix
0.9581s, observed startup34.412s. All three happen to finish at4 outputs; do not
use these short responses as sustained decode rates.

Exact long fixture is26,610UTF8 bytes, rendered SHA256
`8338444d3a49e2e35b3c34393f44fa7f3198845e780e6c537c8e2f7ef6e9e384`,
u32le-token SHA256
`c5e2b138597af3389ccd4b18638e76bf6fce41209f0069f95e933d45f667cfbd`.
These are identities of compiled PUBLIC fixtures, never private content hashes.
Tokenization occurs once on already-rendered input with special parsing; no
second chat template. Upstream reports11,608 expert slots/22,556MiB cache,
47,962MiB arena and2,383MiB free VRAM. Backend allocation/weight-row-cache history
can differ. Input45 warmup differs from the separate instrumented73 warmup.

Important decode bound: upstream `--spec4 --suffix-draft3` automatically sets
MTPcap4 but verifier6 (generate.cpp2218–2220); INFO truthfully reports spec6.
Kuno Config::native(4) is not the same verifier configuration. Prefix batching
is independent of this S and ends before first verification, so the prompt
comparison is useful; a fully matched sustained decode arm still needs explicit
`--mtp-max-t4` upstream with verifier4. Do not label decode parity from this arm.

Sourcef4ee071e / SHA62f088fc (`strata-parity-max8-scope-slm-20261020`) fixes the
real MAX8 projection scope using a live ShortPrefix-derived opaque window bound
to the original admission ledger, model binding and nonzero request nonce. Its
scope checks each count and restores on error/unwind. Original generation flags,
GDN/PLE policy and Verify4 counters are not changed or credited. Original
Verify8 kernels, span/finite/fault checks and complete receipts stay authoritative.

It also reuses operand shared memory only after the final actual workgroup
handoff barrier. All128 actual shader variants plus CFG/lifetime and negative
stripped-barrier CPU controls pass in the preparation. Real f4ee GPU matrix26
plus original39 controls pass, then the previously failing cold45 request and
two4245 requests all answer correctly with original erase/close/recovery.
`strata-parity-max8-slm-complete-20261020` reports load100.556s, cold45 prefix
10.385s (includes first-use compilation), unprofiled long **17.0194/15.1360s**.
This is progress from the installed24.9658s, still about2.83× the fresh upstream
warm long prefix. This artifact is not installed; tools/semantic repeats,
warm-short/tail coverage and final full combined controls remain acceptance work.

The earlier instrumented0a6c58d2 sample reports warm4245 prefix17.3564s,
experts9.2190s and attention6.3144s. Native timestamps for one real4096-token
expert layer cover2,955 calls: matrix58.682ms, dequant19.806ms, ordered copies
29.202ms, gather3.517ms, activation2.761ms. Diagnostic serial recording excludes
preceding barrier gaps and cannot replace complete performance. The active QSA
score-grid bound has little measured family-level speed effect so far; do not
call fewer empty CTAs a proportional speedup. A fresh upstream phase-timing arm
and connected fixed-inner QSA specialization are now being prepared.

Source75211452 / SHA6321485d adds six original GPU QSA selection controls in a
separate `attention_checks` inventory (full26+39 unchanged). It builds; actual
71-control hardware acceptance is pending. See
`kuno-strata-attention-numeric-controls.md`. No private API/protocol/configuration
or frontend consumer changed. The accepted installation is stillfb31e124.

## Compiler specialization and verified timing attribution

Fresh upstream phase timing uses the same pinned binary and same public requests,
adding only `STRATA_PREFILL_TIMING=1`. The separate presence-tested SYNC debugging
flag remains absent. `strata-matched-timing-reference-20261020` completes, validates
both4244-prefix summaries, contains the actual process under swap/core0 and
confirms physical recovery. Instrumented repeat6.290s versus clean5.352s; queue
event intervals include idle time and endpoint overhead, not isolated kernels.
Repeat QSA attention680ms, GDN projections158ms, recurrence227ms and output96ms
highlight remaining math gaps. Expert dequant875ms, GU+SwiGLU545ms and DOWN with
lookahead gaps2358ms cannot be added to independent host times. Host staging319ms
and copy wait87ms are not a measured PCIe bandwidth. Full public source/client
provenance and attribution are retained in the reference directory's TIMING.md.

Sourcee9b6bd4c / SHAcf867264 (`strata-parity-qsa-specialized-controls-20261020`)
expands the complete selected prompt QSA K8/twelve-head scalar pipeline without
changing other attention/decode/MTP shaders or fixed barriers. All original9
actual GPU scalar-versus-candidate partial/stat/fault comparisons complete with
bitwise equality on their bounded fixtures. The combined receipt still fails
because the independent ranked selection control disagrees at start2050/count3.
Full physical recovery/normal admission/zero driver fault delta are confirmed;
this failed receipt is not an80-control acceptance. A stricter non-dyadic and
nonzero query-row-offset extension is being prepared before selection.

The selection failure is now concretely diagnosed. Positive start0 passes; later
ranked case has244 mismatched output words and fault0, with clean owning tails.
A CPU simulation of the old radix key with `v+0` optimized away reproduces exactly
244 mismatches: negative zero and positive zero incorrectly receive different
ordered integer keys. Fix40653af1 canonicalizes IEEE zero bits before the key
transform. The independent numeric oracle, original six cases, finite/status
faults and all comparison thresholds are unchanged. Initial fixture failure from
upload_f32 rejecting deliberate NaN poison was separately repaired by uploading
its exact wiping raw bits; no production upload guard was relaxed.

The whole cf867264 public profile also completes all3 correct requests and actual
close/recovery. Warm4245 prefixes14.4097/14.4597s, QSA family1.8111s versus2.9727s
in the earlier comparable profile. Total expert interval8.3447s, attention4.6148s.
Routing group count26570 versus26673 differs slightly, so do not attribute every
whole-request millisecond solely to QSA. Both use the same public input count and
20.0257-trillion useful-matrix-FLOP estimate. Finer numeric controls and final
semantic/thinking/tools acceptance remain required; nothing new is installed.

GDN source987d1977 similarly expands the complete selected32-scalar state
recurrence while retaining all original FMA/fault/barrier order. CPU Naga/SPIR-V
acceptance only; its original-versus-candidate real-arena10-case golden control
is being integrated before GPU acceptance. An optional pure-Rust CubeK dense
projection path is also under isolated review; it is not yet a selected dependency
or measured speed result. No frontend/API contract changes.
