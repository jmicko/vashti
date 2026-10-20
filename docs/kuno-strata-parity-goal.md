# Strata parity goal — 2026-10-20

The user explicitly requested a new goal to match Strata. Work remains Kuno-only
on `codex/strata-rust-port`, using the agreed integration032e2cdd and the installed
worker c8db19de / SHAfb31e124 as the accepted baseline. Preserve the canonical
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
RAM, different input/configuration and an earlier host. Kuno's current4245-token
compiled-public check takes24.9658s, toolbox4207/4293 prefixes29.506/29.920s.
Do not claim parity by comparing unlike workloads, truncating tools/history,
changing the quant or reporting isolated first-token speed. Before declaring
parity, require an agreed matched local upstream workload/settings/timing scope
or a sufficiently documented equivalent reference; distinguish prompt/decode,
load/first-use compilation, final erasure and whole-response time.

## Latest combined operator checkpoint

The accepted installation above is unchanged. Sourcee9b6bd4c /
SHAcf867264 completes cold73 and warm4245/4245 public requests with correct
arithmetic and actual erasure/close/recovery. Its last **profiled** prefix is
14.4597s, expert interval8.3447s and QSA1.8111s; prior comparable QSA was2.9727s.
Clean upstream5.3515s remains a distinct unprofiled reference; parity is not met.

Source7ada5edd / SHAbba3d0f3 builds the complete GDN scalar-state path and
opt-in CubeK cyclic-double-buffer dense family. All30 full original-weight
FP16/BF16 matrix cases report zero numerical mismatches, but the overall public
gate rejects before its final receipt: three newly added non-dyadic QSA controls
differ from the unchanged scalar shader by up to5.72e-6 in partials and7.63e-6
in statistics. The original nine dyadic cases remain bit-exact; these do not
justify accepting the stronger cases. No numerical threshold is weakened.
Receipt `strata-parity-coupled-complete-operators-20261020` records failure and
actual physical recovery, all services normal and zero added driver faults.

The missing compiler connection is concrete: expanded prompt-only QSA lacked
`STRATA_ROUND_PRODUCTS`, so the native graph did not apply its existing
NoContraction product-rounding pass. Source6256ebea / SHA4b4e5b0b adds that
marker only to the optimized prompt shader, leaves the scalar golden reference
unchanged, and retains explicit chronological PV FMAs. CPU actual-postpass
controls and SPIR-V validation pass. GPU97-control and complete-request
acceptance remain pending. The full inventory is30 matrix (including six
actual CubeK launches),39 original router/dequant,6 causal-selector,
12 unchanged-scalar QSA and10 unchanged-scalar GDN controls. The public helper
requires every invariant and actual recovery before writing `passed:true`.

All artifacts are under the development base. Staged API acceptance additionally
covers actual MAX8 terminal counts1..8 ascending and descending on the same PID,
existing64/65/66 thresholds, thinking/tool/result, repeated semantic code and
privacy/physical recovery. These are prepared gates, not live passes yet.
No API, protocol, model identity, consumer or installation change is claimed.

## Connected high-potential work

- Matrix: improve compiler-visible K64 vector/stage/MMA execution while retaining
  the accepted paired32KiB shared operands. The prior register-first variant
  materially regressed; do not confuse source prefetch intent with proven speed.
- Native graph: investigate redundant global memory boundaries between command
  segments while preserving original per-node physical-range hazards, crossqueue
  timelines, external entrance/exit, host visibility and final actual join.
- Short prefix: extend the existing genuine MAX3/64 path to connected MAX8,
  no vocabulary head, original device residual ownership and exact known-input
  MTP teaching. Preserve separate real nonce/purpose/capture identities, complete
  target/draft cursors, erasure/rebind and the final ordinary count1 input.

Only compiled-public profiling may retain aggregate expert-member histograms
and arithmetic-work estimates; no identities, routes, text, hashes of content
or private request profiling. Production worker logging/persistence stay unchanged.
Use the real GPU numerical controls before accepting complete generation, then
matched speed, thinking/tools/semantic repeat/privacy, actual unload and physical
recovery. Source isolation does not isolate GPUs: only the root runs shared tests.

Current review ownership: matrix agent603a1de7 owns strata_matrix.rs; runtime
agent owns strata_graph.rs; short agent owns controller/target/MTP/IO plus one
immutable known-input headless recording identity in server/base.rs. Root owns
integration, public profiling geometry, GPU testing and installation.

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
