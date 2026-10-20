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
