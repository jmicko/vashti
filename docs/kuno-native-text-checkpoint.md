# Native text checkpoint, 2026-10-06

The user has resumed the goal, authorized both GPUs and asked for continued
native optimization. The objective remains incomplete. Use **10 decode tokens/s**
as the agreed working target and do not repeat Ollama baseline tests. The older
paused receipts below remain historical evidence, not a current instruction to
stop. Root owns GPU/service operations; source-only agents use isolated worktrees.

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
