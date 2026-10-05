# Native text paused checkpoint, 2026-10-05

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
5. Only then connect measured CPU/GPU splits into the complete Rust model and
   establish quality/options/tools/long contexts/private worker acceptance,
   scheduler integration and native-default/managed `ollama.` catalog routing.

System memory cache/accounting references:
[kernel cgroup memory documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)
and [kernel process memory documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html).
Hardware availability and sudo should be checked again after the user resumes;
no credentials belong in this document or inference diagnostics.
