# Complete N128 small-expert temporal carry

Backend-only continuation after the complete local-prefetch acceptance.
Implementation history and completed acceptance are recorded below.

**Completed status, 2026-10-24 UTC:** the full candidate now passes CPU,
numerical and both model acceptance chains. It is neutral/slightly slower and
its carried small kernels spill registers. Keep the complete variant. Exact
results follow the implementation history; no installation or parity claim.

Initial source a7fd8ded fails compilation because the new shared module did not
explicitly import `half::f16`; its parent scope does not export that concrete
type. Preserve the failed full-build log. Add the explicit import and rebuild
the same complete candidate. No GPU test has started.

Forward source79b80db3 compiles. Its first four macro layouts and cohort GU pass
the actual carried proof, then the cohort DOWN negative accepts a vec4-to-vec3
type mutation. The copied stride validator allowed its dynamic-INFO alternative
to bypass the exact typed constant cohort stride. Require that typed stride
for the actual cohort specialization, and do not traverse OpConstant literal
bits as definition IDs when proving dynamic INFO ancestry. Apply the same
strict rule to the retained local validator. No matrix arithmetic changes.
An isolated CPU harness includes the actual forward validator with only test
cfg/visibility adjusted: all six available genuine layout0 modules and their
negative mutations pass. The earlier instrumented harness (v2) changes error
text to identify the accepted mutation; its inherited only-visibility flag is
inaccurate. Preserve it as diagnostics, not an exact-source acceptance claim.
Full422/all48 layout acceptance remains pending.

The new multi-slot pair now names its real reference explicitly as the preceding
local-prefetch arithmetic, rather than inheriting the full-walk reference label.
All original pair reports remain unchanged.

The accepted N128/four-C small macro and cohort consumers have zero spills but
are slower than the earlier N64/two-C carried consumers. Their K16 arithmetic
uses only current-iteration operands. This candidate combines the N128 owner
and publication geometry with genuine chronological K32 loop-backedge carry
for BOTH small paths. Large eight-C projections and experts remain the complete
accepted local-prefetch implementation. All earlier variants remain selectable.

`strata-small-local-carry` / `--small-local-carry` requires the whole
`strata-local-prefetch` bundle. Both actual production selectors choose an
explicit compile-time arithmetic parameter. It uses the retained, unchanged
four-C `native_wide::balanced_wide32::compute`: load bank0 before the loop,
load bank1 before current products, consume K16 in original order, refill
bank0 at K+32 only when another iteration exists, then consume bank1.
Both admitted roles have K divisible by32. No final speculative operand read.
That arithmetic spilled in the earlier large wrapper; its register pressure
in these small wrappers remains unproven.

Original explicit `try_expert`, cohort `launch`, compiler definitions and all
1747 old numerical controls always select carry=false. The production selectors
use the new feature. Compile-time specialization participates in the genuine
CubeCL kernel identity; no runtime scalar, descriptor, owner, group status,
packet word, native role name, guard or physical grid changes. Existing native
roles remain authentic for both arithmetic specializations. No vendor runtime
extension is required. Feature-off rejects a requested carried host launch.

The same initialized padded inputs, N128 grids, four C outputs, independent
eight-member fallback publication, GU checker, ordered DOWN, six owners,
cohort packet, source/release phases and erasure remain. Actual new subcounters
identify carried macro32/64 and GU/DOWN cohort launches, beneath the existing
N128 family counters. No older carry/preload counter receives false credit.
Old tile8 is unchanged. No new request state, GPU allocation or persistence.

Acceptance must retain all421 AppCPU and 1747 GPU controls. One additional CPU
test compiles all48 genuine new small layouts and checks the actual initial,
current and carried operand origins, distinct row banks, chronological four C
chains, bindings, INFO/typed-vec4 stride units, WG/SG, publication and negative
mutations. The original local88 and large carried40 layout controls remain.
Planned full422 / feature-off421 AppCPU; unchanged18/86 runtime and8+9 metadata
and10 native controls. Add116 paired old/new/F64 complete-owner cases and4
mixed-slot cases: planned1867 fresh GPU controls. Existing numerical tolerances,
invalid-route sentinel and nonfinite semantics stay unchanged.

Compose the entire feature, production selection, compiler checks and both
paired consumers before live tests. Then require both original complete model/
compiler/counter/capacity/erasure/recovery chains, actual ISA/spills and the same
original first/repeated CLEAN gate. Do not credit source spelling, estimated
cycles or an isolated microbenchmark as prompt-speed acceptance. No API,
Vashti, installed service, driver, boot or release change is authorized here.

## Completed acceptance and preserved failures

Source `94613ad7929378ea5761ab94ea2037d5530c7768`, worker
`6a4f0eca6253e0285fba4212ec7b2c75cf0875af03d3e44b70697124d4e3d779`
passes all422 full / 421 feature-off same-source AppCPU controls, the unchanged
18 / 86 runtime controls, 8 + 9 metadata and 10 native controls. All48 new
carried layouts, preceding88 local layouts, original429 WGSL / 320 matrix
controls and all previous module/resource checks pass.

Fresh GPU execution passes all1867 controls: every preceding1747 plus116 new
regular/cohort old/new/F64 cases and4 mixed-slot pairs. Both original complete
model/compiler/source/owner/capacity/erasure/recovery chains pass. Every request
produces four outputs; this is not sustained decode throughput. The final proof
explicitly records `fresh_public_numerical_execution=true`. No old numerical
receipt substitutes for the new worker.

| Mode | Warm73 | First4245 | Repeated4245 | Model load |
| --- | ---: | ---: | ---: | ---: |
| Normal scheduling | 3.883731470 s | 11.458690320 s | 9.347526684 s | 81.690894022 s |
| Sampled profile | 3.913913730 s | 11.432224226 s | 9.350972261 s | 78.902582317 s |

Repeated normal time is0.42% longer than the preceding9.308575798 s observation;
this is not variance-controlled or a speed gain. No original CLEAN ran because
its trigger was not met. Best prior CLEAN11.817666824 / 8.323282816 s versus
Strata5.5006 / 5.3515 s remains authoritative; the goal is active and unmet.

CPU failures a7fd8ded and79b80db3 described above remain preserved. Build v3
passes. First acceptance preparation used a missing optional zero counter in a
synthetic metadata fixture and stopped before any GPU execution; its partial
forward-harness-v1 directory remains. Forward harness-v2 uses explicit zero
defaults and passes18 metadata controls. These translated prior public profiles
are CPU bookkeeping fixtures, not new numerical evidence.

Read-only preflight v1 then referenced `runtime-cpu-v1` logs, whereas the actual
completed builder emitted `-v2` logs. Preserve the failed preflight and wrapper;
forward wrapper-v2 names those actual completed files. Read-only preflight v2
passes. Only then was the one fresh complete GPU unit started. All18 original
model-chain scripts are preserved with path relocation only; original numerical
and model assertions remain. New exact counter laws require actual carried
subcounts equal the existing N128 family counts, not invented old-kernel credit.

## Actual compiler and phase evidence

The export contains234 actual compiler programs, including12 small programs:
both genuine arithmetic specializations for each of four macro roles and two
cohort roles. Names alone cannot distinguish the specializations. A separate
read-only audit binds the exact original chunk hashes and classifies the actual
final native instruction inventories: four DPAS instructions for each short
K16 loop, eight for each full-carried K32 loop. It does not guess from export
order, estimated cycles or program names.

All six short programs remain zero-spill. All six carried programs spill:

| Carried role | Instructions | Spills:fills | Sends | GRF |
| --- | ---: | ---: | ---: | ---: |
| Macro32/64 GU, each | 4163 | 12:322 | 485 | 128 |
| Macro32/64 DOWN, each | 5453 | 44:434 | 591 | 128 |
| Cohort GU | 3642 | 11:176 | 468 | 128 |
| Cohort DOWN | 4226 | 9:182 | 574 | 128 |

Both large dense and both large expert programs remain zero-spill; the old22
small programs and original two QSA assembly hashes remain unchanged. These
are actual compiler facts, not measured instruction latency.

Matching serialized layer24 sample: macro26.813109 ms versus previous26.528896,
cohort7.691611 versus7.891718, large GU8.050415 versus8.063954 and large
DOWN3.331454 versus3.334012. Whole normal experts5.696305728 s and
attention2.436403882 s, compared with prior5.663845805 / 2.431003274 s.
Overlapping/serialized diagnostic intervals must not be summed into normal
wall time or extrapolated across both chunks. Spills are real, but these
observations do not prove that they alone explain the missing gain.

Read-only compiler review exports the accepted BF16 dense control's native
code: one K16 loop has8 input-buffer loads,16 weight-buffer loads and8 DPAS
instructions. Mesa26.2.5's actual cooperative-matrix lowering emits packed
scalar dereference loads; vector-pointer width cancels in its stride conversion.
Do not treat a wider Rust vector alone as proof of better loads or blindly
transpose the already matching A/B register layouts. This is a BF16 dense
control, not an expert timing or a dump of oneMKL's opaque GEMM. Pinned Strata's
Intel/non-HIP `prompt_f16()` returns false; no alternate FP16 precision shortcut
was established. Source/code provenance is in the artifact map below.

Next compose [granular workgroups and interleaved operand carry](kuno-prefill-small-granular.md)
for BOTH small consumers. The complete new policy addresses register liveness
and available workgroups together, preserves four-C reuse and keeps every
existing variant. It is not implemented or speed-accepted yet.

## Recovery and artifacts

`kuno-small-local-carry-v1-complete-20261024` exits0. Final17:32:56 UTC and
independent17:34:28 UTC snapshots show Intel42,278,912 B, NVIDIA compute0 and
desktop621 MiB, all five kernel fault classes0, all three services HTTP200,
admission inactive/unpaused/no recovery/waiting0/completed261. No test worker
remains. No reset, driver, boot, installation, release, API or Vashti change.

Old refresher3062861 and its own sleep were identity-verified and stopped.
New refresher3184527,parent3556 is bounded16:55:17..20:55:17 UTC. Verify its
actual identity and credential before reuse. It stores no password.

Artifacts under `/media/jmicko/faststore/kuno-development`:

- `small-local-carry-root-build-receipt-20261024.json`, full AppCPU-v3,
  feature-off AppCPU-v1,72-feature `strata-parity-small-local-carry-v1-20261024`
  release worker and offline log, original runtime/deep-runtime-v2 logs.
- Full build unit-v3 and read-only preflight-v2 pass; earlier failed builds,
  validator harnesses and preflight-v1 remain. Actual-validator-v3 checks all
  six genuine exported layout0 programs and negatives without GPU execution.
- `small-local-carry-forward-harness-v2-20261024/`: all original18 scripts,
  appended exact counter laws and18 CPU metadata controls.
- `strata-parity-small-local-carry-v1-20261024-{operators,profile,normal}` and
  `strata-parity-small-local-carry-v1-20261024-expert-cohorts-public-summary.json`.
- `small-local-carry-v1-{acceptance-freeze,complete-isa-export,normal-safe-phases,
  profile-safe-phases,actual-isa-provenance,final-state}-20261024.json` and
  `small-local-carry-independent-final-state-20261024.json`. No CLEAN receipt.
- `dense-operand-code-audit-20261024/`, `mesa-cmma-load-audit-20261024/` and
  `dense-operand-read-only-findings-20261024.json`: pinned code/source only,
  no new GPU execution or performance measurement.
