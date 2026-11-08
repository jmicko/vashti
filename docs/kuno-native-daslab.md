# Native DASLab IQ3_S and Strata optimization pass

The user approved switching the native Rust laboratory toward the full
ISTA-DASLab IQ3_S model and investigating Strata's implementation, with a
30 accepted-token/s target. This is internal backend work. Installed Kuno,
ordinary Ollama access, Vashti, public contracts and releases are unchanged.
Do not substitute a pruned model, wrong experts or draft-token counts for this
target. The original arithmetic path remains available.

## Source and licensing

Strata is public source under MIT, pinned for this pass at
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`:
[engine license](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/LICENSE).
Its engine may be studied and adapted with the required notice. A clean-room
process is not required by that license. Reused third-party files must be
checked individually: GGML is MIT, vendored SYCLomatic has Apache-2.0 with the
LLVM exception, and optional residual control vectors have separate terms.
No optional behavior-changing control vector is imported here. Runtime
inference stays in Rust; independent C fixture generators are test tools only.

The Intel results reported upstream use MTP/speculation and are not matched
local measurements of Kuno:
[Intel acceptance reports](https://github.com/Niko1221/Strata/blob/fb58e0dbc8399662c0e47c76578c6e878b14f6cf/docs/INTEL_ARC.md).
The useful mechanisms to evaluate separately are packed integer dot products,
reusable GPU buffers/graphs, and multi-draft target verification.

## Original artifact and geometry

Repository `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`, folder `IQ3_S`.
Both original shards are fully SHA256 verified, total83,617,662,656 bytes.
There is no merged copy or duplicated payload. The publisher's model card
declares Apache-2.0. See the [complete original-artifact receipt](../apps/kuno/benchmarks/text/receipts/daslab-iq3-artifact-20261009.json)
for exact file sizes, hashes and header-derived inventory.

The label describes a mixed format, not uniformly three-bit weights:

| New format | GGUF ID | Elements/block | Bytes/block |
| --- | --- | --- | --- |
| IQ2_S | 22 | 256 | 82 |
| IQ3_S | 21 | 256 | 110 |
| Q5_K | 13 | 256 | 176 |
| Q2_0 | 42 | 64 | 18 |

Q2_0 is distinct from Q2_K and ternary TQ2_0. Its consecutive two-bit codes
represent -1,0,+1,+2 multiplied by a half-precision scale. Nine routed down
matrices use it. Other tensors use already supported IQ3_XXS, IQ4_NL, IQ4_XS,
Q4_K, Q6_K, Q8_0, F32, F16 and BF16. The explicit profile validates all1224
canonical tensor names/raw types, all48 routed triples and existing bound
dimensions before allocation. It is never inferred from a failed load.

The initial laboratory retains144 slots/layer in72+72 segments and the checked
31GiB device envelope. Exact expert payload is14,144,716,800 bytes; separately
rounded physical bank charge is18,723,373,056 bytes. Static cache admission is
explicit5 or8GiB; these are not claims about every driver allocation.

## Implemented acceptance

Core source `d11bd15` integrates reviewed original scalar and AVX2 decoders,
F32 activations and the existing64-lane reduction. Independent SHA-pinned
GGML b11232 C records verify all20,032 new scalar/SIMD values, native-shaped
CPU experts, and malformed/no-write guards. Core all-feature tests pass190
with nine intentional opt-in ignores, strict default/all-feature Clippy passes.
See [fixture provenance](../crates/kuno-text/tests/fixtures/daslab-b11232/README.md).

Laboratory connection is `985ffb5`: original byte-addressed CubeCL decoders,
strict `daslab-iq3` profile, generic native grouped MoE, fixed public token0
reference writer, and seven-format original-value/projection probe. Host tests
pass598 plus two shader suites before two added connection tests; all seven
profile/connection tests and strict all-feature/all-target Clippy subsequently
pass. Both-card frozen v92 operator acceptance verifies56128 exact C value
bits,273 projections,28 cooperative and28 subgroup finite/fault cases. Owning
cleanup and unchanged classified GPU faults pass. This establishes operators,
not actual-model throughput or private-runtime acceptance.

Public reference input is compiled token0 (`!`), seed424242/temperature0.
A distinct four-position DASLab CPU reference completes; no Unsloth logit file
is reused. Its SHA256 is
`26deb2d125bfb8a402181c84c687dee77e9f2e2fdf752010c819e66bca24d95a`.
The first whole-model attempt correctly rejected a generic grouped-MoE check
that still admitted only the older quantization triples. `67a01bc` admits the
inspected original DASLab triples; a host test validates all48 actual layers.
Frozen featured v93 accepts every position0 logit, then fails before position1:
shared dense dispatch admitted raw types absent from its WGSL decoder. This
was a real connection bug, not a model or GPU failure. `aae7e3b` restricts that
fast path to its actual decoder whitelist and uses the correct generic packed
kernel for other formats. Frozen v95 then passes all four complete248320-wide
CPU vectors, greedy IDs and original owning closes. Its timing is excluded
because a compiler ran concurrently; it is numerical/lifecycle acceptance only.

The independent complete32-position CPU reference SHA256 is
`635385075cd6a46241c4451726b5a3f95d40bb98e2f6e96e89e2bfbfd5db45d6`.
Original GPU v95 passes positions0..12, then fails position13 full logits with
max absolute error0.3575926/max old-gate-scaled error58.79485. Greedy still
matches. The exact router/layer cause is not located; resemblance to the older
Unsloth near-tie failure is an inference, not diagnosis. Keep the failed run.
No accepted full32 timing and no relaxed tolerance or forced expert selection.

`07be82e` adds an explicit DASLab prefix13, with12 decode continuations and
context14, before that preserved discontinuity. The complete original32 file
is still parsed/validated, including its unused suffix. Fixed coverage,
profile selection, complete vector checks, correct-expert trace membership,
erasure and receipt scope remain enforced. This is a conditional diagnostic,
not ordinary cache throughput, general quality acceptance or a long-run claim.

## Measured original-arithmetic kernel control

Source `b429ea0` adds the opt-in four-row subgroup MoE leaf to the original
Unsloth Q3 diagnostic. Both-card independent three-profile tests agree exactly
with the old GPU path, including failure/gap/owning checks.

Same frozen featured v91 artifact, compiler-idle correct-expert residency
replay over23 positions (22 timed continuations):

| Kernel | Elapsed ns | Accepted continuations/s |
| --- | ---: | ---: |
| Original with empty-segment skipping | 2364290984 | 9.3051 |
| Four-row subgroup with the same skipping | 2267664814 | 9.7016 |

One ordered pair gives4.09% lower latency, not a general speed guarantee.
Both runs match all complete reference vectors/greedy IDs, use10560 GPU expert
calls with zero CPU calls/uploads/physical read bytes, and pass all owner
closes. Thirteen major faults and4096 observed write bytes remain in each
timed public diagnostic. Training/binding/prefill/preload/later full cleanup
and physical recovery are outside the timer; numerical/membership checks and
request I/O close are inside. This is the known23-position prefix before the
preserved Unsloth full32 routing discontinuity, not full32 acceptance or an
absolute hardware maximum. A first control passed arithmetic/cleanup but failed
to write its receipt because the new output directory was root-owned; it is
retained as a failed run, not selected as an accepted timing.

## Strata-inspired candidates and acceptance boundaries

The default original arithmetic is unchanged. No candidate is selected for
production/private requests by these laboratory switches.

- **Packed integer MoE:** `2375412` implements original IQ3_XXS/IQ4_NL/IQ4_XS/Q8
  weight decoding with two residual Q8 activation planes and hardware DOT4I8.
  `b82eefd` adds IQ2_S/IQ3_S/Q2_0 and all seven actual DASLab routed triples.
  Both-card v101 operator tests pass nine profiles, independent integer/C
  oracles, full intermediate/output/gap/fault checks and81 erased owners
  totaling15,630,336 bytes per card. Peak shared storage24,196 bytes. This
  changes arithmetic: original F32 error is separately reported, not asserted
  identical. The protected model connection passes all13 original full vectors and greedy IDs; timing below is conditional ideal replay. A first run
  completed arithmetic but could not write a root-owned receipt directory;
  only the fresh repeat is accepted.
- **Q6_K head candidate:** `8dcec0c` adds original packed weight projection
  with one reusable private two-plane input preparation. Both-card v100 tests
  pass38,416 rows at four geometries,6400 independent C values, strict integer
  oracle/fault/tail/stale/owner checks, with32 erased owners/1,245,184 bytes.
  `f671e6b` preserves an explicit finite-MAX guard, and `bb856a5` configures the
  empty transient stream before public source allocation. Actual248320-row
  head is521,472,000 packed bytes with5760 scratch bytes. `36cf286` connects the protected count1 final head, reusing already-dead NORM storage and restoring its full owner. Four- and13-position original CPU vector/greedy gates pass;13-position timing below is slower than the original control. Original-F32 bit parity is not claimed.
- **Original F32 dense decode:** `7d06efb` expands supported shared
  dense decoders to IQ4_NL/IQ4_XS/Q5_K, retaining original activations. Both-card v102 tests pass120 additional original dense/fault/subview cases and18,112 exact C values. Actual model13-position original gates pass in v103/v104. Source, operator and model gates remain distinct.
- **Original F32 shared MoE codebooks:** `d424727` moves large codebooks
  from Function arrays to workgroup storage while retaining the exact original
  64-lane reduction tree. Lack of Function arrays in SPIR-V alone does not prove
  Intel driver spills or a gain; both-card v104 all-nine-profile operators and protected13-position model tests pass. Initialization cost and physical timing decide.
- **MTP append seam:** reviewed `2dea140` adds teacher-forced KV-only append.
  Six focused host tests check cache bits, subsequent full-step parity,
  repair/replay, preflight preservation and failure poisoning. Eight executor
  calls replace51 full-step calls, not a device speed result. Strata's accepted
  prefix replay is documented in [MTP notes](kuno-native-mtp-prefix-notes.md).
  Rust target batch verification, controller and all-state commit/rollback
  remain necessary before an accepted speculative throughput claim.

Protected helpers independently confirm physical memory recovery and unchanged
classified faults after operator tests; their kernel-level receipts deliberately
do not claim physical host recovery, private execution or model performance.

GPU and service use remains root-coordinated even with separate source worktrees.
Run physical tests without swap/core dumps, keep private requests out of saved
fixtures/logs, restore normal admission after every attempt, and verify actual
physical GPU recovery. Do not reset/unbind/reboot GPUs or change drivers/clocks
to chase this result. Required Vashti changes: none.

## Conditional DASLab timing through v109

All runs below use the same public `!`/seed424242/temperature0 prefix13,12
checked continuations,8GiB cache and exact correct-expert replay. Training and
preloading3,625,113,600 original bytes are outside timing. Replay checks every
original248320-wide CPU logit vector, greedy ID and selected-expert membership.
Each accepted replay records5760 GPU expert ranks, zero CPU misses/uploads/
weight-transfer bytes and observed physical read bytes0. The full original32
failure remains; none of these numbers establish ordinary demand-cache speed,
long-run quality, private execution or30tokens/s.

| Frozen run | Choice | Elapsed ns | Accepted continuations/s |
| --- | --- | ---: | ---: |
| v101 | Original F32 | 1184704305 | 10.1291 |
| v103 | Original F32 plus dense extension | 1149079282 | 10.4431 |
| v103 | Initial integer MoE | 24347370621 | 0.4929 |
| v104 | Original F32 control | 1109469903 | 10.8160 |
| v104 | Repaired integer MoE | 1073637638 | 11.1770 |
| v104 | Original shared-codebook MoE | 1097770028 | 10.9313 |
| v104 | Original MoE plus prepared Q6 head | 1221968213 | 9.8202 |
| v105 | Original F32, with host diagnostics | 1220809483 | 9.8295 |
| v106 | Original, larger append-only cache | 1283836016 | 9.3470 |
| v107 | Original, larger FIFO cache | 1110630256 | 10.8047 |
| v109 | Original, larger FIFO control | 1141680997 | 10.5108 |
| v109 | Once-prepared integer MoE, FIFO | 942769705 | 12.7284 |
| v109 | Prepared MoE plus aligned Q6 head, FIFO | 945901792 | 12.6863 |

The matching v103 control isolates the21.19-fold integer regression. Source
`6189e69`/`5b976e2` replaces repeated256/512/2048-way codeword lookups with one
128-lane initializer and direct constant stores. Every original codeword/index
and arithmetic schedule is independently verified; v104 both-card all-nine
profile probes pass. The repaired model timing confirms the large regression
is gone. Its3.3% advantage over the same-artifact v104 original control is one
ordered comparison, with variation between runs; it is not a sustained gain
claim. The Q6-head candidate is slower in this comparison and stays opt-in.

v104 source is `fb6cd2fa0bb0072c47a33df276b1f5b7ba2ce69d`, featured artifact
SHA256 `4aeff919bf9f03451d8f1c057c852c651df2b056d5533fa66c4012eaa379b16c`.
v105 source `690f0324f2c40edd5ec7e3020cdb97348abef2f1` adds no inference math,
only existing host phase and binding counters plus reference/greedy-check
clocks. Its featured SHA256 is
`13a2bede94b10cc4a0917f1578a6bd814a48764f3045bd49b431772f3195e323`.
Raw `.json`, `.log`, paired source and before/after host receipts are under
`/media/jmicko/faststore/kuno-development/daslab-main-gpu-20261009/`; exact run
names are `daslab-{original,integer,shared,q6-head}-prefix13-v104` and
`daslab-original-prefix13-v105`. All compiler-idle timed runs pass original
full owners and independently restore admission/memory without new GPU faults.

v105 records binding stats `[0,34254,0,272299086,4961847]` (hits, misses,
inserts, creation ns, lookup ns). Zero timed cache hits/inserts are measured. Source review confirms these counters require a present cache: it has reached its entry/key allowance, not necessarily withdrawn. Required-owner pressure may withdraw it separately; these counters do not prove that occurred. Binding creation consumes22.3% of the1.2208-second
replay. Host wall phases: router fences507.59ms, attention/GR enqueue340.77ms,
route/expert enqueue207.79ms, final proofs95.68ms. These are host wall times,
including waits where relevant, not isolated GPU kernel durations. Reference
checks5.33ms and greedy2.25ms do not explain the gap. Next bounded candidates
are retaining complete request bindings, once-per-layer activation preparation,
and aligned Q6 loads; each still needs independent and full-model acceptance.

## Larger exact request-cache experiment

Source `bfadcd4` integrates the reviewed vendor leaf `82aab9c`; `85711cd`
adds the protected profile scope and complete parent host reservation. Default
256/512 entry behavior and256KiB runtime metadata remain unchanged. Only
canonical protected original-lowbit public fixtures may explicitly select
2048/4096 entries with a1MiB/4MiB runtime metadata ceiling. Full actual entry,
bucket, binding-list, INFO and mandatory-owner capacities are accounted before
cloning retained resources; optional cache insertion falls back at the ceiling,
and required-owner pressure keeps its priority. No runtime private-buffer,
family metadata or erasure allowance is reduced or replaced by cache credit.

Main validates the complete selected runtime host ceiling before device creation. Config retains it and rechecks it with each fresh native model plan. With4MiB, exact
combined application/runtime reservation is16,449,536 bytes under16,515,072;
family pending/anchor metadata stays under its own256KiB cap. Default combined
reservation12,517,376 remains under12,582,912. Runtime captures its scope ceiling
once; changing process settings cannot enlarge an active scope. Cached keys and
complete spare capacities retain their existing request-end wiping.

Integrated657 native host tests (one ignored) pass with direct-device/FMA/
work-stealing, including actual-profile/window/owner-policy rejection and exact
host totals. The vendor's105 device-free runtime tests and strict native/runtime
Clippy pass at the reviewed leaf; physical caching throughput is a separate gate.
Featured v106 artifact SHA256
`d0ebc35d6d070b61f816027b007d84de1165304989331b6ecfac084a54808fa9`
is built from `85711cd11e52939754b3a1a9deded75f313af8e2` for that gate.

`b4c07ed` adds bounded FIFO replacement only to explicitly extended full entry
tables. Full old/new key capacities and staging records are charged before
replacement; retired keys are wiped. Default caches stay append-only. The v107
original replay records31,750 hits/2,504 misses and inserts, reducing binding
creation to16.4ms. Compiler-idle v109 independently repeats those counts, with
16.8ms creation and8.8ms lookup. Further binding tuning cannot explain most of
the remaining1.14s: these two costs now represent about2.2% of that replay.
Byte-credit exhaustion before table fullness still safely falls back; the
ceiling does not include opaque driver allocations or whole-process RSS.

## Once-prepared activation planes and aligned loads

`f5d5e97` prepares gate/up input once, then each compact down input once, reusing
the existing complete131,072-byte CPU bridge owner in an ordered GPU-only phase.
All consumers queue before its original same-stream full CPU overwrite. Scratch
uses5,072 logical words/20,288bytes; no extra model owner or weight repacking.
An initial standalone test failed because auto-reflection dropped obsolete
unused bindings while the host still supplied them. Preserve that failed v107
run. `95b31cc` specializes dense live binding layouts and adds a negative
reflection regression. Both-card frozen v108 then passes all nine profiles,
identical prepared-versus-old-integer ACT/PART/merge values, independent integer
oracles, compact-rank planes, complete scratch gap/tail and finite/fault guards.
Each card erases90 owners/15,925,248bytes; helper host checks separately verify
physical memory/admission recovery and unchanged classified faults.

`774f2a9` charges the two extra preparation INFO records per layer separately:
96×2,048=196,608bytes. Hybrid runtime addition is1,245,184bytes, derived ownership
1,212,416, preserving32,768bytes margin. Worst38-tile total65,335,296 remains
under67,108,864. Host/MAIN/owner counts are unchanged.

Featured v109 source is `774f2a9020a03fa350a191fe1825f0ad508bbc13`, SHA256
`1d6b61068d9f491a93404ad941eb4649ac9e407f8f51437babe75f9d035d879f`.
Same-artifact ordered original/prepared pair takes1,141,680,997→942,769,705ns
for12 accepted continuations:17.4% lower latency. Full original CPU vectors,
greedy IDs, actual trace membership, original owning closes and zero CPU calls/
uploads/physical reads pass. Preparation changes arithmetic, as explicitly
recorded; it is not F32 bit equivalence or long/private quality acceptance.

Aligned Q6 loads `bcb9bfc` independently pass both-card v107 tests,38,416 rows,
6,400 C values, fault/partial-view/owner checks. Combined with prepared MoE,
v109 accepts all13 model vectors/greedy IDs but takes945,901,792ns, giving no
gain over prepared-only. Keep the original head for the selected experiment.

## Resident GPU routing: accepted correctness, slower first version

`cae513f` and `8b734fa` add the actual original GPU top10 mapper and sealed
all-resident dispatch. `1742100` restores the original token-zero shared-expert
condition caught by peer review before model testing. The ordinary64MiB IO
limit remains unchanged; explicit ideal-only construction charges two full
owners (128KiB table/2MiB audit),9,183,232 additional runtime bytes and4MiB
host storage under a separate128MiB runtime ceiling. Every token audits all48
literal packets and consumed MAP/PROB/OWNER wires before any family proposal.
Source/graph/slot mappings/bank instances/full physical handles stay frozen.

`f386083`/`536c98e` add/register a compiled-public operator probe. Frozen v112
SHA256 `53e32001997781329b15e5c6c8f2ff1df7fe2c965664786528f17502ba73b95b`
passes both cards:49 independent original top10/canonical cases over all48
frame offsets,24 complete-owner no-publication faults, fresh-owner replay and
16 erased owners/4,849,664bytes. Physical memory, admission and HTTPs recover;
classified GPU fault counts stay zero.

Same-artifact featured v111, source `1742100`, SHA256
`9195a9f7281ffec27d70f571d7144655d4415fd3d6eafecad57199d884ec28de`,
compares both original bank segments in both paths. Original CPU-router bridge
control is1,184,871,333ns (10.1277tokens/s); GPU-routing replay is1,420,875,896ns
(8.4455tokens/s), **a regression**. Both accept all13 full CPU vectors, greedy
IDs and actual trace membership,5760 GPU ranks, zero CPU/uploads/physical reads.
The new path accepts12 owning route audits/576layers and erases1380 complete
owners/820,641,792bytes. Removing the CPU map alone does not prove a speed gain:
GPU waiting moves into other enqueue phases; router-fence time was not solely
readback overhead. Keep prepared-only v109 as the selected experiment.

Next isolate a parallel complete-owner preflight to replace costly single-workgroup
mapper scans without removing checks, then combine the independently accepted
prepared integer projections. Neither next change has source/device/speed
acceptance yet.30tokens/s and full32 original-vector acceptance remain unachieved.


## Parallel routing and prepared composition follow-up, 2026-10-09

Model-tested source `6a32b11` and standalone mapper `416f889` complete the
parallel owning preflight and prepared-integer composition previously pending.
Both retain complete original full-owner/fault/numerical gates. F32 matched
control/routing is11.1205/10.4493tokens/s. Prepared-integer ABBA is
12.7468/14.6064/10.0652/12.8670: one routing run improves, its repeat regresses.
Keep routing optional and prepared-only v109 selected. Thirty accepted tokens/s
is not established. Per-step clocks do not change the original aggregate or
exclude the first dispatch; first three versus later steps can be inspected
without silently redefining the result. B1 observes114,688physical read bytes;
other three observe zero; none upload expert/public weights.

See the active [native checkpoint](kuno-native-text-checkpoint.md) and numeric
foundation receipt for exact artifacts, original13-position gates,12 routing
audits/576layers, checked9,379,840-byte integer routing runtime charge and all
physical closes. Ordinary64MiB ceilings remain unchanged.

Next source-only work is the bounded [MTP all-row target/carry foundation](kuno-native-mtp-all-rows-plan.md)
and original full512-expert Q8 draft on RTX. These do not yet implement accepted
prefix state repair or establish native MTP throughput/private production use.
