# Two-accumulator subgroup cells with authenticated padded inputs

## Completed acceptance (2026-10-24 UTC)

Source `ff74ce03614ec0e565911f2843ce86db31ebdc35` (complete bundle
`79b19229`, include-safe comment fix `ff74ce03`), worker
`7ac745386ddae836ac52ce1d28ded5ae416e37b0707235898e63a5e79a1b28f4`:
404/401 AppCPU,18/73 runtime,8+9 metadata,10 native and all1425 GPU controls
pass. All1305 preceding GPU controls remain. Both original18-script complete
normal/diagnostic model chains retain original capacity, ownership, finite,
status, reuse, rolling and erasure checks. All48 new shader layouts pass actual
SPIR-V/CFG/type/provenance/negative controls. Actual compiler export contains213
programs and22 small macro/cohort programs, all22 zero-spill. The six genuinely
new programs are identified by their actual workgroups128/256/64, not shared
entry basenames; actual scratch4096/8192/2048B and SG16 match. Original two QSA
compiler hashes are unchanged.

Normal warm/first/repeated:3.399544418/11.217620306/9.427012698s;
diagnostic3.387290726/11.376132225/9.530366557s. Normal/diagnostic load
79.785854916/84.232173431s. Normal repeated improves7.104% from preceding
padded-global10.147959844s; first improves13.208%. Warm fixture is73 tokens;
the long requests are4245 tokens, each returns4 output tokens. These are prompt
timings, not sustained decode throughput. This does not beat prior local-route
normal9.360428477s, best original CLEAN8.394039699s, or Strata CLEAN5.3515s.
No new CLEAN threshold was reached, no parity or installation claim.

Same diagnostic sample: macro236 calls25.875936->15.821815ms (38.855% less),
cohort278 calls19.910562->14.076081ms (29.304% less), all matrix708 calls
62.551702->46.626741ms. Hardware ordered-copy608 calls29.173950ms and
dequant542 calls19.256531ms remain essentially unchanged. Tail/wide matrix
families and host capture are also essentially unchanged. These intervals can
overlap; do not add them or compare their boundaries directly to Strata's coarse
phase timers. Retain all complete variants rather than rolling back components.

### Exact acceptance and preserved failed attempts

Artifacts below are under `/media/jmicko/faststore/kuno-development`.
Original AppCPU V1 rejected include-incompatible inner comments; V2 launch
rejected an incorrect supplied source SHA before compilation; V3 completed404.
Original root-build V1 had a Python syntax error before work; V2 completed the
actual build. Earlier CPU preflights rejected stale build/artifact paths; the
V4 exact-prefix preflight passed. Preserve all logs and do not count rejected
launches as executed tests.

The V4 public numerical run exceeded its original900s deadline. Its immediate
cleanup state still had694845440B Intel memory; the independent follow-up at
09:03:11UTC confirmed50688000B, no NVIDIA compute, faults0 and normal services.
The V5 forward harness changes only public numerical deadline900->1800s and
script paths. V5 then completes actual numerical execution and physical recovery;
it subsequently rejects the whole-model accounting law, which still expected
new cohort launches under the old full-walk counter names. This is an acceptance
failure, not a successful whole-model checkpoint. Its original receipts remain.

V6 adds a distinct balanced-cohort partition to the counter validator. The old
helper remains a byte-identical prefix. The new law retains every original
image/launch/capacity/owner law and additionally requires old full-walk cohort
counts0 and exact equality of the two real balanced counter aliases. The actual
long requests report3460 GU and3460 DOWN new cohort launches, old full-walk0;
the73-token requests report613 each. All preceding helper cases stay selected.
An additional21 CPU checks exercise original/new valid profiles and reject
mutated launch, alias, old-family and capacity counters. V6's first preflight
launch omitted its argv and failed before work; the second passes its exact
acceptance prefix404/73/1425.

V6 explicitly REUSES the V5 completed numeric receipt and compiler capture for
the SAME frozen source/worker, revalidating all original numerical checks and
all1425 cases; it does not claim a second GPU numerical execution. Reuse requires
receipt SHA `6c276b7baa247761caf915f387be777faafd322cc34934ebe035354034edc0fb`
and compiler capture SHA
`b46037846a9c8d6bc81189c5e5a3eb65ab7e4a4a5cae6e07a2e87f79c8f28b3f`,
successful exit/recovery and exact source/executable hashes. Both model modes
are freshly executed under the corrected law and all original18-script chains.
No compilation or source modification between V5/V6 is claimed.

Actual completed build artifacts: `balanced-global-app-cpu-v3-20261024`,
`balanced-global-off-app-cpu-v1-20261024`, root build receipt/source seal and
`balanced-global-v6-preflight-safe-20261024.json`. Accepted result:
`strata-parity-balanced-global-v6-20261024-expert-cohorts-public-summary.json`,
`balanced-global-v6-{profile,normal}-safe-phases-20261024.json`,
`balanced-global-v6-complete-isa-export-20261024.json`,
`balanced-global-v5-actual-geometry-isa-safe-20261024.json`, V6 complete log,
acceptance freeze, terminal journal and independent final-state JSON. V6 unit
`kuno-balanced-global-v6-complete-20261024` exits0; final state helper exits0.
The historical V5 driver prints a stale `balanced_global_v4_exit` label; its
artifact paths/unit remain V5 and its failed result is not relabeled.

At09:35:59UTC Intel50688000B, NVIDIA compute0/desktop643MiB, all5 kernel fault
classes0, all3 services HTTP200, admission inactive/unpaused/no recovery/waiting0.
Independent09:39:35UTC check confirms the same state (desktop665MiB).
Installed services, APIs, Vashti/canonical checkout, driver and boot untouched.
Controller was re-read and is ACTIVE/UNMET; an earlier chat summary saying paused
was stale. No automatic goal completion or parity claim is justified.

### Next coherent lead

The current matrix loop loads operands then immediately consumes them; there is
no explicit next-K operand bank. Investigate overlapping genuine next-K16 input
and weight matrix loads with current products in BOTH macro and cohort paths,
using the complete two-C geometry/padded producers/publication/native authority
together. Preserve per-accumulator chronological K16 products, final-read bounds,
all preceding variants and original failure/owner/capacity rules. Authenticate
actual emitted bank dataflow and inspect real spills. This is a source lead,
not implemented or a proven speedup. Copy/dequant remain substantial measured
costs; do not assume further matrix tuning alone closes the whole-model gap.

## Original composition contract

Coherent internal candidate after full1305 padded-global acceptance. Keep the
whole preceding variants and original401 AppCPU/1305 GPU controls. No API,
Vashti, installation, model/context/precision, driver or reboot changes.

`strata-balanced-global` / `--balanced-global` requires padded-global and
propagates `kuno-balanced-global`. Separate genuine matrix identities use
ONE chronological K16 loop, two8x16 FP32 accumulator fragments per SG16,
two independent global FP16 weight-A loads and one shared global input-B
matrix. Each subgroup covers16 real/padded input members x16 output rows.
The existing input producers initialize all necessary padding. Private native
macro admission requires their exact Pad32 identities and complete input
extent, including after checked SwiGLU. Packed cohorts retain their original
proven16-row padding/header/slot authority. No unused-capacity assumption.

Macro32/64 retain original M32/M64,N64 dispatch grids and scalar/owner ABI,
with WG128/256 (8/16 independent subgroups) instead of WG32/64. Cohort16 uses
WG64 (four subgroups) instead of WG32. Subgroup token=row-block*tile+(sg/4)*16,
output row=feature-block*64+(sg%4)*16. Complete real cells are covered once;
fully inactive subgroups terminate without workgroup barriers. All original
independent8-half finite/invalid/sentinel/sticky-fault publication, checked GU
and pending ordered DOWN consumers remain. Tile8 and wide tails stay intact.

New subgroup-local publication scratch is4096/8192/2048B for macro32/64/cohort16.
These are explicit NEW shader resources with actual CPU/compiled proofs and
device-limit checks; preceding kernels keep their exact old resources. No
original private GPU owner, input/output/routing/status allocation, command
tape, retention or public-oracle budget grows. This is an occupancy hypothesis
suggested by the remaining small-GEMM profile and earlier higher-parallelism
cohort path, not a claim about oneMKL's opaque internal tile geometry.

Add actual shader CFG/type/SSA/negative proof, exact subgroup-cell coverage
and actual profile-name control =>404 AppCPU, retaining all401. Actual emitted
FP16 A8x16/B16x16, FP32 C8x16 and subgroup scope are authenticated. Genuine
negative mutations include wrong owner/WG/dtype, aliased second A and missing
second product. Distinct A loads must feed their corresponding products;
both products reuse the one actual input-B matrix in the same K iteration.
Require original AtomicOr-only fault descriptor and zero arithmetic-loop atomics
or input barriers. All original staged/independent-B negatives remain in the
retained complete modules. New48 layouts are32 macros plus16 packed cohorts.

Add four exact native full/cohort role and forged-name/owner/grid/phase tests
=>73 full runtime, preserving69 and original18 feature-off,8+9 metadata and10
native controls. Add116 actual previous/new independent F64 and whole-owner
regular/cohort cases, plus four real eight-slot previous/new cohort pairs
with original tile8 retained, one original GPU output owner, complete erasure
between programs and existing bounded zeroizing host reference. Total1425 GPU
controls. All data-dependent publication rules and tolerances remain unchanged.

Compose this entire producer/padding/geometry/math/publication/native/profiling
bundle before speed comparison. Do not roll back individual parts from isolated
timing. No compilation, GPU correctness or performance claim at composition.
