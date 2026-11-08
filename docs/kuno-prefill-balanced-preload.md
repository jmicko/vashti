# Two-bank K16 operands for the complete balanced expert bundle

## Completed acceptance (2026-10-24 UTC)

Source `aa38762b5e878bf6ed9be809f74fab9453a10230`, worker
`c0f4692b05ff6e36f1464fc6154747f2f0893a97b90c04a031d23af66a6faba2`:
405 full AppCPU and404 feature-off AppCPU pass on the SAME source, retaining
every preceding control. Actual48 layouts and real SSA/owner/type/publication/
bank/accumulator negatives pass. Runtime18/73, metadata8+9, native10 and offline
compiler populations pass. Fresh all1425 GPU numerical controls execute and
pass with this changed worker; no previous worker's GPU result is reused. Both
original18-script whole-model/recovery chains pass, including actual new counters
and all original capacity/owner/finite/status/reuse/rolling/erasure laws.

Actual213-program ISA export includes22 small macro/cohort programs, all zero
spills/fills. All six changed programs retain realWG128/256/64,SG16 and original
scratch4096/8192/2048B. New and feature-off48-row resource tables have the SAME
SHA256 `51d4dc4306f4ae332538372abd2b2957e6a5f299f853bc3d7f09b38f807c7e44`.
Original two QSA compiler hashes remain unchanged. Actual raw-ID macro32 GU
SPIR-V inspection shows zero-initialized K phi, step32, comparison to original
columns, next-operand offsets+16 and a pre-loop divisibility guard. This is an
additional inspection of the first actual module, not a claim of a separate
all-layout bounds proof; the48-layout SSA checks and real F64 controls remain.

Normal warm/first/repeated:3.401840512/11.106082964/9.320359240s;
diagnostic3.413547338/11.089457881/9.368816478s. Observed repeated change is
1.131% normal /1.695% diagnostic versus preceding9.427012698/9.530366557s.
This small single-pass wall-time difference is not a sustained or variance-
controlled speed claim. Normal/diagnostic load81.955154279/115.051349864s;
startup is not improved or controlled by this experiment. Warm fixture73 tokens,
long requests4245, each4 output tokens; no sustained decode inference.
No CLEAN threshold reached. Best original CLEAN8.394039699s versus Strata
5.3515s and first-request comparison remain unchanged; goal ACTIVE/UNMET.

Same diagnostic sample: macro236 calls15.821815->12.781656ms (19.215% less),
cohort278 calls14.076081->12.257438ms (12.920% less), all matrix708 calls
46.626741->41.788045ms. Ordered-copy608 calls29.168375ms and dequant542
calls19.147060ms essentially unchanged, as are wide/tail matrix families.
Intervals may overlap and this is only one sampled layer; do not sum them into
whole-request cost or infer a global bottleneck from that sample alone.
Both modes report actual preload macro32/64 counts4218/6532 on long requests
and3460 each GU/DOWN preload cohorts, matching genuine balanced counters.

Artifacts under `/media/jmicko/faststore/kuno-development`:
`balanced-preload-{app,off-app}-cpu-v1-20261024`, complete root build receipt,
source seal, `balanced-preload-v1-preflight-safe-20261024.json`, acceptance freeze,
`strata-parity-balanced-preload-v1-20261024-expert-cohorts-public-summary.json`,
`balanced-preload-v1-{profile,normal}-safe-phases-20261024.json`, complete ISA and
actual-geometry exports, complete driver log, terminal journal and independent
final-state JSON. Both AppCPU builds, complete worker build, preflight and GPU/
model driver succeed on their first live attempts. A preparatory script assert
was corrected before launch; an auxiliary disassembler path was absent, and the
system `/usr/bin/spirv-dis` then succeeded into a fresh V2 code-only file. Keep
the empty original file; neither preparation error was an engine/GPU failure.

Protected complete unit `kuno-balanced-preload-v1-complete-20261024` exits0,
17min37.431s CPU,27.7GiB peak,swap0; final state helper exits0. At10:23:24UTC
Intel50688000B,NVIDIA compute0/desktop621MiB,all5 kernel fault classes0,
all3 services HTTP200, admission inactive/unpaused/no recovery/waiting0/completed239.
Independent10:26:43UTC confirms the same state, desktop632MiB. No GPU worker
remains. Original bounded sudo refresher2814185 was identity-verified and stopped;
replacement2936238 is parented by Demons shell3556, bounded09:55..13:55UTC.
Installed services, canonical/Vashti/API/driver/boot remain unchanged.

## Next evidence to collect

Retain this entire selectable variant. It exposes current/next K16 operands
within each K32 iteration; it does NOT carry preloaded operands across the loop
backedge. A true carry pipeline is a separate coupled candidate, analogous to
the retained dense global-register implementation, requiring authentic initial
loads, cross-iteration SSA and final-read proofs. Do not relabel this as that.

Before more small matrix tuning, broaden attribution of the remaining request
time. Current `strata_prefill_profile.rs::NativeSelection::selects` admits only
request3/start0/count4096/layer24, and the emitted sample hardcodes those fields.
It excludes the148-token tail chunk and every other layer. Existing normal
request counters confirm real rolling execution (1045 windows/29262 segments)
and27,345,664,000 copy bytes; static raw storage is TWO roots/64MiB, not the
per-layer accumulated6GiB counter. Existing source already uses retained mapped
host sources, distinct transfer queues and real disjoint-range dependencies.
Do not invent missing pinning, per-expert CPU copies or a6GiB raw allocation.

Prefer an explicitly fixed-public broader timeline diagnostic that preserves
original normal scheduling, all request/owner/metadata bounds and original modes.
Keep instrumented attribution separate from the original CLEAN completion test.
Establish whether the remaining gap is matrix work across other shapes/layers,
unhidden copies, scheduling, or non-expert work before selecting the next bundle.

Read-only rejected lead: the current Vulkan adapter constructor enables
`cooperative_matrix(needed)` but not the separate cooperative-matrix robustness
feature. [Khronos distinguishes that feature](https://docs.vulkan.org/spec/latest/chapters/features.html)
from general buffer robustness. No evidence establishes disabling robustness as
a matrix speed fix. No protections or driver settings changed; exact local
source hash/constructor and primary links are in development
`balanced-preload-robustness-source-check-20261024.json`.

## Original composition contract

Internal candidate on top of accepted full1425 balanced-global. Keep the entire
accepted variant selectable feature-off. `strata-balanced-preload` /
`--balanced-preload` requires `strata-balanced-global` / `--balanced-global`.
No new native ABI, role authority, GPU owners, geometry, resource capacity,
precision, API, configuration or Vashti behavior. No installed service changes.

Both macro32/64 and packed cohort16 call the SAME product selector. The old
single-bank product body is retained. The new path genuinely loads two K16
input-B matrices and four K16 weight-A matrices before consuming the current
bank. It then performs A00*B0+C0, A01*B0+C1, A10*B1+C0, A11*B1+C1 and advances
K32. Each output keeps the original chronological K16 accumulation order.
Actual admitted K=2560/640 are divisible by32; a workgroup-uniform divisibility
guard precedes all operand reads. The final next bank is K=columns-16, never a
speculative bank beyond the descriptor extent. Existing authentic input padding,
all route/header guards, independent8-half publication, checked activation and
pending ordered DOWN consumers remain identical. Tile8 and wide tails unchanged.

Separate profile counters `expert_balanced_preload_macro{32,64}_products` and
`expert_balanced_preload_cohort_{gu,down}_launches` are incremented only on real
new launches. Original balanced/semantic counters remain credited exactly as
before. Public fixed-numeric report carries `balanced_preload_selected`.

Original404 AppCPU names remain selected with the feature both absent/present;
one added bounds/chronology control makes405 with the feature. Existing actual
48-layout compiler proof selects exact one-bank/two-bank populations. New path
requires four distinct weight loads, two distinct input banks each used by its
own pair of products, all six loads before the first product, and actual second
K16 accumulator SSA chains back to the corresponding first-bank result. Existing
type/WG/owner/AtomicOr/publication negatives remain; additional real mutations
alias next B or A to the previous bank and crosswire accumulators. Original
single-bank source/proof branch stays intact feature-off. Native controls remain
18/73, metadata8+9, native10 because authority/ABI/geometry have not changed.

All1425 existing GPU controls must freshly execute with the new worker, retaining
independent F64 and complete-owner/status/fault/sentinel/source/erasure checks,
both roles and genuine packed eight-slot cases. No previous worker's numerical
receipt may stand in for this changed implementation. Both original18-script
whole-model/recovery chains must pass and new counters must equal their actual
balanced counterparts. Inspect the actual compiler's bank dataflow and spills;
do not infer prefetch survival or a speedup from source alone. The actual F64
checks are required to validate arithmetic and final K contributions, beyond
the CPU schedule law. No compilation/numerical/performance claim at composition.
