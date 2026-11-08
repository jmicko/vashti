# Row-ordered complete expert cohorts

Complete variant after carried-operand acceptance; see the measured acceptance
below. Original design and source checkpoints are retained for provenance. No API, installation, Vashti or persistent private-content change.
Retain the whole carry/preload/producer/consumer bundle and the old logical-order
policy as independent selectable variants. Do not roll back the carry component
from its isolated or complete timing; this is a different host-scheduling lead.

## Actual evidence

The accepted96-position fixed-public host profile29a3cfe8 reports9100 small groups
in4021 batches in the4096-input chunk:28.28898% of genuine eight-slot capacity.
The148-input tail reports9854 small groups in1375 batches:89.58182%. These are
real `expert_cohort_groups/batches` and native grammar admission counts, not a
model-size guess. Current logical-expert ordering makes large groups split runs
of compatible small groups; raw-bank units join only adjacent count<32 groups.
Per-layer shape-agnostic lower bounds are1158/1252 batches, respectively. These
are mathematical lower bounds, NOT predicted attainable schedules or speedups.
All data is compiled public fixture only; no private routing history is stored.

Pinned Strata fb58e0 default Intel also walks logical expert IDs. Its opt-in
MMQ resident-only row sorting is NOT the path producing our5.3515s reference.
The reference demonstrates that ordering and grouping are distinct from token
rank; do not attribute an untested sorting optimization to its measured result.

## Complete change to design and validate

Order compatible groups by row geometry BEFORE constructing grouped prefixes,
member map, inverse, raw-bank units and actual cohort parameters. Every logical
expert must retain its original physical cell/source and every token/rank member
must stay in its original within-expert order. Ties must be deterministic. Keep
all original validation, header/extents, readonly source authority, ring lifetime,
private erasure and metadata budgets. Explicitly retain feature-off logical ID
ordering. Preserve original tests for that policy and add independent whole-map
permutation/inverse/weighted-result equivalence controls for the new policy.

The final ordered consumer must still sum rank0..9 with the identical expression
order, including original status/fault checks. Reordering independent expert
work must never become a different reduction. Keep every original GPU owner,
publication, status, pending DOWN contract, real slot and actual bounds. A
source mapper, a packed producer and its consumers must be validated together;
merely changing the numerical batch counter is not an implementation.

Grouping by size may reduce matrix padding and dispatches, but can also change
DMA overlap and load balance. No gain is guaranteed. Use normal whole-model
first/repeated runs and the original CLEAN gate, not a favorable microbenchmark.
If GPU math/vendor/role/kernel definitions stay EXACTLY unchanged, any reused
1425 numerical receipt must be explicitly pinned and labeled historical; new
host permutation, ownership and actual full-model/recovery evidence is still
required. If arithmetic/dispatch authority changes, freshly execute all affected
controls and retain the entire existing suite. Do not silently claim GPU
reexecution from source equality.

Source evidence: strata_prompt_groups.rs::group/validate/ordered_inverse,
strata_prompt_raw_banks.rs::Plan::with_lookahead, and
strata_prompt_expert_cohorts.rs::enqueue_unit. The final consumer needs an explicit
source audit before selecting an ordering. Standalone content-free analysis:
/media/jmicko/faststore/kuno-development/next-cohort-fill-analysis-20261024.json.

## Composed source policy

`strata-cohort-row-order` / `--cohort-row-order` requires the COMPLETE carried
bundle and ordered output. Original `groups::group` remains the logical-order
fixture/reference policy; only the real whole-prefill call uses the explicit
`group_for_prefill`. The latter orders by (member_count,logical_id), placing
unused IDs last, BEFORE deriving every prefix/member/inverse and real raw-bank
plan. Decode and all original1425 numerical fixtures retain their old maps.
The1KiB Zeroizing u16 order stack sorts in place, with no retained history or
heap allocation; conservative original2MiB host reservation remains sufficient.

Source audit: `strata_global_moe.rs` uploads this exact map, then derives/uploads
its inverse, then enqueues experts using the SAME map. Original
`strata_prompt_ordered_shared_reduce.wgsl` reads `inverse[token*10+rank]`, checks
the map roundtrip, and accumulates rank0..9 using its untouched FMAs. No GPU
shader/vendor/authority/owner/geometry change is composed. Both macro and cohort
carry remain selected, not rolled back.

Five additional CPU controls cover complete source-cell/member/inverse mappings,
nonassociative rank-FMA bit equality over full public maps, strict production
call-site and original scratch bounds, packet rejection/ties, and the REAL
raw-bank planner's source slots/consumption/release laws with mixed memory tiers.
All409 original controls remain, planned414 full and409 feature-off on the same
source. Sixteen additional actual-GPU paired controls use the real sorted map,
original rank-reduce/SharedMerge reference and actual ordered consumer, checking
whole owners, immutable sources and every original finite/fault/poison/erasure
case. All original1425 GPU controls remain, planned1441 freshly executed.
Original18/73 runtime,8+9 metadata,10 native and BOTH original18-script full-model/
compiler/recovery chains remain required. Actual row-order layer/group counters
must equal real successful grouping and actual model counts. Compilation/live
acceptance and all speed claims are pending at this source checkpoint.

First CPU source88c53361 compiles, and all four new grouping/equivalence/budget/
rejection controls pass. The real-plan control correctly rejects a strict batch
reduction assumption for its148-row fixture: that fixture already has477 small
groups in60 batches, exactly the lower bound. Keep the assertion; use the SAME
public generator's4096-row mixed fixture (405 small groups in55 original versus
51 sorted batches) to test actual fragmentation. This is a fixture correction,
not an engine rollback or a weakened production law. Original failed log remains.

## Completed414/1441 acceptance and original CLEAN comparison

Source148758aedc634f640f7f46f8846426c87d90898a / worker
19c9456a9273cdce6e614798767a15f7d4e8356668e9d4f8391f1746096ce882:
414 full and409 feature-off AppCPU controls pass on the SAME source; every
original409 remains selected. Actual48 shader layout/SSA/owner controls, fresh
18/73 runtime,8+9 metadata,10 native source controls and offline429 WGSL/320
matrix layouts pass. Fresh all1441 GPU controls pass: ALL original1425 plus16
real row-ordered consumer/reference pairs. Original bounds/F64/whole-owner/
status/fault/poison/erasure checks remain. BOTH original18-script full-model/
compiler/recovery chains pass. Actual213 compiler captures include all22 small
programs with zero spills/fills; original two QSA assembly hashes are unchanged.
No numerical result from the previous worker substitutes for these executions.

| Mode |Warm|4245 first|4245 repeated|Load|
|---|---:|---:|---:|---:|
|Normal,73 warm tokens|3.314356839s|10.554253471s|8.772859676s|80.639954731s|
|Sampled profile,73 warm tokens|3.379658115s|10.604465380s|8.813316241s|77.929683916s|
|Original CLEAN,45 warm tokens|6.560603586s|11.817666824s|8.323282816s|78.707864380s|

All requests generate4 outputs; no sustained decode claim. The normal repeat
is8.66% shorter than complete carry9.604607590s. Original CLEAN uses the exact
unchanged helper with `--public-prefill-check`,45/4245/4245, no profile records,
no compiler/PMU/debug capture or overlapping CPU compilation. It ran ONCE after
the normal trigger; no favorable repeats. Relative to previous best CLEAN
12.471177197/8.394039699s, first/repeat improve5.24%/0.84% in these individual
observations. The0.84% difference is not variance-controlled. Strata's CLEAN
5.5006/5.3515s remains substantially faster: goal ACTIVE/UNMET, no installation
or production-selection claim.

Actual original/new long-request group counts are26673/26673, cohort groups
18954/18954 and copy bytes27345664000/27345664000. Batches fall5396->2410
(55.34% fewer), bulk dispatches88390->65576 and rolling segments29262->22805.
Host-copy segments remain39873 and useful/padded matrix FLOP counters are
UNCHANGED. Do not claim arithmetic or transferred-byte reduction from this win.
Serialized layer24 GPU sample: cohort278->58 calls/15.499310->6.684164ms,
macro17.441285->17.328533ms, all matrix49.718236->40.613840ms, dequant
19.252524->17.712324ms, copy29.176798ms essentially unchanged. Sample intervals
can overlap; never sum/extrapolate them as normal whole-request GPU time.
Normal expert interval5.974026836->5.139562731s; attention2.424829186s unchanged.

The SAME accepted worker additionally passes a complete96-position host-only
trace, with no new numerical-reexecution claim. Original scope/counter laws,
normal rolling, zero GPU timestamp calls and original coarse fences remain.
4096 chunk:9100 groups in1158 batches,98.23% occupancy;148 chunk:9854 in1252,
98.38%. Both reach the per-layer lower bound. All expert memberships, copy bytes,
original padded FLOP counters and48 layers per chunk are preserved. Host-only
first/repeat10.660380392/8.927327496s is diagnostic, NOT CLEAN. Expert4k/tail
3.784223162/1.344191276s; attention2.125877950/0.295115730s. Cross-layer metadata
emission can fall into the next original phase's host time; compare matching
instrumentation and do not call host durations isolated GPU costs.

Acceptance unit exits0/final helper0 at13:12:43UTC; extra trace unit exits0/
final helper0 at13:16:43UTC. Independent13:26:38UTC: Intel42278912B,NVIDIA
compute0/desktop620MiB,all5 faults0,all3HTTP200,admission inactive/unpaused/
no recovery/waiting0/completed251. No benchmark/worker remains. Old bounded
refresher2936238 identity-verified/stopped. New3062861,parent3556 starts
13:23:42UTC and ends17:23:42UTC. Verify identity/cache before another root task.
Canonical/Vashti/API/install/release/driver/boot unchanged; no frontend handoff.

Artifacts under /media/jmicko/faststore/kuno-development:
cohort-row-order-{app-cpu-v2,off-app-cpu-v1}-20261024; complete worker/build
receipt; exact selector/native controls; V1 acceptance freeze/driver/final and
independent state; strata-parity-cohort-row-order-v1-20261024-expert-cohorts-public-summary.json;
cohort-row-order-v1-{normal,profile}-safe-phases-20261024.json; complete ISA export;
cohort-row-order-v1-original-clean-safe-20261024.json;
row-order-full-host-v1-{safe-summary,freeze,final-state}-20261024.json.
All failed V1 CPU fixture artifacts remain. The full build and first live GPU/
whole-model/CLEAN/host-trace attempts pass without validator-schema repair.

## Next source investigation, grounded in current evidence

Do not keep optimizing cohort fill: it has reached the actual lower bound.
Current repeated profile still spends~5.14s in experts and2.42s in attention.
An older accepted full-model profile a761763e has GDN input/output projections
0.371493752/0.152830944s versus current0.553253207/0.247051362s; QSA input/output
0.111000414/0.051762920s versus0.160878545/0.083588306s. These four ordinary dense
projection intervals are~0.358s longer; recurrence and QSA attention/merge are
nearly unchanged. This historical comparison is not a controlled causal claim,
but points to actual non-expert dense shape/layout policy rather than more
expert cache guesses. Audit the current producer AND consumer path against the
retained original dense kernels and pinned active Strata oneMKL dispatch. Keep
all complete variants; no isolated rollback or claim that an opaque oneMKL tile
has been exactly ported.

Separately, current native pipeline-creation counters show23 additional prepares
(0.607678049s host wall) and23 native Vulkan creations(1.091666620s) during the
FIRST long request, and zero new ones on its repeat. The distinct preparation
and native-create scopes are in server.rs::pipeline, backend/base.rs::create_pipeline
(native-prepared metadata) and strata_graph.rs::pipeline (actual Vulkan create).
These are HOST stage costs, may overlap other GPU work, and are not a measured
counterfactual saving or clean45-token-warm breakdown. Investigate genuine
model-load preparation for required static kernel definitions, preserving all
runtime owner/admission/binding guards, caps and privacy. Do not fake inference,
completion or reusable state to warm a cache; never store private pipeline data.
Content-free evidence: cohort-row-order-observed-costs-20261024.json and
ordinary-projection-history-comparison-20261024.json.
