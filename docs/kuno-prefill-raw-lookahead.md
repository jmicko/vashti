# Eight-group legacy expert upload lookahead

Optional backend-only continuation from integration8b1fd224,2026-10-23. Root
owns feature/CLI registration, integrated builds, selector controls, hardware
and final coordination. No frontend, API, model, context, configuration,
installation or release change; required Vashti changes:none.

## Source difference and complete change

Pinned MIT Stratafb58e0db `sycl/src/prefill/prefill.cpp:3996` stages eight
experts ahead and releases each raw slot after DOWN dequantization(:3975).
Kuno's two-bank plan formerly made every >=32-member group a singleton upload
unit. Only two such units were outstanding even though each existing bank
contains eight actual raw slots. This is a concrete scheduling difference,
not proof that all remaining native completion wait is transfer time.

`strata-raw-lookahead` groups up to FOUR consecutive >=32-member groups into
each legacy upload unit. Two existing banks then stage up to EIGHT such groups
ahead. Original1..31-member cohorts remain homogeneous and at most eight
groups per upload unit. Neither kind crosses a category boundary. Resident
groups retain sentinel8 and perform no HOST copy; streamed groups retain the
original authentic per-role sources and exact real raw-slot descriptors.

Each original legacy group still queues the SAME Gather, GU/DOWN images,
GU/activation/DOWN products and `native_strata_expert_unit_end` after its actual
final DOWN, including a genuine tail. The existing DOWN-image early release
also remains. Only upload-plan bank retirement/refill is delayed until the
last group in its upload unit. This host bookkeeping grants no copy readiness,
GPU completion, source access or early overwrite authority. Original actual
Graph ranges/timelines and all-queue completion remain authoritative.

A sequential `next_group` cursor authenticates every `current` and
`complete_group` operation. Skips, repeats, stale/altered units and premature
whole-unit retirement fail. Legacy units cannot use the cohort `consumed`
shortcut. The upload-unit kind occupies a spare high bit of its original
two-u16 record (all original group boundaries are<=512), requiring no second
routing allocation. The cursor is included in `size_of<Plan>()`, the original
8KiB HOST_BYTES quote and full zeroization. The two genuine32MiB GPU owners,
64MiB total, unchanged slot capacity, original maps and scratch owners remain.
Feature-off production retains the original singleton legacy upload policy.

No WGSL, native compiler, vendor grammar, transient I/O, Graph or cohort
selector source is changed. Whole-bank cohort DQ still declares its complete
32MiB raw owner; no narrower cohort hazard is inferred. Each legacy DQ already
uses the authentic per-slot view. Original kernel arithmetic, finite/fault
policy, source/role authorities, resident views, cache placement and erasure
are unchanged.

## Public metadata and evidence requirements

Actual fixed-public counters add:

- `expert_raw_lookahead_legacy_units`: number of legacy upload units.
- `expert_raw_lookahead_legacy_groups`: all groups in those units.
- `expert_raw_lookahead_batched_units`: legacy upload units containing>1 group.
- `expert_raw_lookahead_batched_groups`: groups in those multi-group units.

These use the existing public-only count sink; ordinary private requests
persist nothing. Counts come only from a fully finished actual plan. Expected
relationships are legacy_groups=original scratch/legacy groups,
legacy_units<=legacy_groups<=4*legacy_units,
batched_units<=legacy_units and2*batched_units<=batched_groups<=4*batched_units.
Original native per-group release counts must remain unchanged; upload-unit
counts and bank assignments may change. Original HOST source bytes and each
of the three physical role-copy populations must remain exact, including mixed
per-role HOST/resident groups; streamed does not mean all three roles require
a copy. Peak outstanding slots are
at most8 for all-legacy runs and at most16 with original small cohorts.

Preserve the three original plan controls, including their exact singleton
oracle when the new feature is enabled. Add five actual Plan CPU controls:

1. Runs of1/4/5/8/9/31/64/512 legacy groups across nine residency patterns:
   exact sequential execution, physical slot partition, per-role copy coverage,
   upload-unit/geometry counts and eight-group bound.
2. An explicit mixed small/legacy partition, unchanged original cohort grouping,
   residency and category boundaries.
3. Out-of-order/repeated/forged calls, illegal cohort shortcut, stale handles,
   sequential cursor and no early bank refill.
4. Original singleton and selected lookahead plans together: unchanged logical
   group-end population and HOST-copy count with reduced upload-unit count.
5. Full reserved metadata/flag/cursor erasure while allocations remain live,
   with the unchanged8KiB cap. Drop invokes this same erase routine; no freed
   storage is read by the control.

The separately root-owned queue control uses the ACTUAL existing native
Planner and this actual Plan, not a replacement dependency algorithm. It must
verify the initial eight real uploads, original three-role spans, original DQ
release/final-group boundaries and refill waits on actual slot readers before
and after compaction. Original whole-bank cohort and resident/no-copy future-
wait counterexamples remain selected unchanged. No runtime Scope is fabricated.

Source implementation is not build, CPU, GPU or speed acceptance. Root must
run the full original selected CPU population plus these controls, original
runtime controls and unchanged original429 WGSL/layout population; all457 GPU
controls; BOTH unchanged full18-script source/owner/compiler/math/erasure/
recovery chains. Preserve strict original routing counts[7663,26673,26673].
Inspect actual new counters, unchanged group/release/copy populations and
physical recovery. Use completed whole-prompt normal/profile durations to
assess overlap; do not extrapolate the layer24 sample or claim sustained
decode throughput from four outputs. No hardware result is claimed here.

## Complete de309ebf validation

Source de309ebfb6db720c4ba5f9c278a205fb41f6701c; frozen worker
0ccb105762abb62ed1b41d95ce75cf7193b689b184bcb5d5e41f4443b2c1746b.
286 selected App CPU controls and15 runtime controls PASS. Original429 WGSL,
320 matrix,16 raw,32 composite,24 padded,48 prior register CFG and2 norm
layouts remain;8 extra genuine HC CFG and16 HC layouts PASS. All469 GPU
controls, including12 full-owner bit-exact HC down/injection controls, PASS.
BOTH unchanged original18-script chains PASS, original route counts
[7663,26673,26673], genuine model/packed sources, all original numerical,
owner, erasure, queue, native-prepared and physical recovery guards retained.

Normal first/repeated4245-token prompt9.797516084/8.355123182s; instrumented
profile9.728968924/8.419709565s. Prior accepted normalization normal
9.872980496/8.468479392s. Small diagnostic gain only; no clean selection,
installation or Strata parity claim (clean reference5.3515s). Normal experts
5.091678941s; inclusive HC768.747164ms. Combined selected down1.249538ms
includes injection; its separate diagnostic fence0.022271ms is not an unchanged
injection-kernel interval. New HC native producer/finite actual ISA have zero
spills/fills; original reference/optimized QSA hashes remain1814b26a/f49c30e1.

Long requests exercise194 fused products per projection,409728 prefix and1940
suffix token visits; together(4245-1)*97 because PromptPlan batches the prefix
and processes its final scalar token separately. Old logical full-call coverage
is gr_wide_tail_products0 + hc_fused_full_down_products97. Lookahead observes
4415 legacy upload units for7719 groups;1936 multi-group units contain5240
groups. Same13291 streamed groups,39873 HOST role copies and27345664000 bytes
per corresponding long request. No role bytes, model cache or arena quote added.

Evidence under /media/jmicko/faststore/kuno-development:
strata-parity-fused-hc-eight-lookahead-20261023{,-operators,-profile,-normal};
full aggregate suffix-expert-cohorts-public-summary.json;
fused-hc-eight-lookahead-root-build-receipt-20261023.json;
fused-hc-lookahead-complete-isa-export-20261023.json.
Initial build count guard wrongly expected48 rather than56 CFG markers; outer
counter guard wrongly included the last scalar token. Failed logs preserved.
Read-only completions verify every prior control/hash and both existing complete
chains without rerunning hardware or selecting favorable timings.
No public/private API or frontend change; required Vashti changes:none.
