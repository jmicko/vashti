# Next candidate: fill compatible expert cohorts coherently

Design checkpoint after complete carried-operand acceptance; NOT implemented or
speed-tested. No API, installation, Vashti or persistent private-content change.
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
