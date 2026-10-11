# Fixed-public profile-ranked empty-bank filler

This records the approved separate default-off placement experiment
on `77e421c`, after the target-backlog comparison. No numerical, admission or
throughput acceptance follows from this plan. Original expert arithmetic,
routers, probabilities, absolute IDs, source tensors, full owner proofs,
checkpoint/repair transitions and all existing references remain authoritative.
It changes only setup selection for currently empty VRAM slots.

## Current placement and evidence

Each of48 original DASLab layers retains144 VRAM slots, with72+72 original
packed segments per role. Normal language completes its real prompt prefill,
keeps every actual resident in its existing slot, then fills empty slots with
the lowest absent IDs. The immutable source set maps all512 real expert IDs:
actual residents occupy virtual slots0..143, and the sorted368-ID complement
occupies144..511 in unchanged HOST pages144+144+80. No fill, eviction, recency
commit or mapping mutation is allowed after attachment.

Both v164 normal code runs used56,344 VRAM and46,376 HOST rank-member routes:
54.85% and45.15%, with zero demand expert uploads. Setup filled1,769 of6,912
slots (25.59%) after real prefill. Counts describe routed members, not unique
weight loads, PCIe bytes or isolated kernel time. The profile's effect on these
members has not been measured.

The earlier v146 preserve-bank CPU13 diagnostic kept its cold routed residents
and served all9,120 warm members from VRAM. It did not add new normal-language
placement behavior; normal language already preserves prefill residents. The
historical adaptive trials used a different demand-fill/CPU-mixed path and do
not accept this frozen source policy. Strata normally computes RAM misses on
the CPU while the GPU serves cache hits; its throughput does not predict the
benefit of native GPU reads from imported HOST pages.

v164 Intel admission retained the original536,870,912-byte driver margin and
only24,248,320 bytes beyond it. Original bank owners are18,723,373,056 physical
bytes for14,144,716,800 packed bytes. The HOST complement contains36,147,609,600
logical bytes; full512-expert packed GPU payload alone would be50,292,326,400
bytes. Adding slots or moving every expert to GPU is outside this experiment.
No cap relaxation, smaller owner, arithmetic conversion or allocation fallback
is permitted.

## Public artifact and bounded table

Use only the already pinned Strata source at commit
`fb58e0dbc8399662c0e47c76578c6e878b14f6cf`, original
`data/expert-profile.bin`, SHA256
`8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
It is196,632 bytes: `STRP`, version1,48 layers,512 experts,24,576 ranked pairs,
then the original rank table. All24,576 pairs are unique and every layer has512.
The file carries a ranking, not calibrated frequency or coverage probabilities.
Its order is not adjusted from the native fixture or any future routes.

Strata's top-level MIT license, copyright2026 Niko1221 and the Strata
contributors, covers this pinned source and shipped artifact; no data-specific
license override was found. Preserve the complete MIT copyright and permission
notice beside any copied artifact or derived public table, and record the
artifact/derivation hashes in the source contract. No private routing profile,
trace, prompt or selection is saved.

A compiled immutable table needs only the first144 profile-ranked IDs per
layer:6,912 u16 values,13,824 bytes. With `r` current residents, at most `r` of
that prefix are already resident, leaving at least144-r choices for the empty
slots. Walk the prefix in its original per-layer order and take the first
144-r nonresidents. There is no low-ID, random, oracle or partial-table fallback.
The full pinned artifact remains an independent test fixture; it is not a
release runtime input or environment-selected path. The compact table creates
no GPU binding, MAIN/INFO owner or release heap allocation. Its derived
SHA256 is `7bb7a111617d13f3daeed4a3b406e0f7a34b2a7fb07a1ee3845161a08ab57a5d`.
The compiled source and independent original fixture are under
`apps/kuno/benchmarks/native-text/fixtures/strata-bank-profile`; the executor
exports only fixed public provenance constants, never a profile input path.

Charge the fixed13,824-byte public table explicitly within the existing
192-KiB completion-plan allowance, alongside the original selected vector,
plan and source-write metadata. All existing temporary packed-source windows,
full owning bank/pages, HOST metadata and RAM caps remain unchanged. The table
is model-independent public data; resident selections retain the existing
transient wiping owners.

## Immutable leaf and parent contract

The typed policy is `crate::executor::PublicHostBankFillPolicy` with
`Original` (default) and `StrataRankedFiller`. The existing
`prepare_public_host_sources(executor,cancel,preserve_routed_bank)` remains an
Original wrapper. A new
`prepare_public_host_sources_with_fill_policy(executor,cancel,preserve_routed_bank,policy)`
uses the exact existing completed-boundary/setup/fault/write/commit path.
`StrataRankedFiller` rejects `preserve_routed_bank=true`, so the separate old
preserve policy is never silently ignored.

Selection first authenticates every current resident and original slot, fills
only originally empty slots with the fixed public prefix, validates the full
pending original plan, uses the same source tensors and byte-exact writes,
retains the owning completed-fault checkpoints, and publishes only after all48
layers succeed. A failed selected construction remains terminal. An immutable
private publication witness accompanies the PreparedPlan and actual Source;
full graph/model/bank-map/source identity stays joined through attachment.

The setup witness contains only48 opaque bank identity/revision pairs and the
aggregate completed record. It rejects swapped banks or any later recency
commit before the existing PreparedPlan full bank/generation snapshots are
captured, then is dropped with wiping scalar fields. Its inline size plus the
288-byte per-layer prefix is compile-time bounded by the existing4-KiB misc
allowance. The accepted record is retained by that Plan and actual immutable
FrozenSources. Their existing full snapshots/maps remain authoritative; the
getter rechecks actual Source/graph/epoch and every bank map at the original
completed native boundary. No separate route history, resident selection owner
or general lifecycle mechanism is retained.


`native_frozen_source_bank_fill(executor)` returns
`CompletedPublicHostBankFill` containing `policy`, `completed_layers`,
`filler_experts` and `preserved_residents` (u64 counters). Before a selected source
is attached, a healthy original completed boundary returns Original and zero
counters. Positive selected evidence requires actual48-layer publication and
original PreparedPlan/Source identity, with
`filler_experts + preserved_residents == 48*144` and
`filler_experts == FillStats.experts`. Zero filler experts is valid when all
banks were already full. These counters establish completed setup, not inference
speed or dispatch credit, and contain no IDs, maps or ranks.

CPU13 prepares after accepted original cold13 and ended request reuse, at its
real completed cursor13, BEFORE the original warm reset. It preserves cold
residents, prepares the full source map, then resets/attaches at cursor0; the
next warm pass still performs real token0 prefill and unchanged full reference
checks. It may become all-hit and cannot substitute for the separate canonical
0..143 HOST/VRAM numerical diagnostic. No HOST/VRAM coverage may be fabricated.
Language completes the real accepted prompt before filling and attachment.

Separate canonical selector:
`KUNO_LAB_FROZEN_BANK_FILL=profile-ranked-public`. Controllers and protected
labels remain separately owned. Initial scope is explicit original Intel
speculative/fused full-Q8 MTP with frozen GPU router/partition exit/direct fill,
inline Verify3/chunk/request proofs. Existing GU/DOWN/PLE/status/QSA/GDN/IQ4 and
backlog modes compose under their own original guards. Default and old preserve
behavior remain unchanged; no target-only, English or live mutable cache scope.

## Required source and physical checks

Independent tests must validate the original complete artifact/hash/format/rank
table and every derived prefix, preserve original arbitrary noncontiguous
residents and slots, and test all `r=0..144` cases, all-full/no-write completion,
invalid/doubled/out-of-range ranks, stale publication/mapping identity and both
policy construction orders. Selected writes must be exactly the empty slots;
source types, budgets, absolute512-ID maps and sorted368 HOST complement remain
identical in meaning. Keep the original full arithmetic/owner tests intact.

After source checks and strict direct-device Clippy, physical work must first
pass unchanged full CPU13 vectors/carries/greedy/real repair and full owning
closing/recovery gates, while retaining the canonical HOST-path control. Normal
language comparisons must use the same source/artifact, compiler-idle timing,
original real prompt prefill and aggregate tier counters. No gain, transfer or
30-token/s claim is made before those actual trials.

## Source checkpoint

Leaf and immutable parent source are implemented on the isolated branch based
exactly on77e421c. Controllers and protected labels are separate. Physical
accuracy/admission/recovery and throughput acceptance remain pending. The selected path uses original
`validate_empty_plan`, source windows, packed writes, full fault reads and bank
commits; the original and diagnostic constructors retain their default branch.
Source tests cover the full original artifact/rank table and compact SHA256,
all48 layers with resident counts0..144, malformed prefixes/changed plans,
stale/reordered bank generations, exact selected/default metadata charging,
completed counter joins and selected source scope. Linux fixture hash checks
use the installed SHA256 tool and the fixed test artifact, not runtime input.

CPU source checks completed with the existing
`oracle-native-target-20261009` cache, standalone native-text manifest,
`--no-default-features --features direct-device --bin kuno-text-lab -j2`:
`ranked_`6/6, fixed profile module2/2, `public_host_`43/43, original
`bank_slots::tests`15/15 and frozen Source module20/20. These cover8 new and74
existing distinct checks (four new checks repeat in the public-host suite).
Strict `--all-targets -j2` Clippy with warnings denied and owned-file
format/diff checks pass. These are CPU source checks; no model, GPU, service,
private trace or physical admission trial was used by this leaf.
