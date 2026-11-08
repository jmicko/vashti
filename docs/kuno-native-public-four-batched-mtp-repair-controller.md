# Public four-row lookahead with batched MTP repair

This internal, default-off controller composes the already accepted genuine
four-row target path with the unused fused-four MTP repair leaf
`7ea302cc8a547229a1e8d5f35bf98013972af7b1`. It does not enable an installed
service, change an API, or establish physical correctness or throughput.
The existing [four-row controller](kuno-native-public-four-lookahead-controller.md),
[original batched repair](kuno-native-mtp-batched-repair.md), and
[fused-four leaf](kuno-native-mtp-fused-four-repair.md) remain authoritative.

## Exact registration

Select both `KUNO_LAB_TARGET_LOOKAHEAD=four-public` and
`KUNO_LAB_MTP_REPAIR=batch-public`. This combination requires the existing
compiled-public CPU13 warm replay or code/prose language fixture, Intel MTP,
original full-Q8 MTP with fused sweeps, both integer GU and integer DOWN,
`KUNO_LAB_TARGET_PLE_READS=grouped-public`, immutable original frozen HOST
sources, actual GPU router, partition exit, direct fill, inline dispatch,
fused Verify3, chunk I/O, and request reuse. CPU13 requires `WARM_REPLAY=once`.
The existing physical-memory admission and independent target/draft Vulkan
device checks still apply. There is no fallback.

The combination admits an absent or shared-Q8 head, chronological QSA,
integer-GU stride four, grouped status folds, and the previously scoped target
submission backlog. It rejects Nvidia MTP, target-only, Q2 or another MTP
profile, English vocabulary, broadcast-Q8 head, profile-ranked or preserved
bank placement, three-streaming GDN, IQ4 register decoder, old private or
workgroup tables, shared DOWN scales or GU factors, two-row GU, packed HOST
staging, old three-member arithmetic, original four-row/stride DOWN, prepared
experts, and live member reduction. Unknown/duplicate selectors are rejected
by the existing registration rules.

Absence of either selector preserves that selector's previous scope. In
particular scalar four-row lookahead does not newly require grouped PLE;
batch-only one-to-three-row inference does not newly require integer GU/DOWN
or grouped PLE. The old `public_mtp_batch_repair` parser and its two original
receipt functions remain byte-identical.

## Actual execution

CPU13 cold replay and every initial prompt/first-token prefill remain original.
Their observed MTP repair counter deltas must be zero. The selected CPU13 warm
replay still verifies every unchanged original logit/carry vector and real
accepted-prefix recomputation; no draft token or expert is forced.

After a genuine target round, the unchanged full packet/row identity checks,
greedy acceptance, `round.finish(accepted)` repair, complete accepted vector
checks, and actual target cursor proof run before MTP repair. Only when both
selectors are active **and the actual target round count is four** does the
controller call `repair_accepted_cache_fused_four`. Its authentic third-draft
stamp/current cursor is `target_start+2`; the same real accepted inputs,
preceding carry, and accepted target carries are borrowed by the leaf.
Prefixes one through four append zero through three actual KV cells.

Actual count-one/two/three tails always call the unchanged
`repair_accepted_cache`; their existing max-two appended-cell proof remains.
Only after completed MTP repair and the unchanged corrected draft cursor
check does the controller advance to the final accepted target carry/token.
Both methods use the existing owners, 512-dispatch policy, fault gates,
private caps, and cleanup. There are no new GPU owners or changed arithmetic.

## Completion receipts

The leaf methods share the three `Traffic.batched_repair_*` counters. The
controller therefore captures real counter snapshots immediately before and
after each successful repair. Each observed delta must be exactly one
completed call, one completed fault gate, and the actual accepted-prefix
append count. Separate small ledgers accumulate exact-four and original-tail
completion; no artificial `Traffic` snapshots are constructed.

`mtp_fused_four_repair` reports the selected composition, actual four-row
model state, completed four calls/cells/fault gates, expected count-four
rounds and accepted-prefix cells, aggregate completion across both methods,
and the nested unchanged `original_batched_tails` receipt. Four calls must
equal the actual count-four round count. Four cells must equal the actual
prefix histogram weighted by zero/one/two/three. The max-three bound exists
only in this separate four receipt. Its aggregate must exactly equal the
whole measured draft Traffic delta. Default/cold/prefill fused repair is
false with zero completion credit.

The existing `mtp_batch_repair` field remains the original <=2 receipt: in
composed mode it covers only genuine one-to-three-row tails. If no tails ran,
it is false/zero and `original_batched_tails_requested=true` distinguishes
the requested path from actual completion. Batch-only output retains its
previous receipt. `draft_traffic.batched_repair_*` always reports actual
total completion across both methods, joined to the separate receipts.

The four-lookahead receipt identifies the actual choice as
`fused-four-with-original-batched-tails`; scalar four retains
`original-scalar`. The outer requested mode is
`batch-public-fused-four-with-original-tails`. Initial-prefill receipts
separately require actual zero repair traffic; construction reservation is
never execution credit. Counts and timing remain content-free.

## Validation and limitation

Focused tests cover both selectors' narrow composition and unchanged separate
scopes, all four real-prefix cell counts, original count-one/two/three tails,
zero-tail completion, aggregate and histogram joins, counter regressions,
overflow, false/default credit, and the unchanged <=2 rejection of three cells.
The existing reference, tolerances, owners, and source/round gates are unchanged.
CPU source checks cannot establish actual fused-four GPU correctness or a
throughput improvement. The protected runner and physical tests remain
root-owned and are not activated by this controller commit.

Source checkpoint on the controller branch: 225 distinct CPU checks passed
(49 target-helper, 129 DAS controller/profile, 47 resident-MTP), including
seven new controller checks. The five new helper tests were repeated after
a test-initializer lint correction. Pure `direct-device` all-target Clippy
passed with warnings denied and two compilation workers; tests used two
threads. Scoped formatting and diff checks passed. A direct source comparison
confirmed the original MTP parser and both old receipt function bodies are
byte-identical to controller base `6152919`. Physical composed CPU13,
code/prose throughput, and protected-runner activation remain pending.
