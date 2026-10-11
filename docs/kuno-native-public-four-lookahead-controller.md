# Fixed-public four-row target lookahead controller

Status: approved source contract; source and physical acceptance are separate.
This is an internal Intel benchmark selector, not a normal Kuno/API policy.

## Exact selection and unchanged control

`KUNO_LAB_TARGET_LOOKAHEAD=four-public` selects at most three **actual** original
MTP proposals and four actual target inputs. Absence retains the original maximum
of two proposals and three target inputs. Unknown values reject before model
allocation. Selection requires the compiled original CPU13 warm fixture or the
compiled code/prose200 fixture, Intel target and independent Intel full-Q8 MTP,
fused MTP sweeps, frozen original HOST sources, real GPU routing, partition exit,
direct fill, inline dispatch, fused Verify3, chunk I/O and request reuse. Both
integer GU and integer DOWN reservations/owners are mandatory in this initial
four-member source capability. The original 144-slot bank geometry, fixed 8 GiB static cache and budgets remain.

The first scope rejects target-only, Nvidia MTP, Q2, English shortlist, head
broadcast, packed-register IQ4, profile-ranked bank filler, legacy preserve bank,
private/workgroup tables, shared factors/scales, two-row GU, staging,
three-member arithmetic, prepared experts and live-member reduction. The
separate `three-streaming-public` GDN selector and `batch-public` MTP repair also
reject; selected four rows use distinct FourGdn/FourStreaming and the unchanged
individually fenced scalar MTP repair. Integer GU (optionally its admitted
stride-four schedule) and integer DOWN, chronological QSA, grouped PLE and
status folds may compose only through the parent and source leaf's genuine four-member capability.

Construction reserves four-row capability with
`Config::reserve_public_four_rows()` and `GpuExecutor::reserve_public_four_rows()`.
This does not activate four-row execution. Cold CPU13 and real language prompt
prefill still use the original count1..3 path. After accepted prefill and actual
source attachment, `ResidentModel::set_public_four_rows(&mut executor, cancel)`
activates the selected transaction before the request starts. Source capability
is prepared before attachment inside the existing
`attach_public_host_sources_with_down_math` parent wrapper. No controller-side
source permission is fabricated. Existing original count1..3 arithmetic remains the tail/repair implementation.

## Real proposals, verification and repair

At target start `p`, MTP initially ends at `p-1`. The original `step_draft` yields
a real first proposal from cell `p-1`; one original `follow_draft` yields a second
from cell `p`; a second original `follow_draft` yields a third from cell `p+1`,
with completed cursor `p+2`. Every output's binding/position and existing source,
finite and LastForward proof remain mandatory. No proposal, carry or router
result is substituted. The actual target packet is `[confirmed,d1,d2,d3]`.

The earliest true target mismatch defines the accepted prefix. Language EOS
clips this prefix before restoration. All original accepted CPU13 rows retain
the complete unchanged vector/greedy oracle; language rejection repair compares
all saved accepted logits/carries and greedy results with recomputed outputs.
Four-row rejection uses genuine original count1..3 recomputation.

MTP repair retains the original first real KV cell via `rewind(p)`, then calls
original `append_kv` once for each accepted row after the first, using the actual
preceding accepted target carry. Each scalar call has its original fault gate
and resets its own original 512-record ledger. Maximum three appends do not form
one joined dispatch packet. Count1 tails keep their original confirmed append.
The final target/MTP cursors remain `p+a`/`p+a-1`.

## Content-free actual receipts

The selected actual getter is `completed_public_four_rows(executor)` and returns
mode plus completed real count4 chunks. Selected mode requires positive joined
count4 and actual Verify4 dense/fused calls; cold/prefill/default must be false
and zero. The controller joins completed chunks exactly to real count4 rounds,
without crediting initial prefill or count1..3 repair.

Receipts record only scalar count1..4 round and accepted-prefix histograms,
actual third proposals, cases where the preceding two proposals were accepted,
and actual accepted third proposals. Conditional acceptance is the latter two
counts' ratio when its denominator is nonzero; no independent-probability model
is claimed. Aggregate draft, third-followup, full verification and repair times
come from their executed phases. No tokens, expert IDs, routes, prompts, vectors
or carries are added to persisted receipts.

CPU13 has twelve continuations after its original first token. Perfect four-row
acceptance gives three count4 rounds and therefore does not itself exercise every
tail or rejected prefix. Preserve the separate original count3 CPU13 control and
report natural count/prefix coverage. The pinned two-draft MTP CPU reference and
CPU13 header/three-chunk fixture are unchanged; the selected controller uses the
existing per-row full target oracle. No independent third-draft CPU oracle or
whole-model throughput result is claimed by source tests.

## Fixed ownership and validation

The parent must prove four-row packet raw plus decoded capacities and typed
headers remain inside the original 8 MiB host envelope and original two 4 MiB
MAIN output/logits owners. Distinct count4 dense/router/MoE/GR/GDN/QSA/PLE/round
paths must publish all original full-family and full-owner checks. No allocation,
margin, reference, tolerance, default selector or public API is relaxed here.

Required gates: focused canonical scope/prefix/EOS/cursor/statistics and exact
actual receipt tests; strict direct-device all-target Clippy and formatting;
original CPU13 cold/warm controls followed by selected full-vector count4 gate;
actual rejected-prefix repairs when naturally reached; then compiler-idle paired
code/prose timings under the same artifact with only this selector differing.
Source implementation and physical timing status will be recorded separately.

## Controller source acceptance (2026-10-11)

All 143 distinct controller/helper checks pass: 45 speculative registration
tests, 54 language registration/statistics tests and 44 shared receipt/parser
tests. These include ten new four-row tests and 133 existing regressions.
The new checks cover canonical/default-off selection, required integer GU/DOWN
and actual Intel/full-Q8 scope, conflicting policies, genuine third-cell
position/binding, earliest mismatch and EOS clipping, original count1..3
statistics, and exact positive/default/prefill dispatch and conditional
acceptance receipts. The original CPU13 references and old tests are unchanged.

Strict pure `direct-device` all-target Clippy with warnings denied and `-j2`,
scoped Rust formatting and `git diff --check` pass against the final four-row
parent and leaf dependencies. No GPU, installed service or whole-model timing
was run by this controller owner. Original cold/warm controls, selected CPU13
full-vector/real-repair acceptance and compiler-idle paired code/prose timings
remain the integrating owner's physical gates.

### Reserved output capacity correction

The first selected physical CPU13 attempt stopped during the original cold
count1 prefill, before any draft or count4 transaction. Reserving four rows
creates an immutable logical `4*vocabulary` target logits owner; the final GR
head still requested the old `3*vocabulary` full-owner proof. Both logical
layouts occupy the same existing 4 MiB physical allocation, so proving only
physical size would miss this mismatch.

The parent now supplies the actual target-output constructor reservation flag
to the GR full-owner proof. The original head wrapper retains its old capacity;
selected count1..3 prefill/tails keep their original mixer/projection spans while
proving the whole four-row logical owner. Genuine count4 requires the reserved
four-row owner. No allocation, arithmetic, reference or tolerance changes.

The correction passes 18 distinct source checks: ten GR checks (three new
capacity regressions and seven original controls), seven original/selected
target-output checks and the whole-model four-mode/prefill/tail completion
check. Strict pure direct-device all-target Clippy with warnings denied and
`-j2`, scoped formatting and diff checks also pass. A physical rerun is still
required; source acceptance does not establish cold-prefill or selected warm
execution on the card.
