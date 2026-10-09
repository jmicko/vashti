# Native target logits and HC carry for one to three rows

Implemented by `d12b96b`, with fixed-public registration `38c0e52` and
protected launcher `ef2cf9e`. The B70 passes complete CPU logit and actual HC
carry comparison for partitions [1,1,1], [1,2], [2,1], [3] and fresh [3], each
followed by real continuation at position3. All four rows and exact greedy
tokens pass; maximum logit/carry errors are0.000067711/0.000028134. Full closes
erase1,378 owners/826,638,336bytes. See the
[numeric foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).

This acceptance is for the compiled public original-arithmetic target fixture
only. This basic method commits all supplied rows. A separate one-latest-chunk
repair coordinator and the dual-GPU public controller now pass the physical
gates below. Full resident Q8 MTP passes both GPUs. Private production
integration and30tokens/s remain pending.
Whole-model acceptance is B70-only; the full main bank does not fit the3060.
The requirements below describe the implemented target-output contract.

## Output contract

Add an explicit construction-time reservation for a maximum of three target
rows. Add an internal `prefill_chunk_with_carry` result method accepting one to
three fixed public tokens. Preserve the existing `Advance` and `LastLogits`
behavior, their scalar logits allocations and every default setting.

One shared execution transaction should return an internal result enum. The
old method moves its existing logits allocation out; the new method returns a
guarded `NativeTargetRows` owner containing token-major flattened logits and HC
rows, start/count and the actual graph binding identity. Provide read-only row
accessors and position `start + row`; avoid a second head, vocabulary clone,
mutable payload, `Debug` or serialization. Allocate only behind wiping guards,
check exact capacities and wipe payload capacity and scalar metadata on drop.

The carry is the real 10,240-wide residual after layer47 FFN HC combine and
before `output_hc` normalization/mixing: four contiguous 2,560-wide streams.
This matches `Qwen4Exp::step_with_carry` in `crates/kuno-text/src/model.rs`.
Target row p consumes token p, predicts token p+1 and supplies R(p) for MTP
cell p paired with T(p+1), predicting T(p+2). Normal coupled MTP has no extra
zero-hidden/dummy-token cache cell; see the publisher conditioning correction
in `kuno-native-text-mtp-plan.md`. Never substitute the normalized HC owner or the
2,560-wide mixed head input for this carry.

## Smallest source changes

- `resident_model_batch.rs`: add the explicit reservation, two output owners,
  request/count/host/runtime accounting, an `AllRowsWithCarry` purpose and the
  result method. Keep one shared trunk and all-family transaction. After the
  last combine, retain `outer.residual[current]` as the carry source through
  completed owning readback and both proofs.
- `resident_gr_batch.rs`: add a distinct all-row final selection. Existing
  `Kind::Final` still selects only the last source row. The new selection uses
  count rows and input offset0, the same `output_hc` weights, and no injection.
  Consume its prepared capability only after the original head is queued;
  validate the new logits owner against its exact three-row logical capacity.
- `resident_batch_io.rs`, optionally a new focused child module: produce one
  complete padded combined logits/carry/fault packet and map its complete owner
  once inside the original audited readback. Prove all source/packet/fault
  owners are full, distinct, same-client/stream roots. Check exact header,
  count/start, finite active values, zero inactive rows and all owning padding.
  Drop the raw guarded lease before the audit/telemetry/fence-credit boundary.
- `resident_batch_projection.rs` and `executor.rs` need no new arithmetic:
  use the existing original-packed batch projection with
  `BatchSpans(count, 0, 0, 2560, 0, 248320)`. The original head still uses16
  bounded Q6 tiles; count is runtime data, so three rows do not become48 head
  tile dispatches. Retain source identity/cache keys and all bounds/finite
  checks. Do not select Q6 integer-head or single-member policies here.
- Parent-owned registration/fixture code selects only the compiled public
  probe, exact pinned model/profile and its explicitly larger budgets. It
  rejects unrelated inherited experiments and arbitrary input/model settings.

Keep the existing first output proof, PLE preparation, second fault proof,
36 GDN/12 QSA/48 MoE preflights and all-family commits. Hold the guarded result
until these complete. Any failure abandons it and follows the existing terminal
poison/close path. Only then publish the result and advance position by count.
The constructor must reserve both owners before `begin_resident`; close must
include both in the original complete owner erasure/audit.

## Exact proposed reservations

Use fixed maximum-three-row capacity even when a call has one or two active
rows. The combined packet has eight header words, including the accumulated
fault, followed by fixed-capacity logits and HC sections. Header layout and
version must be pinned in the implementation and tests; unused header words
and inactive/padding words are zero.

| Owner | Logical extent | Rounded full owner |
| --- | --- | --- |
| New logits | 3 x248,320 =744,960 f32 words; 2,979,840 B | 4,194,304 B |
| New combined packet | 8 +744,960 +3 x10,240 =775,688 u32 words; 3,102,752 B | 4,194,304 B |
| Existing final residual | MAX8 x10,240 =81,920 f32 words; 327,680 B | Existing524,288 B; no addition |

MAIN grows by **8,388,608 B and two owners**. Keep the old scalar logits and
packet owners for default/legacy behavior. Do not allocate another carry page
or expand/copy public model weights. The base nonhybrid private inventory
818,249,728 B becomes826,638,336 B; independently reserved banks, cache, public
vectors, sources, actual startup baseline and runtime ceiling remain separate.

The worst application output peak is one full4,194,304-byte raw packet lease
plus decoded payload2,979,840 +122,880 =3,102,720 B, totaling **7,297,024 B**.
Reserve an8 MiB output-host category instead of the existing2 MiB category:
**+6,291,456 B**. Its sub-MiB headroom covers checked result metadata/capacities;
no uncharged duplicate payload is permitted. The fixed opt-in application
host cap grows12 ->18 MiB; the default cap remains12 MiB. Existing component
reservation11,206,656 B including original bank metadata becomes17,498,112 B.
With the unchanged262,144-byte runtime-cache metadata reservation the total is
17,760,256 B; retaining hybrid adds1 MiB, producing18,808,832 B and preserving
65,536 B margin under18,874,368 B. A larger runtime-cache metadata ceiling must
retain its separate full reservation/cap extension. A held CPU oracle has its
own bounded public-fixture host reservation, not this category.

The output map and inspection copies change from2 x1 MiB to2 x4 MiB:
**+6,291,456 runtime bytes**. Add the two new startup full-owner clears as
separate INFO work,2 x2,048 =**4,096 B**, even if native clear emits no INFO.
The original16 head tile/validation pairs and one output packet producer retain
their dispatch counts. Total additional runtime reservation is **6,295,552 B**:

| Original head profile | Existing runtime | Proposed all-row runtime |
| --- | --- | --- |
| Q6,16 head tiles | 64,000,000 B | 70,295,552 B |
| Worst supported F32,38 tiles | 64,090,112 B | 70,385,664 B |

These exceed the original64 MiB ceiling. Reserve the explicit protected
**128 MiB runtime ceiling** for this new fixture only; propagate the full chosen
ceiling to construction/admission/runtime configuration, rather than merely
changing the static requirement. Default64 MiB remains unchanged. Preserve
existing rollover/margin and source/target/inspection charges without credit
for earlier fences, omitted scalar output work or unused owners. If an extra
packet validation/producer dispatch is needed, charge its2,048-byte INFO record
explicitly before connecting it. Full close retains its original sequential
owning audit; new4 MiB pages do not increase its existing maximum span.

## Existing batch MoE and configuration constraints

Use original MAX8 chunk policy, public tile8/member8, original F32 activations
and routing,144 slots with actual mixed original-packed72+72 bank segments.
Current generic `batch_checked` can execute three rows: there are at most30
selected rank entries, actual expert unions are pinned, and original uploads
remain split into ten-expert windows with checked40 MiB source precharge and
completed dependency/audit before another window. No selected expert is
substituted. This path routes on CPU from the actual complete router vectors;
it does not establish an all-GPU router claim.

Prefer **no hybrid constructor reservation** for this probe. A reserved hybrid
owner with `cap=None` is nevertheless supported: ordinary packet/batch paths
are selected, and `prepared_shared_overlap` requires `cap.is_some()` before
matching shared `prepare` with `finish_prepared`. Constructor overlap alone
does not trigger the old unprepared-shared contradiction. If retained, keep its
independent163,840 B MAIN,1,245,184 B runtime and1 MiB host charges in full.
Do not credit unused hybrid owners toward the new output budgets.

`cap=Some(0)` and `cap=Some(10)` both reject three-row chunks; zero CPU misses
does not turn the count1 policy into a batch policy. SingleMoe/SingleGdn/
SingleAll/SingleQ4All and native member1 similarly require count1. Do not install
them. The new resident GPU router authorization is also count1-only and must
remain disabled. No adaptive/secondary placement, integer/subgroup expert
experiment or Q6 integer head belongs in the first original-arithmetic probe.

## Focused acceptance and ownership split

One native-owner leaf can own `resident_model_batch.rs`,
`resident_gr_batch.rs` and the new owning-output I/O module. Parent owns
registration/configuration and protected physical runs. A separate reference
leaf can use the unchanged CPU `step_with_carry` to emit an exact fixed-public
three-row oracle; no core model change or second CPU head is necessary.

Host tests should cover counts1/2/3; rejection of0/4 and unreserved output mode;
last-row behavior unchanged; all-row source order/normalization tap; exact
head input/output strides and capacities; short/foreign/aliased owners; header
count/start/fault/nonfinite/inactive/padding rejection; lease destruction before
audit; and result destruction on either proof/late family failure. Check the
literal MAIN/host/runtime arithmetic and default budgets separately.

An independent small-math projector/HC fixture should distinguish every row
and stream, proving pre-normalization carries and one head per active row.
Use an original-packed Q6 projection oracle with three distinct fixed public
inputs to verify complete vocabulary-row and tile-boundary placement. Existing
source/projection tests are prerequisites, not a new full-model acceptance.

Protected whole-model acceptance uses fixed token0 and the next two tokens
from the existing exact CPU public chain, within its accepted prefix. Compare
all248,320 logits and all10,240 HC values at each position against sequential
CPU `step_with_carry`, with explicit unchanged logits tolerance/exact greedies
and an explicit carry tolerance. Check result identities against their own
graph instances, since fresh CPU/GPU instances have distinct binding IDs.
Compare partitions[1,1,1], [1,2], [2,1], [3], fresh replay and continuation at
position3. Verify full owners, both proofs, cleanup, failure retirement and
physical recovery on both cards. This is coupled target-output correctness;
it is neither independent full MTP-head parity nor a speculative speedup.

## Accepted-prefix seam: implemented and physically checked

The original method still commits **all supplied rows**. `LatestTargetRound`
adds an explicit construction-reserved transaction: one immutable tokenpacket,
one exclusive target/executor borrow, original full-row execution, then drop
its output lease before selecting the accepted input prefix. Allaccept consumes
without another forward. Rejection restores actual GDN recurrent AND rawconv
roots, clears exact QSA appended KV/indexer/positions under the original sticky
fault/audit proof, restores real PLE convolution/history, then reruns onlythe
accepted prefix. No generation, poison or ledger resets are permitted.

`87049e4`/`f3899b6` implement the families/coordinator. The completed QSA
temporaryfault association is correctlyNone after originalacceptance; narrow
fix `54c846b` retains the checkpoint's exact captured physical fault owner
while rejecting any live candidate association. OneMiBhost and49,152runtime
INFO bytes are admitted; noadditionalGPUstate is allocated. The old216MiB
per-prefix GDN snapshot estimate is superseded by this one-latest-root restore
and original-prefix recompute approach.

Default artifact `4448591135f69e99576d285ebdc2993b2c11255c9039d99b70a5cdd22b9091c0`
passes actual wrong-future-token cases accepting1/2/3inputs, all completeCPU
accepted-prefix vectors/greedies and furthercorrectcontinuation toposition4.
Full1,378-owner/826,638,336-byte erasure passes. The initial QSA pointer
mismatch run is preserved; physicalmemory/faults/services recover in bothruns.

DualGPUcontroller `66c1078` uses B70target and3060nativeQ8MTP, correct
T(c+1)/R(c) pairing, actualGPU second draft, greedy prefixchecking, real
repair and target-conditioned draftKV rebuilding. The first compiler-idle
13-row run passes all12 accepted output vectors and closes. Sixof10drafts
accept;12outputs/10.595s=1.1326tokens/s undercold normal demand routing,
including2,311expertfills/4.777GB, draft/verification/repair andpublicchecks.
This isnot a speedgain and isnot directlycomparable with the earlier
12.728tokens/s conditionalideal preparedcount1 result. Packedthree-column
dense and preparedexpert candidates remain inprogress. See numericreceipt
for artifact/source hashes and per-round phase clocks.

See `kuno-native-text-mtp-plan.md`, `kuno-native-text-model-batching.md` and
`kuno-native-resident-router.md` for the current causal/owning contracts.

## Fixed Verify3 dense arithmetic candidate

Internal default-off constructor choice `target_verify3_dense` is admitted only
with target-row reservation, original shared/fast dense settings and MAX8/tile8.
Protected target4/repair4/dual-GPU13 registrations accept only canonical
`KUNO_LAB_TARGET_DENSE=verify3`; choices are frozen before any model mutation.
Parent checks actual16/32 subgroups/128 lanes before device creation. Packed
original vec4 weights are reused across1–3 original F32 rows with unchanged
full output finite/fault/padding gates, original owners, INFO/dispatch counts and
head tiles. Whole CPU vectors/greedy/actual repair pass physically at v121;
cold controller comparison does not establish a speed gain. Keep default off.

Canonical optional `KUNO_LAB_TARGET_IO=chunk` compares the already implemented
complete-chunk runtime-I/O cadence. It does not change the128MiB hard ceiling,
whole-model final proof, owner erasure or retirement. Default remains per-layer;
`dea5d48` source and v122 full CPU/repair/owner physical checks pass. Its
1.1404tokens/s cold controller result establishes no speed gain. Neither lab
option changes a public API or Vashti consumer contract.

## Construction-fixed public warm replay and finite fusion

`4b93c99`/`bafd9c0` add one healthy fixed-DAS13 replay, reserved at construction
and consumed only at successful completed13 with no live outputs/checkpoints.
Original private/source full-page erasure and fault/identity fences precede
infallible numerical GDN/QSA/PLE cursor/history0 commits. Public weights/banks,
actual ownership/admission/epochs/ledger/counters remain; poison and retirement
are never cleared. Sanitization clears transient binding/INFO caches. Original
token0 prefill must re-prove start0/count1/cursor1. The existing completed
projection helper still rejects0. Canonical protected selector is
`KUNO_LAB_WARM_REPLAY=once` for speculative13 only; both complete CPU passes,
actual repairs and reset/final full-owner audits pass physically in v125.

`fe196fa`/`6f581b4` add default-off `verify3-fused` under the same target policy.
It retains original F32 weight/reduction arithmetic and full owner/input/source/
fault/status/padding proofs. The final weight tile's bounded4096-group strided
scan covers the complete logical output complement, excluding ALL active
complete matrix intervals; active writers check their own finite results.48-byte
INFO pre-device admission uses the same2048-byte rounded allowance. The immutable
choice and actual fused dispatches are evidenced in each numeric receipt.
Target4 partitions and real accepted-prefix1/2/3 continuation gates pass v126.
The dedicated both-card operator gate is pending corrected inventory acceptance
at `5eb8e82`; no finite-fusion speed selection or30token/s acceptance yet.
These are isolated public lab contracts. Production/Vashti API is unchanged.


## Fixed-public target request I/O reuse (implemented and public-fixture accepted)

Only the three public DASLab target fixture registrations may select
`KUNO_LAB_TARGET_REUSE=request`, with exact `KUNO_LAB_DEVICE_DISPATCH=inline`
and `KUNO_LAB_TARGET_IO=chunk`. The pure direct-device build is required;
default and arithmetic-feature combinations, inherited generic cache flags and
other fixtures reject. Existing absent selectors preserve their behavior.

Configure only the B70 target stream through a dedicated programmatic runtime
constructor before model admission:128MiB original I/O ceiling,4096-byte INFO
slabs, batched INFO, inline readback,4096 FIFO bindings and fully charged4MiB
request metadata. No process environment mutation, clean-readback pool or
GPU-upload-clear selector. Nvidia MTP retains64MiB I/O/16MiB HOST/default
configuration. The original generic cap10 request scope remains separate and
rejects fixed-target activation.

The native Config freezes and charges the separate metadata before allocation.
A scope may start only after completed real public token0 prefill at cursor1,
with actual context4 or13, no live output and the approved original native
DAS144-bank MAX8 graph. End only after actual full context completion, perform
the original full fault/I/O ownership audit, and close the scope before warm
reset/final owning close. At most two scopes for the approved once-warm replay;
new prefill/cursor1 must be proved again. Failures remain terminal; no ledger/
epoch/poison reset or cancellation cleanup bypass.

Target4 request-mode tests use the real token0 prelude then the original five
partitions over remaining three rows. Repair4 starts at actual row1/count3 and
checks actual accepted prefix1/2/3 plus original continuations. Controller13
checks the same complete CPU logits/carries and actual target/draft repairs.
Request begin and full closing I/O audit are included in the12-accepted-token
throughput interval. Numerical receipts record actual active/closed proofs,
reserved capacities and checked process-global content-free binding/INFO
counter deltas; no runtime nonce/key or private content is saved. Physical
target4, actual prefix repair and fullCPU13 cold/warm controller acceptance now
passes at source `12f9cc7` (v129). Warm original-F32 throughput is6.63953 accepted
tokens/s, with zero expert uploads/weight reads and4096 process disk bytes.
This ordered public fixture is not a matchedABBA, private-request or30token/s
acceptance. See the exact receipt and checkpoint.

## Original-F32 live expert-member reductions (source accepted, GPU pending)

Kernel `7ab49b8` and immutable graph integration `13d9929` are default-off.
Only the three fixed-public target registrations accept canonical
`KUNO_LAB_TARGET_MOE=live`. It requires original F32 Generic expert arithmetic,
DAS144-bank MAX8 target-row geometry, and native chunk execution. Prepared
integer experts, CPU/hybrid placement and count1-specific execution selectors
cannot compose with this choice. Absent selection preserves the original path.
The constructor freezes the flag before admission/allocation; all48 layer
geometries must agree. Completed-boundary getters check actual healthy/idle
geometry, rather than echoing an environment value.

For each active expert group the gate/up/down kernels retain the original
packed weights,64-lane reduction tree, operation order, both segment launches,
INFO layout, status/fault/padding proofs, rank fold and owners. Only unused
member-plane shared stores/reduction additions are skipped. Every workgroup
barrier remains unconditional. No GPU owner, capacity, private lifetime or
erasure gate changes. This does not eliminate packed decoding or barriers.

The dedicated default-build `iq-live-members-moe-probe` requires exact
`KUNO_LAB_IQ_LIVE_MEMBERS_PUBLIC_ONLY=1` and a fresh public receipt; inherited
experiments and extra arguments reject before device access. It compares old
and new full ACT/PART/rank merge bits against independent CPU results for all
nine DAS mixed profiles, counts1/2/3 and mixed/single-segment maps. Malformed
maps, nonfinite inputs, inactive planes, full gaps/tails/status and owning closes
are checked under the original2MiB private/64MiB I/O/32MiB HOST/1GiB MAIN bounds.
Source tests and shader validation pass; physical both-card operator acceptance,
target4/repair4 and same-artifact fullCPU13 control/live timing remain pending.

The controller also records checked, nonsaturated process-global scoped-read
setup/flush/map and inline readback batch deltas over the public dual-device
timed interval. These are attribution diagnostics, not isolated GPU kernel
timings or per-stream counters. No prompts, generated content or private IDs
are persisted. Production API, Vashti and installed services remain unchanged.
