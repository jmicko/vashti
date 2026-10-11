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


## Bounded chronological attention selector, 2026-10-09

Internal source `cc4b8aa` and public controller connection `597039d` add
`KUNO_LAB_TARGET_QSA=chronological-public` to the fixed speculative13 and
language200 labs only. Absence preserves the guarded CPU selector. The selector
is constructor-fixed: actual target rows, tile8, no CPU hybrid or resident
router. Duplicate, malformed and unrelated mode selection rejects before device
access. This is not a production setting or public inference API.

For a chunk whose final visible position is at most2048 tokens, all original
four-token blocks fit the QSA budget. The exact retained IDs are chronological;
original indexer/score equations still execute. Two ordered GPU dispatches
validate complete SCORE/status/SELECT/global-Fault owners and then publish IDs
only if the actual sticky global fault remains clear. Finite nonnegative active
scores, status1, inactive/logical-gap/full-owner zero tails, initial empty SELECT
and global-Fault tail are checked. No expert or target token is forced.
Chunks above the conservative2048-token boundary retain the exact original CPU
read/select/upload path, including its score-dependent selection.

There are no new private MAIN owners. Additional full INFO ownership is24×2048
=49,152bytes per MAX8 chunk; temporary handle/resource metadata has a separate
4,096byte host reservation. Both actual device limits and physical full-span/
client/alias proofs precede launches. State acceptance remains the existing
original all-family completed fault/finite/tail checks and actual prefix repair.
The standalone `qsa-chronological-probe` requires exactly its public-only gate
and a new numeric receipt, with swap/core disabled; default or pure direct-device
builds only. No private input/model path is admitted.

Frozen v137/`597039d` passes both physical cards'21 independent numerical cases,
25 no-publication faults and full original owner erasure. Full CPU13 target
vectors/exact greedies/actual MTP repair and cold/warm closes pass on B70 with the
3060 draft. Short all-hit warm replay is7.26311 tokens/s with51 readback batches,
versus prior6.83366/219 in an earlier artifact; this is an ordered diagnostic,
not a matched performance selection or normal-demand30tokens/s claim.
Same-artifact ordered language200 candidate/control both pass real target greedy,
actual original prefix repairs and full owning closes: candidate199/95.5603s=
2.08246tokens/s, control199/98.6940s=2.01633. Both spend about49s on weight reads
and physically read35.2/34.95GB; map batches4163/5939. This modest ordered result
is not matchedABBA, general quality acceptance or a30tokens/s selection. Physical
memory/admission/all threeHTTP200/fault counts0 recover after every trial.

## Authenticated immutable RAM expert sources, 2026-10-09

Fixed public source construction reserves separate immutable source metadata
(+589,824runtime INFO and2,105,344host bytes per chunk) before model admission.
The completed144-slot VRAM banks and their exact model/instance/generation
identity are frozen; sorted complementary original368experts/layer are
materialized into HOST_VISIBLE|COHERENT|CACHED, non-DEVICE_LOCAL storage-only
pages. Five address partitions have capacities72/72/144/144/80. No routing,
expert substitution or bank-generation mutation is allowed after attachment.
The source plan's65,536alignment is conservative budgeting. Typed imported pages
must carry the actual native allocation alignment and extent, already checked
against the logical and full physical envelope. The importer must not reinterpret
budget rounding as the driver's allocation divisor.

Full original CPU13 acceptance passes v143/`f0e088b`, with both actual tiers,
complete target vectors, exact greedies and real repair. All432registered pages
subsequently detach and CPU-zero/free36,147,609,600actual backing bytes. Private
owners retain their original all-family erasure gates. Short deterministic warm
placement gives12continuations/2.845782894s=4.21677tokens/s,0explicit expert
uploads,1,081,344process disk bytes; not ordinary language performance or30.

Optional `KUNO_LAB_TARGET_ROUTER=device-frozen-public` has both-card80-case
operator acceptance, but no full-model acceptance yet. It requires the immutable
source scope, uses original top-k and selected softmax, and defers exact original
CPU-score ID/order and full-map audits until before all-family/source commit.
Its separate complete owners add2,228,224MAIN/9,175,040runtime/5,242,880host bytes.
Original probability bits are not claimed; original vector tolerances remain.
There is no ordinary CPU fallback after source identity/activation failure.

The candidate `KUNO_LAB_FROZEN_PARTITION_EXIT=1` is default-off and requires the
same canonical original frozen public source scope. An immutable source geometry
flag selects a separate inactive-partition kernel entry point; the ordinary
factory/default remain unchanged. Receipts must compare the actual completed
source flag with the requested mode. Inactive groups terminate uniformly before
scratch/reductions; full scratch reset and complete finite/status/fold checks,
active arithmetic, original IDs and all-family publication remain mandatory.
Construction owners, grids and budgets stay the same. Compiler/source checks do
not establish full-model or speed acceptance; see the current checkpoint before
selecting this mode for physical validation. These are laboratory contracts,
not production API/configuration/Vashti changes.

### Operative public lookup-table lowering experiment, 2026-10-10

An explicit laboratory candidate `KUNO_LAB_FROZEN_TABLES=private-public` requires
`KUNO_LAB_TARGET_SOURCES=frozen-host-public` and
`KUNO_LAB_FROZEN_PARTITION_EXIT=1`. It applies only to the fixed public CPU13
and language200 controllers, with unchanged original models/references and Q8
draft comparison. Absent selection preserves the existing source/compilation.

The selected immutable source geometry dispatches a distinct gate/up entrypoint
whose public quantization tables lower as module `var<private>` rather than
by-value WGSL `const` arrays. This targets Naga's repeated whole-array Function
spill on dynamic lookup. The term private here is a shader address space for
the fixed **public codebook**, not retained user content. The guarded compiler
selection must use the exact kernel entrypoint, not ambient environment values,
and admit only the expected original public u32 table geometries/literals.

Packed weights, table entries, lane/column accumulation order, F32 factors,
64-lane reductions, inactive-partition semantics, down kernels, runtime maps,
owner reservations, binding counts, source identity and erasure remain unchanged.
Receipts compare the completed actual source's `frozen_host_private_tables` flag
with the requested mode. Source shader/SPIR-V checks are prerequisites, not GPU
acceptance or a performance claim. Require original CPU13/repair/full-owner gates
before ordinary same-artifact language timing. If GPU compilation or execution
fails, retain the old path and failure evidence; do not weaken numerical guards.

### Pending cooperative kernel leaves, 2026-10-10

Two source-only candidates are committed in isolated worktrees based on
`7364dc8`. Neither is integrated into the parent parser, protected launcher or
receipts, and neither has GPU acceptance. Do not assume an environment selector
exists or silently substitute either for `private-public`.

- Workgroup-table leaf `3594b76`: immutable `with_workgroup_tables` policy
  requires frozen sources and partition exit and excludes private tables. The
  exact named GU entrypoint deduplicates identical public grids and performs
  literal lane-index/value initialization after the uniform inactive exit,
  then one barrier. It uses5/6/12KiB total SLM for IQ3_XXS/IQ3_S/IQ2_S, with
  unchanged original arithmetic and no new MAIN/INFO/binding owners. Three
  focused mixed-profile checked/unchecked shader/SPIR-V/negative-store tests
  and strict direct-device Clippy pass. Root selection remains pending.
- Down-scale leaf `b56af6a`: immutable `with_shared_down_scales` policy requires
  frozen sources and partition exit. The named frozen-down entrypoint stages
  the20 original IQ4_NL scales in80B SLM after the uniform inactive exit, then
  uses the original scale/code multiplication, column/member order and64-lane
  reduction. Q2 and default kernels remain unchanged. Three focused
  checked/unchecked shader/SPIR-V/C-fixture bit/dot tests and strict Clippy pass.
  Root selection and composition with the workgroup-table leaf remain pending.

After explicit GPU-work resumption, integrate leaves and guarded selection
separately, resolve their shared Geometry/Sources policy changes, rerun CPU
proofs, freeze a new artifact, then require complete original CPU13 vectors,
carries, greedies, actual repair and full owner/physical recovery gates before
normal-generation comparisons. Neither source acceptance nor an isolated
operator establishes whole-model speed. See the current checkpoint for exact
worktrees, commits and final host state. No production/API/Vashti change.

### Composed cooperative policies, 2026-10-11

Root source integrates GU workgroup leaf `4f23abd` (original `3594b76`) and
IQ4 down-scale leaf `ce78d39` (original `b56af6a`). Explicit laboratory selection
now uses `KUNO_LAB_FROZEN_TABLES=workgroup-public` for the GU codebooks and
`KUNO_LAB_FROZEN_DOWN_SCALES=shared-public` for the down scales. Both require
frozen original public host sources and inactive-partition exit, plus the
existing fixed public/request/direct-fill guards. Absence preserves the control.
The single table selector excludes private/workgroup coexistence; shared down
scales compose independently with either table mode or the exit-only control.
Invalid spellings, duplicates and missing prerequisites reject before attachment.

Completed actual attachments must match requested `frozen_host_workgroup_tables`
and `frozen_host_shared_down_scales` in numeric receipts; the speculative cold
replay must report false for both. Runtime dispatch explicitly selects the GU
and down entrypoints independently and has an all48-profile composition test.
Original models, routes, arithmetic/reduction, memory/source budgets, owners,
references and tolerances remain unchanged. The launcher retains the same
48GiB CPU13 and48GiB+196MiB language limits and no-swap/core protection.

Source tests and shader proofs do not establish GPU acceptance or throughput.
Require fresh original CPU13 full-logit/carry/greedy/repair and complete closing
proofs before same-artifact language200 control/candidate timing. Preserve failed
trial evidence and verify actual memory recovery. No public API/production or
Vashti change is part of this internal experiment.

### Frozen public factor and Q8 draft-head candidates, 2026-10-11

These independent laboratory policies remain default-off and do not change
installed Kuno, model names or public APIs. The original target model, references,
source hashes, route membership, probabilities, floating-point operation order,
repair checks and owning closure remain authoritative.

`KUNO_LAB_FROZEN_GU_FACTORS=shared-public` selects cooperative IQ2_S/IQ3_XXS/IQ3_S
block-factor staging only for authenticated frozen public sources with explicit
partition exit. It composes with private/workgroup table and shared down-scale
policies; private/workgroup table modes still exclude one another. IQ4_XS is
unchanged. Parent registration rejects wrong spelling, duplicate keys and missing
source/request/exit prerequisites. Runtime dispatch uses the actual geometry
policy, and completed attachment receipt `frozen_host_shared_gate_up_factors`
must match selection. Cold CPU13 replay retains original target dispatch; only
warm replay uses frozen expert policies.

`KUNO_LAB_MTP_Q8_HEAD=broadcast-public` selects the original Q8 output.weight
head with subgroup-shared half-to-F32 scales. It is valid only in the explicit
public speculative/language controllers with the self-contained Q8 draft profile.
The selector must reject alternate Q2 profiles. Selection is fixed before the
first draft/KV operation, and actual completed full-forward receipt
`mtp_q8_head_broadcast` is captured before owner close. No other Q8 projection
or target quantization changes; original accumulation/reduction and fault
coverage remain. Both CPU13 replays use the selected draft head, independently
of target frozen policy replay.

Each candidate needs CPU/shader source checks, full original CPU13 physical
acceptance, same-artifact normal language timing and recovered GPU/service state.
Source-only operation counts are not latency or performance evidence. Reduced
per-lane conversion count may still cost more broadcast/barrier work. The prior
workgroup-table plus shared-down-scale candidate passed CPU13 but regressed the
single normal ordered code200 pair (7.73410 to6.90538tokens/s); do not turn it on
implicitly when evaluating these candidates. Neither30tokens/s nor production
acceptance is established here. Required Vashti changes:none.

### Frozen public row packing and cooperative draft-head scales, 2026-10-11

`KUNO_LAB_FROZEN_DOWN_ROWS=four-public` is an independent default-off frozen
public/partition-exit policy. Four rows share a64-thread workgroup,16physical
lanes per row, each retaining four original virtual64-lane accumulators.
Register folding of stride32 then16 and the remaining16-lane tree preserve
original operation order for IQ4_NL and Q2_0, all640columns and8RHS cells. Original
input, partial/fault owners and finite/coverage checks are unchanged. Geometry's
down grid changes2560 to640 atomically with the selected actual down entrypoint;
completed receipt `frozen_host_four_row_down` must match the requested policy.
This composes with existing table/factor/down-scale choices. Gate grid and
member capacity are unchanged. Scratch is2048B plain and2368B shared-scale,
and accumulator register count rises8 to32 per thread. Scheduled workgroups
fall byfour; packed weight/PCIe volume does not. Neither is a speed claim.

`KUNO_LAB_MTP_Q8_HEAD=shared-public` extends the existing head selector as a
mutually exclusive alternative to `broadcast-public`. It cooperatively stages
320original halfword-safe decoded head scales into1280B shared memory with one
uniform barrier, preserving the original512B reduction scratch, input/code
multiplication order,64virtual-lane dot/reduction, fault/tail logic and owners.
Initialization covers every slot; inactive rows writezero without reading weights.
It remains restricted to the public self-contained Q8 profile and selection
before any draft/KV operation. Actual completed full-forward receipt
`mtp_q8_head_shared_scales` is captured before close; broadcast receipt is false
in this mode. No other draft projection or target quantization changes.

The original cold/warm CPU13 and normal-language gates, no-swap/core constraints,
fixed seed/fixtures and recovered physical-state requirements remain unchanged.
Alternate Q2 and private/production inference do not silently receive these
policies. Required Vashti changes:none. Public APIs/installed defaults unchanged.

### Bounded exact host-expert staging experiment, 2026-10-11

The proposed default-off laboratory selector is
`KUNO_LAB_FROZEN_HOST_STAGE=packed-public`. It requires original authenticated
frozen host sources, partition exit, GPU source routing, public request reuse,
direct fill, chunk I/O, inline dispatch and the existing fixed public CPU13 or
language200 controllers. It is not a production model/API/configuration choice.
Invalid values, duplicates or incomplete prerequisites reject before device
construction. Original model/reference/seed and greedy repair are unchanged.

Construction reserves three transient PrivateArena owners of32MiB each for
packed gate, up and down weights (96MiB MAIN total). They must enter the complete
allocation inventory, stream/epoch/alias checks and full close erasure, separately
from retained original public source pages/pins. Six additional copy dispatches
per48-layer chunk receive589,824B extra runtime INFO reservation. Selection fixes
these reservations before model construction; attachment grants no new uncharged
allocation authority. The control retains its existing owner/budget inventory.

After original GPU routing publishes MAP, each of the three host shards enqueues
exact coalesced packed-word copies followed by its matching consumers, on the
same validated stream. For an originally selected host group, source addressing
is `(MAP.slot-first_slot)*expert_words+word`, scratch addressing is
`group*expert_words+word`; original MAP, expert IDs, slot guards, members, ranks,
probabilities, quantization and arithmetic stay unchanged. A256-thread copy uses
four coalesced256-word stripes per workgroup and grid`ceil(expert_words/1024) x30`. The largest
GU/down logical extents26,112,000/27,648,000B fit32MiB. Consumer staging addressing
is selected only after the original shard/member guard; inactive groups and stale
tails remain unread. The two resident VRAM shards keep their original bindings.

Queue order is router -> host shard copies -> consumers -> later overwrite.
There is no CPU route fence, independent ready flag or source-identity rewrite.
Original deferred route audits and full-family acceptance remain mandatory.
Receipts report completed actual staging policy, host/VRAM routes, copy dispatch
count and logical copied bytes derived from accepted unique host groups; staging
routes remain host-source routes. Transfer bytes are not device bandwidth or a
speed claim. Full original CPU13 vectors/carries/greedy/repair, source owners,
zero audit and recovered host precede normal same-artifact generation timing.
Original immutable RAM sources remain available; staging does not change model
storage, free original backing early or eliminate compulsory PCIe transfer.

Source parent integration now implements this exact contract. All three owners
are covered by full-model and MoE physical proofs and the29-extent frozen-source
proof; constructor limits require256-thread copy support. Actual completed flag
`frozen_packed_host_staging` and counters `staging_dispatches`,
`staged_unique_host_groups`, `staged_packed_bytes` require accepted original route
proofs. Copy traffic remains HOST traffic. No private/production selector exists.
Five leaf proofs,3root parser/actual dispatch checks,110Python checks,9fixed-label
scope/limits checks and strict direct all-target Clippy pass. Physical acceptance
and speed timing remain pending. Required Vashti changes:none; installed services
unchanged.

The initial v152/c3e9403 physical attempt rejected before target allocation:
the executor's separate exact private MAIN extent check still admitted only the
old router inventory. No numerical/staging acceptance is claimed for v152; host
recovered with no new faults. Parent now threads an explicit default-off executor
staging reservation, requires Config parity and frozen-router/target-row scope,
and admits exactly the old routed MAIN plus100,663,296B. The new pure extent
test rejects omitted/extra bytes and wrong scopes;4focused parent checks and
strict direct Clippy pass. A new frozen artifact must rerun the original GPU gate.


### Paired frozen public gate/up parent selection, 2026-10-11

`KUNO_LAB_FROZEN_GU_ROWS=two-public` selects two original GU rows per64-thread
workgroup in fixed public frozen/partition-exit controllers. It defaults off,
rejects private/workgroup tables and GU factors, and permits four-row down and
shared-down scales. Exact runtime GateKernel::TwoRows and grid320 are selected
atomically; cold CPU13 retains original GU dispatch. Actual completed receipt
`frozen_host_two_row_gate_up` requires the completed owner boundary and nonzero
accepted frozen source dispatches, rather than attachment alone. Original MAP,
quantization, F32 accumulation/reduction/SiLU, source identity, full zero-audit
and public-only limits remain. Detailed kernel contract is in the two-row leaf
note. Source checks alone do not establish GPU or throughput acceptance.

v150 four-row DOWN plus shared Q8 head scales passed original cold/warm CPU13,
real repair and full closure. The first ordered same-artifact code200 pair is
8.11952 control /8.48473 candidate (+4.50%); draft5.748s ->5.037s accounts for
most of the difference. This is preliminary, not repeated ABBA or production
acceptance. Both trials restored B70 to50,688,000B and NVIDIA compute0 with all
HTTP services200 and no new GPU faults. Exact frozen source/hash and numeric
receipts are under v150_rows_shared_head_physical. New paired GU physical gates
remain pending. No installed/public API/Vashti change.
