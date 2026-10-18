# Native DASLab/Strata checkpoint — 2026-10-17

Working checkout: `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`.

**Current CPU-only source:** `9a3567e6afe57de88835645786c72a0012b86200`, branch
`codex/strata-rust-port`. Read the [current source audit](kuno-strata-current-source-audit.md)
and [PLE arithmetic/reader receipt](kuno-strata-rewrite-completion.md#native-ple-materialized-products-and-default-reader-audit-2026-10-17).
Actual decode/prompt PLE gate and convolution variants now enforce original
materialized product rounding in native SPIR-V. Original explicit SCALE FMA,
complete nonannotation instructions, owners, state and graph identity remain.
The prior default wide dense target/MTP and prompt-helper fixes remain connected.

602 distinct selected CPU cases / 85 nonempty suites, strict core/off **binary**
Clippy, changed Rust formatting and whitespace pass. This is not an unfiltered
repository test run. Frozen artifact SHA256
`32cc84cc18a4389d373fcafb80a3f63bcb70913af90153763e0a4658564a55b7`, independently source/ordinary/frozen/mode0500/requested-and-resolved-features
verified. Only its early pre-adapter offline branch ran: 397 shader variants /
139 physical-pointer compiler checks / 77 rounded-product modules / 32 matrix
configurations pass. A public cancellation witness and actual five-kernel SPIR-V
regression demonstrate the materialization requirement; they do not establish
physical numerical/model/recovery/throughput acceptance. No new GPU heap owner,
model duplicate, content persistence, public API or configuration change.

**Whole rewrite incomplete; speed gate closed.** A concrete remaining selected
PLE reader gap is now mapped: the saved reference selects direct I/O and defaults
to a1M-row eight-way packed cache/256 inflight jobs, while current Rust has only
per-batch deduplication/four joined positional workers plus prompt prefetch.
Port that complete reader/cache/request lifetime and HOST pricing next; see the
normalized audit for exact current callers, default evidence and required joins/
erasure. This is required source work, not optional speed tuning. Finish the
remaining operator/prompt/queue/ownership audit and necessary substitutions too.
Protected whole-model prompt/count1/multirow, accept/reject/EOS/cap/context,
retained mixed/hot-request and failure/all-resource recovery acceptance remain
required before speed. CLI compiled-public/one-request; private caller separately
gated. Goal active CPU-only; sudo refresh only currently authorized. No GPU/model
inference/service/install/release/frontend operation. Required Vashti/API changes:
none. Canonical checkout and inherited untracked cache preserved.




**Previous coupled-sampler block (`8a4ee5b`):** actual full and CJK MTP paths
mutate penalties once, build fixed 64 lists and merge with request K. Source
counter/history/tie/F32 contracts were audited. 534 CPU cases, strict binary lint
and 360 offline variants/139 physical checks passed then; see the
[historical receipt](kuno-strata-rewrite-completion.md#coupled-draft-penalty-writes-and-fixed-64-selection-2026-10-16).
Physical numerical/model acceptance remains open.

**Previous whole-head block (`308de40`):** target and compact-CJK MTP each
project all original bounded Q6 pages in one dispatch, retaining actual physical
read dependencies and owning one 32 KiB metadata table each. Its total 65,536 B
public increment and unchanged two-original-tensor subtraction remain connected.
532 CPU cases, strict binary lint and 356 offline variants/139 physical checks
passed then. Read the [historical receipt](kuno-strata-rewrite-completion.md#whole-original-target-and-compact-cjk-head-dispatch-2026-10-16)
for exact ownership, artifact and still-open physical acceptance.

**Previous wide-Q6 block (`ffe7293`):** original A2/RPW1/SG32, aligned vector
loads and exact1–8 RHS native math remain connected. 527 CPU cases / 66 suites,
strict core/off binary lint and 340 offline variants/123 physical checks passed then;
see the [historical receipt](kuno-strata-rewrite-completion.md#native-aligned-wide-q6-heads-2026-10-16).
Its per-page dispatch gap is superseded above.

**Previous mapped-embedding block (`5b520bb`):** target and shared MTP gather
original HOST pages once with native32-lane block ownership/two float4 writes,
using existing source table0 tail and actual read dependencies.522 CPU cases,
strict core/off lint and335 offline variants/123 physical checks passed then;
see the [historical receipt](kuno-strata-rewrite-completion.md#original-mapped-embedding-gather-2026-10-16).
Physical numerical/model behavior and recovery remain unverified.

**Previous adaptive block (`53cd14c`):** original expert copies start after
verified row release, before commit/next MTP drafting, and join original copy/
primary visibility fences before physical/private publication. Original scratch
cleanup exclusion and first count1 usage are connected.514 CPU cases, strict
core/off lint and normal frozen artifact passed then; see the
[historical receipt](kuno-strata-rewrite-completion.md#adaptive-copy-flight-across-commit-and-drafting-2026-10-16).
Physical overlap, final substitutions and full-model acceptance remain unproven.

**Previous copy block (`8bd6edf`):** original three-role transfers use dormant
prompt raw scratch, matching HOST import flags and registered source publication.
Its synchronous after-window scheduling is superseded above.508 CPU cases and
334 offline variants/122 physical-pointer checks passed then; exact artifact/hash
are in the [historical receipt](kuno-strata-rewrite-completion.md#original-cell-transfer-copy-path-2026-10-16).

**Previous exact-page block (`ba0cc6a`):** actual exact allocator, shared prompt
inventory and matching model/admission charges connect. Prompt roots756/642/32MiB
(1430MiB total); target private source plan2,778,038,272B is842MiB lower. QSA32K
state saves192MiB. MTP complete private plan62,586,880B stays within128MiB grant.
The current copy path adds no allocation.504 selected CPU cases and54 ordinary
cases passed then; prior frozen artifact/hash and offline334/122 acceptance remain
in the [historical receipt](kuno-strata-rewrite-completion.md#exact-exclusive-pages-and-shared-prompt-inventory-2026-10-16).
No physical pricing/residency or speed conclusion.

**Previous sole-R block (`81ff964`):** all prompt GR/PLE halves update original
R in place, with single writable descriptors, packed mixed-image reuse and a
storage barrier before other-lane packed GR reads. Actual MTP teacher preserves
target carry using idle PLE key scratch.467 CPU cases and334 offline variants
passed. Its160MiB logical reduction is now reflected by the new owning inventory;
its old2080MiB physical-root quote is superseded by the source above.

**Previous subgroup block (`5d668c3`):**463 CPU cases,331 offline variants
(122 physical checks), strict core/off lint and normal frozen artifact pass.
Original selected device/module ownership preserves WGSL32 and cooperative16
through both pipeline builders. See the previous subgroup receipt for exact
artifact hash and physical acceptance limits. This source remains connected.

**Previous PLE region block (`9d7821a`):**
Key normalization uses one writable binding; query/conv regions share storage
after the final gate reader. Ten unique PLE views replace twelve; logical payload
shrinks320MiB inside unchanged2080MiB roots. Actual MTP teacher uses views8/9.
457 CPU cases /57 suites and331 offline shaders passed. The previous subgroup
gap described in that historical receipt is superseded by the current source.
No physical memory refund or numerical model correctness claim.

**Previous mixed-image block (`40fe97e`):**
Read [mixed-image receipt](kuno-strata-rewrite-completion.md#connected-native-mixed-input-image-reuse-2026-10-16)
and [active handoff](kuno-vashti-coordination.md#active-kuno-engine-handoff--mixed-input-image-reuse-2026-10-16).
GR now emits F32 plus native BF16/FP16 pairs in one writer. Matching original
source/count/first/stride and actual queued-writer publication select their
original views for GDN/QSA/indexer/router/shared projections and routed-expert
packed gather. Next GR invalidates the scope marker. No user-provided readiness,
extra model copy, new GPU root or application API is introduced.

455 distinct focused CPU cases /56 nonempty suites, locked selected test build,
strict core/off lint, formatting/whitespace and330 offline shader variants pass.
Frozen artifact for this preceding block `strata-mixed-images-ordinary-20261016/`, SHA256
`9d1bedf93be55ca5447bfa3ca8d40a29736820b7345335d182c0fac571f3a5d8`;
clean/stable source build and independent hash/mode0500 verified. No device
initialized. Linked receipt lists exact logs,122 physical shader checks and
source/CPU limitations. Images use40MiB inside unchanged2080MiB prompt roots;
net logical scratch reduction from pre-F1 is42.4375MiB. No physical VRAM refund,
model numerical correctness or speed claim.

Next complete native in-place GR residual/PLE consumers, allocator/whole-device
pricing and final source/scheduling/substitution audit, then required numerical/
state/full-context/hot-request/failure/recovery acceptance under renewed GPU
authorization. Protected private executable input remains separately gated;
current CLI is compiled-public/one-request. Whole rewrite incomplete, speed gate
closed. Required Vashti changes:none. Preserve canonical work/untracked files.

**Previous packed-GR block (`37dced9`):**
Read [packed prompt GR receipt](kuno-strata-rewrite-completion.md#native-prompt-gr-scales-and-packed-image-consumers-2026-10-16)
and [active handoff](kuno-vashti-coordination.md#active-kuno-engine-handoff--native-packed-prompt-gr-2026-10-16).
Actual default native prompt GR now keeps four row scales/token and writes BF16
normalized/SiLU images directly into original carved owners. Down/injection/up
consume those exact images with the existing Rust cooperative-matrix kernels;
mix reconstructs F32 values from original R/scales/gamma, never rounded BF16.
Standard injected reads remove three Pack dispatches (two without injection).
Logical carved payload shrinks82.4375MiB at4096 rows; fixed prompt roots stay2080MiB.
This is not a physical VRAM refund or speed claim.

453 focused CPU cases /56 nonempty suites, locked selected test build, strict
core/off lint and formatting/whitespace pass. Numerical/device acceptance remains
open. Exact normal artifact/offline shader receipt is in the linked completion
section. Frozen artifact for this preceding block is `strata-packed-gr-ordinary-20261016/`, SHA256
`fc0d1c655302c774427905e36932797c5c45eb3c02b9b2baf79403be8e198d7b`. Clean/stable build, independent hash/mode0500 and330 offline shader
variants (122 physical checks) pass. Previous artifact remains historical.

Next finish mixed BF16/FP16 input reuse, in-place GR/PLE consumers, allocator
and whole-device pricing/substitution/scheduling audit. Full numerical/physical/
hot-reuse/failure/recovery acceptance and protected private executable input stay
open. Whole rewrite incomplete; speed gate closed. Required Vashti changes:none.
Canonical work and inherited untracked files preserved.

**Previous direct-source block (`636c9f3`):** selected Intel direct-source GU/DOWN no longer retain
three unused128MiB fetch roots or an unused32KiB public stage-address table. Dense
Q8_1 preparation has one correctly bounded512KiB original-arena owner instead.
Actual binding schemas, retained source-read dependencies, constructor/queue
attachment and pricing change together. Prompt raw-ring/phase roots stay intact.
Global private inventory2,185,068,544 bytes; complete target private quote
3,660,939,264 bytes. Private reduction402,128,896 bytes (383.5MiB), plus32KiB
public table. These are source allocations, not observed freed memory or speed.

The reproduced nine stale legacy failures are reconciled using independently
pinned profile-specific expectations, preserving production admission/caps.
Final selected locked release test build and452 distinct cases/56 nonempty suites
pass; all31 legacy cases also pass under noncore Strata and original no-feature
reference. Both strict lint configurations and changed format/whitespace pass.
This is focused coverage, not a complete broad-core regression or device proof.
Exact log names and intermediate corrections are in the linked receipt.
Frozen artifact for the preceding block is `strata-direct-source-ordinary-20261016/`,
SHA256 `c0032c4ed9cc065712384deeb624055a7ec1be40af952ee3216986eb6db7b662`;
all330 offline shader variants pass. Clean/stable source build, independent hash
and mode0500 verified; no device initialization. Old12a60bf artifact is superseded.

The retained Session, original physical route restoration/full32K sampler and
bounded rendered-input consumer from the preceding blocks remain connected.
Executable input is still compiled-public/one-request. A separate private caller
must independently verify actual no-swap/core, original sources and exclusive
HOST/GPU admission before receiving private bytes; never widen public receipts.

Next continue native packed/in-place scratch/allocator/whole-device pricing and
final pinned-source scheduling/substitution audits. Numerical/physical hot reuse,
whole-model completion/failure/recovery and remaining old regression acceptance
stay open. Full rewrite incomplete; speed gate closed. No API/config/connector/
UI change; required Vashti changes:none. Preserve canonical work/untracked files.

**Preceding rendered-input block:** `e145841` connects an already-rendered UTF-8
reader/Rust-tokenizer consumer to the actual native Request entry. One wiping
buffer has1MiB text plus a single overflow sentinel. Actual token/context limits
reject with fixed errors; text/scratch drop before generation. Native HOST quotes
include a separate64MiB text/BPE/regex phase allowance;192MiB tokenizer cache and
4MiB generation/suffix owners remain separate. No private executable input or
GPU claim. Its421-case evidence is superseded by the current selected452 cases.

**Previous connected block:** source `a8bbdf0` plus `f1dd77f6a5a5cf98253488edec1fb761c1e9a0a4` fixes two
real dependencies found in the retained-state audit. Next-request binding restores
the actual private GPU route map from the original committed physical placement,
checks its adaptive-cache agreement and full original root/readback/padding/fault
under actual I/O admission before rearming. Swapped weights, owner epochs and
rounds stay intact; usage resets. Live sampler/prompt/PLE HOST scopes now reject
reset/rearm. No profile identity reset or extra weight/GPU source is introduced.

Native sampler request positions, consumed history, handback/controls and verified
prefix admission now reach32K, independently from upstream's unchanged4096-slot
penalty window/BASE/RING/shader ABI. Ordinary sampler bookkeeping remains4096.
419 distinct CPU cases/52 nonempty suites, final locked release test build, both
strict lint configurations and format/whitespace pass. Exact logs and retained
HOST/control inventory are in the linked receipt. No GPU/numerical/speed claim.

Next connect the protected arbitrary-input/tokenization caller. Repeated physical
requests still need actual device/numerical/failure/recovery acceptance; the public
executable remains compiled-public and one-request. Finish the remaining packed/
in-place storage/allocator/whole-device pricing and final pinned scheduling/source
substitution audit, preserving the old regression. No new shader/generator or
normal frozen artifact;12a60bf is stale. Goal active CPU-only; no service/model/
GPU/install/release change or required Vashti change.

**Previous connected block:** actual `Session::load/run/close` owns the original
runtime/tables/executor/target/MTP and immutable public weights. The one-shot
wrapper uses that same lifecycle, with explicit checked cleanup for preflight
rejections and original all-attempt terminal/physical recovery ordering. Session
is thread-bound. Full32K model capacity is separate from actual bounded request
geometry. Repeated requests consume original non-cloneable erasure receipts,
rearm monotonic language scopes and use a distinct retained-ready gate instead of
impersonating generation0 PLE. Selected FourGdn/grouped PLE/status policies and
all owner/epoch/counter/retirement state are preserved. Controller checks its
actual newly issued scope at exit. Successful requests clear/check private owners
before publishing Ready; terminal close independently audits original owners.

404 distinct CPU cases/50 nonempty suites, final locked test build, both strict
lint configurations and formatting/whitespace pass. Exact logs in the receipt.
No shader/generator/arithmetic change or new normal frozen artifact.12a60bf is
stale; build current source before any newly authorized physical acceptance.
No source/component is selected or dropped by throughput; speed gate stays closed.

**Hot repeated model execution is not physically validated.** The current public
executable still uses the one-request wrapper; the internal reusable session is
connected in source. Existing request-end sanitization clears recordings/parameter
and binding/header caches, retaining pipelines/weights. The new retained HOST/control source inventory is above; connect a protected
arbitrary-input/tokenization caller. Remaining
packed/in-place scratch/substitution and whole-device pricing, pinned source/
scheduling audit, old regression and physical/numerical/failure/recovery gates
remain. Full rewrite incomplete; no private/general-service or throughput claim.
No API/config/connector/UI change; required Vashti changes:none. Preserve canonical
application work and inherited untracked files.

**Previous connected block:** `4ceb86f` introduced completed target/MTP retained
private-state erasure, original-owner receipts and dedicated64MiB inspection HOST
pricing before the then-one-shot terminal close.402 distinct CPU cases/50 suites
and both strict lint configurations passed. The loaded-session block above now
connects bounded rearming and preserved-policy reuse without loosening13-only replay.

**Previous connected block:** `cbf4381`/`910cb58` price/preload/seal all784 original
dense projections before expert-source attachment. The actual256-byte dense roots
plus shared head quote4,166,315,008 bytes, replacing5GiB. Ranked expert capacity
uses all remaining shared MAIN, without the uniform240-slot cap. Uploads retain
real source/root admission, actual joins/checkpoints and precise logical views;
cache misses/shape changes cannot evict retained native weights.512KiB metadata
is priced in the original HOST envelope. The original1224-tensor receipt proves
all projection names/formats/shapes and separation from norms/experts/tables.
387 distinct CPU cases/44 nonempty suites and both strict lint configurations pass.
No shader/arithmetic change. No new normal frozen artifact: the12a60bf binary is
stale; freeze current source before any newly authorized execution check. No
current device/residency/performance claim. That earlier entry was request-scoped;
protected arbitrary input and hot public-model/request reuse are still open.

**Previous connected block:** native mutually exclusive PLE/GDN/QSA/MoE phase
scratch, with persistent GR/residual and independent eight-slot raw ring retained.
Actual roots `[1GiB,1GiB,32MiB]` total2080MiB,992MiB below the previous3GiB.
Working images80MiB, GU20MiB and routed partials400MiB; real256-byte view gaps and
all source/erase/allocator/copy bounds remain.256-query score/selection batches
reuse8MiB scores and8MiB flags, retain the full2051-stride selection owner and
feed existing32-query attention. Query views stay aligned; selected logical row
bias avoids unaligned descriptors. Extra INFO records are priced together.
378 distinct CPU cases/42 nonempty suites and both strict lint configs pass;
frozen artifact/offline shader receipt is linked above. No GPU or throughput claim.
Power-of-two private roots and distinct F32/ping-pong scratch remain documented
substitutions; full pricing/lifetime/source acceptance is not closed by this block.

**Previous request owner:** `dc7ace7` adds `strata_engine::Request/execute`. The public
fixture calls that actual source path before its legacy constructor/receipt code.
Raw token buffers and sampler controls have wiping owners, arbitrary bounded
prompt/output partitions reach32K, and tokens are delivered through the caller's
callback. Original shared target/MTP construction and ordered cleanup/actual root
recovery remain coupled. No engine file output or receipt/preview logging; the
public harness owns its synthetic diagnostics separately.374 distinct CPU cases/
41 nonempty suites and both strict lint configs pass. Normal freeze/offline shader
receipt is documented. No GPU or model execution, private-input or throughput
acceptance. The executable caller remains compiled-public; protected arbitrary
input/tokenization and verified hot public-weight/runtime reuse remain open.

**Previous connected block:** 32768 runtime context, while reference graph geometry
stays unchanged. Actual target/GDN/QSA and MTP constructors, KV/positions/RoPE,
8192 pooled blocks, selection, prompt traversal, request/result/router guards and
typed graph controls agree. Sparse QSA retains2051 selection stride/33 chunks;
dense MTP uses512 chunks of64 cells. Genuine full cache sweeps require67584 groups,
so Guard/Rewind alone reshape to65535x2 and consume both coordinates. Full padded
owners/erase/admission remain required. MTP private quote64 ->128MiB, target/language
runtime envelopes256 ->512MiB; actual per-owner charges remain tracked. Suffix
history covers32768 and fits its original4MiB HOST envelope. Sampler sliding
history remains4096. Ordinary engine context/default geometry remain unchanged.

Previous 32K receipt:371 distinct CPU cases/40 nonempty suites and both strict lint configurations
pass. Normal release freeze/offline shaders are detailed in the receipt. These
are not GPU/numerical/model acceptance. An earlier EXTRA old frozen-down test
still fails at its unchanged16K-vs-core32K scratch expectation before its runtime
assertion; full regressions remain open. No new speed result. No consumer/API change.

**Historical capacity (allocation superseded by12a60bf above):** `bcc30dd`/`669437d` connect4096-row prompt owners/ABI, three
owned1GiB roots (3GiB MAIN),128MiB working views, route/group planes, HOST/runtime,
full coverage and actual1GiB buffer/binding plus65535-grid inventory guards. The32K
QSA views still fit those original admitted roots; no optional cache loan enabled.
Physical device support is UNVERIFIED. Conservative scratch differs from upstream
Alloc::carve; exact pricing/aliasing needs review.

**Previous coupled blocks:** `89da25f` shares the original mapped HOST embedding
with target/MTP (head MAIN521474048/HOST337715200/addressable859189248 bytes), with
reader-before-detach lifetime. `f9d1e06` scopes native graph/conflict admission,
preserves INFO visibility after copy completion and pins original allocator
bindings through scope end. Ordinary/cold/final drains and original pending
submission/MAIN debt/R loan/family/numerical/I/O/state proofs remain separate.

**Next:** protected arbitrary-input/tokenization caller, verified request reuse,
remaining private prompt/whole-device ownership pricing,
source scheduling/lifetime/substitution audit and legacy regression reconciliation.
Then full numerical, physical and actual recovery acceptance with renewed GPU
authority. No speed trials until the complete connected rewrite gate passes.

**Historical native uploads/reuse (superseded by7477b66/f9d1e06 above):** `7cb1cbc` connects original data, INFO append and
direct INFO replay to one bounded write-only renderer and explicit original upload
states: Idle, WgpuMapped, NativeReady, NativeWriting, NativeWritten. Native reuse
consumes a real completed grant BEFORE map/render/flush/unmap; only success stages
bytes. Failed/partial writes cannot be recycled or copied. First-use Wgpu map duty
stays real; native reuses flush non-coherent full ranges and unmap locally. Original
source index, full padding clear, typed controls, I/O budget and final erasure stay
owned. No readable mapped slice/pointer is exposed to byte/template producers.

Selected native checkpoints now use `join_all_io_sources` +
`recycle_native_after_all_joined`, with original request/leases/INFO/GPU errors and
full-cohort preflight. No Wgpu upload remap callback or primary reuse transitions/
Readback clears; original native packet fill/copy/fence/map remains required. No
ledger release, erasure or target-ready credit. This remains an ALL-OWNER global
completion boundary. Do not replace its join with a peer/current-lane fence.

276 focused CPU cases across 22 nonempty suites, selected/off strict Clippy and
format/whitespace pass. Actual original I/O/cache/parameter tests use a CPU harness
with device activation disabled and generic lab feature cfg; they are not native
map/device/coherence/erasure or selected shared-budget proof. See exact logs and
failed dependency-test attempts in completion receipt. No shader/layout or frozen
ordinary artifact; `0bf0b5f` is stale. No GPU/model/service/protected test or speed
result. Sudo refresh succeeded. Required Vashti changes:none.

**Earlier next block (selective inventory now implemented by `7477b66` above):** selective original I/O owner inventory, full host-write
reuse conflicts against original flight/INFO-copy ownership and exclusion from
native grants/free-index publication, coupled with current-lane/conflicting graph
admission for entry/push/execute/replay/flush. Preserve original source/descriptor
pins for FUTURE retained replay, not just currently running math. No masked all-
owner grant or metadata-only completion. Keep queued-vs-submitted distinction,
INFO epochs/visibility, cold initialization, exports/MAIN debt and final/failure
drains together. Positive-confidence per-cell checkpoints remain in scope.
Other full rewrite/arbitrary-input/context/embedding-placement/regression/
numerical/physical/recovery gates remain; live GPU work still needs authorization.

**Latest native packet path:** `7ad47ca` connects real fill/copy-only unnamed
recordings and their original selected-lane/copy timeline final fence, explicit
TRANSFER_WRITE -> HOST_READ availability and checked synchronous Vulkan HAL host
mapping. Both readback owner and original lease remain in the same I/O ledger.
Full clear initializes all owned padding; complete original copies stay intact;
only logical bytes go to the existing wiping host controller. Non-coherent maps
invalidate the full actual range. Native unmap is local/guarded; no pointer or HAL
borrow crosses async return. Original lease remains held through returned Bytes.
Error/panic/late map retires, with one 30-second batch map deadline. Actual cold
source initialization remains mandatory. Original packet I/O cannot overwrite a
named math recording/key. Ordinary and terminal cleanup readback keep Wgpu.

165 focused CPU cases across 20 nonempty suites, strict selected/off Clippy and
format/whitespace pass. No shader/layout or immutable artifact change; frozen
`0bf0b5f` is stale. No GPU/model/service/protected tests or speed result. Sudo
refresh succeeded. No API/connector/UI change; required Vashti changes:none.

**Earlier next block (native global reuse now replaced by `7cb1cbc` above):** graph current-lane/conflicting-flight admission plus
selective original I/O reuse/source-remap completion authority. Current scope
entry/push/execute/flush still drain globally; `recycle_reuse` still maps uploads
via Wgpu and submits primary transitions/readback clears. Native packet copying/
mapping itself no longer submits a primary copy or requests a map callback.
Do not fake its legacy Wgpu callback counter. Preserve cold initialization,
INFO full-source/destination copies and exact header epochs, original commit
exports, MAIN debt and terminal failure drains. Positive-confidence per-cell
observations need their surrounding checkpoint fixed too. No physical overlap
claim until that whole scope is connected and verified. All other completion/
arbitrary-input/context/embedding-placement/regression/numerical/recovery gates
remain open; live GPU work still needs renewed authorization.

**Latest original draft scope:** `d2d3228` connects a real MAIN->draft readiness
signal/wait BEFORE accepted-prefix commit, retaining the actual current R resource
and allocator binding. The private original graph/request/generation identity
must match preparation, entry and actual completed end; a token alone cannot
create readiness. Scope entry precedes the Pending observer, including sampler
controls, scratch/token/R-copy/front preparation and every named MTP branch.
Unkeyed queue selection stays on the draft lane; target branches inside it and
premature/repeated/stale entry reject. Actual source ranges remain read-only from
preparation through completion, including partial dependencies. Cold nodes,
physical/copy/INFO accesses and original direct replay plans enforce that pin.
The existing independent carry-generation/window receipt remains mandatory.
Follow-up `dcd783b` also requires the original ordinary cross-lane handoff to
complete at terminal cleanup; failed flush plus empty math is not release proof.

Actual completed scope end restores MAIN visibility before releasing the source.
Unknown/failing private guard Drop retires original flags before its retained
client, without locks or release. Owning terminal drains retain the actual source,
allocator and readiness objects; unknown Graph Drop retains them through process/
device teardown. Target exported acknowledgement, MAIN debt and all numerical/
family/PLE/I/O/erasure obligations remain distinct and required.

163 focused CPU tests across19 nonempty suites and selected/off strict Clippy
pass. See completion receipt for exact logs. No shader change, new immutable
artifact, GPU/model/service/protected test or speed result. Frozen `0bf0b5f` is
stale for current source. Sudo refresh succeeded; no system change was made.

**Earlier next step (packet readback partly resolved by `7ad47ca` above):** complete independent execution and I/O as one connected scope. Current
entry/push/execute/replay/flush/metadata/readback/recycle globally join; entry can
physically join target even before verified delivery. Do not claim overlap. Move
current-lane/conflicting-flight admission, INFO source remap authorization,
confidence packet copy/map and upload/readback reuse together. Distinguish queued
nodes from submitted flights; preserve fresh Wgpu initialization visibility and
positive-confidence per-cell observation. Full completion/arbitrary-input/
context/embedding-placement/regression/numerical/physical/recovery gates remain.

**Latest recording/INFO ownership:** `a8e538f` retains the actual descriptor,
physical read-dependency, copy/fill and indirect intervals in each original plan.
Canonical ranges preserve holes and shared readers; any historical write remains
a write. The original recording owns this inventory through its real lane fence,
including direct replay. Its actual capacity is charged within the existing64MiB
retained-recording metadata bound; the INFO2MiB bound includes temporary accesses.
No GPU data owner, model mirror, request values or serialized roots are added.

INFO copies use their actual whole original source/read and target/write owners.
`Graph::upload_parameters` preflights both original flights, rejects missing/
partial/foreign-family owners and joins only conflicting original lane fences.
It then completes only the previous INFO copy before reusing its copy pool.
Original runtime export/ack and shared MAIN retention remain authoritative. A final
drain now decides outstanding INFO visibility after math joins: an older consumer
cannot stand in for a newer disjoint upload.

156 focused CPU cases and selected/off strict Clippy pass. No shader/layout change,
new immutable artifact, GPU/model/service run or speed result. Latest frozen
`0bf0b5f` still predates both this and the pending-retirement block; build current
source before any protected execution. Only requested sudo timestamp validations
ran in Demons. Required Vashti changes:none.

**Prior next step (partly implemented by `d2d3228` above):** connect the complete independent MTP scope, not only named branch math.
The completion guide records the coupled design and actual source findings:
private uploads already use mapped sources/native copies; generic core INFO pages
are distinct from named branch pages. Keep those owners. Signal verified-R
readiness before commit, then route sampler controls, scratch/token/front work,
math, confidence packet readback and reuse under the same original scope. Global
push/flush/replay/switch/recycle waits remain today. `Graph::pending` includes live
flights as well as queued nodes; do not mistake the former for a new dispatch.

**Earlier pending retirement:** `543bc73` couples the original runtime submission
with its shared MAIN debt hold in `PendingNativeCommit`. Each actual multirow
prefix submission acquires the hold before invoking the scoped drafter. Peer
observations still check real growth/limits but cannot lower the retained upper
bound. The guard's consuming join uses its original client and exact original
runtime token, then releases only its hold; a later real checked boundary must
still validate telemetry before reclaiming MAIN credit. Count1 adds no submission
or hold. No metadata-only completion, reset, clone, send or erasure API is added.

Unknown/failing guards atomically retire the original context/sequence without
locking admission or releasing debt. Their completion field drops before the
retained client. Original close still attempts its best-effort actual drain and
cannot claim clean after retirement. CPU cases exercise retained credit across
peer fences/additional allocations, both bounded holds, duplicate/unbalanced
transitions, unexpected growth/poison and monotonic failed-drop flags. Original
backend flight/export validation remains the actual join authority.

152 focused CPU cases and selected/off strict Clippy pass. This block changes no
shader, GPU root or public API. No new immutable release artifact was frozen;
the latest frozen `0bf0b5f` artifact predates this accounting change. Build current
source before future protected execution. Existing330 shader receipts are
historical, not a new physical validation. No GPU/model/service/sudo work ran.

The global waits remain. Next retain the actual original recording access ranges
under the existing metadata budget, use whole-owner INFO-copy hazards (rather than
invented record slices), and establish R readiness before commit with all MTP
preparation on its own lane. Shared I/O recycle still flushes the whole context;
this hold alone is not permission to remove those waits or claim physical overlap.

**Earlier scoped draft consumer:** `1f46f76` adds a second limited phase to the
original prefix observer. After actual provisional row/fault observation and
release of its host lease, the target submits its original commit, then lends
verified R to the actual MTP consumer before explicit commit acknowledgement.
This grant authenticates original graph, arena root, generation, full proposed
window and accepted prefix. It cannot expose target state, a ready cursor or
reset/erase authority. The existing copy-completion guard remains sticky on any
unjoined consumer. Original target fault/I/O/family proof still precedes cursor
publication. Verified-token delivery is described separately below.

The selected controller now retains a completed next draft, not an unconsumed
prior R between rounds. Original prompt R seeds the first draft; later drafts
run inside the scoped pending phase. Request-local sampler history consumes only
the accepted input prefix before coupled next-draft controls. Confidence trimming
retains the actual producer cursor and first candidate; EOS/output-cap stop skips
the next producer without shrinking state acceptance. Entries authenticate that
same completed MTP instance/cursor before use. Token/control buffers remain
bounded and wiping, with no new GPU data root or persisted request content.

**Physical overlap is not yet established.** Global graph/stream guards still
drain math. Delivery source `0bf0b5f` aligns the source order: after real
whole-target row/fault observation and commit submission, emit the clipped verified
predictions, then run next drafting, then join/prove target state before publishing
its cursor. This is target-output delivery, not completed state or request success.
A later draft/commit failure terminates the stream with partial verified output,
retains normal poison/cleanup, and never publishes a ready target. Original token-
delivery errors survive terminal cleanup. Pinned `generate.cpp:11089..11140` has
this emit-before-draft order; its final wait is at11169.

Next establish source readiness before target commit and put ALL MTP preparation
on its independent lane. MTP observation is still synchronous and global drains
can still join the pending target before independent execution starts. Preserve
original source borrows, target/final joins and error ownership when changing that.

At source `0bf0b5f`, shared MAIN completion still credited the whole context.
The pending target hold above supersedes that behavior for its actual interval.
Full-owner INFO hazards and shared I/O recycle remain open: the latter still
flushes the context and is not an independent-lane observation today.

113 focused CPU cases and selected/off strict Clippy pass. Exact immutable/offline
receipts are in the completion guide. No GPU/numerical/visibility/speed/recovery
acceptance, model/service changes or API/UI changes. Required Vashti changes:none.

**Earlier pending commit receipt block:** `91e1f9e` connects separate pending
cold/replay APIs to the selected prefix path. They return original graph/scope/
submission identity after actual queue submission, without joining its math
fence. The original joined bool API still waits before returning true. Every
native submission now has a monotonic epoch distinct from its reusable recording
ID; stale receipts cannot join a later replay. Exact original request scope,
recording and lane fence are validated at join. A dependent drain can complete
the fence first; original completion metadata survives until acknowledgement.

The engine explicitly joins that receipt immediately before its existing real
fault/I/O/family proof. **No independent draft overlap yet.** Keeping this join
is deliberate until original row readiness, MTP preparation, actual intervals and
INFO destinations have the coupled dependency contract. Pending state does not
publish a model cursor, numeric success or owning erasure. Partial/error owners
remain retained; cleanup can invalidate exports only after actual joins. Pending
profiling is rejected, while the old joined APIs preserve their behavior.

Each original flight also retains its actual INFO timeline wait. Only its real
math-fence join credits that epoch, never a newer unrelated upload. This avoids
mistaking explicitly joined commit math for a copy with no completed consumer;
it does not itself establish independent cross-queue visibility. Fixed flight/
export metadata still fits the original1KiB bound. No GPU data root is added.
107 focused CPU cases, selected/off strict Clippy and formatting pass; immutable
release and330 offline shader receipts are in the completion guide. No GPU/
numerical/visibility/speed/recovery acceptance. Installed services are untouched.

Next is the scoped independent-draft consumer and real verified-row readiness
edge *before* commit. MTP `start_window` already clears/stages/copies and performs
private guards before named math capture; selecting a queue only at branch begin
is too late. Audit global graph/stream drains, INFO uploader reuse, copy/compute
header waits and queue-switch handoffs together. INFO copies cover complete
owners, so merely disjoint record offsets do not prove a live uniform page can
be rewritten. Preserve original family/row/flight ownership and final joins.

**Earlier retained commit graph:** `4083258` connects original preflight and real
capture/direct replay in `resident_native_commit.rs`. Multirow commits have their
own model-instance/proposed-T/context/GDN-PLE-role namespace. Accepted n and current
position are typed invocation INFO; they are never graph keys or frozen constants.
Every warm hit rechecks original armed family/input receipts and their exact
start/proposed extents. Count1 self-commits its existing candidate without a graph.

All133 multirow consumers keep fixed proposed-span geometry:36 native GDN
Conv/Replay/finite-status fold triplets,12 QSA clear/fold pairs and1 PLE History.
Accepted n bounds actual work inside those kernels; all-accept QSA is a real
zero-length no-op within the same graph. This avoids capturing an all-accept body
that omits the later rejection consumers. Actual cold/hit submissions still join
through the original runtime API, and the original fault/I/O/family proof still
precedes state publication. No new GPU state root or I/O budget is added.

104 focused CPU cases and strict selected/off Clippy pass. New actual vendor
parameter tests replay original pages across every prefix1..T, all4 original
binding roles and final-context positions, keeping one erased template per shape;
the one-model122-shape inventory fits the original128-branch bound. These are
not physical GPU/visibility/numerical acceptance. Immutable/offline shader
receipts are in the completion guide. **No async overlap is claimed.**

Next is the actual pending event and independent draft consumer. Review the
single INFO uploader, stream drains, live uniform destinations, queue-switch
handoffs and MTP preparation *before* its named branch together. The current
MTP `start_window`/`prepare_window_first` happen before `begin_native_strata_invocation`:
leaving that work on main behind a pending commit would serialize drafting.
A dependency on verified-row readiness must be signaled before commit, rather
than a new main-queue signal placed behind commit. Original resources/permits,
final proof and failure/cleanup joins must stay owned. Keep the whole rewrite
incomplete and do not use incidental speed to choose or abandon components.

**Earlier direct accepted-prefix block:** `131d29c` connects the selected global
controller to `ResidentModel::verify_native_prefix`, using an exclusive scoped
observer of the real provisional target rows. Every original family transaction
remains armed with its full proposed-window stamp. The observer chooses n and
extracts the original device carry; its host-row lease drops before state work.
Only the accepted prefix is then published. No full-window cursor/state commit
precedes rejection repair in this selected controller anymore.

Native GDN uses the original retained inputs and prewindow OLD to form accepted
NEXT, including all-accept multirow replay. Count1 already has its one accepted
candidate. QSA clears/folds only the rejected raw-cache/pooled suffix, while PLE
forms accepted NEXT on GPU and prepares only the accepted two-token history.
The actual second fault/I/O proof covers these real consumers before all-family
preflight/publication. Original policy accounting still consumes all proposed
receipts; lifecycle and model/PLE cursors publish only n. There is no edited/fake
receipt count, generation/poison reset or new GPU state owner. Additional coexisting
runtime INFO is conservatively charged544,768 bytes in original admission.

102 focused native CPU checks, selected/off strict Clippy and formatting pass;
immutable build/offline shader receipts are in the completion guide. These are
source/CPU checks, not GPU/numerical acceptance. The diagnostic `LatestTargetRound`
path still retains its older completed-window/repair schedule; it is no longer the
selected native controller. **Commits still join synchronously.** Next is the real
pending event/commit graph and independent draft consumer, with actual queue/INFO
hazards and final joins preserved. No live speed trial is authorized yet.

**Earlier GPU PLE history block:** `595393e` plus `57147a8` keep the actual committed PLE float
history in its existing GPU OLD/NEXT owners. Native preparation neither clears nor
uploads committed OLD. Real full-window completion swaps their roles; CPU publication
advances only the original two ngram token IDs with a checked ordered token permit.
Default native decode no longer maps OLD/NORMALIZED/NEXT float histories to the CPU.
Native prompt completion uses the same token-only permit; it no longer copies a
diagnostic tail or reads GPU float history unless diagnostics are explicitly
selected. Full prompt, decode and short prompt continuations share the actual
existing GPU roots rather than rebuilding a stale CPU convolution mirror.
An explicitly enabled grouped-read diagnostic still performs/credits actual reads.

Native latest-window checkpoints retain exact token-owner/control metadata, not a
CPU float-history snapshot. Rejection queues the existing GPU History kernel over
original prewindow history and original normalized accepted rows; zero-prefix restore
queues the actual original history copy. Only the existing joint fault fence can
precede queued PLE/token/cursor publication. Original poison/drop and full-root
validation remain. Legacy CPU history/diagnostics are preserved. No private GPU root
is added. Target recording keys now encode independent GDN bit0/PLE bit1 roles;
all4 binding inventories are distinct, without putting positions/content in keys.

61 focused native CPU checks and9 core transition tests pass, including mixed
2048-row prompt/decode/short-prompt continuation and every decode prefix0..8.
Selected/off strict Clippy and formatting pass. Immutable release/offline shader
receipts are in the completion guide. **No GPU/numerical/speed acceptance claim.**
That earlier block still committed full proposals before prefix repair; `131d29c`
supersedes that schedule for the selected native controller. GPU role swaps
and global joins remain explicit source scheduling substitutions. Do not reupload
`model.ple_conv.history()` during native decode: that legacy CPU float owner is
not authoritative after native GPU-history advancement.

**Earlier lane-flight block:** `86f4c35` replaces the global single math fence and
last-recording completion assumption with2 actual lane fences and a bounded original-
recording flight book. All cold/hit/direct submissions use it. Each recording has a
nonzero monotonic identity; its original family/compute count must match the submit.
The slot is reserved before fence reset or queue submission. Partial submissions
remain owned and get no completion credit. Live recordings cannot be replayed,
replaced or evicted; completion/profile collection resolves the original identity,
not the current LRU tail. Duplicate enqueue handles drop only after successful
submission; actual recording resources stay pinned. `pending()` still includes
in-flight work so original readback/cleanup drains are preserved.

46 focused CPU checks, selected/off strict Clippy and formatting pass. Immutable
release and327 offline shader variants also pass; exact hash/logs are in the
completion guide. No GPU/model
or numerical/speed acceptance. **Global joins remain selected** until the family-
state/consumer contract is ported; two owned fences do not alone enable overlap.
No extra private tensor/state owner or public API is added.

**Earlier native INFO block:** `414e2c3` moves actual current Upload-to-Uniform
copies onto the adopted native transfer queue for cold capture and direct replay.
A reused transfer command buffer retains the actual complete source/target handles
through its fence; the math recording's original header waits on its real timeline.
Upload allocations may change across invocations, so no stale source is baked into
retained math. Original mapping, zero-padding, typed/schema/page and wiping ledger
obligations remain. INFO-only drains establish device visibility for later native
and Wgpu consumers even without a math-header wait.

Native-only flush/replay no longer submits/polls an empty primary command buffer.
Actual primary encoder mutations (initialization/readback/clear/generic compute/
legacy upload/profiling) are tracked, and real work still submits/joins. Feature-off
flush retains its original unconditional policy. The bounded4096-copy metadata
adds2MiB to `NATIVE_QUEUE_HOST_BYTES`, consumed by the original host admission quote.
Mapped Upload owners already have MAP_WRITE and unmap flushes noncoherent writes;
they do not use Wgpu's unmappable initialization staging-copy path.

43 focused CPU checks, selected/off strict Clippy and formatting pass. Immutable release
and327 offline shader variants pass; exact hash/logs are in the completion guide. No GPU/model execution,
numerical/speed/recovery or installed-service claim. **Pending target commit is
still joined globally; the rewrite remains incomplete.** This source completes
native INFO transfer, not asynchronous accepted-prefix publication.

**Earlier queue block:** `b6732c8` connects MTP native branch capture/replay to an
actually requested independent compute family. The saved Oct14 B70 properties have
one main0 queue, compute-only1 and transfer-only2. Production selects by real flags/
counts on device creation, not those numeric IDs. Missing independent compute or
dedicated transfer capability rejects this experimental native device, without
silently aliasing queues. Ordinary device constructors and installed Kuno are unchanged.

Native pools/recordings are family-bound; uniform/storage buffers plus raw host
mirror/prompt-shadow aliases share the actual adopted families concurrently.
The graph retains one shared initialization/resource inventory. Each joined branch
switch and fresh primary-queue upload uses actual timeline semaphore signal/wait
handoffs; a CPU fence alone cannot supply Vulkan cross-queue memory visibility.
The monotonic handoff/fence objects remain owned across partial failure and are
not freed if completion is unknown. Original output/fault/erasure acceptance stays.

40 focused CPU checks, selected/off strict Clippy and formatting pass. Immutable
release and327 offline shader variants pass; exact hash/logs are in the completion
guide. **This
connects the second queue; it does not yet overlap target commit and drafting.**
There is still one in-flight native graph fence, synchronous accepted-prefix repair,
and primary-queue Wgpu INFO upload/readback scheduling. Unkeyed MTP prompt/preparation
work also remains on the main queue. No GPU/model/service/sudo operation ran.

**Latest direct MTP block:** `d0670b0` connects actual retained native submission
for whole zero-threshold chains and independently observed positive-threshold steps.
Cold/unsupported recordings still enqueue original math. A hit bulk-renders the
actual typed parameters and submits/joins original retained commands, skipping the
per-kernel loop. Upload/submission/join errors remain terminal. Real proposal,
probability, padding, fault, source and ledger checks still run before the same
completed-forward publication; no cached output is accepted.

The owner caches at most128 shape/accounting entries (32KiB conservative two-copy
metadata cap within the existing host margin). Entries come from actual captured
traffic only after a successful packet observation. They retain only static keys,
operation counts and per-token budget behavior, never origins, prompts or tokens.
Whole-chain replay preserves first-token budget, intermediate resets and final
counter. CPU sampler handback reproduces original proposal/counter advancement only
after actual completion; the recorded GPU bridge remains the draw/history/token
producer. Existing model identity/request nonce plus new coupled-versus-greedy
key separation preserve original graph inventories. Original roots and source are
validated; first-front tickets are consumed only after real submission completes.

73 focused CPU checks, selected/off strict Clippy and formatting pass. Immutable
release build and327 offline shader variants also pass; exact receipts/hash are
in the completion guide.
No GPU, inference, numerical, throughput or installed-service test ran. The source
connection is implemented, **not physically accepted**. The earlier typed-only
blocks below are historical on the direct-MTP limitation.

**Latest MTP phase block:** `8262367` connects explicit typed MTP round/step
controls to actual queued math. A round's front uses origin p; selected rest and
follow-up drafts use p+rows-1+j. Independently observed steps use their current cell
origin with the matching original step index. The backend drains preceding scheduled
kernel metadata before changing the phase, without explicitly flushing/joining the
GPU. Declared phase fields canonicalize to immutable base displacements before
capture. No old-value matching or position in shader/cache identities.

Readable positions stop at4095; boundary expressions permit4096. Native front and
QSA position records use the actual declared phase; **Identity also uses start**
and is tagged. Select's unrelated zero remains static, Rewind stays outside capture.
Core fused-sweep cache_position metadata is normalized to zero only after verifying
the actual packed-cache INTEGER roles, whose predicate never reads it. Separate
packed-cache guards retain the real explicitly declared current/closing boundary.

MTP graph keys now include its existing immutable nonzero model-instance binding,
restoring the original per-drafter graph ownership on the shared target/draft stream.
Typed MTP controls reject target-only route fields.67 focused CPU tests and strict
selected/off Clippy, immutable release build and327 offline shader variants pass;
exact artifact hash and receipts are in the completion guide. **Superseded by `d0670b0` above: this older block still enqueued math.**
No new GPU/private owner, model/service/sudo operation or speed trial this pass.

**Previous MTP staging block:** `862f5fa` separates first TOKEN/target-residual staging from
native front math. The actual window prepares an opaque original-owner/cursor ticket
before capture and consumes it in the first front. A dropped ticket leaves an
owner-local pending flag set, preventing another preparation, observation or full
forward publication while allowing original failure cleanup. Legacy scalar/teacher
callers compose the same stages. No new private GPU root exists.

Coupled sampling now stages every possible draft's `(cell,j,round base)` once at
`begin_round`, in disjoint aligned4KiB views of the original32KiB CONTROL owner.
`draft_pick` no longer uploads a control packet in the captured math. Original
GPU bridge/RNG/history/CJK mapping and CPU phase checks remain active. Actual
producer packet sealing precedes graph closure; actual private readback and each
positive-threshold decision follow closure. Borrowed target residuals remain held
until the same final completed observation.

This pass:26 focused CPU tests, selected/off strict Clippy and formatting pass.
Frozen release build and327 offline shader variants plus matrix/physical-pointer
checks pass; exact artifact hash and logs are in the completion guide. No GPU,
model, numerical, throughput or installed-service test ran. This older block left
MTP untyped; `8262367` supersedes that limitation. Direct MTP replay is still
pending.

**Previous connected block:** `d40442c` connects actual direct target graph replay.
The first eligible verification window captures normally. A repeat with the same
request nonce, static shape and original GDN state role bulk-renders fresh position,
routing epoch and verification-usage controls into its original INFO roots, joins
the uploads, and submits the original retained Vulkan commands without rebuilding
kernels, nodes, descriptors or signatures. The target skips its math enqueue loop.
Original GDN/QSA/PLE pre-armed transactions then consume the completed submission;
GR/shared must be idle, GDN replay-cache metadata gets the exact current stamp, and
MoE queue bookkeeping still requires the unchanged actual48-layer stamp readback.
Output, fault, accepted-prefix commit and erasure proofs are unchanged.

Source review also fixed missing position declarations in GDN native packets and
QSA selection. Only real GDN position packets are tagged; AB closing status uses
constant zero. Route usage is a separate typed control because target verification
can toggle it; reduction's unrelated zero is not tagged. A source-ready check runs
before replay even though per-layer enqueue is skipped.

**Limits:** only fully sealed typed **single-submission** target recordings qualify.
Position0 initialization stays outside capture. Cold, unsupported or multi-segment
shapes use the existing capture/enqueue path; an upload/submission error is terminal,
never a cache miss. Lab numeric observations, cancellation hooks and grouped-status
experiments retain their actual boundaries. MTP remains untyped and still enqueues
its math. No new GPU owner or content persistence was added.

CPU receipts:52 focused tests pass (11 actual vendor parameter-program tests,
18 GDN lifecycle,16 QSA lifecycle,2 GDN math,4 MTP front,1 admitted metadata test).
Strict selected-core and feature-off release Clippy pass. Immutable release build
and327 offline shader variants plus matrix/physical-pointer validation pass; exact
artifact hash and logs are in the completion guide.
These are not whole-model/GPU/numerical acceptance. Historical full-core regression
failures remain untriaged; do not call the complete regression suite green.

**Next block: independent readiness and queue/event fidelity.** Direct target/MTP
submission is connected; do not repeat typed/staging work or benchmark incomplete
source. Pinned Intel `generate.cpp:3115` enables asynchronous single-device commits;
`verify.cpp:2364–2483` queues its original commit graph, records `commit_done_`, and
joins only for an explicit session consumer/final request boundary. The next target
uses the same queue; the independent drafter reads only original window rows and
its own K/V. Serving calls commit at11089 and drafting at11140, with final wait11169.

The selected `strata_global_controller.rs` now uses the actual scoped native
prefix consumer in `run_global` (`131d29c`), preserving real target predictions and
device carry while original transactions remain open. It forms only accepted
GDN/QSA/PLE candidate state, joins the original fault/I/O proof, then publishes n.
`resident_target_round.rs::replay_prefix` remains the older diagnostic path; it
is no longer selected by this controller. Port pending-commit owner/events next;
do not merely remove the fault read or mark a CPU cursor complete early. Preserve all original roots until
completion, target queue order and draft carry ownership; final/cleanup consumers
must join outstanding commits. Review all-accept and one-token self-commit too.

The backend now adopts real target/draft/copy queues (`b6732c8`) and actual INFO
transfer (`414e2c3`). **Next block: actual independent draft consumers of the owned pending commit.** Preserve the new actual current-source INFO submission
and timeline wait; do not redo it or bake variable Upload handles into math graphs.

`Graph` now has separate actual lane fences and stable original recording flights
(`86f4c35`). `complete_lane_submission` joins the matching lane/recording; existing
`complete_math_submission/complete_submission`, `select_branch_queue`, push/copy
and stream flush still deliberately join prior native math globally;
queue-switch handoffs can also signal behind all primary work. These must change
together for a genuine pending target commit and independent draft. Original
request/model identity, recording resources and INFO pages must stay retained until
their own actual completion. Do not use the existing completed-replay bool as a
submission receipt. `91e1f9e` now exports/join-validates actual pending submissions; the current
engine still joins them immediately. Move that join only with the scoped draft
consumer and real dependent target/session/final/cleanup contract. Do not redo
flight identity or INFO ownership work. The remaining coupling is selecting only
real dependencies, original family-state publication and exact carry consumers.

The scoped native prefix consumer resolves full-window-before-repair publication
and avoids storing self-referential borrowed permits. **It does not resolve async
commit.** Original family transactions and state/carry/INFO roots must remain
owned while commit is pending; original proof/poison/drop obligations cannot be
disarmed just because work was submitted. A scoped post-submit draft consumer
while permits remain on the stack is still one viable design; no model-ready
cursor receipt may escape before the actual dependent join.

Next actual source block: retained INFO, verified-row readiness and independently
scheduled draft, then next-target/session/final/cleanup joins. `4083258` connects
the real dynamic n_keep commit graph/count1 skip; `91e1f9e` exports its original
pending execution. Current engine consumption still immediately joins. Moving that
join without the real data/queue dependencies is not source fidelity. Typed commit count/position fields must be explicit, rather than
inferred from equal old INFO values. Live uniform targets must not be overwritten
by an unrelated flight. Review the single INFO uploader and queue-switch handoffs
together with global stream drains. Already verified carry rows and draft-local
KV can be independent of target state commit, but both data and synchronization
must establish that independence. Global adaptive-cache publication is currently
a completed boundary and must remain ordered before its next target consumer.

Other open work: HOST embedding placement;4096/2048 context/prefill versus source
32768/4096; GDN pair swap versus fixed-copy scheduling; packed-cache physical layout;
startup CJK gather; arbitrary prompt entry; full regression triage; final source,
numerical, owning-erasure and physical recovery acceptance. No live speed trials
before the whole-port gate passes. No installation, service, configuration or model
changes this pass. Required Vashti changes:none.

All older checkpoints below are historical.

**Current checkpoint:** user requested a stop for usage conservation. Standalone
Strata is removed; original model shards, alternate draft and comparison evidence
remain. v143/`f0e088b` passes both-card80-case frozen512 routing operators and
B70 full original CPU13 target vectors/greedies/actual repair using real VRAM
and immutable RAM experts. Warm12 continuations take2.84578s (**4.21677tokens/s**),
with zero explicit demand-expert uploads and1,081,344process disk-read bytes.
This short deterministic placement fixture is **not normal language throughput**
and does not establish30tokens/s. The frozen GPU router has operator acceptance,
not full-model acceptance. An explicit inactive-source-partition kernel candidate
is being checked; its full-model/GPU performance remains pending.

All GPU work stopped with B70 at50,688,000bytes, no NVIDIA compute owners,
normal Kuno admission, all three HTTP200 and unchanged classified GPU faults0.
Installed Kuno/Ollama/Vashti, public API and release selection are unchanged.
Resume from the current RAM-source section below; earlier checkpoints are historical.

## Resumed native source, after standalone Strata teardown

The user authorized the new native optimization pass. Comparison teardown and
retained evidence/draft paths are in [the Strata receipt](kuno-strata-comparison.md).
This supersedes the paused/running comparison states below.

Source integrates bound-target alternate Q2 MTP core `ac8cf41`, exact public
language benchmark/permit `117fa12`, and experimental GPU batch routing `76b0bae`.
The GPU routing experiment activates only after successful original cold CPU13
acceptance and the real one-shot warm reset. It retains actual selected experts,
checks original CPU IDs/order from deferred actual-score audits before all-family
commit, and rejects missing residents or any actual original-ID/order mismatch.
Near ties may execute provisionally; their actual original-ID audit remains
mandatory before commit. Probabilities are explicitly
experimental F32; full original target-vector tolerances remain unchanged. This
is a conditional all-hit warm13 diagnostic, not normal long demand inference.

Construction reserves two complete private owners (+2,228,224 MAIN bytes),
9,175,040 runtime bytes and5 MiB host. Every outer-owner proof and final erasure
includes both. Deferred route owners reserve all48 entries before decoding;
no vector growth may leave un-wiped copies. Independent source review passes.
Integrated pure-direct tests pass849 native tests plus2 auxiliary shader tests
with one existing opt-in ignored. Protected launcher tests pass92; formatting,
whitespace and strict pure-direct all-target Clippy pass. No new GPU speed result
is established by this source checkpoint.

The separate language harness tokenizes the exact pinned Strata ChatML public
code/prose fixtures (35/30 prompt tokens), uses normal demand routing and actual
target-greedy acceptance/repair, and stops at EOS or200 emitted tokens. The first
output is picked inside prefill; timing explicitly distinguishes post-prefill
continuations from total emitted tokens. This is not an independent CPU200
quality comparison. Tokenizer/fixture host owners have their own196 MiB process
reservation; swap/core protection is unchanged. The smaller MTP GPU adapter is
still separate source work; the original Q8 GPU draft remains the control.

Next: freeze this exact source, run both-card routing operators, then full CPU13
cold/warm control/candidate and public language benchmarks. Preserve original
failures and recovery evidence. No driver reset, API, UI, installed production
binary or release change; required Vashti changes:none.

### First public language and device-routing physical checks

Pure-direct `087477d` / v131 (SHA256
`172ab523be256b22679349c9eb09ab4bc265492110e25625e0d5e4f9fd9a948f`)
passes both-card GPU batch router operators:10 numeric and16 sticky-fault cases,
complete owner zero proofs, allocator cleanup and physical memory recovery.
The first whole warm13 candidate rejects with `native batch owning IO`; its
terminal close restores normal services. No accepted GPU warm13 timing exists
from that failed run. Additional static pass markers and exact-tie operator
cases will distinguish conservative pre-rejection from actual-ID mismatch.

The exact public Strata code fixture completes35 prompt tokens and200 emitted
tokens under normal demand routing. Primary199 post-prefill continuations take
47.5798s: **4.18245 tokens/s** (200-emitted count4.20346). Actual draft acceptance
is130/138; five rejected-prefix recomputations pass original full logit/HC vector
checks. There is no independent CPU200 quality reference. Full native/source
closing audits and both-card physical memory recovery pass; all three HTTP
services200 and classified GPU faults unchanged atzero. No output text/IDs saved.

This changing-route run performs15,419 expert uploads/87,301 hits,31,285,683,200
uploaded bytes and9,527,840,768 process disk-read bytes during generation.
Weight reads take17.6865s and router phase9.98992s (includes queued GPU work).
Full timed dual-device readback mapping is10.0868s/5,939 batches. These categories
overlap queued work; do not add them as isolated kernel time. Upstream Strata's
27.3 code/36.0 prose rates remain different-engine comparisons. No30 claim.

The next normal-demand priority is an immutable explicitly host-visible,
non-device-local Vulkan public-weight pool. Strata reads missing packed experts
directly from pinned host RAM; it does not eliminate PCIe traffic. Ordinary
CubeCL STORAGE or adding MAP_WRITE cannot establish the desired placement.
A small Rust allocation/packed-math/physical-heap gate comes before full-mirror
integration. Raw HAL imports also need truthful separate allocation/lifetime
accounting. A frozen144-expert/layer complement is33.67GiB versus46.84GiB for
all experts, before rounding; frozen coverage and a separate public RAM envelope
are required. This is pending source/operator work, not a connected inference
claim. See the [numeric receipt](../apps/kuno/benchmarks/text/receipts/native-strata-optimization-20261009.json).

The revised `b2269ab` / v132 GPU router admits close scores provisionally but
retains exact original CPU IDs/order and probability bounds from actual-score
deferred audits before all-family commit. Both-card13 numeric cases now include
exact internal/boundary ties and signed zero;16 full-owner sticky-fault cases
pass. Whole warm13 then passes full CPU vectors, exact greedy picks, actual
speculative prefix repairs and complete erasure. Warm12 accepted continuations
take1.75601s = **6.83366 tokens/s**, with384 actual device-router calls andzero
expert uploads. This is only a small ordered difference from the older6.64
control, not matchedABBA or normal-demand throughput. Actual target erasure is
1,380 owners/828,866,560bytes. All services/memory recover, classified faultszero.

The router phase is removed, but0.94596s appears in attention/GR enqueue;
whole dual-device map time remains0.74467s/219 readback batches. This suggests
queued compute/backpressure, not pure routing cost. New content-free saturating
submission-wait counters measure actual512-task regulation without changing
queue limits or math. Interpretation awaits a new frozen run; no guessed queue
policy is selected.

The distinct original-packed Q2 CPU oracle writer/parser `e6635a8` passes real
weights:12 scalar executions,2 KV appends,534 projections, zero bit differences
on independent replay. Six complete logit/10,240HC records are6,206,016bytes,
SHA256 `4aef325a3b53b762ab7ebaa663874926ed0dc57ece3446ecad402f1e2e8f3909`.
Source remains BF16 dense/fusedQ2 plus actual borrowed DAS IQ4_XS/Q6 head;
no Strata runtime payload or pruned vocabulary is substituted. GPU parity and
throughput remain pending. CPU fixture execution has swap/core disabled and
no private-runtime claim. The GPU adapter and host-memory placement probe are
parallel isolated source work; neither is installed or promoted here.

### Smaller draft and direct host-source physical follow-up

Original-packed Q2 MTP GPU source `9f66226`, independent probe `bedc06f`
and dispatch connection `f12a252` (v135, SHA256
`a84032473f6f78e9f74ce97678c11f50c8b019317c59d7f66bb48913ca15383b`)
pass both B70 and3060. Each card uses two fresh workspaces: six complete
independent CPU-conditioned vectors, exact greedy choices, actual KV repair,
minimal packets and GPU-follow copies pass. B70 maximum logit/HC errors are
8.49e-5/4.73e-5;3060 7.13e-5/4.39e-5. Each workspace closes42 private owners/
11,698,176bytes and2,539,257,856 public bytes, with reserved MAIN0. This is
standalone draft correctness, not private/production or whole-model speed.
The first two v134 attempts fail before GPU access (helper paths, then dedicated
dispatch admission); both failures and normal recovery are preserved.

Explicit selection `9d99d0d` plus main `f868437` retains absent=original Q8 and
accepts only `KUNO_LAB_MTP_PROFILE=strata-q2`, using `KUNO_TEST_MTP` for the
verified original Q2 file. Both public target controllers identify their selected
profile and validate actual complete close bytes. Fused/ordinary constructors
borrow the actual target graph, never a filename-inferred replacement. Runtime,
private and source-window caps stay unchanged; this is internal lab selection.

Frozen v136 `f868437` / SHA256
`b81e695a6ee22933e93f3db7749487ad3cca5c1060293291a665d403029113d3`
passes Q2 combined cold/warm original CPU13 target vectors, exact greedy picks,
actual repair and complete erasure. Warm12 continuations take2.04545s =
**5.86668 tokens/s**, draft acceptance6/11,480 GPU routers andzero uploads.
This is slower than the earlier Q8 6.83 ordered result; no matched ABBA or speed
improvement is claimed. Actual regulation waits are37.9867ms/74, while
attention/GR enqueue is1.18929s and full dual-device map time978.946ms/281.
The large attention phase includes its own readback barriers; guessed queue-depth
changes are not justified by the regulation counter.

The same real public35-token code fixture completes200 outputs with129/140
drafts accepted, nine original-prefix recomputations (full repair logit/HC error0),
and full closing proofs. Primary199 continuations take58.6855s =
**3.39096 tokens/s**. Load3.78102s, prefill24.9270s, weight reads24.8943s,
15,935 uploads/32,328,524,800bytes and13,107,712,000 process disk-read bytes.
There is no independent CPU200 reference and this is not a matched file-cache
comparison with v131. Q2 preserves memory savings, not an accepted speed gain.
Default Q8 remains available and no private output/text/IDs are saved.

Host-placement source `ceb5315`, connection `740cd91` and v136 pass both-card
original Q2 host-buffer operator acceptance:512 exact decoded values/eight dots,
two normal plusfive full-owner fault fixtures, explicit nonDEVICE_LOCAL heap flags0 and HOST_VISIBLE/COHERENT/CACHED type14, each physical allocation64KiB.
There are no explicit weight uploads; all immutable sources/complete allocation
CPU zero and separate host ledgers pass. HAL counters are unavailable in this
build, so their zero deltas are not physical counter evidence. This only proves
small direct Vulkan host allocation/math. CubeCL registration, the full expert
mirror, portable permanent pinning and throughput remain unaccepted.

Both-card and whole-model trials use swap/core0, recover B70 to50,688,000bytes,
leave zero NVIDIA compute owners, all three HTTP services200 and all classified
GPU faults0. Normal Kuno admission is restored after every trial.
Source checks include94 protected launcher tests, ten focused host-placement
and eight Q2 GPU wire/operator tests plus strict integrated pure-direct Clippy.
Agent default/direct/all-feature profile suites and strict checks pass;
final expanded root integration checks remain a separate gate.

Next highest-potential connections: immutable host-only missing-expert pages
with separate nonreuse CubeCL source ownership, and GPU chronological attention
selection for start+count<=2048 token positions (<=512 complete blocks) after complete score/status validation.
The latter removes the existing two per-layer readbacks without changing selected
IDs; longer contexts keep the exact original CPU fallback. Both are parallel
source work, not a measured full-model optimization yet. No install, public API,
Vashti changes or release promotion occurs here.

**User paused the goal after the overnight pass. Do not start more GPU work
until resumed.** Latest physical model acceptance remains v129/source12f9cc7;
the new live-member kernels and model/operator connection are source-only.
Read the final paused checkpoint below for checks and next physical gates.

Latest model-tested target/MTP source is `12f9cc7` (v129): dedicated request
I/O reuse, pure-direct submission, complete finite fusion and real warm-bank replay pass on
`codex/kuno-native-text`. Selected conditional ideal timing remains v109. The user approved full DASLab IQ3_S, Strata
source study and30 accepted-token/s work. Canonical Vashti source is preserved;
installed services, public API/catalog/configuration and release selection are
unchanged. Required Vashti changes: none.

Read [DASLab notes](kuno-native-daslab.md),
[the numeric/provenance receipt](../apps/kuno/benchmarks/text/receipts/daslab-foundation-20261009.json)
and [resident GPU routing design](kuno-native-resident-router.md).
Both original shards pass full publisher hashes, total83,617,662,656bytes;
there is no merged/duplicate payload. Scalar/SIMD/GPU support covers the
original mixed formats and all48 routed triples. Both GPUs pass independent
original operators and all-nine-profile integer/shared probes. Original full32
GPU comparison on the earlier count1 path fails complete CPU logits at
position13 despite matching greedy IDs; exact cause is unlocated. That historical
failure does not establish the new all-row controller's behavior beyond13.
No relaxed tolerance or forced experts. Only explicit prefix13 is accepted for
the current comparisons; longer-context numerical acceptance remains pending.

Selected public experiment: once-prepared integer MoE plus4096-entry FIFO/
4MiB fully charged request metadata, original output head. Featured v109 SHA256
`1d6b61068d9f491a93404ad941eb4649ac9e407f8f51437babe75f9d035d879f`.
Same-artifact control/prepared ordered pair gives10.511→12.728 accepted
continuations/s (12steps:1,141,680,997→942,769,705ns),17.4% lower latency.
Complete original CPU vectors/greedy IDs, actual trace membership, owning closes
and5760 GPU ranks with zero CPU calls/uploads/physical reads pass. Training,
binding, prefill/preload and later full owning close/recovery are outside
timing; vector/trace checks and request I/O close are inside. This is a short
conditional residency diagnostic, not demand-cache/private/production/full32
quality or30tokens/s acceptance. Arithmetic is explicitly two residual Q8
activation planes, not original F32 bit equality.

`b4c07ed` FIFO now gives31,750 cache hits/2,504 misses instead of zero hits.
Creation+lookup fall to25.6ms in the v109 control (~2.2% of1.14s). Default
256/512 caches and256KiB ceiling stay unchanged. Full Rust metadata capacities
and old/new replacement peaks are charged; opaque driver/RSS is not that bound.

`f5d5e97` prepares activation planes once per stage in the existing complete
CPU bridge owner; all GPU consumers precede its original full overwrite.
`95b31cc` repairs auto-reflected unused binding mismatch caught by first v107
physical test. Both-card v108 nine-profile tests then match old integer ACT/
PART/merge exactly, check full scratch/rank/fault guards and erase90 owners/
15,925,248bytes each. `774f2a9` separately reserves196,608bytes extra prepare
INFO, preserving32KiB runtime margin and ordinary64MiB hard cap. No new model
owner. Aligned Q6 `bcb9bfc` passes both-card independent tests and13-position
combined model gates, but12.686tokens/s gives no gain: keep original head.

Integrated default663 and all-feature666 native tests (one opt-in ignored),
plus both auxiliary suites and strict default/all-feature all-target Clippy
pass. Formatting and whitespace checks pass. All physical trials restore normal admission/services, actual
memory and unchanged classified faults. No GPU reset/unbind/reboot, clocks/
drivers, production installation or release.

Immutable resident GPU routing `cae513f`/`8b734fa` plus token-zero parity fix
`1742100` now passes original DAS13 vectors/greedy/trace,12 owning audits/576
layers and1380 complete-owner closes. Both-card standalone probe `536c98e`
passes49 independent cases and24 full no-publication faults. However, matched
both-segment v111 control/routing takes1,184,871,333→1,420,875,896ns:
10.1277→8.4455tokens/s, a regression. Keep prepared-only v109 selected.
The opt-in charges two full table/audit pages,9,183,232 extra runtime bytes and
4MiB host under an explicit ideal-only128MiB ceiling; ordinary64MiB unchanged.
Parallel complete-owner preflight and prepared integer composition now pass
physical numerical/owner checks. Their performance is variable; keep both
optional and prepared-only v109 selected. See the latest result below.
MTP KV-only append
`2dea140` has focused host acceptance. Target all-row verification now passes
the B70 physical gate below. Whole resident GPU MTP now passes both cards;
controller and accepted-prefix target state repair remain pending. The active goal continues;
if sudo expires and effective protected work cannot continue, save a tested
checkpoint and restore normal GPU/service access.

### Chronological attention and immutable host registration follow-up

Attention leaf`cc4b8aa`, protected/public connection`597039d`, and independent
review pass. Integrated897native+2auxiliary tests (one ignored), focused58QSA,
strict pure-direct all-target Clippy,96 protected launcher controls, fmt/diff pass.
The standalone operator's exact gating accepts default/pure-direct builds only;
an initial parent return-type compile mismatch was fixed before any device use.
Frozenv137 SHA256`7d90d1aa3e9a08501ad938ee8b7c2a9c6c5c3035afa0ebc4ffd201c52c668fe0`
passes both-card21numerical/25fault cases with complete229,376byte owners and
458,752byte reset+close zero proofs. Full CPU13 vectors/exactgreedy/actualrepair
and cold/warm closing gates pass. Warm12/1.65218s=7.26311tokens/s,51readback batches
and zero expert uploads/weight reads/physical read bytes. Previous6.83366/219 is
an ordered earlier artifact, not matchedABBA. Actual GPU submission regulation
now accounts549.4ms/59 waits; removing maps did not remove queued computation.

Sameartifact normal code200 candidate(control absentQSA flag) both pass:
199/95.5603s=2.08246 versus199/98.6940s=2.01633tokens/s. Both perform15,419 uploads
and31,285,683,200bytes,130/138draft acceptance and five original prefix
recomputations with zero recomputed vector difference. Both spend~49s reading
weights, with35,230,142,464/34,949,844,992physical disk bytes. Candidate/control
maps4163/5939; no significant general speed selection or30claim. Timing excludes
initial loading/prefill and includes actual request closing audit. No output
text/token IDs persist and no independent CPU200 quality claim is made.

New immutable nonreuse external-source runtime`dd0db6b` plus one-server/one-stream
restriction`139075e` passes independent source review and checked child ownership
suite133tests/strictClippy. Actual foreign storage-only Buffer identity/class,
complete span/physical RAM/metadata limits, LOAD-only non-atomic shader reflection,
separate mask/cache variants, original ordinary behavior and typed completed
detach are checked. Parent CPU13/language trials do not import foreign pages,
so they do not establish physical CubeCL host consumption. The next tiny both-card
CubeCL operator and full frozen144VRAM+368host sources are source work in progress.
Original source identities and full owning/prefix checks remain mandatory.

Every physical trial restores B70 to50,688,000bytes, zero NVIDIA compute owners,
unpaused healthy admission, allthree HTTP200 and classified faults0. No GPU reset,
installed service/API/catalogue/driver/frontend/release change. Required Vashti
changes:none. Exact pertrial source/artifact/helper/numeric recovery evidence is
in the [optimization receipt](../apps/kuno/benchmarks/text/receipts/native-strata-optimization-20261009.json).

### Original target all-row and actual carry acceptance, 2026-10-09

CPU oracle `68e2c33` preserves all four complete old CPU logit vectors bit-for-bit
and records actual pre-output_hc10,240-wide residuals. Both original83.6GB
shards are independently rehashed against publisher hashes in this run. The
4,137,216-byte public oracle SHA256 is
`18788aee8ccb1defdbba624df3ae914c3cf574ab719c3d2a5f375e4048c67c20`.

Original all-row implementation `d12b96b`, registration `38c0e52`, frozen
default artifact SHA256
`fc4c9f346c67b571585749d5164b391bcdd21ec6d101e28d3fa32e151f172ab7`,
passes partitions [1,1,1], [1,2], [2,1], [3], fresh [3] and each position3
continuation. Every complete248,320-logit/10,240-carry row, exact greedy,
graph identity, actual cursor and owning close passes. Maximum absolute
logit/carry errors are0.000067711/0.000028134 under unchanged0.006+0.0003
bounds. All1,378 owners/826,638,336bytes erase at each close. Added MAIN is
8,388,608bytes; runtime ceiling is explicitly128MiB; ordinary defaults remain.

An initial pre-model cgroup mismatch is preserved. `ef2cf9e` narrowly admits
the exact public target fixture's48GiB host cache/no-swap/core-free launch;
80 launcher controls pass. Default source passes717 native plus two auxiliary
tests (one opt-in ignored), strict all-target release Clippy, fmt and diff.
This is correctness acceptance with ordinary CPU routing, original packed
expert math and F32 activations. Cold durations include uploads and checks;
no speculative speed claim is made. All rows commit; prefix repair and the
native GPU Q8 MTP workspace remain in development. Detailed provenance and
limitations are in the [foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).

At12:08:23UTC, same-boot physical B70 memory is50,688,000bytes, NVIDIA554MiB
desktop-only, no compute process. Admission is resumed with no active/queued
work or recovery latch; all three HTTP services respond200. Classified Xe and
NVIDIA faults remain zero. No reset, reboot, installation, API or Vashti change.

### Parallel routing and prepared combination, 2026-10-09

`416f889` replaces only the complete read-only preflight scan with256-thread
parallel work; original mapper arithmetic/publication and all owning checks
remain. Both cards pass49 independent core cases, all48 frame offsets,24
no-publication fault cases and full16-owner/4,849,664-byte erasure. Extra48 INFO
records charge98,304bytes separately; pre-device dispatch limits are explicit.
Matched v113 F32 control/routing gives11.1205/10.4493tokens/s: improved over the
serial scan, but still slower than its control.

`83d6260` composes sealed routing with the accepted prepared integer pair,
keeping both original72-slot segments active, groups10 and original rank order.
Full CPU_PART is cleared after all consumers and before the original fold;
another98,304bytes INFO is reserved independently. The integer routing runtime
addition is9,379,840bytes, under its explicit ideal-only128MiB ceiling.

Frozen v114 source`6a32b11`, SHA256
`62e082d476cb543907cc8ddb178d1a05d16a5523454ffad0fc90caedc9f47f77`, features
`direct-device,cpu-fma,cpu-work-stealing`, passes an ordered ABBA comparison:

| Prepared-only / routing order | Twelve steps, ns | Accepted continuations/s |
| --- | ---: | ---: |
| Prepared-only A1 | 941,409,513 | 12.7468 |
| Routing + prepared B1 | 821,556,179 | 14.6064 |
| Routing + prepared B2 | 1,192,232,580 | 10.0652 |
| Prepared-only A2 | 932,615,706 | 12.8670 |

All four check every original13-position vector, exact greedy, real expert trace,
5760GPU/0CPU ranks, no weight uploads and complete original closes. Each routing
run additionally accepts12 owning audits/576layers. B1 observes114,688 physical
read bytes; other three observe zero. Per-step clocks retain the original
aggregate timer: no discarded initial dispatch or redefined hot result. The
repeat does not establish a consistent routing speed gain or30tokens/s.

Root checks at`6a32b11`:695 default/698 featured native tests, one opt-in ignored,
both auxiliary suites, strict release all-target default/featured Clippy,
formatting and whitespace pass. New all-row target/carry, public CPU oracle and
full-resident Q8 MTP kernels are being developed in separate worktrees and are
not covered by this result. Accepted-prefix publication/controller are absent.
No API/Vashti/installed-service/driver/reset/release change. Required Vashti
changes:none. Exact receipts and source/artifact hashes are in the DASLab
foundation aggregate.

---

# Native text checkpoint, 2026-10-08

**Latest checkpoint: Q3 correct-expert replay accepted; work paused at the
user's request.** Integrated engine source is `9218638`, on
`codex/kuno-native-text`, in the dedicated Kuno worktree. Installed services,
public API/catalog, Vashti and releases are unchanged. Preserve the canonical
Vashti checkout and its independent coordination work.

The actual MTP GPU projection probe now passes on both cards (`8cba745`);
nonprojection graph operations remain CPU and private whole-stack owning
acceptance/speculative speed remain pending. Independent full72+72 segmented
mixed-IQ MoE operators also pass on both cards (`f116d09`/`3571bcb`). These
operator results do not establish whole-model or MTP speed.

Q3 empty-slot filling is implemented (`b739c32`), with opt-in skipping of wholly
unused bank segments (`397999c`/`c6d2088`). Same-input expert and router diagnostics
locate the preserved full32 failure: tiny arithmetic differences create a tied
router score at position23/layer17, change one selected expert, then amplify.
The CPU top-k run on GPU scores agrees with all GPU selected ranks at all48
layers. No decoder/eviction defect was demonstrated; tolerance is unchanged.
The explicit23-position prefix ends **before** that known discontinuity and
verifies complete248320-wide original CPU logits and exact greedy IDs.

Fresh training → full close → fresh exact original preload → replay (`9218638`)
passes22 decode continuations with10,560 GPU expert calls, zero CPU calls,
zero expert/weight uploads and zero observed physical read bytes. The3102-expert
union fits38–135 of144 slots per layer. Same-artifact one-pair timings are
2311859249ns (**9.516tokens/s**) without unused-segment skipping and2174240700ns
(**10.118tokens/s**) with it. Preload uploads6459852800 original bytes outside
the timer. This is a short conditional residency result, not normal demand-cache
throughput or an absolute hardware limit. Prefix/full32 status and all timing
exclusions are explicit in the receipt; full32 failure is retained. Initial
fill runs are slower and their physical I/O differs, so their timing difference
is not an isolated skip speedup. Keep both candidates opt-in.

Host checks pass591 native tests (one ignored opt-in), two one-test shader
suites, release all-target Clippy with default and all five features, formatting
and whitespace checks. All four v90 physical prefix runs pass the unchanged
five original818413568-byte/1378-buffer owning closes; the two ideal probes add
independent training/replay closes, full-vector/membership/counter checks.

Next candidate is a **leaf-only, uncompiled and untested** four-row subgroup
MoE kernel in isolated commit `6d12ddd`, branch `codex/kuno-q3-subgroup-moe`,
base `9218638`. It is not cherry-picked here. Next resume: connect its opt-in
and device support, extend independent segmented-IQ probe/fault/owning checks,
then run host checks and protected both-card operator/full-vector performance
acceptance. Do not infer acceptance from source or formatting.

At the pause, no lab jobs remain; public admission is resumed with no active
or queued work/latch. Kuno11434/Ollama11435/Vashti7771 returnHTTP200. Same-boot
B70 memory is50688000bytes, NVIDIA has no compute process, and classified Xe/
NVIDIA faults remain zero. No GPU reset/unbind/reboot, driver/clock change or
installation occurred. GPUs and normal application access are available.
See the [exact plan](kuno-native-text-mtp-plan.md) and
[source/artifact/numeric receipt](../apps/kuno/benchmarks/text/receipts/unsloth-q3-foundation-20261008.json).

### Previous ideal-residency replay checkpoint (historical)

**Latest checkpoint: exact short ideal-residency replay accepted.**
Source `aef04f24254b6019fa3a61e1275e2bf38eeee98c` integrates trace/preload
`d45dc4b` and runner `366d847`. Frozen v76 SHA:
`7e18edff2d5413b48b834f391f6637497bf5d157acd3507542afa2602887c8a7`.
Protected run `cpu-144-public-oracle-v76` passes every unchanged original gate,
five original owner closes and two additional full replay/training closes.

Seven correct decode steps take596125512ns: **11.74249tokens/s** with3360 GPU
expert calls, zero CPU calls, zero expert/weight uploads and zero observed
physical disk reads/writes. All eight output IDs match the original prefix.
The transient union contains1826 experts,18–63 per layer; preloading uploads
1420390400 original bytes outside the timer. There are three major faults;
zero physical I/O does not mean zero faults. No wrong-expert substitution.

This is a single short sample of the current ordinary all-GPU expert path
under ideal residency, **not an absolute hardware maximum**. Loading, prefill,
training and preload are excluded; activation transfers, router fences,
numerical/owning checks and unfused single-token kernels remain. Original
127-step decode is12994ms, but differs in length/routes and is not a matched
seven-step control. Perfect residency alone does not establish30–50tokens/s.
Compare remaining GPU execution/synchronization and multi-token work before
treating the cache as the only bottleneck.

Checks pass525 native tests, two one-test shader suites,181 Python controls,
strict release all-target Clippy, formatting and whitespace checks. See the
[exact numeric receipt](../apps/kuno/benchmarks/text/receipts/next-public-oracle-20261008.json),
[internal replay contract](kuno-native-text-count1-hybrid-plan.md#exact-short-resident-replay-diagnostic-2026-10-08)
and [original fast-report comparison](kuno-native-text-disk-access.md#bidirectional-cache-transfer-and-published-fast-reports).
The user suggests trying a smaller quant; the full83.6GB IQ3_S artifact is
assessed in the disk research. It needs split-file loading and seven missing
mixed quantization formats across scalar/GPU/CPU/PLE paths plus a new reference
fixture. No smaller quant was downloaded or run. Preserve native Rust execution
and the original Q4 acceptance while adding support.
Do not conflate pruned Coder models or MTP throughput with our original Q4 run.

Normal admission is resumed; all lab jobs exited. Same-boot post-run state at
2026-10-09T00:54:07.706692+00:00 has B70 physical50688000bytes, no NVIDIA compute
process, zero classified Xe/NVIDIA faults andHTTP200 from all three services.
Swap/core dumps were disabled during the benchmark. No reset/unbind/reboot,
driver/clock, installed runtime, API, release or Vashti change. Required Vashti
changes: none. Both integrated review worktrees are archived; receipt/quant
assessment commit is `3601124`.

### Previous adaptive settling checkpoint (historical)

**Longer adaptive settling and disk research complete.**
Integrated laboratory source is `8d25a9d`; corrected token-time decay and
original existing-slot PUBLIC writes are `cbfa8d1`, long diagnostic source is
`569c8e2`, and runner acceptance is `8d25a9d`. All144 slots remain eligible;
the earlier full-cache decay implementation below is superseded.

Complete unchanged original acceptance passes for v74 fixed/copy/direct modes.
Their127-step decode rates are10.9238/10.4829/10.5202tokens/s with zero physical
reads or writes. Direct writes have no demonstrated total speed advantage.
Two separate v75 1024-step diagnostics also pass every original gate, their
extra owning closes, and exact equality of all1024 greedy outputs. Last128-step
windows reach7.475 fixed versus9.405 adaptive tokens/s; adaptive VRAM expert hits
reach89.18%. Across the whole continuation, rates are7.0137 versus8.0989tokens/s.
These are repeated public teacher-forced inputs with growing context. Adaptive
follows fixed; file-cache warming remains an order confound, not a proved causal
speed gain or general10-token/s acceptance.

Longer runs read20.922/20.430GiB from storage, tapering sharply across windows;
there are zero physical writes. Expected lazy n-gram row access and mapped
expert first touches must be distinguished from eviction. There is no current
proof of hot-page thrashing or a48GiB cgroup-induced shortage. See
[disk research and next attribution measurement](kuno-native-text-disk-access.md)
and [the verified numeric receipt](../apps/kuno/benchmarks/text/receipts/next-adaptive-steady-20261008.json).

Source gates pass519 native tests, two one-test shader suites,69 targeted Python
controls, strict release all-target Clippy and formatting. Frozen v75 artifact:
`ecdb3a5f231d1a81b91398ebcc39378a659d96fce2d3358682da9b78f312a882`.
The exact short trace/replay ceiling remains pending: isolated `990c52d` adds
trace/preload seams, `0740b6b` adds runner controls (79 targeted/181 Python tests
pass). Root must write orchestration/main, integrate, compile and validate before
claiming any oracle measurement. Preserve those two clean review worktrees.

No lab benchmark remains running. Normal admission is resumed, no recovery is
required, B70 physical usage is50688000bytes, NVIDIA has no compute process,
classified Xe/NVIDIA faults are zero, and Kuno/Ollama/Vashti returnHTTP200.
Installed binaries/settings and API remain unchanged; required Vashti changes:
none. Keep fixed placement selected until the adaptive order/working-set concern
and broader quality/context coverage are resolved.

### Historical reasoning-mode checkpoint (superseded)

The preceding pause for reasoning-mode switch used full-cache source
`3d67dbdd3c331b779779406a0fd177d3f459a4cd` (not device accepted yet).
The user rejected permanent 128-slot pins; all 144 slots now use prefill and
accepted decode frequency/recency, with 16-token frequency decay, current-token
protection, grouped four-expert fills and unchanged original acceptance gates.
Tests pass 513 native, two one-test shader suites, 52 Python controls and strict
release all-target Clippy. Frozen v73 artifact is
`20697ff60e7068dbd32f8c1bf1bf1c35c34c7c769847c27cc075e61e5aecd727`,
build `next-offload-build-20261008-v73-adaptive-full`. GPU runs are pending.

Earlier 16-slot v71 passes all original gates but is slower: 12825 ms/127 steps
versus matched fixed-cache 11392 ms (9.9025 versus11.1482 tokens/s), 504 promotions,
14100 CPU misses versus16411. Grouped v72 reduces that to12085 ms (10.509 tokens/s)
and eliminates warm disk reads; it still does not beat the fixed cache.
Raw accepted run names are `cpu-144-adaptive-long-v71`,
`cpu-144-adaptive-off-long-v71`, `cpu-144-adaptive-long-v72-grouped`, under
`/media/jmicko/faststore/kuno-development/next-offload-device-20261008`.
Rejected v70 failed setup because partially populated banks were incorrectly
assumed full; no acceptance record exists. Exact policy history is in the
hybrid plan. Earlier selected-source statements below are historical.

No benchmark remains running. Normal admission is resumed (HTTP200,
paused=false, recovery_required=false). Installed inference binaries/settings,
public API and Vashti remain unchanged. Resume with protected matched v73
adaptive/fixed complete original fixtures; do not call full-cache speed proven.


Latest engine source is **3e9f0e4891c75a09ff40eddd47f167e7ae47bef9**.
The complementary stationary NVIDIA expert bank is implemented and passes the
complete original model checks. It is **excluded from the selected path**:
matched same-binary decode is **11.0406 tokens/s** with original B70/CPU placement
and **8.2586** with NVIDIA experts. More residency saves CPU work but additional
completion/readback and buffer reuse overhead outweigh that saving here.

Keep original intel144/cap10, GPU limit10, float/CPU FMA and binding256, with
secondary experts off. The approved10-token/s reference is met on this fixed
public fixture. No new Ollama baseline was run; these results do not establish
general prompt performance. See **Stationary secondary model acceptance** below
for source, artifacts, matched timings and limitations. Earlier pending and
selected-source statements are historical.

Normal inference is restored and all lab jobs exited. This remains a standalone
native laboratory. Installed binaries/settings, public API, Vashti and release
artifacts are unchanged. Required Vashti changes: none.

Backend owner:
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, branch
`codex/kuno-native-text`; preserve the canonical Vashti checkout and its work.

## Active overnight offload run, 2026-10-08

The current accepted best is7.1654 tokens/s (144 slots, original full fixture);
10 tokens/s remains unmet. The preceding block established3.5006 tokens/s. Source
`4cf40f3` adds specialized F16C original CPU packed loops with120 core tests,
3 ignored and strict Clippy. Its normal frozen direct-device artifact
`e12a2b2788eabcb2f5d4c6b0588ba7cb18bd531bba4ae8c61ca676fb650a0179`
passes the complete B70/intel96 fixed75/128 model and all original lifecycle/
owning gates: all128 IDs,127 steps in37118ms (3.4215 tokens/s),61491ms prefill.
It establishes correctness but no whole-model speed improvement. Results are
under `kuno-development/next-offload-device-20261008/f16c96-long-v1.*`.

The next candidate expands explicit protected CPU caps to canonical0..10 so
misses beyond the cap can enter the GPU cache during decode. Cap10, which sends
all misses to CPU, leaves its prefilling bank stationary. No default, model
geometry, rank arithmetic, private/source budget or publication/erasure gate
changes. See the
[cache-admission contract](kuno-native-text-count1-hybrid-plan.md#bounded-dynamic-cache-admission-experiment-2026-10-08).
482 direct native tests plus two offline shader suites and23 Python controls
pass. Mixed-cap physical correctness and timing now pass (source5ed5c8c,
protected-launch correction3f72acb, artifact8733aaec…): cap5/intel96 completes
all128 IDs with37595ms decode (3.3781 tokens/s),50682ms prefill,38728 warm
GPU hits and1703 warm GPU expert uploads (5632921600 bytes), and20529 CPU
misses. Warm weight reads add3923ms. It is not a speed win over cap10.

Source d9d5193 ports the GPU count1 expert projections to packed-vector/
subgroup math on original banks and owning spans. Source6253828 batches the
seven original MoE scratch clears via ordered native buffer clears; the same
completed fences, transient-I/O errors, and whole-owning zero audits remain.
Both full75/128 B70/intel96 runs pass all128 IDs plus fresh/poison/cancellation/
full-owning gates. Vector-only decode36821ms (3.4491 tokens/s), prefill48214ms;
plus clears decode37318ms (3.4032 tokens/s), prefill52038ms. These single ordered
samples establish correctness, not a speed improvement. Frozen artifacts are
d99bcffb… and4ffe5a7b… under next-offload-build-20261008-v3/v4; receipts live in
the same next-offload-device directory.484 native tests plus two shader suites
and strict Clippy pass. The updated runtime patch SHA is
0f4137174f5de0e53caf1e30fc517c6e9822ca1861ca628080a78045e707b0d9.
Physical Intel memory returns50688000 bytes after each completed fixture.

The next CPU candidate is a separate compiled `cpu-fma` feature, explicitly
selected in the protected runner. It keeps original Q4_K/Q5_0/Q8_0 weights,
finite/cancel/ownership guards, and call-local wiping scratch, while using
FMA and32 accumulated lanes. Its reduction round points differ; ordinary
Avx2/Portable and installed inference remain unchanged. Independent f64 row
checks use an L1 forward-error bound; complete synthetic expert checks use an
infinity-norm error bound because cancelled elements can approach zero.
The original128 model IDs remain a mandatory, unchanged acceptance gate.
122 core tests pass with3 ignored. Source56340c2, frozen artifact4fa9fbfd…
passes all128 IDs and original owning/lifecycle gates, but decode37323ms
(3.4027 tokens/s), prefill54966ms does not improve the selected control.

Source98df150 adds numeric-only warm decode process counters. Artifact
a2724725… passes the same complete fixture with36790ms decode (3.4520 tokens/s)
and53251ms prefill. During warm decode it incurs61267 major faults and reads
11601731584 bytes from disk, despite zero warm GPU expert uploads. CPU cycle
profiles alone did not expose this stall. Counters report process CPU ticks,
not inferred GPU times; their scope starts after prefill and before decode.

The next explicitly pinned cap10/mapped experiment advises complete original
file pages for weights already retained on GPU at a proved completed boundary.
It uses Linux64 POSIX_FADV_DONTNEED on the original read-only descriptor,
leaves all mappings and borrowed tensor bytes valid, and never drops arbitrary
host caches. Successfully advised bytes are not a claim of actual reclamation.
Advice cost is separately reported outside decode time. All128 original IDs,
full private budgets, fences, erasure and physical recovery remain required.
123 core tests (3 ignored),485 native tests plus two shader suites,25 Python
controls and strict all-target Clippy pass. Source6e74f91 artifact861c20cc… passes the full75/128 fixture with37142ms
decode (3.4193 tokens/s),50760ms prefill. Advice covers18080460800 source bytes
in549ms, but warm decode still reads11718238208 bytes and incurs60783 major
faults: no effective reduction or speed win. Recovery returns50688000B.

The next candidate is explicit `intel128`: two original64-slot shards, exact
24GiB bank charge,5GiB static cache and31GiB reserved ceiling. Existing profiles
retain their26/10GiB ceiling. The original source metadata, all-ten-rank merge,
private inventories and owning proof gates remain. Source83002c9 first rejects at the original512KiB host ledger guard before
allocation. Correction6925024 accounts640KiB separately within the unchanged
12MiB parent envelope. Artifact4f777e06… passes all128 IDs and original owning/
lifecycle gates:35075ms decode (3.6208 tokens/s),49375ms prefill,19238 CPU
misses and41722 GPU hits. Warm decode still reads11150995456 bytes and incurs
63966 major faults. The modest single-sample gain does not meet10 tokens/s or
establish significance. B70 recovers50688000B.486 native tests plus two shader
suites,25 Python controls and strict all-target Clippy pass.

The next explicit candidate uses original full-owner native clears across
GDN/QSA/GR and coordinator scratch, batching adjacent clears within16 owners/
64MiB. Larger public spans keep the original shader path. Final arena erasure,
parent fences and every original full-owning audit remain unchanged. Host gates
pass486 native tests plus two shader suites,26 Python controls and strictClippy;
Physical source d8c1993/artifact74a696a7… passes all original output and
owning/lifecycle gates,35849ms decode (3.5427 tokens/s),50405ms prefill.40767
actual native clear calls are reported. Nested host timings shift among enqueue
and wait phases but total time does not improve. No selected speed win.

The next candidate combines register integer reductions and vector Q5 unpacking
with explicit three-pass residual Q8 CPU activations on original weights. This
is approximate and separately pinned; the original128-ID gate is unchanged.
Actual private metadata is charged inside the original1MiB CPU envelope.124
core tests (3 ignored),486 native tests plus two shader suites,27 Python controls
and strict all-target Clippy pass. Source662a4bf/artifactf48737d4 passes all128
IDs and original owning/fresh/lifecycle gates:36182ms decode (3.5100 tokens/s),
51308ms prefill. Warm reads11150274560 bytes and64084 major faults remain.

Selected CPU source read-ahead (a045454, artifact68052dcf) passes the same
full gates with34944ms decode (3.6344 tokens/s),57714 advice calls for19238
CPU misses. Warm reads10837012480 bytes and58098 major faults remain. This
single sample is only a small change, far below10 tokens/s.126 core tests
(3 ignored),486 native tests plus two shader suites,28 Python controls and
strict all-target Clippy pass. Physical memory returns50688000 bytes.

The next explicit `--pageout-gpu-sources` candidate also asks Linux to reclaim
mapped immutable original model pages after their GPU copies are ready. Only
complete validated GPU source pages are addressed at the original completed
cap10 boundary. Linux COLD/PAGEOUT preserve read-only file contents and borrowed
slices; anonymous/private inference memory is never addressed. This remains
best-effort advice, with no actual reclaimed-byte credit. Mandatory original
owning/output gates remain. Physical source09dcdd9/artifact7a139c08 passes all original gates with35169ms
decode (3.6111 tokens/s),51648ms prefill; readbytes10623344640 remain. The same
frozen binary under the protected root cgroup passes all gates with34984ms
(3.6302 tokens/s),51385ms prefill,10556477440 readbytes. Original blob belongs
to UID997; Linux can silently filter nonowner file PAGEOUT, but temporary root
execution also produces no useful improvement. No permission/model/service
change. Physical B70 recovers50688000 bytes.

Source2bd4f70/artifactcf2a9eee expands to144 original experts in two72-slot
shards within the same24GiB bank/5GiB cache/31GiB device ceiling. All128 IDs,
full original owning and fresh/lifecycle gates pass with34343ms decode
(3.6980 tokens/s),50668ms prefill,16411 CPU misses. Readbytes10352029696 and
58553 major faults remain.487 native tests plus two shader suites,29 Python
controls and strict all-target Clippy pass. This remains below10 tokens/s.

Source de6d086 adds explicit4/8/16 physical worker calibration and separate
compiled two-pass residual Q8.127 core tests (3 ignored),487 native tests plus
both shader suites and30 Python controls pass. The first8-worker launch
(artifact00215ff3) rejects at the original fixed16-worker constructor guard,
before model inference; no accepted speed receipt. Recovery returns50688000B.
The correction retains a typed4/8/16 maximum, requires pooled4/8 and independently
observes actual constructed geometry. The corrected source5295faf/artifactcbe79139 passes all128 IDs and original
owning/fresh/lifecycle gates with8 actual workers:35199ms decode
(3.6081 tokens/s),51175ms prefill. User/system CPU ticks drop to3237/751,
but readbytes10334400512 remain and wall speed does not improve. Sixteen workers
remain the control. Two-pass physical acceptance is pending.

Source ff8d92b adds the original160-slot/72+72+16 profile within27GiB bank,
3GiB static cache and the unchanged31GiB ceiling.490 native tests and both
shader suites,30 Python controls and strict Clippy pass. Frozen artifact
006771c6… rejects its first prefill at a physical owning check before any
accepted timing; no receipt or speed claim. GPU memory recovers50688000B.
Inspection found the actual resident MoE launch proof still assumed two shards.
The correction derives and independently validates all three actual projection
inventories from the checked public table, without weakening full-page/alias/
geometry checks. A host regression rejects missing/extra shards in each
projection.491 native tests plus both shader suites pass. The fixed-public
terminal cleanup now preserves the original content-free typed error instead
of replacing every failure with an opaque chunk rejection; panic payloads
remain suppressed. Corrected sourcec3ff595/artifact8dfbc305 passes all128 IDs and original fresh,
full-owning and cancellation gates, but decode234567ms (0.5414 tokens/s),
prefill61257ms. Three-GiB dense cache thrashes: warm weight reuploads386165831680B,
read59030ms/upload95104ms. It is excluded from speed selection; the accepted
144-slot/5GiB control remains selected. Physical B70 returns50688000B.

A further internal CPU overlap candidate schedules its coordination task in
Rayon's existing pinned pool with `in_place_scope`, retaining the GPU parent
on its original caller thread instead of creating an OS thread every layer.
Original32KiB orchestration /1MiB CPU reservations and full-capacity wiping
metadata/Partials ownership remain. The existing genuine overlap test proves
parent-thread identity and joined completion through all original failures.
127 core tests (3 ignored),491 native tests plus both shader suites and strict
all-target Clippy pass with compiled two-pass arithmetic. Source17101d1/artifact0c908311 passes the full original75/128 model and owning/
fresh/lifecycle gates with two actual residual passes and16 workers:34888ms
decode (3.6402 tokens/s),59177ms prefill. Readbytes10927468544 and62447 major
faults remain; no useful speed win. B70 returns50688000B.
Original144-slot GPU profiling from frozen c3ff595/FMA artifact rejects during
prefill because profiling exhausts the original private-I/O budget; no timing
or acceptance claim, and no budget bypass. Physical memory returns50688000B.
A further buffered original CPU job candidate keeps mapped GPU sources, but
uses bounded call-local original packed vectors for actual CPU positional reads.
127 core tests (3 ignored),491 native tests plus both shader suites,31 Python
controls and strict all-target Clippy pass; physical acceptance now passes: source14a1c92/artifact1864cd13 completes all128 IDs,
original owning/fresh/lifecycle gates with51413ms decode (2.4702 tokens/s),
50962ms prefill and49233 actual original CPU reads. Major faults drop to361,
but readbytes12011974656 and copy/allocation costs worsen wall time. Mapped
reads remain selected. B70 returns50688000B.

The next diagnostic pauses profiling through initialization/prefill, and arms
only the first following native decode with a drop guard. No I/O/admission cap
or original owner proof is changed. Original full-lifetime profiling is rejected
and excluded from timing selection. Sourcee8270af passes491 native tests, both shader suites,32 Python controls,
50 runtime host tests plus two isolated logger configurations and strict Clippy.
The first-following-step diagnostic still rejects with deferred router I/O
under the unchanged cap; same frozen49d016ca artifact with original per-router
cleanup passes all short fixture/owning/fresh/cancel gates. It reports one
profiled step, 64.183445ms total completed GPU kernels (generic custom kernels
33.254779ms, clears6.821887ms), not a clean speed result. Profiling changes
scheduling and is excluded from speed selection. GPU recovers50688000B.

The next explicit source-pageout experiment replaces ineffective COLD/PAGEOUT
with clean original file PTE release before descriptor cache advice. No private
anonymous allocation, source/file contents, inode permission or original proof
budget is changed. A live borrowed slice across full interior pages must refault
identical bytes at the same address; the receipt distinguishes this method from
historical pageout.127 core tests (3 ignored),491 native tests plus both shader
suites,32 Python controls and strict all-target Clippy pass. Physical
source02e1a9b/artifactd53c7a5c passes all128 IDs and original owning/fresh/
lifecycle gates with29994ms decode (4.2342 tokens/s),19879ms prefill. Warm
readbytes fall to5560762368, majorfaults31597, user/systemticks4493/981.
Advice covers25519091712 bytes in1195ms, not an actual freed-byte receipt.
A single ordered sample improves the observed rate; matched significance is
not claimed. Physical B70 returns50688000B.

Next, aggregate successful-call CPU prepare/gate-up/down/total host intervals
are explicitly enabled only in the fixed-public native lab. This isolates
CPU math/read wait from the broader host enqueue interval; no private trace
or installed/API change.128 core tests (3 ignored),491 native tests and both
shader suites,33 Python controls and strict all-target Clippy pass. Physical
source9c4ddef/artifact412f3940 passes all original full-model/owning/lifecycle
gates,24659ms decode (5.1502 tokens/s). CPU prepare347049742ns, gate/up
1988165674ns, down1076907721ns, total3484611488ns over127 steps. Warm
readbytes17580032, majorfaults572; the CPU set is now largely warm. Native
CPU experts consume27.44ms/token and host total194.17ms/token, so remaining
work focuses on execution/fences rather than more CPU math tuning.
The existing fixed-public scheduling diagnostic measures128 broadcasts in
1654/1813us versus scoped60173/57425us; this is not model speed.

The next explicit shared vector projection fuses the original finite fault
coverage into unchanged arithmetic and last-tile owner reads, reducing one
separate kernel per projection. Source checks pass492 native tests, both offline
shader suites,33 Python controls and strict all-target Clippy; physical
acceptance source5f6527f/artifactbcd0e49b passes all128 IDs and original
owning/fresh/lifecycle gates,23457ms decode (5.4142 tokens/s),18688ms
prefill. Warm uploaded bytes0, diskread67944448 bytes. CPU total3701666222ns
over127 steps. Physical memory returns50688000B and classified kernel
fault counters remain0. This is one ordered warm sample, not a paired
comparison. The warm original native-clear control on the exactv20
artifact passes all128 IDs and owning/lifecycle gates, but4.9075 tokens/s
with16384 warm read bytes does not improve the5.1502 sample. Native clears
remain unselected.

The user granted exclusive GPUs. Installed Kuno had an idle managed worker
holding15978409984 B70 bytes with no active request; a graceful Kuno restart
released it and physical VRAM returned50688000. There was no GPU reset/unbind
or reboot. Kuno admission is currently paused for this run, active=false,
waiting0, recovery_required=false. Normal admission must be restored at the
next checkpoint. Source isolation is the same owning worktree; do not touch
the canonical Vashti task. No installed binary/configuration/API change.

## Complete Next shared execution acceptance, 2026-10-08

The user authorized transferring the resident Ornith improvements into complete
Flash-Next inference. This block reaches **3.5006 decode tokens/s** on the B70,
versus the earlier1.691 artifact. The approximately10-token/s target remains
unmet. This is an observed laboratory improvement, not a matched Ollama or
paired historical-artifact comparison. No Ollama baseline was repeated.

Source commits `775d985`, `fc1e89e`, `d351bd6`, `62a52ba` and `1a80d5b` connect
shared packed dense/router kernels, vector recurrent attention, batched private
parameter uploads, the exact original CPU reduction in SIMD registers, and
explicit direct caller device dispatch. Original mapped Ollama weights, routed
GPU/CPU experts, rank merge, full private budgets and owning erasure remain.
The uploaded INFO records have distinct slots and ordered copy/task submission;
the direct device handle retains the original serialized WGPU server and fences.
These remain opt-in fixed-public lab modes, outside installed dependencies.

| Complete75/128 fixture | Prefill ms | Following127 decode ms | Decode tokens/s |
| --- | ---: | ---: | ---: |
| Shared dense/router | 60243 | 83606 | 1.5190 |
| Plus vector GDN | 53080 | 81255 | 1.5630 |
| Plus batched private parameters | 49319 | 44935 | 2.8263 |
| Exact CPU finish,96-slot banks | 54263 | 41974 | 3.0257 |
| Direct device dispatch, standard banks | 48254 | 37941 | 3.3473 |
| Direct device dispatch,96-slot banks | 65978 | 36280 | 3.5006 |

Each row is one ordered compiler-idle sample with its actual source/artifact;
do not infer statistical significance or independent component percentages.
The final96-slot choice completes29205 CPU misses and31755 GPU hits, versus
36916/24044 with standard banks. It improves decode modestly but worsens
prefill. Model bind3272ms follows file warming, not a matched cold-load test.
Seed424242, temperature0 and all128 original native output IDs are unchanged
for every accepted long run. The separate CPU-Q8 trial fails the original-ID
gate and is excluded; its full receipt and failed acceptance are retained.

Selected fixed-public artifact: source
`1a80d5bfc4b01e5984c10daf5c7002869aa26895`, Cargo feature `direct-device`, SHA256
`fe49069954829b2f8514b889b66967564937a67fdaabffe60a58418d437b8ec2`, frozen in
`/media/jmicko/faststore/kuno-development/next-shared-build-20261008-v5-inline`.
Use explicit inline dispatch/intel96, cap10, pooled CPU workers, overlap,
mapped weights, large weight cache, compact INFO, task batching, GPU router,
deferred router IO, retained metadata, fast GPU, shared kernels/GDN and batched
parameter upload. Exact requested/observed fields, hashes, source identities,
all13 samples, tests and postflight are in the
[numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-shared-speed-20261008.json).
The [count1 contract](kuno-native-text-count1-hybrid-plan.md#complete-model-shared-execution-acceptance-2026-10-08)
records reproduction and remaining gates.

119 core tests pass with3 ignored; normal/direct native suites pass479/481
tests plus two separate offline shader suites each.50 imported runtime tests,
22 hybrid Python controls, strict all-target Clippy, formatting and provenance
checks pass. All accepted physical samples pass original row/router/GDN probes,
fresh continuation, the existing cancellation/poison recovery checks and full
1378-owner/818413568-byte MAIN zero audits, source0, without context retirement.
Actual48-GiB cgroups enforce swap0/core0. No real prompt, output or route trace
is stored. These tests do not establish arbitrary-request quality, full-logit
equivalence, long-context/soak or production worker acceptance.

The next speed work should address CPU expert execution and remaining
fine-grained GPU handoffs. A12-second public CPU profile samples52.30% of cycles
in CPU AVX experts and13.30% in the old threaded device server. These are CPU
cycle shares, not wall-time shares; its profiled model run is excluded from
clean speed selection. Final cgroup snapshots show major faults/page-cache
refaults during decode with no swap or OOM, but IO counters are unavailable and
I/O causality is unproven. GPU phase counters are nested host intervals, not
isolated kernel timestamps. Mixed-card text execution and speculative expert
prefetch/MTP remain future experiments.

At postflight all lab jobs exited and Kuno admission is resumed: active=false,
waiting0, paused=false, recovery_required=false. Kuno11434/direct Ollama11435
return HTTP200 with no resident model; Vashti7771 returns HTTP200. Intel
physical VRAM recovers to50688000 bytes; NVIDIA541MiB includes the desktop.
The same boot has zero classified Xe wedge/timeout/GT-reset/memory-fault
messages. No installed binary/configuration, API/catalog/default inference,
Vashti or release change occurred. Required Vashti changes: none. The broader
goal remains paused and incomplete; this directly authorized trial is finished.

The final checkpoint also formats two earlier imported runtime functions and
regenerates provenance; those whitespace changes do not alter the benchmarked
artifact or behavior. The receipt distinguishes its original runtime patch
hash from the checkpoint patch hash. Imported runtime tests/Clippy are rerun
after formatting.

The resident/operator checkpoint below is historical. Its statements that
shared Next arithmetic is not connected and whole-model speed remains1.691
are superseded by this acceptance.

## Resident speed acceptance, 2026-10-08

Ornith now meets the approximately80-token/s resident target:85.35/85.38 for
two fresh75/128 sequences and82.6–83.0 on two EOS-finished public prompts.
All128 IDs match the native scalar reference. The shared-kernel refactor
recheck measures86.00/85.27 with the same IDs. See
[the acceptance contract](kuno-native-ornith.md#resident-speed-acceptance-2026-10-08)
and [aggregate receipt](../apps/kuno/benchmarks/text/receipts/ornith-resident-speed-20261008.json)
for exact source, artifacts, modes, counters and numerical limits.

The main measured fix is cached parallel expert selection (instrumented router
42.064→0.419ms across40 layers), combined with vector packed Q4/Q6 calculations.
Explicit unchecked runtime bounds add roughly9% to the fixed-geometry lab;
bounds-checked vector/router still measures77.5–78.2. Defaults are unchanged.
Teacher-forced128-token comparison to Ollama gives121 top1 and7 rank2 matches,
with identical scalar/fast choices. Free-running Ollama parity is not claimed.

At that earlier checkpoint, backend source through `ffe1cf0` shares these arithmetic kernels and
validates original Flash-Next Q4/Q5/Q8/F32/Q6 slabs plus512/top10 routing. This
is an operator port, not yet complete-model integration. Earlier complete Next
speed remains1.691 tokens/s. The next milestone is approximately10 whole-model
Next tokens/s before speculative FreeToken/Strata work. Ordinary dispatch,
buffer reuse, arithmetic and CPU/GPU offload remain in scope; avoid repeating
Ollama baselines or expanding API/lifecycle work.

Accepted normal artifacts: Ornithv7`55cdb1fa…` at`26829f6`; shared Ornithv8
`b6b81d72…` at`4a06b83`; Next operatorv1`e9a3c833…` at`ffe1cf0`. Two binary
offline shader suites and strict all-bin Clippy pass. All GPU jobs use48-GiB
user cgroups with swap0/core0 and normal explicit private-buffer clearing.
B70 physical memory recovers to50,688,000 bytes with no classified Xe fault.
No installed binary/configuration, gateway/API/catalog, frontend or release
change. Required Vashti changes: none.

At this completed block's postflight all benchmark units exited and inference
admission was resumed: active=false, waiting=0, paused=false and
recovery_required=false. Kuno11434/direct Ollama11435 return HTTP200 with no
resident model; Vashti7771 returns HTTP200. Intel physical VRAM is50,688,000
bytes; NVIDIA596MiB includes the desktop. Kernel classification reports zero
wedged, timeout, GT reset or memory-fault messages on the current boot.

For the next whole-model port, retain the previously measured complete Next
phase boundaries: the old long run spends about186ms/token in attention/GR
enqueue,172ms in router fences,108ms in final proofs and111ms in routed-expert
enqueue. These are host intervals including dependent queued work, not isolated
kernel durations. Port the accepted packed arithmetic and cached512/top10
selection into the original whole-model owners and reduce repeated dispatch/
metadata work. A new operator test alone does not retire this whole-model gate.

The historical checkpoint below describes the original13.9-token/s artifact,
not the currently accepted fast laboratory. Its first result and source-only
block candidate statements are superseded by this section.

## Fully resident Ornith checkpoint, 2026-10-07

The user changed the immediate diagnostic model to the already downloaded
`ornith:35b`, to isolate core performance without offloading. Its Qwen3.5 MoE
adapter is implemented in Rust/WGPU Vulkan, using original packed GGUF weights
and the existing Rust tokenizer. There are no intermediate CPU routing reads:
all40 layers, all256 experts per layer, GDN/full attention and the final head
are GPU resident. This is a separate fixed-public laboratory, not a deployed
engine registration. See [the architecture/measurement contract](kuno-native-ornith.md).

**Source and exact artifact:** `3047d24` introduces the graph; `c174aa9`
handles omitted optional BPE boundary flags and numeric public reference
capture. The accepted execution artifact is
`6917d349c17bbac3eba4e97cfb67a031e96e2c10396a9f473d166f49caa55093`,
frozen in `/media/jmicko/faststore/kuno-development/ornith-resident-build-20261007-v2`.
It retains21,155,768,832 packed weight bytes and235,737,136 private GPU bytes;
915 dispatches execute a complete token. The normal48-GiB cgroup has swap0
and core0. No weight copy, driver change, reset, reboot or install occurred.

**Measured outcome:** two75-prompt/128-output fresh native sequences reach
13.9408 and13.9184 decode tokens/s (127 following steps), with4.987/4.968-second
serial prefill. Native first eight outputs match Ollama; long outputs match the
first14 IDs, then diverge. Both native fresh sequences reproduce identical128
IDs. Thus functioning inference is observed but full-model numerical parity is
**not accepted**. Different activation/reduction arithmetic is a possible
explanation, not an established diagnosis.

Ollama's same public raw prompt/seed424242 runs fully on GPU at75.3222 cold and
83.3857 warm reported decode tokens/s. One numeric-reference run measures
83.5502 and reproduces128 BPE IDs from its in-memory public response. The
identical warm request may reuse Ollama's internal prefix; native resets all
sequence state. Native load3.954seconds follows file warming and is not a
matched cold-load win. All receipts are fixed-public numeric data; no real
prompt, output, image or routing trace enters persistence. See
[the aggregate receipt](../apps/kuno/benchmarks/text/receipts/ornith-resident-foundations-20261007.json).

**Next candidate:** the subsequent explicit `KUNO_LAB_ORNITH_BLOCK_DOT=1`
reuses Q4/Q6 headers and packed byte pairs during subgroup dots. It is off by
default and source-tested only. Do not claim a performance or device-parity
win for it or silently select it. On resume, first compare its individual
packed projections numerically, then teacher-force identical public tokens to
locate long-run divergence and use GPU timestamps to identify the remaining
core bottleneck. Keep the work on speed and arithmetic, not API/lifecycle
expansion. Ornith's14-token/s result does not satisfy Flash-Next's10-token/s
goal or establish Ollama parity.

**Tests:** 118 core tests,477 existing native tests, the new offline shader
inventory,118 Python control tests and strict release Clippy pass. The first
Ornith trial rejected missing optional BOS metadata before GPU creation;
the corrected artifact completes both short and long sequences. No extended
cancellation, multimodal, arbitrary-request, MTP, quality/soak or production
acceptance follows from these tests.

**Earlier heavy-pass outcome:** `9d625d7`'s integrated Flash-Next cap10 path,
with mapped weights, exact AVX2 CPU experts, pooled workers, compact dispatch
metadata, token-level IO, GPU routing and shared-expert overlap, completes the
original75/128 fixture at1.691tokens/s. It remains far below10. Its larger-bank
profiles and aligned upload erasure are source-tested through `9718587`;
expanded-bank device timing is outstanding. The approximate Q8 CPU-activation
long trial diverged and is not selected. Do not combine individual operator
improvements into an unmeasured whole-model gain.

**Restored system:** all benchmark units exited; B70 physical usage recovered
to50,688,000 bytes, NVIDIA398MiB includes the desktop, and no classified Xe
wedged/timeout/reset/memory fault appeared. Kuno admission is unpaused with no
active/waiting request or recovery latch. Kuno11434 and direct Ollama11435
answerHTTP200 with no loaded model; Vashti's existing process on7771 serves
HTTP200. The failed metadata-only test unit was cleared. Installed binaries,
configuration, API/catalog/routing, Vashti and release artifacts are unchanged.
Required Vashti changes: none.

## Joined overlap short ABBA checkpoint, 2026-10-07

The user-authorized pooled-worker and overlap block is finished. Goal remains
paused; no GPU/compiler job is running and installed admission is resumed.
Source`44fd9bc506eedfd701f68c71cb07f47c423cb533` adds explicit pooled count1 CPU/GPU
submission overlap, bounded borrowed orchestrator/parent header capacity, joined
failure paths and unchanged original rank merge/all-family publication.111 core
unit/integration tests,465 native tests,108 Python controls, strict release
all-target Clippy for both crates and formatting pass. Four actual normal-binary
invalid worker/overlap configurations reject before adapter/model access.

Normal artifact SHA`ae9c293443445458ff02faaa79bd88aa29d3222b5dfc3144ab5fa0caa21a61f9`;
source/build`native-count1-overlap-build-20261007-v1`; physical samples/postflight
`native-count1-overlap-device-20261007-v1`, both under faststore's
`kuno-development`. All eight compiler-idle same-artifact cap10 pooled
serial/overlap/overlap/serial short fixtures pass original output IDs, actual CPU
miss counts/fresh continuation, layer0/3 cancellation, poison/recovery and all
four1378-owner/818413568-byte owning erasure audits/source0. Same-stream flush
submits GPU expert work before CPU join; it is not a completion proof or an
instrumented physical-kernel simultaneity claim. The protected serial control
uses the same post-join CPU upload ordering as the overlap candidate.

| Card | Serial short decode ms | Overlap short decode ms | Mean duration difference |
| --- | --- | --- | --- |
| B70 | 4777,4861 | 4653,4796 | 1.96% shorter |
| 3060 | 6460,5979 | 6075,5941 | 3.40% shorter |

These small differences overlap the observed sample variation, especially the
active NVIDIA desktop. Do not select overlap as a reliable performance win or
multiply percentages across distinct artifacts. Pooled worker reuse's prior
short paired benefit remains7.22%/8.31%; overlap is an opt-in scheduling
experiment. Current overlap short rates are approximately1.48 B70 /1.17 NVIDIA
accepted following steps/s, far below the10-token/s working target. No new
Ollama baseline ran. Full-vocabulary logit, hybrid late-finality and original
75/128 gates remain unaccepted, followed by matched repeated long comparisons
and profiling of the remaining GPU/fence work. Short eight IDs do not prove
full numerical equivalence or long stability. Defaults/production are unchanged.

Exact aggregate/source/hash/sample receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-overlap-20261007.json`.
Postflight Intel physical memory returns50688000/34242297856B and Xe classified
wedged/timeout/reset/memory-fault counts remain zero on the same boot. NVIDIA
retains desktop allocations. Both inference listeners respond with no loaded
models; admission active=false/waiting0/paused=false/recovery_required=false.
Sudo cache was refreshed throughout. No installed binary/configuration change,
service restart, hardware reset/reboot, public API/catalog, Vashti or release
change. Required Vashti changes:none. Source is isolated on
`codex/kuno-native-text`; preserve the canonical frontend checkout.

## Pooled worker short ABBA acceptance, 2026-10-07

User-authorized bounded optimization while the goal remains paused. Source
`408abb9e26ed873e740f264db264297025e28539` adds explicit scoped/pooled CPU workers
without changing arithmetic, default execution, frontend or gateway.110 core
unit/integration tests,464 native tests,107 Python controls, strict release
all-target Clippy for both crates and formatting pass. Fixed-public scheduling
control establishes worker setup savings only. Normal artifact SHA
`926abe803dab2d8cb92789ddfda10690a7a1aeb1438a03499cf3d0f4c62132cf`, build
`native-count1-pool-build-20261007-v1`; eight protected compiler-idle same-artifact
short cap10 scoped/pooled/pooled/scoped samples under
`native-count1-pool-device-20261007-v1` (both directories in faststore's
`kuno-development`). All outputs, actual CPU-miss counts, fresh continuation,
layer0/3 cancel, poison/recovery and four complete1378-owner erasure audits pass.

| Card | Scoped short decode ms | Pooled short decode ms | Mean duration reduction |
| --- | --- | --- | --- |
| B70 | 5318,5374 | 4978,4942 | 7.22% |
| 3060 | 6903,6493 | 6173,6110 | 8.31% |

These are seven-step short repeated comparisons, not long throughput or the
10-token/s target. Full-vocabulary logits, hybrid late-finality and75/128 gates
remain unaccepted. Exact receipts:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-pool-20261007.json`.
Intel physical memory recovered to50688000B; content-free Xe classifications
remain zero. NVIDIA retains the active desktop. Admission resumed with no
active/waiting work or recovery latch. No installation/config/service restart,
hardware reset/reboot, API or Vashti change. The next separately authorized
slice is concurrent CPU work and parent GPU expert submission, with bounded
joined ownership and unchanged final merge. No overlap is implemented by408abb9.

## Both-card short hybrid acceptance and constructor correction, 2026-10-07

The user restored sudo and authorized this bounded GPU block; the goal remains
paused. The initial `a8cb5e0` Intel cap0 trial stalled during model construction:
the outer complete-model allocation held MAIN admission while hybrid reservation
called `healthy()` -> `admission.active()` on that same non-reentrant mutex.
The user lab unit was stopped after181 seconds, with no accepted receipt or
owning-erasure claim. Intel physical VRAM returned to50688000 bytes and the
content-free Xe fault classification stayed zero. Preserve that failed attempt
at `native-count1-hybrid-device-20261007-v1`; do not count its misleading systemd
"success"/TERM summary as model acceptance.

Correction `ff3c7d186b5828b6e7b43e9a68b99f048167ca70` supplies a borrowed sealed
construction-allocation proof from inside the existing guarded callback. It
binds the original ledger/stream/sequence and checks poison/retirement flags
without relocking. Arena IDs, worker plan and page checks remain in place;
normal runtime locks and all acceptance/cleanup checks are unchanged.
463 Rust host/offline tests, strict release all-target Clippy, formatting and a
normal Cargo release build pass. The Python validator source is unchanged from
its106-control acceptance, and all four actual receipts pass it again.

Normal binary SHA:
`4382edf25862e600c82eaf59a141fdba37a8fc056a47105da601a9dc3654de4c`.
Build/provenance: `native-count1-hybrid-build-20261007-v2`; physical receipts,
logs, run records and postflight: `native-count1-hybrid-device-20261007-v2`, both
under `/media/jmicko/faststore/kuno-development`. The shared normal target now
contains this new binary; use its own Cargo transcript, never the earlier one.
Aggregate receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-hybrid-device-20261007.json`.

The corrected same-binary cap0 and cap10 short fixtures pass on **both cards**:
original15-token8/7 prefill and all eight chat output IDs, seed424242/temperature0,
seven accepted native steps/scalar0, original SingleQ4All/tile1 policy, actual
completed CPU counts, pre-cancel/poisoned retry rejection, fresh-first and next
output matches, layer0/3 cancellation, all four1378-owner/818413568-byte MAIN
zero audits/source0, normal exit, no retirement and swap0/core0 execution.

| Card | CPU cap | Seven-step decode ms | Completed CPU misses | Fresh CPU misses | Warm expert-weight uploads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intel B70 | 0 | 5440 | 0 | 0 | 2619187200 B |
| Intel B70 | 10 | 5448 | 1256 | 102 | 0 B |
| NVIDIA3060 | 0 | 9056 | 0 | 0 | 4829081600 B |
| NVIDIA3060 | 10 | 6849 | 2259 | 273 | 0 B |

These are **single-order, compiler-idle short observations**, not repeated ABBA,
long throughput, cold-load or production selection. Intel is effectively
unchanged in this pair; NVIDIA duration is24.37% shorter. CPU reads and private
activation/result bridges still occur, so zero expert-weight uploads is not
zero I/O. CPU-assigned misses remain outside the GPU bank; later actual miss
counts differ from cap0, as designed. No routing trace or private vectors are
saved. Scope is the fixed public fixture only. No new Ollama baseline was run;
the10-tokens/s target is unmet and the earlier long1.4258 result remains the
applicable accepted long observation.

**Still outstanding before selection:** full-vocabulary relative logit gates,
hybrid late-layer/second-proof/error/panic/partial-family-commit failure gates,
and the original75/128 whole-model fixture, followed by repeated long cap0/10
ABBA. The short cap10 outputs are not full-vocabulary numerical acceptance.
CPU jobs still join before GPU enqueue, and `cpu_expert::parallel_rows` still
starts/joins two16-worker scopes for each nonempty CPU operator call. A reusable
scoped worker pool or better overlap remains a measured next optimization;
this inspection does not isolate their costs or establish an improvement.
Preserve bounded job/scratch/header ownership and failure joins when changing
workers; do not silently change activation precision or original reduction order.

All four lab units exited; no lab/compiler job from this block remains running.
Installed Kuno admission is restored and verified active=false/waiting0/
paused=false/recovery_required=false. Kuno11434 and the currently configured
direct Ollama11435 both return HTTP200 `/api/ps` with no loaded models (11433
is not the current upstream; no port configuration was changed here). Root xe
query reports50688000/34242297856 bytes; NVIDIA has388/391MiB in two postflight
snapshots including the active desktop. Xe kernel7.0.0-38 classifications for
wedged/timeout/GT-reset/memory-fault messages are all zero in the same boot.
No reset/unbind/reboot, service restart, installed binary/configuration, public
API/catalog/routing, application or release change occurred. Required Vashti
changes: none. This section supersedes the prior source-only status only for
corrected **short** whole-model output/lifecycle/erasure acceptance.

## No-root CPU-miss model registration checkpoint, 2026-10-07

The goal remains paused. The user authorized a bounded source/host pass while
at work, with no sudo or sudo workarounds. Source `a8cb5e00b00f6bb8938df59bc819cef68ad0a7ca`
on `codex/kuno-native-text` registers the separately protected fixed-public
`resident-count1-hybrid-chat` entry. It reserves the existing original count1
CPU-miss workspace, keeps8/7 native prefill, then selects SingleQ4All/dense1 and
cap exactly0 or10 at the completed original boundary. Default inference and
ordinary1376-owner constructors are unchanged.

An actual all48 joined/source-validated MoE permit supplies the checked
zeroizing CPU-miss delta. Overflow is preflighted before any family commit;
the count advances only with final accepted model position after all finishes,
PLE, original bank metadata and health/cancel checks. Cleanup erases it and
failed owners cannot expose it. The compiled public fixture alone records
aggregate counts, never expert/rank traces. Cap10 must prove actual CPU work
both in original decode and fresh continuation; cap0 must prove zero. Both
layer0/3 cancellation probes run the selected cap. The separate validator pins
four1378-owner/818413568-byte audits and all original and hybrid lifecycle
proofs, preserving ordinary1376-owner acceptance.

**Completed without root:**461 Rust host/offline tests,106 Python rejection/
runner controls, strict release all-target Clippy, formatting/diff checks and
normal Cargo release build. Four actual normal-binary negative entry cases
reject missing protections, invalid cap, extra argv and disabled-policy override
before adapter/model access, without a receipt. Python synthetic documents are
validator controls, not hybrid device evidence. The normal binary SHA is
`1c44a37a3c2cd44d41a03946bdd2286bfdab1c727c0843fbe925619cf0ff2197`;
artifact directory is
`/media/jmicko/faststore/kuno-development/native-count1-hybrid-build-20261007-v1`.
It contains the pinned binary, successful Cargo transcript, source identity and
`predevice-rejections.json`. Aggregate source-only receipt:
`apps/kuno/benchmarks/text/receipts/rust-native-count1-hybrid-source-20261007.json`.
The shared normal target executable now belongs to this new build; do not use
it with older9d4f4277 build transcripts. Earlier frozen/durable evidence remains
unchanged.

**Next requires sudo/shared GPU coordination:** pause installed inference
admission, unload/verify actual memory and exclude other GPU work. Run cap0 and
cap10 sequentially on each card using the exact new artifact, for example:

```sh
python3 apps/kuno/benchmarks/text/run-native-count1-hybrid-lab.py \
  --cargo-artifacts /media/jmicko/faststore/kuno-development/native-count1-hybrid-build-20261007-v1/cargo-artifacts.jsonl \
  --model /path/to/the-existing-original-Ollama-blob \
  --card intel --cpu-miss-cap 10 --run-id hybrid-intel-10 \
  --receipt /media/jmicko/faststore/kuno-development/hybrid-intel-10.json
```

The wrapper itself never acquires sudo, pauses services or claims exclusive GPU
access. It verifies swap0/core0/8-GiB execution and exact public eight-token
outputs, then records acceptance only after full lifecycle/audit validation.
Continue hybrid-specific late-final-layer, second-proof and partial-family-
commit fault gates before production selection; matched long timing follows
numerical/lifecycle acceptance. These remain physically unaccepted. CPU jobs
still join before GPU enqueue: no overlap, persistent CPU pool, mapped public
reader, automatic split, hot-request reuse or speed win is established.
The10-token/s target remains unmet; preserve the earlier1.4258 Q4 observation
and do not repeat Ollama baselines.

No GPU/model job, sudo operation, service restart/pause, installation, driver/
hardware reset, API/catalog/routing or Vashti change occurred in this pass.
No compiler or lab job started by this pass remains running. Installed health
and physical GPU memory were not re-audited this turn. Required Vashti changes:
none. This section supersedes the earlier unexposed CPU-miss registration and
counter proposals; it does not supersede their remaining physical acceptance
or introduce a production capability.

## Latest owned-public upload and long comparison, 2026-10-07

`feb76b3` records both-card finite-grid acceptance. `79ba326` adds the narrowly
scoped `create_public_owned(Vec<u8>)`: move immutable original public model bytes
into the ordinary server write queue, avoiding redundant host Vec copies.
Static original GGUF uploads and individual expert sources use it; private-data,
ordered-source, reused bulk-buffer, admission, error and erasure paths remain
unchanged. This is not zero-copy GPU staging. The runtime README records the
exact boundary; the18-file patch SHA is
`64e243409cde4685bc0207876156b595e0081b8d830c1faf66117bb1deb8278f`.
It reconstructs against pinned archives with all modified hashes matching.
459 Rust native tests,49 runtime tests, strict all-target Clippy and formatting
pass; the Python controls are unchanged from their earlier99-test acceptance.

Normal source `79ba326b4de1ff0ee5f7aa4244619bc47378cfb6` release SHA is
`9d4f4277efdcbfcc2349657b9c1bc5d6bd7aef8f3a531c756d38099a4a187144`.
Both adapters pass52 routed/14 original-format/12 expected-failure cases and
the complete eight-token model, fresh continuation, layer0/3 cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits, source0, normal exit/no retirement. Short observations are
Intel5,096ms (read884/upload7ms), NVIDIA7,808ms (read613/upload23ms).
The upload timer is asynchronous enqueue time, not device completion; the
previous Intel run was profiled. These are not matched copy-before/after timing.
Exact evidence: `rust-native-owned-public-upload-device-20261007.json` and
`native-public-owned-build-20261007-v1` under the existing receipt/artifact roots.

The **complete compiler-idle same-binary Intel Q4 ABBA** passes all four original
75/128-output, seed424242/temperature0 model and lifecycle gates. Every sample
uses native tile1, prefill MAX8, actual52-GiB host cap/swap0/core0, the same
borrowed unmodified model, a fresh model owner, and all four original full audits.

| Sample | Profile | Prefill ms | 127-step decode ms | Decode tokens/s |
| --- | --- | ---: | ---: | ---: |
| 1 | single-all | 30704 | 90578 | 1.4021 |
| 2 | single-q4-all | 30019 | 89408 | 1.4205 |
| 3 | single-q4-all | 30308 | 88740 | 1.4311 |
| 4 | single-all | 30281 | 90183 | 1.4082 |

Generic mean1.4052, Q4 mean1.4258 tokens/s: a small observed1.47% rate change,
not the required10 tokens/s. Two repetitions on one card establish only this
bounded comparison, not broad quality or production selection. OS caches were
not flushed, prefill includes lazy loading, and this does not measure hot
cross-request session reuse or the upload copy change in isolation. Process
model caches do not survive samples. No new Ollama baseline was run.
All128 output IDs match, no OOM or swap is recorded, and actual cumulative
unit memory peak is about8.3GiB despite the52-GiB allowance. Preserve numerical
proofs and telemetry at `native-owned-q4-performance-intel-20261007-v1`;
aggregate identities/results: `rust-native-owned-q4-abba-intel-20261007.json`.

Each completed public sample uploads50,656,768,000 expert-weight bytes during
decode:15,413 misses,45,547 hits. Warm original reads take roughly15 seconds;
phase waits also include queued GPU work and cannot isolate kernel times.
The next priority is the existing default-off count1 CPU-miss seam's explicit
whole-model numerical/lifecycle trial, with actual completed-miss evidence,
separate1378-owner/818,413,568-byte validation and cap0 control. The earlier
both-card packed CPU/GPU numeric/calibration acceptance below is already done;
do not rerun that study simply to repeat it. CPU jobs currently join before GPU
enqueue, so no overlap or automatic placement gain is established. A prepared
host/pinned positional or mapped public-weight reader and safe persistent CPU
workers are further measured candidates, not implemented by this pass.

Read-only PCIe inspection identifies Intel's host-facing2b:00.0 bridge and root
port at8GT/s x16; its internal2c:01.0/2d:00.0 fields report2.5GT/s x1 and should
not be used alone as the physical host-link bandwidth. NVIDIA's actual endpoint
reports5GT/s x4 through the chipset. These link fields are not transfer benchmark
measurements; no hardware setting/reset/unbind/reboot occurred.

The goal remains paused. Installed admission is resumed and verified
paused=false/recovery_required=false; both Kuno and direct Ollama /api/ps return
HTTP200 with no models resident, and Kuno /api/tags returns HTTP200. No lab unit
or compiler remains running. A privileged read-only xe query reports50,688,000
VRAM bytes used out of34,242,297,856; NVIDIA reports399MiB including the desktop.
No installed binary/configuration, API/routing,
Vashti, driver or release change was made; required Vashti changes: none.

## Clean paused checkpoint, 2026-10-07

This is the earlier checkpoint, before the additional authorized pass above.
The latest section supersedes its source/artifact and finite-grid status.

Owner branch remains `codex/kuno-native-text` in
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`. Source through `d13e95d`
passes459 native Rust host/offline tests, strict all-target Clippy, formatting
and diff checks. All99 Python native-lab tests pass; runtime49 fixture sources
are unchanged. The ordinary main checkout and installed binaries/configuration
are unchanged; installed admission reports paused=false, recovery_required=false.
All source-only agents stopped with clean isolated branches.

**Physically accepted dense candidate:** `b0c83c9`, `ee9434c`, `0741420` remove
seven inactive tile1 dense dispatch slices at the completed `SingleAll`/
`SingleQ4All` boundary, observe actual capacity and reject mismatches before
private clears. Prefill remains MAX8; active packed math, complete owners,
inactive stored values and fault/audit scopes are unchanged. The exact normal
`0741420` binary SHA256 is
`3098e9cb84ed7563f5160030717e9868a759932ee60769492d764a536337ad36`.

Both cards pass all91 matmul cases (84 prior cases plus7 real same-owner MAX8
prefill/count1 replays). The replay's paired GPU maximum absolute error is zero
on both cards; maximum CPU error is1.1920928955078125e-7. The independent strict
validator `7df78cd` passes both actual receipts against shapes independently
read from the original GGUF header. Operator receipts have no vendor field;
actual vendor selection is joined through the pinned unique-discrete-adapter
selection and separate same-binary model receipt. Per-element numerical gates
remain producer checks; aggregate maxima cannot reconstruct their relative
threshold. No timing authority follows from this validator.

Both complete eight-token chat runs pass the same IDs, seven actual native
continuations, fresh prefill/second-token replay, layer0/3 queued cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits with source0, normal exit and no retirement under actual8-GiB/
swap0/core0 protection. Intel generation is5,563ms (read962/upload392ms);
NVIDIA8,675ms (read540/upload770ms). No additional speed gain is established;
Intel has more read time than the prior short run. Exact proofs and identities:
`apps/kuno/benchmarks/text/receipts/rust-native-single-dense-device-20261007.json`,
backed by `/media/jmicko/faststore/kuno-development/native-single-dense-build-20261007-v1`.
The earlier long Intel75/128 fixture at `4d4ecbe` remains1.311tokens/s; it is
neither this new binary's long result nor matched ABBA performance acceptance.

**Source-only follow-ups, not physically accepted:**

- `ec0cae5` narrows count1 grouped finite/reducer public grids toY10/Y1, keeping
  defaultY80/Y8, the same kernel bodies, complete MAX80 owners/status/fault
  scans and active storage writes. Before timing, run the existing52 routed/
  14-format/12-failure numeric inventory and original complete model/lifecycle
  gates on both cards from a fresh normal binary.
- `b6269b4` plus stable assertion fix `3d65a44` derives the unchanged additional
  1MiB bridge runtime envelope (1,015,808bytes). `7546186` normalizes only
  validated inactive logical MIXED rows1..7 in the experimental count1 input
  scan; active arithmetic, owning gaps/padding and router/fault checks remain.
  `d13e95d` keeps the grid fixture and necessary CubeCL NaN/guard spelling
  lint-clean. This hybrid path is still unexposed and device-unaccepted; the
  default GPU path uses its original owners. Do not infer hybrid acceptance
  from ordinary dense tests.

No production native worker/default routing, API, installation, release,
Vashti change or hardware reset/reboot is accepted. No new long timing or CPU/
GPU overlap win is claimed. The normal shared target binary is still the
`3098e9cb…` artifact; Clippy/test binaries are not a new normal Cargo build.

**Next source proposals preserved, implementation not started:**

- After finite device gates, an explicit fixed-public `resident-count1-hybrid-chat`
  CLI can pin cap exactly0|10, SingleQ4All/dense1, the original chat/seed424242/
  temperature0/eight outputs, full zero audit and cancellation checks. It needs
  a separate strict1378-owner/818,413,568-byte receipt rather than weakening the
  ordinary1376-owner validator. Require actual completed CPU-miss count>0 for
  cap10 and zero for cap0; no routing IDs/masks/per-layer content may persist.
- A private zeroizing completed CPU-miss counter can derive a checked delta
  only after all48 actual joined/source/bank proofs pass, check overflow before
  any family commit, and publish only alongside final accepted position after
  all commits/finish/PLE/health/cancel checks. Failed or partial attempts never
  advance it; a healthy completed-boundary getter is required. This is a
  proposal, not a current counter or CLI capability.
- Existing one-shot lifecycle sites can test late_combine47, second-proof
  error/panic after actual PLE preparation, and cancellation after the first
  genuine GDN commit. Each needs exact-site evidence, terminal erasure/retry
  rejection and fresh-owner replay. Existing layer0/3 tests alone do not prove
  joined hybrid work or late publication safety.
- A standalone exact-tree subgroup proposal can first publish64 original lane
  partials to shared memory, keep one unconditional barrier, reconstruct the
  exact original tree independently within each complete16/32/64 subgroup,
  use uniform explicit shuffles (never subgroupAdd), and keep a second
  unconditional barrier. Original local lane0 either publishes its subgroup
  root or performs the exact serial shared-tree fallback. Unsupported/incomplete
  shuffles target each invocation's own active lane; final target0 is selected
  only for complete groups. Feature absence selects the untouched original
  kernel. Never assume local/subgroup lane mapping or width. Require offline
  WGSL/Naga uniformity, paired bitwise finite roots, signed-zero/nonfinite
  classification, forced fallback and actual width/mask/device/full-owner
  gates before dense or grouped integration; GDN sequential sums stay unchanged.
  No source files were created on the isolated `codex/kuno-exact-tree-shuffle`
  branch at `b6269b4`. Pinned source supports PLANE_DIM/UNIT_POS_PLANE/
  plane_ballot/plane_shuffle, but actual card widths are unverified. See the
  [WGSL subgroup specification](https://www.w3.org/TR/WGSL/#subgroups) and
  [shuffle semantics](https://www.w3.org/TR/WGSL/#subgroupshuffle-builtin).

## Additional finite-grid device acceptance, 2026-10-07

The user authorized another bounded optimization turn while leaving the goal
paused. The normal `91efea6a87b05b475a16aa4dfb8e8fa9cba03fef` artifact is
`d6b834687ac0865bd81f2c1eb732c69d4434a849bb3228487a3cfd9dd2ad6018`.
The count1 grouped finite/reducer grids now pass all52 routed cases,
14 original-format cases and12 expected failures on both actual Vulkan adapters.
The independent strict Q4 validator accepts original Q5 layer6/Q8 layer0.
Operator erasure covers1078 owners/153,944,064 bytes, with normal exit and
actual8-GiB/swap0/core0 protection. The operator run proofs now preserve the
actual public adapter marker and independently check the expected vendor.

Both complete eight-token chat/lifecycle runs pass the original IDs, seven
native steps, fresh prefill/second-token replay, layer0/3 queued cancellation,
poisoned retry rejection, fresh recovery and all four1376-owner/818,249,728-byte
MAIN audits with source0. NVIDIA short generation observes8,525ms, read545ms,
upload767ms. Intel observes5,739ms, read962ms, upload363ms, but its run overlaps
CPU profiling and is **not** a comparable unprofiled timing. The CPU sample
covers30 seconds spanning startup/prefill/decode and possibly recovery/cleanup:
30.64% of leaf samples are memmove,10.97% memset. This is a lead, not isolated
decode attribution or permission to remove private clearing. No copy-path or
whole-model speed gain follows from these results.

Exact identities, receipts, owning audits and CPU sample scope are recorded in
`apps/kuno/benchmarks/text/receipts/rust-native-single-finite-device-20261007.json`
and the durable `native-single-finite-build-20261007-v1` artifact directory.
The experimental hybrid seam remains unexposed and device-unaccepted. There
is no API, installed service, Vashti, release, driver or hardware reset change.

## Resumed native continuation evidence, 2026-10-06

Root `codex/kuno-native-text` integrates `3735a2c` aggregate native phase timers,
`22c1ac6` protected native count1 continuation, `484fe41` same-binary 52-GiB
decode comparison, and `9fb6a2b` bounded CPU/GPU numeric CLI registration. The
current source passes 400 host/offline tests and strict all-target Clippy. There
is no production native worker, routing change, installation or Vashti change.

The release binary built from `22c1ac6` is frozen as SHA256
`6dcd3e2e59b2d184ff83b264b2f6110883daac116a1e07c9cf1c01c04cca44fd`.
Both cards pass the protected eight-token chat reference with native prefill
8+7 and seven native continuation steps, zero scalar steps. Cancellation,
poisoned retry rejection, fresh first-token recovery, all four original MAIN
audits (1376 pages/818,249,728 bytes each), source audits and normal cleanup pass.
The short runs include warm-up and do not establish a throughput improvement.
See `apps/kuno/benchmarks/text/receipts/rust-native-decode-device-20261006.json`.

The Intel scalar/native/native/scalar comparison ended after its second child,
with an aggregate supervisor failure. Preserve the complete failed directory
`/media/jmicko/faststore/kuno-development/native-decode-performance-intel-20261006-v1`.
Do not claim completed ABBA acceptance or replay it merely to obtain the same
result. Scalar sample1 is accepted at 71.032 seconds for 127 decode steps
(1.788 tokens/s). The native sample2 model receipt records 224.924 seconds
(0.565 tokens/s), the same 128 IDs and all four completed owning audits; its
supervisor sample record was not published. The exact failure location is not
proven. A `/proc/<pid>/io` exit race is a source-identified possibility; fix
bounded process telemetry without weakening model/lifecycle validation.

Native router-fence wall time is 151.757 seconds versus scalar 21.031. These
timers include dependent queued work, not isolated GPU execution. The native
eight-token static projection tile is a candidate bottleneck; a protected
public tile1 experiment is in source development, not yet built/device-accepted.
Its default must remain eight, prefill stays MAX8, and private count/positions
must remain runtime data. Independent numeric checks precede timing.

The new CPU/GPU numeric CLI builds but Intel currently rejects its fixed
operator fixture before aggregate publication. Both failed launches retain
actual swap0/core0/8-GiB protection and content-free run records under
`/media/jmicko/faststore/kuno-development/native-cpu-gpu-build-20261006-v1`.
Do not grant a timing or whole-model placement permit from source tests alone.
Constant stage diagnostics are being prepared without changing numerical gates.

## Current priority amendment, 2026-10-06

### Batched audit and staged CPU-miss acceptance

Root subsequently integrates `4d4ecbe` one mandatory full audited admission
readback, preserving completed lease drop, ordered-source release, context/
cancellation/sequence checks, telemetry and MAIN fence credit. Ordinary
readbacks remain unchanged; there is no general empty-epoch skip. All439 native
host tests and strict all-target Clippy pass. Runtime49 and Python81 fixture
sources are unchanged by this seam. Normal release SHA256 is
`3b192df34bf0c9cfa45d462afc14e341cd270c7cdf03f64ffd7197d3a035d71a`.
Both-card eight-token model/fresh replay/cancellation/recovery/full erasure
checks pass; short generation observes4,820ms Intel and8,701ms NVIDIA. See
`rust-native-audited-fence-device-20261006.json` in the receipt directory below
and `/media/jmicko/faststore/kuno-development/native-audited-fence-build-20261006-v1`.
The longer75-token/128-output Intel fixture also passes the same exact output,
native continuation, fresh replay, cancellation/recovery and four full owning
audits under actual8-GiB/swap0/core0 containment. Prefill is33,486ms;127 decode
steps take96,865ms (**1.311 tokens/s**). Weight reads/uploads are13,858/7,441ms.
See `rust-native-audited-fence-long-intel-20261006.json`. This is a protected
lifecycle observation, not matched ABBA performance acceptance; the10-token/s
target remains unmet. Explicit public count1 dense dispatch remains a separate
source candidate.

Root source through `97d50bc` adds the default-off, explicitly reserved count1
CPU-miss seam (`fc5fea3`, `b9b3f19`, compiler wiring `f391168`, corrected invalid
capacity fixture `97d50bc`). All450 native host/offline tests pass; strict
all-target Clippy passed at `f391168`. This source is not CLI-exposed or
device-accepted. Its selected path joins bounded CPU misses before GPU enqueue;
no CPU/GPU overlap or whole-model speed gain is claimed. Default owners remain
1376/818,249,728 bytes; explicit reservation adds two owners,1378/818,413,568.
No production, API, installation or Vashti behavior changes.

The subsequent upload-erasure batching commit `200b38a` also passes all of the
same both-card10/52/14/12/eight-case operator inventories and complete eight-token
model/cancellation/recovery/erasure gates. Native host tests pass436 and runtime
fault fixtures pass49; strict native/runtime-host Clippy and provenance pass.
Its normal release binary is SHA256
`3ce1dbd109d0251a6e6c6a72ea67bdc779a41451accfce2fea41b00f45db3b68`.
Exact build/run evidence is under
`/media/jmicko/faststore/kuno-development/native-erase-batch-build-20261006-v1`,
with aggregate `rust-native-erase-batch-device-20261006.json` in the receipt
directory below. Intel short generation observes4,916ms with router fence1.602s;
NVIDIA observes9,418ms with fence2.224s. The latter has greater route/read time
and establishes no additional gain. These short observations do not grant
matched performance acceptance. One mandatory audited admission fence and
whole-model count1 CPU-miss connection are the next separate source candidates.

Root integrates `f9e9006` bounded full audit copy/map batching, `ebd0046`
staged CPU-miss/GPU-hit numeric selection, `91d4dd9` protected eight-case CLI
and `793c351` its independent strict receipt validator. Native host/offline
tests pass436, runtime ownership fixtures pass45, Python controls pass81;
strict native/runtime-host Clippy and provenance reconstruction pass.

The normal `91d4dd9` release binary is SHA256
`8f08b872111d04fd97e7bc77c1675242257761386ddfd9a7490fa0d82764cded`.
Both cards pass10 GDN,52 routed/14 packed-format/12 expected failure cases,
and the eight original Q5/Q8 staged cases at CPU caps0/1/5/10. The latter
classifies actual noncontiguous misses, preserves resident GPU hits and merges
original ranks; it grants no cache publication, model or timing authority.
Independent validators pin vendor and original layers6/0.

Both complete eight-token `single-q4-all` model/lifecycle runs also pass unchanged
IDs, seven native continuation steps, fresh continuation replay, queued layer0/3
cancellation, poisoned retry rejection, fresh recovery and all four full
1,376-owner /818,249,728-byte MAIN audits. Processes exit normally without
retirement under actual8-GiB/swap0/core0 containment. Durable evidence is under
`/media/jmicko/faststore/kuno-development/native-audit-staged-build-20261006-v1`;
aggregate identities/proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-audit-staged-device-20261006.json`.

Short generation observations fell12,017→5,307ms Intel and13,478→8,838ms NVIDIA
for seven steps. Router-fence wall time fell8.391→1.861s and6.715→2.225s,
respectively; actual warm file reads remain separately recorded. These are
short lifecycle observations, not matched long-run throughput acceptance or
achievement of the10-tokens/s target. Upload erasure batching and the explicit
whole-model count1 CPU-miss seam remain separate candidates. No production,
API, Vashti, installation, release, driver or hardware change follows.

### Q4 block candidate and both-card operator acceptance

Root integrates `3e20935` original Q4 header reuse, `3f38840` strict protected
profile/comparison controls, `cfe8fa0` completed model-boundary selection and
`22a2413` a narrowly normalized shader-declaration test. 426 Rust host/offline
tests, strict all-target Clippy, formatting and diff checks pass. The existing
62 Python control tests passed at `3f38840`; no later Python change is part of
this build. The count1 CPU-miss plan at `dc440d8` is still a proposal.

The normal `22a2413` release binary has SHA256
`b8008cfe72009a2f8856b9bc39ba6d5c543cee371130365507a0c410e4b1f7da`.
Both cards pass 52 routed-MoE cases, 14 independent packed-format cases and
12 expected failure cases. Six Q4Block cases per card retain paired generic
tile1 references for both down formats and capacities16/64/96, including the
cross-segment boundary. Observed generic maximum absolute error is zero in all
twelve comparisons; the original numerical tolerances are unchanged.
Full owning offsets/tails/status, independent tree/core/scalar references,
unaltered bank ledgers and erasure remain enforced. Each probe audits1,078 MAIN
owners /153,944,064 bytes; controlled host scratch is17,543,168 bytes. Each process
exits normally under actual8-GiB, swap0/core0 containment without retirement.

Durable build/run evidence is under
`/media/jmicko/faststore/kuno-development/native-q4-build-20261006-v1`; aggregate
identities and proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-q4-block-operator-device-20261006.json`.
Both-card complete-model eight-token `single-q4-all` now also passes original
IDs, seven actual native continuation steps, full fresh prefill plus second-token
replay, queued cancellation at layers0/3, poisoned retry rejection and four full
1,376-owner /818,249,728-byte MAIN audits. Original source audits remain empty;
explicit normal cleanup passes without retirement. The exact additional profile
proofs and actual8-GiB/swap0/core0 protection are checked by the public runner.
See `rust-native-single-q4-model-device-20261006.json` in the same receipt directory.

Generation observations are12,017ms Intel and13,478ms NVIDIA for seven steps.
Intel warm weight-read time fell from1,018 to237ms versus the earlier short
single-all run, so its lower wall time does not establish a kernel improvement;
NVIDIA is essentially unchanged. No long Q4 ABBA is launched from these short
checks. Full75/128 timing, batched audit cleanup and a real staged CPU/GPU model
seam remain unaccepted. Strict Q4 operator validation is integrated at `abf9b9e`;
both actual reports pass and all72 Python native-lab tests pass. No production,
default/API/Vashti, install, release or hardware change follows. Admission is
resumed and verified quiescent while the next source candidate is prepared.

### Later owner acceptance

Root integrates `64027cf` public static projection tiles, `dd4aac7` protected
tile controls/exact validators/bounded telemetry exit-race handling, `3122e44`
constant CPU/GPU numeric failure stages, `e7bc6d4` actual tile1 recovery proofs,
and `1946020` narrow full-page public bank allocation for the CPU/GPU lab. The
latest source passes **409 host/offline tests**, strict all-target Clippy,
formatting/diff checks, and **51 Python native laboratory tests**.

Both cards pass **84 projection cases** (all seven original formats, counts
1/3/5/8, public tiles1/4/8), retaining the independent oracle, gaps/tails,
original-weight preservation and full owning erasure. Maximum absolute error
is 1.193e-7 on both cards. Frozen operator binary from `3122e44` is
`51581ade3fa1d2e0fe6ccd9a1ff89c31249721f85f76befd7bee9bfe78690ba5`.

Both cards also pass tile1 whole-model eight-token chat, fresh **tile1** second
token replay and cancellation after queued **tile1** work at layers0/3, with
all four original audits and zero scalar source owners. Frozen `e7bc6d4` binary
is `fe74ea739a03cfa4e1e3a555af912db65d1617c4d17bacc7cbaa243cc2e88ef3`.
Short generation durations are 13.012 seconds Intel and 14.187 seconds NVIDIA
for seven continuations; they include warm-up/cache effects and show no useful
improvement over the earlier short native tile8 runs. No long tile1-only ABBA
was started. Do not claim a matched throughput gain from these correctness runs.

The joint expert fixture failure is localized to strict public-bank physical
extent checks. Logical `client.empty(n)` requests did not guarantee a complete
power-of-two backing allocation under the exclusive allocator's rolling mean.
`1946020` requests the already charged full pages in this fixed lab and retains
the original bounded logical shader views. Ordinary model-bank allocation,
numerical tolerances, budgets and strict physical/nonalias checks are unchanged.
Frozen binary
`bf25aa3ae587b2364871496e8057baf168d9ed487b440a50b7a5aa4b893b9d2b`
passes all four numeric configurations on both cards: Q5 GPU-only/mixed,
Q8 mixed/CPU-only, original ranks0..9 including zero probabilities, concurrent
AVX2/16 physical workers. Every case completes18 MAIN pages/884,736 bytes plus
I/O audits. This permits only the stated subset's operator numeric claim;
other groups/splits/workers, timing, automatic placement and whole-model CPU
dispatch are still unaccepted.

All live jobs use actual8-GiB memory/swap0/core0 guards. Detailed identities,
hashes, counters and limitations are in
`apps/kuno/benchmarks/text/receipts/rust-native-tile-and-cpu-gpu-device-20261006.json`.
Single-member grouped MoE and streaming single-token GDN are the next isolated
candidates. Their later operator acceptance follows below; whole-model and
performance acceptance remain separate. No new Ollama runs, frontend/API/routing,
installation, release or hardware reset occurs.

### Single-token kernel operator acceptance

Root integrates streaming GDN `a0b9d5f`, single-member MoE `d07154f`, and
model-boundary policy/count gates with two probe compile corrections `1b486e7`.
416 host tests pass. Independent source reviews cover both candidates and the
model seam. The profile controls and their strict validators are still being
integrated; no production selection changes.

The exact normal release build at `1b486e7` has SHA256
`61aee450ea2b0b8bd2fbd57feb310a7268607929e8f681f6fbc0d85efa353a35`.
Both GPUs pass the ten-case GDN probe, including odd/actual single-streaming
shapes against unchanged chunk, scalar GPU and independent core references.
Both pass 46 routed-MoE cases, 12 expected failure cases, and 14 independent
packed-format cases (seven original formats, tiles8/1). Count/map guards,
immutable old state, finite checks, inactive offsets/tails/status, full owning
erasure and runtime I/O audits remain enforced. GDN audits160 MAIN owners /
113,868,800 bytes; grouped MoE audits994 /141,950,976 bytes across the probe.
Every protected process exits normally with actual8-GiB, swap0/core0 guards and
no context retirement. These are numerical receipts, not benchmark samples.

Durable build/transcript/run records are under
`/media/jmicko/faststore/kuno-development/native-single-operator-build-20261006-v1`;
aggregate hashes and numerical proofs are in
`apps/kuno/benchmarks/text/receipts/rust-native-single-kernel-operator-20261006.json`.
The following later model acceptance supersedes that gate for `single-all`;
compiler-idle matched timing remains separate.

### Combined single-token model acceptance

Controls `1d174d3` and lint fixes `776d58e` connect the public `single-all`
profile. Source tests pass417 Rust /60 Python and strict all-target Clippy.
The exact normal model build at `776d58e`, SHA256
`f05ab6e17de098d4c9f714a39839a5274695f68ab044bb9536ce400b1c2e6142`,
passes both-card original eight-token chat, seven native continuation steps,
actual observed single-all policies, full fresh prefill plus second-token
continuation, queued native cancellations at layers0/3, poisoned retry rejection,
all four complete1376-owner /818,249,728-byte audits, and explicit normal cleanup
without retirement. Each source audit is empty. Actual8-GiB, swap0/core0
containment and the three additional profile completion proofs pass.

Seven-step generation observations are Intel13,053ms and NVIDIA13,566ms,
essentially unchanged from prior short chunk runs. These are correctness/lifecycle
samples with file-cache/JIT variability, not a matched throughput result.
No long kernel-only ABBA is run or speed win selected. Intel's warm weight
read/upload observations total1,018/415ms while the router fence accumulator is
8,508,872,971ns; dependent kernel work and readback also require investigation.
Timers are host interval observations, not isolated GPU kernel execution.

Durable build and records are under
`/media/jmicko/faststore/kuno-development/native-single-model-build-20261006-v1`;
the exact aggregate is
`apps/kuno/benchmarks/text/receipts/rust-native-single-all-model-device-20261006.json`.
Independent whole-model acceptance of the two single-family profiles is not
claimed. The full75/128 fixture and performance selection remain unaccepted.

Root subsequently integrates `cc82731` fixed public CPU/GPU calibration helper
and `4195238` protected registration, preserving `1946020` complete public banks.
420 Rust tests and strict all-target Clippy pass; earlier60 Python tests still
cover unchanged controls. The following bounded timing acceptance is complete.
No production worker/default/API/frontend/install changes.

### Fixed original-read CPU/GPU calibration

The exact normal `4195238` build has SHA256
`b0a2cacda8b3dff0b4b14e2891870f1f9ff6424e9c294aa736e5b78929f82393`.
Both cards complete six fixed Q5/Q8 points, 12 exact-mode numeric gates and24
Concurrent/CPU-first/CPU-first/Concurrent samples, retaining each original
permit through its two tickets. All cases preserve ten original ranks, complete
18 MAIN owners /884,736 bytes plus I/O audits, unchanged tolerances and file
identity. Processes exit normally with actual8-GiB/swap0/core0 guards and no
retirement. No discarded or partial successful sample is accepted.

Two Concurrent samples per point give these **completed candidate** means in
milliseconds (including reads/fills, joined work, private candidate validation
and completed bridge/merge; excluding independent references and final erasure):

| Card / down | GPU10 / CPU0 | GPU5 / CPU5 | GPU0 / CPU10 |
| --- | ---: | ---: | ---: |
| Intel Q5 | 37.180 | 30.782 | 26.970 |
| Intel Q8 | 48.290 | 34.844 | 32.918 |
| NVIDIA Q5 | 40.265 | 33.174 | 25.447 |
| NVIDIA Q8 | 55.296 | 41.922 | 29.205 |

Fixed group0/vector0 and prior numeric/reference work warm the original weight
working set. This narrow operator trial supports further CPU-miss integration;
it is not prepared-host, cold storage, the32-group study, automatic placement or
full-model token throughput. Cleanup and full-with-reference walls are separately
recorded. No private-worker/native-default integration follows from these data.

Durable records are under
`/media/jmicko/faststore/kuno-development/native-cpu-gpu-calibration-build-20261006-v1`;
raw bounded samples, numeric gates, exact identities and hashes are in
`apps/kuno/benchmarks/text/receipts/rust-native-cpu-gpu-calibration-device-20261006.json`.
Next source work is an explicit single-token Q4 header-reuse candidate and a
reviewed CPU-miss/GPU-hit owner/ledger proposal. They remain unaccepted.

The user selected **10 decode tokens/s** as the working native performance target
and explicitly deferred additional/repeated Ollama baseline tests to conserve
development usage. Prioritize measured Rust-engine optimizations after the
applicable numerical/lifecycle checks; expanding the Ollama matrix is not a
resume prerequisite. The completed Intel host-limit experiment must not be
replayed merely to establish the same cache-reclaim result. A corresponding
NVIDIA containment study can wait unless it answers a concrete optimization
question. Preserve historical receipts and label results against the agreed
target rather than claiming a new matched Ollama comparison.

The goal was still paused when this amendment was recorded. No source/runtime
change, GPU load, benchmark or service operation accompanies it. A read-only
source review is preparing the existing native count1 continuation path as a
candidate; acceptance and performance remain unmeasured. Read the
[measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06)
before interpreting the historical practical steps and gates below.

## Latest integrated source and device evidence

Root worktree/branch remain
`/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`, `codex/kuno-native-text`.
The original path resolves through the new `~/dev` symlink; managed worktree Git
pointers, source/read/write access and shared Demons shell are verified.

- `195d8a5`: integrates reviewed numeric candidate `2439d73`.
- `962c37d`: named protected `resident-numeric-boundaries-probe`, rejecting missing
  owned I/O/full audits before creating a device; no production route.
- `80aef9d`: exact aggregate-only numeric validator (author `36d2121`).
- `5ffffa4`: separate same-binary host-limit diagnostic (author `13844b5`).

396 host/offline tests passed before extracting the explicit CLI guard; its
additional focused test passed afterward. Strict all-target Clippy, formatting
and diff checks pass. The numeric validator and memory diagnostic each pass 11
focused Python tests. No existing correctness tolerance or 8-GiB guard changed.

The Cargo-selected binary is frozen durably outside the build target at
`/media/jmicko/faststore/kuno-development/public-native-artifacts/73ce13b2b9fdac27cb85b655bc5543960b9decafafb326900d9656ee931a53a2/kuno-text-lab`.
Its SHA256 is the parent directory name; build snapshot SHA256 is
`79e0f766678c2fd80b0d9826d78017d1992c783026b2b8295010d94225b04a29`.
The missing `/tmp` copies were not treated as live artifacts. A durable older
`110b9926...` release was subsequently found, but the current tests use the new
binary and its actual Cargo identity.

Both cards complete all eight fresh scalar/native owners, full owning maps,
audit/erasure and ordinary family acceptance at position0/count1, protected by
8 GiB, swap0 and core0. All numeric validators and worker units exit0. See
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-device-20261006.json`.
The first Intel launch failed before GPU work because systemd starts in home and
the wrapper path was relative; terminal exit2 is retained. The corrected launch
uses absolute wrapper and frozen executable paths.

| Boundary | Intel bit mismatches / maximum absolute | NVIDIA bit mismatches / maximum absolute |
| --- | --- | --- |
| layer0 combined PLE input | 750 / 7.45e-9 | 0 / 0 |
| projected KEY | 0 / 0 | 0 / 0 |
| projected VALUE | 0 / 0 | 0 / 0 |
| injected residual | 6,579 / 1.49e-8 | 4,185 / 5.96e-8 |

KEY/VALUE depend on separate per-layer token embeddings, not the differing
residual. PLE has known scalar CPU-f64 versus native GPU-f32 topology differences.
Those cannot explain Intel's already differing input. Same nominal f32 schedules
in distinct compiled upstream shaders permit contraction/codegen differences;
source inspection alone does not identify their first offender. These results
locate differences only in this position0/count1 fixture. Whole-model bitwise
acceptance and faster-than-Ollama claims remain false; no arithmetic changed.

## Current host-limit diagnostic

The dedicated public lane fixes one actual executable and fixture, changes only
its declared user cgroup's MemoryMax 8/52/52/8 GiB, keeps swap/core0 and all owning
cleanup/cancellation gates, and records memory/refault/pressure/process-I/O
counters. It does not flush global caches or change production limits. Its
strict 8-GiB correctness checker remains separate. No guard output is fabricated.

The completed Intel experiment is
`kuno-public-performance-intel-20261006-v1.service`, with durable records under
`/media/jmicko/faststore/kuno-development/native-performance-intel-20261006-v1`.
The unit finishes in 1,046.203 seconds, swap peak0, inactive/dead/exit0. All four
model/public-output/full-cleanup/cancellation checks pass, with no cgroup OOM.
Exact identities, model receipts, bounded cgroup records and telemetry hashes are
in `apps/kuno/benchmarks/text/receipts/rust-native-host-limit-intel-20261006.json`.

| Sample / cap | Prefill seconds | 127-step decode seconds | Decode tokens/s | Sampled child disk reads GiB |
| --- | --- | --- | --- | --- |
| 1 / 8 GiB | 91.199 | 128.939 | 0.985 | 123.35 |
| 2 / 52 GiB | 72.232 | 80.438 | 1.579 | 35.46 |
| 3 / 52 GiB | 43.520 | 63.189 | 2.010 | 5.79 |
| 4 / 8 GiB | 91.992 | 130.799 | 0.971 | 126.54 |

The 8-GiB runs repeatedly hit the cgroup limit (58,052/62,003 `max` events),
while both 52-GiB runs have zero such events. Warm weight-read times fall from
65.414/66.696 seconds to 17.996/5.063; upload times stay 7.560–8.432 seconds.
The limit materially biases this laboratory through model file-cache reclaim;
it is not an isolated GPU compute cost. The 52→8 transition also reclaims about
35 GiB before the last child. Naturally changing caches explain some variation
within a cap, so this is not a pure cache-controlled cap-only estimate.

No global caches were flushed. Each child starts a fresh owner; sampled disk
reads and cgroup deltas include later cancellation/fresh replay/audits, outside
the internal decode interval. Per-child current peaks are sampled; unit
`memory.peak` is cumulative. NVIDIA's host-limit experiment is not started.
Matched Ollama production throughput and a true second hot full request remain
unaccepted. Read `apps/kuno/benchmarks/text/performance-lane.md`.

At checkpoint, no public Kuno laboratory unit or native engine child is running,
Ollama `/api/ps` is empty, and NVIDIA has returned to desktop-only baseline.
Main `codex/image-flow-refinements` stays clean and unchanged. No installation,
production limit/configuration, public contract, frontend or release changed.

## Practical next steps after the user resumes

1. Establish basic calculation speed against actual Ollama/ggml before advanced
   expert scheduling. The user explicitly wants this baseline: same resident
   original weights, actual matrix shapes, fixed public inputs, precision and
   batch counts on the same device, with matched CPU workers. Existing operator
   timing wins compare Rust variants only; CPU physical acceptance makes no
   speed claim. Separate completed operator timings, full-layer coordination,
   and whole-model matched GPU/CPU placement; do not subtract nested enqueue
   timers to infer a compute floor. See the later CLI/offload requirement in the
   coordination guide; it changes no installed inference behavior yet.
2. Review the completed Intel cgroup evidence, then obtain NVIDIA evidence under
   the same protocol. All four declared Intel cases have finished; do not replay
   them merely because an earlier observation timed out.
3. The existing complete native `prefill_chunk(&[token], LastLogits)` can continue
   decode after native prefill on the same owner, retaining GPU-f32 PLE. A strict
   default-off public-lab flag is a small next experiment. Require native MAX8,
   owned transient I/O and complete audits; record observed native/scalar decode
   counts. Once native decode starts, never insert a scalar step then switch back:
   scalar advances the model but not the native PLE workspace cursor.
4. Keep the independent scalar/higher-precision oracle. Both-card complete
   fixtures, all-family cancellation/recovery/full cleanup and score comparisons
   precede an old/new timing comparison. Matching the same native implementation
   against itself is not independent numeric acceptance. MAX8 scratch/bridges
   at count1 can be slower; no benefit is established by source inspection.
5. Joint CPU/GPU expert source remains integrated but physically unaccepted;
   preserve the earlier gates below. Long contexts/options/quality/private
   worker, scheduler/catalog/native-default and managed `ollama.` routing remain
   outstanding. No Vashti change, install, public API, release/reset/reboot is
   part of this checkpoint.

## Historical paused checkpoint, 2026-10-05


The user explicitly paused the goal and requested completion of current source
work only. No new GPU benchmark, installation, service/configuration change,
driver reset or reboot is part of this checkpoint. The goal is **paused**, not
complete. No faster-than-Ollama claim or production native-default dispatch is
accepted. Required Vashti changes: **none**.

## Integrated source

Use managed worktree `/home/jmicko/.codex/worktrees/kuno-image-speed/vashti`,
branch `codex/kuno-native-text`. The source checkpoint is `e386c79`:

| Root commit | Author commit | Change |
| --- | --- | --- |
| `5f2ed09` | `6cd9a246` | Read-only fixed-public MoE owner inventory |
| `b2b69cc` | `c16359f5` | Real-read CPU/GPU expert operator laboratory |
| `1ef151f` | `dc915f6b` | Borrow the existing caller graph; avoid duplicate binding |
| `e386c79` | `db3a18f7` | Complete measured work before independent calibration references |

Root completed 391 host/offline tests (zero failures or ignored tests), strict
all-target Clippy, native-lab formatting and `git diff --check`. These are source
checks, not physical acceptance. Exact source hashes, budgets and check evidence
are in `apps/kuno/benchmarks/text/receipts/rust-cpu-gpu-calibration-source-20261005.json`.
The author separately completed 384 release host/offline tests, strict Clippy,
formatting and a normal release build in its isolated branch. That unregistered
author binary is not a root device-tested or installed artifact. The root's
older frozen device artifacts retain their original source/build identities.
The focused independent read-only review of the final calibration-order change
found no material blocker: references follow completed measured consumers and
merge/audit; unchanged numerical gates precede local plan publication; retained
values remain within the charged reference and bridge envelopes.

Read the implemented section of [the calibration contract](kuno-native-text-cpu-gpu-calibration.md).
This operator is source-integrated but has no registered root CLI and has not
executed actual GPU/model payload work. No production model calls the new CPU
operator. Numeric acceptance precedes timing; the sealed local permit alone is
not both-card acceptance or an automatic routing policy.

## Existing physical evidence

- Both cards passed the complete fixed-public 75-prompt/128-output fixture under
  frozen binary `b1ba040f…`; all original output IDs and complete cleanup match.
  See the full Intel/NVIDIA receipts in the performance guide.
- All 44 actual context4/8 lifecycle cases completed under frozen `110b9926…`,
  including late error, actual PLE-ticket panic, cancellation after the first
  GDN commit, poisoned retry rejection and fresh recovery. Both worker units
  exit 0, with full audits and retirement=false.
- All 24 final greedy comparisons matched, but full scores differ: maximum
  absolute delta about 1.04e-5 on Intel and 8.70e-6 on NVIDIA. Both outer
  validators correctly exit 1; whole-model bitwise acceptance remains false.
  No tolerance was relaxed. Matching aggregate statistics do not prove that
  two native partitions produce bit-identical vectors.
- The packed Rust CPU operator passed 16 fixed-public actual original-weight
  AVX2 cases and portable comparisons. These checks do not establish all-ten
  CPU/GPU parity, joint overlap benefit or a connected hybrid model.

See [the performance guide](kuno-text-performance.md), especially
`rust-native-model-lifecycle-device-20261005.json`,
`rust-native-model-full-intel-20261005.json`,
`rust-native-model-full-nvidia-20261005.json` and
`rust-packed-cpu-experts-actual-20261005.json`. No device job is left running by
this checkpoint; do not resume already-terminal recorded sessions.

## Numerical investigation candidate

An isolated source candidate is frozen at
`2439d732966c066d39ef9ee745d208206d98e5ee` in
`/home/jmicko/.codex/worktrees/kuno-text-loader/vashti`, branch
`codex/kuno-native-numeric-boundaries`, based on `8621308`. It is not integrated
or registered on the root branch at this checkpoint. Five focused tests pass
(376 unrelated tests filtered), as do strict all-target Clippy, formatting and
diff checks. No full suite, release build or device test was run for this
candidate. Both its branch and the earlier failure-fixture branch are clean
and preserved. Root's focused source review found no material blocker; an
additional peer review was requested but has not returned. Exact candidate
provenance is recorded in
`apps/kuno/benchmarks/text/receipts/rust-native-numeric-boundaries-candidate-20261005.json`.

The intended default-unarmed collector compares actual existing layer0 combined
residual, PLE projected KEY, projected VALUE and injected residual, one fixed
public position0/count1 boundary per scalar/fresh-native pair. All eight complete
normal owners must close and audit before aggregate-only results are published.
Actual KEY width is 10,240; VALUE width is 2,560. No state import, math change,
new MAIN pages, user data or whole-model numerical tolerance is introduced.
Controlled diagnostic host storage is bounded at 610,304 bytes within its
independent 1-MiB envelope. Additional audited runtime is 1,114,112 bytes;
actual original worst bound plus this component is 65,204,224 bytes, below the
unchanged 67,108,864-byte hard cap. These source bounds do not establish physical
map/audit acceptance.

The existing scalar PLE tail uses CPU f64 accumulation while the native tail
uses the already-approved GPU f32 path. This is a candidate explanation for
small full-score differences, not proof of first divergence. Scalar/native
projection kernels use the same 64-lane schedule. Observe their actual input
and output boundaries before attributing differences to the PLE tail or shader
lowering.

## Resume gates

1. Review and validate the frozen numerical-boundary candidate, then register
   only its explicit protected public diagnostic. Run actual paired boundaries
   on both cards before accepting a cause or changing any arithmetic.
2. Register the corrected joint expert lab in an explicit protected Numeric-only
   entry using one caller graph/client/retirement owner. Start with a small
   declared subset; do not infer the whole matrix from it. Accept actual original
   Q5/Q8 expert stages, all-ten ownership/rank math, full maps/tails/faults and
   completed erasure on both cards before timing.
3. Correctness containment remains 8 GiB, swap disabled and core dumps disabled.
   Before speed comparisons, design a separate verified performance lane with
   matched actual engine memory containment and captured numeric cgroup counters.
   The original Ollama baseline used 52 GiB while current native correctness
   units use 8 GiB; a prior native 52-GiB receipt exceeded 8 GiB. Buffered file
   cache is cgroup-accounted. This is a plausible benchmark bias, not a proven
   cause: current runs ended before pressure/refault counters were captured.
   Low RSS and the unreliable 256-KiB systemd summary do not settle it. First
   quantify the same-binary 8/52-GiB effect, then compare engines sequentially
   under matched conditions. Do not drop global caches or weaken private limits.
4. Preserve the recorded cold definition: model unloaded, OS cache unflushed.
   Hot full-prefill comparisons require zero Ollama cached prompt tokens. The
   previous repeat reused 71 tokens and is not a fresh-native-prefill comparator.
   Compile outside measured runs and preserve actual binary/source/settings.
   Follow the [current measurement decision](kuno-text-performance.md#baseline-variability-and-next-comparison-2026-10-06):
   use 10 tokens/s as the practical target, without further Ollama baseline runs.
   The original 7.6–8.0 tokens/s receipt is not Ollama's ceiling. Preserve native
   timing/identity/cache-state evidence and include observed stalls rather than
   claiming a new matched-engine speedup from the working target alone.
5. Only then connect measured CPU/GPU splits into the complete Rust model and
   establish quality/options/tools/long contexts/private worker acceptance,
   scheduler integration and native-default/managed `ollama.` catalog routing.

System memory cache/accounting references:
[kernel cgroup memory documentation](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html#memory)
and [kernel process memory documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html).
Hardware availability and sudo should be checked again after the user resumes;
no credentials belong in this document or inference diagnostics.

### Pending packed runtime audit experiment

The batched-INFO laboratory will pack full post-erasure source copies into
fewer bounded staging owners. Every original full source byte, new full stage
and padding remains charged, mapped/inspected and finally erased under the
original caps. No inference math, original MAIN proof or installed path changes.
Sourceb5fafef/artifact2d8f4a9c passes all128 IDs and original owning/fresh/
lifecycle gates:22697ms decode (5.5955 tokens/s),16885ms prefill. Warm
readbytes10211328, uploaded bytes0; CPU total3528544550ns. Final proof
interval falls from4172079892ns to3845052045ns in these single samples.
Physical B70 returns50688000B.51 runtime host tests plus two isolated logger
controls,492 native tests/both shaders,33 Python controls and strict Clippy
pass. This does not meet10 tokens/s or establish a paired speed comparison.

The next candidate shares the same three typed GPU error scopes across each
already-owned single-stream scheduler batch. It adds no deferred task queue or
private owner and preserves all before-acceptance errors/fences. Source/physical
acceptance source4495d8f/artifactb64b9cc5 passes all original128 IDs and
owning/fresh/lifecycle gates:22665ms decode (5.6034 tokens/s),16890ms
prefill. This is essentially the same observed speed as5.5955; no meaningful
speed benefit is established.53 runtime tests/two logger controls,492 native
tests/both shaders,33 Python controls and strict Clippy pass. Physical memory
returns50688000B.

The next diagnostic replaces ambiguous custom `Kernel` profiler names with
static public operator/packing families. No arithmetic, identity, owner or
speed-selection change. First-following profiling remains excluded from speed
selection.

### Actual metadata batches and updated host profile

Source a3248d6 / frozen artifact d461f52b passes the original128-ID full
fixture, fresh continuation, poison and owning/lifecycle gates:22260ms decode
(5.7053 tokens/s),19126ms prefill. Metadata boundary calls fall from635131
to159643; original actual extents, aliases and shard/dtype checks remain.
492 native tests and two offline shader suites,33 Python controls, strict
all-target Clippy pass. Physical B70 memory returns50688000 bytes.

Static family labels source599b523 / artifact2c4f5195 also pass the short
fixture and original owning/lifecycle gates. First-following diagnostic4309
kernels total59.76967ms; dense Q4K13.299167ms, clears6.44854ms, grouped
RMS5.944636ms, GDN4.477242ms, dense Q6K3.967663ms. Profiling forces
extra flushes and differs from the long fixture; it is excluded from speed
selection and does not establish an unprofiled GPU execution floor.

A separate twelve-second root CPU sample of the public v25b fixture keeps raw
perf stacks in private tmpfs, removes them, and saves only static symbols/sample
aggregates. It points to CPU packed expert math, allocator cleanup, and buffer
initialization/inspection. Cycle shares are not elapsed-time shares; the entire
profiled sample is excluded from speed selection even though its output matches.
Next candidate compacts exclusive allocation pages in place, preserving the
original free aging, release conditions and every retained descriptor identity.

Source fd1b864 / artifact22309cf4 accepts the unchanged full128-ID and
owning/fresh/lifecycle fixture with21648ms decode (5.8666 tokens/s),15544ms
prefill. Physical memory returns50688000 bytes; no new classified Xe faults
as of2026-10-08T12:11:53.071467+00:00. Three actual imported allocator host
controls,53 original runtime tests/two logger controls,492 native tests/two
shader suites and strict Clippy pass. One sample;10 tokens/s remains unmet.
The next candidate uses word-sized full-span zero inspection. A standalone
public16-MiB zero-array scan,100 passes, measures754881065ns for byte checks
and46726019ns for safe8-byte words; this is a CPU microbenchmark only, not
whole-model speed acceptance. The original spans/tails/checks remain.

Source998fdcd / frozen artifact8444d9db completes original128 IDs and all
owning/fresh/lifecycle gates:19777ms decode (6.4216 tokens/s),15504ms prefill.
The completed-final-proof interval is2756719001ns/127 steps. Physical B70
returns50688000 bytes.55 original runtime/zero-scan tests,three actual allocator
controls/two logger controls,492 native tests/two shaders and strict Clippy pass.
Next candidate groups consecutive full scratch shader clears (explicit
`--compute-scratch-clear`) without ending the existing compute pass. Final
erasure, every full byte and all original completion/ownership checks remain.

Sourcea6cac90 / frozen artifact6ad5e5ed completes the original full128 IDs
and owning/fresh/lifecycle gates:18654ms decode (6.8082 tokens/s),15494ms
prefill. Independent groups2/3/4 full-owner/padding/unselected controls pass.
494 native tests/two original shader suites,34 Python controls and strict
Clippy pass. Next candidate narrows only active-row dispatch grids; all shader
math and full MAX8 owning allocations remain.10 tokens/s is still unmet.

Source b6049a3/artifactbb1c45f637c2d64926267bd943a5e96a31234b8890d7a96b5fd8b17c99fe27b1
limits GR/GDN/QSA and status grids to their validated active rows. The full
original75/128 fixture and fresh/poison/full-owning/lifecycle gates pass with
17948ms decode (7.0760 tokens/s),17044ms prefill. Physical B70 returns50688000B.
494 native tests, both original offline shader suites and strict Clippy pass.
The next candidate groups statuses at their existing ready-before-reuse
boundaries, preserving active words, sticky faults and complete original gates;
physical acceptance is pending. No installed service/API/frontend changes.

Grouped status source9fbda89/artifact3c7346fb119a96ccd631e2a96eec2c4a199d9fc5f46ffd6250311471ba310044
passes the full original75/128 output and owning/fresh/lifecycle gates:
17724ms decode (7.1654 tokens/s),15279ms prefill,18415 grouped status calls.
495 native tests, both offline shader suites,35 Python controls and strict
Clippy pass. B70 returns50688000B; classified Xe faults remain zero on kernel38.
The next candidate ports GR RMS to max-scaled vector/subgroup reductions;
source gates pass496 native tests plus both shaders,35 Python controls and
strict Clippy; complete physical acceptance is pending. API/installed inference
remain unchanged. The10-token/s target remains unmet.

Shared GR source901f35b/artifact809aace6a5f6d86a5d6d2effa118c685f46d1f9a074f610ae8309fbe6f31739e
passes independent actual-shape/count/invalid/overflow/padding probes and all
original75/128 output, fresh, full-owning and lifecycle gates. Decode17805ms
(7.1328 tokens/s), prefill15240ms,12319 warm shared-GR calls; no speed gain over
v30 in this single sample. B70 returns50688000B. The next candidate batches
three completed PLE diagnostic readbacks, retaining every original validation
and enforcing the simultaneous peak against the unchanged4MiB host cap.
496 native tests, both shaders and strict Clippy pass; physical acceptance
pending. Installed inference and public contracts remain unchanged.

PLE batch source7a2dab6/artifact23c4ccf7ab5775995f449f1f4ac40f167b93211324110091bc3f807f5aafad94
passes all original75/128 output, fresh, full-owning and lifecycle gates:
17745ms decode (7.1569 tokens/s),15431ms prefill. No measurable gain in this
single sample; B70 returns50688000B. The same source recompiled for original
compiled two-pass CPU Q8 (artifact3885e2a9aa2ef3e2e6f001ea85753876d06abe7aff03e3d3bac4b9d02fb2c6c0)
also passes all original gates, but18358ms decode (6.9179 tokens/s),16022ms
prefill is slower. Its CPU operator total rises3006→3686ms against v30 and warm
disk reads remain zero; retain original float FMA as the control. Both feature
sets pass496 native tests, both shader suites and strict Clippy. The next
candidate factors original Q4_K vector dots for dense/expert kernels; independent
rows and full original device acceptance are pending. No API/installed changes.

Factored Q4_K source1f0e078/artifacte71731c34dffd23e94001a21f820f00e7bd5bc9bfbea6b3cfe5f8ba706269a39
passes all original75/128 output, independent rows and owning/fresh/lifecycle
gates, but18110ms decode (7.0127 tokens/s) does not improve v30. Eight workers
on the same frozen artifact also pass all original gates with17943ms decode
(7.0780 tokens/s). CPU user/system ticks fall4026/612→2726/417 against v30,
but CPU operator wall total rises3006→3256ms; retain sixteen workers. Both
physical postflights return50688000B. The next source restores original Q4_K
arithmetic and removes count1 hybrid inactive merge workgroups/repeated
immutable owner-header reads. Source gates pass496 native tests, both shader
suites and strict Clippy; complete device acceptance and a new bounded CPU
profile are pending.10 tokens/s remains unmet; no installation/API/UI changes.

Hybrid fold source5ee51a5/artifact21dba701f30a3a90731f8862b141a6db580dc754bfe507505458c0490939ed14
passes all original75/128 IDs and owning/fresh/lifecycle gates:18126ms decode
(7.0065 tokens/s). No gain; B70 returns50688000B. Best remains v30 at7.1654.
The next candidate replaces byte-at-a-time heap erasure with full-capacity
volatile word stores and directly copies into preallocated guarded owners.
496 native tests/two shaders,57 runtime controls/three actual allocator controls/
two logger controls and strict all-target Clippy pass. Exact unchanged pins and
regenerated23-file provenance patch3f6c6be96aa16958293769027304cf27d4453c6ac3a77edd4415793b1257ad93.
Complete physical acceptance and a refreshed bounded host profile are pending.

Heap erasure source13f591c/artifact54526913e4c600e32a828530de408fe6c99a1e466e5bc80e7a3f4b26c3ecfe76
passes all original gates with17730ms decode (7.1630 tokens/s),15579ms prefill.
B70 returns50688000B. Two bounded host profiles, v36b(all threads) and v36c(main
thread), pass original complete model/lifecycle acceptance and are excluded from
speed selection. Raw stacks are root-owned tmpfs and removed; only static
symbol/sample aggregates remain. Main-thread clearing/copying/allocation remains
visible; cycle shares do not establish elapsed-time attribution.

First-following GPU profile v36d rejects with retired private I/O during its
profiling path; no acceptance or timing selected. Physical B70 returns50688000B
and same-boot classified Xe faults remain zero at2026-10-08T14:00:58.573310+00:00.
Do not bypass its original cap or relax ownership gates to obtain a profile.
The next candidate explicitly reuses only exact-size inspected-zero Readback
owners inside the original I/O cap. Source and complete device acceptance pending.

Readback reuse candidate source gates pass496 native tests/two original shaders,
61 runtime host controls/three actual allocator controls/two logger controls,
36 Python count1 controls and strict all-target Clippy. Provenance23-file patch
fe6c6cfa60fa34f2b784dfb0292a15103f4cb61c608949201e022fec639eccc0;
original four upstream pins/licenses and both locks remain unchanged. Device
acceptance is pending. No source edits or compiler overlap during timings.

Readback reuse source3f16450/artifact0162e03533dc58025dbf4ca146b9552a32590490bf3493473bda565705937155
passes original128 IDs and all fresh/poison/full-owning/lifecycle gates with
17491ms decode (7.2609 tokens/s),48883ms prefill and1900 warm actual reuses.
B70 returns50688000B. The prefill regression is recorded, not attributed or
hidden;10 tokens/s remains unmet. Next candidate retains up to64 small owners
in the same8MiB cap, with explicit actual pointer storage plus owner header
limited to the original4KiB envelope before content. Device acceptance pending.

Small Readback pool source4d17149/artifactef05a2f88c95184639f18c1d45e7eec384f5c3b9055fd6b575918422d3083801
passes all original128-ID/owning/fresh/lifecycle gates with17643ms decode
(7.1983 tokens/s),15251ms prefill,7948 warm reuses. Source gates496 native/two
shaders,62 runtime/three allocator/two logger controls,36 Python controls and
strict Clippy pass. B70 returns50688000B. More reuse does not establish a speed
gain; the existing single best remains v37 at7.2609.

Next candidate uses native primary mapping only for Upload owners: full GPU
clearing twice and direct CPU READ mapping/full zero inspection between passes.
It removes repeated WRITE remaps and Upload verification stages while retaining
all original source bytes, both erase passes, error fences and failure retirement.
Exact capability and original complete-model acceptance are pending; never relax
the original owning gates or budgets. Installed execution remains unchanged.

Primary Upload source gates pass496 native/two shader suites,65 runtime/three
actual allocator/two logger controls,37 Python controls and strict all-target
Clippy. Regenerated23-file provenance patch
f46b89e75ba029cd435f3b0cd0a71715300c306fb4377b69dfa003414d8efd6a
retains original upstream archives/licenses and both locks. Device pending.

Primary Upload source5b4eceb/artifact83e4292a2994430bfe312f1f3c349a9a89fa2e08426672e18d72beca6acdf9cd
starts the correct capability/probes but its all-size clearing run is materially
slower and is deliberately stopped before full output/owning/lifecycle acceptance.
No receipt or speed selection exists. It is a fixed-public fixture; no private
user content was involved. Ordinary process stop returns B70 to50688000B without
GPU reset. Do not enable that candidate by default or represent it as accepted.

The next candidate uses original volatile Upload erasure and inline Readback
map completion on the compiled inline calling thread, preserving original
guarded leases, callbacks, failure retirement and all owning/copy/clear bounds.
Source and complete device acceptance are pending. Original async worker/API
behavior, installed inference and the10-token/s target are unchanged.

Inline Readback source gates pass496 native/two shaders,66 runtime/three actual
allocator/two logger controls,38 Python controls and strict all-target Clippy.
Regenerated23-file provenance a93143eaaa8f6f88155ad8323145214f26a7e51b3107e060ef2f8135a40978e9
retains original upstream pins/licenses and both locks. Device pending.

Inline Readback sourceb345f9c/artifact3e707443f0452d00f9b4bc24608efbad406a2fde1f24bc58acc0eb6d658ef709
passes unchanged original128 IDs, independent arithmetic/fresh/poison and full
1378-owner/818413568-byte normal cleanup/lifecycle gates. Following127 steps
take17047ms (7.4499912 tokens/s), prefill48856ms,9525 warm inline map batches.
Physical B70 returns50688000B and same-boot Xe classifications remain zero.
This is the best single sample so far;10 tokens/s remains unmet.

The next candidate partitions GDN key reductions across four lanes. Arithmetic
summation order changes and complete original acceptance is pending. Installed
execution and all full-owner/cleanup gates remain unchanged.

Partitioned GDN sourceaa3681c/artifactb2efc9e3be1d353687059dbe1a77fb19eb9ff231f8e212063929a2f7791f5f26
passes original128 IDs, independent scalar numeric/invalid-column probes and
all owning/fresh/poison/lifecycle gates:17079ms decode (7.4360 tokens/s),15463ms
prefill. B70 returns50688000B. No measured overall gain; current best remains
v40 at7.4499912.497 native tests/two shaders and strict Clippy pass.

The next candidate combines two original expert-bank shards into two total
projection dispatches rather than four, retaining original packed arithmetic,
full physical proofs and all cleanup gates. Source/device acceptance pending.

Paired original banks source07308b8/artifact6351c8bdf632c9db5ee3b536995ecee41b0312fffe5ad9ddabc8a9550062c401
passes unchanged original128 IDs and all owning/fresh/poison/lifecycle gates.
Following127 steps16933ms (7.5001476 tokens/s), prefill15222ms. B70 returns50688000B.
Single small improvement;10 tokens/s remains unmet. Expanded offline paired
shaders,497 native tests/two other shaders and strict Clippy pass.

Next candidate is separately compiled joined work-stealing CPU row scheduling.
Original arithmetic and all failure/cleanup gates remain; source/device pending.

Work-stealing source gates pass497 native tests/two shaders,128 core controls
(three ignored) with both default broadcast and the separate work-stealing
feature,39 Python runner rejection controls and strict all-target Clippy.
Default and selected core behavior preserve exact original-format bits, bounded
private capacities, joined panic/error/cancel cleanup, erased row headers,
fixed physical worker topology and parent affinity. Complete device pending.

Work-stealing sourceceb88d9/artifactfd3d36bcc38f7cf978a74666e2e132751cf9a7698685f754065b4759b1e2148a
passes all original128 IDs and owning/fresh/poison/lifecycle gates:16410ms decode
(7.7391834 tokens/s),15111ms prefill. CPU phase total2614583418ns, down895490110ns,
gate/up1342271873ns. B70 returns50688000B. Single improved sample;10 remains
unmet. Next candidate shares original Q4 activations in bounded workgroup
memory, retaining exact arithmetic. Complete source/device acceptance pending.

Q4 activation-sharing source gates pass497 native tests/two shaders and strict
all-target Clippy on the separately selected work-stealing build. Shared-input
load spans are runtime-checked and unchanged original row/guard probes remain
required on device. Whole-model device acceptance is pending.

Q4 sharing source7c40189/artifact9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2
passes original128 IDs and all owning/fresh/poison/lifecycle gates:16247ms decode,
15464ms prefill. B70 returns50688000B. Next candidate packs four fixed16-lane
Q4 rows into64 invocations, eliminating unused trailing workgroups independently
of the native subgroup width. Reduction order changes; original full acceptance
remains mandatory. Source/device pending;10 tokens/s remains unmet.

Fixed16 Q4 source gates pass498 native controls/two shaders,39 Python controls
and strict all-target Clippy. Tail enumeration covers both64/128 threads and
16/32 native subgroups; every inactive owner index is read exactly once and
active writers are excluded. Fixed16 shuffle peers remain in their own logical
row; shaders validate at256/4096/8192 columns. Device acceptance pending.

Fixed16 Q4 source6a32c25/artifact9a035f53ccf3f35082a1dc6d9ae07f04895f730019c87ab16d57e4d7adb45b1a
passes original128 IDs and owning/fresh/poison/lifecycle gates but is slower:
16700ms decode,16952ms prefill. B70 returns50688000B. Withdraw that geometry
from selection and restore accepted128-lane Q4 source with activation sharing.
Do not assume native subgroup choice or attribute a measured bottleneck from
subgroup bounds alone. Next candidate dispatches only original GPU subset ranks
in shared MoE, retaining private transient counts and all original bounds.
Source/device acceptance pending. Best remains v44 at7.8168277 tokens/s.


### Overnight run paused: 2026-10-08 final checkpoint

User explicitly paused the goal for a debrief. No further optimization runs are
scheduled. The last candidate, active GPU-only MoE dispatch source1a0179b /
frozen02da13e947e1003d365532715a21b71fbcf4a732676b0358b0bc0e006e15ec6b,
passes original128 generated IDs, independent guards, fresh continuation,
poison recovery, full owning erasure and protected lifecycle completion.
127 following steps16616ms (7.6432354 tokens/s), prefill15772ms. It does not
improve the best sample; withdraw it alongside fixed16 Q4. Selected engine source
is restored exactly to7c40189 across native-text and core text code.

Best remains v44:127 following steps16247ms (7.8168277 tokens/s), prefill15464ms,
source7c401893b481d006067d502cc19274b40bc7536f, frozen artifact
9e41bc5ed1f870d4649db476a9e4ee5b03323213ed1d076518ea4cb7c44a75f2.
The first accepted overnight long sample was3.3781088 tokens/s. These are single
fixed-public samples with different selected placement/runtime options, not a
statistical comparison or arbitrary-prompt/long-context/soak acceptance. Target
10 remains unmet. The aggregate now records51 accepted fixtures, including
explicitly excluded diagnostic profiles; stopped/rejected experiments remain
excluded from speed selection.

Selected source gates:497 native controls/two shader suites;128 core controls
(three ignored) in default and separately compiled work-stealing modes;66
runtime host controls, three allocator controls and two logger controls;
39 Python runner controls and strict Clippy. Last candidate also passes complete
protected device acceptance. Restored source matches the already accepted v44
engine exactly; no extra physical run is needed to test identical source.

After all benchmark processes exited, physical B70 usage returned50688000B.
Root Xe classification at2026-10-08T15:45:04.155894+00:00 on boot
665344a9-0e39-4e45-90de-465fa11df0ca / kernel7.0.0-38-generic reports zero
wedged, timeout, GT-reset and memory-fault messages. Public admission resumed:
active=false, waiting=0, paused=false, recovery_required=false. Kuno11434,
direct Ollama11435 and Vashti7771 return HTTP200. No GPU reset or reboot used.
Installed binaries/configuration/model blob/API/catalog/default routing and
Vashti application remain unchanged. No release performed.

Remaining work after a future explicit resume: reduce measured routing fences,
GPU execution/dispatch costs and CPU expert row overhead. The accepted warm
v44 run has zero expert weight uploads/reads, so expert transfer is not the
current dominant measured cost on this fixture. Host-phase timings include
synchronization and overlap and must not be read as pure GPU compute times.
Keep original privacy/correctness/lifecycle bounds; no API or frontend work is
needed for this internal optimization checkpoint.


### Astra request-buffer reuse candidate, 2026-10-08

User authorized another concentrated implementation/benchmark pass toward10
 tokens/s. Owner remains the isolated Kuno checkout. db9f13c adds exact-math
inactive-workgroup Q4 activation-load skipping;324bc92 adds an explicit
request-scoped runtime I/O pool. Original maps/fences/finite checks remain;
private buffers stay owned/charged through the request and receive the original
full two-pass CPU-inspected audit at request close/error. The count1 contract
above is explicitly revised for this request lifetime. No API/frontend changes.

The attempted v44 first-token GPU profile failed the existing private-I/O
bounded runtime gate and is NOT an accepted speed sample. It left B7050688000B
and same-boot Xe classifications zero at2026-10-08T16:02:16.609881+00:00.
No cap was raised. Source work proceeded from accepted v44.

Integrated source tests pass498 native controls/two shader suites,70 runtime
host controls/three allocator/two logger controls and41 Python runner controls.
Runtime strict all-target Clippy passes; integrated Clippy/device acceptance
are pending. Benchmark generation duration will include the final request audit,
with its separate duration and actual reuse counters. Public admission is
paused only for this exclusive benchmark block; restore it at the stopping point.

The first physical same-build control/reuse runs from63a68bb both pass original
128 IDs and full owning/lifecycle acceptance:16108/16357ms respectively. PLE's
independent sanitizer erased the pool each token;126 reuses were insufficient
to exercise the intended optimization. bb678d9 joins that completed boundary
to the request owner. Its corrected run passes every gate but regresses to
23143ms with84673 reuses,381 checkpoints and28ms closing audit. Final proofs
fall2.816→1.179s but routing fences rise3.854→10.396s; this is not selected.
All completed runs leave B7050688000B. Numeric source/artifact/phase receipts
are in`next-request-io-reuse-20261008.json`.

The next candidate a2f1562 batches reused Uniform shader-read→COPY_DST
transitions at a completed checkpoint, removes repeated linear owner searches,
and records scoped readback setup/flush/map-wait aggregates. Original ownership,
full audit and numeric acceptance stay mandatory. Source gates pass498 native
controls/two shader suites,74 runtime host/three allocator/two logger controls;
runtime strict Clippy passes. Physical performance remains pending.


The indexed transition candidate passes original128 IDs but takes22214ms.
INFO interning plus Readback COPY_DST transitions takes21103ms; same-build
reuse-OFF control remains16130ms. Actual full-span Readback clears at the
completed recycle (e9d8032) remove the regression and establish9.3348034 tokens/s:
127 steps13605ms including request-close audit; prefill15314ms. Original128 IDs,
independent numeric/finite guards, fresh continuation, actual cancellation,
poison recovery, owning erasure and protected lifecycle all pass. Map/callback
wait falls9.331→2.454s; this is measured, while the proposed GPU memory-coherence
explanation is an inference. Every completed candidate leaves B7050688000B.
No GPU reset or driver/clock change is involved.

The next candidate preserves original completed checks but uses accept-only
logits/PLE boundaries and one full recycle after the final fault per token.
It also increases exact immutable INFO entries512→1024 under unchanged64KiB
keys and256KiB combined metadata. New runtime API is internal to the isolated
benchmark, not a gateway or frontend contract. Runtime82+3 allocator+2 logger
controls pass with strict Clippy; integrated498 controls/two shader suites and
41 Python controls pass. Physical speed acceptance is pending.


The combined accept-only/INFO1024 candidate3e367ef passes all original device
gates but regresses to16958ms (final receipt authoritative), with5.617s mapped
read waits. It is not selected. Next isolate the same accept-only boundary with
the previously accepted512-entry cache. No new controls or admission limits are
changed. All-family publication and request-close audits remain mandatory.


The512-entry isolated one-recycle candidate8151cf6 also passes every original
gate but regresses (see numeric aggregate). Restore the previously accepted
three completed recycles and512-entry INFO cache. The unused internal
accept-only API does not alter selected execution. Next investigate bounded
nonblocking polling before the original inline map wait, retaining the same
30-second deadline, callbacks and all error acceptance.


Candidate406cf80+a5a0d34 combines bounded bucket lookup with up-to1ms active
native polling only in request-scoped inline readback. Host integration passes
88 unit+3 allocator+2 logger controls. Core math, device placement, full owning
bounds, original map callbacks, completed errors and terminal audits are
unchanged. Normal/default paths retain blocking waits. Integrated native gates
and the protected full device fixture are the remaining acceptance steps.


The combined candidate5df607c/frozenba9844cafee773457a9ac1efa5f5db000fcda2bb22b12b1a72131f9a38dde589
passes twice:127 steps13277/13234ms (about9.57/9.60 tokens/s), with original128
IDs, numeric/finite/owning/fresh/poison/cancellation/lifecycle acceptance intact.
Physical B70 returns50688000B after both runs. One final bounded candidate
removes the SIMD worker's unused decoded-row allocation and constructor GPU
clears that are completely overwritten by full-span Readback copies. Tests and
physical acceptance of those changes remain pending; target10 is still unmet.


Source708f5fe removes only fully overwritten constructor clears;9b51ace makes
CPU decoded scratch lazy for Portable. Runtime88+3 allocator+2 logger controls
and strictClippy pass. Core131 controls (three intentional ignored) pass in
both default and cpu-work-stealing-lab modes, with strict all-targetClippy in
both. The original conservative private budget remains. Integrated498/native
shader validation, frozen normal build and protected physical run are next.


The copy/lazy candidate91cd57e passes all original device gates at13280ms,
with no measurable additional gain over the repeated poll/index build. Its
changes remain byte/arithmetic-equivalent; do not claim a throughput win for
them. The next candidate reduces actual request-scoped zero-tail uploads while
preserving full input/destination ownership and all audits. Warm command-byte
counters are added to distinguish genuine transfer savings from timing noise.
No public inference API or frontend work is involved.


### Astra pass complete: selected source and restored host, 2026-10-08

This final checkpoint supersedes the candidate/pending statements above.
Selected source is `79739d174af41990258a9571c1a8e720d77edc8a`, frozen normal
locked Cargo release artifact
`0df72bd223c7afda807f9fed0793572153bbe40db8ac3fd77ab621f01590d9e6`, with
`direct-device,cpu-fma,cpu-work-stealing`. Its build directory is
`/media/jmicko/faststore/kuno-development/next-offload-build-20261008-v47-zero-tail`.
Original Ollama weights remain read-only and unchanged.

| Selected fixed-public run | Following steps | Decode including closing audit | Decode tokens/s | Prefill |
| --- | ---: | ---: | ---: | ---: |
| `cpu-144-zero-tail-long-v47` | 127 | 13391 ms | 9.4839818 | 15550 ms |
| `cpu-144-zero-tail-repeat-long-v47` | 127 | 13298 ms | 9.5503083 | 15305 ms |

This is about 21–22% faster than the preceding accepted 7.8168277-token/s
sample, but still below the user-approved 10-token/s reference. These two short
fixed-public samples do not establish arbitrary prompt, long-context, quality,
soak or matched-Ollama performance. All original 128 generated IDs, independent
arithmetic/finiteness/guard checks, fresh continuation, actual mid-token
cancellation, poison recovery, full owned-memory erasure and protected lifecycle
checks pass. Timings include the terminal request audit. Both runs enforce
swap zero and core dumps disabled, without persisting real user content.

The main improvement is request-owned I/O reuse with actual completed Readback
clears (`e9d8032`), followed by bounded INFO buckets (`406cf80`) and short bounded
active polling (`a5a0d34`). Selected execution retains all three completed
recycle checkpoints and the 512-entry INFO limit; the correct but slower
one-recycle experiments are excluded. `708f5fe` removes only fully overwritten
constructor clears; `9b51ace` avoids unused SIMD decoded-row scratch. Neither
of the last two has a separately established throughput gain.

`fbae6f7` plus measurement/validation `79739d1` retains request-scoped zero-tail
uploads at the user's explicit preference. Both selected runs encode original
1747845120 bytes as 229218304 copied bytes plus 1518626816 GPU-cleared bytes:
86.8857% fewer upload-copy bytes. The same artifact repeated full correctness.
There is no established additional throughput gain over the immediately prior
13280-ms result. Counters measure encoded command extents, not hardware PCIe
traffic, mapped-store traffic or transfer/compute overlap. This optimization
starts after completed prefill; model loading and unscoped uploads still use
the original path. No startup improvement is claimed.

Final source gates: 498 native controls plus two one-test shader suites;
131 core controls with three intentionally ignored, in both default and
`cpu-work-stealing-lab` builds; 90 runtime host controls, three allocator and
two logger fixtures; 42 Python benchmark/receipt controls. Strict release
all-target Clippy passes for the native lab, runtime and both core feature
modes. Zero-tail host checks include 2048 byte-model cases, original spans,
offsets/padding, signed zero, NaN payloads, alignment and overflow rejection.
Independent source reviews found no blocker. Runtime provenance accounts for
23 modified vendor files and four unchanged published dependency pins/licenses;
composed patch SHA256 is
`e3bb92933bdd4d4cd62e027d0a9a954a3cc3a6f0bc717507158de9788afad386`.

The [numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-request-io-reuse-20261008.json)
records all 14 accepted physical fixtures, including candidates excluded from
speed selection, with exact source/artifact/raw-receipt/run-proof hashes and
numeric phases. Raw public receipts and lifecycle proofs are under
`/media/jmicko/faststore/kuno-development/next-offload-device-20261008`.

All benchmark jobs exited and physical B70 memory returned 50688000 bytes.
Root Xe classification at `2026-10-08T18:16:29.880861+00:00` on boot
`665344a9-0e39-4e45-90de-465fa11df0ca`, kernel `7.0.0-38-generic`, reports
zero wedged, timeout, GT-reset and memory-fault messages. Public admission is
restored: active=false, waiting=0, paused=false, recovery_required=false.
Kuno 11434, direct Ollama 11435 and Vashti 7771 return HTTP 200. No GPU reset,
unbind, reboot, driver or clock change was used. Installed binaries/settings,
model blob, public API/catalog/default routing and Vashti remain unchanged.
The goal stays paused; this explicitly requested pass is complete, with the
speed target still unmet. Required Vashti changes: none.

Next work should start from this selected engine and remaining measured routing/
readback and CPU expert costs, preserving complete output/privacy acceptance.
Warm expert-weight upload/read counters are zero for this fixture; this does
not establish that all transfer costs are irrelevant. Extending zero-tail
handling to loading would require separate correctness and startup measurements.


### Private startup uploads accepted, 2026-10-08

User requested extending the retained zero-tail optimization to loading and
comparing current Kuno with installed Ollama/llama.cpp again. Runtime change
`909ab3b` (reviewed agent source `80f058e`) and instrumentation/provenance
`3b701d4` extend byte-exact prefix/ordered-clear planning to every configured
private upload. Independent constructor/prefill completion and erasure remain;
request reuse is not broadened. Public original weight uploads remain on their
original path. Bounds, full destination ownership, source guards and stage
initialization/audits remain unchanged. No math or model artifact changes.

Frozen normal locked release source
`3b701d4d9bbeb794351502022b21c30734d501eb`, features
`direct-device,cpu-fma,cpu-work-stealing`, artifact SHA256
`26871ebd5f3518f2d8e8e0734dd171527edd6865d68f1e4208e2b55c49a11178`, completes
`cpu-144-startup-zero-tail-long-v47` with all original128 IDs, independent
numerical/finite/guard checks, fresh continuation, actual mid-token cancellation,
poison recovery, all five owning audits and protected lifecycle acceptance.
Swap remains zero and core dumps disabled. Bind1365ms, prefill15324ms,
127 following decode steps13237ms (9.59431895 tokens/s), including9ms closing
audit. This single sample does not establish a further throughput/startup win
against the preceding selected13391/13298ms samples;10 remains unmet.

| Encoded private upload phase | Original bytes | Copied bytes | GPU-cleared tail bytes |
| --- | ---: | ---: | ---: |
| Model bind/probes/constructor | 20643840 | 20643840 | 0 |
| Prefill | 58982400 | 14672064 | 44310336 |
| Warm decode | 1747845120 | 229218304 | 1518626816 |

Prefill private copy payload falls75.12%; bind has no saving in this fixture.
These are encoded command extents, not hardware PCIe measurements. Dense public
weight uploads are excluded; do not describe this as compressing the model.
Bind includes the lab's startup probes, and prefill includes lazy weight use;
neither is a cold-storage end-to-end startup benchmark.

Source gates:498 native controls plus two one-test shader suites;92 runtime
controls plus three allocator/two logger fixtures;43 Python runner controls;
strict release all-target Clippy for native and runtime. The common byte planner
passes2048 byte-model cases plus dense/empty/original-maximum/overflow/counter
separation cases. Parent source review found no blocker. Runtime composed patch
SHA256 `8d32c58ea7f78118e01e57773e4d9f93408d6d723851b18db6735fd7699ab457`;
23 modified vendor files and original dependency/lock/license pins remain.
Exact raw/run/Cargo/artifact hashes and numeric results are in
[the receipt](../apps/kuno/benchmarks/text/receipts/next-startup-zero-tail-20261008.json).

The [updated source comparison](kuno-native-text-count1-hybrid-plan.md#updated-ollamallamacpp-comparison-2026-10-08-research-only)
pins installed Ollama0.35.1 to llama.cpp b11232. First recommended experiment:
cache identical GPU binding objects within the existing request scope, measuring
hits and construction cost. Both backends already pool descriptor storage;
Kuno still rebuilds/validates each binding object, while llama retains descriptor
sets across graph evaluations but updates them per dispatch. Complete binding
memoization is a proposed Kuno optimization, not an established upstream feature
or measured speedup. Further candidates are register-resident recurrent state,
retesting refined CPU integer dots after the disk-fault fix, then overlapping
device-resident expert launch with CPU routing readback. No comparison candidate
was implemented; no new Ollama benchmark or arbitrary-prompt collection ran.

All benchmark processes exited and B70 returned50688000B. Root same-boot Xe
classification at2026-10-08T18:43:11.889831+00:00 reports zero wedged, timeout,
GT-reset and memory-fault messages on kernel7.0.0-38-generic, boot
665344a9-0e39-4e45-90de-465fa11df0ca. Admission resumed with active=false,
waiting=0, paused=false, recovery_required=false. Kuno11434, direct Ollama11435
and Vashti7771 return HTTP200. No reset/reboot or installed service change.
Goal remains paused. Required Vashti changes: none.


### Ranked optimization pass complete, 2026-10-08

User authorized the three ranked experiments from the pinned upstream review.
Binding reuse `37fe3ca843abdb41c5b47a7b39c002c3ec67fee5` is selected. Retained
GDN trial `f73970f9706a2a99835495850c8a77cdb7571213` is preserved in history but
withdrawn by `89ea1bc03ab22c1e834363d46c2736627d03b8da`; its original shader is
restored byte-for-byte. That source also documents retesting the existing refined
CPU integer paths, without new CPU arithmetic changes. Current engine sources
equal the initial binding candidate; the final normal locked rebuild produces
the same executable bytes. No production dependency, API or consumer changed.

All samples use the original public 75-token/128-output fixture, seed 424242,
temperature 0, B70 `intel144`, cap 10 and 16 joined work-stealing CPU workers.
Generation measures the 127 following decode steps and original closing audit.
Binding, prefill, CPU phases and encoded transfer bytes remain separate.

| Run | Change from selected float + binding path | Decode ms | Tokens/s | Prefill ms | Selection |
| --- | --- | ---: | ---: | ---: | --- |
| v48 binding | Initial binding reuse | 11933 | 10.6428 | 15272 | Selected |
| v48 off control | Binding reuse disabled, same executable | 13324 | 9.5317 | 15051 | Control |
| v49 GDN shard | Retain decayed state products across GDN prediction/update | 12086 | 10.5080 | 15809 | Excluded |
| v50 Q8 three | Three-pass residual Q8 activations, original packed weights | 13014 | 9.7587 | 15065 | Excluded |
| v51 Q8 two | Compiled two-pass residual Q8 activations | 12663 | 10.0292 | 15490 | Excluded |
| v52 binding | Original float/GDN selected path rebuilt and repeated | 11836 | 10.7300 | 15887 | Selected |

Selected throughput is 11.66–12.57% above the same-binary off control. This
clears the user's 10-token/s reference on the fixed fixture, without another
Ollama baseline or a claim of broad model/quality/soak acceptance. GDN is one
ordered trial; actual register pressure/spills were not measured, so it provides
no established extra gain. Q8 preparation takes 582–715 million host ns versus
308–310 million for float; whole CPU operator time is 3.11–3.36 s versus
2.38–2.40 s. These overlap GPU work and cannot be summed into decode time.
The original float path remains selected. No single-pass Q8 or changed output
expectation was accepted. Actual residual pass counts are respectively 3 and 2.

The cache retains exact pipeline/buffer identities and ordered offsets/sizes
for at most 256 entries/32 bindings/512 buckets inside the unchanged 256 KiB
metadata cap. Required owner/INFO growth drops the optional cache first. Keys
are fully wiped and retained handles dropped before original full erasure.
Original completion/error scopes and dirty-owner/publication checks remain.
Each selected warm run records 216192 hits, 146831 misses and 256 inserts;
creation is about 1.30 s and lookup 66 ms of host time. The off control records
all zeros. Warm expert-weight upload/read and process disk-read bytes are zero
in all six samples; that is not a claim that all transfers are irrelevant.

Frozen normal locked release artifact for v48 and v52 is
`04f01a611209fb1bb526d48613deade7e2ec7413b9d6b69e7f03de1b9ea61d8b`, features
`direct-device,cpu-fma,cpu-work-stealing`. GDN v49 artifact is
`64669a1b8c63e775eb8f2ca9265edf60d2d6d1146a54f7d8b7d7e0cb7fbbddc5`.
Q8 v50 artifact is
`88fb8aa7cd55e001b57458ba746bf8dc7ddaa614d6404926ea8f489eeeed89b4`, features
`direct-device,cpu-work-stealing`; v51 is
`a18fbb04b6a99ade0640d3596b683aa7924637718aa64d0ff0bb34c569512355`, adding
`cpu-q8-two-pass`. Q8 builds deliberately omit incompatible `cpu-fma`.

All six complete fixtures pass original 128 output IDs, independent numerical/
finite/guard checks, fresh continuation, actual mid-token cancellation, poison
recovery, full owned-memory audits and protected lifecycle completion. Swap is
zero and core dumps disabled. Final request close/audit is 8–9 ms. No private
prompt/image or generated private content is saved. Only the fixed public lab
fixture and content-free numeric/provenance receipts are retained.

Source checks: 498 native controls plus two one-test shader suites; 97 runtime
host controls plus three allocator/two logger fixtures; 44 Python hybrid
controls. The two-pass/work-stealing core combination passes 131 controls with
three intentionally ignored. Existing ordinary/work-stealing core modes remain
unchanged. Strict release all-target Clippy passes for runtime, native standard
and Q8-two builds, and the two-pass/work-stealing core build. Formatting and
whitespace/provenance checks pass. No new external agent review was performed
for this pass. Runtime provenance records 24 modified vendor files and the four
unchanged published dependency pins/licenses; composed patch SHA256 is
`5af5a45bbc3d11c976411a2d89408fc9995fd9de3f751306e1634e3767af0ade`.

The [six-sample numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-ranked-optimizations-20261008.json)
records exact source, frozen executable, Cargo transcript, raw receipt and run
proof hashes. Original raw/run files are in
`/media/jmicko/faststore/kuno-development/next-offload-device-20261008`; normal
frozen builds are sibling `next-offload-build-20261008-v48-binding` through
`next-offload-build-20261008-v52-binding-selected` directories. Longer context,
multiple prompts, nondeterministic sampling, mixed-card execution, general
quality, soak and installed-worker integration are not established here.

All benchmark jobs exited. Physical B70 memory returned 50688000 bytes after
every trial and the final repeat. Root Xe classification at
`2026-10-08T19:21:59.447749+00:00`, boot
`665344a9-0e39-4e45-90de-465fa11df0ca`, kernel `7.0.0-38-generic`, reports zero
wedged, timeout, GT-reset and memory-fault messages. Public admission is resumed:
active=false, waiting=0, paused=false, recovery_required=false. Kuno 11434,
direct Ollama 11435 and Vashti 7771 return HTTP 200. No reset/unbind/reboot,
clock/driver or installed-service/configuration change. The goal remains paused;
this user-authorized pass is complete. Required Vashti changes: none.

### Placement trials and secondary resident calibration complete (2026-10-08)

This supersedes the preceding selected-source checkpoint. Owner source is
`34cdeaf71d8b29ffe186439a35e77b19be1150af` in `codex/kuno-native-text`.
Keep stationary B70 intel144, cap10/GPU limit10, original GPU float/CPU FMA
arithmetic and binding256. New fused projections `048426b`, CPU-resident
balancing `12d1f85`/prefetch-accounting correction `ff45fe5`, packed integer
GPU projections `c0d3420`, and binding512 `eec1797` pass their applicable original
checks but establish no matched warm throughput gain. All remain opt-in labs.

| Complete-model run | Decode ms /127 steps | Tokens/s | Outcome |
| --- | ---: | ---: | --- |
| Warm ordinary v53 | 11548 | 10.9976 | Original path |
| Fused projections v53 | 11677 | 10.8761 | Excluded |
| Ordinary limit10 v54 | 11397 | 11.1433 | Original placement |
| Limit6 accepted repeat | 12712 | 9.9906 | Excluded |
| Packed integer v55 | 11654 | 10.8975 | Excluded |
| Binding256 v56 | 11596 | 10.9521 | Original capacity |
| Binding512 v56 | 12946 | 9.8100 | Excluded |
| Latest ordinary v60 | 11577 | 10.9700 | Selected control |

Binding512 inserts419 then withdraws its optional cache for required metadata
growth inside the unchanged256-KiB cap. The first limit6 wrapper rejected an
incorrect prefetch identity; correcting the count to include CPU-resident ranks
does not relax original output/ownership acceptance. Its repeat passes. First
post-reboot ordinary decode takes20018ms and reads9867210752B, so it is excluded
from matched warm comparisons. Latest warm control reads/writes zero, prefill
14534ms, with44549/60960 resident expert calls (73.08%) and zero expert uploads.
No demonstrated steady-decode RAM shortage justifies adding16GB to current64GB.

Protected secondary operator calibration is added by `da72160`, with compiled
scheduler selection `c1f8f72`, independent exact FMA32 reference `d8245e4` and
mapped/prefetched original CPU sources `34cdeaf`. The original64-lane reference,
numerical tolerances, full ownership/erasure gates and memory budgets remain.
RTX3060 mapped trial completed resident/upload/CPU medians are4.264/43.606/4.939ms
for Q5 and7.300/52.622/5.026ms for Q8. Earlier buffered CPU results are excluded
as a current CPU baseline. Stationary banks avoid upload cost; the isolated
shader does not prove that GPU execution always beats the selected CPU path.

Next priority is a disjoint48-expert/layer NVIDIA bank,32+16 shards totaling9GiB
across48 layers, using optimized native shared kernels and concurrent B70/GPU/CPU
work. **The full two-device engine is not implemented or accepted.** It needs
original-rank membership/source proofs, both-device lifecycle/fault/erasure,
full original-token/numerical/fresh/cancel/poison acceptance, matched total
timings and physical recovery. No private routes/history will be persisted.

Normal locked v60 release artifact is
`21ed239b5c71551ae36e55ea39781b19f8abbe64f0bcfa3ab42ae8367b7e0881`, features
`direct-device,cpu-fma,cpu-work-stealing`. Latest complete-model control passes
all original128 IDs and owning/lifecycle checks. Local calibration passes six
points with two numerical gates and four ABBA samples per point. Source checks
pass502 native tests plus two one-test shader suites,49 Python hybrid controls
and3 resident-calibration rejection controls, strict release all-target native
Clippy, formatting and whitespace checks. Unchanged runtime host passes98 tests
and strict Clippy. Runtime composed patch SHA256 is
`6a122c160a1c58bc0f67a89b98dc5ef89e7df6ea448c27d0b7841eddb723d2d4`.
No broader quality/context/soak, new Ollama baseline or production integration
is established. The approved10-token/s reference is met on this fixed fixture.

Exact source/artifact/raw/run/log hashes are in the
[ten-model-sample aggregate](../apps/kuno/benchmarks/text/receipts/next-placement-kernel-trials-20261008.json)
and [separate operator calibration](../apps/kuno/benchmarks/text/receipts/next-resident-expert-calibration-20261008.json).
Raw public-fixture records remain under
`/media/jmicko/faststore/kuno-development/next-offload-device-20261008`.

All lab jobs exited. At2026-10-08T21:54:25.400396+00:00, boot
`b32654b7-a49b-493d-8305-46da15b6eac6`, kernel`7.0.0-38-generic`, B70 physical
memory is50688000/34242297856B; NVIDIA is372MiB of desktop allocations with no
compute processes. Root same-boot Xe wedged/timeout/reset-failure/memory-fault
and NVIDIA Xid classifications are zero. Admission is resumed with no active
or queued work/recovery latch. Kuno11434, direct Ollama11435 and Vashti7771
return HTTP200; public residency and running lab units are empty. No GPU reset,
driver/clock/reboot, model-file, installed-service/configuration, API/catalog,
Vashti, release or installation change. **Required Vashti changes: none.**

### Stationary secondary model acceptance (2026-10-08)

This supersedes the preceding pending two-device checkpoint. The implemented
stationary NVIDIA bank passes complete original model acceptance but is slower,
so keep it off. Engine source is `3e9f0e4891c75a09ff40eddd47f167e7ae47bef9`;
post-run receipt partition/provenance validation is `bfdd986`. Integrate the
complete backend range after `1e16862`, not just the timing counter commit.
No production native-text/API deployment is part of this checkpoint.

| Complete original75/128 fixture | Decode ms /127 | Tokens/s | Prefill ms |
| --- | ---: | ---: | ---: |
| Secondary accepted first v67 | 15779 | 8.0487 | 48658 |
| Ordinary control v68 | 11587 | 10.9606 | 14616 |
| Secondary profiled v69 | 15378 | 8.2586 | 14433 |
| Same-artifact ordinary v69 | 11503 | 11.0406 | 14626 |

The matched v69 pair uses normal frozen artifact
`0267ae0af32888a0c0d4c86847a016180ffbb81496c2a62a85ef6113c98784e8`, features
`direct-device,cpu-fma,cpu-work-stealing`. Keep intel144/cap10, GPU limit10,
original float/FMA and binding256. Every run passes all original128 IDs,
numerical, whole-owning, fresh, cancel and poisoned-retry gates. Both secondary
runs additionally pass five complete nine-owner closes and five independent
Q5/Q8 probes across slots0/31/32/47. Warm disk reads/writes are zero throughout.
The first secondary prefill is disk-cold; do not treat48658ms as a warm baseline.
Current64GB RAM is sufficient for these measured warm decodes.

The9-GiB32+16 bank holds48 complementary experts per layer. Transient prefill
frequency chooses them; no private history is saved. NVIDIA serves3900 additional
expert calls, reducing CPU calls16411 to12511 and raising combined GPU coverage
73.0791% to79.4767%. Secondary construction takes6036ms in profiled v69, outside
decode. Across2196 dispatches, secondary enqueue/readback/join/reuse cost
0.434/2.442/0.144/1.040 seconds in host wall time. Readback includes GPU waiting;
these are not isolated kernel/PCIe timings. The extra synchronization outweighs
the CPU saving, despite no warm expert weight upload. Reduce these bridges
before selecting a secondary cache from hit rate alone.

Explicit secondary geometry and status branches fix early device rejections;
160-KiB enqueue and separately checked384-KiB post-worker join reservations fix
an overly conservative precharge inside the unchanged1-MiB CPU host cap.
No tolerance, owner/fault or publication check was weakened. Full source,
limits, failures and matched-pair evidence are in the
[internal contract](kuno-native-text-count1-hybrid-plan.md#stationary-secondary-model-acceptance-2026-10-08)
and [numeric aggregate](../apps/kuno/benchmarks/text/receipts/next-secondary-stationary-model-20261008.json).
Checks pass505 native plus two one-test shader suites,50 Python controls,
strict release all-target Clippy and formatting/whitespace checks. Runtime
source/patch is unchanged. Broader quality/context/soak and production
integration remain unverified; no new Ollama tests were run.

All jobs exited; public admission is restored with active=false, waiting=0,
paused=false, recovery_required=false. Final same-boot snapshot at
2026-10-08T22:42:05.444314+00:00 shows B70 at50688000/34242297856B,
NVIDIA373MiB desktop allocations and no compute process. No running lab unit
or public Kuno/Ollama model remains; Kuno11434/Ollama11435/Vashti7771 return
HTTP200. Xe wedged/timeout/reset-failure/memory-fault and NVIDIA Xid counts
remain zero. No installed binary/configuration, model-file, public API/catalog,
Vashti, release or installation change. No reset/unbind/reboot/driver/clock
change. **Required Vashti changes: none.**

### Publisher-aligned full resident MTP acceptance, 2026-10-09,12:48UTC

Input normalization `724ca2f` follows publisher/vLLM global10240-wide RMS;
aligned reference `99386e6` pairs T(c+1)/R(c) at MTP cell c. Historical
`0c1cbfb` dummy-cell/per-stream artifacts are preserved and superseded for
conditioning/quality. The v2 six-row CPU fixture is6,205,952bytes, SHA256
`8ebabbf3e60addf6c9138c699837b2fd4d03bbc5f3495dd3462600884f34490d`.
All12 CPU steps and2 real KV appends replay full vectors bit-for-bit; the
complete original Q8 sidecar hash is reverified. Teacher-forced agreement is
2/3 scored rows, and the initial two provisional drafts both miss. Longer
steady draft quality remains unmeasured; no accepted-token speed is claimed.

Full resident Q8 workspace `a32f241` and exclusive public consumer `1761b06`
pass both RTX3060 and B70. Frozen default source `f652d53`, binary SHA256
`7f6a8e12690b2ad2c68827bc26d00808ebe4657deea28f384a59318a9dc266d4`,
executes4 complete teacher rows,2 actual provisional rows and real target-KV
repair/continuation, in two fresh workspaces per card. Every248320-logit and
10240-hidden vector and exact greedy matches unchanged tolerances. Intel
maximum errors are0.000064612/0.000024796. All42 private owners/11,698,176bytes
erase, and5,011,668,992public bytes release. Default739 native+2auxiliary
tests, strict all-target Clippy/fmt and81 protected launcher controls pass.
The guarded stream reserves64MiB runtime and no CPU projection fallback.

Numerical/provenance records are in the
[foundation receipt](../apps/kuno/benchmarks/text/receipts/daslab-target-mtp-foundation-20261009.json).
At12:48:12UTC actual B70 memory returned to50,688,000bytes, NVIDIA568MiB
desktop-only with no compute process, all HTTP services200, normal admission
and zero classified GPU faults. No API/frontend/install/release change.
Target-prefix repair source `f3899b6` and root public fixture `211b282` are
implemented with747+2 tests/strict Clippy; their physical gate is next.

### Actual target repair, minimal MTP and first dual-GPU run,13:08UTC

`54c846b` fixes the new QSA checkpoint's completed-pointer mismatch while
retaining exact captured physicalowner and originalfaultaudit. Protectedv118
passes wrongfutureinput cases accepting1/2/3tokens and allCPUvectors/greedies
through actualposition4 continuation and complete owningcloses. OneMiBhost/
49,152INFO/zero additionalGPUstate remain the reserved repaircost. Preserved
v117failure fullyrestored the host.

Minimaldraft/follow `241415b`/`da69fe0` passes bothGPUs: allCPUhidden/greedies,
realGPUfollow, exact64KiBpacket/onehostread perforward and nofollowupinput
upload. Existingfullvectorpath and fullowner/fault/padding checks pass too.
Teacher4-row checks take~100–119ms includingpublicchecks, correctness-only.

Separate13-row CPUoracle `d6a7df7` preserves old4/header unchanged; everyfull
legacyCPUlogit vector is bitwiseequal, with actualpreheadHC. Bothshards are
rehashverified. Payload13,445,592bytes SHA256
`80650af818ae00344af5b4b3979e9272cc833c641ea4f80d2934cdb6ffa67e02`.
Native762+2 tests, strictdefaultClippy/fmt and82launchercontrols pass at
`66c1078`. First frozen dualGPUcontroller defaultSHA256
`de3120b39f366914fecb96515c55154a13d314356d38a7377d3108a4074f1ce5`
passes all12actualtarget outputs,6/10actualdraft acceptance, realtarget/draft
state repair and completecloses. Coldnormaldemand speed is1.132565581tokens/s,
10.595412929s,2,311expertfills/4,776,960,000uploadedbytes. Initialprefill/load
is separately10.555s. Targetverification dominates; last fullyaccepted
3-output round is.945s (.816verify/.083draft/.046repair). This isnot a
matchedcontrol or warmedceiling, and30tokens/s remains unmet. Do not replace
the selected12.728conditionalideal preparedcount1 speed with this distinctscope.
No privateinput/API/catalog/frontend/install/release readiness is claimed.

At13:08:29UTC, actualB70memory50,688,000bytes/NVIDIA571MiBdesktop-only,
no compute process, normaladmission and allHTTP200, allclassifiedfaults zero.
The next candidates are fixedthree-column packed dense and prepared expert
inputs; no source timings should be claimed before their physicalgates.

### Fixed three-column dense candidate, 2026-10-09,13:34UTC

Packed dense leaf `25ccfce` (reviewed `8181125`) and protected registration
`7ec4114` add canonical `KUNO_LAB_TARGET_DENSE=verify3` inside the fixed public
B70 target4/repair4/dual-GPU13 modes. Absence retains original projections.
Choice is frozen at construction; actual SUBGROUP16/32 and128-lane admission
precedes request_device. Each decoded original vec4 is reused across1–3 F32
rows; MAX8 owners,16 head tiles, complete finite/fault/padding validation,
MAIN/runtime ceilings and dispatch counts stay unchanged. Actual numeric
`target_verify3_dense_calls` proves selection. No production API/config change.

Default v121 source `7ec4114`, SHA256
`1c0d4ab9ea08dd9bdedbceb5507013ca0e79781b9a7bd08a49b5dfac5e9073a2`,
passes every CPU full-logit/HC vector and exact greedy across all target4
partitions, actual accepted-prefix1/2/3 repair and continuation, and all12 real
accepted outputs in the dual-GPU13 controller. Complete1,378-owner erasure and
physical recovery pass with no GPU faults.767 default native+2 auxiliary tests,
strict all-target Clippy/fmt/diff and83 protected launcher controls pass.

Compiler-idle cold normal-demand controller candidate1.2081tokens/s versus
same-artifact original control1.4074 establishes **no speed gain**; keep this
candidate default off. Both use6/10 actual draft acceptance and identical
2,311 expert fills/4,776,960,000 uploaded bytes. Target phases show3.819s expert
route/enqueue and2.714s router fences in the candidate; weights/dispatch/cache
cost dominates. These ordinary cold timings remain separate from selected
12.728conditional-ideal prepared count1.30tokens/s remains unmet.

Existing completed-chunk I/O cadence is now explicitly testable using canonical
`KUNO_LAB_TARGET_IO=chunk` at `dea5d48`, preserving original complete final I/O
proof and128MiB owning ceiling. Default remains per-layer. Source29 focused
DASLab tests, strict all-target Clippy and83 launcher controls pass. v122 passes
all target4 partitions, actual prefix repair and12 accepted controller outputs
with full CPU vectors, erasure and physical recovery. Cold normal-demand speed
1.1404tokens/s establishes no gain; keep optional. Expert read-call time is
2.319s versus20.5ms upload enqueue, but these read calls alone do not prove
physical disk traffic. A separate process read-bytes counter is now implemented
for the next run. Prepared Verify3 experts and fused draft sweeps are source
checked, with physical gates in progress and no combined speed acceptance.
Required Vashti changes:none. Source/receipts/host evidence is in the foundation
numeric aggregate. No install/release/reset/driver/clock changes.


### Prepared expert and fused draft checks, 2026-10-09,14:10UTC

Frozen v123 default source `a99b63b`, SHA256
`bdc9f04f889ef6662b2d4377410d1a99f81c5eacc5487c137c7931f45ff9ca9b`,
passes prepared expert target4 partitions, actual prefix1/2/3 repair and all12
real dual-card controller outputs under unchanged full CPU bounds. Maximum
target4 logits/carry errors are0.0007592/0.0002008; this candidate changes
activation arithmetic explicitly and is not F32 bitwise equivalence. It erases
1,379 owners/826,703,872bytes. Cold controller0.8632tokens/s during parallel
CPU checks establishes no speed gain. Its actual process counter records
3,039,633,408 disk-read bytes inside generation and5,069,881,344 during load
plus initial prefill. These are actual physical reads, distinct from4.777GB
uploaded and source read-call duration. Keep the candidate optional.

Fused MTP sweeps pass both GPUs: complete teacher/provisional CPU vectors,
minimal/follow, real KV repair and two fresh42-owner closes. Four minimal
teacher steps execute664 explicit dispatches (166each), versus original268
each. Follow executes167. No combined throughput acceptance yet. Privacy,
physical memory, normal admission/services and unchanged classified faults
recover after every run. Root785 native+2auxiliary tests, strict default
Clippy/fmt/diff and85 launcher controls pass at this checkpoint.

Direct existing-view public fills `1090a4a`/`a6fcdf5` eliminate temporary GPU
source copies while retaining original precharges, telemetry and full gates.
The immutable choice is `KUNO_LAB_TARGET_FILL=direct-public` in the same fixed
public target modes; default remains original. New counters separate first
touch from eviction and count actual queued direct batches/bytes. Original64
barrier leaf `b75af2c` moves only the unused shared store/barrier into the
16-wide branch, preserving exact original shuffle arithmetic;84 generated
variants and202 independent mixed-scale host reduction cases pass. Root790
native+2auxiliary tests and strict default Clippy pass. Physical checks are
in progress; no new selected speed is claimed. Exact-bank admission and a
healthy one-shot all-private warm replay remain research/in-progress work.
Required Vashti changes:none; no production installation/API/release change.

### Original arithmetic and direct fills,2026-10-09,14:37UTC

Frozen v124 default source `a6fcdf5`, SHA256
`38296f8f909cb6791cc249a5180d2fdd2f833c64c6206cfcbb0636d4c19d1c5b`,
passes all14-format original fast operators on both cards, all target4 partitions,
actual accepted-prefix1/2/3 repair, and fused MTP teacher/provisional/minimal/
follow/KV repair on both cards. Independent full CPU vectors and original
bounds remain unchanged. Full private erasure, actual GPU memory recovery,
HTTP/admission and zero new classified GPU faults pass. Root790 native+2auxiliary
checks, strict default Clippy/fmt and85 protected launcher controls pass.
The initial root-created operator output directory caused PermissionDenied only
when publishing the numeric receipt; it is preserved. Fresh user-owned v124b
passes both cards, with no shader/device fault.

A compiler-idle same-artifact ordered cold controller pair improves1.50087 to
1.74852 accepted tokens/s (16.5%) using checked direct public fills and fused
draft sweeps, with unchanged arithmetic/routing/full CPU checks and actual
state repair. Both accept6/10 drafts, upload2,311 experts/4,776,960,000bytes,
and show zero evictions: all misses are first touches. Enqueue time falls from
21.53ms to1.16ms; target expert-phase time falls from1.885s to1.319s. Actual
timed disk reads are131,072bytes/0 respectively. These are distinct cold
demand runs, not an ABBA or steady-state result; selected12.728 conditional
ideal count1 remains unchanged.30tokens/s is still unmet.

The prepared expert comparison gives1.37538tokens/s, with598,609,920 actual
disk-read bytes and1.274s source read time versus.436s in direct+fused.
Paging differs, so no arithmetic speed conclusion or default selection follows.
Full CPU gates still pass. Numeric/source/helper hashes and preserved failure
are in the foundation receipt. No capacity expansion is justified by this
first-touch-only fixture.

A construction-fixed one-shot public warm replay is source-checked at `4b93c99`
with registration `bafd9c0`: same banks/IDs/admission/ledger/epochs remain owned;
all private and source pages are erased and independently verified before real
GDN/QSA/PLE state returns to0. The existing completed projection proof is kept
unchanged; original token0 prefill proves actual start0/count1/cursor1 again.
Transient binding/INFO caches are cleared, so only model weights and compiled
kernels are warm. Root796 native+2auxiliary checks and85 launcher controls pass;
physical warm replay is pending. Optional fixed-public selector is
`KUNO_LAB_WARM_REPLAY=once`, rejected outside the speculative13 fixture.
No production API/config/install/release change. Required Vashti changes:none.

### Actual warm-bank replay,2026-10-09,14:55UTC

Frozen v125 default `bafd9c0`, SHA256
`f61b547ed7b9a4a9b9e7daa98674de550b95468db30fef26922ebffc61d7023c`,
passes the real one-shot warm replay with all original CPU13 vectors/greedies,
actual target/draft repairs, complete retained1,378-owner826,638,336byte erase
and independent final closes. No ledger/epoch/poison reset occurs. Same public
banks remain owned; transient binding/INFO caches are cleared. Compiler-idle
cold1.20749/warm2.34472 tokens/s; cold reads2,117,292,032 physical process bytes,
warm reads12,288 and makes zero expert uploads/zero weight reads. Initial
prefill is separately6.265s/.504s. This is real normal demand, not forced
experts or oracle routing. Target warm enqueue/fence work dominates.

Prepared expert replay also passes full gates with1,379 owners: warm2.34773
establishes no material improvement; cold had source-check CPU overlap and
2.29GB paging, so is correctness-only. Existing Verify3 dense warm replay
passes2.61000,6,016 actual dispatches,zero expert uploads. None is a matched
ABBA selection or30tokens/s acceptance. Selected conditional ideal12.728 count1
is a different path/scope and remains unchanged. Both GPUs/services/admission
recover after every run, classified GPU faults stayzero.

Full Verify3 logical-finite fusion `fe196fa` with registration `6f581b4` is
constructor-fixed/defaultoff; canonical `KUNO_LAB_TARGET_DENSE=verify3-fused`
sets both immutable flags.48-byte INFO limit is checked before device creation,
unchanged rounded charge; bounded last-tile scan excludes every active complete
matrix interval while covering the full logical complement. Original physical
padding/source/status/input/fault/final owner checks remain. Whole target4
partitions/continuation pass v126 with original full CPU vectors, max logit
error0.00006676, carry0.00002862, all actual fused counters/erasure/recovery.
Standalone operator initially rejected a new fault case at its stale inventory
guard; preserved v126failure has no accepted operator receipt/GPU fault. One-file
`5eb8e82` corrects only the compiled inventory guard with tested exact original/
fused branches; physical retry and actual repair/controller are next.
Root803 default native+2auxiliary tests, focused11 registrations and strict
Clippy/fmt,86 launcher controls pass before this pure inventory correction.
Required Vashti changes:none. No API, installation/release or GPU reset.

Corrected v127 default source `5eb8e82`, SHA256
`7f3fe8a0a9a24c6543784cfb1937c8bde3db7e6507cf612ee363b59318c2baba`,
now passes the dedicated finite operator on both B70 and RTX3060:60 normal+
210 invalid/sticky-value cases each, all1–3 counts, independent CPU and old-GPU
results, source/full padding, complete private erasure and allocator release.
No external candidate finite scan hides the fused result. Original actual
prefix1/2/3 target repair/continuation also passes whole CPU vectors at v126.
All faults remainzero and memory/services/admission recover. Full controller
finite timing is next. No tolerance or GPU driver/reset change was made.

Source review distinguishes the historical full32 DAS failure from this new
ordinary target path. Old v95 consumed raw-score GPU top10, while current
nonhybrid MAX8 uses `router_topk_guarded` on actual GPU scores with original
rounded global softmax. DAS13 route divergence was never captured; therefore
it remains an unresolved historical failure, not proof that today's target path
fails beyond13. QSA selects all14 chronological tokens here, and144 bank slots
cannot evict before14×10 experts; neither is an evidenced cause. Next numerical
extension should separately compare current target-only row13 against existing
complete CPU32 gold, then capture exact DAS13 routes if needed. Keep the old
failure and unchanged bounds; current acceptance still covers13 positions only.

Compiler-idle v127 complete-controller finite fusion now passes both fullCPU13
passes, actual target/draft repairs, warm-reset and final owning audits. Cold
normal-demand1.31906/warm2.96368 tokens/s; warm6016actualfuseddispatches,zero
expert uploads,16,384 physical process read bytes. Compared to the earlier
ordered warm Verify3-only2.61000 this is promising, but not matchedABBA.
No30tokens/s claim. Host enqueue/fence still dominates; separate pure
`direct-device` comparison is being implemented for these public target modes.
All normal services/admission/actual memory and zero GPU faults recover.


### Pure-direct target acceptance,2026-10-09,15:31UTC

Frozen pure `direct-device` source `834a299`, SHA256
`cef8f61dd7dd0125efd93dca5d74a09827626c4ac9c0d28db30817edadea38c0`,
passes target4 partitions/continuations, actual prefix1/2/3 repairs and both
fullCPU13 controller passes. All original arithmetic, full vector bounds,
actual state repair and complete private/source erasure remain. Actual inline
submission is checked before device/model work and recorded afterward. Ordinary
default, draft allocation bounds and public API are unchanged.

Compiler-idle cold1.41475/warm3.18530 accepted tokens/s; warmzero expert uploads,
weight reads and physical process disk bytes. Warm12 outputs take3.76730s, with
2.11111s router fences and.93779s attention/GR enqueue. This ordered comparison
is no matchedABBA or30tokens/s acceptance. Warmreset and final1378-owner
826,638,336byte audits pass; actual memory/services/admission recover, classified
GPU faults stayzero. Source agent default806+2auxiliary tests (oneignored),
pure-direct target registrations/runtime policy tests, strict default/direct/
all-feature Clippy/fmt and87 launcher controls pass. Source review accepted.

Next source-only work is a dedicated fixed-public target request-I/O scope:
constructor-charged4MiB/4096 FIFO metadata, batchedINFO and inline readback on
the B70 target client only. NVIDIA draft/default/cap10 paths must retain their
existing configuration; no global environment mutation or private persistence.
It is not implemented or physically accepted in this checkpoint. Required
Vashti changes:none. No production install/release or GPU reset.


### Dedicated target request reuse,2026-10-09,15:55UTC

Runtime `0cfdbe0`/native permit `9f3b8cc` with registration `12f9cc7` are now
physically accepted, superseding the preceding source-only note. Frozen pure
`direct-device` SHA256
`70ce929d488a5aea1892f51b3016156de69096225db48fad09fbdcba3e617cda`
passes all target4 request-mode partitions after real token0 prelude, actual
row1/count3 prefix1/2/3 repairs/continuations, and fullCPU13 cold/warm controller
checks. Actual scoped begin/end is proved against admission/runtime ownership.
Scopes1/2 close with original full audit before warm reset/final close.

Compiler-idle cold1.54364/warm6.63953 accepted tokens/s. Warm12 outputs take
1.80736s,zero expert uploads/weight reads,4,096 process disk bytes. Request begin
and7.449ms full closing audit are inside timing. Warm24,266 bindinghits/3,526
misses and24,133 INFOhits/267 misses establish actual reuse; original complete
fault checkpoints remain411. Checked binding/INFO/reuse deltas are process
global; checkpoint counts are admission-local. Router phase falls to.87469s,
attention/GR enqueue to.28737s. This is an ordered improvement over v128,
not matchedABBA or30tokens/s acceptance. Cold includes2.226GB physical disk
reads/2.637s source-read time and cannot establish steady-state math speed.

Target-only constructor captures128MiB I/O,4096-byte INFO/batched inline maps
and fully charged4MiB/4096 FIFO metadata. No global environment mutation;
Nvidia draft remains64MiB I/O/16MiB HOST, original generic cap10 unchanged.
Native complete HOST stays22,806,528bytes, no new MAIN/GPU owner. Full retained
and final1378-owner826,638,336byte erasure, physical memory/HTTP/admission
recovery andzero classified GPU faults pass. Root814 default+2auxiliary tests
(oneignored), pure-direct3 runtime/policy tests, strict default/direct all-target
Clippy/fmt and88 launcher controls pass; independent source review passes.

Next is an opt-in original-F32 grouped expert scratch reduction: skip shared
stores/adds for unused members while preserving each live64-lane tree and every
unconditional barrier. Kernel, both-card multirow operator and immutable model
plan work are isolated; none is physically accepted here yet. No API/frontend/
production installation/release/reset change. Required Vashti changes:none.

### Paused overnight checkpoint,2026-10-09

The user paused the goal after the DASLab/Strata pass. Latest integrated source
is `bbe630a`, after live kernel `7ab49b8`, model policy `13d9929`, independent
multirow probe `a13ebc3` and style-only `91dd416`. Latest **physically tested
model artifact remains12f9cc7/v129** with SHA256
`70ce929d488a5aea1892f51b3016156de69096225db48fad09fbdcba3e617cda`.
No further GPU benchmark or new artifact freeze was started after the pause.

The same v129 artifact's optional prepared expert mode passes all target4
partitions, real prefix1/2/3 repairs and complete CPU13 cold/warm controller
checks. Compiler-idle cold1.73282/warm6.77829 accepted tokens/s; warm12 outputs
take1.77036s, zero expert uploads, weight-read time and process disk bytes.
Full closing request audit remains inside timing (13.596ms). Warm and final
1379-owner826,703,872-byte erasures pass; all actual target/draft repairs retain
the original vector tolerances. Four-row logit/carry max absolute errors are
0.00059509/0.00027323. This arithmetic uses two residual Q8 activation planes,
not original-F32 bit equality. Its ordered2.09% gain over6.63953 is too small
to establish a material win; original F32 remains the normal comparison path.
No matchedABBA, private/production or30token/s acceptance is claimed. Exact
numeric/source/hash/host records are in the foundation receipt's
`public_target_request_reuse_prepared_v129` entry.

Source-only live-member work preserves the original64-lane F32 reduction tree
and every unconditional barrier while skipping unused expert-member shared
stores/additions. Selection is immutable/default-off; all48 actual geometries
are checked at completed boundaries. The new operator covers all nine mixed
profiles and counts1/2/3, old/new bits, independent CPU vectors, faults, inactive
planes and owning tails. The public controller gains checked process-global
readback attribution counters; these are not isolated kernel timings.
See the normative [all-row plan](kuno-native-mtp-all-rows-plan.md).

Root integrated source passes829 native tests plus2 auxiliary tests (one
existing opt-in ignored),89 protected launcher tests, strict default/pure-direct/
all-feature all-target Clippy, formatting/whitespace and independent source
review. Child immutable integration and probe each pass821
native+2auxiliary, strict default/all-feature Clippy and focused tests. The tiny
probe let-chain lint repair passes its five affected tests with unchanged
coverage. Exact final integrated lint/build checks are recorded in the paused
foundation receipt. Physical live-member correctness/performance is pending.

At16:21:01UTC, boot `b32654b7-a49b-493d-8305-46da15b6eac6`, kernel7.0.0-38:
B70 memory50,688,000/34,242,297,856 bytes, NVIDIA502MiB desktop-only and zero
compute processes. Admission is open/idle with no queue/recovery latch; Kuno,
Ollama and Vashti HTTP all200. Classified Xe/NVIDIA faults remainzero. No owned
lab processes remain. Snapshot is preserved in the paused foundation receipt.
Installed binaries/services, canonical Vashti work, API/config/catalog and
release selection remain unchanged; no reset/unbind/reboot/driver change.

On resume, refresh sudo only when the shared shell is idle; freeze fresh default
operator and pure-direct model artifacts from this integrated source. Run both
card live-member operator gates, then target4/repair4 and same-artifact CPU13
control/live cold/warm comparison with compilers idle. Existing full public
references/tolerances/actual repairs/erasure/memory recovery remain mandatory.
Then prioritize remaining GPU/CPU handshakes: v129 warm has no expert transfers,
so its ~0.875s router phase includes queued preceding math/map/audit overhead,
not isolated top-k computation. Do not promote the old count1 raw-score GPU
router to MAX8 without matching rounded-global-softmax ranking semantics.
Longer-context acceptance remains separate. Stop here until the user resumes.

## RAM-source full-model acceptance and stop checkpoint, 2026-10-09

The user requested ending this pass to conserve usage. Frozen v143 source
`f0e088b28038646ad34fcc358a63344c0bbbb5d0`, binary SHA256
`ddbd8c7dba93944146203c72f6fb235eaddca49ae5d4b53478ee5d69ef9794fe`,
passes both GPUs'80 standalone router cases and the B70/3060 full original
CPU13 target/MTP controller with actual immutable RAM expert sources.

Integrated source includes CPU source consumer/plan/attachment `f2d383b`,
authenticated512 GPU routing `9538601`, independent operator `01dad20`,
corrected selected-fault fixture `fad1b21`, and actual allocation alignment
`f0e088b`. The prior139–142 full-model setup failures are retained. v142 stage
markers prove preparation and full warm reset succeed; the later importer had
confused conservative65,536-byte envelope rounding with actual Vulkan alignment.
Layer0's90,316,800-byte gate page is valid at smaller driver alignment. The
sealed native page now carries that measured alignment into registration;
logical/full physical bounds, source identity and ownership checks remain.
`be3e0e8` preserves the original authored public failure after mandatory close
attempts; failed attempts never publish accepted receipts.

The accepted CPU13 warm run uses deterministic PHYSICAL placement0..143 to
exercise both tiers; original routing/weights/tokens are not substituted.
Twelve real target-verified continuations take2.845782894s, **4.216765806tokens/s**.
There are0 demand expert uploads/bytes and1,081,344process disk-read bytes.
Actual rank-member routes:2,590VRAM and6,530RAM; these are not unique reads or
PCIe byte counts. Direct GPU RAM reads still cross PCIe. Preparation fills6,132
original experts/12,535,603,200bytes; RAM attachment takes41.434147362s.
All complete original CPU13 logits/carries, exact greedies and actual repair pass.
The target final close audits1,378owners/826,638,336private bytes. All432host
pages detach and CPU-zero/free36,147,609,600actual backing bytes; live backing0.
This is not ordinary language200, a matched speed comparison, independentCPU200
quality, private-request or30tokens/s acceptance. Frozen GPU512 has standalone
routing acceptance only, not full-model acceptance. Q8 draft remains the default.

Both-card router cases preserve original selected ID/order and the full owner
checks; F32 probabilities remain explicitly experimental under2e-6absolute/
2e-5relative tolerances and deferred exact original-ID audit. This is source
addressing across512experts, not512VRAM residents. Each card audits480operator
owners/188,743,680bytes. The corrected sentinel fixture expects original
MISSING4, derives actual selected IDs independently and changes no kernel guards.

Exact source/CPU13/operator/failure/numeric recovery records are retained in
`/media/jmicko/faststore/kuno-development/native-strata-v143-host-reference-20261009`
and `native-strata-v143-router-operator-20261009` under that same parent;
compact checked-in data is in the native-Strata optimization receipt.
The aligned integrated source passes945native tests,2auxiliary shader tests,
1existing ignored test, nine focused plan regressions and strict pure-direct
all-target Clippy; formatting/whitespace pass. Later candidate checks are recorded
separately below and must not be substituted for these physical source hashes.

At21:23:38UTC final recovery confirms unchanged boot/kernel, B70 used50,688,000B,
no NVIDIA compute owners, normal/unpaused/no-latch Kuno, allthree HTTP200, and
classified Xe/NVIDIA faults0. No GPU reset, driver replacement, service binary
installation, API/frontend/catalogue or release changes. Required Vashti changes:
none. Preserve the canonical checkout's unrelated/uncommitted coordination work.

The next isolated candidate skips reductions/barriers for a frozen source
partition that does not own an actual group. It keeps original active64-lane
arithmetic,30-group grids, complete scratch reset/finite/status/rank fold and
all-family/deferred-original audits. It is default-off and needs whole-model
CPU13/GPU acceptance before ordinary code200 timing. Do not assume the ratio of
scheduled inactive workgroups translates directly into a throughput multiplier.

### Final inactive-partition source checkpoint (no new GPU trial)

Integrated candidate source`389fb6b388b450f05e0bf05e74859ecc1956af6f` contains
leaf`612b7f6` (original peer commit`e80dd80`) and parser/launcher`6413b16`
(original`9eeca92`). Explicit canonical`KUNO_LAB_FROZEN_PARTITION_EXIT=1` requires
`KUNO_LAB_TARGET_SOURCES=frozen-host-public`; absent/default remainsfalse.
Actual completed source geometry is compared against the requested flag before
numeric receipts. It composes with CPU routing or the separately reserved
`device-frozen-public` router. No grid/count trimming, quant arithmetic change,
new owner, budget or default-mode change is included.

Parent integrated checks pass45native public-parser/reference tests,
30frozen-source/routing/shader tests,102protected-launcher Python tests, strict
pure-direct all-target Clippy, formatting and whitespace. The leaf independently
passes47grouped kernel/guard tests and4source tests, including all actual DAS
profiles' checked/unchecked WGSL+SPIRV lowering and preserved active arithmetic
structure; peer source review passes. No CPU-inference features were selected.
This candidate has **no physical kernel/full-model/performance acceptance**.
The physical143 hash remains the accepted alignment/RAM-source control.

Next pass: extend the local protected comparison helper to select this explicit
flag (it currently emits the control), then validate fullCPU13 GPU512/partition
candidate against original vectors/greedies/real repair and complete owning
closes. Only after that gate, benchmark normal exact public code200 and a matched
control. Record prompt/seed, accepted token counts, measured actual routes,
readbacks/disk reads, source setup and physical RAM/VRAM recovery; do not present
rank-member route counts as unique PCIe bytes. The user's stop request leaves
no GPU trial or automatic work running. The optimization goal remains paused.

### v144 partition-exit physical acceptance and code200 ABBA (2026-10-10)

The user explicitly resumed the optimization goal and confirmed idle GPUs. No
engine source changed this pass; HEAD remains `cab8ec5` with candidate source
`389fb6b` and documents only. The protected comparison helper was extended
with four explicit fast labels (`warm-fast-host`, `warm-fast-gpu-host`,
`language-code-fast-host`, `language-code-fast-gpu-host`) that select canonical
`KUNO_LAB_FROZEN_PARTITION_EXIT=1` only together with frozen-host sources.
Helper sha256 `00abc51fdacc16535b42360c91a11e6a31d4f7e98dd4bf9af095b6f1a890ee23`
(prior `dc72bded...`); all twenty existing labels were oracle-compared
byte-identical and the protected launcher accepts every new environment
(`test-run-native-strata-fast-labels-20261009.py`, 24 trials, zero failures).
A fresh direct-device build was frozen from `cab8ec5`:
binary sha256 `0f14244454f8c9046c80ca330da00f8bd7eb08069e97c38100d702bec02ff158`
(build `native-next-v144-build-20261009`). This is a new binary; no v143
physical acceptance was transferred to it.

Four same-artifact whole-model CPU13 gates all passed complete vectors/carries,
exact greedies, real prefix repair, both source tiers (VRAM 2590 / host 6530,
dispatches 4320), zero expert uploads, clean closes and baseline recovery
(Intel 50,688,000 B, no classified faults, admission normal):

| Gate | Trial | Router | Exit | tokens/s |
| --- | --- | --- | --- | --- |
| 1 | warm-host | false | false | 4.249597 |
| 2 | warm-fast-host | false | true | 4.522345 |
| 3 | warm-gpu-host | true | false | 4.610436 |
| 4 | warm-fast-gpu-host | true | true | 5.478617 |

Gate 3/4 receipts carry positive actual `gpu_router_calls` (384) with full CPU13
acceptance, establishing the whole-model GPU512 routing acceptance that was
still pending at v143. The composed candidate is the fastest accepted warm
configuration (+29.2% over the CPU control).

Normal public code200 was then measured with matched same-day ABBA runs on the
CPU-routed frozen-host pair (35-prompt first-code fixture, seed 424242,
temperature 0, 200 emitted, 199 timed continuations, identical 69 rounds and
138/130 draft proposals/acceptances in every run):

| Run | Trial | tokens/s |
| --- | --- | --- |
| A1 | language-code-host | 6.700894 |
| B1 | language-code-fast-host | 6.951033 |
| B2 | language-code-fast-host | 6.963472 |
| A2 | language-code-host | 6.599900 |

Controls mean 6.6504 (stdev 0.0714), candidates mean 6.9573 (stdev 0.0088):
**matched ABBA improvement +4.61%** from the exit kernel alone, well outside
the control noise band. Ordinary generation is now ~6.6-7.0 tokens/s, up from
the older 2-4 tokens/s artifacts.

New defect found and preserved: `language-code-gpu-host` fails deterministically
(two failed evidence dirs `native-next-v144-lang-abba-a1-20261009` and
`...-a1-retry-20261009`) with the content-free authored error
`InvalidFormat("native model chunk rejected")` at the first
completed-native-projection-boundary call (`native_public_fill_counters`,
daslab_language_probe.rs:684) immediately after
`activate_public_frozen_host_batch_router`. The same activation passes in the
warm probe only because its replay performs a fresh prefill commit before the
next bare boundary call; the CPU-routed language control completes normally
(`native-next-v144-lang-cpuhost-bisect-20261009`). This is a language-probe /
frozen-router integration defect, not an exit-candidate failure: it reproduces
on the control label and the exit kernel passed every gate including the
GPU-routed composed gate 4.

All runs ended with admission normal, Intel at 50,688,000 B, no NVIDIA compute
owners, and unchanged classified fault counts. Required Vashti changes: none.

Next concrete steps, in order:
1. Pinpoint and fix the post-router-activation completed-boundary violation
   (one of prove_outer/checks in resident_model_batch.rs:1646-1672) so the
   language probe can begin request I/O immediately after activation; focused
   CPU tests first, then rebuild, then repeat the four gates plus a GPU-routed
   language ABBA to measure the composed fastest combination on code200.
2. Then continue toward the 30 tokens/s objective with the packed IQ3 block
   decoding and row-grouping kernel investigation (MIT-noticed Strata
   inspiration), keeping the Q8 draft and one-change-per-experiment discipline.

### Language router-activation defect: root cause and source fix (2026-10-10)

The deterministic `language-code-gpu-host` failure diagnosed to completion. The
authored error `InvalidFormat("native model chunk rejected")` was misleading:
`lab_oracle_attempt` (resident_model_batch.rs:1746) poisons the model and
replaces any operation error with that generic message, hiding the real
rejection. A separate diagnostic worktree (`codex/kuno-router-lang-diag`,
instrumented boundary checks plus an oracle catch-arm print) reproduced the
failure and revealed the original errors: admission
"request I/O scope rejected", then vendor
"private reuse begin requires empty exclusive owner".

Root cause: `activate_public_frozen_host_batch_router` uploads the 48-layer
routing table through client `write_private`, which charges the private I/O
exclusive-owner ledger. Beginning a request I/O reuse scope requires that
ledger completely empty (transient_io.rs `begin_reuse_configured`). The warm
probe never observed the dirty window because its replay performs a fresh
prefill commit - whose readback checkpoint sanitizes the ledger - before
`begin_public_target_request_io`; the language probe begins request I/O
immediately after activation.

Fix (commit `8c7d838` on codex/kuno-native-text): a new admission method
`checkpoint_public_router_activation` runs the existing checkpoint/sanitize at
the end of both `activate_public_frozen_host_batch_router` and
`activate_public_warm_batch_router`, so a completed activation leaves the
exclusive-owner ledger empty. Focused checks pass: 45 daslab_, 30 frozen and
66 router tests, Clippy all-targets -D warnings, formatting and whitespace.

Diagnostic validation (not acceptance evidence): the instrumented build
`native-next-diag-routerlang-build4-20261010` (source `ded5afc`, binary
`293bb2c0...`) completed the full `language-code-gpu-host` run in
`native-next-diag-routerlang-run4-20261010`: 3552 actual gpu_router_calls,
199 continuations, complete real target-greedy verification and owner erasure,
diagnostic preview 7.1716 continuation tokens/s. Diagnostic runs 1-3 preserve
the failure progression; build/run directories are under
/media/jmicko/faststore/kuno-development with the diag- prefix.

Next session, in order: freeze a fresh v145 build from `8c7d838`, repeat the
four same-artifact whole-model CPU13 gates (no acceptance transfers from
v144), then run the GPU-routed language ABBA
(language-code-gpu-host vs language-code-fast-gpu-host) to measure the
composed fastest combination on code200. After that, continue toward the
30 tokens/s objective with the packed IQ3 block decoding and row-grouping
kernel investigation. No GPU trial is running; the host was left at baseline
with admission normal.

### v145 fix acceptance: gates re-run and GPU-routed language ABBA (2026-10-10)

With user authorization the optimization goal resumed. The resident Kuno image
worker held 16.8 GiB on the B70 (retained image model from earlier
generation, no idle expiry by design); the helper's idle guard correctly
refused the first gate attempt (directory
`native-next-v145-gate1-cpu-control-20261010`, source.json only). The user
authorized restarting kuno.service, which cleared the worker and returned the
card to the 50,688,000 B baseline; the next image generation pays one model
reload.

A fresh v145 build was frozen from `fd39cd1` (ledger fix `8c7d838`): binary
`f8f23497bb318c12c91b74bfa4d3fe9bb7b70af32b0bdbeeca373a7ad445a554`
(`native-next-v145-build-20261010`). No acceptance was transferred from v144;
all four whole-model CPU13 gates were re-run on the new artifact and passed
complete vectors/carries, exact greedies, real repair, both source tiers
(VRAM 2590 / host 6530), zero uploads and baseline recovery:

| Gate | Trial | Router | Exit | tokens/s |
| --- | --- | --- | --- | --- |
| 1 | warm-host | false | false | 4.248511 |
| 2 | warm-fast-host | false | true | 4.640277 |
| 3 | warm-gpu-host | true | false | 4.542013 |
| 4 | warm-fast-gpu-host | true | true | 5.253543 |

The previously blocked GPU-routed language ABBA then ran on the same artifact
(language-code-gpu-host control vs language-code-fast-gpu-host candidate;
35-prompt first-code fixture, seed 424242, temperature 0, 199 timed
continuations, identical 69 rounds, 130/138 accepted drafts and 3552 actual
router calls in every run):

| Run | Trial | tokens/s |
| --- | --- | --- |
| A1 | language-code-gpu-host | 7.512568 |
| B1 | language-code-fast-gpu-host | 8.250165 |
| B2 | language-code-fast-gpu-host | 8.425974 |
| A2 | language-code-gpu-host | 7.577262 |

GPU-routed controls mean 7.5449 (stdev 0.0457); composed candidates mean
8.3381 (stdev 0.1244): **matched ABBA improvement +10.51%** from the exit
kernel, and **+25.4% over the v144 CPU-routed control** (6.6504). The frozen
GPU-routed language path itself - deterministically broken before the fix -
completed with full real target-greedy verification, owner erasure and
baseline recovery on the frozen artifact. Ordinary public code200 decode is
now 6.65 (CPU control) to 8.34 (composed candidate) tokens/s, up from the
2-4 tokens/s of the older artifacts.

All runs ended with admission normal, Intel at 50,688,000 B, no NVIDIA
compute owners and unchanged classified fault counts. Required Vashti
changes: none. The kuno service restart is the only service action this
pass, explicitly user-authorized.

Next concrete step toward the 30 tokens/s objective: the packed IQ3 block
decoding and row-grouping kernel investigation (MIT-noticed Strata
inspiration in pinned sycl/src/kernels/cuda/iq_kernels.dp.cpp), one change
per experiment with the same gate discipline; the Q2 draft remains a separate
later experiment.

### IQ3 kernel investigation: Strata study and experiment design (2026-10-10)

The pinned Strata reference (fb58e0d, v0.1.41,
sycl/src/kernels/cuda/iq_kernels.dp.cpp) was studied against the lab's
current IQ MoE kernels. Findings:

1. **Decode-once split is already present in the lab.** Strata's central
   trick - Split<TY>::load/apply, decoding each weight block once and
   applying it to every activation column, bitwise-equal by construction -
   corresponds to the lab's verify3 prepare/project pair: Q8 activation
   planes are prepared once per packet/layer, and the project kernel applies
   each decoded weight word to all three frames (THREE_PACKED_VALUE
   workgroup planes). The packet dimension needs no new work.
2. **Grid staging is the measured anti-pattern.** Strata stages the format
   grid in workgroup memory behind STRATA_GRID_SLM and measured it a LOSS,
   leaving it off: "copying the grid into each of ~1,200 work-groups per
   call costs more than the lookups save (IQ2_S 52.5 -> 62.7 us, IQ3_S
   60.8 -> 70.0, IQ3_XXS 60.3 -> 62.1; unitrace 2026-10-01)". The lab does
   exactly this: grid_source (iq_moe_integer.rs:52-73) emits a 128-case
   switch that stores every grid word into workgroup memory per workgroup
   invocation, used by both the single-token and verify3 kernels
   (iq_moe_integer_verify3.rs:108-144).
3. **Byte-addressed decode vs packed loads.** Strata loads containing
   words once (load8_a2/load4_a2, page-safe unaligned handling) and
   extracts bytes in registers. The lab's decode (projection() packed
   expressions) performs three scattered NAME_byte word loads per quartet.
   The SWAR sign trick (iq3_signed_word) already matches Strata's
   swar_sub4; the loads are the difference.
4. **Row grouping.** Strata: STRATA_EXPERT_LANES=8 lanes per row, 32 rows
   per workgroup, 10 block-calls per lane for a 2560 row. The lab:
   subgroup-per-row with block stride = subgroup width (80/width blocks
   per lane; B70 subgroup width must be measured, expected 16 -> 5 blocks
   per lane, 8 rows per 128-lane workgroup). Geometry change is a
   measurement question, not a proven win.
5. **Mixed formats confirmed.** The GSQ-RCO IQ3_S file keeps one layer's
   routed gate/up experts in IQ4_XS (Strata comment line 419), matching
   the lab's all-profile mixed lowering tests.

Phase accounting from the v145 composed run (B2): verify is 72% of code200
elapsed (16.9 s of 23.6 s; draft 4.9 s on the 3060 is a separate floor);
within verify, attention_and_gr_enqueue ~11.1 s carries the MoE work,
completed proofs ~4.5 s and router audits ~0.8 s are lab correctness
overheads. The IQ kernels live in the dominant phase but attention shares
it; the exit candidate's 2.4 s savings confirmed the MoE slice is real.

Experiment queue, one isolated change per pass with oracle proof, then
whole-model gates and matched ABBA (no acceptance claims before all four):

- **E1 (first): grid to read-only storage buffer.** Replace per-workgroup
  SLM staging with a read-only storage buffer of original_grid_words(dtype)
  bound at launch; remove GRID_DECLARATION/GRID_INITIALIZE substitution.
  Bitwise-equal by construction: same words, same lookups, storage class
  only. Files: iq_moe_integer.rs grid_source + dense BINDINGS strings,
  iq_moe_integer_verify3.rs binding assembly and launch path, WGSL/SPIR-V
  lowering tests asserting on the switch text. Expected gain: removes a
  128-case switch and 256 word stores per workgroup per launch; Strata's
  measurement implies a double-digit percent kernel gain for IQ3_S.
- **E2: packed per-block decode.** Load the containing words once per
  (row, block) and extract index/high/sign bytes in registers, replacing
  the per-quartet scattered NAME_byte loads. Same values in the same
  order; oracle must pass unchanged.
- **E3: row-grouping geometry.** Evaluate subgroup-per-row against
  8-lane/row grouping under the existing scratch/reduction guards;
  measure, do not assume the scheduling ratio.

No engine source was changed in this investigation; the working tree is at
9020b5e and clean. The next session should implement E1 through the lab's
full discipline: oracle and lowering tests first, clean commit, fresh
build freeze, four whole-model gates, then the code200 ABBA with the
composed GPU-routed+exit configuration.

### Correction to the IQ3 investigation: the operative path (2026-10-10)

Deeper tracing corrects the section above before implementation. The E1
experiment as scoped there is wrong, and no engine source has been changed.

1. **The operative frozen MoE kernels are NOT the iq_moe_integer WGSL
   family.** The accepted warm/language runs use
   `grouped_gate_up_frozen`/`grouped_down_frozen` (grouped_moe.rs, CubeCL,
   the original64 F32 contract) decoding each weight through
   `quant_matvec::packed_value` -> `quant_daslab` per value. The
   iq_moe_integer/verify3/prepared family - including the WGSL SLM grid
   staging - is the separate integer family, inactive in every accepted run
   (`prepared_verify3_experts: false` in all language receipts). Implementing
   E1 would optimize a family that does not run and would show ~0% in the
   operative ABBA; the staging remains a latent anti-pattern only if that
   family is ever enabled.
2. **Decode-once is already present in the operative kernel** across the
   member dimension (grouped_moe.rs: "Decode each original weight only ONCE,
   reused across RHS"). The actual decode inefficiency is per-VALUE block
   metadata: each of the 32 lanes covering an IQ block independently
   re-reads the same scale/sign/high-bit words from the weight array, so
   block metadata is loaded ~32x redundantly per block per row. An
   arithmetic-identical amortization (stage a row's block metadata in
   workgroup memory once) is the real "packed IQ3 block decoding" candidate,
   and it lives in quant_matvec/quant_daslab plus the grouped_moe inner
   loop, not in the WGSL templates.
3. **The VRAM placement is a frozen fixture contract**: public_host_bank_completion
   validates the resident set is exactly experts 0..CAPACITY (0..143),
   documented as the diagnostic placement exercising both tiers. A
   route-aware (hot-expert) placement is a fixture-contract change with its
   own opt-in, validation, acceptance-description update and re-gating - a
   separate experiment from kernel work, though potentially the largest
   lever if host-source PCIe streaming dominates (71.5% of routes hit host
   sources; existing receipts do not instrument PCIe bytes, so this remains
   unproven).
4. **Decision for the next implementation pass**: operative-path per-block
   metadata staging (arithmetic-identical, oracle-testable) vs the
   hot-placement fixture variant (measurement first, larger protocol
   change). The weight_read_ns receipt counters track only demand uploads,
   which are zero in the frozen design, so neither experiment's payoff can
   be predicted from existing receipts; the placement variant would decide
   the memory-versus-compute question directly.

### HANDOFF: preserve-placement experiment implemented, NOT yet run (2026-10-10)

User selected the hot-placement measurement over kernel decode work. Source
landed and committed; the physical trial has NOT run yet (user hit usage
limit). Everything a fresh session needs to go straight to the trial:

## What was implemented (commit 4fad434 on codex/kuno-native-text)

`KUNO_LAB_FROZEN_BANK_PLACEMENT=preserve-public` keeps the warm replay's
actual routed VRAM residents (left by replay=0's demand uploads) instead of
the canonical CPU13 diagnostic set 0..143. Architecture verified sound: the
five frozen source partitions cover ALL 512 experts with immutable host
pages; VRAM residency is a per-expert overlay (by_expert slot < 144) that
check_mapping already validates against the actual bank - the language path
runs arbitrary LRU resident sets through the same machinery every trial.

Files changed:
- `resident_model_batch.rs prepare_public_host_sources`: new
  `preserve_routed_bank: bool` parameter; canonical 0..143 branch only for
  context-13 warm replay AND NOT preserve; otherwise `complete_public_host_banks`.
- `daslab_speculative_probe.rs`: allowlist `KUNO_LAB_FROZEN_BANK_PLACEMENT`;
  parse canonical `preserve-public`, requires frozen sources + warm replay;
  new `preserve_routed_bank` input; receipt field
  `frozen_bank_placement` (`preserve-routed` | `cpu13-diagnostic`);
  `timing_scope` string names the actual placement; tier check: canonical
  requires host>0 AND vram>0, preserve requires vram>0 only (host may be 0 -
  that IS the measurement).
- `daslab_language_probe.rs`: passes `false` (behavior unchanged).
- `protected-lab-exec.py`: `bank_placement_scope()` early check in
  `memory_limit` (no-mode envs reach the fallback without the speculative
  block, so the flag must be validated up front) + pop block in
  `selected_target_policy` so the permitted-set check never sees it.
- Local helper labels: `warm-preservebank-host` (placement only) and
  `warm-preservebank-fast-host` (placement + exit kernel, composed).
- Tests all pass: 46 daslab / 31 frozen / 66 router, Clippy -D warnings,
  fmt, 103 Python launcher tests, 26-trial helper env oracle test.

## To run the experiment (next session)

1. Freeze a fresh build (`build-direct-target.py native-next-v146-...`) from
   HEAD - the placement variant is source-new, no acceptance transfers.
2. Host gates for the variant, in order (sudo in the Demons shell; fresh
   output dirs each):
   - `mkdir native-next-v146-preserve-gate && sudo -n python3
     /media/jmicko/faststore/kuno-development/run-native-strata-public-protected-20261009.py
     <build> native-next-v146-preserve-gate warm-preservebank-host`
     Expect: FULL CPU13 vectors/carries/greedies/repair acceptance (placement
     changes only WHERE weights are read from, never the arithmetic), receipt
     `frozen_bank_placement: preserve-routed`, both tier counters recorded.
   - Optional control side-by-side on the SAME artifact:
     `warm-host` (already run for v145) and/or `warm-preservebank-fast-host`
     for placement+exit.
3. Read the decision from the receipts: compare `accepted_tokens_per_second`
   and `frozen_source_routes` host vs vram counts. If the preserve runs are
   much faster with host_routes near 0, PCIe host streaming was the
   bottleneck and VRAM-resident routing is the roadmap (then consider
   increasing bank capacity beyond 144, or hot-set policies for production).
   If speed is unchanged, decode/ALU dominates and the operative-path
   per-block metadata staging work (see the correction section above,
   quant_matvec `packed_value` block-metadata redundancy) is the next kernel
   target.
4. Record in the receipt and checkpoint with the same discipline: artifact
   hashes, actual route counts, exact rates, limitations, and keep terminal
   recovery verification (Intel 50,688,000 B, admission normal, fault counts
   unchanged).

## Do NOT

- Do not treat warm-gate numbers with the preserve placement as the accepted
  CPU13 control: the canonical control is the 0..143 placement; the preserve
  variant is a diagnostic that answers the memory-versus-compute question.
- Do not delete the tier-check branch: the canonical gates still require
  both tiers, which is what keeps the accepted fixture honest.

Host state at handoff: baseline 50,688,000 B, admission normal, no GPU trial
launched for the placement variant yet. Working tree clean at 4fad434.

### Resumed shader experiments and usable stop checkpoint, 2026-10-10

The user resumed after GLM/free-model work, then explicitly asked for a few
remaining tests and a stop with the system usable. GPU and CPU work are now
stopped. No source acceptance authorizes further GPU occupancy or installation.
The existing optimization goal remains paused.

Review found no blocking defect in the router-activation ledger fix8c7d838 or
preserve-bank diagnostic4fad434. Prior v145 GPU-routed code200 ABBA remains the
accepted repeated normal-language result:7.54493 control /8.33810 exit candidate
continuation tokens/s (+10.51%). The new results do not reach20 tokens/s.

**Preserved placement diagnostic:** fresh v146 source1d0d7c8, binary SHA256
`d5a449cd87ce6a12b24de5611adddf47b179006af9607f5f010206ac7ef16549`, build
`/media/jmicko/faststore/kuno-development/native-next-v146-resume-build-20261010`.
Both original CPU13 full-vector/carry/greedy/real-repair gates and full closes
pass. Preserve replay12/2.095918815s=5.72541 tokens/s, all9120 rank-member routes
in VRAM; canonical12/2.783044584s=4.31182,6530 host/2590 VRAM. Both upload0 experts.
These are short diagnostic rates, not normal-generation throughput. Source
checks may have overlapped the canonical control, so the ordered difference is
not an isolated causal estimate. Normal language already preserves residents
left by its actual prefill; this is not a new language placement optimization.

**Operative module-private table experiment:** leaf120e94d integrated ase054d70,
then guarded public controller/launcher and element-mutation proof7364dc8. Frozen
v147 source`7364dc8df4dc645bb3df588f7a52b6e1b9751646`, binary SHA256
`4a6f3c657941a50200f751028d3b32871c552219de02a530ceb9dfdb135e21e1`, build
`/media/jmicko/faststore/kuno-development/native-next-v147-private-build-20261010`.
Exact flag`KUNO_LAB_FROZEN_TABLES=private-public` requires original frozen public
sources and explicit partition exit. Actual completed source policy is recorded
and compared; default unchanged. Original grids, weights, F32 arithmetic order,
64-lane reductions, ranks, source owners/bindings and down kernels are unchanged.
Naga tests establish removal of per-access whole-array Function stores for all
actual mixed IQ profiles; they do not establish physical driver traffic savings.

Software checks passed:48 daslab parser/reference tests,5 private-table focused
tests (two overlap the48),104 protected-launcher Python tests, strict
pure-direct-device all-target Clippy, formatting and whitespace checks. The
negative element-store fixture checks recursively resolved Private pointer bases
in the Naga AST. Actual macro-generated runtime kernel name was independently
verified to match the guarded compiler selector.

Full original CPU13 hardware gate`warm-private-gpu-host` passed:12 verified
continuations/2.339232169s=5.12989 tokens/s, actual private-table flagtrue,
6530 host/2590 VRAM routes, zero demand uploads. Original complete logits,
carries, greedies, real repair and all target/draft/host-source closes pass.

One same-artifact ordered normal public code200 pair:

| Trial | Timed continuations | Seconds | Continuation tokens/s |
| --- | ---: | ---: | ---: |
| GPU router + partition exit control |199|25.577232506|7.780357|
| Same + module-private public grids |199|26.109952147|7.621615|

Both emit200 tokens,69 rounds,130/138 draft acceptance,46376 host/56344 VRAM
rank-member routes and0 expert uploads. Seed424242, temperature0, thinkingoff,
original DASLab target + unchanged Q8 MTP, compiled35-token merge-sorted-lists
prompt. Setup/hash/attachment excluded from generation timing. Real target
verification/repair and inside-timing request close remain included. No ABBA,
prose/soak or independent CPU200 reference gate; do not claim a precise causal
regression. This experiment shows no speed gain and remains default-off.

Exact content-free source, ownership, counters, closes and recovery evidence
are in the native-Strata receipt and these local output directories:

- `native-next-v146-preserve-gate-20261010`
- `native-next-v146-canonical-control-20261010`
- `native-next-v147-private-gpu-gate-20261010`
- `native-next-v147-code-control-a1-20261010`
- `native-next-v147-code-private-b1-20261010`

All are under`/media/jmicko/faststore/kuno-development`. Both original target
shards are fully hashed before each frozen-host trial; public reference files
and all numerical tolerances remain unchanged. No request/generated text,
raw output tokens, routing IDs, prompts, images or credentials were recorded.

**Two clean source-only follow-ups, not integrated or GPU accepted:**

1. Workgroup-table commit`3594b76de116fc72f536cc296572a281d54231f2`, based7364dc8,
   at`/home/jmicko/.codex/worktrees/kuno-constant-array-probe/vashti`.
   Deduplicated original public IQ grids are initialized by literal lane/value
   stores after uniform inactive exit and one barrier.5/6/12KiB total SLM;
   original arithmetic/owners retained. Three focused checked/unchecked
   shader/SPIR-V/immutability/geometry tests and strict direct Clippy pass.
   See that commit's`docs/kuno-native-public-workgroup-tables-leaf.md`.
2. IQ4 down-scale commit`b56af6af0ef70f30401ae0b881fc4178c9807951`, based7364dc8,
   branch`codex/kuno-iq4-down-scales`, at
   `/home/jmicko/.codex/worktrees/kuno-iq4-down-scales/vashti`.
   Twenty original IQ4_NL scales decoded once into80B SLM; column/member
   order and64-lane dots unchanged. Q2/default text unchanged. Three focused
   shader/SPIR-V/original C-fixture bit/dot/owner tests, strict direct Clippy,
   formatting and independent peer review pass. See`iq4_down_scale_tests.rs`.

After fresh authorization/availability, integrate these leaves with guarded
parser/launcher/receipt selection; neither has a registered runtime flag yet.
Resolve their shared Geometry/Sources policy changes carefully, retaining
private/workgroup mutual exclusion and orthogonal scale-sharing composition.
Re-run CPU proofs, freeze a NEW clean artifact, run full original CPU13 gates
and actual owner/physical recovery, then compare normal generation on the same
artifact. Do not reuse v147 acceptance for new kernels. Do not weaken guards,
replace selected experts, or move public tables into uncharged INFO/MAP storage.
Do not increase144 slots blindly:160 slots plus bank rounding,8GiB cache and
margin exceed the current31GiB ceiling. Zero demand uploads does not measure
all PCIe traffic; observed normal language has45.15% host routes.

**Final live state at2026-10-11 01:27:07UTC:** same boot, B70baseline50,688,000B,
no NVIDIA compute owners (570MiB graphics), admission idle/unpaused with no
recovery latch, Kuno/Ollama/Vashti HTTP200, all classified GPU fault counts0.
All owned benchmark and CPU jobs ended. Services/binaries/public APIs were not
changed; no reset/reboot/install/release occurred. Required Vashti changes:none.
Canonical frontend checkout remains untouched. Preserve the pre-existing native
untracked Python bytecode cache rather than treating it as source changes.
