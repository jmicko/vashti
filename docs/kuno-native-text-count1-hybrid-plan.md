# Count1 CPU-miss / resident-GPU plan

Initial source proposal against root `4195238`, 2026-10-06; the refined
reservation and executable source contract below supersede the original three
page estimate and overlap sequence. Source establishes no speed benefit.
The scope is the existing fixed-public whole-model
lab, with a completed native prefill and count1 continuation. MAX8 prefill,
production requests, automatic placement and other GPUs are outside this slice.
The [approved decode plan](kuno-native-text-decode.md) remains the contract.

Use one explicit public CPU-miss cap, initially one calibration-supported cap
from 0/1/5/10 with AVX2/16 proven physical cores. Select that policy only at the
existing completed model boundary, then reject every hybrid count other than1
before launch. The baseline remains the ordinary native GPU path and its actual
resident cache. Every resident selected hit stays on GPU; only selected misses
may go to CPU. Private miss counts never select a shader specialization.

The smallest ownership addition is one reusable hybrid workspace on the original
client/arena/admission/sequence-poison owner, allocated in the constructor plan
before private work. It borrows the original file descriptor and public packed
layer tensors; it adds no expert bank, weight conversion or host payload cache.

| Resource | Proposed additional reservation | Current aggregate / resulting bound |
| --- | ---: | --- |
| MAIN combined packet | 32,768 B: 512 router + 2560 mixed-input + fault, full owning tail | |
| MAIN CPU partials | 131,072 B: full ten-rank logical capacity, distinct from GPU PART | |
| MAIN owner/readiness packet | 32,768 B: ten runtime owner/index/ready records and zero tail | |
| MAIN total | 196,608 B, three pages reused across 48 layers | 818,249,728 → 818,446,336 B; 1376 → 1379 owners |
| Private application host | Shared 1,048,576 B CPU/bridge peak, with exact live capacities checked | 11,075,584 → 12,124,160 B within existing 12 MiB; 458,752 B remaining |
| Public CPU jobs | Existing 40 MiB cap; maximum actual ten Q8 experts: 35,840,000 B | Original packed bytes, released after each joined layer |
| Owned runtime I/O | Additional 1 MiB conservative component peak | Existing 64,000,000 / 64,090,112 B envelopes → 65,048,576 / 65,138,688 B, below 64 MiB |

The host reservation is not CPU1 MiB plus another bridge1 MiB. Reuse the existing
64 KiB router host allowance for the combined packet: its32 KiB map lease plus
12,288 B guarded values/header fits after actual-capacity validation. Drop the
raw map and audit inside the original admission readback before CPU jobs start.
During CPU execution, the core operator owns input/activation/results/workers
under its existing1 MiB guard. After join, those workers/inputs/activations/jobs
are destroyed and only Partials remains; padded upload/owner staging must be
charged against the same new1 MiB live envelope. Recheck every actual capacity,
including retained per-layer rank/completion headers, against the existing
32 KiB per-pending and256 KiB metadata limits. CPU's1 MiB cap alone does not prove
the combined envelope: a narrow admission seam must subtract live caller input,
packet and split headers from the shared allowance before core private fill,
then reuse the released worker allowance for bridge staging. The original
numeric API/math can remain unchanged. This combined proof is a code gate.

The I/O component counts 32 KiB packet read +128 KiB CPU upload +32 KiB owner
upload +128 INFO records, each with 512 B source and target, =327,680 B. Duplicate
audit staging and two 8 KiB slab tails duplicated for inspection give
2×327,680 +2×2×8192 =688,128 B, rounded to 1 MiB. It must be derived from
the actual APIs/dispatches before allocation. Router readbacks already drain the
preceding layer; that completed proof permits scratch reuse and bounds the new
live debt to one layer. The final layer uses the original whole-model proof.
Do not multiply reuse by48, credit sources early, remove a checkpoint or raise
either independent64 MiB hard cap. GPU miss source precharges remain the real
original subset of the existing maximum40 MiB window. CPU public packed jobs and
GPU public upload sources can coexist; record their separate host peak bounds.

The concrete layer sequence is:

1. Replace only the hybrid branch's existing router packet with a combined owning
   read of the exact prepared GR mixed input, original logits and global fault.
   Keep `router_topk_guarded` normalization/ties/probability bits unchanged.
2. Inside the existing native registry, construct and validate a canonical
   all-ten staged BankSlots plan against the real bank. Its upload IDs identify
   actual misses; select the first capped miss ranks in original rank order,
   including zero-probability ranks. Discard that uncommitted classification
   plan, construct a canonical plan for all GPU hits plus unchosen GPU misses,
   and validate it against the same untouched origin. Protect every GPU hit
   before choosing victims. Only this second plan may fill real slots.
3. Retain an opaque hybrid selection tying the immutable original ten-rank wave,
   CPU miss proof, GPU subset, actual pending generation and original bank
   instance/full physical handles together. Revalidate it before every fill,
   launch and final preflight. CPU-assigned misses stay absent from the bank.
4. Start the scoped CPU orchestrator outside admission locks. The parent alone
   fills real GPU misses and dispatches resident/fetched GPU experts. Reuse the
   original64-lane packed arithmetic, sign-stable SiLU and complete MAX8/MAX80
   scratch owners with a public fixed ten-rank/grid envelope and runtime subset
   guard. GPU-only/CPU-only endpoints skip their empty job, never use fake slots.
5. Join CPU, check Partials count/shape/finite values/full zero suffix and original
   file consistency, then upload the distinct CPU owner. One rank merge reads
   exactly one ready unweighted GPU or CPU partial per original rank and performs
   the unchanged rank0..9 probability multiply/add. Full finite/status/tail folds
   validate both actual owners, even when probability is+0/-0, and OR only the
   existing global fault. Enqueue original shared expert and GR combine unchanged.

There are two concrete adapter blockers to handle explicitly. The numeric lab's
`Split::new` assumes a contiguous miss suffix, which real router ranks do not
have. Existing grouped `GroupMap`, `Pin` and `fold` require every one of the ten
ranks to be GPU-ready. A narrow hybrid subset/map/readiness adapter and merge
are needed; filtering the original route plan, weakening canonical validation,
writing fake GPU-ready bits or calling a scalar cache entry are invalid seams.
All count/rank/slot/owner data stays runtime-private, with no pipeline/cache keys.

Extend existing routed Pending metadata, not its authority. CPU jobs are joined
per layer and scratch consumers are completed before reuse; no48-layer CPU
vectors/jobs remain live. Every pending pins the original bank and staged ledger
through the existing all48 routed Acceptance. Keep the whole-model output/fault
and second PLE proof,36 GDN/12 QSA permits, PLE preflight and GR/shared queue
validation before any family publishes. Only that all-family acceptance may
publish the48 GPU-subset ledgers and causal position. CPU success alone cannot.

On CPU panic/cancel/read error, GPU fill/enqueue/map/fault failure, stale origin,
missing owner or failed cleanup: abort the start gate, join every scoped worker
outside locks, set the shared sticky poison and invoke original drain/MAIN
erasure/I/O audit. Failed cleanup retires/quarantines the context. There is no
GPU-to-CPU fallback, slot rollback, accepted partial output or same-owner retry.
Drop only marks poison and wipes owned headers; it does not reenter admission.

Before full-model integration, gate the one-layer adapter with real primed
hits/noncontiguous misses, both actual Q5/Q8 down formats, every rank position,
K0/K1/K5/all-misses and signed-zero probabilities. Compare raw gate/up,
activations, unweighted partials and original-rank output against independent
packed64-tree and separate original GPU references at existing tolerances.
Read full owning CPU/GPU/status packets, gaps/tails and public bank sentinels;
prove abandoned/altered/stale plans and late faults cannot publish. Then test
sequential fresh baseline/hybrid whole-model owners at count1, original fixture
IDs/logit gates, cancellation/poisoned retry/fresh recovery and all original
owning audits plus the three new pages. Timing follows those gates. Receipts
retain only aggregate counts/errors/times/erasure, never rank traces or vectors.

Source anchors: `resident_model_batch.rs` Plan/router/all-family commit,
`resident_batch_io.rs` completed owning readback, `resident_moe_batch.rs`
staged plans/physical pins/Pending/Acceptance, `bank_slots.rs` canonical source
validation and `cpu_expert.rs` guarded positional reads/scoped execution. The
existing fixed-public calibration alone does not connect this model seam.

## Implemented one-layer source slice

The isolated `codex/kuno-one-layer-hybrid` source adds an executable numeric-only
acceptance adapter; compilation and device acceptance remain owner gates. It
does not construct a whole-model hybrid workspace or establish a speed benefit.
The helper is
`cpu_gpu_expert_probe::run_staged_hybrid_acceptance(client, file, graph, limits,
retired, cancel) -> Result<StagedHybridReport>` on the original inspected file,
bound graph and configured client. No model path reopen, hardware/route controls,
new private diagnostics, timing sample or NumericPermit is returned.

The opaque child `staged_hybrid::Selection::prepare(&ExpertBanks, &[usize; 10],
cpu_miss_cap)` borrows the real bank. It retains the original BankInstance,
unchanged StagedSlots origin, canonical all-ten classification plan and a
separately canonical GPU subset plan. CPU assignment uses the first capped
actual misses in original rank order. Every selected resident hit remains on
GPU in its original slot, protected before choosing a GPU miss victim. Its
immutable split supplies the actual CPU IDs, actual GPU slots and complete
runtime rank owner/index map to the accepted producer. Neither plan advances or
publishes the source. Empty CPU/GPU endpoints skip that empty job and use no fake
slot or readiness bit. Revalidation precedes fill and launch, follows joined
consumers and merge, and accompanies the final physical/owning preflight. The
original bank owner remains alive through erasure/audit, then is disposed before
the completed report; a modified physical bank cannot be reused against its
unpublished original ledger.

Eight declared cases use actual original Q5_0/Q8_0 down tensors, group 0,
signed-zero vector 1 and AVX2/16 workers. For each down format, odd ranks
1/3/5/7/9 are actual noncontiguous misses at CPU caps0/1/5; the cap10 case has
all ten actual misses and no GPU route. Priming uses completed original GPU
consumers and a real committed local source ledger. K0 is the real primed
resident-hit/fetched-miss GPU baseline. All ten original probabilities and ranks
remain in the unchanged rank0..9 merge, including positive/negative zero.
Independent packed64-tree/core arithmetic, raw gate/up, activations, unweighted
partials and the separate original all-GPU reference remain required at the
existing tolerances. Full owning maps/status/tails, shared fault, file checks,
public bank sentinels, 18-page MAIN erasure and transient I/O audits precede
aggregate publication. Errors/panics use the original join, sticky poison,
drain/erasure and retirement path. This first eight-case slice does not claim
every rank/owner permutation or whole-model finality.

There is no new MAIN allocation: the existing full-page 16-slot candidate and
reference bank pair remains 134,217,728 B, with 18 owning private buffers totaling
884,736 B and the existing 256 MiB aggregate physical gate. The existing 40 MiB
public source cap, 4 MiB private host envelope and 1 MiB bridge allowance remain.
Both retained plans, actual canonical-vector/state capacities, fixed split and
selector headers are conservatively checked within the existing 256 KiB planner
allowance (shared staged state is deliberately counted twice); no 48-layer
accumulation or model-budget credit is asserted. Host test sources cover actual
noncontiguous classification, full-bank victim protection, both empty endpoints,
foreign bank instance/origin, stale revision, altered canonical plans/ownership,
capacity rejection, signed-zero nonfinite rejection and all-or-error eight-case
publication. No host test, compilation or device execution was run by this
source-only agent. Root must compile/test/review and register a strict protected
numeric-only mode, then accept both cards before any model seam or timing work.

### Protected one-layer numeric registration

The separate `resident-staged-hybrid-expert` mode accepts only card and mode,
requires `KUNO_PUBLIC_RECEIPT`, `KUNO_LAB_TRANSIENT_IO=1` and
`KUNO_LAB_OWNED_ZERO_AUDIT=1` before device/model access, and rejects model/native
policy selections. The tile/profile defaults may be pinned to `8`/`chunk`;
runtime INFO slab configuration remains the independently protected I/O policy.
No route, miss-cap, worker, probability or input override is read. Main opens and
binds the original GGUF once, then calls the fixed helper on the same configured
client, limits and shared retirement owner. It rechecks file/cancel/retirement
after all eight complete owning cleanups before writing a new public receipt.

Scope `fixed-public-original-staged-cpu-gpu-expert-numeric` contains exactly the
eight independently checked configurations and existing exact bounds. Reports
with timings, unexpected splits, incomplete audits, nonfinite errors or exceeded
actual-capacity bounds reject the aggregate. No selected IDs, masks, slots,
ownership maps, vectors, permits or tensors are serialized. Cache publication,
whole-model acceptance, timing, performance, production and automatic split
policy remain explicitly false. File-write failure can leave an incomplete
public file but emits no completion marker. Formatting and source test coverage
do not establish compilation or physical acceptance; root owns those checks.

## Approved bounded count1 model source contract

This source slice starts from `91d4dd9`. Ordinary `Config::new` reserves no
hybrid pages and keeps the exact1376 owners /818249728-byte legacy MAIN plan.
Only explicit `Config::reserve_count1_hybrid() -> Result<Config>` on an initial
public tile8 configuration reserves a CPU128KiB owner
(25600 logical F32 words) and an owner32KiB page (244 logical U32 words).
The existing4097-word /32KiB router packet holds2560 prepared mixed values,
512 original router logits, one fault word and zero remainder. Consequently the
additional MAIN reservation is163840 bytes and two owners, giving1378 owners
and818413568 bytes. The original three-page estimate above is superseded.

Reservation alone selects no CPU work. Completed original native prefill is
followed by the existing opaque model boundary proof and a public cap0 or10
`ResidentModel::set_lab_count1_cpu_misses(executor, cap, cancel) -> Result<usize>`
setter. `native_cpu_miss_cap() -> Result<Option<usize>>` observes defaultNone or
the selected public cap. Every other cap and every hybrid count other than1
rejects. Completed public tile1/8 policies are supported; default
native prefill/decode and existing modes continue through the unchanged GPU
path. The protected whole-model CLI and strict receipt described below are
now source/host-tested; physical whole-model acceptance remains separate.

The first executable seam intentionally joins the original AVX2/16 physical
worker operator before parent-only GPU filling and enqueue. It does not claim
CPU/GPU overlap or speed. All CPU public packed jobs release before the original
GPU source window; each separately remains within40MiB. Only joined Partials
survive into padded CPU/owner upload. This supersedes the proposed overlap in
step4 above; a later scoped overlap experiment needs its own measured proof.

The shared additional private host peak remains1MiB. Before CPU private fill,
the caller subtracts actual live guarded mixed/logit storage, split/map and
selection headers from that limit through a subtract-only operator API. After
join, actual Partials capacity plus padded CPU and owner encoding and all live
headers must fit that same1MiB. No CPU job/vector is retained in48 Pending
receipts. One additional1MiB runtime component is conservatively precharged
alongside the existing64090112-byte maximum, giving65138688 below64MiB.
The additional component explicitly counts128 full2048-byte INFO envelopes
(the actual additional clears need96 records), four ownership/inspection copies
of the160KiB CPU+owner upload, two32KiB packet readback copies and four8KiB
slab tails:1015808 bytes. Rounded1MiB leaves32768 bytes. This supersedes the
initial688128-byte estimate, which did not spell out full INFO ownership.
The mandatory audited router dependency drains the preceding layer before
scratch reuse; the final original whole-model proof covers the last layer.

The owner packet has header `[1,cpu_count,gpu_count,0]`, then80 triples
`[owner,index,ready]`. Active GPU records use owner0/index=original rank/ready0,
with real readiness only in the existing GPU STATUS. Active CPU records use
owner1/index=compact CPU ordinal/ready1 only after joined Partials validation.
The70 inactive records, gaps and owning tail are zero. The GPU804-word map
contains only canonical GPU groups with original-rank members, while all ten
original probability bits remain in PROB. A true empty GPU plan is allowed only
for ten actual CPU misses. No fake slot/readiness or zero-probability omission
is allowed. Packed gate/down math and64-lane trees remain unchanged; the new
merge reads one actual owner in rank0..9 order and full fold checks every owning
view, status, finite value, gap and tail against the original global fault.

The explicit hybrid packet's existing owning input scan normalizes only the
seven inactive logical MIXED rows before copying the active row. Original GR
scratch legitimately retains prior MAX8 rows; GR's count1 mix overwrites row0
and its combine consumes neither MIXED nor inactive rows. The clear therefore
uses the actual validated base/stride and touches only rows1..7, establishing
the hybrid full-fold zero-inactive invariant without changing active arithmetic.
Count1 and the complete MAX8 view are checked before launch and again before
the shader's only input store. Active finite checks, prefix/stride gaps, logical
trailing gaps, allocator padding, original router words and the full fault page
remain inspected and cannot be normalized into acceptance. Default GPU scratch
is unchanged; no owner, dispatch, source/readback capacity or audit is added.
Regression source fixtures cover preserved MAX8 values, signed-zero row0,
corrupt active/gap/tail and malformed count/view, plus actual offline shader
lowering. Host/offline execution and physical MAX8-to-count1 acceptance remain
root's gates; source fixtures alone do not prove device mutation or speed.

Each compact Pending retains the original BankInstance/full physical pins,
immutable original ranks/probability bits and actual-miss/hit-slot proof, plus
one canonical GPU subset candidate. The temporary all-ten classification plan
is destroyed before retaining the Pending. Sequential final preflight rebuilds
that classification against the unchanged origin under existing working space,
checks the original-rank ownership and protected hit slots, then destroys it.
Each Pending stays within32KiB, routed metadata within2MiB and model metadata
within256KiB. These proofs confer no CPU/GPU publication authority: the original
whole-model output/fault, second PLE proof,36 GDN/12 QSA/48 MoE preflights and
GR/shared/cancel/context checks still precede every family commit. Faults,
panic/cancel or cleanup failure poison and erase the original owner; no retry,
fallback, private persistence or partial publication is added.

Root must compile/run meaningful host and offline shader checks, then perform
independent rank/format/status/tail numeric acceptance, actual count1 fixture
IDs/logits and both-card cancellation/poison/fresh recovery/full owning audits.
Source-only test additions and integrated operator calibration do not establish
this model seam or a performance benefit. Timing follows those acceptance gates.


## Protected whole-model registration, 2026-10-07

Implemented internal laboratory entry (not gateway/API inference):
`kuno-text-lab intel|nvidia resident-count1-hybrid-chat 0|10`.
Exactly four argv entries and a fresh `KUNO_PUBLIC_RECEIPT` destination are
required. Before any adapter creation/model payload access, require owned
transient I/O and full zero audits, native MAX8/decode/cancellation checks,
decode tile1 and `single-q4-all`. Prefill batch count is exactly8; the default
block kernel remains enabled. Bank slots/static cache/source-window/scatter
cannot be overridden. The other model/ordered/scalar/block-MoE/word-down/fused
fault/token-checkpoint controls remain disabled. The wrapper pins these settings
rather than copying the ambient environment.

This mode constructs `Config::reserve_count1_hybrid()`, runs the original
15-token chat prompt in8/7 chunks, then selects dense1/SingleQ4All and the
requested cap at the completed original prefill boundary. The fixture is
seed424242/temperature0, eight output IDs
`[9419,0,2500,628,353,1438,488,3242]`, with seven actual decode transactions.
No arbitrary prompt, routing trace, model rewrite, production policy or new
HTTP route is added. Ordinary constructors/modes still use1376 MAIN owners.

A transient `Zeroizing<u64>` completion counter starts at zero. The exclusive
all48-MoE acceptance permit derives its checked delta from the actual Pending
proofs only after joined CPU/source/bank validation and duplicate rejection.
All48 layers must preflight. Counter overflow and the480-expert per-token
maximum are checked before ANY family commit. The proposal is published only
alongside accepted model position, after all GDN/QSA/MoE finishes, PLE commit,
original bank metadata, health and cancellation checks. A failed/partial
transaction does not advance it; explicit cleanup erases it. Its getter requires
a healthy completed original-model boundary. No counter is persisted for
private inference.

Only this compiled public fixture can emit the new aggregate receipt fields:
`count1_hybrid_cpu_miss_cap`, `observed_count1_hybrid_cpu_miss_cap`,
`completed_cpu_misses`, `fresh_completed_cpu_misses`, and
`count1_hybrid_cpu_gpu_overlap=false`. Cap0 requires both counts zero; cap10
requires both positive, bounded by3360 and480 respectively. Fresh recovery
must run the same cap and match the next original token, not just replay prefill.
Layer0/3 cancellation probes also select the same cap and reject the failed
owner's completion getter. Existing pre-cancel, poisoned retry rejection and
fresh-first-token checks remain required. Four full1378-owner/818413568-byte
MAIN audits, empty source audits and normal cleanup/no retirement precede
receipt persistence. No per-layer counts, expert IDs, masks or probability
bits may appear in the receipt.

`run-native-count1-hybrid-lab.py` separately pins this inventory, exact original
outputs, observed policy, actual completed work, swap0/core0/8-GiB limits and
all ordinary plus hybrid lifecycle markers. The ordinary public MAX8 validator
still pins1376/818249728; it cannot opt into hybrid inventory. Both entries
share only an internal validation helper with independently supplied constants.
Malformed/incomplete receipts cannot create an accepted run record.

Host/offline tests and a normal build are separate from physical acceptance.
Run cap0 and cap10 sequentially on each card after coordinating exclusive
admission and verifying memory; do not time concurrently with compilation.
Then complete hybrid-specific late final-layer/second-proof/partial-commit
fault tests before production selection, followed by matched long timing.
The current CPU operator joins before GPU enqueue; no overlap, hot-request
reuse, automatic placement, performance gain or10-tokens/s acceptance is claimed.


### Guarded constructor reservation correction

The first protected Intel cap0 trial of `a8cb5e0` caught a recursive MAIN mutex
acquisition before prefill. `ResidentModel` already held the complete
`ResidentAdmission::allocate` callback lock; the new reserved hybrid workspace
called ordinary `healthy()`, which tried `admission.active()` on that same lock.
It was stopped without a device reset. No whole-model/erasure/performance
acceptance is inferred from this stopped attempt or its systemd exit status.
Physical Intel VRAM returned to50688000 bytes and content-free Xe fault
classification remained zero.

The correction passes a non-cloneable borrowed `ModelConstructionAllocation`
from inside the existing guarded complete-model allocation callback, through
native workspace construction to hybrid reservation. It checks exact ledger,
retirement and sequence-poison identities and flags without locking again.
Reservation still validates original arena IDs, fresh policy, AVX2/16 workers
and all page sizes. Ordinary runtime `healthy()` and allocation/fence/cleanup
locks remain unchanged. A held-mutex regression test requires this exact origin
validator to complete, with foreign-ledger/stream/sequence and poisoned flags
rejected. Host checks alone do not establish the corrected physical trial.


### Corrected short physical acceptance

The `ff3c7d1` normal `4382edf2…` build passes the protected cap0/10 chat fixture
on both Intel and NVIDIA: all eight original output IDs, seven native steps,
actual cap0 count0 and cap10 counts1256/2259, fresh continuation counts102/273,
pre/mid cancellation, poisoned retry/fresh recovery, four complete1378-owner
MAIN zero audits/source0, normal cleanup and no context retirement. Exact
receipts and limits are recorded in the
[latest checkpoint](kuno-native-text-checkpoint.md#both-card-short-hybrid-acceptance-and-constructor-correction-2026-10-07).
This supersedes the earlier corrected-constructor physical pending statement
for the short fixture only. It does not establish full-vocabulary logit gates,
hybrid late-finality failure gates,75/128 acceptance, repeated long performance,
CPU/GPU overlap, automatic placement, production routing or10 tokens/s.

## Reusable CPU worker experiment, 2026-10-07

The user authorized worker reuse followed by an overlap experiment. This first
slice changes only CPU scheduling, with an explicit protected-lab selector
`KUNO_LAB_CPU_WORKERS=scoped|pooled`. Absent means scoped. Other lab modes,
unknown values and invalid encodings reject before device/model access. The
runner pins `--cpu-workers` (default scoped), and requires both requested and
completed worker modes in the receipt. Historical receipts remain immutable;
this stronger runner requires the newly explicit fields on new executions.
There is no gateway, model catalog, API or Vashti change.

`ExpertOperator::new_pooled` builds one bounded 16-worker Rayon pool shared by
the model's layers. Physical affinity is set and checked on each worker during
construction; it never changes the parent thread. Each synchronous broadcast
owns one disjoint original contiguous row chunk per fixed worker index, catches
worker errors/panics, completes every callback, and inspects every completion
before returning. A failed call clears all output. Per-call original scratch,
packed jobs, activations and partials retain their zeroizing ownership; retained
row pointers/bounds/results use the existing full-capacity typed header erasure.
The original private precharge conservatively includes the larger of scoped
handles and pooled row headers. Pool topology survives, request jobs do not.
Opaque Rayon job storage, OS stacks and compiler/register copies are outside
the application-owned erasure claim, as opaque std thread storage was before.
No math, precision, original rank reduction, cancellation, CPU cap or publication
rules change. This slice still joins CPU before GPU enqueue and claims no overlap.

A fixed-public scheduling-only ABBA control ran 128 calls with 16 physical
workers and original scratch allocation but no expert arithmetic. Scoped:
50,534/57,111 microseconds; pooled:1,505/1,500 microseconds. These measurements
exclude pool construction and establish scheduling savings only, not actual
model speed. New host checks cover repeated bitwise original seven-format
results, fixed worker-thread reuse and parent affinity, joined panic/cancel/error
with subsequent clean work, and full-capacity row-header erasure. Whole-model
short device comparisons and existing full-logit/late-finality/long gates remain
required before performance selection.

Source acceptance:464 native-lab tests;110 kuno-text unit/integration tests
(two installed-model tests and one timing test ignored in the ordinary suite);
107 Python controls; both crates strict release all-target Clippy; formatting
and whitespace checks pass. The separate ignored public timing control passed.
Device acceptance is still pending at this source commit.

## Explicit joined CPU/GPU overlap slice, 2026-10-07

The separately approved next experiment adds `KUNO_LAB_CPU_GPU_OVERLAP=0|1`
(default absent/0). Only the protected count1 hybrid mode accepts this setting;
1 requires pooled workers. The runner explicitly pins `--cpu-gpu-overlap` and
validates requested/completed booleans together with the worker selection.
Unknown encodings, other modes and scoped+overlap reject before device access.
No production/default, public API, model catalog or Vashti change is selected.

A single original scoped orchestrator borrows the existing reader/selection/
input, invokes the pooled core operator, and joins all pool callbacks. Its
handle uses the original typed full-capacity metadata owner and an additional
32-KiB charge subtracted from the same1-MiB CPU/private remainder. The parent
stays on the original GPU stream and exclusively fills actual GPU misses,
encodes original map/probability owning pages with exact-capacity checks, and
launches the real GPU subset/finite check. Both actual owning header capacities
plus4096B parent frame are additionally subtracted before concurrent CPU fill.
Public CPU packed jobs remain at most40MiB; simultaneous GPU public source
precharge remains separately bounded at40MiB. Existing MAIN and independent
runtime-I/O hard caps do not change. Parent flush submits original stream work
without waiting; it does not grant completion, release source debt, remove a
fence, publish bank state or reuse scratch.

Before CPU join, dispatch uses the existing original-source/canonical-plan
validation with `require_join=false`; this supplies no acceptance authority.
After join, `Proof::complete` and the unchanged joined source proof are mandatory
before encoding CPU results/readiness. CPU readiness and results use distinct
original owners; rank0..9 multiply/add, shared path and whole-model all-family
publication remain unchanged. The final pending/owning/PLE/fault proofs still
require the actual joined CPU owner. Both serial and overlap controls use the
same GPU-subset preparation and post-join CPU upload ordering; only concurrent
execution/early flush differs.

Parent error or panic is held until the CPU orchestrator joins outside admission
locks; CPU error/panic/cancellation likewise prevents both outputs returning.
No failed GPU work falls back to CPU, no partial token is accepted, and original
sticky poison/drain/erase/retirement applies. A bounded fixed-public host test
synchronizes a live CPU row with parent work and checks success, parent error/
panic, cancellation and CPU panic followed by clean pool reuse. Physical device
submission overlap is a scheduling capability, not an instrumented claim of
simultaneous kernel execution or a speed improvement. Full-logit, late-finality
and long gates remain outstanding before selection, even if short outputs pass.

Source gate:111 core unit/integration tests,465 native tests and108 Python
controls pass; both crates strict release all-target Clippy, formatting and
whitespace pass. Physical overlap execution remains pending at this source commit.

Both-card protected short ABBA execution now passes for44fd9bc and normal
artifactae9c2934. All output/lifecycle/owning erasure checks pass. Observed mean
duration differences are1.96% B70 and3.40%3060, within sample variation; no
reliable overlap speed selection, long/production or full-logit claim. See the
[latest checkpoint](kuno-native-text-checkpoint.md#joined-overlap-short-abba-checkpoint-2026-10-07)
and exact rust-native-count1-overlap-20261007.json receipt. Admission is restored
and physical Intel memory recovers without classified Xe faults.

## Shared resident arithmetic in complete count1 Next, 2026-10-08

The explicit laboratory selection `KUNO_LAB_NEXT_SHARED=1` connects the accepted
Ornith vector packed arithmetic and cached parallel512/top10 router to the
complete original Next graph. It requires protected cap10, fast GPU projections
and GPU routing; fused finite checks are excluded. Defaults remain off. All
operations use the existing CubeCL stream, original model files, public weight
tiles and complete private input/output owners. Public dimensions/dtype alone
identify pipelines; offsets travel in the transient16-byte INFO ledger with
the original full-capacity erasure. No second backend/device or API is added.

Only count1 dense projections and routing change. Original grouped GPU experts,
CPU misses, rank merge, all-family acceptance, finite/fault checks, normal
cleanup and swap0/core0 limits remain. The shared parallel router accepts the
complete finite f32 range, uses negative infinity for eliminated candidates and
preserves deterministic lower-ID ties. GPU startup checks compare six original
tensor slabs to independently decoded Rust rows at absolute5e-4 plus relative
5e-5 tolerance and check eight router patterns including finite extrema and
complete owning intervals. Changed reduction order is not assumed to imply
whole-model output parity or a performance win.

The normal runner requires actual shared dense calls covering every fast dense
dispatch plus the original-row/router probe markers. Source tests:478 native
tests, both separate offline shader suites,20 hybrid Python controls and strict
release all-bin Clippy pass. Complete-model device acceptance and timing are
pending at this source checkpoint. No installed service, consumer or release
change; required Vashti changes: none.

The further explicit `KUNO_LAB_NEXT_SHARED_GDN=1` requires shared Next kernels
and replaces only the count1 recurrent update. Four adjacent columns execute
per invocation, with query/key vectors cached once per head. Sequential f32
decay/prediction/update/readout order, immutable old state, per-column failure
zeroing and original status/finite/all-family gates remain. The original eight
full distinct owners and every MAX8 extent are checked before dispatch; the
new shader additionally requires head128 and aligned contiguous vec4 columns.
Runtime bases/strides use the original transient64-byte INFO ownership. No
state/publication or cleanup authority is granted to this arithmetic function.
The counter must cover all36 recurrent stages in each completed decode step.
Startup compares full candidate/output/status owners and unchanged old state
against the original checked scalar shader for normal, zero-beta and nonfinite
columns, including nonzero surrounding intervals. Source gates pass; device
acceptance and whole-model performance remain pending for this additional mode.

## Batched private parameter uploads, 2026-10-08

`KUNO_LAB_INFO_BATCH_UPLOAD=1` is an explicit protected compact-cap10 experiment,
off by default. It appends each kernel's parameters into distinct aligned
regions of an initialized mapped source page and its original uniform target.
At submission, a separate copy command buffer precedes the graph command
buffer. Submission closes every append page; no in-flight uniform is changed.
This removes per-kernel source allocations and compute-pass interruptions,
without changing arithmetic or graph acceptance. Full source and target pages,
including audit duplicates, stay charged to the existing runtime-I/O cap and
tracked through normal erasure and inspection. Failed preparation keeps the
original owners and unmaps sources before cleanup. See the
[runtime ownership details](../apps/kuno/benchmarks/native-text/runtime-ownership/README.md).

The runner pins the flag and checks both the actual runtime marker and receipt
selection. Source gates pass:478 native tests, both offline shader suites,
50 imported runtime tests,21 hybrid runner controls and strict Clippy for both
native and imported runtime code. Physical complete-model acceptance and
performance remain pending for this mode. No API, installed service or Vashti
consumer changes are included.

## Direct device enqueue experiment, 2026-10-08

The standalone Cargo feature `direct-device` selects the pinned common
runtime's existing reentrant mutex service handle instead of its threaded
channel. It is disabled in normal builds. A compiled candidate requires
`--device-dispatch inline`/`KUNO_LAB_DEVICE_DISPATCH=inline` in protected cap10
mode, checked before device access against the actual runtime handle property.
Normal builds accept the default `threaded` choice; mismatched artifacts, other
encodings, cap0 and other entry points reject the inline choice. The runner
checks both the actual marker and final selection/property receipt.

The original WGPU server and queue still execute all graph work. Caller-side
server enqueues remain serialized by the runtime's existing reentrant device
lock; GPU commands remain asynchronous until their original fences. No math,
logical/physical ownership, full-budget, finite/publication or normal erasure
gate is removed. The only additional vendored dependency is the same pinned
CubeCL0.10 common archive, with original licenses/provenance. It is excluded
from deployed Kuno dependencies. Distributed collectives and arbitrary private
requests are outside this experiment. Device acceptance/performance are pending.
